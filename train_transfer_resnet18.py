from pathlib import Path
import json
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from age_config import NUM_AGE_CLASSES
from models.transfer_resnet18 import ResNet18MultiTask
from preprocessing import build_loaders

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "models" / "transfer_resnet18"
EPOCHS = 50
PATIENCE = 8
WEIGHT_DECAY = 1e-4

def run_epoch(model, loader, age_criterion, gender_criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_age_loss = 0.0
    total_gender_loss = 0.0
    age_true, age_pred = [], []
    gender_true, gender_pred = [], []

    for batch in loader:
        images = batch["image"].to(device)
        ages = batch["age_class"].to(device)
        genders = batch["gender"].to(device)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            age_logits = outputs["age"]
            gender_logits = outputs["gender"]
            age_loss = age_criterion(age_logits, ages)
            gender_loss = gender_criterion(gender_logits, genders)
            loss = 1.5 * age_loss + 0.5 * gender_loss

            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

        size = images.size(0)
        total_loss += loss.item() * size
        total_age_loss += age_loss.item() * size
        total_gender_loss += gender_loss.item() * size

        age_true.extend(ages.cpu().tolist())
        age_pred.extend(age_logits.argmax(dim=1).cpu().tolist())
        gender_true.extend(genders.cpu().tolist())
        gender_pred.extend(gender_logits.argmax(dim=1).cpu().tolist())

    count = len(age_true)

    return {
        "loss": total_loss / count,
        "age_loss": total_age_loss / count,
        "gender_loss": total_gender_loss / count,
        "age_accuracy": sum(a == b for a, b in zip(age_true, age_pred)) / count,
        "age_macro_f1": f1_score(
            age_true, age_pred,
            labels=list(range(NUM_AGE_CLASSES)),
            average="macro",
            zero_division=0
        ),
        "gender_accuracy": sum(a == b for a, b in zip(gender_true, gender_pred)) / count,
        "gender_f1": f1_score(
            gender_true, gender_pred,
            average="binary",
            zero_division=0
        )
    }


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    loaders = build_loaders(model_type="resnet18")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        age_labels = loaders["train"].dataset.data["age_class"].tolist()
        counts = torch.bincount(
            torch.tensor(age_labels, dtype=torch.long),
            minlength=NUM_AGE_CLASSES
        ).float()

        age_weights = (counts.sum() / (NUM_AGE_CLASSES * counts.clamp_min(1))).sqrt()
        age_weights = (age_weights / age_weights.mean()).to(device)

        model = ResNet18MultiTask(
            num_age_classes=NUM_AGE_CLASSES,
            num_gender_classes=2,
            pretrained=True
        ).to(device)

        age_criterion = nn.CrossEntropyLoss(weight=age_weights)
        gender_criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            [
                {"params": model.backbone.parameters(), "lr": 1e-4},
                {"params": model.age_head.parameters(), "lr": 3e-4},
                {"params": model.gender_head.parameters(), "lr": 3e-4}
            ],
            weight_decay=WEIGHT_DECAY
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        best_score = float("-inf")
        epochs_without_improvement = 0
        history = []

        for epoch in range(1, EPOCHS + 1):
            train_metrics = run_epoch(
                model, loaders["train"], age_criterion,
                gender_criterion, device, optimizer
            )
            val_metrics = run_epoch(
                model, loaders["val"], age_criterion,
                gender_criterion, device
            )

            score = val_metrics["age_macro_f1"]
            scheduler.step(val_metrics["age_macro_f1"])

            history.append({
                "epoch": epoch,
                "train": train_metrics,
                "validation": val_metrics,
                "checkpoint_score": score
            })

            print(
                f"Epoch {epoch:02d}/{EPOCHS} | "
                f"train_loss={train_metrics['loss']:.4f} | "
                f"val_loss={val_metrics['loss']:.4f} | "
                f"val_age_acc={val_metrics['age_accuracy']:.4f} | "
                f"val_age_macro_f1={val_metrics['age_macro_f1']:.4f} | "
                f"val_gender_acc={val_metrics['gender_accuracy']:.4f}"
            )

            if score > best_score:
                best_score = score
                epochs_without_improvement = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "validation_metrics": val_metrics,
                        "checkpoint_score": score
                    },
                    OUTPUT_DIR / "best_model.pt"
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= PATIENCE:
                break

        (OUTPUT_DIR / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8"
        )

    finally:
        for loader in loaders.values():
            loader.dataset.close()


if __name__ == "__main__":
    main()