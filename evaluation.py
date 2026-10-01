import torch

from training import (
    AGE_LOSS_WEIGHT,
    GENDER_LOSS_WEIGHT,
    MultitaskLoss,
    run_epoch,
    save_json
)


def evaluate_test(
    model,
    test_loader,
    device,
    output_dir,
    age_class_weights=None
):
    """Evaluate a loaded best checkpoint once on the untouched test split."""
    metrics = run_epoch(
        model=model,
        loader=test_loader,
        criterion=MultitaskLoss(
            age_class_weights=age_class_weights,
            age_weight=AGE_LOSS_WEIGHT,
            gender_weight=GENDER_LOSS_WEIGHT
        ),
        device=device,
        optimizer=None
    )
    save_json(
        {
            "metrics": metrics,
            "age_loss_weight": AGE_LOSS_WEIGHT,
            "gender_loss_weight": GENDER_LOSS_WEIGHT
        },
        output_dir / "test_metrics.json"
    )
    return metrics


def print_metrics(metrics):
    for name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"{name}: {value:.4f}")
        else:
            print(f"{name}: {value}")


def load_best_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint