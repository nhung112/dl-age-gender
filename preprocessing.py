"""
FILE 2: PREPROCESSING UTKFACE 

Chạy:
    python prepare_utkface_part2.py

Thư viện:
    pandas, numpy, Pillow, torch, torchvision, matplotlib

Đầu vào:
    Các CSV đã được file 1 chia và ảnh trong ZIP.

Đầu ra:
    DataLoader dùng cho Simple CNN, Complex CNN và ResNet18.
    Cấu hình preprocessing, báo cáo ảnh trùng, preview augmentation.

Không sửa ảnh gốc. Không chia lại train/val/test.
Hash chỉ phát hiện ảnh RGB giống hệt, không kiểm chứng khác người.
"""

from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
import hashlib
import json
import random
import tempfile

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt


# ============================================================
# 1. CẤU HÌNH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

ZIP_PATH = PROJECT_ROOT / "data/raw/UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data/metadata"
OUTPUT_DIR = PROJECT_ROOT / "data/preprocessing"

IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42

COLUMNS = ["member", "age", "gender", "race"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def set_seed(seed=SEED):
    """Cố định trạng thái ngẫu nhiên trước mỗi model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ============================================================
# 2. ĐỌC VÀ KIỂM TRA SPLIT CỦA FILE 1
# ============================================================

def read_metadata(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")

    data = pd.read_csv(path, encoding="utf-8-sig")
    data.columns = [
        str(column).replace("\ufeff", "").strip().lower()
        for column in data.columns
    ]

    if not set(COLUMNS).issubset(data.columns):
        raise ValueError(f"CSV is missing a required column: {path}")

    data = data[COLUMNS].copy()

    if data.empty or data.isna().any().any():
        raise ValueError(f"CSV is empty or has missing values in columns: {path}")

    if data["member"].duplicated().any():
        raise ValueError(f"CSV has duplicate member values: {path}")

    for column in ["age", "gender", "race"]:
        values = pd.to_numeric(data[column], errors="raise")

        if not np.isfinite(values).all() or not (values % 1 == 0).all():
            raise ValueError(f"Column {column} must be finite integers.")

        data[column] = values.astype(int)

    if not data["age"].between(1, 116).all():
        raise ValueError("Age must be between range 1 and 116.")

    if not data["gender"].isin([0, 1]).all():
        raise ValueError("Gender must be either 0 or 1.")

    if not data["race"].isin([0, 1, 2, 3, 4]).all():
        raise ValueError("Race is invalid.")


    normalized_members = []
    for member in data["member"]:
        if not isinstance(member, str) or not member.strip():
            raise ValueError(f"Invalid ZIP member in {path}: {member!r}")

        normalized = member.strip().replace("\\", "/")
        parts = normalized.split("/")

        if (
            normalized.startswith("/")
            or len(normalized) >= 2 and normalized[1] == ":"
            or ".." in parts
            or not normalized.lower().endswith(".jpg")
        ):
            raise ValueError(f"Invalid ZIP member in {path}: {member!r}")

        normalized_members.append(normalized)

    data["member"] = normalized_members

    return data


def load_splits(metadata_dir=METADATA_DIR):
    """Kiểm tra overlap, coverage và nhãn; không chia lại."""
    metadata_dir = Path(metadata_dir)

    clean = read_metadata(metadata_dir / "clean_metadata.csv")

    splits = {
        name: read_metadata(metadata_dir / f"{name}.csv")
        for name in ["train", "val", "test"]
    }

    combined = pd.concat(splits.values(), ignore_index=True)

    if combined["member"].duplicated().any():
        raise ValueError("Member appears in multiple splits.")

    pd.testing.assert_frame_equal(
        combined.sort_values("member").reset_index(drop=True),
        clean.sort_values("member").reset_index(drop=True)
    )

    for name, data in splits.items():
        print(f"{name}: {len(data):,} images")

    return splits


# ============================================================
# 3. TÌM ẢNH BÊN TRONG ZIP
# ============================================================

def create_zip_index(zip_path, members):
    """Validate exact ZIP member paths stored by file 1."""
    zip_path = Path(zip_path)

    if not zip_path.is_file():
        raise FileNotFoundError(f"UTKFace ZIP not found: {zip_path}")

    requested_members = {
        str(member).strip().replace("\\", "/")
        for member in members
    }

    with ZipFile(zip_path) as archive:
        zip_members = set(archive.namelist())

    missing = requested_members - zip_members

    if missing:
        raise FileNotFoundError(
            f"ZIP is missing {len(missing)} images. Example: {sorted(missing)[:3]}"
        )

    return {member: member for member in requested_members}


# ============================================================
# 4. KIỂM TRA ẢNH LỖI VÀ ẢNH GIỐNG HỆT XUYÊN TẬP
# ============================================================

def audit_images(splits, zip_path, index, output_dir):
    records = []

    with ZipFile(zip_path) as archive:
        for split_name, data in splits.items():
            for member in data["member"]:
                try:
                    content = archive.read(index[member])

                    with Image.open(BytesIO(content)) as image:
                        image = image.convert("RGB")
                        image.load()

                        payload = str(image.size).encode() + image.tobytes()
                        image_hash = hashlib.sha256(payload).hexdigest()

                except (OSError, ValueError) as error:
                    raise ValueError(
                        f"Failed to read image: {member}"
                    ) from error

                records.append({
                    "member": member,
                    "split": split_name,
                    "image_hash": image_hash
                })

    table = pd.DataFrame(records)

    duplicates = table[
        table.duplicated("image_hash", keep=False)
    ]

    duplicates.to_csv(
        output_dir / "exact_duplicates.csv",
        index=False
    )

    split_counts = table.groupby("image_hash")["split"].nunique()
    bad_hashes = split_counts[split_counts > 1].index

    cross_split = table[table["image_hash"].isin(bad_hashes)]

    if not cross_split.empty:
        cross_split.to_csv(
            output_dir / "cross_split_duplicates.csv",
            index=False
        )

        raise ValueError(
            "Cross-split duplicates found. "
            "See cross_split_duplicates.csv and process in file 1."
        )

    print("Images checked.")


# ============================================================
# 5. RESIZE, AUGMENTATION VÀ NORMALIZATION
# ============================================================

def create_transform(model_type, training, image_size=IMAGE_SIZE):
    if model_type not in ["scratch", "resnet18"]:
        raise ValueError("model_type must be either scratch or resnet18.")

    steps = [
        transforms.Resize((image_size, image_size))
    ]

    if training:
        steps.extend([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1
            )
        ])

    steps.append(transforms.ToTensor())

    if model_type == "resnet18":
        steps.append(
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        )

    return transforms.Compose(steps)


# ============================================================
# 6. DATASET: ĐỌC MỘT ẢNH VÀ NHÃN TƯƠNG ỨNG
# ============================================================

class UTKFaceDataset(Dataset):
    def __init__(self, data, zip_path, index, transform):
        self.data = data.reset_index(drop=True)
        self.zip_path = Path(zip_path)
        self.index = index
        self.transform = transform
        self.archive = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, position):
        row = self.data.iloc[position]

        if self.archive is None:
            self.archive = ZipFile(self.zip_path)

        content = self.archive.read(self.index[row["member"]])

        with Image.open(BytesIO(content)) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        return {
            "image": image,
            "age": torch.tensor(
                [float(row["age"])],
                dtype=torch.float32
            ),
            "gender": torch.tensor(
                int(row["gender"]),
                dtype=torch.long
            ),
            "member": row["member"]
        }

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None


# ============================================================
# 7. DATALOADER: GOM ẢNH THÀNH BATCH
# ============================================================

def build_loaders(
    model_type="scratch",
    metadata_dir=METADATA_DIR,
    zip_path=ZIP_PATH,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    seed=SEED
):
    if image_size < 32 or batch_size < 1:
        raise ValueError("image_size >= 32 and batch_size >= 1.")

    set_seed(seed)

    splits = load_splits(metadata_dir)

    members = pd.concat(splits.values())["member"]
    index = create_zip_index(zip_path, members)

    loaders = {}

    for name, data in splits.items():
        transform = create_transform(
            model_type=model_type,
            training=(name == "train"),
            image_size=image_size
        )

        dataset = UTKFaceDataset(
            data=data,
            zip_path=zip_path,
            index=index,
            transform=transform
        )

        loaders[name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(name == "train"),
            drop_last=False,
            num_workers=0,
            generator=torch.Generator().manual_seed(seed)
        )

    return loaders


# ============================================================
# 8. KIỂM TRA BATCH VÀ AUGMENTATION
# ============================================================

def check_batches(loaders, model_type, image_size=IMAGE_SIZE):
    for name, loader in loaders.items():
        batch = next(iter(loader))
        size = len(batch["member"])

        assert batch["image"].shape == (size, 3, image_size, image_size)
        assert batch["age"].shape == (size, 1)
        assert batch["gender"].shape == (size,)

        assert batch["image"].dtype == torch.float32
        assert batch["age"].dtype == torch.float32
        assert batch["gender"].dtype == torch.int64

        assert torch.isfinite(batch["image"]).all()
        assert batch["age"].ge(1).all() and batch["age"].le(116).all()
        assert ((batch["gender"] == 0) | (batch["gender"] == 1)).all()

        if model_type == "scratch":
            assert batch["image"].min() >= 0
            assert batch["image"].max() <= 1

        print(model_type, name, tuple(batch["image"].shape))

    for name in ["val", "test"]:
        dataset = loaders[name].dataset

        assert torch.equal(
            dataset[0]["image"],
            dataset[0]["image"]
        )


def save_augmentation_preview(loader, model_type, output_path):
    """Xem cùng một ảnh train qua nhiều lần augmentation"""
    dataset = loader.dataset

    figure, axes = plt.subplots(1, 4, figsize=(12, 3))

    for axis in axes:
        image = dataset[0]["image"].permute(1, 2, 0)

        if model_type == "resnet18":
            image = (
                image * torch.tensor(IMAGENET_STD)
                + torch.tensor(IMAGENET_MEAN)
            )

        axis.imshow(image.clamp(0, 1).numpy())
        axis.axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


# ============================================================
# 9. LƯU CẤU HÌNH ĐỂ INFERENCE XỬ LÝ GIỐNG TRAINING
# ============================================================

def save_config(output_dir):
    config = {
        "image_size": [IMAGE_SIZE, IMAGE_SIZE],
        "color": "RGB",
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "gender_encoding": {"0": "Male", "1": "Female"},
        "age_unit": "years",
        "scratch": {"pixel_range": [0, 1]},
        "resnet18": {
            "pixel_range_before_normalize": [0, 1],
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD
        },
        "train_augmentation": {
            "horizontal_flip_probability": 0.5,
            "brightness": 0.1,
            "contrast": 0.1
        },
        "validation_test_augmentation": False,
        "identity_disjoint_verified": False,
        "split_sha256": {
            name: hashlib.sha256(
                (METADATA_DIR / f"{name}.csv").read_bytes()
            ).hexdigest()
            for name in ["train", "val", "test"]
        }
    }

    path = output_dir / "preprocessing_config.json"
    path.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8"
    )


# ============================================================
# 10. CHẠY TOÀN BỘ PREPROCESSING
# ============================================================

def main():
    splits = load_splits()

    members = pd.concat(splits.values())["member"]
    index = create_zip_index(ZIP_PATH, members)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix="run_", dir=OUTPUT_DIR)
    )

    audit_images(splits, ZIP_PATH, index, run_dir)
    save_config(run_dir)

    for model_type in ["scratch", "resnet18"]:
        loaders = build_loaders(model_type=model_type)

        try:
            check_batches(loaders, model_type)

            save_augmentation_preview(
                loaders["train"],
                model_type,
                run_dir / f"augmentation_{model_type}.png"
            )
        finally:
            for loader in loaders.values():
                loader.dataset.close()

    print("Preprocessing completed.")
    print("Output:", run_dir)
    print("Create new DataLoader before training each model.")


if __name__ == "__main__":
    main()
