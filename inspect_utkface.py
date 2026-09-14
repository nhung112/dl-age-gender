"""
UTKFace Data Preparation Pipeline

Pipeline:
1. Dataset inspection
2. Parse labels from filename
3. Data cleaning
4. Train / Validation / Test split

Processed images are NOT duplicated on disk.
Metadata and split information are stored as CSV files.

Target:
    Age    -> Regression or binned classification (0-116)
    Gender -> Binary classification (0 = Male, 1 = Female)
"""

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import re
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from PIL import Image
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

RAW_ARCHIVE = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

IMAGE_SIZE = (224, 224)
RANDOM_STATE = 42
STRATIFY_BY_AGE = True

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

METADATA_COLUMNS = ["member", "date", "age", "gender", "race"]

@dataclass
class ArchiveImage:
    member: str
    filename: str
    data: bytes

def parse_label(filename: str) -> tuple[int, int, int] | None:
    match = re.match(r"^(\d+)_([01])_([0-4])_", filename)
    return tuple(map(int, match.groups())) if match else None

def select_unique_members(images: list[ArchiveImage]) -> set[str]:
    """Select metadata members without modifying the raw archive."""
    groups = defaultdict(list)
    for image in images:
        groups[sha256(image.data).hexdigest()].append(image)

    kept_members = set()
    same_label_images = 0
    conflict_images = 0

    for group in groups.values():
        if len(group) == 1:
            kept_members.add(group[0].member)
            continue

        labels = {parse_label(image.filename) for image in group}
        if None not in labels and len(labels) == 1:
            kept_members.add(group[0].member)
            same_label_images += len(group)
        else:
            conflict_images += len(group)

    print(f"Same-label duplicate images : {same_label_images:,}")
    print(f"Conflict duplicate images   : {conflict_images:,}")

    return kept_members

# ============================================================
# DIRECTORY SETUP
# ============================================================

METADATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATASET INSPECTION + METADATA EXTRACTION
# ============================================================

def inspect_dataset():
    """
    Scan all JPG files, parse labels from filenames,
    validate image files and collect image metadata.
    """

    records = []
    invalid_filenames = []
    corrupted_images = []

    filename_pattern = re.compile(
        r"^(?P<age>\d+)_(?P<gender>[01])_(?P<race>[0-4])_"
        r"(?P<timestamp>\d+)\.jpg\.chip\.jpg$"
    )

    with ZipFile(RAW_ARCHIVE, "r") as archive:
        image_files = [
            ArchiveImage(member, Path(member).name, archive.read(member))
            for member in sorted(archive.namelist())
            if member.lower().endswith(".jpg")
        ]
        kept_members = select_unique_members(image_files)

        for image_file in image_files:
            if image_file.member not in kept_members:
                continue

            filename = image_file.filename
            match = filename_pattern.match(filename)

            if match is None:
                invalid_filenames.append(filename)
                continue

            age = int(match.group("age"))
            gender = int(match.group("gender"))
            race = int(match.group("race"))

            try:
                image_bytes = image_file.data

                with Image.open(BytesIO(image_bytes)) as image:
                    image.load()
                    width, height = image.size
                    channels = len(image.getbands())

            except (OSError, ValueError) as exc:
                corrupted_images.append(
                    {
                        "member": image_file.member,
                        "reason": str(exc)
                    }
                )
                continue

            records.append(
                {
                    "member": image_file.member,
                    "date": pd.to_datetime(
                        match.group("timestamp")[:8],
                        format="%Y%m%d",
                    ),
                    "age": age,
                    "gender": gender,
                    "race": race,
                    "width": width,
                    "height": height,
                    "channels": channels
                }
            )

    metadata = pd.DataFrame(records)

    print("\n" + "=" * 60)
    print("DATASET INSPECTION")
    print("=" * 60)

    print(f"JPG files found   : {len(image_files):,}")
    print(f"Valid images      : {len(metadata):,}")
    print(f"Invalid filenames : {len(invalid_filenames):,}")
    if invalid_filenames:
        for filename in invalid_filenames:
            print(f"  - {filename}")

    print(f"Corrupted images  : {len(corrupted_images):,}")

    if not metadata.empty:
        dimensions = (
                    metadata[["width", "height"]]
                    .value_counts()
                    .to_dict()
                )
        print(f"Image dimensions  : {dimensions}")
        
        print(
            f"Channels          : "
            f"{metadata['channels'].value_counts().sort_index().to_dict()}"
        )

        print(
            f"Age range         : "
            f"{metadata['age'].min()}-{metadata['age'].max()}"
        )

        print("\nGender:")
        print(
            metadata["gender"]
            .map({0: "Male", 1: "Female"})
            .value_counts()
        )

        print("\nRace:")
        print(
            metadata["race"]
            .map({
                0: "White",
                1: "Black",
                2: "Asian",
                3: "Indian",
                4: "Others"
            })
            .value_counts()
        )

    return metadata

# ============================================================
# 2. DATA CLEANING
# ============================================================

def clean_metadata(metadata):
    """
    Remove invalid labels and unreasonable image records.
    Raw images remain untouched.
    """

    before = len(metadata)

    clean = metadata.copy()

    valid_records = (
        clean["age"].between(1, 116)
        & clean["gender"].isin([0, 1])
        & clean["race"].isin([0, 1, 2, 3, 4])
        & clean["channels"].eq(3)
    )
    clean = clean.loc[valid_records].drop_duplicates("member")

    clean[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "clean_metadata.csv",
        index=False
    )

    print("\n" + "=" * 60)
    print("DATA CLEANING")
    print("=" * 60)
    print(f"Before cleaning  : {before:,}")
    print(f"After cleaning   : {len(clean):,}")
    print(f"Removed          : {before - len(clean):,}")

    return clean

# ============================================================
# 3. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def create_splits(metadata):
    """
    Create a fixed 70/15/15 split.

    Gender is always stratified. When enabled, age is grouped into
    five bins and combined with gender for stratification.
    """

    stratify_labels = metadata["gender"].astype(str)

    if STRATIFY_BY_AGE:
        age_bins = [0, 18, 30, 45, 60, float("inf")]
        age_labels = ["0-17", "18-29", "30-44", "45-59", "60+"]
        age_groups = pd.cut(
            metadata["age"],
            bins=age_bins,
            labels=age_labels,
            right=False,
            include_lowest=True
        ).astype(str)
        stratify_labels = age_groups + "_" + stratify_labels

    label_counts = stratify_labels.value_counts()
    if label_counts.min() < 2:
        raise ValueError(
            "Stratification groups must contain at least two samples. "
            "Set STRATIFY_BY_AGE = False or use wider age bins."
        )

    train, temp = train_test_split(
        metadata,
        test_size=VAL_RATIO + TEST_RATIO,
        random_state=RANDOM_STATE,
        stratify=stratify_labels
    )

    relative_test_ratio = TEST_RATIO / (
        VAL_RATIO + TEST_RATIO
    )

    temp_stratify_labels = stratify_labels.loc[temp.index].copy()
    rare_temp_groups = set(
        temp_stratify_labels.value_counts().loc[lambda counts: counts < 2].index
    )
    if rare_temp_groups:
        temp_stratify_labels = temp_stratify_labels.where(
            ~temp_stratify_labels.isin(rare_temp_groups),
            temp["gender"].astype(str)
        )

    val, test = train_test_split(
        temp,
        test_size=relative_test_ratio,
        random_state=RANDOM_STATE,
        stratify=temp_stratify_labels
    )

    train = train.copy()
    val = val.copy()
    test = test.copy()

    train["split"] = "train"
    val["split"] = "validation"
    test["split"] = "test"

    train[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "train.csv",
        index=False
    )

    val[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "val.csv",
        index=False
    )

    test[METADATA_COLUMNS].to_csv(
        METADATA_DIR / "test.csv",
        index=False
    )

    combined = pd.concat(
        [train, val, test],
        ignore_index=True
    )

    print("\n" + "=" * 60)
    print("DATA SPLIT")
    print("=" * 60)

    print(f"Train      : {len(train):,}")
    print(f"Validation : {len(val):,}")
    print(f"Test       : {len(test):,}")

    print("\nGender distribution:")
    print(
        pd.crosstab(
            combined["split"],
            combined["gender"],
            normalize="index"
        ).rename(
            columns={
                0: "Male",
                1: "Female"
            }
        )
    )

    return train, val, test

# ============================================================
# 4. PIPELINE COMPLETION
# ============================================================

def main():

    print("=" * 60)
    print("UTKFACE DATA PREPARATION")
    print("=" * 60)

    # Step 1 + 2: Inspect dataset and extract labels from filenames.
    metadata = inspect_dataset()
    if metadata.empty:
        raise RuntimeError(
            "No valid UTKFace images were found."
        )

    # Step 3: Clean invalid records.
    metadata = clean_metadata(metadata)

    # Step 4: Fixed train / validation / test split.
    train, val, test = create_splits(metadata)

if __name__ == "__main__":
    main()
