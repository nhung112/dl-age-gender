"""Remove exact duplicate images from a UTKFace dataset."""

from __future__ import annotations

import argparse
import hashlib
import re
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator
from zipfile import ZipFile


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "raw" / "UTKFace.zip"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
FILENAME_PATTERN = re.compile(r"^(\d+)_([01])_([0-4])_")


@dataclass
class ImageFile:
    name: str
    data: bytes
    path: Path


def iter_images(source: Path) -> Iterator[ImageFile]:
    """Yield supported images from a directory or ZIP archive."""
    if source.is_dir():
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield ImageFile(path.name, path.read_bytes(), path)
        return

    if source.suffix.lower() == ".zip":
        with ZipFile(source) as archive:
            for member in sorted(archive.infolist(), key=lambda item: item.filename):
                name = Path(member.filename).name
                if not member.is_dir() and Path(name).suffix.lower() in IMAGE_SUFFIXES:
                    yield ImageFile(name, archive.read(member), Path(member.filename))
        return

    raise ValueError(f"Source must be an image folder or ZIP file: {source}")


def parse_label(filename: str) -> tuple[int, int, int] | None:
    match = FILENAME_PATTERN.match(filename)
    return tuple(map(int, match.groups())) if match else None


def group_by_hash(images: Iterable[ImageFile]) -> dict[str, list[ImageFile]]:
    groups: dict[str, list[ImageFile]] = defaultdict(list)
    for image in images:
        digest = hashlib.sha256(image.data).hexdigest()
        groups[digest].append(image)
    return groups


def is_same_label(group: list[ImageFile]) -> bool:
    labels = [parse_label(image.name) for image in group]
    return None not in labels and len(set(labels)) == 1


def clean_directory(groups: dict[str, list[ImageFile]]) -> tuple[int, int]:
    same_label_images = conflict_images = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        if is_same_label(group):
            same_label_images += len(group)
            for image in group[1:]:
                image.path.unlink()
        else:
            conflict_images += len(group)
            for image in group:
                image.path.unlink()
    return same_label_images, conflict_images


def clean_zip(source: Path, groups: dict[str, list[ImageFile]]) -> tuple[int, int]:
    """Rewrite the archive, keeping one same-label image per group."""
    kept_members: set[Path] = set()
    removed_members: set[Path] = set()
    same_label_images = conflict_images = 0

    for group in groups.values():
        if len(group) < 2:
            continue
        if is_same_label(group):
            same_label_images += len(group)
            kept_members.add(group[0].path)
            removed_members.update(image.path for image in group[1:])
        else:
            conflict_images += len(group)
            removed_members.update(image.path for image in group)

    with tempfile.NamedTemporaryFile(
        dir=source.parent, suffix=".zip", delete=False
    ) as temporary_file:
        temporary_zip = Path(temporary_file.name)

    try:
        with ZipFile(source) as original, ZipFile(temporary_zip, "w") as cleaned:
            for member in original.infolist():
                if Path(member.filename) in removed_members:
                    continue
                cleaned.writestr(member, original.read(member))
        temporary_zip.replace(source)
    finally:
        temporary_zip.unlink(missing_ok=True)

    return same_label_images, conflict_images


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    return parser


def main() -> None:
    source = build_parser().parse_args().source
    
    images = list(iter_images(source))
    groups = group_by_hash(images)

    if source.is_dir():
        same_label_images, conflict_images = clean_directory(groups)
    else:
        same_label_images, conflict_images = clean_zip(source, groups)

    print("=" * 60)
    print("UTKFACE DUPLICATE CLEANING")
    print("=" * 60)
    print(f"Source              : {source}")
    print(f"Images read         : {len(images):,}")
    print(f"Duplicate groups    : {sum(len(group) > 1 for group in groups.values()):,}")
    print(f"Same-label images   : {same_label_images:,}")
    print(f"Conflict images     : {conflict_images:,}")
    print("Same-label action   : kept 1 image per group")
    print("Conflict action     : removed entire group")

if __name__ == "__main__":
    main()
