import random
import shutil
from pathlib import Path
from typing import Dict, List

SPLIT_NAMES: List[str] = ["train", "val", "test"]
ALLOWED_SUFFIXES: List[str] = [".jpg", ".jpeg", ".png"]


def discover_class_names(source_directory: Path) -> List[str]:
    return sorted([
        entry.name
        for entry in source_directory.iterdir()
        if entry.is_dir()
    ])


def clear_split_directories(output_directory: Path) -> None:
    for split_name in SPLIT_NAMES:
        split_directory: Path = output_directory / split_name
        if split_directory.exists():
            shutil.rmtree(split_directory)


def create_split_structure(output_directory: Path, class_names: List[str]) -> None:
    for split_name in SPLIT_NAMES:
        for class_name in class_names:
            destination_dir: Path = output_directory / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)


def collect_class_images(class_directory: Path) -> List[Path]:
    return [
        file_path
        for file_path in sorted(class_directory.iterdir())
        if file_path.is_file() and file_path.suffix.lower() in ALLOWED_SUFFIXES
    ]


def partition_images(
    images_list: List[Path],
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, List[Path]]:
    random.shuffle(images_list)
    total_samples: int = len(images_list)
    train_end_index: int = int(total_samples * train_ratio)
    val_end_index: int = int(total_samples * (train_ratio + val_ratio))

    return {
        "train": images_list[:train_end_index],
        "val": images_list[train_end_index:val_end_index],
        "test": images_list[val_end_index:],
    }


def copy_partitioned_images(
    partitions: Dict[str, List[Path]],
    output_directory: Path,
    class_name: str,
) -> None:
    for split_name, split_images in partitions.items():
        target_directory: Path = output_directory / split_name / class_name
        for image_path in split_images:
            shutil.copy2(image_path, target_directory / image_path.name)


def build_splits(
    source_directory: Path,
    output_directory: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Dict[str, int]]:
    assert train_ratio > 0.0, "train_ratio must be positive"
    assert val_ratio > 0.0, "val_ratio must be positive"
    assert train_ratio + val_ratio <= 1.0, (
        "train_ratio + val_ratio must not exceed 1.0, "
        "it must leave room for the test split"
    )
    assert source_directory.is_dir(), f"source directory not found: {source_directory}"
    assert source_directory.resolve() != output_directory.resolve(), (
        "source and output directories must be different"
    )

    random.seed(random_seed)

    class_names: List[str] = discover_class_names(source_directory)
    assert len(class_names) > 0, f"no class directories found in {source_directory}"

    clear_split_directories(output_directory)
    create_split_structure(output_directory, class_names)

    counts: Dict[str, Dict[str, int]] = {}

    for class_name in class_names:
        class_images: List[Path] = collect_class_images(source_directory / class_name)
        assert len(class_images) > 0, f"no images found for class: {class_name}"

        partitions: Dict[str, List[Path]] = partition_images(
            class_images, train_ratio, val_ratio
        )
        copy_partitioned_images(partitions, output_directory, class_name)

        counts[class_name] = {
            split_name: len(split_images)
            for split_name, split_images in partitions.items()
        }

    return counts


def print_summary(counts: Dict[str, Dict[str, int]]) -> None:
    class_column_width: int = 12
    count_column_width: int = 10

    header: str = f"{'class':<{class_column_width}}"
    header += "".join(
        f"{split_name:>{count_column_width}}" for split_name in SPLIT_NAMES
    )
    separator: str = "-" * len(header)

    print(header)
    print(separator)

    totals: Dict[str, int] = {split_name: 0 for split_name in SPLIT_NAMES}

    for class_name in sorted(counts):
        row: str = f"{class_name:<{class_column_width}}"
        for split_name in SPLIT_NAMES:
            sample_count: int = counts[class_name][split_name]
            totals[split_name] += sample_count
            row += f"{sample_count:>{count_column_width}}"
        print(row)

    print(separator)

    total_row: str = f"{'total':<{class_column_width}}"
    for split_name in SPLIT_NAMES:
        total_row += f"{totals[split_name]:>{count_column_width}}"
    print(total_row)


if __name__ == "__main__":
    split_counts = build_splits(
        source_directory=Path("data_master"),
        output_directory=Path("data"),
    )
    print_summary(split_counts)
