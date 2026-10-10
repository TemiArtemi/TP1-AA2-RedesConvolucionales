import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple
import kagglehub


def create_split_structure(base_dir: Path, class_names: List[str]) -> None:
    for split_name in ["train", "val", "test"]:
        for class_name in class_names:
            destination_dir: Path = base_dir / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)


def partition_and_copy_images(
    images_list: List[Path],
    destination_base: Path,
    class_name: str,
    train_ratio: float,
    val_ratio: float,
) -> None:
    random.shuffle(images_list)
    total_samples: int = len(images_list)
    train_end_index: int = int(total_samples * train_ratio)
    val_end_index: int = int(total_samples * (train_ratio + val_ratio))

    partitions: List[Tuple[str, List[Path]]] = [
        ("train", images_list[:train_end_index]),
        ("val", images_list[train_end_index:val_end_index]),
        ("test", images_list[val_end_index:]),
    ]

    for split_name, split_images in partitions:
        target_directory: Path = destination_base / split_name / class_name
        for image_path in split_images:
            shutil.copy2(image_path, target_directory / image_path.name)


def download_and_organize_dataset(
    dataset_identifier: str,
    output_directory: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    random_seed: int = 42,
) -> None:
    random.seed(random_seed)
    download_path_str: str = kagglehub.dataset_download(dataset_identifier)
    source_root: Path = Path(download_path_str)
    output_root: Path = Path(output_directory)

    classes: List[str] = [
        "buildings",
        "forest",
        "glacier",
        "mountain",
        "sea",
        "street",
    ]

    create_split_structure(output_root, classes)

    for class_name in classes:
        collected_images: List[Path] = []
        possible_subfolders: List[Path] = [
            source_root / "seg_train" / "seg_train" / class_name,
            source_root / "seg_test" / "seg_test" / class_name,
        ]

        for folder in possible_subfolders:
            if folder.exists():
                collected_images.extend(
                    [
                        file_path
                        for file_path in folder.iterdir()
                        if file_path.is_file()
                        and file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                    ]
                )

        partition_and_copy_images(
            collected_images, output_root, class_name, train_ratio, val_ratio
        )


if __name__ == "__main__":
    download_and_organize_dataset(
        dataset_identifier="puneet6060/intel-image-classification",
        output_directory="data",
        train_ratio=0.7,
        val_ratio=0.2,
        random_seed=42,
    )