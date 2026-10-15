import random
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.config import AugmentationConfig, DataConfig

DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_RANDOM_SEED: int = 42
SPLIT_NAMES: List[str] = ["train", "val", "test"]
SHUFFLE_BY_SPLIT: Dict[str, bool] = {"train": True, "val": False, "test": False}
NORMALIZATION_MEAN: List[float] = [0.485, 0.456, 0.406]
NORMALIZATION_STD: List[float] = [0.229, 0.224, 0.225]


def set_random_seed(random_seed: int) -> None:
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)


def discover_class_names(data_directory: Path) -> List[str]:
    train_directory: Path = data_directory / "train"
    assert train_directory.is_dir(), f"train directory not found: {train_directory}"
    return sorted([
        entry.name
        for entry in train_directory.iterdir()
        if entry.is_dir()
    ])


def build_augmentation_operations(
    augmentation: AugmentationConfig,
) -> List[Callable]:
    operations: List[Callable] = []

    if augmentation.horizontal_flip_probability > 0.0:
        operations.append(
            transforms.RandomHorizontalFlip(p=augmentation.horizontal_flip_probability)
        )
    if augmentation.vertical_flip_probability > 0.0:
        operations.append(
            transforms.RandomVerticalFlip(p=augmentation.vertical_flip_probability)
        )
    if augmentation.rotation_degrees > 0.0:
        operations.append(
            transforms.RandomRotation(degrees=augmentation.rotation_degrees)
        )
    if any([
        augmentation.brightness_jitter,
        augmentation.contrast_jitter,
        augmentation.saturation_jitter,
        augmentation.hue_jitter,
    ]):
        operations.append(
            transforms.ColorJitter(
                brightness=augmentation.brightness_jitter,
                contrast=augmentation.contrast_jitter,
                saturation=augmentation.saturation_jitter,
                hue=augmentation.hue_jitter,
            )
        )
    if (
        augmentation.translation_fraction > 0.0
        or augmentation.scale_minimum != 1.0
        or augmentation.scale_maximum != 1.0
    ):
        operations.append(
            transforms.RandomAffine(
                degrees=0.0,
                translate=(
                    augmentation.translation_fraction,
                    augmentation.translation_fraction,
                ),
                scale=(augmentation.scale_minimum, augmentation.scale_maximum),
            )
        )

    return operations


def build_image_transform(
    data_config: DataConfig,
    is_training: bool,
) -> transforms.Compose:
    operations: List[Callable] = [
        transforms.Resize((data_config.image_size, data_config.image_size)),
    ]

    if is_training and data_config.augmentation is not None:
        operations.extend(build_augmentation_operations(data_config.augmentation))

    operations.append(transforms.ToTensor())

    if data_config.normalize:
        operations.append(
            transforms.Normalize(
                mean=NORMALIZATION_MEAN,
                std=NORMALIZATION_STD,
            )
        )

    return transforms.Compose(operations)


def build_dataloader(
    split_directory: Path,
    split_name: str,
    data_config: DataConfig,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    image_folder: datasets.ImageFolder = datasets.ImageFolder(
        root=str(split_directory),
        transform=build_image_transform(
            data_config=data_config,
            is_training=split_name == "train",
        ),
    )
    return DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=SHUFFLE_BY_SPLIT[split_name],
        num_workers=num_workers,
    )


def build_dataloaders(
    data_directory: Path,
    data_config: DataConfig,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> Dict[str, DataLoader]:
    assert batch_size > 0, "batch_size must be positive"
    assert num_workers >= 0, "num_workers must be non-negative"
    assert data_directory.is_dir(), f"data directory not found: {data_directory}"

    set_random_seed(random_seed)

    dataloaders: Dict[str, DataLoader] = {}

    for split_name in SPLIT_NAMES:
        split_directory: Path = data_directory / split_name
        assert split_directory.is_dir(), f"split directory not found: {split_directory}"
        dataloaders[split_name] = build_dataloader(
            split_directory=split_directory,
            split_name=split_name,
            data_config=data_config,
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return dataloaders
