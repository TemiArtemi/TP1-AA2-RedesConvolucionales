import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

IMAGE_SIZE: int = 96
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_NUM_WORKERS: int = 0
DEFAULT_RANDOM_SEED: int = 42
SPLIT_NAMES: List[str] = ["train", "val", "test"]
SHUFFLE_BY_SPLIT: Dict[str, bool] = {"train": True, "val": False, "test": False}


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


def build_image_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
    ])


def build_dataloader(
    split_directory: Path,
    shuffle: bool,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    image_folder: datasets.ImageFolder = datasets.ImageFolder(
        root=str(split_directory),
        transform=build_image_transform(),
    )
    return DataLoader(
        dataset=image_folder,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def build_dataloaders(
    data_directory: Path,
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
            shuffle=SHUFFLE_BY_SPLIT[split_name],
            batch_size=batch_size,
            num_workers=num_workers,
        )

    return dataloaders
