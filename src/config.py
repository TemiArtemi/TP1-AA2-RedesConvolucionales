from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class AugmentationConfig:
    horizontal_flip_probability: float = 0.5
    vertical_flip_probability: float = 0.0
    rotation_degrees: float = 10.0
    brightness_jitter: float = 0.3
    contrast_jitter: float = 0.3
    saturation_jitter: float = 0.3
    hue_jitter: float = 0.05
    translation_fraction: float = 0.1
    scale_minimum: float = 0.9
    scale_maximum: float = 1.1


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 96
    normalize: bool = False
    augmentation: Optional[AugmentationConfig] = None


@dataclass(frozen=True)
class ArchitectureConfig:
    conv_channels: Tuple[int, ...] = (16, 32, 64)
    kernel_size: int = 5
    fc_hidden_units: int = 128
    dropout_probability: float = 0.25
    activation: str = "relu"


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 10
    min_delta: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    random_seed: int
    data: DataConfig
    architecture: ArchitectureConfig
    training: TrainingConfig


BASELINE_CONFIG: ExperimentConfig = ExperimentConfig(
    experiment_id="E1",
    random_seed=42,
    data=DataConfig(
        image_size=96,
        normalize=False,
        augmentation=None,
    ),
    architecture=ArchitectureConfig(
        conv_channels=(16, 32, 64),
        kernel_size=5,
        fc_hidden_units=128,
        dropout_probability=0.25,
        activation="relu",
    ),
    training=TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        patience=10,
        min_delta=0.0,
    ),
)

AUGMENTED_CONFIG: ExperimentConfig = ExperimentConfig(
    experiment_id="E2",
    random_seed=42,
    data=DataConfig(
        image_size=96,
        normalize=False,
        augmentation=AugmentationConfig(),
    ),
    architecture=ArchitectureConfig(
        conv_channels=(16, 32, 64),
        kernel_size=5,
        fc_hidden_units=128,
        dropout_probability=0.25,
        activation="relu",
    ),
    training=TrainingConfig(
        learning_rate=0.001,
        batch_size=32,
        num_epochs=50,
        patience=10,
        min_delta=0.0,
    ),
)

CONFIG_REGISTRY: Dict[str, ExperimentConfig] = {
    "E1": BASELINE_CONFIG,
    "E2": AUGMENTED_CONFIG,
}
