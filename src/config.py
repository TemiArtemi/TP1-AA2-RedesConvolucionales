from dataclasses import dataclass, replace
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

NORMALIZED_CONFIG: ExperimentConfig = replace(
    AUGMENTED_CONFIG,
    experiment_id="E3",
    data=replace(AUGMENTED_CONFIG.data, normalize=True),
)

HIGH_RESOLUTION_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E4",
    data=replace(NORMALIZED_CONFIG.data, image_size=128),
)

WIDER_CONFIG: ExperimentConfig = replace(
    HIGH_RESOLUTION_CONFIG,
    experiment_id="E5",
    architecture=replace(
        HIGH_RESOLUTION_CONFIG.architecture,
        conv_channels=(32, 64, 128),
        fc_hidden_units=256,
    ),
)

LOWER_LEARNING_RATE_CONFIG: ExperimentConfig = replace(
    WIDER_CONFIG,
    experiment_id="E6",
    training=replace(WIDER_CONFIG.training, learning_rate=0.0005),
)

ELU_ACTIVATION_CONFIG: ExperimentConfig = replace(
    LOWER_LEARNING_RATE_CONFIG,
    experiment_id="E7",
    architecture=replace(LOWER_LEARNING_RATE_CONFIG.architecture, activation="elu"),
)

CONFIG_REGISTRY: Dict[str, ExperimentConfig] = {
    "E1": BASELINE_CONFIG,
    "E2": AUGMENTED_CONFIG,
    "E3": NORMALIZED_CONFIG,
    "E4": HIGH_RESOLUTION_CONFIG,
    "E5": WIDER_CONFIG,
    "E6": LOWER_LEARNING_RATE_CONFIG,
    "E7": ELU_ACTIVATION_CONFIG,
}

PRIMARY_EXPERIMENT_IDS: Tuple[str, ...] = ("E1", "E2", "E3", "E4", "E5")
RESERVE_EXPERIMENT_IDS: Tuple[str, ...] = ("E6", "E7")
