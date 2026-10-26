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
    fc_hidden_layer_count: int = 1
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
        fc_hidden_layer_count=1,
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
        fc_hidden_layer_count=1,
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

MORE_FILTERS_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E5",
    architecture=replace(
        NORMALIZED_CONFIG.architecture,
        conv_channels=(32, 64, 128),
    ),
)

MORE_NEURONS_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E6",
    architecture=replace(NORMALIZED_CONFIG.architecture, fc_hidden_units=256),
)

LOWER_LEARNING_RATE_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E7",
    training=replace(NORMALIZED_CONFIG.training, learning_rate=0.0005),
)

LARGER_BATCH_SIZE_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E8",
    training=replace(NORMALIZED_CONFIG.training, batch_size=64),
)

ELU_ACTIVATION_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E9",
    architecture=replace(NORMALIZED_CONFIG.architecture, activation="elu"),
)

DEEPER_CONVOLUTION_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E10",
    architecture=replace(
        NORMALIZED_CONFIG.architecture,
        conv_channels=(16, 32, 64, 128),
    ),
)

DEEPER_HEAD_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E11",
    architecture=replace(NORMALIZED_CONFIG.architecture, fc_hidden_layer_count=2),
)

LONGER_TRAINING_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E12",
    training=replace(
        NORMALIZED_CONFIG.training,
        num_epochs=100,
        patience=15,
    ),
)

ASSEMBLED_CONFIG: ExperimentConfig = replace(
    NORMALIZED_CONFIG,
    experiment_id="E13",
    architecture=replace(NORMALIZED_CONFIG.architecture, activation="elu"),
    training=replace(
        NORMALIZED_CONFIG.training,
        learning_rate=0.0005,
        batch_size=64,
        num_epochs=100,
        patience=15,
    ),
)

CONFIG_REGISTRY: Dict[str, ExperimentConfig] = {
    "E1": BASELINE_CONFIG,
    "E2": AUGMENTED_CONFIG,
    "E3": NORMALIZED_CONFIG,
    "E4": HIGH_RESOLUTION_CONFIG,
    "E5": MORE_FILTERS_CONFIG,
    "E6": MORE_NEURONS_CONFIG,
    "E7": LOWER_LEARNING_RATE_CONFIG,
    "E8": LARGER_BATCH_SIZE_CONFIG,
    "E9": ELU_ACTIVATION_CONFIG,
    "E10": DEEPER_CONVOLUTION_CONFIG,
    "E11": DEEPER_HEAD_CONFIG,
    "E12": LONGER_TRAINING_CONFIG,
    "E13": ASSEMBLED_CONFIG,
}

PHASE_ONE_IDS: Tuple[str, ...] = ("E1", "E2", "E3", "E4")
PHASE_TWO_IDS: Tuple[str, ...] = ("E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12")
ASSEMBLY_IDS: Tuple[str, ...] = ("E13",)
