from typing import Callable, Dict

import torch
import torch.nn as nn

from src.config import ArchitectureConfig

ACTIVATION_BUILDERS: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}
IMAGE_CHANNELS: int = 3


def build_activation(activation_name: str) -> nn.Module:
    assert activation_name in ACTIVATION_BUILDERS, (
        f"unknown activation: {activation_name}"
    )
    return ACTIVATION_BUILDERS[activation_name]()


class ConvNet(nn.Module):
    def __init__(
        self,
        architecture: ArchitectureConfig,
        image_size: int,
        n_classes: int,
    ) -> None:
        super(ConvNet, self).__init__()
        assert n_classes > 0, "n_classes must be positive"
        assert len(architecture.conv_channels) > 0, "at least one conv layer required"
        assert architecture.fc_hidden_layer_count > 0, (
            "fc_hidden_layer_count must be positive"
        )
        assert image_size % (2 ** len(architecture.conv_channels)) == 0, (
            "image_size must be divisible by the total pooling factor"
        )

        padding: int = architecture.kernel_size // 2

        self.conv_layers: nn.ModuleList = nn.ModuleList()
        in_channels: int = IMAGE_CHANNELS
        for out_channels in architecture.conv_channels:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=architecture.kernel_size,
                    stride=1,
                    padding=padding,
                )
            )
            in_channels = out_channels

        self.pool: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.activation: nn.Module = build_activation(architecture.activation)

        spatial_size: int = image_size // (2 ** len(architecture.conv_channels))
        flattened_features: int = in_channels * spatial_size * spatial_size

        self.dropout: nn.Dropout = nn.Dropout(p=architecture.dropout_probability)

        self.fc_layers: nn.ModuleList = nn.ModuleList()
        in_features: int = flattened_features
        for _ in range(architecture.fc_hidden_layer_count):
            self.fc_layers.append(
                nn.Linear(in_features, architecture.fc_hidden_units)
            )
            in_features = architecture.fc_hidden_units
        self.classifier: nn.Linear = nn.Linear(in_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for conv_layer in self.conv_layers:
            x = self.pool(self.activation(conv_layer(x)))
        x = x.flatten(1)
        for fc_layer in self.fc_layers:
            x = self.dropout(self.activation(fc_layer(x)))
        return self.classifier(x)
