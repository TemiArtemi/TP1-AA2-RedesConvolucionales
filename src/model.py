import torch
import torch.nn as nn
import torch.nn.functional as F

FLATTENED_FEATURES: int = 64 * 12 * 12
CONV_CHANNELS: int = 64
HIDDEN_UNITS: int = 128
DROPOUT_PROBABILITY: float = 0.25
IMAGE_CHANNELS: int = 3


class ConvNet(nn.Module):
    def __init__(self, n_classes: int) -> None:
        super(ConvNet, self).__init__()
        assert n_classes > 0, "n_classes must be positive"

        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=IMAGE_CHANNELS,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.conv3: nn.Conv2d = nn.Conv2d(
            in_channels=32,
            out_channels=CONV_CHANNELS,
            kernel_size=5,
            stride=1,
            padding=2,
        )
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.dropout1: nn.Dropout = nn.Dropout(p=DROPOUT_PROBABILITY)
        self.fc1: nn.Linear = nn.Linear(FLATTENED_FEATURES, HIDDEN_UNITS)
        self.fc2: nn.Linear = nn.Linear(HIDDEN_UNITS, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        x = self.pool1(F.relu(self.conv3(x)))
        x = x.view(-1, FLATTENED_FEATURES)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        return self.fc2(x)
