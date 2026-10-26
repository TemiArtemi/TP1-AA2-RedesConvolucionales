import torch.nn as nn
from torchvision import models

from src.model import build_activation

HEAD_HIDDEN_UNITS: int = 128
HEAD_DROPOUT_PROBABILITY: float = 0.5
PRETRAINED_WEIGHTS: models.ResNet18_Weights = (
    models.ResNet18_Weights.IMAGENET1K_V1
)


def build_resnet18_with_head(
    n_classes: int,
    activation: str,
    unfreeze_last_block: bool,
) -> nn.Module:
    model: nn.Module = models.resnet18(weights=PRETRAINED_WEIGHTS)

    for parameter in model.parameters():
        parameter.requires_grad = False

    if unfreeze_last_block:
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True

    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, HEAD_HIDDEN_UNITS),
        build_activation(activation),
        nn.Dropout(HEAD_DROPOUT_PROBABILITY),
        nn.Linear(HEAD_HIDDEN_UNITS, n_classes),
    )

    return model


def build_head_only_model(n_classes: int, activation: str) -> nn.Module:
    return build_resnet18_with_head(
        n_classes=n_classes,
        activation=activation,
        unfreeze_last_block=False,
    )


def build_head_and_last_layer_model(n_classes: int, activation: str) -> nn.Module:
    return build_resnet18_with_head(
        n_classes=n_classes,
        activation=activation,
        unfreeze_last_block=True,
    )
