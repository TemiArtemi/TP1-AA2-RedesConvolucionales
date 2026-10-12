import copy
from typing import Dict

import torch
import torch.nn as nn

DEFAULT_PATIENCE: int = 10
DEFAULT_MIN_DELTA: float = 0.0


class EarlyStopping:
    def __init__(
        self,
        patience: int = DEFAULT_PATIENCE,
        min_delta: float = DEFAULT_MIN_DELTA,
    ) -> None:
        assert patience > 0, "patience must be positive"
        assert min_delta >= 0.0, "min_delta must be non-negative"

        self.patience: int = patience
        self.min_delta: float = min_delta
        self.best_validation_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_state_dict: Dict[str, torch.Tensor] = {}
        self.epochs_without_improvement: int = 0
        self.current_epoch: int = 0

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience

    def step(self, validation_loss: float, model: nn.Module) -> bool:
        self.current_epoch += 1
        improved: bool = validation_loss < self.best_validation_loss - self.min_delta

        if improved:
            self.best_validation_loss = validation_loss
            self.best_epoch = self.current_epoch
            self.epochs_without_improvement = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
        else:
            self.epochs_without_improvement += 1

        return improved

    def restore_best_weights(self, model: nn.Module) -> None:
        assert len(self.best_state_dict) > 0, "no best weights recorded yet"
        model.load_state_dict(self.best_state_dict)
