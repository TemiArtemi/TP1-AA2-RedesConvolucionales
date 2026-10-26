import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.early_stopping import EarlyStopping

DEFAULT_LEARNING_RATE: float = 0.001
DEFAULT_NUM_EPOCHS: int = 50
DEFAULT_PATIENCE: int = 10
DEFAULT_MIN_DELTA: float = 0.0
DEFAULT_DEVICE: torch.device = torch.device("cpu")


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()

    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (outputs.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()

    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(model: nn.Module, checkpoint_path: Path) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


def train_model(
    model: nn.Module,
    dataloaders: Dict[str, DataLoader],
    device: torch.device = DEFAULT_DEVICE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    num_epochs: int = DEFAULT_NUM_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    checkpoint_path: Optional[Path] = None,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    assert num_epochs > 0, "num_epochs must be positive"
    assert learning_rate > 0.0, "learning_rate must be positive"
    assert "train" in dataloaders, "dataloaders must contain a 'train' split"
    assert "val" in dataloaders, "dataloaders must contain a 'val' split"

    trainable_parameters: List[nn.Parameter] = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    assert trainable_parameters, "model has no trainable parameters"

    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer: torch.optim.Optimizer = torch.optim.Adam(
        trainable_parameters,
        lr=learning_rate,
    )
    early_stopping: EarlyStopping = EarlyStopping(
        patience=patience,
        min_delta=min_delta,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    started_at: float = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        epoch_started_at: float = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = validate_one_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)

        improved: bool = early_stopping.step(val_loss, model)

        if improved and checkpoint_path is not None:
            save_checkpoint(model, checkpoint_path)

        if verbose:
            elapsed: float = time.perf_counter() - epoch_started_at
            marker: str = "  * best" if improved else ""
            print(
                f"epoch {epoch:3d}/{num_epochs} ({elapsed:5.1f}s) | "
                f"train loss {train_loss:.4f} acc {train_accuracy:.4f} | "
                f"val loss {val_loss:.4f} acc {val_accuracy:.4f}"
                f"{marker}"
            )

        if early_stopping.should_stop:
            if verbose:
                print(f"early stopping triggered at epoch {epoch}")
            break

    model.load_state_dict(early_stopping.best_state_dict)

    if verbose:
        total_elapsed: float = time.perf_counter() - started_at
        epochs_per_second: float = early_stopping.current_epoch / max(total_elapsed, 1e-9)
        print(
            f"restored epoch {early_stopping.best_epoch} | "
            f"best val loss {early_stopping.best_validation_loss:.4f} | "
            f"{early_stopping.current_epoch} epochs in {total_elapsed:.1f}s "
            f"({epochs_per_second:.2f} epochs/s)"
        )

    return history
