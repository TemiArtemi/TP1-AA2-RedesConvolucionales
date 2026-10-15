from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay

FIGURE_DPI: int = 150


def plot_training_history(
    history: Dict[str, List[float]],
    best_epoch: int,
    experiment_id: str,
    output_directory: Path,
) -> Path:
    epochs: List[int] = list(range(1, len(history["train_loss"]) + 1))

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="val")
    axes[0].axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"best epoch ({best_epoch})",
    )
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="val")
    axes[1].axvline(
        best_epoch,
        color="red",
        linestyle="--",
        label=f"best epoch ({best_epoch})",
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(experiment_id)
    figure.tight_layout()

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_directory / f"{experiment_id}_curves.png"
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return output_path


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: List[str],
    experiment_id: str,
    split_name: str,
    output_directory: Path,
) -> Path:
    display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=class_names,
    )

    figure, axis = plt.subplots(figsize=(8, 8))
    display.plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        xticks_rotation=45,
        values_format="d",
    )
    axis.set_title(f"{experiment_id} - {split_name}")
    figure.tight_layout()

    output_directory.mkdir(parents=True, exist_ok=True)
    output_path: Path = output_directory / (
        f"{experiment_id}_{split_name}_confusion.png"
    )
    figure.savefig(output_path, dpi=FIGURE_DPI)
    plt.close(figure)

    return output_path
