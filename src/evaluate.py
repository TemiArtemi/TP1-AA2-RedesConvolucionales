from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

REPORT_DIGITS: int = 4


@dataclass
class EvaluationResult:
    loss: float
    accuracy: float
    predictions: np.ndarray
    labels: np.ndarray
    confusion_matrix: np.ndarray
    report: str
    precision_macro: float
    recall_macro: float
    f1_macro: float
    precision_weighted: float
    recall_weighted: float
    f1_weighted: float


def run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()

    total_loss: float = 0.0
    total_samples: int = 0
    all_predictions: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_samples += labels.size(0)
            all_predictions.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())

    predictions: np.ndarray = torch.cat(all_predictions).numpy()
    targets: np.ndarray = torch.cat(all_labels).numpy()

    return total_loss / total_samples, predictions, targets


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: List[str],
) -> EvaluationResult:
    loss, predictions, targets = run_inference(model, dataloader, criterion, device)

    n_classes: int = len(class_names)
    accuracy: float = float((predictions == targets).mean())
    matrix: np.ndarray = confusion_matrix(
        targets,
        predictions,
        labels=list(range(n_classes)),
    )
    report: str = classification_report(
        targets,
        predictions,
        labels=list(range(n_classes)),
        target_names=class_names,
        digits=REPORT_DIGITS,
        zero_division=0,
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        targets,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return EvaluationResult(
        loss=loss,
        accuracy=accuracy,
        predictions=predictions,
        labels=targets,
        confusion_matrix=matrix,
        report=report,
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        precision_weighted=float(precision_weighted),
        recall_weighted=float(recall_weighted),
        f1_weighted=float(f1_weighted),
    )


def print_evaluation_summary(result: EvaluationResult, class_names: List[str]) -> None:
    print(f"loss              {result.loss:.4f}")
    print(f"accuracy          {result.accuracy:.4f}")
    print(
        f"precision macro   {result.precision_macro:.4f}  "
        f"weighted {result.precision_weighted:.4f}"
    )
    print(
        f"recall    macro   {result.recall_macro:.4f}  "
        f"weighted {result.recall_weighted:.4f}"
    )
    print(
        f"f1-score  macro   {result.f1_macro:.4f}  "
        f"weighted {result.f1_weighted:.4f}"
    )
    print()
    print(result.report)
    print("confusion matrix (rows = true, columns = predicted)")
    print(f"{'':<12}" + "".join(f"{name[:9]:>10}" for name in class_names))
    for index, class_name in enumerate(class_names):
        row: str = f"{class_name:<12}"
        row += "".join(f"{value:>10}" for value in result.confusion_matrix[index])
        print(row)
