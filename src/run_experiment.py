import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.config import CONFIG_REGISTRY, ExperimentConfig
from src.dataset import build_dataloaders, discover_class_names, set_random_seed
from src.evaluate import EvaluationResult, evaluate_model, print_evaluation_summary
from src.model import ConvNet
from src.plotting import plot_confusion_matrix, plot_training_history
from src.train import train_model

DATA_DIRECTORY: Path = Path("data")
REPORTS_DIRECTORY: Path = Path("reports")
CHECKPOINTS_DIRECTORY: Path = Path("checkpoints")
DEVICE: torch.device = torch.device("cpu")


def find_best_epoch(history: Dict[str, List[float]]) -> int:
    validation_losses: List[float] = history["val_loss"]
    best_index: int = min(
        range(len(validation_losses)),
        key=lambda index: validation_losses[index],
    )
    return best_index + 1


def format_config_markdown(
    config: ExperimentConfig,
    class_names: List[str],
) -> str:
    augmentation = config.data.augmentation
    augmentation_text: str = (
        "ninguna" if augmentation is None else str(augmentation)
    )
    channels_text: str = " - ".join(str(value) for value in config.architecture.conv_channels)

    lines: List[str] = [
        f"## {config.experiment_id}",
        "",
        "### Configuración",
        "",
        "**Datos**",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Clases | {len(class_names)} ({', '.join(class_names)}) |",
        f"| Resolución | {config.data.image_size}×{config.data.image_size} |",
        f"| Normalización | {'sí' if config.data.normalize else 'ninguna'} |",
        f"| Augmentación | {augmentation_text} |",
        f"| Batch size | {config.training.batch_size} |",
        f"| Semilla | {config.random_seed} |",
        "",
        "**Arquitectura**",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Canales convolucionales | {channels_text} |",
        f"| Kernel | {config.architecture.kernel_size} |",
        f"| Neuronas FC | {config.architecture.fc_hidden_units} |",
        f"| Dropout | {config.architecture.dropout_probability} |",
        f"| Activación | {config.architecture.activation} |",
        "",
        "**Entrenamiento**",
        "",
        "| Parámetro | Valor |",
        "|---|---|",
        f"| Learning rate | {config.training.learning_rate} |",
        f"| Épocas (techo) | {config.training.num_epochs} |",
        f"| Paciencia | {config.training.patience} |",
        f"| min_delta | {config.training.min_delta} |",
        f"| Optimizador | Adam |",
        f"| Pérdida | CrossEntropyLoss |",
        "",
    ]
    return "\n".join(lines)


def format_evaluation_markdown(
    result: EvaluationResult,
    split_name: str,
    class_names: List[str],
) -> str:
    lines: List[str] = [
        f"### Resultados — {split_name}",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| loss | {result.loss:.4f} |",
        f"| accuracy | {result.accuracy:.4f} |",
        f"| precision macro | {result.precision_macro:.4f} |",
        f"| precision weighted | {result.precision_weighted:.4f} |",
        f"| recall macro | {result.recall_macro:.4f} |",
        f"| recall weighted | {result.recall_weighted:.4f} |",
        f"| f1 macro | {result.f1_macro:.4f} |",
        f"| f1 weighted | {result.f1_weighted:.4f} |",
        "",
        f"**Matriz de confusión — {split_name}** (filas = clase real, columnas = predicha)",
        "",
        "```text",
    ]

    header: str = f"{'':<12}" + "".join(f"{name[:9]:>10}" for name in class_names)
    lines.append(header)
    for index, class_name in enumerate(class_names):
        row: str = f"{class_name:<12}"
        row += "".join(
            f"{int(value):>10}" for value in result.confusion_matrix[index]
        )
        lines.append(row)
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def run_experiment(config: ExperimentConfig) -> Dict[str, float]:
    set_random_seed(config.random_seed)

    class_names: List[str] = discover_class_names(DATA_DIRECTORY)
    dataloaders: Dict[str, object] = build_dataloaders(
        data_directory=DATA_DIRECTORY,
        data_config=config.data,
        batch_size=config.training.batch_size,
        random_seed=config.random_seed,
    )

    model: ConvNet = ConvNet(
        architecture=config.architecture,
        image_size=config.data.image_size,
        n_classes=len(class_names),
    )

    checkpoint_path: Path = CHECKPOINTS_DIRECTORY / f"{config.experiment_id}_best.pt"

    history: Dict[str, List[float]] = train_model(
        model=model,
        dataloaders=dataloaders,
        device=DEVICE,
        learning_rate=config.training.learning_rate,
        num_epochs=config.training.num_epochs,
        patience=config.training.patience,
        min_delta=config.training.min_delta,
        checkpoint_path=checkpoint_path,
    )

    best_epoch: int = find_best_epoch(history)

    REPORTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    history_path: Path = REPORTS_DIRECTORY / f"{config.experiment_id}_history.json"
    with history_path.open("w", encoding="utf-8") as history_file:
        json.dump(history, history_file, indent=2)

    plot_training_history(
        history=history,
        best_epoch=best_epoch,
        experiment_id=config.experiment_id,
        output_directory=REPORTS_DIRECTORY,
    )

    criterion: nn.Module = nn.CrossEntropyLoss()

    val_result: EvaluationResult = evaluate_model(
        model=model,
        dataloader=dataloaders["val"],
        criterion=criterion,
        device=DEVICE,
        class_names=class_names,
    )
    test_result: EvaluationResult = evaluate_model(
        model=model,
        dataloader=dataloaders["test"],
        criterion=criterion,
        device=DEVICE,
        class_names=class_names,
    )

    plot_confusion_matrix(
        matrix=val_result.confusion_matrix,
        class_names=class_names,
        experiment_id=config.experiment_id,
        split_name="val",
        output_directory=REPORTS_DIRECTORY,
    )
    plot_confusion_matrix(
        matrix=test_result.confusion_matrix,
        class_names=class_names,
        experiment_id=config.experiment_id,
        split_name="test",
        output_directory=REPORTS_DIRECTORY,
    )

    print()
    print(f"=== {config.experiment_id} — VALIDACIÓN ===")
    print_evaluation_summary(val_result, class_names)
    print()
    print(f"=== {config.experiment_id} — TEST ===")
    print_evaluation_summary(test_result, class_names)

    summary_lines: List[str] = [
        format_config_markdown(config, class_names),
        "### Resultados",
        "",
        "| Métrica | Valor |",
        "|---|---:|",
        f"| best_epoch | {best_epoch} |",
        f"| best_val_loss | {history['val_loss'][best_epoch - 1]:.4f} |",
        f"| épocas ejecutadas | {len(history['val_loss'])} |",
        f"| val accuracy | {val_result.accuracy:.4f} |",
        f"| **test accuracy** | **{test_result.accuracy:.4f}** |",
        "",
        format_evaluation_markdown(val_result, "validación", class_names),
        format_evaluation_markdown(test_result, "test", class_names),
    ]
    summary_path: Path = REPORTS_DIRECTORY / f"{config.experiment_id}_summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print()
    print(f"resumen escrito en {summary_path}")

    return {
        "best_epoch": float(best_epoch),
        "val_accuracy": val_result.accuracy,
        "test_accuracy": test_result.accuracy,
    }


def parse_arguments() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        choices=sorted(CONFIG_REGISTRY),
        help="identificador de la configuración a ejecutar",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments: argparse.Namespace = parse_arguments()
    run_experiment(CONFIG_REGISTRY[arguments.config])
