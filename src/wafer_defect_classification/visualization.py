from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np


def save_sample_grid(
    maps: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    map_type: str,
    output_path: str | Path,
) -> Path:
    """Save one representative sample grid for a map type."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(3, 3, figsize=(9, 9), constrained_layout=True)
    flat_axes = axes.ravel()

    for class_index, class_name in enumerate(class_names):
        axis = flat_axes[class_index]
        sample_index = int(np.flatnonzero(labels == class_index)[0])
        sample = maps[sample_index]

        if map_type == "wafer":
            cmap = ListedColormap(["#d9d9d9", "#ffffff", "#d62728"])
            norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)
            axis.imshow(sample, cmap=cmap, norm=norm, interpolation="nearest")
        elif map_type == "dram":
            cmap = ListedColormap(["#111111", "#ffd447"])
            norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
            axis.imshow(sample, cmap=cmap, norm=norm, interpolation="nearest")
        else:
            raise ValueError(f"Unsupported map type for visualization: {map_type}")

        axis.set_title(class_name, fontsize=9)
        axis.set_xticks([])
        axis.set_yticks([])

    figure.suptitle(
        f"Synthetic {map_type.upper()} defect pattern examples",
        fontsize=12,
    )
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    return output_file


def save_confusion_matrix_figure(
    confusion_matrix: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a compact matplotlib confusion matrix figure."""
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(confusion_matrix, dtype=float)

    figure, axis = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = axis.imshow(matrix, cmap="Blues", interpolation="nearest")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_title(title)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_xticks(np.arange(len(class_names)))
    axis.set_yticks(np.arange(len(class_names)))
    axis.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    axis.set_yticklabels(class_names, fontsize=8)

    max_value = matrix.max() if matrix.size else 0.0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = int(matrix[row_index, column_index])
            color = "white" if matrix[row_index, column_index] > threshold else "black"
            axis.text(column_index, row_index, value, ha="center", va="center", color=color, fontsize=7)

    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    return output_file


def save_random_forest_feature_importance(
    model,
    feature_names: list[str],
    output_path: str | Path,
    top_n: int = 20,
) -> Path:
    """Save a feature-importance bar chart for a RandomForest pipeline."""
    classifier = _classifier_from_model(model)
    importances = np.asarray(classifier.feature_importances_, dtype=float)
    return _save_ranked_bar_chart(
        values=importances,
        names=feature_names,
        output_path=output_path,
        title="Synthetic validation random forest feature importance",
        x_label="Importance",
        top_n=top_n,
    )


def save_logistic_regression_coefficients(
    model,
    feature_names: list[str],
    output_path: str | Path,
    top_n: int = 20,
) -> Path:
    """Save a mean absolute coefficient bar chart for a logistic pipeline."""
    classifier = _classifier_from_model(model)
    coefficients = np.asarray(classifier.coef_, dtype=float)
    magnitudes = np.mean(np.abs(coefficients), axis=0)
    return _save_ranked_bar_chart(
        values=magnitudes,
        names=feature_names,
        output_path=output_path,
        title="Synthetic validation logistic regression coefficient magnitude",
        x_label="Mean absolute coefficient",
        top_n=top_n,
    )


def _save_ranked_bar_chart(
    values: np.ndarray,
    names: list[str],
    output_path: str | Path,
    title: str,
    x_label: str,
    top_n: int,
) -> Path:
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if values.size == 0:
        ranked_indices = np.asarray([], dtype=int)
    else:
        ranked_indices = np.argsort(values)[-top_n:]
    ranked_values = values[ranked_indices]
    ranked_names = [names[index] for index in ranked_indices]

    figure_height = max(4.0, 0.35 * max(1, len(ranked_names)))
    figure, axis = plt.subplots(figsize=(8, figure_height), constrained_layout=True)
    axis.barh(np.arange(len(ranked_names)), ranked_values, color="#4c78a8")
    axis.set_yticks(np.arange(len(ranked_names)))
    axis.set_yticklabels(ranked_names, fontsize=8)
    axis.set_xlabel(x_label)
    axis.set_title(title)
    figure.savefig(output_file, dpi=160)
    plt.close(figure)
    return output_file


def _classifier_from_model(model):
    if hasattr(model, "named_steps"):
        return model.named_steps["classifier"]
    return model
