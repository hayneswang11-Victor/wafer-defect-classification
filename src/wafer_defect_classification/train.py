from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from wafer_defect_classification.evaluate import compute_classification_metrics
from wafer_defect_classification.features import extract_feature_table, get_feature_names
from wafer_defect_classification.visualization import (
    save_confusion_matrix_figure,
    save_logistic_regression_coefficients,
    save_random_forest_feature_importance,
)

SUPPORTED_MAP_TYPES = {"wafer", "dram"}
SUPPORTED_MODELS = {"logistic_regression", "random_forest"}
DEFAULT_DATA_PATHS = {
    "wafer": Path("data/synthetic/wafer_maps/wafer_maps.npz"),
    "dram": Path("data/synthetic/dram_fail_bitmaps/dram_fail_bitmaps.npz"),
}


@dataclass(frozen=True)
class TrainingResult:
    map_type: str
    model_name: str
    model_file: Path
    metrics_file: Path
    split_file: Path
    confusion_matrix_figure: Path
    model_diagnostic_figure: Path | None
    metrics: dict[str, Any]


def train_model(
    map_type: str,
    model_name: str,
    data_path: str | Path | None = None,
    model_dir: str | Path = "models",
    metrics_dir: str | Path = "reports/metrics",
    figure_dir: str | Path = "reports/figures",
    split_dir: str | Path = "data/splits",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    n_estimators: int = 200,
) -> TrainingResult:
    """Train one classical synthetic defect classifier for one map type."""
    normalized_map_type = _validate_map_type(map_type)
    normalized_model_name = _validate_model_name(model_name)

    dataset = _load_dataset(data_path or DEFAULT_DATA_PATHS[normalized_map_type])
    maps = dataset["maps"]
    labels = np.asarray(dataset["labels"], dtype=int)
    class_names = [str(name) for name in dataset["class_names"].tolist()]

    feature_table = extract_feature_table(
        maps=maps,
        labels=labels,
        class_names=class_names,
        map_type=normalized_map_type,
    )
    feature_names = get_feature_names()
    x = feature_table[feature_names].to_numpy(dtype=float)
    y = labels

    split_indices = create_stratified_splits(
        labels=y,
        test_size=test_size,
        validation_size=validation_size,
        random_seed=random_seed,
    )
    model = build_model(
        model_name=normalized_model_name,
        random_seed=random_seed,
        n_estimators=n_estimators,
    )

    train_indices = np.asarray(split_indices["train"], dtype=int)
    validation_indices = np.asarray(split_indices["validation"], dtype=int)
    test_indices = np.asarray(split_indices["test"], dtype=int)

    model.fit(x[train_indices], y[train_indices])
    validation_predictions = model.predict(x[validation_indices])
    test_predictions = model.predict(x[test_indices])

    validation_metrics = compute_classification_metrics(
        y_true=y[validation_indices],
        y_pred=validation_predictions,
        class_names=class_names,
    )
    test_metrics = compute_classification_metrics(
        y_true=y[test_indices],
        y_pred=test_predictions,
        class_names=class_names,
    )

    model_dir_path = Path(model_dir)
    metrics_dir_path = Path(metrics_dir)
    figure_dir_path = Path(figure_dir)
    split_dir_path = Path(split_dir)
    for path in (model_dir_path, metrics_dir_path, figure_dir_path, split_dir_path):
        path.mkdir(parents=True, exist_ok=True)

    model_file = model_dir_path / f"{normalized_map_type}_{normalized_model_name}.joblib"
    metrics_file = metrics_dir_path / f"{normalized_map_type}_{normalized_model_name}_metrics.json"
    split_file = split_dir_path / f"{normalized_map_type}_splits.json"
    confusion_matrix_figure = (
        figure_dir_path / f"{normalized_map_type}_{normalized_model_name}_confusion_matrix.png"
    )

    artifact = {
        "model": model,
        "map_type": normalized_map_type,
        "model_name": normalized_model_name,
        "feature_names": feature_names,
        "class_names": class_names,
        "synthetic_data_only": True,
        "interpretation": "Synthetic prototype validation only.",
    }
    joblib.dump(artifact, model_file)

    split_payload = {
        "map_type": normalized_map_type,
        "random_seed": random_seed,
        "requested_test_size": test_size,
        "requested_validation_size_relative_to_remaining": validation_size,
        "train": split_indices["train"],
        "validation": split_indices["validation"],
        "test": split_indices["test"],
    }
    _write_json(split_file, split_payload)

    metrics_payload = {
        "map_type": normalized_map_type,
        "model_name": normalized_model_name,
        "synthetic_data_only": True,
        "interpretation": (
            "Synthetic validation metrics for a prototype; not real semiconductor "
            "manufacturing performance."
        ),
        "feature_names": feature_names,
        "class_names": class_names,
        "validation": validation_metrics,
        "test": test_metrics,
    }
    _write_json(metrics_file, metrics_payload)

    save_confusion_matrix_figure(
        confusion_matrix=test_metrics["confusion_matrix"],
        class_names=class_names,
        output_path=confusion_matrix_figure,
        title=f"{normalized_map_type} {normalized_model_name} synthetic validation",
    )

    diagnostic_figure: Path | None = None
    if normalized_model_name == "random_forest":
        diagnostic_figure = (
            figure_dir_path / f"{normalized_map_type}_random_forest_feature_importance.png"
        )
        save_random_forest_feature_importance(
            model=model,
            feature_names=feature_names,
            output_path=diagnostic_figure,
        )
    elif normalized_model_name == "logistic_regression":
        diagnostic_figure = (
            figure_dir_path / f"{normalized_map_type}_logistic_regression_coefficients.png"
        )
        save_logistic_regression_coefficients(
            model=model,
            feature_names=feature_names,
            output_path=diagnostic_figure,
        )

    return TrainingResult(
        map_type=normalized_map_type,
        model_name=normalized_model_name,
        model_file=model_file,
        metrics_file=metrics_file,
        split_file=split_file,
        confusion_matrix_figure=confusion_matrix_figure,
        model_diagnostic_figure=diagnostic_figure,
        metrics=metrics_payload,
    )


def build_model(
    model_name: str,
    random_seed: int = 42,
    n_estimators: int = 200,
) -> Pipeline:
    normalized_model_name = _validate_model_name(model_name)
    if normalized_model_name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=random_seed,
                    ),
                ),
            ]
        )

    return Pipeline(
        steps=[
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_seed,
                    class_weight="balanced",
                ),
            )
        ]
    )


def create_stratified_splits(
    labels: np.ndarray,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, list[int]]:
    labels = np.asarray(labels, dtype=int)
    indices = np.arange(len(labels))
    unique_labels, counts = np.unique(labels, return_counts=True)
    class_count = len(unique_labels)
    if class_count < 2:
        raise ValueError("Training requires at least two classes.")
    if np.any(counts < 3):
        raise ValueError(
            "Stratified train/validation/test split requires at least 3 samples per class."
        )

    test_count = _split_count(
        total_count=len(labels),
        class_count=class_count,
        fraction=test_size,
        reserve_per_class=2,
    )
    train_validation_indices, test_indices = train_test_split(
        indices,
        test_size=test_count,
        random_state=random_seed,
        stratify=labels,
    )

    train_validation_labels = labels[train_validation_indices]
    validation_count = _split_count(
        total_count=len(train_validation_indices),
        class_count=class_count,
        fraction=validation_size,
        reserve_per_class=1,
    )
    train_indices, validation_indices = train_test_split(
        train_validation_indices,
        test_size=validation_count,
        random_state=random_seed,
        stratify=train_validation_labels,
    )

    return {
        "train": sorted(int(index) for index in train_indices),
        "validation": sorted(int(index) for index in validation_indices),
        "test": sorted(int(index) for index in test_indices),
    }


def _split_count(
    total_count: int,
    class_count: int,
    fraction: float,
    reserve_per_class: int,
) -> int:
    requested = max(int(math.ceil(total_count * fraction)), class_count)
    maximum = total_count - reserve_per_class * class_count
    if maximum < class_count:
        raise ValueError("Not enough samples for the requested stratified split.")
    return min(requested, maximum)


def _load_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Synthetic dataset not found: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as data:
        return {
            "maps": data["maps"],
            "labels": data["labels"],
            "class_names": data["class_names"],
        }


def _validate_map_type(map_type: str) -> str:
    normalized = map_type.lower().strip()
    if normalized not in SUPPORTED_MAP_TYPES:
        raise ValueError(
            f"Unsupported map_type {map_type!r}. Expected one of: {sorted(SUPPORTED_MAP_TYPES)}"
        )
    return normalized


def _validate_model_name(model_name: str) -> str:
    normalized = model_name.lower().strip()
    if normalized not in SUPPORTED_MODELS:
        raise ValueError(
            f"Unsupported model_name {model_name!r}. Expected one of: {sorted(SUPPORTED_MODELS)}"
        )
    return normalized


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
