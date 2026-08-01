from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from wafer_defect_classification.features import extract_features


@dataclass(frozen=True)
class PredictionResult:
    model_path: Path
    input_path: Path
    output_path: Path | None
    map_type: str
    model_name: str
    sample_count: int
    predictions: list[dict[str, Any]]


def predict_file(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path | None = None,
    array_key: str = "maps",
) -> PredictionResult:
    """Load a saved model artifact and classify one map or a batch of maps.

    Supported input formats:
    - ``.npy``: one 2D map or a 3D batch ``(n_samples, height, width)``
    - ``.npz``: an array stored under ``array_key`` (default: ``maps``)
    - ``.csv``: one 2D numeric map

    The model artifact determines whether the input is interpreted as a wafer
    map or a DRAM fail bit map. This inference path is for the synthetic
    prototype only and does not imply real-fab validation.
    """
    model_file = Path(model_path)
    input_file = Path(input_path)
    output_file = Path(output_path) if output_path is not None else None

    artifact = _load_artifact(model_file)
    maps = load_maps(input_file, array_key=array_key)
    feature_frame = _build_feature_frame(
        maps=maps,
        map_type=artifact["map_type"],
        feature_names=artifact["feature_names"],
    )

    model = artifact["model"]
    feature_matrix = feature_frame.to_numpy(dtype=float)
    labels = np.asarray(model.predict(feature_matrix), dtype=int)
    probabilities = _predict_probabilities(model, feature_matrix)

    class_names = artifact["class_names"]
    rows: list[dict[str, Any]] = []
    for sample_index, label_value in enumerate(labels.tolist()):
        class_name = _class_name(class_names, label_value)
        row: dict[str, Any] = {
            "sample_index": sample_index,
            "predicted_label": label_value,
            "predicted_class": class_name,
        }

        if probabilities is not None:
            probability_row = probabilities[sample_index]
            probability_map = {
                _class_name(class_names, class_index): float(probability)
                for class_index, probability in enumerate(probability_row)
            }
            row["confidence"] = float(np.max(probability_row))
            row["probabilities"] = probability_map
        else:
            row["confidence"] = None
            row["probabilities"] = None

        rows.append(row)

    result = PredictionResult(
        model_path=model_file,
        input_path=input_file,
        output_path=output_file,
        map_type=artifact["map_type"],
        model_name=artifact["model_name"],
        sample_count=len(rows),
        predictions=rows,
    )

    if output_file is not None:
        write_predictions(result, output_file)

    return result


def load_maps(input_path: str | Path, array_key: str = "maps") -> np.ndarray:
    """Load one 2D map or a 3D batch from a supported file."""
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Prediction input not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npy":
        maps = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            if array_key not in data.files:
                raise KeyError(
                    f"Array key {array_key!r} not found in {path}. "
                    f"Available keys: {sorted(data.files)}"
                )
            maps = data[array_key]
    elif suffix == ".csv":
        maps = np.loadtxt(path, delimiter=",")
    else:
        raise ValueError(
            f"Unsupported prediction input format {suffix!r}. "
            "Expected .npy, .npz, or .csv."
        )

    array = np.asarray(maps)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(
            "Prediction input must be one 2D map or a 3D batch "
            f"(n_samples, height, width); got shape {array.shape}."
        )
    if array.shape[0] == 0:
        raise ValueError("Prediction input contains no maps.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Prediction input must be numeric; got dtype {array.dtype}.")

    return array


def write_predictions(result: PredictionResult, output_path: str | Path) -> Path:
    """Write predictions as JSON or CSV, selected by the output suffix."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    if suffix == ".json":
        payload = {
            "model_path": str(result.model_path),
            "input_path": str(result.input_path),
            "map_type": result.map_type,
            "model_name": result.model_name,
            "synthetic_data_only": True,
            "interpretation": (
                "Synthetic prototype inference only; not validated on real "
                "semiconductor manufacturing data."
            ),
            "sample_count": result.sample_count,
            "predictions": result.predictions,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    if suffix == ".csv":
        class_names = _ordered_probability_names(result.predictions)
        fieldnames = [
            "sample_index",
            "predicted_label",
            "predicted_class",
            "confidence",
            *[f"probability_{name}" for name in class_names],
        ]
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for prediction in result.predictions:
                row = {
                    "sample_index": prediction["sample_index"],
                    "predicted_label": prediction["predicted_label"],
                    "predicted_class": prediction["predicted_class"],
                    "confidence": prediction["confidence"],
                }
                probabilities = prediction.get("probabilities") or {}
                for name in class_names:
                    row[f"probability_{name}"] = probabilities.get(name)
                writer.writerow(row)
        return path

    raise ValueError(
        f"Unsupported prediction output format {suffix!r}. "
        "Expected .json or .csv."
    )


def _load_artifact(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        raise TypeError("Model artifact must be a dictionary.")

    required_keys = {
        "model",
        "map_type",
        "model_name",
        "feature_names",
        "class_names",
    }
    missing = sorted(required_keys - set(artifact))
    if missing:
        raise KeyError(f"Model artifact is missing required keys: {missing}")

    map_type = str(artifact["map_type"]).lower().strip()
    if map_type not in {"wafer", "dram"}:
        raise ValueError(f"Unsupported map_type in model artifact: {map_type!r}")

    feature_names = [str(name) for name in artifact["feature_names"]]
    class_names = [str(name) for name in artifact["class_names"]]
    if not feature_names:
        raise ValueError("Model artifact contains no feature names.")
    if not class_names:
        raise ValueError("Model artifact contains no class names.")

    return {
        **artifact,
        "map_type": map_type,
        "model_name": str(artifact["model_name"]),
        "feature_names": feature_names,
        "class_names": class_names,
    }


def _build_feature_frame(
    maps: np.ndarray,
    map_type: str,
    feature_names: list[str],
) -> pd.DataFrame:
    feature_rows = [extract_features(map_array, map_type) for map_array in maps]
    missing = [
        name for name in feature_names
        if any(name not in row for row in feature_rows)
    ]
    if missing:
        raise KeyError(f"Feature extractor did not produce required features: {missing}")
    return pd.DataFrame(feature_rows, columns=feature_names, dtype=float)


def _predict_probabilities(model: Any, feature_matrix: np.ndarray) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    probabilities = np.asarray(model.predict_proba(feature_matrix), dtype=float)
    if probabilities.ndim != 2:
        raise ValueError(
            f"predict_proba returned an unexpected shape: {probabilities.shape}"
        )
    return probabilities


def _class_name(class_names: list[str], label_value: int) -> str:
    if 0 <= label_value < len(class_names):
        return class_names[label_value]
    return f"unknown:{label_value}"


def _ordered_probability_names(predictions: list[dict[str, Any]]) -> list[str]:
    for prediction in predictions:
        probabilities = prediction.get("probabilities")
        if isinstance(probabilities, dict):
            return list(probabilities)
    return []
