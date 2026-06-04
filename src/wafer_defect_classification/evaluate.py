from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute JSON-friendly synthetic validation classification metrics."""
    labels = list(range(len(class_names)))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    per_class_precision, per_class_recall, per_class_f1, per_class_support = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": _float(accuracy_score(y_true, y_pred)),
        "macro_precision": _float(macro_precision),
        "macro_recall": _float(macro_recall),
        "macro_f1": _float(macro_f1),
        "weighted_f1": _float(weighted_f1),
        "per_class": [
            {
                "class_index": int(index),
                "class_name": class_name,
                "precision": _float(per_class_precision[index]),
                "recall": _float(per_class_recall[index]),
                "f1": _float(per_class_f1[index]),
                "support": int(per_class_support[index]),
            }
            for index, class_name in enumerate(class_names)
        ],
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def _float(value: float) -> float:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return 0.0
    return numeric_value
