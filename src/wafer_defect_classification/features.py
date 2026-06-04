from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from skimage.measure import label, regionprops

from wafer_defect_classification.preprocessing import PreparedMap, prepare_map

FEATURE_NAMES = [
    "fail_density",
    "fail_count",
    "valid_area",
    "centroid_y",
    "centroid_x",
    "centroid_distance_to_center",
    "radial_mean",
    "radial_std",
    "radial_q25",
    "radial_q75",
    "edge_fail_ratio",
    "center_fail_ratio",
    "row_fail_mean",
    "row_fail_max",
    "row_fail_std",
    "column_fail_mean",
    "column_fail_max",
    "column_fail_std",
    "horizontal_projection_std",
    "vertical_projection_std",
    "connected_component_count",
    "largest_component_area",
    "largest_component_area_ratio",
    "largest_component_bbox_height",
    "largest_component_bbox_width",
    "largest_component_eccentricity",
    "bounding_box_height",
    "bounding_box_width",
    "bounding_box_area_ratio",
    "aspect_ratio",
]


def get_feature_names() -> list[str]:
    return FEATURE_NAMES.copy()


def extract_features(map_array: np.ndarray, map_type: str) -> dict[str, float]:
    """Extract finite numeric features from one wafer map or DRAM fail bit map."""
    prepared = prepare_map(map_array, map_type)
    features = _extract_from_prepared_map(prepared)
    return {name: _finite_float(features[name]) for name in FEATURE_NAMES}


def extract_feature_table(
    maps: Iterable[np.ndarray],
    labels: Iterable[int],
    class_names: list[str],
    map_type: str,
) -> pd.DataFrame:
    """Extract feature rows and attach label metadata."""
    rows: list[dict[str, object]] = []
    for sample_index, (map_array, label_value) in enumerate(zip(maps, labels, strict=True)):
        label_int = int(label_value)
        class_name = (
            class_names[label_int]
            if 0 <= label_int < len(class_names)
            else f"unknown:{label_int}"
        )
        row: dict[str, object] = {
            "sample_index": sample_index,
            "map_type": map_type,
            "label": label_int,
            "class_name": class_name,
        }
        row.update(extract_features(map_array, map_type))
        rows.append(row)

    return pd.DataFrame(rows, columns=["sample_index", "map_type", "label", "class_name", *FEATURE_NAMES])


def _extract_from_prepared_map(prepared: PreparedMap) -> dict[str, float]:
    valid_mask = prepared.valid_mask
    fail_mask = prepared.fail_mask
    height, width = fail_mask.shape

    valid_area = int(valid_mask.sum())
    fail_count = int(fail_mask.sum())
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    yy, xx = np.indices(fail_mask.shape)
    valid_distances = _normalized_distances(yy, xx, valid_mask, center_y, center_x)
    fail_distances = valid_distances[fail_mask]

    features: dict[str, float] = {
        "fail_density": _safe_div(fail_count, valid_area),
        "fail_count": float(fail_count),
        "valid_area": float(valid_area),
    }

    features.update(
        _centroid_features(
            fail_mask=fail_mask,
            center_y=center_y,
            center_x=center_x,
        )
    )
    features.update(_radial_features(fail_distances))
    features.update(
        _region_ratio_features(
            fail_mask=fail_mask,
            valid_mask=valid_mask,
            normalized_distances=valid_distances,
            fail_count=fail_count,
        )
    )
    features.update(_projection_features(fail_mask=fail_mask, valid_mask=valid_mask))
    features.update(_connected_component_features(fail_mask=fail_mask, valid_area=valid_area))
    features.update(_bounding_box_features(fail_mask=fail_mask, valid_area=valid_area))
    return features


def _normalized_distances(
    yy: np.ndarray,
    xx: np.ndarray,
    valid_mask: np.ndarray,
    center_y: float,
    center_x: float,
) -> np.ndarray:
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    max_distance = float(distances[valid_mask].max()) if valid_mask.any() else 0.0
    if max_distance == 0.0:
        return np.zeros_like(distances, dtype=float)
    return distances / max_distance


def _centroid_features(
    fail_mask: np.ndarray,
    center_y: float,
    center_x: float,
) -> dict[str, float]:
    fail_coords = np.argwhere(fail_mask)
    if len(fail_coords) == 0:
        return {
            "centroid_y": center_y,
            "centroid_x": center_x,
            "centroid_distance_to_center": 0.0,
        }

    centroid_y = float(fail_coords[:, 0].mean())
    centroid_x = float(fail_coords[:, 1].mean())
    distance = float(np.sqrt((centroid_y - center_y) ** 2 + (centroid_x - center_x) ** 2))
    return {
        "centroid_y": centroid_y,
        "centroid_x": centroid_x,
        "centroid_distance_to_center": distance,
    }


def _radial_features(fail_distances: np.ndarray) -> dict[str, float]:
    if fail_distances.size == 0:
        return {
            "radial_mean": 0.0,
            "radial_std": 0.0,
            "radial_q25": 0.0,
            "radial_q75": 0.0,
        }

    return {
        "radial_mean": float(fail_distances.mean()),
        "radial_std": float(fail_distances.std()),
        "radial_q25": float(np.quantile(fail_distances, 0.25)),
        "radial_q75": float(np.quantile(fail_distances, 0.75)),
    }


def _region_ratio_features(
    fail_mask: np.ndarray,
    valid_mask: np.ndarray,
    normalized_distances: np.ndarray,
    fail_count: int,
) -> dict[str, float]:
    center_region = valid_mask & (normalized_distances <= 0.35)
    edge_region = valid_mask & (normalized_distances >= 0.75)
    center_fail_count = int((fail_mask & center_region).sum())
    edge_fail_count = int((fail_mask & edge_region).sum())
    return {
        "edge_fail_ratio": _safe_div(edge_fail_count, fail_count),
        "center_fail_ratio": _safe_div(center_fail_count, fail_count),
    }


def _projection_features(
    fail_mask: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    row_counts = fail_mask.sum(axis=1).astype(float)
    column_counts = fail_mask.sum(axis=0).astype(float)
    valid_row_lengths = valid_mask.sum(axis=1).astype(float)
    valid_column_lengths = valid_mask.sum(axis=0).astype(float)

    row_values = row_counts[valid_row_lengths > 0]
    column_values = column_counts[valid_column_lengths > 0]
    row_density = _density_projection(row_counts, valid_row_lengths)
    column_density = _density_projection(column_counts, valid_column_lengths)

    return {
        "row_fail_mean": _mean_or_zero(row_values),
        "row_fail_max": _max_or_zero(row_values),
        "row_fail_std": _std_or_zero(row_values),
        "column_fail_mean": _mean_or_zero(column_values),
        "column_fail_max": _max_or_zero(column_values),
        "column_fail_std": _std_or_zero(column_values),
        "horizontal_projection_std": _std_or_zero(row_density),
        "vertical_projection_std": _std_or_zero(column_density),
    }


def _connected_component_features(
    fail_mask: np.ndarray,
    valid_area: int,
) -> dict[str, float]:
    labeled = label(fail_mask, connectivity=1)
    regions = regionprops(labeled)
    if not regions:
        return {
            "connected_component_count": 0.0,
            "largest_component_area": 0.0,
            "largest_component_area_ratio": 0.0,
            "largest_component_bbox_height": 0.0,
            "largest_component_bbox_width": 0.0,
            "largest_component_eccentricity": 0.0,
        }

    largest = max(regions, key=lambda region: region.area)
    min_row, min_col, max_row, max_col = largest.bbox
    return {
        "connected_component_count": float(len(regions)),
        "largest_component_area": float(largest.area),
        "largest_component_area_ratio": _safe_div(largest.area, valid_area),
        "largest_component_bbox_height": float(max_row - min_row),
        "largest_component_bbox_width": float(max_col - min_col),
        "largest_component_eccentricity": _finite_float(largest.eccentricity),
    }


def _bounding_box_features(
    fail_mask: np.ndarray,
    valid_area: int,
) -> dict[str, float]:
    fail_coords = np.argwhere(fail_mask)
    if len(fail_coords) == 0:
        return {
            "bounding_box_height": 0.0,
            "bounding_box_width": 0.0,
            "bounding_box_area_ratio": 0.0,
            "aspect_ratio": 0.0,
        }

    min_row = int(fail_coords[:, 0].min())
    max_row = int(fail_coords[:, 0].max())
    min_col = int(fail_coords[:, 1].min())
    max_col = int(fail_coords[:, 1].max())
    bbox_height = max_row - min_row + 1
    bbox_width = max_col - min_col + 1
    bbox_area = bbox_height * bbox_width
    return {
        "bounding_box_height": float(bbox_height),
        "bounding_box_width": float(bbox_width),
        "bounding_box_area_ratio": _safe_div(bbox_area, valid_area),
        "aspect_ratio": _safe_div(bbox_width, bbox_height),
    }


def _density_projection(counts: np.ndarray, valid_lengths: np.ndarray) -> np.ndarray:
    valid = valid_lengths > 0
    if not valid.any():
        return np.asarray([], dtype=float)
    return counts[valid] / valid_lengths[valid]


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def _mean_or_zero(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else 0.0


def _max_or_zero(values: np.ndarray) -> float:
    return float(values.max()) if values.size else 0.0


def _std_or_zero(values: np.ndarray) -> float:
    return float(values.std()) if values.size else 0.0


def _finite_float(value: float) -> float:
    numeric_value = float(value)
    if not np.isfinite(numeric_value):
        return 0.0
    return numeric_value
