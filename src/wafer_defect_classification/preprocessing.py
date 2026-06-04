from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WAFER_OUTSIDE_VALUE = -1
WAFER_NORMAL_VALUE = 0
WAFER_DEFECT_VALUE = 1

DRAM_NORMAL_VALUE = 0
DRAM_FAILED_VALUE = 1


@dataclass(frozen=True)
class PreparedMap:
    array: np.ndarray
    map_type: str
    valid_mask: np.ndarray
    fail_mask: np.ndarray


def prepare_map(map_array: np.ndarray, map_type: str) -> PreparedMap:
    """Build validity and failure masks for supported defect-map types."""
    normalized_type = map_type.lower().strip()
    if normalized_type not in {"wafer", "dram"}:
        raise ValueError(f"Unsupported map_type: {map_type!r}")

    array = np.asarray(map_array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D map array, got shape {array.shape}")

    if normalized_type == "wafer":
        valid_mask = array != WAFER_OUTSIDE_VALUE
        fail_mask = valid_mask & (array == WAFER_DEFECT_VALUE)
    else:
        valid_mask = np.ones(array.shape, dtype=bool)
        fail_mask = array == DRAM_FAILED_VALUE

    return PreparedMap(
        array=array,
        map_type=normalized_type,
        valid_mask=valid_mask.astype(bool, copy=False),
        fail_mask=fail_mask.astype(bool, copy=False),
    )
