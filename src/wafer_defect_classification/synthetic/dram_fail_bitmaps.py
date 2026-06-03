from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFECT_CLASSES = [
    "random defect",
    "center defect",
    "edge defect",
    "ring defect",
    "scratch defect",
    "cluster defect",
    "row failure",
    "column failure",
    "block failure",
]

NORMAL_BIT = 0
FAILED_BIT = 1
DEFAULT_ROWS = 128
DEFAULT_COLUMNS = 128


@dataclass(frozen=True)
class GeneratedDramDataset:
    maps: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    output_dir: Path
    data_file: Path
    metadata_file: Path
    label_mapping_file: Path
    info_file: Path


def generate_dram_fail_bitmap(
    defect_class: str,
    rng: np.random.Generator,
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
) -> np.ndarray:
    """Generate one synthetic bit/address-level DRAM fail bit map."""
    if defect_class not in DEFECT_CLASSES:
        raise ValueError(f"Unknown DRAM fail bit map defect class: {defect_class}")

    failures = _generate_failures(defect_class, rng, rows, columns)
    bitmap = np.zeros((rows, columns), dtype=np.uint8)
    bitmap[failures] = FAILED_BIT
    return bitmap


def generate_dram_dataset(
    output_dir: str | Path = "data/synthetic/dram_fail_bitmaps",
    samples_per_class: int = 100,
    rows: int = DEFAULT_ROWS,
    columns: int = DEFAULT_COLUMNS,
    seed: int = 42,
) -> GeneratedDramDataset:
    """Generate and save a synthetic DRAM fail bit map dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    maps: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(DEFECT_CLASSES):
        for sample_index in range(samples_per_class):
            sample_id = f"dram_{class_index:02d}_{sample_index:04d}"
            sample = generate_dram_fail_bitmap(class_name, rng, rows, columns)
            maps.append(sample)
            labels.append(class_index)
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "map_type": "dram",
                    "class_index": class_index,
                    "class_name": class_name,
                    "rows": rows,
                    "columns": columns,
                    "synthetic_data": True,
                }
            )

    maps_array = np.stack(maps, axis=0)
    labels_array = np.asarray(labels, dtype=np.int64)

    data_file = output_path / "dram_fail_bitmaps.npz"
    metadata_file = output_path / "metadata.csv"
    label_mapping_file = output_path / "label_mapping.json"
    info_file = output_path / "dataset_info.json"

    np.savez_compressed(
        data_file,
        maps=maps_array,
        labels=labels_array,
        class_names=np.asarray(DEFECT_CLASSES),
    )

    pd.DataFrame(metadata_rows).to_csv(metadata_file, index=False)
    _write_label_mapping(label_mapping_file)
    _write_info(
        info_file=info_file,
        data_file=data_file,
        metadata_file=metadata_file,
        label_mapping_file=label_mapping_file,
        samples_per_class=samples_per_class,
        rows=rows,
        columns=columns,
        seed=seed,
    )

    return GeneratedDramDataset(
        maps=maps_array,
        labels=labels_array,
        class_names=DEFECT_CLASSES.copy(),
        output_dir=output_path,
        data_file=data_file,
        metadata_file=metadata_file,
        label_mapping_file=label_mapping_file,
        info_file=info_file,
    )


def _generate_failures(
    defect_class: str,
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    yy, xx = np.indices((rows, columns))
    center_y = (rows - 1) / 2.0
    center_x = (columns - 1) / 2.0
    radius = min(rows, columns) / 2.0
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    if defect_class == "random defect":
        failures = _sample_from_mask(
            np.ones((rows, columns), dtype=bool),
            rng,
            fraction=rng.uniform(0.003, 0.010),
        )
    elif defect_class == "center defect":
        candidate = distances <= radius * rng.uniform(0.18, 0.32)
        failures = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.20, 0.45))
    elif defect_class == "edge defect":
        band = int(rng.integers(4, 12))
        candidate = np.zeros((rows, columns), dtype=bool)
        candidate[:band, :] = True
        candidate[-band:, :] = True
        candidate[:, :band] = True
        candidate[:, -band:] = True
        failures = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.12, 0.28))
    elif defect_class == "ring defect":
        ring_radius = radius * rng.uniform(0.42, 0.68)
        thickness = radius * rng.uniform(0.025, 0.055)
        candidate = np.abs(distances - ring_radius) <= thickness
        failures = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.25, 0.50))
    elif defect_class == "scratch defect":
        failures = _line_failure(rng, rows, columns, center_y, center_x)
    elif defect_class == "cluster defect":
        failures = _cluster_failure(rng, rows, columns, yy, xx)
    elif defect_class == "row failure":
        failures = _row_failure(rng, rows, columns)
    elif defect_class == "column failure":
        failures = _column_failure(rng, rows, columns)
    elif defect_class == "block failure":
        failures = _block_failure(rng, rows, columns)
    else:
        raise ValueError(f"Unknown DRAM fail bit map defect class: {defect_class}")

    return _add_sparse_noise(failures, rng, fraction=0.0008)


def _sample_from_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> np.ndarray:
    failures = np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return failures
    count = max(1, int(round(len(coords) * fraction)))
    count = min(count, len(coords))
    chosen = rng.choice(len(coords), size=count, replace=False)
    selected = coords[chosen]
    failures[selected[:, 0], selected[:, 1]] = True
    return failures


def _line_failure(
    rng: np.random.Generator,
    rows: int,
    columns: int,
    center_y: float,
    center_x: float,
) -> np.ndarray:
    yy, xx = np.indices((rows, columns))
    angle = rng.uniform(0.0, np.pi)
    offset = rng.uniform(-12.0, 12.0)
    thickness = rng.uniform(0.7, 1.8)
    signed_distance = (
        (xx - center_x) * np.cos(angle) + (yy - center_y) * np.sin(angle) - offset
    )
    candidate = np.abs(signed_distance) <= thickness
    keep = rng.random((rows, columns)) < rng.uniform(0.75, 0.95)
    return candidate & keep


def _cluster_failure(
    rng: np.random.Generator,
    rows: int,
    columns: int,
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    center_y = int(rng.integers(rows // 8, rows - rows // 8))
    center_x = int(rng.integers(columns // 8, columns - columns // 8))
    radius = rng.uniform(5.0, 14.0)
    candidate = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
    return _sample_from_mask(candidate, rng, fraction=rng.uniform(0.35, 0.70))


def _row_failure(
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    failures = np.zeros((rows, columns), dtype=bool)
    row = int(rng.integers(0, rows))
    thickness = int(rng.integers(1, 4))
    row_stop = min(rows, row + thickness)
    failures[row:row_stop, :] = True
    if row_stop - row < thickness:
        failures[max(0, row - (thickness - (row_stop - row))) : row, :] = True
    return failures


def _column_failure(
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    failures = np.zeros((rows, columns), dtype=bool)
    column = int(rng.integers(0, columns))
    thickness = int(rng.integers(1, 4))
    column_stop = min(columns, column + thickness)
    failures[:, column:column_stop] = True
    if column_stop - column < thickness:
        failures[:, max(0, column - (thickness - (column_stop - column))) : column] = True
    return failures


def _block_failure(
    rng: np.random.Generator,
    rows: int,
    columns: int,
) -> np.ndarray:
    failures = np.zeros((rows, columns), dtype=bool)
    block_rows = int(rng.integers(max(4, rows // 16), max(5, rows // 5)))
    block_columns = int(rng.integers(max(4, columns // 16), max(5, columns // 5)))
    y0 = int(rng.integers(0, max(1, rows - block_rows + 1)))
    x0 = int(rng.integers(0, max(1, columns - block_columns + 1)))
    failures[y0 : y0 + block_rows, x0 : x0 + block_columns] = True
    return failures


def _add_sparse_noise(
    failures: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> np.ndarray:
    available = ~failures
    noise = _sample_from_mask(available, rng, fraction=fraction)
    return failures | noise


def _write_label_mapping(path: Path) -> None:
    mapping = {
        "class_to_index": {name: index for index, name in enumerate(DEFECT_CLASSES)},
        "index_to_class": {str(index): name for index, name in enumerate(DEFECT_CLASSES)},
    }
    path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")


def _write_info(
    info_file: Path,
    data_file: Path,
    metadata_file: Path,
    label_mapping_file: Path,
    samples_per_class: int,
    rows: int,
    columns: int,
    seed: int,
) -> None:
    info = {
        "map_type": "dram",
        "description": (
            "Synthetic bit/address-level defect distribution inside one memory array."
        ),
        "synthetic_data_only": True,
        "normal_bit_value": NORMAL_BIT,
        "failed_bit_value": FAILED_BIT,
        "rows": rows,
        "columns": columns,
        "samples_per_class": samples_per_class,
        "total_samples": samples_per_class * len(DEFECT_CLASSES),
        "random_seed": seed,
        "data_file": str(data_file),
        "metadata_file": str(metadata_file),
        "label_mapping_file": str(label_mapping_file),
    }
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
