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

OUTSIDE_WAFER = -1
NORMAL_DIE = 0
DEFECT_DIE = 1
DEFAULT_HEIGHT = 64
DEFAULT_WIDTH = 64


@dataclass(frozen=True)
class GeneratedWaferDataset:
    maps: np.ndarray
    labels: np.ndarray
    class_names: list[str]
    wafer_mask: np.ndarray
    output_dir: Path
    data_file: Path
    metadata_file: Path
    label_mapping_file: Path
    info_file: Path


def make_circular_wafer_mask(
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> np.ndarray:
    """Return a circular die-validity mask for a wafer-map grid."""
    yy, xx = np.indices((height, width))
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(height, width) * 0.46
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)
    return distances <= radius


def generate_wafer_map(
    defect_class: str,
    rng: np.random.Generator,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> np.ndarray:
    """Generate one synthetic die-level wafer map for a defect class."""
    if defect_class not in DEFECT_CLASSES:
        raise ValueError(f"Unknown wafer defect class: {defect_class}")

    wafer_mask = make_circular_wafer_mask(height, width)
    defects = _generate_defects(defect_class, wafer_mask, rng)

    wafer_map = np.full((height, width), OUTSIDE_WAFER, dtype=np.int8)
    wafer_map[wafer_mask] = NORMAL_DIE
    wafer_map[defects & wafer_mask] = DEFECT_DIE
    return wafer_map


def generate_wafer_dataset(
    output_dir: str | Path = "data/synthetic/wafer_maps",
    samples_per_class: int = 100,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    seed: int = 42,
) -> GeneratedWaferDataset:
    """Generate and save a synthetic wafer-map dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    wafer_mask = make_circular_wafer_mask(height, width)
    maps: list[np.ndarray] = []
    labels: list[int] = []
    metadata_rows: list[dict[str, object]] = []

    for class_index, class_name in enumerate(DEFECT_CLASSES):
        for sample_index in range(samples_per_class):
            sample_id = f"wafer_{class_index:02d}_{sample_index:04d}"
            sample = generate_wafer_map(class_name, rng, height, width)
            maps.append(sample)
            labels.append(class_index)
            metadata_rows.append(
                {
                    "sample_id": sample_id,
                    "map_type": "wafer",
                    "class_index": class_index,
                    "class_name": class_name,
                    "height": height,
                    "width": width,
                    "synthetic_data": True,
                    "outside_wafer_value": OUTSIDE_WAFER,
                }
            )

    maps_array = np.stack(maps, axis=0)
    labels_array = np.asarray(labels, dtype=np.int64)

    data_file = output_path / "wafer_maps.npz"
    metadata_file = output_path / "metadata.csv"
    label_mapping_file = output_path / "label_mapping.json"
    info_file = output_path / "dataset_info.json"

    np.savez_compressed(
        data_file,
        maps=maps_array,
        labels=labels_array,
        class_names=np.asarray(DEFECT_CLASSES),
        wafer_mask=wafer_mask.astype(np.uint8),
        outside_wafer_value=np.asarray([OUTSIDE_WAFER], dtype=np.int8),
    )

    pd.DataFrame(metadata_rows).to_csv(metadata_file, index=False)
    _write_label_mapping(label_mapping_file)
    _write_info(
        info_file=info_file,
        data_file=data_file,
        metadata_file=metadata_file,
        label_mapping_file=label_mapping_file,
        samples_per_class=samples_per_class,
        height=height,
        width=width,
        seed=seed,
    )

    return GeneratedWaferDataset(
        maps=maps_array,
        labels=labels_array,
        class_names=DEFECT_CLASSES.copy(),
        wafer_mask=wafer_mask,
        output_dir=output_path,
        data_file=data_file,
        metadata_file=metadata_file,
        label_mapping_file=label_mapping_file,
        info_file=info_file,
    )


def _generate_defects(
    defect_class: str,
    wafer_mask: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = wafer_mask.shape
    yy, xx = np.indices((height, width))
    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0
    radius = min(height, width) * 0.46
    distances = np.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    if defect_class == "random defect":
        defects = _sample_from_mask(wafer_mask, rng, fraction=rng.uniform(0.025, 0.055))
    elif defect_class == "center defect":
        candidate = wafer_mask & (distances <= radius * rng.uniform(0.18, 0.30))
        defects = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.35, 0.60))
    elif defect_class == "edge defect":
        candidate = wafer_mask & (distances >= radius * rng.uniform(0.72, 0.84))
        defects = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.22, 0.42))
    elif defect_class == "ring defect":
        ring_radius = radius * rng.uniform(0.48, 0.68)
        thickness = radius * rng.uniform(0.035, 0.075)
        candidate = wafer_mask & (np.abs(distances - ring_radius) <= thickness)
        defects = _sample_from_mask(candidate, rng, fraction=rng.uniform(0.35, 0.65))
    elif defect_class == "scratch defect":
        defects = _line_defect(wafer_mask, rng, center_y, center_x)
    elif defect_class == "cluster defect":
        defects = _cluster_defect(wafer_mask, rng, yy, xx)
    elif defect_class == "row failure":
        defects = _row_defect(wafer_mask, rng)
    elif defect_class == "column failure":
        defects = _column_defect(wafer_mask, rng)
    elif defect_class == "block failure":
        defects = _block_defect(wafer_mask, rng)
    else:
        raise ValueError(f"Unknown wafer defect class: {defect_class}")

    return _add_sparse_noise(defects, wafer_mask, rng, fraction=0.003)


def _sample_from_mask(
    mask: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> np.ndarray:
    defects = np.zeros(mask.shape, dtype=bool)
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return defects
    count = max(1, int(round(len(coords) * fraction)))
    count = min(count, len(coords))
    chosen = rng.choice(len(coords), size=count, replace=False)
    selected = coords[chosen]
    defects[selected[:, 0], selected[:, 1]] = True
    return defects


def _line_defect(
    wafer_mask: np.ndarray,
    rng: np.random.Generator,
    center_y: float,
    center_x: float,
) -> np.ndarray:
    height, width = wafer_mask.shape
    yy, xx = np.indices((height, width))
    angle = rng.uniform(0.0, np.pi)
    offset = rng.uniform(-6.0, 6.0)
    thickness = rng.uniform(0.8, 1.8)
    signed_distance = (
        (xx - center_x) * np.cos(angle) + (yy - center_y) * np.sin(angle) - offset
    )
    candidate = wafer_mask & (np.abs(signed_distance) <= thickness)
    keep = rng.random((height, width)) < rng.uniform(0.70, 0.92)
    return candidate & keep


def _cluster_defect(
    wafer_mask: np.ndarray,
    rng: np.random.Generator,
    yy: np.ndarray,
    xx: np.ndarray,
) -> np.ndarray:
    coords = np.argwhere(wafer_mask)
    center_y, center_x = coords[rng.integers(0, len(coords))]
    radius = rng.uniform(3.5, 8.0)
    candidate = wafer_mask & ((yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2)
    return _sample_from_mask(candidate, rng, fraction=rng.uniform(0.45, 0.80))


def _row_defect(wafer_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    defects = np.zeros(wafer_mask.shape, dtype=bool)
    row_lengths = wafer_mask.sum(axis=1)
    min_length = 0.60 * row_lengths.max()
    valid_rows = np.flatnonzero(row_lengths >= min_length)
    row = int(rng.choice(valid_rows))
    thickness = int(rng.integers(1, 3))
    row_start = max(0, row - thickness // 2)
    row_stop = min(wafer_mask.shape[0], row_start + thickness)
    candidate = np.zeros_like(wafer_mask)
    candidate[row_start:row_stop, :] = True
    keep = rng.random(wafer_mask.shape) < rng.uniform(0.92, 1.00)
    defects[candidate & wafer_mask & keep] = True
    return defects


def _column_defect(wafer_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    defects = np.zeros(wafer_mask.shape, dtype=bool)
    column_heights = wafer_mask.sum(axis=0)
    min_height = 0.60 * column_heights.max()
    valid_columns = np.flatnonzero(column_heights >= min_height)
    column = int(rng.choice(valid_columns))
    thickness = int(rng.integers(1, 3))
    column_start = max(0, column - thickness // 2)
    column_stop = min(wafer_mask.shape[1], column_start + thickness)
    candidate = np.zeros_like(wafer_mask)
    candidate[:, column_start:column_stop] = True
    keep = rng.random(wafer_mask.shape) < rng.uniform(0.92, 1.00)
    defects[candidate & wafer_mask & keep] = True
    return defects


def _block_defect(wafer_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    block_height = int(rng.integers(8, 17))
    block_width = int(rng.integers(8, 17))
    y0, x0 = _choose_valid_block_origin(wafer_mask, block_height, block_width, rng)
    y1 = y0 + block_height
    x1 = x0 + block_width
    candidate = np.zeros_like(wafer_mask)
    candidate[y0:y1, x0:x1] = True
    keep = rng.random(wafer_mask.shape) < rng.uniform(0.94, 1.00)
    return candidate & wafer_mask & keep


def _choose_valid_block_origin(
    wafer_mask: np.ndarray,
    block_height: int,
    block_width: int,
    rng: np.random.Generator,
) -> tuple[int, int]:
    height, width = wafer_mask.shape
    full_inside: list[tuple[int, int]] = []
    mostly_inside: list[tuple[int, int]] = []

    for y0 in range(0, height - block_height + 1):
        for x0 in range(0, width - block_width + 1):
            region = wafer_mask[y0 : y0 + block_height, x0 : x0 + block_width]
            valid_fraction = float(region.mean())
            if valid_fraction == 1.0:
                full_inside.append((y0, x0))
            elif valid_fraction >= 0.95:
                mostly_inside.append((y0, x0))

    candidates = full_inside or mostly_inside
    if not candidates:
        raise RuntimeError("Could not place a meaningful wafer block defect.")

    choice = int(rng.integers(0, len(candidates)))
    return candidates[choice]


def _add_sparse_noise(
    defects: np.ndarray,
    wafer_mask: np.ndarray,
    rng: np.random.Generator,
    fraction: float,
) -> np.ndarray:
    noisy = defects.copy()
    noise = _sample_from_mask(wafer_mask & ~noisy, rng, fraction=fraction)
    return noisy | noise


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
    height: int,
    width: int,
    seed: int,
) -> None:
    info = {
        "map_type": "wafer",
        "description": "Synthetic die-level spatial defect distribution across a wafer.",
        "synthetic_data_only": True,
        "outside_wafer_value": OUTSIDE_WAFER,
        "normal_die_value": NORMAL_DIE,
        "defect_die_value": DEFECT_DIE,
        "height": height,
        "width": width,
        "samples_per_class": samples_per_class,
        "total_samples": samples_per_class * len(DEFECT_CLASSES),
        "random_seed": seed,
        "data_file": str(data_file),
        "metadata_file": str(metadata_file),
        "label_mapping_file": str(label_mapping_file),
    }
    info_file.write_text(json.dumps(info, indent=2), encoding="utf-8")
