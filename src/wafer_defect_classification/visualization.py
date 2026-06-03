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
