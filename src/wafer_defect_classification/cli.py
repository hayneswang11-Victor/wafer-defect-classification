from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console

from wafer_defect_classification.synthetic.dram_fail_bitmaps import (
    generate_dram_dataset,
)
from wafer_defect_classification.synthetic.wafer_maps import generate_wafer_dataset
from wafer_defect_classification.visualization import save_sample_grid

DEFAULT_CONFIG = Path("configs/default.yaml")


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "generate":
        _run_generate(args)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wafer_defect_classification",
        description="Synthetic wafer map and DRAM fail bit map prototype tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate synthetic wafer maps and/or DRAM fail bit maps.",
    )
    generate_parser.add_argument(
        "--map-type",
        choices=["wafer", "dram", "both"],
        required=True,
        help="Synthetic map type to generate.",
    )
    generate_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to the YAML configuration file.",
    )
    generate_parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Override the configured number of samples per defect class.",
    )
    generate_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override the configured random seed.",
    )
    return parser


def _run_generate(args: argparse.Namespace) -> None:
    console = Console()
    config = _load_config(Path(args.config))
    seed = int(args.seed if args.seed is not None else config["project"]["random_seed"])
    figure_dir = Path(config["outputs"]["figure_dir"])

    console.print("[bold]Synthetic data disclosure:[/bold] generated data only.")

    if args.map_type in {"wafer", "both"}:
        wafer_config = config["map_types"]["wafer"]
        wafer_dataset = generate_wafer_dataset(
            output_dir=wafer_config["output_dir"],
            samples_per_class=_samples_per_class(args, wafer_config),
            height=int(wafer_config["height"]),
            width=int(wafer_config["width"]),
            seed=seed,
        )
        wafer_figure = save_sample_grid(
            maps=wafer_dataset.maps,
            labels=wafer_dataset.labels,
            class_names=wafer_dataset.class_names,
            map_type="wafer",
            output_path=figure_dir / "wafer_sample_grid.png",
        )
        _print_dataset_summary(console, "wafer", wafer_dataset, wafer_figure)

    if args.map_type in {"dram", "both"}:
        dram_config = config["map_types"]["dram"]
        dram_dataset = generate_dram_dataset(
            output_dir=dram_config["output_dir"],
            samples_per_class=_samples_per_class(args, dram_config),
            rows=int(dram_config["rows"]),
            columns=int(dram_config["columns"]),
            seed=seed,
        )
        dram_figure = save_sample_grid(
            maps=dram_dataset.maps,
            labels=dram_dataset.labels,
            class_names=dram_dataset.class_names,
            map_type="dram",
            output_path=figure_dir / "dram_sample_grid.png",
        )
        _print_dataset_summary(console, "dram", dram_dataset, dram_figure)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError(f"Configuration file did not contain a mapping: {path}")
    return config


def _samples_per_class(args: argparse.Namespace, map_config: dict[str, Any]) -> int:
    if args.samples_per_class is not None:
        return int(args.samples_per_class)
    return int(map_config["samples_per_class"])


def _print_dataset_summary(
    console: Console,
    map_type: str,
    dataset: Any,
    figure_path: Path,
) -> None:
    console.print(f"[bold]{map_type}[/bold] dataset generated")
    console.print(f"  data: {dataset.data_file}")
    console.print(f"  metadata: {dataset.metadata_file}")
    console.print(f"  labels: {dataset.label_mapping_file}")
    console.print(f"  info: {dataset.info_file}")
    console.print(f"  figure: {figure_path}")


if __name__ == "__main__":
    raise SystemExit(main())
