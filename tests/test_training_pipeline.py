import json

import pytest

from wafer_defect_classification.synthetic.dram_fail_bitmaps import generate_dram_dataset
from wafer_defect_classification.synthetic.wafer_maps import generate_wafer_dataset
from wafer_defect_classification.train import train_model


def test_training_pipeline_runs_on_tiny_wafer_dataset(tmp_path):
    dataset = generate_wafer_dataset(
        output_dir=tmp_path / "synthetic" / "wafer_maps",
        samples_per_class=3,
        seed=101,
    )

    result = train_model(
        map_type="wafer",
        model_name="random_forest",
        data_path=dataset.data_file,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        figure_dir=tmp_path / "figures",
        split_dir=tmp_path / "splits",
        n_estimators=10,
    )

    assert result.model_file.exists()
    assert result.metrics_file.exists()
    assert result.confusion_matrix_figure.exists()
    assert result.model_diagnostic_figure is not None
    assert result.model_diagnostic_figure.exists()
    assert result.split_file.exists()

    metrics = json.loads(result.metrics_file.read_text(encoding="utf-8"))
    assert metrics["synthetic_data_only"] is True
    assert "Synthetic validation metrics" in metrics["interpretation"]
    assert "test" in metrics
    assert "accuracy" in metrics["test"]


def test_training_pipeline_runs_on_tiny_dram_dataset(tmp_path):
    dataset = generate_dram_dataset(
        output_dir=tmp_path / "synthetic" / "dram_fail_bitmaps",
        samples_per_class=3,
        rows=32,
        columns=32,
        seed=202,
    )

    result = train_model(
        map_type="dram",
        model_name="logistic_regression",
        data_path=dataset.data_file,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        figure_dir=tmp_path / "figures",
        split_dir=tmp_path / "splits",
    )

    assert result.model_file.exists()
    assert result.metrics_file.exists()
    assert result.confusion_matrix_figure.exists()
    assert result.model_diagnostic_figure is not None
    assert result.model_diagnostic_figure.exists()
    assert result.split_file.exists()


def test_training_pipeline_creates_split_file(tmp_path):
    dataset = generate_dram_dataset(
        output_dir=tmp_path / "synthetic" / "dram_fail_bitmaps",
        samples_per_class=3,
        rows=32,
        columns=32,
        seed=303,
    )

    result = train_model(
        map_type="dram",
        model_name="random_forest",
        data_path=dataset.data_file,
        model_dir=tmp_path / "models",
        metrics_dir=tmp_path / "metrics",
        figure_dir=tmp_path / "figures",
        split_dir=tmp_path / "splits",
        n_estimators=10,
    )

    splits = json.loads(result.split_file.read_text(encoding="utf-8"))
    assert sorted(splits) == [
        "map_type",
        "random_seed",
        "requested_test_size",
        "requested_validation_size_relative_to_remaining",
        "test",
        "train",
        "validation",
    ]
    assert len(splits["train"]) > 0
    assert len(splits["validation"]) > 0
    assert len(splits["test"]) > 0


def test_unknown_map_type_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported map_type"):
        train_model(
            map_type="mixed",
            model_name="random_forest",
            data_path=tmp_path / "missing.npz",
        )


def test_unknown_model_name_raises_clear_error(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model_name"):
        train_model(
            map_type="wafer",
            model_name="svm",
            data_path=tmp_path / "missing.npz",
        )
