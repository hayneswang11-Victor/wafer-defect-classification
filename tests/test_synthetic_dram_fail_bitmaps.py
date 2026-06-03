import numpy as np

from wafer_defect_classification.synthetic.dram_fail_bitmaps import (
    DEFECT_CLASSES,
    generate_dram_dataset,
)


def test_dram_generation_includes_all_classes_and_expected_shape(tmp_path):
    dataset = generate_dram_dataset(
        output_dir=tmp_path / "dram_fail_bitmaps",
        samples_per_class=2,
        rows=128,
        columns=128,
        seed=123,
    )

    assert dataset.maps.shape == (len(DEFECT_CLASSES) * 2, 128, 128)
    assert dataset.labels.shape == (len(DEFECT_CLASSES) * 2,)
    assert dataset.class_names == DEFECT_CLASSES
    assert set(dataset.labels.tolist()) == set(range(len(DEFECT_CLASSES)))


def test_dram_generation_saves_rectangular_arrays_metadata_and_labels(tmp_path):
    dataset = generate_dram_dataset(
        output_dir=tmp_path / "dram_fail_bitmaps",
        samples_per_class=1,
        rows=128,
        columns=128,
        seed=123,
    )

    assert dataset.maps.ndim == 3
    assert dataset.maps.shape[1:] == (128, 128)
    assert set(np.unique(dataset.maps)).issubset({0, 1})
    assert dataset.data_file.exists()
    assert dataset.metadata_file.exists()
    assert dataset.label_mapping_file.exists()
    assert dataset.info_file.exists()


def test_dram_generation_is_reproducible_with_same_seed(tmp_path):
    first = generate_dram_dataset(
        output_dir=tmp_path / "first",
        samples_per_class=2,
        seed=999,
    )
    second = generate_dram_dataset(
        output_dir=tmp_path / "second",
        samples_per_class=2,
        seed=999,
    )

    assert np.array_equal(first.maps, second.maps)
    assert np.array_equal(first.labels, second.labels)
