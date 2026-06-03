import numpy as np
from skimage.measure import label, regionprops

from wafer_defect_classification.synthetic.wafer_maps import (
    DEFECT_CLASSES,
    OUTSIDE_WAFER,
    generate_wafer_dataset,
    generate_wafer_map,
    make_circular_wafer_mask,
)


def test_wafer_generation_includes_all_classes_and_expected_shape(tmp_path):
    dataset = generate_wafer_dataset(
        output_dir=tmp_path / "wafer_maps",
        samples_per_class=2,
        height=64,
        width=64,
        seed=123,
    )

    assert dataset.maps.shape == (len(DEFECT_CLASSES) * 2, 64, 64)
    assert dataset.labels.shape == (len(DEFECT_CLASSES) * 2,)
    assert dataset.class_names == DEFECT_CLASSES
    assert set(dataset.labels.tolist()) == set(range(len(DEFECT_CLASSES)))


def test_wafer_generation_saves_mask_metadata_and_labels(tmp_path):
    dataset = generate_wafer_dataset(
        output_dir=tmp_path / "wafer_maps",
        samples_per_class=1,
        seed=123,
    )

    assert dataset.wafer_mask.shape == (64, 64)
    assert dataset.wafer_mask.dtype == np.bool_
    assert np.all(dataset.maps[:, ~dataset.wafer_mask] == OUTSIDE_WAFER)
    assert set(np.unique(dataset.maps[:, dataset.wafer_mask])).issubset({0, 1})
    assert dataset.data_file.exists()
    assert dataset.metadata_file.exists()
    assert dataset.label_mapping_file.exists()
    assert dataset.info_file.exists()


def test_wafer_generation_is_reproducible_with_same_seed(tmp_path):
    first = generate_wafer_dataset(
        output_dir=tmp_path / "first",
        samples_per_class=2,
        seed=999,
    )
    second = generate_wafer_dataset(
        output_dir=tmp_path / "second",
        samples_per_class=2,
        seed=999,
    )

    assert np.array_equal(first.maps, second.maps)
    assert np.array_equal(first.labels, second.labels)
    assert np.array_equal(first.wafer_mask, second.wafer_mask)


def test_wafer_block_failure_has_large_connected_region():
    sample = generate_wafer_map(
        "block failure",
        rng=np.random.default_rng(77),
        height=64,
        width=64,
    )
    defective = sample == 1
    regions = regionprops(label(defective, connectivity=1))
    largest = max(regions, key=lambda region: region.area)
    min_row, min_col, max_row, max_col = largest.bbox
    bbox_height = max_row - min_row
    bbox_width = max_col - min_col

    assert largest.area >= 55
    assert bbox_height >= 8
    assert bbox_width >= 8
    assert largest.area >= 0.70 * bbox_height * bbox_width


def test_wafer_row_failure_produces_long_horizontal_pattern():
    wafer_mask = make_circular_wafer_mask(64, 64)
    sample = generate_wafer_map(
        "row failure",
        rng=np.random.default_rng(88),
        height=64,
        width=64,
    )
    defective = sample == 1
    row_counts = defective.sum(axis=1)
    column_counts = defective.sum(axis=0)

    assert row_counts.max() >= 0.55 * wafer_mask.sum(axis=1).max()
    assert row_counts.max() >= 2.5 * column_counts.max()


def test_wafer_column_failure_produces_long_vertical_pattern():
    wafer_mask = make_circular_wafer_mask(64, 64)
    sample = generate_wafer_map(
        "column failure",
        rng=np.random.default_rng(99),
        height=64,
        width=64,
    )
    defective = sample == 1
    row_counts = defective.sum(axis=1)
    column_counts = defective.sum(axis=0)

    assert column_counts.max() >= 0.55 * wafer_mask.sum(axis=0).max()
    assert column_counts.max() >= 2.5 * row_counts.max()
