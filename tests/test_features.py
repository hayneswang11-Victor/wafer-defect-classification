import numpy as np

from wafer_defect_classification.features import (
    extract_feature_table,
    extract_features,
    get_feature_names,
)
from wafer_defect_classification.synthetic.dram_fail_bitmaps import (
    DEFECT_CLASSES as DRAM_CLASSES,
    generate_dram_fail_bitmap,
)
from wafer_defect_classification.synthetic.wafer_maps import (
    DEFECT_CLASSES as WAFER_CLASSES,
    OUTSIDE_WAFER,
    generate_wafer_dataset,
    generate_wafer_map,
    make_circular_wafer_mask,
)


def test_random_wafer_map_feature_extraction():
    sample = generate_wafer_map("random defect", np.random.default_rng(1))
    features = extract_features(sample, map_type="wafer")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] > 0
    assert features["valid_area"] == float(make_circular_wafer_mask().sum())


def test_random_dram_map_feature_extraction():
    sample = generate_dram_fail_bitmap("random defect", np.random.default_rng(2))
    features = extract_features(sample, map_type="dram")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] > 0
    assert features["valid_area"] == 128.0 * 128.0


def test_empty_wafer_map_feature_extraction_does_not_crash():
    wafer_mask = make_circular_wafer_mask()
    sample = np.full(wafer_mask.shape, OUTSIDE_WAFER, dtype=np.int8)
    sample[wafer_mask] = 0
    features = extract_features(sample, map_type="wafer")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] == 0.0
    assert features["fail_density"] == 0.0
    assert features["valid_area"] == float(wafer_mask.sum())


def test_empty_dram_map_feature_extraction_does_not_crash():
    sample = np.zeros((128, 128), dtype=np.uint8)
    features = extract_features(sample, map_type="dram")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] == 0.0
    assert features["fail_density"] == 0.0
    assert features["valid_area"] == 128.0 * 128.0


def test_single_pixel_wafer_defect_does_not_crash():
    sample = np.asarray(
        [
            [-1, -1, -1, -1],
            [-1, 0, 0, -1],
            [-1, 0, 1, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    features = extract_features(sample, map_type="wafer")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] == 1.0


def test_single_pixel_dram_defect_does_not_crash():
    sample = np.zeros((16, 16), dtype=np.uint8)
    sample[8, 8] = 1
    features = extract_features(sample, map_type="dram")

    _assert_valid_feature_dict(features)
    assert features["fail_count"] == 1.0


def test_row_failure_has_high_row_fail_max_relative_to_column_fail_max():
    sample = generate_dram_fail_bitmap("row failure", np.random.default_rng(3))
    features = extract_features(sample, map_type="dram")

    assert features["row_fail_max"] >= 10.0 * features["column_fail_max"]


def test_column_failure_has_high_column_fail_max_relative_to_row_fail_max():
    sample = generate_dram_fail_bitmap("column failure", np.random.default_rng(4))
    features = extract_features(sample, map_type="dram")

    assert features["column_fail_max"] >= 10.0 * features["row_fail_max"]


def test_block_failure_has_large_connected_component_area():
    sample = generate_dram_fail_bitmap("block failure", np.random.default_rng(5))
    features = extract_features(sample, map_type="dram")

    assert features["connected_component_count"] >= 1.0
    assert features["largest_component_area"] >= 60.0
    assert features["largest_component_area_ratio"] > 0.0


def test_wafer_outside_area_is_ignored():
    sample = np.asarray(
        [
            [-1, -1, -1, -1],
            [-1, 0, 1, -1],
            [-1, 0, 0, -1],
            [-1, -1, -1, -1],
        ],
        dtype=np.int8,
    )
    features = extract_features(sample, map_type="wafer")

    assert features["valid_area"] == 4.0
    assert features["fail_count"] == 1.0
    assert features["fail_density"] == 0.25


def test_feature_table_has_one_row_per_sample_and_label_metadata(tmp_path):
    dataset = generate_wafer_dataset(
        output_dir=tmp_path / "wafer",
        samples_per_class=1,
        seed=6,
    )
    table = extract_feature_table(
        maps=dataset.maps,
        labels=dataset.labels,
        class_names=dataset.class_names,
        map_type="wafer",
    )

    assert len(table) == len(dataset.maps)
    assert {"label", "class_name"}.issubset(table.columns)
    assert table.loc[0, "label"] == 0
    assert table.loc[0, "class_name"] == WAFER_CLASSES[0]
    assert get_feature_names()[0] in table.columns


def test_feature_extraction_is_deterministic_for_same_input():
    sample = generate_dram_fail_bitmap("cluster defect", np.random.default_rng(7))

    first = extract_features(sample, map_type="dram")
    second = extract_features(sample, map_type="dram")

    assert first == second


def _assert_valid_feature_dict(features):
    assert list(features) == get_feature_names()
    assert len(features) == 30
    for value in features.values():
        assert isinstance(value, float)
        assert np.isfinite(value)


def test_feature_names_match_required_list():
    assert get_feature_names() == [
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
