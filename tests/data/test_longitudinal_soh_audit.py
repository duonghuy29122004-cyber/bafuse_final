import pytest

from bafuse.labeling.longitudinal_soh_audit import (
    _integral_after_duplicate_collapse,
    _extreme_text,
    _mad_scores,
    _positive_interval_summary,
    _ratio_percent,
    _time_axis_scores,
    _theil_sen,
    _verify_split,
)


def test_capacity_retention_ratio_is_scale_invariant():
    assert _ratio_percent(2.0, 2.5) == _ratio_percent(2000.0, 2500.0)


def test_modified_z_flags_extreme_value_only_after_minimum_sample_size():
    scores, metadata = _mad_scores([1.0, 1.02, 0.99, 1.01, 1.03, 1.0])
    assert metadata["status"] == "scored"
    assert scores[-1] is not None

    scores, metadata = _mad_scores([1.0, 1.01, 0.99, 1.0, 1.02, 30.0])
    assert metadata["status"] == "scored"
    assert scores[-1] is not None and abs(scores[-1]) > 3.5


def test_modified_z_does_not_invent_scale_for_small_or_zero_mad_series():
    scores, metadata = _mad_scores([1.0, 1.0, 1.0, 1.0])
    assert scores == [None] * 4
    assert metadata["status"] == "insufficient_sample_size"

    scores, metadata = _mad_scores([1.0, 1.0, 1.0, 1.0, 2.0])
    assert scores == [None] * 5
    assert metadata["status"] == "zero_mad_unscorable"


def test_time_axis_rounding_is_not_misclassified_as_a_physical_outlier():
    scores, metadata = _time_axis_scores([0.0, 5e-11, -5e-11, 1e-10, -1e-10])
    assert scores == [None] * 5
    assert metadata["status"] == "below_numerical_comparison_resolution"


def test_positive_interval_summary_is_descriptive_without_a_gap_threshold():
    summary = _positive_interval_summary([0.0, 0.1, 0.1, 0.3])
    assert summary["minimum"] == pytest.approx(0.1)
    assert summary["median"] == pytest.approx(0.1)
    assert summary["maximum"] == pytest.approx(0.3)
    assert summary["maximum_to_median"] == pytest.approx(3.0)


def test_theil_sen_uses_rpt_ordinal_as_descriptive_axis():
    slope, intercept = _theil_sen([1.0, 2.0, 4.0], [10.0, 12.0, 16.0])
    assert slope == 2.0
    assert intercept == 8.0


def test_trajectory_summary_reports_empty_increase_or_decrease_without_crashing():
    assert _extreme_text([], True) == "none"
    assert _extreme_text([-2.0, -1.0], True) == "-1.000000"
    assert _extreme_text([-2.0, -1.0], False) == "-2.000000"


def test_collapsing_exact_duplicate_rows_preserves_step4_integral():
    rows = [
        {"testtime": 0.0, "steptime": 0.0, "amps": -2.0},
        {"testtime": 0.0, "steptime": 0.0, "amps": -2.0},
        {"testtime": 1.0, "steptime": 1.0, "amps": -2.0},
    ]
    assert _integral_after_duplicate_collapse(rows, [b"same", b"same", b"next"], "testtime") == 2.0
    assert _integral_after_duplicate_collapse(rows, [b"same", b"same", b"next"], "steptime") == 2.0


def test_split_verification_keeps_parts_of_a_cell_together_and_part3_external():
    split = [
        {"physical_cell_id": "G1-Cell1", "partition": "train"},
        {"physical_cell_id": "G3-Cell1", "partition": "external_test"},
    ]
    candidates = [
        {"physical_cell_id": "G1-Cell1", "partition": "train", "dataset_part": "1"},
        {"physical_cell_id": "G1-Cell1", "partition": "train", "dataset_part": "2"},
        {"physical_cell_id": "G3-Cell1", "partition": "external_test", "dataset_part": "3"},
    ]
    _verify_split(candidates, split)

    overlapping = candidates + [
        {"physical_cell_id": "G1-Cell1", "partition": "validation", "dataset_part": "2"}
    ]
    with pytest.raises(ValueError, match="split mismatch|overlap"):
        _verify_split(overlapping, split)

    part3_split = [{"physical_cell_id": "G3-Cell1", "partition": "train"}]
    part3_in_train = [
        {"physical_cell_id": "G3-Cell1", "partition": "train", "dataset_part": "3"}
    ]
    with pytest.raises(ValueError, match="Part 3 record"):
        _verify_split(part3_in_train, part3_split)
