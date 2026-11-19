import hashlib

import pytest

from bafuse.labeling.independent_soh_audit import (
    BlockRequest,
    BlockSamples,
    _guide_mapping_status,
    _integrate,
    _same_mapped_file,
    parse_block_reference,
    recompute_physics_features,
)


def test_block_reference_requires_exact_step_and_row_interval():
    assert parse_block_reference(
        "Group_1/Group 1/TPG1_4_Cell9.csv#Step4:segment1:rows320489-327336"
    ).__dict__ == {
        "source": "Group_1/Group 1/TPG1_4_Cell9.csv",
        "step": 4,
        "start_line": 320489,
        "end_line": 327336,
    }
    with pytest.raises(ValueError, match="Unparseable"):
        parse_block_reference("TPG1_4_Cell9.csv")
    with pytest.raises(ValueError, match="Invalid"):
        parse_block_reference("x.csv#Step4:segment1:rows5-4")


def test_independent_integral_uses_absolute_current_and_positive_intervals():
    integral, timing = _integrate(
        [0.0, 1.0, 1.0, 3.0],
        [-2.0, -4.0, -6.0, -2.0],
    )

    assert integral == pytest.approx(11.0)
    assert timing["zero"] == 1
    assert timing["negative"] == 0


def test_independent_integral_reports_time_resets_without_reordering():
    integral, timing = _integrate(
        [0.0, 2.0, 1.0, 3.0],
        [1.0, 1.0, 1.0, 1.0],
    )

    assert integral == pytest.approx(4.0)
    assert timing["negative"] == 1


def test_block_scan_checks_step_identity_finite_values_and_following_transition():
    row = ["4", "0", "0", "0.0", "-1.5", "4.1", "24.0"]
    block_hash = hashlib.sha256(("\x1f".join(row) + "\n").encode()).hexdigest()
    request = BlockRequest(
        source_file="data/raw/oxford/part1/example.csv",
        source_part=1,
        step=4,
        start_line=2,
        end_line=2,
        expected_hash=block_hash,
        expected_source_size=100,
        expected_source_mtime_ns=1,
    )
    samples = BlockSamples(
        request,
        {
            "step": 0,
            "testtime": 1,
            "steptime": 2,
            "amphr": 3,
            "amps": 4,
            "volts": 5,
            "temperature": 6,
        },
        width=len(row),
    )
    samples.add(row)

    result = samples.result(next_step=5)

    assert result["row_count_exact"]
    assert result["block_hash_exact"]
    assert result["step_ids_exact"]
    assert result["current_negative_rows"] == 1
    assert result["missing_temperature"] == 0
    assert result["next_row_transitions_step"]
    assert result["wrong_width_rows"] == 0


def test_guide_relative_and_raw_absolute_paths_resolve_to_same_source():
    assert _same_mapped_file(
        "Group_1/Group 1/TPG1_4_Cell9.csv",
        "data/raw/oxford/part1/Group_1/Group 1/TPG1_4_Cell9.csv",
        1,
    )


def test_guide_mapping_uses_observation_keys_and_same_cell_bol_source():
    candidate = {
        "dataset_part": "1",
        "physical_cell_id": "G1-Cell9",
        "measurement_id": "RPT1",
        "rpt_index": "1",
        "source_file": "data/raw/oxford/part1/Group_1/Group 1/TPG1_4_Cell9.csv",
        "bol_source_part": "1",
        "bol_source_file": "data/raw/oxford/part1/Group_1/Group 1/TPG1_1_Cell9.csv",
    }
    rpt_guide = {
        "dataset_part": "1",
        "physical_cell_id": "G1-Cell9",
        "measurement_id": "RPT1",
        "rpt_order": "1",
        "source_file": "Group_1/Group 1/TPG1_4_Cell9.csv",
        "measurement_mapping_status": "single_checkpoint",
        "source_mapping_status": "unique_source_file",
    }
    bol_guide = {
        "dataset_part": "1",
        "physical_cell_id": "G1-Cell9",
        "measurement_id": "BoL",
        "source_file": (
            "Group_1/Group 1/TPG1_1_Cell9.csv;"
            "Group_1/Group 1/TPG1_2_Cell9.csv"
        ),
        "measurement_mapping_status": "single_checkpoint",
        "source_mapping_status": "multiple_source_files",
    }

    status, note = _guide_mapping_status(
        candidate,
        {("1", "G1-Cell9", "RPT1"): [rpt_guide]},
        {("1", "G1-Cell9"): [rpt_guide, bol_guide]},
    )

    assert status == "ACCEPTED"
    assert "raw hashes" in note

    candidate["rpt_index"] = "2"
    status, note = _guide_mapping_status(
        candidate,
        {("1", "G1-Cell9", "RPT1"): [rpt_guide]},
        {("1", "G1-Cell9"): [rpt_guide, bol_guide]},
    )
    assert status == "REVIEW"
    assert "RPT order" in note


def test_part3_bol_mapping_uses_corrected_zero_suffix_crosswalk():
    candidate = {
        "dataset_part": "3",
        "physical_cell_id": "G7-Cell19",
        "measurement_id": "RPT1",
        "rpt_index": "1",
        "source_file": "data/raw/oxford/part3/Group_7/Group 7/TPG7.3-Cell19.csv",
        "bol_source_part": "3",
        "bol_source_file": "data/raw/oxford/part3/Group_7/Group 7/TPG7.0-Cell19.csv",
    }
    rpt_guide = {
        "dataset_part": "3",
        "physical_cell_id": "G7-Cell19",
        "measurement_id": "RPT1",
        "rpt_order": "1",
        "source_file": "Group_7/Group 7/TPG7.3-Cell19.csv",
        "measurement_mapping_status": "single_checkpoint",
        "source_mapping_status": "unique_source_file",
    }
    base_guide_bol = {
        "dataset_part": "3",
        "physical_cell_id": "G7-Cell19",
        "measurement_id": "BoL",
        "source_file": "Group_7/Group 7/TPG7.1-Cell19.csv",
        "measurement_mapping_status": "single_checkpoint",
        "source_mapping_status": "unique_source_file",
    }
    corrected_crosswalk_bol = {
        **base_guide_bol,
        "source_file": "Group_7/Group 7/TPG7.0-Cell19.csv",
    }

    status, note = _guide_mapping_status(
        candidate,
        {("3", "G7-Cell19", "RPT1"): [rpt_guide]},
        {("3", "G7-Cell19"): [rpt_guide, base_guide_bol]},
        {("3", "G7-Cell19"): [rpt_guide, corrected_crosswalk_bol]},
    )

    assert status == "ACCEPTED"
    assert "crosswalk" in note


def test_independent_physics_features_are_finite_and_have_expected_means():
    features = recompute_physics_features(
        [4.2, 4.0, 3.8, 3.5, 3.2, 2.8, 2.5],
        [-1.5, -1.6, -1.4, -1.5, -1.5, -1.6, -1.5],
        [24.0, 24.5, 25.0, 25.5, 26.0, 26.5, 27.0],
    )

    assert features["mean_discharge_temperature_native"] == pytest.approx(25.5)
    assert features["discharge_temperature_rise_native"] == pytest.approx(3.0)
    assert features["mean_absolute_discharge_current_native"] == pytest.approx(1.5142857142857142)
    assert all(value == value for value in features.values())
