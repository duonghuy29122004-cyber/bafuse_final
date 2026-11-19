from __future__ import annotations

import hashlib

import numpy as np
import pytest

from bafuse.hybrid.final2_dataset import (
    Final2Arrays,
    _candidate_status,
    _filter_zero_current_step10,
    _hash_raw_segments,
    compute_soh_percent,
    ensure_cell_disjoint,
    expected_bol_source_part,
    fit_final2_scalers,
    transform_final2_arrays,
)


def _arrays(
    discharge: np.ndarray | None = None,
    charge: np.ndarray | None = None,
    physics: np.ndarray | None = None,
    target: np.ndarray | None = None,
) -> Final2Arrays:
    return Final2Arrays(
        X_discharge=(
            np.asarray(discharge, dtype=np.float32)
            if discharge is not None
            else np.asarray([[1.0] * 128, [3.0] * 128], dtype=np.float32)
        ),
        X_charge=(
            np.asarray(charge, dtype=np.float32)
            if charge is not None
            else np.asarray([[5.0] * 128, [9.0] * 128], dtype=np.float32)
        ),
        X_physics_raw=(
            np.asarray(physics, dtype=np.float32)
            if physics is not None
            else np.asarray([[1.0] * 5, [3.0] * 5], dtype=np.float32)
        ),
        y_soh=(
            np.asarray(target, dtype=np.float32)
            if target is not None
            else np.asarray([[80.0], [100.0]], dtype=np.float32)
        ),
    )


def test_p5_capacity_ratio_and_bol_policy_are_explicit():
    assert compute_soh_percent(2.4, 3.0) == pytest.approx(80.0)
    assert expected_bol_source_part(1, 3) == 1
    assert expected_bol_source_part(2, 4) == 1
    assert expected_bol_source_part(2, 5) is None
    assert expected_bol_source_part(2, 6) is None
    assert expected_bol_source_part(3, 10) == 3
    with pytest.raises(ValueError, match="must be positive"):
        compute_soh_percent(2.0, 0.0)


def test_scaler_fit_is_blocked_until_cohort_and_training_are_authorized():
    with pytest.raises(RuntimeError, match="cohort is complete"):
        fit_final2_scalers(
            _arrays(),
            accepted_cohort_complete=False,
            training_authorized=True,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        fit_final2_scalers(
            _arrays(),
            accepted_cohort_complete=True,
            training_authorized=False,
        )


def test_variant_d_scalers_fit_only_the_supplied_final2_train_rows():
    train = _arrays()
    scalers = fit_final2_scalers(
        train,
        accepted_cohort_complete=True,
        training_authorized=True,
    )
    assert scalers.discharge.fit_row_count == 2
    assert scalers.charge.fit_row_count == 2
    assert scalers.physics.fit_row_count == 2
    assert scalers.target.fit_row_count == 2

    validation = _arrays(
        discharge=np.full((1, 128), 1000.0),
        charge=np.full((1, 128), -1000.0),
        physics=np.full((1, 5), 500.0),
        target=np.asarray([[10000.0]]),
    )
    transformed = transform_final2_arrays(validation, scalers)
    np.testing.assert_allclose(scalers.discharge.mean, np.full(128, 2.0))
    np.testing.assert_allclose(scalers.charge.mean, np.full(128, 7.0))
    np.testing.assert_allclose(scalers.target.mean, [90.0])
    assert transformed["X_discharge"].shape == (1, 128)
    assert transformed["X_charge"].shape == (1, 128)
    assert transformed["X_physics"].shape == (1, 5)
    assert transformed["y_soh"].shape == (1, 1)


def test_step10_zero_current_filter_preserves_all_positive_rows_and_voltage_dips():
    voltage = np.asarray([3.84, 3.85, 3.83, 3.90], dtype=np.float64)
    block = {
        "current_values": [1.0, 0.0, 1.0, 1.0],
        "voltage_values": voltage.tolist(),
        "capacity_values": [0.0, 0.5, 0.7, 1.2],
        "testtime_values": [0.0, 0.2, 0.3, 1.0],
        "steptime_values": [0.0, 0.2, 0.3, 1.0],
        "row_count": 4,
        "start_line": 20,
        "raw_block_sha256": "a" * 64,
        "positive_current_row_count": 3,
        "positive_current_rows_sha256": "b" * 64,
        "zero_current_rows": [
            {
                "line": 21,
                "testtime": 0.2,
                "steptime": 0.2,
                "amphr": 0.5,
                "amps": 0.0,
                "volts": 3.85,
            }
        ],
    }

    filtered_voltage, audit = _filter_zero_current_step10(
        block,
        reference="TPG10.19-Cell2.csv#Step10:segment1:rows20-23",
    )

    assert filtered_voltage is not None
    np.testing.assert_array_equal(filtered_voltage, [3.84, 3.83, 3.90])
    assert audit["raw_block_sha256"] == "a" * 64
    assert audit["positive_current_rows_sha256"] == "b" * 64
    assert audit["zero_current_excluded_count"] == 1
    assert audit["retained_voltage_decrease_count"] == 1
    assert audit["joined_intervals_across_excluded_rows"][0]["left_line"] == 20
    assert audit["joined_intervals_across_excluded_rows"][0]["right_line"] == 22
    assert audit["reasons"] == []


def test_cell_split_guard_rejects_cross_partition_cells():
    with pytest.raises(ValueError, match="Cell leakage detected"):
        ensure_cell_disjoint(
            {
                "train": ["G1-Cell1", "G1-Cell2"],
                "validation": ["G1-Cell2"],
                "external_test": ["G3-Cell4"],
            }
        )


def test_candidate_status_quarantines_groups_and_rejects_missing_step(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("placeholder", encoding="utf-8")
    relative = "source.csv"
    p5_records = {
        (1, "G1-Cell1", 1): {
            "source_file": relative,
            "source_available": True,
        }
    }
    row = {
        "group": "1",
        "source_file": relative,
        "partition": "train",
        "source_exists": "True",
        "step4_valid": "True",
        "step10_valid": "True",
        "final_mapping_status": "verified",
    }
    assert _candidate_status(
        row,
        key=(1, "G1-Cell1", 1),
        p5_records=p5_records,
        project_root=tmp_path,
    )[0] == "ACCEPTED"

    group5 = dict(row, group="5")
    assert _candidate_status(
        group5,
        key=(2, "G5-Cell1", 1),
        p5_records={},
        project_root=tmp_path,
    )[0] == "REVIEW"

    invalid_curve = dict(row, step10_valid="False")
    assert _candidate_status(
        invalid_curve,
        key=(1, "G1-Cell2", 1),
        p5_records={},
        project_root=tmp_path,
    )[0] == "REJECTED"


def test_guide_verified_mapping_requires_a_p5_label_before_acceptance(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("placeholder", encoding="utf-8")
    row = {
        "group": "10",
        "source_file": "source.csv",
        "partition": "external_test",
        "source_exists": "True",
        "step4_valid": "True",
        "step10_valid": "True",
        "final_mapping_status": "GUIDE_VERIFIED",
    }

    status, _, _ = _candidate_status(
        row,
        key=(3, "G10-Cell2", 1),
        p5_records={},
        project_root=tmp_path,
    )
    assert status == "REVIEW"

    status, reason, partition = _candidate_status(
        row,
        key=(3, "G10-Cell2", 1),
        p5_records={
            (3, "G10-Cell2", 1): {
                "source_file": "source.csv",
                "source_available": True,
                "label_source": "final2_g10_p5_extension",
            }
        },
        project_root=tmp_path,
    )
    assert (status, partition) == ("ACCEPTED", "external_test")
    assert "Final2-only P5 label extension" in reason


def test_raw_curve_ineligibility_rejects_owner_verified_mapping(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("placeholder", encoding="utf-8")
    row = {
        "group": "10",
        "source_file": "source.csv",
        "partition": "external_test",
        "source_exists": "True",
        "step4_valid": "True",
        "step10_valid": "True",
        "final_mapping_status": "GUIDE_VERIFIED",
    }

    status, reason, partition = _candidate_status(
        row,
        key=(3, "G10-Cell2", 10),
        p5_records={},
        project_root=tmp_path,
        curve_eligibility={
            (3, "G10-Cell2", 10): {
                "step4_valid": True,
                "step10_valid": False,
                "step4_reasons": [],
                "step10_reasons": ["current_direction_not_consistent"],
            }
        },
    )
    assert status == "REJECTED"
    assert partition == "external_test"
    assert "Step 10" in reason
    assert "current_direction_not_consistent" in reason


def test_hash_raw_segments_uses_exact_csv_rows(tmp_path):
    source = tmp_path / "segments.csv"
    rows = [
        "Step,TestTime,Amps",
        "4,0,-1.0",
        "4,1,-1.0",
        "10,2,1.0",
    ]
    source.write_text("\n".join(rows) + "\n", encoding="utf-8")
    expected = hashlib.sha256(
        "4\x1f0\x1f-1.0\n4\x1f1\x1f-1.0\n".encode("utf-8")
    ).hexdigest()

    assert _hash_raw_segments(source, [(2, 3)]) == {(2, 3): expected}
