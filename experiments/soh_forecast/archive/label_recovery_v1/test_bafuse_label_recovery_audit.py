import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_adaptive_v1"
    / "phase_a_b_label_audit_v1"
)


def _read(name):
    with (OUTPUT / name).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def test_phase_a_b_accounts_for_every_missing_train_validation_label():
    rows = _read("missing_soh_label_audit.csv")
    assert len(rows) == 228
    assert sum(row["partition"] == "train" for row in rows) == 161
    assert sum(row["partition"] == "validation" for row in rows) == 67
    assert all(row["candidate_reason"] for row in rows)


def test_only_verified_part1_step4_candidates_are_accepted():
    rows = _read("recovered_soh_label_candidates.csv")
    accepted = [row for row in rows if row["candidate_decision"] == "accepted"]
    assert {(row["physical_cell_id"], row["rpt_index"]) for row in accepted} == {
        ("G3-Cell14", "11"),
        ("G3-Cell14", "12"),
    }
    for row in accepted:
        assert row["step4_step_ids"] == "4"
        assert row["step4_current_signs"] == "negative"
        assert row["step4_rows_observed"] == row["step4_capacity_finite_rows"]
        assert row["capacity_ratio_reliable"] == "TRUE"
        assert row["candidate_calculation"].startswith("100 * ")
        assert float(row["candidate_soh_percent"]) > 80


def test_unresolved_part2_candidates_are_not_promoted():
    rows = _read("recovered_soh_label_candidates.csv")
    part2 = [row for row in rows if row["dataset_part"] == "2"]
    assert len(part2) == 223
    assert all(row["candidate_decision"] == "requires_manual_review" for row in part2)
    assert sum(
        row["guide_mapping_confidence"] == "ambiguous_compound_stage"
        for row in part2
    ) == 210
    assert sum(
        row["guide_mapping_confidence"] == "single_checkpoint"
        and row["step4_boundary_status"] == "multiple_step4_segments"
        for row in part2
    ) == 13
    assert all(not row["step4_native_capacity_span"] for row in part2)
    assert all(not row["candidate_soh_percent"] for row in part2)


def test_g3_recovery_values_match_raw_step4_capacity_ratios():
    rows = {
        row["rpt_index"]: row
        for row in _read("recovered_soh_label_candidates.csv")
        if row["physical_cell_id"] == "G3-Cell14"
        and row["candidate_decision"] == "accepted"
    }
    expected = {"11": 82.497975278416, "12": 81.651045606845}
    assert set(rows) == set(expected)
    for rpt, value in expected.items():
        assert math.isclose(
            float(rows[rpt]["candidate_soh_percent"]),
            value,
            rel_tol=0,
            abs_tol=1e-9,
        )
