from pathlib import Path
import json
import math
import re

import pytest

from app.part3_modality_eligibility import (
    CSV_FIELDS,
    REQUIRED_CSV_FIELDS,
    build_audit_rows,
    summarize,
    write_audit,
    write_report_from_csv,
)


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def audit_rows():
    return build_audit_rows(ROOT)


def _find(rows, cell, rpt):
    return next(
        row
        for row in rows
        if row["physical_cell_id"] == cell and row["rpt_index"] == str(rpt)
    )


def test_audit_covers_every_canonical_part3_rpt_and_final2_status(audit_rows):
    counts = summarize(audit_rows)

    assert counts["rpt_rows"] == 87
    assert counts["cells"] == 12
    assert counts["groups"] == {"10": 34, "7": 17, "8": 18, "9": 18}
    assert counts["final2_status"] == {"ACCEPTED": 74, "REJECTED": 3, "REVIEW": 10}
    assert counts["raw_scanned"] == 87
    assert counts["raw_present"] == 87
    assert counts["discharge_only_inference_eligible"] == 84
    assert counts["two_modality_inference_eligible"] == 82
    assert counts["physics5_valid"] == 82
    assert counts["three_modality_inference_eligible"] == 82
    assert counts["bol_reference_valid"] == 87
    assert counts["soh_label_available"] == 84
    assert counts["label_evaluation_eligible"] == 74
    assert all(row["source_size_matches_reference"] == "true" for row in audit_rows)
    assert sum(row["source_mtime_matches_reference"] == "true" for row in audit_rows) == 77
    assert all(row["temperature_inference_eligible"] == "false" for row in audit_rows)
    assert all(row["eis_inference_eligible"] == "false" for row in audit_rows)


def test_review_candidate_can_be_two_modality_eligible_but_not_evaluable(audit_rows):
    row = _find(audit_rows, "G7-Cell19", 1)

    assert row["final2_status"] == "REVIEW"
    assert row["step4_valid"] == "true"
    assert row["step10_valid"] == "true"
    assert row["two_modality_inference_eligible"] == "true"
    assert row["physics5_valid"] == "true"
    assert row["three_modality_inference_eligible"] == "true"
    assert row["soh_label_available"] == "true"
    assert row["soh_label_status"] == "candidate_only_review"
    assert row["label_evaluation_eligible"] == "false"
    assert row["physics5_value_source"] == "computed_raw_step4"


def test_curve_blockers_are_not_hidden_by_other_valid_modality(audit_rows):
    rejected = _find(audit_rows, "G7-Cell19", 2)
    unresolved = _find(audit_rows, "G10-Cell22", 12)

    assert rejected["final2_status"] == "REJECTED"
    assert rejected["step4_inference_eligible"] == "false"
    assert rejected["step10_inference_eligible"] == "true"
    assert rejected["step4_source_scan_status"] == "too_few_step4_capacity_rows"
    assert rejected["inference_eligible"] == "false"
    assert rejected["label_evaluation_eligible"] == "false"

    assert unresolved["final2_mapping_status"] == "needs_curve_audit"
    assert unresolved["step4_inference_eligible"] == "true"
    assert unresolved["step10_inference_eligible"] == "false"
    assert "Raw complete protocol blocks=0" in unresolved["raw_mapping_audit_notes"]
    assert unresolved["soh_label_available"] == "true"
    assert unresolved["label_evaluation_eligible"] == "false"


def test_only_approved_isolated_step10_zero_current_row_is_excluded(audit_rows):
    row = _find(audit_rows, "G10-Cell2", 10)
    ordinary = _find(audit_rows, "G10-Cell22", 10)

    assert row["final2_status"] == "ACCEPTED"
    assert row["step10_zero_current_policy"] == "exclude_zero_current"
    assert row["step10_zero_current_excluded_count"] == "1"
    assert row["step10_zero_current_source_line"] == "2880746"
    assert row["step10_inference_eligible"] == "true"
    assert row["step10_block_sha256"] == (
        "9018c7b5306f2793f978d3aa912ec6c5c4da1570667c9ea0c5b4dccb0a016888"
    )
    assert "voltage decreases kept" in row["step10_evidence"]
    assert row["label_evaluation_eligible"] == "true"
    assert row["soh_evaluation_eligible"] == "true"
    assert ordinary["step10_zero_current_policy"] == "strict"
    assert ordinary["step10_zero_current_excluded_count"] == "0"


def test_five_physics_features_are_verified_independently_of_step4(audit_rows):
    review = _find(audit_rows, "G7-Cell19", 1)
    extension = _find(audit_rows, "G10-Cell2", 10)

    assert review["step4_valid"] == "true"
    assert review["step10_valid"] == "true"
    assert review["physics5_valid"] == "true"
    assert "Computed by the existing build_p5_electrical_features" in review["physics5_evidence"]
    assert review["physics5_segment_sha256"]
    assert int(review["physics5_segment_row_count"]) > 0
    assert extension["physics5_valid"] == "true"
    assert "G10-Cell2 extension" in extension["physics5_evidence"]
    assert sum(row["physics5_valid"] == "true" for row in audit_rows) == 82
    assert sum("exact float32 match" in row["physics5_evidence"] for row in audit_rows) == 64
    assert sum("G10-Cell2 extension" in row["physics5_evidence"] for row in audit_rows) == 10
    recomputed = [row for row in audit_rows if row["physics5_value_source"] == "computed_raw_step4"]
    expected_recomputed = {
        ("G7-Cell4", 1),
        ("G7-Cell19", 1),
        ("G7-Cell27", 1),
        ("G8-Cell20", 1),
        ("G10-Cell22", 2),
        ("G10-Cell22", 4),
        ("G10-Cell22", 11),
        ("G10-Cell25", 11),
    }
    assert {(row["physical_cell_id"], int(row["rpt_index"])) for row in recomputed} == expected_recomputed
    for row in recomputed:
        values = json.loads(row["physics5_values_json"])
        assert tuple(values) == tuple(row["physics5_feature_names"].split(";"))
        assert len(values) == 5
        assert all(math.isfinite(value) for value in values.values())
        assert re.fullmatch(r"[0-9a-f]{64}", row["physics5_segment_sha256"])
        assert row["soh_evaluation_eligible"] == "false"


def test_3m_inference_does_not_depend_on_saved_tensor_or_review_labels(audit_rows):
    review_rows = [
        row
        for row in audit_rows
        if row["final2_status"] == "REVIEW"
        and row["step4_valid"] == "true"
        and row["step10_valid"] == "true"
    ]

    assert len(review_rows) == 8
    assert all(row["three_modality_inference_eligible"] == "true" for row in review_rows)
    assert all(row["soh_evaluation_eligible"] == "false" for row in review_rows)
    assert summarize(audit_rows)["label_evaluation_eligible"] == 74


def test_raw_p5_keeps_step4_zero_current_samples_and_exact_mapped_span(audit_rows):
    for rpt in (2, 4):
        row = _find(audit_rows, "G10-Cell22", rpt)
        reference = row["step4_segment_reference"].rsplit("rows", 1)[1]
        start, end = (int(value) for value in reference.split("-"))

        assert row["step4_valid"] == "true"
        assert row["physics5_value_source"] == "computed_raw_step4"
        assert int(row["physics5_segment_row_count"]) == end - start + 1
        assert "neutral zero-current samples retained" in row["physics5_evidence"]


def test_bol_reference_and_pydma_charge_are_row_mapped(audit_rows):
    assert all(row["bol_reference_valid"] == "true" for row in audit_rows)
    assert all(row["pydma_charge_available"] == "true" for row in audit_rows)
    assert {
        status: sum(row["pydma_charge_qc_status"] == status for row in audit_rows)
        for status in ("accepted", "rejected", "missing")
    } == {"accepted": 86, "rejected": 1, "missing": 0}

    rpt10 = _find(audit_rows, "G10-Cell2", 10)
    assert rpt10["pydma_charge_qc_status"] == "rejected"
    assert rpt10["pydma_charge_saved_qc_status"] == "diagnostic_only_final2_rejected"
    assert rpt10["pydma_charge_fit_gate_passed"] == "true"
    assert rpt10["pydma_charge_accepted_runs"] == "3"
    assert float(rpt10["pydma_charge_fit_rmse_mV"]) < 15
    assert rpt10["final2_observation_status"] == "ACCEPTED"


def test_csv_writer_preserves_auditable_columns():
    destination = ROOT / "app/part3_modality_eligibility.csv"
    rows = write_audit(ROOT)

    with destination.open("r", encoding="utf-8-sig", newline="") as stream:
        import csv

        reader = csv.DictReader(stream)
        written = list(reader)

    assert tuple(reader.fieldnames) == CSV_FIELDS
    assert tuple(reader.fieldnames[: len(REQUIRED_CSV_FIELDS)]) == REQUIRED_CSV_FIELDS
    assert len(written) == len(rows) == 87
    assert written[0]["canonical_record_id"]
    assert written[0]["audit_evidence_paths"]


def test_report_counts_and_ineligible_lists_are_read_from_csv(audit_rows):
    import csv

    csv_path = ROOT / "app/.part3_modality_report_test.csv"
    report_path = ROOT / "app/.part3_modality_report_test.md"
    modified = [dict(row) for row in audit_rows]
    changed = _find(modified, "G7-Cell19", 3)
    changed["three_modality_inference_eligible"] = "false"
    changed["soh_evaluation_eligible"] = "false"
    try:
        with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(modified)
        report = write_report_from_csv(csv_path, report_path, ROOT)
        assert "Đủ điều kiện: **81**; không đủ điều kiện: **6**." in report
        assert "G7-Cell19/RPT3" in report
        assert "86 accepted, 1 rejected, 0 missing" in report
        assert "CNN1D-Step4-Baseline" in report
        assert "CNN-3M" in report
        assert "### Baseline —" in report
        assert "### 2M —" in report
        assert "### 3M —" in report
        assert "| Part | Group | Cell | RPTs | Step4 |" in report
        assert "trained/frozen" in report
        assert "7 checkpoint" in report
        assert "**74** sample / **12** cell" in report
    finally:
        csv_path.unlink(missing_ok=True)
        report_path.unlink(missing_ok=True)
