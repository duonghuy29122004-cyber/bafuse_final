import csv

import pytest

from bafuse.labeling import multimethod_soh_recovery as recovery


def test_native_capacity_ratio_and_method_difference_summary():
    assert recovery._ratio_percent(2.4, 3.0) == pytest.approx(80.0)
    assert recovery._ratio_percent(2.4, 0.0) is None

    summary = recovery._metric_summary(
        [
            {"discharge": 90.0, "charge": 92.0},
            {"discharge": 80.0, "charge": 79.0},
        ],
        "discharge",
        "charge",
    )

    assert summary["n"] == 2
    assert summary["mean_absolute_difference_pp"] == pytest.approx(1.5)
    assert summary["median_absolute_difference_pp"] == pytest.approx(1.5)
    assert summary["max_absolute_difference_pp"] == pytest.approx(2.0)
    assert summary["bias_right_minus_left_pp"] == pytest.approx(0.5)


def test_summary_reports_only_valid_pairs_without_dropping_accepted_exclusions():
    comparison = [
        {
            "original_disposition": "ACCEPTED",
            "m1_status": "VALID",
            "m2_status": "VALID",
            "m3_discharge_status": "VALID",
            "m3_charge_status": "VALID",
            "m1_invalid_reasons": "",
            "m2_invalid_reasons": "",
            "soh_discharge_integrated_percent": 90.0,
            "soh_charge_integrated_percent": 92.0,
            "soh_native_discharge_percent": 90.1,
            "soh_native_charge_percent": 92.1,
            "historical_m1_reference_percent_comparison_only": 90.0,
            "physical_cell_id": "G1-Cell1",
            "protocol_domain": "Maccor 4200; Parts 1-2",
        },
        {
            "original_disposition": "ACCEPTED",
            "m1_status": "REVIEW",
            "m2_status": "REVIEW",
            "m3_discharge_status": "REVIEW",
            "m3_charge_status": "REVIEW",
            "m1_invalid_reasons": "ambiguous_bol",
            "m2_invalid_reasons": "ambiguous_bol",
        },
    ]
    agreement = [
        {
            key: value
            for key, value in comparison[0].items()
            if key
            in {
                "soh_discharge_integrated_percent",
                "soh_charge_integrated_percent",
                "soh_native_discharge_percent",
                "soh_native_charge_percent",
                "physical_cell_id",
                "protocol_domain",
            }
        }
    ]

    summary = recovery._summaries(comparison, agreement)

    assert summary["accepted_M1_vs_M2"]["n"] == 1
    assert summary["accepted_records_not_in_valid_M1_M2_pairs"] == 1
    assert summary["accepted_pair_exclusion_reasons"] == {"ambiguous_bol": 1}


def test_step10_bol_selects_unique_complete_positive_charge_segment(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "PROJECT_ROOT", tmp_path)
    source = "data/raw/oxford/part1/Group_1/example.csv"
    path = tmp_path / source.replace("/", "\\")
    path.parent.mkdir(parents=True)
    rows = [
        ["10", "0", "0", "0.0", "0.1", "2.9"],
        ["10", "1", "1", "0.1", "0.1", "3.4"],
        ["10", "2", "2", "0.2", "0.1", "4.2"],
        ["11", "3", "3", "0.2", "0.0", "4.2"],
        ["10", "4", "4", "0.2", "-0.1", "3.9"],
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts"])
        writer.writerows(rows)

    selected, segments, status = recovery._select_primary_step10_bol(source, 1)

    assert status == "unique_complete_positive_step10_segment"
    assert len(segments) == 2
    assert selected is not None
    assert selected["row_count"] == 3
    assert selected["current_signs"] == {"positive": 3}
    assert selected["testtime_integral_abs_current_native"] == pytest.approx(0.2)
    assert selected["voltage_max"] == pytest.approx(4.2)


def test_step10_charge_block_rejects_repeated_time_and_wrong_direction():
    segment = recovery._Step10Segment(start_line=2)
    columns = {
        "step": 0,
        "testtime": 1,
        "steptime": 2,
        "amphr": 3,
        "amps": 4,
        "volts": 5,
    }
    segment.update(["10", "0", "0", "0", "0.1", "3.0"], columns, 2)
    segment.update(["10", "0", "1", "0.1", "-0.1", "4.2"], columns, 3)
    result = segment.finish("example.csv", 1, 11)

    reasons = recovery._step10_segment_reasons(result)

    assert "current_direction_not_consistently_positive" in reasons
    assert "testtime_not_strictly_increasing" in reasons


def test_review_outcome_keeps_charge_proxy_separate_from_discharge():
    outcome, explanation = recovery._review_outcome(
        {
            "m1_status": "REVIEW",
            "m2_status": "VALID",
            "m3_discharge_status": "VALID",
            "m1_invalid_reasons": "testtime_repeated_timestamp",
        }
    )

    assert outcome == "CHARGE_PROXY_ONLY"
    assert "does not resolve or replace" in explanation
