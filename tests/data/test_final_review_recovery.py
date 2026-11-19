import pytest

from bafuse.labeling import final_review_recovery as recovery


def test_guide_mapping_requires_one_cell_and_measurement_entry():
    rows = [
        {
            "dataset_part": "3",
            "physical_cell_id": "G9-Cell9",
            "measurement_id": "BoL",
            "source_file": "Group_9/TPG9.0-Cell9.csv",
        }
    ]

    assert recovery._guide_entry(
        rows, part=3, cell="G9-Cell9", measurement_id="BoL"
    ) == rows[0]
    with pytest.raises(ValueError, match="found 0"):
        recovery._guide_entry(rows, part=3, cell="G9-Cell9", measurement_id="RPT1")
    with pytest.raises(ValueError, match="found 2"):
        recovery._guide_entry(
            rows + [rows[0]], part=3, cell="G9-Cell9", measurement_id="BoL"
        )


def test_step4_protocol_gate_requires_monotone_discharge_to_documented_cutoff():
    rpt = {
        "testtime_zero_intervals": 0,
        "testtime_negative_intervals": 0,
        "steptime_zero_intervals": 0,
        "steptime_negative_intervals": 0,
        "current_max": -1.49,
        "voltage_first": 4.2,
        "voltage_last": 2.5,
        "voltage_min": 2.5,
    }
    bol = dict(rpt)

    assert recovery._validate_step4_pair(rpt, bol)
    with pytest.raises(ValueError, match="strictly increasing"):
        recovery._validate_step4_pair(
            {**rpt, "testtime_zero_intervals": 1}, bol
        )
    with pytest.raises(ValueError, match="discharge-only"):
        recovery._validate_step4_pair(
            {**rpt, "current_max": 0.0}, bol
        )
    with pytest.raises(ValueError, match="lowest sampled voltage"):
        recovery._validate_step4_pair({**rpt, "voltage_last": 2.51}, bol)


def test_trapezoidal_capacity_uses_absolute_current_and_ignores_zero_time_interval():
    area, zero_intervals, negative_intervals = recovery._integral(
        [0.0, 1.0, 1.0, 3.0], [-2.0, -2.0, -4.0, -4.0]
    )

    assert area == pytest.approx(10.0)
    assert zero_intervals == 1
    assert negative_intervals == 0
    assert recovery._ratio_percent(10.0, 12.5) == pytest.approx(80.0)


def test_proposal_status_requires_all_evidence_and_does_not_promote_by_default():
    original_label = {"label_disposition": "REVIEW", "status_promoted": False}

    assert recovery._proposal_status(True, True, True) == "PROMOTION_SUPPORTED"
    assert recovery._proposal_status(True, False, True) == "REVIEW_RETAINED"
    assert original_label == {"label_disposition": "REVIEW", "status_promoted": False}


def test_g2_timestamp_sensitivity_reports_both_axes_and_zero_change_after_collapse():
    rows = []
    for time in range(27):
        row = {"testtime": float(time), "steptime": float(time), "amps": -1.0}
        rows.append(row)
        if time < 26:
            rows.append(dict(row))
    rpt_testtime, _, _ = recovery._integral(
        [row["testtime"] for row in rows], [row["amps"] for row in rows]
    )
    rpt_steptime, _, _ = recovery._integral(
        [row["steptime"] for row in rows], [row["amps"] for row in rows]
    )
    summary = recovery._g2_investigation(
        {
            "rows": rows,
            "testtime_integrated_abs_current_native": rpt_testtime,
            "steptime_integrated_abs_current_native": rpt_steptime,
        },
        {
            "testtime_integrated_abs_current_native": 30.0,
            "steptime_integrated_abs_current_native": 30.0,
        },
    )

    assert summary["exact_duplicate_pairs_testtime"] == 26
    assert summary["exact_duplicate_pairs_steptime"] == 26
    assert summary["integral_delta_after_collapse_testtime"] == 0.0
    assert summary["integral_delta_after_collapse_steptime"] == 0.0
    assert summary["soh_testtime_percent"] == pytest.approx(100 * rpt_testtime / 30)
    assert summary["soh_steptime_percent"] == pytest.approx(100 * rpt_steptime / 30)


def test_split_verification_keeps_physical_cell_and_part3_external_only():
    candidates = [
        {
            "physical_cell_id": "G1-Cell1",
            "partition": "train",
            "dataset_part": "1",
        },
        {
            "physical_cell_id": "G1-Cell1",
            "partition": "train",
            "dataset_part": "2",
        },
        {
            "physical_cell_id": "G9-Cell9",
            "partition": "external_test",
            "dataset_part": "3",
        },
    ]
    splits = [
        {"physical_cell_id": "G1-Cell1", "partition": "train"},
        {"physical_cell_id": "G9-Cell9", "partition": "external_test"},
    ]

    recovery._verify_split(candidates, splits)
    with pytest.raises(ValueError, match="Partition mismatch"):
        recovery._verify_split(
            candidates + [
                {
                    "physical_cell_id": "G1-Cell1",
                    "partition": "validation",
                    "dataset_part": "2",
                }
            ],
            splits,
        )
    with pytest.raises(ValueError, match="External Test"):
        recovery._verify_split(
            [
                *candidates[:2],
                {
                    "physical_cell_id": "G9-Cell9",
                    "partition": "train",
                    "dataset_part": "3",
                },
            ],
            splits,
        )
