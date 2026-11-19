from __future__ import annotations

import json

import pandas as pd
import pytest

from bafuse.data.eol_coverage_v1 import (
    _consistency_row,
    _parameter_bound_flags,
    _parse_segment_ref,
    _read_pocv_components,
    _resolve_pocv_curve,
    _scan_tpg,
    _write_csv,
    candidate_capacity_ratio,
    checkpoint_identity,
    validate_eol_keys,
)
from app.hybrid_dashboard import _render_eol_checkpoint_section


def test_csv_writer_preserves_union_of_record_fields(tmp_path) -> None:
    output = tmp_path / "records.csv"
    _write_csv(
        output,
        [{"kind": "baseline", "rmse_mV": 8.0}, {"kind": "eol", "seed": 211}],
    )

    records = pd.read_csv(output)
    assert list(records.columns) == ["kind", "rmse_mV", "seed"]
    assert records.loc[0, "rmse_mV"] == 8.0
    assert pd.isna(records.loc[0, "seed"])
    assert pd.isna(records.loc[1, "rmse_mV"])
    assert records.loc[1, "seed"] == 211


def test_part2_eol_identity_remains_unnumbered() -> None:
    checkpoint = checkpoint_identity("2", "G1-Cell20", "TPG1.28", "")

    assert checkpoint == "EOL-P2-G1-Cell20-TPG1.28-UNNUMBERED"
    with pytest.raises(ValueError, match="must not be assigned"):
        checkpoint_identity("2", "G1-Cell20", "TPG1.28", "19")


def test_part3_eol_identity_preserves_guide_rpt_assignment() -> None:
    checkpoint = checkpoint_identity("3", "G7-Cell27", "TPG7.EoL", "6")

    assert checkpoint == "EOL-P3-G7-Cell27-RPT6-TPG7.EoL"


def test_capacity_ratio_requires_positive_verified_spans() -> None:
    value, status = candidate_capacity_ratio(2.4, 3.0)

    assert value == pytest.approx(80.0)
    assert status == "provisional_capacity_ratio_candidate_not_official"
    assert candidate_capacity_ratio(2.4, 0)[0] is None
    assert candidate_capacity_ratio(float("nan"), 3.0)[0] is None


def test_segment_ref_keeps_exact_rows_and_subsegment() -> None:
    parsed = _parse_segment_ref(
        "data/raw/oxford/part3/Group_7/TPG7.EoL-Cell27.csv"
        "#Step8:segment2:rows105-190:sub3"
    )

    assert parsed == {
        "source": "data/raw/oxford/part3/Group_7/TPG7.EoL-Cell27.csv",
        "step": 8,
        "segment_index": 2,
        "source_row_start": 105,
        "source_row_end": 190,
        "subsegment_index": 3,
    }


def test_pocv_curve_joins_only_monotone_ranges_across_missing_signal_rows(
    tmp_path,
) -> None:
    path = tmp_path / "tpg.csv"
    columns = [
        "Step",
        "Cyc",
        "TestTime",
        "StepTime",
        "Amphr",
        "Amps",
        "Volts",
        "VARx2",
        "VARx4",
    ]
    rows = [",".join(columns)]
    for index in range(12):
        rows.append(
            f"8,0,{index},{index},{index / 1000},-0.1,{4-index/1000},0,0"
        )
    rows.extend(["NaN,NaN,NaN,NaN,NaN,NaN,NaN,0,0"] * 2)
    for index in range(12, 24):
        rows.append(
            f"8,0,{index},{index},{index / 1000},-0.1,{4-index/1000},0,0"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    scan = _scan_tpg(path)
    segment, status = _resolve_pocv_curve(scan, path, 8)
    assert segment is not None
    assert status == "verified_unique_curve_stitched_across_missing_documented_channels"
    assert len(segment["components"]) == 2

    relative = path.relative_to(tmp_path).as_posix()
    capacity, voltage, audit = _read_pocv_components(
        tmp_path,
        relative,
        json.dumps(segment["components"]),
        8,
        "discharge",
    )

    assert len(capacity) == len(voltage) == 24
    assert audit["status"] == (
        "verified_exact_ranges_stitched_across_missing_documented_channels"
    )
    assert audit["capacity_span_native"] == pytest.approx(0.023)


def test_part3_rpt_assignment_cannot_duplicate_canonical_key() -> None:
    eol = [
        {
            "checkpoint_id": "EOL-P3-G7-Cell27-RPT6-TPG7.EoL",
            "part": "3",
            "physical_cell_id": "G7-Cell27",
            "rpt_index": "6",
            "guide_rpt_assignment": "6",
            "source_file": "data/raw/oxford/part3/TPG7.EoL-Cell27.csv",
        }
    ]
    canonical = [
        {"part": "3", "physical_cell_id": "G7-Cell27", "rpt_index": "6"}
    ]

    with pytest.raises(ValueError, match="duplicates a canonical RPT key"):
        validate_eol_keys(eol, canonical)


def test_charge_discharge_differences_are_descriptive_and_unclipped() -> None:
    row = _consistency_row(
        "Part 3",
        "G7-Cell27",
        "RPT6",
        [
            {
                "direction": "charge",
                "lli_percent": -2.0,
                "lam_anode_percent": 1.0,
                "lam_cathode_percent": 0.5,
                "accepted": True,
            },
            {
                "direction": "discharge",
                "lli_percent": 3.0,
                "lam_anode_percent": -1.0,
                "lam_cathode_percent": 0.5,
                "accepted": True,
            },
        ],
        "test",
    )

    assert row["lli_percent_charge"] == -2.0
    assert row["lli_percent_absolute_difference"] == 5.0
    assert row["lam_anode_percent_discharge"] == -1.0
    assert row["consistency_status"] == "difference_reported_no_acceptance_threshold"


def test_parameter_bound_flags_report_only_marked_parameters() -> None:
    assert (
        _parameter_bound_flags(
            '[{"parameter":"alpha_an","exact_or_near_bound":true},'
            '{"parameter":"beta_an","exact_or_near_bound":false}]'
        )
        == "alpha_an"
    )


def test_eol_dashboard_escapes_values_and_keeps_missingness_blank() -> None:
    record = {
        "checkpoint_id": "EOL-P2-G1-Cell20-TPG1.28-UNNUMBERED",
        "part_label": "Part 2 continuation",
        "physical_cell_id": "G1-Cell20",
        "guide_rpt_assignment": "",
        "direction": "charge",
        "modality_status": "D+C+P",
        "predicted_soh_percent": "",
        "raw_model_output_soh_percent": "-76.1",
        "provisional_reference_soh_percent": "81.25",
        "prediction_plausibility_status": (
            "negative_output_incompatible_with_positive_capacity_ratio"
        ),
        "soh_status": "not available",
        "fit_status": "<unsafe>",
        "fit_rmse_mV": "",
        "accepted_runs": "0",
        "lli_percent": "",
        "lam_anode_percent": "",
        "lam_cathode_percent": "",
        "negative_estimate_flags": "",
        "parameter_at_bound_flags": "",
        "baseline_reference_status": "verified",
        "mapping_status": "unnumbered EoL",
    }

    rendered = _render_eol_checkpoint_section(
        pd.DataFrame([record]),
        "eol_dashboard_records.csv",
    )

    assert "Separate EoL checkpoints" in rendered
    assert "&lt;unsafe&gt;" in rendered
    assert ">81.250<" in rendered
    assert ">-76.100<" in rendered
    assert "remain unnumbered" in rendered
