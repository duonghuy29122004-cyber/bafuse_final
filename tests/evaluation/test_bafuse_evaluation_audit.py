from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
AUDIT_DIR = PHASE2_DIR / "evaluation_audit_v1"
sys.path.insert(0, str(PHASE2_DIR))
sys.path.insert(0, str(AUDIT_DIR))

import audit_soh_evaluation as audit
import run_phase2 as phase2
from app.hybrid_dashboard import _render_soh_audit_section


def test_overall_metrics_use_observation_level_formula() -> None:
    metrics = audit._metrics(
        np.asarray([1.0, 3.0]),
        np.asarray([2.0, 3.0]),
    )

    assert metrics["mae_soh_percentage_points"] == pytest.approx(0.5)
    assert metrics["rmse_soh_percentage_points"] == pytest.approx(2**-0.5)
    assert metrics["r2"] == pytest.approx(0.5)


def test_full_coverage_cnn1d_diagnostics_retain_cell_ids() -> None:
    rows = [
        {
            "partition": "external_test",
            "physical_cell_id": "G7-Cell4",
            "reference_soh_percent": 1.0,
            "predicted_soh_percent": 1.0,
        },
        {
            "partition": "external_test",
            "physical_cell_id": "G8-Cell5",
            "reference_soh_percent": 3.0,
            "predicted_soh_percent": 4.0,
        },
    ]

    metrics = audit._full_cnn_metrics(rows)

    assert metrics[0]["metric_scope"] == "overall"
    assert metrics[0]["physical_cell_id"] == ""
    assert {row["physical_cell_id"] for row in metrics[1:]} == {
        "G7-Cell4",
        "G8-Cell5",
    }


def test_unlabeled_segments_are_opt_in_for_audit_extraction() -> None:
    source = "data/raw/oxford/part3/Group_10/TPG10.2.csv"
    row = {
        "part": "3",
        "physical_cell_id": "G10-Cell2",
        "rpt_index": "1",
        "source_file": source,
        "soh_percent": None,
        "step4_status": "verified",
        "step4_segment_ref": f"{source}#Step4:segment1:rows2-10",
        "step10_status": "verified",
        "step10_segment_ref": f"{source}#Step10:segment1:rows12-20",
    }

    assert phase2._segment_requests([row]) == {}
    requests = phase2._segment_requests([row], include_unlabeled=True)

    assert len(requests) == 1
    assert len(next(iter(requests.values()))) == 2


def test_dashboard_renders_overall_coverage_and_escaped_exclusions() -> None:
    metrics = pd.DataFrame(
        [
            {
                "metric_scope": "overall",
                "partition": "external_test",
                "model": "three_encoder_fusion",
                "observation_count": 74,
                "physical_cell_count": 12,
                "mae_soh_percentage_points": 1.2,
                "rmse_soh_percentage_points": 1.8,
                "r2": 0.77,
            }
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "physical_cell_id": "G10-Cell2",
                "eligible_fusion_predictions": 10,
                "complete_modality_observations": 10,
                "eligible_full_coverage_cnn1d_predictions": 10,
                "available_step4_observations": 10,
                "valid_soh_reference_observations": 10,
                "verified_step10_charge_observations": 10,
                "valid_physics_feature_observations": 10,
            }
        ]
    )
    exclusions = pd.DataFrame(
        [{"exact_exclusion_reason": "<script>alert(1)</script>"}]
    )
    full_cnn1d = pd.DataFrame(
        [
            {
                "metric_scope": "overall",
                "observation_count": 82,
                "physical_cell_count": 12,
                "mae_soh_percentage_points": 1.1,
                "rmse_soh_percentage_points": 1.5,
                "r2": 0.82,
            }
        ]
    )

    markup = _render_soh_audit_section(
        metrics, coverage, exclusions, full_cnn1d, "report.md"
    )

    assert "External fusion cells<strong>1/12</strong>" in markup
    assert "Complete-modality observations<strong>10</strong>" in markup
    assert "82" in markup
    assert "&lt;script&gt;" in markup
    assert "<script>alert(1)</script>" not in markup
