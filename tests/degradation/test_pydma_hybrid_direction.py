import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments" / "pydma_hybrid_direction_v1" / "run_hybrid_direction.py"
SPEC = importlib.util.spec_from_file_location("pydma_hybrid_direction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
hybrid = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hybrid)


def _fit(charge, discharge):
    return {
        "charge_accepted": charge is not None,
        "charge_LLI_percent": charge,
        "discharge_accepted": discharge is not None,
        "discharge_LLI_percent": discharge,
    }


def test_rpt_parser_does_not_read_part_number_as_rpt():
    assert hybrid._rpt("P2/RPT18 (TPG1.27)") == 18
    assert hybrid._rpt("7.0") == 7
    assert hybrid._rpt("UNNUMBERED") is None


def test_verified_stitched_eol_curve_is_not_rejected_for_missing_documented_channels():
    assert hybrid._valid_segment(
        "verified_unique_curve_stitched_across_missing_documented_channels"
    )
    assert not hybrid._valid_segment("no_valid_segment")
    assert not hybrid._valid_segment("ambiguous_verified_curve")


def test_h2_positive_rate_ignores_unnumbered_part2_eol():
    import pandas as pd

    frame = pd.DataFrame(
        [
            {
                "physical_cell_id": "G1-Cell1",
                "mapping_verified": True,
                "is_unnumbered_eol": False,
                "charge_accepted": True,
                "charge_LLI_percent": 1.0,
                "discharge_accepted": True,
                "discharge_LLI_percent": -1.0,
            },
            {
                "physical_cell_id": "G1-Cell1",
                "mapping_verified": True,
                "is_unnumbered_eol": True,
                "charge_accepted": True,
                "charge_LLI_percent": -2.0,
                "discharge_accepted": True,
                "discharge_LLI_percent": 2.0,
            },
        ]
    )

    result = hybrid._direction_rates(frame).iloc[0]

    assert result["paired_accepted_checkpoints"] == 1
    assert result["h2_primary_direction"] == "charge"


def test_bound_parser_uses_saved_parameter_bounds_json():
    bounds = [
        {"parameter": "alpha_an", "exact_or_near_bound": True},
        {"parameter": "beta_an", "exact_or_near_bound": False},
    ]

    raw, count, any_bound = hybrid._bound_details(
        {"parameter_bounds_json": json.dumps(bounds)}
    )

    assert raw
    assert count == 1
    assert any_bound is True


def test_h1_uses_charge_for_complete_nonnegative_sequence():
    sequence = [_fit(2.0, -1.0), _fit(3.0, 4.0)]

    selected, policy = hybrid._select_h1(sequence, "LLI")

    assert policy == {
        "primary_direction": "charge",
        "charge_complete_coverage": True,
        "all_charge_nonnegative": True,
    }
    assert [row["h1_selected_percent"] for row in selected] == [2.0, 3.0]
    assert all(row["h1_source_direction"] == "charge" for row in selected)


def test_h1_only_falls_back_for_negative_discharge():
    sequence = [_fit(2.0, -1.0), _fit(0.0, 4.0), _fit(None, -2.0)]

    selected, _ = hybrid._select_h1(sequence, "LLI")

    assert [row["h1_selected_percent"] for row in selected] == [2.0, 4.0, -2.0]
    assert [row["h1_fallback_used"] for row in selected] == [True, False, False]


def test_h2_leaves_ties_unselected_and_uses_conditional_fallback():
    sequence = [_fit(2.0, -1.0)]

    tie = hybrid._select_h2(sequence, "LLI", "tie")[0]
    fallback = hybrid._select_h2(sequence, "LLI", "discharge")[0]

    assert tie["h2_selected_percent"] is None
    assert tie["h2_selection_status"] == "primary_tie_no_direction_selected"
    assert fallback["h2_selected_percent"] == 2.0
    assert fallback["h2_source_direction"] == "charge"
    assert fallback["h2_fallback_used"] is True


def test_temporal_summary_does_not_bridge_missing_checkpoint():
    import pandas as pd

    rows = []
    for position, estimate in enumerate((1.0, None, 3.0), start=1):
        rows.append(
            {
                "physical_cell_id": "G1-Cell1",
                "metric": "LLI",
                "method": "H1",
                "mapping_verified": True,
                "position": position,
                "part": 1,
                "rpt_or_checkpoint": f"RPT{position}",
                "checkpoint_id": "",
                "estimate_percent": estimate,
                "source_direction": "charge" if estimate is not None else "",
                "is_unnumbered_eol": False,
            }
        )

    result = hybrid._temporal_consistency(pd.DataFrame(rows)).iloc[0]

    assert result["missing_value_count"] == 1
    assert result["adjacent_evaluable_transitions"] == 0
    assert result["decrease_count"] == 0
    assert result["direction_switches"] == 0
