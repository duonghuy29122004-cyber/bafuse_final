import unittest

import pandas as pd

from run_trajectory_audit import (
    METHODS,
    _classify_delta,
    _mark_common_checkpoints,
    _trajectory_stats,
)


def _point(rpt, position, value, *, eol=False, checkpoint=None):
    return {
        "evaluation_domain": "parts1_2_internal",
        "group": 1,
        "physical_cell_id": "G1-Cell1",
        "method": "H1 Hybrid",
        "metric": "LLI",
        "sensitivity": "regular_only",
        "checkpoint_scope": "all_available",
        "checkpoint_key": checkpoint or f"P1|G1-Cell1|RPT{rpt}",
        "rpt_index": rpt,
        "rpt_or_checkpoint": checkpoint or f"RPT{rpt}",
        "part": 1,
        "position": position,
        "chronological_position": position,
        "is_eol": eol,
        "mapping_verified": True,
        "value_percent": value,
        "method_source_direction": "charge",
        "method_primary_direction": "charge",
        "method_fallback_applied": False,
        "method_fit_status": "accepted",
        "method_fit_rmse_mV": 8.0,
        "method_fit_accepted_runs": 3,
    }


class TrajectoryAuditTests(unittest.TestCase):
    def test_increase_decrease_and_tie_classification(self):
        self.assertEqual(_classify_delta(0.5), "increase")
        self.assertEqual(_classify_delta(-0.5), "decrease")
        self.assertEqual(_classify_delta(0.0), "tie")

    def test_gap_is_reported_but_not_counted_as_consecutive_rpt(self):
        points = pd.DataFrame(
            [_point(1, 1, 1.0), _point(3, 3, 3.0), _point(4, 4, 2.0)]
        )
        stats, transitions = _trajectory_stats(
            points,
            "regular_only",
            "all_available",
            {
                "evaluation_domain": "parts1_2_internal",
                "group": 1,
                "physical_cell_id": "G1-Cell1",
                "metric": "LLI",
                "method": "H1 Hybrid",
            },
        )
        self.assertEqual(stats["valid_checkpoint_count"], 3)
        self.assertEqual(stats["adjacent_observed_transition_count"], 2)
        self.assertEqual(stats["consecutive_verified_rpt_transition_count"], 1)
        self.assertEqual(stats["nonconsecutive_observed_transition_count"], 1)
        self.assertEqual(transitions[0]["missing_intermediate_rpt_count"], 1)
        self.assertFalse(transitions[0]["is_consecutive_verified_rpt_transition"])

    def test_unnumbered_eol_is_ordered_without_inventing_an_rpt_index(self):
        points = pd.DataFrame(
            [
                _point(1, 1, 1.0),
                _point(2, 2, 2.0),
                _point(None, 3, 2.5, eol=True, checkpoint="EOL-UNNUMBERED"),
            ]
        )
        points.loc[2, "rpt_index"] = None
        stats, transitions = _trajectory_stats(
            points,
            "eol_inclusive",
            "all_available",
            {
                "evaluation_domain": "parts1_2_internal",
                "group": 1,
                "physical_cell_id": "G1-Cell1",
                "metric": "LLI",
                "method": "H1 Hybrid",
            },
        )
        self.assertEqual(stats["eol_adjacent_transition_count"], 1)
        self.assertEqual(transitions[-1]["transition_type"], "eol_adjacent")
        self.assertIsNone(transitions[-1]["rpt_gap"])
        self.assertIsNone(transitions[-1]["right_rpt_index"])

    def test_trend_classification_requires_three_valid_checkpoints(self):
        points = pd.DataFrame([_point(1, 1, 1.0), _point(2, 2, 2.0)])
        stats, _ = _trajectory_stats(
            points,
            "regular_only",
            "all_available",
            {
                "evaluation_domain": "parts1_2_internal",
                "group": 1,
                "physical_cell_id": "G1-Cell1",
                "metric": "LLI",
                "method": "H1 Hybrid",
            },
        )
        self.assertFalse(stats["eligible_for_trend_classification"])
        self.assertIsNone(stats["overall_increasing_trend"])

    def test_common_checkpoint_set_is_intersection_of_all_four_methods(self):
        rows = []
        for method in METHODS:
            for rpt in (1, 2):
                if method == "H2 Hybrid" and rpt == 2:
                    continue
                rows.append(
                    {
                        "evaluation_domain": "parts1_2_internal",
                        "group": 1,
                        "physical_cell_id": "G1-Cell1",
                        "metric": "LLI",
                        "method": method,
                        "checkpoint_key": f"P1|G1-Cell1|RPT{rpt}",
                        "valid_value": True,
                        "is_eol": False,
                    }
                )
        marked = _mark_common_checkpoints(pd.DataFrame(rows))
        shared = marked[marked["common_regular_checkpoint"]]
        self.assertEqual(set(shared["checkpoint_key"]), {"P1|G1-Cell1|RPT1"})
        self.assertEqual(shared["method"].nunique(), 4)


if __name__ == "__main__":
    unittest.main()
