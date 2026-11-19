import unittest

import numpy as np
import pandas as pd

from run_hybrid_direction_v3 import (
    METRICS,
    _rate_policy,
    _select_v3_row,
    _validate_known_eol,
)


class H2HybridDirectionV3Tests(unittest.TestCase):
    def test_equal_nonnegative_rates_prefer_charge(self):
        rows = []
        prior_rates = []
        for metric in METRICS:
            for rpt, charge, discharge in ((1, 2.0, 3.0), (2, -2.0, -3.0)):
                rows.append(
                    {
                        "evaluation_domain": "external_part3",
                        "group": 7,
                        "physical_cell_id": "G7-Cell4",
                        "metric": metric,
                        "rpt_index": rpt,
                        "is_unnumbered_eol": False,
                        "charge_accepted": True,
                        "discharge_accepted": True,
                        f"charge_{metric}_percent": charge,
                        f"discharge_{metric}_percent": discharge,
                    }
                )
            prior_rates.append(
                {
                    "physical_cell_id": "G7-Cell4",
                    "metric": metric,
                    "paired_accepted_checkpoints": 2,
                    "charge_nonnegative_count": 1,
                    "charge_nonnegative_rate": 0.5,
                    "discharge_nonnegative_count": 1,
                    "discharge_nonnegative_rate": 0.5,
                    "h2_primary_direction": "tie",
                }
            )
        policies = _rate_policy(pd.DataFrame(rows), pd.DataFrame(prior_rates))
        self.assertEqual(len(policies), len(METRICS))
        self.assertTrue(policies["primary_direction"].eq("charge").all())
        self.assertTrue(policies["selection_reason"].eq("tie_prefer_charge").all())

    def test_no_paired_accepted_fits_remain_undetermined(self):
        rows = []
        prior_rates = []
        for metric in METRICS:
            rows.append(
                {
                    "evaluation_domain": "external_part3",
                    "group": 10,
                    "physical_cell_id": "G10-Cell2",
                    "metric": metric,
                    "rpt_index": 1,
                    "is_unnumbered_eol": False,
                    "charge_accepted": False,
                    "discharge_accepted": False,
                    f"charge_{metric}_percent": np.nan,
                    f"discharge_{metric}_percent": np.nan,
                }
            )
            prior_rates.append(
                {
                    "physical_cell_id": "G10-Cell2",
                    "metric": metric,
                    "paired_accepted_checkpoints": 0,
                    "charge_nonnegative_count": 0,
                    "charge_nonnegative_rate": np.nan,
                    "discharge_nonnegative_count": 0,
                    "discharge_nonnegative_rate": np.nan,
                    "h2_primary_direction": "undetermined",
                }
            )
        policies = _rate_policy(pd.DataFrame(rows), pd.DataFrame(prior_rates))
        self.assertTrue(policies["primary_direction"].eq("undetermined").all())
        self.assertTrue(policies["selection_reason"].eq("no_paired_accepted_checkpoints").all())

    def test_positive_primary_is_not_replaced_by_larger_opposite(self):
        result = _select_v3_row(
            {
                "metric": "Anode_LAM",
                "charge_accepted": True,
                "discharge_accepted": True,
                "charge_Anode_LAM_percent": 30.0,
                "discharge_Anode_LAM_percent": 4.0,
                "charge_status": "accepted",
                "charge_rmse_mV": 8.0,
                "charge_accepted_runs": 3,
                "discharge_status": "accepted",
                "discharge_rmse_mV": 9.0,
                "discharge_accepted_runs": 3,
            },
            "discharge",
        )
        self.assertEqual(result["h2_v3_selected_percent"], 4.0)
        self.assertEqual(result["h2_v3_source_direction"], "discharge")
        self.assertFalse(result["h2_v3_fallback_applied"])

    def test_negative_primary_uses_only_accepted_nonnegative_opposite(self):
        row = {
            "metric": "Anode_LAM",
            "charge_accepted": True,
            "discharge_accepted": True,
            "charge_Anode_LAM_percent": 16.8616,
            "discharge_Anode_LAM_percent": -50.0482,
            "charge_status": "accepted",
            "charge_rmse_mV": 10.0,
            "charge_accepted_runs": 3,
            "discharge_status": "accepted",
            "discharge_rmse_mV": 10.4,
            "discharge_accepted_runs": 3,
        }
        original = (row["charge_Anode_LAM_percent"], row["discharge_Anode_LAM_percent"])
        result = _select_v3_row(row, "discharge")
        self.assertAlmostEqual(result["h2_v3_selected_percent"], 16.8616)
        self.assertEqual(result["h2_v3_source_direction"], "charge")
        self.assertTrue(result["h2_v3_fallback_applied"])
        self.assertEqual(
            (row["charge_Anode_LAM_percent"], row["discharge_Anode_LAM_percent"]),
            original,
        )

    def test_negative_values_are_preserved_without_positive_opposite(self):
        result = _select_v3_row(
            {
                "metric": "LLI",
                "charge_accepted": True,
                "discharge_accepted": True,
                "charge_LLI_percent": -8.0,
                "discharge_LLI_percent": -3.0,
                "discharge_status": "accepted",
                "discharge_rmse_mV": 9.0,
                "discharge_accepted_runs": 3,
            },
            "discharge",
        )
        self.assertEqual(result["h2_v3_selected_percent"], -3.0)
        self.assertEqual(result["h2_v3_fallback_reason"], "unresolved_negative")

    def test_missing_primary_does_not_get_a_positive_opposite_substitution(self):
        result = _select_v3_row(
            {
                "metric": "Cathode_LAM",
                "charge_accepted": True,
                "discharge_accepted": False,
                "charge_Cathode_LAM_percent": 5.0,
                "discharge_Cathode_LAM_percent": np.nan,
            },
            "discharge",
        )
        self.assertTrue(np.isnan(result["h2_v3_selected_percent"]))
        self.assertFalse(result["h2_v3_fallback_applied"])
        self.assertEqual(result["h2_v3_fallback_reason"], "primary_fit_missing_or_rejected")

    def test_undetermined_policy_does_not_select_a_direction(self):
        result = _select_v3_row(
            {
                "metric": "LLI",
                "charge_accepted": True,
                "discharge_accepted": True,
                "charge_LLI_percent": 3.0,
                "discharge_LLI_percent": 4.0,
            },
            "undetermined",
        )
        self.assertTrue(np.isnan(result["h2_v3_selected_percent"]))
        self.assertFalse(result["h2_v3_selected_accepted"])
        self.assertEqual(result["h2_v3_fallback_reason"], "no_paired_accepted_checkpoints")

    def test_both_known_unnumbered_eol_values_are_charge_fits(self):
        rows = []
        for cell, value in (("G1-Cell20", 16.8616), ("G3-Cell14", 19.1026)):
            rows.append(
                {
                    "physical_cell_id": cell,
                    "is_unnumbered_eol": True,
                    "metric": "Anode_LAM",
                    "h2_v3_selected_percent": value,
                    "h2_v3_source_direction": "charge",
                    "charge_accepted": True,
                    "rpt_index": np.nan,
                }
            )
        _validate_known_eol(pd.DataFrame(rows))


if __name__ == "__main__":
    unittest.main()
