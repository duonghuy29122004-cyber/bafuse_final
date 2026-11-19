import unittest

import pandas as pd

from run_hybrid_direction_v2 import assert_unique_selection_keys, select_positive_fallback


class PositiveFallbackTests(unittest.TestCase):
    def test_two_nonnegative_fits_keep_the_primary(self):
        result = select_positive_fallback("discharge", 8.0, 4.0, True, True)
        self.assertEqual(result["selected_percent"], 4.0)
        self.assertEqual(result["source_direction"], "discharge")
        self.assertFalse(result["fallback_applied"])

    def test_negative_discharge_uses_accepted_nonnegative_charge(self):
        result = select_positive_fallback("discharge", 8.0, -12.0, True, True)
        self.assertEqual(result["selected_percent"], 8.0)
        self.assertEqual(result["source_direction"], "charge")
        self.assertTrue(result["fallback_applied"])
        self.assertEqual(result["fallback_reason"], "negative_primary_positive_opposite")

    def test_negative_charge_uses_accepted_nonnegative_discharge(self):
        result = select_positive_fallback("charge", -9.0, 2.0, True, True)
        self.assertEqual(result["selected_percent"], 2.0)
        self.assertEqual(result["source_direction"], "discharge")
        self.assertTrue(result["fallback_applied"])

    def test_both_negative_preserves_signed_primary(self):
        result = select_positive_fallback("discharge", -50.0, -3.0, True, True)
        self.assertEqual(result["selected_percent"], -3.0)
        self.assertEqual(result["source_direction"], "discharge")
        self.assertEqual(result["fallback_reason"], "unresolved_negative")
        self.assertFalse(result["fallback_applied"])

    def test_missing_or_rejected_opposite_does_not_replace_primary(self):
        missing = select_positive_fallback("discharge", None, -6.0, False, True)
        rejected = select_positive_fallback("discharge", 9.0, -6.0, False, True)
        for result in (missing, rejected):
            self.assertEqual(result["selected_percent"], -6.0)
            self.assertEqual(result["source_direction"], "discharge")
            self.assertEqual(result["fallback_reason"], "fallback_unavailable")
            self.assertFalse(result["fallback_applied"])

    def test_eol_fallback_is_not_gated_by_previous_checkpoint(self):
        # Selection has no temporal input, so an EoL decrease cannot veto an accepted fallback.
        result = select_positive_fallback("discharge", 16.8616, -50.0482, True, True)
        self.assertEqual(result["selected_percent"], 16.8616)
        self.assertTrue(result["fallback_applied"])

    def test_metrics_are_selected_independently(self):
        metric_pairs = {
            "LLI": (5.0, -1.0, "discharge"),
            "Anode_LAM": (7.0, -2.0, "charge"),
            "Cathode_LAM": (-3.0, 4.0, "discharge"),
        }
        selected = {
            metric: select_positive_fallback(primary, charge, discharge, True, True)
            for metric, (charge, discharge, primary) in metric_pairs.items()
        }
        self.assertEqual(selected["LLI"]["source_direction"], "charge")
        self.assertEqual(selected["Anode_LAM"]["source_direction"], "charge")
        self.assertEqual(selected["Cathode_LAM"]["source_direction"], "discharge")
        self.assertEqual(selected["Cathode_LAM"]["selected_percent"], 4.0)

    def test_h2_tie_does_not_choose_a_direction(self):
        result = select_positive_fallback("tie", 4.0, 5.0, True, True)
        self.assertIsNone(result["selected_percent"])
        self.assertEqual(result["fallback_reason"], "primary_direction_tie")
        self.assertFalse(result["fallback_applied"])

    def test_original_signed_values_are_not_modified(self):
        original = {"charge": 16.8616, "discharge": -50.0482}
        select_positive_fallback(
            "discharge",
            original["charge"],
            original["discharge"],
            True,
            True,
        )
        self.assertEqual(original, {"charge": 16.8616, "discharge": -50.0482})

    def test_selection_keys_must_be_unique(self):
        row = {
            "part": 2,
            "physical_cell_id": "G1-Cell20",
            "rpt_or_checkpoint": "EOL-P2-G1-Cell20",
            "checkpoint_id": "EOL-P2-G1-Cell20",
            "metric": "Anode_LAM",
        }
        assert_unique_selection_keys(pd.DataFrame([row]))
        with self.assertRaises(ValueError):
            assert_unique_selection_keys(pd.DataFrame([row, row]))


if __name__ == "__main__":
    unittest.main()
