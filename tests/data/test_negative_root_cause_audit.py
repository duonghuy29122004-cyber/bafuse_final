from __future__ import annotations

import unittest

import pandas as pd

from bafuse.data.negative_root_cause_audit import (
    _cross_seed_matrix,
    _formula,
    _historical_records,
    _rpt,
)


class NegativeRootCauseAuditTests(unittest.TestCase):
    def test_rpt_normalization_handles_guide_spacing(self) -> None:
        self.assertEqual(_rpt("RPT 24"), "24")
        self.assertEqual(_rpt("RPT8"), "8")
        self.assertEqual(_rpt("BoL"), "0")

    def test_formula_preserves_negative_estimates(self) -> None:
        reference = {"alpha_an": 1.0, "beta_an": 0.0, "alpha_ca": 1.0, "beta_ca": 0.0}
        current = {"alpha_an": 1.2, "beta_an": 0.0, "alpha_ca": 1.1, "beta_ca": 0.0}
        labels = _formula(reference, current, 1.0, 1.0)
        self.assertAlmostEqual(labels["lam_anode_percent"], -20.0)
        self.assertAlmostEqual(labels["lam_cathode_percent"], -10.0)
        self.assertAlmostEqual(labels["lli_percent"], -10.0)

    def test_historical_snapshots_deduplicate_without_counting_attempts(self) -> None:
        row = {
            "result_table": "current",
            "dataset_part": "1",
            "physical_cell_id": "G1-Cell9",
            "checkup": "RPT2",
            "direction": "discharge",
            "test_step": "8",
            "source_file": "TPG1_8_Cell9.csv",
            "fit_rmse_mV": "14.1",
            "accepted_runs": "3",
            "total_runs": "3",
            "diagnostic_label_accepted": "True",
            "fit_status_original": "accepted",
            "lli_percent": "-2.0",
            "lam_anode_percent": "",
            "lam_cathode_percent": "1.0",
        }
        snapshot = {**row, "result_table": "pre_append"}
        records, raw_negative = _historical_records(pd.DataFrame([row, snapshot]))
        self.assertEqual(len(raw_negative), 2)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["historical_row_multiplicity"], 2)
        self.assertEqual(records[0]["negative_metrics"], "lli_percent")

    def test_historical_outputs_with_distinct_precision_are_not_merged(self) -> None:
        row = {
            "result_table": "current",
            "dataset_part": "1",
            "physical_cell_id": "G1-Cell9",
            "checkup": "RPT2",
            "direction": "discharge",
            "test_step": "8",
            "source_file": "TPG1_8_Cell9.csv",
            "fit_rmse_mV": "14.1",
            "accepted_runs": "3",
            "total_runs": "3",
            "diagnostic_label_accepted": "True",
            "fit_status_original": "accepted",
            "lli_percent": "-2.0",
            "lam_anode_percent": "",
            "lam_cathode_percent": "1.0",
        }
        alternate = {**row, "result_table": "alternate", "lli_percent": "-2.000000000001"}
        records, _ = _historical_records(pd.DataFrame([row, alternate]))
        self.assertEqual(len(records), 2)

    def test_cross_seed_matrix_keeps_baseline_and_target_seeds_separate(self) -> None:
        rows = []
        for role, seeds, alpha_an in (
            ("bol", ("211", "212"), 1.0),
            ("rpt", ("211", "212"), None),
        ):
            for seed in seeds:
                alpha = alpha_an if alpha_an is not None else (1.2 if seed == "211" else 0.9)
                fit_id = (
                    f"bol-p1-g1-cell9-rpt0-discharge-seed{seed}"
                    if role == "bol"
                    else f"rpt-p1-g1-cell9-rpt2-discharge-seed{seed}"
                )
                rows.append(
                    {
                        "fit_id": fit_id,
                        "seed": seed,
                        "fitted_parameters_json": (
                            '{"alpha_an": %s, "beta_an": 0, "alpha_ca": 1, "beta_ca": 0}'
                            % alpha
                        ),
                        "capacity_span_native": "1",
                        "rmse_mV": "8",
                        "accepted_runs": "3",
                    }
                )
        matrix = _cross_seed_matrix(pd.DataFrame(rows))
        self.assertEqual(len(matrix), 4)
        self.assertEqual(
            {row["rpt_seed"] for row in matrix if row["bol_seed"] == "211"},
            {"211", "212"},
        )
        self.assertTrue(
            all(row["lam_anode_percent"] < 0 for row in matrix if row["rpt_seed"] == "211")
        )
        self.assertTrue(
            all(row["lam_anode_percent"] > 0 for row in matrix if row["rpt_seed"] == "212")
        )


if __name__ == "__main__":
    unittest.main()
