from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from bafuse.hybrid.hybrid_battery_health import (
    _aggregate_pydma,
    _build_checkpoint_manifest,
    _build_hybrid_results,
    _load_feature_rows,
    _load_pydma_records,
    _map_pydma_to_features,
    _predict_soh,
    _select_curve_records,
    _validate_prediction_parity,
)
from bafuse.labeling.pydma_policy import pydma_fit_passes_gate


class HybridBatteryHealthTests(unittest.TestCase):
    def test_existing_checkpoint_reproduces_saved_predictions(self) -> None:
        root = Path.cwd()
        frame, curves, _, model, checkpoint, _ = _load_feature_rows(root)
        predictions = _predict_soh(curves, model, checkpoint)
        self.assertEqual(len(predictions), 231)
        self.assertLess(_validate_prediction_parity(root, frame, predictions), 1e-4)
        self.assertTrue(frame["observation_step4_status"].eq("verified").all())
        self.assertTrue(frame["observation_ready_soh"].map(str.lower).eq("true").all())
        self.assertEqual(
            int(frame["observation_mapping_status"].eq("needs_curve_audit").sum()),
            13,
        )
        self.assertTrue(frame.loc[frame["dataset_part"] == "3", "partition"].eq("external_test").all())

    def test_pydma_outputs_carry_mapping_and_reference_provenance(self) -> None:
        rows = _load_pydma_records(Path.cwd())
        part2_cell3 = rows[
            (rows["dataset_part"] == 2)
            & (rows["physical_cell_id"] == "G2-Cell3")
            & rows["rpt_id"].ne("")
        ]
        self.assertFalse(part2_cell3.empty)
        self.assertTrue(
            part2_cell3["degradation_reference_status"]
            .eq("part1_origin_selected_step8_10_sequence_remains_provisional")
            .all()
        )
        self.assertTrue(part2_cell3["observation_mapping_status"].ne("").all())
        self.assertTrue(
            part2_cell3["observation_source_match_basis"]
            .eq("full_source_path_suffix")
            .all()
        )

    def test_mapping_provenance_reaches_dashboard_outputs_and_gates_fits(self) -> None:
        root = Path.cwd()
        frame, feature_curves, _, model, checkpoint, _ = _load_feature_rows(root)
        frame["predicted_soh_percent"] = _predict_soh(
            feature_curves, model, checkpoint
        )
        curve_rows = _map_pydma_to_features(_load_pydma_records(root), frame)
        aggregates = _aggregate_pydma(curve_rows)
        manifest = _build_checkpoint_manifest(frame, curve_rows)
        hybrid = _build_hybrid_results(frame, aggregates)

        checkpoint_row = manifest[
            (manifest["dataset_part"] == 2)
            & (manifest["physical_cell_id"] == "G2-Cell3")
            & (manifest["rpt_id"] == "RPT9")
        ].iloc[0]
        self.assertEqual(checkpoint_row["observation_step4_status"], "verified")
        self.assertEqual(
            checkpoint_row["degradation_reference_status"],
            "part1_origin_selected_step8_10_sequence_remains_provisional",
        )
        hybrid_row = hybrid[
            (hybrid["dataset_part"] == "2")
            & (hybrid["physical_cell_id"] == "G2-Cell3")
            & (hybrid["measurement_id"] == "RPT9")
        ].iloc[0]
        self.assertEqual(
            hybrid_row["pydma_discharge_reference_status"],
            "part1_origin_selected_step8_10_sequence_remains_provisional",
        )

        unresolved_fit = curve_rows[
            (curve_rows["physical_cell_id"] == "G1-Cell9")
            & (curve_rows["rpt_id"] == "RPT3")
            & (curve_rows["direction"] == "discharge")
            & curve_rows["strict_gate_pass"]
        ]
        self.assertFalse(unresolved_fit.empty)
        self.assertTrue(
            unresolved_fit["observation_curve_step_status"]
            .eq("needs_curve_audit")
            .all()
        )
        aggregate = aggregates[
            (aggregates["physical_cell_id"] == "G1-Cell9")
            & (aggregates["rpt_id"] == "RPT3")
            & (aggregates["direction"] == "discharge")
        ].iloc[0]
        self.assertEqual(aggregate["n_accepted_curves"], 1)
        self.assertEqual(aggregate["n_mapping_eligible_accepted_curves"], 0)
        self.assertEqual(aggregate["n_aggregated_curves"], 0)
        self.assertTrue(pd.isna(aggregate["lli_estimated_pct"]))

    def test_strict_gate_requires_three_runs_and_rmse_below_15(self) -> None:
        self.assertTrue(
            pydma_fit_passes_gate(rmse_mV=14.999, accepted_runs=3, is_accepted=True)
        )
        self.assertFalse(
            pydma_fit_passes_gate(rmse_mV=15.0, accepted_runs=3, is_accepted=True)
        )
        self.assertFalse(
            pydma_fit_passes_gate(rmse_mV=10.0, accepted_runs=2, is_accepted=True)
        )

    def test_candidate_only_group5_fits_have_no_consistency_claim(self) -> None:
        rows = _load_pydma_records(Path.cwd())
        candidate_only = rows[
            rows["table_source"] == "part2_g5_candidate_audit_only"
        ]
        self.assertFalse(candidate_only.empty)
        self.assertTrue(candidate_only["candidate_values_consistent"].isna().all())

    def test_curve_mapping_requires_exact_source_file(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "dataset_part": "1",
                    "physical_cell_id": "G1-Cell9",
                    "measurement_id": "RPT1",
                    "source_file": "Group_1/TPG1_4_Cell9.csv",
                }
            ]
        )
        rows = pd.DataFrame(
            [
                self._curve_record("TPG1_4_Cell9.csv", "accepted"),
                self._curve_record("TPG1_5_Cell9.csv", "wrong_source"),
            ]
        )
        mapped = _map_pydma_to_features(rows, frame)
        self.assertEqual(
            mapped.loc[0, "feature_mapping_status"],
            "verified_exact_step4_source_match",
        )
        self.assertEqual(mapped.loc[1, "feature_mapping_status"], "no_exact_step4_source_match")

    def test_repeated_optimizer_rows_do_not_count_as_distinct_curves(self) -> None:
        rows = pd.DataFrame(
            [
                self._aggregate_record("curve-a", rmse=9.0, lli=2.0),
                self._aggregate_record("curve-a", rmse=8.0, lli=2.1),
            ]
        )
        selected = _select_curve_records(rows)
        self.assertEqual(len(selected), 1)
        aggregate = _aggregate_pydma(rows).iloc[0]
        self.assertEqual(aggregate["n_available_curves"], 1)
        self.assertEqual(aggregate["n_accepted_curves"], 1)
        self.assertAlmostEqual(aggregate["median_voltage_fit_rmse_mV"], 8.0)
        self.assertAlmostEqual(aggregate["lli_estimated_pct"], 2.1)

    def test_negative_estimates_are_preserved_and_flagged(self) -> None:
        rows = pd.DataFrame(
            [self._aggregate_record("curve-a", rmse=8.0, lli=-0.5)]
        )
        aggregate = _aggregate_pydma(rows).iloc[0]
        self.assertAlmostEqual(aggregate["lli_estimated_pct"], -0.5)
        self.assertIn("REVIEW_NEGATIVE_ESTIMATE", aggregate["quality_flags"])

    def test_outlying_curve_group_is_not_forced_into_a_consensus(self) -> None:
        rows = pd.DataFrame(
            [
                self._aggregate_record(f"curve-{i}", rmse=8.0 + i / 10, lli=value)
                for i, value in enumerate([1.0, 1.0, 1.0, 100.0])
            ]
        )
        aggregate = _aggregate_pydma(rows).iloc[0]
        self.assertEqual(aggregate["quality_flag"], "AMBIGUOUS")
        self.assertEqual(aggregate["n_accepted_curves"], 4)
        self.assertEqual(aggregate["n_aggregated_curves"], 0)
        self.assertIsNone(aggregate["lli_estimated_pct"])

    @staticmethod
    def _curve_record(source_file: str, suffix: str) -> dict[str, object]:
        return {
            "result_record_id": suffix,
            "dataset_part": 1,
            "group_id": "1",
            "physical_cell_id": "G1-Cell9",
            "rpt_id": "RPT1",
            "rpt_order": 1,
            "direction": "discharge",
            "test_step": "8",
            "source_file": source_file,
            "source_file_basename": source_file.casefold(),
            "candidate_mapping_status": "",
            "candidate_segment_count": 1,
            "strict_gate_pass": True,
            "lli_estimated_pct": 1.0,
            "lam_anode_estimated_pct": 1.0,
            "lam_cathode_estimated_pct": 1.0,
            "candidate_values_consistent": True,
        }

    @staticmethod
    def _aggregate_record(
        curve_id: str, *, rmse: float, lli: float, direction: str = "discharge"
    ) -> dict[str, object]:
        return {
            "dataset_part": 1,
            "group_id": "1",
            "physical_cell_id": "G1-Cell9",
            "rpt_id": "RPT1",
            "rpt_order": 1,
            "protocol_group": "maccor4200_parts1_2",
            "direction": direction,
            "curve_id": curve_id,
            "strict_gate_pass": True,
            "fit_rmse_mV": rmse,
            "lli_estimated_pct": lli,
            "lam_anode_estimated_pct": 1.0,
            "lam_cathode_estimated_pct": 1.0,
            "feature_mapping_status": "verified_exact_step4_source_match",
            "is_exact_feature_mapped": True,
            "candidate_mapping_status": "",
            "candidate_segment_count": 1,
            "statistical_outlier_targets": "",
            "quality_flags": "",
        }


if __name__ == "__main__":
    unittest.main()
