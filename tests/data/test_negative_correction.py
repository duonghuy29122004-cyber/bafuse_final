from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bafuse.data.negative_correction import (
    _read_bol_reference_overrides,
    _output_status,
    _select_run_targets,
    _sign_instability,
    select_best_fit,
)


class NegativeCorrectionSelectionTests(unittest.TestCase):
    def test_selects_by_rmse_without_preferring_positive_degradation(self) -> None:
        candidates = [
            {
                "candidate_valid": True,
                "rmse_mV": 14.2,
                "fit_id": "negative",
                "source": "saved_existing_fit",
                "labels": {
                    "lli_percent": -25.0,
                    "lam_anode_percent": -40.0,
                    "lam_cathode_percent": 10.0,
                },
            },
            {
                "candidate_valid": True,
                "rmse_mV": 8.8,
                "fit_id": "positive",
                "source": "saved_existing_fit",
                "labels": {
                    "lli_percent": 4.0,
                    "lam_anode_percent": 8.0,
                    "lam_cathode_percent": 7.0,
                },
            },
        ]
        selected = select_best_fit(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["fit_id"], "positive")
        self.assertEqual(_output_status(candidates, selected, existing_only=True), "OPTIMIZER_UNSTABLE")

    def test_rejects_candidates_that_fail_quality_or_bounds_checks(self) -> None:
        candidates = [
            {
                "candidate_valid": False,
                "rmse_mV": 1.0,
                "fit_id": "invalid-low-rmse",
                "labels": {"lli_percent": 2.0},
            },
            {
                "candidate_valid": True,
                "rmse_mV": 12.0,
                "fit_id": "valid",
                "labels": {"lli_percent": -1.0},
            },
        ]
        selected = select_best_fit(candidates)
        self.assertIsNotNone(selected)
        self.assertEqual(selected["fit_id"], "valid")
        self.assertEqual(_output_status(candidates, selected), "NEGATIVE_BEST_FIT_REVIEW")

    def test_reports_sign_instability_without_a_numeric_tolerance(self) -> None:
        candidates = [
            {
                "candidate_valid": True,
                "labels": {"lli_percent": -0.01, "lam_anode_percent": 4.0},
            },
            {
                "candidate_valid": True,
                "labels": {"lli_percent": 0.01, "lam_anode_percent": 4.1},
            },
        ]
        self.assertEqual(
            _sign_instability(candidates),
            {
                "lli_percent": True,
                "lam_anode_percent": False,
                "lam_cathode_percent": False,
            },
        )

    def test_selects_only_part1_deferred_train_and_validation_keys(self) -> None:
        worklist = [
            {
                "status": "REFIT_NOT_ATTEMPTED",
                "dataset_part": "1",
                "physical_cell_id": "G1-Cell15",
                "rpt_index": "3",
                "partition": "validation",
            },
            {
                "status": "MAPPING_UNRESOLVED",
                "dataset_part": "1",
                "physical_cell_id": "G1-Cell15",
                "rpt_index": "10",
                "partition": "validation",
            },
            {
                "status": "REFIT_NOT_ATTEMPTED",
                "dataset_part": "3",
                "physical_cell_id": "G10-Cell2",
                "rpt_index": "4",
                "partition": "external_test",
            },
        ]
        with self.assertRaisesRegex(ValueError, "Part 1 train/validation"):
            _select_run_targets("deferred", worklist, worklist)

        review, targets = _select_run_targets(
            "deferred", worklist[:2], worklist[:1]
        )
        self.assertEqual(
            review,
            (("1", "G1-Cell15", "3"),),
        )
        self.assertEqual(targets, review)

    def test_explicit_target_is_limited_to_a_mapped_part1_train_key(self) -> None:
        worklist = [
            {
                "status": "BOL_STEP8_UNRESOLVED",
                "dataset_part": "1",
                "physical_cell_id": "G2-Cell4",
                "rpt_index": "6",
                "partition": "train",
            },
            {
                "status": "BOL_UNRESOLVED",
                "dataset_part": "2",
                "physical_cell_id": "G2-Cell4",
                "rpt_index": "7",
                "partition": "train",
            },
        ]
        review, targets = _select_run_targets(
            "pilot", worklist, explicit_target_keys=(("1", "G2-Cell4", "6"),)
        )
        self.assertEqual(review, (("1", "G2-Cell4", "6"),))
        self.assertEqual(targets, review)
        with self.assertRaisesRegex(ValueError, "locked Part 1 train split"):
            _select_run_targets(
                "pilot", worklist, explicit_target_keys=(("2", "G2-Cell4", "7"),)
            )

    def test_selected_bol_manifest_loads_only_explicit_same_source_references(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "bol_reference_decisions.csv"
            source = "data/raw/oxford/part1/Group_2/TPG2_1_Cell4.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "original_selected_source",
                        "selected_step8_reference",
                        "selected_step10_reference",
                        "selection_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_part": "1",
                        "physical_cell_id": "G2-Cell4",
                        "original_selected_source": source,
                        "selected_step8_reference": source + "#Step8:segment1:rows20-40:sub1",
                        "selected_step10_reference": source + "#Step10:segment1:rows50-70:sub1",
                        "selection_status": "selected_provisional",
                    }
                )
                writer.writerow(
                    {
                        "dataset_part": "2",
                        "physical_cell_id": "G5-Cell5",
                        "original_selected_source": "",
                        "selected_step8_reference": "",
                        "selected_step10_reference": "",
                        "selection_status": "not_promoted_policy_unresolved",
                    }
                )
            overrides = _read_bol_reference_overrides(root, manifest)
        self.assertEqual(list(overrides), [("1", "G2-Cell4")])
        self.assertIn("rows20-40", overrides[("1", "G2-Cell4")]["selected_step8_reference"])

    def test_selected_bol_manifest_rejects_cross_source_step_pairing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "bol_reference_decisions.csv"
            with manifest.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "original_selected_source",
                        "selected_step8_reference",
                        "selected_step10_reference",
                        "selection_status",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_part": "1",
                        "physical_cell_id": "G2-Cell4",
                        "original_selected_source": "data/raw/part1/first.csv",
                        "selected_step8_reference": "data/raw/part1/first.csv#Step8:segment1:rows20-40:sub1",
                        "selected_step10_reference": "data/raw/part1/second.csv#Step10:segment1:rows50-70:sub1",
                        "selection_status": "selected_provisional",
                    }
                )
            with self.assertRaisesRegex(ValueError, "do not match"):
                _read_bol_reference_overrides(root, manifest)


if __name__ == "__main__":
    unittest.main()
