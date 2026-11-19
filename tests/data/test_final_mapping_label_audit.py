from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from bafuse.data.final_mapping_label_audit import (
    _compare_locked_split,
    _fit_source_path,
    _partition_split_audit,
    _soh_audit,
    expected_rpts,
    is_user_policy_mapping,
    passes_fit_gate,
)


class FinalMappingLabelAuditTests(unittest.TestCase):
    def test_g5_cell6_confirmed_ranges_are_explicit(self) -> None:
        self.assertEqual(
            expected_rpts("2", "G5-Cell6", "TPG5.9", "Profile + RPT17-RPT23 (Guide lists RPT17-RPT24)"),
            list(range(17, 24)),
        )
        self.assertEqual(
            expected_rpts("2", "G5-Cell6", "TPG5.10", "Profile + RPT24-RPT34"),
            list(range(24, 35)),
        )
        self.assertTrue(is_user_policy_mapping("2", "G5-Cell6", "TPG5.9", 17))
        self.assertTrue(is_user_policy_mapping("2", "G5-Cell6", "TPG5.10", 24))
        self.assertFalse(is_user_policy_mapping("2", "G5-Cell6", "TPG5.10", 25))

    def test_pydma_gate_is_strict_and_requires_three_runs(self) -> None:
        self.assertTrue(passes_fit_gate(True, 9.99, 10.0, 3))
        self.assertFalse(passes_fit_gate(True, 10.0, 10.0, 3))
        self.assertFalse(passes_fit_gate(True, 9.0, 10.0, 2))
        self.assertFalse(passes_fit_gate(False, 9.0, 10.0, 3))

    def test_locked_split_comparison_reports_mismatches_and_missing_cells(self) -> None:
        joined = pd.DataFrame(
            [
                {"dataset_part": "1", "physical_cell_id": "G1-Cell1", "partition": "train"},
                {"dataset_part": "1", "physical_cell_id": "G1-Cell2", "partition": "train"},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            locked_path = Path(temp_dir) / "split_manifest.csv"
            pd.DataFrame(
                [
                    {"split_version": "test-v1", "physical_cell_id": "G1-Cell1", "split": "train"},
                    {"split_version": "test-v1", "physical_cell_id": "G1-Cell2", "split": "test"},
                    {"split_version": "test-v1", "physical_cell_id": "G1-Cell3", "split": "validation"},
                ]
            ).to_csv(locked_path, index=False)
            result = _compare_locked_split(joined, locked_path)

        self.assertEqual(result["status"], "compared")
        self.assertEqual(result["matched_cell_count"], 1)
        self.assertEqual(result["mismatched_cell_count"], 2)
        self.assertEqual(result["mismatches"][0]["baseline_partition"], "train")
        self.assertEqual(result["mismatches"][1]["baseline_partition"], "not_present")

    def test_part3_cell_names_are_separate_from_parts12_continuations(self) -> None:
        joined = pd.DataFrame(
            [
                {"dataset_part": "1", "physical_cell_id": "G1-Cell1", "partition": "train"},
                {"dataset_part": "2", "physical_cell_id": "G1-Cell1", "partition": "train"},
                {"dataset_part": "3", "physical_cell_id": "G1-Cell1", "partition": "external_test"},
            ]
        )
        self.assertTrue(_partition_split_audit(joined)["consistent"])

        joined.loc[joined["dataset_part"] == "2", "partition"] = "validation"
        result = _partition_split_audit(joined)
        self.assertFalse(result["consistent"])
        self.assertEqual(result["mismatches"][0]["split_scope_cell"], "parts12:G1-Cell1")

    def test_combined_pydma_source_resolves_only_when_unique_across_parts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            part1 = root / "data" / "raw" / "oxford" / "part1" / "Group_1" / "Group 1"
            part1.mkdir(parents=True)
            source = part1 / "TPG1_1_Cell15.csv"
            source.touch()
            self.assertEqual(
                _fit_source_path(
                    root, "2", source.name, "1", "15", search_all_parts=True
                ),
                source.resolve(),
            )

            part2 = root / "data" / "raw" / "oxford" / "part2" / "Group_1" / "Group 1"
            part2.mkdir(parents=True)
            (part2 / source.name).touch()
            self.assertIsNone(
                _fit_source_path(
                    root, "2", source.name, "1", "15", search_all_parts=True
                )
            )
            spaced_source = part1 / "TPG1 - Cell 20.csv"
            spaced_source.touch()
            self.assertEqual(
                _fit_source_path(
                    root, "2", spaced_source.name, "1", "20", search_all_parts=True
                ),
                spaced_source.resolve(),
            )

    def test_soh_source_uses_only_unique_cell_rpt_observation_match(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = "data/raw/oxford/part1/Group_3/Group 3/TPG3_17_Cell14.csv"
            source_path = root / source
            source_path.parent.mkdir(parents=True)
            source_path.touch()
            labels = pd.DataFrame(
                [
                    {
                        "dataset_part": "1",
                        "physical_cell_id": "G3-Cell14",
                        "measurement_id": "RPT11",
                        "source_file": "",
                    },
                    {
                        "dataset_part": "1",
                        "physical_cell_id": "G3-Cell15",
                        "measurement_id": "RPT2",
                        "source_file": "",
                    },
                ]
            )
            observations = pd.DataFrame(
                [
                    {
                        "part": "1",
                        "physical_cell_id": "G3-Cell14",
                        "rpt_index": "11",
                        "source_file": source,
                    },
                    {
                        "part": "1",
                        "physical_cell_id": "G3-Cell15",
                        "rpt_index": "2",
                        "source_file": "candidate-a.csv",
                    },
                    {
                        "part": "1",
                        "physical_cell_id": "G3-Cell15",
                        "rpt_index": "2",
                        "source_file": "candidate-b.csv",
                    },
                ]
            )

            result = _soh_audit(root, [("test", labels, "1")], {}, observations)

        unique, ambiguous = result.to_dict("records")
        self.assertEqual(unique["source_file"], source)
        self.assertEqual(unique["source_resolution"], "unique_cell_rpt_observation_match")
        self.assertTrue(unique["source_exists"])
        self.assertEqual(ambiguous["source_file"], "")
        self.assertEqual(ambiguous["source_resolution"], "ambiguous_cell_rpt_observation_match")
        self.assertFalse(ambiguous["source_exists"])


if __name__ == "__main__":
    unittest.main()
