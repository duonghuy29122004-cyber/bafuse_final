from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from bafuse.data.stage_validation import (
    _prepare_bol,
    _prepare_mapping,
    _prepare_split_integrity,
    _prepare_target_masks,
    _scan_raw_step_file,
    _source_exists,
)


class StageValidationTests(unittest.TestCase):
    def test_source_resolution_handles_manifest_and_part_relative_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "raw" / "oxford" / "part2" / "Group_5" / "TPG.csv"
            source.parent.mkdir(parents=True)
            source.write_text("Step\n4\n", encoding="utf-8")
            self.assertTrue(_source_exists(root, "2", "Group_5/TPG.csv"))
            self.assertTrue(_source_exists(root, "2", "data/raw/oxford/part2/Group_5/TPG.csv"))
            self.assertFalse(_source_exists(root, "2", "Group_5/missing.csv"))

    def test_mapping_does_not_promote_user_confirmed_or_curve_audit_rows(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "data" / "raw" / "oxford" / "part1" / "a.csv"
            source.parent.mkdir(parents=True)
            source.write_text("Step\n4\n", encoding="utf-8")
            mapping = pd.DataFrame(
                [
                    {
                        "part": "1",
                        "physical_cell_id": "G1-Cell1",
                        "rpt_index": "1",
                        "source_file": "data/raw/oxford/part1/a.csv",
                        "mapping_status_v2": "verified",
                    },
                    {
                        "part": "1",
                        "physical_cell_id": "G1-Cell1",
                        "rpt_index": "2",
                        "source_file": "data/raw/oxford/part1/a.csv",
                        "mapping_status_v2": "user_confirmed_identity_not_independently_verified",
                    },
                ]
            )
            verified, unresolved = _prepare_mapping(root, mapping)
            self.assertEqual(verified.rpt_index.tolist(), ["1"])
            self.assertEqual(unresolved.rpt_index.tolist(), ["2"])
            self.assertEqual(unresolved.stage1_mapping_status.iloc[0], "user_confirmed_identity_not_independently_verified")

    def test_reference_defined_bol_values_are_not_verified_targets(self) -> None:
        availability = pd.DataFrame(
            [
                {
                    "part": "1",
                    "group": "1",
                    "physical_cell_id": "G1-Cell1",
                    "rpt_index": "1",
                    "source_file": "rpt.csv",
                    "partition": "train",
                    "mapping_status": "verified",
                    "step4_available": "True",
                    "step8_available": "True",
                    "step10_available": "True",
                    "step4_boundary_status": "verified",
                    "step8_boundary_status": "verified",
                    "step10_boundary_status": "verified",
                    "step6_charge_status": "not_audited",
                    "soh_label_status": "no_existing_primary_label_row",
                    "historical_pydma_accepted_direction_records": "0",
                }
            ]
        )
        soh = pd.DataFrame(columns=["dataset_part", "physical_cell_id", "measurement_id", "validation_status_v2"])
        bol = pd.DataFrame(
            [
                {
                    "dataset_part": "1",
                    "group": "1",
                    "physical_cell_id": "G1-Cell1",
                    "primary_bol_reference_file": "bol.csv",
                    "partition": "train",
                    "bol_source_exists": True,
                    "bol_verified": True,
                    "bol_step4_status": "verified",
                    "bol_policy_status": "part1_provisional_earliest_complete_bol_selected",
                }
            ]
        )
        result = _prepare_target_masks(availability, soh, bol)
        bol_row = result.loc[result.observation_type.eq("BOL_REFERENCE")].iloc[0]
        self.assertEqual(bol_row.soh_percent, "100.0")
        self.assertEqual(bol_row.lli_percent, "0.0")
        self.assertFalse(bol_row.has_verified_soh)
        self.assertFalse(bol_row.has_verified_lli)
        self.assertFalse(bol_row.target_verified)

    def test_priority_step_scan_reports_segments_without_rpt_assignment(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "priority.csv"
            frame = pd.DataFrame(
                [
                    *[
                        {"Step": 4, "Amphr": value, "Amps": -1.0, "Volts": 4.1 - value / 20}
                        for value in range(12)
                    ],
                    *[
                        {"Step": 8, "Amphr": value, "Amps": -0.1, "Volts": 4.1 - value / 20}
                        for value in range(12)
                    ],
                    *[
                        {"Step": 10, "Amphr": value, "Amps": 0.1, "Volts": 3.0 + value / 20}
                        for value in range(12)
                    ],
                ]
            )
            frame.to_csv(path, index=False)
            rows = _scan_raw_step_file(
                path,
                "2",
                "5",
                "G5-Cell6",
                "TPG5.9",
                "Group_5/priority.csv",
            )
            self.assertEqual([row["step"] for row in rows], [4, 8, 10])
            self.assertTrue(all(row["curve_candidate"] for row in rows))
            self.assertTrue(all(row["mapping_decision"] == "not_assigned_to_an_RPT_by_step_count" for row in rows))
            self.assertEqual(rows[0]["source_row_start"], 2)
            self.assertEqual(rows[0]["source_row_end"], 13)
            self.assertEqual(rows[1]["low_valid_subsegments"], 1)
            self.assertEqual(json.loads(rows[1]["low_rate_subsegments"])[0]["source_row_start"], 14)
            self.assertEqual(json.loads(rows[1]["low_rate_subsegments"])[0]["source_row_end"], 25)

    def test_split_integrity_detects_observation_partition_mismatch(self) -> None:
        cells = [f"G1-Cell{i}" for i in range(12)]
        cells.extend(f"G5-Cell{i}" for i in range(4))
        cells.extend(f"G7-Cell{i}" for i in range(12))
        parts = ["1"] * 12 + ["2"] * 4 + ["3"] * 12
        partitions = ["train"] * 12 + ["validation"] * 4 + ["external_test"] * 12
        split = pd.DataFrame({"physical_cell_id": cells, "partition": partitions})
        mapping = pd.DataFrame(
            {
                "part": parts,
                "physical_cell_id": cells,
                "source_file": ["unused.csv"] * len(cells),
            }
        )
        masks = pd.DataFrame(
            {
                "dataset_part": parts,
                "group": ["1"] * 12 + ["5"] * 4 + ["7"] * 12,
                "physical_cell_id": cells,
                "normalized_rpt_index": ["1"] * len(cells),
                "partition": partitions,
                "observation_type": ["RPT"] * len(cells),
            }
        )
        _, passed = _prepare_split_integrity(split, mapping, masks)
        self.assertTrue(passed)
        masks.loc[0, "partition"] = "validation"
        _, passed = _prepare_split_integrity(split, mapping, masks)
        self.assertFalse(passed)


if __name__ == "__main__":
    unittest.main()
