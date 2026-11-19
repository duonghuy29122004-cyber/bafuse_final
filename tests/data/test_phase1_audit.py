from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from bafuse.data.phase1_audit import (
    _eis_csv_audit,
    _hash_manifest,
)


class Phase1AuditTests(unittest.TestCase):
    def test_eis_row_count_does_not_create_a_frequency_axis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory)
            source = dataset / "EIS" / "example.csv"
            source.parent.mkdir(parents=True)
            values = ["ReZ_Ohm,N_A"] + [f"{0.04 + index / 1000},{index / 10000}" for index in range(35)]
            source.write_text("\n".join(values) + "\n", encoding="utf-8")
            manifest = [
                {
                    "modality": "EIS",
                    "file_type": "csv",
                    "relative_path": "EIS/example.csv",
                    "stage": "BoL",
                    "soc": "20",
                    "physical_cell_id": "G1-Cell1",
                }
            ]

            report, _, mismatches = _eis_csv_audit(dataset, manifest)

        self.assertEqual(report["csv_rows_per_file"], {"35": 1})
        self.assertFalse(report["csv_frequency_grid_check"]["actual_frequency_values_available"])
        self.assertEqual(report["frequency_columns_in_csv"], [])
        self.assertEqual(len(mismatches), 1)
        self.assertEqual(mismatches[0]["csv_relative_path"], "EIS/example.csv")

    def test_hash_snapshot_can_resume_when_path_size_and_mtime_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset = workspace / "Path_Dependent_Battery_Degradation_Dataset_Part 1"
            dataset.mkdir()
            (dataset / "source.txt").write_text("part1", encoding="utf-8")
            result = workspace / "experiments" / "results" / "benchmarks" / "test_result"
            result.mkdir(parents=True)
            (result / "result.csv").write_text("metric\n1\n", encoding="utf-8")
            output = workspace / "experiments" / "results" / "data_audits" / "hash_audit"
            output.mkdir(parents=True)

            first = _hash_manifest(workspace, dataset, output)
            second = _hash_manifest(workspace, dataset, output, reuse_existing=True)

        self.assertEqual(len(first), 2)
        self.assertEqual([row["sha256"] for row in first], [row["sha256"] for row in second])
        self.assertTrue(all(row["modified_ns"] for row in second))


if __name__ == "__main__":
    unittest.main()