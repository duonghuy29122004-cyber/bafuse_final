from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bafuse.data.bol_procedure_audit import (
    _candidate_records,
    _compare_step4_segments,
    _guide_source_path,
    _scan_timeline,
)


class BoLProcedureAuditTests(unittest.TestCase):
    def test_existing_part_relative_reference_resolves_to_raw_source_path(self) -> None:
        self.assertEqual(
            _guide_source_path("1", "Group_2/Group 2/TPG2_1_Cell4.csv"),
            "data/raw/oxford/part1/Group_2/Group 2/TPG2_1_Cell4.csv",
        )

    def test_timeline_preserves_repeated_step_runs_and_testtime_decreases(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "timeline.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["Step", "TestTime", "Amphr", "Amps", "Volts"],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"Step": 4, "TestTime": 1, "Amphr": 1, "Amps": -1, "Volts": 4},
                        {"Step": 4, "TestTime": 2, "Amphr": 2, "Amps": -1, "Volts": 3},
                        {"Step": 8, "TestTime": 3, "Amphr": 1, "Amps": -0.1, "Volts": 4},
                        {"Step": 8, "TestTime": 2, "Amphr": 2, "Amps": -0.1, "Volts": 3},
                        {"Step": 10, "TestTime": 4, "Amphr": 1, "Amps": 0.1, "Volts": 3},
                    ]
                )

            result = _scan_timeline(path)

        self.assertEqual(result["testtime_decrease_count"], 1)
        self.assertEqual(
            [
                (run["step"], run["segment_index"], run["source_row_start"], run["source_row_end"])
                for run in result["step_runs"]
            ],
            [(4, 1, 2, 3), (8, 1, 4, 5), (10, 1, 6, 6)],
        )
        self.assertEqual(result["step_runs"][1]["preceding_step4_segment_index"], 1)
        self.assertEqual(result["step_runs"][1]["testtime_decrease_count"], 1)

    def test_candidates_in_one_source_remain_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            path = root / "bol.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=["Step", "TestTime", "Amphr", "Amps", "Volts"],
                )
                writer.writeheader()
                for run, step in enumerate((8, 4, 8), start=1):
                    for index in range(12):
                        writer.writerow(
                            {
                                "Step": step,
                                "TestTime": run * 100 + index,
                                "Amphr": index / 1000,
                                "Amps": -0.1 if step == 8 else -1.5,
                                "Volts": 4.1 - index / 100,
                            }
                        )

            timeline = _scan_timeline(path)
            candidates = _candidate_records(
                root, path, 8, "discharge", timeline["step_runs"]
            )

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate["valid"] for candidate in candidates))
        self.assertEqual(
            [candidate["step_segment_index"] for candidate in candidates], [1, 2]
        )
        self.assertEqual(
            [
                (candidate["source_row_start"], candidate["source_row_end"])
                for candidate in candidates
            ],
            [(2, 13), (26, 37)],
        )
        self.assertEqual(
            [candidate["preceding_step4_segment_index"] for candidate in candidates],
            [None, 1],
        )

    def test_step4_signal_comparison_uses_exact_csv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "first.csv"
            second = root / "second.csv"
            header = "Step,Amphr,Amps,Volts,Temp1\n"
            rows = (
                "4,1.0,-1.0,4.0,24.0\n"
                "4,2.0,-1.0,3.0,24.1\n"
                "4,3.0,-1.0,2.5,24.2\n"
                "5,3.0,0.0,2.5,24.2\n"
            )
            first.write_text(header + rows, encoding="utf-8")
            second.write_text(header + rows, encoding="utf-8")
            segment = {"source_row_start": 2, "source_row_end": 4}

            result = _compare_step4_segments(first, segment, second, segment)

        self.assertEqual(result["status"], "matching_within_1e-8")
        self.assertEqual(result["signals"], "Amphr;Amps;Volts;Temp1")
        self.assertEqual(result["max_abs_difference"], 0)


if __name__ == "__main__":
    unittest.main()
