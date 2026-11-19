from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from bafuse.data.soh_baseline_review import (
    build_part1_bol_comparison,
)


class SOHBaselineReviewTests(unittest.TestCase):
    def test_comparison_matches_capacity_only_without_selecting_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "bol_first.csv"
            second = root / "bol_second.csv"
            first.write_text(
                "Step,Amphr,Amps,Volts,Cyc,TestTime\n"
                "4,0,-1,4,0,20\n"
                "4,2,-1,3.8,0,21\n"
                "3,5,0,4,0,25\n"
                "4,0,-1,4,0,30\n"
                "4,1,-1,3.8,0,31\n",
                encoding="utf-8",
            )
            second.write_text(
                "Step,Amphr,Amps,Volts,Cyc,TestTime\n"
                "4,0,-1,4,0,5\n"
                "4,1,-1,3.8,0,6\n",
                encoding="utf-8",
            )
            observation_path = root / "observations.csv"
            with observation_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "measurement_id",
                        "procedure",
                        "procedure_file_id",
                        "source_file",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "dataset_part": "1",
                            "physical_cell_id": "G1-Cell1",
                            "measurement_id": "BoL",
                            "procedure": "TPG1",
                            "procedure_file_id": "1",
                            "source_file": first.name,
                        },
                        {
                            "dataset_part": "1",
                            "physical_cell_id": "G1-Cell1",
                            "measurement_id": "BoL",
                            "procedure": "TPG1.2",
                            "procedure_file_id": "2",
                            "source_file": second.name,
                        },
                    ]
                )
            scan_path = root / "source_scan.csv"
            scan_rows = []
            for path, spans in ((first, [2.0, 1.0]), (second, [1.0])):
                stat = path.stat()
                scan_rows.append(
                    {
                        "dataset_parts": "1",
                        "relative_path": path.name,
                        "source_size_bytes": str(stat.st_size),
                        "source_modified_ns": str(stat.st_mtime_ns),
                        "step4_segments_json": json.dumps(
                            [
                                {
                                    "segment_index": index,
                                    "capacity_span": span,
                                    "step4_rows": 2,
                                }
                                for index, span in enumerate(spans, 1)
                            ]
                        ),
                    }
                )
            with scan_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=scan_rows[0].keys())
                writer.writeheader()
                writer.writerows(scan_rows)

            comparisons, summary = build_part1_bol_comparison(
                observation_path, scan_path, root, chunk_rows=2
            )

        self.assertEqual(len(comparisons), 2)
        self.assertEqual(
            [row["capacity_match_atol_1e-12"] for row in comparisons],
            [False, True],
        )
        self.assertTrue(comparisons[1]["core_signals_match_except_time"])
        self.assertTrue(comparisons[1]["testtime_constant_offset"])
        self.assertEqual(comparisons[1]["testtime_offset_native_units"], 25)
        self.assertEqual(
            summary["decision"], "review_only_no_baseline_selected"
        )
        self.assertTrue(
            all(
                "does not establish measurement identity"
                in row["interpretation"]
                for row in comparisons
            )
        )

    def test_non_unique_bol_source_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "candidate.csv"
            source.write_text("Step,Amphr\n4,0\n4,1\n", encoding="utf-8")
            other_source = root / "other_candidate.csv"
            other_source.write_text("Step,Amphr\n4,0\n4,2\n", encoding="utf-8")
            observation_path = root / "observations.csv"
            with observation_path.open(
                "w", encoding="utf-8-sig", newline=""
            ) as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "measurement_id",
                        "procedure",
                        "procedure_file_id",
                        "source_file",
                    ],
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {
                            "dataset_part": "1",
                            "physical_cell_id": "G1-Cell1",
                            "measurement_id": "BoL",
                            "procedure": "TPG1",
                            "procedure_file_id": "1",
                            "source_file": "candidate.csv;other_candidate.csv",
                        },
                        {
                            "dataset_part": "1",
                            "physical_cell_id": "G1-Cell1",
                            "measurement_id": "BoL",
                            "procedure": "TPG1.2",
                            "procedure_file_id": "2",
                            "source_file": "candidate.csv",
                        },
                    ]
                )
            scan_path = root / "source_scan.csv"
            scan_rows = []
            for path, span in ((source, 1), (other_source, 2)):
                stat = path.stat()
                scan_rows.append(
                    {
                        "dataset_parts": "1",
                        "relative_path": path.name,
                        "source_size_bytes": str(stat.st_size),
                        "source_modified_ns": str(stat.st_mtime_ns),
                        "step4_segments_json": json.dumps(
                            [
                                {
                                    "segment_index": 1,
                                    "capacity_span": span,
                                    "step4_rows": 2,
                                }
                            ]
                        ),
                    }
                )
            with scan_path.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=scan_rows[0].keys(),
                )
                writer.writeheader()
                writer.writerows(scan_rows)

            comparisons, summary = build_part1_bol_comparison(
                observation_path, scan_path, root
            )

        self.assertEqual(
            comparisons[0]["comparison_status"],
            "blocked_non_unique_step4_source",
        )
        self.assertEqual(summary["blocked_pairs"], 1)


if __name__ == "__main__":
    unittest.main()
