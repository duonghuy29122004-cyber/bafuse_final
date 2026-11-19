from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from bafuse.data.soh_step4_scan import (
    build_step4_scan,
    scan_step4_file,
    write_step4_scan,
)


class SOHStep4ScanTests(unittest.TestCase):
    def test_scan_merges_step4_runs_across_chunk_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "single_segment.csv"
            path.write_text(
                "Step,Amphr,Amps\n4,2.0,-1\n4,1.9,-1\n4,1.8,-1\n4,1.7,-1\n",
                encoding="utf-8",
            )

            result = scan_step4_file(path, chunk_rows=2)

        self.assertEqual(result["step4_segment_count"], 1)
        self.assertEqual(result["step4_rows"], 4)
        self.assertEqual(result["step4_capacity_finite_rows"], 4)
        self.assertAlmostEqual(result["step4_capacity_span"], 0.3)
        self.assertEqual(result["step4_current_signs"], "negative")
        self.assertEqual(result["scan_status"], "too_few_step4_capacity_rows")
        segment = json.loads(result["step4_segments_json"])[0]
        self.assertEqual(segment["source_row_start"], 2)
        self.assertEqual(segment["source_row_end"], 5)

    def test_scan_reports_multiple_step4_segments_and_mixed_current(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "multiple_segments.csv"
            path.write_text(
                "Step,Amphr,Amps\n"
                "4,2.0,-1\n4,1.8,-1\n"
                "5,1.8,0\n"
                "4,1.7,1\n4,1.5,1\n",
                encoding="utf-8",
            )

            result = scan_step4_file(path, chunk_rows=3)

        self.assertEqual(result["step4_segment_count"], 2)
        self.assertEqual(result["step4_rows"], 4)
        self.assertEqual(result["step4_current_signs"], "negative;positive")
        self.assertEqual(result["scan_status"], "multiple_step4_segments")
        segments = json.loads(result["step4_segments_json"])
        self.assertEqual([segment["step4_rows"] for segment in segments], [2, 2])
        self.assertEqual(
            [(segment["source_row_start"], segment["source_row_end"]) for segment in segments],
            [(2, 3), (5, 6)],
        )

    def test_scan_reports_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "missing_columns.csv"
            path.write_text("Amphr,Amps\n2.0,-1\n", encoding="utf-8")

            result = scan_step4_file(path)

        self.assertEqual(result["scan_status"], "not_scanned")
        self.assertEqual(result["csv_status"], "missing_required_columns")
        self.assertIn("Step", result["scan_error"])

    def test_manifest_step4_status_requires_unique_eligible_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "part2"
            (root / "Group_5").mkdir(parents=True)
            guide_path = root / "Guide_to_Datafiles_2.xlsx"
            guide_path.write_bytes(b"guide fixture")
            eligible = root / "Group_5" / "TPG5_1_Cell6.csv"
            ineligible = root / "Group_5" / "TPG5_1_Cell6_extra.csv"
            eligible.write_text(
                "Step,Amphr,Amps\n"
                + "".join(f"4,{3 - index * 0.001},-1\n" for index in range(20)),
                encoding="utf-8",
            )
            ineligible.write_text(
                "Step,Amphr,Amps\n"
                + "".join(f"10,{index},1\n" for index in range(20)),
                encoding="utf-8",
            )
            source_manifest = root / "guide_observation_manifest.csv"
            with source_manifest.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "measurement_id",
                        "measurement_mapping_status",
                        "source_file",
                        "source_file_count",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_part": "2",
                        "physical_cell_id": "G5-Cell6",
                        "measurement_id": "BoL",
                        "measurement_mapping_status": "single_checkpoint",
                        "source_file": (
                            "Group_5/TPG5_1_Cell6.csv;"
                            "Group_5/TPG5_1_Cell6_extra.csv"
                        ),
                        "source_file_count": "2",
                    }
                )
                writer.writerow(
                    {
                        "dataset_part": "2",
                        "physical_cell_id": "G5-Cell6",
                        "measurement_id": "",
                        "measurement_mapping_status": "ambiguous_compound_stage",
                        "source_file": "Group_5/TPG5_1_Cell6.csv",
                        "source_file_count": "1",
                    }
                )
            guide_summary = {
                "dataset_roots": {"2": str(root.resolve())},
                "guide_sha256": {
                    "2": hashlib.sha256(guide_path.read_bytes()).hexdigest()
                },
            }
            (root / "summary.json").write_text(
                json.dumps(guide_summary), encoding="utf-8"
            )

            file_results, observations, summary = build_step4_scan(
                source_manifest, {2: root}, chunk_rows=7
            )

        self.assertEqual(len(file_results), 2)
        self.assertEqual(observations[0]["step4_boundary_status"], "unique_source_among_multiple_candidates")
        self.assertEqual(
            observations[0]["step4_candidate_source_file"],
            "Group_5/TPG5_1_Cell6.csv",
        )
        self.assertEqual(
            observations[1]["step4_boundary_status"], "blocked_compound_guide_stage"
        )
        self.assertEqual(summary["distinct_source_files_scanned"], 2)

    def test_output_directory_must_be_empty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            (output / "keep.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                write_step4_scan(
                    output / "missing.csv",
                    {},
                    output,
                )

    def test_manifest_size_mismatch_is_not_scanned(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "part2"
            (root / "Group_5").mkdir(parents=True)
            guide_path = root / "Guide_to_Datafiles_2.xlsx"
            guide_path.write_bytes(b"guide fixture")
            source = root / "Group_5" / "TPG5_1_Cell6.csv"
            source.write_text("Step,Amphr\n4,2.0\n", encoding="utf-8")
            source_manifest = root / "guide_observation_manifest.csv"
            with source_manifest.open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=[
                        "dataset_part",
                        "physical_cell_id",
                        "measurement_id",
                        "measurement_mapping_status",
                        "source_file",
                        "source_file_count",
                        "source_mapping_status",
                        "source_file_size_bytes",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_part": "2",
                        "physical_cell_id": "G5-Cell6",
                        "measurement_id": "BoL",
                        "measurement_mapping_status": "single_checkpoint",
                        "source_file": "Group_5/TPG5_1_Cell6.csv",
                        "source_file_count": "1",
                        "source_mapping_status": "unique_source_file",
                        "source_file_size_bytes": str(source.stat().st_size + 1),
                    }
                )
            (root / "summary.json").write_text(
                json.dumps(
                    {
                        "dataset_roots": {"2": str(root.resolve())},
                        "guide_sha256": {
                            "2": hashlib.sha256(guide_path.read_bytes()).hexdigest()
                        },
                    }
                ),
                encoding="utf-8",
            )

            file_results, observations, _ = build_step4_scan(
                source_manifest, {2: root}
            )

        self.assertEqual(file_results[0]["scan_status"], "source_size_mismatch")
        self.assertEqual(file_results[0]["rows_scanned"], 0)
        self.assertEqual(
            observations[0]["step4_boundary_status"], "source_size_mismatch"
        )


if __name__ == "__main__":
    unittest.main()
