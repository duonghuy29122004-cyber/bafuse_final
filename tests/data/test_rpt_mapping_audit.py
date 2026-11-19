from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bafuse.data.rpt_mapping_audit import (
    _finalize_raw_segment,
    _half_cell_reference_audit,
    _map_step_segments,
    _resolve_source,
    parse_rpt_stage,
    scan_tpg_file,
)


class RPTMappingAuditTests(unittest.TestCase):
    def test_parse_rpt_ranges_and_explicit_compounds_without_tpg_inference(self) -> None:
        self.assertEqual(parse_rpt_stage("Profile + RPT17-RPT23"), (list(range(17, 24)), False))
        self.assertEqual(parse_rpt_stage("Profile+RPT10-15"), (list(range(10, 16)), False))
        self.assertEqual(parse_rpt_stage("RPT10 to 15"), (list(range(10, 16)), False))
        self.assertEqual(parse_rpt_stage("RPT7 + RPT8"), ([7, 8], False))
        self.assertEqual(parse_rpt_stage("TPG5.10 Profile"), ([], False))
        self.assertEqual(parse_rpt_stage("RPT24-RPT34; RPT24"), (list(range(24, 35)) + [24], True))
        self.assertEqual(
            parse_rpt_stage("Profile + RPT17-RPT23 (provisional; Guide lists RPT17-RPT24)"),
            (list(range(17, 24)), False),
        )

    def test_source_path_resolution_accepts_manifest_and_part_relative_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data" / "raw" / "oxford" / "part1" / "Group 1" / "source.csv"
            source.parent.mkdir(parents=True)
            source.write_text("source", encoding="utf-8")
            full = _resolve_source(
                root,
                "1",
                "data/raw/oxford/part1/Group 1/source.csv",
            )
            relative = _resolve_source(root, "1", "Group 1/source.csv")

        self.assertEqual(full, source.resolve())
        self.assertEqual(relative, source.resolve())

    def test_scan_keeps_step_boundaries_and_discarded_fragments(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tpg.csv"
            rows = ["Step,Amphr,Volts,Amps"]
            rows.extend(f"4,{i / 1000:.3f},{4.2 - i / 500:.3f},-1" for i in range(1200))
            rows.extend(f"8,{i / 1000:.3f},{4.2 - i / 1000:.3f},-0.1" for i in range(1200))
            rows.extend(f"10,{i / 1000:.3f},{3.0 + i / 1000:.3f},0.1" for i in range(1200))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            result = scan_tpg_file(path, chunk_rows=97)

        self.assertEqual(result["scan_status"], "scanned")
        self.assertEqual(result["rows_scanned"], 3600)
        self.assertEqual(len(result["steps"]["4"]), 1)
        self.assertEqual(len(result["steps"]["8"]), 1)
        self.assertEqual(len(result["steps"]["10"]), 1)
        self.assertTrue(result["steps"]["4"][0]["valid"])
        self.assertTrue(result["steps"]["8"][0]["valid"])
        self.assertTrue(result["steps"]["10"][0]["valid"])
        self.assertLess(result["steps"]["4"][0]["source_row_end"], result["steps"]["8"][0]["source_row_start"])

    def test_scan_separates_repeated_steps_and_uses_direction_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "repeated.csv"
            rows = ["Step,Amphr,Volts,Amps"]
            rows.extend(f"8,{i / 1000:.3f},{4.2 - i / 1000:.3f},-0.1" for i in range(1200))
            rows.extend(f"9,{i / 1000:.3f},3.0,0" for i in range(5))
            rows.extend(f"10,{i / 1000:.3f},{3.0 + i / 1000:.3f},-0.1" for i in range(1200))
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")

            result = scan_tpg_file(path, chunk_rows=113)

        self.assertEqual(len(result["steps"]["8"]), 1)
        self.assertEqual(len(result["steps"]["10"]), 0)

    def test_map_compound_segments_only_when_counts_match_explicit_rpt_list(self) -> None:
        segments = [
            {"segment_index": index, "valid": True}
            for index in range(1, 4)
        ]
        mapped, status, note = _map_step_segments(
            segments,
            expected_count=3,
            exact_single_rpt=False,
        )
        self.assertEqual(status, "verified")
        self.assertEqual([mapped[i]["segment_index"] for i in range(3)], [1, 2, 3])
        self.assertIn("observed source-row order", note)

        mapped, status, note = _map_step_segments(
            segments[:2],
            expected_count=3,
            exact_single_rpt=False,
        )
        self.assertEqual(mapped, {})
        self.assertEqual(status, "needs_curve_audit")
        self.assertIn("expected 3", note)

    def test_empty_or_invalid_segment_is_not_marked_valid(self) -> None:
        raw = {
            "step": 8,
            "segment_index": 1,
            "source_row_start": 0,
            "source_rows_indices": list(range(6)),
            "source_rows": 6,
            "capacity_parts": [np.arange(6, dtype=float)],
            "voltage_parts": [np.linspace(4.0, 3.8, 6)],
            "current_parts": [-np.full(6, 0.1)],
        }
        result = _finalize_raw_segment(raw)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0]["valid"])
        self.assertEqual(result[0]["status"], "invalid_or_partial")

    def test_half_cell_references_accept_the_observed_zero_to_100_soc_scale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference_dir = root / "data" / "raw" / "oxford" / "part1" / "Half_Cell" / "Half Cell"
            reference_dir.mkdir(parents=True)
            names = [
                "CH_HC_anode.csv",
                "CH_HC_cathode.csv",
                "DCH_HC_anode.csv",
                "DHC_HC_cathode.csv",
            ]
            for name in names:
                with (reference_dir / name).open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["SoC", "Volts"])
                    writer.writerows((index, 0.1 + index / 100) for index in range(101))

            results = _half_cell_reference_audit(root)

        self.assertTrue(
            all(
                result["status"] == "verified_reference_columns_and_finite_pairs"
                for result in results.values()
            )
        )


if __name__ == "__main__":
    unittest.main()
