from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from bafuse.data.ambiguous_rpt_resolution import (
    _current_direction_status,
    _effective_stage_claims,
    analyze_rpt_blocks,
    map_ordered_blocks,
    run_resolution,
    scan_raw_tpg,
    _curve_quality,
    _stage_claims,
)


def _protocol_rows(start_time: float, *, short_step4: bool = False) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    steps = [4, 5, 6, 7, 8, 9, 10, 11, 14, 16, 18, 22, 24, 26, 30, 32, 34]
    for step in steps:
        count = 3 if step in {5, 14, 16, 18, 22, 24, 26, 30, 32, 34} else 20
        if step == 4 and short_step4:
            count = 3
        for index in range(count):
            capacity = index / 100
            if step == 4:
                voltage, current = 4.2 - index / 10, -1.0
            elif step == 6:
                voltage, current = 4.0 + index / 100, 1.0
            elif step == 8:
                voltage, current = 4.1 - index / 100, -0.1
            elif step == 10:
                voltage, current = 3.1 + index / 100, 0.1
            else:
                voltage, current = 3.5, 0.0
            rows.append(
                {
                    "Step": step,
                    "TestTime": start_time + len(rows),
                    "StepTime": index,
                    "Amphr": capacity,
                    "Amps": current,
                    "Volts": voltage,
                }
            )
    return rows


def _short_fragment(step: int, start_time: float) -> list[dict[str, float]]:
    return [
        {
            "Step": step,
            "TestTime": start_time + index,
            "StepTime": index,
            "Amphr": index / 100,
            "Amps": -0.1 if step == 8 else 0.1,
            "Volts": 3.5,
        }
        for index in range(6)
    ]


class AmbiguousRPTResolutionTests(unittest.TestCase):
    def test_scanner_keeps_step6_and_reconstructs_ordered_rpt_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tpg.csv"
            columns = ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts"]
            rows = (
                _protocol_rows(0.0)
                + _short_fragment(10, 500.0)
                + _short_fragment(8, 510.0)
                + _short_fragment(9, 520.0)
                + _protocol_rows(1000.0)
            )
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)

            scan = scan_raw_tpg(path, chunk_rows=37)

        self.assertEqual(scan.status, "scanned")
        self.assertEqual(len([run for run in scan.runs if run.step == 4]), 2)
        blocks = analyze_rpt_blocks(scan.runs)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(all(block["core_sequence_verified"] for block in blocks))
        self.assertTrue(
            all(
                block["pulse_signature_status"]
                == "complete_ordered_pulse_signature"
                for block in blocks
            )
        )
        step6 = next(run for run in scan.runs if run.step == 6)
        self.assertEqual(
            _current_direction_status(step6), "verified_charge_direction"
        )
        mapped = map_ordered_blocks(blocks, [4, 5])
        self.assertIsNotNone(mapped)
        self.assertEqual([rpt for rpt, _ in mapped or []], [4, 5])

    def test_invalid_step4_does_not_invalidate_block_identity_but_fails_curve_qc(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.csv"
            columns = ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts"]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(_protocol_rows(0.0, short_step4=True))

            scan = scan_raw_tpg(path, chunk_rows=29)

        block = analyze_rpt_blocks(scan.runs)[0]
        self.assertTrue(block["core_sequence_verified"])
        self.assertEqual(
            _curve_quality(block["step4"], "partial.csv")[1], "needs_curve_audit"
        )

    def test_ordinal_mapping_is_rejected_when_core_block_count_does_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tpg.csv"
            columns = ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts"]
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(_protocol_rows(0.0) + _protocol_rows(1000.0))
            scan = scan_raw_tpg(path, chunk_rows=41)

        blocks = analyze_rpt_blocks(scan.runs)
        self.assertIsNone(map_ordered_blocks(blocks, [4]))

    def test_guide_range_annotation_conflict_is_preserved(self) -> None:
        primary, guide, duplicate = _stage_claims(
            "Profile + RPT17-RPT23 (provisional; Guide lists RPT17-RPT24)"
        )
        self.assertEqual(primary, list(range(17, 24)))
        self.assertEqual(guide, list(range(17, 25)))
        self.assertFalse(duplicate)

    def test_confirmed_g5_cell6_ranges_resolve_rpt24_guide_overlap(self) -> None:
        primary, guide, duplicate, annotation_differs = _effective_stage_claims(
            "G5-Cell6",
            "TPG5.9",
            "Profile + RPT17-RPT23 (provisional; Guide lists RPT17-RPT24)",
        )
        self.assertEqual(primary, list(range(17, 24)))
        self.assertEqual(guide, list(range(17, 24)))
        self.assertFalse(duplicate)
        self.assertTrue(annotation_differs)

        primary, guide, duplicate, annotation_differs = _effective_stage_claims(
            "G5-Cell6",
            "TPG5.10",
            "Profile + RPT24-RPT34 (provisional; Guide lists RPT24-RPT34)",
        )
        self.assertEqual(primary, list(range(24, 35)))
        self.assertEqual(guide, list(range(24, 35)))
        self.assertFalse(duplicate)
        self.assertFalse(annotation_differs)

    def test_run_resolution_writes_detail_update_and_report_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = (
                root
                / "data"
                / "raw"
                / "oxford"
                / "part2"
                / "Group_5"
                / "Group 5"
                / "TPG5.1-Cell5.csv"
            )
            source.parent.mkdir(parents=True)
            columns = ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts"]
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(_protocol_rows(0.0))

            observations = root / "observations.csv"
            row = {
                "part": "2",
                "group": "5",
                "physical_cell_id": "G5-Cell5",
                "rpt_index": "1",
                "source_tpg": "TPG5.1",
                "source_file": "data/raw/oxford/part2/Group_5/Group 5/TPG5.1-Cell5.csv",
                "source_record_id": "guide-row-1",
                "guide_stage": "Profile + RPT1",
                "split": "train",
                "bol_reference_file": "TPG5.1-BoL.csv",
                "bol_step4_segment_ref": "TPG5.1-BoL.csv#Step4:segment1:rows2-21",
                "mapping_status": "ambiguous_rpt",
            }
            with observations.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)

            detail = root / "reports" / "detail.csv"
            updates = root / "reports" / "updates.csv"
            report = root / "reports" / "report.md"
            summary = run_resolution(
                root, observations, detail, updates, report, chunk_rows=37
            )

            self.assertTrue(detail.is_file())
            self.assertTrue(updates.is_file())
            self.assertTrue(report.is_file())
            self.assertEqual(summary["initial_ambiguous_observations"], 1)
            self.assertEqual(summary["proposed_mapping_statuses"], {"verified": 1})
            self.assertEqual(summary["step4_mapping_candidates"], 1)
            self.assertEqual(summary["ready_soh_candidates"], 0)
            with detail.open(encoding="utf-8-sig", newline="") as stream:
                generated = next(csv.DictReader(stream))
            self.assertEqual(generated["step6_status"], "verified_charge_direction")
            self.assertEqual(generated["step4_mapping_candidate"], "true")
            self.assertEqual(generated["ready_soh_candidate"], "false")
            self.assertIn("BoL denominator remains provisional", generated["soh_block_reason"])


if __name__ == "__main__":
    unittest.main()
