from __future__ import annotations

import unittest
from pathlib import Path

from bafuse.data.bol_procedure_audit import _scan_timeline
from bafuse.data.bol_sequence_resolution import (
    _guide_sources,
    reconstruct_step_sequences,
)
from bafuse.data.pydma_stage2 import CurvePreflightError, _read_verified_curve


def _raw_run(
    step: int,
    index: int,
    start: int,
    end: int,
    *,
    signs: dict[str, int],
    voltage_min: float,
    voltage_max: float,
) -> dict[str, object]:
    return {
        "step": step,
        "segment_index": index,
        "source_row_start": start,
        "source_row_end": end,
        "row_count": end - start + 1,
        "testtime_first": float(start),
        "testtime_last": float(end),
        "testtime_decrease_count": 0,
        "current_sign_counts": signs,
        "current_direction_changes": 0,
        "amps_min": -0.1 if signs.get("negative") else 0.1 if signs.get("positive") else 0,
        "amps_max": -0.1 if signs.get("negative") else 0.1 if signs.get("positive") else 0,
        "volts_min": voltage_min,
        "volts_max": voltage_max,
    }


def _step4_segment(index: int, start: int, end: int) -> dict[str, object]:
    return {
        "segment_index": index,
        "source_row_start": start,
        "source_row_end": end,
        "step4_rows": end - start + 1,
        "capacity_finite_rows": max(10, end - start + 1),
        "capacity_span": 3.0,
        "current_signs": ["negative"],
    }


def _candidate(
    step: int,
    segment: int,
    start: int,
    end: int,
    *,
    voltage_min: float,
    voltage_max: float,
    valid: bool = True,
) -> dict[str, object]:
    return {
        "step_segment_index": segment,
        "subsegment_index": 1,
        "source_row_start": start,
        "source_row_end": end,
        "sample_count": end - start + 1,
        "capacity_min": 0.0,
        "capacity_max": 2.0,
        "capacity_span": 2.0,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "valid": valid,
        "source_reference": (
            f"data/raw/part{1}/test.csv#Step{step}:segment{segment}:"
            f"rows{start}-{end}:sub1"
        ),
    }


def _reconstruct(
    step4_segments: list[dict[str, object]],
    step_runs: list[dict[str, object]],
    step8_candidates: list[dict[str, object]],
    step10_candidates: list[dict[str, object]],
) -> list[dict[str, object]]:
    all_step_runs = []
    for run in step_runs:
        all_step_runs.append(
            {key: run[key] for key in ("step", "segment_index", "source_row_start", "source_row_end", "row_count")}
        )
    return reconstruct_step_sequences(
        part=1,
        group_id="2",
        physical_cell_id="G2-Cell4",
        procedure="TPG2",
        procedure_file_id=1,
        source_file="data/raw/part1/test.csv",
        source_sha256="hash",
        guide_stage="BoL RPT",
        timeline={
            "source_row_count": 120,
            "testtime_decrease_count": 0,
            "step_runs": step_runs,
            "all_step_runs": all_step_runs,
        },
        step4_segments=step4_segments,
        step8_candidates=step8_candidates,
        step10_candidates=step10_candidates,
    )


class BoLSequenceResolutionTests(unittest.TestCase):
    def test_one_complete_sequence_reconstructs_exact_references(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(10, 1, 40, 49, signs={"positive": 9}, voltage_min=2.9, voltage_max=4.2),
        ]
        result = _reconstruct(
            [_step4_segment(1, 2, 9)],
            runs,
            [_candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17)],
            [_candidate(10, 1, 40, 49, voltage_min=2.9, voltage_max=4.2)],
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["sequence_status"], "complete")
        self.assertEqual(result[0]["protocol_order_status"], "step4_then_step8_then_step10")
        self.assertIn("rows20-29:sub1", result[0]["step8_reference"])
        self.assertIn("rows40-49:sub1", result[0]["step10_reference"])

    def test_two_complete_sequences_remain_chronological_and_separate(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(10, 1, 40, 49, signs={"positive": 9}, voltage_min=2.9, voltage_max=4.2),
            _raw_run(4, 2, 70, 79, signs={"negative": 10}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 2, 82, 89, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.18),
            _raw_run(10, 2, 92, 99, signs={"positive": 8}, voltage_min=2.9, voltage_max=4.2),
        ]
        step4 = [_step4_segment(1, 2, 9), _step4_segment(2, 70, 79)]
        step8 = [
            _candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17),
            _candidate(8, 2, 82, 89, voltage_min=2.5, voltage_max=4.18),
        ]
        step10 = [
            _candidate(10, 1, 40, 49, voltage_min=2.9, voltage_max=4.2),
            _candidate(10, 2, 92, 99, voltage_min=2.9, voltage_max=4.2),
        ]
        result = _reconstruct(step4, runs, step8, step10)
        self.assertEqual([row["sequence_id"] for row in result], ["TPG2.1-SEQ1", "TPG2.1-SEQ2"])
        self.assertEqual([row["sequence_status"] for row in result], ["complete", "complete"])
        self.assertIn("rows20-29:sub1", result[0]["step8_reference"])
        self.assertIn("rows82-89:sub1", result[1]["step8_reference"])

    def test_complete_sequence_followed_by_partial_step10_is_not_promoted_as_complete(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(10, 1, 40, 49, signs={"positive": 9}, voltage_min=2.9, voltage_max=4.2),
            _raw_run(4, 2, 70, 79, signs={"negative": 10}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 2, 82, 89, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.18),
            _raw_run(10, 2, 92, 99, signs={"positive": 8}, voltage_min=2.9, voltage_max=3.36),
        ]
        result = _reconstruct(
            [_step4_segment(1, 2, 9), _step4_segment(2, 70, 79)],
            runs,
            [
                _candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17),
                _candidate(8, 2, 82, 89, voltage_min=2.5, voltage_max=4.18),
            ],
            [
                _candidate(10, 1, 40, 49, voltage_min=2.9, voltage_max=4.2),
                _candidate(10, 2, 92, 99, voltage_min=2.9, voltage_max=3.36),
            ],
        )
        self.assertEqual(result[0]["sequence_status"], "complete")
        self.assertEqual(result[1]["step10_status"], "partial")
        self.assertEqual(result[1]["sequence_status"], "partial")
        self.assertEqual(result[1]["step10_reference"], "")

    def test_multiple_step8_candidates_in_one_interval_are_ambiguous(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(8, 2, 31, 39, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.18),
            _raw_run(10, 1, 40, 49, signs={"positive": 9}, voltage_min=2.9, voltage_max=4.2),
        ]
        result = _reconstruct(
            [_step4_segment(1, 2, 9)],
            runs,
            [
                _candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17),
                _candidate(8, 2, 31, 39, voltage_min=2.5, voltage_max=4.18),
            ],
            [_candidate(10, 1, 40, 49, voltage_min=2.9, voltage_max=4.2)],
        )
        self.assertEqual(result[0]["sequence_status"], "ambiguous")
        self.assertEqual(result[0]["step8_status"], "ambiguous")
        self.assertEqual(result[0]["step8_reference"], "")

    def test_step10_after_the_next_step4_is_not_cross_paired(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(4, 2, 40, 49, signs={"negative": 10}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(10, 1, 60, 69, signs={"positive": 8}, voltage_min=2.9, voltage_max=4.2),
        ]
        result = _reconstruct(
            [_step4_segment(1, 2, 9), _step4_segment(2, 40, 49)],
            runs,
            [_candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17)],
            [_candidate(10, 1, 60, 69, voltage_min=2.9, voltage_max=4.2)],
        )
        self.assertEqual([row["sequence_status"] for row in result], ["partial", "partial"])
        self.assertEqual(result[0]["step10_reference"], "")
        self.assertEqual(result[1]["step8_reference"], "")

    def test_selection_is_deterministic_and_preserves_source_boundaries(self) -> None:
        runs = [
            _raw_run(4, 1, 2, 9, signs={"negative": 8}, voltage_min=2.5, voltage_max=4.2),
            _raw_run(8, 1, 20, 29, signs={"negative": 9}, voltage_min=2.5, voltage_max=4.17),
            _raw_run(10, 1, 40, 49, signs={"positive": 9}, voltage_min=2.9, voltage_max=4.2),
        ]
        args = (
            [_step4_segment(1, 2, 9)],
            runs,
            [_candidate(8, 1, 20, 29, voltage_min=2.5, voltage_max=4.17)],
            [_candidate(10, 1, 40, 49, voltage_min=2.9, voltage_max=4.2)],
        )
        self.assertEqual(_reconstruct(*args), _reconstruct(*args))
        self.assertEqual(_reconstruct(*args)[0]["step4_reference"].split("rows")[-1], "2-9")

    def test_both_guide_labeled_bol_procedures_are_retained(self) -> None:
        root = Path(__file__).resolve().parents[2]
        sources = _guide_sources(root)
        for cell, part in (("G2-Cell4", 1), ("G5-Cell16", 2)):
            matched = [
                row
                for row in sources
                if row["physical_cell_id"] == cell and row["part"] == part
            ]
            self.assertEqual([row["procedure_file_id"] for row in matched], [1, 2])

    def test_g2_cell4_original_multiple_step8_preflight_is_resolved_by_exact_reference(self) -> None:
        root = Path(__file__).resolve().parents[2]
        source = (
            root
            / "data"
            / "raw"
            / "oxford"
            / "part1"
            / "Group_2"
            / "Group 2"
            / "TPG2_1_Cell4.csv"
        )
        with self.assertRaisesRegex(CurvePreflightError, "found 2"):
            _read_verified_curve(root, source, 8, "discharge")
        reference8 = (
            "data/raw/oxford/part1/Group_2/Group 2/TPG2_1_Cell4.csv"
            "#Step8:segment1:rows50221-139442:sub1"
        )
        capacity8, voltage8, audit8 = _read_verified_curve(
            root, source, 8, "discharge", reference8
        )
        self.assertEqual(audit8["source_row_start"], 50221)
        self.assertEqual(audit8["source_row_end"], 139442)
        self.assertEqual(len(capacity8), len(voltage8))

        reference10 = (
            "data/raw/oxford/part1/Group_2/Group 2/TPG2_1_Cell4.csv"
            "#Step10:segment1:rows141244-230843:sub1"
        )
        _, _, audit10 = _read_verified_curve(root, source, 10, "charge", reference10)
        self.assertEqual(audit10["source_row_start"], 141244)
        self.assertEqual(audit10["source_row_end"], 230843)

    def test_step4_timeline_tracks_current_reversals_and_all_intervening_steps(self) -> None:
        with self.subTest("synthetic timeline"):
            from tempfile import TemporaryDirectory
            import csv

            with TemporaryDirectory() as temporary:
                path = Path(temporary) / "timeline.csv"
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=["Step", "TestTime", "Amphr", "Amps", "Volts"],
                    )
                    writer.writeheader()
                    writer.writerows(
                        [
                            {"Step": 4, "TestTime": 1, "Amphr": 1, "Amps": -1, "Volts": 4},
                            {"Step": 4, "TestTime": 2, "Amphr": 2, "Amps": 1, "Volts": 3},
                            {"Step": 6, "TestTime": 3, "Amphr": 2, "Amps": 0, "Volts": 3},
                            {"Step": 8, "TestTime": 4, "Amphr": 1, "Amps": -0.1, "Volts": 4},
                        ]
                    )
                result = _scan_timeline(path)
        step4 = result["step_runs"][0]
        self.assertEqual(step4["current_direction_changes"], 1)
        self.assertEqual(step4["current_sign_counts"], {"negative": 1, "zero": 0, "positive": 1})
        self.assertEqual([run["step"] for run in result["all_step_runs"]], [4, 6, 8])
        self.assertEqual(result["source_row_count"], 4)


if __name__ == "__main__":
    unittest.main()
