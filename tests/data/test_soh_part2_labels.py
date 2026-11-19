from __future__ import annotations

import json
import unittest

from bafuse.labeling.soh_part2_labels import build_part2_soh_labels


def _observation(group: str, cell: str, measurement: str, source: str, **overrides: str) -> dict[str, str]:
    row = {
        "dataset_part": "2",
        "group_id": group,
        "cell_id": cell,
        "physical_cell_id": f"G{group}-Cell{cell}",
        "procedure": "TPG1.20",
        "procedure_file_id": "20",
        "measurement_id": measurement,
        "rpt_order": measurement[3:],
        "measurement_mapping_status": "single_checkpoint",
        "source_file": source,
        "source_file_count": "1",
        "source_mapping_status": "unique_source_file",
        "step4_candidate_source_file": "",
        "nominal_cycle_age": "",
        "cycle_age_status": "nominal_48_cycles_per_RPT",
    }
    row.update(overrides)
    return row


def _scan(source: str, span: float, status: str = "one_contiguous_step4_candidate") -> dict[str, str]:
    segment = {
        "segment_index": 1,
        "step4_rows": 20,
        "capacity_finite_rows": 20,
        "capacity_span": span,
        "current_signs": ["negative"],
        "positive_capacity_deltas": 0,
        "negative_capacity_deltas": 0,
    }
    return {
        "relative_path": source,
        "dataset_parts": "2",
        "scan_status": status,
        "step4_segments_json": json.dumps([segment]),
    }


_BASELINES = {
    "G1-Cell1": {
        "capacity": 3.0,
        "source_file": "p1.csv",
        "procedure": "TPG1",
        "segment": {"segment_index": 1},
    }
}


class Part2SOHLabelTests(unittest.TestCase):
    def test_groups_1_to_4_use_part1_same_cell_baseline_and_groups_5_6_are_withheld(self) -> None:
        observations = [
            _observation("1", "1", "RPT13", "a.csv"),
            _observation("1", "1", "RPT14", "b.csv", measurement_mapping_status="ambiguous_compound_stage"),
            _observation("5", "6", "RPT1", "c.csv"),
        ]
        scans = [_scan("a.csv", 2.4), _scan("b.csv", 2.0), _scan("c.csv", 2.0)]
        labels, summary = build_part2_soh_labels(observations, scans, _BASELINES)
        by_status = {row["measurement_id"]: row for row in labels}
        self.assertAlmostEqual(by_status["RPT13"]["soh_percent"], 80.0)
        self.assertEqual(by_status["RPT13"]["reference_capacity_raw"], 3.0)
        self.assertEqual(by_status["RPT13"]["baseline_source_file"], "p1.csv")
        self.assertEqual(by_status["RPT14"]["validation_status"], "unavailable_ambiguous_rpt_mapping")
        self.assertIsNone(by_status["RPT14"]["soh_percent"])
        self.assertEqual(by_status["RPT1"]["validation_status"], "withheld_group5_6_not_labeled")
        self.assertIsNone(by_status["RPT1"]["soh_percent"])
        self.assertEqual(summary["numeric_rpt_label_count"], 1)

    def test_missing_part1_baseline_and_multi_segment_are_not_labeled(self) -> None:
        observations = [
            _observation("2", "3", "RPT7", "a.csv"),
            _observation("1", "1", "RPT13", "b.csv"),
        ]
        scans = [_scan("a.csv", 2.4), _scan("b.csv", 2.4, status="multiple_step4_segments")]
        labels, _ = build_part2_soh_labels(observations, scans, _BASELINES)
        statuses = {row["physical_cell_id"]: row["validation_status"] for row in labels}
        self.assertEqual(statuses["G2-Cell3"], "unavailable_no_part1_same_cell_bol_reference")
        self.assertEqual(statuses["G1-Cell1"], "unavailable_multiple_step4_segments")

    def test_low_ratio_is_kept_for_review_but_not_counted(self) -> None:
        labels, summary = build_part2_soh_labels(
            [_observation("1", "1", "RPT13", "a.csv")], [_scan("a.csv", 1.0)], _BASELINES
        )
        self.assertEqual(labels[0]["validation_status"], "review_capacity_span_outlier")
        self.assertEqual(summary["numeric_rpt_label_count"], 0)


if __name__ == "__main__":
    unittest.main()
