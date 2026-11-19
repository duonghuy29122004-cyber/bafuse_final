from __future__ import annotations

import json
import unittest

from bafuse.labeling.soh_label_generation import (
    build_part3_soh_labels,
    build_part1_soh_labels,
)


def _observation(
    measurement_id: str,
    procedure_file_id: int,
    source_file: str,
    *,
    boundary_status: str = "one_contiguous_step4_candidate",
    source_mapping_status: str = "unique_source_file",
    source_file_count: str = "1",
    dataset_part: str = "1",
) -> dict[str, str]:
    return {
        "dataset_part": dataset_part,
        "group_id": "1",
        "cell_id": "1",
        "physical_cell_id": "G1-Cell1",
        "procedure": f"TPG1.{procedure_file_id}",
        "procedure_file_id": str(procedure_file_id),
        "measurement_id": measurement_id,
        "rpt_order": "0" if measurement_id == "BoL" else measurement_id[3:],
        "measurement_mapping_status": "single_checkpoint",
        "source_file": source_file,
        "source_file_count": source_file_count,
        "source_mapping_status": source_mapping_status,
        "step4_candidate_source_file": "",
        "nominal_cycle_age": "",
        "cycle_age_status": "nominal_48_cycles_per_RPT",
        "step4_boundary_status": boundary_status,
    }


def _scan_row(
    source_file: str,
    segments: list[dict[str, object]],
    *,
    scan_status: str,
    dataset_parts: str = "1",
) -> dict[str, str]:
    return {
        "relative_path": source_file,
        "dataset_parts": dataset_parts,
        "scan_status": scan_status,
        "step4_segments_json": json.dumps(segments),
    }


def _segment(
    index: int,
    span: float,
    *,
    finite_rows: int = 20,
    current_signs: list[str] | None = None,
    positive_deltas: int = 0,
    negative_deltas: int = 0,
) -> dict[str, object]:
    return {
        "segment_index": index,
        "step4_rows": finite_rows,
        "capacity_finite_rows": finite_rows,
        "capacity_span": span,
        "current_signs": current_signs or ["negative"],
        "positive_capacity_deltas": positive_deltas,
        "negative_capacity_deltas": negative_deltas,
    }


class SOHLabelGenerationTests(unittest.TestCase):
    def test_part3_uses_only_its_same_cell_baseline_and_scan(self) -> None:
        observations = [
            _observation(
                "BoL",
                0,
                "TPG7.0-Cell4.csv",
                dataset_part="3",
            ),
            _observation(
                "RPT1",
                3,
                "TPG7.3-Cell4.csv",
                dataset_part="3",
            ),
        ]
        scan_rows = [
            _scan_row(
                "TPG7.0-Cell4.csv",
                [_segment(1, 2.0)],
                scan_status="one_contiguous_step4_candidate",
                dataset_parts="3",
            ),
            _scan_row(
                "TPG7.3-Cell4.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
                dataset_parts="3",
            ),
        ]

        labels, summary = build_part3_soh_labels(observations, scan_rows)
        rpt = next(row for row in labels if row["measurement_id"] == "RPT1")

        self.assertEqual(rpt["soh_percent"], 90.0)
        self.assertEqual(rpt["baseline_source_file"], "TPG7.0-Cell4.csv")
        self.assertEqual(rpt["label_source"], "Oxford Part 3 RPT Step 4 CSV")
        self.assertEqual(summary["dataset_part"], "3")
        self.assertEqual(summary["baseline_cell_count"], 1)

    def test_unidentified_part3_observations_are_quarantined(self) -> None:
        observations = [
            _observation(
                "BoL",
                0,
                "TPG7.0-Cell4.csv",
                dataset_part="3",
            ),
            _observation(
                "RPT1",
                3,
                "TPG7.3-Cell4.csv",
                dataset_part="3",
            ),
            {
                **_observation(
                    "BoL",
                    0,
                    "unidentified.csv",
                    dataset_part="3",
                ),
                "physical_cell_id": "",
            },
        ]
        scan_rows = [
            _scan_row(
                "TPG7.0-Cell4.csv",
                [_segment(1, 2.0)],
                scan_status="one_contiguous_step4_candidate",
                dataset_parts="3",
            ),
            _scan_row(
                "TPG7.3-Cell4.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
                dataset_parts="3",
            ),
        ]

        labels, summary = build_part3_soh_labels(observations, scan_rows)

        self.assertEqual(len(labels), 2)
        self.assertEqual(
            summary["quarantined_missing_physical_cell_id_observation_count"],
            1,
        )

    def test_capacity_span_outlier_is_retained_but_not_eligible_for_training(self) -> None:
        observations = [
            _observation("BoL", 1, "bol_first.csv"),
            _observation("BoL", 2, "bol_second.csv"),
            _observation("RPT1", 3, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol_first.csv",
                [_segment(1, 2.0, finite_rows=5000)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "bol_second.csv",
                [_segment(1, 1.99, finite_rows=5000)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 0.0076)],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, summary = build_part1_soh_labels(observations, scan_rows)
        rpt1 = next(row for row in labels if row["measurement_id"] == "RPT1")

        self.assertAlmostEqual(rpt1["soh_percent"], 0.38)
        self.assertEqual(
            rpt1["validation_status"],
            "review_capacity_span_outlier",
        )
        self.assertEqual(summary["numeric_rpt_label_count"], 0)
        self.assertEqual(
            summary["validation_status_counts"]["review_capacity_span_outlier"],
            1,
        )

    def test_selects_first_complete_bol_segment_and_emits_sensitivity(self) -> None:
        observations = [
            _observation("BoL", 1, "bol_first.csv", boundary_status="multiple_step4_segments"),
            _observation("BoL", 2, "bol_second.csv"),
            _observation("RPT1", 3, "rpt1.csv"),
            _observation(
                "RPT2",
                4,
                "rpt2.csv",
                boundary_status="multiple_step4_segments",
            ),
        ]
        scan_rows = [
            _scan_row(
                "bol_first.csv",
                [
                    _segment(1, 0.1, finite_rows=5),
                    _segment(2, 2.0),
                    _segment(3, 1.0),
                ],
                scan_status="multiple_step4_segments",
            ),
            _scan_row(
                "bol_second.csv",
                [_segment(1, 1.99)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt2.csv",
                [_segment(1, 1.7), _segment(2, 1.6)],
                scan_status="multiple_step4_segments",
            ),
        ]

        labels, summary = build_part1_soh_labels(observations, scan_rows)
        by_measurement = {
            (row["measurement_id"], row["measurement_order"]): row for row in labels
        }

        self.assertEqual(by_measurement[("BoL", "1")]["capacity_raw"], 2.0)
        self.assertEqual(
            by_measurement[("BoL", "1")]["baseline_segment_index"], 2
        )
        self.assertEqual(by_measurement[("BoL", "1")]["soh_percent"], 100.0)
        self.assertEqual(
            by_measurement[("BoL", "2")]["validation_status"],
            "secondary_bol_qc_only",
        )
        self.assertIsNone(by_measurement[("BoL", "2")]["soh_percent"])
        self.assertEqual(
            by_measurement[("BoL", "2")]["soh_percent_using_secondary_bol"],
            100.0,
        )
        self.assertEqual(by_measurement[("RPT1", "3")]["soh_percent"], 90.0)
        self.assertAlmostEqual(
            by_measurement[("RPT1", "3")]["soh_percent_using_secondary_bol"],
            100.0 * 1.8 / 1.99,
        )
        self.assertEqual(
            by_measurement[("RPT2", "4")]["validation_status"],
            "unavailable_multiple_step4_segments",
        )
        self.assertEqual(summary["baseline_cell_count"], 1)
        self.assertEqual(summary["numeric_rpt_label_count"], 1)

    def test_selects_unique_file_with_step4_among_duplicate_sources(self) -> None:
        baseline = _observation(
            "BoL",
            1,
            "bol_with_step4.csv;bol_without_step4.csv",
            boundary_status="multiple_step4_segments;no_step4_rows",
            source_mapping_status="multiple_source_files",
            source_file_count="2",
        )
        observations = [
            baseline,
            _observation("BoL", 2, "bol_second.csv"),
            _observation("RPT1", 3, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol_with_step4.csv",
                [_segment(1, 2.0), _segment(2, 1.0)],
                scan_status="multiple_step4_segments",
            ),
            _scan_row("bol_without_step4.csv", [], scan_status="no_step4_rows"),
            _scan_row(
                "bol_second.csv",
                [_segment(1, 1.99)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, _ = build_part1_soh_labels(observations, scan_rows)
        baseline_label = next(
            row
            for row in labels
            if row["measurement_id"] == "BoL" and row["measurement_order"] == "1"
        )

        self.assertEqual(
            baseline_label["validation_status"], "baseline_reference"
        )
        self.assertEqual(baseline_label["source_file"], "bol_with_step4.csv")
        self.assertEqual(
            baseline_label["source_candidates"],
            "bol_with_step4.csv;bol_without_step4.csv",
        )

    def test_multiple_step4_source_candidates_remain_ambiguous(self) -> None:
        observations = [
            _observation(
                "BoL",
                1,
                "bol_a.csv;bol_b.csv",
                boundary_status="multiple_step4_source_candidates",
                source_mapping_status="multiple_source_files",
                source_file_count="2",
            ),
            _observation("BoL", 2, "bol_second.csv"),
            _observation("RPT1", 3, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol_a.csv",
                [_segment(1, 2.0)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "bol_b.csv",
                [_segment(1, 2.1)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "bol_second.csv",
                [_segment(1, 1.99)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, summary = build_part1_soh_labels(observations, scan_rows)
        by_order = {row["measurement_order"]: row for row in labels}

        self.assertEqual(
            by_order["1"]["validation_status"],
            "unavailable_ambiguous_source_mapping",
        )
        self.assertEqual(
            by_order["3"]["validation_status"],
            "unavailable_no_measured_bol_reference",
        )
        self.assertEqual(summary["baseline_cell_count"], 0)

    def test_ambiguous_earliest_bol_does_not_fall_forward_to_later_bol(self) -> None:
        observations = [
            _observation(
                "BoL",
                1,
                "bol_ambiguous.csv;bol_other.csv",
                source_mapping_status="multiple_source_files",
                source_file_count="2",
            ),
            _observation("BoL", 2, "bol_later.csv"),
            _observation("RPT1", 3, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol_later.csv",
                [_segment(1, 1.9)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 1.8)],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, summary = build_part1_soh_labels(observations, scan_rows)
        by_order = {row["measurement_order"]: row for row in labels}

        self.assertEqual(
            by_order["1"]["validation_status"],
            "unavailable_ambiguous_source_mapping",
        )
        self.assertEqual(
            by_order["2"]["validation_status"],
            "secondary_bol_qc_only",
        )
        self.assertEqual(
            by_order["3"]["validation_status"],
            "unavailable_no_measured_bol_reference",
        )
        self.assertEqual(summary["baseline_cell_count"], 0)

    def test_mixed_current_signs_are_withheld(self) -> None:
        observations = [
            _observation("BoL", 1, "bol.csv"),
            _observation("RPT1", 2, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol.csv",
                [_segment(1, 2.0, current_signs=["negative"])],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [_segment(1, 1.8, current_signs=["negative", "positive"])],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, _ = build_part1_soh_labels(observations, scan_rows)
        rpt = next(row for row in labels if row["measurement_id"] == "RPT1")

        self.assertEqual(rpt["validation_status"], "review_mixed_current_signs")
        self.assertIsNone(rpt["soh_percent"])

    def test_capacity_direction_reversal_is_withheld(self) -> None:
        observations = [
            _observation("BoL", 1, "bol.csv"),
            _observation("RPT1", 2, "rpt1.csv"),
        ]
        scan_rows = [
            _scan_row(
                "bol.csv",
                [_segment(1, 2.0)],
                scan_status="one_contiguous_step4_candidate",
            ),
            _scan_row(
                "rpt1.csv",
                [
                    _segment(
                        1,
                        1.8,
                        positive_deltas=2,
                        negative_deltas=100,
                    )
                ],
                scan_status="one_contiguous_step4_candidate",
            ),
        ]

        labels, _ = build_part1_soh_labels(observations, scan_rows)
        rpt = next(row for row in labels if row["measurement_id"] == "RPT1")

        self.assertEqual(
            rpt["validation_status"], "review_capacity_direction_reversal"
        )
        self.assertIsNone(rpt["soh_percent"])

    def test_missing_source_is_not_reported_as_ambiguous(self) -> None:
        observations = [
            _observation("BoL", 1, "bol.csv"),
            _observation(
                "RPT1",
                2,
                "",
                source_mapping_status="source_file_not_found",
                source_file_count="0",
            ),
        ]
        scan_rows = [
            _scan_row(
                "bol.csv",
                [_segment(1, 2.0)],
                scan_status="one_contiguous_step4_candidate",
            )
        ]

        labels, _ = build_part1_soh_labels(observations, scan_rows)
        rpt = next(row for row in labels if row["measurement_id"] == "RPT1")

        self.assertEqual(
            rpt["validation_status"], "unavailable_source_file_not_found"
        )


if __name__ == "__main__":
    unittest.main()
