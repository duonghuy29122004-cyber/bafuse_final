from __future__ import annotations

import unittest

import numpy as np

from bafuse.labeling.run_part2_g5_candidate_pydma_audit import (
    CurveSegment,
    _map_candidate_rpts,
    _require_four_candidate_curves,
    _split_monotone_segments,
)


class Group5CandidatePyDMAAuditTests(unittest.TestCase):
    def test_splits_capacity_resets_and_preserves_source_row_boundaries(self) -> None:
        capacity = np.concatenate((np.arange(10), np.arange(10))).astype(float)
        voltage = np.linspace(3.0, 4.0, 20)
        source_rows = np.concatenate((np.arange(10, 20), np.arange(30, 40)))

        segments, audits = _split_monotone_segments(
            capacity, voltage, source_rows, "discharge", 8
        )

        self.assertEqual(len(segments), 2)
        self.assertEqual(len(audits), 2)
        self.assertEqual(
            [
                (segment.audit["source_row_start"], segment.audit["source_row_end"])
                for segment in segments
            ],
            [(10, 19), (30, 39)],
        )
        self.assertTrue(all(audit["valid_monotone_curve"] for audit in audits))

    def test_rejects_non_monotone_or_too_short_segments(self) -> None:
        capacity = np.asarray([0.0, 0.1, 0.2, 0.0, 0.0])
        voltage = np.asarray([3.0, 3.1, 3.2, 3.0, 3.1])
        source_rows = np.arange(5)

        segments, audits = _split_monotone_segments(
            capacity, voltage, source_rows, "charge", 10
        )

        self.assertEqual(segments, [])
        self.assertEqual(
            [audit["valid_monotone_curve"] for audit in audits],
            [False, False, False],
        )

    def test_requires_exactly_four_curves_in_each_direction(self) -> None:
        valid_segment = CurveSegment(
            capacity=np.arange(10, dtype=np.float64),
            voltage=np.arange(10, dtype=np.float64),
            audit={"segment_index": 1},
        )
        with self.assertRaisesRegex(ValueError, "expected exactly 4"):
            _require_four_candidate_curves(
                {"discharge": [valid_segment], "charge": [valid_segment] * 4},
                "TPG5.3-Cell5.csv",
            )

    def test_maps_candidate_rpts_in_valid_file_order(self) -> None:
        curves = {
            direction: [
                CurveSegment(
                    capacity=np.arange(10, dtype=np.float64),
                    voltage=np.arange(10, dtype=np.float64),
                    audit={"segment_index": segment_index},
                )
                for segment_index in (2, 5, 7, 10)
            ]
            for direction in ("discharge", "charge")
        }

        mapping = _map_candidate_rpts(curves, "TPG5.3-Cell5.csv")

        self.assertEqual(
            {
                rpt: segment.audit["segment_index"]
                for rpt, segment in mapping["discharge"].items()
            },
            {1: 2, 2: 5, 3: 7, 4: 10},
        )
        self.assertEqual(
            {
                rpt: segment.audit["segment_index"]
                for rpt, segment in mapping["charge"].items()
            },
            {1: 2, 2: 5, 3: 7, 4: 10},
        )


if __name__ == "__main__":
    unittest.main()
