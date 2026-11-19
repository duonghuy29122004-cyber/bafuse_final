from __future__ import annotations

import unittest

import pandas as pd

from bafuse.labeling.soh_part2_g5_g6_labels import (
    _capacity_ratio,
    _is_complete_discharge,
    _is_group5_rpt_signature,
    _select_unique_discharge_segment,
)
from bafuse.data.cycle_age import validated_cycle_age


def _segment(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "segment_index": 1,
        "step4_rows": 7000,
        "capacity_finite_rows": 7000,
        "capacity_span": 2.8,
        "positive_capacity_deltas": 6999,
        "negative_capacity_deltas": 0,
        "current_signs": ["negative"],
        "finite_voltage_samples": 7000,
        "voltage_min_v": 2.5,
        "voltage_max_v": 4.2,
        "post_steps": list(range(5, 37)),
    }
    result.update(overrides)
    return result


class Part2G5G6LabelTests(unittest.TestCase):
    def test_group5_mapping_requires_ordered_rpt_diagnostic_signature(self) -> None:
        self.assertTrue(_is_group5_rpt_signature(_segment()))
        self.assertFalse(_is_group5_rpt_signature(_segment(post_steps=[5, 6, 7, 8])))
        self.assertFalse(
            _is_group5_rpt_signature(_segment(post_steps=[*range(5, 35), 37, 36]))
        )

    def test_complete_discharge_qc_rejects_partial_or_wrong_direction_segments(self) -> None:
        self.assertTrue(_is_complete_discharge(_segment()))
        self.assertFalse(_is_complete_discharge(_segment(step4_rows=284)))
        self.assertFalse(_is_complete_discharge(_segment(current_signs=["positive"])))
        self.assertFalse(_is_complete_discharge(_segment(voltage_min_v=2.8)))

    def test_g6_mapping_requires_one_complete_discharge(self) -> None:
        partial = _segment(step4_rows=6, capacity_finite_rows=6, capacity_span=0.001)
        selected = _select_unique_discharge_segment(
            [_segment(current_signs=["positive"]), partial, _segment(segment_index=3)]
        )
        self.assertEqual(selected["segment_index"], 3)
        with self.assertRaisesRegex(ValueError, "found 0"):
            _select_unique_discharge_segment([partial])
        with self.assertRaisesRegex(ValueError, "found 2"):
            _select_unique_discharge_segment(
                [_segment(segment_index=1), _segment(segment_index=2)]
            )

    def test_capacity_ratio_uses_same_cell_reference(self) -> None:
        self.assertAlmostEqual(_capacity_ratio(2.5, 2.9), 100 * 2.5 / 2.9)
        with self.assertRaisesRegex(ValueError, "positive"):
            _capacity_ratio(2.5, 0)

    def test_non_nominal_age_is_opt_in_and_limited_to_part2_groups5_and6(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset_part": ["2", "2", "2"],
                "group_id": ["1", "5", "6"],
                "rpt_order": [1, 1, 2],
                "nominal_cycle_age": [48, None, None],
                "cycle_age_status": [
                    "nominal_48_cycles_per_RPT",
                    "cadence_not_established",
                    "calendar_aging_only",
                ],
            }
        )
        ages = validated_cycle_age(frame, allow_non_nominal_age=True)
        self.assertEqual(ages[0], 48)
        self.assertTrue(pd.isna(ages[1:]).all())
        with self.assertRaisesRegex(ValueError, "Nominal-age proxy"):
            validated_cycle_age(frame, allow_non_nominal_age=False)

    def test_non_nominal_age_rejects_unsupported_or_filled_rows(self) -> None:
        frame = pd.DataFrame(
            {
                "dataset_part": ["2"],
                "group_id": ["5"],
                "rpt_order": [1],
                "nominal_cycle_age": [48],
                "cycle_age_status": ["cadence_not_established"],
            }
        )
        with self.assertRaisesRegex(ValueError, "must not contain"):
            validated_cycle_age(frame, allow_non_nominal_age=True)
        frame.loc[0, "nominal_cycle_age"] = None
        frame.loc[0, "group_id"] = "4"
        with self.assertRaisesRegex(ValueError, "Unsupported cycle-age"):
            validated_cycle_age(frame, allow_non_nominal_age=True)


if __name__ == "__main__":
    unittest.main()
