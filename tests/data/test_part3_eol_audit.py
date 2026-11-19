from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from app.part3_eol_audit import _audit_step10_blocks


class Part3EolAuditTests(unittest.TestCase):
    def test_discontiguous_step10_blocks_are_not_promoted_to_one_curve(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "eol.csv"
            with source.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Step", "TestTime", "Amps", "Volts"))
                writer.writerows(
                    (
                        (10, 0.0, 1.0, 3.0),
                        (10, 1 / 60, 1.0, 3.1),
                        ("NaN", "NaN", "NaN", "NaN"),
                        ("NaN", "NaN", "NaN", "NaN"),
                        (10, 2 / 60, 1.0, 3.2),
                        (10, 3 / 60, 1.0, 3.3),
                        ("NaN", "NaN", "NaN", "NaN"),
                        ("NaN", "NaN", "NaN", "NaN"),
                        (10, 4 / 60, 1.0, 3.4),
                        (10, 5 / 60, 1.0, 3.5),
                    )
                )

            segments = json.dumps(
                [
                    {"start": 2, "end": 3},
                    {"start": 6, "end": 7},
                    {"start": 10, "end": 11},
                ]
            )
            result = _audit_step10_blocks(source, segments)

        self.assertTrue(result["blocks_valid"])
        self.assertTrue(result["separator_rows_nonfinite"])
        self.assertTrue(result["testtime_strictly_increasing_within_blocks"])
        self.assertTrue(result["testtime_strictly_increasing_across_blocks"])
        self.assertFalse(result["contiguous_extractor_eligible"])


if __name__ == "__main__":
    unittest.main()
