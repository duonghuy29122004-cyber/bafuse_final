from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from bafuse.data.pydma_stage2 import (
    CurvePreflightError,
    _build_negative_label_audit,
    _read_verified_curve,
)


class PyDMAPilotCurveTests(unittest.TestCase):
    def test_negative_audit_preserves_fit_ids_and_flags_seed_sign_changes(self) -> None:
        rows = [
            {
                "dataset_part": "1",
                "physical_cell_id": "G1-Cell9",
                "rpt_index": "2",
                "direction": "discharge",
                "rpt_fit_id": "seed211",
                "lli_percent": "-25",
                "lam_anode_percent": "-40",
                "lam_cathode_percent": "10",
                "historical_lli_percent": "-25",
                "historical_lam_anode_percent": "-40",
                "historical_lam_cathode_percent": "10",
            },
            {
                "dataset_part": "1",
                "physical_cell_id": "G1-Cell9",
                "rpt_index": "2",
                "direction": "discharge",
                "rpt_fit_id": "seed212",
                "lli_percent": "4",
                "lam_anode_percent": "8",
                "lam_cathode_percent": "7",
                "historical_lli_percent": "-25",
                "historical_lam_anode_percent": "-40",
                "historical_lam_cathode_percent": "10",
            },
        ]
        audited = _build_negative_label_audit(rows)
        self.assertEqual(len(audited), 2)
        self.assertEqual(audited[0]["rpt_fit_id"], "seed211")
        self.assertEqual(audited[1]["rpt_fit_id"], "seed212")
        self.assertEqual(
            audited[0]["negative_estimate_classification"], "OPTIMIZER_UNSTABLE"
        )
        self.assertTrue(audited[0]["historical_negative_present"])

    def test_exact_manifest_segment_uses_source_line_boundaries(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "raw" / "oxford" / "part1" / "test.csv"
            path.parent.mkdir(parents=True)
            rows = [
                {"Step": 8, "Amphr": value, "Amps": -0.1, "Volts": 4.1 - value / 20}
                for value in range(12)
            ]
            rows.extend(
                {"Step": 10, "Amphr": value, "Amps": 0.1, "Volts": 3.0 + value / 20}
                for value in range(12)
            )
            pd.DataFrame(rows).to_csv(path, index=False)
            reference = (
                "data/raw/oxford/part1/test.csv#Step8:segment1:rows2-13:sub1"
            )
            capacity, voltage, audit = _read_verified_curve(
                root, path, 8, "discharge", reference
            )
            self.assertEqual(len(capacity), 12)
            self.assertEqual(len(voltage), 12)
            self.assertEqual(audit["source_row_start"], 2)
            self.assertEqual(audit["source_row_end"], 13)
            self.assertEqual(audit["status"], "verified_exact_manifest_boundary")

    def test_baseline_requires_one_unique_valid_curve(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "data" / "raw" / "oxford" / "part1" / "test.csv"
            path.parent.mkdir(parents=True)
            rows = [
                {"Step": 8, "Amphr": value, "Amps": -0.1, "Volts": 4.1 - value / 20}
                for value in range(12)
            ]
            pd.DataFrame(rows).to_csv(path, index=False)
            _, _, audit = _read_verified_curve(root, path, 8, "discharge")
            self.assertEqual(audit["status"], "unique_bol_curve_candidate")
            with self.assertRaises(CurvePreflightError):
                _read_verified_curve(
                    root,
                    path,
                    8,
                    "discharge",
                    "data/raw/oxford/part1/test.csv#Step8:segment1:rows3-13:sub1",
                )


if __name__ == "__main__":
    unittest.main()
