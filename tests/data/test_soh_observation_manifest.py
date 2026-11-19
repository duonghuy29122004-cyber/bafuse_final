from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bafuse.labeling.audit import GuideEntry
from bafuse.data.soh_observation_manifest import (
    _cycle_age,
    _measurement_identity,
    build_observation_manifest,
)


class SOHObservationManifestTests(unittest.TestCase):
    def test_measurement_identity_only_accepts_single_checkpoint_stages(self) -> None:
        self.assertEqual(_measurement_identity("BoL"), ("BoL", 0, "single_checkpoint"))
        self.assertEqual(_measurement_identity("BoL RPT"), ("BoL", 0, "single_checkpoint"))
        self.assertEqual(
            _measurement_identity("RPT 12"), ("RPT12", 12, "single_checkpoint")
        )
        self.assertEqual(
            _measurement_identity("Profile + RPT5-RPT8"),
            ("", None, "ambiguous_compound_stage"),
        )
        self.assertEqual(
            _measurement_identity("RPT 19 RPT 20"),
            ("", None, "ambiguous_compound_stage"),
        )

    def test_cycle_age_keeps_documented_and_unsupported_groups_distinct(self) -> None:
        self.assertEqual(_cycle_age(2, "3", 12), ("576", "nominal_48_cycles_per_RPT"))
        self.assertEqual(_cycle_age(2, "5", 12), ("", "cadence_not_established"))
        self.assertEqual(_cycle_age(2, "6", 12), ("", "calendar_aging_only"))
        self.assertEqual(_cycle_age(2, "4", None), ("", "unresolved_checkpoint"))

    def test_manifest_crosswalks_unique_sources_and_retains_ambiguous_stages(self) -> None:
        entries = [
            GuideEntry("5", "6", "TPG5", "BoL", 1),
            GuideEntry("5", "6", "TPG5.2", "Profile + RPT5-RPT8", 2),
            GuideEntry("5", "6", "TPG5.3", "RPT 9", 3),
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "part2"
            (root / "Group_5").mkdir(parents=True)
            (root / "Guide_to_Datafiles_2.xlsx").write_bytes(b"guide fixture")
            for name in (
                "TPG5_1_Cell6.csv",
                "TPG5_2_Cell6.csv",
                "TPG5_3_Cell6.csv",
            ):
                (root / "Group_5" / name).write_text("", encoding="utf-8")
            with patch(
                "bafuse.data.soh_observation_manifest.read_guide_entries",
                return_value=entries,
            ):
                rows, summary = build_observation_manifest({2: root})

        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["measurement_id"], "BoL")
        self.assertEqual(rows[0]["nominal_cycle_age"], "")
        self.assertEqual(rows[0]["cycle_age_status"], "cadence_not_established")
        compound = next(row for row in rows if row["guide_stage"] == "Profile + RPT5-RPT8")
        self.assertEqual(compound["measurement_mapping_status"], "ambiguous_compound_stage")
        self.assertEqual(compound["step4_boundary_status"], "blocked_compound_guide_stage")
        rpt = next(row for row in rows if row["guide_stage"] == "RPT 9")
        self.assertEqual(rpt["measurement_id"], "RPT9")
        self.assertEqual(rpt["nominal_cycle_age"], "")
        self.assertEqual(rpt["source_mapping_status"], "unique_source_file")
        self.assertEqual(summary["candidate_rows"], 3)
        self.assertEqual(summary["unique_source_file_rows"], 3)

    def test_part3_unsuffixed_bol_maps_to_zero_suffix_source(self) -> None:
        entry = GuideEntry("7", "4", "TPG7", "BoL", 1)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "part3"
            source_dir = root / "Group_7" / "Group 7"
            source_dir.mkdir(parents=True)
            (root / "Guide_to_Datafiles_3.xlsx").write_bytes(b"guide fixture")
            for name in (
                "TPG7_0_Cell4.csv",
                "TPG7_1_Cell4.csv",
                "TPG7_3_Cell4.csv",
            ):
                (source_dir / name).write_text("", encoding="utf-8")
            with patch(
                "bafuse.data.soh_observation_manifest.read_guide_entries",
                return_value=[entry],
            ):
                rows, _ = build_observation_manifest({3: root})

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["procedure_file_id"], 1)
        self.assertEqual(rows[0]["source_file_id"], 0)
        self.assertEqual(
            rows[0]["source_mapping_rule"],
            "part3_unsuffixed_bol_to_zero_suffix",
        )
        self.assertTrue(rows[0]["source_file"].endswith("TPG7_0_Cell4.csv"))


if __name__ == "__main__":
    unittest.main()
