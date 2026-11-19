from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bafuse.labeling.audit import FileRecord
from bafuse.labeling.run_part2_pydma import _build_cell_plans
from bafuse.labeling.pydma_runner import _record_source_path


def _row(
    *,
    part: str,
    group: str,
    cell: str,
    measurement: str,
    file_id: str,
    stage: str,
    mapping: str = "single_checkpoint",
    source_mapping: str = "unique_source_file",
) -> dict[str, str]:
    return {
        "dataset_part": part,
        "group_id": group,
        "cell_id": cell,
        "physical_cell_id": f"G{group}-Cell{cell}",
        "measurement_id": measurement,
        "procedure_file_id": file_id,
        "guide_stage": stage,
        "procedure": f"TPG{group}",
        "measurement_mapping_status": mapping,
        "source_mapping_status": source_mapping,
        "source_file_count": "1",
        "source_file": f"Group_{group}/source_{file_id}.csv",
    }


class Part2PyDMACrosswalkTests(unittest.TestCase):
    def test_selects_same_cell_part1_baseline_and_part2_group6_baseline(self) -> None:
        rows = [
            _row(
                part="1",
                group="1",
                cell="9",
                measurement="BoL",
                file_id="2",
                stage="BoL RPT",
            ),
            _row(
                part="1",
                group="1",
                cell="9",
                measurement="BoL",
                file_id="1",
                stage="BoL RPT",
            ),
            _row(
                part="2",
                group="1",
                cell="9",
                measurement="RPT1",
                file_id="3",
                stage="RPT1",
            ),
            _row(
                part="2",
                group="6",
                cell="1",
                measurement="BoL",
                file_id="1",
                stage="BoL",
            ),
            _row(
                part="2",
                group="6",
                cell="1",
                measurement="RPT1",
                file_id="2",
                stage="RPT1",
            ),
        ]

        plans = _build_cell_plans(rows)

        self.assertEqual(set(plans), {"G1-Cell9", "G6-Cell1"})
        self.assertEqual(plans["G1-Cell9"]["baseline"]["dataset_part"], "1")
        self.assertEqual(plans["G1-Cell9"]["baseline"]["procedure_file_id"], "1")
        self.assertEqual(len(plans["G1-Cell9"]["targets"]), 1)
        self.assertEqual(plans["G6-Cell1"]["baseline"]["dataset_part"], "2")

    def test_excludes_compound_group5_and_non_unique_source_mappings(self) -> None:
        rows = [
            _row(
                part="2",
                group="5",
                cell="6",
                measurement="BoL",
                file_id="1",
                stage="BoL RPT",
            ),
            _row(
                part="2",
                group="5",
                cell="6",
                measurement="RPT1+RPT2",
                file_id="2",
                stage="RPT1 + RPT2",
                mapping="ambiguous_compound_stage",
            ),
            _row(
                part="1",
                group="2",
                cell="3",
                measurement="BoL",
                file_id="1",
                stage="BoL RPT",
            ),
            _row(
                part="2",
                group="2",
                cell="3",
                measurement="RPT1",
                file_id="2",
                stage="RPT1",
                source_mapping="ambiguous_source_mapping",
            ),
        ]

        self.assertEqual(_build_cell_plans(rows), {})

    def test_requires_same_cell_part1_baseline_for_continuation_group(self) -> None:
        rows = [
            _row(
                part="2",
                group="1",
                cell="9",
                measurement="RPT1",
                file_id="2",
                stage="RPT1",
            )
        ]

        with self.assertRaisesRegex(ValueError, "No unique same-cell BoL source"):
            _build_cell_plans(rows)

    def test_pydma_runner_resolves_absolute_cross_part_reference_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            shared = Path(temporary_directory)
            part1_source = shared / "part1" / "bol.csv"
            part1_source.parent.mkdir()
            part1_source.write_text("source", encoding="utf-8")
            part2_root = shared / "part2"
            part2_root.mkdir()
            record = FileRecord(
                relative_path=str(part1_source),
                file_type="csv",
                size_bytes=part1_source.stat().st_size,
                modified_ns=part1_source.stat().st_mtime_ns,
                format="CSV",
                decode_status="test",
            )

            self.assertEqual(
                _record_source_path(part2_root, record),
                part1_source,
            )


if __name__ == "__main__":
    unittest.main()
