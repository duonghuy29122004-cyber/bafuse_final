from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bafuse.labeling.pydma_crosswalk import cell_inputs
from bafuse.labeling.run_part3_pydma import _build_cell_plans


def _row(
    *,
    part: str = "3",
    group: str = "7",
    cell: str = "4",
    measurement: str,
    file_id: str,
    stage: str,
    mapping: str = "single_checkpoint",
    source_mapping: str = "unique_source_file",
    source_file_count: str = "1",
    source_file_id: str | None = None,
    source_mapping_rule: str = "guide_procedure_file_id",
) -> dict[str, str]:
    is_bol = measurement.casefold() == "bol"
    mapped_source_id = source_file_id or ("0" if is_bol else file_id)
    mapping_rule = (
        source_mapping_rule
        if not is_bol or source_mapping_rule != "guide_procedure_file_id"
        else "part3_unsuffixed_bol_to_zero_suffix"
    )
    return {
        "dataset_part": part,
        "group_id": group,
        "cell_id": cell,
        "physical_cell_id": f"G{group}-Cell{cell}",
        "measurement_id": measurement,
        "procedure_file_id": file_id,
        "guide_stage": stage,
        "procedure": f"TPG{group}" if is_bol else f"TPG{group}.{file_id}",
        "measurement_mapping_status": mapping,
        "source_file": (
            f"Group_{group}/TPG{group}.0-Cell{cell}.csv"
            if is_bol
            else f"Group_{group}/TPG{group}.{file_id}-Cell{cell}.csv"
        ),
        "source_file_id": mapped_source_id,
        "source_mapping_rule": mapping_rule,
        "source_file_count": source_file_count,
        "source_mapping_status": source_mapping,
        "source_file_size_bytes": "4",
    }


class Part3PyDMACrosswalkTests(unittest.TestCase):
    def test_selects_unique_same_cell_part3_bol_and_single_checkpoint_rpt(self) -> None:
        rows = [
            _row(
                part="1",
                measurement="BoL",
                file_id="1",
                stage="BoL RPT",
            ),
            _row(
                measurement="BoL",
                file_id="1",
                stage="BoL",
            ),
            _row(
                measurement="RPT1",
                file_id="3",
                stage="RPT1",
            ),
            _row(
                measurement="RPT2+RPT3",
                file_id="5",
                stage="RPT2 + RPT3",
                mapping="ambiguous_compound_stage",
            ),
        ]

        plans = _build_cell_plans(rows)

        self.assertEqual(set(plans), {"G7-Cell4"})
        self.assertEqual(plans["G7-Cell4"]["baseline"]["dataset_part"], "3")
        self.assertEqual(
            [row["measurement_id"] for row in plans["G7-Cell4"]["targets"]],
            ["RPT1"],
        )

    def test_rejects_part3_bol_that_does_not_use_corrected_zero_suffix_mapping(self) -> None:
        baseline = _row(
            measurement="BoL",
            file_id="1",
            stage="BoL",
            source_file_id="1",
            source_mapping_rule="guide_procedure_file_id",
        )
        target = _row(measurement="RPT1", file_id="3", stage="RPT1")

        with self.assertRaisesRegex(ValueError, "corrected unsuffixed-to-.0 crosswalk"):
            _build_cell_plans([baseline, target])

    def test_requires_unique_same_cell_bol_baseline(self) -> None:
        target = _row(measurement="RPT1", file_id="3", stage="RPT1")

        with self.assertRaisesRegex(ValueError, "Expected one unique same-cell Part 3 BoL"):
            _build_cell_plans([target])

    def test_builds_pydma_inputs_and_directional_targets_from_manifest_files(self) -> None:
        baseline = _row(measurement="BoL", file_id="1", stage="BoL")
        target = _row(measurement="RPT1", file_id="3", stage="RPT1")
        plans = _build_cell_plans([baseline, target])
        plan = plans["G7-Cell4"]

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for row in (baseline, target):
                path = root / Path(row["source_file"])
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"data")
            records, entries, fit_targets = cell_inputs(plan, {"3": root})

        self.assertEqual(len(records), 2)
        self.assertEqual(len(entries), 2)
        self.assertEqual(fit_targets, {(2, "charge"), (2, "discharge")})
        self.assertTrue(all(Path(record.relative_path).is_absolute() for record in records))


if __name__ == "__main__":
    unittest.main()
