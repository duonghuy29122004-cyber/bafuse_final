from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch
from xml.etree import ElementTree

from bafuse.labeling.audit import (
    FileRecord,
    GuideEntry,
    _identify_file,
    read_guide_entries,
)
from bafuse.labeling.labels import (
    _stage_order,
    build_degradation_labels,
    build_soh_labels,
)
from bafuse.labeling.pipeline import _cell_identity_inventory
from bafuse.labeling.pydma_runner import run_pydma_for_available_cells


class DatasetIdentityTests(unittest.TestCase):
    def test_guide_parser_detects_table_headers_and_resets_cell_between_groups(self) -> None:
        namespace = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
        relationship_namespace = (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
        )
        package_namespace = (
            "http://schemas.openxmlformats.org/package/2006/relationships"
        )
        strings = [
            "Group 1 - cycling",
            "Cell",
            "Procedure",
            "RPT/Profile",
            "9",
            "TPG1",
            "BoL RPT",
            "TPG1.1",
            "RPT 1",
            "Group 2 - cycling",
            "3",
            "TPG2",
            "TPG2.1",
            "Group 3 - unresolved",
            "TPG3",
            "BoL RPT",
        ]
        shared_strings = ElementTree.Element(f"{{{namespace}}}sst")
        for value in strings:
            item = ElementTree.SubElement(shared_strings, f"{{{namespace}}}si")
            ElementTree.SubElement(item, f"{{{namespace}}}t").text = value

        workbook = ElementTree.Element(f"{{{namespace}}}workbook")
        sheets = ElementTree.SubElement(workbook, f"{{{namespace}}}sheets")
        ElementTree.SubElement(
            sheets,
            f"{{{namespace}}}sheet",
            {
                "name": "Summary",
                "sheetId": "1",
                f"{{{relationship_namespace}}}id": "rId1",
            },
        )
        ElementTree.SubElement(
            sheets,
            f"{{{namespace}}}sheet",
            {
                "name": "Sheet1",
                "sheetId": "2",
                f"{{{relationship_namespace}}}id": "rId2",
            },
        )

        relationships = ElementTree.Element(
            f"{{{package_namespace}}}Relationships"
        )
        for relationship_id, target in (
            ("rId1", "worksheets/sheet1.xml"),
            ("rId2", "worksheets/sheet2.xml"),
        ):
            ElementTree.SubElement(
                relationships,
                f"{{{package_namespace}}}Relationship",
                {
                    "Id": relationship_id,
                    "Type": (
                        "http://schemas.openxmlformats.org/officeDocument/2006/"
                        "relationships/worksheet"
                    ),
                    "Target": target,
                },
            )

        worksheet = ElementTree.Element(f"{{{namespace}}}worksheet")
        sheet_data = ElementTree.SubElement(
            worksheet, f"{{{namespace}}}sheetData"
        )
        rows = [
            {"A": 0},
            {"A": 1, "B": 2, "C": 3},
            {"A": 4, "B": 5, "C": 6},
            {"B": 7, "C": 8},
            {"A": 9},
            {"A": 1, "B": 2, "C": 3},
            {"A": 10, "B": 11, "C": 6},
            {"B": 12, "C": 8},
            {"A": 13},
            {"B": 14, "C": 15},
        ]
        for row_number, values in enumerate(rows, start=1):
            row = ElementTree.SubElement(
                sheet_data, f"{{{namespace}}}row", {"r": str(row_number)}
            )
            for column, string_index in values.items():
                cell = ElementTree.SubElement(
                    row,
                    f"{{{namespace}}}c",
                    {"r": f"{column}{row_number}", "t": "s"},
                )
                ElementTree.SubElement(
                    cell, f"{{{namespace}}}v"
                ).text = str(string_index)

        with tempfile.TemporaryDirectory() as temporary_directory:
            guide_path = Path(temporary_directory) / "guide.xlsx"
            with zipfile.ZipFile(guide_path, "w") as archive:
                archive.writestr("xl/workbook.xml", ElementTree.tostring(workbook))
                archive.writestr(
                    "xl/_rels/workbook.xml.rels",
                    ElementTree.tostring(relationships),
                )
                archive.writestr(
                    "xl/sharedStrings.xml",
                    ElementTree.tostring(shared_strings),
                )
                archive.writestr(
                    "xl/worksheets/sheet1.xml",
                    ElementTree.tostring(
                        ElementTree.Element(f"{{{namespace}}}worksheet")
                    ),
                )
                archive.writestr(
                    "xl/worksheets/sheet2.xml",
                    ElementTree.tostring(worksheet),
                )
            entries = read_guide_entries(guide_path)

        self.assertEqual(
            [
                (entry.group_id, entry.cell_id, entry.procedure, entry.stage)
                for entry in entries
            ],
            [
                ("1", "9", "TPG1", "BoL RPT"),
                ("1", "9", "TPG1.1", "RPT 1"),
                ("2", "3", "TPG2", "BoL RPT"),
                ("2", "3", "TPG2.1", "RPT 1"),
            ],
        )

    def test_cell_csv_identity_uses_group_and_cell(self) -> None:
        from pathlib import Path

        root = Path(r"D:\dataset")
        path = root / "Group_1" / "Group 1" / "TPG1_4_Cell15.csv"
        result = _identify_file(path, root)
        self.assertEqual(result["group_id"], "1")
        self.assertEqual(result["cell_id"], "15")
        self.assertEqual(result["physical_cell_id"], "G1-Cell15")
        self.assertEqual(result["file_id"], 4)

    def test_eis_identity_keeps_soc_and_stage(self) -> None:
        from pathlib import Path

        root = Path(r"D:\dataset")
        path = root / "EIS" / "EIS" / "MoL" / "Group1_MoL_Cell15_SOC50.csv"
        result = _identify_file(path, root)
        self.assertEqual(result["modality"], "EIS")
        self.assertEqual(result["condition"], "MoL EIS at SOC 50%")
        self.assertEqual(result["physical_cell_id"], "G1-Cell15")

    def test_rpt_order_uses_documented_rpt_number(self) -> None:
        self.assertLess(_stage_order("BoL RPT", 1), _stage_order("RPT 1", 4))
        self.assertLess(_stage_order("RPT 2", 6), _stage_order("RPT 10", 22))
        self.assertEqual(_stage_order("RPT 19 RPT 20", 30)[0], 10_000)

    def test_multi_rpt_guide_entry_is_retained_but_not_labeled_as_one_observation(
        self,
    ) -> None:
        record = FileRecord(
            relative_path="Group_5/TPG5_Cell6.csv",
            file_type="csv",
            size_bytes=1,
            modified_ns=1,
            format="CSV",
            decode_status="csv_scanned",
            group_id="5",
            cell_id="6",
            physical_cell_id="G5-Cell6",
            file_id=1,
            stage="Profile + RPT5-RPT8",
            modality="cycling_and_RPT",
            measurement_summary_json=(
                '{"step4_row_count": 100, "step4_capacity_raw_span": 2.8, '
                '"step4_current_signs": ["negative"]}'
            ),
        )
        guide_entry = GuideEntry(
            group_id="5",
            cell_id="6",
            procedure="TPG5",
            stage="Profile + RPT5-RPT8",
            file_id=1,
        )

        labels, validation = build_soh_labels([record], [guide_entry])

        self.assertEqual(len(labels), 1)
        self.assertIsNone(labels[0]["capacity_raw"])
        self.assertIsNone(labels[0]["soh_percent"])
        self.assertEqual(
            labels[0]["validation_status"], "unavailable_ambiguous_rpt_mapping"
        )
        self.assertEqual(validation[0]["status"], "unavailable_ambiguous_rpt_mapping")

    def test_duplicate_procedure_files_select_the_only_valid_step4_source(self) -> None:
        valid_record = FileRecord(
            relative_path="Group_4/TPG4_1_Cell18_1.csv",
            file_type="csv",
            size_bytes=1,
            modified_ns=1,
            format="CSV",
            decode_status="csv_scanned",
            group_id="4",
            cell_id="18",
            physical_cell_id="G4-Cell18",
            file_id=1,
            stage="BoL RPT",
            measurement_summary_json=(
                '{"step4_row_count": 50, "step4_capacity_raw_span": 2.9, '
                '"step4_current_signs": ["negative"]}'
            ),
        )
        other_record = FileRecord(
            relative_path="Group_4/TPG4_1_Cell18_2.csv",
            file_type="csv",
            size_bytes=1,
            modified_ns=1,
            format="CSV",
            decode_status="csv_scanned",
            group_id="4",
            cell_id="18",
            physical_cell_id="G4-Cell18",
            file_id=1,
            stage="BoL RPT",
            measurement_summary_json='{"step_counts": {"10": 20}}',
        )
        guide_entry = GuideEntry("4", "18", "TPG4", "BoL RPT", 1)

        labels, _ = build_soh_labels([valid_record, other_record], [guide_entry])

        self.assertEqual(labels[0]["source_file"], valid_record.relative_path)
        self.assertEqual(labels[0]["capacity_raw"], 2.9)
        self.assertEqual(labels[0]["validation_status"], "baseline_reference")
        self.assertIn("selected the unique CSV", labels[0]["validation_notes"])

    def test_duplicate_procedure_files_without_unique_step4_are_withheld(self) -> None:
        records = [
            FileRecord(
                relative_path=f"Group_4/source_{index}.csv",
                file_type="csv",
                size_bytes=1,
                modified_ns=1,
                format="CSV",
                decode_status="csv_scanned",
                group_id="4",
                cell_id="18",
                physical_cell_id="G4-Cell18",
                file_id=1,
                stage="BoL RPT",
                measurement_summary_json=json_summary,
            )
            for index, json_summary in enumerate(
                (
                    '{"step4_row_count": 50, "step4_capacity_raw_span": 2.9}',
                    '{"step4_row_count": 40, "step4_capacity_raw_span": 2.8}',
                ),
                start=1,
            )
        ]
        guide_entry = GuideEntry("4", "18", "TPG4", "BoL RPT", 1)

        labels, _ = build_soh_labels(records, [guide_entry])

        self.assertIsNone(labels[0]["capacity_raw"])
        self.assertIsNone(labels[0]["soh_percent"])
        self.assertEqual(
            labels[0]["validation_status"], "unavailable_ambiguous_source_mapping"
        )

    def test_filename_pairs_not_in_guide_are_not_counted_as_documented_cells(self) -> None:
        records = [
            FileRecord(
                relative_path="cycling.csv",
                file_type="csv",
                size_bytes=1,
                modified_ns=1,
                format="CSV",
                decode_status="csv_scanned",
                group_id="1",
                cell_id="15",
                physical_cell_id="G1-Cell15",
                modality="cycling_and_RPT",
            ),
            FileRecord(
                relative_path="extra_eis.csv",
                file_type="csv",
                size_bytes=1,
                modified_ns=1,
                format="CSV",
                decode_status="csv_scanned",
                group_id="10",
                cell_id="22",
                physical_cell_id="G10-Cell22",
                modality="EIS",
            ),
        ]
        guide = [GuideEntry("1", "15", "TPG1", "BoL RPT", 1)]

        counts = _cell_identity_inventory(records, guide)

        self.assertEqual(counts["documented_physical_cells"], 1)
        self.assertEqual(counts["filename_group_cell_pairs"], 2)
        self.assertEqual(counts["filename_pairs_not_in_guide"], 1)
        self.assertEqual(counts["unlinked_eis_group_cell_pairs"], 1)

    def test_batch_pydma_visits_only_guide_cells_with_matching_csv(self) -> None:
        records = [
            FileRecord(
                relative_path="g1c15.csv",
                file_type="csv",
                size_bytes=1,
                modified_ns=1,
                format="CSV",
                decode_status="csv_scanned",
                group_id="1",
                cell_id="15",
                file_id=1,
                modality="cycling_and_RPT",
            ),
            FileRecord(
                relative_path="g10c22.csv",
                file_type="csv",
                size_bytes=1,
                modified_ns=1,
                format="CSV",
                decode_status="csv_scanned",
                group_id="10",
                cell_id="22",
                modality="EIS",
            ),
        ]
        guide = [
            GuideEntry("1", "15", "TPG1", "BoL RPT", 1),
            GuideEntry("1", "20", "TPG1", "BoL RPT", 1),
        ]

        with patch(
            "bafuse.labeling.pydma_runner.run_pydma_for_cell",
            side_effect=lambda _root, _records, _guide, group, cell: [
                {"group_id": group, "cell_id": cell}
            ],
        ) as run_cell:
            result = run_pydma_for_available_cells(
                Path(r"D:\dataset"),
                records,
                guide,
            )

        self.assertEqual(result, [{"group_id": "1", "cell_id": "15"}])
        run_cell.assert_called_once()

    def test_imported_pydma_labels_use_strict_15_mv_gate(self) -> None:
        record = FileRecord(
            relative_path="Group_1/Group 1/TPG1.2-Cell9.csv",
            file_type="csv",
            size_bytes=1,
            modified_ns=1,
            format="CSV",
            decode_status="csv_scanned",
            group_id="1",
            cell_id="9",
            physical_cell_id="G1-Cell9",
            file_id=2,
        )
        guide = [GuideEntry("1", "9", "TPG1.2", "RPT 1", 2)]

        for rmse_mV, expected_success in ((14.999, True), (15.0, False)):
            with self.subTest(rmse_mV=rmse_mV), tempfile.TemporaryDirectory() as temp_dir:
                imported_path = Path(temp_dir) / "pydma_results.csv"
                with imported_path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(
                        stream,
                        fieldnames=(
                            "group_id",
                            "cell_id",
                            "checkup",
                            "direction",
                            "source_file",
                            "diagnostic_label_status",
                            "fit_rmse_mV",
                            "accepted_runs",
                            "total_runs",
                            "LLI_percent_vs_BoL",
                            "LAM_anode_percent_vs_BoL",
                            "LAM_cathode_percent_vs_BoL",
                        ),
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "group_id": "Group 1",
                            "cell_id": "Cell 9",
                            "checkup": "RPT1",
                            "direction": "charge",
                            "source_file": "TPG1.2-Cell9.csv",
                            "diagnostic_label_status": "diagnostic_estimate_accepted",
                            "fit_rmse_mV": rmse_mV,
                            "accepted_runs": 3,
                            "total_runs": 3,
                            "LLI_percent_vs_BoL": 1.0,
                            "LAM_anode_percent_vs_BoL": 2.0,
                            "LAM_cathode_percent_vs_BoL": 3.0,
                        }
                    )

                labels, _ = build_degradation_labels(
                    [record],
                    guide,
                    imported_path,
                )

            charge_label = labels[0]
            self.assertEqual(charge_label["fit_threshold_mV"], 15.0)
            self.assertEqual(charge_label["fit_success"], expected_success)
            if expected_success:
                self.assertEqual(charge_label["lli_percent"], 1.0)
            else:
                self.assertIsNone(charge_label["lli_percent"])
                self.assertEqual(
                    charge_label["identifiability_status"],
                    "rejected_import_validation",
                )


if __name__ == "__main__":
    unittest.main()
