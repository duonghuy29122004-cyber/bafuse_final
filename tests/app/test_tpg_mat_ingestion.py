from __future__ import annotations

import tempfile
import unittest
import csv
import shutil
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from array import array

from app.tpg_mat_ingestion import (
    MATLAB_CONVERTER_SCRIPT,
    ROOT,
    MatlabRuntimeUnavailable,
    TpgIngestionError,
    analyze_tpg_file,
    analyze_tpg_mat,
    calculate_soh_metrics,
    guide_mapping_for_file,
    guide_mapping_for_mat,
    inspect_tpg_mat,
    validate_mat_csv_equivalence,
    _reference_labels,
    _scan_approved_rpt10_step10,
    _approved_rpt10_zero_current_evidence,
    _inspection_result,
    _demo_eol_reference_labels,
    _verified_upload_context,
    _verified_feature_rows,
)
from experiments.soh_forecast.bafuse_three_encoder_v1.run_phase2 import (
    _scan_source_requests,
)
from app.inference_dashboard import InferenceEngine


REAL_RPT_MAT = (
    ROOT
    / "data"
    / "raw"
    / "oxford"
    / "part2"
    / "Group_1"
    / "Group 1"
    / "TPG1.21-Cell20.mat"
)


class TpgMatIngestionTests(unittest.TestCase):
    def test_part3_rpt_source_uses_checksum_from_verified_frozen_manifest(self) -> None:
        mapping, observations = guide_mapping_for_file(
            "TPG7.6-Cell4.csv",
            require_source_sha256=True,
        )

        self.assertEqual(mapping["physical_cell_id"], "G7-Cell4")
        self.assertEqual(mapping["guide_entry_ids"], "P3|Sheet1|row10|G7-Cell4|TPG7.6")
        self.assertEqual(mapping["guide_rpt_refs"], "RPT3")
        self.assertEqual(mapping["source_sha256"], "dbb18bb421055b2f2be9f582a4873f1fe9211ca8d291c86dabb49dee4a2a0814")
        self.assertEqual(
            mapping["source_sha256_provenance"],
            "verified_frozen_final2_observation_manifest",
        )
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["rpt_index"], "3")

    def test_uploaded_csv_with_incorrect_source_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "data" / "raw.csv"
            reference.parent.mkdir(parents=True)
            reference.write_text(
                "Step,Amphr,Amps,Volts\n4,0,-1,4.0\n", encoding="utf-8"
            )
            uploaded = root / "uploaded.csv"
            uploaded.write_text(
                "Step,Amphr,Amps,Volts\n4,0,-1,3.9\n", encoding="utf-8"
            )
            mapping = {
                "source_present": "true",
                "repository_path": "data/raw.csv",
                "source_sha256": "0" * 64,
            }
            with (
                patch("app.tpg_mat_ingestion.ROOT", root),
                patch(
                    "app.tpg_mat_ingestion.guide_mapping_for_file",
                    return_value=(mapping, []),
                ),
            ):
                with self.assertRaisesRegex(TpgIngestionError, "do not match"):
                    _verified_upload_context(uploaded, "raw.csv")

    def test_analyze_tpg_file_extracts_only_the_explicitly_selected_rpt(self) -> None:
        observations = [
            {"part": "3", "physical_cell_id": "G7-Cell4", "rpt_index": "3"},
            {"part": "3", "physical_cell_id": "G7-Cell4", "rpt_index": "4"},
        ]
        mapping = {
            "dataset_part": "3",
            "physical_cell_id": "G7-Cell4",
            "procedure": "TPG7",
            "source_checksum_verified": "true",
        }
        captured: dict[str, object] = {}

        def capture_analysis(
            source_csv: Path,
            original_filename: str,
            selected_mapping: dict[str, str],
            selected_observations: list[dict[str, str]],
            validated_rows: int,
            engine: object,
            model_name: str | None,
        ) -> dict[str, object]:
            captured["observations"] = selected_observations
            return {"selected_model": model_name}

        with (
            patch(
                "app.tpg_mat_ingestion._verified_upload_context",
                return_value=(mapping, observations, Path("reference.csv")),
            ),
            patch("app.tpg_mat_ingestion._count_verified_csv_rows", return_value=2),
            patch(
                "app.tpg_mat_ingestion._analyze_verified_csv",
                side_effect=capture_analysis,
            ),
        ):
            result = analyze_tpg_file(
                Path("uploaded.csv"),
                "TPG7.csv",
                object(),
                model_name="CNN1D-Step4-Baseline",
                selected_rpt_index="RPT4",
                require_source_sha256=True,
            )

        self.assertEqual(result["selected_model"], "CNN1D-Step4-Baseline")
        self.assertEqual(
            [row["rpt_index"] for row in captured["observations"]],
            ["4"],
        )

    def test_multiple_mapped_rpts_require_an_explicit_selection(self) -> None:
        observations = [
            {"rpt_index": "3"},
            {"rpt_index": "4"},
        ]
        with patch(
            "app.tpg_mat_ingestion._verified_upload_context",
            return_value=({}, observations, Path("reference.csv")),
        ):
            with self.assertRaisesRegex(TpgIngestionError, "explicit RPT selection"):
                analyze_tpg_file(
                    Path("uploaded.csv"),
                    "TPG7.csv",
                    object(),
                    require_source_sha256=True,
                )

    def test_legacy_analyzer_keeps_all_rpts_without_final2_selection(self) -> None:
        observations = [
            {"rpt_index": "3"},
            {"rpt_index": "4"},
        ]
        captured: dict[str, object] = {}

        def capture_analysis(
            source_csv: Path,
            original_filename: str,
            mapping: dict[str, str],
            selected_observations: list[dict[str, str]],
            validated_rows: int,
            engine: object,
            model_name: str | None,
        ) -> dict[str, object]:
            captured["observations"] = selected_observations
            return {"selected_model": model_name}

        with (
            patch(
                "app.tpg_mat_ingestion._verified_upload_context",
                return_value=({}, observations, Path("reference.csv")),
            ),
            patch("app.tpg_mat_ingestion._validate_source_equivalence", return_value=2),
            patch(
                "app.tpg_mat_ingestion._analyze_verified_csv",
                side_effect=capture_analysis,
            ),
        ):
            analyze_tpg_file(
                Path("uploaded.csv"),
                "TPG7.csv",
                object(),
            )

        self.assertEqual(
            [row["rpt_index"] for row in captured["observations"]],
            ["3", "4"],
        )

    def test_owner_confirmed_frozen_mapping_is_exact_and_preserves_raw_ambiguity(self) -> None:
        mapping, observations = guide_mapping_for_file("TPG10.19-Cell2.csv")
        self.assertEqual(mapping["dataset_part"], "3")
        self.assertEqual(mapping["group_id"], "10")
        self.assertEqual(mapping["physical_cell_id"], "G10-Cell2")
        self.assertEqual(mapping["mapping_resolution_status"], "owner_confirmed_frozen_final2_exception")
        self.assertEqual(
            mapping["guide_source_mapping_status"],
            "candidate_only_for_unassigned_Guide_cell; quarantine identity",
        )
        self.assertEqual(mapping["source_sha256"], "7af1228384504c2e54b32088528cae7b5a4a043e899a365a1d4bcf6ed28e8fcb")
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0]["rpt_index"], "10")
        self.assertEqual(
            observations[0]["mapping_resolution_provenance"],
            "Group 10 Cell 2 identity confirmed by project owner; explicit Cell 2 label omitted from printed Guide.",
        )
        inspection = _inspection_result(mapping, observations)
        self.assertIn("not_independently_guide_mapped", inspection["mapping_status"])

        with self.assertRaisesRegex(TpgIngestionError, "ambiguous Guide-to-cell mapping"):
            guide_mapping_for_file("TPG10.0-Cell2.csv")

    def test_guide_verified_eol_mapping_is_demo_only_and_keeps_step10_review(self) -> None:
        mapping, observations = guide_mapping_for_file("TPG7.EoL-Cell4.csv")

        self.assertEqual(mapping["mapping_resolution_status"], "GUIDE_VERIFIED")
        self.assertEqual(mapping["inventory_scope"], "VERIFIED_DEMO_ONLY")
        self.assertEqual(
            mapping["source_sha256"],
            "5ce33bff5aad11f4cfa039ffd755d45bfbe1d576272947c5065d16817c8bcf15",
        )
        self.assertEqual(len(observations), 1)
        observation = observations[0]
        self.assertEqual(observation["physical_cell_id"], "G7-Cell4")
        self.assertEqual(observation["rpt_index"], "7")
        self.assertEqual(observation["step4_status"], "verified")
        self.assertEqual(observation["step10_status"], "not_mapped")
        self.assertEqual(observation["inventory_scope"], "VERIFIED_DEMO_ONLY")

    def test_eol_provisional_reference_uses_same_cell_bol_and_is_not_evaluable(self) -> None:
        _, observations = guide_mapping_for_file("TPG7.EoL-Cell4.csv")
        observations[0]["source_checksum_verified"] = "true"
        source = ROOT / Path(observations[0]["source_file"])

        reference = _demo_eol_reference_labels(source, observations)["7"]

        self.assertEqual(
            reference["status"],
            "demo_only_provisional_capacity_ratio_not_evaluation_eligible",
        )
        self.assertFalse(reference["evaluation_eligible"])
        self.assertEqual(
            reference["bol_source_file"],
            "data/raw/oxford/part3/Group_7/Group 7/TPG7.0-Cell4.csv",
        )
        self.assertGreater(reference["value_percent"], 0)

    def test_only_approved_g10_cell2_rpt10_excludes_its_zero_current_row(self) -> None:
        scratch = ROOT / "tests" / "_rpt10_zero_current_fixture"
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir()
        try:
            source = scratch / "TPG10.19-Cell2.csv"
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("Step", "Volts", "Amps"))
                writer.writerows(
                    (
                        (10, 3.8, 1.0),
                        (10, 3.9, 0.0),
                        (10, 4.0, 1.0),
                        (10, 4.1, 1.0),
                    )
                )
            eligibility = scratch / "eligibility.csv"
            fields = (
                "dataset_part",
                "group",
                "physical_cell_id",
                "rpt_index",
                "source_tpg",
                "source_file",
                "step10_segment_reference",
                "step10_inference_eligible",
                "step10_zero_current_policy",
                "step10_zero_current_excluded_count",
                "step10_zero_current_source_line",
            )
            with eligibility.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(
                    {
                        "dataset_part": "3",
                        "group": "10",
                        "physical_cell_id": "G10-Cell2",
                        "rpt_index": "10",
                        "source_tpg": "TPG10.19",
                        "source_file": (
                            "data/raw/oxford/part3/Group_10/Group 10/"
                            "TPG10.19-Cell2.csv"
                        ),
                        "step10_segment_reference": (
                            "data/raw/oxford/part3/Group_10/Group 10/"
                            "TPG10.19-Cell2.csv:rows2833891-2910695"
                        ),
                        "step10_inference_eligible": "true",
                        "step10_zero_current_policy": "exclude_zero_current",
                        "step10_zero_current_excluded_count": "1",
                        "step10_zero_current_source_line": "2880746",
                    }
                )

            def request(rpt: str) -> dict[str, object]:
                row = {
                    "part": "3",
                    "group": "10",
                    "physical_cell_id": "G10-Cell2",
                    "rpt_index": rpt,
                    "source_file": (
                        "data/raw/oxford/part3/Group_10/Group 10/"
                        "TPG10.19-Cell2.csv"
                    ),
                    "mapping_status": "GUIDE_VERIFIED",
                    "step10_status": "verified",
                }
                return {
                    "row": row,
                    "step": 10,
                    "start": 2,
                    "end": 5,
                    "expected_rows": 4,
                    "observed_rows": 0,
                    "step_ids": set(),
                    "voltage": array("f"),
                    "current": array("f"),
                    "temperature": array("f"),
                    "step_time": array("d"),
                    "missing_voltage": 0,
                    "missing_current": 0,
                    "missing_temperature": 0,
                    "missing_step_time": 0,
                    "source": str(source),
                }

            approved_request = request("10")
            approved_request["start"] = 2833891
            approved_request["end"] = 2910695
            approved_request["expected_rows"] = 76805
            evidence = _approved_rpt10_zero_current_evidence(
                approved_request, source, eligibility
            )
            self.assertEqual(
                evidence,
                {"source_line": 2880746, "start": 2833891, "end": 2910695},
            )

            scan_request = request("10")
            _scan_approved_rpt10_step10(
                ROOT,
                scan_request,
                {"source_line": 3, "start": 2, "end": 5},
            )
            result = scan_request["row"]["step10_result"]
            self.assertTrue(result["valid"])
            self.assertEqual(result["observed_rows"], 4)
            self.assertEqual(result["zero_current_excluded_count"], 1)
            self.assertEqual(result["zero_current_source_line"], 3)
            self.assertEqual(len(result["voltage_raw"]), 3)
            self.assertEqual(len(result["curve"]), 128)

            ordinary_request = request("9")
            self.assertIsNone(
                _approved_rpt10_zero_current_evidence(
                    ordinary_request, source, eligibility
                )
            )
            _scan_source_requests(ROOT, [ordinary_request])
            self.assertFalse(ordinary_request["row"]["step10_result"]["valid"])
            self.assertEqual(
                ordinary_request["row"]["step10_result"]["status"],
                "unexpected_current_direction",
            )
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def test_converter_script_uses_matlab_table_and_timetable_apis(self) -> None:
        script = MATLAB_CONVERTER_SCRIPT.read_text(encoding="utf-8")
        self.assertIn("load(inputPath)", script)
        self.assertIn("istable(value) || istimetable(value)", script)
        self.assertIn("writetable(measurementTable, outputPath", script)

    @unittest.skipUnless(REAL_RPT_MAT.is_file(), "Guide-mapped MAT fixture is unavailable")
    def test_real_mat_maps_to_guide_verified_rpt13(self) -> None:
        mapping, observations = guide_mapping_for_mat(REAL_RPT_MAT.name)
        inspection = inspect_tpg_mat(REAL_RPT_MAT, REAL_RPT_MAT.name)

        self.assertEqual(mapping["physical_cell_id"], "G1-Cell20")
        self.assertEqual(mapping["dataset_part"], "2")
        self.assertEqual([row["rpt_index"] for row in observations], ["13"])
        self.assertEqual(observations[0]["step4_status"], "verified")
        self.assertEqual(observations[0]["step10_status"], "verified")
        self.assertEqual(inspection["rpt_checkpoints"], ["RPT13"])

    def test_profile_mat_is_not_treated_as_an_rpt(self) -> None:
        with self.assertRaisesRegex(TpgIngestionError, "not an RPT checkpoint"):
            guide_mapping_for_mat("TPG1.20-Cell20.mat")

    def test_real_part1_part2_and_part3_csv_files_have_verified_rpt_mappings(self) -> None:
        expected = {
            "TPG1_4_Cell9.csv": ("1", "G1-Cell9", "1"),
            "TPG1.26-Cell9.csv": ("2", "G1-Cell9", "13"),
            "TPG7.6-Cell4.csv": ("3", "G7-Cell4", "3"),
        }
        for filename, identity in expected.items():
            with self.subTest(filename=filename):
                mapping, observations = guide_mapping_for_file(filename)
                self.assertEqual(
                    (
                        mapping["dataset_part"],
                        mapping["physical_cell_id"],
                        observations[0]["rpt_index"],
                    ),
                    identity,
                )
                self.assertEqual(len(observations), 1)

    def test_mat_csv_equivalence_checks_ordered_measurement_values(self) -> None:
        frame = pd.DataFrame(
            {
                "Step": [4, 4, 10],
                "Amphr": [0.0, 0.1, 0.2],
                "Amps": [-0.5, -0.5, 0.5],
                "Volts": [4.1, 4.0, 4.2],
                "Temp1": [24.0, 24.1, 24.2],
                "StepTime": [0.0, 1.0, 2.0],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            converted = Path(directory) / "converted.csv"
            reference = Path(directory) / "reference.csv"
            frame.to_csv(converted, index=False)
            frame.to_csv(reference, index=False)

            self.assertEqual(validate_mat_csv_equivalence(converted, reference), 3)

            changed = frame.copy()
            changed.loc[1, "Volts"] += 0.01
            changed.to_csv(converted, index=False)
            with self.assertRaisesRegex(TpgIngestionError, "values differ"):
                validate_mat_csv_equivalence(converted, reference)

    def test_raw_csv_equivalence_allows_missing_optional_temperature(self) -> None:
        frame = pd.DataFrame(
            {
                "Step": [4, 10],
                "Amphr": [0.0, 0.1],
                "Amps": [-0.5, 0.5],
                "Volts": [4.1, 4.2],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            uploaded = Path(directory) / "uploaded.csv"
            reference = Path(directory) / "reference.csv"
            frame.to_csv(uploaded, index=False)
            frame.to_csv(reference, index=False)

            self.assertEqual(validate_mat_csv_equivalence(uploaded, reference), 2)

    def test_reference_labels_match_multiple_rpts_by_part_cell_and_rpt(self) -> None:
        labels = pd.read_csv(
            ROOT
            / "data"
            / "processed"
            / "label_soh_part1_2"
            / "run_007_part1_capacity_qc"
            / "labels_soh.csv",
            dtype=str,
        )
        cell_labels = labels[
            labels["physical_cell_id"].eq("G1-Cell9")
            & labels["validation_status"].eq("calculated_from_measured_capacity")
        ].sort_values("rpt_order", key=pd.to_numeric)
        observations = [
            {"rpt_index": str(int(row["rpt_order"]))}
            for _, row in cell_labels.head(3).iterrows()
        ]
        matched = _reference_labels("1", "G1-Cell9", observations)

        self.assertEqual(set(matched), {"1", "2", "3"})
        self.assertTrue(
            all(item["status"] == "calculated_from_measured_capacity" for item in matched.values())
        )
        self.assertTrue(all(item["value_percent"] is not None for item in matched.values()))

    def test_evaluation_metrics_use_only_finite_matched_pairs(self) -> None:
        rows = [
            {
                "predicted_soh_percent": 90.0,
                "reference_soh": {"value_percent": 88.0},
            },
            {
                "predicted_soh_percent": 80.0,
                "reference_soh": {"value_percent": 82.0},
            },
            {
                "predicted_soh_percent": None,
                "reference_soh": {"value_percent": 70.0},
            },
            {
                "predicted_soh_percent": 75.0,
                "reference_soh": {"value_percent": None},
            },
        ]
        metrics = calculate_soh_metrics(rows)

        self.assertEqual(metrics["n"], 2)
        self.assertEqual(metrics["mae_percentage_points"]["value"], 2.0)
        self.assertEqual(metrics["rmse_percentage_points"]["value"], 2.0)
        self.assertAlmostEqual(metrics["r2"]["value"], 5 / 9)

    def test_r2_is_withheld_for_one_pair_or_zero_reference_variance(self) -> None:
        one = calculate_soh_metrics(
            [{"predicted_soh_percent": 90, "reference_soh": {"value_percent": 88}}]
        )
        constant = calculate_soh_metrics(
            [
                {"predicted_soh_percent": 90, "reference_soh": {"value_percent": 88}},
                {"predicted_soh_percent": 80, "reference_soh": {"value_percent": 88}},
            ]
        )

        self.assertIsNone(one["r2"]["value"])
        self.assertIn("at least two", one["r2"]["reason"])
        self.assertIsNone(constant["r2"]["value"])
        self.assertIn("zero variance", constant["r2"]["reason"])

    def test_demo_only_eol_reference_is_excluded_from_evaluation_metrics(self) -> None:
        rows = [
            {
                "predicted_soh_percent": 90.0,
                "reference_soh": {"value_percent": 80.0},
            },
            {
                "predicted_soh_percent": 20.0,
                "inventory_scope": "VERIFIED_DEMO_ONLY",
                "reference_soh": {
                    "value_percent": 100.0,
                    "evaluation_eligible": False,
                },
            },
        ]

        metrics = calculate_soh_metrics(rows)

        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["mae_percentage_points"]["value"], 10.0)

    @unittest.skipUnless(REAL_RPT_MAT.is_file(), "Guide-mapped MAT fixture is unavailable")
    def test_matlab_runtime_absence_fails_explicitly_before_inference(self) -> None:
        with patch(
            "app.tpg_mat_ingestion._matlab_executable",
            side_effect=MatlabRuntimeUnavailable(
                "MATLAB is required to decode TPG MAT files."
            ),
        ):
            with self.assertRaisesRegex(MatlabRuntimeUnavailable, "MATLAB is required"):
                analyze_tpg_mat(REAL_RPT_MAT, REAL_RPT_MAT.name, engine=object())

    @unittest.skipUnless(REAL_RPT_MAT.is_file(), "Guide-mapped MAT fixture is unavailable")
    def test_guide_linked_rpt13_csv_extracts_three_steps_and_runs_frozen_model(self) -> None:
        mapping, observations = guide_mapping_for_mat(REAL_RPT_MAT.name)
        reference_csv = ROOT / Path(mapping["repository_path"]).with_suffix(".csv")

        model_rows, processing_rows = _verified_feature_rows(reference_csv, observations)
        self.assertEqual(len(model_rows), 1)
        self.assertEqual(processing_rows[0]["rpt"], "RPT13")
        self.assertEqual(processing_rows[0]["step8_mapping_status"], "verified")
        self.assertEqual(
            processing_rows[0]["step8_extraction_status"], "verified_raw_segment"
        )
        self.assertEqual(processing_rows[0]["processing_status"], "features_ready")
        self.assertTrue(processing_rows[0]["inputs"]["step4"]["available"])
        self.assertTrue(processing_rows[0]["inputs"]["step10"]["available"])
        self.assertEqual(
            len(processing_rows[0]["inputs"]["step4"]["processed_128_voltage"]), 128
        )
        self.assertEqual(
            set(processing_rows[0]["inputs"]["physics_features"]),
            {
                "mean_discharge_temperature_native",
                "discharge_temperature_rise_native",
                "discharge_plateau_slope_v_per_progress",
                "discharge_slope_variability_v_per_progress",
                "mean_absolute_discharge_current_native",
            },
        )
        self.assertEqual(
            sum(key.startswith("step4_voltage_progress_") for key in model_rows[0]), 128
        )
        self.assertEqual(
            sum(key.startswith("step10_voltage_progress_") for key in model_rows[0]), 128
        )

        prediction = InferenceEngine().predict(model_rows)[0]
        self.assertTrue(pd.notna(prediction["predicted_soh_percent"]))
        self.assertEqual(prediction["model_version"], "bafuse-three-encoder-v1")


if __name__ == "__main__":
    unittest.main()
