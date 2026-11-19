from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from app.model_runtime import (
    Final2ModelRuntime,
    MODEL_ARCHITECTURES,
    MODEL_NAMES,
    STEP4_BASELINE,
    load_shared_charge_degradation,
)
from app.tpg_mat_ingestion import _analyze_verified_csv
from app.dashboard_page import HTML_PAGE
from app.inference_dashboard import InferenceEngine, LegacyCheckpointUnavailable


ROOT = Path(__file__).resolve().parents[2]


def _feature_row(
    *,
    part: str = "1",
    group: str = "1",
    cell: str = "G1-Cell15",
    rpt: str = "1",
) -> dict[str, object]:
    row: dict[str, object] = {
        "part": part,
        "group": group,
        "physical_cell_id": cell,
        "rpt_index": rpt,
    }
    for index in range(128):
        row[f"step4_voltage_progress_{index:03d}"] = 4.1 - index * 1.6 / 127
        row[f"step10_voltage_progress_{index:03d}"] = 2.9 + index * 1.3 / 127
    row["p5_physics_features"] = [0.6, 0.1, -1.0, 0.1, 0.2]
    return row


class _ConstantModel(nn.Module):
    def forward(self, *inputs: torch.Tensor) -> torch.Tensor:
        return torch.zeros((inputs[0].shape[0], 1), device=inputs[0].device)


class Final2ModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.runtime = Final2ModelRuntime()

    def test_validation_metadata_routes_all_seven_frozen_checkpoints(self) -> None:
        self.assertEqual(self.runtime.default_model(), "TCN-2M")
        catalog = {row["name"]: row for row in self.runtime.model_catalog()}
        self.assertEqual(set(catalog), set(MODEL_NAMES))
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                result = self.runtime.preprocess_rows([_feature_row()], name)["results"][0]
                self.assertEqual(result["model_name"], name)
                self.assertEqual(result["soh_status"], "frozen_checkpoint_inference")
                self.assertTrue(torch.isfinite(torch.tensor(result["predicted_soh_percent"])))
                self.assertTrue(catalog[name]["checkpoint_status"]["available"])
                self.assertTrue(result["checkpoint_sha256"])

    def test_catalog_exposes_saved_model_details_without_external_selection(self) -> None:
        catalog = {row["name"]: row for row in self.runtime.model_catalog()}
        selected_default = self.runtime.default_model()
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                row = catalog[name]
                self.assertEqual(row["architecture"], MODEL_ARCHITECTURES[name])
                self.assertTrue(row["modalities"])
                self.assertTrue(row["checkpoint_id"])
                self.assertTrue(row["checkpoint_path"])
                self.assertEqual(
                    row["dataset_identifier"],
                    "bafuse-final2-p5-g10-guide-verified-v2",
                )
                self.assertIn("validation_rmse_pp", row["validation_metrics"])
                self.assertIn("external_rmse_pp", row["external_metrics"])
        self.runtime.external_metrics = {
            name: {"external_rmse_pp": 0.0} for name in MODEL_NAMES
        }
        self.assertEqual(self.runtime.default_model(), selected_default)

    def test_unavailable_models_are_marked_for_selector_disabling(self) -> None:
        runtime = Final2ModelRuntime()
        with patch.object(
            runtime,
            "_checkpoint_status",
            side_effect=lambda name: (
                {"available": False, "reason": "checkpoint_missing:legacy.pt"}
                if name == "CNN-2M"
                else {"available": True, "reason": "available"}
            ),
        ):
            model = next(row for row in runtime.model_catalog() if row["name"] == "CNN-2M")
        self.assertFalse(model["checkpoint_status"]["available"])
        self.assertEqual(model["checkpoint_status"]["reason"], "checkpoint_missing:legacy.pt")
        self.assertIn("model.checkpoint_status?.reason", HTML_PAGE)
        self.assertIn("option.disabled=!available", HTML_PAGE)

    def test_missing_legacy_checkpoint_does_not_block_final2_catalog(self) -> None:
        engine = InferenceEngine(checkpoint_path=ROOT / "missing-legacy-checkpoint.pt")
        catalog = engine.final2_runtime.model_catalog()
        self.assertEqual({row["name"] for row in catalog}, set(MODEL_NAMES))
        self.assertTrue(any(row["checkpoint_status"]["available"] for row in catalog))
        with self.assertRaisesRegex(
            LegacyCheckpointUnavailable, "Legacy /api/infer.*missing"
        ):
            engine.predict([])

    def test_charge_degradation_is_primary_and_shows_exact_missing_message(self) -> None:
        self.assertIn("Accepted Charge-only PyDMA degradation (primary display)", HTML_PAGE)
        self.assertIn("row.charge_degradation?.[s.chargeKey]", HTML_PAGE)
        self.assertIn(
            "Chưa có kết quả PyDMA Charge hợp lệ cho RPT này.", HTML_PAGE
        )
        self.assertIn(
            "H2 v3 supplemental results and per-RPT fit provenance", HTML_PAGE
        )
        self.assertIn("Charge / same-cell BoL provenance", HTML_PAGE)
        self.assertIn("charge_artifact_parse_failed", HTML_PAGE)
        self.assertIn("charge_artifact_source_mismatch", HTML_PAGE)
        self.assertIn("same_cell_bol_fit_accepted=", HTML_PAGE)
        self.assertIn("same_cell_bol_source=", HTML_PAGE)
        self.assertIn("Charge source=", HTML_PAGE)
        self.assertIn("BoL QC=", HTML_PAGE)

    def test_dashboard_exposes_demo_eol_provenance_fields(self) -> None:
        for field in (
            "rptProvenance",
            "VERIFIED_DEMO_ONLY",
            "mapping_provenance",
            "source_sha256",
            "demo_step10_review_reason",
            "reference_soh?.label_method",
        ):
            with self.subTest(field=field):
                self.assertIn(field, HTML_PAGE)

    def test_cuda_detection_has_cpu_fallback(self) -> None:
        with patch("app.model_runtime.torch.cuda.is_available", return_value=False):
            runtime = Final2ModelRuntime()
        self.assertEqual(str(runtime.device), "cpu")

    def test_tcn_default_requires_current_validation_selection_support(self) -> None:
        runtime = Final2ModelRuntime()
        self.assertEqual(runtime.default_model(), "TCN-2M")
        runtime.validation_selection = {
            **runtime.validation_selection,
            "selected_primary_model": "CNN-3M",
        }
        self.assertEqual(runtime.default_model(), STEP4_BASELINE)

    def test_part3_eligibility_is_reloaded_and_applied_per_model_modality(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            eligibility = Path(directory) / "part3_modality_eligibility.csv"
            columns = [
                "dataset_part",
                "group",
                "physical_cell_id",
                "rpt_index",
                "step4_inference_eligible",
                "step10_inference_eligible",
                "physics_inference_eligible",
            ]

            def write_row(physics_available: str) -> None:
                with eligibility.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    writer.writerow(
                        {
                            "dataset_part": "3",
                            "group": "10",
                            "physical_cell_id": "G10-Cell22",
                            "rpt_index": "1",
                            "step4_inference_eligible": "true",
                            "step10_inference_eligible": "true",
                            "physics_inference_eligible": physics_available,
                        }
                    )

            write_row("false")
            runtime = Final2ModelRuntime(eligibility_path=eligibility)
            entry = {"best_sha256": "test", "seed": 123, "best_epoch": 1}
            with patch.object(
                runtime,
                "_load_model",
                return_value=(_ConstantModel(), entry, "frozen.pt"),
            ):
                tcn = runtime.preprocess_rows(
                    [_feature_row(part="3", group="10", cell="G10-Cell22")],
                    "TCN-2M",
                )["results"][0]
                cnn3m = runtime.preprocess_rows(
                    [_feature_row(part="3", group="10", cell="G10-Cell22")],
                    "CNN-3M",
                )["results"][0]
                self.assertIsNotNone(tcn["predicted_soh_percent"])
                self.assertIsNone(cnn3m["predicted_soh_percent"])
                self.assertIn("p5_physics", cnn3m["reason"])
                self.assertIn("not_eligible", cnn3m["reason"])

                write_row("true")
                accepted = runtime.preprocess_rows(
                    [_feature_row(part="3", group="10", cell="G10-Cell22")],
                    "CNN-3M",
                )["results"][0]
                self.assertIsNotNone(accepted["predicted_soh_percent"])

    def test_part3_missing_manifest_and_exact_row_withhold_predictions(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            missing_path = Path(directory) / "not-created.csv"
            runtime = Final2ModelRuntime(eligibility_path=missing_path)
            result = runtime.preprocess_rows(
                [_feature_row(part="3", group="10", cell="G10-Cell22")],
                "TCN-2M",
            )["results"][0]
            self.assertIsNone(result["predicted_soh_percent"])
            self.assertIn("eligibility_manifest_missing", result["reason"])

    def test_verified_demo_only_eol_uses_raw_modality_validation_not_frozen_audit_row(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            missing_path = Path(directory) / "not-created.csv"
            runtime = Final2ModelRuntime(eligibility_path=missing_path)
            row = _feature_row(part="3", group="7", cell="G7-Cell4", rpt="7")
            row.update(
                {
                    "inventory_scope": "VERIFIED_DEMO_ONLY",
                    "source_checksum_verified": True,
                    "demo_step4_valid": True,
                    "demo_step10_valid": False,
                    "demo_p5_valid": True,
                }
            )
            entry = {"best_sha256": "test", "seed": 123, "best_epoch": 1}
            with patch.object(
                runtime,
                "_load_model",
                return_value=(_ConstantModel(), entry, "frozen.pt"),
            ):
                baseline = runtime.preprocess_rows(
                    [row], STEP4_BASELINE
                )["results"][0]
                two_modality = runtime.preprocess_rows(
                    [row], "TCN-2M"
                )["results"][0]

        self.assertIsNotNone(baseline["predicted_soh_percent"])
        self.assertEqual(baseline["eligibility_status"], "verified_demo_only")
        self.assertFalse(baseline["modality_eligibility"]["step10_voltage"])
        self.assertIsNone(two_modality["predicted_soh_percent"])
        self.assertIn(
            "part3_modality_not_eligible:step10_voltage",
            two_modality["reason"],
        )

    def test_missing_modality_and_checkpoint_return_explicit_reasons(self) -> None:
        runtime = Final2ModelRuntime()
        entry = {"best_sha256": "test", "seed": 123, "best_epoch": 1}
        row = _feature_row()
        del row["step10_voltage_progress_127"]
        with patch.object(
            runtime,
            "_load_model",
            return_value=(_ConstantModel(), entry, "frozen.pt"),
        ):
            result = runtime.preprocess_rows([row], "TCN-2M")["results"][0]
        self.assertIsNone(result["predicted_soh_percent"])
        self.assertIn("missing_required_modalities:step10_voltage", result["reason"])

        with (
            patch.object(
                runtime,
                "_checkpoint_status",
                return_value={"available": False, "reason": "checkpoint_missing:test.pt"},
            ),
            patch.object(runtime, "_load_model", wraps=runtime._load_model) as load_model,
        ):
            result = runtime.preprocess_rows([_feature_row()], "TCN-2M")["results"][0]
        self.assertIsNone(result["predicted_soh_percent"])
        self.assertIn("checkpoint_missing", result["reason"])
        load_model.assert_called_once()

    def test_shared_charge_pydma_joins_exact_identity_and_gate(self) -> None:
        result = load_shared_charge_degradation("1", "1", "G1-Cell15", "RPT1")
        self.assertEqual(
            result["status"], "accepted_charge_pydma_diagnostic_not_ground_truth"
        )
        self.assertEqual(result["source_direction"], "charge")
        self.assertLess(result["fit_rmse_mV"], 15)
        self.assertGreaterEqual(result["accepted_runs"], 3)

        missing = load_shared_charge_degradation("1", "1", "G1-Cell15", "RPT999")
        self.assertIsNone(missing["value_percent"])
        self.assertIn("no_existing_charge_fit", missing["reason"])

        g10_rpt10 = load_shared_charge_degradation(
            "3", "10", "G10-Cell2", "RPT10"
        )
        self.assertEqual(
            g10_rpt10["status"],
            "accepted_charge_fit_final2_observation_rejected_diagnostic_only",
        )
        self.assertLess(g10_rpt10["fit_rmse_mV"], 15)
        self.assertEqual(
            g10_rpt10["same_cell_bol_source"],
            "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
        )
        self.assertIs(g10_rpt10["same_cell_bol_fit_accepted"], True)
        self.assertEqual(g10_rpt10["same_cell_bol_status"], "same_cell_bol_fit_accepted")
        self.assertEqual(g10_rpt10["source_name"], "phase_c2_charge")
        g10_rpt1 = load_shared_charge_degradation(
            "3", "10", "G10-Cell2", "RPT1"
        )
        self.assertEqual(g10_rpt1["source_name"], "phase_c2_charge")
        self.assertEqual(
            g10_rpt1["same_cell_bol_source"],
            "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
        )

        part3_run009 = load_shared_charge_degradation(
            "3", "10", "G10-Cell22", "RPT1"
        )
        self.assertEqual(part3_run009["source_name"], "part3_run009_charge")
        self.assertEqual(part3_run009["source_file"], "TPG10.3-Cell22.csv")
        self.assertEqual(
            part3_run009["same_cell_bol_source"], "TPG10.0-Cell22.csv"
        )
        self.assertIs(part3_run009["fit_accepted_by_pydma_15mV"], True)
        self.assertIs(part3_run009["same_cell_bol_fit_accepted"], True)
        self.assertEqual(
            part3_run009["same_cell_bol_quality_control_status"],
            "reference_baseline",
        )
        self.assertLess(part3_run009["same_cell_bol_fit_rmse_mV"], 15)
        self.assertGreaterEqual(part3_run009["same_cell_bol_accepted_runs"], 3)

    def test_part3_run009_joins_exact_cell_bol_and_withholds_missing_rpt(self) -> None:
        fields = [
            "group_id", "cell_id", "checkup", "direction", "source_file",
            "fit_rmse_mV", "fit_accepted_by_pydma_15mV", "fit_threshold_mV",
            "accepted_runs", "total_runs", "diagnostic_label_status",
            "LLI_percent_vs_BoL", "LAM_anode_percent_vs_BoL",
            "LAM_cathode_percent_vs_BoL",
        ]

        def row(cell: str, checkpoint: str, source: str) -> dict[str, object]:
            is_bol = checkpoint == "BoL"
            return {
                "group_id": "Group 7",
                "cell_id": cell,
                "checkup": checkpoint,
                "direction": "charge",
                "source_file": source,
                "fit_rmse_mV": "8.0",
                "fit_accepted_by_pydma_15mV": "true",
                "fit_threshold_mV": "15",
                "accepted_runs": "3",
                "total_runs": "3",
                "diagnostic_label_status": "reference_baseline" if is_bol else "diagnostic_estimate_accepted",
                "LLI_percent_vs_BoL": "0" if is_bol else "2.5",
                "LAM_anode_percent_vs_BoL": "0" if is_bol else "1.5",
                "LAM_cathode_percent_vs_BoL": "0" if is_bol else "1.0",
            }

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "pydma_results.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        row("Cell 2", "BoL", "TPG7.0-Cell2.csv"),
                        row("Cell 22", "BoL", "TPG7.0-Cell22.csv"),
                        row("Cell 22", "RPT1", "TPG7.3-Cell22.csv"),
                        row("Cell 2", "RPT1", "TPG7.3-Cell2.csv"),
                    ]
                )
            result = load_shared_charge_degradation(
                "3", "7", "G7-Cell2", "RPT1", part3_path=path
            )
            self.assertEqual(result["source_file"], "TPG7.3-Cell2.csv")
            self.assertEqual(result["same_cell_bol_source"], "TPG7.0-Cell2.csv")

            missing = load_shared_charge_degradation(
                "3", "7", "G7-Cell2", "RPT2", part3_path=path
            )
            self.assertIsNone(missing["value_percent"])
            self.assertEqual(
                missing["status"],
                "accepted_charge_pydma_unavailable_for_exact_part_group_cell_rpt",
            )

            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row("Cell 2", "RPT1", "TPG7.3-Cell2.csv"))
            missing_bol = load_shared_charge_degradation(
                "3", "7", "G7-Cell2", "RPT1", part3_path=path
            )
            self.assertIsNone(missing_bol["value_percent"])
            self.assertEqual(missing_bol["status"], "charge_artifact_source_mismatch")
            self.assertIn("bol_missing_or_ambiguous", missing_bol["reason"])

            rejected_bol = row("Cell 2", "BoL", "TPG7.0-Cell2.csv")
            rejected_bol["fit_accepted_by_pydma_15mV"] = "false"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        rejected_bol,
                        row("Cell 2", "RPT1", "TPG7.3-Cell2.csv"),
                    ]
                )
            rejected = load_shared_charge_degradation(
                "3", "7", "G7-Cell2", "RPT1", part3_path=path
            )
            self.assertIsNone(rejected["value_percent"])
            self.assertIn("same_cell_bol_did_not_pass", rejected["reason"])

            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(
                    [
                        row("Cell 2", "BoL", "TPG7.0-Cell2.csv"),
                        row("Cell 2", "RPT1", "TPG7.3-Cell2.csv"),
                        row("Cell 2", "RPT1", "TPG7.4-Cell2.csv"),
                    ]
                )
            duplicate = load_shared_charge_degradation(
                "3", "7", "G7-Cell2", "RPT1", part3_path=path
            )
            self.assertIsNone(duplicate["value_percent"])
            self.assertIn("duplicate_exact_charge_rows", duplicate["reason"])

    def test_charge_artifact_preserves_bol_provenance_and_reports_failures(self) -> None:
        h2_fields = [
            "part", "group", "physical_cell_id", "rpt_index", "source_file",
            "charge_source_file", "charge_accepted", "charge_rmse_mV",
            "charge_accepted_runs", "charge_LLI_percent",
            "charge_Anode_LAM_percent", "charge_Cathode_LAM_percent",
        ]
        phase_fields = [
            "physical_cell_id", "rpt_index", "source_file", "fit_direction",
            "step", "fit_rmse_mV", "lli_percent", "lam_anode_percent",
            "lam_cathode_percent", "accepted_runs", "quality_control_status",
            "same_cell_bol_source", "same_cell_bol_fit_accepted",
        ]

        def write_csv(path: Path, fields: list[str], row: dict[str, object]) -> None:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerow(row)

        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            h2_path, phase_path = folder / "h2.csv", folder / "phase.csv"
            write_csv(
                h2_path,
                h2_fields,
                {
                    "part": 3, "group": 10, "physical_cell_id": "G10-Cell2",
                    "rpt_index": "RPT10",
                    "source_file": "data/raw/oxford/part3/Group_10/Group 10/TPG10.19-Cell2.csv",
                    "charge_accepted": "false",
                },
            )
            write_csv(
                phase_path,
                phase_fields,
                {
                    "physical_cell_id": "G10-Cell2", "rpt_index": "RPT10",
                    "source_file": "data/raw/oxford/part3/Group_10/Group 10/TPG10.19-Cell2.csv",
                    "fit_direction": "charge", "step": 10, "fit_rmse_mV": 11.0,
                    "lli_percent": 1.0, "lam_anode_percent": 2.0,
                    "lam_cathode_percent": 3.0, "accepted_runs": 3,
                    "quality_control_status": "diagnostic_only_final2_rejected",
                    "same_cell_bol_source": "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
                    "same_cell_bol_fit_accepted": "true",
                },
            )
            phase_result = load_shared_charge_degradation(
                "3", "10", "G10-Cell2", "RPT10",
                h2_path=h2_path, g10_path=phase_path,
            )
            self.assertEqual(phase_result["source_name"], "phase_c2_charge")
            self.assertEqual(
                phase_result["same_cell_bol_source"],
                "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
            )
            self.assertIs(phase_result["same_cell_bol_fit_accepted"], True)
            wrong_phase_source = {
                "physical_cell_id": "G10-Cell2", "rpt_index": "RPT10",
                "source_file": "different.csv", "fit_direction": "charge",
                "step": 10, "fit_rmse_mV": 11.0, "lli_percent": 1.0,
                "lam_anode_percent": 2.0, "lam_cathode_percent": 3.0,
                "accepted_runs": 3,
                "quality_control_status": "diagnostic_only_final2_rejected",
                "same_cell_bol_source": "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
                "same_cell_bol_fit_accepted": "true",
            }
            write_csv(phase_path, phase_fields, wrong_phase_source)
            phase_mismatch = load_shared_charge_degradation(
                "3", "10", "G10-Cell2", "RPT10",
                h2_path=h2_path, g10_path=phase_path,
            )
            self.assertEqual(
                phase_mismatch["status"], "charge_artifact_source_mismatch"
            )
            self.assertEqual(
                phase_mismatch["source_errors"][0]["reason"],
                "phase_c2_source_file_does_not_match_rpt10_source",
            )

            write_csv(
                h2_path,
                h2_fields,
                {
                    "part": 1, "group": 1, "physical_cell_id": "G1-Cell15",
                    "rpt_index": "RPT1", "source_file": "TPG1.1.csv",
                    "charge_source_file": "TPG1.1.csv", "charge_accepted": "true",
                    "charge_rmse_mV": 10, "charge_accepted_runs": 3,
                    "charge_LLI_percent": 1, "charge_Anode_LAM_percent": 2,
                    "charge_Cathode_LAM_percent": 3,
                },
            )
            h2_result = load_shared_charge_degradation(
                "1", "1", "G1-Cell15", "RPT1",
                h2_path=h2_path,
                g10_path=phase_path,
            )
            self.assertEqual(h2_result["same_cell_bol_status"], "not_recorded_in_h2_v3_artifact")
            self.assertIsNone(h2_result["same_cell_bol_source"])
            self.assertIn("not asserted", h2_result["same_cell_bol_note"])

            write_csv(
                h2_path,
                h2_fields,
                {
                    "part": 1, "group": 1, "physical_cell_id": "G1-Cell15",
                    "rpt_index": "RPT1", "source_file": "TPG1.1.csv",
                    "charge_source_file": "different.csv", "charge_accepted": "true",
                    "charge_rmse_mV": 10, "charge_accepted_runs": 3,
                    "charge_LLI_percent": 1, "charge_Anode_LAM_percent": 2,
                    "charge_Cathode_LAM_percent": 3,
                },
            )
            mismatch = load_shared_charge_degradation(
                "1", "1", "G1-Cell15", "RPT1", h2_path=h2_path
            )
            self.assertEqual(mismatch["status"], "charge_artifact_source_mismatch")
            self.assertTrue(mismatch["source_errors"])

            h2_path.write_text("part,group\n1,1\n", encoding="utf-8")
            parse_failure = load_shared_charge_degradation(
                "1", "1", "G1-Cell15", "RPT1", h2_path=h2_path
            )
            self.assertEqual(parse_failure["status"], "charge_artifact_parse_failed")
            self.assertTrue(parse_failure["artifact_errors"])

    def test_switching_model_preserves_reference_and_degradation_joins(self) -> None:
        model_row = _feature_row()
        processing = {
            "part": "1",
            "group": "1",
            "physical_cell_id": "G1-Cell15",
            "rpt_index": "1",
            "rpt": "RPT1",
            "processing_status": "features_ready",
            "inputs": {
                "step4": {},
                "step8": {},
                "step10": {},
                "physics_features": {},
                "p5_physics_features": {},
            },
        }
        reference = {
            "value_percent": 96.5,
            "status": "calculated_from_measured_capacity",
            "label_scope": "part1",
            "source_file": "source.csv",
            "label_method": "Step4 capacity ratio",
            "capacity_unit": "native ratio",
        }
        saved = {
            ("1", "LLI"): {"value_percent": 2.0, "status": "saved"},
            ("1", "Anode_LAM"): {"value_percent": 3.0, "status": "saved"},
            ("1", "Cathode_LAM"): {"value_percent": 4.0, "status": "saved"},
        }
        charge = {"LLI_percent": 1.0, "status": "accepted", "source_direction": "charge"}

        class FakeEngine:
            final2_runtime = SimpleNamespace(
                validation_metrics={"TCN-2M": {}, "CNN-2M": {}}
            )

            def predict_for_model(self, rows, model_name):
                return {
                    "model_name": model_name,
                    "results": [
                        {
                            "part": row["part"],
                            "physical_cell_id": row["physical_cell_id"],
                            "rpt_index": row["rpt_index"],
                            "predicted_soh_percent": 90.0 if model_name == "TCN-2M" else 91.0,
                            "soh_status": "frozen_checkpoint_inference",
                            "model_name": model_name,
                        }
                        for row in rows
                    ],
                }

        mapping = {
            "dataset_part": "1",
            "group_id": "1",
            "physical_cell_id": "G1-Cell15",
            "procedure": "TPG1.4",
            "guide_stage": "RPT1",
            "guide_rpt_refs": "RPT1",
        }
        outputs = []
        with (
            patch(
                "app.tpg_mat_ingestion._verified_feature_rows",
                return_value=([model_row], [processing]),
            ),
            patch("app.tpg_mat_ingestion._saved_degradation", return_value=saved),
            patch(
                "app.tpg_mat_ingestion._reference_labels",
                return_value={"1": reference},
            ),
            patch(
                "app.model_runtime.load_shared_charge_degradation",
                return_value=charge,
            ),
        ):
            for model_name in ("TCN-2M", "CNN-2M"):
                outputs.append(
                    _analyze_verified_csv(
                        Path("source.csv"),
                        "source.csv",
                        mapping,
                        [{"rpt_index": "1"}],
                        10,
                        FakeEngine(),
                        model_name=model_name,
                    )["rpts"][0]
                )
        self.assertNotEqual(
            outputs[0]["predicted_soh_percent"], outputs[1]["predicted_soh_percent"]
        )
        for field in ("reference_soh", "degradation", "charge_degradation"):
            self.assertEqual(outputs[0][field], outputs[1][field])


if __name__ == "__main__":
    unittest.main()
