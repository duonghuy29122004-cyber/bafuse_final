from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from app.final2_streamlit_components import (
    MODEL_OPTIONS,
    DashboardInputError,
    build_prediction_result,
    build_rag_battery_context,
    invalidate_stale_result,
    make_upload_signature,
    resolve_single_rpt,
    resolve_uploaded_rpt,
    resolve_uploaded_rpts,
    validate_csv_upload,
)
from app.final2_streamlit_rag import (
    RagServiceUnavailable,
    ask_rag,
    build_rag_request,
    rag_service_available,
)
from app.model_runtime import Final2ModelRuntime, MODEL_NAMES


def _identity() -> dict[str, str]:
    return {
        "part": "3",
        "group": "7",
        "physical_cell_id": "G7-Cell4",
        "rpt_index": "3",
        "source_file": "data/raw/oxford/part3/Group_7/Group 7/TPG7.6-Cell4.csv",
        "source_sha256": "a" * 64,
        "procedure": "TPG7.6",
        "mapping_status": "GUIDE_VERIFIED",
    }


def _analysis() -> dict:
    identity = _identity()
    return {
        "selected_model": "TCN-2M",
        "provenance": {"dataset_identifier": "frozen"},
        "rpts": [
            {
                "part": identity["part"],
                "group_id": identity["group"],
                "physical_cell_id": identity["physical_cell_id"],
                "rpt_index": identity["rpt_index"],
                "source_file": identity["source_file"],
                "predicted_soh_percent": 91.25,
                "soh_status": "frozen_checkpoint_inference",
                "model_prediction": {
                    "checkpoint_sha256": "b" * 64,
                    "reason": "",
                },
                "reference_soh": {
                    "value_percent": 90.5,
                    "status": "calculated_from_measured_capacity",
                    "source_file": "Group_7/Group 7/TPG7.6-Cell4.csv",
                },
                "inputs": {
                    "step4": {
                        "status": "verified",
                        "processed_128_voltage": [3.0] * 128,
                    },
                    "step10": {
                        "status": "unavailable",
                        "processed_128_voltage": [],
                    },
                },
                "charge_degradation": {
                    "status": "accepted_charge_pydma_diagnostic_not_ground_truth",
                    "source_direction": "charge",
                    "charge_source_file": "TPG7.6-Cell4.csv",
                    "LLI_percent": -0.5,
                    "Anode_LAM_percent": 1.25,
                    "Cathode_LAM_percent": 2.5,
                    "fit_rmse_mV": 8.2,
                    "same_cell_bol_status": "not_recorded_in_h2_v3_artifact",
                },
            }
        ],
    }


class Final2StreamlitDashboardTests(unittest.TestCase):
    def test_explain_navigation_and_session_api_settings(self) -> None:
        root = Path(__file__).resolve().parents[2]
        app = AppTest.from_file(str(root / "app/final2_streamlit_dashboard.py"), default_timeout=40)
        first = build_prediction_result(
            _analysis(), _identity(), "TCN-2M",
            SimpleNamespace(validation_metrics={}, external_metrics={}),
        )
        app.session_state["battery_section"] = "Prediction"
        app.session_state["battery_result"] = first

        def button(label: str):
            return next(item for item in app.button if item.label == label)

        def response(payload: dict):
            return SimpleNamespace(
                ok=True, is_redirect=False, is_permanent_redirect=False, json=lambda: payload,
            )

        vector = [1.0] + [0.0] * 3583

        def post(url: str, **kwargs):
            if url.endswith("/embed"):
                return response({"embedding": vector})
            return response({"choices": [{"message": {"content": "See (according to P01.md)."}}]})

        with (
            patch.dict("os.environ", {"BAFUSE_RAG_MODEL_NAME": "qwen25-battery-lora"}),
            patch("rag.rag_service.requests.get", return_value=response(
                {"data": [{"id": "qwen25-battery-lora"}]}
            )) as get,
            patch("rag.rag_service.requests.post", side_effect=post) as send,
        ):
            app.run()
            button("Explain with RAG").click().run()
            self.assertEqual(app.session_state["main_navigation"], "RAG Assistant")
            self.assertEqual(app.session_state["rag_battery_context"]["predicted_soh"], 91.25)
            self.assertIn("Predicted SoH: 91.250%", [item.value for item in app.text])
            self.assertIn("Label SoH: 90.500%", [item.value for item in app.text])
            app.text_input(key="_rag_generation").set_value("https://vllm-one.trycloudflare.com")
            app.text_input(key="_rag_embedding").set_value("https://embed-one.trycloudflare.com").run()
            button("Test API Connections").click().run()
            get.assert_not_called()
            send.assert_not_called()
            button("Approve generation host vllm-one.trycloudflare.com").click().run()
            button("Test API Connections").click().run()
            self.assertEqual(len(app.success), 1)
            self.assertEqual(len(app.error), 1)
            send.assert_not_called()
            button("Approve embedding host embed-one.trycloudflare.com").click().run()
            button("Test API Connections").click().run()
            self.assertEqual(len(app.success), 2)
            self.assertEqual(get.call_args.args[0], "https://vllm-one.trycloudflare.com/v1/models")
            self.assertEqual(send.call_args.kwargs["json"], {"query": "SOH"})

            app.chat_input[0].set_value("Explain lithium inventory loss").run()
            payload = send.call_args.kwargs["json"]
            self.assertEqual(send.call_args_list[-2].kwargs["json"], {"query": "Explain lithium inventory loss"})
            self.assertEqual(send.call_args.args[0], "https://vllm-one.trycloudflare.com/v1/chat/completions")
            self.assertIn('"predicted_soh": 91.25', payload["messages"][0]["content"])
            self.assertIn('"part": "3"', payload["messages"][0]["content"])
            self.assertIn('"lli_percent": -0.5', payload["messages"][0]["content"])
            self.assertEqual(len(app.session_state["rag_retrieved_chunks"]), 4)
            self.assertTrue(app.session_state["rag_retrieved_chunks"][0]["doc_name"].endswith(".md"))
            app.chat_input[0].set_value("How does that work?").run()
            follow_up = send.call_args.kwargs["json"]["messages"]
            self.assertEqual(follow_up[1]["content"], "Explain lithium inventory loss")
            self.assertEqual(follow_up[-1]["content"], "How does that work?")

            app.text_input(key="_rag_embedding").set_value("https://embed-two.trycloudflare.com/embed").run()
            self.assertNotIn("rag_connection_test", app.session_state)
            self.assertEqual(app.session_state["battery_result"]["predicted_soh_percent"], 91.25)
            button("Approve embedding host embed-two.trycloudflare.com").click().run()
            app.chat_input[0].set_value("Another question").run()
            self.assertEqual(send.call_args_list[-2].args[0], "https://embed-two.trycloudflare.com/embed")

            app.radio(key="main_navigation").set_value("Battery Analysis").run()
            button("Explain with RAG").click().run()
            self.assertEqual(len(app.session_state["rag_messages"]), 6)
            self.assertEqual(app.text_input(key="_rag_embedding").value, "https://embed-two.trycloudflare.com/embed")
            app.radio(key="main_navigation").set_value("Battery Analysis").run()
            app.radio(key="battery_section").set_value("Input").run()
            app.selectbox(key="selected_model").set_value("ResNet-2M").run()
            self.assertNotIn("rag_battery_context", app.session_state)
            self.assertNotIn("rag_messages", app.session_state)
            app.session_state["battery_result"] = {
                **first,
                "identity": {**first["identity"], "physical_cell_id": "G7-Cell19"},
                "predicted_soh_percent": 88.0,
            }
            app.radio(key="battery_section").set_value("Prediction").run()
            button("Explain with RAG").click().run()
            self.assertEqual(app.session_state["rag_battery_context"]["cell"], "Cell19")
            self.assertEqual(app.session_state["rag_messages"], [])
            self.assertFalse(app.exception)

    def test_csv_only_upload_validation(self) -> None:
        self.assertEqual(validate_csv_upload("TPG7.6-Cell4.csv", 10), "TPG7.6-Cell4.csv")
        with self.assertRaisesRegex(DashboardInputError, "CSV"):
            validate_csv_upload("battery.mat", 10)
        with self.assertRaisesRegex(DashboardInputError, "1 GiB"):
            validate_csv_upload("battery.csv", 0)

    def test_rpt_identity_requires_one_exact_guide_source(self) -> None:
        identity = _identity()
        mapping = {
            "source_present": "true",
            "repository_path": identity["source_file"],
            "source_sha256": identity["source_sha256"],
            "dataset_part": "3",
            "group_id": "7",
            "physical_cell_id": "G7-Cell4",
            "procedure": "TPG7.6",
            "mapping_resolution_status": "GUIDE_VERIFIED",
        }
        observation = {
            "part": "3",
            "group": "7",
            "physical_cell_id": "G7-Cell4",
            "rpt_index": "3",
            "source_file": identity["source_file"],
        }
        self.assertEqual(
            resolve_single_rpt("TPG7.6-Cell4.csv", mapping, [observation])["rpt_index"],
            "3",
        )
        with self.assertRaisesRegex(DashboardInputError, "maps to 2"):
            resolve_single_rpt("TPG7.6-Cell4.csv", mapping, [observation, observation])
        selected = dict(observation, rpt_index="4")
        self.assertEqual(
            resolve_single_rpt(
                "TPG7.6-Cell4.csv",
                mapping,
                [observation, selected],
                rpt_index="4",
            )["rpt_index"],
            "4",
        )

    def test_part3_csv_uses_exact_frozen_hash_and_guide_rpt(self) -> None:
        identity = resolve_uploaded_rpt("TPG7.6-Cell4.csv")
        self.assertEqual(identity["physical_cell_id"], "G7-Cell4")
        self.assertEqual(identity["rpt_index"], "3")
        self.assertEqual(
            identity["source_sha256"],
            "dbb18bb421055b2f2be9f582a4873f1fe9211ca8d291c86dabb49dee4a2a0814",
        )

    def test_exact_frozen_final2_label_takes_precedence_after_hash_verification(self) -> None:
        identity = resolve_uploaded_rpt("TPG7.6-Cell4.csv")
        payload = _analysis()
        payload["provenance"].update(
            {
                "source_checksum_verified": True,
                "source_file_sha256": identity["source_sha256"],
            }
        )
        runtime = SimpleNamespace(
            validation_metrics={},
            external_metrics={},
        )

        result = build_prediction_result(
            payload, identity, "TCN-2M", runtime
        )

        self.assertAlmostEqual(result["label_soh_percent"], 91.23074494078035)
        self.assertEqual(
            result["label_details"]["source_sha256"],
            identity["source_sha256"],
        )
        self.assertEqual(
            result["label_details"]["dataset_identifier"],
            "bafuse-final2-p5-g10-guide-verified-v2",
        )

    def test_multi_rpt_source_exposes_each_exact_selection(self) -> None:
        source = _identity()["source_file"]
        mapping = {
            "source_present": "true",
            "repository_path": source,
            "source_sha256": "a" * 64,
            "dataset_part": "3",
            "group_id": "7",
            "physical_cell_id": "G7-Cell4",
            "procedure": "TPG7",
            "mapping_resolution_status": "GUIDE_VERIFIED",
        }
        observations = [
            {
                "part": "3",
                "group": "7",
                "physical_cell_id": "G7-Cell4",
                "rpt_index": rpt,
                "source_file": source,
            }
            for rpt in ("3", "4")
        ]
        with patch(
            "app.final2_streamlit_components.guide_mapping_for_file",
            return_value=(mapping, observations),
        ):
            resolved = resolve_uploaded_rpts("TPG7.6-Cell4.csv")
            selected = resolve_uploaded_rpt("TPG7.6-Cell4.csv", "4")

        self.assertEqual([row["rpt_index"] for row in resolved], ["3", "4"])
        self.assertEqual(selected["rpt_index"], "4")

    def test_prediction_requires_exact_identity_and_source_for_label(self) -> None:
        runtime = SimpleNamespace(
            validation_metrics={"TCN-2M": {"validation_rmse_pp": 0.38}},
            external_metrics={"TCN-2M": {"external_rmse_pp": 1.4}},
        )
        result = build_prediction_result(_analysis(), _identity(), "TCN-2M", runtime)
        self.assertEqual(result["label_soh_percent"], 90.5)
        self.assertEqual(result["mae_percentage_points"], 0.75)
        self.assertEqual(result["rmse_percentage_points"], 0.75)
        self.assertEqual(result["validation_metrics"]["validation_rmse_pp"], 0.38)
        self.assertEqual(result["external_metrics"]["external_rmse_pp"], 1.4)

        payload = _analysis()
        payload["rpts"][0]["reference_soh"]["source_file"] = "TPG7.3-Cell4.csv"
        missing = build_prediction_result(payload, _identity(), "TCN-2M", runtime)
        self.assertIsNone(missing["label_soh_percent"])
        self.assertIsNone(missing["mae_percentage_points"])
        self.assertIsNone(missing["rmse_percentage_points"])

    def test_exact_identity_mismatch_is_rejected(self) -> None:
        runtime = SimpleNamespace(validation_metrics={}, external_metrics={})
        payload = _analysis()
        payload["rpts"][0]["rpt_index"] = "4"
        with self.assertRaisesRegex(DashboardInputError, "identity"):
            build_prediction_result(payload, _identity(), "TCN-2M", runtime)

        payload = _analysis()
        payload["rpts"][0]["physical_cell_id"] = "G7-Cell19"
        with self.assertRaisesRegex(DashboardInputError, "identity"):
            build_prediction_result(payload, _identity(), "TCN-2M", runtime)

    def test_missing_saved_label_is_reported_as_unavailable(self) -> None:
        payload = _analysis()
        payload["rpts"][0]["reference_soh"] = {
            "value_percent": None,
            "status": "missing_reference_soh_label_for_exact_part_cell_rpt",
            "source_file": "",
        }
        result = build_prediction_result(
            payload,
            _identity(),
            "TCN-2M",
            SimpleNamespace(validation_metrics={}, external_metrics={}),
        )
        self.assertIsNone(result["label_soh_percent"])
        self.assertIn("missing_reference", result["label_status"])

    def test_model_selector_and_modality_contract_are_exact(self) -> None:
        self.assertEqual(tuple(MODEL_OPTIONS), tuple(
            name for name in MODEL_NAMES if name not in {"CNN1D-Step4-Baseline", "CNN-2M"}
        ))
        payload = _analysis()
        payload["rpts"][0]["predicted_soh_percent"] = None
        payload["rpts"][0]["soh_status"] = "prediction_unavailable"
        payload["rpts"][0]["model_prediction"]["reason"] = (
            "missing_required_modalities:step10_voltage"
        )
        runtime = SimpleNamespace(validation_metrics={}, external_metrics={})
        result = build_prediction_result(payload, _identity(), "TCN-2M", runtime)
        self.assertIsNone(result["predicted_soh_percent"])
        self.assertIn("step10_voltage", result["prediction_reason"])

    def test_saved_degradation_requires_charge_and_matching_source(self) -> None:
        runtime = SimpleNamespace(validation_metrics={}, external_metrics={})
        payload = _analysis()
        result = build_prediction_result(payload, _identity(), "TCN-2M", runtime)
        self.assertEqual(result["degradation"]["LLI (%)"], -0.5)
        self.assertEqual(result["degradation"]["LAM Anode (%)"], 1.25)
        self.assertTrue(result["degradation_details"]["verified_saved_charge_result"])

        payload["rpts"][0]["charge_degradation"]["charge_source_file"] = "different.csv"
        result = build_prediction_result(payload, _identity(), "TCN-2M", runtime)
        self.assertIsNone(result["degradation"]["LLI (%)"])

    def test_missing_step10_is_n_a_not_a_synthesized_curve(self) -> None:
        runtime = SimpleNamespace(validation_metrics={}, external_metrics={})
        result = build_prediction_result(_analysis(), _identity(), "TCN-2M", runtime)
        self.assertEqual(len(result["step4_curve"]), 128)
        self.assertEqual(result["step10_curve"], [])

    def test_session_result_is_invalidated_on_file_or_model_change(self) -> None:
        old_signature = make_upload_signature("file-1", "rpt.csv", 10, "TCN-2M")
        state = {
            "battery_result": {"model_name": "TCN-2M"},
            "battery_result_signature": old_signature,
            "rag_messages": [{"role": "user", "content": "keep"}],
        }
        self.assertFalse(invalidate_stale_result(state, old_signature))
        new_signature = make_upload_signature("file-2", "rpt.csv", 10, "TCN-2M")
        self.assertTrue(invalidate_stale_result(state, new_signature))
        self.assertNotIn("battery_result", state)
        self.assertNotIn("rag_battery_context", state)
        self.assertEqual(len(state["rag_messages"]), 1)

    def test_frozen_selected_checkpoint_and_saved_metrics_load(self) -> None:
        runtime = Final2ModelRuntime()
        self.assertEqual(runtime.default_model(), "TCN-2M")
        model, entry, _ = runtime._load_model("TCN-2M")
        self.assertIsNotNone(model)
        self.assertEqual(
            entry["best_sha256"],
            "901a2db5305f8209796de2fedbc31e561a45b989087bf24742ec05ec56d6ce5f",
        )
        self.assertIn("validation_rmse_pp", runtime.validation_metrics["TCN-2M"])
        self.assertIn("external_rmse_pp", runtime.external_metrics["TCN-2M"])

    def test_rag_is_disabled_without_local_backend(self) -> None:
        self.assertFalse(rag_service_available({}))
        with self.assertRaisesRegex(RagServiceUnavailable, "RAG service not connected"):
            ask_rag("Explain", None, environ={})
        with self.assertRaisesRegex(RagServiceUnavailable, "localhost"):
            ask_rag(
                "Explain",
                None,
                environ={"BAFUSE_RAG_ENDPOINT": "https://example.com"},
            )

    def test_rag_context_payload_contains_only_available_values(self) -> None:
        result = build_prediction_result(
            _analysis(),
            _identity(),
            "TCN-2M",
            SimpleNamespace(validation_metrics={}, external_metrics={}),
        )
        context = build_rag_battery_context(result)
        self.assertEqual(context["cell"], "Cell4")
        self.assertEqual(context["predicted_soh"], 91.25)
        self.assertEqual(context["saved_label_soh"], 90.5)
        payload = build_rag_request("Explain this result", context)
        self.assertEqual(payload["battery_context"]["predicted_soh"], 91.25)
        self.assertEqual(payload["battery_context"]["lli_percent"], -0.5)
        self.assertEqual(set(payload["battery_context"]), {
            "part", "group", "cell", "rpt", "model", "predicted_soh", "saved_label_soh",
            "lli_percent", "lam_anode_percent", "lam_cathode_percent",
        })

    def test_rag_context_preserves_missing_label_and_pydma_values(self) -> None:
        analysis = _analysis()
        analysis["rpts"][0]["reference_soh"] = {"status": "saved_label_unavailable"}
        analysis["rpts"][0]["charge_degradation"] = {"status": "not_available"}
        result = build_prediction_result(
            analysis,
            _identity(),
            "TCN-2M",
            SimpleNamespace(validation_metrics={}, external_metrics={}),
        )
        context = build_rag_battery_context(result)
        self.assertIsNone(context["saved_label_soh"])
        self.assertIsNone(context["lli_percent"])
        self.assertIsNone(context["lam_anode_percent"])
        self.assertIsNone(context["lam_cathode_percent"])

    def test_frozen_dataset_selection_runs_verified_arrays_through_the_shared_engine(self) -> None:
        from app.final2_streamlit_dashboard import _run_dataset_prediction
        from app.tpg_mat_ingestion import _verified_frozen_observation_rows

        row = next(row for row in _verified_frozen_observation_rows()
                   if row["status"] == "ACCEPTED" and row["physical_cell_id"] == "G10-Cell2")
        engine = SimpleNamespace(
            final2_runtime=SimpleNamespace(validation_metrics={}, external_metrics={}),
        )
        def predict(features, model):
            self.assertEqual(model, "CNN-3M")
            self.assertEqual(len(features[0]["p5_physics_features"]), 5)
            return {"results": [{"predicted_soh_percent": 80.0, "soh_status": "frozen_checkpoint_inference"}]}

        engine.predict_for_model = predict
        result = _run_dataset_prediction(row, "CNN-3M", engine)
        self.assertEqual(result["identity"]["rpt_index"], row["rpt_index"])
        self.assertEqual(len(result["step4_curve"]), 128)
        self.assertEqual(len(result["step10_curve"]), 128)
        self.assertAlmostEqual(result["label_soh_percent"], float(row["soh_percent"]))

    def test_existing_dashboard_files_are_unchanged_from_baseline(self) -> None:
        baseline = Path(
            r"C:\Users\duong\Documents\final2_cleanup_backup_20261117\project"
        )
        root = Path(__file__).resolve().parents[2]
        for relative in (
            "app/inference_dashboard.py",
            "app/part3_analysis_dashboard.py",
            "app/run_inference_demo.ps1",
            "app/run_part3_analysis.ps1",
        ):
            with self.subTest(path=relative):
                self.assertEqual(
                    (root / relative).read_bytes(),
                    (baseline / relative).read_bytes(),
                )


if __name__ == "__main__":
    unittest.main()
