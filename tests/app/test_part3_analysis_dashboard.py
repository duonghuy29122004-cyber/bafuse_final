from __future__ import annotations

import json
import csv
import threading
import tempfile
import unittest
from pathlib import Path
from urllib.request import urlopen
from unittest.mock import patch

from app.part3_analysis_data import Part3AnalysisData
from app.part3_analysis_dashboard import HTML_PAGE, create_server
from app.model_runtime import Final2ModelRuntime
from app.part3_review_preprocessing import extract_rpt12_step10


class _FakeService:
    def health(self) -> dict[str, object]:
        return {
            "status": "ok",
            "available_model_count": 1,
            "default_model": "TCN-2M",
        }

    def metadata(self) -> dict[str, object]:
        return {"cell_count": 12, "default_model": "TCN-2M"}


class Part3AnalysisDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = Part3AnalysisData()

    def test_inventory_and_frozen_external_cohort_are_exact(self) -> None:
        metadata = self.data.metadata()

        self.assertEqual(metadata["observation_count"], 96)
        self.assertEqual(metadata["canonical_rpt_count"], 87)
        self.assertEqual(metadata["demo_only_eol_count"], 9)
        self.assertEqual(metadata["cell_count"], 12)
        self.assertEqual(metadata["frozen_external_count"], 74)
        self.assertEqual(len(self.data._frozen_inputs), 74)
        self.assertEqual(set(self.data._frozen_inputs), set(self.data._frozen_references))

    def test_owner_confirmed_g10_mapping_provenance_is_not_rewritten(self) -> None:
        row = self.data._records[("3", "10", "G10-Cell2", "10")]

        self.assertEqual(row["mapping_status"], "GUIDE_VERIFIED")
        self.assertIn("user confirms Cell2", row["guide_mapping_status"])
        self.assertEqual(
            row["identity_provenance"],
            "Group 10 Cell 2 identity confirmed by project owner; "
            "explicit Cell 2 label omitted from printed Guide.",
        )
        self.assertEqual(
            row["canonical_mapping_status"],
            "user_confirmed_identity_by_group_membership",
        )
        self.assertEqual(
            row["canonical_cell_identity_status"],
            "user_confirmed_Cell2_by_group_membership",
        )
        self.assertEqual(row["procedure"], "TPG10.19")
        self.assertEqual(row["status"], "ACCEPTED")

    def test_review_resolution_keeps_data_qc_separate_from_frozen_membership(self) -> None:
        review_rows = [
            row
            for row in self.data._records.values()
            if row.get("data_qc_status") is not None
        ]

        self.assertEqual(len(review_rows), 10)
        self.assertEqual(
            sum(row["data_qc_status"] == "ACCEPTED_DATA_QC" for row in review_rows),
            8,
        )
        self.assertEqual(
            sum(row["data_qc_status"] == "REVIEW_DATA_QC" for row in review_rows),
            2,
        )
        self.assertEqual(
            sum(row["data_qc_status"] == "REJECTED_DATA_QC" for row in review_rows),
            0,
        )
        self.assertTrue(all(row["status"] == "REVIEW" for row in review_rows))
        self.assertTrue(
            all(
                row["evaluation_membership"] == "EXCLUDED_FROZEN_FINAL2"
                and row["evaluation_eligible"] is False
                for row in review_rows
            )
        )
        self.assertEqual(len(self.data._frozen_inputs), 74)
        self.assertFalse(set(row["key"] for row in review_rows) & set(self.data._frozen_inputs))

        rpt2 = self.data._records[("3", "10", "G10-Cell22", "2")]
        rpt11 = self.data._records[("3", "10", "G10-Cell22", "11")]
        rpt12 = self.data._records[("3", "10", "G10-Cell22", "12")]
        self.assertEqual(rpt2["data_qc_status"], "REVIEW_DATA_QC")
        self.assertFalse(rpt2["step4_eligible"])
        self.assertTrue(rpt2["step10_eligible"])
        self.assertFalse(rpt2["two_modality_eligible"])
        self.assertTrue(rpt11["step4_eligible"])
        self.assertFalse(rpt11["step10_eligible"])
        self.assertTrue(rpt12["step4_eligible"])
        self.assertTrue(rpt12["step10_eligible"])
        self.assertTrue(rpt12["p5_eligible"])

    def test_review_runtime_requires_verified_source_and_uses_audit_modalities(self) -> None:
        runtime = Final2ModelRuntime()
        resolution = runtime.review_resolution[("3", "7", "G7-Cell4", "1")]
        row = {
            "part": "3",
            "group": "7",
            "physical_cell_id": "G7-Cell4",
            "rpt_index": "1",
            "source_file": resolution["raw_source_path"],
            "source_sha256": resolution["source_sha256"],
            "source_checksum_verified": True,
        }

        result = runtime._part3_eligibility(row)

        self.assertFalse(result["blocked"])
        self.assertEqual(
            result["modalities"],
            {
                "step4_voltage": True,
                "step10_voltage": True,
                "p5_physics": True,
            },
        )
        row["source_checksum_verified"] = False
        blocked = runtime._part3_eligibility(row)
        self.assertTrue(blocked["blocked"])
        self.assertIn("source_or_mapping_not_verified", blocked["reason"])

    def test_review_step10_repair_resamples_verified_signal_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rpt12.csv"
            columns = ["Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts", "VARx2"]
            rows = [
                ["10", "0", "0", "0", "1", "4.0", ""],
                ["10", "1", "1", "1", "1", "3.9", ""],
                ["", "", "", "", "", "", "1"],
                ["", "", "", "", "", "", ""],
                ["10", "2", "2", "2", "1", "3.8", ""],
                ["10", "3", "3", "3", "1", "3.7", ""],
            ]
            with source.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerows(rows)
            resolution = {
                "part": "3",
                "group": "10",
                "cell": "G10-Cell22",
                "rpt": "12",
                "step10_block_ranges": '[{"start":2,"end":3},{"start":6,"end":7}]',
                "step10_separator_rows": "[4,5]",
            }

            with patch(
                "app.part3_review_preprocessing.verify_review_source",
                return_value="verified-test-hash",
            ):
                result = extract_rpt12_step10(source, resolution)

        self.assertEqual(result["status"], "verified_review_step10_contiguous_blocks_v1")
        self.assertEqual(result["observed_rows"], 4)
        self.assertEqual(result["row_ranges"], [(2, 3), (6, 7)])
        self.assertEqual(len(result["processed_128_voltage"]), 128)
        self.assertEqual(result["capacity_span_native_amphr"], 3.0)

    def test_inference_eligibility_uses_selected_model_modalities(self) -> None:
        rows = [
            {
                "key": ("3", "7", f"G7-Cell{i + 1}", "1"),
                "is_eol": False,
                "evaluation_eligible": True,
                "reference_soh_percent": 90.0,
                "step4_eligible": True,
                "step10_eligible": index != 1,
                "p5_eligible": index == 0,
                "predictions": {
                    "CNN1D-Step4-Baseline": {"predicted_soh_percent": 89.0},
                    "CNN-2M": {"predicted_soh_percent": 89.0},
                    "CNN-3M": {"predicted_soh_percent": 89.0},
                },
            }
            for index, i in enumerate(range(3))
        ]
        inputs = {row["key"]: {} for row in rows}

        baseline = self.data._summary(
            rows, "CNN1D-Step4-Baseline", {"n": 3}, inputs, ["step4_voltage"]
        )
        two_modality = self.data._summary(
            rows,
            "CNN-2M",
            {"n": 3},
            inputs,
            ["step4_voltage", "step10_voltage"],
        )
        three_modality = self.data._summary(
            rows,
            "CNN-3M",
            {"n": 3},
            inputs,
            ["step4_voltage", "step10_voltage", "p5_physics"],
        )

        self.assertEqual(baseline["inference_eligible_count"], 3)
        self.assertEqual(two_modality["inference_eligible_count"], 2)
        self.assertEqual(three_modality["inference_eligible_count"], 1)

    def test_demo_eol_and_review_rows_are_excluded_from_evaluation_metrics(self) -> None:
        rows = [
            {
                "reference_soh_percent": 90.0,
                "predictions": {"TCN-2M": {"predicted_soh_percent": 89.0}},
                "evaluation_eligible": True,
                "is_eol": False,
                "status": "ACCEPTED",
            },
            {
                "reference_soh_percent": 80.0,
                "predictions": {"TCN-2M": {"predicted_soh_percent": 10.0}},
                "evaluation_eligible": False,
                "is_eol": True,
                "status": "DEMO_ONLY",
            },
            {
                "reference_soh_percent": 70.0,
                "predictions": {"TCN-2M": {"predicted_soh_percent": 20.0}},
                "evaluation_eligible": False,
                "is_eol": False,
                "status": "REVIEW",
            },
        ]

        metrics = self.data._exploratory_metrics(rows, "TCN-2M")

        self.assertEqual(metrics["n"], 1)
        self.assertEqual(metrics["mae_pp"], 1.0)

    def test_g10_rpt10_pydma_values_are_diagnostic_only(self) -> None:
        record = self.data._records[("3", "10", "G10-Cell2", "10")]
        saved = {
            "status": "accepted_charge_fit_final2_observation_rejected_diagnostic_only",
            "LLI_percent": -0.5,
            "Anode_LAM_percent": 1.0,
            "Cathode_LAM_percent": 2.0,
            "fit_rmse_mV": 10.0,
            "accepted_runs": 3,
            "total_runs": 5,
            "same_cell_bol_fit_accepted": True,
            "same_cell_bol_status": "same_cell_bol_fit_accepted",
            "same_cell_bol_source": "BoL fit",
        }

        with patch("app.part3_analysis_data.load_shared_charge_degradation", return_value=saved):
            degradation = self.data._degradation(record)

        self.assertFalse(degradation["chart_eligible"])
        self.assertIsNone(degradation["lli_percent"])
        self.assertEqual(degradation["diagnostic_values"]["lli_percent"], -0.5)

    def test_server_serves_standalone_read_only_page_and_json_health(self) -> None:
        server = create_server(port=0, service=_FakeService())
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base_url, timeout=5) as response:
                page = response.read().decode("utf-8")
            with urlopen(f"{base_url}/api/health", timeout=5) as response:
                health = json.loads(response.read())

            self.assertIn("Battery Health Analysis - Part 3", page)
            self.assertIn("External Test results are never used for model selection", page)
            self.assertIn("Original / data QC", page)
            self.assertIn("Frozen evaluation membership", HTML_PAGE)
            self.assertEqual(health["available_model_count"], 1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_server_rejects_non_loopback_bind(self) -> None:
        with self.assertRaisesRegex(ValueError, "loopback only"):
            create_server(host="0.0.0.0", port=0, service=_FakeService())


if __name__ == "__main__":
    unittest.main()
