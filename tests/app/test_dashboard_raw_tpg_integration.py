from __future__ import annotations

import http.client
import json
import os
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

from app.inference_dashboard import DashboardHandler, InferenceEngine
from app.tpg_mat_ingestion import ROOT, guide_mapping_for_file


REAL_TPG_FILES = (
    "TPG1_4_Cell9.csv",
    "TPG1.26-Cell9.csv",
    "TPG7.6-Cell4.csv",
)
PART3_MISSING_TEMPERATURE_FILE = "TPG7.3-Cell4.csv"


@unittest.skipUnless(
    os.environ.get("BAFUSE_RUN_REAL_TPG_UPLOAD_TESTS") == "1",
    "Set BAFUSE_RUN_REAL_TPG_UPLOAD_TESTS=1 to stream real large TPG files.",
)
class RawTpgUploadIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        handler = type(
            "BoundDashboardHandler",
            (DashboardHandler,),
            {"engine": InferenceEngine()},
        )
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever,
            daemon=True,
        )
        cls.server_thread.start()
        cls.host, cls.port = cls.server.server_address

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.server_thread.join(timeout=10)

    def _post_inspection(self, filename: str) -> dict:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=120)
        try:
            body = json.dumps({"filename": filename})
            connection.request(
                "POST",
                "/api/inspect",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200, payload)
            return payload
        finally:
            connection.close()

    def _post_file(self, path: Path) -> dict:
        connection = http.client.HTTPConnection(self.host, self.port, timeout=1800)
        try:
            connection.putrequest("POST", "/api/analyze")
            connection.putheader("Content-Type", "application/octet-stream")
            connection.putheader("X-Upload-Filename", path.name)
            connection.putheader("Content-Length", str(path.stat().st_size))
            connection.endheaders()
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    connection.send(block)
            response = connection.getresponse()
            payload = json.loads(response.read())
            self.assertEqual(response.status, 200, payload)
            return payload
        finally:
            connection.close()

    def test_part1_part2_and_part3_raw_csv_uploads(self) -> None:
        results = []
        for filename in REAL_TPG_FILES:
            with self.subTest(filename=filename):
                mapping, observations = guide_mapping_for_file(filename)
                source_path = ROOT / Path(mapping["repository_path"])
                self.assertTrue(source_path.is_file(), source_path)
                inspection = self._post_inspection(filename)
                result = self._post_file(source_path)
                self.assertEqual(result["part"], mapping["dataset_part"])
                self.assertEqual(result["cell_id"], mapping["physical_cell_id"])
                self.assertEqual(
                    [row["rpt_index"] for row in result["rpts"]],
                    [row["rpt_index"] for row in observations],
                )
                self.assertGreater(result["source_row_count_verified"], 0)
                self.assertEqual(
                    inspection["detected_rpt_count"],
                    result["detected_rpt_count"],
                )
                self.assertEqual(result["successful_prediction_count"], 1)
                self.assertEqual(result["matched_evaluation_count"], 1)
                self.assertEqual(
                    result["evaluation_metrics"]["n"],
                    result["matched_evaluation_count"],
                )
                output = result["rpts"][0]
                self.assertEqual(
                    output["reference_soh"]["status"],
                    "calculated_from_measured_capacity",
                )
                self.assertAlmostEqual(
                    output["signed_error_percentage_points"],
                    output["predicted_soh_percent"]
                    - output["reference_soh"]["value_percent"],
                )
                self.assertEqual(len(output["inputs"]["step4"]["processed_128_voltage"]), 128)
                self.assertEqual(len(output["inputs"]["step10"]["processed_128_voltage"]), 128)
                self.assertEqual(len(output["inputs"]["model_normalized"]["step4_voltage_normalized"]), 128)
                available_degradation = sum(
                    metric["value_percent"] is not None
                    for row in result["rpts"]
                    for metric in row["degradation"].values()
                )
                results.append(
                    {
                        "file": filename,
                        "part": result["part"],
                        "cell": result["cell_id"],
                        "verified_rpts": result["detected_rpt_count"],
                        "soh_predictions": result["successful_prediction_count"],
                        "matched_references": result["matched_evaluation_count"],
                        "evaluation_metrics": result["evaluation_metrics"],
                        "available_degradation_estimates": available_degradation,
                        "processing_status": [
                            row["processing_status"] for row in result["rpts"]
                        ],
                    }
                )
        print("REAL_TPG_UPLOAD_RESULTS=" + json.dumps(results, sort_keys=True))

    def test_part3_missing_temperature_keeps_rpt_but_withholds_soh(self) -> None:
        mapping, observations = guide_mapping_for_file(
            PART3_MISSING_TEMPERATURE_FILE
        )
        source_path = ROOT / Path(mapping["repository_path"])
        self.assertTrue(source_path.is_file(), source_path)
        result = self._post_file(source_path)
        self.assertEqual(result["part"], "3")
        self.assertEqual(
            [row["rpt_index"] for row in result["rpts"]],
            [row["rpt_index"] for row in observations],
        )
        self.assertEqual(result["successful_prediction_count"], 0)
        self.assertEqual(
            result["rpts"][0]["processing_status"],
            "missing_or_nonfinite_step4_temperature",
        )
        self.assertEqual(
            result["rpts"][0]["inputs"]["step4"]["temperature_status"],
            "missing_or_nonfinite_step4_temperature",
        )
        self.assertEqual(
            result["rpts"][0]["inputs"]["physics_features"][
                "mean_discharge_temperature_native"
            ]["status"],
            "missing_step4_temperature",
        )
        self.assertEqual(
            result["rpts"][0]["reference_soh"]["status"],
            "calculated_from_measured_capacity",
        )
        self.assertEqual(result["matched_evaluation_count"], 0)
        self.assertIsNone(result["rpts"][0]["signed_error_percentage_points"])
        print(
            "PART3_MISSING_TEMPERATURE_RESULT="
            + json.dumps(
                {
                    "file": PART3_MISSING_TEMPERATURE_FILE,
                    "cell": result["cell_id"],
                    "verified_rpts": result["detected_rpt_count"],
                    "soh_predictions": result["successful_prediction_count"],
                    "processing_status": result["rpts"][0]["processing_status"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
