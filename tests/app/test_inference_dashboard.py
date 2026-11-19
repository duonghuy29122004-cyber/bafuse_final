from __future__ import annotations

import csv
import io
import json
import threading
import unittest
from types import SimpleNamespace
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer
from unittest.mock import patch

from app.inference_dashboard import (
    DashboardHandler,
    InferenceEngine,
    MODEL_NAMES,
    MODEL_INPUT_COLUMNS,
    UPLOAD_COLUMNS,
    UploadValidationError,
    build_sample_upload,
    lookup_h2_v3,
    lookup_h2_v3_trajectory,
    load_h2_v4_candidate_audit,
    parse_upload_csv,
)
from app.upload_pydma import UploadedCurveError, run_uploaded_pydma


class InferenceDashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sample = build_sample_upload()
        cls.rows = parse_upload_csv(cls.sample)

    def test_prepared_sample_uses_frozen_checkpoint_and_has_no_target(self) -> None:
        engine = InferenceEngine()
        result = engine.predict(self.rows)[0]

        self.assertEqual(result["model_version"], "bafuse-three-encoder-v1")
        self.assertTrue(engine.checkpoint_sha256)
        self.assertTrue(0 <= result["predicted_soh_percent"] <= 100)
        self.assertNotIn("soh_percent", result)

    def test_missing_legacy_checkpoint_keeps_catalog_up_and_errors_legacy_endpoint(self) -> None:
        engine = InferenceEngine(checkpoint_path=Path("missing-legacy-checkpoint.pt"))
        handler = type(
            "TestDashboardHandler",
            (DashboardHandler,),
            {"engine": engine},
        )
        server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base_url = f"http://127.0.0.1:{server.server_port}"
            with urlopen(f"{base_url}/api/models", timeout=10) as response:
                catalog = json.loads(response.read())
            self.assertEqual(
                {row["name"] for row in catalog["models"]}, set(MODEL_NAMES)
            )
            request = Request(
                f"{base_url}/api/infer",
                data=self.sample.encode("utf-8"),
                headers={"X-Upload-Filename": "prepared.csv"},
                method="POST",
            )
            with self.assertRaises(HTTPError) as raised:
                urlopen(request, timeout=10)
            self.assertEqual(raised.exception.code, 503)
            error = json.loads(raised.exception.read())
            self.assertIn("Legacy /api/infer", error["error"])
            self.assertIn("missing", error["error"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_identity_metadata_is_not_a_predictive_input(self) -> None:
        engine = InferenceEngine()
        original = engine.predict(self.rows)[0]["predicted_soh_percent"]
        changed = [dict(self.rows[0])]
        changed[0]["part"] = "2"
        changed[0]["group"] = "99"
        changed[0]["physical_cell_id"] = "not-a-model-feature"
        changed[0]["rpt_index"] = "999"

        prediction = engine.predict(changed)[0]["predicted_soh_percent"]
        self.assertEqual(original, prediction)

    def test_parser_rejects_target_labels(self) -> None:
        lines = self.sample.splitlines()
        lines[0] += ",soh_percent"
        lines[1] += ",99"
        csv_with_label = "\n".join(lines) + "\n"
        with self.assertRaisesRegex(UploadValidationError, "target/label"):
            parse_upload_csv(csv_with_label)

    def test_parser_rejects_missing_modalities(self) -> None:
        lines = self.sample.splitlines()
        headers = lines[0].split(",")
        index = headers.index("step10_voltage_progress_127")
        headers.pop(index)
        values = lines[1].split(",")
        values.pop(index)
        malformed = ",".join(headers) + "\n" + ",".join(values) + "\n"

        with self.assertRaisesRegex(UploadValidationError, "missing required columns"):
            parse_upload_csv(malformed)

    def test_saved_h2_v3_lookup_returns_all_metrics(self) -> None:
        rows = lookup_h2_v3(
            {
                "part": "1",
                "group": "1",
                "physical_cell_id": "G1-Cell15",
                "checkpoint": "RPT1",
            }
        )

        self.assertEqual({row["metric"] for row in rows}, {"LLI", "Anode_LAM", "Cathode_LAM"})
        self.assertTrue(all(row["h2_v3_selected_accepted"] for row in rows))

    def test_trajectory_excludes_eol_until_sensitivity_is_requested(self) -> None:
        identity = {
            "part": "2",
            "group": "1",
            "physical_cell_id": "G1-Cell15",
            "metric": "LLI",
        }
        regular = lookup_h2_v3_trajectory(identity)
        including_eol = lookup_h2_v3_trajectory({**identity, "include_eol": True})

        self.assertTrue(regular)
        self.assertTrue(all(not row["is_eol"] for row in regular))
        self.assertTrue(any(row["is_eol"] for row in including_eol))

    def test_h2_v4_audit_remains_separate_from_dashboard_default(self) -> None:
        audit = load_h2_v4_candidate_audit()

        self.assertEqual(audit["candidate_status"], "experimental_not_dashboard_default")
        self.assertEqual(len(audit["policies"]), 6)
        self.assertEqual(audit["counts"]["newly_selected_estimates"], 27)

    def test_sample_columns_match_the_model_input_contract(self) -> None:
        self.assertEqual(set(self.rows[0]), set(UPLOAD_COLUMNS))
        self.assertEqual(len(MODEL_INPUT_COLUMNS), 261)

    @staticmethod
    def _pydma_bundle(part: str = "1", group: str = "1") -> str:
        columns = [
            "role",
            "part",
            "group",
            "physical_cell_id",
            "checkpoint_id",
            "source_file",
            "direction",
            "Step",
            "Amphr",
            "Amps",
            "Volts",
        ]
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for role, checkpoint, source, offset in (
            ("bol", "BoL", "TPG1.1.csv", 0.0),
            ("target", "RPT2", "TPG1.1.csv", 0.01),
        ):
            for index in range(20):
                writer.writerow(
                    {
                        "role": role,
                        "part": part,
                        "group": group,
                        "physical_cell_id": "G1-Cell15",
                        "checkpoint_id": checkpoint,
                        "source_file": source,
                        "direction": "discharge",
                        "Step": 8,
                        "Amphr": offset + index * 0.01,
                        "Amps": -0.1,
                        "Volts": 4.2 - index * 0.05,
                    }
                )
        return stream.getvalue()

    def test_uploaded_pydma_preserves_signed_estimates_and_provenance(self) -> None:
        accepted = SimpleNamespace(
            rmse=0.010,
            n_accepted_runs=3,
            n_total_runs=4,
            is_accepted=True,
            fitted_params=object(),
        )
        comparison = SimpleNamespace(
            lli=-0.12,
            lam_anode=0.08,
            lam_cathode=0.10,
        )
        with (
            patch(
                "bafuse.labeling.pydma_runner._fit_one",
                side_effect=[accepted, accepted],
            ),
            patch("pydma.DMAAnalyzer") as analyzer,
        ):
            analyzer.return_value.compare_with_reference.return_value = comparison
            result = run_uploaded_pydma(self._pydma_bundle())

        self.assertEqual(result["mapping_status"], "user_provided_unverified_not_guide_audited")
        self.assertEqual(result["estimate_status"], "user_uploaded_diagnostic_estimates")
        self.assertEqual(result["degradation_estimates"]["LLI_percent"], -12.0)
        self.assertTrue(result["negative_estimate_flags"]["LLI_percent"])

    def test_uploaded_pydma_withholds_estimates_when_a_fit_fails_gate(self) -> None:
        accepted = SimpleNamespace(
            rmse=0.010,
            n_accepted_runs=3,
            n_total_runs=4,
            is_accepted=True,
            fitted_params=object(),
        )
        rejected = SimpleNamespace(
            rmse=0.016,
            n_accepted_runs=2,
            n_total_runs=4,
            is_accepted=False,
            fitted_params=object(),
        )
        with (
            patch(
                "bafuse.labeling.pydma_runner._fit_one",
                side_effect=[accepted, rejected],
            ),
            patch("pydma.DMAAnalyzer") as analyzer,
        ):
            result = run_uploaded_pydma(self._pydma_bundle())

        self.assertEqual(result["estimate_status"], "withheld_fit_rejected")
        self.assertIsNone(result["degradation_estimates"])
        analyzer.assert_not_called()

    def test_uploaded_pydma_excludes_part2_groups_5_and_6(self) -> None:
        with self.assertRaisesRegex(UploadedCurveError, "Groups 5 and 6"):
            run_uploaded_pydma(self._pydma_bundle(part="2", group="5"))


if __name__ == "__main__":
    unittest.main()
