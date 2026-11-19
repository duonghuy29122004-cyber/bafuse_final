from __future__ import annotations

import json
import unittest

from experiments.soh_forecast.bafuse_architecture_benchmark_v1.prepare_benchmark import (
    BENCHMARK_DIR,
    _build_preflight_data,
    load_strict_part1_labels,
    reextract_g1_cell9_rpt8,
)


class BenchmarkPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prepared = _build_preflight_data()

    def test_strict_part1_whitelist_contains_exactly_94_approved_keys(self) -> None:
        labels = load_strict_part1_labels()

        self.assertEqual(len(labels), 94)
        self.assertIn(("1", "G1-Cell9", "8"), labels)
        self.assertTrue(all(key[0] == "1" for key in labels))

    def test_only_exact_part1_audit_keys_enter_dataset_construction(self) -> None:
        approved = set(self.prepared["strict_p1_labels"])
        part1_candidates = [
            row for row in self.prepared["admitted"] if row["part"] == "1"
        ]

        self.assertTrue(part1_candidates)
        self.assertTrue(
            all(
                (row["part"], row["physical_cell_id"], row["rpt_index"])
                in approved
                for row in part1_candidates
            )
        )
        self.assertTrue(self.prepared["rpt8"]["strict_part1_key"])
        self.assertFalse(self.prepared["rpt8"]["benchmark_label_eligible"])
        self.assertFalse(
            any(
                row["part"] == "1"
                and row["physical_cell_id"] == "G1-Cell9"
                and row["rpt_index"] == "8"
                for row in part1_candidates
            )
        )

    def test_rpt8_raw_reextraction_checks_step4_cutoff_and_step10(self) -> None:
        result = reextract_g1_cell9_rpt8()

        self.assertEqual(result["step4"]["row_count"], 20)
        self.assertEqual(result["step4"]["observed_step_ids"], [4])
        self.assertAlmostEqual(
            result["step4"]["voltage_min_v"], 3.738765545129, delta=1e-12
        )
        self.assertAlmostEqual(
            result["step4"]["capacity_span_native_amphr"],
            0.007603714301,
            delta=1e-12,
        )
        self.assertEqual(result["step4"]["temperature_missing_rows"], 0)
        self.assertFalse(result["step4_reaches_documented_cutoff"])
        self.assertEqual(result["step10"]["row_count"], 80638)
        self.assertEqual(result["step10"]["observed_step_ids"], [10])
        self.assertEqual(result["step10"]["current_positive_rows"], 80638)
        self.assertTrue(result["label_arithmetic_matches"])
        self.assertFalse(result["benchmark_label_eligible"])

    def test_part3_is_external_only_and_eis_is_not_joined_to_an_rpt(self) -> None:
        self.assertTrue(
            all(
                row["partition"] == "external_test"
                for row in self.prepared["records"]
                if row["part"] == "3"
            )
        )
        self.assertFalse(any(row["part"] == "3" for row in self.prepared["admitted"]))
        self.assertTrue(self.prepared["eis"])
        self.assertTrue(all(row["rpt_index"] == "" for row in self.prepared["eis"]))
        self.assertTrue(
            all(
                row["eis_same_rpt_feature_allowed"] is False
                for row in self.prepared["alignment"]
            )
        )
        self.assertTrue(
            all(
                row["eis_alignment_status"]
                == "checkpoint_only_no_exact_RPT_or_timestamp_join"
                for row in self.prepared["alignment"]
            )
        )
        alignment_by_key = {
            (row["part"], row["physical_cell_id"], row["rpt_index"]): row
            for row in self.prepared["alignment"]
        }
        self.assertTrue(
            all(
                alignment_by_key[
                    (row["part"], row["physical_cell_id"], row["rpt_index"])
                ]["step4_same_cell_rpt_mapping_verified"]
                for row in self.prepared["admitted"]
            )
        )

    def test_fresh_baseline_config_remains_untrained_and_uses_step10(self) -> None:
        config = json.loads(
            (BENCHMARK_DIR / "baseline_config.json").read_text(encoding="utf-8")
        )

        self.assertFalse(config["training_authorized"])
        self.assertFalse(config["training_started"])
        self.assertEqual(
            config["split_version"], "bafuse-architecture-benchmark-9-3-0-12-v1"
        )
        self.assertEqual(
            config["model"]["charge"]["source_step_by_part"],
            {"1": 10, "2": 10, "3": 10},
        )
        self.assertEqual(
            config["model"]["discharge"]["source_step_by_part"],
            {"1": 4, "2": 4, "3": 4},
        )


if __name__ == "__main__":
    unittest.main()
