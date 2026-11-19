from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from bafuse.hybrid.soh_fusion import resample_curve
from train_formula_ablation import (
    VARIANTS,
    fit_physics,
    forward_smoke_test,
    read_internal_data,
)


class FormulaAblationDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = read_internal_data()

    def test_internal_archive_has_only_train_and_validation(self) -> None:
        self.assertEqual(len(self.data["y_soh"]), 139)
        self.assertEqual(set(self.data["dataset_part"].tolist()), {"1", "2"})
        self.assertEqual(
            {name: int(np.sum(self.data["partition"] == name)) for name in ("train", "validation")},
            {"train": 102, "validation": 37},
        )
        self.assertEqual(
            {
                name: int(np.sum(self.data["legacy_partition"] == name))
                for name in ("train", "validation")
            },
            {"train": 99, "validation": 40},
        )
        self.assertTrue(
            np.isfinite(self.data["X_discharge"]).all()
            and np.isfinite(self.data["X_charge"]).all()
            and np.isfinite(self.data["X_physics_p5_raw"]).all()
            and np.isfinite(self.data["X_physics_legacy_raw"]).all()
            and np.isfinite(self.data["y_soh"]).all()
        )

    def test_physical_cells_are_isolated_under_both_splits(self) -> None:
        for split in ("partition", "legacy_partition"):
            train_cells = set(
                self.data["physical_cell_id"][self.data[split] == "train"].tolist()
            )
            validation_cells = set(
                self.data["physical_cell_id"][self.data[split] == "validation"].tolist()
            )
            self.assertFalse(train_cells & validation_cells)

    def test_physics_scaler_is_fitted_from_train_only(self) -> None:
        train_mask = self.data["partition"] == "train"
        scaled, scaler = fit_physics(self.data["X_physics_legacy_raw"], train_mask)
        expected_mean = self.data["X_physics_legacy_raw"][train_mask].astype(
            np.float64
        ).mean(axis=0)
        self.assertEqual(scaler["fit_rows"], 102)
        np.testing.assert_allclose(scaler["mean"], expected_mean, rtol=0, atol=1e-7)
        altered = self.data["X_physics_legacy_raw"].copy()
        altered[~train_mask] += 1e5
        _, altered_scaler = fit_physics(altered, train_mask)
        np.testing.assert_array_equal(scaler["mean"], altered_scaler["mean"])
        np.testing.assert_array_equal(
            scaler["scale_population_std"], altered_scaler["scale_population_std"]
        )
        self.assertEqual(scaled.shape, (139, 5))

    def test_train_constant_physics_feature_is_zeroed_for_all_partitions(self) -> None:
        raw = np.array(
            [
                [1, 7, 2, 3, 4],
                [2, 7, 4, 4, 5],
                [3, 7, 6, 5, 6],
                [4, 7, 8, 6, 7],
                [5, 9, 10, 7, 8],
                [6, 11, 12, 8, 9],
            ],
            dtype=np.float32,
        )
        train_mask = np.array([True, True, True, True, False, False])
        scaled, scaler = fit_physics(raw, train_mask)
        self.assertEqual(scaler["scale_population_std"][1], 0.0)
        self.assertEqual(scaler["scale_used"][1], 1.0)
        self.assertEqual(
            scaler["constant_train_features_zeroed"],
            ["discharge_current_variability"],
        )
        np.testing.assert_array_equal(scaled[:, 1], np.zeros(6, dtype=np.float32))

    def test_all_new_model_branches_accept_synthetic_shapes(self) -> None:
        results = forward_smoke_test(torch.device("cpu"))
        self.assertEqual(len(results), len(VARIANTS))
        for result in results:
            self.assertEqual(result["output_shape"], [2, 1])
            self.assertGreater(result["parameter_count"], 0)
        branch_counts = {
            result["parameter_count"]
            for result in results
            if result["variant"] in {"A1_discharge_only", "A2_discharge_charge", "A3_discharge_p5_physics"}
        }
        self.assertGreater(len(branch_counts), 1)

    def test_curve_resampling_is_deterministic_and_target_independent(self) -> None:
        raw = [4.1, 3.8, 3.4, 2.5]
        first = resample_curve(raw, n_points=128)
        second = resample_curve(raw, n_points=128)
        self.assertEqual(len(first), 128)
        self.assertEqual(first, second)
        self.assertEqual(first[0], raw[0])
        self.assertEqual(first[-1], raw[-1])

    def test_training_source_does_not_reference_external_npz(self) -> None:
        source = (Path(__file__).with_name("train_formula_ablation.py")).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("bafuse_external_p5.npz", source)
        self.assertNotIn("external_predictions.csv", source)


if __name__ == "__main__":
    unittest.main()
