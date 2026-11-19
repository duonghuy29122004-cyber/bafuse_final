from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from bafuse.features.physics_features import (
    PHYSICS_ENCODER_INPUT_COUNT,
    PHYSICS_FEATURE_NAMES,
    PhysicsFeatureScaler,
    build_physics_feature_rows,
    physics_feature_matrix,
    write_physics_feature_artifacts,
)


def _source_row(
    source_file: str = "Group_1/TPG1_4_Cell9.csv",
    offset: float = 0.0,
) -> dict[str, str]:
    row = {
        "physical_cell_id": "G1-Cell9",
        "measurement_id": "RPT 1",
        "source_file": source_file,
        "physics_temp1_available": "True",
    }
    row.update(
        {
            name: str(float(index + 1) + offset)
            for index, name in enumerate(PHYSICS_FEATURE_NAMES)
        }
    )
    row["soh_pct"] = "95.0"
    row["lli_pct"] = "2.0"
    return row


class PhysicsFeatureDatasetTests(unittest.TestCase):
    def test_collapses_direction_rows_and_excludes_targets(self) -> None:
        first = _source_row()
        duplicate = dict(first, direction="discharge", test_step="8")

        rows = build_physics_feature_rows([first, duplicate])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["physics_feature_status"], "available")
        self.assertNotIn("soh_pct", rows[0])
        self.assertNotIn("lli_pct", rows[0])
        self.assertTrue(all(rows[0][f"{name}_available"] for name in PHYSICS_FEATURE_NAMES))

    def test_accepts_native_numeric_and_boolean_values_from_feature_runner(self) -> None:
        row = _source_row()
        row["physics_temp1_available"] = True
        for name in PHYSICS_FEATURE_NAMES:
            row[name] = float(row[name])

        features = build_physics_feature_rows([row])

        self.assertEqual(features[0]["physics_feature_status"], "available")
        self.assertEqual(features[0][PHYSICS_FEATURE_NAMES[0]], 1.0)

    def test_artifact_writer_accepts_feature_runner_rows(self) -> None:
        row = _source_row()
        row["physics_temp1_available"] = True
        for name in PHYSICS_FEATURE_NAMES:
            row[name] = float(row[name])
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)

            manifest = write_physics_feature_artifacts(
                output_dir, [row], output_dir / "source_manifest.csv"
            )

            self.assertEqual(manifest["unique_cell_rpt_rows"], 1)
            self.assertTrue((output_dir / "physics_features.csv").is_file())
            self.assertTrue((output_dir / "physics_feature_manifest.json").is_file())

    def test_feature_matrix_preserves_missing_values_for_train_fold_imputation(self) -> None:
        row = _source_row()
        row["physics_temp1_available"] = ""
        row[PHYSICS_FEATURE_NAMES[2]] = ""

        matrix = physics_feature_matrix(build_physics_feature_rows([row]))

        self.assertEqual(matrix.shape, (1, len(PHYSICS_FEATURE_NAMES)))
        self.assertTrue(np.isnan(matrix[0, 2]))

    def test_rejects_conflicting_duplicate_feature_values(self) -> None:
        first = _source_row()
        duplicate = _source_row(source_file="group_1\\tpg1_4_cell9.csv", offset=1.0)

        with self.assertRaisesRegex(ValueError, "disagree on step4_temp1_mean_C"):
            build_physics_feature_rows([first, duplicate])

    def test_scaler_fits_only_training_rows_and_preserves_missingness(self) -> None:
        train = np.asarray(
            [
                np.arange(1, 9, dtype=np.float64),
                np.arange(3, 11, dtype=np.float64),
            ]
        )
        scaler = PhysicsFeatureScaler.fit(train)
        evaluation = np.asarray(
            [[101.0, 102.0, np.nan, 104.0, 105.0, 106.0, 107.0, 108.0]]
        )

        transformed = scaler.transform(evaluation)

        self.assertEqual(PHYSICS_ENCODER_INPUT_COUNT, 16)
        self.assertEqual(transformed.shape, (1, PHYSICS_ENCODER_INPUT_COUNT))
        self.assertAlmostEqual(float(scaler.mean[0]), 2.0)
        self.assertAlmostEqual(float(transformed[0, 0]), 99.0)
        self.assertEqual(float(transformed[0, 2]), 0.0)
        self.assertEqual(float(transformed[0, 8 + 2]), 1.0)
        self.assertTrue(np.isfinite(transformed).all())

    def test_scaler_rejects_features_missing_from_entire_training_split(self) -> None:
        train = np.full((2, len(PHYSICS_FEATURE_NAMES)), np.nan)
        train[:, 0] = [1.0, 2.0]

        with self.assertRaisesRegex(ValueError, "no finite observations"):
            PhysicsFeatureScaler.fit(train)


if __name__ == "__main__":
    unittest.main()
