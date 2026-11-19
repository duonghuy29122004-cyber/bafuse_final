from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from bafuse.config import load_project_paths


class ProjectPathConfigurationTests(unittest.TestCase):
    def test_repository_configuration_resolves_paths_from_project_root(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        paths = load_project_paths(repository_root)

        self.assertEqual(
            paths.dataset_parts["1"],
            repository_root / "data" / "raw" / "oxford" / "part1",
        )
        self.assertEqual(
            paths.checkpoint,
            repository_root
            / "models"
            / "checkpoints"
            / "part1_cnn1d"
            / "cnn1d_checkpoint.pt",
        )
        self.assertEqual(
            paths.rpt_observation_manifest,
            repository_root / "data" / "manifests" / "bafuse_rpt_observations.csv",
        )
        self.assertEqual(
            paths.part2_bol_policy_decisions,
            repository_root
            / "reports"
            / "bafuse_part2_bol_resolution_v1"
            / "part2_bol_policy_decisions.csv",
        )

    def test_invalid_part_mapping_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            config_dir = root / "configs"
            config_dir.mkdir()
            (config_dir / "project_paths.json").write_text(
                json.dumps({"dataset_parts": {"1": "part1"}}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "dataset_parts"):
                load_project_paths(root)


if __name__ == "__main__":
    unittest.main()
