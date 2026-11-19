from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from bafuse.data.artifacts import (
    artifact_output_path,
    resolve_run_artifact,
)


class BenchmarkArtifactPathTests(unittest.TestCase):
    def test_experiment_artifact_maps_to_preserving_artifact_tree(self) -> None:
        repository_root = Path(__file__).resolve().parents[2]
        experiment_run = repository_root / "experiments" / "test_artifact_path"
        artifact = artifact_output_path(experiment_run, "checkpoint.pt")

        self.assertEqual(
            artifact,
            repository_root
            / "models"
            / "legacy_experiments"
            / "test_artifact_path"
            / "checkpoint.pt",
        )
        self.assertTrue(artifact.parent.is_dir())
        artifact.write_bytes(b"test")
        try:
            self.assertEqual(
                resolve_run_artifact(experiment_run, "checkpoint.pt"),
                artifact,
            )
        finally:
            artifact.unlink()
            artifact.parent.rmdir()

    def test_non_experiment_outputs_remain_in_the_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            run_dir.mkdir()
            artifact = artifact_output_path(run_dir, "checkpoint.pt")

            self.assertEqual(artifact, run_dir / "checkpoint.pt")

    def test_artifact_names_cannot_escape_the_run_directory(self) -> None:
        with self.assertRaisesRegex(ValueError, "basename"):
            artifact_output_path(Path("run"), "..\\outside.pt")


if __name__ == "__main__":
    unittest.main()
