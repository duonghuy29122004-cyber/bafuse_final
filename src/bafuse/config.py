from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    dataset_parts: dict[str, Path]
    processed_root: Path
    interim_root: Path
    manifests_root: Path
    unified_observation_manifest: Path
    rpt_observation_manifest: Path
    source_availability_manifest: Path
    part2_bol_policy_decisions: Path
    feature_run: Path
    baseline_run: Path
    checkpoint: Path
    soh_labels: dict[str, Path]
    pydma_part12: Path
    pydma_part3: Path
    pydma_g5_audit: Path
    split_manifest: Path
    pydma_visualization: Path
    visualization_part12_labels: Path
    visualization_part3_labels: Path
    visualization_stage2_predictions: Path
    hybrid_output: Path


def _resolve(root: Path, value: Any, key: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Project path {key!r} must be a non-empty string.")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


@lru_cache(maxsize=8)
def load_project_paths(root: Path | str | None = None) -> ProjectPaths:
    repository_root = (
        Path(root).resolve()
        if root is not None
        else Path(__file__).resolve().parents[2]
    )
    config_path = repository_root / "configs" / "project_paths.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Project path config must be an object: {config_path}")

    def path(key: str) -> Path:
        return _resolve(repository_root, config.get(key), key)

    parts = config.get("dataset_parts")
    labels = config.get("soh_labels")
    if not isinstance(parts, dict) or set(parts) != {"1", "2", "3"}:
        raise ValueError("dataset_parts must define exactly parts 1, 2, and 3.")
    if not isinstance(labels, dict) or set(labels) != {"1", "2", "3"}:
        raise ValueError("soh_labels must define exactly parts 1, 2, and 3.")

    return ProjectPaths(
        root=repository_root,
        dataset_parts={
            part: _resolve(repository_root, value, f"dataset_parts.{part}")
            for part, value in parts.items()
        },
        processed_root=path("processed_root"),
        interim_root=path("interim_root"),
        manifests_root=path("manifests_root"),
        unified_observation_manifest=path("unified_observation_manifest"),
        rpt_observation_manifest=path("rpt_observation_manifest"),
        source_availability_manifest=path("source_availability_manifest"),
        part2_bol_policy_decisions=path("part2_bol_policy_decisions"),
        feature_run=path("feature_run"),
        baseline_run=path("baseline_run"),
        checkpoint=path("checkpoint"),
        soh_labels={
            part: _resolve(repository_root, value, f"soh_labels.{part}")
            for part, value in labels.items()
        },
        pydma_part12=path("pydma_part12"),
        pydma_part3=path("pydma_part3"),
        pydma_g5_audit=path("pydma_g5_audit"),
        split_manifest=path("split_manifest"),
        pydma_visualization=path("pydma_visualization"),
        visualization_part12_labels=path("visualization_part12_labels"),
        visualization_part3_labels=path("visualization_part3_labels"),
        visualization_stage2_predictions=path(
            "visualization_stage2_predictions"
        ),
        hybrid_output=path("hybrid_output"),
    )
