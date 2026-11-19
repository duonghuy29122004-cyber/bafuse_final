from __future__ import annotations

import json

import pytest

from bafuse.hybrid.training_authorization import (
    TrainingAuthorizationError,
    build_training_scope,
    require_training_authorization,
)


def _inputs(tmp_path):
    paths = {}
    for name in ("labels.csv", "features.csv", "split.csv", "config.json"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        paths[name] = path
    return paths


def _scope(paths, output):
    return build_training_scope(
        label_manifest=paths["labels.csv"],
        feature_matrix=paths["features.csv"],
        split_manifest=paths["split.csv"],
        experiment_config=paths["config.json"],
        compute_budget={"device": "cpu", "max_hours": 1},
        output_directory=output,
        architectures=["cnn1d_cnn1d_mlp"],
    )


def test_authorization_fails_closed_when_missing_or_not_approved(tmp_path):
    paths = _inputs(tmp_path)
    scope = _scope(paths, tmp_path / "run")
    authorization = tmp_path / "authorization.json"

    with pytest.raises(TrainingAuthorizationError, match="not found"):
        require_training_authorization(authorization, scope)

    authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "not_approved",
                "approved_by_user": "",
                "approved_at_utc": "",
                "scope": scope,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TrainingAuthorizationError, match="not explicitly"):
        require_training_authorization(authorization, scope)


def test_authorization_requires_exact_current_hash_bound_scope(tmp_path):
    paths = _inputs(tmp_path)
    output = tmp_path / "run"
    scope = _scope(paths, output)
    authorization = tmp_path / "authorization.json"
    authorization.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "approved",
                "approved_by_user": "user",
                "approved_at_utc": "2026-11-13T00:00:00Z",
                "scope": scope,
            }
        ),
        encoding="utf-8",
    )
    require_training_authorization(authorization, scope)

    paths["labels.csv"].write_text("changed", encoding="utf-8")
    current_scope = _scope(paths, output)
    with pytest.raises(TrainingAuthorizationError, match="does not exactly match"):
        require_training_authorization(authorization, current_scope)


def test_scope_rejects_incomplete_budget_and_nonempty_output(tmp_path):
    paths = _inputs(tmp_path)
    with pytest.raises(TrainingAuthorizationError, match="fully specified"):
        build_training_scope(
            label_manifest=paths["labels.csv"],
            feature_matrix=paths["features.csv"],
            split_manifest=paths["split.csv"],
            experiment_config=paths["config.json"],
            compute_budget={"device": None, "max_hours": 1},
            output_directory=tmp_path / "run",
            architectures=["cnn1d_cnn1d_mlp"],
        )

    output = tmp_path / "existing"
    output.mkdir()
    (output / "checkpoint.pt").write_bytes(b"keep")
    with pytest.raises(TrainingAuthorizationError, match="already exists"):
        _scope(paths, output)
