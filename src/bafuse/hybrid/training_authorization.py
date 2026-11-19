from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


class TrainingAuthorizationError(RuntimeError):
    pass


def _contains_unset(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_contains_unset(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_unset(item) for item in value)
    return False


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Authorization input is missing: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_training_scope(
    *,
    label_manifest: Path,
    feature_matrix: Path,
    split_manifest: Path,
    experiment_config: Path,
    compute_budget: dict[str, Any],
    output_directory: Path,
    architectures: list[str],
) -> dict[str, Any]:
    output_path = output_directory.resolve()
    if output_path.exists():
        raise TrainingAuthorizationError(
            f"Training output directory already exists: {output_path}"
        )
    if not architectures or len(set(architectures)) != len(architectures):
        raise TrainingAuthorizationError("Architecture scope must be nonempty and unique.")
    if not compute_budget or _contains_unset(compute_budget):
        raise TrainingAuthorizationError(
            "Compute budget must be explicit and fully specified before authorization."
        )
    if "device" in compute_budget and not str(compute_budget["device"]).strip():
        raise TrainingAuthorizationError("Compute budget must identify a device.")
    if "max_total_device_hours" in compute_budget:
        try:
            hours = float(compute_budget["max_total_device_hours"])
            if not math.isfinite(hours) or hours <= 0:
                raise TrainingAuthorizationError(
                    "Maximum total device-hours must be finite and positive."
                )
        except (TypeError, ValueError) as error:
            raise TrainingAuthorizationError(
                "Maximum total device-hours must be numeric."
            ) from error
    for key in ("max_epochs_per_run", "batch_size"):
        if key in compute_budget:
            try:
                if int(compute_budget[key]) <= 0:
                    raise TrainingAuthorizationError(f"{key} must be positive.")
            except (TypeError, ValueError) as error:
                raise TrainingAuthorizationError(f"{key} must be an integer.") from error
    if "seeds" in compute_budget:
        seeds = compute_budget["seeds"]
        if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds):
            raise TrainingAuthorizationError("Compute budget must define unique random seeds.")
    return {
        "schema_version": 1,
        "label_manifest_sha256": _sha256(label_manifest.resolve()),
        "feature_matrix_sha256": _sha256(feature_matrix.resolve()),
        "split_manifest_sha256": _sha256(split_manifest.resolve()),
        "experiment_config_sha256": _sha256(experiment_config.resolve()),
        "architectures": architectures,
        "compute_budget": compute_budget,
        "output_directory": str(output_path),
    }


def require_training_authorization(
    authorization_file: Path, expected_scope: dict[str, Any]
) -> None:
    if not authorization_file.is_file():
        raise TrainingAuthorizationError(
            f"Training is blocked: explicit authorization file not found: {authorization_file}"
        )
    try:
        authorization = json.loads(authorization_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise TrainingAuthorizationError(
            f"Training is blocked: authorization file is unreadable: {error}"
        ) from error
    if not isinstance(authorization, dict):
        raise TrainingAuthorizationError("Training is blocked: authorization must be a JSON object.")
    if authorization.get("schema_version") != 1:
        raise TrainingAuthorizationError("Training is blocked: unsupported authorization schema.")
    if authorization.get("status") != "approved":
        raise TrainingAuthorizationError(
            "Training is blocked: authorization status is not explicitly 'approved'."
        )
    if not str(authorization.get("approved_by_user", "")).strip():
        raise TrainingAuthorizationError("Training is blocked: approver is not identified.")
    if not str(authorization.get("approved_at_utc", "")).strip():
        raise TrainingAuthorizationError("Training is blocked: approval timestamp is missing.")
    if authorization.get("scope") != expected_scope:
        raise TrainingAuthorizationError(
            "Training is blocked: authorization scope does not exactly match the current "
            "label manifest, feature matrix, split, experiment config, architectures, "
            "compute budget, and output directory."
        )
