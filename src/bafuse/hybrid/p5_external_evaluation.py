from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from bafuse.hybrid.p5_benchmark_models import (
    ARCHITECTURE_NAMES,
    architecture_config,
    build_model,
    trainable_parameter_count,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_external_evaluation_v1_20261114"
    / "external_evaluation_config.json"
)
SEEDS = (42, 123, 2026)
EXPECTED_DATASET_MANIFEST_SHA256 = (
    "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
)
EXPECTED_EXTERNAL_NPZ_SHA256 = (
    "f97cbd5eb9682380dc296f1aa518adf4bfa82845e5d07d76a8544eba3b84d29b"
)
EXPECTED_BENCHMARK_MANIFEST_SHA256 = (
    "4af99b1cde2a3ce24e8ba066f38ab8c2a7e83e0b20896c5b478716daa77f1e7c"
)
EXPECTED_PART_COUNTS = {"train": 102, "validation": 37, "external_test": 64}
AUDIT_FEATURE_COLUMNS = (
    "p5_scaled_mean_abs_discharge_current",
    "p5_scaled_discharge_current_variability",
    "p5_scaled_discharge_voltage_slope_20_80",
    "p5_scaled_discharge_voltage_derivative_mad_10_90",
    "p5_scaled_discharge_voltage_segment_contrast_10_30_vs_70_90",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve(path_value: str | Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty evaluation artifact: {path}.")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset_part"], row["physical_cell_id"], row["rpt_index"]


def _verify_hash_section(
    manifest: dict[str, Any],
    section: str,
    *,
    local_output_directory: Path | None = None,
) -> list[dict[str, str]]:
    entries = manifest.get(section)
    if not isinstance(entries, dict):
        raise ValueError(f"Manifest is missing the {section} mapping.")
    verified: list[dict[str, str]] = []
    for name, entry in entries.items():
        if section != "outputs" and not entry.get("path"):
            if section == "inputs" and name == "authorization_scope":
                verified.append(
                    {
                        "name": name,
                        "sha256": entry["sha256"],
                        "verified_via": "benchmark authorization_scope.json",
                    }
                )
                continue
            raise ValueError(f"Manifest {section} entry {name} has no file path.")
        if section == "outputs" and local_output_directory is not None:
            path = local_output_directory / name
        else:
            path = _resolve(entry.get("path", name))
        if not path.is_file():
            raise FileNotFoundError(f"Manifest {section} file is missing: {path}.")
        observed = _sha256(path)
        expected = entry.get("sha256")
        if observed != expected:
            raise ValueError(
                f"Manifest {section} hash mismatch for {name}: "
                f"expected {expected}, got {observed}."
            )
        verified.append(
            {
                "name": name,
                "path": str(path),
                "sha256": observed,
            }
        )
    return verified


def _validate_config(config: dict[str, Any]) -> None:
    if config.get("evaluation_identifier") != "bafuse-p5-external-evaluation-v1-20261114":
        raise ValueError("Unexpected External evaluation identifier.")
    if config.get("protocol_status") != "locked_before_external_inference":
        raise ValueError("Evaluation protocol must be locked before inference.")
    if config.get("architectures") != list(ARCHITECTURE_NAMES):
        raise ValueError("External evaluation architecture list changed.")
    if config.get("seeds") != list(SEEDS):
        raise ValueError("External evaluation seed list changed.")
    if config.get("required_model_seeds") != {
        architecture: list(SEEDS) for architecture in ARCHITECTURE_NAMES
    }:
        raise ValueError("Required per-architecture checkpoint seeds changed.")
    expected_auth = {
        "external_prediction_evaluation_authorized": True,
        "external_metrics_authorized": True,
        "external_diagnostics_authorized": True,
        "training_authorized": False,
        "fine_tuning_authorized": False,
        "hyperparameter_tuning_authorized": False,
        "checkpoint_modification_authorized": False,
        "dataset_modification_authorized": False,
        "feature_schema_modification_authorized": False,
        "label_modification_authorized": False,
        "pydma_fitting_authorized": False,
    }
    if config.get("authorization") != expected_auth:
        raise ValueError("External evaluation authorization flags changed.")
    if config.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Unexpected frozen dataset manifest SHA-256.")
    if config.get("external_npz_sha256") != EXPECTED_EXTERNAL_NPZ_SHA256:
        raise ValueError("Unexpected External NPZ SHA-256.")
    if config.get("benchmark_manifest_sha256") != EXPECTED_BENCHMARK_MANIFEST_SHA256:
        raise ValueError("Unexpected Train/Validation benchmark SHA-256.")
    if config.get("predeclared_validation_selections") != {
        "best_individual_checkpoint": {
            "architecture": "BaFuse_CNN1D_P5",
            "seed": 123,
        },
        "best_architecture_by_mean_validation_RMSE": "BaFuse_TCN_P5",
        "best_TCN_checkpoint_by_validation_RMSE": {
            "architecture": "BaFuse_TCN_P5",
            "seed": 42,
        },
        "external_results_may_change_selection": False,
        "seed_predictions_may_be_ensembled": False,
    }:
        raise ValueError("Predeclared Validation-based selection protocol changed.")
    if config.get("inference") != {
        "device": "cuda:0",
        "model_eval_mode": True,
        "torch_inference_mode": True,
        "optimizer_created": False,
        "backward_called": False,
        "posthoc_calibration": False,
        "prediction_clipping": False,
        "preprocessing_refit": False,
        "checkpoint_or_dataset_modification": False,
        "reload_prediction_absolute_tolerance": 1e-6,
    }:
        raise ValueError("Locked inference-only settings changed.")
    if config.get("external_tensor_contract") != {
        "X_discharge": {"shape": [64, 128], "dtype": "float32"},
        "X_charge": {"shape": [64, 128], "dtype": "float32"},
        "X_physics": {"shape": [64, 5], "dtype": "float32"},
        "y_soh": {"shape": [64, 1], "dtype": "float32"},
    }:
        raise ValueError("Locked External tensor contract changed.")


def _verify_frozen_dataset(
    config: dict[str, Any], dataset_dir: Path
) -> dict[str, Any]:
    manifest_path = dataset_dir / "bafuse_p5_sha256_manifest.json"
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError(
            f"Frozen dataset manifest mismatch: {manifest_hash}."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_identifier") != config["dataset_identifier"]:
        raise ValueError("Frozen dataset identifier mismatch.")
    outputs = _verify_hash_section(
        manifest, "outputs", local_output_directory=dataset_dir
    )
    inputs = _verify_hash_section(manifest, "inputs")
    code = _verify_hash_section(manifest, "code")
    external_path = dataset_dir / "bafuse_external_p5.npz"
    external_hash = _sha256(external_path)
    if external_hash != EXPECTED_EXTERNAL_NPZ_SHA256:
        raise ValueError(
            f"External tensor hash mismatch: expected {EXPECTED_EXTERNAL_NPZ_SHA256}, "
            f"got {external_hash}."
        )
    if manifest["outputs"]["bafuse_external_p5.npz"]["sha256"] != external_hash:
        raise ValueError("External tensor hash differs from the frozen manifest entry.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "external_npz_path": external_path,
        "external_npz_sha256": external_hash,
        "verified_outputs": outputs,
        "verified_inputs": inputs,
        "verified_code": code,
    }


def _verify_benchmark(
    config: dict[str, Any], benchmark_dir: Path
) -> dict[str, Any]:
    manifest_path = benchmark_dir / "benchmark_sha256_manifest.json"
    manifest_hash = _sha256(manifest_path)
    if manifest_hash != EXPECTED_BENCHMARK_MANIFEST_SHA256:
        raise ValueError(
            f"Train/Validation benchmark manifest mismatch: {manifest_hash}."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("benchmark_identifier") != config["benchmark_identifier"]:
        raise ValueError("Train/Validation benchmark identifier mismatch.")
    if manifest.get("dataset_identifier") != config["dataset_identifier"]:
        raise ValueError("Train/Validation benchmark references another dataset.")
    if manifest.get("dataset_manifest_sha256") != EXPECTED_DATASET_MANIFEST_SHA256:
        raise ValueError("Train/Validation benchmark has another dataset hash.")
    outputs = _verify_hash_section(
        manifest, "outputs", local_output_directory=benchmark_dir
    )
    inputs = _verify_hash_section(manifest, "inputs")
    code = _verify_hash_section(manifest, "code")
    scope_record = _read_json(benchmark_dir / "authorization_scope.json")
    scope_hash = _hash_json(scope_record.get("scope"))
    if (
        scope_hash != manifest["inputs"]["authorization_scope"]["sha256"]
        or scope_record.get("scope_sha256") != scope_hash
    ):
        raise ValueError("Train/Validation authorization scope hash mismatch.")
    run_rows = _read_csv(benchmark_dir / "benchmark_run_metrics.csv")
    if len(run_rows) != 12:
        raise ValueError(f"Expected 12 locked training runs, found {len(run_rows)}.")
    expected_pairs = {
        (architecture, seed)
        for architecture in ARCHITECTURE_NAMES
        for seed in SEEDS
    }
    actual_pairs = {(row["architecture"], int(row["seed"])) for row in run_rows}
    if actual_pairs != expected_pairs:
        raise ValueError("Train/Validation benchmark does not contain all 12 model seeds.")
    checkpoint_paths: set[str] = set()
    checkpoint_records: list[dict[str, Any]] = []
    config_sha256 = manifest["inputs"]["training_config"]["sha256"]
    source_code_sha256 = _hash_json(
        {
            "training_runner": manifest["code"]["training_runner"]["sha256"],
            "model_definitions": manifest["code"]["model_definitions"]["sha256"],
        }
    )
    benchmark_config = _read_json(benchmark_dir / "benchmark_config.json")
    if benchmark_config.get("benchmark_identifier") != config["benchmark_identifier"]:
        raise ValueError("Recorded training config identifier mismatch.")
    for row in run_rows:
        checkpoint_relpath = Path(row["checkpoint_path"]).as_posix()
        checkpoint_paths.add(checkpoint_relpath)
        checkpoint_path = benchmark_dir / Path(row["checkpoint_path"])
        observed_hash = _sha256(checkpoint_path)
        if observed_hash != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch for {checkpoint_relpath}.")
        manifest_hash_for_checkpoint = manifest["outputs"].get(checkpoint_relpath, {}).get(
            "sha256"
        )
        if observed_hash != manifest_hash_for_checkpoint:
            raise ValueError(
                f"Checkpoint does not match the benchmark manifest: {checkpoint_relpath}."
            )
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        expected_seed = int(row["seed"])
        expected_parameters = int(row["parameter_count"])
        expected_validation_rmse = float(row["validation_RMSE"])
        if (
            row["dataset_identifier"] != config["dataset_identifier"]
            or row["dataset_manifest_sha256"] != EXPECTED_DATASET_MANIFEST_SHA256
            or row["training_configuration_sha256"] != config_sha256
            or row["model_source_code_sha256"] != source_code_sha256
        ):
            raise ValueError(
                f"Run-metric provenance mismatch for {checkpoint_relpath}."
            )
        checks = (
            checkpoint.get("architecture") == row["architecture"],
            checkpoint.get("seed") == expected_seed,
            checkpoint.get("best_epoch") == int(row["best_epoch"]),
            checkpoint.get("parameter_count") == expected_parameters,
            checkpoint.get("frozen_dataset_identifier") == config["dataset_identifier"],
            checkpoint.get("frozen_dataset_manifest_sha256")
            == EXPECTED_DATASET_MANIFEST_SHA256,
            checkpoint.get("benchmark_config_sha256") == config_sha256,
            checkpoint.get("model_source_code_sha256") == source_code_sha256,
            checkpoint.get("architecture_config")
            == architecture_config(row["architecture"]),
            checkpoint.get("training_configuration")
            == benchmark_config["training"],
            checkpoint.get("target_transform") is None,
            abs(
                float(checkpoint.get("best_validation_RMSE", math.inf))
                - expected_validation_rmse
            )
            <= 1e-7,
        )
        if not all(checks):
            raise ValueError(
                f"Checkpoint metadata mismatch for {checkpoint_relpath}; "
                "External inference is blocked."
            )
        if trainable_parameter_count(build_model(row["architecture"])) != expected_parameters:
            raise ValueError(
                f"Model parameter count mismatch for {checkpoint_relpath}."
            )
        checkpoint_records.append(
            {
                "architecture": row["architecture"],
                "seed": expected_seed,
                "path": checkpoint_relpath,
                "sha256": observed_hash,
                "parameter_count": expected_parameters,
                "best_epoch": int(row["best_epoch"]),
                "best_validation_RMSE": expected_validation_rmse,
                "dataset_identifier": checkpoint["frozen_dataset_identifier"],
                "dataset_manifest_sha256": checkpoint[
                    "frozen_dataset_manifest_sha256"
                ],
                "architecture_configuration_valid": True,
                "optimizer_state_present": isinstance(
                    checkpoint.get("optimizer_state_dict"), dict
                ),
            }
        )
    if len(checkpoint_paths) != 12:
        raise ValueError("Expected 12 unique checkpoint paths.")
    best_individual = min(
        run_rows,
        key=lambda row: (float(row["validation_RMSE"]), float(row["validation_MAE"])),
    )
    if (
        best_individual["architecture"],
        int(best_individual["seed"]),
    ) != ("BaFuse_CNN1D_P5", 123):
        raise ValueError("Predeclared best individual checkpoint does not match Validation.")
    summary_rows = _read_csv(benchmark_dir / "benchmark_model_summary.csv")
    if len(summary_rows) != 4:
        raise ValueError("Expected four locked architecture summary rows.")
    best_architecture = min(
        summary_rows,
        key=lambda row: (
            float(row["validation_RMSE_mean"]),
            float(row["validation_MAE_mean"]),
        ),
    )
    if best_architecture["architecture"] != "BaFuse_TCN_P5":
        raise ValueError("Predeclared Validation-best architecture does not match.")
    tcn_best = min(
        (row for row in run_rows if row["architecture"] == "BaFuse_TCN_P5"),
        key=lambda row: float(row["validation_RMSE"]),
    )
    if int(tcn_best["seed"]) != 42:
        raise ValueError("Predeclared best TCN seed does not match Validation.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": manifest_hash,
        "verified_outputs": outputs,
        "verified_inputs": inputs,
        "verified_code": code,
        "run_rows": run_rows,
        "checkpoint_records": checkpoint_records,
        "benchmark_config": benchmark_config,
        "validation_selection_checks": {
            "best_individual_checkpoint": {
                "architecture": best_individual["architecture"],
                "seed": int(best_individual["seed"]),
            },
            "best_architecture_by_mean_validation_RMSE": best_architecture[
                "architecture"
            ],
            "best_TCN_checkpoint_by_validation_RMSE": {
                "architecture": tcn_best["architecture"],
                "seed": int(tcn_best["seed"]),
            },
            "matches_predeclared_protocol": True,
        },
    }


def _validate_external_arrays(
    config: dict[str, Any],
    dataset_dir: Path,
    split_rows: list[dict[str, str]],
    label_rows: list[dict[str, str]],
    feature_rows: list[dict[str, str]],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], np.ndarray]:
    split_by_key = {_key(row): row for row in split_rows}
    if len(split_by_key) != len(split_rows) or len(split_rows) != 203:
        raise ValueError("Frozen split manifest must have 203 unique observation keys.")
    label_by_key = {_key(row): row for row in label_rows}
    if len(label_by_key) != len(label_rows) or set(label_by_key) != set(split_by_key):
        raise ValueError("Frozen labels and split keys are not a unique one-to-one mapping.")
    if any(row["final_status"] != "ACCEPTED" for row in label_rows):
        raise ValueError("The frozen label cohort contains a non-accepted label.")
    if any(row["partition"] != split_by_key[_key(row)]["partition"] for row in label_rows):
        raise ValueError("Label and split partition assignments differ.")
    partition_counts = {
        partition: sum(row["partition"] == partition for row in split_rows)
        for partition in EXPECTED_PART_COUNTS
    }
    if partition_counts != EXPECTED_PART_COUNTS:
        raise ValueError(f"Unexpected frozen partition counts: {partition_counts}.")
    cell_partition: dict[str, str] = {}
    for row in split_rows:
        previous = cell_partition.setdefault(row["physical_cell_id"], row["partition"])
        if previous != row["partition"]:
            raise ValueError(f"Physical cell overlaps splits: {row['physical_cell_id']}.")
    external_rows = sorted(
        (row for row in split_rows if row["partition"] == "external_test"),
        key=lambda row: int(row["split_row_index"]),
    )
    if len(external_rows) != int(config["expected_external_rows"]):
        raise ValueError("External split has an unexpected observation count.")
    if any(row["dataset_part"] != config["expected_external_part"] for row in external_rows):
        raise ValueError("External split includes an observation outside Part 3.")
    external_cells = {row["physical_cell_id"] for row in external_rows}
    internal_cells = {
        row["physical_cell_id"]
        for row in split_rows
        if row["partition"] in {"train", "validation"}
    }
    if len(external_cells) != int(config["expected_external_physical_cells"]):
        raise ValueError(f"Unexpected External physical-cell count: {len(external_cells)}.")
    if external_cells.intersection(internal_cells):
        raise ValueError("External physical cells overlap Train or Validation.")
    if any(
        label_by_key[_key(row)]["final_status"] != "ACCEPTED"
        for row in external_rows
    ):
        raise ValueError("External reference includes a non-accepted label.")

    audit_by_key = {_key(row): row for row in feature_rows}
    if len(audit_by_key) != len(feature_rows) or set(audit_by_key) != set(split_by_key):
        raise ValueError("Frozen feature-audit keys do not match split keys.")
    for row in external_rows:
        audit = audit_by_key[_key(row)]
        if audit["partition"] != "external_test":
            raise ValueError(f"External feature audit partition mismatch for {_key(row)}.")
        if audit["predictive_label_or_bol_feature"] != "False":
            raise ValueError(f"Target/BoL feature leakage is flagged for {_key(row)}.")
        if audit["future_rpt_feature"] != "False":
            raise ValueError(f"Future-RPT leakage is flagged for {_key(row)}.")
        if audit["predictive_temperature_feature"] != "False":
            raise ValueError(f"Temperature entered the frozen inputs for {_key(row)}.")
        if audit["same_rpt_step_order_valid"] != "True":
            raise ValueError(f"Same-RPT modality alignment failed for {_key(row)}.")

    npz_path = dataset_dir / "bafuse_external_p5.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        expected_names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected_names:
            raise ValueError("External NPZ has an unexpected member list.")
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "X_discharge": (64, 128),
        "X_charge": (64, 128),
        "X_physics": (64, 5),
        "y_soh": (64, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected_shapes[name]:
            raise ValueError(
                f"External array {name} has unexpected dtype/shape "
                f"{array.dtype}/{array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"External array {name} contains non-finite values.")
    target_values = [
        float(label_by_key[_key(row)]["soh_percent"]) for row in external_rows
    ]
    if not np.isfinite(target_values).all():
        raise ValueError("External reference labels contain non-finite values.")
    expected_targets = np.asarray(target_values, dtype=np.float32).reshape(-1, 1)
    if not np.array_equal(arrays["y_soh"], expected_targets):
        raise ValueError("External NPZ target order does not match the frozen split.")
    expected_physics = np.asarray(
        [
            [float(audit_by_key[_key(row)][name]) for name in AUDIT_FEATURE_COLUMNS]
            for row in external_rows
        ],
        dtype=np.float32,
    )
    if not np.allclose(arrays["X_physics"], expected_physics, rtol=0.0, atol=1e-6):
        raise ValueError("External physics features do not match the row-level audit.")
    external_keys = [_key(row) for row in external_rows]
    if len(external_keys) != 64 or len(set(external_keys)) != 64:
        raise ValueError("External output keys are not exactly 64 unique records.")
    return arrays, external_rows, expected_targets.reshape(-1)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    denominator = float(np.sum((target - np.mean(target)) ** 2))
    if denominator <= 0:
        raise ValueError("External SoH targets have zero variance; R² is undefined.")
    return {
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": float(1.0 - np.sum(residual**2) / denominator),
    }


def _residual_statistics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    return {
        "mean_residual_pp": float(np.mean(residual)),
        "median_residual_pp": float(np.median(residual)),
        "residual_sample_sd_pp": float(np.std(residual, ddof=1)),
        "minimum_residual_pp": float(np.min(residual)),
        "maximum_residual_pp": float(np.max(residual)),
        "maximum_absolute_error_pp": float(np.max(np.abs(residual))),
    }


def _seed_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_NAMES:
        rows = [row for row in metric_rows if row["architecture"] == architecture]
        if len(rows) != len(SEEDS):
            raise ValueError(f"Expected three External runs for {architecture}.")
        result: dict[str, Any] = {
            "architecture": architecture,
            "seed_count": len(rows),
        }
        for name in ("MAE", "RMSE", "R2"):
            values = [float(row[name]) for row in rows]
            result[f"{name}_mean"] = statistics.mean(values)
            result[f"{name}_sample_sd"] = statistics.stdev(values)
        summary.append(result)
    return summary


def _write_checkpoint_plot(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    architecture: str,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    residual = prediction - target
    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(target, prediction, alpha=0.8)
    axes[0].plot([low, high], [low, high], linestyle="--", color="black")
    axes[0].set_title("Actual vs predicted")
    axes[0].set_xlabel("Reconstructed SoH (%)")
    axes[0].set_ylabel("Predicted SoH (%)")
    axes[1].scatter(target, residual, alpha=0.8)
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_title("Residual vs actual")
    axes[1].set_xlabel("Reconstructed SoH (%)")
    axes[1].set_ylabel("Prediction - reference (pp)")
    axes[2].hist(residual, bins=12)
    axes[2].axvline(0.0, linestyle="--", color="black")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("Prediction - reference (pp)")
    axes[2].set_ylabel("Observations")
    figure.suptitle(f"{architecture}, seed {seed}, Part 3 External Test")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_cell_trajectory_plot(
    path: Path,
    rows: list[dict[str, str]],
    target: np.ndarray,
    prediction: np.ndarray,
    architecture: str,
    seed: int,
) -> None:
    import matplotlib.pyplot as plt

    by_cell: dict[str, list[tuple[int, float, float]]] = {}
    for index, row in enumerate(rows):
        by_cell.setdefault(row["physical_cell_id"], []).append(
            (
                int(row["rpt_index"]),
                float(target[index]),
                float(prediction[index]),
            )
        )
    cells = sorted(by_cell)
    columns = 3
    row_count = math.ceil(len(cells) / columns)
    figure, axes = plt.subplots(
        row_count,
        columns,
        figsize=(5 * columns, 3.5 * row_count),
        squeeze=False,
    )
    for axis, cell in zip(axes.flat, cells):
        points = sorted(by_cell[cell])
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label="Reference",
        )
        axis.plot(
            [point[0] for point in points],
            [point[2] for point in points],
            marker="x",
            linestyle="--",
            label="Prediction",
        )
        axis.set_title(cell)
        axis.set_xlabel("RPT order")
        axis.set_ylabel("SoH (%)")
        axis.legend()
    for axis in list(axes.flat)[len(cells):]:
        axis.set_visible(False)
    figure.suptitle(f"{architecture}, seed {seed}, External cell trajectories")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_rmse_comparison_plot(
    path: Path,
    comparison_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    labels = [
        f"{row['architecture'].replace('BaFuse_', '').replace('_P5', '')}\n{row['seed']}"
        for row in comparison_rows
    ]
    positions = np.arange(len(comparison_rows))
    validation = [float(row["validation_RMSE"]) for row in comparison_rows]
    external = [float(row["external_RMSE"]) for row in comparison_rows]
    figure, axis = plt.subplots(figsize=(13, 5))
    axis.plot(positions, validation, marker="o", label="Validation")
    axis.plot(positions, external, marker="s", label="External")
    axis.set_xticks(positions, labels, rotation=35, ha="right")
    axis.set_ylabel("RMSE (SoH percentage points)")
    axis.set_title("Locked Validation vs Part 3 External RMSE")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_comparison_report(
    path: Path,
    comparison_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Locked Validation and External comparison",
        "",
        "Every row is one previously locked checkpoint. External results are",
        "reported for all 12 checkpoints; no checkpoint or architecture was",
        "removed or selected based on External performance.",
        "",
        "| Model | Seed | Validation MAE | Validation RMSE | Validation R² | External MAE | External RMSE | External R² | Δ MAE | Δ RMSE | Δ R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_rows:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | "
            f"{float(row['validation_MAE']):.4f} | {float(row['validation_RMSE']):.4f} | "
            f"{float(row['validation_R2']):.4f} | {float(row['external_MAE']):.4f} | "
            f"{float(row['external_RMSE']):.4f} | {float(row['external_R2']):.4f} | "
            f"{float(row['external_minus_validation_MAE']):+.4f} | "
            f"{float(row['external_minus_validation_RMSE']):+.4f} | "
            f"{float(row['external_minus_validation_R2']):+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Architecture-level External seed summary",
            "",
            "| Model | External MAE mean ± SD | External RMSE mean ± SD | External R² mean ± SD |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['architecture']} | {row['MAE_mean']:.4f} ± {row['MAE_sample_sd']:.4f} | "
            f"{row['RMSE_mean']:.4f} ± {row['RMSE_sample_sd']:.4f} | "
            f"{row['R2_mean']:.4f} ± {row['R2_sample_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "External seed summaries describe independent checkpoints, not an ensemble. "
            "Differences are descriptive of this one Part 3 cohort; they do not establish "
            "universal generalization.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cross_part_report(
    path: Path,
    labels: list[dict[str, str]],
    split_rows: list[dict[str, str]],
    dataset_dir: Path,
) -> None:
    cross_part_path = dataset_dir / "bafuse_p5_cross_part_audit.md"
    cross_part_text = cross_part_path.read_text(encoding="utf-8")
    if "Maccor 4200" not in cross_part_text or "Maccor 4000" not in cross_part_text:
        raise ValueError("Cross-Part protocol audit lacks the verified instrument comparison.")
    values_by_part: dict[str, list[float]] = {}
    cells_by_part: dict[str, set[str]] = {}
    rows_by_part: dict[str, int] = {}
    for row in labels:
        part = row["dataset_part"]
        values_by_part.setdefault(part, []).append(float(row["soh_percent"]))
        cells_by_part.setdefault(part, set()).add(row["physical_cell_id"])
        rows_by_part[part] = rows_by_part.get(part, 0) + 1
    lines = [
        "# Cross-Part evaluation context",
        "",
        "This is a post-hoc interpretation of the locked evaluation. External",
        "labels and scores were not used to select checkpoints, alter features,",
        "fit preprocessing, or tune any model.",
        "",
        "## Cohort and reconstructed target distributions",
        "",
        "| Part | Observations | Physical cells | SoH min / median / max (%) |",
        "|---:|---:|---:|---:|",
    ]
    for part in ("1", "2", "3"):
        values = values_by_part.get(part, [])
        if not values:
            continue
        lines.append(
            f"| {part} | {rows_by_part[part]} | {len(cells_by_part[part])} | "
            f"{min(values):.3f} / {float(np.median(values)):.3f} / {max(values):.3f} |"
        )
    external_count = sum(row["partition"] == "external_test" for row in split_rows)
    external_cells = {
        row["physical_cell_id"] for row in split_rows if row["partition"] == "external_test"
    }
    internal_cells = {
        row["physical_cell_id"]
        for row in split_rows
        if row["partition"] in {"train", "validation"}
    }
    lines.extend(
        [
            "",
            f"The frozen External cohort has {external_count} Part 3 RPT observations "
            f"from {len(external_cells)} physical cells. Train/Validation cover "
            f"{len(internal_cells)} cells from Parts 1–2; there is no cell overlap. "
            "These labels are reconstructed capacity-retention references, not official "
            "Oxford ground truth.",
            "",
            "## Protocol and measurement context",
            "",
            "- The reviewed dataset record describes Parts 1–2 cycling/RPT data from Maccor 4200 and Part 3 from Maccor 4000.",
            "- The documented Step 4 discharge protocol differs: Parts 1–2 use CC-CV discharge, while Part 3 uses CC discharge. Voltage endpoints are close in the frozen audit, but the protocol is not identical.",
            "- The P5 inputs are same-RPT Step 4 discharge voltage/current summaries and Step 10 charge voltage, resampled over normalized source-row progress. That progress coordinate is not time or capacity; changes in sampling/protocol can still alter the represented curve.",
            "- The P5 cross-Part audit reports Step 4 row-count medians of 6,416 (Part 1), 5,759 (Part 2), and 6,393 (Part 3); median native TestTime intervals are approximately 0.0166667 in all three. The native time unit is not asserted here.",
            "- Mean absolute discharge current is near 1.5 in all Parts, while voltage-shape summaries have different Part-level ranges. These are descriptive engineered features, not independent sensors or validated electrochemical states.",
            "- Parts 1–2 report a 24 °C chamber; the reviewed Part 3 README does not state its temperature. Temperature was excluded from P5.",
            "",
            "Part 3 therefore tests cross-Part transfer with potential instrument and",
            "protocol shift. Those differences are documented; attributing a score",
            "change to a particular cause remains a hypothesis, not a result of this",
            "evaluation.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _verify_post_inference_reproducibility(
    config: dict[str, Any],
    dataset_dir: Path,
    benchmark_dir: Path,
    evaluation_dir: Path,
    verification: dict[str, Any],
    external_inputs: dict[str, torch.Tensor],
    external_rows: list[dict[str, str]],
    target: np.ndarray,
    predictions: dict[tuple[str, int], np.ndarray],
) -> dict[str, Any]:
    tolerance = float(config["inference"]["reload_prediction_absolute_tolerance"])
    reload_results: list[dict[str, Any]] = []
    device = torch.device(config["inference"]["device"])
    selected_seed = 42
    for architecture in ARCHITECTURE_NAMES:
        checkpoint_rel = next(
            row["path"]
            for row in verification["checkpoint_records"]
            if row["architecture"] == architecture and row["seed"] == selected_seed
        )
        checkpoint = torch.load(
            benchmark_dir / Path(checkpoint_rel),
            map_location="cpu",
            weights_only=False,
        )
        model = build_model(architecture).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.inference_mode():
            reloaded = model(
                external_inputs["X_discharge"].to(device),
                external_inputs["X_charge"].to(device),
                external_inputs["X_physics"].to(device),
            ).cpu().numpy()
        expected = predictions[(architecture, selected_seed)].reshape(-1, 1)
        maximum_delta = float(np.max(np.abs(reloaded - expected)))
        if maximum_delta > tolerance:
            raise RuntimeError(
                f"Reload prediction mismatch for {architecture}: {maximum_delta}."
            )
        reload_results.append(
            {
                "architecture": architecture,
                "seed": selected_seed,
                "checkpoint_path": checkpoint_rel,
                "observations": len(target),
                "max_abs_prediction_delta": maximum_delta,
                "tolerance": tolerance,
                "passed": True,
            }
        )

    prediction_rows = _read_csv(evaluation_dir / "external_predictions.csv")
    independent_metrics: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_NAMES:
        for seed in SEEDS:
            rows = [
                row
                for row in prediction_rows
                if row["architecture"] == architecture and int(row["seed"]) == seed
            ]
            rows.sort(key=lambda row: int(row["prediction_row_index"]))
            if len(rows) != 64:
                raise ValueError(f"Saved prediction count mismatch for {architecture}/{seed}.")
            saved_target = np.asarray(
                [float(row["reference_soh_percent"]) for row in rows],
                dtype=np.float32,
            )
            saved_prediction = np.asarray(
                [float(row["predicted_soh_percent"]) for row in rows],
                dtype=np.float32,
            )
            if not np.array_equal(saved_target, target):
                raise ValueError("Saved External reference order changed.")
            independent_metrics.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **_metrics(saved_target, saved_prediction),
                }
            )
    recorded = {
        (row["architecture"], int(row["seed"])): row
        for row in _read_csv(evaluation_dir / "external_metrics_per_run.csv")
    }
    for row in independent_metrics:
        original = recorded[(row["architecture"], row["seed"])]
        for name in ("MAE", "RMSE", "R2"):
            if abs(float(original[name]) - float(row[name])) > 1e-10:
                raise ValueError(
                    f"Independent {name} does not match saved External predictions."
                )

    dataset_manifest = _read_json(dataset_dir / "bafuse_p5_sha256_manifest.json")
    dataset_outputs = _verify_hash_section(
        dataset_manifest,
        "outputs",
        local_output_directory=dataset_dir,
    )
    benchmark_manifest = _read_json(
        benchmark_dir / "benchmark_sha256_manifest.json"
    )
    benchmark_outputs = _verify_hash_section(
        benchmark_manifest,
        "outputs",
        local_output_directory=benchmark_dir,
    )
    if _sha256(dataset_dir / "bafuse_p5_sha256_manifest.json") != (
        EXPECTED_DATASET_MANIFEST_SHA256
    ):
        raise ValueError("Frozen dataset manifest changed during inference.")
    if _sha256(benchmark_dir / "benchmark_sha256_manifest.json") != (
        EXPECTED_BENCHMARK_MANIFEST_SHA256
    ):
        raise ValueError("Train/Validation benchmark manifest changed during inference.")
    return {
        "checkpoint_reload_checks": reload_results,
        "independent_metric_recalculation": {
            "all_12_runs_passed": True,
            "prediction_rows_per_checkpoint": 64,
            "prediction_rows_total": len(prediction_rows),
        },
        "frozen_dataset_outputs_unchanged": len(dataset_outputs) == 13,
        "train_validation_outputs_and_checkpoints_unchanged": len(benchmark_outputs)
        == 97,
        "external_predictions_reloaded_from_saved_csv": True,
    }


def _write_reproducibility_report(
    path: Path,
    environment: dict[str, Any],
    reproducibility: dict[str, Any],
    protocol_sha256: str,
) -> None:
    lines = [
        "# External evaluation reproducibility report",
        "",
        f"- Locked evaluation protocol SHA-256: `{protocol_sha256}`.",
        f"- Python: `{environment['python_version']}`.",
        f"- PyTorch: `{environment['pytorch_version']}`; CUDA runtime: `{environment['cuda_runtime_version']}`.",
        f"- Inference device: `{environment['device_name']}`.",
        f"- Deterministic algorithms: `{environment['deterministic_algorithms']}`; cuDNN deterministic: `{environment['cudnn_deterministic']}`.",
        "- Inference used `model.eval()` and `torch.inference_mode()`; no optimizer, backward pass, training-mode update, or preprocessing fit was used.",
        "- All 12 saved prediction rows per checkpoint were independently re-scored for MAE, RMSE, and R².",
        "- At least one checkpoint per architecture was reloaded and its predictions reproduced against the saved CSV.",
        "- Frozen dataset and Train/Validation benchmark output hashes were rechecked after inference.",
        "",
        "| Architecture | Reloaded seed | Max absolute prediction difference | Tolerance | Passed |",
        "|---|---:|---:|---:|---|",
    ]
    for row in reproducibility["checkpoint_reload_checks"]:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | "
            f"{row['max_abs_prediction_delta']:.9g} | {row['tolerance']:.1e} | "
            f"{row['passed']} |"
        )
    lines.extend(
        [
            "",
            "Exact bitwise agreement across different hardware, drivers, or library builds is not claimed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _all_output_hashes(evaluation_dir: Path) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in sorted(item for item in evaluation_dir.rglob("*") if item.is_file()):
        if path.name == "external_sha256_manifest.json":
            continue
        result[path.relative_to(evaluation_dir).as_posix()] = {
            "sha256": _sha256(path)
        }
    return result


def run_external_evaluation(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    evaluation_dir = config_path.parent
    expected_dir = _resolve(
        "experiments/soh_forecast/bafuse_architecture_benchmark_v1/"
        "bafuse_p5_external_evaluation_v1_20261114"
    )
    if evaluation_dir != expected_dir:
        raise ValueError("Evaluation config is outside the approved versioned directory.")
    existing_files = {
        path.relative_to(evaluation_dir).as_posix()
        for path in evaluation_dir.rglob("*")
        if path.is_file()
    }
    partial_files = {
        "external_dataset_verification.json",
        "external_checkpoint_verification.json",
        "external_environment.json",
        "external_metrics_per_run.csv",
        "external_metrics_model_summary.csv",
        "external_validation_comparison.csv",
        "external_predictions.csv",
        "external_residual_diagnostics.csv",
        "external_model_comparison.md",
        "external_cross_part_analysis.md",
    }
    for architecture in ARCHITECTURE_NAMES:
        for seed in SEEDS:
            stem = f"plots/{architecture}_seed_{seed}"
            partial_files.add(f"{stem}_diagnostics.png")
            partial_files.add(f"{stem}_cell_trajectories.png")
    partial_files.add("plots/validation_external_rmse.png")
    allowed_existing = {"external_evaluation_config.json"} | partial_files
    has_final_outputs = {
        "external_evaluation_status.json",
        "external_sha256_manifest.json",
    }.intersection(existing_files)
    if (
        "external_evaluation_config.json" not in existing_files
        or existing_files - allowed_existing
        or has_final_outputs
    ):
        raise FileExistsError(
            "Refusing to overwrite unexpected or finalized External evaluation artifacts."
        )
    protocol_sha256 = _sha256(config_path)
    dataset_dir = _resolve(config["dataset_directory"])
    benchmark_dir = _resolve(config["benchmark_directory"])

    dataset_verification = _verify_frozen_dataset(config, dataset_dir)
    benchmark_verification = _verify_benchmark(config, benchmark_dir)
    environment_device = torch.device(config["inference"]["device"])
    if environment_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"Approved inference device is unavailable: {environment_device}.")
    if environment_device.index not in (None, 0):
        raise RuntimeError("External evaluation is scoped to CUDA device 0.")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Deterministic CUDA workspace configuration is missing.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    environment = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": str(environment_device),
        "device_name": torch.cuda.get_device_name(environment_device),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    dataset_verification_record = {
            "dataset_identifier": config["dataset_identifier"],
            "dataset_manifest_sha256": dataset_verification["manifest_sha256"],
            "external_npz_sha256": dataset_verification["external_npz_sha256"],
            "verified_dataset_outputs": dataset_verification["verified_outputs"],
            "verified_dataset_inputs": dataset_verification["verified_inputs"],
            "verified_dataset_code": dataset_verification["verified_code"],
            "external_npz_hash_verified_before_loading": True,
            "external_npz_arrays_loaded": False,
            "protocol_sha256": protocol_sha256,
            "verification_status": "passed_before_external_inference",
        }
    _write_json(
        evaluation_dir / "external_dataset_verification.json",
        dataset_verification_record,
    )
    _write_json(
        evaluation_dir / "external_checkpoint_verification.json",
        {
            "benchmark_identifier": config["benchmark_identifier"],
            "benchmark_manifest_sha256": benchmark_verification["manifest_sha256"],
            "verified_benchmark_outputs": benchmark_verification["verified_outputs"],
            "verified_benchmark_inputs": benchmark_verification["verified_inputs"],
            "verified_benchmark_code": benchmark_verification["verified_code"],
            "checkpoint_records": benchmark_verification["checkpoint_records"],
            "validation_selection_checks": benchmark_verification[
                "validation_selection_checks"
            ],
            "checkpoint_verification_status": "passed_before_external_inference",
            "protocol_sha256": protocol_sha256,
        },
    )

    split_rows = _read_csv(dataset_dir / "bafuse_p5_split_manifest.csv")
    label_rows = _read_csv(dataset_dir / "bafuse_soh_labels_frozen.csv")
    feature_rows = _read_csv(dataset_dir / "bafuse_p5_feature_audit.csv")
    arrays, external_rows, target = _validate_external_arrays(
        config, dataset_dir, split_rows, label_rows, feature_rows
    )
    dataset_verification_record.update(
        {
            "external_npz_arrays_loaded": True,
            "external_npz_shapes_and_dtypes_validated": True,
            "external_observation_count": len(external_rows),
            "external_unique_physical_cells": len(
                {row["physical_cell_id"] for row in external_rows}
            ),
            "external_keys_unique": True,
            "external_cells_disjoint_from_train_validation": True,
            "external_targets_aligned_to_frozen_split": True,
            "external_physics_aligned_to_row_audit": True,
            "verification_status": "passed",
        }
    )
    _write_json(
        evaluation_dir / "external_dataset_verification.json",
        dataset_verification_record,
    )
    environment["numpy_version"] = np.__version__
    environment["inference_started_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_json(evaluation_dir / "external_environment.json", environment)
    external_inputs = {
        name: torch.from_numpy(arrays[name])
        for name in ("X_discharge", "X_charge", "X_physics")
    }
    device = environment_device
    per_run_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    predictions: dict[tuple[str, int], np.ndarray] = {}
    validation_by_key = {
        (row["architecture"], int(row["seed"])): row
        for row in benchmark_verification["run_rows"]
    }
    keys_seen: set[tuple[str, int, tuple[str, str, str]]] = set()
    for architecture in ARCHITECTURE_NAMES:
        for seed in SEEDS:
            verification_record = next(
                row
                for row in benchmark_verification["checkpoint_records"]
                if row["architecture"] == architecture and row["seed"] == seed
            )
            checkpoint_path = benchmark_dir / Path(verification_record["path"])
            if _sha256(checkpoint_path) != verification_record["sha256"]:
                raise ValueError(
                    f"Checkpoint changed after preflight: {verification_record['path']}."
                )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            model = build_model(architecture).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            with torch.inference_mode():
                prediction_tensor = model(
                    external_inputs["X_discharge"].to(device),
                    external_inputs["X_charge"].to(device),
                    external_inputs["X_physics"].to(device),
                )
            if prediction_tensor.shape != (64, 1):
                raise ValueError(
                    f"{architecture}/{seed} returned shape {tuple(prediction_tensor.shape)}."
                )
            prediction = prediction_tensor.detach().cpu().numpy().reshape(-1)
            if not np.isfinite(prediction).all():
                raise ValueError(f"{architecture}/{seed} returned non-finite predictions.")
            predictions[(architecture, seed)] = prediction
            metrics = _metrics(target, prediction)
            residual_stats = _residual_statistics(target, prediction)
            run_row = {
                "architecture": architecture,
                "seed": seed,
                "parameter_count": verification_record["parameter_count"],
                "best_epoch": verification_record["best_epoch"],
                "validation_MAE": float(
                    validation_by_key[(architecture, seed)]["validation_MAE"]
                ),
                "validation_RMSE": float(
                    validation_by_key[(architecture, seed)]["validation_RMSE"]
                ),
                "validation_R2": float(
                    validation_by_key[(architecture, seed)]["validation_R2"]
                ),
                **metrics,
                "prediction_count": len(prediction),
                "checkpoint_path": verification_record["path"],
                "checkpoint_sha256": verification_record["sha256"],
            }
            per_run_rows.append(run_row)
            comparison_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "validation_MAE": run_row["validation_MAE"],
                    "validation_RMSE": run_row["validation_RMSE"],
                    "validation_R2": run_row["validation_R2"],
                    "external_MAE": metrics["MAE"],
                    "external_RMSE": metrics["RMSE"],
                    "external_R2": metrics["R2"],
                    "external_minus_validation_MAE": (
                        metrics["MAE"] - run_row["validation_MAE"]
                    ),
                    "external_minus_validation_RMSE": (
                        metrics["RMSE"] - run_row["validation_RMSE"]
                    ),
                    "external_minus_validation_R2": (
                        metrics["R2"] - run_row["validation_R2"]
                    ),
                }
            )
            residual_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **residual_stats,
                    "observation_count": len(target),
                    "residual_definition": "prediction - reference_soh",
                }
            )
            for index, row in enumerate(external_rows):
                observation_key = _key(row)
                combination = (architecture, seed, observation_key)
                if combination in keys_seen:
                    raise ValueError(f"Duplicate prediction identity: {combination}.")
                keys_seen.add(combination)
                prediction_rows.append(
                    {
                        "prediction_row_index": index,
                        "architecture": architecture,
                        "seed": seed,
                        "dataset_part": row["dataset_part"],
                        "physical_cell_id": row["physical_cell_id"],
                        "rpt_index": row["rpt_index"],
                        "partition": row["partition"],
                        "reference_soh_percent": float(target[index]),
                        "predicted_soh_percent": float(prediction[index]),
                        "residual_prediction_minus_reference_pp": float(
                            prediction[index] - target[index]
                        ),
                    }
                )
            del model, checkpoint, prediction_tensor

    if len(prediction_rows) != 768 or len(keys_seen) != 768:
        raise ValueError("Expected exactly 768 unique model/seed/observation predictions.")
    if any(row["partition"] != "external_test" for row in prediction_rows):
        raise ValueError("Non-External rows entered the External prediction artifact.")

    summary_rows = _seed_summary(per_run_rows)
    _write_csv(evaluation_dir / "external_metrics_per_run.csv", per_run_rows)
    _write_csv(evaluation_dir / "external_metrics_model_summary.csv", summary_rows)
    _write_csv(evaluation_dir / "external_validation_comparison.csv", comparison_rows)
    _write_csv(evaluation_dir / "external_predictions.csv", prediction_rows)
    _write_csv(evaluation_dir / "external_residual_diagnostics.csv", residual_rows)
    plots_dir = evaluation_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for architecture in ARCHITECTURE_NAMES:
        for seed in SEEDS:
            prediction = predictions[(architecture, seed)]
            stem = f"{architecture}_seed_{seed}"
            _write_checkpoint_plot(
                plots_dir / f"{stem}_diagnostics.png",
                target,
                prediction,
                architecture,
                seed,
            )
            _write_cell_trajectory_plot(
                plots_dir / f"{stem}_cell_trajectories.png",
                external_rows,
                target,
                prediction,
                architecture,
                seed,
            )
    _write_rmse_comparison_plot(
        plots_dir / "validation_external_rmse.png", comparison_rows
    )
    _write_comparison_report(
        evaluation_dir / "external_model_comparison.md",
        comparison_rows,
        summary_rows,
    )
    _write_cross_part_report(
        evaluation_dir / "external_cross_part_analysis.md",
        label_rows,
        split_rows,
        dataset_dir,
    )

    reproducibility = _verify_post_inference_reproducibility(
        config,
        dataset_dir,
        benchmark_dir,
        evaluation_dir,
        benchmark_verification,
        external_inputs,
        external_rows,
        target,
        predictions,
    )
    _write_reproducibility_report(
        evaluation_dir / "external_reproducibility_report.md",
        environment,
        reproducibility,
        protocol_sha256,
    )
    _write_json(
        evaluation_dir / "external_evaluation_status.json",
        {
            "status": "complete",
            "evaluation_identifier": config["evaluation_identifier"],
            "protocol_sha256": protocol_sha256,
            "external_metrics_authorized": True,
            "training_performed": False,
            "fine_tuning_performed": False,
            "external_model_selection_performed": False,
            "prediction_rows": len(prediction_rows),
            "model_seed_runs": len(per_run_rows),
            "reproducibility_checks_passed": True,
        },
    )
    report_lines = [
        "# BaFuse P5 Part 3 External Test evaluation",
        "",
        f"- Evaluation: `{config['evaluation_identifier']}`.",
        f"- Dataset manifest SHA-256: `{dataset_verification['manifest_sha256']}`.",
        f"- Train/Validation benchmark manifest SHA-256: `{benchmark_verification['manifest_sha256']}`.",
        f"- External NPZ SHA-256: `{dataset_verification['external_npz_sha256']}`.",
        "- Protocol was recorded before inference. All 12 locked checkpoints were evaluated; no training, tuning, calibration, clipping, or model selection from External results occurred.",
        "- Target is reconstructed same-cell Step 4 discharge capacity retention, not official Oxford ground truth.",
        "- External cohort: 64 observations from 11 Part 3 physical cells, with no Train/Validation physical-cell overlap.",
        "",
        "## External metrics by checkpoint",
        "",
        "| Model | Seed | MAE (pp) | RMSE (pp) | R² | Mean residual (pp) | Median residual (pp) | Residual SD (pp) | Min / max residual (pp) | Max |error| (pp) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    residual_by_run = {
        (row["architecture"], int(row["seed"])): row for row in residual_rows
    }
    for row in per_run_rows:
        residual = residual_by_run[(row["architecture"], int(row["seed"]))]
        report_lines.append(
            f"| {row['architecture']} | {row['seed']} | {row['MAE']:.4f} | "
            f"{row['RMSE']:.4f} | {row['R2']:.4f} | "
            f"{residual['mean_residual_pp']:+.4f} | "
            f"{residual['median_residual_pp']:+.4f} | "
            f"{residual['residual_sample_sd_pp']:.4f} | "
            f"{residual['minimum_residual_pp']:+.4f} / "
            f"{residual['maximum_residual_pp']:+.4f} | "
            f"{residual['maximum_absolute_error_pp']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Architecture-level seed summaries",
            "",
            "| Model | External MAE mean ± SD (pp) | External RMSE mean ± SD (pp) | External R² mean ± SD |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        report_lines.append(
            f"| {row['architecture']} | {row['MAE_mean']:.4f} ± "
            f"{row['MAE_sample_sd']:.4f} | {row['RMSE_mean']:.4f} ± "
            f"{row['RMSE_sample_sd']:.4f} | {row['R2_mean']:.4f} ± "
            f"{row['R2_sample_sd']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "## Validation-to-External findings",
            "",
        ]
    )
    best_external = min(per_run_rows, key=lambda row: float(row["RMSE"]))
    best_external_arch = min(summary_rows, key=lambda row: float(row["RMSE_mean"]))
    val_selected = benchmark_verification["validation_selection_checks"]
    tcn_external = next(
        row for row in summary_rows if row["architecture"] == "BaFuse_TCN_P5"
    )
    cnn123_external = next(
        row
        for row in per_run_rows
        if row["architecture"] == "BaFuse_CNN1D_P5" and row["seed"] == 123
    )
    report_lines.extend(
        [
            f"- Best checkpoint on External by RMSE (descriptive only): `{best_external['architecture']}`, seed {best_external['seed']}, RMSE {best_external['RMSE']:.4f} pp. This does not change model selection.",
            f"- The preselected Validation best individual remains `{val_selected['best_individual_checkpoint']['architecture']}`, seed {val_selected['best_individual_checkpoint']['seed']}; its External MAE/RMSE/R² are {cnn123_external['MAE']:.4f}/{cnn123_external['RMSE']:.4f}/{cnn123_external['R2']:.4f}.",
            f"- The preselected TCN checkpoint remains seed {val_selected['best_TCN_checkpoint_by_validation_RMSE']['seed']}. TCN External architecture mean RMSE is {tcn_external['RMSE_mean']:.4f} ± {tcn_external['RMSE_sample_sd']:.4f} pp; best External architecture mean RMSE is `{best_external_arch['architecture']}` at {best_external_arch['RMSE_mean']:.4f} pp. No External ranking was used to retune or choose a deployment model.",
            (
                "- TCN retains the lowest architecture-level mean RMSE on External."
                if best_external_arch["architecture"] == "BaFuse_TCN_P5"
                else f"- TCN does not have the lowest architecture-level mean RMSE on External; `{best_external_arch['architecture']}` is lower. This is reported without changing the Validation-based selection."
            ),
            f"- External results for all seeds are reported; seed-level variation is summarized by the sample SD in the table. These values are not an ensemble.",
            "- Positive mean residual denotes overprediction; negative mean residual denotes underprediction. Per-checkpoint bias and residual spread are listed in the residual table and CSV.",
            (
                lambda largest: (
                    f"- Largest absolute architecture-level mean Validation-to-External RMSE change: `{largest['architecture']}` "
                    f"({largest['mean_delta']:+.4f} pp)."
                )
            )(
                max(
                    (
                        {
                            "architecture": architecture,
                            "mean_delta": statistics.mean(
                                float(row["external_minus_validation_RMSE"])
                                for row in comparison_rows
                                if row["architecture"] == architecture
                            ),
                        }
                        for architecture in ARCHITECTURE_NAMES
                    ),
                    key=lambda item: abs(item["mean_delta"]),
                )
            ),
            "- See `external_validation_comparison.csv` and `external_model_comparison.md` for every checkpoint’s Validation metrics, External metrics, and External-minus-Validation differences.",
            "",
            "## Interpretation limits",
            "",
            "- This is one Part 3 held-out dataset-part evaluation, with 64 observations but only 11 physical cells. Observations within a cell are not independent; the requested metrics are observation-level and do not estimate uncertainty across a large population of cells.",
            "- The Train/Validation cohorts are from Parts 1–2 and Part 3 is from another Maccor instrument and documented Step 4 discharge protocol. The frozen audit reports similar median native TestTime intervals and voltage endpoints, but curve row counts and voltage-shape summaries differ. These metadata support possible domain shift; they do not establish the cause of any score change.",
            "- Part 3 README temperature is unspecified; P5 excludes temperature. The target remains a reconstructed capacity ratio, not official ground truth.",
            "- Prediction plots and per-cell trajectories are diagnostics only. No per-cell metric aggregation, calibration, clipping, or model change was applied.",
            "",
            "## Reproducibility",
            "",
            f"- Checkpoint reload checks passed for all four architectures; independent metric recalculation and frozen-file hash rechecks are recorded in `external_reproducibility_report.md`.",
            "- External output manifest SHA-256 is recorded in the session summary after manifest generation.",
            "- Training, fine-tuning, hyperparameter tuning, PyDMA, Half-Cell fitting, and dataset modification were not performed.",
            "",
            "## Artifacts",
            "",
            "- `external_metrics_per_run.csv`, `external_metrics_model_summary.csv`, `external_validation_comparison.csv`, `external_predictions.csv`, and `external_residual_diagnostics.csv` contain the row-level evaluation results.",
            "- `plots/` contains per-checkpoint actual-vs-predicted/residual diagnostics, per-cell External trajectories, and Validation-vs-External RMSE.",
            "",
        ]
    )
    (evaluation_dir / "external_evaluation_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    final_dataset_manifest = _read_json(dataset_verification["manifest_path"])
    final_dataset_outputs = _verify_hash_section(
        final_dataset_manifest, "outputs", local_output_directory=dataset_dir
    )
    final_benchmark_manifest = _read_json(benchmark_verification["manifest_path"])
    final_benchmark_outputs = _verify_hash_section(
        final_benchmark_manifest, "outputs", local_output_directory=benchmark_dir
    )
    if len(final_dataset_outputs) != 13 or len(final_benchmark_outputs) != 97:
        raise ValueError("Post-inference frozen artifact verification is incomplete.")

    manifest = {
        "schema_version": 1,
        "evaluation_identifier": config["evaluation_identifier"],
        "protocol_sha256": protocol_sha256,
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": dataset_verification["manifest_sha256"],
        "benchmark_identifier": config["benchmark_identifier"],
        "benchmark_manifest_sha256": benchmark_verification["manifest_sha256"],
        "external_prediction_evaluation_authorized": True,
        "training_performed": False,
        "external_model_selection_performed": False,
        "code": {
            "evaluation_script": {
                "path": Path(__file__).resolve().relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(Path(__file__).resolve()),
            },
            "model_definitions": {
                "path": Path(__file__)
                .with_name("p5_benchmark_models.py")
                .resolve()
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "sha256": _sha256(
                    Path(__file__).with_name("p5_benchmark_models.py")
                ),
            },
        },
        "inputs": {
            "evaluation_config": {
                "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": protocol_sha256,
            },
            "frozen_dataset_manifest": {
                "path": dataset_verification["manifest_path"]
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "sha256": dataset_verification["manifest_sha256"],
            },
            "trainval_benchmark_manifest": {
                "path": benchmark_verification["manifest_path"]
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "sha256": benchmark_verification["manifest_sha256"],
            },
        },
        "outputs": _all_output_hashes(evaluation_dir),
        "manifest_self_hash": "excluded; calculate after writing this file",
    }
    manifest_path = evaluation_dir / "external_sha256_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "evaluation_identifier": config["evaluation_identifier"],
        "output_directory": str(evaluation_dir),
        "external_runs": len(per_run_rows),
        "external_predictions": len(prediction_rows),
        "best_external_rmse_run_descriptive_only": {
            "architecture": best_external["architecture"],
            "seed": best_external["seed"],
            "RMSE": best_external["RMSE"],
        },
        "manifest_sha256": _sha256(manifest_path),
        "external_model_selection_performed": False,
        "training_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the 12 locked BaFuse P5 checkpoints on Part 3 External only; "
            "does not train or select models."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    arguments = parser.parse_args(argv)
    print(json.dumps(run_external_evaluation(arguments.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
