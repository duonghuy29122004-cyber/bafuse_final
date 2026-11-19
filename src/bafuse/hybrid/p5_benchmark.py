from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bafuse.hybrid.p5_benchmark_models import (
    ARCHITECTURE_NAMES,
    architecture_config,
    build_model,
    trainable_parameter_count,
)
from bafuse.hybrid.training_authorization import (
    build_training_scope,
    require_training_authorization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_PARENT = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
)
DEFAULT_CONFIG = (
    BENCHMARK_PARENT / "p5_training_config_v1_20261114.json"
)
P5_DATASET_ID = "bafuse-p5-electrical-v1-20261114"
EXPECTED_P5_MANIFEST_SHA256 = (
    "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
)
EXPECTED_PARTITION_COUNTS = {
    "train": 102,
    "validation": 37,
    "external_test": 64,
}
P5_PHYSICS_FEATURE_NAMES = (
    "mean_abs_discharge_current",
    "discharge_current_variability",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)
P5_AUDIT_FEATURE_COLUMNS = (
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


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty benchmark artifact: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset_part"], row["physical_cell_id"], row["rpt_index"]


def _verify_manifest(dataset_dir: Path, manifest_path: Path) -> dict[str, Any]:
    observed_manifest_hash = _sha256(manifest_path)
    if observed_manifest_hash != EXPECTED_P5_MANIFEST_SHA256:
        raise ValueError(
            "Frozen P5 manifest hash mismatch: "
            f"expected {EXPECTED_P5_MANIFEST_SHA256}, got {observed_manifest_hash}."
        )
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_identifier") != P5_DATASET_ID:
        raise ValueError("Frozen P5 dataset identifier does not match the approved dataset.")

    verified: dict[str, list[str]] = {"outputs": [], "inputs": [], "code": []}
    for section in ("outputs", "inputs", "code"):
        entries = manifest.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f"Frozen P5 manifest lacks the {section} mapping.")
        for name, entry in entries.items():
            path_value = entry.get("path", name)
            path = _resolve(path_value)
            if section == "outputs":
                path = dataset_dir / name
            if not path.is_file():
                raise FileNotFoundError(f"Frozen P5 {section} file is missing: {path}.")
            observed = _sha256(path)
            expected = entry.get("sha256")
            if observed != expected:
                raise ValueError(
                    f"Frozen P5 {section} hash mismatch for {name}: "
                    f"expected {expected}, got {observed}."
                )
            verified[section].append(name)
    return {
        "manifest": manifest,
        "manifest_sha256": observed_manifest_hash,
        "verified_hashes": verified,
    }


def _validate_config(config: dict[str, Any]) -> None:
    required_flags = {
        "training_authorized": True,
        "train_evaluation_authorized": True,
        "validation_evaluation_authorized": True,
        "checkpoint_selection_authorized": True,
        "external_prediction_evaluation_authorized": False,
        "dataset_modification_authorized": False,
        "feature_schema_modification_authorized": False,
        "label_modification_authorized": False,
        "pydma_fitting_authorized": False,
        "half_cell_fitting_authorized": False,
    }
    for name, expected_value in required_flags.items():
        if config.get(name) is not expected_value:
            raise ValueError(
                f"Training authorization flag {name} must be {expected_value}."
            )
    if config.get("architectures") != list(ARCHITECTURE_NAMES):
        raise ValueError("Training config does not match the four P5 architectures.")
    training = config.get("training")
    if not isinstance(training, dict):
        raise ValueError("Training configuration is missing.")
    expected = {
        "seeds": [42, 123, 2026],
        "max_epochs": 300,
        "batch_size": 16,
        "optimizer": "AdamW",
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "loss": "MSELoss",
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.0,
        "checkpoint_metric": "validation_RMSE",
        "target_transform": None,
    }
    if training != expected:
        raise ValueError(f"Training configuration differs from the approved setup: {training}.")
    if config.get("dataset_identifier") != P5_DATASET_ID:
        raise ValueError("Training config refers to a different P5 dataset.")
    if config.get("dataset_manifest_sha256") != EXPECTED_P5_MANIFEST_SHA256:
        raise ValueError("Training config has an unexpected frozen dataset manifest hash.")


def _read_internal_inputs(
    dataset_dir: Path,
    manifest: dict[str, Any],
    dataset_verification: dict[str, Any],
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    list[dict[str, str]],
    list[dict[str, str]],
]:
    label_path = dataset_dir / "bafuse_soh_labels_frozen.csv"
    split_path = dataset_dir / "bafuse_p5_split_manifest.csv"
    feature_audit_path = dataset_dir / "bafuse_p5_feature_audit.csv"
    schema_path = dataset_dir / "bafuse_p5_feature_schema.json"
    preprocess_path = dataset_dir / "bafuse_p5_preprocessing_config.json"
    labels = _read_csv(label_path)
    split_rows = _read_csv(split_path)
    feature_audit = _read_csv(feature_audit_path)
    schema = _read_json(schema_path)
    preprocessing = _read_json(preprocess_path)

    if len(labels) != 203 or len(split_rows) != 203 or len(feature_audit) != 203:
        raise ValueError("Frozen labels and row manifests must each contain 203 records.")
    label_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in labels:
        key = _key(row)
        if key in label_by_key:
            raise ValueError(f"Duplicate frozen label key: {key}.")
        if row["final_status"] != "ACCEPTED":
            raise ValueError(f"Non-accepted frozen label found: {key}.")
        label_by_key[key] = row
    split_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    cell_partition: dict[str, str] = {}
    for row in split_rows:
        key = _key(row)
        if key in split_by_key:
            raise ValueError(f"Duplicate frozen split key: {key}.")
        partition = row["partition"]
        if partition not in EXPECTED_PARTITION_COUNTS:
            raise ValueError(f"Unexpected frozen partition {partition!r} for {key}.")
        previous = cell_partition.setdefault(row["physical_cell_id"], partition)
        if previous != partition:
            raise ValueError(
                f"Physical cell {row['physical_cell_id']} overlaps partitions."
            )
        if row["label_status"] != "ACCEPTED":
            raise ValueError(f"Split row is not ACCEPTED: {key}.")
        split_by_key[key] = row
    if set(label_by_key) != set(split_by_key):
        raise ValueError("Frozen label and split observation keys do not match.")
    actual_counts = {
        part: sum(row["partition"] == part for row in split_rows)
        for part in EXPECTED_PARTITION_COUNTS
    }
    if actual_counts != EXPECTED_PARTITION_COUNTS:
        raise ValueError(f"Frozen split counts changed: {actual_counts}.")
    for partition in EXPECTED_PARTITION_COUNTS:
        indices = sorted(
            int(row["split_row_index"])
            for row in split_rows
            if row["partition"] == partition
        )
        if indices != list(range(EXPECTED_PARTITION_COUNTS[partition])):
            raise ValueError(f"Frozen {partition} row indices are not contiguous.")
    if len(cell_partition) != 23:
        raise ValueError(
            f"Expected 23 labeled physical cells, found {len(cell_partition)}."
        )
    for key, label in label_by_key.items():
        if label["partition"] != split_by_key[key]["partition"]:
            raise ValueError(f"Label and split partition differ for {key}.")

    if schema.get("dataset_identifier") != P5_DATASET_ID:
        raise ValueError("Feature schema is not for the approved P5 dataset.")
    if schema.get("label", {}).get("definition") != (
        "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)"
    ):
        raise ValueError("Frozen label definition changed.")
    modalities = schema.get("modalities", {})
    for modality in ("X_discharge", "X_charge"):
        definition = modalities.get(modality, {})
        if definition.get("capacity_or_time_axis_used") is not False:
            raise ValueError(f"{modality} may encode target capacity.")
        if definition.get("dtype") != "float32" or definition.get(
            "shape_per_split"
        ) != ["N", 128]:
            raise ValueError(f"Frozen {modality} schema changed.")
    if modalities.get("X_physics", {}).get("dtype") != "float32" or modalities.get(
        "X_physics", {}
    ).get("shape_per_split") != ["N", 5]:
        raise ValueError("Frozen physics tensor schema changed.")
    if modalities.get("y_soh", {}).get("dtype") != "float32" or modalities.get(
        "y_soh", {}
    ).get("shape_per_split") != ["N", 1]:
        raise ValueError("Frozen SoH target schema changed.")
    physics_schema = schema.get("modalities", {}).get("X_physics", {})
    if physics_schema.get("target_or_bol_derived") is not False:
        raise ValueError("Physics features are marked as target- or BoL-derived.")
    if physics_schema.get("temperature_included") is not False:
        raise ValueError("Temperature must remain excluded from the P5 inputs.")
    if tuple(physics_schema.get("feature_names_in_order", [])) != (
        P5_PHYSICS_FEATURE_NAMES
    ):
        raise ValueError("Frozen P5 physics feature order changed.")
    if preprocessing.get("fit_partition") != "train":
        raise ValueError("Frozen P5 preprocessing was not fitted on Train only.")
    if preprocessing.get("fit_row_count") != EXPECTED_PARTITION_COUNTS["train"]:
        raise ValueError("Frozen P5 preprocessing fit count changed.")
    if preprocessing.get("validation_and_external_fit_or_update") is not False:
        raise ValueError("Frozen P5 preprocessing was updated outside Train.")

    audit_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in feature_audit:
        key = _key(row)
        if key in audit_by_key:
            raise ValueError(f"Duplicate P5 feature audit key: {key}.")
        if key not in split_by_key:
            raise ValueError(f"Unexpected P5 feature audit key: {key}.")
        if row["partition"] != split_by_key[key]["partition"]:
            raise ValueError(f"P5 feature audit partition differs for {key}.")
        if row["partition"] in {"train", "validation"}:
            if row["curve_progress_basis"] != "normalized source-row index":
                raise ValueError(f"Unexpected curve progress basis for {key}.")
            if row["predictive_label_or_bol_feature"] != "False":
                raise ValueError(f"Target/BoL feature leakage is flagged for {key}.")
            if row["future_rpt_feature"] != "False":
                raise ValueError(f"Future-RPT leakage is flagged for {key}.")
            if row["predictive_temperature_feature"] != "False":
                raise ValueError(f"Temperature entered the predictors for {key}.")
            if row["same_rpt_step_order_valid"] != "True":
                raise ValueError(f"Same-RPT modality mapping is invalid for {key}.")
        audit_by_key[key] = row
    if set(audit_by_key) != set(split_by_key):
        raise ValueError("P5 feature audit does not cover the frozen split keys.")

    loaded: dict[str, dict[str, np.ndarray]] = {}
    for partition, filename in (
        ("train", "bafuse_train_p5.npz"),
        ("validation", "bafuse_validation_p5.npz"),
    ):
        with np.load(dataset_dir / filename, allow_pickle=False) as archive:
            expected_members = {"X_discharge", "X_charge", "X_physics", "y_soh"}
            if set(archive.files) != expected_members:
                raise ValueError(f"Unexpected NPZ members in {filename}.")
            arrays = {name: archive[name].copy() for name in archive.files}
        count = EXPECTED_PARTITION_COUNTS[partition]
        expected_shapes = {
            "X_discharge": (count, 128),
            "X_charge": (count, 128),
            "X_physics": (count, 5),
            "y_soh": (count, 1),
        }
        for name, array in arrays.items():
            if array.dtype != np.float32 or array.shape != expected_shapes[name]:
                raise ValueError(
                    f"{filename}:{name} has unexpected dtype/shape "
                    f"{array.dtype}/{array.shape}."
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{filename}:{name} contains non-finite values.")

        ordered_rows = sorted(
            (row for row in split_rows if row["partition"] == partition),
            key=lambda row: int(row["split_row_index"]),
        )
        expected_targets = np.asarray(
            [float(label_by_key[_key(row)]["soh_percent"]) for row in ordered_rows],
            dtype=np.float32,
        ).reshape(-1, 1)
        if not np.array_equal(arrays["y_soh"], expected_targets):
            raise ValueError(f"{filename} target rows do not align with frozen labels.")
        expected_physics = np.asarray(
            [
                [float(audit_by_key[_key(row)][name]) for name in P5_AUDIT_FEATURE_COLUMNS]
                for row in ordered_rows
            ],
            dtype=np.float32,
        )
        if not np.allclose(
            arrays["X_physics"], expected_physics, rtol=0.0, atol=1e-6
        ):
            raise ValueError(f"{filename} physics features do not align with the audit.")
        loaded[partition] = arrays

    source_label_path = _resolve(
        manifest["inputs"]["authoritative_labels"]["path"]
    )
    if label_path.read_bytes() != source_label_path.read_bytes():
        raise ValueError("Frozen labels differ from the approved authoritative copy.")

    train_arrays = loaded["train"]
    validation_arrays = loaded["validation"]
    train_x = {
        name: torch.from_numpy(train_arrays[name])
        for name in ("X_discharge", "X_charge", "X_physics")
    }
    validation_x = {
        name: torch.from_numpy(validation_arrays[name])
        for name in ("X_discharge", "X_charge", "X_physics")
    }
    train_y = torch.from_numpy(train_arrays["y_soh"].reshape(-1))
    validation_y = torch.from_numpy(validation_arrays["y_soh"].reshape(-1))
    train_rows = sorted(
        (row for row in split_rows if row["partition"] == "train"),
        key=lambda row: int(row["split_row_index"]),
    )
    validation_rows = sorted(
        (row for row in split_rows if row["partition"] == "validation"),
        key=lambda row: int(row["split_row_index"]),
    )
    dataset_verification.update(
        {
            "label_count": len(labels),
            "unique_observation_keys": len(label_by_key),
            "physical_cell_count": len(cell_partition),
            "partition_counts": actual_counts,
            "partition_cell_counts": {
                part: len(
                    {
                        row["physical_cell_id"]
                        for row in split_rows
                        if row["partition"] == part
                    }
                )
                for part in EXPECTED_PARTITION_COUNTS
            },
            "physical_cell_overlap": False,
            "all_labels_accepted": True,
            "train_validation_targets_match_label_table": True,
            "train_validation_physics_match_row_audit": True,
            "external_npz_loaded": False,
            "external_targets_read_for_selection": False,
            "frozen_labels_byte_match_authoritative_source": True,
        }
    )
    return (
        train_x,
        train_y,
        validation_x,
        validation_y,
        train_rows,
        validation_rows,
    )


def _configure_determinism() -> dict[str, Any]:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": workspace,
        "dataloader_num_workers": 0,
        "dataloader_worker_seed_policy": (
            "No worker processes are used; each DataLoader has a seed-specific generator."
        ),
    }


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _metric_values(target: torch.Tensor, prediction: torch.Tensor) -> dict[str, float | None]:
    error = prediction - target
    mae = torch.mean(torch.abs(error)).item()
    rmse = torch.sqrt(torch.mean(error.square())).item()
    denominator = torch.sum((target - torch.mean(target)).square()).item()
    r2 = (
        1.0 - torch.sum(error.square()).item() / denominator
        if denominator > 0
        else None
    )
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def _predict(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        prediction = model(
            inputs["X_discharge"].to(device),
            inputs["X_charge"].to(device),
            inputs["X_physics"].to(device),
        ).squeeze(1)
    if not torch.isfinite(prediction).all():
        raise ValueError("Model produced a non-finite prediction.")
    return prediction.detach().cpu()


def _optimizer_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            copied[key] = value.detach().cpu().clone()
        elif isinstance(value, dict):
            copied[key] = _optimizer_state_to_cpu(value)
        elif isinstance(value, list):
            copied[key] = [
                item.detach().cpu().clone()
                if isinstance(item, torch.Tensor)
                else item
                for item in value
            ]
        else:
            copied[key] = value
    return copied


def _atomic_torch_save(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_learning_curve(history: list[dict[str, Any]], base: Path, stem: str) -> None:
    _write_csv(base / "training_curves" / f"{stem}_history.csv", history)
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_mse_loss"]) for row in history])
    axes[0].set_title("Training MSE loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (SoH percentage points squared)")
    axes[1].plot(epochs, [float(row["validation_RMSE"]) for row in history])
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (SoH percentage points)")
    figure.tight_layout()
    figure.savefig(base / "training_curves" / f"{stem}_learning_curve.png", dpi=150)
    plt.close(figure)


def _write_prediction_plot(
    path: Path,
    target: np.ndarray,
    prediction: np.ndarray,
    architecture: str,
    seed: int,
    partition: str,
) -> None:
    import matplotlib.pyplot as plt

    residual = prediction - target
    low = float(min(np.min(target), np.min(prediction)))
    high = float(max(np.max(target), np.max(prediction)))
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].scatter(target, prediction, alpha=0.8)
    axes[0].plot([low, high], [low, high], linestyle="--", color="black")
    axes[0].set_title("Actual vs predicted")
    axes[0].set_xlabel("Actual SoH (%)")
    axes[0].set_ylabel("Predicted SoH (%)")
    axes[1].scatter(target, residual, alpha=0.8)
    axes[1].axhline(0.0, linestyle="--", color="black")
    axes[1].set_title("Residual vs actual")
    axes[1].set_xlabel("Actual SoH (%)")
    axes[1].set_ylabel("Prediction - actual (pp)")
    axes[2].hist(residual, bins=min(12, max(5, len(residual) // 4)))
    axes[2].axvline(0.0, linestyle="--", color="black")
    axes[2].set_title("Residual distribution")
    axes[2].set_xlabel("Prediction - actual (pp)")
    axes[2].set_ylabel("Observations")
    figure.suptitle(f"{architecture}, seed {seed}, {partition}")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_trajectory_plot(
    path: Path,
    rows: list[dict[str, str]],
    target: np.ndarray,
    prediction: np.ndarray,
    architecture: str,
    seed: int,
    partition: str,
) -> None:
    import matplotlib.pyplot as plt

    cells: dict[str, list[tuple[int, float, float]]] = {}
    for index, row in enumerate(rows):
        cells.setdefault(row["physical_cell_id"], []).append(
            (
                int(row["rpt_index"]),
                float(target[index]),
                float(prediction[index]),
            )
        )
    cell_names = sorted(cells)
    columns = min(3, max(1, len(cell_names)))
    figure, axes = plt.subplots(
        math.ceil(len(cell_names) / columns),
        columns,
        figsize=(5 * columns, 3.5 * math.ceil(len(cell_names) / columns)),
        squeeze=False,
    )
    for axis, cell in zip(axes.flat, cell_names):
        observations = sorted(cells[cell])
        rpt = [value[0] for value in observations]
        actual = [value[1] for value in observations]
        predicted = [value[2] for value in observations]
        axis.plot(rpt, actual, marker="o", label="Actual")
        axis.plot(rpt, predicted, marker="x", linestyle="--", label="Predicted")
        axis.set_title(cell)
        axis.set_xlabel("Verified RPT order")
        axis.set_ylabel("SoH (%)")
        axis.legend()
    for axis in list(axes.flat)[len(cell_names):]:
        axis.set_visible(False)
    figure.suptitle(f"{architecture}, seed {seed}, {partition} cells")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _environment_record(device: torch.device, deterministic: dict[str, Any]) -> dict[str, Any]:
    cuda_device: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_device = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "major": properties.major,
            "minor": properties.minor,
        }
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "cuda_device": cuda_device,
        "numpy_version": np.__version__,
        "determinism": deterministic,
        "known_reproducibility_limitations": (
            "Determinism is configured for this PyTorch/CUDA build and device; "
            "bitwise equivalence across different hardware, drivers, or library versions "
            "is not claimed."
        ),
    }


def _append_log(path: Path, message: str) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(f"{datetime.now(timezone.utc).isoformat()} {message}\n")
        stream.flush()


def _train_one_run(
    *,
    architecture: str,
    seed: int,
    config: dict[str, Any],
    train_inputs: dict[str, torch.Tensor],
    train_targets: torch.Tensor,
    validation_inputs: dict[str, torch.Tensor],
    validation_targets: torch.Tensor,
    train_rows: list[dict[str, str]],
    validation_rows: list[dict[str, str]],
    output_dir: Path,
    device: torch.device,
    code_hash: str,
    dataset_manifest_sha256: str,
    config_sha256: str,
    authorization_scope: dict[str, Any],
    budget_started: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training = config["training"]
    generator = _seed_everything(seed)
    model = build_model(architecture).to(device)
    parameter_count = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    dataset = TensorDataset(
        train_inputs["X_discharge"],
        train_inputs["X_charge"],
        train_inputs["X_physics"],
        train_targets,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    stem = f"{architecture}_seed_{seed}"
    checkpoint_path = output_dir / "checkpoints" / architecture / f"seed_{seed}_best.pt"
    best_rmse = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    run_started = time.perf_counter()
    patience = int(training["early_stopping_patience"])
    min_delta = float(training["early_stopping_min_delta"])
    max_epochs = int(training["max_epochs"])
    for epoch in range(1, max_epochs + 1):
        elapsed = time.perf_counter() - budget_started
        if elapsed >= float(config["compute_budget"]["max_total_device_hours"]) * 3600:
            raise TimeoutError("Approved total device-hour budget was exhausted.")
        model.train()
        squared_error_sum = 0.0
        observations = 0
        for discharge, charge, physics, target in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(
                discharge.to(device),
                charge.to(device),
                physics.to(device),
            ).squeeze(1)
            loss = nn.functional.mse_loss(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss for {architecture}, seed {seed}, epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach()) * len(target)
            observations += len(target)

        train_mse = squared_error_sum / observations
        validation_prediction = _predict(model, validation_inputs, device)
        current_metrics = _metric_values(validation_targets, validation_prediction)
        validation_rmse = float(current_metrics["RMSE"])
        if not math.isfinite(validation_rmse):
            raise FloatingPointError(
                f"Non-finite Validation RMSE for {architecture}, seed {seed}, epoch {epoch}."
            )
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss": train_mse,
                "validation_RMSE": validation_rmse,
            }
        )
        if validation_rmse + min_delta < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_optimizer_state = _optimizer_state_to_cpu(optimizer.state_dict())
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None or best_optimizer_state is None:
        raise RuntimeError(f"No Validation checkpoint was selected for {architecture}, seed {seed}.")
    _atomic_torch_save(
        checkpoint_path,
        {
            "model_state_dict": best_state,
            "optimizer_state_dict": best_optimizer_state,
            "architecture": architecture,
            "architecture_config": architecture_config(architecture),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_RMSE": best_rmse,
            "parameter_count": parameter_count,
            "training_configuration": training,
            "frozen_dataset_identifier": P5_DATASET_ID,
            "frozen_dataset_manifest_sha256": dataset_manifest_sha256,
            "benchmark_config_sha256": config_sha256,
            "model_source_code_sha256": code_hash,
            "authorization_scope_sha256": _sha256_json(authorization_scope),
            "target_transform": None,
        },
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    reloaded_model = build_model(architecture).to(device)
    reloaded_model.load_state_dict(checkpoint["model_state_dict"])
    reloaded_model.eval()

    train_prediction = _predict(model, train_inputs, device)
    validation_prediction = _predict(model, validation_inputs, device)
    reloaded_train = _predict(reloaded_model, train_inputs, device)
    reloaded_validation = _predict(reloaded_model, validation_inputs, device)
    train_reload_max_delta = float(torch.max(torch.abs(train_prediction - reloaded_train)))
    validation_reload_max_delta = float(
        torch.max(torch.abs(validation_prediction - reloaded_validation))
    )
    if train_reload_max_delta > 1e-6 or validation_reload_max_delta > 1e-6:
        raise RuntimeError(
            "Checkpoint reload prediction mismatch exceeded 1e-6: "
            f"Train={train_reload_max_delta}, Validation={validation_reload_max_delta}."
        )

    train_metrics = _metric_values(train_targets, train_prediction)
    validation_metrics = _metric_values(validation_targets, validation_prediction)
    duration = time.perf_counter() - run_started
    _write_learning_curve(history, output_dir, stem)
    prediction_rows: list[dict[str, Any]] = []
    for partition, rows, targets, predictions in (
        ("train", train_rows, train_targets, train_prediction),
        ("validation", validation_rows, validation_targets, validation_prediction),
    ):
        target_array = targets.numpy()
        prediction_array = predictions.numpy()
        if len(rows) != len(target_array) or len(rows) != len(prediction_array):
            raise ValueError(f"{architecture}, seed {seed}: prediction count mismatch.")
        _write_prediction_plot(
            output_dir / "prediction_plots" / f"{stem}_{partition}_diagnostics.png",
            target_array,
            prediction_array,
            architecture,
            seed,
            partition,
        )
        _write_trajectory_plot(
            output_dir / "prediction_plots" / f"{stem}_{partition}_trajectories.png",
            rows,
            target_array,
            prediction_array,
            architecture,
            seed,
            partition,
        )
        for index, row in enumerate(rows):
            prediction_rows.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "dataset_part": row["dataset_part"],
                    "physical_cell_id": row["physical_cell_id"],
                    "rpt_index": row["rpt_index"],
                    "partition": partition,
                    "target_soh_percent": float(target_array[index]),
                    "predicted_soh_percent": float(prediction_array[index]),
                    "residual_prediction_minus_actual_pp": float(
                        prediction_array[index] - target_array[index]
                    ),
                }
            )

    run_metrics = {
        "architecture": architecture,
        "seed": seed,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_duration_seconds": duration,
        "best_validation_RMSE": best_rmse,
        "train_MAE": train_metrics["MAE"],
        "train_RMSE": train_metrics["RMSE"],
        "train_R2": train_metrics["R2"],
        "validation_MAE": validation_metrics["MAE"],
        "validation_RMSE": validation_metrics["RMSE"],
        "validation_R2": validation_metrics["R2"],
        "train_mean_residual_pp": float(torch.mean(train_prediction - train_targets)),
        "validation_mean_residual_pp": float(
            torch.mean(validation_prediction - validation_targets)
        ),
        "validation_curve_range_RMSE": max(
            float(row["validation_RMSE"]) for row in history
        )
        - min(float(row["validation_RMSE"]) for row in history),
        "checkpoint_path": checkpoint_path.relative_to(output_dir).as_posix(),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "train_reload_max_abs_prediction_delta": train_reload_max_delta,
        "validation_reload_max_abs_prediction_delta": validation_reload_max_delta,
        "training_configuration_sha256": config_sha256,
        "dataset_identifier": P5_DATASET_ID,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "model_source_code_sha256": code_hash,
    }
    return run_metrics, history, prediction_rows


def _model_summary(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    metric_names = (
        "train_MAE",
        "train_RMSE",
        "train_R2",
        "validation_MAE",
        "validation_RMSE",
        "validation_R2",
    )
    for architecture in ARCHITECTURE_NAMES:
        rows = [row for row in run_rows if row["architecture"] == architecture]
        if len(rows) != 3:
            raise ValueError(
                f"Expected exactly three seeds for {architecture}, found {len(rows)}."
            )
        summary: dict[str, Any] = {
            "architecture": architecture,
            "parameter_count": rows[0]["parameter_count"],
            "seed_count": len(rows),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows if row[metric] is not None]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_stdev"] = (
                statistics.stdev(values) if len(values) > 1 else 0.0
            )
        summary["best_epoch_mean"] = statistics.mean(
            float(row["best_epoch"]) for row in rows
        )
        summary["validation_curve_range_RMSE_mean"] = statistics.mean(
            float(row["validation_curve_range_RMSE"]) for row in rows
        )
        summary["validation_curve_range_RMSE_max"] = max(
            float(row["validation_curve_range_RMSE"]) for row in rows
        )
        summary["train_validation_RMSE_gap_mean"] = (
            summary["validation_RMSE_mean"] - summary["train_RMSE_mean"]
        )
        summary["validation_mean_residual_pp_mean"] = statistics.mean(
            float(row["validation_mean_residual_pp"]) for row in rows
        )
        summary["checkpoint_paths"] = ";".join(
            str(row["checkpoint_path"]) for row in rows
        )
        result.append(summary)
    return result


def _write_model_comparison(
    path: Path,
    summaries: list[dict[str, Any]],
    baseline: dict[str, Any],
    best_run: dict[str, Any],
    best_architecture: dict[str, Any],
    stable_architecture: dict[str, Any],
) -> None:
    lines = [
        "# BaFuse P5 Train/Validation architecture comparison",
        "",
        "Selection uses Validation RMSE. Metrics are observation-level and are in",
        "SoH percentage points for MAE/RMSE. No External predictions or metrics",
        "were produced.",
        "",
        "| Model | Parameters | Train MAE | Train RMSE | Train R² | Validation MAE | Validation RMSE | Validation R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {architecture} | {parameter_count} | {train_MAE_mean:.4f} ± {train_MAE_stdev:.4f} | "
            "{train_RMSE_mean:.4f} ± {train_RMSE_stdev:.4f} | {train_R2_mean:.4f} ± {train_R2_stdev:.4f} | "
            "{validation_MAE_mean:.4f} ± {validation_MAE_stdev:.4f} | "
            "{validation_RMSE_mean:.4f} ± {validation_RMSE_stdev:.4f} | "
            "{validation_R2_mean:.4f} ± {validation_R2_stdev:.4f} |".format(**row)
        )
    lines.extend(
        [
            "",
            f"Train-mean baseline (fit on Train only): Train MAE/RMSE/R² "
            f"{baseline['train']['MAE']:.4f}/{baseline['train']['RMSE']:.4f}/"
            f"{baseline['train']['R2']:.4f}; Validation MAE/RMSE/R² "
            f"{baseline['validation']['MAE']:.4f}/{baseline['validation']['RMSE']:.4f}/"
            f"{baseline['validation']['R2']:.4f}.",
            "",
            f"- Best individual checkpoint: `{best_run['architecture']}`, seed "
            f"{best_run['seed']}, Validation RMSE {best_run['validation_RMSE']:.4f} pp.",
            f"- Best architecture by mean Validation RMSE: "
            f"`{best_architecture['architecture']}` "
            f"({best_architecture['validation_RMSE_mean']:.4f} ± "
            f"{best_architecture['validation_RMSE_stdev']:.4f} pp).",
            f"- Most stable architecture by seed-to-seed Validation RMSE standard "
            f"deviation: `{stable_architecture['architecture']}` "
            f"(SD {stable_architecture['validation_RMSE_stdev']:.4f} pp).",
            "",
            "The best-mean architecture is a candidate for a separately authorized",
            "External evaluation, not evidence of performance on unseen external cells.",
            "Validation contains only 37 observations from three physical cells, so",
            "architecture selection may be sensitive to this limited split.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_training_report(
    path: Path,
    config: dict[str, Any],
    summaries: list[dict[str, Any]],
    run_rows: list[dict[str, Any]],
    best_run: dict[str, Any],
    best_architecture: dict[str, Any],
    stable_architecture: dict[str, Any],
) -> None:
    lines = [
        "# BaFuse P5 Train/Validation benchmark report",
        "",
        f"- Dataset: `{P5_DATASET_ID}`.",
        f"- Dataset manifest SHA-256: `{config['dataset_manifest_sha256']}`.",
        "- Target: same-RPT reconstructed Step 4 discharge capacity retention in percent; not official Oxford ground truth.",
        "- Relationship: same-RPT operating-state measurements to same-RPT SoH estimation, not future forecasting.",
        "- Data usage: 102 Train and 37 Validation observations; External arrays were not loaded.",
        "- Preprocessing: frozen P5 preprocessing only; scaler remains the pre-fitted Train-only transform.",
        "- Target transform: none; optimization is directly on SoH percentage values.",
        "- Loss/optimizer: MSELoss / AdamW with the shared configured hyperparameters.",
        "- Checkpoint selection: lowest Validation RMSE, independent for each architecture and seed.",
        "",
        "## Per-run metrics",
        "",
        "| Model | Seed | Parameters | Best epoch | Duration (s) | Train MAE | Train RMSE | Train R² | Validation MAE | Validation RMSE | Validation R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | {row['parameter_count']} | "
            f"{row['best_epoch']} | {row['training_duration_seconds']:.2f} | "
            f"{row['train_MAE']:.4f} | {row['train_RMSE']:.4f} | {row['train_R2']:.4f} | "
            f"{row['validation_MAE']:.4f} | {row['validation_RMSE']:.4f} | "
            f"{row['validation_R2']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Seed summaries and diagnostics",
            "",
            "| Model | Validation RMSE mean ± SD (pp) | Validation MAE mean ± SD (pp) | Train–Validation RMSE gap (pp) | Validation RMSE curve range mean/max (pp) | Mean validation residual (pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summaries:
        lines.append(
            f"| {row['architecture']} | {row['validation_RMSE_mean']:.4f} ± "
            f"{row['validation_RMSE_stdev']:.4f} | {row['validation_MAE_mean']:.4f} ± "
            f"{row['validation_MAE_stdev']:.4f} | {row['train_validation_RMSE_gap_mean']:.4f} | "
            f"{row['validation_curve_range_RMSE_mean']:.4f}/{row['validation_curve_range_RMSE_max']:.4f} | "
            f"{row['validation_mean_residual_pp_mean']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Positive validation residual means predictions exceed observed SoH on average; negative means underprediction.",
            "The curve-range column is the max-minus-min Validation RMSE observed during each run, averaged and maximized across seeds. It is descriptive, not a threshold-based instability diagnosis.",
            "",
            "## Selection and limitations",
            "",
            f"- Best individual model/seed: `{best_run['architecture']}`, seed {best_run['seed']}, "
            f"Validation RMSE {best_run['validation_RMSE']:.4f} pp.",
            f"- Best architecture by mean Validation RMSE: `{best_architecture['architecture']}`.",
            f"- Lowest seed-to-seed Validation RMSE standard deviation: `{stable_architecture['architecture']}`.",
            "- Reported R² is computed separately over the observations in each Train or Validation partition.",
            "- Seed mean and sample SD summarize run variability; they are not pooled prediction metrics.",
            "- This split has only three Validation physical cells. Results are not evidence of generalization to Part 3 or other unseen cells.",
            "- Per-cell trajectory plots are diagnostic only; no per-cell metric aggregation is used for model selection.",
            "- No hyperparameter search, PyDMA, Half-Cell fitting, or External prediction evaluation was performed.",
            "",
            "## Checkpoints",
            "",
        ]
    )
    lines.extend(
        f"- `{row['architecture']}`, seed {row['seed']}: "
        f"`{row['checkpoint_path']}` (SHA-256 `{row['checkpoint_sha256']}`)."
        for row in run_rows
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reproducibility_report(
    path: Path,
    environment: dict[str, Any],
    run_rows: list[dict[str, Any]],
    config_sha256: str,
    code_hash: str,
    scope_sha256: str,
) -> None:
    lines = [
        "# BaFuse benchmark reproducibility report",
        "",
        f"- Training configuration SHA-256: `{config_sha256}`.",
        f"- Model/runner source SHA-256: `{code_hash}`.",
        f"- Authorization scope SHA-256: `{scope_sha256}`.",
        f"- PyTorch: `{environment['pytorch_version']}`; CUDA runtime: `{environment['cuda_runtime_version']}`.",
        f"- Device: `{environment['cuda_device']['name']}`.",
        f"- Deterministic algorithms enabled: `{environment['determinism']['torch_deterministic_algorithms']}`.",
        f"- cuDNN deterministic: `{environment['determinism']['cudnn_deterministic']}`; benchmark: `{environment['determinism']['cudnn_benchmark']}`.",
        f"- CUBLAS_WORKSPACE_CONFIG: `{environment['determinism']['cublas_workspace_config']}`.",
        "- Python, NumPy, PyTorch CPU, and CUDA RNGs are seeded per run. Each model is freshly instantiated for every seed; DataLoader uses a seed-specific generator with zero worker processes.",
        "- No target or feature preprocessing is fitted during training. P5 physics scaling is the frozen Train-only transform.",
        "- Checkpoint reload comparison covers all Train and Validation predictions for every run, with maximum absolute tolerance 1e-6.",
        "",
        "| Model | Seed | Reload max |Δ| Train | Reload max |Δ| Validation |",
        "|---|---:|---:|---:|",
    ]
    for row in run_rows:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | "
            f"{row['train_reload_max_abs_prediction_delta']:.9g} | "
            f"{row['validation_reload_max_abs_prediction_delta']:.9g} |"
        )
    lines.extend(
        [
            "",
            str(environment["known_reproducibility_limitations"]),
            "External Test NPZ was hash-verified as part of the frozen dataset but was not opened by the training runner.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _manifest_outputs(output_dir: Path) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        if path.name == "benchmark_sha256_manifest.json":
            continue
        outputs[path.relative_to(output_dir).as_posix()] = {
            "sha256": _sha256(path)
        }
    return outputs


def run_benchmark(
    config_path: Path = DEFAULT_CONFIG,
    authorization_file: Path | None = None,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    config = _read_json(config_path)
    _validate_config(config)
    dataset_dir = _resolve(config["dataset_directory"])
    output_dir = _resolve(config["output_directory"])
    if output_dir.exists():
        raise FileExistsError(
            f"Versioned benchmark output already exists; refusing to overwrite: {output_dir}"
        )
    dataset_manifest_path = dataset_dir / "bafuse_p5_sha256_manifest.json"
    verified = _verify_manifest(dataset_dir, dataset_manifest_path)
    if verified["manifest_sha256"] != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen P5 manifest hash differs from the training config.")

    compute_budget = config["compute_budget"]
    scope = build_training_scope(
        label_manifest=dataset_dir / "bafuse_soh_labels_frozen.csv",
        feature_matrix=dataset_manifest_path,
        split_manifest=dataset_dir / "bafuse_p5_split_manifest.csv",
        experiment_config=config_path,
        compute_budget=compute_budget,
        output_directory=output_dir,
        architectures=config["architectures"],
    )
    if authorization_file is None:
        authorization_value = config.get("authorization_file")
        if not authorization_value:
            raise ValueError("Provide the approved scope-bound authorization file.")
        authorization_file = _resolve(authorization_value)
    require_training_authorization(authorization_file.resolve(), scope)

    device = torch.device(compute_budget["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Authorized CUDA device is unavailable: {device}.")
    if device.type != "cuda":
        raise RuntimeError("The approved P5 benchmark configuration requires CUDA.")
    if device.index not in (None, 0):
        raise RuntimeError("The benchmark is scoped to the verified CUDA device 0.")
    deterministic = _configure_determinism()
    environment = _environment_record(device, deterministic)

    dataset_verification: dict[str, Any] = {
        "dataset_identifier": P5_DATASET_ID,
        "dataset_manifest_sha256": verified["manifest_sha256"],
        "expected_dataset_manifest_sha256": EXPECTED_P5_MANIFEST_SHA256,
        "frozen_artifact_hashes_verified": verified["verified_hashes"],
        "frozen_artifact_hash_verification_passed": True,
        "train_validation_inputs_loaded": True,
        "external_npz_hash_verified_but_not_loaded": True,
        "external_prediction_evaluation_authorized": False,
        "target_semantics": (
            "reconstructed same-cell Step 4 discharge capacity retention; "
            "not official Oxford ground truth"
        ),
    }
    input_started = time.perf_counter()
    (
        train_inputs,
        train_targets,
        validation_inputs,
        validation_targets,
        train_rows,
        validation_rows,
    ) = _read_internal_inputs(dataset_dir, verified["manifest"], dataset_verification)
    dataset_verification["internal_tensor_validation_seconds"] = (
        time.perf_counter() - input_started
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "checkpoints").mkdir()
    (output_dir / "training_curves").mkdir()
    (output_dir / "prediction_plots").mkdir()
    _write_json(output_dir / "benchmark_config.json", config)
    _write_json(output_dir / "benchmark_environment.json", environment)
    _write_json(
        output_dir / "authorization_scope.json",
        {"scope": scope, "scope_sha256": _sha256_json(scope)},
    )
    code_paths = {
        "training_runner": Path(__file__).resolve(),
        "model_definitions": Path(__file__).with_name("p5_benchmark_models.py"),
    }
    code_hashes = {name: _sha256(path) for name, path in code_paths.items()}
    code_hash = _sha256_json(code_hashes)
    config_sha256 = _sha256(config_path)
    _write_json(output_dir / "benchmark_dataset_verification.json", dataset_verification)
    _write_json(
        output_dir / "benchmark_training_status.json",
        {
            "status": "running",
            "training_authorized": True,
            "external_prediction_evaluation_performed": False,
            "planned_runs": len(ARCHITECTURE_NAMES) * len(config["training"]["seeds"]),
            "completed_runs": 0,
        },
    )

    baseline_value = float(train_targets.mean())
    baseline = {
        "train": _metric_values(
            train_targets,
            torch.full_like(train_targets, baseline_value),
        ),
        "validation": _metric_values(
            validation_targets,
            torch.full_like(validation_targets, baseline_value),
        ),
        "train_mean_percent": baseline_value,
        "fit_scope": "Train observations only",
    }
    _write_json(output_dir / "benchmark_baseline_metrics.json", baseline)

    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    log_path = output_dir / "benchmark_training_log.txt"
    budget_started = time.perf_counter()
    total_runs = len(ARCHITECTURE_NAMES) * len(config["training"]["seeds"])
    for architecture in ARCHITECTURE_NAMES:
        for seed in config["training"]["seeds"]:
            run_index = len(run_rows) + 1
            _append_log(
                log_path,
                f"START run {run_index}/{total_runs}: {architecture}, seed {seed}",
            )
            try:
                run_metrics, _, run_predictions = _train_one_run(
                    architecture=architecture,
                    seed=int(seed),
                    config=config,
                    train_inputs=train_inputs,
                    train_targets=train_targets,
                    validation_inputs=validation_inputs,
                    validation_targets=validation_targets,
                    train_rows=train_rows,
                    validation_rows=validation_rows,
                    output_dir=output_dir,
                    device=device,
                    code_hash=code_hash,
                    dataset_manifest_sha256=verified["manifest_sha256"],
                    config_sha256=config_sha256,
                    authorization_scope=scope,
                    budget_started=budget_started,
                )
            except Exception:
                error_path = output_dir / f"error_{architecture}_seed_{seed}.txt"
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                _write_json(
                    output_dir / "benchmark_training_status.json",
                    {
                        "status": "failed",
                        "failed_run": {"architecture": architecture, "seed": seed},
                        "completed_runs": len(run_rows),
                        "error_log": error_path.name,
                        "external_prediction_evaluation_performed": False,
                    },
                )
                _append_log(
                    log_path,
                    f"FAILED run {run_index}/{total_runs}; traceback saved to {error_path.name}",
                )
                raise
            run_rows.append(run_metrics)
            prediction_rows.extend(run_predictions)
            _write_csv(output_dir / "benchmark_run_metrics.csv", run_rows)
            _write_csv(output_dir / "train_validation_predictions.csv", prediction_rows)
            _write_json(
                output_dir / "benchmark_training_status.json",
                {
                    "status": "running",
                    "training_authorized": True,
                    "external_prediction_evaluation_performed": False,
                    "planned_runs": total_runs,
                    "completed_runs": len(run_rows),
                    "last_completed_run": {
                        "architecture": architecture,
                        "seed": seed,
                        "best_epoch": run_metrics["best_epoch"],
                        "validation_RMSE": run_metrics["validation_RMSE"],
                    },
                },
            )
            _append_log(
                log_path,
                f"COMPLETE run {run_index}/{total_runs}: best_epoch={run_metrics['best_epoch']} "
                f"validation_RMSE={run_metrics['validation_RMSE']:.8f} "
                f"duration_s={run_metrics['training_duration_seconds']:.2f}",
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if len(run_rows) != total_runs or len(prediction_rows) != total_runs * 139:
        raise ValueError("The benchmark did not produce the expected run/prediction counts.")
    summaries = _model_summary(run_rows)
    best_run = min(
        run_rows,
        key=lambda row: (float(row["validation_RMSE"]), float(row["validation_MAE"])),
    )
    best_architecture = min(
        summaries,
        key=lambda row: (
            float(row["validation_RMSE_mean"]),
            float(row["validation_MAE_mean"]),
        ),
    )
    stable_architecture = min(
        summaries,
        key=lambda row: (
            float(row["validation_RMSE_stdev"]),
            float(row["validation_RMSE_mean"]),
        ),
    )
    _write_csv(output_dir / "benchmark_model_summary.csv", summaries)
    _write_model_comparison(
        output_dir / "benchmark_model_comparison.md",
        summaries,
        baseline,
        best_run,
        best_architecture,
        stable_architecture,
    )
    _write_training_report(
        output_dir / "benchmark_training_report.md",
        config,
        summaries,
        run_rows,
        best_run,
        best_architecture,
        stable_architecture,
    )
    scope_sha256 = _sha256_json(scope)
    _write_reproducibility_report(
        output_dir / "benchmark_reproducibility_report.md",
        environment,
        run_rows,
        config_sha256,
        code_hash,
        scope_sha256,
    )
    _write_json(
        output_dir / "benchmark_training_status.json",
        {
            "status": "complete",
            "training_authorized": True,
            "external_prediction_evaluation_performed": False,
            "planned_runs": total_runs,
            "completed_runs": len(run_rows),
            "prediction_rows_train_validation_only": len(prediction_rows),
            "best_checkpoint": {
                "architecture": best_run["architecture"],
                "seed": best_run["seed"],
                "path": best_run["checkpoint_path"],
            },
            "recommended_architecture_for_future_external_authorization": (
                best_architecture["architecture"]
            ),
        },
    )
    manifest = {
        "schema_version": 1,
        "benchmark_identifier": config["benchmark_identifier"],
        "dataset_identifier": P5_DATASET_ID,
        "dataset_manifest_sha256": verified["manifest_sha256"],
        "training_configuration_sha256": config_sha256,
        "authorization_scope_sha256": scope_sha256,
        "training_authorized": True,
        "external_prediction_evaluation_authorized": False,
        "external_npz_loaded": False,
        "code": {
            name: {"path": path.relative_to(PROJECT_ROOT).as_posix(), "sha256": code_hashes[name]}
            for name, path in code_paths.items()
        },
        "inputs": {
            "frozen_labels": {
                "path": (
                    dataset_dir / "bafuse_soh_labels_frozen.csv"
                ).relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(dataset_dir / "bafuse_soh_labels_frozen.csv"),
            },
            "frozen_split_manifest": {
                "path": (
                    dataset_dir / "bafuse_p5_split_manifest.csv"
                ).relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(dataset_dir / "bafuse_p5_split_manifest.csv"),
            },
            "frozen_dataset_manifest": {
                "path": dataset_manifest_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": verified["manifest_sha256"],
            },
            "training_config": {
                "path": config_path.relative_to(PROJECT_ROOT).as_posix(),
                "sha256": config_sha256,
            },
            "authorization_file": {
                "path": authorization_file.resolve().relative_to(PROJECT_ROOT).as_posix(),
                "sha256": _sha256(authorization_file.resolve()),
            },
            "authorization_scope": {"sha256": scope_sha256},
        },
        "outputs": _manifest_outputs(output_dir),
        "manifest_self_hash": "excluded; calculate after writing this file",
    }
    manifest_path = output_dir / "benchmark_sha256_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha256 = _sha256(manifest_path)
    return {
        "benchmark_identifier": config["benchmark_identifier"],
        "output_directory": str(output_dir),
        "dataset_manifest_sha256": verified["manifest_sha256"],
        "benchmark_manifest_sha256": manifest_sha256,
        "completed_runs": len(run_rows),
        "best_run": {
            "architecture": best_run["architecture"],
            "seed": best_run["seed"],
            "validation_RMSE": best_run["validation_RMSE"],
            "checkpoint_path": best_run["checkpoint_path"],
        },
        "best_architecture_by_mean_validation_RMSE": best_architecture[
            "architecture"
        ],
        "most_stable_architecture_by_validation_RMSE_sd": stable_architecture[
            "architecture"
        ],
        "external_prediction_evaluation_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Train the approved BaFuse P5 models on Train and select checkpoints "
            "using Validation RMSE; never loads External arrays."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--authorization-file", type=Path)
    arguments = parser.parse_args(argv)
    result = run_benchmark(arguments.config, arguments.authorization_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
