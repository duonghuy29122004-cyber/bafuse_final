from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from bafuse.hybrid.p5_normalization import PopulationZScore  # noqa: E402
from extended_models import (  # noqa: E402
    ARCHITECTURES,
    architecture_details,
    build_model,
    trainable_parameter_count,
)


CONFIG_PATH = EXPERIMENT_DIR / "benchmark_config.json"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_electrical_v1_20261114"
)
BASELINE_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_p5_architecture_comparison_v1"
    / "artifacts"
)
EXPECTED_DATASET_SHA = "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
EXPECTED_NORMALIZATION_SHA = (
    "eb1d4090ce3d7b5c84d3181750d99af773fa6e6561a576c65831a86d690a0478"
)
EXPECTED_COUNTS = {"train": 102, "validation": 37, "external_test": 64}
SEEDS = (42, 123, 2026)
MODALITIES = ("3M", "2M")
MODEL_NAMES = tuple(
    f"{architecture}-{modality}"
    for architecture in ARCHITECTURES
    for modality in MODALITIES
)
OFFICIAL_BASELINE_NAME = "CNN1D-3M"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    actual = np.asarray(target, dtype=np.float64).reshape(-1)
    predicted = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(actual) or actual.shape != predicted.shape:
        raise ValueError("Metric vectors must be non-empty and aligned.")
    error = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": (
            float(1 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
        "mean_signed_error_pp": float(np.mean(error)),
    }


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _configure_device() -> tuple[torch.device, dict[str, Any]]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must enable deterministic CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed training protocol requires CUDA device 0.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")
    return device, {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "dataloader_num_workers": 0,
    }


def _validate_config(config: dict[str, Any]) -> None:
    if (
        config.get("dataset_manifest_sha256") != EXPECTED_DATASET_SHA
        or config.get("normalization_parameters_sha256") != EXPECTED_NORMALIZATION_SHA
        or config.get("normalization_variant") != "D"
        or config.get("split_counts") != EXPECTED_COUNTS
        or tuple(config.get("seeds", ())) != SEEDS
    ):
        raise ValueError("Frozen dataset, split, normalization, or seed config changed.")
    expected_models = [
        {"architecture": architecture, "modalities": ["2M", "3M"]}
        for architecture in ARCHITECTURES
    ]
    if config.get("models") != expected_models:
        raise ValueError("Extended architecture model list changed.")
    if config.get("official_baseline", {}).get("model") != OFFICIAL_BASELINE_NAME:
        raise ValueError("CNN1D-3M must remain the only official baseline.")
    if config.get("official_baseline", {}).get("retrain_or_copy_checkpoint") is not False:
        raise ValueError("The official CNN1D-3M checkpoint may not be copied or retrained.")
    expected_training = {
        "max_epochs": 300,
        "batch_size": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "MSELoss",
        "target_space": "normalized SoH using frozen Variant D Train target statistics",
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.0,
        "checkpoint_metric": "validation_RMSE_soh_percentage_points",
        "device": "cuda:0",
    }
    if config.get("training") != expected_training:
        raise ValueError("Training protocol differs from the requested benchmark.")
    scope = config.get("scope", {})
    if any(scope.get(name) is not False for name in (
        "dataset_modification_authorized",
        "label_or_split_modification_authorized",
        "normalization_refit_authorized",
        "existing_checkpoint_modification_authorized",
        "existing_checkpoint_copy_authorized",
        "historical_result_modification_authorized",
    )):
        raise ValueError("Frozen data and existing artifacts must remain unchanged.")


def _load_npz(filename: str, expected_n: int) -> dict[str, np.ndarray]:
    if filename == "bafuse_external_p5.npz":
        raise ValueError("Training stage must not load External arrays.")
    with np.load(DATASET_DIR / filename, allow_pickle=False) as archive:
        expected_names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected_names:
            raise ValueError(f"Unexpected NPZ members in {filename}.")
        arrays = {name: archive[name].copy() for name in expected_names}
    expected_shapes = {
        "X_discharge": (expected_n, 128),
        "X_charge": (expected_n, 128),
        "X_physics": (expected_n, 5),
        "y_soh": (expected_n, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected_shapes[name]:
            raise ValueError(f"Unexpected dtype/shape in {filename}:{name}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {filename}:{name}.")
    return arrays


def _load_internal_data(
    config: dict[str, Any],
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, list[dict[str, str]]], dict[str, Any]]:
    manifest_path = DATASET_DIR / "bafuse_p5_sha256_manifest.json"
    if _sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen P5 dataset manifest checksum mismatch.")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_identifier") != config["dataset_identifier"]:
        raise ValueError("Unexpected frozen dataset identifier.")

    verified: dict[str, str] = {}
    for filename in (
        "bafuse_train_p5.npz",
        "bafuse_validation_p5.npz",
        "bafuse_p5_split_manifest.csv",
    ):
        entry = manifest.get("outputs", {}).get(filename)
        path = DATASET_DIR / filename
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"Frozen internal input checksum mismatch: {filename}.")
        verified[filename] = entry["sha256"]

    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    counts = Counter(row["partition"] for row in split_rows)
    if dict(counts) != EXPECTED_COUNTS or len(split_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Frozen split counts changed: {dict(counts)}.")
    keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in split_rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Frozen split contains duplicate sample keys.")
    cells_by_partition = {
        partition: {
            row["physical_cell_id"]
            for row in split_rows
            if row["partition"] == partition
        }
        for partition in EXPECTED_COUNTS
    }
    partitions = tuple(EXPECTED_COUNTS)
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            if cells_by_partition[first] & cells_by_partition[second]:
                raise ValueError(f"Frozen cell split overlaps: {first}/{second}.")

    ordered = {
        partition: sorted(
            (row for row in split_rows if row["partition"] == partition),
            key=lambda row: int(row["split_row_index"]),
        )
        for partition in ("train", "validation")
    }
    data = {
        "train": _load_npz("bafuse_train_p5.npz", 102),
        "validation": _load_npz("bafuse_validation_p5.npz", 37),
    }
    for partition, arrays in data.items():
        if len(ordered[partition]) != arrays["y_soh"].shape[0]:
            raise ValueError(f"{partition} sample keys do not match frozen array rows.")
    return data, ordered, {
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "split_counts": dict(counts),
        "physical_cell_counts": {
            partition: len(cells)
            for partition, cells in cells_by_partition.items()
        },
        "physical_cell_overlap": False,
        "verified_internal_input_hashes": verified,
        "external_npz_loaded": False,
        "external_targets_loaded": False,
    }


def _load_variant_d(
    config: dict[str, Any],
) -> tuple[dict[str, Any], PopulationZScore, PopulationZScore, PopulationZScore]:
    path = PROJECT_ROOT / config["normalization_parameters"]
    if not path.is_file() or _sha256(path) != config["normalization_parameters_sha256"]:
        raise ValueError("Frozen Variant D normalization checksum mismatch.")
    bundle = _read_json(path)
    if (
        bundle.get("dataset_identifier") != config["dataset_identifier"]
        or bundle.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or bundle.get("fit_partition") != "train"
        or bundle.get("fit_row_count") != 102
    ):
        raise ValueError("Variant D preprocessing provenance changed.")
    variant = bundle.get("variants", {}).get("D")
    if not isinstance(variant, dict):
        raise ValueError("Variant D scalers are missing.")
    discharge = PopulationZScore.from_dict(variant["discharge"])
    charge = PopulationZScore.from_dict(variant["charge"])
    target = PopulationZScore.from_dict(variant["target"])
    if tuple(scaler.mean.size for scaler in (discharge, charge, target)) != (128, 128, 1):
        raise ValueError("Variant D scaler dimensions changed.")
    if any(scaler.fit_row_count != 102 for scaler in (discharge, charge, target)):
        raise ValueError("Variant D scaler Train row counts changed.")
    return bundle, discharge, charge, target


def _prepare_inputs(
    arrays: dict[str, np.ndarray],
    modality: str,
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    result = {
        "X_discharge": discharge_scaler.transform(arrays["X_discharge"]),
        "X_charge": charge_scaler.transform(arrays["X_charge"]),
    }
    if modality == "3M":
        result["X_physics"] = np.array(arrays["X_physics"], copy=True)
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"{modality} preprocessing produced NaN/Inf.")
    return result


def _to_tensors(inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(value.astype(np.float32, copy=False))
        for name, value in inputs.items()
    }


def _forward(
    model: nn.Module,
    modality: str,
    inputs: dict[str, torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    discharge = inputs["X_discharge"].to(device)
    charge = inputs["X_charge"].to(device)
    if modality == "3M":
        return model(discharge, charge, inputs["X_physics"].to(device))
    return model(discharge, charge)


def _predict(
    model: nn.Module,
    modality: str,
    inputs: dict[str, torch.Tensor],
    target_scaler: PopulationZScore,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        normalized = _forward(model, modality, inputs, device).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("Model produced non-finite SoH predictions.")
    return prediction


def _prediction_rows(
    model_name: str,
    seed: int,
    partition: str,
    sample_rows: list[dict[str, str]],
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    if len(sample_rows) != len(target) or len(target) != len(prediction):
        raise ValueError(f"Prediction alignment failed for {model_name}/{partition}.")
    records: list[dict[str, Any]] = []
    for row, actual, predicted in zip(sample_rows, target, prediction):
        error = float(predicted - actual)
        records.append(
            {
                "model": model_name,
                "seed": seed,
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "partition": partition,
                "target_soh_percent": float(actual),
                "predicted_soh_percent": float(predicted),
                "residual_prediction_minus_actual_pp": error,
                "absolute_error_pp": abs(error),
            }
        )
    return records


def _official_baseline_reference(config: dict[str, Any]) -> dict[str, Any]:
    freeze_path = BASELINE_DIR / "checkpoint_freeze_manifest.json"
    final_manifest_path = BASELINE_DIR / "final_sha256_manifest.json"
    if not freeze_path.is_file() or not final_manifest_path.is_file():
        raise FileNotFoundError("Official CNN1D-3M baseline manifests are missing.")
    final_manifest = _read_json(final_manifest_path)
    freeze_rel = str(freeze_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    if final_manifest.get("files", {}).get(freeze_rel) != _sha256(freeze_path):
        raise ValueError("Official baseline freeze manifest does not match its final hash.")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "complete"
        or freeze.get("frozen") is not True
        or freeze.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or freeze.get("normalization_parameters_sha256")
        != config["normalization_parameters_sha256"]
    ):
        raise ValueError("Official CNN1D-3M baseline provenance changed.")
    baseline_entries = [
        entry
        for entry in freeze["checkpoints"]
        if entry.get("architecture") == "CNN1D"
    ]
    if {int(entry["seed"]) for entry in baseline_entries} != set(SEEDS):
        raise ValueError("Official CNN1D-3M baseline is missing a seed.")
    for entry in baseline_entries:
        checkpoint = BASELINE_DIR / entry["path"]
        if not checkpoint.is_file() or _sha256(checkpoint) != entry["sha256"]:
            raise ValueError(f"Official baseline checkpoint hash changed: {entry['path']}.")

    train_summary_path = BASELINE_DIR / "train_validation_summary.csv"
    for path in (train_summary_path,):
        relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if final_manifest.get("files", {}).get(relative) != _sha256(path):
            raise ValueError(f"Official baseline metrics do not match final hash: {path}.")
    train_rows = _read_csv(train_summary_path)
    train = next(row for row in train_rows if row["architecture"] == "CNN1D")
    return {
        "model": OFFICIAL_BASELINE_NAME,
        "role": "only_official_baseline",
        "source_experiment": config["official_baseline"]["source_experiment"],
        "checkpoint_freeze_manifest_sha256": _sha256(freeze_path),
        "checkpoint_retrained": False,
        "checkpoint_copied": False,
        "parameter_count": int(baseline_entries[0]["parameter_count"]),
        "checkpoints": [
            {
                "seed": int(entry["seed"]),
                "path": str((BASELINE_DIR / entry["path"]).relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": entry["sha256"],
            }
            for entry in sorted(baseline_entries, key=lambda item: int(item["seed"]))
        ],
        "train_MAE_mean": float(train["train_MAE_mean"]),
        "train_MAE_sd": float(train["train_MAE_sd"]),
        "train_RMSE_mean": float(train["train_RMSE_mean"]),
        "train_RMSE_sd": float(train["train_RMSE_sd"]),
        "train_R2_mean": float(train["train_R2_mean"]),
        "train_R2_sd": float(train["train_R2_sd"]),
        "validation_MAE_mean": float(train["validation_MAE_mean"]),
        "validation_MAE_sd": float(train["validation_MAE_sd"]),
        "validation_RMSE_mean": float(train["validation_RMSE_mean"]),
        "validation_RMSE_sd": float(train["validation_RMSE_sd"]),
        "validation_R2_mean": float(train["validation_R2_mean"]),
        "validation_R2_sd": float(train["validation_R2_sd"]),
        "metrics_sources": {
            "train_validation_summary": str(train_summary_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
    }


def _train_one(
    architecture: str,
    modality: str,
    seed: int,
    config: dict[str, Any],
    config_sha256: str,
    data: dict[str, dict[str, np.ndarray]],
    sample_rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = config["training"]
    discharge_scaler, charge_scaler, target_scaler = scalers
    model_name = f"{architecture}-{modality}"
    generator = _seed_everything(seed)

    prepared = {
        partition: _to_tensors(
            _prepare_inputs(
                arrays, modality, discharge_scaler, charge_scaler
            )
        )
        for partition, arrays in data.items()
    }
    normalized_targets = {
        partition: target_scaler.transform(arrays["y_soh"]).reshape(-1)
        for partition, arrays in data.items()
    }
    train_fields = [
        prepared["train"]["X_discharge"],
        prepared["train"]["X_charge"],
    ]
    validation_fields = [
        prepared["validation"]["X_discharge"],
        prepared["validation"]["X_charge"],
    ]
    if modality == "3M":
        train_fields.append(prepared["train"]["X_physics"])
        validation_fields.append(prepared["validation"]["X_physics"])
    train_fields.append(torch.from_numpy(normalized_targets["train"]))
    train_loader = DataLoader(
        TensorDataset(*train_fields),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    model = build_model(architecture, modality).to(device)
    parameter_count = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    best_rmse = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    validation_target = data["validation"]["y_soh"].reshape(-1)

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            discharge = batch[0].to(device)
            charge = batch[1].to(device)
            target = batch[-1].to(device)
            if modality == "3M":
                prediction = model(discharge, charge, batch[2].to(device)).squeeze(1)
            else:
                prediction = model(discharge, charge).squeeze(1)
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss for {model_name}, seed {seed}, epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            sample_count += len(target)
        train_loss = loss_sum / sample_count
        with torch.inference_mode():
            normalized_validation = _forward(
                model, modality, prepared["validation"], device
            ).detach().cpu().numpy()
        validation_loss = float(
            np.mean(
                (
                    normalized_validation.reshape(-1)
                    - normalized_targets["validation"]
                )
                ** 2
            )
        )
        validation_percent = target_scaler.inverse_transform(
            normalized_validation
        ).reshape(-1)
        validation_metrics = _metrics(validation_target, validation_percent)
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss_normalized_target": train_loss,
                "validation_mse_loss_normalized_target": validation_loss,
                "validation_RMSE_soh_pp": validation_metrics["RMSE"],
            }
        )
        if validation_metrics["RMSE"] < best_rmse:
            best_rmse = validation_metrics["RMSE"]
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                break

    if best_state is None:
        raise RuntimeError(f"No Validation-selected checkpoint for {model_name}/{seed}.")
    checkpoint_path = (
        ARTIFACT_DIR / "checkpoints" / model_name / f"seed_{seed}_best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": best_state,
            "model_name": model_name,
            "architecture": architecture,
            "modality": modality,
            "architecture_details": architecture_details(architecture, modality),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_RMSE_soh_pp": best_rmse,
            "parameter_count": parameter_count,
            "training_configuration": training,
            "dataset_identifier": config["dataset_identifier"],
            "dataset_manifest_sha256": config["dataset_manifest_sha256"],
            "normalization_variant": "D",
            "normalization_parameters_sha256": config[
                "normalization_parameters_sha256"
            ],
            "target_transform": _read_json(
                PROJECT_ROOT / config["normalization_parameters"]
            )["variants"]["D"]["target"],
            "experiment_config_sha256": config_sha256,
            "external_evaluation_performed": False,
            "physics_input_used": modality == "3M",
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    model.load_state_dict(best_state)
    model.to(device).eval()

    metric_fields: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        prediction = _predict(
            model, modality, prepared[partition], target_scaler, device
        )
        target = data[partition]["y_soh"].reshape(-1)
        metrics = _metrics(target, prediction)
        metric_fields.update(
            {f"{partition}_{name}": value for name, value in metrics.items()}
        )
        prediction_rows.extend(
            _prediction_rows(
                model_name,
                seed,
                partition,
                sample_rows[partition],
                target,
                prediction,
            )
        )
    duration = time.perf_counter() - started
    _write_csv(
        ARTIFACT_DIR / "histories" / f"{model_name}_seed_{seed}.csv",
        history,
    )
    best_history = history[best_epoch - 1]
    return (
        {
            "model": model_name,
            "architecture": architecture,
            "modality": modality,
            "seed": seed,
            "status": "trained_from_scratch",
            "parameter_count": parameter_count,
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "training_duration_seconds": duration,
            "train_mse_loss_normalized_target": best_history[
                "train_mse_loss_normalized_target"
            ],
            "validation_mse_loss_normalized_target": best_history[
                "validation_mse_loss_normalized_target"
            ],
            **metric_fields,
            "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "normalization_parameters_sha256": config[
                "normalization_parameters_sha256"
            ],
        },
        prediction_rows,
    )


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        selected = [row for row in rows if row["model"] == model_name]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected three seeds for {model_name}.")
        counts = {int(row["parameter_count"]) for row in selected}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent parameter counts for {model_name}.")
        summary: dict[str, Any] = {
            "model": model_name,
            "parameter_count": counts.pop(),
            "seed_count": len(selected),
        }
        for partition in ("train", "validation"):
            for metric in ("MAE", "RMSE", "R2"):
                values = [float(row[f"{partition}_{metric}"]) for row in selected]
                summary[f"{partition}_{metric}_mean"] = statistics.mean(values)
                summary[f"{partition}_{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def run() -> dict[str, Any]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {ARTIFACT_DIR}.")
    config = _read_json(CONFIG_PATH)
    _validate_config(config)
    data, sample_rows, dataset_info = _load_internal_data(config)
    normalization_bundle, discharge, charge, target = _load_variant_d(config)
    scalers = (discharge, charge, target)
    for partition in ("train", "validation"):
        for modality in MODALITIES:
            transformed = _prepare_inputs(
                data[partition], modality, discharge, charge
            )
            if not all(np.isfinite(value).all() for value in transformed.values()):
                raise ValueError(f"Invalid normalized {partition}/{modality} inputs.")
        if target.transform(data[partition]["y_soh"]).shape != (
            EXPECTED_COUNTS[partition],
            1,
        ):
            raise ValueError(f"Unexpected normalized target shape for {partition}.")
    baseline_reference = _official_baseline_reference(config)
    device, environment = _configure_device()

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    (ARTIFACT_DIR / "histories").mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ARTIFACT_DIR / "training.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    config_sha256 = _sha256(CONFIG_PATH)
    normalizer_path = PROJECT_ROOT / config["normalization_parameters"]
    _write_json(ARTIFACT_DIR / "training_environment.json", environment)
    _write_json(
        ARTIFACT_DIR / "dataset_verification.json",
        {
            **dataset_info,
            "train_shapes": {
                name: list(value.shape) for name, value in data["train"].items()
            },
            "validation_shapes": {
                name: list(value.shape) for name, value in data["validation"].items()
            },
            "normalization_parameters_sha256": _sha256(normalizer_path),
            "normalization_fit_partition": "train",
            "normalization_fit_row_count": 102,
            "physics_refit": False,
        },
    )
    _write_json(ARTIFACT_DIR / "official_baseline_reference.json", baseline_reference)
    _write_json(
        ARTIFACT_DIR / "model_architectures.json",
        {
            name: {
                "parameter_count": trainable_parameter_count(
                    build_model(name.rsplit("-", 1)[0], name.rsplit("-", 1)[1])
                ),
                "details": architecture_details(
                    name.rsplit("-", 1)[0], name.rsplit("-", 1)[1]
                ),
            }
            for name in MODEL_NAMES
        },
    )
    _write_json(
        ARTIFACT_DIR / "source_hashes.json",
        {
            "benchmark_config_sha256": config_sha256,
            "model_source_sha256": _sha256(EXPERIMENT_DIR / "extended_models.py"),
            "training_script_sha256": _sha256(EXPERIMENT_DIR / "train_extended_models.py"),
            "normalization_implementation_sha256": _sha256(
                PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_normalization.py"
            ),
            "normalization_parameters_sha256": _sha256(normalizer_path),
            "dataset_manifest_sha256": _sha256(
                DATASET_DIR / "bafuse_p5_sha256_manifest.json"
            ),
            "official_baseline_freeze_manifest_sha256": baseline_reference[
                "checkpoint_freeze_manifest_sha256"
            ],
        },
    )

    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        architecture, modality = model_name.rsplit("-", 1)
        for seed in SEEDS:
            logging.info("START %s seed %s", model_name, seed)
            row, predictions = _train_one(
                architecture,
                modality,
                seed,
                config,
                config_sha256,
                data,
                sample_rows,
                scalers,
                device,
            )
            run_rows.append(row)
            prediction_rows.extend(predictions)
            logging.info(
                "COMPLETE %s seed %s best_epoch=%s validation_RMSE=%.4f pp",
                model_name,
                seed,
                row["best_epoch"],
                row["validation_RMSE"],
            )
    run_rows.sort(key=lambda row: (MODEL_NAMES.index(row["model"]), int(row["seed"])))
    prediction_rows.sort(
        key=lambda row: (
            MODEL_NAMES.index(row["model"]),
            int(row["seed"]),
            row["partition"],
            int(row["rpt_index"]),
        )
    )
    _write_csv(ARTIFACT_DIR / "run_metrics.csv", run_rows)
    _write_csv(ARTIFACT_DIR / "train_validation_summary.csv", _summaries(run_rows))
    _write_csv(ARTIFACT_DIR / "train_validation_predictions.csv", prediction_rows)

    entries: list[dict[str, Any]] = []
    for row in run_rows:
        checkpoint_path = PROJECT_ROOT / row["checkpoint_path"]
        if _sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {checkpoint_path}.")
        entries.append(
            {
                "model": row["model"],
                "architecture": row["architecture"],
                "modality": row["modality"],
                "seed": int(row["seed"]),
                "path": row["checkpoint_path"],
                "sha256": row["checkpoint_sha256"],
                "parameter_count": int(row["parameter_count"]),
                "best_epoch": int(row["best_epoch"]),
                "validation_RMSE_soh_pp": float(row["validation_RMSE"]),
            }
        )
    if len(entries) != 18 or len({(entry["model"], entry["seed"]) for entry in entries}) != 18:
        raise ValueError("Expected 18 unique new checkpoints before External evaluation.")
    freeze = {
        "status": "complete",
        "frozen": True,
        "experiment_identifier": config["experiment_identifier"],
        "experiment_config_sha256": config_sha256,
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "normalization_parameters_sha256": config[
            "normalization_parameters_sha256"
        ],
        "official_baseline": OFFICIAL_BASELINE_NAME,
        "official_baseline_freeze_manifest_sha256": baseline_reference[
            "checkpoint_freeze_manifest_sha256"
        ],
        "new_checkpoint_count": len(entries),
        "checkpoint_count": len(entries),
        "external_evaluation_performed": False,
        "external_used_for_training_or_selection": False,
        "checkpoints": entries,
    }
    _write_json(ARTIFACT_DIR / "checkpoint_freeze_manifest.json", freeze)
    logging.info("All 18 new checkpoints frozen; External data has not been loaded.")
    training_files = {
        str(path.relative_to(ARTIFACT_DIR)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "training_stage_sha256_manifest.json"
    }
    _write_json(
        ARTIFACT_DIR / "training_stage_sha256_manifest.json",
        {
            "hash_scope": "All training-stage files after the 18-checkpoint freeze.",
            "files": training_files,
        },
    )
    return freeze


if __name__ == "__main__":
    run()
