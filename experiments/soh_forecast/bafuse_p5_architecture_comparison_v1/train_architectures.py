from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import platform
import random
import shutil
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

from architecture_models import (  # noqa: E402
    ARCHITECTURES,
    architecture_details,
    build_architecture,
    trainable_parameter_count,
)
from bafuse.hybrid.p5_normalization import (  # noqa: E402
    PopulationZScore,
    apply_voltage_normalization,
)


ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
CONFIG_PATH = EXPERIMENT_DIR / "benchmark_config.json"
DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_electrical_v1_20261114"
)
NORMALIZED_EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_p5_normalized_v1"
)
NORMALIZED_ARTIFACT_DIR = NORMALIZED_EXPERIMENT_DIR / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_NORMALIZED_ARCHITECTURE_BENCHMARK.md"
EXPECTED_DATASET_SHA = (
    "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
)
EXPECTED_NORMALIZATION_SHA = (
    "eb1d4090ce3d7b5c84d3181750d99af773fa6e6561a576c65831a86d690a0478"
)
EXPECTED_COUNTS = {"train": 102, "validation": 37, "external_test": 64}
ARCHITECTURE_ORDER = ARCHITECTURES


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return result


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if not len(y) or y.shape != y_hat.shape:
        raise ValueError("Metric vectors must be non-empty and have the same shape.")
    error = y_hat - y
    denominator = float(np.sum((y - y.mean()) ** 2))
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
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be deterministic.")
    if not torch.cuda.is_available():
        raise RuntimeError("The frozen benchmark config requires CUDA device 0.")
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


def _load_internal_npz(filename: str, expected_n: int) -> dict[str, np.ndarray]:
    if filename == "bafuse_external_p5.npz":
        raise ValueError("External arrays are prohibited in the training stage.")
    with np.load(DATASET_DIR / filename, allow_pickle=False) as archive:
        names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != names:
            raise ValueError(f"Unexpected NPZ members in {filename}.")
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "X_discharge": (expected_n, 128),
        "X_charge": (expected_n, 128),
        "X_physics": (expected_n, 5),
        "y_soh": (expected_n, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected_shapes[name]:
            raise ValueError(f"Unexpected dtype/shape for {filename}:{name}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite values in {filename}:{name}.")
    return arrays


def _load_frozen_training_data(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = DATASET_DIR / "bafuse_p5_sha256_manifest.json"
    if _sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen P5 manifest hash does not match the benchmark config.")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_identifier") != config["dataset_identifier"]:
        raise ValueError("Unexpected frozen dataset identifier.")
    verified: dict[str, str] = {}
    for name in (
        "bafuse_train_p5.npz",
        "bafuse_validation_p5.npz",
        "bafuse_p5_split_manifest.csv",
    ):
        entry = manifest.get("outputs", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Frozen manifest lacks {name}.")
        actual = _sha256(DATASET_DIR / name)
        if actual != entry.get("sha256"):
            raise ValueError(f"Frozen input hash mismatch: {name}.")
        verified[name] = actual

    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    counts = Counter(row["partition"] for row in split_rows)
    if dict(counts) != EXPECTED_COUNTS or len(split_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Frozen P5 split counts changed: {dict(counts)}.")
    keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in split_rows
    ]
    if len(set(keys)) != len(keys):
        raise ValueError("Frozen split contains duplicate observation keys.")
    partitions = ("train", "validation", "external_test")
    cells = {
        part: {row["physical_cell_id"] for row in split_rows if row["partition"] == part}
        for part in partitions
    }
    for index, first in enumerate(partitions):
        for second in partitions[index + 1 :]:
            if cells[first] & cells[second]:
                raise ValueError(f"Frozen cell split overlaps: {first}/{second}.")

    ordered = {
        part: sorted(
            (row for row in split_rows if row["partition"] == part),
            key=lambda row: int(row["split_row_index"]),
        )
        for part in ("train", "validation")
    }
    train = _load_internal_npz("bafuse_train_p5.npz", 102)
    validation = _load_internal_npz("bafuse_validation_p5.npz", 37)
    internal_keys = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for part in ("train", "validation")
        for row in ordered[part]
    }
    label_rows = _read_csv(DATASET_DIR / "bafuse_soh_labels_frozen.csv")
    internal_labels: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in label_rows:
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in internal_keys:
            if key in internal_labels:
                raise ValueError(f"Duplicate frozen label key: {key}.")
            internal_labels[key] = row
    if set(internal_labels) != internal_keys:
        raise ValueError("Frozen Train/Validation labels do not match split keys.")
    for partition, arrays in (("train", train), ("validation", validation)):
        expected = np.asarray(
            [
                float(
                    internal_labels[
                        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                    ]["soh_percent"]
                )
                for row in ordered[partition]
            ],
            dtype=np.float32,
        ).reshape(-1, 1)
        if not np.array_equal(expected, arrays["y_soh"]):
            raise ValueError(f"{partition} frozen targets are misaligned.")
        if any(
            internal_labels[
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            ]["final_status"]
            != "ACCEPTED"
            for row in ordered[partition]
        ):
            raise ValueError(f"{partition} contains a non-accepted label.")

    return (
        {"train": train, "validation": validation},
        {
            "counts": dict(counts),
            "cell_counts": {part: len(value) for part, value in cells.items()},
            "physical_cell_overlap": False,
            "verified_internal_hashes": verified,
            "external_npz_loaded": False,
            "external_targets_used": False,
        },
    )


def _load_variant_d(
    config: dict[str, Any],
) -> tuple[dict[str, Any], PopulationZScore, PopulationZScore, PopulationZScore]:
    normalizer_path = PROJECT_ROOT / config["normalization_parameters"]
    if _sha256(normalizer_path) != config["normalization_parameters_sha256"]:
        raise ValueError("Frozen Variant D normalization parameters changed.")
    bundle = _read_json(normalizer_path)
    if (
        bundle.get("dataset_identifier") != config["dataset_identifier"]
        or bundle.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or bundle.get("fit_partition") != "train"
        or bundle.get("fit_row_count") != 102
    ):
        raise ValueError("Variant D preprocessing provenance is not valid.")
    variant = bundle.get("variants", {}).get("D")
    if not isinstance(variant, dict):
        raise ValueError("Variant D normalizers are missing.")
    if not variant.get("voltage_normalization") or not variant.get("target_normalization"):
        raise ValueError("Variant D does not enable full voltage/target normalization.")
    discharge = PopulationZScore.from_dict(variant["discharge"])
    charge = PopulationZScore.from_dict(variant["charge"])
    target = PopulationZScore.from_dict(variant["target"])
    if (
        len(discharge.mean) != 128
        or len(charge.mean) != 128
        or len(target.mean) != 1
        or any(
            scaler.fit_row_count != 102
            for scaler in (discharge, charge, target)
        )
    ):
        raise ValueError("Variant D scaler shapes or fit counts changed.")
    return bundle, discharge, charge, target


def _prepare_inputs(
    arrays: dict[str, np.ndarray],
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    transformed = apply_voltage_normalization(
        {name: arrays[name] for name in ("X_discharge", "X_charge", "X_physics")},
        discharge_scaler=discharge_scaler,
        charge_scaler=charge_scaler,
    )
    if not np.array_equal(transformed["X_physics"], arrays["X_physics"]):
        raise ValueError("Frozen P5 Physics inputs changed during Variant D preparation.")
    return transformed


def _tensor_inputs(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(arrays[name].astype(np.float32, copy=False))
        for name in ("X_discharge", "X_charge", "X_physics")
    }


def _predict_percent(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    target_scaler: PopulationZScore,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        normalized = model(
            inputs["X_discharge"].to(device),
            inputs["X_charge"].to(device),
            inputs["X_physics"].to(device),
        ).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite SoH predictions.")
    return prediction


def _prediction_rows(
    architecture: str,
    seed: int,
    partition: str,
    rows: list[dict[str, str]],
    targets: np.ndarray,
    predictions: np.ndarray,
) -> list[dict[str, Any]]:
    if len(rows) != len(targets) or len(rows) != len(predictions):
        raise ValueError(f"{architecture}/{seed}/{partition} count mismatch.")
    return [
        {
            "architecture": architecture,
            "seed": seed,
            "dataset_part": row["dataset_part"],
            "physical_cell_id": row["physical_cell_id"],
            "rpt_index": row["rpt_index"],
            "partition": partition,
            "target_soh_percent": float(targets[index]),
            "predicted_soh_percent": float(predictions[index]),
            "residual_prediction_minus_actual_pp": float(
                predictions[index] - targets[index]
            ),
        }
        for index, row in enumerate(rows)
    ]


def _evaluate_internal_model(
    model: nn.Module,
    seed: int,
    data: dict[str, Any],
    rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    discharge_scaler, charge_scaler, target_scaler = scalers
    run_metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        arrays = data[partition]
        inputs = _tensor_inputs(
            _prepare_inputs(arrays, discharge_scaler, charge_scaler)
        )
        prediction = _predict_percent(model, inputs, target_scaler, device)
        target = arrays["y_soh"].reshape(-1)
        metrics = _metrics(target, prediction)
        run_metrics.update({f"{partition}_{key}": value for key, value in metrics.items()})
        prediction_rows.extend(
            _prediction_rows(
                "",
                seed,
                partition,
                rows[partition],
                target,
                prediction,
            )
        )
    return run_metrics, prediction_rows


def _reuse_cnn1d_d(
    config: dict[str, Any],
    data: dict[str, Any],
    dataset_info: dict[str, Any],
    rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
    normalization_bundle: dict[str, Any],
    normalizer_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline_metrics = _read_csv(NORMALIZED_ARTIFACT_DIR / "run_metrics.csv")
    baseline_predictions = _read_csv(
        NORMALIZED_ARTIFACT_DIR / "train_validation_predictions.csv"
    )
    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    target_scaler = scalers[2]
    for seed in config["training"]["seeds"]:
        source_row = next(
            row
            for row in baseline_metrics
            if row["variant"] == "D" and int(row["seed"]) == seed
        )
        source_checkpoint = NORMALIZED_ARTIFACT_DIR / source_row["checkpoint_path"]
        if _sha256(source_checkpoint) != source_row["checkpoint_sha256"]:
            raise ValueError(f"Frozen CNN1D-D checkpoint hash mismatch for seed {seed}.")
        checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("normalization_parameters_sha256")
            != config["normalization_parameters_sha256"]
            or checkpoint.get("frozen_dataset_manifest_sha256")
            != config["dataset_manifest_sha256"]
            or checkpoint.get("target_transform")
            != normalization_bundle["variants"]["D"]["target"]
            or checkpoint.get("best_epoch") != int(source_row["best_epoch"])
        ):
            raise ValueError(f"Frozen CNN1D-D checkpoint provenance mismatch for {seed}.")
        model = build_architecture("CNN1D").to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        copied = ARTIFACT_DIR / "checkpoints" / "CNN1D" / f"seed_{seed}_best.pt"
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_checkpoint, copied)
        if _sha256(copied) != _sha256(source_checkpoint):
            raise ValueError("Copied CNN1D D checkpoint checksum mismatch.")

        per_seed_saved = [
            row
            for row in baseline_predictions
            if row["variant"] == "D" and int(row["seed"]) == seed
        ]
        if any(row["partition"] not in {"train", "validation"} for row in per_seed_saved):
            raise ValueError("Variant D baseline predictions contain non-internal rows.")
        max_delta = 0.0
        record: dict[str, Any] = {}
        for partition in ("train", "validation"):
            expected_rows = [row for row in per_seed_saved if row["partition"] == partition]
            expected_by_key = {
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                : row
                for row in expected_rows
            }
            actual_keys = [
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                for row in rows[partition]
            ]
            if (
                len(expected_by_key) != len(expected_rows)
                or len(set(actual_keys)) != len(actual_keys)
                or set(expected_by_key) != set(actual_keys)
            ):
                raise ValueError(f"CNN1D D saved {partition} keys are misaligned.")
            expected_rows = [expected_by_key[key] for key in actual_keys]
            inputs_np = _prepare_inputs(data[partition], scalers[0], scalers[1])
            prediction = _predict_percent(
                model, _tensor_inputs(inputs_np), target_scaler, device
            )
            saved = np.asarray(
                [float(row["predicted_soh_percent"]) for row in expected_rows],
                dtype=np.float32,
            )
            delta = np.abs(prediction - saved)
            current_delta = float(delta.max(initial=0.0))
            max_delta = max(max_delta, current_delta)
            if current_delta > 1e-6:
                raise ValueError(
                    f"CNN1D D prediction mismatch for seed {seed}/{partition}: "
                    f"{current_delta}."
                )
            target = data[partition]["y_soh"].reshape(-1)
            metric = _metrics(target, prediction)
            record.update({f"{partition}_{key}": value for key, value in metric.items()})
            prediction_rows.extend(
                _prediction_rows(
                    "CNN1D",
                    seed,
                    partition,
                    rows[partition],
                    target,
                    prediction,
                )
            )
        run_rows.append(
            {
                "architecture": "CNN1D",
                "seed": seed,
                "status": "reused_variant_D_checkpoint",
                "parameter_count": int(source_row["parameter_count"]),
                "best_epoch": int(source_row["best_epoch"]),
                "epochs_completed": int(source_row["epochs_completed"]),
                "training_duration_seconds": float(
                    source_row["training_duration_seconds"]
                ),
                "train_mse_loss_normalized_target": float(
                    source_row["train_mse_loss_target_space"]
                ),
                "validation_mse_loss_normalized_target": float(
                    source_row["validation_mse_loss_target_space"]
                ),
                **record,
                "checkpoint_path": str(copied.relative_to(ARTIFACT_DIR)),
                "checkpoint_sha256": _sha256(copied),
                "source_checkpoint_path": str(
                    source_checkpoint.relative_to(PROJECT_ROOT)
                ),
                "source_checkpoint_sha256": _sha256(source_checkpoint),
                "preprocessing_prediction_max_abs_delta": max_delta,
                "normalization_parameters_sha256": _sha256(normalizer_path),
            }
        )
        shutil.copy2(
            NORMALIZED_ARTIFACT_DIR
            / "histories"
            / f"D_seed_{seed}.csv",
            ARTIFACT_DIR / "histories" / f"CNN1D_seed_{seed}.csv",
        )
    return run_rows, prediction_rows


def _train_one(
    architecture: str,
    seed: int,
    config: dict[str, Any],
    data: dict[str, Any],
    rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
    config_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training = config["training"]
    discharge_scaler, charge_scaler, target_scaler = scalers
    generator = _seed_everything(seed)
    train_inputs_np = _prepare_inputs(data["train"], discharge_scaler, charge_scaler)
    validation_inputs_np = _prepare_inputs(
        data["validation"], discharge_scaler, charge_scaler
    )
    train_inputs = _tensor_inputs(train_inputs_np)
    validation_inputs = _tensor_inputs(validation_inputs_np)
    train_target_normalized = target_scaler.transform(
        data["train"]["y_soh"]
    ).reshape(-1)
    validation_target_normalized = target_scaler.transform(
        data["validation"]["y_soh"]
    ).reshape(-1)
    train_y_tensor = torch.from_numpy(train_target_normalized)
    loader = DataLoader(
        TensorDataset(
            train_inputs["X_discharge"],
            train_inputs["X_charge"],
            train_inputs["X_physics"],
            train_y_tensor,
        ),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = build_architecture(architecture).to(device)
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
    histories: list[dict[str, Any]] = []
    started = time.perf_counter()
    y_validation = data["validation"]["y_soh"].reshape(-1)

    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        count = 0
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
                    f"Non-finite {architecture} training loss at epoch {epoch}."
                )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            count += len(target)
        train_loss = loss_sum / count

        model.eval()
        with torch.inference_mode():
            validation_prediction_normalized = model(
                validation_inputs["X_discharge"].to(device),
                validation_inputs["X_charge"].to(device),
                validation_inputs["X_physics"].to(device),
            ).detach().cpu().numpy()
        validation_prediction_percent = target_scaler.inverse_transform(
            validation_prediction_normalized
        ).reshape(-1)
        validation_loss = float(
            np.mean(
                (
                    validation_prediction_normalized.reshape(-1)
                    - validation_target_normalized
                )
                ** 2
            )
        )
        validation_metric = _metrics(y_validation, validation_prediction_percent)
        validation_rmse = validation_metric["RMSE"]
        histories.append(
            {
                "epoch": epoch,
                "train_mse_loss_normalized_target": train_loss,
                "validation_mse_loss_normalized_target": validation_loss,
                "validation_RMSE_soh_pp": validation_rmse,
            }
        )
        if validation_rmse < best_rmse:
            best_rmse = validation_rmse
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
        raise RuntimeError(f"No Validation checkpoint was selected for {architecture}.")
    checkpoint_path = (
        ARTIFACT_DIR / "checkpoints" / architecture / f"seed_{seed}_best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": architecture,
            "architecture_details": architecture_details(architecture),
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
            "normalization_parameters_reference": (
                f"{config['normalization_parameters']}#variants.D"
            ),
            "target_transform": _read_json(
                PROJECT_ROOT / config["normalization_parameters"]
            )["variants"]["D"]["target"],
            "experiment_config_sha256": config_sha256,
            "external_evaluation_performed": False,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    prediction_rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    for partition, inputs, targets, partition_rows in (
        (
            "train",
            train_inputs,
            data["train"]["y_soh"].reshape(-1),
            rows["train"],
        ),
        (
            "validation",
            validation_inputs,
            y_validation,
            rows["validation"],
        ),
    ):
        prediction_percent = _predict_percent(
            model, inputs, target_scaler, device
        )
        current_metrics = _metrics(targets, prediction_percent)
        metrics.update(
            {f"{partition}_{name}": value for name, value in current_metrics.items()}
        )
        prediction_rows.extend(
            _prediction_rows(
                architecture,
                seed,
                partition,
                partition_rows,
                targets,
                prediction_percent,
            )
        )

    duration = time.perf_counter() - started
    history_path = ARTIFACT_DIR / "histories" / f"{architecture}_seed_{seed}.csv"
    _write_csv(history_path, histories)
    best_history = histories[best_epoch - 1]
    run_row = {
        "architecture": architecture,
        "seed": seed,
        "status": "trained",
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(histories),
        "training_duration_seconds": duration,
        "train_mse_loss_normalized_target": best_history[
            "train_mse_loss_normalized_target"
        ],
        "validation_mse_loss_normalized_target": best_history[
            "validation_mse_loss_normalized_target"
        ],
        **metrics,
        "checkpoint_path": str(checkpoint_path.relative_to(ARTIFACT_DIR)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_path": "",
        "source_checkpoint_sha256": "",
        "preprocessing_prediction_max_abs_delta": "",
        "normalization_parameters_sha256": config[
            "normalization_parameters_sha256"
        ],
    }
    return run_row, prediction_rows, histories


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_ORDER:
        model_rows = [row for row in rows if row["architecture"] == architecture]
        if len(model_rows) != 3:
            raise ValueError(f"Expected 3 seeds for {architecture}.")
        counts = {int(row["parameter_count"]) for row in model_rows}
        if len(counts) != 1:
            raise ValueError(f"Inconsistent parameter counts for {architecture}.")
        summary: dict[str, Any] = {
            "architecture": architecture,
            "parameter_count": counts.pop(),
            "seed_count": len(model_rows),
        }
        for metric in (
            "train_MAE",
            "train_RMSE",
            "train_R2",
            "validation_MAE",
            "validation_RMSE",
            "validation_R2",
        ):
            values = [float(row[f"train_{metric[6:]}"] if metric.startswith("train_") else row[metric]) for row in model_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        results.append(summary)
    return results


def run() -> dict[str, Any]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {ARTIFACT_DIR}.")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {REPORT_PATH}.")
    config = _read_json(CONFIG_PATH)
    if config.get("dataset_manifest_sha256") != EXPECTED_DATASET_SHA:
        raise ValueError("Unexpected P5 frozen manifest hash.")
    if config.get("normalization_parameters_sha256") != EXPECTED_NORMALIZATION_SHA:
        raise ValueError("Unexpected Variant D scaler hash.")
    if config.get("architectures") != list(ARCHITECTURE_ORDER):
        raise ValueError("Architecture order or cohort differs from the request.")
    training = config["training"]
    if training != {
        "seeds": [42, 123, 2026],
        "max_epochs": 300,
        "batch_size": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "MSELoss",
        "target_space": "normalized SoH using Variant D Train target statistics",
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.0,
        "checkpoint_metric": "validation_RMSE_soh_percentage_points",
        "device": "cuda:0",
    }:
        raise ValueError("Training protocol differs from the user-approved settings.")
    if config.get("external_evaluation", {}).get("authorized") is not True:
        raise ValueError("External evaluation authorization is missing.")
    if config.get("scope", {}).get("normalization_refit_authorized") is not False:
        raise ValueError("Variant D normalization must remain frozen.")

    data, dataset_info = _load_frozen_training_data(config)
    normalization_bundle, discharge, charge, target = _load_variant_d(config)
    scalers = (discharge, charge, target)
    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    rows = {
        partition: sorted(
            (row for row in split_rows if row["partition"] == partition),
            key=lambda row: int(row["split_row_index"]),
        )
        for partition in ("train", "validation")
    }
    transformed_train = _prepare_inputs(data["train"], discharge, charge)
    transformed_validation = _prepare_inputs(data["validation"], discharge, charge)
    for arrays in (transformed_train, transformed_validation):
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Variant D input contains NaN/Inf.")
    if target.transform(data["train"]["y_soh"]).shape != (102, 1):
        raise ValueError("Variant D target transform returned an unexpected shape.")

    device, environment = _configure_device()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    for folder in ("checkpoints", "histories"):
        (ARTIFACT_DIR / folder).mkdir()
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
            "dataset_identifier": config["dataset_identifier"],
            "dataset_manifest_sha256": config["dataset_manifest_sha256"],
            **dataset_info,
            "train_shapes": {
                key: list(value.shape) for key, value in data["train"].items()
            },
            "validation_shapes": {
                key: list(value.shape) for key, value in data["validation"].items()
            },
            "normalization_parameters_sha256": _sha256(normalizer_path),
            "normalization_fit_partition": "train",
            "normalization_fit_row_count": 102,
            "physics_features_refit": False,
        },
    )
    source_hashes = {
        "benchmark_config_sha256": config_sha256,
        "architecture_models_sha256": _sha256(EXPERIMENT_DIR / "architecture_models.py"),
        "p5_model_implementation_sha256": _sha256(
            PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark_models.py"
        ),
        "normalization_implementation_sha256": _sha256(
            PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_normalization.py"
        ),
        "normalization_parameters_sha256": _sha256(normalizer_path),
        "dataset_manifest_sha256": _sha256(
            DATASET_DIR / "bafuse_p5_sha256_manifest.json"
        ),
    }
    _write_json(ARTIFACT_DIR / "source_hashes.json", source_hashes)
    _write_json(
        ARTIFACT_DIR / "architecture_parameter_counts.json",
        {
            architecture: {
                "parameter_count": trainable_parameter_count(
                    build_architecture(architecture)
                ),
                "details": architecture_details(architecture),
            }
            for architecture in ARCHITECTURE_ORDER
        },
    )

    baseline_metrics, baseline_predictions = _reuse_cnn1d_d(
        config,
        data,
        dataset_info,
        rows,
        scalers,
        device,
        normalization_bundle,
        normalizer_path,
    )
    run_rows = list(baseline_metrics)
    all_predictions = list(baseline_predictions)
    for architecture in ("LSTM", "GRU", "TCN", "CNN-LSTM"):
        for seed in training["seeds"]:
            logging.info("START %s seed %s", architecture, seed)
            run_row, predictions, _ = _train_one(
                architecture,
                int(seed),
                config,
                data,
                rows,
                scalers,
                device,
                config_sha256,
            )
            run_rows.append(run_row)
            all_predictions.extend(predictions)
            logging.info(
                "COMPLETE %s seed %s best_epoch=%s val_RMSE=%.4f pp",
                architecture,
                seed,
                run_row["best_epoch"],
                run_row["validation_RMSE"],
            )
    run_rows.sort(key=lambda row: (ARCHITECTURE_ORDER.index(row["architecture"]), row["seed"]))
    _write_csv(ARTIFACT_DIR / "run_metrics.csv", run_rows)
    summaries = _summary(run_rows)
    _write_csv(ARTIFACT_DIR / "train_validation_summary.csv", summaries)
    all_predictions.sort(
        key=lambda row: (
            ARCHITECTURE_ORDER.index(row["architecture"]),
            int(row["seed"]),
            row["partition"],
            int(row["rpt_index"]),
        )
    )
    _write_csv(ARTIFACT_DIR / "train_validation_predictions.csv", all_predictions)

    checkpoints: list[dict[str, Any]] = []
    for row in run_rows:
        checkpoint_path = ARTIFACT_DIR / row["checkpoint_path"]
        if _sha256(checkpoint_path) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint changed before freeze: {row['checkpoint_path']}.")
        checkpoints.append(
            {
                "architecture": row["architecture"],
                "seed": int(row["seed"]),
                "path": row["checkpoint_path"].replace("\\", "/"),
                "sha256": row["checkpoint_sha256"],
                "parameter_count": int(row["parameter_count"]),
                "best_epoch": int(row["best_epoch"]),
                "validation_RMSE_soh_pp": float(row["validation_RMSE"]),
                "status": row["status"],
                "source_checkpoint_path": row["source_checkpoint_path"],
                "source_checkpoint_sha256": row["source_checkpoint_sha256"],
            }
        )
    if len(checkpoints) != 15:
        raise ValueError(f"Expected 15 frozen checkpoints, got {len(checkpoints)}.")
    logging.info("All 15 model/seed checkpoints frozen; External not loaded.")
    freeze = {
        "status": "complete",
        "frozen": True,
        "experiment_identifier": config["experiment_identifier"],
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "normalization_parameters_sha256": config[
            "normalization_parameters_sha256"
        ],
        "experiment_config_sha256": config_sha256,
        "checkpoint_count": len(checkpoints),
        "external_evaluation_performed": False,
        "external_evaluation_authorized": True,
        "checkpoints": checkpoints,
    }
    _write_json(ARTIFACT_DIR / "checkpoint_freeze_manifest.json", freeze)
    training_files = {
        str(path.relative_to(ARTIFACT_DIR)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file()
        and path.name not in {"training_stage_sha256_manifest.json"}
    }
    _write_json(
        ARTIFACT_DIR / "training_stage_sha256_manifest.json",
        {
            "hash_scope": "Training-stage artifacts after checkpoint freeze.",
            "files": training_files,
        },
    )
    return freeze


if __name__ == "__main__":
    run()
