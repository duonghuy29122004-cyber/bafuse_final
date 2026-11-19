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
sys.path.insert(
    0,
    str(
        PROJECT_ROOT
        / "experiments"
        / "soh_forecast"
        / "bafuse_p5_architecture_comparison_v1"
    ),
)

from architecture_models import (  # noqa: E402
    build_architecture as build_tcn_3m_model,
    trainable_parameter_count as count_tcn_3m_parameters,
)
from bafuse.hybrid.p5_normalization import PopulationZScore  # noqa: E402
from tcn_2m_model import (  # noqa: E402
    architecture_details as tcn_2m_details,
    build_tcn_2m_model,
    trainable_parameter_count as count_tcn_2m_parameters,
)


CONFIG_PATH = EXPERIMENT_DIR / "benchmark_config.json"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_TCN_TWO_MODALITY_ABLATION.md"
DATASET_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_electrical_v1_20261114"
)
NORMALIZED_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_p5_normalized_v1"
    / "artifacts"
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
MODEL_ORDER = ("TCN-2M", "TCN-3M")


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
        raise ValueError(f"Refusing to write an empty CSV: {path}.")
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
        raise ValueError("Metric vectors must be non-empty and aligned.")
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


def _load_npz(filename: str, expected_n: int) -> dict[str, np.ndarray]:
    if filename == "bafuse_external_p5.npz":
        raise ValueError("The training stage must not load External arrays.")
    path = DATASET_DIR / filename
    with np.load(path, allow_pickle=False) as archive:
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


def _load_frozen_internal_data(
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
                raise ValueError(f"Frozen physical-cell split overlaps: {first}/{second}.")

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
            raise ValueError(f"{partition} split keys do not match frozen array rows.")

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
        "tcn_2m_physics_input_dependency": False,
        "physics_arrays_used_only_for_tcn_3m_comparison": True,
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
        raise ValueError("Frozen Variant D scalers are missing.")
    discharge = PopulationZScore.from_dict(variant["discharge"])
    charge = PopulationZScore.from_dict(variant["charge"])
    target = PopulationZScore.from_dict(variant["target"])
    if (
        tuple(scaler.mean.size for scaler in (discharge, charge, target))
        != (128, 128, 1)
        or any(
            scaler.fit_row_count != 102
            for scaler in (discharge, charge, target)
        )
    ):
        raise ValueError("Frozen Variant D scaler shape or fit count changed.")
    return bundle, discharge, charge, target


def _prepare_2m_inputs(
    arrays: dict[str, np.ndarray],
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    transformed = {
        "X_discharge": discharge_scaler.transform(arrays["X_discharge"]),
        "X_charge": charge_scaler.transform(arrays["X_charge"]),
    }
    if any(not np.isfinite(value).all() for value in transformed.values()):
        raise ValueError("TCN-2M voltage preprocessing produced NaN/Inf.")
    return transformed


def _prepare_3m_inputs(
    arrays: dict[str, np.ndarray],
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    transformed = _prepare_2m_inputs(arrays, discharge_scaler, charge_scaler)
    transformed["X_physics"] = np.array(arrays["X_physics"], copy=True)
    if not np.isfinite(transformed["X_physics"]).all():
        raise ValueError("Frozen TCN-3M Physics values contain NaN/Inf.")
    return transformed


def _to_tensor_inputs(inputs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(value.astype(np.float32, copy=False))
        for name, value in inputs.items()
    }


def _predict_2m(
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
        ).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("TCN-2M produced non-finite SoH predictions.")
    return prediction


def _predict_3m(
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
        raise ValueError("TCN-3M produced non-finite SoH predictions.")
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


def _baseline_files_and_entries() -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[tuple[str, int], dict[str, str]],
]:
    freeze_path = BASELINE_DIR / "checkpoint_freeze_manifest.json"
    training_manifest_path = BASELINE_DIR / "training_stage_sha256_manifest.json"
    freeze = _read_json(freeze_path)
    training_manifest = _read_json(training_manifest_path)
    if freeze.get("status") != "complete" or freeze.get("frozen") is not True:
        raise ValueError("Existing TCN-3M checkpoint freeze is incomplete.")
    if (
        freeze.get("dataset_manifest_sha256") != EXPECTED_DATASET_SHA
        or freeze.get("normalization_parameters_sha256") != EXPECTED_NORMALIZATION_SHA
    ):
        raise ValueError("Existing TCN-3M checkpoints use different data/preprocessing.")
    for relative, expected in training_manifest["files"].items():
        path = BASELINE_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Existing TCN-3M training artifact changed: {relative}.")

    entries = [
        entry for entry in freeze["checkpoints"]
        if entry["architecture"] == "TCN"
    ]
    if {int(entry["seed"]) for entry in entries} != set(SEEDS):
        raise ValueError("Existing TCN-3M freeze lacks one or more requested seeds.")
    metrics_by_seed: dict[tuple[str, int], dict[str, str]] = {}
    for row in _read_csv(BASELINE_DIR / "run_metrics.csv"):
        if row["architecture"] == "TCN":
            metrics_by_seed[("TCN", int(row["seed"]))] = row
    if set(metrics_by_seed) != {("TCN", seed) for seed in SEEDS}:
        raise ValueError("Existing TCN-3M metrics lack one or more requested seeds.")
    for entry in entries:
        source = BASELINE_DIR / entry["path"]
        if not source.is_file() or _sha256(source) != entry["sha256"]:
            raise ValueError(f"Existing TCN-3M checkpoint changed: {entry['path']}.")
        row = metrics_by_seed[("TCN", int(entry["seed"]))]
        if row["checkpoint_sha256"] != entry["sha256"]:
            raise ValueError("TCN-3M checkpoint hash differs from its frozen metrics.")
    return freeze, entries, metrics_by_seed


def _reuse_tcn_3m(
    config: dict[str, Any],
    data: dict[str, dict[str, np.ndarray]],
    sample_rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
    baseline_freeze: dict[str, Any],
    baseline_entries: list[dict[str, Any]],
    baseline_metrics: dict[tuple[str, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_predictions = _read_csv(BASELINE_DIR / "train_validation_predictions.csv")
    run_rows: list[dict[str, Any]] = []
    all_predictions: list[dict[str, Any]] = []
    entries_by_seed = {int(entry["seed"]): entry for entry in baseline_entries}
    discharge_scaler, charge_scaler, target_scaler = scalers
    for seed in SEEDS:
        entry = entries_by_seed[seed]
        source_checkpoint = BASELINE_DIR / entry["path"]
        checkpoint = torch.load(
            source_checkpoint, map_location="cpu", weights_only=True
        )
        if (
            checkpoint.get("architecture") != "TCN"
            or checkpoint.get("seed") != seed
            or checkpoint.get("parameter_count") != int(entry["parameter_count"])
            or checkpoint.get("normalization_parameters_sha256")
            != config["normalization_parameters_sha256"]
            or checkpoint.get(
                "dataset_manifest_sha256",
                checkpoint.get("frozen_dataset_manifest_sha256"),
            )
            != config["dataset_manifest_sha256"]
            or checkpoint.get("target_transform")
            != _read_json(PROJECT_ROOT / config["normalization_parameters"])[
                "variants"
            ]["D"]["target"]
        ):
            raise ValueError(f"TCN-3M checkpoint provenance mismatch for seed {seed}.")
        model = build_tcn_3m_model("TCN").to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if count_tcn_3m_parameters(model) != int(entry["parameter_count"]):
            raise ValueError("TCN-3M parameter count differs from the frozen baseline.")

        per_seed_saved = [
            row
            for row in source_predictions
            if row["architecture"] == "TCN" and int(row["seed"]) == seed
        ]
        max_delta = 0.0
        metrics_for_seed: dict[str, Any] = {}
        for partition in ("train", "validation"):
            expected_rows = [
                row for row in per_seed_saved if row["partition"] == partition
            ]
            expected_by_key = {
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
                for row in expected_rows
            }
            actual_keys = [
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                for row in sample_rows[partition]
            ]
            if (
                len(expected_by_key) != len(expected_rows)
                or set(expected_by_key) != set(actual_keys)
                or len(set(actual_keys)) != len(actual_keys)
            ):
                raise ValueError(
                    f"TCN-3M saved {partition} keys do not match frozen P5 keys."
                )
            aligned_saved = [expected_by_key[key] for key in actual_keys]
            inputs = _to_tensor_inputs(
                _prepare_3m_inputs(data[partition], discharge_scaler, charge_scaler)
            )
            prediction = _predict_3m(model, inputs, target_scaler, device)
            saved_prediction = np.asarray(
                [float(row["predicted_soh_percent"]) for row in aligned_saved],
                dtype=np.float32,
            )
            current_delta = float(
                np.max(np.abs(prediction - saved_prediction), initial=0.0)
            )
            max_delta = max(max_delta, current_delta)
            if current_delta > 1e-6:
                raise ValueError(
                    f"TCN-3M prediction parity failed for seed {seed}/{partition}: "
                    f"{current_delta}."
                )
            target = data[partition]["y_soh"].reshape(-1)
            metrics = _metrics(target, prediction)
            baseline_row = baseline_metrics[("TCN", seed)]
            for metric_name in ("MAE", "RMSE", "R2"):
                expected_metric = float(baseline_row[f"{partition}_{metric_name}"])
                if abs(metrics[metric_name] - expected_metric) > 1e-6:
                    raise ValueError(
                        f"TCN-3M {partition} {metric_name} mismatch for seed {seed}."
                    )
            metrics_for_seed.update(
                {f"{partition}_{name}": value for name, value in metrics.items()}
            )
            all_predictions.extend(
                _prediction_rows(
                    "TCN-3M",
                    seed,
                    partition,
                    sample_rows[partition],
                    target,
                    prediction,
                )
            )
        baseline_row = baseline_metrics[("TCN", seed)]
        run_rows.append(
            {
                "model": "TCN-3M",
                "seed": seed,
                "status": "reused_existing_checkpoint",
                "parameter_count": int(entry["parameter_count"]),
                "best_epoch": int(entry["best_epoch"]),
                "epochs_completed": int(baseline_row["epochs_completed"]),
                "training_duration_seconds": float(
                    baseline_row["training_duration_seconds"]
                ),
                "train_mse_loss_normalized_target": float(
                    baseline_row["train_mse_loss_normalized_target"]
                ),
                "validation_mse_loss_normalized_target": float(
                    baseline_row["validation_mse_loss_normalized_target"]
                ),
                **metrics_for_seed,
                "checkpoint_path": str(
                    source_checkpoint.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "checkpoint_sha256": entry["sha256"],
                "source_checkpoint_path": str(
                    source_checkpoint.relative_to(PROJECT_ROOT)
                ).replace("\\", "/"),
                "source_checkpoint_sha256": entry["sha256"],
                "preprocessing_prediction_max_abs_delta": max_delta,
                "normalization_parameters_sha256": config[
                    "normalization_parameters_sha256"
                ],
            }
        )
        source_history = BASELINE_DIR / "histories" / f"TCN_seed_{seed}.csv"
        if not source_history.is_file():
            raise FileNotFoundError(f"TCN-3M history is missing for seed {seed}.")
        shutil.copy2(
            source_history,
            ARTIFACT_DIR / "histories" / f"TCN-3M_seed_{seed}.csv",
        )
    return run_rows, all_predictions, [
        {
            "model": "TCN-3M",
            "seed": seed,
            "source_checkpoint_path": str(
                (BASELINE_DIR / entries_by_seed[seed]["path"]).relative_to(
                    PROJECT_ROOT
                )
            ).replace("\\", "/"),
            "source_checkpoint_sha256": entries_by_seed[seed]["sha256"],
            "source_freeze_manifest_sha256": _sha256(
                BASELINE_DIR / "checkpoint_freeze_manifest.json"
            ),
            "preprocessing_prediction_max_abs_delta": next(
                row["preprocessing_prediction_max_abs_delta"]
                for row in run_rows
                if int(row["seed"]) == seed
            ),
        }
        for seed in SEEDS
    ]


def _train_one(
    seed: int,
    config: dict[str, Any],
    data: dict[str, dict[str, np.ndarray]],
    sample_rows: dict[str, list[dict[str, str]]],
    scalers: tuple[PopulationZScore, PopulationZScore, PopulationZScore],
    device: torch.device,
    config_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    training = config["training"]
    discharge_scaler, charge_scaler, target_scaler = scalers
    generator = _seed_everything(seed)
    train_inputs = _to_tensor_inputs(
        _prepare_2m_inputs(data["train"], discharge_scaler, charge_scaler)
    )
    validation_inputs = _to_tensor_inputs(
        _prepare_2m_inputs(data["validation"], discharge_scaler, charge_scaler)
    )
    train_target_normalized = target_scaler.transform(
        data["train"]["y_soh"]
    ).reshape(-1)
    validation_target_normalized = target_scaler.transform(
        data["validation"]["y_soh"]
    ).reshape(-1)
    train_loader = DataLoader(
        TensorDataset(
            train_inputs["X_discharge"],
            train_inputs["X_charge"],
            torch.from_numpy(train_target_normalized),
        ),
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = build_tcn_2m_model().to(device)
    parameter_count = count_tcn_2m_parameters(model)
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
        for discharge, charge, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            prediction = model(discharge.to(device), charge.to(device)).squeeze(1)
            loss = nn.functional.mse_loss(prediction, target.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite TCN-2M loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(target)
            sample_count += len(target)
        train_loss = loss_sum / sample_count
        model.eval()
        with torch.inference_mode():
            validation_prediction_normalized = model(
                validation_inputs["X_discharge"].to(device),
                validation_inputs["X_charge"].to(device),
            ).detach().cpu().numpy()
        validation_loss = float(
            np.mean(
                (
                    validation_prediction_normalized.reshape(-1)
                    - validation_target_normalized
                )
                ** 2
            )
        )
        validation_prediction_percent = target_scaler.inverse_transform(
            validation_prediction_normalized
        ).reshape(-1)
        validation_metrics = _metrics(
            validation_target, validation_prediction_percent
        )
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
        raise RuntimeError("No Validation-selected TCN-2M checkpoint was produced.")
    checkpoint_path = (
        ARTIFACT_DIR / "checkpoints" / "TCN-2M" / f"seed_{seed}_best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": best_state,
            "model": "TCN-2M",
            "architecture": tcn_2m_details(),
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
            "physics_input_used": False,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()

    metric_fields: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    for partition in ("train", "validation"):
        inputs = train_inputs if partition == "train" else validation_inputs
        target = data[partition]["y_soh"].reshape(-1)
        prediction = _predict_2m(model, inputs, target_scaler, device)
        metrics = _metrics(target, prediction)
        metric_fields.update(
            {f"{partition}_{name}": value for name, value in metrics.items()}
        )
        prediction_rows.extend(
            _prediction_rows(
                "TCN-2M",
                seed,
                partition,
                sample_rows[partition],
                target,
                prediction,
            )
        )

    duration = time.perf_counter() - started
    history_path = ARTIFACT_DIR / "histories" / f"TCN-2M_seed_{seed}.csv"
    _write_csv(history_path, history)
    best_history = history[best_epoch - 1]
    row = {
        "model": "TCN-2M",
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
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "source_checkpoint_path": "",
        "source_checkpoint_sha256": "",
        "preprocessing_prediction_max_abs_delta": "",
        "normalization_parameters_sha256": config[
            "normalization_parameters_sha256"
        ],
    }
    return row, prediction_rows, history


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        model_rows = [row for row in rows if row["model"] == model_name]
        if len(model_rows) != len(SEEDS):
            raise ValueError(f"Expected three seeds for {model_name}.")
        parameter_counts = {int(row["parameter_count"]) for row in model_rows}
        if len(parameter_counts) != 1:
            raise ValueError(f"Inconsistent parameter count for {model_name}.")
        summary: dict[str, Any] = {
            "model": model_name,
            "parameter_count": parameter_counts.pop(),
            "seed_count": len(model_rows),
        }
        for partition in ("train", "validation"):
            for metric in ("MAE", "RMSE", "R2"):
                values = [
                    float(row[f"{partition}_{metric}"]) for row in model_rows
                ]
                summary[f"{partition}_{metric}_mean"] = statistics.mean(values)
                summary[f"{partition}_{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def run() -> dict[str, Any]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {ARTIFACT_DIR}.")
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite {REPORT_PATH}.")
    config = _read_json(CONFIG_PATH)
    if (
        config.get("dataset_manifest_sha256") != EXPECTED_DATASET_SHA
        or config.get("normalization_parameters_sha256") != EXPECTED_NORMALIZATION_SHA
    ):
        raise ValueError("Unexpected frozen dataset or Variant D checksum.")
    training = config["training"]
    expected_training = {
        "seeds": list(SEEDS),
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
    if training != expected_training:
        raise ValueError("Training protocol differs from the requested ablation.")
    if config.get("external_evaluation", {}).get("authorized") is not True:
        raise ValueError("External evaluation authorization is missing.")
    if config.get("scope", {}).get("normalization_refit_authorized") is not False:
        raise ValueError("Variant D normalization must not be refitted.")

    data, sample_rows, dataset_info = _load_frozen_internal_data(config)
    normalization_bundle, discharge, charge, target = _load_variant_d(config)
    scalers = (discharge, charge, target)
    for partition in ("train", "validation"):
        transformed = _prepare_2m_inputs(data[partition], discharge, charge)
        if not all(np.isfinite(value).all() for value in transformed.values()):
            raise ValueError(f"Invalid normalized {partition} inputs.")
        if target.transform(data[partition]["y_soh"]).shape != (
            EXPECTED_COUNTS[partition],
            1,
        ):
            raise ValueError(f"Unexpected normalized target shape for {partition}.")

    baseline_freeze, baseline_entries, baseline_metrics = _baseline_files_and_entries()
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
            "tcn_2m_model_physics_dependency": False,
        },
    )
    source_hashes = {
        "benchmark_config_sha256": config_sha256,
        "tcn_2m_model_sha256": _sha256(EXPERIMENT_DIR / "tcn_2m_model.py"),
        "training_script_sha256": _sha256(EXPERIMENT_DIR / "train_tcn_2m.py"),
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
        "tcn_3m_checkpoint_freeze_manifest_sha256": _sha256(
            BASELINE_DIR / "checkpoint_freeze_manifest.json"
        ),
        "tcn_3m_training_stage_manifest_sha256": _sha256(
            BASELINE_DIR / "training_stage_sha256_manifest.json"
        ),
    }
    _write_json(ARTIFACT_DIR / "source_hashes.json", source_hashes)
    _write_json(
        ARTIFACT_DIR / "model_architecture.json",
        {
            "TCN-2M": {
                "parameter_count": count_tcn_2m_parameters(build_tcn_2m_model()),
                "details": tcn_2m_details(),
            },
            "TCN-3M": {
                "parameter_count": count_tcn_3m_parameters(
                    build_tcn_3m_model("TCN")
                ),
                "details": {
                    "source": "Existing frozen P5 TCN-3M implementation and checkpoints.",
                    "architecture_config": _read_json(
                        BASELINE_DIR / "architecture_parameter_counts.json"
                    )["TCN"]["details"],
                },
            },
        },
    )
    _write_json(
        ARTIFACT_DIR / "baseline_checkpoint_verification.json",
        {
            "baseline_freeze_manifest_sha256": _sha256(
                BASELINE_DIR / "checkpoint_freeze_manifest.json"
            ),
            "baseline_training_stage_manifest_sha256": _sha256(
                BASELINE_DIR / "training_stage_sha256_manifest.json"
            ),
            "tcn_3m_entries": [
                {
                    "seed": int(entry["seed"]),
                    "path": entry["path"],
                    "sha256": entry["sha256"],
                    "parameter_count": int(entry["parameter_count"]),
                    "best_epoch": int(entry["best_epoch"]),
                }
                for entry in baseline_entries
            ],
            "external_predictions_or_metrics_read": False,
        },
    )

    baseline_rows, baseline_predictions, baseline_references = _reuse_tcn_3m(
        config,
        data,
        sample_rows,
        scalers,
        device,
        baseline_freeze,
        baseline_entries,
        baseline_metrics,
    )
    run_rows = list(baseline_rows)
    prediction_rows = list(baseline_predictions)
    for seed in SEEDS:
        logging.info("START TCN-2M seed %s", seed)
        row, predictions, _ = _train_one(
            seed,
            config,
            data,
            sample_rows,
            scalers,
            device,
            config_sha256,
        )
        run_rows.append(row)
        prediction_rows.extend(predictions)
        logging.info(
            "COMPLETE TCN-2M seed %s best_epoch=%s validation_RMSE=%.4f pp",
            seed,
            row["best_epoch"],
            row["validation_RMSE"],
        )

    run_rows.sort(key=lambda row: (MODEL_ORDER.index(row["model"]), int(row["seed"])))
    prediction_rows.sort(
        key=lambda row: (
            MODEL_ORDER.index(row["model"]),
            int(row["seed"]),
            row["partition"],
            int(row["rpt_index"]),
        )
    )
    _write_csv(ARTIFACT_DIR / "run_metrics.csv", run_rows)
    _write_csv(ARTIFACT_DIR / "train_validation_summary.csv", _summaries(run_rows))
    _write_csv(ARTIFACT_DIR / "train_validation_predictions.csv", prediction_rows)
    _write_csv(ARTIFACT_DIR / "baseline_checkpoint_references.csv", baseline_references)

    checkpoint_entries: list[dict[str, Any]] = []
    for row in run_rows:
        if row["model"] == "TCN-2M":
            path = PROJECT_ROOT / row["checkpoint_path"]
            if _sha256(path) != row["checkpoint_sha256"]:
                raise ValueError(f"TCN-2M checkpoint hash mismatch: {path}.")
            checkpoint_entries.append(
                {
                    "model": "TCN-2M",
                    "seed": int(row["seed"]),
                    "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                    "path_scope": "project_root",
                    "sha256": row["checkpoint_sha256"],
                    "source": "newly_trained",
                    "parameter_count": int(row["parameter_count"]),
                    "best_epoch": int(row["best_epoch"]),
                    "validation_RMSE_soh_pp": float(row["validation_RMSE"]),
                }
            )
    for entry in baseline_entries:
        source_checkpoint = BASELINE_DIR / entry["path"]
        checkpoint_entries.append(
            {
                "model": "TCN-3M",
                "seed": int(entry["seed"]),
                "path": str(source_checkpoint.relative_to(PROJECT_ROOT)).replace(
                    "\\", "/"
                ),
                "path_scope": "project_root",
                "sha256": entry["sha256"],
                "source": "previously_frozen_baseline",
                "parameter_count": int(entry["parameter_count"]),
                "best_epoch": int(entry["best_epoch"]),
                "validation_RMSE_soh_pp": float(entry["validation_RMSE_soh_pp"]),
            }
        )
    if len(checkpoint_entries) != 6:
        raise ValueError("Expected three TCN-2M and three TCN-3M checkpoints.")
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
        "checkpoint_count": len(checkpoint_entries),
        "tcn_2m_checkpoint_count": 3,
        "tcn_3m_baseline_checkpoint_count": 3,
        "external_evaluation_performed": False,
        "external_evaluation_authorized": True,
        "external_used_for_training_or_selection": False,
        "checkpoints": checkpoint_entries,
        "baseline_freeze_manifest_sha256": _sha256(
            BASELINE_DIR / "checkpoint_freeze_manifest.json"
        ),
    }
    _write_json(ARTIFACT_DIR / "checkpoint_freeze_manifest.json", freeze)
    logging.info("TCN-2M checkpoints frozen; External data has not been loaded.")
    training_files = {
        str(path.relative_to(ARTIFACT_DIR)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "training_stage_sha256_manifest.json"
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
