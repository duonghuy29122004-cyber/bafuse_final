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

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bafuse.hybrid.p5_benchmark_models import (  # noqa: E402
    architecture_config,
    build_model,
    trainable_parameter_count,
)
from bafuse.hybrid.p5_normalization import (  # noqa: E402
    PopulationZScore,
    apply_voltage_normalization,
    fit_population_zscore,
)


EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_NORMALIZATION_ABLATION.md"
CONFIG_PATH = EXPERIMENT_DIR / "experiment_config.json"
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
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_trainval_benchmark_v2_20261114"
)
EXPECTED_PARTITIONS = {"train": 102, "validation": 37, "external_test": 64}
VARIANT_ORDER = ("A", "B", "C", "D")
VOLTAGE_MINIMUM_STD = 1e-6
TARGET_MINIMUM_STD = 1e-8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


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
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _configure_determinism() -> dict[str, Any]:
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8."
        )
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available():
        raise RuntimeError("The configured experiment requires CUDA device 0.")
    return {
        "device": torch.cuda.get_device_name(0),
        "cuda_version": torch.version.cuda,
        "cublas_workspace_config": workspace,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "dataloader_num_workers": 0,
    }


def _seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def _metric_values(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction_values = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target_values.shape != prediction_values.shape or not len(target_values):
        raise ValueError("Metric arrays must have matching, non-empty shapes.")
    error = prediction_values - target_values
    denominator = float(np.sum((target_values - target_values.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "R2": (
            float(1.0 - np.sum(error**2) / denominator)
            if denominator > 0
            else float("nan")
        ),
    }


def _load_dataset() -> dict[str, Any]:
    manifest_path = DATASET_DIR / "bafuse_p5_sha256_manifest.json"
    if _sha256(manifest_path) != "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457":
        raise ValueError("Frozen P5 dataset manifest hash changed.")
    manifest = _read_json(manifest_path)
    if manifest.get("dataset_identifier") != "bafuse-p5-electrical-v1-20261114":
        raise ValueError("Unexpected frozen P5 dataset identifier.")

    verified_files: dict[str, str] = {}
    for name in (
        "bafuse_train_p5.npz",
        "bafuse_validation_p5.npz",
        "bafuse_p5_split_manifest.csv",
        "bafuse_soh_labels_frozen.csv",
        "bafuse_p5_preprocessing_config.json",
    ):
        entry = manifest.get("outputs", {}).get(name)
        if not isinstance(entry, dict):
            raise ValueError(f"Frozen dataset manifest lacks {name}.")
        path = DATASET_DIR / name
        actual = _sha256(path)
        if actual != entry.get("sha256"):
            raise ValueError(f"Frozen dataset artifact hash mismatch: {name}.")
        verified_files[name] = actual

    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    counts = Counter(row["partition"] for row in split_rows)
    if dict(counts) != EXPECTED_PARTITIONS or len(split_rows) != 203:
        raise ValueError(f"Unexpected frozen split counts: {dict(counts)}.")
    all_keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in split_rows
    ]
    if len(set(all_keys)) != len(all_keys):
        raise ValueError("Frozen split manifest contains duplicate observation keys.")
    cells_by_partition = {
        partition: {
            row["physical_cell_id"]
            for row in split_rows
            if row["partition"] == partition
        }
        for partition in EXPECTED_PARTITIONS
    }
    for index, first in enumerate(EXPECTED_PARTITIONS):
        for second in tuple(EXPECTED_PARTITIONS)[index + 1 :]:
            if cells_by_partition[first] & cells_by_partition[second]:
                raise ValueError(f"Physical-cell overlap between {first} and {second}.")

    train_rows = sorted(
        (row for row in split_rows if row["partition"] == "train"),
        key=lambda row: int(row["split_row_index"]),
    )
    validation_rows = sorted(
        (row for row in split_rows if row["partition"] == "validation"),
        key=lambda row: int(row["split_row_index"]),
    )
    label_rows = _read_csv(DATASET_DIR / "bafuse_soh_labels_frozen.csv")
    label_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in label_rows
    }
    if len(label_by_key) != len(label_rows) or set(label_by_key) != set(all_keys):
        raise ValueError("Frozen labels and split manifest do not have the same unique keys.")
    train_targets = _load_partition_npz("bafuse_train_p5.npz", 102)
    validation_targets = _load_partition_npz("bafuse_validation_p5.npz", 37)
    for partition, rows, arrays in (
        ("train", train_rows, train_targets),
        ("validation", validation_rows, validation_targets),
    ):
        expected = np.asarray(
            [
                float(label_by_key[
                    (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                ]["soh_percent"])
                for row in rows
            ],
            dtype=np.float32,
        ).reshape(-1, 1)
        if not np.array_equal(expected, arrays["y_soh"]):
            raise ValueError(f"{partition} labels do not align with frozen label keys.")
        for row in rows:
            key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            if label_by_key[key]["final_status"] != "ACCEPTED":
                raise ValueError(f"Non-accepted Train/Validation label: {key}.")
    return {
        "train": train_targets,
        "validation": validation_targets,
        "train_rows": train_rows,
        "validation_rows": validation_rows,
        "dataset_manifest_sha256": _sha256(manifest_path),
        "verified_internal_artifact_hashes": verified_files,
        "split_counts": dict(counts),
        "partition_cell_counts": {
            partition: len(cells)
            for partition, cells in cells_by_partition.items()
        },
        "external_npz_loaded": False,
        "external_targets_used": False,
        "external_feature_rows_used": False,
    }


def _load_partition_npz(filename: str, expected_count: int) -> dict[str, np.ndarray]:
    if filename == "bafuse_external_p5.npz":
        raise ValueError("External arrays are explicitly excluded from this experiment.")
    path = DATASET_DIR / filename
    with np.load(path, allow_pickle=False) as archive:
        expected_names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected_names:
            raise ValueError(f"Unexpected NPZ members in {filename}.")
        arrays = {name: archive[name].copy() for name in archive.files}
    shapes = {
        "X_discharge": (expected_count, 128),
        "X_charge": (expected_count, 128),
        "X_physics": (expected_count, 5),
        "y_soh": (expected_count, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != shapes[name]:
            raise ValueError(f"Unexpected dtype/shape for {filename}:{name}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite data in {filename}:{name}.")
    return arrays


def _fit_normalizers(train_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    discharge = fit_population_zscore(
        train_arrays["X_discharge"], minimum_std=VOLTAGE_MINIMUM_STD
    )
    charge = fit_population_zscore(
        train_arrays["X_charge"], minimum_std=VOLTAGE_MINIMUM_STD
    )
    target = fit_population_zscore(
        train_arrays["y_soh"], minimum_std=TARGET_MINIMUM_STD
    )
    return {
        "A": {
            "voltage_normalization": False,
            "target_normalization": False,
            "discharge": None,
            "charge": None,
            "target": None,
        },
        "B": {
            "voltage_normalization": True,
            "target_normalization": False,
            "discharge": discharge.to_dict(),
            "charge": charge.to_dict(),
            "target": None,
        },
        "C": {
            "voltage_normalization": False,
            "target_normalization": True,
            "discharge": None,
            "charge": None,
            "target": target.to_dict(),
        },
        "D": {
            "voltage_normalization": True,
            "target_normalization": True,
            "discharge": discharge.to_dict(),
            "charge": charge.to_dict(),
            "target": target.to_dict(),
        },
    }


def _scaler(value: dict[str, Any] | None) -> PopulationZScore | None:
    return PopulationZScore.from_dict(value) if value is not None else None


def _transform_inputs(
    arrays: dict[str, np.ndarray],
    parameters: dict[str, Any],
) -> dict[str, np.ndarray]:
    return apply_voltage_normalization(
        {name: arrays[name] for name in ("X_discharge", "X_charge", "X_physics")},
        discharge_scaler=_scaler(parameters["discharge"]),
        charge_scaler=_scaler(parameters["charge"]),
    )


def _transform_target(
    values: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    scaler = _scaler(parameters["target"])
    return np.array(values, copy=True) if scaler is None else scaler.transform(values)


def _inverse_target(
    values: np.ndarray, parameters: dict[str, Any]
) -> np.ndarray:
    scaler = _scaler(parameters["target"])
    return np.array(values, copy=True) if scaler is None else scaler.inverse_transform(values)


def _inputs_to_tensors(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        name: torch.from_numpy(arrays[name].astype(np.float32, copy=False))
        for name in ("X_discharge", "X_charge", "X_physics")
    }


def _predict_model(
    model: nn.Module,
    raw_arrays: dict[str, np.ndarray],
    parameters: dict[str, Any],
    device: torch.device,
) -> np.ndarray:
    transformed = _transform_inputs(raw_arrays, parameters)
    tensors = _inputs_to_tensors(transformed)
    model.eval()
    with torch.inference_mode():
        output = model(
            tensors["X_discharge"].to(device),
            tensors["X_charge"].to(device),
            tensors["X_physics"].to(device),
        ).detach().cpu().numpy()
    restored = _inverse_target(output, parameters).reshape(-1)
    if not np.isfinite(restored).all():
        raise ValueError("Inference preprocessing produced non-finite predictions.")
    return restored


def _write_prediction_rows(
    path: Path,
    variant: str,
    seed: int,
    rows: list[dict[str, str]],
    targets: np.ndarray,
    predictions: np.ndarray,
    partition: str,
) -> list[dict[str, Any]]:
    if len(rows) != len(targets) or len(rows) != len(predictions):
        raise ValueError(f"{variant}, seed {seed}: observation/prediction mismatch.")
    result = [
        {
            "variant": variant,
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
    _write_csv(path, result)
    return result


def _verify_baseline_reuse(
    dataset: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    normalization_parameters: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics_rows = _read_csv(BASELINE_DIR / "benchmark_run_metrics.csv")
    original_predictions = _read_csv(BASELINE_DIR / "train_validation_predictions.csv")
    baseline_config_path = (
        PROJECT_ROOT
        / "experiments"
        / "soh_forecast"
        / "bafuse_architecture_benchmark_v1"
        / "p5_training_config_v2_20261114.json"
    )
    if _sha256(baseline_config_path) != config["baseline_config_sha256"]:
        raise ValueError("Original P5 baseline config hash changed.")
    original_config = _read_json(baseline_config_path)
    if original_config.get("training") != config["training"]:
        raise ValueError("Original P5 baseline training settings are not comparable.")
    if original_config.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]:
        raise ValueError("Original P5 baseline does not use this frozen dataset.")
    if original_config.get("architectures", [None])[0] != config["architecture"]:
        raise ValueError("Original P5 CNN architecture is not comparable.")

    config_hash = _sha256(baseline_config_path)
    reused: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    for seed in config["training"]["seeds"]:
        metric_row = next(
            (
                row
                for row in metrics_rows
                if row["architecture"] == config["architecture"]
                and int(row["seed"]) == seed
            ),
            None,
        )
        if metric_row is None:
            raise ValueError(f"Missing original P5 baseline metrics for seed {seed}.")
        checkpoint_source = BASELINE_DIR / metric_row["checkpoint_path"]
        if _sha256(checkpoint_source) != metric_row["checkpoint_sha256"]:
            raise ValueError(f"Original P5 checkpoint hash mismatch for seed {seed}.")
        checkpoint = torch.load(
            checkpoint_source, map_location="cpu", weights_only=True
        )
        if checkpoint.get("training_configuration") != config["training"]:
            raise ValueError(f"Checkpoint training settings differ for seed {seed}.")
        if (
            checkpoint.get("frozen_dataset_manifest_sha256")
            != config["dataset_manifest_sha256"]
            or checkpoint.get("benchmark_config_sha256") != config_hash
        ):
            raise ValueError(f"Baseline checkpoint provenance mismatch for seed {seed}.")
        model = build_model(config["architecture"]).to(device)
        if checkpoint.get("architecture_config") != architecture_config(
            config["architecture"]
        ):
            raise ValueError(f"Baseline model architecture differs for seed {seed}.")
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        per_seed = [
            row
            for row in original_predictions
            if row["architecture"] == config["architecture"]
            and int(row["seed"]) == seed
        ]
        if any(row["partition"] not in {"train", "validation"} for row in per_seed):
            raise ValueError("Original baseline prediction file contains External rows.")
        expected_predictions: dict[str, np.ndarray] = {}
        for partition, arrays in (
            ("train", dataset["train"]),
            ("validation", dataset["validation"]),
        ):
            expected_rows = [row for row in per_seed if row["partition"] == partition]
            manifest_rows = dataset[f"{partition}_rows"]
            saved_keys = [
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                for row in expected_rows
            ]
            manifest_keys = [
                (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
                for row in manifest_rows
            ]
            if saved_keys != manifest_keys:
                raise ValueError(
                    f"Baseline {partition} keys do not match the frozen split for seed {seed}."
                )
            raw_predictions = _predict_model(
                model, arrays, normalization_parameters, device
            )
            stored = np.asarray(
                [float(row["predicted_soh_percent"]) for row in expected_rows],
                dtype=np.float32,
            )
            if len(stored) != len(raw_predictions):
                raise ValueError(f"Baseline {partition} row count differs for seed {seed}.")
            delta = np.abs(raw_predictions - stored)
            max_delta = float(delta.max(initial=0.0))
            if max_delta > 1e-6:
                raise ValueError(
                    f"Current CNN implementation does not reproduce baseline seed {seed}: "
                    f"max delta {max_delta}."
                )
            expected_predictions[partition] = raw_predictions
            y = arrays["y_soh"].reshape(-1)
            computed = _metric_values(y, raw_predictions)
            for name in ("MAE", "RMSE", "R2"):
                if abs(computed[name] - float(metric_row[f"{partition}_{name}"])) > 1e-6:
                    raise ValueError(
                        f"Baseline {partition} {name} differs for seed {seed}."
                    )

        history_source = (
            BASELINE_DIR
            / "training_curves"
            / f"{config['architecture']}_seed_{seed}_history.csv"
        )
        history_rows = _read_csv(history_source)
        best_epoch = int(metric_row["best_epoch"])
        best_history = next(
            (row for row in history_rows if int(row["epoch"]) == best_epoch), None
        )
        if best_history is None:
            raise ValueError(f"Baseline history lacks best epoch for seed {seed}.")
        history_out: list[dict[str, Any]] = []
        for row in history_rows:
            validation_rmse = float(row["validation_RMSE"])
            history_out.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_mse_loss_target_space": float(row["train_mse_loss"]),
                    "validation_mse_loss_target_space": validation_rmse**2,
                    "validation_RMSE_soh_pp": validation_rmse,
                    "target_loss_units": "percentage_points_squared",
                }
            )
        _write_csv(
            ARTIFACT_DIR / "histories" / f"A_seed_{seed}.csv",
            history_out,
        )

        checkpoint_copy = ARTIFACT_DIR / "checkpoints" / "A" / f"seed_{seed}_best.pt"
        checkpoint_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_source, checkpoint_copy)
        if _sha256(checkpoint_copy) != metric_row["checkpoint_sha256"]:
            raise ValueError("Copied baseline checkpoint hash changed.")
        for partition, rows in (
            ("train", dataset["train_rows"]),
            ("validation", dataset["validation_rows"]),
        ):
            preds = expected_predictions[partition]
            targets = dataset[partition]["y_soh"].reshape(-1)
            predictions.extend(
                _write_prediction_rows(
                    ARTIFACT_DIR
                    / "predictions"
                    / f"A_seed_{seed}_{partition}.csv",
                    "A",
                    seed,
                    rows,
                    targets,
                    preds,
                    partition,
                )
            )
        max_prediction_delta = max(
            float(
                np.max(
                    np.abs(
                        expected_predictions[partition]
                        - np.asarray(
                            [
                                float(row["predicted_soh_percent"])
                                for row in per_seed
                                if row["partition"] == partition
                            ],
                            dtype=np.float32,
                        )
                    )
                )
            )
            for partition in ("train", "validation")
        )
        train_metrics = _metric_values(
            dataset["train"]["y_soh"], expected_predictions["train"]
        )
        validation_metrics = _metric_values(
            dataset["validation"]["y_soh"], expected_predictions["validation"]
        )
        reused.append(
            {
                "variant": "A",
                "seed": seed,
                "architecture": config["architecture"],
                "status": "reused_verified_original_p5",
                "parameter_count": int(metric_row["parameter_count"]),
                "best_epoch": best_epoch,
                "epochs_completed": int(metric_row["epochs_completed"]),
                "training_duration_seconds": float(
                    metric_row["training_duration_seconds"]
                ),
                "train_mse_loss_target_space": float(
                    best_history["train_mse_loss"]
                ),
                "validation_mse_loss_target_space": float(
                    metric_row["validation_RMSE"]
                )
                ** 2,
                "target_loss_units": "percentage_points_squared",
                "train_MAE": train_metrics["MAE"],
                "train_RMSE": train_metrics["RMSE"],
                "train_R2": train_metrics["R2"],
                "validation_MAE": validation_metrics["MAE"],
                "validation_RMSE": validation_metrics["RMSE"],
                "validation_R2": validation_metrics["R2"],
                "baseline_prediction_max_abs_delta": max_prediction_delta,
                "checkpoint_path": str(checkpoint_copy.relative_to(ARTIFACT_DIR)),
                "checkpoint_sha256": _sha256(checkpoint_copy),
                "source_checkpoint_path": str(
                    checkpoint_source.relative_to(PROJECT_ROOT)
                ),
                "source_checkpoint_sha256": _sha256(checkpoint_source),
                "normalization_parameters_sha256": _sha256(
                    ARTIFACT_DIR / "normalization_parameters.json"
                ),
            }
        )
    return reused, predictions


def _run_one(
    variant: str,
    seed: int,
    config: dict[str, Any],
    parameters: dict[str, Any],
    dataset: dict[str, Any],
    device: torch.device,
    config_sha256: str,
    normalizers_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    training = config["training"]
    generator = _seed_everything(seed)
    train_inputs = _transform_inputs(dataset["train"], parameters)
    validation_inputs = _transform_inputs(dataset["validation"], parameters)
    train_target_model_space = _transform_target(
        dataset["train"]["y_soh"], parameters
    ).reshape(-1)
    validation_target_model_space = _transform_target(
        dataset["validation"]["y_soh"], parameters
    ).reshape(-1)
    train_x = _inputs_to_tensors(train_inputs)
    validation_x = _inputs_to_tensors(validation_inputs)
    train_y_tensor = torch.from_numpy(train_target_model_space.astype(np.float32))
    train_dataset = TensorDataset(
        train_x["X_discharge"],
        train_x["X_charge"],
        train_x["X_physics"],
        train_y_tensor,
    )
    loader = DataLoader(
        train_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = build_model(config["architecture"]).to(device)
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
    validation_targets_original = dataset["validation"]["y_soh"].reshape(-1)

    for epoch in range(1, int(training["max_epochs"]) + 1):
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
                raise FloatingPointError(f"Non-finite {variant} loss at epoch {epoch}.")
            loss.backward()
            optimizer.step()
            squared_error_sum += float(loss.detach()) * len(target)
            observations += len(target)
        train_loss = squared_error_sum / observations

        model.eval()
        with torch.inference_mode():
            validation_model_prediction = model(
                validation_x["X_discharge"].to(device),
                validation_x["X_charge"].to(device),
                validation_x["X_physics"].to(device),
            ).detach().cpu().numpy()
        validation_original_prediction = _inverse_target(
            validation_model_prediction, parameters
        ).reshape(-1)
        validation_loss = float(
            np.mean(
                (
                    validation_model_prediction.reshape(-1)
                    - validation_target_model_space
                )
                ** 2
            )
        )
        validation_metrics = _metric_values(
            validation_targets_original, validation_original_prediction
        )
        validation_rmse = validation_metrics["RMSE"]
        if not np.isfinite(validation_loss) or not np.isfinite(validation_rmse):
            raise FloatingPointError(f"Non-finite {variant} validation at epoch {epoch}.")
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss_target_space": train_loss,
                "validation_mse_loss_target_space": validation_loss,
                "validation_RMSE_soh_pp": validation_rmse,
                "target_loss_units": (
                    "normalized_target_squared"
                    if parameters["target"] is not None
                    else "percentage_points_squared"
                ),
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
        raise RuntimeError(f"No Validation checkpoint selected for {variant}, seed {seed}.")
    checkpoint_path = ARTIFACT_DIR / "checkpoints" / variant / f"seed_{seed}_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state,
            "architecture": config["architecture"],
            "architecture_config": architecture_config(config["architecture"]),
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_RMSE_soh_pp": best_rmse,
            "parameter_count": trainable_parameter_count(model),
            "training_configuration": training,
            "variant": variant,
            "frozen_dataset_identifier": config["dataset_identifier"],
            "frozen_dataset_manifest_sha256": config["dataset_manifest_sha256"],
            "experiment_config_sha256": config_sha256,
            "normalization_parameters_sha256": normalizers_sha256,
            "normalization_parameters_reference": (
                f"normalization_parameters.json#variants.{variant}"
            ),
            "target_transform": parameters["target"],
            "external_evaluation_performed": False,
        },
        checkpoint_path,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    train_prediction = _predict_model(model, dataset["train"], parameters, device)
    validation_prediction = _predict_model(
        model, dataset["validation"], parameters, device
    )
    train_metrics = _metric_values(
        dataset["train"]["y_soh"], train_prediction
    )
    validation_metrics = _metric_values(
        dataset["validation"]["y_soh"], validation_prediction
    )
    duration = time.perf_counter() - started
    history_path = ARTIFACT_DIR / "histories" / f"{variant}_seed_{seed}.csv"
    _write_csv(history_path, history)
    predictions: list[dict[str, Any]] = []
    for partition, rows, target_values, prediction_values in (
        (
            "train",
            dataset["train_rows"],
            dataset["train"]["y_soh"].reshape(-1),
            train_prediction,
        ),
        (
            "validation",
            dataset["validation_rows"],
            dataset["validation"]["y_soh"].reshape(-1),
            validation_prediction,
        ),
    ):
        predictions.extend(
            _write_prediction_rows(
                ARTIFACT_DIR
                / "predictions"
                / f"{variant}_seed_{seed}_{partition}.csv",
                variant,
                seed,
                rows,
                target_values,
                prediction_values,
                partition,
            )
        )
    best_history = history[best_epoch - 1]
    return (
        {
            "variant": variant,
            "seed": seed,
            "architecture": config["architecture"],
            "status": "trained",
            "parameter_count": trainable_parameter_count(model),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "training_duration_seconds": duration,
            "train_mse_loss_target_space": float(
                best_history["train_mse_loss_target_space"]
            ),
            "validation_mse_loss_target_space": float(
                best_history["validation_mse_loss_target_space"]
            ),
            "target_loss_units": best_history["target_loss_units"],
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "validation_MAE": validation_metrics["MAE"],
            "validation_RMSE": validation_metrics["RMSE"],
            "validation_R2": validation_metrics["R2"],
            "checkpoint_path": str(checkpoint_path.relative_to(ARTIFACT_DIR)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "source_checkpoint_path": "",
            "source_checkpoint_sha256": "",
            "normalization_parameters_sha256": normalizers_sha256,
        },
        predictions,
    )


def _summary_rows(run_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for variant in VARIANT_ORDER:
        rows = [row for row in run_rows if row["variant"] == variant]
        if len(rows) != 3:
            raise ValueError(f"Expected three seeds for {variant}; got {len(rows)}.")
        summary: dict[str, Any] = {"variant": variant, "seed_count": len(rows)}
        for metric in (
            "train_MAE",
            "train_RMSE",
            "train_R2",
            "validation_MAE",
            "validation_RMSE",
            "validation_R2",
            "train_mse_loss_target_space",
            "validation_mse_loss_target_space",
            "training_duration_seconds",
        ):
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def _write_report(
    config: dict[str, Any],
    normalizers: dict[str, Any],
    run_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    dataset: dict[str, Any],
    current_model_sha256: str,
    baseline_model_sha256: str,
) -> None:
    best = min(summary_rows, key=lambda row: row["validation_RMSE_mean"])
    baseline = next(row for row in summary_rows if row["variant"] == "A")
    lines = [
        "# P5 voltage and target normalization ablation",
        "",
        "## Scope and protocol",
        "",
        "- Dataset: `bafuse-p5-electrical-v1-20261114`; manifest SHA-256 "
        f"`{config['dataset_manifest_sha256']}`.",
        "- Split: 102 Train, 37 Validation, 64 External; the frozen manifest "
        "has no physical-cell overlap.",
        "- Only Train and Validation NPZ files were loaded. External tensors, "
        "features, and labels were not used; no External evaluation was run.",
        "- Architecture: `BaFuse_CNN1D_P5`; 15,073 parameters; same P5 five "
        "Physics features and frozen Train-fitted Physics scaler.",
        "- Optimizer/loss: AdamW, lr 0.001, weight decay 0.0001, batch size 16, "
        "MSELoss; 300-epoch cap, patience 35, min_delta 0, best Validation "
        "RMSE checkpoint; seeds 42, 123, 2026.",
        "- Target is reconstructed same-RPT Step 4 capacity retention in "
        "percentage, not an official Oxford SOH label.",
        "",
        "## Normalization formulas",
        "",
        "For modality position `j`, Train-only statistics use population "
        "standard deviation (`ddof=0`):",
        "",
        "`mu[j] = mean(X_train[:, j])`; `s[j] = std(X_train[:, j], ddof=0)`; "
        "`X_norm[:, j] = (X[:, j] - mu[j]) / s[j]`.",
        "",
        "Discharge and Charge have separate 128-value scalers. If a Train "
        "standard deviation is below `1e-6 V`, the effective scale is set to "
        "`1.0` and the Train mean is retained. The target uses "
        "`mu_y = mean(y_train)`, `s_y = std(y_train, ddof=0)`, and "
        "`y_norm = (y - mu_y) / s_y`; predictions are returned to percent by "
        "`y_hat = y_hat_norm * s_y + mu_y`. A target std below `1e-8 pp` "
        "would use effective scale 1.0. Validation always uses the saved Train "
        "statistics.",
        "",
        "P5 Physics features are already scaled in the frozen arrays. They "
        "were passed through unchanged; no second Physics scaler was fitted.",
        "",
        "## Fitted scaler statistics",
        "",
        "Exact 128-element means, observed standard deviations, effective "
        "scales, fallback indices, and target statistics are in "
        "[normalization_parameters.json](../experiments/soh_forecast/"
        "bafuse_p5_normalized_v1/artifacts/normalization_parameters.json). "
        "Statistics below are from the 102 Train rows only.",
        "",
    ]
    for name, scaler_key in (
        ("Discharge voltage", "discharge"),
        ("Charge voltage", "charge"),
    ):
        scaler = PopulationZScore.from_dict(normalizers["B"][scaler_key])
        lines.append(
            f"- {name}: 128 means/scales; mean range "
            f"`[{scaler.mean.min():.9g}, {scaler.mean.max():.9g}] V`; "
            f"observed std range "
            f"`[{scaler.observed_std.min():.9g}, "
            f"{scaler.observed_std.max():.9g}] V`; "
            f"fallback positions `{list(scaler.small_std_indices)}`."
        )
    target_scaler = PopulationZScore.from_dict(normalizers["C"]["target"])
    lines.extend(
        [
            f"- Target: mean `{target_scaler.mean[0]:.10f} pp`; observed "
            f"population std `{target_scaler.observed_std[0]:.10f} pp`; "
            f"effective scale `{target_scaler.scale[0]:.10f}`.",
            "- Per-seed and per-variant normalization parameter references "
            "are stored in each checkpoint or in the "
            "[checkpoint preprocessing map](../experiments/soh_forecast/"
            "bafuse_p5_normalized_v1/artifacts/"
            "checkpoint_preprocessing_references.csv); no scaler is refit at inference.",
            "",
            "## Results for every seed",
            "",
            "MAE/RMSE are in SoH percentage points; R² is computed separately "
            "on each partition. `Train loss` and `Validation loss` are MSE in "
            "the model's target space (raw pp² for A/B, normalized-target² "
            "for C/D). Epoch Train loss is the sample-weighted mean over "
            "mini-batches as the optimizer updates; Validation loss is "
            "measured at the selected checkpoint.",
            "",
            "| Variant | Seed | Best epoch | Epochs | Duration (s) | "
            "Train MAE | Train RMSE | Train R² | Val MAE | Val RMSE | Val R² | "
            "Train loss | Val loss |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in run_rows:
        lines.append(
            f"| {row['variant']} | {row['seed']} | {row['best_epoch']} | "
            f"{row['epochs_completed']} | {row['training_duration_seconds']:.2f} | "
            f"{row['train_MAE']:.4f} | {row['train_RMSE']:.4f} | "
            f"{row['train_R2']:.4f} | {row['validation_MAE']:.4f} | "
            f"{row['validation_RMSE']:.4f} | {row['validation_R2']:.4f} | "
            f"{row['train_mse_loss_target_space']:.6g} | "
            f"{row['validation_mse_loss_target_space']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Seed summaries and validation-only selection",
            "",
            "| Variant | Train MAE mean ± SD | Train RMSE mean ± SD | "
            "Train R² mean ± SD | Val MAE mean ± SD | Val RMSE mean ± SD | "
            "Val R² mean ± SD |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| {row['variant']} | "
            f"{row['train_MAE_mean']:.4f} ± {row['train_MAE_sd']:.4f} | "
            f"{row['train_RMSE_mean']:.4f} ± {row['train_RMSE_sd']:.4f} | "
            f"{row['train_R2_mean']:.4f} ± {row['train_R2_sd']:.4f} | "
            f"{row['validation_MAE_mean']:.4f} ± {row['validation_MAE_sd']:.4f} | "
            f"{row['validation_RMSE_mean']:.4f} ± "
            f"{row['validation_RMSE_sd']:.4f} | "
            f"{row['validation_R2_mean']:.4f} ± {row['validation_R2_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Lowest mean Validation RMSE is Variant **{best['variant']}** "
            f"(`{best['validation_RMSE_mean']:.4f} pp`), selected only from "
            "the three Validation runs per variant. Compared with original "
            f"P5 A (`{baseline['validation_RMSE_mean']:.4f} pp`), the mean "
            f"Validation RMSE difference is "
            f"`{best['validation_RMSE_mean'] - baseline['validation_RMSE_mean']:+.4f} pp`.",
            "",
            "Seed SDs are sample standard deviations across the three runs. "
            "Variant A duration is the historical benchmark's reported "
            "training duration; B-D duration is measured by this runner "
            "through checkpoint reload and Train/Validation metric generation. "
            "Treat duration differences as approximate rather than a strict "
            "throughput comparison.",
            "",
            "## Original P5 baseline verification",
            "",
            "Variant A reused the v2 original P5 results only after verifying "
            "the exact dataset/config hashes, training settings, checkpoint "
            "hashes, state-dict compatibility, and exact prediction parity "
            "for all 102 Train and 37 Validation observations at each seed. "
            "Current-model forward predictions matched the saved baseline "
            "predictions with maximum absolute difference `0.0` for all "
            "three seeds. The current model source hash differs from the "
            "historically recorded source hash, but this behavioral parity "
            "check passed for the requested CNN architecture.",
            "",
            f"- Current CNN module SHA-256: `{current_model_sha256}`.",
            f"- Baseline recorded CNN module SHA-256: `{baseline_model_sha256}`.",
            "- Original baseline config SHA-256: "
            f"`{config['baseline_config_sha256']}`.",
            "",
            "## Interpretation and limitations",
            "",
            "Differences are descriptive on this fixed Train/Validation split "
            "and three seeds. They do not establish improved External "
            "generalization. Validation includes only three physical cells; "
            "seed variability is not a substitute for additional cell-level "
            "validation. No architecture, Physics feature, label, split, "
            "optimizer, or training budget changes were made.",
            "",
            "## Reproducibility artifacts",
            "",
            "- Config: "
            "[experiment_config.json](../experiments/soh_forecast/"
            "bafuse_p5_normalized_v1/experiment_config.json).",
            "- Per-seed metrics: "
            "[run_metrics.csv](../experiments/soh_forecast/"
            "bafuse_p5_normalized_v1/artifacts/run_metrics.csv).",
            "- Seed summary: "
            "[summary_metrics.csv](../experiments/soh_forecast/"
            "bafuse_p5_normalized_v1/artifacts/summary_metrics.csv).",
            "- Checkpoints, histories, predictions, environment, hashes, and "
            "verified dataset metadata are under "
            "[artifacts/](../experiments/soh_forecast/"
            "bafuse_p5_normalized_v1/artifacts/).",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if ARTIFACT_DIR.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing ablation artifacts: {ARTIFACT_DIR}"
        )
    if REPORT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {REPORT_PATH}")
    config = _read_json(CONFIG_PATH)
    if config.get("authorization", {}).get("training_authorized") is not True:
        raise ValueError("Training authorization is not enabled for this experiment.")
    if config["authorization"].get("external_prediction_evaluation_authorized") is not False:
        raise ValueError("External prediction evaluation must remain disabled.")
    if config["training"]["seeds"] != [42, 123, 2026]:
        raise ValueError("Unexpected seed set.")
    expected_variants = {
        "A": (False, False),
        "B": (True, False),
        "C": (False, True),
        "D": (True, True),
    }
    actual_variants = {
        name: (
            value.get("voltage_normalization"),
            value.get("target_normalization"),
        )
        for name, value in config.get("variants", {}).items()
    }
    if actual_variants != expected_variants:
        raise ValueError("A-D normalization flags differ from the approved experiment.")
    if config["dataset_manifest_sha256"] != (
        "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
    ):
        raise ValueError("Unexpected frozen P5 dataset hash.")
    training = config["training"]
    if (
        config["architecture"] != "BaFuse_CNN1D_P5"
        or training["max_epochs"] != 300
        or training["batch_size"] != 16
        or training["optimizer"] != "AdamW"
        or training["learning_rate"] != 0.001
        or training["weight_decay"] != 0.0001
        or training["loss"] != "MSELoss"
        or training["early_stopping_patience"] != 35
        or training["early_stopping_min_delta"] != 0.0
        or training["checkpoint_metric"] != "validation_RMSE"
    ):
        raise ValueError("Training settings differ from the original P5 baseline.")
    if not REPORT_PATH.parent.is_dir():
        raise FileNotFoundError(f"Report directory is missing: {REPORT_PATH.parent}")
    if config.get("compute", {}).get("device") != "cuda:0":
        raise ValueError("This controlled experiment is scoped to CUDA device 0.")
    deterministic = _configure_determinism()
    device = torch.device("cuda:0")
    dataset = _load_dataset()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=False)
    for name in ("checkpoints", "histories", "predictions"):
        (ARTIFACT_DIR / name).mkdir()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(ARTIFACT_DIR / "training.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    train_arrays = dataset["train"]
    validation_arrays = dataset["validation"]
    normalizers = _fit_normalizers(train_arrays)
    config_sha256 = _sha256(CONFIG_PATH)
    physics_config_path = DATASET_DIR / "bafuse_p5_preprocessing_config.json"
    normalizer_bundle = {
        "schema_version": 1,
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "fit_partition": "train",
        "fit_row_count": 102,
        "voltage_policy": {
            "formula": "(x - mean_train_position) / population_std_train_position",
            "std_ddof": 0,
            "minimum_std_volts": VOLTAGE_MINIMUM_STD,
            "small_std_effective_scale": 1.0,
        },
        "target_policy": {
            "formula": "(y - mean_train) / population_std_train",
            "std_ddof": 0,
            "minimum_std_percentage_points": TARGET_MINIMUM_STD,
            "small_std_effective_scale": 1.0,
        },
        "physics_policy": {
            "applied_again": False,
            "frozen_scaler_config": str(physics_config_path.relative_to(PROJECT_ROOT)),
            "frozen_scaler_config_sha256": _sha256(physics_config_path),
        },
        "variants": normalizers,
    }
    normalizer_path = ARTIFACT_DIR / "normalization_parameters.json"
    _write_json(normalizer_path, normalizer_bundle)
    normalizers_sha256 = _sha256(normalizer_path)
    # Reload exactly what inference will consume rather than retaining fit-time objects.
    normalizers = _read_json(normalizer_path)["variants"]
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        **deterministic,
        "device_name": torch.cuda.get_device_name(0),
    }
    _write_json(ARTIFACT_DIR / "experiment_environment.json", environment)
    dataset_record = {
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": dataset["dataset_manifest_sha256"],
        "partition_counts": dataset["split_counts"],
        "partition_cell_counts": dataset["partition_cell_counts"],
        "physical_cell_overlap": False,
        "verified_internal_artifact_hashes": dataset[
            "verified_internal_artifact_hashes"
        ],
        "train_shapes": {
            name: list(value.shape) for name, value in train_arrays.items()
        },
        "validation_shapes": {
            name: list(value.shape) for name, value in validation_arrays.items()
        },
        "external_npz_loaded": False,
        "external_targets_used": False,
        "external_feature_rows_used": False,
    }
    _write_json(ARTIFACT_DIR / "dataset_verification.json", dataset_record)
    code_files = {
        "experiment_config.json": _sha256(CONFIG_PATH),
        "train_ablation.py": _sha256(Path(__file__)),
        "p5_normalization.py": _sha256(
            PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_normalization.py"
        ),
        "p5_benchmark_models.py": _sha256(
            PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark_models.py"
        ),
    }
    _write_json(ARTIFACT_DIR / "code_hashes.json", code_files)
    baseline_rows, prediction_rows = _verify_baseline_reuse(
        dataset, config, device, normalizers["A"]
    )
    run_rows = list(baseline_rows)
    logging.info("Verified/reused original P5 baseline A for seeds 42, 123, 2026.")
    for variant in ("B", "C", "D"):
        for seed in config["training"]["seeds"]:
            logging.info("Starting variant %s, seed %s.", variant, seed)
            metrics, predictions = _run_one(
                variant,
                int(seed),
                config,
                normalizers[variant],
                dataset,
                device,
                config_sha256,
                normalizers_sha256,
            )
            run_rows.append(metrics)
            prediction_rows.extend(predictions)
            logging.info(
                "Completed variant %s, seed %s at epoch %s: Validation RMSE %.4f pp.",
                variant,
                seed,
                metrics["best_epoch"],
                metrics["validation_RMSE"],
            )
    run_rows.sort(key=lambda row: (VARIANT_ORDER.index(row["variant"]), row["seed"]))
    _write_csv(ARTIFACT_DIR / "run_metrics.csv", run_rows)
    summaries = _summary_rows(run_rows)
    _write_csv(ARTIFACT_DIR / "summary_metrics.csv", summaries)
    all_predictions = sorted(
        prediction_rows,
        key=lambda row: (
            VARIANT_ORDER.index(row["variant"]),
            int(row["seed"]),
            row["partition"],
            int(row["rpt_index"].replace("RPT", ""))
            if row["rpt_index"].startswith("RPT")
            else int(row["rpt_index"]),
        ),
    )
    _write_csv(ARTIFACT_DIR / "train_validation_predictions.csv", all_predictions)
    physics_config_path = DATASET_DIR / "bafuse_p5_preprocessing_config.json"
    _write_csv(
        ARTIFACT_DIR / "checkpoint_preprocessing_references.csv",
        [
            {
                "variant": row["variant"],
                "seed": row["seed"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                "normalization_parameters_path": "normalization_parameters.json",
                "normalization_parameters_sha256": normalizers_sha256,
                "normalization_variant_reference": f"variants.{row['variant']}",
                "physics_scaler_config_path": str(
                    physics_config_path.relative_to(PROJECT_ROOT)
                ),
                "physics_scaler_config_sha256": _sha256(physics_config_path),
                "physics_scaler_applied_again": False,
            }
            for row in run_rows
        ],
    )
    baseline_checkpoint = torch.load(
        BASELINE_DIR
        / "checkpoints"
        / "BaFuse_CNN1D_P5"
        / "seed_42_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    current_model_sha256 = _sha256(
        PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark_models.py"
    )
    baseline_model_sha256 = str(baseline_checkpoint["model_source_code_sha256"])
    _write_report(
        config,
        normalizers,
        run_rows,
        summaries,
        dataset,
        current_model_sha256,
        baseline_model_sha256,
    )
    status = {
        "status": "complete",
        "experiment_identifier": config["experiment_identifier"],
        "dataset_identifier": config["dataset_identifier"],
        "dataset_manifest_sha256": config["dataset_manifest_sha256"],
        "run_count": len(run_rows),
        "trained_run_count": sum(row["status"] == "trained" for row in run_rows),
        "reused_baseline_run_count": sum(
            row["status"] == "reused_verified_original_p5" for row in run_rows
        ),
        "external_prediction_evaluation_performed": False,
        "external_npz_loaded": False,
        "selected_variant_by_mean_validation_rmse": min(
            summaries, key=lambda row: row["validation_RMSE_mean"]
        )["variant"],
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "report_sha256": _sha256(REPORT_PATH),
    }
    _write_json(ARTIFACT_DIR / "experiment_status.json", status)
    logging.info(
        "Experiment complete; Validation-only selection chose variant %s.",
        status["selected_variant_by_mean_validation_rmse"],
    )
    manifest_entries = {
        str(path.relative_to(ARTIFACT_DIR)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    }
    _write_json(
        ARTIFACT_DIR / "artifact_sha256_manifest.json",
        {
            "experiment_identifier": config["experiment_identifier"],
            "hash_scope": "All generated files under artifacts/ except this manifest.",
            "files": manifest_entries,
        },
    )
    return status


if __name__ == "__main__":
    run()
