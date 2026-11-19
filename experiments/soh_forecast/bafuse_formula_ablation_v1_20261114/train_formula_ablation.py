from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bafuse.hybrid.p5_benchmark_models import CNNVoltageEncoder
from bafuse.hybrid.soh_fusion import CurveEncoder


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA_PATH = OUT / "data" / "internal_trainval_raw.npz"
SEEDS = (42, 123, 2026)
MAX_EPOCHS = 300
BATCH_SIZE = 16
PATIENCE = 35
MIN_DELTA = 0.0
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
FEATURE_NAMES = {
    "p5": (
        "mean_abs_discharge_current",
        "discharge_current_variability",
        "discharge_voltage_slope_20_80",
        "discharge_voltage_derivative_mad_10_90",
        "discharge_voltage_segment_contrast_10_30_vs_70_90",
    ),
    "legacy": (
        "mean_discharge_temperature_native",
        "discharge_temperature_rise_native",
        "discharge_plateau_slope_v_per_progress",
        "discharge_slope_variability_v_per_progress",
        "mean_absolute_discharge_current_native",
    ),
}
VARIANTS = {
    "A1_discharge_only": {
        "split": "p5",
        "architecture": "p5",
        "charge": False,
        "physics": None,
    },
    "A2_discharge_charge": {
        "split": "p5",
        "architecture": "p5",
        "charge": True,
        "physics": None,
    },
    "A3_discharge_p5_physics": {
        "split": "p5",
        "architecture": "p5",
        "charge": False,
        "physics": "p5",
    },
    "C1_legacy_arch_legacy_physics": {
        "split": "p5",
        "architecture": "legacy",
        "charge": True,
        "physics": "legacy",
    },
    "C2_legacy_arch_p5_physics": {
        "split": "p5",
        "architecture": "legacy",
        "charge": True,
        "physics": "p5",
    },
    "C3_p5_arch_legacy_physics": {
        "split": "p5",
        "architecture": "p5",
        "charge": True,
        "physics": "legacy",
    },
    "D_legacy_split_p5_full": {
        "split": "legacy",
        "architecture": "p5",
        "charge": True,
        "physics": "p5",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def configure_determinism() -> dict[str, Any]:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise RuntimeError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return {
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "dataloader_num_workers": 0,
    }


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def read_internal_data() -> dict[str, np.ndarray]:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Sanitized internal-only data is missing: {DATA_PATH}")
    with np.load(DATA_PATH, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    required = {
        "X_discharge",
        "X_charge",
        "X_physics_p5_raw",
        "X_physics_p5_scaled",
        "X_physics_legacy_raw",
        "y_soh",
        "dataset_part",
        "physical_cell_id",
        "rpt_index",
        "partition",
        "legacy_partition",
    }
    if set(data) != required:
        raise ValueError(f"Unexpected sanitized input fields: {set(data) ^ required}")
    count = 139
    shapes = {
        "X_discharge": (count, 128),
        "X_charge": (count, 128),
        "X_physics_p5_raw": (count, 5),
        "X_physics_p5_scaled": (count, 5),
        "X_physics_legacy_raw": (count, 5),
        "y_soh": (count, 1),
        "dataset_part": (count,),
        "physical_cell_id": (count,),
        "rpt_index": (count,),
        "partition": (count,),
        "legacy_partition": (count,),
    }
    for name, values in data.items():
        if values.shape != shapes[name]:
            raise ValueError(f"Unexpected sanitized input shape for {name}: {values.shape}")
        if name.startswith("X_") or name == "y_soh":
            if values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid numeric data in {name}")
    if set(data["dataset_part"].tolist()) != {"1", "2"}:
        raise ValueError("Part 3 is present in the training input.")
    if Counter(data["partition"].tolist()) != {"train": 102, "validation": 37}:
        raise ValueError("The staged P5 Train/Validation assignment changed.")
    if Counter(data["legacy_partition"].tolist()) != {"train": 99, "validation": 40}:
        raise ValueError("The staged historical Train/Validation assignment changed.")
    for partition_name in ("partition", "legacy_partition"):
        cells_by_partition: dict[str, set[str]] = {}
        for cell, partition in zip(data["physical_cell_id"], data[partition_name]):
            cells_by_partition.setdefault(str(partition), set()).add(str(cell))
        if cells_by_partition["train"] & cells_by_partition["validation"]:
            raise ValueError(f"Physical-cell leakage in {partition_name}.")
    return data


def fit_physics(
    raw: np.ndarray,
    train_mask: np.ndarray,
    *,
    feature_names: tuple[str, ...] | None = None,
    frozen_scaled: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    names = feature_names or FEATURE_NAMES["p5"]
    if len(names) != raw.shape[1]:
        raise ValueError("Physics feature names do not match the input width.")
    train_values = raw[train_mask].astype(np.float64)
    mean = train_values.mean(axis=0)
    scale = train_values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError("Train-only physics scaler is nonfinite or has negative variance.")
    constant_features = scale == 0
    scale_used = scale.copy()
    scale_used[constant_features] = 1.0
    scaled = ((raw.astype(np.float64) - mean) / scale_used).astype(np.float32)
    scaled[:, constant_features] = 0.0
    if frozen_scaled is not None and not np.allclose(
        scaled[train_mask], frozen_scaled[train_mask], rtol=0.0, atol=1e-6
    ):
        raise ValueError("Recomputed P5 Train scaler differs from frozen P5 values.")
    return scaled, {
        "fit_partition": "train",
        "fit_rows": int(train_mask.sum()),
        "feature_names": list(names),
        "mean": mean.tolist(),
        "scale_population_std": scale.tolist(),
        "scale_used": scale_used.tolist(),
        "constant_train_features_zeroed": [
            names[index]
            for index in np.flatnonzero(constant_features)
        ],
        "fit_scope": "corresponding Train rows only",
    }


class FusionAblationModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        *,
        use_charge: bool,
        physics_family: str | None,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.use_charge = use_charge
        self.physics_family = physics_family
        curve_encoder = CurveEncoder if architecture == "legacy" else CNNVoltageEncoder
        self.discharge_encoder = curve_encoder()
        self.charge_encoder = curve_encoder() if use_charge else None
        if physics_family is None:
            self.physics_encoder = None
        elif architecture == "legacy":
            self.physics_encoder = nn.Sequential(
                nn.Linear(5, 32),
                nn.GELU(),
                nn.Linear(32, 32),
                nn.GELU(),
            )
        else:
            self.physics_encoder = nn.Sequential(
                nn.Linear(5, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
            )
        branches = 1 + int(use_charge) + int(physics_family is not None)
        activation: type[nn.Module] = nn.GELU if architecture == "legacy" else nn.ReLU
        layers: list[nn.Module] = [
            nn.Linear(branches * 32, 64),
            activation(),
        ]
        if architecture == "legacy":
            layers.append(nn.Dropout(0.2))
        layers.extend((nn.Linear(64, 32), activation(), nn.Linear(32, 1)))
        self.fusion_head = nn.Sequential(*layers)

    def forward(
        self,
        discharge: torch.Tensor,
        charge: torch.Tensor | None,
        physics: torch.Tensor | None,
    ) -> torch.Tensor:
        embeddings = [self.discharge_encoder(discharge)]
        if self.use_charge:
            if charge is None:
                raise ValueError("Charge branch is enabled but no charge tensor was passed.")
            embeddings.append(self.charge_encoder(charge))
        if self.physics_family is not None:
            if physics is None:
                raise ValueError("Physics branch is enabled but no physics tensor was passed.")
            embeddings.append(self.physics_encoder(physics))
        return self.fusion_head(torch.cat(embeddings, dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    errors = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": None if denominator <= 0 else float(1 - np.sum(errors**2) / denominator),
    }


def forward_smoke_test(device: torch.device) -> list[dict[str, Any]]:
    rows = []
    for variant, config in VARIANTS.items():
        model = FusionAblationModel(
            config["architecture"],
            use_charge=config["charge"],
            physics_family=config["physics"],
        ).to(device)
        with torch.inference_mode():
            output = model(
                torch.zeros((2, 128), dtype=torch.float32, device=device),
                torch.zeros((2, 128), dtype=torch.float32, device=device)
                if config["charge"]
                else None,
                torch.zeros((2, 5), dtype=torch.float32, device=device)
                if config["physics"] is not None
                else None,
            )
        if output.shape != (2, 1) or not torch.isfinite(output).all():
            raise RuntimeError(f"Synthetic forward-pass failed for {variant}.")
        rows.append(
            {
                "variant": variant,
                "input_shapes": {
                    "X_discharge": [2, 128],
                    "X_charge": [2, 128] if config["charge"] else None,
                    "X_physics": [2, 5] if config["physics"] is not None else None,
                },
                "output_shape": [2, 1],
                "parameter_count": parameter_count(model),
                "synthetic_only": True,
            }
        )
    return rows


def predict(
    model: nn.Module,
    data: dict[str, np.ndarray],
    indices: np.ndarray,
    config: dict[str, Any],
    physics: np.ndarray | None,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        output = model(
            torch.from_numpy(data["X_discharge"][indices]).to(device),
            torch.from_numpy(data["X_charge"][indices]).to(device)
            if config["charge"]
            else None,
            torch.from_numpy(physics[indices]).to(device)
            if physics is not None
            else None,
        ).squeeze(1)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Nonfinite prediction.")
    return output.detach().cpu().numpy()


def write_learning_curve(
    rows: list[dict[str, Any]], variant: str, seed: int
) -> None:
    path = OUT / "training_histories" / f"{variant}_seed_{seed}.csv"
    write_csv(path, rows)
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(row["train_mse_loss"]) for row in rows])
    axes[0].set_title("Train MSE")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE (SoH percentage points squared)")
    axes[1].plot(epochs, [float(row["validation_RMSE"]) for row in rows])
    axes[1].set_title("Validation RMSE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("RMSE (SoH percentage points)")
    figure.tight_layout()
    figure.savefig(
        OUT / "training_curves" / f"{variant}_seed_{seed}.png", dpi=150
    )
    plt.close(figure)


def train_one(
    variant: str,
    seed: int,
    data: dict[str, np.ndarray],
    device: torch.device,
    source_code_sha256: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    config = VARIANTS[variant]
    split_column = "partition" if config["split"] == "p5" else "legacy_partition"
    train_mask = data[split_column] == "train"
    validation_mask = data[split_column] == "validation"
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    train_cells = set(data["physical_cell_id"][train_indices].tolist())
    validation_cells = set(data["physical_cell_id"][validation_indices].tolist())
    if train_cells & validation_cells:
        raise ValueError(f"Train/Validation cell overlap for {variant}.")
    if not len(train_indices) or not len(validation_indices):
        raise ValueError(f"Empty Train/Validation partition for {variant}.")

    physics: np.ndarray | None = None
    scaler: dict[str, Any] | None = None
    if config["physics"] == "p5" and config["split"] == "p5":
        physics = data["X_physics_p5_scaled"]
        scaler = {
            "fit_partition": "frozen_p5_train",
            "fit_rows": 102,
            "feature_names": list(FEATURE_NAMES["p5"]),
            "source": "frozen P5 preprocessing_config; verified against staged Train values",
        }
    elif config["physics"] == "p5":
        physics, scaler = fit_physics(
            data["X_physics_p5_raw"],
            train_mask,
            feature_names=FEATURE_NAMES["p5"],
        )
        scaler["feature_names"] = list(FEATURE_NAMES["p5"])
    elif config["physics"] == "legacy":
        physics, scaler = fit_physics(
            data["X_physics_legacy_raw"],
            train_mask,
            feature_names=FEATURE_NAMES["legacy"],
        )
        scaler["feature_names"] = list(FEATURE_NAMES["legacy"])

    model = FusionAblationModel(
        config["architecture"],
        use_charge=config["charge"],
        physics_family=config["physics"],
    ).to(device)
    count = parameter_count(model)
    generator = seed_everything(seed)
    target = data["y_soh"].reshape(-1)
    dataset = TensorDataset(
        torch.from_numpy(data["X_discharge"][train_indices]),
        torch.from_numpy(data["X_charge"][train_indices]),
        torch.from_numpy(physics[train_indices]) if physics is not None else torch.zeros((len(train_indices), 1)),
        torch.from_numpy(target[train_indices]),
    )
    loader = DataLoader(
        dataset,
        batch_size=min(BATCH_SIZE, len(train_indices)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    best_rmse = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_squared_sum = 0.0
        n_seen = 0
        for discharge_batch, charge_batch, physics_batch, target_batch in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(
                discharge_batch.to(device),
                charge_batch.to(device) if config["charge"] else None,
                physics_batch.to(device) if physics is not None else None,
            ).squeeze(1)
            loss = nn.functional.mse_loss(output, target_batch.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss in {variant}, seed {seed}.")
            loss.backward()
            optimizer.step()
            train_squared_sum += float(loss.detach()) * len(target_batch)
            n_seen += len(target_batch)
        train_mse = train_squared_sum / n_seen
        validation_prediction = predict(
            model, data, validation_indices, config, physics, device
        )
        validation_rmse = metrics(
            target[validation_indices], validation_prediction
        )["rmse"]
        if validation_rmse is None or not math.isfinite(validation_rmse):
            raise FloatingPointError(f"Nonfinite Validation RMSE in {variant}, seed {seed}.")
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss": train_mse,
                "validation_RMSE": validation_rmse,
            }
        )
        if validation_rmse + MIN_DELTA < best_rmse:
            best_rmse = validation_rmse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError(f"No Validation checkpoint for {variant}, seed {seed}.")
    model.load_state_dict(best_state)
    model.eval()
    run_id = f"{variant}_seed_{seed}"
    checkpoint_path = OUT / "checkpoints" / f"{run_id}.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = checkpoint_path.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": best_state,
            "variant": variant,
            "seed": seed,
            "architecture": config["architecture"],
            "use_charge": config["charge"],
            "physics_family": config["physics"],
            "parameter_count": count,
            "best_epoch": best_epoch,
            "best_validation_RMSE": best_rmse,
            "training_configuration": {
                "max_epochs": MAX_EPOCHS,
                "batch_size": BATCH_SIZE,
                "optimizer": "AdamW",
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "loss": "MSELoss",
                "early_stopping_patience": PATIENCE,
                "early_stopping_min_delta": MIN_DELTA,
                "checkpoint_metric": "validation_RMSE",
                "target_transform": None,
            },
            "scaler": scaler,
            "internal_data_sha256": sha256(DATA_PATH),
            "training_code_sha256": source_code_sha256,
            "external_data_loaded": False,
        },
        temporary,
    )
    temporary.replace(checkpoint_path)
    train_prediction = predict(model, data, train_indices, config, physics, device)
    validation_prediction = predict(model, data, validation_indices, config, physics, device)
    rows: list[dict[str, Any]] = []
    per_cell: list[dict[str, Any]] = []
    for partition, indices, predictions in (
        ("train", train_indices, train_prediction),
        ("validation", validation_indices, validation_prediction),
    ):
        actual = target[indices]
        overall = metrics(actual, predictions)
        cells: dict[str, list[int]] = defaultdict(list)
        for local_index, index in enumerate(indices):
            cells[str(data["physical_cell_id"][index])].append(local_index)
            rows.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "seed": seed,
                    "dataset_part": str(data["dataset_part"][index]),
                    "physical_cell_id": str(data["physical_cell_id"][index]),
                    "rpt_index": str(data["rpt_index"][index]),
                    "partition": partition,
                    "target_soh_percent": float(actual[local_index]),
                    "predicted_soh_percent": float(predictions[local_index]),
                    "residual_prediction_minus_actual_pp": float(
                        predictions[local_index] - actual[local_index]
                    ),
                }
            )
        baseline_prediction = np.full_like(actual, float(np.mean(target[train_indices])))
        baseline = metrics(actual, baseline_prediction)
        for cell, local_indices in cells.items():
            cell_metrics = metrics(actual[local_indices], predictions[local_indices])
            per_cell.append(
                {
                    "run_id": run_id,
                    "variant": variant,
                    "seed": seed,
                    "partition": partition,
                    "physical_cell_id": cell,
                    "observation_count": len(local_indices),
                    **{f"model_{name}": value for name, value in cell_metrics.items()},
                    "train_mean_baseline_mae": metrics(
                        actual[local_indices],
                        np.full(len(local_indices), float(np.mean(target[train_indices]))),
                    )["mae"],
                    "train_mean_baseline_rmse": metrics(
                        actual[local_indices],
                        np.full(len(local_indices), float(np.mean(target[train_indices]))),
                    )["rmse"],
                }
            )
        if partition == "train":
            train_metrics = overall
            train_baseline = baseline
        else:
            validation_metrics = overall
            validation_baseline = baseline
    macro_validation = {
        f"validation_macro_cell_{metric_name}": float(
            np.mean(
                [
                    row[f"model_{metric_name}"]
                    for row in per_cell
                    if row["partition"] == "validation"
                ]
            )
        )
        for metric_name in ("mae", "rmse")
    }
    run_metrics = {
        "run_id": run_id,
        "variant": variant,
        "seed": seed,
        "split_policy": config["split"],
        "architecture_family": config["architecture"],
        "physics_family": config["physics"] or "none",
        "charge_branch": config["charge"],
        "parameter_count": count,
        "train_observations": len(train_indices),
        "validation_observations": len(validation_indices),
        "train_cell_count": len(train_cells),
        "validation_cell_count": len(validation_cells),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "best_validation_RMSE": best_rmse,
        "train_MAE": train_metrics["mae"],
        "train_RMSE": train_metrics["rmse"],
        "train_R2": train_metrics["r2"],
        "validation_MAE": validation_metrics["mae"],
        "validation_RMSE": validation_metrics["rmse"],
        "validation_R2": validation_metrics["r2"],
        "validation_macro_cell_MAE": macro_validation["validation_macro_cell_mae"],
        "validation_macro_cell_RMSE": macro_validation["validation_macro_cell_rmse"],
        "train_mean_baseline_MAE": train_baseline["mae"],
        "train_mean_baseline_RMSE": train_baseline["rmse"],
        "validation_train_mean_baseline_MAE": validation_baseline["mae"],
        "validation_train_mean_baseline_RMSE": validation_baseline["rmse"],
        "checkpoint_path": checkpoint_path.relative_to(OUT).as_posix(),
        "checkpoint_sha256": sha256(checkpoint_path),
        "data_sha256": sha256(DATA_PATH),
        "training_code_sha256": source_code_sha256,
        "external_data_loaded": False,
    }
    write_learning_curve(history, variant, seed)
    return run_metrics, history, rows, per_cell


def write_run_config(
    *,
    data: dict[str, np.ndarray],
    determinism: dict[str, Any],
    device: torch.device,
    smoke_rows: list[dict[str, Any]],
    code_hash: str,
) -> None:
    if device.type == "cuda":
        device_record = {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "cuda_runtime": torch.version.cuda,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    else:
        device_record = {"device": str(device), "name": None, "cuda_runtime": None}
    config = {
        "experiment_identifier": "bafuse-formula-ablation-v1-20261114",
        "dataset_identifier": "bafuse-p5-electrical-v1-20261114",
        "dataset_manifest_sha256": "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457",
        "training_authorized": True,
        "train_evaluation_authorized": True,
        "validation_evaluation_authorized": True,
        "checkpoint_selection_authorized": True,
        "external_prediction_evaluation_authorized": False,
        "dataset_modification_authorized": False,
        "label_modification_authorized": False,
        "pydma_fitting_authorized": False,
        "half_cell_fitting_authorized": False,
        "training": {
            "seeds": list(SEEDS),
            "max_epochs": MAX_EPOCHS,
            "batch_size": BATCH_SIZE,
            "optimizer": "AdamW",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "loss": "MSELoss",
            "early_stopping_patience": PATIENCE,
            "early_stopping_min_delta": MIN_DELTA,
            "checkpoint_metric": "validation_RMSE",
            "target_transform": None,
        },
        "variant_matrix": VARIANTS,
        "training_process_input": {
            "path": "data/internal_trainval_raw.npz",
            "sha256": sha256(DATA_PATH),
            "observations": len(data["y_soh"]),
            "parts": sorted(set(data["dataset_part"].tolist())),
            "p5_counts": {
                "train": int(np.sum(data["partition"] == "train")),
                "validation": int(np.sum(data["partition"] == "validation")),
            },
            "legacy_counts": {
                "train": int(np.sum(data["legacy_partition"] == "train")),
                "validation": int(np.sum(data["legacy_partition"] == "validation")),
            },
            "external_arrays_or_labels_loaded": False,
        },
        "environment": {
            "python": "recorded by environment output",
            "pytorch": torch.__version__,
            "numpy": np.__version__,
            "determinism": determinism,
            "device": device_record,
        },
        "synthetic_forward_passes": smoke_rows,
        "training_code_sha256": code_hash,
    }
    path = OUT / "ablation_training_config.json"
    path.write_text(
        json.dumps(config, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    data = read_internal_data()
    expected_run_ids = {
        f"{variant}_seed_{seed}" for variant in VARIANTS for seed in SEEDS
    }
    metrics_path = OUT / "training_run_metrics.csv"
    predictions_path = OUT / "training_predictions.csv"
    per_cell_path = OUT / "per_cell_metrics.csv"
    config_path = OUT / "ablation_training_config.json"
    output_exists = any(
        (OUT / folder).exists()
        for folder in ("checkpoints", "training_histories", "training_curves")
    )
    existing_metrics: list[dict[str, str]] = []
    existing_predictions: list[dict[str, str]] = []
    existing_per_cell: list[dict[str, str]] = []
    if output_exists:
        if not all(path.is_file() for path in (
            metrics_path, predictions_path, per_cell_path, config_path
        )):
            raise FileExistsError(
                "Partial training outputs lack their run tables/config; refusing to resume."
            )
        existing_metrics = read_csv(metrics_path)
        existing_predictions = read_csv(predictions_path)
        existing_per_cell = read_csv(per_cell_path)
        run_ids = [row["run_id"] for row in existing_metrics]
        completed_run_ids = set(run_ids)
        if len(run_ids) != len(completed_run_ids) or not completed_run_ids <= expected_run_ids:
            raise ValueError("Existing run metrics contain duplicate or unexpected run IDs.")
        stored_config = json.loads(config_path.read_text(encoding="utf-8"))
        if (
            stored_config["training_process_input"]["sha256"] != sha256(DATA_PATH)
            or stored_config["external_prediction_evaluation_authorized"] is not False
            or stored_config["dataset_manifest_sha256"]
            != "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
        ):
            raise ValueError("Existing run configuration does not match the authorized inputs.")
        for row in existing_metrics:
            checkpoint = OUT / row["checkpoint_path"]
            if (
                row["data_sha256"] != sha256(DATA_PATH)
                or row["external_data_loaded"].lower() != "false"
                or not checkpoint.is_file()
                or sha256(checkpoint) != row["checkpoint_sha256"]
            ):
                raise ValueError(f"Existing run provenance failed for {row['run_id']}.")
        prediction_run_ids = {row["run_id"] for row in existing_predictions}
        per_cell_run_ids = {row["run_id"] for row in existing_per_cell}
        prediction_counts = Counter(row["run_id"] for row in existing_predictions)
        if (
            prediction_run_ids != completed_run_ids
            or per_cell_run_ids != completed_run_ids
            or prediction_counts != {run_id: 139 for run_id in completed_run_ids}
            or any(row["dataset_part"] == "3" for row in existing_predictions)
            or any(row["partition"] not in {"train", "validation"} for row in existing_predictions)
            or any(row["partition"] not in {"train", "validation"} for row in existing_per_cell)
        ):
            raise ValueError("Existing Train/Validation output tables are incomplete or contain External rows.")
        if any(row["run_id"] not in completed_run_ids for row in existing_per_cell):
            raise ValueError("Existing per-cell metrics contain an unexpected run.")
    elif any(path.exists() for path in (metrics_path, predictions_path, per_cell_path, config_path)):
        raise FileExistsError("Training tables exist without output directories; refusing to overwrite.")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    determinism = configure_determinism()
    smoke_rows = forward_smoke_test(device)
    code_hash = sha256(Path(__file__).resolve())
    if not config_path.exists():
        write_run_config(
            data=data,
            determinism=determinism,
            device=device,
            smoke_rows=smoke_rows,
            code_hash=code_hash,
        )
    (OUT / "training_histories").mkdir(parents=True, exist_ok=True)
    (OUT / "training_curves").mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict[str, Any]] = list(existing_metrics)
    all_predictions: list[dict[str, Any]] = list(existing_predictions)
    all_per_cell: list[dict[str, Any]] = list(existing_per_cell)
    completed_run_ids = {row["run_id"] for row in all_metrics}
    for variant, config in VARIANTS.items():
        for seed in SEEDS:
            run_id = f"{variant}_seed_{seed}"
            if run_id in completed_run_ids:
                continue
            run_metrics, _, predictions, per_cell = train_one(
                variant, seed, data, device, code_hash
            )
            all_metrics.append(run_metrics)
            all_predictions.extend(predictions)
            all_per_cell.extend(per_cell)
            completed_run_ids.add(run_id)
            write_csv(metrics_path, all_metrics)
            write_csv(predictions_path, all_predictions)
            write_csv(per_cell_path, all_per_cell)
            print(
                f"{variant} seed={seed} best_epoch={run_metrics['best_epoch']} "
                f"val_rmse={run_metrics['validation_RMSE']:.6f}",
                flush=True,
            )
    if len(all_metrics) != 21:
        raise RuntimeError(f"Expected 21 new runs, completed {len(all_metrics)}.")
    env = {
        "python_version": os.sys.version.split()[0],
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "determinism": determinism,
    }
    (OUT / "training_environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"completed_runs": len(all_metrics), "device": env["device_name"]}, indent=2))


if __name__ == "__main__":
    main()
