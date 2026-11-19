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
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bafuse.hybrid.final2_dataset import (
    Final2Arrays,
    compute_soh_percent,
    ensure_cell_disjoint,
    fit_final2_scalers_for_frozen_cohort,
    transform_final2_arrays,
)
from bafuse.hybrid.final2_models import (
    FINAL2_MODEL_NAMES,
    build_final2_model,
    final2_parameter_counts,
    forward_final2_model,
)
from bafuse.hybrid.final2_runtime import (
    autocast_context,
    resolve_device,
    seed_everything,
)
from bafuse.hybrid.p5_benchmark_models import (
    CNNVoltageEncoder,
    trainable_parameter_count,
)
from bafuse.hybrid.p5_normalization import PopulationZScore


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "final2" / "training_6_models"
DATASET_IDENTIFIER = "bafuse-final2-p5-g10-guide-verified-v2"
STEP4_BASELINE_NAME = "CNN1D-Step4-Baseline"
TRAIN_MEAN_NAME = "Train-Mean"
SEEDS = (42, 123, 2026)
EXPECTED_SAMPLE_COUNTS = {"train": 99, "validation": 40, "external_test": 74}
EXPECTED_ACTIVE_CELL_COUNTS = {"train": 9, "validation": 3, "external_test": 12}
EXPECTED_ASSIGNED_CELL_COUNTS = {"train": 12, "validation": 4, "external_test": 12}


@dataclass(frozen=True)
class Partition:
    arrays: Final2Arrays
    dataset_part: np.ndarray
    physical_cell_id: np.ndarray
    rpt_index: np.ndarray
    keys: tuple[str, ...]

    @property
    def sample_count(self) -> int:
        return self.arrays.sample_count

    @property
    def cell_count(self) -> int:
        return len(set(self.physical_cell_id.astype(str)))


class Step4CNN1DBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = CNNVoltageEncoder()
        self.head = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, discharge: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(discharge))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty artifact: {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _load_partition(path: Path) -> Partition:
    stored = _load_npz(path)
    arrays = Final2Arrays(
        X_discharge=np.asarray(stored["X_discharge"], dtype=np.float32),
        X_charge=np.asarray(stored["X_charge"], dtype=np.float32),
        X_physics_raw=np.asarray(stored["X_physics_raw"], dtype=np.float32),
        y_soh=np.asarray(stored["y_soh"], dtype=np.float32).reshape(-1, 1),
    )
    arrays.validate()
    dataset_part = np.asarray(stored["dataset_part"], dtype=np.int16)
    physical_cell_id = np.asarray(stored["physical_cell_id"]).astype(str)
    rpt_index = np.asarray(stored["rpt_index"], dtype=np.int16)
    lengths = {
        len(dataset_part),
        len(physical_cell_id),
        len(rpt_index),
        arrays.sample_count,
    }
    if len(lengths) != 1:
        raise ValueError(f"Partition arrays and identity keys are misaligned: {path}.")
    keys = tuple(
        f"{part}|{cell}|{rpt}"
        for part, cell, rpt in zip(dataset_part, physical_cell_id, rpt_index)
    )
    if len(keys) != len(set(keys)):
        raise ValueError(f"Partition contains duplicate observation keys: {path}.")
    return Partition(arrays, dataset_part, physical_cell_id, rpt_index, keys)


def _snapshot_directory(path: Path) -> dict[str, str]:
    if not path.is_dir():
        raise FileNotFoundError(f"Historical artifact directory is missing: {path}.")
    return {
        item.relative_to(PROJECT_ROOT).as_posix(): _sha256(item)
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _validate_dataset(output_root: Path) -> tuple[dict[str, Any], Partition, Partition, dict[str, Any]]:
    dataset_dir = output_root / "frozen_dataset"
    summary = _read_json(dataset_dir / "dataset_summary.json")
    if summary.get("dataset_identifier") != DATASET_IDENTIFIER:
        raise ValueError("Dataset identifier does not match the authorized v2 cohort.")
    if not summary.get("training_authorized") or not summary.get(
        "external_inference_authorized"
    ):
        raise RuntimeError("Training and external evaluation are not authorized in the dataset.")
    if summary.get("candidate_mapping_rows") != 477:
        raise ValueError("Expected 477 mapping candidates in the versioned dataset.")
    if summary.get("status_counts") != {
        "ACCEPTED": 213,
        "REVIEW": 259,
        "REJECTED": 5,
    }:
        raise ValueError(f"Observed mapping disposition counts differ: {summary.get('status_counts')}.")
    if summary.get("accepted_observations_by_partition") != EXPECTED_SAMPLE_COUNTS:
        raise ValueError(
            "Observed accepted observation counts differ from the approved "
            f"pre-training check: {summary.get('accepted_observations_by_partition')}."
        )
    if summary.get("accepted_cells_by_partition") != EXPECTED_ACTIVE_CELL_COUNTS:
        raise ValueError(
            "Observed active cell counts differ from the approved pre-training check: "
            f"{summary.get('accepted_cells_by_partition')}."
        )
    if summary.get("split_cell_counts") != EXPECTED_ASSIGNED_CELL_COUNTS:
        raise ValueError(
            f"Assigned cell split changed: {summary.get('split_cell_counts')}."
        )
    if summary.get("g10_cell2_p5_extension", {}).get("observations") != 10:
        raise ValueError("G10-Cell2 RPT1-RPT10 were not all rebuilt as eligible observations.")

    checksum_manifest = _read_json(dataset_dir / "dataset_sha256_manifest.json")
    for name, entry in checksum_manifest["artifacts"].items():
        artifact = dataset_dir / name
        if not artifact.is_file() or _sha256(artifact) != entry["sha256"]:
            raise ValueError(f"Versioned dataset checksum mismatch: {name}.")

    train = _load_partition(dataset_dir / "train.npz")
    validation = _load_partition(dataset_dir / "validation.npz")
    if (train.sample_count, validation.sample_count) != (
        EXPECTED_SAMPLE_COUNTS["train"],
        EXPECTED_SAMPLE_COUNTS["validation"],
    ):
        raise ValueError("Train/Validation NPZ counts differ from the measured manifest.")
    if (train.cell_count, validation.cell_count) != (
        EXPECTED_ACTIVE_CELL_COUNTS["train"],
        EXPECTED_ACTIVE_CELL_COUNTS["validation"],
    ):
        raise ValueError("Train/Validation active-cell counts differ from the manifest.")
    if set(train.keys).intersection(validation.keys):
        raise ValueError("Train and Validation contain duplicate observation keys.")
    split_integrity = _read_json(dataset_dir / "split_integrity.json")
    train_cells = sorted(set(train.physical_cell_id.astype(str)))
    validation_cells = sorted(set(validation.physical_cell_id.astype(str)))
    external_cells = split_integrity["accepted_cell_ids_by_partition"]["external_test"]
    ensure_cell_disjoint(
        {
            "train": train_cells,
            "validation": validation_cells,
            "external_test": external_cells,
        }
    )
    assigned_cells = split_integrity["assigned_cell_ids_by_partition"]
    if not set(train_cells).issubset(assigned_cells["train"]) or not set(
        validation_cells
    ).issubset(assigned_cells["validation"]):
        raise ValueError("Train/Validation NPZ cells differ from the frozen split.")
    if not (dataset_dir / "external_test.npz").is_file():
        raise FileNotFoundError("The frozen External Test partition file is missing.")
    if summary["accepted_observations_by_partition"]["external_test"] != (
        EXPECTED_SAMPLE_COUNTS["external_test"]
    ):
        raise ValueError("External Test count failed preflight.")

    extension_rows = _read_csv(dataset_dir / "g10_cell2_p5_label_extension.csv")
    rpt10 = [row for row in extension_rows if row["rpt_index"] == "10"]
    if len(rpt10) != 1:
        raise ValueError("G10-Cell2 RPT10 is missing or duplicated in the new label extension.")
    rpt10_row = rpt10[0]
    q_rpt = float(rpt10_row["q_rpt_step4_testtime_native_current_time"])
    q_bol = float(rpt10_row["q_bol_step4_testtime_native_current_time"])
    if not math.isclose(
        float(rpt10_row["soh_percent"]),
        compute_soh_percent(q_rpt, q_bol),
        rel_tol=0,
        abs_tol=1e-8,
    ):
        raise ValueError("G10-Cell2 RPT10 target does not match the P5 capacity-ratio formula.")
    if not math.isclose(
        float(rpt10_row["soh_percent"]),
        79.59825353062423,
        rel_tol=0,
        abs_tol=1e-5,
    ):
        raise ValueError("G10-Cell2 RPT10 Step 4 target changed from the audited value.")
    if (
        rpt10_row["mapping_status"] != "GUIDE_VERIFIED"
        or rpt10_row["final2_partition"] != "external_test"
        or rpt10_row["step10_zero_current_excluded_count"] != "1"
        or rpt10_row["step10_zero_current_policy"] != "exclude_zero_current"
    ):
        raise ValueError("RPT10 mapping, partition, or Step 10 exclusion provenance changed.")
    zero_rows = json.loads(rpt10_row["step10_zero_current_rows"])
    if len(zero_rows) != 1 or zero_rows[0]["line"] != 2880746:
        raise ValueError("RPT10 zero-current row audit does not match the verified source row.")

    old_v1_eligibility = _read_csv(
        PROJECT_ROOT
        / "experiments"
        / "final2"
        / "v1"
        / "dataset"
        / "g10_cell2_curve_eligibility.csv"
    )
    old_rpt10 = [row for row in old_v1_eligibility if row["rpt_index"] == "10"]
    if len(old_rpt10) != 1 or old_rpt10[0]["step10_eligible"].casefold() != "false":
        raise ValueError("The historical v1 RPT10 rejection was not preserved.")

    external_path = dataset_dir / "external_test.npz"
    dataset_hashes = {
        name: _sha256(dataset_dir / name)
        for name in (
            "train.npz",
            "validation.npz",
            "external_test.npz",
            "dataset_summary.json",
            "split_integrity.json",
            "observation_manifest.csv",
            "dataset_sha256_manifest.json",
        )
    }
    historical_hashes = {
        "final2_v1": _snapshot_directory(PROJECT_ROOT / "experiments" / "final2" / "v1"),
        "phase_c2": _snapshot_directory(
            PROJECT_ROOT / "experiments" / "final2" / "phase_c2_g10_cell2_charge"
        ),
    }
    freeze_manifest = {
        "schema_version": 1,
        "dataset_identifier": DATASET_IDENTIFIER,
        "accepted_cohort_frozen": True,
        "training_authorized": True,
        "external_evaluation_authorized": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mapping_candidate_counts": summary["status_counts"],
        "accepted_observations_by_partition": summary[
            "accepted_observations_by_partition"
        ],
        "accepted_cells_by_partition": summary["accepted_cells_by_partition"],
        "assigned_cells_by_partition": summary["split_cell_counts"],
        "train_and_validation_npz_loaded": True,
        "external_npz_loaded": False,
        "external_npz_sha256": _sha256(external_path),
        "step4_label_formula": (
            "100 * trapezoidal_integral(abs(Amps), native TestTime) over verified "
            "Step 4 RPT / same-cell earliest mapped Step 4 BoL"
        ),
        "rpt10_soh_reference_percent": float(rpt10_row["soh_percent"]),
        "rpt10_zero_current_source_row": zero_rows[0],
        "review_rejected_policy": (
            "REVIEW and REJECTED rows remain in the audit manifest but are excluded "
            "from the frozen Accepted training cohort."
        ),
        "dataset_artifact_sha256": dataset_hashes,
        "historical_artifact_sha256": historical_hashes,
    }
    return summary, train, validation, freeze_manifest


def _runtime_record(device: torch.device, amp_enabled: bool) -> dict[str, Any]:
    cuda_device: dict[str, Any] | None = None
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        cuda_device = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
        }
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": cuda_device,
        "amp_enabled": amp_enabled,
        "numpy_version": np.__version__,
        "determinism": {
            "seeded_python_numpy_torch": True,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "known_reproducibility_limitation": (
            "Bitwise equivalence across different GPUs, CUDA drivers, or library builds "
            "is not claimed."
        ),
    }


def _tensor_inputs(
    arrays: Final2Arrays,
    scalers: Any,
) -> dict[str, torch.Tensor]:
    normalized = transform_final2_arrays(arrays, scalers)
    return {
        key: torch.as_tensor(value, dtype=torch.float32)
        for key, value in normalized.items()
    }


def _forward(
    model: nn.Module,
    name: str,
    discharge: torch.Tensor,
    charge: torch.Tensor,
    physics: torch.Tensor,
) -> torch.Tensor:
    if name == STEP4_BASELINE_NAME:
        return model(discharge)
    return forward_final2_model(
        model,
        name,
        discharge,
        charge,
        physics if name == "CNN-3M" else None,
    )


def _predict_percent(
    model: nn.Module,
    name: str,
    inputs: dict[str, torch.Tensor],
    target_scaler: PopulationZScore,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(inputs["X_discharge"]), 64):
            stop = min(start + 64, len(inputs["X_discharge"]))
            discharge = inputs["X_discharge"][start:stop].to(device)
            charge = inputs["X_charge"][start:stop].to(device)
            physics = inputs["X_physics"][start:stop].to(device)
            with autocast_context(device, enabled=amp_enabled):
                prediction = _forward(model, name, discharge, charge, physics)
            normalized = prediction.detach().float().cpu().numpy()
            output.append(
                np.asarray(target_scaler.inverse_transform(normalized)).reshape(-1)
            )
    return np.concatenate(output)


def _metric_values(actual: np.ndarray, predicted: np.ndarray, cells: int) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    predicted = np.asarray(predicted, dtype=np.float64).reshape(-1)
    if actual.shape != predicted.shape or not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs are misaligned or contain non-finite values.")
    residual = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "N": int(actual.size),
        "N_cells": int(cells),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "R2": None if denominator <= 0 else float(1.0 - np.sum(residual**2) / denominator),
    }


def _per_cell_rows(
    *,
    name: str,
    seed: int | str,
    partition_name: str,
    partition: Partition,
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexes: defaultdict[str, list[int]] = defaultdict(list)
    for index, cell in enumerate(partition.physical_cell_id.astype(str)):
        indexes[cell].append(index)
    rows: list[dict[str, Any]] = []
    for cell, indices in sorted(indexes.items()):
        selected = np.asarray(indices, dtype=np.int64)
        rows.append(
            {
                "model": name,
                "seed": seed,
                "partition": partition_name,
                "physical_cell_id": cell,
                **_metric_values(actual[selected], predicted[selected], 1),
            }
        )
    macro = {
        "model": name,
        "seed": seed,
        "partition": partition_name,
        "N_cells": len(rows),
        "macro_MAE": statistics.mean(row["MAE"] for row in rows),
        "macro_RMSE": statistics.mean(row["RMSE"] for row in rows),
        "macro_R2": (
            statistics.mean(row["R2"] for row in rows if row["R2"] is not None)
            if any(row["R2"] is not None for row in rows)
            else None
        ),
    }
    return rows, macro


def _optimizer_state_to_cpu(state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: (
            value.detach().cpu().clone()
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in state.items()
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _train_one_run(
    *,
    name: str,
    seed: int,
    train: Partition,
    validation: Partition,
    train_inputs: dict[str, torch.Tensor],
    validation_inputs: dict[str, torch.Tensor],
    train_targets_normalized: torch.Tensor,
    target_scaler: PopulationZScore,
    output_dir: Path,
    dataset_manifest_sha256: str,
    config_sha256: str,
    device: torch.device,
    amp_enabled: bool,
    training: dict[str, Any],
    selection_metric: str = "RMSE",
    print_progress: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    seed_everything(seed)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model = (
        Step4CNN1DBaseline()
        if name == STEP4_BASELINE_NAME
        else build_final2_model(name)
    ).to(device)
    parameter_count = trainable_parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    training_dataset = TensorDataset(
        train_inputs["X_discharge"],
        train_inputs["X_charge"],
        train_inputs["X_physics"],
        train_targets_normalized,
    )
    loader = DataLoader(
        training_dataset,
        batch_size=int(training["batch_size"]),
        shuffle=True,
        generator=generator,
        num_workers=0,
        drop_last=False,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    stem = f"{name.replace('-', '_')}_seed_{seed}"
    checkpoint_dir = output_dir / "checkpoints" / name
    best_path = checkpoint_dir / f"seed_{seed}_best.pt"
    last_path = checkpoint_dir / f"seed_{seed}_last.pt"
    best_rmse = math.inf
    best_score = math.inf
    best_epoch = 0
    stale_epochs = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_optimizer_state: dict[str, Any] | None = None
    best_grad_scaler_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.perf_counter()
    for epoch in range(1, int(training["max_epochs"]) + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for discharge, charge, physics, target in loader:
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=amp_enabled):
                prediction = _forward(
                    model,
                    name,
                    discharge.to(device),
                    charge.to(device),
                    physics.to(device),
                ).squeeze(1)
                loss = nn.functional.mse_loss(prediction, target.to(device))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"Non-finite training loss for {name}, seed {seed}, epoch {epoch}."
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.detach()) * len(target)
            train_count += len(target)

        validation_prediction = _predict_percent(
            model,
            name,
            validation_inputs,
            target_scaler,
            device,
            amp_enabled,
        )
        actual_validation = validation.arrays.y_soh.reshape(-1)
        validation_metrics = _metric_values(
            actual_validation,
            validation_prediction,
            validation.cell_count,
        )
        validation_rmse = float(validation_metrics["RMSE"])
        if not math.isfinite(validation_rmse):
            raise FloatingPointError(
                f"Non-finite Validation RMSE for {name}, seed {seed}, epoch {epoch}."
            )
        if print_progress:
            val_actual_z = target_scaler.transform(actual_validation.reshape(-1, 1))
            val_pred_z = target_scaler.transform(validation_prediction.reshape(-1, 1))
            val_loss = float(np.mean((val_actual_z - val_pred_z) ** 2))
            print(
                f"{name} seed={seed} epoch={epoch:03d} "
                f"train_loss={train_loss_sum / train_count:.6f} "
                f"validation_loss={val_loss:.6f} "
                f"validation_MAE_pp={float(validation_metrics['MAE']):.6f} "
                f"lr={optimizer.param_groups[0]['lr']:.6g}",
                flush=True,
            )
        history.append(
            {
                "epoch": epoch,
                "train_mse_loss_normalized": train_loss_sum / train_count,
                "validation_MAE_pp": validation_metrics["MAE"],
                "validation_RMSE_pp": validation_rmse,
                "validation_R2": validation_metrics["R2"],
            }
        )
        score = float(validation_metrics[selection_metric])
        if not math.isfinite(score):
            raise FloatingPointError(f"Non-finite Validation {selection_metric}.")
        if score + float(training["early_stopping_min_delta"]) < best_score:
            best_score = score
            best_rmse = validation_rmse
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            best_optimizer_state = _optimizer_state_to_cpu(optimizer.state_dict())
            best_grad_scaler_state = scaler.state_dict()
        else:
            stale_epochs += 1
            if stale_epochs >= int(training["early_stopping_patience"]):
                break

    if (
        best_state is None
        or best_optimizer_state is None
        or best_grad_scaler_state is None
    ):
        raise RuntimeError(f"No Validation checkpoint was selected for {name}, seed {seed}.")
    payload = {
        "model_state_dict": best_state,
        "optimizer_state_dict": best_optimizer_state,
        "grad_scaler_state_dict": best_grad_scaler_state,
        "model": name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_RMSE_pp": best_rmse,
        "parameter_count": parameter_count,
        "training_configuration": training,
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_sha256_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": config_sha256,
        "target_transform": "Final2 Train-only population z-score; inverse-transform for metrics",
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_torch_save(best_path, payload)
    last_payload = {
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "optimizer_state_dict": _optimizer_state_to_cpu(optimizer.state_dict()),
        "grad_scaler_state_dict": scaler.state_dict(),
        "model": name,
        "seed": seed,
        "last_epoch": len(history),
        "parameter_count": parameter_count,
        "training_configuration": training,
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_sha256_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": config_sha256,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _atomic_torch_save(last_path, last_payload)

    model.load_state_dict(best_state)
    model.eval()
    best_validation = _predict_percent(
        model,
        name,
        validation_inputs,
        target_scaler,
        device,
        amp_enabled,
    )
    best_model = (
        Step4CNN1DBaseline()
        if name == STEP4_BASELINE_NAME
        else build_final2_model(name)
    ).to(device)
    loaded = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model.load_state_dict(loaded["model_state_dict"])
    best_model.eval()
    reloaded_validation = _predict_percent(
        best_model,
        name,
        validation_inputs,
        target_scaler,
        device,
        amp_enabled,
    )
    validation_delta = float(np.max(np.abs(best_validation - reloaded_validation)))
    if validation_delta > 1e-6:
        raise RuntimeError(
            f"Checkpoint reload mismatch for {name}, seed {seed}: {validation_delta}."
        )
    train_prediction = _predict_percent(
        best_model,
        name,
        train_inputs,
        target_scaler,
        device,
        amp_enabled,
    )
    train_actual = train.arrays.y_soh.reshape(-1)
    train_metrics = _metric_values(train_actual, train_prediction, train.cell_count)
    validation_metrics = _metric_values(
        actual_validation, reloaded_validation, validation.cell_count
    )
    prediction_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    for partition_name, partition, actual, predicted in (
        ("train", train, train_actual, train_prediction),
        ("validation", validation, actual_validation, reloaded_validation),
    ):
        cells, macro = _per_cell_rows(
            name=name,
            seed=seed,
            partition_name=partition_name,
            partition=partition,
            actual=actual,
            predicted=predicted,
        )
        per_cell_rows.extend(cells)
        macro_rows.append(macro)
        for index, key in enumerate(partition.keys):
            part, cell, rpt = key.split("|")
            prediction_rows.append(
                {
                    "model": name,
                    "seed": seed,
                    "partition": partition_name,
                    "dataset_part": part,
                    "physical_cell_id": cell,
                    "rpt_index": rpt,
                    "target_soh_percent": float(actual[index]),
                    "predicted_soh_percent": float(predicted[index]),
                    "residual_prediction_minus_actual_pp": float(
                        predicted[index] - actual[index]
                    ),
                }
            )
    run_metrics = {
        "model": name,
        "seed": seed,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "duration_seconds": time.perf_counter() - started,
        "best_validation_RMSE_pp": best_rmse,
        "train": train_metrics,
        "validation": validation_metrics,
        "checkpoint_path": best_path.relative_to(output_dir).as_posix(),
        "last_checkpoint_path": last_path.relative_to(output_dir).as_posix(),
        "checkpoint_sha256": _sha256(best_path),
        "last_checkpoint_sha256": _sha256(last_path),
        "validation_reload_max_abs_delta_pp": validation_delta,
    }
    _write_csv(output_dir / "history" / f"{stem}.csv", history)
    return run_metrics, prediction_rows, per_cell_rows, macro_rows


def _train(output_root: Path, requested_device: str | None) -> None:
    run_dir = output_root / "run_001"
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing Final2 training run: {run_dir}."
        )
    summary, train, validation, freeze_manifest = _validate_dataset(output_root)
    device = resolve_device(requested_device)
    amp_enabled = device.type == "cuda"
    model_counts = final2_parameter_counts()
    config = {
        "schema_version": 1,
        "experiment_identifier": "bafuse-final2-six-models-rpt10-recovered-v2",
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_summary_sha256": _sha256(
            output_root / "frozen_dataset" / "dataset_summary.json"
        ),
        "dataset_sha256_manifest_sha256": _sha256(
            output_root / "frozen_dataset" / "dataset_sha256_manifest.json"
        ),
        "training_authorized": True,
        "external_evaluation_authorized": True,
        "split": {
            "partition_cell_counts": EXPECTED_ASSIGNED_CELL_COUNTS,
            "observed_accepted_counts": EXPECTED_SAMPLE_COUNTS,
            "active_cells": EXPECTED_ACTIVE_CELL_COUNTS,
        },
        "models": list(FINAL2_MODEL_NAMES),
        "primary_comparison_models": list(FINAL2_MODEL_NAMES),
        "train_only_learned_baselines": [STEP4_BASELINE_NAME, TRAIN_MEAN_NAME],
        "parameter_counts": model_counts,
        "training": {
            "seeds": list(SEEDS),
            "max_epochs": 300,
            "batch_size": 16,
            "optimizer": "AdamW",
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "loss": "MSELoss on Train-normalized SoH",
            "early_stopping_patience": 35,
            "early_stopping_min_delta": 0.0,
            "checkpoint_selection": "lowest Validation RMSE in SoH percentage points",
            "mixed_precision_requested": True,
            "mixed_precision_enabled": amp_enabled,
            "scaler_fit_scope": "Train observations only",
            "fresh_initialization": True,
        },
        "target": {
            "definition": (
                "100 * Step 4 discharge integrated abs(Amps) against native TestTime "
                "at RPT / same-cell earliest mapped Step 4 BoL"
            ),
            "unit": "SoH percentage points",
            "semantics": "same-time reconstructed capacity retention, not Oxford ground truth",
        },
        "selection_policy": (
            "For each model, select the seed checkpoint with lowest Validation RMSE; "
            "select the best of the six primary models using Validation only."
        ),
        "external_test_policy": (
            "Do not load external_test.npz during training or model selection. "
            "Run one external evaluation only after selection is frozen."
        ),
        "runtime": _runtime_record(device, amp_enabled),
    }
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True)
    config_path = output_root / "training_config_v2.json"
    _write_json(config_path, config)
    config_sha256 = _sha256(config_path)
    freeze_manifest["training_config_sha256"] = config_sha256
    freeze_manifest["external_npz_loaded"] = False
    _write_json(output_root / "frozen_dataset_manifest_v2.json", freeze_manifest)
    dataset_manifest_sha256 = _sha256(
        output_root / "frozen_dataset" / "dataset_sha256_manifest.json"
    )

    scalers = fit_final2_scalers_for_frozen_cohort(
        train.arrays,
        accepted_cohort_frozen=freeze_manifest["accepted_cohort_frozen"],
        training_authorized=freeze_manifest["training_authorized"],
    )
    scaler_payload = scalers.to_dict(dataset_identifier=DATASET_IDENTIFIER)
    scaler_path = run_dir / "normalization_parameters.json"
    _write_json(scaler_path, scaler_payload)
    train_inputs = _tensor_inputs(train.arrays, scalers)
    validation_inputs = _tensor_inputs(validation.arrays, scalers)
    train_targets_normalized = torch.as_tensor(
        scalers.target.transform(train.arrays.y_soh), dtype=torch.float32
    ).reshape(-1)
    baseline_mean = float(np.mean(train.arrays.y_soh))

    training = config["training"]
    run_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME):
        for seed in SEEDS:
            result, predictions, cells, macros = _train_one_run(
                name=name,
                seed=seed,
                train=train,
                validation=validation,
                train_inputs=train_inputs,
                validation_inputs=validation_inputs,
                train_targets_normalized=train_targets_normalized,
                target_scaler=scalers.target,
                output_dir=run_dir,
                dataset_manifest_sha256=dataset_manifest_sha256,
                config_sha256=config_sha256,
                device=device,
                amp_enabled=amp_enabled,
                training=training,
            )
            run_rows.append(result)
            prediction_rows.extend(predictions)
            per_cell_rows.extend(cells)
            macro_rows.extend(macros)

    selected_runs: dict[str, dict[str, Any]] = {}
    for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME):
        selected_runs[name] = min(
            (row for row in run_rows if row["model"] == name),
            key=lambda row: row["validation"]["RMSE"],
        )
    selected_six = [selected_runs[name] for name in FINAL2_MODEL_NAMES]
    selected_model = min(
        selected_six, key=lambda row: row["validation"]["RMSE"]
    )
    best_run_metrics = []
    for row in selected_runs.values():
        best_run_metrics.append(
            {
                "model": row["model"],
                "seed": row["seed"],
                "parameter_count": row["parameter_count"],
                "best_epoch": row["best_epoch"],
                "checkpoint_path": row["checkpoint_path"],
                "checkpoint_sha256": row["checkpoint_sha256"],
                **{
                    f"train_{key}": value
                    for key, value in row["train"].items()
                },
                **{
                    f"validation_{key}": value
                    for key, value in row["validation"].items()
                },
            }
        )
    mean_rows = []
    for partition_name, partition in (("train", train), ("validation", validation)):
        actual = partition.arrays.y_soh.reshape(-1)
        predicted = np.full_like(actual, baseline_mean, dtype=np.float64)
        mean_rows.append(
            {
                "model": TRAIN_MEAN_NAME,
                "seed": "train_mean",
                "parameter_count": 0,
                "best_epoch": None,
                "checkpoint_path": "",
                "checkpoint_sha256": "",
                **{
                    f"{partition_name}_{key}": value
                    for key, value in _metric_values(
                        actual, predicted, partition.cell_count
                    ).items()
                },
            }
        )
    train_mean, validation_mean = mean_rows
    best_run_metrics.append(
        {
            "model": TRAIN_MEAN_NAME,
            "seed": "train_mean",
            "parameter_count": 0,
            "best_epoch": None,
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            **train_mean,
            **validation_mean,
        }
    )
    flat_run_rows = [
        {
            **{
                key: value
                for key, value in row.items()
                if key not in {"train", "validation"}
            },
            **{f"train_{key}": value for key, value in row["train"].items()},
            **{
                f"validation_{key}": value
                for key, value in row["validation"].items()
            },
        }
        for row in run_rows
    ]
    _write_csv(run_dir / "per_seed_train_validation_metrics.csv", flat_run_rows)
    _write_csv(run_dir / "selected_validation_metrics.csv", best_run_metrics)
    _write_csv(run_dir / "train_validation_predictions.csv", prediction_rows)
    _write_csv(run_dir / "per_cell_train_validation_metrics.csv", per_cell_rows)
    _write_csv(run_dir / "macro_train_validation_metrics.csv", macro_rows)
    _write_csv(run_dir / "train_mean_baseline_metrics.csv", mean_rows)

    selection = {
        "schema_version": 1,
        "selection_frozen": True,
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_metric": "overall Validation RMSE (SoH percentage points)",
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_sha256_manifest_sha256": dataset_manifest_sha256,
        "training_config_sha256": config_sha256,
        "normalization_parameters_sha256": _sha256(scaler_path),
        "selected_best_seed_by_model": selected_runs,
        "selected_primary_model": selected_model["model"],
        "selected_primary_seed": selected_model["seed"],
        "selected_primary_validation_rmse_pp": selected_model["validation"]["RMSE"],
        "train_mean_soh_percent": baseline_mean,
        "external_evaluation_authorized": True,
        "external_evaluated": False,
        "primary_models": list(FINAL2_MODEL_NAMES),
        "evaluation_models": [
            *FINAL2_MODEL_NAMES,
            STEP4_BASELINE_NAME,
            TRAIN_MEAN_NAME,
        ],
    }
    _write_json(run_dir / "validation_selection.json", selection)
    checkpoint_manifest = {
        name: {
            "best_path": row["checkpoint_path"],
            "best_sha256": row["checkpoint_sha256"],
            "last_path": row["last_checkpoint_path"],
            "last_sha256": row["last_checkpoint_sha256"],
            "seed": row["seed"],
            "best_epoch": row["best_epoch"],
            "validation_rmse_pp": row["validation"]["RMSE"],
        }
        for name, row in selected_runs.items()
    }
    _write_json(run_dir / "checkpoint_manifest.json", checkpoint_manifest)
    report_lines = [
        "# BaFuse Final2 - training and validation",
        "",
        f"- Dataset: `{DATASET_IDENTIFIER}`.",
        "- Objective: same-time SOH estimation from Step 4/Step 10 voltage-shape curves.",
        "- Target: same-cell Step 4 capacity ratio against earliest mapped Step 4 BoL; reconstructed proxy, not Oxford ground truth.",
        "- Cohort: 213 Accepted observations; 259 REVIEW and 5 REJECTED candidates remain excluded.",
        "- Split: Train 99/9 active cells; Validation 40/3; External Test 74/12.",
        "- External Test arrays were not loaded in this training/selection phase.",
        "- Normalization: separate discharge, charge, P5 physics, and target scalers fitted on Train only.",
        "- Model selection: lowest Validation RMSE; no external outcomes informed selection.",
        f"- Selected primary architecture/seed: `{selected_model['model']}` / {selected_model['seed']}.",
        "",
        "## Selected seed metrics",
        "",
        "| Model | Seed | Train N/cells | Train MAE | Train RMSE | Train R2 | Validation N/cells | Validation MAE | Validation RMSE | Validation R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in best_run_metrics:
        report_lines.append(
            f"| {row['model']} | {row['seed']} | "
            f"{row['train_N']}/{row['train_N_cells']} | {row['train_MAE']:.4f} | "
            f"{row['train_RMSE']:.4f} | {row['train_R2']:.4f} | "
            f"{row['validation_N']}/{row['validation_N_cells']} | "
            f"{row['validation_MAE']:.4f} | {row['validation_RMSE']:.4f} | "
            f"{row['validation_R2']:.4f} |"
        )
    report_lines.extend(
        [
            "",
            "External evaluation is pending and must be run exactly once after this selection freeze.",
            "",
        ]
    )
    (run_dir / "training_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    for snapshot_name, paths in freeze_manifest["historical_artifact_sha256"].items():
        current = _snapshot_directory(
            PROJECT_ROOT
            / (
                "experiments/final2/v1"
                if snapshot_name == "final2_v1"
                else "experiments/final2/phase_c2_g10_cell2_charge"
            )
        )
        if current != paths:
            raise RuntimeError(f"Historical {snapshot_name} artifacts changed during training.")
    print(
        json.dumps(
            {
                "status": "training_and_validation_complete",
                "run_directory": str(run_dir),
                "selected_model": selection["selected_primary_model"],
                "selected_seed": selection["selected_primary_seed"],
                "validation_rmse_pp": selection["selected_primary_validation_rmse_pp"],
                "external_evaluated": False,
            },
            indent=2,
        )
    )


def _evaluate_external(output_root: Path) -> None:
    run_dir = output_root / "run_001"
    evaluation_dir = output_root / "external_evaluation"
    if evaluation_dir.exists():
        raise FileExistsError(
            "External evaluation artifacts already exist; refusing a second evaluation."
        )
    selection = _read_json(run_dir / "validation_selection.json")
    if not selection.get("selection_frozen") or selection.get("external_evaluated"):
        raise RuntimeError("Validation selection is not frozen or was already evaluated.")
    if not selection.get("external_evaluation_authorized"):
        raise RuntimeError("External evaluation is not authorized.")
    checkpoint_manifest = _read_json(run_dir / "checkpoint_manifest.json")
    for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME):
        entry = checkpoint_manifest.get(name)
        if not entry:
            raise ValueError(f"Selected model checkpoint is missing from manifest: {name}.")
        checkpoint = run_dir / entry["best_path"]
        if _sha256(checkpoint) != entry["best_sha256"]:
            raise ValueError(f"Selected checkpoint hash mismatch before evaluation: {name}.")

    dataset_dir = output_root / "frozen_dataset"
    freeze_manifest = _read_json(output_root / "frozen_dataset_manifest_v2.json")
    if freeze_manifest.get("external_npz_loaded") is not False:
        raise ValueError("External partition was loaded before validation selection.")
    external = _load_partition(dataset_dir / "external_test.npz")
    if (
        external.sample_count != EXPECTED_SAMPLE_COUNTS["external_test"]
        or external.cell_count != EXPECTED_ACTIVE_CELL_COUNTS["external_test"]
    ):
        raise ValueError("External Test sample/cell counts differ from the frozen manifest.")
    external_manifest = _read_csv(dataset_dir / "observation_manifest.csv")
    expected_external_keys = {
        "|".join((row["dataset_part"], row["physical_cell_id"], row["rpt_index"]))
        for row in external_manifest
        if row["status"] == "ACCEPTED" and row["final2_partition"] == "external_test"
    }
    if set(external.keys) != expected_external_keys:
        raise ValueError("External Test NPZ keys differ from its Accepted manifest keys.")

    scaler_payload = _read_json(run_dir / "normalization_parameters.json")
    scalers = {
        name: PopulationZScore.from_dict(scaler_payload[name])
        for name in ("discharge", "charge", "physics", "target")
    }
    from bafuse.hybrid.final2_dataset import Final2Scalers

    final2_scalers = Final2Scalers(**scalers)
    external_inputs = _tensor_inputs(external.arrays, final2_scalers)
    device = resolve_device()
    amp_enabled = device.type == "cuda"
    actual = external.arrays.y_soh.reshape(-1)
    metrics: dict[str, Any] = {}
    prediction_rows: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    macro_rows: list[dict[str, Any]] = []
    prediction_by_model: dict[str, np.ndarray] = {}
    for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME):
        entry = checkpoint_manifest[name]
        checkpoint_path = run_dir / entry["best_path"]
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = (
            Step4CNN1DBaseline()
            if name == STEP4_BASELINE_NAME
            else build_final2_model(name)
        ).to(device)
        model.load_state_dict(payload["model_state_dict"])
        model.eval()
        predicted = _predict_percent(
            model,
            name,
            external_inputs,
            scalers["target"],
            device,
            amp_enabled,
        )
        prediction_by_model[name] = predicted
        metrics[name] = _metric_values(actual, predicted, external.cell_count)
        cell_rows, macro = _per_cell_rows(
            name=name,
            seed=entry["seed"],
            partition_name="external_test",
            partition=external,
            actual=actual,
            predicted=predicted,
        )
        per_cell_rows.extend(cell_rows)
        macro_rows.append(macro)
        for index, key in enumerate(external.keys):
            part, cell, rpt = key.split("|")
            prediction_rows.append(
                {
                    "model": name,
                    "seed": entry["seed"],
                    "dataset_part": part,
                    "physical_cell_id": cell,
                    "rpt_index": rpt,
                    "target_soh_percent": float(actual[index]),
                    "predicted_soh_percent": float(predicted[index]),
                    "residual_prediction_minus_actual_pp": float(
                        predicted[index] - actual[index]
                    ),
                }
            )
    mean_prediction = np.full(
        external.sample_count,
        float(selection["train_mean_soh_percent"]),
        dtype=np.float64,
    )
    metrics[TRAIN_MEAN_NAME] = _metric_values(
        actual, mean_prediction, external.cell_count
    )
    cell_rows, macro = _per_cell_rows(
        name=TRAIN_MEAN_NAME,
        seed="train_mean",
        partition_name="external_test",
        partition=external,
        actual=actual,
        predicted=mean_prediction,
    )
    per_cell_rows.extend(cell_rows)
    macro_rows.append(macro)
    for index, key in enumerate(external.keys):
        part, cell, rpt = key.split("|")
        prediction_rows.append(
            {
                "model": TRAIN_MEAN_NAME,
                "seed": "train_mean",
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "target_soh_percent": float(actual[index]),
                "predicted_soh_percent": float(mean_prediction[index]),
                "residual_prediction_minus_actual_pp": float(
                    mean_prediction[index] - actual[index]
                ),
            }
        )
    for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME):
        entry = checkpoint_manifest[name]
        if _sha256(run_dir / entry["best_path"]) != entry["best_sha256"]:
            raise RuntimeError(f"Checkpoint changed during external evaluation: {name}.")

    comparison: list[dict[str, Any]] = []
    selected_validation = {
        row["model"]: row
        for row in _read_csv(run_dir / "selected_validation_metrics.csv")
    }
    for name in FINAL2_MODEL_NAMES:
        val = selected_validation[name]
        ext = metrics[name]
        comparison.append(
            {
                "model": name,
                "seed": val["seed"],
                "train_N": val["train_N"],
                "train_N_cells": val["train_N_cells"],
                "train_MAE": val["train_MAE"],
                "train_RMSE": val["train_RMSE"],
                "train_R2": val["train_R2"],
                "validation_N": val["validation_N"],
                "validation_N_cells": val["validation_N_cells"],
                "validation_MAE": val["validation_MAE"],
                "validation_RMSE": val["validation_RMSE"],
                "validation_R2": val["validation_R2"],
                "external_N": ext["N"],
                "external_N_cells": ext["N_cells"],
                "external_MAE": ext["MAE"],
                "external_RMSE": ext["RMSE"],
                "external_R2": ext["R2"],
            }
        )
    baseline_comparison = []
    for name in (STEP4_BASELINE_NAME, TRAIN_MEAN_NAME):
        val = selected_validation[name] if name in selected_validation else None
        ext = metrics[name]
        baseline_comparison.append(
            {
                "model": name,
                "seed": val["seed"] if val else "train_mean",
                "train_N": val["train_N"] if val else EXPECTED_SAMPLE_COUNTS["train"],
                "train_N_cells": val["train_N_cells"] if val else EXPECTED_ACTIVE_CELL_COUNTS["train"],
                "train_MAE": val["train_MAE"] if val else None,
                "train_RMSE": val["train_RMSE"] if val else None,
                "train_R2": val["train_R2"] if val else None,
                "validation_N": val["validation_N"] if val else EXPECTED_SAMPLE_COUNTS["validation"],
                "validation_N_cells": val["validation_N_cells"] if val else EXPECTED_ACTIVE_CELL_COUNTS["validation"],
                "validation_MAE": val["validation_MAE"] if val else None,
                "validation_RMSE": val["validation_RMSE"] if val else None,
                "validation_R2": val["validation_R2"] if val else None,
                "external_N": ext["N"],
                "external_N_cells": ext["N_cells"],
                "external_MAE": ext["MAE"],
                "external_RMSE": ext["RMSE"],
                "external_R2": ext["R2"],
            }
        )

    evaluation_dir.mkdir(parents=True)
    _write_csv(evaluation_dir / "external_predictions.csv", prediction_rows)
    _write_csv(evaluation_dir / "per_cell_external_metrics.csv", per_cell_rows)
    _write_csv(evaluation_dir / "macro_external_metrics.csv", macro_rows)
    _write_csv(evaluation_dir / "final_comparison.csv", comparison)
    _write_csv(evaluation_dir / "baseline_comparison.csv", baseline_comparison)
    _write_json(
        evaluation_dir / "external_metrics.json",
        {
            "dataset_identifier": DATASET_IDENTIFIER,
            "external_test_loaded_after_validation_selection": True,
            "external_test_evaluation_count": 1,
            "sample_count": external.sample_count,
            "cell_count": external.cell_count,
            "metrics_by_model": metrics,
        },
    )
    external_run_manifest = {
        "schema_version": 1,
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_identifier": DATASET_IDENTIFIER,
        "dataset_sha256_manifest_sha256": selection[
            "dataset_sha256_manifest_sha256"
        ],
        "validation_selection_sha256": _sha256(run_dir / "validation_selection.json"),
        "external_npz_sha256": _sha256(dataset_dir / "external_test.npz"),
        "checkpoint_sha256": {
            name: checkpoint_manifest[name]["best_sha256"]
            for name in (*FINAL2_MODEL_NAMES, STEP4_BASELINE_NAME)
        },
        "external_test_evaluation_count": 1,
        "sample_count": external.sample_count,
        "cell_count": external.cell_count,
    }
    _write_json(evaluation_dir / "external_evaluation_manifest.json", external_run_manifest)
    final_lines = [
        "# BaFuse Final2 - Final2 model comparison",
        "",
        f"- Dataset version: `{DATASET_IDENTIFIER}`.",
        "- Target: same-time reconstructed Step 4 capacity retention, using same-cell earliest mapped Step 4 BoL; not Oxford official ground truth.",
        "- G10-Cell2 RPT10: GUIDE_VERIFIED, External Test; its Step 10 zero-current row was excluded only from the Step 10 input after the continuity checks passed. Its Step 4 target is unchanged.",
        "- Split: Train 99 observations/9 active cells; Validation 40/3; External Test 74/12.",
        "- External Test was not loaded for training, scaler fitting, early stopping, or model selection; it was evaluated once after validation selection froze.",
        "- Metrics are overall observation-level MAE/RMSE in SoH percentage points and R2; per-cell and macro-cell results are supplementary.",
        "",
        "## Six-model comparison",
        "",
        "| Model | Train N/cells | Train MAE | Train RMSE | Train R2 | Validation N/cells | Validation MAE | Validation RMSE | Validation R2 | External N/cells | External MAE | External RMSE | External R2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        final_lines.append(
            f"| {row['model']} | {row['train_N']}/{row['train_N_cells']} | "
            f"{float(row['train_MAE']):.4f} | {float(row['train_RMSE']):.4f} | "
            f"{float(row['train_R2']):.4f} | "
            f"{row['validation_N']}/{row['validation_N_cells']} | "
            f"{float(row['validation_MAE']):.4f} | "
            f"{float(row['validation_RMSE']):.4f} | "
            f"{float(row['validation_R2']):.4f} | "
            f"{row['external_N']}/{row['external_N_cells']} | "
            f"{float(row['external_MAE']):.4f} | "
            f"{float(row['external_RMSE']):.4f} | "
            f"{float(row['external_R2']):.4f} |"
        )
    final_lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| Baseline | Train MAE/RMSE/R2 | Validation MAE/RMSE/R2 | External MAE/RMSE/R2 |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in baseline_comparison:
        def display(partition: str) -> str:
            values = [row[f"{partition}_{metric}"] for metric in ("MAE", "RMSE", "R2")]
            if any(value is None for value in values):
                return "not recorded"
            return "/".join(f"{float(value):.4f}" for value in values)

        final_lines.append(
            f"| {row['model']} | {display('train')} | "
            f"{display('validation')} | {display('external')} |"
        )
    final_lines.extend(
        [
            "",
            f"Best validation-selected primary model: `{selection['selected_primary_model']}` "
            f"(seed {selection['selected_primary_seed']}, Validation RMSE "
            f"{float(selection['selected_primary_validation_rmse_pp']):.4f} pp).",
            "",
            "External evaluation reports unseen-cell domain-shift performance. It is not a future-RPT forecast.",
            "",
        ]
    )
    (output_root / "final_report.md").write_text(
        "\n".join(final_lines), encoding="utf-8"
    )
    selection["external_evaluated"] = True
    selection["external_evaluation_manifest_sha256"] = _sha256(
        evaluation_dir / "external_evaluation_manifest.json"
    )
    _write_json(run_dir / "validation_selection.json", selection)
    print(
        json.dumps(
            {
                "status": "external_evaluation_complete",
                "evaluation_directory": str(evaluation_dir),
                "sample_count": external.sample_count,
                "cell_count": external.cell_count,
                "best_validation_model": selection["selected_primary_model"],
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Final2 models or perform the single authorized external evaluation."
    )
    parser.add_argument("--phase", choices=("train", "external"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    if arguments.phase == "train":
        _train(arguments.output_root.resolve(), arguments.device)
    else:
        _evaluate_external(arguments.output_root.resolve())


if __name__ == "__main__":
    main()
