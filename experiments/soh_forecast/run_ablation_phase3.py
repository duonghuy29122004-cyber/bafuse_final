from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
AUDIT_DIR = PHASE2_DIR / "evaluation_audit_v1"
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_ablation_v1"
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PHASE2_DIR))
sys.path.insert(0, str(AUDIT_DIR))

import run_phase2 as phase2
from audit_soh_evaluation import _load_supplemental_cell2
from bafuse.hybrid.soh_fusion import CurveEncoder, PHYSICS_FEATURE_NAMES


SEEDS = (211, 212, 213)
PARTITIONS = ("train", "validation", "external_test")
CURVE_COLUMNS = tuple(f"voltage_progress_{index:03d}" for index in range(128))
MODALITIES = ("discharge", "charge", "physics")
CONFIGURATIONS: dict[str, tuple[str, ...]] = {
    "M1": ("discharge",),
    "M2": ("discharge", "charge"),
    "M3": ("discharge", "physics"),
    "M4": ("discharge", "charge", "physics"),
    "M5": ("charge", "physics"),
    "M6": ("charge",),
    "M7": ("physics",),
}
MODALITY_FEATURES: dict[str, tuple[str, ...]] = {
    "discharge": CURVE_COLUMNS,
    "charge": CURVE_COLUMNS,
    "physics": PHYSICS_FEATURE_NAMES,
}
FEATURE_FILES = {
    "discharge": PHASE2_DIR / "discharge_features.csv",
    "charge": PHASE2_DIR / "charge_features.csv",
    "physics": PHASE2_DIR / "physics_features.csv",
}
PROTECTED_FILES = (
    ROOT / "models" / "checkpoints" / "part1_cnn1d" / "cnn1d_checkpoint.pt",
    PHASE2_DIR / "fusion_checkpoint.pt",
    ROOT / "src" / "bafuse" / "labeling" / "torch_pydma.py",
    ROOT / "src" / "bafuse" / "labeling" / "pydma_policy.py",
    ROOT / "src" / "bafuse" / "labeling" / "pydma_runner.py",
    ROOT / "reports" / "bafuse_three_encoder_v1_hybrid" / "pydma_curve_level_results.csv",
    ROOT / "reports" / "bafuse_three_encoder_v1_hybrid" / "pydma_quality_report.csv",
    ROOT / "reports" / "bafuse_three_encoder_v1_hybrid" / "pydma_rpt_aggregated.csv",
    ROOT / "reports" / "bafuse_final_data_validation_v2" / "pydma_optimized_parameters.csv",
    ROOT / "reports" / "bafuse_final_data_validation_v2" / "pydma_bol_parameters.csv",
)


class AblationSoH(nn.Module):
    def __init__(self, modalities: tuple[str, ...]) -> None:
        super().__init__()
        if not modalities or any(name not in MODALITIES for name in modalities):
            raise ValueError(f"Unsupported ablation modalities: {modalities}")
        if len(set(modalities)) != len(modalities):
            raise ValueError("A modality cannot appear more than once.")
        self.modalities = modalities
        encoders: dict[str, nn.Module] = {}
        for name in modalities:
            if name in ("discharge", "charge"):
                encoders[name] = CurveEncoder()
            else:
                encoders[name] = nn.Sequential(
                    nn.Linear(len(PHYSICS_FEATURE_NAMES), 32),
                    nn.GELU(),
                    nn.Linear(32, 32),
                    nn.GELU(),
                )
        self.encoders = nn.ModuleDict(encoders)
        self.regression_head = nn.Sequential(
            nn.Linear(32 * len(modalities), 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(inputs) != set(self.modalities):
            raise ValueError(
                f"Expected inputs for {self.modalities}, got {tuple(inputs)}"
            )
        batch_sizes = {int(value.shape[0]) for value in inputs.values()}
        if len(batch_sizes) != 1:
            raise ValueError("All active modalities must have the same batch size.")
        encoded = [self.encoders[name](inputs[name]) for name in self.modalities]
        return self.regression_head(torch.cat(encoded, dim=1))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required Phase 3 input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(part: Any, cell: Any, rpt: Any) -> tuple[str, str, str]:
    part_text = str(int(float(part)))
    cell_text = str(cell).strip()
    rpt_text = str(rpt).strip()
    if rpt_text.startswith("RPT"):
        rpt_text = rpt_text[3:]
    return part_text, cell_text, str(int(float(rpt_text)))


def _finite_float(value: Any, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric value for {field}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite numeric value for {field}: {value!r}")
    return parsed


def _feature_map(
    path: Path, columns: tuple[str, ...]
) -> dict[tuple[str, str, str], np.ndarray]:
    rows = _read_csv(path)
    result: dict[tuple[str, str, str], np.ndarray] = {}
    for row in rows:
        key = _key(
            row.get("dataset_part", row.get("part", "")),
            row["physical_cell_id"],
            row.get("rpt_index", row.get("measurement_id", "")),
        )
        if key in result:
            raise ValueError(f"Duplicate modality key in {path.name}: {key}")
        values = np.asarray(
            [_finite_float(row[column], f"{path.name}:{column}") for column in columns],
            dtype=np.float32,
        )
        result[key] = values
    return result


def _load_common_cohort() -> tuple[list[dict[str, Any]], str, dict[str, str]]:
    split_by_cell, split_version = phase2._load_split()
    feature_manifest = _read_csv(PHASE2_DIR / "multimodal_feature_manifest.csv")
    modality_maps = {
        name: _feature_map(FEATURE_FILES[name], MODALITY_FEATURES[name])
        for name in MODALITIES
    }
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in feature_manifest:
        if row["fusion_eligible"].casefold() != "true":
            continue
        key = _key(row.get("dataset_part", row.get("part", "")), row["physical_cell_id"], row["rpt_index"])
        if key in records:
            raise ValueError(f"Duplicate complete-modality observation: {key}")
        if key[1] not in split_by_cell:
            raise ValueError(f"Cell absent from locked split manifest: {key[1]}")
        partition = split_by_cell[key[1]]
        if row["partition"] != partition:
            raise ValueError(f"Feature manifest split mismatch for {key}")
        target = _finite_float(row["soh_percent"], f"SOH label for {key}")
        features: dict[str, np.ndarray] = {}
        for modality, mapping in modality_maps.items():
            if key not in mapping:
                raise ValueError(f"Missing {modality} features for eligible key {key}")
            features[modality] = mapping[key]
        if row["step4_raw_status"] != "verified_raw_segment":
            raise ValueError(f"Unverified Step 4 source for eligible key {key}")
        if row["step10_raw_status"] != "verified_raw_segment":
            raise ValueError(f"Unverified Step 10 source for eligible key {key}")
        if row["label_scope"] == "not_labeled":
            raise ValueError(f"Unlabeled observation marked fusion-eligible: {key}")
        records[key] = {
            "key": key,
            "part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": int(key[2]),
            "measurement_id": row["measurement_id"],
            "partition": partition,
            "soh_percent": target,
            "label_scope": row["label_scope"],
            "source_file": row["source_file"],
            "step4_segment_ref": row["step4_segment_ref"],
            "step10_segment_ref": row["step10_segment_ref"],
            "feature_source": "saved_verified_phase2_feature_tables",
            "identity_provenance": "canonical_observation_manifest",
            "features": features,
            "split_version": split_version,
        }

    supplemental, _ = _load_supplemental_cell2()
    supplemental_label_rows = _read_csv(
        AUDIT_DIR / "supplemental_g10_cell2_labels.csv"
    )
    saved_supplemental = {
        _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"]): row
        for row in supplemental_label_rows
    }
    eligible_supplemental = 0
    for item in supplemental:
        label = item["label"]
        row = item["row"]
        if label["validation_status"] != "calculated_from_measured_capacity":
            continue
        key = _key(label["dataset_part"], label["physical_cell_id"], label["measurement_id"])
        if key in records:
            raise ValueError(f"Supplemental label duplicates an existing feature key: {key}")
        prior_label = saved_supplemental.get(key)
        if prior_label is None:
            raise ValueError(f"Missing audited supplemental label for {key}")
        if not math.isclose(
            _finite_float(prior_label["soh_percent"], f"saved label {key}"),
            _finite_float(label["soh_percent"], f"reconstructed label {key}"),
            rel_tol=0.0,
            abs_tol=1e-8,
        ):
            raise ValueError(f"Supplemental SOH provenance mismatch for {key}")
        if not label["fusion_eligible"]:
            continue
        if key[1] not in split_by_cell or split_by_cell[key[1]] != "external_test":
            raise ValueError(f"Supplemental Part 3 key is not external-only: {key}")
        step4 = row["step4_result"]
        step10 = row["step10_result"]
        if not step4 or not step4["valid"] or not step10 or not step10["valid"]:
            raise ValueError(f"Supplemental key has invalid raw curves: {key}")
        if not step4.get("temperature_complete", False):
            raise ValueError(f"Supplemental key lacks complete temperature: {key}")
        if step4["row_start"] != int(label["source_row_start"]) or step4["row_end"] != int(label["source_row_end"]):
            raise ValueError(f"Supplemental Step 4 boundaries disagree for {key}")
        if step10["row_start"] != int(label["step10_source_row_start"]) or step10["row_end"] != int(label["step10_source_row_end"]):
            raise ValueError(f"Supplemental Step 10 boundaries disagree for {key}")
        features = {
            "discharge": np.asarray(step4["curve"], dtype=np.float32),
            "charge": np.asarray(step10["curve"], dtype=np.float32),
            "physics": np.asarray(
                [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES],
                dtype=np.float32,
            ),
        }
        if any(not np.isfinite(values).all() for values in features.values()):
            raise ValueError(f"Non-finite supplemental feature values for {key}")
        records[key] = {
            "key": key,
            "part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": int(key[2]),
            "measurement_id": label["measurement_id"],
            "partition": "external_test",
            "soh_percent": _finite_float(label["soh_percent"], f"SOH label for {key}"),
            "label_scope": "supplemental_provisional_part3_capacity_ratio",
            "source_file": label["source_file"],
            "step4_segment_ref": label["step4_segment_ref"],
            "step10_segment_ref": label["step10_segment_ref"],
            "feature_source": "reconstructed_from_verified_G10_raw_segments",
            "identity_provenance": label["identity_provenance"],
            "features": features,
            "split_version": split_version,
        }
        eligible_supplemental += 1

    ordered = sorted(
        records.values(),
        key=lambda item: (
            PARTITIONS.index(item["partition"]),
            item["key"][0],
            item["physical_cell_id"],
            item["rpt_index"],
        ),
    )
    by_partition = {
        partition: [item for item in ordered if item["partition"] == partition]
        for partition in PARTITIONS
    }
    expected_counts = {"train": 106, "validation": 41, "external_test": 73}
    observed_counts = {name: len(items) for name, items in by_partition.items()}
    if observed_counts != expected_counts:
        raise ValueError(f"Unexpected common cohort coverage: {observed_counts}")
    if eligible_supplemental != 9:
        raise ValueError(
            f"Expected 9 eligible G10-Cell2 supplemental keys, got {eligible_supplemental}"
        )
    if any(item["part"] == "3" for item in by_partition["train"] + by_partition["validation"]):
        raise ValueError("Part 3 observations cannot enter training or validation.")

    partition_cells = {
        partition: {item["physical_cell_id"] for item in rows}
        for partition, rows in by_partition.items()
    }
    if (
        partition_cells["train"] & partition_cells["validation"]
        or partition_cells["train"] & partition_cells["external_test"]
        or partition_cells["validation"] & partition_cells["external_test"]
    ):
        raise ValueError("Physical-cell leakage detected across partitions.")

    shared_external = _read_csv(AUDIT_DIR / "shared_subset_model_comparison.csv")
    expected_external_keys = {
        _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        for row in shared_external
        if row["partition"] == "external_test"
    }
    actual_external_keys = {
        item["key"] for item in by_partition["external_test"]
    }
    if actual_external_keys != expected_external_keys:
        missing = sorted(expected_external_keys - actual_external_keys)
        extra = sorted(actual_external_keys - expected_external_keys)
        raise ValueError(
            f"External key mismatch against corrected audit; missing={missing}, extra={extra}"
        )

    saved_predictions = _read_csv(PHASE2_DIR / "fusion_predictions.csv")
    saved_internal_keys = {
        _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        for row in saved_predictions
    }
    for item in ordered:
        if item["physical_cell_id"] == "G10-Cell2":
            continue
        if item["key"] not in saved_internal_keys:
            raise ValueError(f"Missing historical keyed prediction for {item['key']}")

    return ordered, split_version, split_by_cell


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def _fit_scaler(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    if not np.isfinite(mean).all() or not np.isfinite(scale).all():
        raise ValueError(f"Non-finite training-only {name} scaler.")
    if np.any(scale < 1e-8):
        raise ValueError(f"Constant or near-constant training-only {name} input.")
    return mean.astype(np.float32), scale.astype(np.float32)


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Metrics require non-empty aligned one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics cannot be computed from non-finite values.")
    errors = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae_soh_percentage_points": float(np.mean(np.abs(errors))),
        "rmse_soh_percentage_points": float(np.sqrt(np.mean(errors**2))),
        "r2": 1.0 - float(np.sum(errors**2)) / denominator if denominator > 0 else None,
    }


def _arrays_for(
    records: list[dict[str, Any]], modalities: tuple[str, ...]
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    arrays = {
        name: np.stack([record["features"][name] for record in records]).astype(
            np.float32
        )
        for name in modalities
    }
    target = np.asarray([record["soh_percent"] for record in records], dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Target values include non-finite labels.")
    return arrays, target


def _predict_scaled(
    model: AblationSoH,
    arrays: dict[str, np.ndarray],
    device: torch.device,
) -> np.ndarray:
    model.eval()
    tensors = {
        name: torch.from_numpy(values.astype(np.float32)).to(device)
        for name, values in arrays.items()
    }
    with torch.no_grad():
        outputs = model(tensors).reshape(-1)
    predictions = outputs.detach().cpu().numpy()
    if not np.isfinite(predictions).all():
        raise ValueError("Model produced a non-finite prediction.")
    return predictions


def _train_one(
    records: list[dict[str, Any]],
    config_id: str,
    modalities: tuple[str, ...],
    seed: int,
    checkpoint_dir: Path,
    max_epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    _seed_everything(seed)
    train_records = [row for row in records if row["partition"] == "train"]
    validation_records = [row for row in records if row["partition"] == "validation"]
    if any(row["part"] == "3" for row in train_records + validation_records):
        raise ValueError("Part 3 is evaluation-only.")
    train_cells = {row["physical_cell_id"] for row in train_records}
    validation_cells = {row["physical_cell_id"] for row in validation_records}
    if train_cells & validation_cells:
        raise ValueError("Train and validation contain the same physical cell.")

    train_inputs, train_target = _arrays_for(train_records, modalities)
    validation_inputs, validation_target = _arrays_for(validation_records, modalities)
    scalers: dict[str, dict[str, list[float]]] = {}
    normalized_train: dict[str, np.ndarray] = {}
    normalized_validation: dict[str, np.ndarray] = {}
    for name in modalities:
        mean, scale = _fit_scaler(train_inputs[name], name)
        scalers[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
        normalized_train[name] = (train_inputs[name] - mean) / scale
        normalized_validation[name] = (validation_inputs[name] - mean) / scale
    target_mean = float(train_target.mean())
    target_scale = float(train_target.std())
    if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale < 1e-8:
        raise ValueError("Training-only target scaler is not usable.")
    normalized_train_target = (train_target - target_mean) / target_scale
    normalized_validation_target = (validation_target - target_mean) / target_scale

    tensors = [
        torch.from_numpy(normalized_train[name].astype(np.float32))
        for name in modalities
    ]
    tensors.append(torch.from_numpy(normalized_train_target.astype(np.float32)))
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, len(train_records)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = AblationSoH(modalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    validation_tensors = {
        name: torch.from_numpy(values.astype(np.float32)).to(device)
        for name, values in normalized_validation.items()
    }
    validation_target_tensor = torch.from_numpy(
        normalized_validation_target.astype(np.float32)
    ).to(device)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in loader:
            inputs = {
                name: batch[index].to(device)
                for index, name in enumerate(modalities)
            }
            target = batch[-1].to(device).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(inputs)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_tensors)
            validation_loss = float(
                criterion(
                    validation_prediction,
                    validation_target_tensor.reshape(-1, 1),
                ).item()
            )
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": statistics.mean(batch_losses),
                "validation_mse_scaled": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a validation-selected checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "ablation_checkpoint.pt"
    checkpoint_payload = {
        "model_configuration": config_id,
        "modalities": modalities,
        "model_state_dict": best_state,
        "input_scalers": scalers,
        "target_mean_train_only": target_mean,
        "target_scale_train_only": target_scale,
        "training_seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "training_observation_keys": [row["key"] for row in train_records],
        "validation_observation_keys": [row["key"] for row in validation_records],
        "split_version": records[0]["split_version"],
        "pretraining": "none; all Phase 3 ablation models initialized independently",
        "preprocessing_fit_scope": "training observations only",
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "loss": "MSE on train-standardized SOH percent",
        "device": str(device),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    with (checkpoint_dir / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    all_predictions: dict[tuple[str, str, str], float] = {}
    partition_arrays: dict[str, dict[str, np.ndarray]] = {}
    for partition in PARTITIONS:
        selected = [row for row in records if row["partition"] == partition]
        raw_inputs, _ = _arrays_for(selected, modalities)
        normalized_inputs = {
            name: (
                raw_inputs[name] - np.asarray(scalers[name]["mean"], dtype=np.float32)
            )
            / np.asarray(scalers[name]["scale"], dtype=np.float32)
            for name in modalities
        }
        partition_arrays[partition] = normalized_inputs
        scaled_predictions = _predict_scaled(model, normalized_inputs, device)
        predictions = scaled_predictions * target_scale + target_mean
        for row, prediction in zip(selected, predictions):
            all_predictions[row["key"]] = float(prediction)

    return {
        "model": model,
        "modalities": modalities,
        "seed": seed,
        "checkpoint_path": checkpoint_path,
        "checkpoint_payload": checkpoint_payload,
        "history": history,
        "predictions": all_predictions,
        "partition_arrays": partition_arrays,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "scalers": scalers,
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
    }


def _metric_rows(
    records: list[dict[str, Any]],
    run_results: dict[tuple[str, int], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for (config_id, seed), result in sorted(run_results.items()):
        by_key = result["predictions"]
        if set(by_key) != {record["key"] for record in records}:
            raise ValueError(f"Prediction key coverage mismatch for {config_id}/{seed}")
        for partition in PARTITIONS:
            selected = [record for record in records if record["partition"] == partition]
            actual = np.asarray([record["soh_percent"] for record in selected])
            predicted = np.asarray([by_key[record["key"]] for record in selected])
            scores = _metrics(actual, predicted)
            rows.append(
                {
                    "model_configuration": config_id,
                    "training_seed": seed,
                    "partition": partition,
                    "metric_scope": "overall",
                    "observation_count": len(selected),
                    "physical_cell_count": len(
                        {record["physical_cell_id"] for record in selected}
                    ),
                    **scores,
                }
            )
        for record in records:
            prediction = by_key[record["key"]]
            prediction_rows.append(
                {
                    "model_configuration": config_id,
                    "training_seed": seed,
                    "partition": record["partition"],
                    "dataset_part": record["part"],
                    "physical_cell_id": record["physical_cell_id"],
                    "measurement_id": record["measurement_id"],
                    "rpt_index": record["rpt_index"],
                    "source_file": record["source_file"],
                    "label_scope": record["label_scope"],
                    "reference_soh_percent": record["soh_percent"],
                    "predicted_soh_percent": prediction,
                    "absolute_error_soh_percentage_points": abs(
                        prediction - record["soh_percent"]
                    ),
                    "checkpoint": str(result["checkpoint_path"].relative_to(ROOT)),
                }
            )
    return rows, prediction_rows


def _seed_summary(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id in CONFIGURATIONS:
        for partition in PARTITIONS:
            selected = [
                row
                for row in metric_rows
                if row["model_configuration"] == config_id
                and row["partition"] == partition
            ]
            summary: dict[str, Any] = {
                "model_configuration": config_id,
                "partition": partition,
                "successful_training_seeds": len(selected),
            }
            for metric in (
                "mae_soh_percentage_points",
                "rmse_soh_percentage_points",
                "r2",
            ):
                values = [
                    float(row[metric]) for row in selected if row[metric] is not None
                ]
                summary[f"mean_{metric}"] = (
                    float(np.mean(values)) if values else None
                )
                summary[f"std_{metric}"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else None
                )
            rows.append(summary)
    return rows


def _per_cell_metrics(
    prediction_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    groups: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        groups[
            (
                row["model_configuration"],
                int(row["training_seed"]),
                row["partition"],
                row["physical_cell_id"],
            )
        ].append(row)
    rows: list[dict[str, Any]] = []
    for (config_id, seed, partition, cell), selected in sorted(groups.items()):
        actual = np.asarray([float(row["reference_soh_percent"]) for row in selected])
        predicted = np.asarray([float(row["predicted_soh_percent"]) for row in selected])
        rows.append(
            {
                "model_configuration": config_id,
                "training_seed": seed,
                "partition": partition,
                "physical_cell_id": cell,
                "observation_count": len(selected),
                "metric_scope": "per_cell_diagnostic",
                **_metrics(actual, predicted),
            }
        )
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[
            (
                row["model_configuration"],
                row["partition"],
                row["physical_cell_id"],
            )
        ].append(row)
    summary_rows: list[dict[str, Any]] = []
    for (config_id, partition, cell), selected in sorted(by_cell.items()):
        result: dict[str, Any] = {
            "model_configuration": config_id,
            "partition": partition,
            "physical_cell_id": cell,
            "successful_training_seeds": len(selected),
        }
        for metric in (
            "mae_soh_percentage_points",
            "rmse_soh_percentage_points",
            "r2",
        ):
            values = [float(row[metric]) for row in selected if row[metric] is not None]
            result[f"mean_{metric}"] = float(np.mean(values)) if values else None
            result[f"std_{metric}"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else None
            )
        summary_rows.append(result)
    return rows, summary_rows


def _contribution_rows(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_map: dict[tuple[str, int, str, str, str, str], dict[str, Any]] = {}
    for row in prediction_rows:
        key = (
            row["model_configuration"],
            int(row["training_seed"]),
            row["partition"],
            row["dataset_part"],
            row["physical_cell_id"],
            row["measurement_id"],
        )
        if key in pred_map:
            raise ValueError(f"Duplicate prediction in paired contribution set: {key}")
        pred_map[key] = row
    leave_one_out = {
        "discharge": "M5",
        "charge": "M3",
        "physics": "M2",
    }
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for partition in PARTITIONS:
            for encoder, loo_model in leave_one_out.items():
                full_rows = {
                    key[3:]: value
                    for key, value in pred_map.items()
                    if key[:3] == ("M4", seed, partition)
                }
                loo_rows = {
                    key[3:]: value
                    for key, value in pred_map.items()
                    if key[:3] == (loo_model, seed, partition)
                }
                if set(full_rows) != set(loo_rows):
                    raise ValueError(
                        f"Unpaired prediction keys for M4 vs {loo_model}, "
                        f"{partition}, seed {seed}"
                    )
                keys = sorted(full_rows)
                actual = np.asarray(
                    [float(full_rows[key]["reference_soh_percent"]) for key in keys]
                )
                full_pred = np.asarray(
                    [float(full_rows[key]["predicted_soh_percent"]) for key in keys]
                )
                loo_pred = np.asarray(
                    [float(loo_rows[key]["predicted_soh_percent"]) for key in keys]
                )
                full_metric = _metrics(actual, full_pred)
                loo_metric = _metrics(actual, loo_pred)
                rows.append(
                    {
                        "encoder_name": encoder,
                        "full_model_identifier": "M4",
                        "leave_one_encoder_out_model": loo_model,
                        "partition": partition,
                        "training_seed": seed,
                        "observation_count": len(keys),
                        "delta_mae_soh_percentage_points": (
                            loo_metric["mae_soh_percentage_points"]
                            - full_metric["mae_soh_percentage_points"]
                        ),
                        "delta_rmse_soh_percentage_points": (
                            loo_metric["rmse_soh_percentage_points"]
                            - full_metric["rmse_soh_percentage_points"]
                        ),
                        "positive_delta_means_including_encoder_improves": True,
                        "paired_observation_keys": True,
                    }
                )
    return rows


def _permutation_rows(
    records: list[dict[str, Any]],
    run_results: dict[tuple[str, int], dict[str, Any]],
    repetitions: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = run_results.get(("M4", seed))
        if result is None:
            continue
        validation = [row for row in records if row["partition"] == "validation"]
        actual = np.asarray([row["soh_percent"] for row in validation])
        base_inputs = result["partition_arrays"]["validation"]
        baseline_pred = (
            _predict_scaled(result["model"], base_inputs, device)
            * result["target_scale"]
            + result["target_mean"]
        )
        baseline_mae = _metrics(actual, baseline_pred)["mae_soh_percentage_points"]
        for modality in MODALITIES:
            for repetition in range(1, repetitions + 1):
                random_seed = 1_000_000 + seed * 100 + repetition
                permutation = np.random.default_rng(random_seed).permutation(
                    len(validation)
                )
                perturbed_inputs = {
                    name: values.copy() for name, values in base_inputs.items()
                }
                perturbed_inputs[modality] = base_inputs[modality][permutation]
                perturbed_scaled = _predict_scaled(
                    result["model"], perturbed_inputs, device
                )
                perturbed_prediction = (
                    perturbed_scaled * result["target_scale"] + result["target_mean"]
                )
                perturbed_mae = _metrics(
                    actual, perturbed_prediction
                )["mae_soh_percentage_points"]
                rows.append(
                    {
                        "encoder_name": modality,
                        "training_seed": seed,
                        "permutation_repetition": repetition,
                        "permutation_seed": random_seed,
                        "partition": "validation",
                        "observation_count": len(validation),
                        "physical_cell_count": len(
                            {row["physical_cell_id"] for row in validation}
                        ),
                        "baseline_validation_mae": baseline_mae,
                        "perturbed_validation_mae": perturbed_mae,
                        "change_in_mae": perturbed_mae - baseline_mae,
                        "permuted_unit": "entire modality vector across validation observations",
                    }
                )
    return rows


def _external_coverage_rows(
    records: list[dict[str, Any]],
    split_by_cell: dict[str, str],
    successful_seeds: dict[str, int],
) -> list[dict[str, Any]]:
    audit_coverage = _read_csv(AUDIT_DIR / "external_12_cell_coverage.csv")
    audit_exclusions = _read_csv(AUDIT_DIR / "external_exclusion_reasons.csv")
    audit_by_cell = {row["physical_cell_id"]: row for row in audit_coverage}
    if set(audit_by_cell) != {
        cell for cell, partition in split_by_cell.items() if partition == "external_test"
    }:
        raise ValueError("Coverage audit does not list exactly the 12 locked external cells.")
    excluded_reasons: dict[str, set[str]] = defaultdict(set)
    excluded_counts: dict[str, int] = defaultdict(int)
    for row in audit_exclusions:
        if row["eligible_full_coverage_cnn1d_prediction"].casefold() == "true" and row[
            "eligible_fusion_prediction"
        ].casefold() != "true":
            cell = row["physical_cell_id"]
            excluded_counts[cell] += 1
            reason = row["exact_exclusion_reason"] or row["current_status"]
            excluded_reasons[cell].add(reason)
    common_counts = CounterRecord(records, "external_test")
    rows: list[dict[str, Any]] = []
    for cell in sorted(audit_by_cell):
        audit = audit_by_cell[cell]
        row: dict[str, Any] = {
            "physical_cell_id": cell,
            "external_split_physical_cells": 12,
            "available_step4_labeled_observations": audit[
                "eligible_full_coverage_cnn1d_predictions"
            ],
            "complete_modality_eligible_observations": audit[
                "eligible_fusion_predictions"
            ],
            "common_ablation_observations_per_configuration": common_counts.get(cell, 0),
            "missing_modality_exclusion_count": excluded_counts.get(cell, 0),
            "missing_modality_exclusion_reasons": "; ".join(
                sorted(excluded_reasons.get(cell, set()))
            ),
            "supplemental_g10_cell2_label_count": (
                10 if cell == "G10-Cell2" else 0
            ),
            "supplemental_g10_cell2_complete_modality_count": (
                common_counts.get(cell, 0) if cell == "G10-Cell2" else 0
            ),
            "supplemental_identity_provenance": (
                "user-confirmed; Guide omits explicit cell number; Part 3 labels provisional"
                if cell == "G10-Cell2"
                else ""
            ),
            "exact_audit_exclusion_reasons": audit["exact_exclusion_reasons"],
        }
        for config_id in CONFIGURATIONS:
            predicted = common_counts.get(cell, 0) if successful_seeds[config_id] else 0
            row[f"predicted_observations_per_seed_{config_id}"] = predicted
            row[f"total_predictions_across_successful_seeds_{config_id}"] = (
                predicted * successful_seeds[config_id]
            )
        rows.append(row)
    if sum(common_counts.values()) != 73:
        raise ValueError(f"Expected 73 external common-cohort predictions, got {sum(common_counts.values())}")
    return rows


def CounterRecord(
    records: list[dict[str, Any]], partition: str
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        if record["partition"] == partition:
            counts[record["physical_cell_id"]] += 1
    return counts


def _summary_lookup(
    summary_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["model_configuration"], row["partition"]): row for row in summary_rows}


def _format_mean_std(
    row: dict[str, Any], metric: str, decimals: int = 3
) -> str:
    mean = row.get(f"mean_{metric}")
    std = row.get(f"std_{metric}")
    if mean is None:
        return "not available"
    if std is None:
        return f"{float(mean):.{decimals}f} (SD n/a)"
    return f"{float(mean):.{decimals}f} ± {float(std):.{decimals}f}"


def _mean_paired_differences(
    prediction_rows: list[dict[str, Any]],
    left: str,
    right: str,
    partition: str,
) -> tuple[float | None, float | None]:
    values: list[float] = []
    by_config_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["partition"] == partition and row["model_configuration"] in (left, right):
            by_config_seed[
                (row["model_configuration"], int(row["training_seed"]))
            ].append(row)
    for seed in SEEDS:
        left_rows = by_config_seed.get((left, seed), [])
        right_rows = by_config_seed.get((right, seed), [])
        left_map = {
            (r["dataset_part"], r["physical_cell_id"], r["measurement_id"]): r
            for r in left_rows
        }
        right_map = {
            (r["dataset_part"], r["physical_cell_id"], r["measurement_id"]): r
            for r in right_rows
        }
        if not left_map or set(left_map) != set(right_map):
            continue
        keys = sorted(left_map)
        y = np.asarray([float(left_map[key]["reference_soh_percent"]) for key in keys])
        left_pred = np.asarray([float(left_map[key]["predicted_soh_percent"]) for key in keys])
        right_pred = np.asarray([float(right_map[key]["predicted_soh_percent"]) for key in keys])
        values.append(
            _metrics(y, left_pred)["mae_soh_percentage_points"]
            - _metrics(y, right_pred)["mae_soh_percentage_points"]
        )
    if not values:
        return None, None
    return float(np.mean(values)), (
        float(np.std(values, ddof=1)) if len(values) > 1 else None
    )


def _historical_reference_rows() -> list[dict[str, Any]]:
    rows = _read_csv(AUDIT_DIR / "soh_overall_metrics.csv")
    return [
        {
            "reference_type": "historical_single_run",
            "model": row["model"],
            "partition": row["partition"],
            "metric_scope": row["metric_scope"],
            "evaluation_population": row["evaluation_population"],
            "observation_count": row["observation_count"],
            "physical_cell_count": row["physical_cell_count"],
            "mae_soh_percentage_points": row["mae_soh_percentage_points"],
            "rmse_soh_percentage_points": row["rmse_soh_percentage_points"],
            "r2": row["r2"],
            "not_part_of_seed_average": True,
        }
        for row in rows
        if row["model"] in ("historical_cnn1d", "three_encoder_fusion")
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protected_integrity_rows(
    before: dict[str, str], after: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256_before": before[str(path)],
            "sha256_after": after[str(path)],
            "unchanged": before[str(path)] == after[str(path)],
        }
        for path in PROTECTED_FILES
    ]


def _input_checksum_rows() -> list[dict[str, str]]:
    paths = [
        PHASE2_DIR / "multimodal_feature_manifest.csv",
        *FEATURE_FILES.values(),
        PHASE2_DIR / "fusion_predictions.csv",
        AUDIT_DIR / "shared_subset_model_comparison.csv",
        AUDIT_DIR / "external_12_cell_coverage.csv",
        AUDIT_DIR / "external_exclusion_reasons.csv",
        AUDIT_DIR / "supplemental_g10_cell2_labels.csv",
        AUDIT_DIR / "soh_overall_metrics.csv",
        ROOT / phase2.SPLIT_PATH,
        ROOT / phase2.LABEL_PATHS["1"],
        ROOT / phase2.LABEL_PATHS["2"],
        ROOT / phase2.LABEL_PATHS["3"],
        PHASE2_DIR / "run_phase2.py",
        AUDIT_DIR / "audit_soh_evaluation.py",
        ROOT / "src" / "bafuse" / "hybrid" / "soh_fusion.py",
        Path(__file__).resolve(),
    ]
    return [
        {
            "path": str(path.relative_to(ROOT)),
            "sha256": _sha256(path),
        }
        for path in paths
    ]


def _configuration_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config_id, modalities in CONFIGURATIONS.items():
        model = AblationSoH(modalities)
        rows.append(
            {
                "model_configuration": config_id,
                "discharge_encoder": "yes" if "discharge" in modalities else "no",
                "charge_encoder": "yes" if "charge" in modalities else "no",
                "physics_encoder": "yes" if "physics" in modalities else "no",
                "modalities": ";".join(modalities),
                "active_encoder_count": len(modalities),
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                "input_schema": "approved v1 Step 4/Step 10 voltage curves and five approved Step 4 physics features",
                "preprocessing": "per-feature train-only standardization; target standardized on train only",
                "training_observations": sum(
                    row["partition"] == "train" for row in records
                ),
                "validation_observations": sum(
                    row["partition"] == "validation" for row in records
                ),
                "external_observations": sum(
                    row["partition"] == "external_test" for row in records
                ),
                "common_key_cohort": True,
                "initialization": "independent random initialization; no checkpoint warm start",
            }
        )
    return rows


def _training_failure_rows(
    failures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if failures:
        return failures
    return [
        {
            "model_configuration": "",
            "training_seed": "",
            "error_type": "",
            "error_message": "No failures",
            "traceback": "",
        }
    ]


def _contribution_summary_rows(
    contribution_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for encoder in MODALITIES:
        for partition in PARTITIONS:
            selected = [
                row
                for row in contribution_rows
                if row["encoder_name"] == encoder and row["partition"] == partition
            ]
            for metric in ("delta_mae_soh_percentage_points", "delta_rmse_soh_percentage_points"):
                values = [float(row[metric]) for row in selected if row[metric] is not None]
                rows.append(
                    {
                        "encoder_name": encoder,
                        "partition": partition,
                        "difference_metric": metric,
                        "mean_across_paired_seeds": (
                            float(np.mean(values)) if values else None
                        ),
                        "std_across_paired_seeds": (
                            float(np.std(values, ddof=1)) if len(values) > 1 else None
                        ),
                        "paired_seed_count": len(values),
                    }
                )
    return rows


def _plot_results(
    output_dir: Path,
    summary_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "ablation_figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    lookup = _summary_lookup(summary_rows)
    config_ids = list(CONFIGURATIONS)

    for partition, metric, title, filename, ylabel in (
        (
            ("validation", "mae_soh_percentage_points", "Overall validation MAE", "01_validation_mae.png", "MAE (SOH percentage points)"),
            ("validation", "rmse_soh_percentage_points", "Overall validation RMSE", "02_validation_rmse.png", "RMSE (SOH percentage points)"),
            ("validation", "r2", "Overall validation R²", "03_validation_r2.png", "R²"),
            ("external_test", "mae_soh_percentage_points", "External Part 3 MAE (provisional labels)", "04_external_mae.png", "MAE (SOH percentage points)"),
            ("external_test", "rmse_soh_percentage_points", "External Part 3 RMSE (provisional labels)", "05_external_rmse.png", "RMSE (SOH percentage points)"),
        )
    ):
        means = [
            lookup[(config_id, partition)].get(f"mean_{metric}")
            for config_id in config_ids
        ]
        stds = [
            lookup[(config_id, partition)].get(f"std_{metric}")
            for config_id in config_ids
        ]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(
            config_ids,
            [np.nan if value is None else float(value) for value in means],
            yerr=[0.0 if value is None else float(value or 0.0) for value in stds],
            capsize=4,
            color="#4c78a8",
        )
        ax.set_title(title)
        ax.set_xlabel("Ablation configuration")
        ax.set_ylabel(ylabel)
        if partition == "external_test":
            ax.text(
                0.99,
                0.98,
                "Descriptive only; Part 3 labels are provisional",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color="#8c2d04",
            )
        fig.tight_layout()
        fig.savefig(figure_dir / filename, dpi=160)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, metric, title in (
        (axes[0], "delta_mae_soh_percentage_points", "Δ MAE"),
        (axes[1], "delta_rmse_soh_percentage_points", "Δ RMSE"),
    ):
        means: list[float] = []
        stds: list[float] = []
        for encoder in MODALITIES:
            selected = [
                float(row[metric])
                for row in contribution_rows
                if row["encoder_name"] == encoder
                and row["partition"] == "validation"
                and row[metric] is not None
            ]
            means.append(float(np.mean(selected)) if selected else float("nan"))
            stds.append(float(np.std(selected, ddof=1)) if len(selected) > 1 else 0.0)
        ax.bar(MODALITIES, means, yerr=stds, capsize=4, color="#59a14f")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.set_ylabel("LOO model error minus M4 error (pp)")
        ax.set_xlabel("Encoder included in M4")
    fig.suptitle("Paired leave-one-encoder-out validation differences (mean ± SD)")
    fig.tight_layout()
    fig.savefig(figure_dir / "06_leave_one_encoder_out_contribution.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for encoder in MODALITIES:
        values = [
            float(row["change_in_mae"])
            for row in permutation_rows
            if row["encoder_name"] == encoder
        ]
        if values:
            ax.bar(
                encoder,
                float(np.mean(values)),
                yerr=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                capsize=4,
                color="#f28e2b",
            )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Grouped modality permutation importance on validation")
    ax.set_ylabel("Change in validation MAE (pp)")
    ax.set_xlabel("Permuted modality vector")
    fig.tight_layout()
    fig.savefig(figure_dir / "07_grouped_modality_permutation.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(config_ids))
    for offset, partition, label, color in (
        (-0.18, "validation", "Validation", "#4c78a8"),
        (0.18, "external_test", "External Part 3 (provisional)", "#e15759"),
    ):
        means = [
            lookup[(config_id, partition)].get("mean_mae_soh_percentage_points")
            for config_id in config_ids
        ]
        stds = [
            lookup[(config_id, partition)].get("std_mae_soh_percentage_points")
            for config_id in config_ids
        ]
        ax.errorbar(
            x + offset,
            [np.nan if value is None else float(value) for value in means],
            yerr=[0.0 if value is None else float(value or 0.0) for value in stds],
            fmt="o",
            capsize=4,
            label=label,
            color=color,
        )
    ax.set_xticks(x, config_ids)
    ax.set_ylabel("Overall MAE (SOH percentage points)")
    ax.set_title("Validation versus external generalization (mean ± SD across seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "08_validation_external_generalization.png", dpi=160)
    plt.close(fig)


def _write_report(
    output_dir: Path,
    records: list[dict[str, Any]],
    split_version: str,
    metric_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    contribution_rows: list[dict[str, Any]],
    contribution_summary: list[dict[str, Any]],
    permutation_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    configuration_rows: list[dict[str, Any]],
    historical_rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
) -> None:
    summary = _summary_lookup(summary_rows)

    def score(config: str, partition: str, metric: str) -> float | None:
        value = summary[(config, partition)].get(f"mean_{metric}")
        return float(value) if value is not None else None

    def delta_for(left_model: str, right_model: str, partition: str) -> tuple[float | None, float | None]:
        return _mean_paired_differences(
            prediction_rows, left_model, right_model, partition
        )

    val_ranked = sorted(
        CONFIGURATIONS,
        key=lambda model: (
            float("inf")
            if score(model, "validation", "mae_soh_percentage_points") is None
            else score(model, "validation", "mae_soh_percentage_points")
        ),
    )
    ext_ranked = sorted(
        CONFIGURATIONS,
        key=lambda model: (
            float("inf")
            if score(model, "external_test", "mae_soh_percentage_points") is None
            else score(model, "external_test", "mae_soh_percentage_points")
        ),
    )
    individual_ranked = sorted(
        ("M1", "M6", "M7"),
        key=lambda model: (
            float("inf")
            if score(model, "validation", "mae_soh_percentage_points") is None
            else score(model, "validation", "mae_soh_percentage_points")
        ),
    )
    best_by_seed: dict[int, str] = {}
    for seed in SEEDS:
        seed_rows = [
            row
            for row in metric_rows
            if row["training_seed"] == seed and row["partition"] == "validation"
        ]
        if seed_rows:
            best_by_seed[seed] = min(
                seed_rows,
                key=lambda row: float(row["mae_soh_percentage_points"]),
            )["model_configuration"]

    count_by_partition = {
        partition: (
            sum(row["partition"] == partition for row in records),
            len(
                {
                    row["physical_cell_id"]
                    for row in records
                    if row["partition"] == partition
                }
            ),
        )
        for partition in PARTITIONS
    }
    lines = [
        "# BaFuse Phase 3: Three-Encoder Ablation Study",
        "",
        "## Executive summary",
        "",
        f"- Locked split `{split_version}` was preserved. The common complete-modality cohort contains 106 train observations / 9 labeled cells, 41 validation observations / 3 labeled cells, and 73 external observations / 12 cells.",
        "- All seven configurations were evaluated on the same 220 observation keys. The external set includes 9 supplemental G10-Cell2 observations; their cell identity is user-confirmed because the Guide omits the cell number, and their capacity-ratio labels remain provisional.",
        "- Part 3 was not used for training, preprocessing, early stopping, feature selection, or model selection. External metrics are descriptive only.",
        "- No formal statistical significance tests were performed. Three training seeds and three labeled validation cells support descriptive variability summaries only.",
        f"- Successful training runs: {len(metric_rows) // 3} of 21 configuration/seed runs; recorded failures: {len(failures)}.",
        "",
        "## Experimental protocol",
        "",
        "- M1: discharge; M2: discharge + charge; M3: discharge + physics; M4: discharge + charge + physics; M5: charge + physics; M6: charge; M7: physics.",
        "- Each configuration was independently initialized from random weights; the original CNN1D and saved fusion checkpoints were not used to initialize Phase 3 models.",
        "- Input features and raw-curve validation were reused from the approved Step 4/Step 10 pipeline. Physics inputs are the existing five approved features; no feature selection or external-performance tuning was introduced.",
        "- Each seed used the same batch size 16, maximum 200 epochs, AdamW (lr 0.001, weight decay 0.0001), scaled-target MSE, validation-loss checkpoint selection, and patience 20 with a 1e-8 improvement threshold.",
        "- Input and target standardization were fitted separately for each run using training observations only. No Part 3 observation entered preprocessing or training.",
        "- Checkpoints and training histories are separate by configuration and seed under `checkpoints/`.",
        "",
        "## Overall metrics across training seeds",
        "",
        "Values are mean ± sample standard deviation over successful seeds. MAE/RMSE are SOH percentage points. R² is computed over all observations within each partition for each seed before summarizing.",
        "",
    ]
    for partition, heading in (
        ("train", "### Train"),
        ("validation", "### Validation"),
        ("external_test", "### External Part 3"),
    ):
        lines.extend(
            [
                heading,
                "",
                "| Model | Seeds | Observations | Cells | MAE | RMSE | R² |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        n_obs, n_cells = count_by_partition[partition]
        for config_id in CONFIGURATIONS:
            row = summary[(config_id, partition)]
            lines.append(
                f"| {config_id} | {row['successful_training_seeds']} | {n_obs} | {n_cells} | "
                f"{_format_mean_std(row, 'mae_soh_percentage_points')} | "
                f"{_format_mean_std(row, 'rmse_soh_percentage_points')} | "
                f"{_format_mean_std(row, 'r2')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Historical reference models (single-run references, not seed averages)",
            "",
            "These saved-checkpoint results are kept separate from the 3-seed ablations. The historical CNN1D and three-encoder checkpoint results use the same keyed cohort, including the 9 supplemental G10-Cell2 external observations. The latter checkpoint is one saved run, not the new M4 ablation.",
            "",
            "| Model | Partition | Observations | Cells | MAE | RMSE | R² |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in historical_rows:
        lines.append(
            f"| {row['model']} | {row['partition']} | {row['observation_count']} | "
            f"{row['physical_cell_count']} | {float(row['mae_soh_percentage_points']):.4f} | "
            f"{float(row['rmse_soh_percentage_points']):.4f} | {float(row['r2']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-encoder-out contribution",
            "",
            "For each seed and partition, contribution is calculated on paired identical keys as `MAE(LOO) - MAE(M4)` and `RMSE(LOO) - RMSE(M4)`. Positive deltas mean the encoder's inclusion in M4 improves the metric. These are descriptive differences, not causal contribution percentages.",
            "",
            "| Encoder included | LOO model | Validation ΔMAE mean ± SD | Validation ΔRMSE mean ± SD | External ΔMAE mean ± SD | External ΔRMSE mean ± SD |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    loo_models = {"discharge": "M5", "charge": "M3", "physics": "M2"}
    for encoder in MODALITIES:
        items = []
        for partition in ("validation", "external_test"):
            values_by_metric = []
            for metric in (
                "delta_mae_soh_percentage_points",
                "delta_rmse_soh_percentage_points",
            ):
                values = [
                    float(row[metric])
                    for row in contribution_rows
                    if row["encoder_name"] == encoder
                    and row["partition"] == partition
                    and row[metric] is not None
                ]
                if values:
                    mean = float(np.mean(values))
                    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
                    values_by_metric.append(
                        f"{mean:.3f} ± {std:.3f}" if std is not None else f"{mean:.3f} (SD n/a)"
                    )
                else:
                    values_by_metric.append("not available")
            items.extend(values_by_metric)
        lines.append(
            f"| {encoder} | {loo_models[encoder]} | {items[0]} | {items[1]} | {items[2]} | {items[3]} |"
        )

    lines.extend(
        [
            "",
            "## Grouped modality permutation importance",
            "",
            "The entire 128-point discharge vector, 128-point charge vector, or 5-feature physics vector was permuted together across validation observations, independently for each repetition. Change in MAE is perturbed minus baseline; positive values indicate increased validation error. This is permutation importance on M4, not retraining-based ablation.",
            "",
            "| Modality | Repetitions | Mean Δ validation MAE | SD |",
            "|---|---:|---:|---:|",
        ]
    )
    for modality in MODALITIES:
        values = [
            float(row["change_in_mae"])
            for row in permutation_rows
            if row["encoder_name"] == modality
        ]
        mean = float(np.mean(values)) if values else None
        std = float(np.std(values, ddof=1)) if len(values) > 1 else None
        mean_text = f"{mean:.3f}" if mean is not None else "not available"
        std_text = f"{std:.3f}" if std is not None else "n/a"
        lines.append(f"| {modality} | {len(values)} | {mean_text} | {std_text} |")

    lines.extend(
        [
            "",
            "## External 12-cell coverage",
            "",
            "All 12 locked external physical cells are listed in `ablation_external_coverage.csv`. The primary ablation cohort requires complete discharge, charge, physics, and provisional/reference labels; rows with missing required modalities are excluded from every configuration rather than imputed. Each configuration has the same 73 eligible external keys per successful seed.",
            "",
            "| Cell | Common cohort observations | Missing-modality exclusions | Exclusion reason |",
            "|---|---:|---:|---|",
        ]
    )
    coverage_rows = _external_coverage_rows(
        records,
        phase2._load_split()[0],
        {
            config_id: sum(
                (config_id, seed) in {
                    (row["model_configuration"], int(row["training_seed"]))
                    for row in metric_rows
                }
                for seed in SEEDS
            )
            for config_id in CONFIGURATIONS
        },
    )
    for row in coverage_rows:
        lines.append(
            f"| {row['physical_cell_id']} | {row['common_ablation_observations_per_configuration']} | "
            f"{row['missing_modality_exclusion_count']} | {row['missing_modality_exclusion_reasons'] or 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Generalization and research questions",
            "",
            f"1. **Best individual encoder:** {individual_ranked[0]} ({CONFIGURATIONS[individual_ranked[0]][0]}) has the lowest mean validation MAE among M1, M6, and M7 at {_format_mean_std(summary[(individual_ranked[0], 'validation')], 'mae_soh_percentage_points')} pp.",
        ]
    )
    add_charge_mean, add_charge_sd = delta_for("M1", "M2", "validation")
    add_physics_mean, add_physics_sd = delta_for("M1", "M3", "validation")

    def diff_text(mean: float | None, sd: float | None) -> str:
        if mean is None:
            return "not available"
        return f"{mean:.3f} pp" + (f" ± {sd:.3f}" if sd is not None else "")

    lines.append(
        f"2. **Does Charge improve Discharge-only?** Paired `MAE(M1) - MAE(M2)` is {diff_text(add_charge_mean, add_charge_sd)} on validation; positive favors adding Charge."
    )
    lines.append(
        f"3. **Does Physics improve Discharge-only?** Paired `MAE(M1) - MAE(M3)` is {diff_text(add_physics_mean, add_physics_sd)} on validation; positive favors adding Physics."
    )
    pairwise = []
    for two_model in ("M2", "M3", "M5"):
        mean, sd = delta_for(two_model, "M4", "validation")
        pairwise.append(f"`MAE({two_model}) - MAE(M4)` = {diff_text(mean, sd)}")
    lines.append(
        f"4. **Does full fusion outperform two-encoder models?** {', '.join(pairwise)}; positive favors full M4."
    )
    contrib_val = {
        row["encoder_name"]: row
        for row in contribution_summary
        if row["partition"] == "validation"
        and row["difference_metric"] == "delta_mae_soh_percentage_points"
    }
    strongest = max(
        MODALITIES,
        key=lambda name: (
            float("-inf")
            if contrib_val[name]["mean_across_paired_seeds"] is None
            else float(contrib_val[name]["mean_across_paired_seeds"])
        ),
    )
    lines.append(
        f"5. **Largest marginal improvement in M4:** {strongest}, with validation paired ΔMAE "
        f"{diff_text(contrib_val[strongest]['mean_across_paired_seeds'], contrib_val[strongest]['std_across_paired_seeds'])}."
    )
    removal = []
    for model in ("M2", "M3", "M5"):
        mean, sd = delta_for(model, "M4", "external_test")
        removal.append(f"{model}: {diff_text(mean, sd)} (`MAE({model}) - MAE(M4)`)")
    lines.append(
        f"6. **Does removing an encoder improve external descriptive MAE?** {', '.join(removal)}; positive favors the full model. These external differences do not guide selection."
    )
    lines.append(
        f"7. **Best validation MAE:** {val_ranked[0]} at {_format_mean_std(summary[(val_ranked[0], 'validation')], 'mae_soh_percentage_points')} pp."
    )
    lines.append(
        f"8. **Best external Part 3 MAE (descriptive):** {ext_ranked[0]} at {_format_mean_std(summary[(ext_ranked[0], 'external_test')], 'mae_soh_percentage_points')} pp."
    )
    lines.append(
        f"9. **Seed consistency:** validation-best configuration by seed: "
        + ", ".join(f"{seed}: {model}" for seed, model in sorted(best_by_seed.items()))
        + ". Seed-level means/SDs are descriptive; no significance testing is claimed."
    )
    val_best_score = score(val_ranked[0], "validation", "mae_soh_percentage_points")
    m4_score = score("M4", "validation", "mae_soh_percentage_points")
    m4_params = next(
        row["trainable_parameter_count"]
        for row in configuration_rows
        if row["model_configuration"] == "M4"
    )
    if val_ranked[0] == "M4":
        runner_up = val_ranked[1] if len(val_ranked) > 1 else "M4"
        full_gain, full_gain_sd = delta_for(
            runner_up, "M4", "validation"
        )
        gain_text = diff_text(full_gain, full_gain_sd)
        runner_up_params = next(
            row["trainable_parameter_count"]
            for row in configuration_rows
            if row["model_configuration"] == runner_up
        )
        complexity_answer = (
            f"M4 has the lowest numerical validation MAE ({m4_score:.3f} pp) with "
            f"{m4_params} parameters, compared with {runner_up} at "
            f"{runner_up_params} parameters; paired `MAE({runner_up}) - MAE(M4)` "
            f"is {gain_text}. The validation ranking supports M4 numerically, but "
            "three validation cells and three seeds do not establish statistical superiority."
        )
    else:
        complexity_answer = (
            f"M4 is not the numerical validation winner ({m4_score:.3f} pp vs "
            f"{val_best_score:.3f} pp for {val_ranked[0]}); the available validation results do not justify selecting the extra encoder complexity over the smaller validation winner."
        )
    lines.append(f"10. **Is full three-encoder complexity justified?** {complexity_answer}")
    lines.extend(
        [
            "",
            "## Artifacts and reproduction",
            "",
            "- `ablation_configurations.csv`, `ablation_overall_metrics.csv`, `ablation_predictions.csv`, `ablation_seed_summary.csv`, `encoder_contribution.csv`, `modality_permutation_importance.csv`, and `ablation_external_coverage.csv` are the primary outputs.",
            "- `ablation_per_cell_metrics.csv` and `ablation_per_cell_seed_summary.csv` provide per-cell diagnostics. Their cell/seed variation is descriptive, not an inferential confidence interval.",
            "- `historical_reference_metrics.csv` keeps the historical CNN1D and saved fusion checkpoint as single-run references, outside the ablation seed averages.",
            "- `ablation_figures/` contains the eight required plots. External plots are explicitly marked provisional/descriptive.",
            "- `input_checksums.csv` records the feature tables, labels, locked split, audit outputs, and source code used. `protected_artifact_integrity.csv` verifies selected historical checkpoints and PyDMA code/results remained byte-identical.",
            "",
            "Reproduction command (use a new empty output directory to preserve this run):",
            "",
            "```powershell",
            ".venv-torch-cuda\\Scripts\\python.exe experiments\\soh_forecast\\run_ablation_phase3.py --output-dir experiments\\soh_forecast\\bafuse_three_encoder_ablation_v1_reproduction",
            "```",
            "",
            "Focused tests:",
            "",
            "```powershell",
            ".venv-torch-cuda\\Scripts\\python.exe -m pytest -q tests\\test_bafuse_ablation.py",
            "```",
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        lines.append(
            "One or more configuration/seed runs failed. See `training_failures.csv`; missing metrics and predictions are not imputed."
        )
        for failure in failures:
            lines.append(
                f"- {failure['model_configuration']} seed {failure['training_seed']}: {failure['error_type']}: {failure['error_message']}"
            )
    else:
        lines.append("No configuration/seed training failures were recorded.")

    (output_dir / "final_ablation_study_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_metadata(
    output_dir: Path,
    split_version: str,
    records: list[dict[str, Any]],
    configuration_rows: list[dict[str, Any]],
    max_epochs: int,
    patience: int,
    batch_size: int,
    permutation_repetitions: int,
) -> None:
    metadata = {
        "experiment": "BaFuse Phase 3 three-encoder ablation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_version": split_version,
        "training_seeds": list(SEEDS),
        "partitions": {
            partition: {
                "observations": sum(row["partition"] == partition for row in records),
                "physical_cells": len(
                    {
                        row["physical_cell_id"]
                        for row in records
                        if row["partition"] == partition
                    }
                ),
            }
            for partition in PARTITIONS
        },
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "early_stopping_minimum_improvement": 1e-8,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "loss": "MSE on training-standardized target",
        "permutation_repetitions_per_modality_per_seed": permutation_repetitions,
        "permutation_seed_rule": "1_000_000 + training_seed * 100 + repetition",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "configuration_parameter_counts": {
            row["model_configuration"]: row["trainable_parameter_count"]
            for row in configuration_rows
        },
        "external_part3_used_for_any_fit_or_selection": False,
        "external_label_status": "provisional capacity-ratio candidates",
        "formal_significance_testing": False,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def run(
    output_dir: Path,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 16,
    permutation_repetitions: int = 10,
) -> None:
    if max_epochs < 1 or patience < 1 or batch_size < 1 or permutation_repetitions < 1:
        raise ValueError("Epochs, patience, batch size, and repetitions must be positive.")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty experiment directory: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    protected_before = {
        str(path): _sha256(path)
        for path in PROTECTED_FILES
        if path.is_file()
    }
    if len(protected_before) != len(PROTECTED_FILES):
        missing = [str(path) for path in PROTECTED_FILES if not path.is_file()]
        raise FileNotFoundError(f"Cannot verify protected historical artifacts: {missing}")

    records, split_version, split_by_cell = _load_common_cohort()
    configuration_rows = _configuration_rows(records)
    _write_csv(output_dir / "ablation_configurations.csv", configuration_rows)
    input_manifest_rows = [
        {
            "dataset_part": row["part"],
            "physical_cell_id": row["physical_cell_id"],
            "measurement_id": row["measurement_id"],
            "partition": row["partition"],
            "source_file": row["source_file"],
            "step4_segment_ref": row["step4_segment_ref"],
            "step10_segment_ref": row["step10_segment_ref"],
            "soh_percent": row["soh_percent"],
            "label_scope": row["label_scope"],
            "feature_source": row["feature_source"],
            "identity_provenance": row["identity_provenance"],
            "all_three_modalities_available": True,
        }
        for row in records
    ]
    _write_csv(output_dir / "ablation_input_manifest.csv", input_manifest_rows)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_results: dict[tuple[str, int], dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []
    for config_id, modalities in CONFIGURATIONS.items():
        for seed in SEEDS:
            checkpoint_dir = (
                output_dir / "checkpoints" / config_id / f"seed_{seed}"
            )
            try:
                result = _train_one(
                    records,
                    config_id,
                    modalities,
                    seed,
                    checkpoint_dir,
                    max_epochs,
                    patience,
                    batch_size,
                    device,
                )
                run_results[(config_id, seed)] = result
                print(
                    f"{config_id} seed={seed}: best epoch {result['best_epoch']}, "
                    f"validation MSE={result['best_validation_mse_scaled']:.6f}"
                )
            except Exception as error:
                failure = {
                    "model_configuration": config_id,
                    "training_seed": seed,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "traceback": traceback.format_exc(),
                }
                failures.append(failure)
                print(
                    f"FAILED {config_id} seed={seed}: "
                    f"{failure['error_type']}: {failure['error_message']}"
                )

    if not run_results:
        _write_csv(output_dir / "training_failures.csv", failures)
        raise RuntimeError("No Phase 3 ablation configuration completed successfully.")

    metric_rows, prediction_rows = _metric_rows(records, run_results)
    summary_rows = _seed_summary(metric_rows)
    per_cell_rows, per_cell_summary = _per_cell_metrics(prediction_rows)
    contribution_rows = _contribution_rows(prediction_rows)
    contribution_summary = _contribution_summary_rows(contribution_rows)
    permutation_rows = _permutation_rows(
        records, run_results, permutation_repetitions, device
    )
    success_counts = {
        config_id: sum((config_id, seed) in run_results for seed in SEEDS)
        for config_id in CONFIGURATIONS
    }
    coverage_rows = _external_coverage_rows(records, split_by_cell, success_counts)
    historical_rows = _historical_reference_rows()

    _write_csv(output_dir / "ablation_overall_metrics.csv", metric_rows)
    _write_csv(output_dir / "ablation_predictions.csv", prediction_rows)
    _write_csv(output_dir / "ablation_seed_summary.csv", summary_rows)
    _write_csv(output_dir / "ablation_per_cell_metrics.csv", per_cell_rows)
    _write_csv(output_dir / "ablation_per_cell_seed_summary.csv", per_cell_summary)
    _write_csv(output_dir / "encoder_contribution.csv", contribution_rows)
    _write_csv(
        output_dir / "encoder_contribution_seed_summary.csv", contribution_summary
    )
    _write_csv(
        output_dir / "modality_permutation_importance.csv", permutation_rows
    )
    _write_csv(output_dir / "ablation_external_coverage.csv", coverage_rows)
    _write_csv(output_dir / "historical_reference_metrics.csv", historical_rows)
    _write_csv(
        output_dir / "training_failures.csv", _training_failure_rows(failures)
    )

    protected_after = {
        str(path): _sha256(path)
        for path in PROTECTED_FILES
    }
    integrity_rows = _protected_integrity_rows(protected_before, protected_after)
    _write_csv(output_dir / "protected_artifact_integrity.csv", integrity_rows)
    if any(not row["unchanged"] for row in integrity_rows):
        raise RuntimeError(
            "A protected checkpoint or PyDMA artifact changed during Phase 3."
        )
    _write_csv(output_dir / "input_checksums.csv", _input_checksum_rows())
    _plot_results(
        output_dir, summary_rows, contribution_rows, permutation_rows
    )
    _write_metadata(
        output_dir,
        split_version,
        records,
        configuration_rows,
        max_epochs,
        patience,
        batch_size,
        permutation_repetitions,
    )
    _write_report(
        output_dir,
        records,
        split_version,
        metric_rows,
        summary_rows,
        contribution_rows,
        contribution_summary,
        permutation_rows,
        prediction_rows,
        configuration_rows,
        historical_rows,
        failures,
    )
    print(
        f"Phase 3 artifacts written to {output_dir}; "
        f"{len(run_results)}/{len(CONFIGURATIONS) * len(SEEDS)} runs succeeded."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the seven BaFuse Phase 3 ablation models."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--permutation-repetitions", type=int, default=10)
    args = parser.parse_args()
    run(
        args.output_dir,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        permutation_repetitions=args.permutation_repetitions,
    )


if __name__ == "__main__":
    main()
