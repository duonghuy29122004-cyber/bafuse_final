from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR))

from bafuse.hybrid.p5_normalization import PopulationZScore  # noqa: E402
from extended_models import (  # noqa: E402
    ARCHITECTURES,
    build_model,
    trainable_parameter_count,
)


CONFIG_PATH = EXPERIMENT_DIR / "benchmark_config.json"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_EXTENDED_ARCHITECTURE_BENCHMARK.md"
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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


def _verify_checkpoint_freeze(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    freeze_path = ARTIFACT_DIR / "checkpoint_freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("All 18 new checkpoints must be frozen before External.")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "complete"
        or freeze.get("frozen") is not True
        or freeze.get("external_evaluation_performed") is not False
        or freeze.get("external_used_for_training_or_selection") is not False
        or freeze.get("checkpoint_count") != 18
        or freeze.get("new_checkpoint_count") != 18
        or freeze.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or freeze.get("normalization_parameters_sha256")
        != config["normalization_parameters_sha256"]
        or freeze.get("experiment_config_sha256") != _sha256(CONFIG_PATH)
        or freeze.get("official_baseline") != OFFICIAL_BASELINE_NAME
    ):
        raise ValueError("18-checkpoint freeze manifest is incomplete or inconsistent.")

    training_manifest = _read_json(
        ARTIFACT_DIR / "training_stage_sha256_manifest.json"
    )
    for relative, expected in training_manifest["files"].items():
        path = ARTIFACT_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Training-stage artifact changed after freeze: {relative}.")

    normalizer_path = PROJECT_ROOT / config["normalization_parameters"]
    if _sha256(normalizer_path) != config["normalization_parameters_sha256"]:
        raise ValueError("Frozen Variant D normalizer changed.")
    normalization_bundle = _read_json(normalizer_path)
    if (
        normalization_bundle.get("dataset_identifier") != config["dataset_identifier"]
        or normalization_bundle.get("dataset_manifest_sha256")
        != config["dataset_manifest_sha256"]
        or normalization_bundle.get("fit_partition") != "train"
        or normalization_bundle.get("fit_row_count") != 102
    ):
        raise ValueError("Frozen normalization provenance changed.")
    variant_d = normalization_bundle.get("variants", {}).get("D")
    if not isinstance(variant_d, dict):
        raise ValueError("Variant D scalers are missing.")
    scalers = tuple(
        PopulationZScore.from_dict(variant_d[name])
        for name in ("discharge", "charge", "target")
    )
    if tuple(scaler.mean.size for scaler in scalers) != (128, 128, 1):
        raise ValueError("Variant D scaler dimensions changed.")
    if any(scaler.fit_row_count != 102 for scaler in scalers):
        raise ValueError("Variant D scalers are not Train-fitted on 102 observations.")

    baseline_freeze_path = BASELINE_DIR / "checkpoint_freeze_manifest.json"
    if (
        _sha256(baseline_freeze_path)
        != freeze["official_baseline_freeze_manifest_sha256"]
    ):
        raise ValueError("Official CNN1D-3M baseline freeze manifest changed.")
    baseline_freeze = _read_json(baseline_freeze_path)
    baseline_entries = [
        entry
        for entry in baseline_freeze["checkpoints"]
        if entry.get("architecture") == "CNN1D"
    ]
    if {int(entry["seed"]) for entry in baseline_entries} != set(SEEDS):
        raise ValueError("Official CNN1D-3M baseline no longer covers all seeds.")
    for entry in baseline_entries:
        path = BASELINE_DIR / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError("Official CNN1D-3M checkpoint changed.")

    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in freeze["checkpoints"]:
        model_name = entry["model"]
        seed = int(entry["seed"])
        key = (model_name, seed)
        if key in entries:
            raise ValueError(f"Duplicate checkpoint entry: {key}.")
        if model_name not in MODEL_NAMES or seed not in SEEDS:
            raise ValueError(f"Unexpected frozen model/seed: {key}.")
        checkpoint_path = PROJECT_ROOT / entry["path"]
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError(f"Frozen checkpoint hash mismatch: {entry['path']}.")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        architecture, modality = model_name.rsplit("-", 1)
        expected_physics = modality == "3M"
        if (
            checkpoint.get("model_name") != model_name
            or checkpoint.get("architecture") != architecture
            or checkpoint.get("modality") != modality
            or checkpoint.get("seed") != seed
            or checkpoint.get("parameter_count") != int(entry["parameter_count"])
            or checkpoint.get("dataset_identifier") != config["dataset_identifier"]
            or checkpoint.get("dataset_manifest_sha256")
            != config["dataset_manifest_sha256"]
            or checkpoint.get("normalization_parameters_sha256")
            != config["normalization_parameters_sha256"]
            or checkpoint.get("experiment_config_sha256")
            != freeze["experiment_config_sha256"]
            or checkpoint.get("external_evaluation_performed") is not False
            or checkpoint.get("physics_input_used") is not expected_physics
            or checkpoint.get("target_transform") != variant_d["target"]
        ):
            raise ValueError(f"Checkpoint provenance mismatch: {key}.")
        model = build_model(architecture, modality)
        if trainable_parameter_count(model) != int(entry["parameter_count"]):
            raise ValueError(f"Parameter count changed after freeze: {key}.")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        entries[key] = entry

    expected_keys = {(model_name, seed) for model_name in MODEL_NAMES for seed in SEEDS}
    if set(entries) != expected_keys:
        raise ValueError("Freeze manifest does not cover all 18 model/seed checkpoints.")
    return freeze, normalization_bundle, entries


def _load_external_data(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, Any]]:
    manifest_path = DATASET_DIR / "bafuse_p5_sha256_manifest.json"
    if _sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen P5 dataset manifest checksum mismatch.")
    manifest = _read_json(manifest_path)
    for filename in (
        "bafuse_external_p5.npz",
        "bafuse_p5_split_manifest.csv",
        "bafuse_soh_labels_frozen.csv",
    ):
        entry = manifest.get("outputs", {}).get(filename)
        path = DATASET_DIR / filename
        if (
            not isinstance(entry, dict)
            or not path.is_file()
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"Frozen External data checksum mismatch: {filename}.")
    with np.load(DATASET_DIR / "bafuse_external_p5.npz", allow_pickle=False) as archive:
        expected_names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected_names:
            raise ValueError("Unexpected frozen External NPZ members.")
        arrays = {name: archive[name].copy() for name in expected_names}
    expected_shapes = {
        "X_discharge": (64, 128),
        "X_charge": (64, 128),
        "X_physics": (64, 5),
        "y_soh": (64, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected_shapes[name]:
            raise ValueError(f"Unexpected External shape/dtype for {name}.")
        if not np.isfinite(array).all():
            raise ValueError(f"Non-finite External values for {name}.")

    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    counts = Counter(row["partition"] for row in split_rows)
    if dict(counts) != EXPECTED_COUNTS or len(split_rows) != sum(EXPECTED_COUNTS.values()):
        raise ValueError(f"Frozen split counts changed: {dict(counts)}.")
    all_keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in split_rows
    ]
    if len(set(all_keys)) != len(all_keys):
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
    external_rows = sorted(
        (row for row in split_rows if row["partition"] == "external_test"),
        key=lambda row: int(row["split_row_index"]),
    )
    external_keys = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in external_rows
    }
    if len(external_keys) != 64:
        raise ValueError("Expected 64 unique External sample keys.")
    labels: dict[tuple[str, str, str], float] = {}
    for row in _read_csv(DATASET_DIR / "bafuse_soh_labels_frozen.csv"):
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in external_keys:
            if row["final_status"] != "ACCEPTED":
                raise ValueError(f"External frozen label is not accepted: {key}.")
            if key in labels:
                raise ValueError(f"Duplicate External frozen label: {key}.")
            labels[key] = float(row["soh_percent"])
    if set(labels) != external_keys:
        raise ValueError("External labels do not align with frozen sample keys.")
    expected_y = np.asarray(
        [
            labels[(row["dataset_part"], row["physical_cell_id"], row["rpt_index"])]
            for row in external_rows
        ],
        dtype=np.float32,
    ).reshape(-1, 1)
    if not np.array_equal(expected_y, arrays["y_soh"]):
        raise ValueError("External targets differ from the frozen accepted labels.")
    return arrays, external_rows, {
        "external_npz_sha256": _sha256(DATASET_DIR / "bafuse_external_p5.npz"),
        "external_observation_count": len(external_rows),
        "external_physical_cell_count": len(
            {row["physical_cell_id"] for row in external_rows}
        ),
        "external_targets_aligned": True,
    }


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
        raise ValueError(f"External {modality} preprocessing produced NaN/Inf.")
    return result


def _predict(
    model: torch.nn.Module,
    modality: str,
    inputs: dict[str, np.ndarray],
    target_scaler: PopulationZScore,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    discharge = torch.from_numpy(inputs["X_discharge"]).to(device)
    charge = torch.from_numpy(inputs["X_charge"]).to(device)
    with torch.inference_mode():
        if modality == "3M":
            normalized = model(
                discharge,
                charge,
                torch.from_numpy(inputs["X_physics"]).to(device),
            ).detach().cpu().numpy()
        else:
            normalized = model(discharge, charge).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("External prediction contains NaN/Inf.")
    return prediction


def _summarize_external(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in MODEL_NAMES:
        selected = [row for row in rows if row["model"] == model_name]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected three External runs for {model_name}.")
        summary: dict[str, Any] = {"model": model_name, "seed_count": len(selected)}
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def _official_baseline_external(config: dict[str, Any]) -> dict[str, Any]:
    path = BASELINE_DIR / "external_summary_metrics.csv"
    final_manifest_path = BASELINE_DIR / "final_sha256_manifest.json"
    final_manifest = _read_json(final_manifest_path)
    relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    if final_manifest.get("files", {}).get(relative) != _sha256(path):
        raise ValueError("Official CNN1D-3M External metrics failed checksum verification.")
    row = next(
        item for item in _read_csv(path) if item["architecture"] == "CNN1D"
    )
    return {
        "MAE_mean": float(row["MAE_mean"]),
        "MAE_sd": float(row["MAE_sd"]),
        "RMSE_mean": float(row["RMSE_mean"]),
        "RMSE_sd": float(row["RMSE_sd"]),
        "R2_mean": float(row["R2_mean"]),
        "R2_sd": float(row["R2_sd"]),
        "signed_mean": float(row["mean_signed_error_pp_mean"]),
        "signed_sd": float(row["mean_signed_error_pp_sd"]),
        "source": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    }


def _historical_comparisons() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    sources = (
        {
            "model": "CNN1D-2M",
            "experiment": "bafuse_p5_cnn_two_modality_ablation_v1",
            "filename": "cnn1d_ablation_comparison.csv",
            "role": "historical comparison — not baseline",
        },
        {
            "model": "TCN-3M",
            "experiment": "bafuse_p5_tcn_two_modality_ablation_v1",
            "filename": "tcn_ablation_comparison.csv",
            "role": "historical comparison — not baseline",
        },
        {
            "model": "TCN-2M",
            "experiment": "bafuse_p5_tcn_two_modality_ablation_v1",
            "filename": "tcn_ablation_comparison.csv",
            "role": "historical comparison — not baseline",
        },
    )
    rows: list[dict[str, Any]] = []
    references: list[dict[str, str]] = []
    for source in sources:
        directory = (
            PROJECT_ROOT
            / "experiments"
            / "soh_forecast"
            / source["experiment"]
            / "artifacts"
        )
        path = directory / source["filename"]
        manifest_path = directory / "final_sha256_manifest.json"
        final_manifest = _read_json(manifest_path)
        relative = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if final_manifest.get("files", {}).get(relative) != _sha256(path):
            raise ValueError(f"Historical metrics failed checksum verification: {path}.")
        row = next(item for item in _read_csv(path) if item["model"] == source["model"])
        rows.append(
            {
                "model": source["model"],
                "role": source["role"],
                "parameter_count": int(row["parameter_count"]),
                "validation_MAE_mean": float(row["validation_MAE_mean"]),
                "validation_MAE_sd": float(row["validation_MAE_sd"]),
                "validation_RMSE_mean": float(row["validation_RMSE_mean"]),
                "validation_RMSE_sd": float(row["validation_RMSE_sd"]),
                "validation_R2_mean": float(row["validation_R2_mean"]),
                "validation_R2_sd": float(row["validation_R2_sd"]),
                "external_MAE_mean": float(row["external_MAE_mean"]),
                "external_MAE_sd": float(row["external_MAE_sd"]),
                "external_RMSE_mean": float(row["external_RMSE_mean"]),
                "external_RMSE_sd": float(row["external_RMSE_sd"]),
                "external_R2_mean": float(row["external_R2_mean"]),
                "external_R2_sd": float(row["external_R2_sd"]),
                "external_signed_error_mean": float(
                    row["external_mean_signed_error_pp_mean"]
                ),
                "external_signed_error_sd": float(
                    row["external_mean_signed_error_pp_sd"]
                ),
                "source": relative,
            }
        )
        references.append(
            {
                "model": source["model"],
                "role": source["role"],
                "metrics_path": relative,
                "metrics_sha256": _sha256(path),
                "final_manifest_path": str(manifest_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )
    return rows, references


def _prediction_rows(
    model_name: str,
    seed: int,
    sample_rows: list[dict[str, str]],
    target: np.ndarray,
    prediction: np.ndarray,
) -> list[dict[str, Any]]:
    if len(sample_rows) != len(target) or len(target) != len(prediction):
        raise ValueError(f"External alignment failed for {model_name}/{seed}.")
    result: list[dict[str, Any]] = []
    for row, actual, predicted in zip(sample_rows, target, prediction):
        error = float(predicted - actual)
        result.append(
            {
                "model": model_name,
                "seed": seed,
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "partition": "external_test",
                "target_soh_percent": float(actual),
                "predicted_soh_percent": float(predicted),
                "residual_prediction_minus_actual_pp": error,
                "absolute_error_pp": abs(error),
            }
        )
    return result


def _per_cell_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model"], row["physical_cell_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (model, cell), selected in sorted(groups.items()):
        residuals = np.asarray(
            [float(row["residual_prediction_minus_actual_pp"]) for row in selected],
            dtype=np.float64,
        )
        unique_keys = {
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            for row in selected
        }
        result.append(
            {
                "model": model,
                "physical_cell_id": cell,
                "unique_observation_count": len(unique_keys),
                "prediction_count_across_seeds": len(selected),
                "MAE_pp": float(np.mean(np.abs(residuals))),
                "RMSE_pp": float(np.sqrt(np.mean(residuals**2))),
                "mean_signed_error_pp": float(np.mean(residuals)),
                "max_abs_error_pp": float(np.max(np.abs(residuals))),
            }
        )
    return result


def _comparison_rows(
    run_summaries: list[dict[str, str]],
    external_summaries: list[dict[str, Any]],
    baseline_reference: dict[str, Any],
    baseline_external: dict[str, Any],
    historical: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "model": OFFICIAL_BASELINE_NAME,
            "role": "ONLY OFFICIAL BASELINE",
            "parameter_count": int(baseline_reference["parameter_count"]),
            **{
                f"{partition}_{metric}_{stat}": baseline_reference[
                    f"{partition}_{metric}_{stat}"
                ]
                for partition in ("validation",)
                for metric in ("MAE", "RMSE", "R2")
                for stat in ("mean", "sd")
            },
            **{
                f"external_{metric}_{stat}": baseline_external[
                    f"{metric}_{stat}"
                ]
                for metric in ("MAE", "RMSE", "R2")
                for stat in ("mean", "sd")
            },
            "external_signed_error_mean": baseline_external["signed_mean"],
            "external_signed_error_sd": baseline_external["signed_sd"],
        }
    ]
    external_by_model = {row["model"]: row for row in external_summaries}
    for row in run_summaries:
        model_name = row["model"]
        external = external_by_model[model_name]
        rows.append(
            {
                "model": model_name,
                "role": "new candidate",
                "parameter_count": int(row["parameter_count"]),
                **{
                    f"validation_{metric}_{stat}": float(
                        row[f"validation_{metric}_{stat}"]
                    )
                    for metric in ("MAE", "RMSE", "R2")
                    for stat in ("mean", "sd")
                },
                **{
                    f"external_{metric}_{stat}": float(
                        external[f"{metric}_{stat}"]
                    )
                    for metric in ("MAE", "RMSE", "R2")
                    for stat in ("mean", "sd")
                },
                "external_signed_error_mean": float(
                    external["mean_signed_error_pp_mean"]
                ),
                "external_signed_error_sd": float(
                    external["mean_signed_error_pp_sd"]
                ),
            }
        )
    rows.extend(historical)
    return rows


def _ablation_rows(comparison: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {row["model"]: row for row in comparison}
    return [
        {
            "architecture": architecture,
            "validation_RMSE_2M": by_name[f"{architecture}-2M"][
                "validation_RMSE_mean"
            ],
            "validation_RMSE_3M": by_name[f"{architecture}-3M"][
                "validation_RMSE_mean"
            ],
            "delta_validation_RMSE_2M_minus_3M": (
                by_name[f"{architecture}-2M"]["validation_RMSE_mean"]
                - by_name[f"{architecture}-3M"]["validation_RMSE_mean"]
            ),
            "external_RMSE_2M": by_name[f"{architecture}-2M"][
                "external_RMSE_mean"
            ],
            "external_RMSE_3M": by_name[f"{architecture}-3M"][
                "external_RMSE_mean"
            ],
            "delta_external_RMSE_2M_minus_3M": (
                by_name[f"{architecture}-2M"]["external_RMSE_mean"]
                - by_name[f"{architecture}-3M"]["external_RMSE_mean"]
            ),
        }
        for architecture in ARCHITECTURES
    ]


def _plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [
        row
        for row in rows
        if row["role"] in {"ONLY OFFICIAL BASELINE", "new candidate"}
    ]
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, (metric, title) in zip(
        axes, (("MAE", "MAE"), ("RMSE", "RMSE"), ("R2", "R²"))
    ):
        means = [row[f"external_{metric}_mean"] for row in selected]
        deviations = [row[f"external_{metric}_sd"] for row in selected]
        labels = [row["model"] for row in selected]
        positions = np.arange(len(selected))
        axis.bar(positions, means, yerr=deviations, capsize=3, color="#4472C4")
        axis.set_title(f"External {title}")
        axis.set_xticks(positions, labels, rotation=45, ha="right")
        if metric != "R2":
            axis.set_ylabel("SoH percentage points")
        axis.grid(axis="y", alpha=0.25)
    figure.suptitle("New architectures vs. official CNN1D-3M baseline")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_report(
    config: dict[str, Any],
    normalization_bundle: dict[str, Any],
    freeze: dict[str, Any],
    comparison: list[dict[str, Any]],
    ablations: list[dict[str, Any]],
    external_info: dict[str, Any],
    historical_refs: list[dict[str, str]],
) -> None:
    eligible = [
        row for row in comparison
        if row["role"] in {"ONLY OFFICIAL BASELINE", "new candidate"}
    ]
    best_overall = min(eligible, key=lambda row: row["validation_RMSE_mean"])
    new_candidates = [row for row in eligible if row["role"] == "new candidate"]
    best_new = min(new_candidates, key=lambda row: row["validation_RMSE_mean"])
    variant_d = normalization_bundle["variants"]["D"]
    target = variant_d["target"]
    parameter_counts = {
        row["model"]: int(row["parameter_count"]) for row in comparison
    }
    baseline_path = "../experiments/soh_forecast/bafuse_p5_architecture_comparison_v1"
    tcn_path = "../experiments/soh_forecast/bafuse_p5_tcn_two_modality_ablation_v1"
    cnn_path = "../experiments/soh_forecast/bafuse_p5_cnn_two_modality_ablation_v1"
    lines = [
        "# BaFuse P5 Extended Architecture Benchmark",
        "",
        "## Protocol and baseline",
        "",
        f"- Frozen dataset `{config['dataset_identifier']}`; manifest SHA-256 "
        f"`{config['dataset_manifest_sha256']}`; split 102 Train / 37 Validation / "
        "64 Part 3 External observations.",
        "- **CNN1D-3M (Discharge + Charge + Physics) is the ONLY official baseline.** "
        "Its existing checkpoints and reported metrics were referenced, not copied, "
        "retrained, or modified.",
        "- Existing CNN1D-2M and TCN-2M/3M are historical comparison models only; "
        "they are not additional official baselines and their artifacts remain unchanged.",
        "- Six new variants (three families × 2M/3M) were trained for seeds "
        "42, 123, and 2026 using AdamW (lr 0.001, weight decay 0.0001), batch 16, "
        "normalized-target MSE, at most 300 epochs, patience 35, min_delta 0. "
        "Each checkpoint was selected by lowest Validation RMSE.",
        "- All 18 new checkpoints were frozen and hash-verified before the frozen "
        "External NPZ and target labels were loaded. External did not influence "
        "training, tuning, or checkpoint selection.",
        f"- Variant D normalization SHA-256: "
        f"`{config['normalization_parameters_sha256']}`. Separate Train-fitted "
        "128-position Discharge and Charge Z-scores and the Train-fitted target "
        "transform were reused unchanged; Physics inputs were not rescaled.",
        f"- Target transform: `y_norm=(SOH_percent-{float(target['mean'][0]):.9g})/"
        f"{float(target['effective_scale'][0]):.9g}`; all reported metrics are "
        "after inverse transformation and are in SOH percentage points except R².",
        f"- External cohort: {external_info['external_physical_cell_count']} "
        "Part 3 cells, an instrument/protocol domain-shift cohort.",
        "",
        "## Architectures",
        "",
        f"- **ResNet1D:** Conv1D stem (1→16, kernel 7), then three residual blocks "
        f"(16→16, 16→32 with stride 2, 32→32); global average pooling to 32; "
        f"ReLU, no dropout; parameters 3M/2M: "
        f"{parameter_counts['ResNet1D-3M']:,}/{parameter_counts['ResNet1D-2M']:,}.",
        "- **InceptionTime:** two residual blocks, each with three parallel "
        "Inception modules. Each module has kernel branches 3/7/15 (8 channels "
        "each), a max-pool projection branch, and 32 output channels; global "
        f"average pooling to 32; GELU, no dropout; parameters 3M/2M: "
        f"{parameter_counts['InceptionTime-3M']:,}/"
        f"{parameter_counts['InceptionTime-2M']:,}.",
        f"- **TCN-Attention:** causal residual TCN blocks "
        "(kernel 3; dilations 1 and 2; 32 output channels), followed by learned "
        "position embeddings and one 4-head Transformer encoder layer over the "
        "observed 128-point curve. Attention is noncausal only within that "
        f"same-RPT curve; it does not use later RPTs; GELU, Transformer dropout "
        f"0.1; parameters 3M/2M: "
        f"{parameter_counts['TCN-Attention-3M']:,}/"
        f"{parameter_counts['TCN-Attention-2M']:,}.",
        "- All voltage encoders are independent for Discharge and Charge and "
        "produce 32-dimensional embeddings. For 3M, the Physics MLP is "
        "`5→32→32` and fusion is `96→64→32→1`; for 2M there is no Physics input "
        "or encoder and fusion is `64→64→32→1`.",
        "- Input/output shapes: each voltage curve is `[B,128]`; 3M Physics is "
        "`[B,5]`; predictions are `[B,1]` in normalized SoH space.",
        "",
        "## Validation and External comparison",
        "",
        "Values are mean ± sample standard deviation over three seeds. "
        "Historical rows are included for context only, not for baseline or "
        "new-model selection.",
        "",
        "| Role | Model | Validation MAE | Validation RMSE | Validation R² | "
        "External MAE | External RMSE | External R² |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['role']} | {row['model']} | "
            f"{row['validation_MAE_mean']:.3f} ± {row['validation_MAE_sd']:.3f} | "
            f"{row['validation_RMSE_mean']:.3f} ± {row['validation_RMSE_sd']:.3f} | "
            f"{row['validation_R2_mean']:.3f} ± {row['validation_R2_sd']:.3f} | "
            f"{row['external_MAE_mean']:.3f} ± {row['external_MAE_sd']:.3f} | "
            f"{row['external_RMSE_mean']:.3f} ± {row['external_RMSE_sd']:.3f} | "
            f"{row['external_R2_mean']:.3f} ± {row['external_R2_sd']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"**Validation-RMSE selection:** {best_overall['model']} has the lowest "
            f"mean Validation RMSE ({best_overall['validation_RMSE_mean']:.4f} pp) "
            "among the official baseline and six new candidates. Among newly trained "
            f"models, the best is {best_new['model']} "
            f"({best_new['validation_RMSE_mean']:.4f} pp). Historical TCN and 2M "
            "rows were excluded from this selection.",
            "",
            "External metrics are reported after all checkpoints were frozen and "
            "are descriptive only; they did not determine the selected model.",
            "",
            "## New 2M vs. 3M ablations",
            "",
            "Negative deltas favor the 2M variant. Differences are descriptive for "
            "this frozen split and do not establish that Physics is harmful or "
            "irrelevant outside this cohort.",
            "",
            "| Family | Validation RMSE 2M | Validation RMSE 3M | Δ Val (2M−3M) | "
            "External RMSE 2M | External RMSE 3M | Δ Ext (2M−3M) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in ablations:
        lines.append(
            f"| {row['architecture']} | {row['validation_RMSE_2M']:.3f} | "
            f"{row['validation_RMSE_3M']:.3f} | "
            f"{row['delta_validation_RMSE_2M_minus_3M']:+.3f} | "
            f"{row['external_RMSE_2M']:.3f} | {row['external_RMSE_3M']:.3f} | "
            f"{row['delta_external_RMSE_2M_minus_3M']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility and historical references",
            "",
            f"- Freeze manifest: `checkpoint_count={freeze['checkpoint_count']}`; "
            "all 18 new checkpoints have SHA-256 values recorded there.",
            f"- External NPZ SHA-256: `{external_info['external_npz_sha256']}`; "
            "predictions cover the same 64 frozen sample keys for all six new variants "
            "and all three seeds.",
            f"- [Official CNN1D-3M benchmark metrics]({baseline_path}/artifacts/run_metrics.csv) "
            "and [original report](../reports/P5_NORMALIZED_ARCHITECTURE_BENCHMARK.md).",
            f"- Historical CNN1D-2M report: "
            "[CNN two-modality ablation](../reports/P5_CNN1D_TWO_MODALITY_ABLATION.md).",
            f"- Historical TCN reports: "
            "[TCN architecture benchmark](../reports/P5_NORMALIZED_ARCHITECTURE_BENCHMARK.md) "
            "and [TCN two-modality ablation](../reports/P5_TCN_TWO_MODALITY_ABLATION.md).",
            "- Historical metric source files: "
            + ", ".join(f"`{ref['metrics_path']}`" for ref in historical_refs),
            f"- [Configuration](../experiments/soh_forecast/"
            "bafuse_p5_extended_architecture_benchmark_v1/benchmark_config.json), "
            "[model definitions](../experiments/soh_forecast/"
            "bafuse_p5_extended_architecture_benchmark_v1/extended_models.py), "
            "[training metrics](../experiments/soh_forecast/"
            "bafuse_p5_extended_architecture_benchmark_v1/artifacts/run_metrics.csv), "
            "[External predictions](../experiments/soh_forecast/"
            "bafuse_p5_extended_architecture_benchmark_v1/artifacts/external_predictions.csv).",
            "",
            "The dataset's reconstructed capacity-retention target is not an "
            "official Oxford SOH ground-truth definition. The 64-observation "
            "Part 3 cohort is small and protocol-shifted; per-seed standard "
            "deviation is not uncertainty over new data splits.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if not ARTIFACT_DIR.is_dir():
        raise FileNotFoundError("Complete training and freeze all 18 checkpoints first.")
    status_path = ARTIFACT_DIR / "external_evaluation_status.json"
    if status_path.exists() or REPORT_PATH.exists():
        raise FileExistsError("Refusing to overwrite extended benchmark outputs.")
    config = _read_json(CONFIG_PATH)
    if (
        config.get("external_evaluation", {}).get("authorized") is not True
        or config.get("external_evaluation", {}).get(
            "external_used_for_training_or_selection"
        ) is not False
    ):
        raise ValueError("External evaluation authorization/prohibition is inconsistent.")
    if config.get("external_evaluation", {}).get(
        "must_wait_for_all_18_checkpoint_freeze"
    ) is not True:
        raise ValueError("External evaluation requires the 18-checkpoint freeze.")
    freeze, normalization_bundle, entries = _verify_checkpoint_freeze(config)
    if not torch.cuda.is_available():
        raise RuntimeError("The fixed evaluation protocol requires CUDA device 0.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda:0")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(
                ARTIFACT_DIR / "external_evaluation.log", encoding="utf-8"
            ),
            logging.StreamHandler(sys.stdout),
        ],
    )
    external, external_rows, external_info = _load_external_data(config)
    variant_d = normalization_bundle["variants"]["D"]
    discharge_scaler = PopulationZScore.from_dict(variant_d["discharge"])
    charge_scaler = PopulationZScore.from_dict(variant_d["charge"])
    target_scaler = PopulationZScore.from_dict(variant_d["target"])
    external_predictions: list[dict[str, Any]] = []
    external_metrics: list[dict[str, Any]] = []

    for model_name in MODEL_NAMES:
        architecture, modality = model_name.rsplit("-", 1)
        inputs = _prepare_inputs(
            external, modality, discharge_scaler, charge_scaler
        )
        for seed in SEEDS:
            entry = entries[(model_name, seed)]
            checkpoint_path = PROJECT_ROOT / entry["path"]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model = build_model(architecture, modality).to(device)
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            if trainable_parameter_count(model) != int(entry["parameter_count"]):
                raise ValueError(f"Parameter count changed for {model_name}/{seed}.")
            prediction = _predict(model, modality, inputs, target_scaler, device)
            target = external["y_soh"].reshape(-1)
            metrics = _metrics(target, prediction)
            external_metrics.append(
                {
                    "model": model_name,
                    "architecture": architecture,
                    "modality": modality,
                    "seed": seed,
                    **metrics,
                    "best_epoch": int(entry["best_epoch"]),
                    "parameter_count": int(entry["parameter_count"]),
                    "checkpoint_path": entry["path"],
                    "checkpoint_sha256": entry["sha256"],
                    "external_npz_sha256": external_info["external_npz_sha256"],
                }
            )
            external_predictions.extend(
                _prediction_rows(model_name, seed, external_rows, target, prediction)
            )
            logging.info(
                "Evaluated frozen %s seed %s: External RMSE %.4f pp.",
                model_name,
                seed,
                metrics["RMSE"],
            )

    external_summaries = _summarize_external(external_metrics)
    per_cell = _per_cell_summary(external_predictions)
    train_summaries = _read_csv(ARTIFACT_DIR / "train_validation_summary.csv")
    run_metrics = _read_csv(ARTIFACT_DIR / "run_metrics.csv")
    if len(run_metrics) != 18 or len(train_summaries) != 6:
        raise ValueError("Expected all 18 training runs and six model summaries.")
    baseline_reference = _read_json(
        ARTIFACT_DIR / "official_baseline_reference.json"
    )
    baseline_external = _official_baseline_external(config)
    historical, historical_refs = _historical_comparisons()
    comparisons = _comparison_rows(
        train_summaries,
        external_summaries,
        baseline_reference,
        baseline_external,
        historical,
    )
    ablations = _ablation_rows(comparisons)

    external_by_key = {
        (row["model"], int(row["seed"])): row for row in external_metrics
    }
    per_seed: list[dict[str, Any]] = []
    for row in run_metrics:
        joined = dict(row)
        external_row = external_by_key[(row["model"], int(row["seed"]))]
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            joined[f"external_{metric}"] = external_row[metric]
        joined["external_checkpoint_sha256"] = external_row["checkpoint_sha256"]
        joined["external_npz_sha256"] = external_row["external_npz_sha256"]
        per_seed.append(joined)

    _write_csv(ARTIFACT_DIR / "external_per_seed_metrics.csv", external_metrics)
    _write_csv(ARTIFACT_DIR / "external_summary_metrics.csv", external_summaries)
    _write_csv(ARTIFACT_DIR / "external_predictions.csv", external_predictions)
    _write_csv(ARTIFACT_DIR / "external_per_cell_metrics.csv", per_cell)
    _write_csv(ARTIFACT_DIR / "per_seed_metrics.csv", per_seed)
    _write_csv(ARTIFACT_DIR / "extended_architecture_comparison.csv", comparisons)
    _write_csv(ARTIFACT_DIR / "modality_ablation_summary.csv", ablations)
    _write_json(
        ARTIFACT_DIR / "historical_comparison_references.json",
        {
            "official_baseline": OFFICIAL_BASELINE_NAME,
            "historical_comparisons_are_not_baselines": True,
            "references": historical_refs,
        },
    )
    _plot_comparison(
        comparisons, ARTIFACT_DIR / "plots" / "extended_architecture_external.png"
    )
    _write_report(
        config,
        normalization_bundle,
        freeze,
        comparisons,
        ablations,
        external_info,
        historical_refs,
    )
    status = {
        "status": "complete",
        "external_evaluation_performed": True,
        "external_used_for_training_or_selection": False,
        "verified_new_checkpoint_count_before_external_load": freeze["checkpoint_count"],
        "external_npz_sha256": external_info["external_npz_sha256"],
        "external_observation_count": external_info["external_observation_count"],
        "external_physical_cell_count": external_info["external_physical_cell_count"],
        "external_prediction_count": len(external_predictions),
        "official_baseline": OFFICIAL_BASELINE_NAME,
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "report_sha256": _sha256(REPORT_PATH),
        "best_by_validation_RMSE": min(
            (
                row
                for row in comparisons
                if row["role"] in {"ONLY OFFICIAL BASELINE", "new candidate"}
            ),
            key=lambda row: row["validation_RMSE_mean"],
        )["model"],
    }
    _write_json(status_path, status)
    files = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file()
        and path.name not in {"final_sha256_manifest.json", "external_evaluation.log"}
    }
    files[str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")] = _sha256(
        REPORT_PATH
    )
    source_files = {
        str((EXPERIMENT_DIR / name).relative_to(PROJECT_ROOT)).replace("\\", "/"):
        _sha256(EXPERIMENT_DIR / name)
        for name in (
            "benchmark_config.json",
            "extended_models.py",
            "train_extended_models.py",
            "evaluate_extended_models.py",
        )
    }
    _write_json(
        ARTIFACT_DIR / "final_sha256_manifest.json",
        {
            "experiment_identifier": config["experiment_identifier"],
            "hash_scope": (
                "All artifacts under artifacts/ plus the final report, excluding this "
                "manifest and the append-only evaluation log."
            ),
            "files": files,
            "source_files": source_files,
        },
    )
    logging.info("Extended architecture benchmark evaluation and report complete.")
    return status


if __name__ == "__main__":
    run()
