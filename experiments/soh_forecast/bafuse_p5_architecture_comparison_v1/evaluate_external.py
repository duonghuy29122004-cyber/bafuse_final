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
NORMALIZED_ARTIFACT_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_p5_normalized_v1"
    / "artifacts"
)
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_NORMALIZED_ARCHITECTURE_BENCHMARK.md"
EXPECTED_COUNTS = {"train": 102, "validation": 37, "external_test": 64}


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
        raise ValueError(f"Expected JSON object in {path}.")
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
        raise ValueError(f"Cannot write empty CSV: {path}.")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    y_hat = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != y_hat.shape or not len(y):
        raise ValueError("Target and prediction vectors must be non-empty and aligned.")
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


def _verify_checkpoint_freeze(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    freeze_path = ARTIFACT_DIR / "checkpoint_freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Cannot evaluate External before checkpoint freeze.")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "complete"
        or freeze.get("frozen") is not True
        or freeze.get("external_evaluation_performed") is not False
        or freeze.get("external_evaluation_authorized") is not True
    ):
        raise ValueError("Checkpoint freeze manifest is incomplete or invalid.")
    if (
        freeze.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or freeze.get("normalization_parameters_sha256")
        != config["normalization_parameters_sha256"]
        or freeze.get("experiment_config_sha256") != _sha256(CONFIG_PATH)
        or freeze.get("checkpoint_count") != 15
    ):
        raise ValueError("Frozen checkpoint manifest does not match this benchmark.")

    training_manifest_path = ARTIFACT_DIR / "training_stage_sha256_manifest.json"
    training_manifest = _read_json(training_manifest_path)
    for relative, expected in training_manifest["files"].items():
        path = ARTIFACT_DIR / relative
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"Training-stage artifact changed after freeze: {relative}.")

    norm_path = PROJECT_ROOT / config["normalization_parameters"]
    if _sha256(norm_path) != config["normalization_parameters_sha256"]:
        raise ValueError("Variant D normalization parameters changed.")
    norm_bundle = _read_json(norm_path)
    variant = norm_bundle.get("variants", {}).get("D")
    if (
        not isinstance(variant, dict)
        or variant.get("discharge", {}).get("fit_row_count") != 102
        or variant.get("charge", {}).get("fit_row_count") != 102
        or variant.get("target", {}).get("fit_row_count") != 102
    ):
        raise ValueError("Frozen Variant D scalers are not Train-fitted.")
    saved_scalers = (
        PopulationZScore.from_dict(variant["discharge"]),
        PopulationZScore.from_dict(variant["charge"]),
        PopulationZScore.from_dict(variant["target"]),
    )
    if tuple(scaler.mean.size for scaler in saved_scalers) != (128, 128, 1):
        raise ValueError("Frozen Variant D scaler dimensions changed.")
    checkpoints = freeze.get("checkpoints")
    if not isinstance(checkpoints, list) or len(checkpoints) != 15:
        raise ValueError("Freeze manifest must enumerate all 15 checkpoints.")
    seen: set[tuple[str, int]] = set()
    for entry in checkpoints:
        architecture = entry["architecture"]
        seed = int(entry["seed"])
        key = (architecture, seed)
        if key in seen:
            raise ValueError(f"Duplicate frozen model/seed entry: {key}.")
        seen.add(key)
        checkpoint_path = ARTIFACT_DIR / entry["path"]
        if not checkpoint_path.is_file() or _sha256(checkpoint_path) != entry["sha256"]:
            raise ValueError(f"Frozen checkpoint hash mismatch: {entry['path']}.")
        checkpoint = torch.load(
            checkpoint_path, map_location="cpu", weights_only=True
        )
        recorded_dataset_hash = checkpoint.get(
            "dataset_manifest_sha256",
            checkpoint.get("frozen_dataset_manifest_sha256"),
        )
        target_transform = norm_bundle["variants"]["D"]["target"]
        if (
            int(checkpoint.get("seed", -1)) != seed
            or int(checkpoint.get("parameter_count", -1))
            != int(entry["parameter_count"])
            or checkpoint.get("normalization_parameters_sha256")
            != config["normalization_parameters_sha256"]
            or recorded_dataset_hash != config["dataset_manifest_sha256"]
            or checkpoint.get("target_transform") != target_transform
        ):
            raise ValueError(f"Checkpoint provenance mismatch: {architecture}/{seed}.")
        if architecture != "CNN1D" and (
            checkpoint.get("experiment_config_sha256")
            != freeze["experiment_config_sha256"]
        ):
            raise ValueError(
                f"Training config provenance mismatch: {architecture}/{seed}."
            )
        if architecture == "CNN1D":
            source = PROJECT_ROOT / entry["source_checkpoint_path"]
            if not source.is_file() or _sha256(source) != entry["source_checkpoint_sha256"]:
                raise ValueError(f"Original CNN1D D checkpoint changed for seed {seed}.")
            if _sha256(source) != entry["sha256"]:
                raise ValueError("Reused CNN1D checkpoint copy differs from its source.")
    expected = {(architecture, seed) for architecture in ARCHITECTURES for seed in (42, 123, 2026)}
    if seen != expected:
        raise ValueError("Freeze manifest does not cover every requested architecture/seed.")
    return freeze, norm_bundle, checkpoints


def _load_external_data(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, str]], dict[str, Any]]:
    manifest_path = DATASET_DIR / "bafuse_p5_sha256_manifest.json"
    if _sha256(manifest_path) != config["dataset_manifest_sha256"]:
        raise ValueError("Frozen P5 dataset manifest changed.")
    manifest = _read_json(manifest_path)
    entry = manifest.get("outputs", {}).get("bafuse_external_p5.npz")
    if not isinstance(entry, dict):
        raise ValueError("Frozen manifest has no External NPZ entry.")
    for filename in (
        "bafuse_p5_split_manifest.csv",
        "bafuse_soh_labels_frozen.csv",
    ):
        frozen_entry = manifest.get("outputs", {}).get(filename)
        path = DATASET_DIR / filename
        if (
            not isinstance(frozen_entry, dict)
            or not path.is_file()
            or _sha256(path) != frozen_entry.get("sha256")
        ):
            raise ValueError(f"Frozen dataset artifact checksum mismatch: {filename}.")
    external_path = DATASET_DIR / "bafuse_external_p5.npz"
    external_hash = _sha256(external_path)
    if external_hash != entry.get("sha256"):
        raise ValueError("Frozen External NPZ checksum mismatch.")
    with np.load(external_path, allow_pickle=False) as archive:
        expected_names = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected_names:
            raise ValueError("Unexpected External NPZ members.")
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "X_discharge": (64, 128),
        "X_charge": (64, 128),
        "X_physics": (64, 5),
        "y_soh": (64, 1),
    }
    for name, array in arrays.items():
        if array.dtype != np.float32 or array.shape != expected_shapes[name]:
            raise ValueError(f"External {name} has unexpected dtype/shape.")
        if not np.isfinite(array).all():
            raise ValueError(f"External {name} contains non-finite values.")

    split_rows = _read_csv(DATASET_DIR / "bafuse_p5_split_manifest.csv")
    partition_counts = Counter(row["partition"] for row in split_rows)
    if dict(partition_counts) != EXPECTED_COUNTS or len(split_rows) != 203:
        raise ValueError(f"Frozen split counts changed: {dict(partition_counts)}.")
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
                raise ValueError(f"Frozen cell split overlaps: {first}/{second}.")
    external_rows = sorted(
        (row for row in split_rows if row["partition"] == "external_test"),
        key=lambda row: int(row["split_row_index"]),
    )
    if len(external_rows) != 64:
        raise ValueError(f"Expected 64 External keys, found {len(external_rows)}.")
    keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in external_rows
    ]
    if len(set(keys)) != 64:
        raise ValueError("External split contains duplicate observation keys.")
    external_key_set = set(keys)
    label_values: dict[tuple[str, str, str], float] = {}
    for row in _read_csv(DATASET_DIR / "bafuse_soh_labels_frozen.csv"):
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in external_key_set:
            if row["final_status"] != "ACCEPTED":
                raise ValueError(f"External label is not accepted: {key}.")
            if key in label_values:
                raise ValueError(f"Duplicate frozen External label key: {key}.")
            label_values[key] = float(row["soh_percent"])
    if set(label_values) != external_key_set:
        raise ValueError("Frozen External labels do not match External split keys.")
    expected_y = np.asarray(
        [label_values[key] for key in keys], dtype=np.float32
    ).reshape(-1, 1)
    if not np.array_equal(expected_y, arrays["y_soh"]):
        raise ValueError("External NPZ targets are misaligned with frozen labels.")
    return arrays, external_rows, {
        "external_npz_sha256": external_hash,
        "external_observation_count": 64,
        "external_physical_cell_count": len(
            {row["physical_cell_id"] for row in external_rows}
        ),
        "external_target_alignment_verified": True,
    }


def _predict_external(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    parameters: tuple[
        PopulationZScore,
        PopulationZScore,
        PopulationZScore,
    ],
    device: torch.device,
) -> np.ndarray:
    discharge_scaler, charge_scaler, target_scaler = parameters
    inputs = apply_voltage_normalization(
        {name: arrays[name] for name in ("X_discharge", "X_charge", "X_physics")},
        discharge_scaler=discharge_scaler,
        charge_scaler=charge_scaler,
    )
    if not np.array_equal(inputs["X_physics"], arrays["X_physics"]):
        raise ValueError("External P5 Physics inputs were unexpectedly modified.")
    tensors = {
        name: torch.from_numpy(inputs[name].astype(np.float32, copy=False))
        for name in ("X_discharge", "X_charge", "X_physics")
    }
    model.eval()
    with torch.inference_mode():
        prediction_normalized = model(
            tensors["X_discharge"].to(device),
            tensors["X_charge"].to(device),
            tensors["X_physics"].to(device),
        ).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(prediction_normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("External inference produced non-finite SoH predictions.")
    return prediction


def _summarize_external(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        selected = [row for row in rows if row["architecture"] == architecture]
        if len(selected) != 3:
            raise ValueError(f"Expected three External runs for {architecture}.")
        item: dict[str, Any] = {
            "architecture": architecture,
            "seed_count": len(selected),
        }
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            values = [float(row[metric]) for row in selected]
            item[f"{metric}_mean"] = statistics.mean(values)
            item[f"{metric}_sd"] = statistics.stdev(values)
        summary.append(item)
    return summary


def _per_cell_summary(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[(row["architecture"], row["physical_cell_id"])].append(row)
    result: list[dict[str, Any]] = []
    for (architecture, cell), rows in sorted(groups.items()):
        residuals = np.asarray(
            [float(row["residual_prediction_minus_actual_pp"]) for row in rows]
        )
        unique_keys = {
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            for row in rows
        }
        result.append(
            {
                "architecture": architecture,
                "physical_cell_id": cell,
                "unique_observation_count": len(unique_keys),
                "prediction_count_across_seeds": len(rows),
                "MAE_pp": float(np.mean(np.abs(residuals))),
                "RMSE_pp": float(np.sqrt(np.mean(residuals**2))),
                "mean_signed_error_pp": float(np.mean(residuals)),
                "max_abs_error_pp": float(np.max(np.abs(residuals))),
            }
        )
    return result


def _plot_metrics(
    comparison: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = (
        ("MAE", "MAE_mean", "MAE_sd"),
        ("RMSE", "RMSE_mean", "RMSE_sd"),
        ("R²", "R2_mean", "R2_sd"),
    )
    positions = np.arange(len(ARCHITECTURES))
    for axis, (title, mean_key, sd_key) in zip(axes, metrics):
        for partition, color, label in (
            ("train", "#4472C4", "Train"),
            ("validation", "#ED7D31", "Validation"),
            ("external", "#70AD47", "External"),
        ):
            means = [
                float(next(row for row in comparison if row["architecture"] == name)[
                    f"{partition}_{mean_key}"
                ])
                for name in ARCHITECTURES
            ]
            deviations = [
                float(next(row for row in comparison if row["architecture"] == name)[
                    f"{partition}_{sd_key}"
                ])
                for name in ARCHITECTURES
            ]
            offset = {"train": -0.22, "validation": 0.0, "external": 0.22}[partition]
            axis.errorbar(
                positions + offset,
                means,
                yerr=deviations,
                fmt="o",
                capsize=3,
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xticks(positions, ARCHITECTURES, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if title != "R²":
            axis.set_ylabel("SoH percentage points")
    axes[0].legend()
    figure.suptitle("P5 Variant D architecture benchmark (mean ± sample SD across seeds)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _write_report(
    config: dict[str, Any],
    norm_bundle: dict[str, Any],
    freeze: dict[str, Any],
    run_rows: list[dict[str, Any]],
    external_runs: list[dict[str, Any]],
    external_summaries: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    per_cell: list[dict[str, Any]],
    worst_cases: list[dict[str, Any]],
    parameter_info: dict[str, Any],
    external_info: dict[str, Any],
) -> None:
    best_validation = min(
        external_summaries,
        key=lambda row: next(
            item["validation_RMSE_mean"]
            for item in comparison
            if item["architecture"] == row["architecture"]
        ),
    )
    best_external = min(external_summaries, key=lambda row: row["RMSE_mean"])
    best_individual = min(external_runs, key=lambda row: row["RMSE"])
    params = np.asarray(
        [float(row["parameter_count"]) for row in comparison], dtype=np.float64
    )
    external_rmse = np.asarray(
        [float(row["external_RMSE_mean"]) for row in comparison], dtype=np.float64
    )
    complexity_correlation = (
        float(np.corrcoef(params, external_rmse)[0, 1])
        if np.std(params) > 0 and np.std(external_rmse) > 0
        else float("nan")
    )
    variant_d = norm_bundle["variants"]["D"]
    scaler_details: list[str] = []
    for name in ("discharge", "charge"):
        scaler = variant_d[name]
        mean = np.asarray(scaler["mean"], dtype=np.float64)
        scale = np.asarray(scaler["effective_scale"], dtype=np.float64)
        scaler_details.append(
            f"- {name.title()} voltage: {len(mean)} Train-fitted positions; "
            f"mean range [{mean.min():.6g}, {mean.max():.6g}], effective-scale "
            f"range [{scale.min():.6g}, {scale.max():.6g}], "
            f"{len(scaler['small_std_indices'])} positions use the small-std "
            f"policy (minimum std {float(scaler['minimum_std']):.6g})."
        )
    target_scaler = variant_d["target"]
    target_mean = float(target_scaler["mean"][0])
    target_scale = float(target_scaler["effective_scale"][0])
    lines = [
        "# P5 Normalized Architecture Benchmark",
        "",
        "## Experiment setup",
        "",
        f"- Frozen dataset: `{config['dataset_identifier']}`; manifest SHA-256 "
        f"`{config['dataset_manifest_sha256']}`.",
        "- Task: same-RPT sequence-to-one SoH estimation from the frozen "
        "Discharge, Charge, and five P5 Physics inputs; this is not future-RPT "
        "forecasting.",
        f"- Split: 102 Train, 37 Validation, 64 Part 3 External observations; "
        f"{external_info['external_physical_cell_count']} External physical cells; "
        "cell membership remains frozen and disjoint.",
        f"- Variant D normalization SHA-256: "
        f"`{config['normalization_parameters_sha256']}`.",
        "- Discharge and Charge use their separate frozen 128-position Train "
        "scalers; the target uses the frozen Train target scaler and is "
        "inverse-transformed before all reported metrics. Frozen Physics "
        "features are passed through unchanged.",
        "- Training: AdamW (lr 0.001, weight decay 0.0001), batch 16, "
        "normalized-target MSE, max 300 epochs, patience 35, min_delta 0, "
        "seeds 42/123/2026; checkpoint selection is lowest Validation RMSE.",
        "- All 15 checkpoints were frozen and hash-verified before the External "
        "NPZ was loaded. External results were not used for training, tuning, "
        "checkpoint selection, or architecture selection.",
        "- Part 3 uses a different Maccor instrument/protocol from Parts 1–2; "
        "External results are a domain-shift test, not an ordinary random test.",
        "",
        "## Frozen Variant D preprocessing",
        "",
        "- For each voltage modality and position `j`, "
        "`x_norm[j] = (x[j] - mean_train[j]) / scale_train[j]`; means and "
        "population standard deviations (`ddof=0`) were fit on the 102 Train "
        "observations only. Each modality uses its own 128-element vectors.",
        *scaler_details,
        f"- Target: `y_norm = (SOH_percent - {target_mean:.9g}) / "
        f"{target_scale:.9g}`; predictions are inverse-transformed as "
        f"`SOH_percent = y_norm * {target_scale:.9g} + {target_mean:.9g}` "
        "before metrics. Both values are from the frozen Train target scaler.",
        "- Where a Train-fitted voltage standard deviation is below the saved "
        "minimum threshold, the frozen preprocessing uses effective scale 1.0 "
        "and retains the Train mean. Exact vectors and thresholds are in "
        "[normalization_parameters.json](../experiments/soh_forecast/"
        "bafuse_p5_normalized_v1/artifacts/normalization_parameters.json).",
        "- Physics inputs remain the five frozen P5 features, with their "
        "original frozen normalization; they are not standardized again here.",
        "",
        "## Architectures",
        "",
        "| Model | Parameters | Voltage encoders and head |",
        "|---|---:|---|",
    ]
    for architecture in ARCHITECTURES:
        count = parameter_info[architecture]["parameter_count"]
        details = architecture_details(architecture)
        if architecture in {"CNN1D", "LSTM", "TCN"}:
            description = json.dumps(
                details["p5_model"]["discharge_and_charge_encoders"]
                if "discharge_and_charge_encoders" in details["p5_model"]
                else details["p5_model"],
                separators=(",", ":"),
            )
        else:
            description = json.dumps(details["encoder"], separators=(",", ":"))
        lines.append(f"| {architecture} | {count} | `{description}` |")
    lines.extend(
        [
            "",
            "All models have independent Discharge and Charge encoders, a shared "
            "P5 Physics MLP (`5→32→32` with ReLU), concatenated 96-dimensional "
            "fusion, and `96→64→32→1` ReLU head. Output shape is `(B,1)` in "
            "normalized SoH space. CNN1D, LSTM and TCN are the existing P5 "
            "implementations unchanged. TCN uses left-padded causal residual "
            "blocks (dilations 1 and 2, receptive field 13). GRU is a separate "
            "two-layer, 32-hidden-unit unidirectional encoder. CNN-LSTM uses "
            "Conv1D(1→16,k9), ReLU, max-pool-2, Conv1D(16→32,k5), ReLU, then "
            "a one-layer, 32-hidden-unit unidirectional LSTM. Recurrent dropout "
            "is 0; existing CNN and TCN branches also have no dropout.",
            "",
            "## Train and Validation results",
            "",
            "| Architecture | Parameters | Train MAE | Train RMSE | Train R² | "
            "Validation MAE | Validation RMSE | Validation R² |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['architecture']} | {row['parameter_count']} | "
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
            "## External results",
            "",
            "| Architecture | External MAE | External RMSE | External R² | "
            "Mean signed error (prediction − target) | Validation→External RMSE gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['architecture']} | "
            f"{row['external_MAE_mean']:.4f} ± {row['external_MAE_sd']:.4f} | "
            f"{row['external_RMSE_mean']:.4f} ± {row['external_RMSE_sd']:.4f} | "
            f"{row['external_R2_mean']:.4f} ± {row['external_R2_sd']:.4f} | "
            f"{row['external_mean_signed_error_pp_mean']:+.4f} ± "
            f"{row['external_mean_signed_error_pp_sd']:.4f} pp | "
            f"{row['validation_to_external_RMSE_gap_pp']:+.4f} pp |"
        )
    lines.extend(
        [
            "",
            "Per-seed details (all metrics are in SoH percentage points except R²):",
            "",
            "Loss values below are MSE in normalized target space.",
            "",
            "| Architecture | Seed | Best epoch | Duration (s) | Train loss | "
            "Validation loss | Train MAE/RMSE/R² | Validation MAE/RMSE/R² | "
            "External MAE/RMSE/R² | External mean signed error |",
            "|---|---:|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    for external in external_runs:
        internal = next(
            row
            for row in run_rows
            if row["architecture"] == external["architecture"]
            and int(row["seed"]) == int(external["seed"])
        )
        lines.append(
            f"| {external['architecture']} | {external['seed']} | "
            f"{int(internal['best_epoch'])} | "
            f"{float(internal['training_duration_seconds']):.2f} | "
            f"{float(internal['train_mse_loss_normalized_target']):.6f} | "
            f"{float(internal['validation_mse_loss_normalized_target']):.6f} | "
            f"{float(internal['train_MAE']):.3f}/"
            f"{float(internal['train_RMSE']):.3f}/"
            f"{float(internal['train_R2']):.3f} | "
            f"{float(internal['validation_MAE']):.3f}/"
            f"{float(internal['validation_RMSE']):.3f}/"
            f"{float(internal['validation_R2']):.3f} | "
            f"{external['MAE']:.3f}/{external['RMSE']:.3f}/{external['R2']:.3f} | "
            f"{external['mean_signed_error_pp']:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "## Selection and generalization analysis",
            "",
            f"- Best architecture by mean Validation RMSE: **{best_validation['architecture']}** "
            f"({next(row['validation_RMSE_mean'] for row in comparison if row['architecture'] == best_validation['architecture']):.4f} pp). "
            "This is the Validation-only selection result.",
            f"- Lowest observed mean External RMSE: **{best_external['architecture']}** "
            f"({best_external['RMSE_mean']:.4f} pp). This is descriptive only and "
            "was not used for model selection.",
            f"- Lowest single External RMSE run: **{best_individual['architecture']}**, "
            f"seed {best_individual['seed']} ({best_individual['RMSE']:.4f} pp); "
            "also descriptive and not selected.",
            "- Validation-to-External gap is reported as mean External RMSE minus "
            "mean Validation RMSE for each architecture. Because the External "
            "cohort is Part 3 and uses a different instrument/protocol, the gap "
            "is evidence of domain shift on this cohort, not a causal estimate.",
            f"- Across only five architectures, the descriptive Pearson "
            f"correlation between parameter count and mean External RMSE is "
            f"`{complexity_correlation:.3f}`. This small architecture count "
            "cannot establish that complexity causes performance differences.",
            "- TCN External performance should be compared directly with CNN1D "
            "in the tables; these descriptive results do not prove TCN is "
            "generally more robust to protocol shift.",
            "",
            "## Per-cell diagnostics and largest errors",
            "",
            "Per-cell rows aggregate the three seeds and are diagnostic only. "
            "No universal error threshold was applied; the worst-case table "
            "reports observed extremes for investigation.",
            "",
            "| Architecture | Cell | Unique observations | MAE | RMSE | Mean signed error | Max absolute error |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(per_cell, key=lambda item: (item["architecture"], item["physical_cell_id"])):
        lines.append(
            f"| {row['architecture']} | {row['physical_cell_id']} | "
            f"{row['unique_observation_count']} | {row['MAE_pp']:.3f} | "
            f"{row['RMSE_pp']:.3f} | {row['mean_signed_error_pp']:+.3f} | "
            f"{row['max_abs_error_pp']:.3f} |"
        )
    lines.extend(
        [
            "",
            "| Architecture | Seed | Worst observation key | Absolute error (pp) | Signed error (pp) |",
            "|---:|---:|---|---:|---:|",
        ]
    )
    for row in worst_cases:
        lines.append(
            f"| {row['architecture']} | {row['seed']} | "
            f"Part {row['dataset_part']} / {row['physical_cell_id']} / "
            f"RPT {row['rpt_index']} | {row['absolute_error_pp']:.3f} | "
            f"{row['residual_prediction_minus_actual_pp']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Limitations and conclusion",
            "",
            f"- External observations: {external_info['external_observation_count']} "
            f"across {external_info['external_physical_cell_count']} Part 3 cells; "
            "the per-cell sample size is small and the domain differs from Parts 1–2.",
            "- This is one frozen split and three seeds. Mean ± sample SD describes "
            "run variability, not uncertainty over new train/validation splits.",
            "- The target is reconstructed capacity retention, not an official "
            "Oxford SOH ground-truth label. P5 labels and target provenance were "
            "used unchanged.",
            "- No external-based tuning, retraining, checkpoint selection, or "
            "architecture selection was performed. No labels, splits, dataset "
            "arrays, Variant D scalers, or source CNN1D checkpoints were modified.",
            "",
            "## Reproducibility artifacts",
            "",
            "- Training configuration: "
            "[benchmark_config.json](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/benchmark_config.json).",
            "- Model and training source: "
            "[architecture_models.py](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/architecture_models.py), "
            "[train_architectures.py](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/train_architectures.py), "
            "[evaluate_external.py](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/evaluate_external.py).",
            "- Frozen checkpoint manifest: "
            "[checkpoint_freeze_manifest.json](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/artifacts/"
            "checkpoint_freeze_manifest.json).",
            "- Per-seed metrics: "
            "[run_metrics.csv](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/artifacts/run_metrics.csv).",
            "- External metrics: "
            "[external_per_seed_metrics.csv](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/artifacts/"
            "external_per_seed_metrics.csv).",
            "- Sample-key predictions: "
            "[external_predictions.csv](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/artifacts/"
            "external_predictions.csv).",
            "- Comparison plot: "
            "[architecture_metrics.png](../experiments/soh_forecast/"
            "bafuse_p5_architecture_comparison_v1/artifacts/plots/"
            "architecture_metrics.png).",
            "- Checkpoint count frozen before External: "
            f"{freeze['checkpoint_count']}. External NPZ SHA-256: "
            f"`{external_info['external_npz_sha256']}`.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if not ARTIFACT_DIR.is_dir():
        raise FileNotFoundError("Complete the training stage before External evaluation.")
    status_path = ARTIFACT_DIR / "external_evaluation_status.json"
    if status_path.exists() or REPORT_PATH.exists():
        raise FileExistsError("External evaluation artifacts already exist.")
    config = _read_json(CONFIG_PATH)
    if config.get("external_evaluation", {}).get("authorized") is not True:
        raise ValueError("External evaluation is not authorized by the benchmark config.")
    if config.get("external_evaluation", {}).get(
        "must_wait_for_checkpoint_freeze_manifest"
    ) is not True:
        raise ValueError("Checkpoint freeze must precede External evaluation.")
    freeze, norm_bundle, checkpoint_entries = _verify_checkpoint_freeze(config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("The configured experiment requires CUDA device 0.")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
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
    variant_d = norm_bundle["variants"]["D"]
    scalers = (
        PopulationZScore.from_dict(variant_d["discharge"]),
        PopulationZScore.from_dict(variant_d["charge"]),
        PopulationZScore.from_dict(variant_d["target"]),
    )
    checkpoint_by_key = {
        (entry["architecture"], int(entry["seed"])): entry
        for entry in checkpoint_entries
    }
    external_predictions: list[dict[str, Any]] = []
    external_metrics: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        for seed in (42, 123, 2026):
            entry = checkpoint_by_key[(architecture, seed)]
            path = ARTIFACT_DIR / entry["path"]
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            recorded_dataset_hash = checkpoint.get(
                "dataset_manifest_sha256",
                checkpoint.get("frozen_dataset_manifest_sha256"),
            )
            if (
                checkpoint.get("normalization_parameters_sha256")
                != config["normalization_parameters_sha256"]
                or recorded_dataset_hash != config["dataset_manifest_sha256"]
                or checkpoint.get("target_transform")
                != norm_bundle["variants"]["D"]["target"]
                or (
                    architecture != "CNN1D"
                    and checkpoint.get("experiment_config_sha256")
                    != freeze["experiment_config_sha256"]
                )
            ):
                raise ValueError(
                    f"Checkpoint preprocessing provenance mismatch: {architecture}/{seed}."
                )
            model = build_architecture(architecture).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            if trainable_parameter_count(model) != int(entry["parameter_count"]):
                raise ValueError(
                    f"Parameter count changed after freeze: {architecture}/{seed}."
                )
            prediction = _predict_external(model, external, scalers, device)
            target = external["y_soh"].reshape(-1)
            metric = _metrics(target, prediction)
            external_metrics.append(
                {
                    "architecture": architecture,
                    "seed": seed,
                    **metric,
                    "checkpoint_path": entry["path"],
                    "checkpoint_sha256": entry["sha256"],
                    "normalization_parameters_sha256": config[
                        "normalization_parameters_sha256"
                    ],
                    "external_npz_sha256": external_info["external_npz_sha256"],
                }
            )
            for index, row in enumerate(external_rows):
                residual = float(prediction[index] - target[index])
                external_predictions.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "dataset_part": row["dataset_part"],
                        "physical_cell_id": row["physical_cell_id"],
                        "rpt_index": row["rpt_index"],
                        "partition": "external_test",
                        "target_soh_percent": float(target[index]),
                        "predicted_soh_percent": float(prediction[index]),
                        "residual_prediction_minus_actual_pp": residual,
                        "absolute_error_pp": abs(residual),
                    }
                )
            logging.info(
                "External evaluated frozen %s seed %s: RMSE %.4f pp.",
                architecture,
                seed,
                metric["RMSE"],
            )

    external_summaries = _summarize_external(external_metrics)
    per_cell = _per_cell_summary(external_predictions)
    worst_cases: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        worst_cases.extend(
            sorted(
                (
                    row
                    for row in external_predictions
                    if row["architecture"] == architecture
                ),
                key=lambda row: float(row["absolute_error_pp"]),
                reverse=True,
            )[:3]
        )
    parameter_info = _read_json(
        ARTIFACT_DIR / "architecture_parameter_counts.json"
    )
    trainval_rows = _read_csv(ARTIFACT_DIR / "run_metrics.csv")
    trainval_summaries = _read_csv(ARTIFACT_DIR / "train_validation_summary.csv")
    comparison: list[dict[str, Any]] = []
    for architecture in ARCHITECTURES:
        internal = next(
            row
            for row in trainval_summaries
            if row["architecture"] == architecture
        )
        external_summary = next(
            row
            for row in external_summaries
            if row["architecture"] == architecture
        )
        item: dict[str, Any] = {
            "architecture": architecture,
            "parameter_count": int(internal["parameter_count"]),
        }
        for metric in (
            "train_MAE",
            "train_RMSE",
            "train_R2",
            "validation_MAE",
            "validation_RMSE",
            "validation_R2",
        ):
            item[f"{metric}_mean"] = float(internal[f"{metric}_mean"])
            item[f"{metric}_sd"] = float(internal[f"{metric}_sd"])
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            item[f"external_{metric}_mean"] = external_summary[f"{metric}_mean"]
            item[f"external_{metric}_sd"] = external_summary[f"{metric}_sd"]
        item["validation_to_external_RMSE_gap_pp"] = (
            item["external_RMSE_mean"] - item["validation_RMSE_mean"]
        )
        comparison.append(item)

    _write_csv(ARTIFACT_DIR / "external_per_seed_metrics.csv", external_metrics)
    _write_csv(ARTIFACT_DIR / "external_summary_metrics.csv", external_summaries)
    _write_csv(ARTIFACT_DIR / "external_per_cell_metrics.csv", per_cell)
    _write_csv(
        ARTIFACT_DIR / "external_predictions.csv",
        sorted(
            external_predictions,
            key=lambda row: (
                ARCHITECTURES.index(row["architecture"]),
                int(row["seed"]),
                int(row["rpt_index"]),
            ),
        ),
    )
    _write_csv(ARTIFACT_DIR / "external_worst_cases.csv", worst_cases)
    _write_csv(ARTIFACT_DIR / "architecture_comparison.csv", comparison)
    plot_path = ARTIFACT_DIR / "plots" / "architecture_metrics.png"
    _plot_metrics(comparison, plot_path)
    _write_report(
        config,
        norm_bundle,
        freeze,
        trainval_rows,
        external_metrics,
        external_summaries,
        comparison,
        per_cell,
        worst_cases,
        parameter_info,
        external_info,
    )
    logging.info("External evaluation complete for all 15 frozen checkpoints.")
    final_status = {
        "status": "complete",
        "external_evaluation_performed": True,
        "external_evaluation_authorized": True,
        "checkpoint_count_verified_before_external_load": freeze["checkpoint_count"],
        "external_npz_sha256": external_info["external_npz_sha256"],
        "external_observation_count": external_info["external_observation_count"],
        "external_prediction_count": len(external_predictions),
        "checkpoint_freeze_manifest_sha256": _sha256(
            ARTIFACT_DIR / "checkpoint_freeze_manifest.json"
        ),
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)),
        "report_sha256": _sha256(REPORT_PATH),
        "selected_by_validation_architecture": min(
            comparison, key=lambda row: row["validation_RMSE_mean"]
        )["architecture"],
        "lowest_mean_external_rmse_descriptive_only": min(
            comparison, key=lambda row: row["external_RMSE_mean"]
        )["architecture"],
        "external_used_for_selection_or_tuning": False,
    }
    _write_json(status_path, final_status)
    hashes = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "final_sha256_manifest.json"
    }
    hashes[str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")] = _sha256(
        REPORT_PATH
    )
    source_files = {
        str((EXPERIMENT_DIR / filename).relative_to(PROJECT_ROOT)).replace("\\", "/"):
        _sha256(EXPERIMENT_DIR / filename)
        for filename in (
            "benchmark_config.json",
            "architecture_models.py",
            "train_architectures.py",
            "evaluate_external.py",
        )
    }
    _write_json(
        ARTIFACT_DIR / "final_sha256_manifest.json",
        {
            "experiment_identifier": config["experiment_identifier"],
            "hash_scope": "All artifacts under artifacts/ plus the final report, excluding this manifest.",
            "files": hashes,
            "source_files": source_files,
        },
    )
    return final_status


if __name__ == "__main__":
    run()
