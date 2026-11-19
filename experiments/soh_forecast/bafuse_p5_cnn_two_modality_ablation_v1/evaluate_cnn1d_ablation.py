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
    build_architecture as build_cnn1d_3m_model,
    trainable_parameter_count as count_cnn1d_3m_parameters,
)
from bafuse.hybrid.p5_normalization import PopulationZScore  # noqa: E402
from cnn1d_2m_model import (  # noqa: E402
    architecture_details as cnn1d_2m_details,
    build_cnn1d_2m_model,
    trainable_parameter_count as count_cnn1d_2m_parameters,
)


CONFIG_PATH = EXPERIMENT_DIR / "benchmark_config.json"
ARTIFACT_DIR = EXPERIMENT_DIR / "artifacts"
REPORT_PATH = PROJECT_ROOT / "reports" / "P5_CNN1D_TWO_MODALITY_ABLATION.md"
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
EXPECTED_COUNTS = {"train": 102, "validation": 37, "external_test": 64}
SEEDS = (42, 123, 2026)
MODEL_ORDER = ("CNN1D-2M", "CNN1D-3M")


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


def _verify_checkpoint_freeze(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[tuple[str, int], dict[str, Any]]]:
    freeze_path = ARTIFACT_DIR / "checkpoint_freeze_manifest.json"
    if not freeze_path.is_file():
        raise FileNotFoundError("Checkpoint freeze manifest must precede External.")
    freeze = _read_json(freeze_path)
    if (
        freeze.get("status") != "complete"
        or freeze.get("frozen") is not True
        or freeze.get("external_evaluation_performed") is not False
        or freeze.get("external_used_for_training_or_selection") is not False
        or freeze.get("checkpoint_count") != 6
        or freeze.get("cnn1d_2m_checkpoint_count") != 3
        or freeze.get("cnn1d_3m_baseline_checkpoint_count") != 3
        or freeze.get("dataset_manifest_sha256") != config["dataset_manifest_sha256"]
        or freeze.get("normalization_parameters_sha256")
        != config["normalization_parameters_sha256"]
        or freeze.get("experiment_config_sha256") != _sha256(CONFIG_PATH)
    ):
        raise ValueError("CNN1D ablation checkpoint freeze is incomplete or inconsistent.")

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
        raise ValueError("Variant D scalers are not fitted on all 102 Train rows.")

    baseline_freeze_path = BASELINE_DIR / "checkpoint_freeze_manifest.json"
    if (
        _sha256(baseline_freeze_path)
        != freeze["baseline_freeze_manifest_sha256"]
    ):
        raise ValueError("The frozen CNN1D-3M baseline manifest changed.")
    baseline_freeze = _read_json(baseline_freeze_path)
    baseline_entries = {
        int(entry["seed"]): entry
        for entry in baseline_freeze["checkpoints"]
        if entry["architecture"] == "CNN1D"
    }
    if set(baseline_entries) != set(SEEDS):
        raise ValueError("The original CNN1D-3M baseline does not cover all seeds.")

    entries: dict[tuple[str, int], dict[str, Any]] = {}
    for entry in freeze["checkpoints"]:
        model_name = entry["model"]
        seed = int(entry["seed"])
        key = (model_name, seed)
        if key in entries:
            raise ValueError(f"Duplicate checkpoint-freeze entry: {key}.")
        path = PROJECT_ROOT / entry["path"]
        if not path.is_file() or _sha256(path) != entry["sha256"]:
            raise ValueError(f"Frozen checkpoint hash mismatch: {entry['path']}.")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if model_name == "CNN1D-2M":
            if (
                seed not in SEEDS
                or checkpoint.get("model") != "CNN1D-2M"
                or checkpoint.get("seed") != seed
                or checkpoint.get("parameter_count") != int(entry["parameter_count"])
                or checkpoint.get("dataset_manifest_sha256")
                != config["dataset_manifest_sha256"]
                or checkpoint.get("normalization_parameters_sha256")
                != config["normalization_parameters_sha256"]
                or checkpoint.get("experiment_config_sha256")
                != freeze["experiment_config_sha256"]
                or checkpoint.get("external_evaluation_performed") is not False
                or checkpoint.get("physics_input_used") is not False
                or checkpoint.get("target_transform")
                != variant_d["target"]
            ):
                raise ValueError(f"CNN1D-2M checkpoint provenance mismatch: {key}.")
        elif model_name == "CNN1D-3M":
            baseline_entry = baseline_entries.get(seed)
            if (
                baseline_entry is None
                or entry["sha256"] != baseline_entry["sha256"]
                or entry["path"]
                != str(
                    (BASELINE_DIR / baseline_entry["path"]).relative_to(PROJECT_ROOT)
                ).replace("\\", "/")
                or checkpoint.get("architecture") != "BaFuse_CNN1D_P5"
                or checkpoint.get("seed") != seed
                or checkpoint.get("normalization_parameters_sha256")
                != config["normalization_parameters_sha256"]
                or checkpoint.get("frozen_dataset_identifier")
                != config["dataset_identifier"]
                or checkpoint.get(
                    "dataset_manifest_sha256",
                    checkpoint.get("frozen_dataset_manifest_sha256"),
                )
                != config["dataset_manifest_sha256"]
                or checkpoint.get("target_transform") != variant_d["target"]
            ):
                raise ValueError(f"CNN1D-3M baseline provenance mismatch: {key}.")
        else:
            raise ValueError(f"Unexpected model in freeze manifest: {model_name}.")
        entries[key] = entry

    expected_keys = {
        (model_name, seed) for model_name in MODEL_ORDER for seed in SEEDS
    }
    if set(entries) != expected_keys:
        raise ValueError("Freeze manifest does not cover both models and all seeds.")
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
    if len(external_rows) != 64:
        raise ValueError("Expected 64 External sample keys.")
    external_keys = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in external_rows
    }
    labels: dict[tuple[str, str, str], float] = {}
    for row in _read_csv(DATASET_DIR / "bafuse_soh_labels_frozen.csv"):
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in external_keys:
            if row["final_status"] != "ACCEPTED":
                raise ValueError(f"Frozen External label is not accepted: {key}.")
            if key in labels:
                raise ValueError(f"Duplicate External frozen label: {key}.")
            labels[key] = float(row["soh_percent"])
    if set(labels) != external_keys:
        raise ValueError("External target labels do not align to frozen sample keys.")
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


def _prepare_2m(
    arrays: dict[str, np.ndarray],
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    result = {
        "X_discharge": discharge_scaler.transform(arrays["X_discharge"]),
        "X_charge": charge_scaler.transform(arrays["X_charge"]),
    }
    if not all(np.isfinite(value).all() for value in result.values()):
        raise ValueError("CNN1D-2M preprocessing produced NaN/Inf.")
    return result


def _prepare_3m(
    arrays: dict[str, np.ndarray],
    discharge_scaler: PopulationZScore,
    charge_scaler: PopulationZScore,
) -> dict[str, np.ndarray]:
    result = _prepare_2m(arrays, discharge_scaler, charge_scaler)
    result["X_physics"] = np.array(arrays["X_physics"], copy=True)
    if not np.isfinite(result["X_physics"]).all():
        raise ValueError("Frozen CNN1D-3M Physics values contain NaN/Inf.")
    return result


def _predict_2m(
    model: torch.nn.Module,
    inputs: dict[str, np.ndarray],
    target_scaler: PopulationZScore,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        normalized = model(
            torch.from_numpy(inputs["X_discharge"]).to(device),
            torch.from_numpy(inputs["X_charge"]).to(device),
        ).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("CNN1D-2M External predictions contain NaN/Inf.")
    return prediction


def _predict_3m(
    model: torch.nn.Module,
    inputs: dict[str, np.ndarray],
    target_scaler: PopulationZScore,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        normalized = model(
            torch.from_numpy(inputs["X_discharge"]).to(device),
            torch.from_numpy(inputs["X_charge"]).to(device),
            torch.from_numpy(inputs["X_physics"]).to(device),
        ).detach().cpu().numpy()
    prediction = target_scaler.inverse_transform(normalized).reshape(-1)
    if not np.isfinite(prediction).all():
        raise ValueError("CNN1D-3M External predictions contain NaN/Inf.")
    return prediction


def _summarize_external(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        selected = [row for row in rows if row["model"] == model_name]
        if len(selected) != len(SEEDS):
            raise ValueError(f"Expected three External runs for {model_name}.")
        summary: dict[str, Any] = {
            "model": model_name,
            "seed_count": len(selected),
        }
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            values = [float(row[metric]) for row in selected]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sd"] = statistics.stdev(values)
        summaries.append(summary)
    return summaries


def _per_cell_summary(
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[(row["model"], row["physical_cell_id"])].append(row)
    results: list[dict[str, Any]] = []
    for (model_name, cell_id), rows in sorted(groups.items()):
        residuals = np.asarray(
            [float(row["residual_prediction_minus_actual_pp"]) for row in rows],
            dtype=np.float64,
        )
        observation_keys = {
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            for row in rows
        }
        results.append(
            {
                "model": model_name,
                "physical_cell_id": cell_id,
                "unique_observation_count": len(observation_keys),
                "prediction_count_across_seeds": len(rows),
                "MAE_pp": float(np.mean(np.abs(residuals))),
                "RMSE_pp": float(np.sqrt(np.mean(residuals**2))),
                "mean_signed_error_pp": float(np.mean(residuals)),
                "max_abs_error_pp": float(np.max(np.abs(residuals))),
            }
        )
    return results


def _plot_comparison(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    positions = np.arange(len(MODEL_ORDER))
    for axis, (title, metric) in zip(
        axes,
        (("MAE", "MAE"), ("RMSE", "RMSE"), ("R²", "R2")),
    ):
        for partition, color, offset in (
            ("train", "#4472C4", -0.22),
            ("validation", "#ED7D31", 0.0),
            ("external", "#70AD47", 0.22),
        ):
            means = [
                float(
                    next(row for row in rows if row["model"] == name)[
                        f"{partition}_{metric}_mean"
                    ]
                )
                for name in MODEL_ORDER
            ]
            deviations = [
                float(
                    next(row for row in rows if row["model"] == name)[
                        f"{partition}_{metric}_sd"
                    ]
                )
                for name in MODEL_ORDER
            ]
            axis.errorbar(
                positions + offset,
                means,
                yerr=deviations,
                fmt="o",
                capsize=4,
                color=color,
                label=partition.title(),
            )
        axis.set_title(title)
        axis.set_xticks(positions, MODEL_ORDER)
        axis.grid(axis="y", alpha=0.25)
        if title != "R²":
            axis.set_ylabel("SoH percentage points")
    axes[0].legend()
    figure.suptitle("CNN1D two-modality ablation: mean ± sample SD across seeds")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _plot_per_cell(rows: list[dict[str, Any]], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = sorted({row["physical_cell_id"] for row in rows})
    positions = np.arange(len(cells))
    figure, axis = plt.subplots(figsize=(13, 5))
    width = 0.38
    for model_name, color, offset in (
        ("CNN1D-2M", "#4472C4", -width / 2),
        ("CNN1D-3M", "#ED7D31", width / 2),
    ):
        values = {
            row["physical_cell_id"]: float(row["MAE_pp"])
            for row in rows
            if row["model"] == model_name
        }
        axis.bar(
            positions + offset,
            [values[cell] for cell in cells],
            width,
            color=color,
            label=model_name,
        )
    axis.set_xticks(positions, cells, rotation=35, ha="right")
    axis.set_ylabel("External MAE (SoH percentage points)")
    axis.set_title("Part 3 per-cell External MAE (aggregated across seeds)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)


def _write_report(
    config: dict[str, Any],
    normalization_bundle: dict[str, Any],
    freeze: dict[str, Any],
    run_rows: list[dict[str, str]],
    external_runs: list[dict[str, Any]],
    external_summaries: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    per_cell: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    external_info: dict[str, Any],
) -> None:
    two = next(row for row in comparison if row["model"] == "CNN1D-2M")
    three = next(row for row in comparison if row["model"] == "CNN1D-3M")
    delta_mean = two["external_RMSE_mean"] - three["external_RMSE_mean"]
    paired_delta_values = [float(row["delta_external_RMSE_pp"]) for row in deltas]
    paired_delta_sd = statistics.stdev(paired_delta_values)
    best_validation = min(comparison, key=lambda row: row["validation_RMSE_mean"])
    best_external = min(comparison, key=lambda row: row["external_RMSE_mean"])
    variant_d = normalization_bundle["variants"]["D"]
    discharge = variant_d["discharge"]
    charge = variant_d["charge"]
    target = variant_d["target"]
    d_scale = np.asarray(discharge["effective_scale"], dtype=np.float64)
    c_scale = np.asarray(charge["effective_scale"], dtype=np.float64)
    target_mean = float(target["mean"][0])
    target_scale = float(target["effective_scale"][0])
    lines = [
        "# P5 CNN1D Two-Modality Ablation",
        "",
        "## Experiment setup",
        "",
        f"- Frozen dataset: `{config['dataset_identifier']}`; manifest SHA-256 "
        f"`{config['dataset_manifest_sha256']}`.",
        "- Same-RPT SoH estimation on the frozen cell-disjoint split: 102 Train, "
        "37 Validation, and 64 Part 3 External observations.",
        f"- External cohort: {external_info['external_physical_cell_count']} "
        "physical cells. Part 3 uses Maccor 4000, unlike Parts 1–2 (Maccor 4200); "
        "treat this as domain-shift evaluation.",
        "- CNN1D-2M was trained from scratch with seeds 42, 123, 2026. CNN1D-3M "
        "checkpoints for those seeds were reused unchanged.",
        "- Protocol: AdamW (lr 0.001, weight decay 0.0001), batch 16, "
        "normalized-target MSE, at most 300 epochs, early-stopping patience 35, "
        "min_delta 0, and checkpoint selection by lowest Validation RMSE.",
        "- All six checkpoints (three new CNN1D-2M and three frozen CNN1D-3M) were "
        "hash-verified before the External NPZ was opened. External data were "
        "not used for training, tuning, checkpoint selection, or architecture changes.",
        "",
        "## Architecture comparison",
        "",
        "| Model | Voltage path | Physics path | Fusion head | Parameters |",
        "|---|---|---|---|---:|",
        "| CNN1D-3M | Two independent existing CNN1D encoders; each outputs 32 | "
        "Existing `5→32→32` Physics MLP | `96→64→32→1`, ReLU after hidden layers | "
        f"{int(three['parameter_count'])} |",
        "| CNN1D-2M | Same two independent existing CNN1D encoders; each outputs 32 | "
        "None; no Physics input or Physics encoder | `64→64→32→1`, ReLU after hidden layers | "
        f"{int(two['parameter_count'])} |",
        "",
        "Each voltage encoder reuses the unchanged P5 CNN1D implementation: "
        "`Conv1D(1,16,kernel=9,padding=4)`, ReLU, max-pool-2, "
        "`Conv1D(16,32,kernel=5,padding=2)`, ReLU, adaptive average pooling, "
        "and flattening to a 32-dimensional embedding. The CNN1D-2M model "
        "concatenates only the independent Discharge and Charge embeddings; "
        "its forward contract accepts `(B,128)` for each modality and returns "
        "`(B,1)` normalized SoH.",
        "",
        "## Frozen Variant D normalization",
        "",
        f"- Normalizer SHA-256: `{config['normalization_parameters_sha256']}`; "
        "all three statistics were fitted on the 102 Train observations.",
        "- Discharge and Charge each use their own 128-position Train-fitted "
        "population Z-score (`ddof=0`): `x_norm[j]=(x[j]-mean_train[j])/scale_train[j]`.",
        f"- Effective discharge scale range: [{d_scale.min():.6g}, {d_scale.max():.6g}]; "
        f"effective charge scale range: [{c_scale.min():.6g}, {c_scale.max():.6g}].",
        f"- Target transform: `y_norm=(SOH_percent-{target_mean:.9g})/"
        f"{target_scale:.9g}`; inverse transform is applied before metrics.",
        "- No scaler was refit. CNN1D-2M receives only the normalized Discharge "
        "and Charge arrays. CNN1D-3M receives the same normalized voltage arrays "
        "and its frozen P5 Physics inputs unchanged.",
        "",
        "## Train and Validation metrics",
        "",
        "Values are mean ± sample standard deviation across three seeds. MAE and "
        "RMSE are in SoH percentage points; R² is unitless.",
        "",
        "| Model | Parameters | Train MAE | Train RMSE | Train R² | "
        "Validation MAE | Validation RMSE | Validation R² |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {int(row['parameter_count'])} | "
            f"{row['train_MAE_mean']:.4f} ± {row['train_MAE_sd']:.4f} | "
            f"{row['train_RMSE_mean']:.4f} ± {row['train_RMSE_sd']:.4f} | "
            f"{row['train_R2_mean']:.4f} ± {row['train_R2_sd']:.4f} | "
            f"{row['validation_MAE_mean']:.4f} ± {row['validation_MAE_sd']:.4f} | "
            f"{row['validation_RMSE_mean']:.4f} ± "
            f"{row['validation_RMSE_sd']:.4f} | "
            f"{row['validation_R2_mean']:.4f} ± "
            f"{row['validation_R2_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## External metrics and ablation comparison",
            "",
            "| Model | External MAE | External RMSE | External R² | "
            "Mean signed error (prediction − target) | Validation→External RMSE gap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['external_MAE_mean']:.4f} ± "
            f"{row['external_MAE_sd']:.4f} | "
            f"{row['external_RMSE_mean']:.4f} ± {row['external_RMSE_sd']:.4f} | "
            f"{row['external_R2_mean']:.4f} ± {row['external_R2_sd']:.4f} | "
            f"{row['external_mean_signed_error_pp_mean']:+.4f} ± "
            f"{row['external_mean_signed_error_pp_sd']:.4f} pp | "
            f"{row['validation_to_external_RMSE_gap_pp']:+.4f} pp |"
        )
    lines.extend(
        [
            "",
            f"**Delta External RMSE (CNN1D-2M − CNN1D-3M): "
            f"{delta_mean:+.4f} pp** (paired-seed difference sample SD "
            f"{paired_delta_sd:.4f} pp). A negative value favors CNN1D-2M on this "
            "External cohort; a positive value favors CNN1D-3M. This difference "
            "is descriptive for the frozen split, not causal or universal.",
            "",
            "| Seed | CNN1D-2M External RMSE | CNN1D-3M External RMSE | "
            "Delta (2M − 3M), pp |",
            "|---:|---:|---:|---:|",
        ]
    )
    for row in deltas:
        lines.append(
            f"| {row['seed']} | {row['CNN1D-2M_external_RMSE']:.4f} | "
            f"{row['CNN1D-3M_external_RMSE']:.4f} | "
            f"{row['delta_external_RMSE_pp']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed results",
            "",
            "| Model | Seed | Parameters | Best epoch | Duration (s) | "
            "Train MAE/RMSE/R² | Validation MAE/RMSE/R² | "
            "External MAE/RMSE/R² | External signed error (pp) |",
            "|---|---:|---:|---:|---:|---|---|---|---:|",
        ]
    )
    external_by_key = {
        (row["model"], int(row["seed"])): row for row in external_runs
    }
    for row in run_rows:
        external = external_by_key[(row["model"], int(row["seed"]))]
        lines.append(
            f"| {row['model']} | {row['seed']} | {row['parameter_count']} | "
            f"{row['best_epoch']} | {float(row['training_duration_seconds']):.2f} | "
            f"{float(row['train_MAE']):.3f}/{float(row['train_RMSE']):.3f}/"
            f"{float(row['train_R2']):.3f} | "
            f"{float(row['validation_MAE']):.3f}/"
            f"{float(row['validation_RMSE']):.3f}/"
            f"{float(row['validation_R2']):.3f} | "
            f"{external['MAE']:.3f}/{external['RMSE']:.3f}/"
            f"{external['R2']:.3f} | "
            f"{external['mean_signed_error_pp']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "Checkpoint SHA-256 values for the three new checkpoints and the "
            "reused baselines are in "
            "[checkpoint_freeze_manifest.json](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/artifacts/"
            "checkpoint_freeze_manifest.json). The baseline Train/Validation "
            "prediction parity check against its original saved predictions had "
            "maximum absolute difference "
            f"{max(float(row['preprocessing_prediction_max_abs_delta']) for row in run_rows if row['model'] == 'CNN1D-3M'):.2g}.",
            "",
            "## Seed stability and prediction bias",
            "",
            f"- Validation RMSE sample SD: CNN1D-2M "
            f"{two['validation_RMSE_sd']:.4f} pp; CNN1D-3M "
            f"{three['validation_RMSE_sd']:.4f} pp.",
            f"- External RMSE sample SD: CNN1D-2M {two['external_RMSE_sd']:.4f} pp; "
            f"CNN1D-3M {three['external_RMSE_sd']:.4f} pp.",
            f"- External mean signed error: CNN1D-2M "
            f"{two['external_mean_signed_error_pp_mean']:+.4f} pp; CNN1D-3M "
            f"{three['external_mean_signed_error_pp_mean']:+.4f} pp. Positive "
            "bias denotes average overprediction; negative bias denotes "
            "average underprediction.",
            "- The reported seed standard deviations describe variability across "
            "these three fixed seeds, not uncertainty over different data splits.",
            "",
            "## Cell-level External errors",
            "",
            "Per-cell metrics aggregate predictions across the three seeds and "
            "are diagnostic; overall metrics remain the primary comparison.",
            "",
            "| Model | Cell | Observations | MAE | RMSE | Mean signed error | Max absolute error |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(
        per_cell, key=lambda item: (MODEL_ORDER.index(item["model"]), item["physical_cell_id"])
    ):
        lines.append(
            f"| {row['model']} | {row['physical_cell_id']} | "
            f"{row['unique_observation_count']} | {row['MAE_pp']:.3f} | "
            f"{row['RMSE_pp']:.3f} | {row['mean_signed_error_pp']:+.3f} | "
            f"{row['max_abs_error_pp']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Analysis and limitations",
            "",
            f"- Best mean Validation RMSE: **{best_validation['model']}** "
            f"({best_validation['validation_RMSE_mean']:.4f} pp).",
            f"- Lowest mean External RMSE: **{best_external['model']}** "
            f"({best_external['external_RMSE_mean']:.4f} pp); this is a "
            "descriptive result and did not influence model choice.",
            "- The Validation-to-External gap may reflect protocol/instrument and "
            "cohort shifts; this single split cannot establish that removing "
            "Physics caused a performance change or generalizes to other cells.",
            "- The target is the frozen capacity-retention label, not an official "
            "Oxford SOH ground-truth definition. No labels or sample memberships "
            "were regenerated.",
            "- CNN1D-2M used no Physics feature or Physics-derived input. CNN1D-3M "
            "checkpoints were reused byte-for-byte; no LSTM, GRU, or CNN-LSTM "
            "models were trained or modified.",
            "",
            "## Reproducibility artifacts",
            "",
            "- [Configuration](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/benchmark_config.json)",
            "- [CNN1D-2M model](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/cnn1d_2m_model.py)",
            "- [Training script](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/train_cnn1d_2m.py)",
            "- [External evaluator](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/evaluate_cnn1d_ablation.py)",
            "- [Per-seed metrics](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/artifacts/per_seed_metrics.csv)",
            "- [Sample-key External predictions](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/artifacts/external_predictions.csv)",
            "- [Metric comparison figure](../experiments/soh_forecast/"
            "bafuse_p5_cnn_two_modality_ablation_v1/artifacts/plots/"
            "cnn1d_ablation_metrics.png)",
            f"- Freeze manifest verified {freeze['checkpoint_count']} checkpoints "
            "before External; External NPZ SHA-256 "
            f"`{external_info['external_npz_sha256']}`.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    if not ARTIFACT_DIR.is_dir():
        raise FileNotFoundError("Complete the CNN1D-2M training stage first.")
    status_path = ARTIFACT_DIR / "external_evaluation_status.json"
    if status_path.exists() or REPORT_PATH.exists():
        raise FileExistsError("Refusing to overwrite CNN1D ablation evaluation artifacts.")
    config = _read_json(CONFIG_PATH)
    if config.get("external_evaluation", {}).get("authorized") is not True:
        raise ValueError("External evaluation is not authorized by config.")
    if config.get("external_evaluation", {}).get(
        "must_wait_for_checkpoint_freeze_manifest"
    ) is not True:
        raise ValueError("Checkpoint freeze must precede External loading.")

    freeze, normalization_bundle, frozen_entries = _verify_checkpoint_freeze(config)
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
    for model_name in MODEL_ORDER:
        for seed in SEEDS:
            entry = frozen_entries[(model_name, seed)]
            checkpoint_path = PROJECT_ROOT / entry["path"]
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if model_name == "CNN1D-2M":
                model = build_cnn1d_2m_model().to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                if count_cnn1d_2m_parameters(model) != int(entry["parameter_count"]):
                    raise ValueError("CNN1D-2M parameter count changed after freeze.")
                inputs = _prepare_2m(external, discharge_scaler, charge_scaler)
                prediction = _predict_2m(model, inputs, target_scaler, device)
            else:
                model = build_cnn1d_3m_model("CNN1D").to(device)
                model.load_state_dict(checkpoint["model_state_dict"])
                if count_cnn1d_3m_parameters(model) != int(entry["parameter_count"]):
                    raise ValueError("CNN1D-3M parameter count changed after freeze.")
                inputs = _prepare_3m(external, discharge_scaler, charge_scaler)
                prediction = _predict_3m(model, inputs, target_scaler, device)
            target_values = external["y_soh"].reshape(-1)
            metrics = _metrics(target_values, prediction)
            external_metrics.append(
                {
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                    "best_epoch": int(entry["best_epoch"]),
                    "parameter_count": int(entry["parameter_count"]),
                    "checkpoint_path": entry["path"],
                    "checkpoint_sha256": entry["sha256"],
                    "external_npz_sha256": external_info["external_npz_sha256"],
                }
            )
            for index, row in enumerate(external_rows):
                residual = float(prediction[index] - target_values[index])
                external_predictions.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "dataset_part": row["dataset_part"],
                        "physical_cell_id": row["physical_cell_id"],
                        "rpt_index": row["rpt_index"],
                        "partition": "external_test",
                        "target_soh_percent": float(target_values[index]),
                        "predicted_soh_percent": float(prediction[index]),
                        "residual_prediction_minus_actual_pp": residual,
                        "absolute_error_pp": abs(residual),
                    }
                )
            logging.info(
                "External evaluated frozen %s seed %s: RMSE %.4f pp.",
                model_name,
                seed,
                metrics["RMSE"],
            )

    external_summaries = _summarize_external(external_metrics)
    per_cell = _per_cell_summary(external_predictions)
    run_rows = _read_csv(ARTIFACT_DIR / "run_metrics.csv")
    if len(run_rows) != 6:
        raise ValueError("Expected six internal Train/Validation model-seed rows.")
    internal_summaries = _read_csv(ARTIFACT_DIR / "train_validation_summary.csv")
    comparison: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        internal = next(row for row in internal_summaries if row["model"] == model_name)
        external_summary = next(
            row for row in external_summaries if row["model"] == model_name
        )
        item: dict[str, Any] = {
            "model": model_name,
            "parameter_count": int(internal["parameter_count"]),
        }
        for partition in ("train", "validation"):
            for metric in ("MAE", "RMSE", "R2"):
                item[f"{partition}_{metric}_mean"] = float(
                    internal[f"{partition}_{metric}_mean"]
                )
                item[f"{partition}_{metric}_sd"] = float(
                    internal[f"{partition}_{metric}_sd"]
                )
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            item[f"external_{metric}_mean"] = float(
                external_summary[f"{metric}_mean"]
            )
            item[f"external_{metric}_sd"] = float(
                external_summary[f"{metric}_sd"]
            )
        item["validation_to_external_RMSE_gap_pp"] = (
            item["external_RMSE_mean"] - item["validation_RMSE_mean"]
        )
        comparison.append(item)

    cnn1d_2m_by_seed = {
        int(row["seed"]): row
        for row in external_metrics
        if row["model"] == "CNN1D-2M"
    }
    cnn1d_3m_by_seed = {
        int(row["seed"]): row
        for row in external_metrics
        if row["model"] == "CNN1D-3M"
    }
    deltas = [
        {
            "seed": seed,
            "CNN1D-2M_external_RMSE": cnn1d_2m_by_seed[seed]["RMSE"],
            "CNN1D-3M_external_RMSE": cnn1d_3m_by_seed[seed]["RMSE"],
            "delta_external_RMSE_pp": (
                cnn1d_2m_by_seed[seed]["RMSE"] - cnn1d_3m_by_seed[seed]["RMSE"]
            ),
        }
        for seed in SEEDS
    ]
    worst_cases: list[dict[str, Any]] = []
    for model_name in MODEL_ORDER:
        worst_cases.extend(
            sorted(
                (row for row in external_predictions if row["model"] == model_name),
                key=lambda row: float(row["absolute_error_pp"]),
                reverse=True,
            )[:5]
        )

    _write_csv(ARTIFACT_DIR / "external_per_seed_metrics.csv", external_metrics)
    _write_csv(ARTIFACT_DIR / "external_summary_metrics.csv", external_summaries)
    _write_csv(ARTIFACT_DIR / "external_predictions.csv", external_predictions)
    _write_csv(ARTIFACT_DIR / "external_per_cell_metrics.csv", per_cell)
    _write_csv(ARTIFACT_DIR / "external_worst_cases.csv", worst_cases)
    _write_csv(ARTIFACT_DIR / "per_seed_metrics.csv", _join_per_seed_metrics(run_rows, external_metrics))
    _write_csv(ARTIFACT_DIR / "external_rmse_deltas_by_seed.csv", deltas)
    _write_csv(ARTIFACT_DIR / "cnn1d_ablation_comparison.csv", comparison)
    _plot_comparison(comparison, ARTIFACT_DIR / "plots" / "cnn1d_ablation_metrics.png")
    _plot_per_cell(per_cell, ARTIFACT_DIR / "plots" / "cnn1d_ablation_per_cell_mae.png")
    _write_report(
        config,
        normalization_bundle,
        freeze,
        run_rows,
        external_metrics,
        external_summaries,
        comparison,
        per_cell,
        deltas,
        external_info,
    )
    logging.info("CNN1D two-modality External evaluation complete.")
    status = {
        "status": "complete",
        "external_evaluation_performed": True,
        "external_used_for_training_or_selection": False,
        "verified_checkpoint_count_before_external_load": freeze["checkpoint_count"],
        "external_npz_sha256": external_info["external_npz_sha256"],
        "external_observation_count": external_info["external_observation_count"],
        "external_physical_cell_count": external_info[
            "external_physical_cell_count"
        ],
        "external_prediction_count": len(external_predictions),
        "report_path": str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "report_sha256": _sha256(REPORT_PATH),
        "delta_external_rmse_cnn1d_2m_minus_cnn1d_3m_pp": (
            comparison[0]["external_RMSE_mean"]
            - comparison[1]["external_RMSE_mean"]
        ),
    }
    _write_json(status_path, status)
    files = {
        str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"): _sha256(path)
        for path in sorted(ARTIFACT_DIR.rglob("*"))
        if path.is_file() and path.name != "final_sha256_manifest.json"
    }
    files[str(REPORT_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/")] = _sha256(
        REPORT_PATH
    )
    source_files = {
        str((EXPERIMENT_DIR / name).relative_to(PROJECT_ROOT)).replace("\\", "/"):
        _sha256(EXPERIMENT_DIR / name)
        for name in (
            "benchmark_config.json",
            "cnn1d_2m_model.py",
            "train_cnn1d_2m.py",
            "evaluate_cnn1d_ablation.py",
        )
    }
    _write_json(
        ARTIFACT_DIR / "final_sha256_manifest.json",
        {
            "experiment_identifier": config["experiment_identifier"],
            "hash_scope": "All artifacts under artifacts/ plus the final report, excluding this manifest.",
            "files": files,
            "source_files": source_files,
        },
    )
    return status


def _join_per_seed_metrics(
    run_rows: list[dict[str, str]],
    external_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    external_by_key = {
        (row["model"], int(row["seed"])): row for row in external_rows
    }
    joined: list[dict[str, Any]] = []
    for row in run_rows:
        external = external_by_key[(row["model"], int(row["seed"]))]
        result: dict[str, Any] = dict(row)
        for metric in ("MAE", "RMSE", "R2", "mean_signed_error_pp"):
            result[f"external_{metric}"] = external[metric]
        result["external_checkpoint_sha256"] = external["checkpoint_sha256"]
        result["external_npz_sha256"] = external["external_npz_sha256"]
        joined.append(result)
    return joined


if __name__ == "__main__":
    run()
