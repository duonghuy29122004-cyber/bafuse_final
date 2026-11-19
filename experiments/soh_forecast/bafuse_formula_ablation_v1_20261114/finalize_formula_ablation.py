from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
P5 = BENCHMARK / "bafuse_p5_electrical_v1_20261114"
P5_RUN = BENCHMARK / "bafuse_p5_trainval_benchmark_v2_20261114"
P5_MANIFEST = P5 / "bafuse_p5_sha256_manifest.json"
TRAINING_MANIFEST = P5_RUN / "benchmark_sha256_manifest.json"
LEGACY_AUDIT = ROOT / "experiments" / "soh_forecast" / "bafuse_legacy_p5_forensic_audit_v1_20261114"
SEEDS = (42, 123, 2026)
EXPERIMENTS = {
    "A1_discharge_only": ("A1_discharge_only", "discharge-only P5 reference"),
    "A2_discharge_charge": ("A2_discharge_charge", "add same-RPT Step 10 charge"),
    "A3_discharge_p5_physics": ("A3_discharge_p5_physics", "add P5 physics without charge"),
    "A4_full_p5": ("A4_full_p5", "reuse exact saved P5 full model"),
    "C1_legacy_arch_legacy_physics": ("C1_legacy_arch_legacy_physics", "legacy architecture and physics"),
    "C2_legacy_arch_p5_physics": ("C2_legacy_arch_p5_physics", "legacy architecture and P5 physics"),
    "C3_p5_arch_legacy_physics": ("C3_p5_arch_legacy_physics", "P5 architecture and legacy physics"),
    "C4_p5_arch_p5_physics": ("A4_full_p5", "same fitted model as A4"),
    "D_legacy_split_p5_full": ("D_legacy_split_p5_full", "P5 full model on historical split"),
    "D_p5_split_p5_full": ("A4_full_p5", "same fitted model as A4"),
}
METRIC_FIELDS = (
    "train_MAE",
    "train_RMSE",
    "train_R2",
    "validation_MAE",
    "validation_RMSE",
    "validation_R2",
    "validation_macro_cell_MAE",
    "validation_macro_cell_RMSE",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def observation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return str(row["dataset_part"]), str(row["physical_cell_id"]), str(row["rpt_index"])


def metric_values(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    errors = predicted - actual
    denominator = float(np.sum((actual - np.mean(actual)) ** 2))
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "r2": None if denominator <= 0 else float(1 - np.sum(errors**2) / denominator),
    }


def _verify_saved_p5_runs(data: dict[str, np.ndarray]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    manifest = read_json(TRAINING_MANIFEST)
    p5_config = read_json(P5_RUN / "benchmark_config.json")
    if manifest["dataset_manifest_sha256"] != sha256(P5_MANIFEST):
        raise ValueError("Saved P5 run refers to a different frozen dataset manifest.")
    if p5_config["dataset_manifest_sha256"] != sha256(P5_MANIFEST):
        raise ValueError("P5 run config refers to a different frozen dataset manifest.")
    expected_training = {
        "seeds": list(SEEDS),
        "max_epochs": 300,
        "batch_size": 16,
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "loss": "MSELoss",
        "early_stopping_patience": 35,
        "early_stopping_min_delta": 0.0,
        "checkpoint_metric": "validation_RMSE",
        "target_transform": None,
    }
    if p5_config["training"] != expected_training:
        raise ValueError("Saved P5 full-model protocol differs from the preregistered protocol.")
    if p5_config["external_prediction_evaluation_authorized"] is not False:
        raise ValueError("Saved P5 run has unexpected External authorization.")
    outputs = manifest["outputs"]

    def verify_output(relative: str) -> Path:
        path = P5_RUN / relative
        manifest_entry = outputs.get(relative)
        expected = (
            manifest_entry.get("sha256")
            if isinstance(manifest_entry, dict)
            else manifest_entry
        )
        if expected is None or sha256(path) != expected:
            raise ValueError(f"Saved P5 run output hash mismatch: {relative}")
        return path

    metrics_path = verify_output("benchmark_run_metrics.csv")
    predictions_path = verify_output("train_validation_predictions.csv")
    run_rows = read_csv(metrics_path)
    prediction_rows = read_csv(predictions_path)
    cnn_rows = [
        row for row in run_rows if row["architecture"] == "BaFuse_CNN1D_P5"
    ]
    if len(cnn_rows) != 3 or {int(row["seed"]) for row in cnn_rows} != set(SEEDS):
        raise ValueError("Expected three saved P5 CNN1D runs.")
    expected_keys = {
        (
            str(part),
            str(cell),
            str(rpt),
        ): float(target)
        for part, cell, rpt, target in zip(
            data["dataset_part"],
            data["physical_cell_id"],
            data["rpt_index"],
            data["y_soh"][:, 0],
        )
    }
    p5_predictions = [
        row for row in prediction_rows if row["architecture"] == "BaFuse_CNN1D_P5"
    ]
    if len(p5_predictions) != 139 * 3:
        raise ValueError("Saved P5 CNN1D predictions are not exactly Train/Validation.")
    if any(row["partition"] not in {"train", "validation"} for row in p5_predictions):
        raise ValueError("External prediction rows found in saved P5 train/validation outputs.")
    if any(row["dataset_part"] == "3" for row in p5_predictions):
        raise ValueError("Part 3 prediction found in saved P5 train/validation outputs.")
    predictions_by_seed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in p5_predictions:
        predictions_by_seed[int(row["seed"])].append(row)
        sample_key = observation_key(row)
        if sample_key not in expected_keys:
            raise ValueError(f"Saved P5 prediction has a non-internal key: {sample_key}")
        if not math.isclose(
            float(row["target_soh_percent"]),
            expected_keys[sample_key],
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"Saved P5 target does not align for {sample_key}")
    by_seed_metrics = {int(row["seed"]): row for row in cnn_rows}
    reused_metrics: list[dict[str, Any]] = []
    reused_predictions: list[dict[str, Any]] = []
    per_cell_rows: list[dict[str, Any]] = []
    p5_train_mask = data["partition"] == "train"
    train_mean = float(np.mean(data["y_soh"][p5_train_mask, 0]))
    for seed in SEEDS:
        metric_row = by_seed_metrics[seed]
        checkpoint_relative = metric_row["checkpoint_path"]
        history_relative = f"training_curves/BaFuse_CNN1D_P5_seed_{seed}_history.csv"
        verify_output(checkpoint_relative)
        history_path = verify_output(history_relative)
        copy_path = OUT / "training_histories" / f"A4_full_p5_seed_{seed}.csv"
        copy_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(history_path, copy_path)
        seed_rows = predictions_by_seed[seed]
        for row in seed_rows:
            reused_predictions.append(
                {
                    "run_id": f"A4_full_p5_seed_{seed}",
                    "variant": "A4_full_p5",
                    **row,
                }
            )
        for partition in ("train", "validation"):
            subset = [row for row in seed_rows if row["partition"] == partition]
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in subset:
                grouped[row["physical_cell_id"]].append(row)
            for cell, cell_rows in grouped.items():
                actual = np.asarray([float(row["target_soh_percent"]) for row in cell_rows])
                predicted = np.asarray([float(row["predicted_soh_percent"]) for row in cell_rows])
                model_metrics = metric_values(actual, predicted)
                baseline = metric_values(
                    actual, np.full(len(actual), train_mean, dtype=np.float64)
                )
                per_cell_rows.append(
                    {
                        "run_id": f"A4_full_p5_seed_{seed}",
                        "variant": "A4_full_p5",
                        "seed": seed,
                        "partition": partition,
                        "physical_cell_id": cell,
                        "observation_count": len(cell_rows),
                        **{f"model_{name}": value for name, value in model_metrics.items()},
                        "train_mean_baseline_mae": baseline["mae"],
                        "train_mean_baseline_rmse": baseline["rmse"],
                    }
                )
        val_cells = [
            row for row in per_cell_rows
            if row["run_id"] == f"A4_full_p5_seed_{seed}"
            and row["partition"] == "validation"
        ]
        if len(val_cells) != 3:
            raise ValueError("P5 saved validation run does not cover three cells.")
        val_predictions = [row for row in seed_rows if row["partition"] == "validation"]
        val_actual = np.asarray([float(row["target_soh_percent"]) for row in val_predictions])
        val_baseline = metric_values(
            val_actual, np.full(len(val_actual), train_mean, dtype=np.float64)
        )
        train_predictions = [row for row in seed_rows if row["partition"] == "train"]
        train_actual = np.asarray([float(row["target_soh_percent"]) for row in train_predictions])
        train_prediction = np.asarray([float(row["predicted_soh_percent"]) for row in train_predictions])
        val_prediction = np.asarray([float(row["predicted_soh_percent"]) for row in val_predictions])
        reused_metrics.append(
            {
                "run_id": f"A4_full_p5_seed_{seed}",
                "variant": "A4_full_p5",
                "seed": seed,
                "split_policy": "p5",
                "architecture_family": "p5",
                "physics_family": "p5",
                "charge_branch": True,
                "parameter_count": int(metric_row["parameter_count"]),
                "train_observations": len(train_predictions),
                "validation_observations": len(val_predictions),
                "train_cell_count": len({row["physical_cell_id"] for row in train_predictions}),
                "validation_cell_count": len({row["physical_cell_id"] for row in val_predictions}),
                "best_epoch": int(metric_row["best_epoch"]),
                "epochs_completed": int(metric_row["epochs_completed"]),
                "duration_seconds": float(metric_row["training_duration_seconds"]),
                "best_validation_RMSE": float(metric_row["best_validation_RMSE"]),
                "train_MAE": float(metric_row["train_MAE"]),
                "train_RMSE": float(metric_row["train_RMSE"]),
                "train_R2": float(metric_row["train_R2"]),
                "validation_MAE": float(metric_row["validation_MAE"]),
                "validation_RMSE": float(metric_row["validation_RMSE"]),
                "validation_R2": float(metric_row["validation_R2"]),
                "validation_macro_cell_MAE": statistics.mean(
                    float(row["model_mae"]) for row in val_cells
                ),
                "validation_macro_cell_RMSE": statistics.mean(
                    float(row["model_rmse"]) for row in val_cells
                ),
                "train_mean_baseline_MAE": metric_values(
                    train_actual, np.full(len(train_actual), train_mean)
                )["mae"],
                "train_mean_baseline_RMSE": metric_values(
                    train_actual, np.full(len(train_actual), train_mean)
                )["rmse"],
                "validation_train_mean_baseline_MAE": val_baseline["mae"],
                "validation_train_mean_baseline_RMSE": val_baseline["rmse"],
                "checkpoint_reference": (
                    "experiments/soh_forecast/bafuse_architecture_benchmark_v1/"
                    + P5_RUN.name
                    + "/"
                    + checkpoint_relative
                ),
                "checkpoint_sha256": metric_row["checkpoint_sha256"],
                "data_sha256": sha256(P5_MANIFEST),
                "training_code_sha256": metric_row["model_source_code_sha256"],
                "external_data_loaded": False,
                "reused_saved_run": True,
            }
        )
    return reused_metrics, reused_predictions, per_cell_rows


def _experiment_map() -> list[dict[str, Any]]:
    rows = []
    for experiment_id, (run_family, description) in EXPERIMENTS.items():
        for seed in SEEDS:
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "description": description,
                    "seed": seed,
                    "run_id": f"{run_family}_seed_{seed}",
                    "same_fit_as": (
                        "A4_full_p5"
                        if experiment_id in {"C4_p5_arch_p5_physics", "D_p5_split_p5_full"}
                        else ""
                    ),
                    "reused": experiment_id in {
                        "A4_full_p5",
                        "C4_p5_arch_p5_physics",
                        "D_p5_split_p5_full",
                    },
                }
            )
    return rows


def _summaries(
    all_runs: list[dict[str, Any]], experiment_map: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_run = {row["run_id"]: row for row in all_runs}
    rows: list[dict[str, Any]] = []
    for experiment_id, (run_family, description) in EXPERIMENTS.items():
        run_rows = [
            by_run[row["run_id"]]
            for row in experiment_map
            if row["experiment_id"] == experiment_id
        ]
        if len(run_rows) != 3:
            raise ValueError(f"Expected three seeds for {experiment_id}.")
        summary: dict[str, Any] = {
            "experiment_id": experiment_id,
            "description": description,
            "parameter_count": run_rows[0]["parameter_count"],
            "seed_count": len(run_rows),
            "reused_saved_fit": all(bool(row.get("reused_saved_run")) for row in run_rows),
            "best_epoch_mean": statistics.mean(float(row["best_epoch"]) for row in run_rows),
        }
        for metric in METRIC_FIELDS:
            values = [float(row[metric]) for row in run_rows]
            summary[f"{metric}_mean"] = statistics.mean(values)
            summary[f"{metric}_sample_sd"] = statistics.stdev(values)
        summary["train_observations"] = run_rows[0]["train_observations"]
        summary["validation_observations"] = run_rows[0]["validation_observations"]
        summary["train_cell_count"] = run_rows[0]["train_cell_count"]
        summary["validation_cell_count"] = run_rows[0]["validation_cell_count"]
        rows.append(summary)
    return rows


def _paired_comparisons(
    all_runs: list[dict[str, Any]], experiment_map: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_run = {row["run_id"]: row for row in all_runs}
    experiment_run = {
        (row["experiment_id"], int(row["seed"])): by_run[row["run_id"]]
        for row in experiment_map
    }
    ablation_rows = []
    for seed in SEEDS:
        full = experiment_run[("A4_full_p5", seed)]
        for experiment_id in (
            "A1_discharge_only",
            "A2_discharge_charge",
            "A3_discharge_p5_physics",
        ):
            variant = experiment_run[(experiment_id, seed)]
            ablation_rows.append(
                {
                    "seed": seed,
                    "comparison": f"{experiment_id}_minus_A4_full_p5",
                    "variant_parameter_count": variant["parameter_count"],
                    "reference_parameter_count": full["parameter_count"],
                    "validation_RMSE_difference_pp": float(variant["validation_RMSE"])
                    - float(full["validation_RMSE"]),
                    "validation_MAE_difference_pp": float(variant["validation_MAE"])
                    - float(full["validation_MAE"]),
                    "validation_macro_cell_RMSE_difference_pp": float(
                        variant["validation_macro_cell_RMSE"]
                    )
                    - float(full["validation_macro_cell_RMSE"]),
                }
            )
    factorial_rows = []
    for seed in SEEDS:
        c1 = experiment_run[("C1_legacy_arch_legacy_physics", seed)]
        c2 = experiment_run[("C2_legacy_arch_p5_physics", seed)]
        c3 = experiment_run[("C3_p5_arch_legacy_physics", seed)]
        c4 = experiment_run[("C4_p5_arch_p5_physics", seed)]
        factorial_rows.append(
            {
                "seed": seed,
                "legacy_arch_physics_effect_C2_minus_C1_pp": float(c2["validation_RMSE"]) - float(c1["validation_RMSE"]),
                "p5_arch_physics_effect_C4_minus_C3_pp": float(c4["validation_RMSE"]) - float(c3["validation_RMSE"]),
                "legacy_physics_arch_effect_C3_minus_C1_pp": float(c3["validation_RMSE"]) - float(c1["validation_RMSE"]),
                "p5_physics_arch_effect_C4_minus_C2_pp": float(c4["validation_RMSE"]) - float(c2["validation_RMSE"]),
                "physics_by_architecture_interaction_pp": (
                    float(c4["validation_RMSE"])
                    - float(c3["validation_RMSE"])
                    - float(c2["validation_RMSE"])
                    + float(c1["validation_RMSE"])
                ),
            }
        )
    split_rows = []
    for seed in SEEDS:
        historical = experiment_run[("D_legacy_split_p5_full", seed)]
        p5 = experiment_run[("D_p5_split_p5_full", seed)]
        split_rows.append(
            {
                "seed": seed,
                "legacy_train_n": historical["train_observations"],
                "legacy_validation_n": historical["validation_observations"],
                "p5_train_n": p5["train_observations"],
                "p5_validation_n": p5["validation_observations"],
                "legacy_validation_RMSE_pp": historical["validation_RMSE"],
                "p5_validation_RMSE_pp": p5["validation_RMSE"],
                "p5_minus_legacy_validation_RMSE_pp": float(p5["validation_RMSE"])
                - float(historical["validation_RMSE"]),
                "legacy_validation_macro_cell_RMSE_pp": historical["validation_macro_cell_RMSE"],
                "p5_validation_macro_cell_RMSE_pp": p5["validation_macro_cell_RMSE"],
                "validation_cell_sets_identical": False,
                "causal_isolation": "not identified; validation cells and checkpoint-selection populations differ",
            }
        )
    return ablation_rows, factorial_rows, split_rows


def _plot_results(summaries: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    selected_ids = (
        "A1_discharge_only",
        "A2_discharge_charge",
        "A3_discharge_p5_physics",
        "A4_full_p5",
        "C1_legacy_arch_legacy_physics",
        "C2_legacy_arch_p5_physics",
        "C3_p5_arch_legacy_physics",
        "D_legacy_split_p5_full",
    )
    by_id = {row["experiment_id"]: row for row in summaries}
    values = [float(by_id[name]["validation_RMSE_mean"]) for name in selected_ids]
    errors = [float(by_id[name]["validation_RMSE_sample_sd"]) for name in selected_ids]
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(range(len(selected_ids)), values, yerr=errors, capsize=4)
    axis.set_xticks(range(len(selected_ids)), [name.replace("_", "\n") for name in selected_ids])
    axis.set_ylabel("Validation RMSE (SoH percentage points)")
    axis.set_title("Train-only ablations; mean +/- sample SD across three seeds")
    figure.tight_layout()
    figure.savefig(OUT / "plots" / "validation_rmse_by_variant.png", dpi=150)
    plt.close(figure)

    factorial_names = (
        "C1_legacy_arch_legacy_physics",
        "C2_legacy_arch_p5_physics",
        "C3_p5_arch_legacy_physics",
        "C4_p5_arch_p5_physics",
    )
    factorial_values = [
        float(by_id[name]["validation_RMSE_mean"]) for name in factorial_names
    ]
    figure, axis = plt.subplots(figsize=(8, 5))
    positions = [0, 1, 0, 1]
    offsets = [-0.06, -0.06, 0.06, 0.06]
    labels = ["Legacy physics", "P5 physics"]
    colors = ["#4c78a8", "#f58518"]
    for index, name in enumerate(factorial_names):
        physics_index = index % 2
        axis.scatter(
            positions[index] + offsets[index],
            factorial_values[index],
            color=colors[physics_index],
            label=labels[physics_index] if index < 2 else None,
            s=80,
        )
        axis.annotate(
            name.replace("_", " "),
            (positions[index] + offsets[index], factorial_values[index]),
            xytext=(4, 6),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xticks([0, 1], ["Legacy architecture", "P5 architecture"])
    axis.set_ylabel("Validation RMSE mean (percentage points)")
    axis.set_title("Architecture x physics factorial")
    axis.legend()
    figure.tight_layout()
    figure.savefig(OUT / "plots" / "architecture_physics_factorial.png", dpi=150)
    plt.close(figure)


def _write_reports(
    all_runs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    ablation_rows: list[dict[str, Any]],
    factorial_rows: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
    experiment_map: list[dict[str, Any]],
) -> None:
    by_summary = {row["experiment_id"]: row for row in summaries}
    lines = [
        "# BaFuse Train-only formula, population, and ablation results",
        "",
        "## Authorization and scope",
        "",
        "- New Train/Validation runs were executed under the current explicit authorization.",
        "- No new External model prediction evaluation, External training, or External checkpoint selection was performed.",
        "- The training runner opened only `data/internal_trainval_raw.npz` (139 Part 1-2 Train/Validation rows). External arrays/labels were not loaded in that process.",
        "- Frozen P5 labels, splits, arrays, historical checkpoints, and legacy artifacts were not modified.",
        "- No PyDMA, Half-Cell fitting, or RAG was performed.",
        "",
        "## Dataset and population audit",
        "",
        "- P5: 203 accepted labels across 23 cells; 102 Train / 37 Validation / 64 Part 3 External Test.",
        "- Historical legacy comparison: 220 rows (106/41/73), comprising 211 primary saved-model rows plus nine supplemental G10-Cell2 saved-checkpoint rows. Common keys: 203; legacy-only: 17; P5-only: 0.",
        "- Of the legacy-only keys, eight internal rows have ambiguous Step 4 segment cardinality and no ordinal assignment. This is upstream source/label mapping ambiguity, not a P5 modality QC failure.",
        "- Nine historical G10-Cell2 rows lack candidate RPT rows in the M1 Guide-mapped manifest. The P5 cohort quarantines them rather than fabricating labels/features.",
        "- Six internal cells changed assignment between the legacy and P5 split manifests. Both assignments are explicit manifests; no random split seed regenerated them.",
        "- The 139 shared internal keys have legacy 99/40 and P5 102/37 Train/Validation assignment. The Validation cell sets are disjoint, preventing a clean separation of training-composition, validation-difficulty, and checkpoint-selection effects.",
        "",
        "## Formula findings",
        "",
        "- Step 4 discharge and same-RPT Step 10 charge curves are linearly resampled to 128 points using normalized source-row progress. Capacity/time are not used in the progress coordinate.",
        "- Legacy and P5 curve arrays match within the P5 row-audit tolerance. P5 physics avoids legacy temperature inputs with unresolved native units/calibration.",
        "- Legacy physics comprises Step 4 temperature mean/rise, discharge voltage slope/derivative spread, and mean absolute current. P5 physics comprises mean absolute current, robust current MAD, voltage slope, voltage-derivative MAD, and voltage segment contrast. All are same-RPT engineered summaries, not independent sensors.",
        "- Step 4 Amphr, integrated current-time capacity, duration, capacity/BoL-derived quantities, SoH-derived inputs, and identifiers are not predictive features.",
        "- New runs use common raw-voltage curves and raw-percent target across architecture families; only physics is Train-standardized. This isolates branch architecture/feature content within the P5 protocol, but does not reproduce every historical legacy preprocessing difference.",
        "- The P5 mean-absolute-current Train scale is `3.883535206084406e-06` A (inferred units); 46/64 External observations lie outside the Train raw range. This is reported as descriptive domain shift, not used to define a threshold or select a feature.",
        "- Temperature is not considered universally comparable across Parts; units/calibration and wider missingness remain unresolved. No temperature imputation was applied.",
        "",
        "## Train-only scaler edge case",
        "",
        "- The historical split has one P5 physics feature, `discharge_current_variability`, with exactly zero population standard deviation across its 99 Train rows. Validation has two values, including one absent from Train.",
        "- After the first 18 runs, the scaler correctly stopped rather than dividing by zero. The documented addendum specifies a Train-only constant-feature mask: that feature is mapped to zero for all rows in the historical split, so Validation-only variation cannot enter a model with no Train-estimated feature effect.",
        "- The 18 completed runs were preserved; only the three remaining historical-split seeds used the constant-feature handling. Per-run code hashes and scaler metadata are retained. This edge-case resolution is a disclosed protocol addendum, not an outcome-driven feature choice.",
        "",
        "## Per-experiment Validation RMSE",
        "",
        "| Experiment | Parameters | Train/Validation n | Validation RMSE mean +/- sample SD (pp) | Validation MAE mean +/- sample SD (pp) | Macro-cell RMSE mean +/- sample SD (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in summaries:
        lines.append(
            f"| {summary['experiment_id']} | {summary['parameter_count']} | "
            f"{summary['train_observations']}/{summary['validation_observations']} | "
            f"{summary['validation_RMSE_mean']:.4f} +/- {summary['validation_RMSE_sample_sd']:.4f} | "
            f"{summary['validation_MAE_mean']:.4f} +/- {summary['validation_MAE_sample_sd']:.4f} | "
            f"{summary['validation_macro_cell_RMSE_mean']:.4f} +/- {summary['validation_macro_cell_RMSE_sample_sd']:.4f} |"
        )
    lines.extend(
        [
            "",
            "A4, C4, and D-P5 are alternate matrix references to the same three saved P5 CNN1D runs. They are not independent replications.",
            "",
            "## Controlled comparisons",
            "",
            "### Branch ablations",
            "",
        ]
    )
    for comparison in ("A1_discharge_only", "A2_discharge_charge", "A3_discharge_p5_physics"):
        rowset = [row for row in ablation_rows if row["comparison"].startswith(comparison)]
        values = [float(row["validation_RMSE_difference_pp"]) for row in rowset]
        lines.append(
            f"- {comparison} minus full A4 Validation RMSE: "
            f"{statistics.mean(values):+.4f} pp mean paired delta "
            f"(sample SD {statistics.stdev(values):.4f}); parameter counts are "
            f"{rowset[0]['variant_parameter_count']} vs {rowset[0]['reference_parameter_count']}."
        )
    lines.extend(
        [
            "",
            "These are branch-removal ablations; parameter count changes with branch count, so they do not isolate information content from capacity.",
            "",
            "### Architecture x physics factorial",
            "",
        ]
    )
    for field in (
        "legacy_arch_physics_effect_C2_minus_C1_pp",
        "p5_arch_physics_effect_C4_minus_C3_pp",
        "legacy_physics_arch_effect_C3_minus_C1_pp",
        "p5_physics_arch_effect_C4_minus_C2_pp",
        "physics_by_architecture_interaction_pp",
    ):
        values = [float(row[field]) for row in factorial_rows]
        lines.append(
            f"- `{field}`: {statistics.mean(values):+.4f} pp mean paired Validation RMSE difference "
            f"(sample SD {statistics.stdev(values):.4f})."
        )
    lines.extend(
        [
            "",
            "All four factorial cells use the same P5 Train/Validation observations, labels, raw-voltage inputs, seed set, and selection rule. These contrasts estimate sensitivity on this one small cell split; they do not establish population-level superiority.",
            "",
            "### Split sensitivity",
            "",
        ]
    )
    split_deltas = [float(row["p5_minus_legacy_validation_RMSE_pp"]) for row in split_rows]
    lines.append(
        f"- P5 minus historical-split own-Validation RMSE: {statistics.mean(split_deltas):+.4f} pp mean paired delta "
        f"(sample SD {statistics.stdev(split_deltas):.4f})."
    )
    lines.append(
        "- This is not a pure training-composition effect: the validation cell sets differ and each policy selects checkpoints on its own Validation set. No shared held-out cell set exists under the two authorized manifests."
    )
    lines.extend(
        [
            "",
            "## Conclusions and boundaries",
            "",
            "- The new Train/Validation experiments isolate branch/feature and network-block changes more cleanly than the historical-versus-P5 comparison. Feature/architecture effects can be summarized for these internal cells, with per-cell results and seed uncertainty.",
            "- The legacy-vs-P5 external performance gap is not causally resolved here. No new External predictions were generated or evaluated, as required.",
            "- The legacy-only internal observations remain quarantined pending unique Step 4 segment assignment; G10-Cell2 remains excluded pending Guide-backed identity/RPT mapping.",
            "- No conclusions promote temperature features, alter the frozen cohort, or authorize model selection on Part 3.",
            "- Dataset freeze/training readiness is not assessed; this is a controlled diagnostic benchmark only.",
            "",
            f"- New runs: {sum(not row.get('reused_saved_run', False) for row in all_runs)}; reused exact P5 full runs: {sum(bool(row.get('reused_saved_run')) for row in all_runs)}.",
            "- All metrics are in SoH percentage points where applicable. Validation macro-cell metrics give equal weight to the three validation cells; overall metrics weight observations equally.",
            "",
        ]
    )
    (OUT / "formula_ablation_results.md").write_text("\n".join(lines), encoding="utf-8")

    environment = read_json(OUT / "training_environment.json")
    repro = [
        "# Formula ablation reproducibility",
        "",
        f"- Frozen P5 dataset manifest SHA-256: `{sha256(P5_MANIFEST)}`.",
        f"- P5 saved Train/Validation benchmark manifest SHA-256: `{sha256(TRAINING_MANIFEST)}`.",
        f"- Prior forensic population/alignment manifest SHA-256: `{sha256(LEGACY_AUDIT / 'forensic_audit_sha256_manifest.json')}`.",
        f"- Sanitized internal Train/Validation archive SHA-256: `{sha256(OUT / 'data' / 'internal_trainval_raw.npz')}`.",
        f"- Current training source code SHA-256: `{sha256(OUT / 'train_formula_ablation.py')}`.",
        f"- Preparation code SHA-256: `{sha256(OUT / 'prepare_formula_ablation.py')}`.",
        f"- Finalization code SHA-256: `{sha256(OUT / 'finalize_formula_ablation.py')}`.",
        f"- Config SHA-256: `{sha256(OUT / 'ablation_training_config.json')}`.",
        f"- Device: `{environment['device_name']}`; CUDA `{environment['cuda_runtime']}`; PyTorch `{environment['pytorch_version']}`.",
        "- Per-run seeds: 42, 123, 2026; RNGs were seeded, deterministic PyTorch algorithms enabled, cuDNN deterministic enabled, and DataLoader workers disabled.",
        "- Physics scaling is Train-only for each split/feature family. Curves and target remain unscaled. No preprocessing was fit on Validation or External.",
        "- New checkpoints are separate from historical directories. A4/C4/D-P5 reference saved P5 checkpoint hashes verified against the P5 training manifest.",
        "- P5 reuse included only the saved CNN1D Train/Validation predictions and histories (139 rows per seed); no External NPZ was opened by the training process or reused for evaluation.",
        "- Reported source statuses include `CURRENT_SNAPSHOT_ONLY`; hashes establish identity at audit time, not an immutable raw-data archive.",
        "",
        "| Training code SHA-256 recorded in run | New runs |",
        "|---|---:|",
    ]
    for code_hash, run_count in sorted(
        Counter(
            row["training_code_sha256"]
            for row in all_runs
            if not row.get("reused_saved_run")
        ).items()
    ):
        repro.append(f"| `{code_hash}` | {run_count} |")
    repro.extend(
        [
            "",
            "The 18 runs completed before the scaler edge-case stop retain their original code hash. The three resumed historical-split runs use the hash recorded for the constant-feature handling; see `scaler_edge_case_addendum.md`.",
            "",
        "| Experiment | Seed | Checkpoint SHA-256 | Reused saved run |",
        "|---|---:|---|---|",
        ]
    )
    for row in all_runs:
        repro.append(
            f"| {row['variant']} | {row['seed']} | `{row['checkpoint_sha256']}` | "
            f"{bool(row.get('reused_saved_run'))} |"
        )
    repro.append("")
    (OUT / "formula_ablation_reproducibility.md").write_text(
        "\n".join(repro), encoding="utf-8"
    )


def _write_hash_manifest() -> None:
    source_rows = read_csv(OUT / "source_sha256_manifest.csv")
    inputs = {
        row["path"]: row["sha256"] for row in source_rows
    }
    inputs["experiments/soh_forecast/bafuse_formula_ablation_v1_20261114/data/internal_trainval_raw.npz"] = sha256(
        OUT / "data" / "internal_trainval_raw.npz"
    )
    code = {}
    for name in (
        "prepare_formula_ablation.py",
        "train_formula_ablation.py",
        "finalize_formula_ablation.py",
        "test_formula_ablation.py",
    ):
        path = OUT / name
        if path.is_file():
            code[path.relative_to(ROOT).as_posix()] = sha256(path)
    outputs = {}
    for path in sorted(OUT.rglob("*")):
        if not path.is_file() or path.name == "experiment_sha256_manifest.json":
            continue
        if "__pycache__" in path.parts:
            continue
        outputs[path.relative_to(OUT).as_posix()] = sha256(path)
    manifest = {
        "experiment_identifier": "bafuse-formula-ablation-v1-20261114",
        "self_hash_excluded": True,
        "training_authorized": True,
        "external_prediction_evaluation_authorized": False,
        "training_process_external_npz_loaded": False,
        "dataset_identifier": "bafuse-p5-electrical-v1-20261114",
        "frozen_dataset_manifest_sha256": sha256(P5_MANIFEST),
        "saved_p5_training_manifest_sha256": sha256(TRAINING_MANIFEST),
        "prior_forensic_manifest_sha256": sha256(
            LEGACY_AUDIT / "forensic_audit_sha256_manifest.json"
        ),
        "inputs_sha256": inputs,
        "code_sha256": code,
        "outputs_sha256": outputs,
    }
    (OUT / "experiment_sha256_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    data_path = OUT / "data" / "internal_trainval_raw.npz"
    with np.load(data_path, allow_pickle=False) as archive:
        data = {name: archive[name].copy() for name in archive.files}
    if len(data["y_soh"]) != 139:
        raise ValueError("Sanitized training input is not the 139-row internal cohort.")
    training_runs = read_csv(OUT / "training_run_metrics.csv")
    if len(training_runs) != 21:
        raise ValueError(f"Expected 21 new trained runs, found {len(training_runs)}.")
    new_predictions = read_csv(OUT / "training_predictions.csv")
    new_per_cell = read_csv(OUT / "per_cell_metrics.csv")
    if any(row["dataset_part"] == "3" for row in new_predictions):
        raise ValueError("External prediction rows found in newly trained outputs.")
    reused_metrics, reused_predictions, reused_per_cell = _verify_saved_p5_runs(data)
    all_runs = training_runs + reused_metrics
    all_predictions = new_predictions + reused_predictions
    all_per_cell = new_per_cell + reused_per_cell
    if len(all_runs) != 24 or len(all_predictions) != 24 * 139:
        raise ValueError("Unexpected complete run/prediction counts.")
    if any(row["dataset_part"] == "3" for row in all_predictions):
        raise ValueError("External predictions found in combined Train/Validation outputs.")
    experiment_map = _experiment_map()
    summaries = _summaries(all_runs, experiment_map)
    ablation_rows, factorial_rows, split_rows = _paired_comparisons(
        all_runs, experiment_map
    )
    write_csv(OUT / "experiment_matrix.csv", experiment_map)
    write_csv(OUT / "all_run_metrics.csv", all_runs)
    write_csv(OUT / "all_train_validation_predictions.csv", all_predictions)
    write_csv(OUT / "all_per_cell_metrics.csv", all_per_cell)
    write_csv(OUT / "variant_summary.csv", summaries)
    write_csv(OUT / "branch_ablation_paired_deltas.csv", ablation_rows)
    write_csv(OUT / "architecture_physics_factorial.csv", factorial_rows)
    write_csv(OUT / "split_sensitivity.csv", split_rows)
    _plot_results(summaries)
    _write_reports(
        all_runs,
        summaries,
        ablation_rows,
        factorial_rows,
        split_rows,
        experiment_map,
    )
    _write_hash_manifest()
    print(
        json.dumps(
            {
                "trained_runs": len(training_runs),
                "reused_runs": len(reused_metrics),
                "total_unique_runs": len(all_runs),
                "prediction_rows": len(all_predictions),
                "experiment_cells": len(experiment_map),
                "manifest_sha256": sha256(OUT / "experiment_sha256_manifest.json"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
