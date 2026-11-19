from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from bafuse.hybrid.architecture_models import ARCHITECTURE_NAMES, architecture_parameter_counts, build_architecture
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES
from bafuse.hybrid.training_authorization import (
    build_training_scope,
    require_training_authorization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPARATION_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "architecture_matrix_v1"
)
CONFIG_PATH = PREPARATION_DIR / "architecture_config.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _iter_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        yield from csv.DictReader(stream)


def _resolve(path: str) -> Path:
    return PROJECT_ROOT / Path(path.replace("/", "\\"))


def _curve_columns(prefix: str) -> list[str]:
    return [f"{prefix}_v_{index:03d}" for index in range(128)]


def load_internal_cohort(
    label_manifest: Path,
    feature_matrix: Path,
    split_manifest: Path,
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    split_rows = _read_csv(split_manifest)
    split_by_cell: dict[str, str] = {}
    for row in split_rows:
        cell = row["physical_cell_id"]
        if cell in split_by_cell:
            raise ValueError(f"Duplicate physical cell in split manifest: {cell}")
        split_by_cell[cell] = row["partition"]
    if dict(Counter(split_by_cell.values())) != {
        "train": 9,
        "validation": 3,
        "external_test": 12,
    }:
        raise ValueError("Training requires the unchanged 9/3/0/12 physical-cell split.")

    labels: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _iter_csv(label_manifest):
        if row["partition"] not in {"train", "validation"}:
            continue
        if row["label_disposition"] != "ACCEPTED":
            continue
        if row["final_eligibility_status"] != "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD":
            continue
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in labels:
            raise ValueError(f"Duplicate accepted target identity: {key}")
        if split_by_cell.get(row["physical_cell_id"]) != row["partition"]:
            raise ValueError(f"Cell split mismatch for accepted target {key}.")
        labels[key] = row

    required_features = (
        *_curve_columns("discharge"),
        *_curve_columns("charge"),
        *PHYSICS_FEATURE_NAMES,
    )
    features: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _iter_csv(feature_matrix):
        if row["partition"] not in {"train", "validation"}:
            continue
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key not in labels:
            continue
        if row["partition"] != labels[key]["partition"]:
            raise ValueError(f"Feature/label partition mismatch for {key}.")
        if key in features:
            raise ValueError(f"Duplicate raw feature identity: {key}")
        missing = [name for name in required_features if not row.get(name)]
        if missing:
            raise ValueError(f"Feature row {key} lacks required predictors: {missing}")
        features[key] = row

    if not labels:
        raise ValueError("No accepted complete Train/Validation records are available.")
    if set(features) != set(labels):
        missing_features = sorted(set(labels).difference(features))
        missing_labels = sorted(set(features).difference(labels))
        raise ValueError(
            f"Label/feature cohort mismatch; missing features={missing_features}; "
            f"unexpected features={missing_labels}."
        )
    rows = sorted(
        labels.values(),
        key=lambda row: (row["partition"], row["physical_cell_id"], int(row["rpt_index"])),
    )
    train_cells = {
        row["physical_cell_id"] for row in rows if row["partition"] == "train"
    }
    validation_cells = {
        row["physical_cell_id"] for row in rows if row["partition"] == "validation"
    }
    if not train_cells or not validation_cells or train_cells.intersection(validation_cells):
        raise ValueError("Train and Validation require nonempty, physically isolated cell sets.")
    return rows, {
        "|".join(key): value for key, value in features.items()
    }


def _tensor_cohort(
    rows: list[dict[str, str]],
    feature_map: dict[str, dict[str, str]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, list[dict[str, str]], dict[str, Any]]:
    keys = [
        "|".join((row["dataset_part"], row["physical_cell_id"], row["rpt_index"]))
        for row in rows
    ]
    discharge = torch.tensor(
        [[float(feature_map[key][name]) for name in _curve_columns("discharge")] for key in keys],
        dtype=torch.float32,
    )
    charge = torch.tensor(
        [[float(feature_map[key][name]) for name in _curve_columns("charge")] for key in keys],
        dtype=torch.float32,
    )
    physics = torch.tensor(
        [[float(feature_map[key][name]) for name in PHYSICS_FEATURE_NAMES] for key in keys],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [float(row["reconstructed_soh_percent"]) for row in rows],
        dtype=torch.float32,
    )
    if not all(
        bool(torch.isfinite(tensor).all())
        for tensor in (discharge, charge, physics, targets)
    ):
        raise ValueError("Internal cohort contains non-finite features or targets.")

    train_indices = [
        index for index, row in enumerate(rows) if row["partition"] == "train"
    ]
    train_physics = physics[train_indices]
    means = train_physics.mean(dim=0)
    scales = train_physics.std(dim=0, unbiased=False)
    scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    physics = (physics - means) / scales
    preprocessing = {
        "physics_feature_names": list(PHYSICS_FEATURE_NAMES),
        "fit_scope": "Train records only",
        "means": means.tolist(),
        "scales": scales.tolist(),
        "curve_scaling": "No voltage-value normalization; preserve V signal.",
    }
    return (
        {"discharge": discharge, "charge": charge, "physics": physics},
        targets,
        rows,
        preprocessing,
    )


def _metric_values(
    target: torch.Tensor,
    prediction: torch.Tensor,
    *,
    physical_cell_count: int,
) -> dict[str, float | int | None]:
    error = prediction - target
    mae = torch.mean(torch.abs(error)).item()
    rmse = torch.sqrt(torch.mean(error.square())).item()
    denominator = torch.sum((target - torch.mean(target)).square()).item()
    r2 = (
        None
        if denominator <= 0
        else 1.0 - torch.sum(error.square()).item() / denominator
    )
    return {
        "N": int(target.numel()),
        "N_cells": physical_cell_count,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
    }


def _prediction_metrics_by_cell(
    rows: list[dict[str, str]],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    *,
    architecture: str,
    seed: int | str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexes: defaultdict[tuple[str, str], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indexes[(row["partition"], row["physical_cell_id"])].append(index)
    output: list[dict[str, Any]] = []
    for (partition, cell), selected in sorted(indexes.items()):
        indices = torch.tensor(selected, dtype=torch.long)
        metrics = _metric_values(
            targets[indices], predictions[indices], physical_cell_count=1
        )
        output.append(
            {
                "architecture": architecture,
                "seed": seed,
                "partition": partition,
                "physical_cell_id": cell,
                **metrics,
            }
        )
    macro: dict[str, Any] = {}
    for partition in ("train", "validation"):
        cell_rows = [row for row in output if row["partition"] == partition]
        cell_r2 = [row["R2"] for row in cell_rows if row["R2"] is not None]
        macro[partition] = {
            "cells": len(cell_rows),
            "macro_MAE": statistics.mean(row["MAE"] for row in cell_rows) if cell_rows else None,
            "macro_RMSE": statistics.mean(row["RMSE"] for row in cell_rows) if cell_rows else None,
            "macro_R2": statistics.mean(cell_r2) if cell_r2 else None,
            "macro_R2_cells": len(cell_r2),
        }
    return output, macro


def _predict(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    selected = torch.tensor(indices, dtype=torch.long)
    with torch.inference_mode():
        return model(
            inputs["discharge"][selected].to(device),
            inputs["charge"][selected].to(device),
            inputs["physics"][selected].to(device),
        ).squeeze(1).cpu()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty training artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_training(
    config_path: Path = CONFIG_PATH,
    authorization_file: Path | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("The checked-in experiment config must keep training_authorized false.")
    if config.get("architectures") != list(ARCHITECTURE_NAMES):
        raise ValueError("Experiment config does not match the implemented four architectures.")
    proposal = config["training_proposal_not_approved"]
    if proposal.get("approval_required") is not True:
        raise ValueError("Training proposal must remain explicitly approval-gated.")
    label_path = _resolve(config["label_manifest"])
    feature_path = _resolve(config["feature_matrix"])
    split_path = _resolve(config["split_manifest"])
    output_dir = _resolve(config["output_directory"])
    scope = build_training_scope(
        label_manifest=label_path,
        feature_matrix=feature_path,
        split_manifest=split_path,
        experiment_config=config_path,
        compute_budget=proposal,
        output_directory=output_dir,
        architectures=config["architectures"],
    )
    if authorization_file is None:
        raise ValueError("Provide a scope-bound --authorization-file before training.")
    require_training_authorization(authorization_file, scope)
    if float(proposal["max_total_device_hours"]) <= 0:
        raise ValueError("The approved total compute budget must be positive.")
    if int(proposal["batch_size"]) <= 0 or int(proposal["max_epochs_per_run"]) <= 0:
        raise ValueError("The approved batch size and epoch limit must be positive.")

    # No target tensor, scaler, checkpoint, or optimizer is created before the
    # explicit authorization above has been verified.
    rows, feature_map = load_internal_cohort(label_path, feature_path, split_path)
    inputs, targets, rows, preprocessing = _tensor_cohort(rows, feature_map)
    train_indices = [i for i, row in enumerate(rows) if row["partition"] == "train"]
    validation_indices = [i for i, row in enumerate(rows) if row["partition"] == "validation"]
    device = torch.device(proposal["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Authorized CUDA device is unavailable: {device}")

    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "preprocessing.json").write_text(
        json.dumps(preprocessing, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "authorization_scope.json").write_text(
        json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    train_mean = targets[train_indices].mean()
    baseline_predictions = torch.full_like(targets, float(train_mean))
    baseline_rows: list[dict[str, Any]] = []
    baseline_cells, baseline_macro = _prediction_metrics_by_cell(
        rows,
        targets,
        baseline_predictions,
        architecture="train_mean_baseline",
        seed="none",
    )
    for partition, indices in (("train", train_indices), ("validation", validation_indices)):
        tensor_indices = torch.tensor(indices, dtype=torch.long)
        cell_macro = baseline_macro[partition]
        baseline_rows.append(
            {
                "architecture": "train_mean_baseline",
                "seed": "none",
                "partition": partition,
                **_metric_values(
                    targets[tensor_indices],
                    baseline_predictions[tensor_indices],
                    physical_cell_count=cell_macro["cells"],
                ),
                "macro_cell_MAE": cell_macro["macro_MAE"],
                "macro_cell_RMSE": cell_macro["macro_RMSE"],
                "macro_cell_R2": cell_macro["macro_R2"],
                "macro_cell_R2_cells": cell_macro["macro_R2_cells"],
            }
        )

    metrics: list[dict[str, Any]] = list(baseline_rows)
    cell_metrics: list[dict[str, Any]] = list(baseline_cells)
    prediction_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        prediction_rows.append(
            {
                "architecture": "train_mean_baseline",
                "seed": "none",
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "partition": row["partition"],
                "target_soh_percent": float(targets[index]),
                "predicted_soh_percent": float(baseline_predictions[index]),
            }
        )

    started_all = time.perf_counter()
    time_budget_seconds = float(proposal["max_total_device_hours"]) * 3600.0
    seed_values = [int(seed) for seed in proposal["seeds"]]
    parameter_counts = architecture_parameter_counts()
    torch.use_deterministic_algorithms(True)
    for architecture in ARCHITECTURE_NAMES:
        for seed in seed_values:
            if time.perf_counter() - started_all >= time_budget_seconds:
                raise TimeoutError("Approved total training time budget was exhausted.")
            random.seed(seed)
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            model = build_architecture(architecture).to(device)
            parameter_count = parameter_counts[architecture]
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=float(proposal["learning_rate"]),
                weight_decay=float(proposal["weight_decay"]),
            )
            train_dataset = TensorDataset(
                inputs["discharge"][train_indices],
                inputs["charge"][train_indices],
                inputs["physics"][train_indices],
                targets[train_indices],
            )
            generator = torch.Generator().manual_seed(seed)
            loader = DataLoader(
                train_dataset,
                batch_size=int(proposal["batch_size"]),
                shuffle=True,
                generator=generator,
            )
            best_state: dict[str, torch.Tensor] | None = None
            best_validation_mae = math.inf
            best_epoch = 0
            stale_epochs = 0
            run_started = time.perf_counter()
            for epoch in range(1, int(proposal["max_epochs_per_run"]) + 1):
                if time.perf_counter() - started_all >= time_budget_seconds:
                    break
                model.train()
                for discharge, charge, physics, target in loader:
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(
                        discharge.to(device),
                        charge.to(device),
                        physics.to(device),
                    ).squeeze(1)
                    loss = nn.functional.mse_loss(prediction, target.to(device))
                    loss.backward()
                    optimizer.step()

                validation_prediction = _predict(
                    model, inputs, validation_indices, device
                )
                validation_target = targets[validation_indices]
                validation_mae = torch.mean(
                    torch.abs(validation_prediction - validation_target)
                ).item()
                if validation_mae < best_validation_mae:
                    best_validation_mae = validation_mae
                    best_epoch = epoch
                    stale_epochs = 0
                    best_state = {
                        name: value.detach().cpu().clone()
                        for name, value in model.state_dict().items()
                    }
                else:
                    stale_epochs += 1
                    if stale_epochs >= int(proposal["early_stopping_patience"]):
                        break
            if best_state is None:
                raise RuntimeError(
                    f"No checkpoint was selected for {architecture}, seed {seed}."
                )
            model.load_state_dict(best_state)
            model.to(device)
            run_seconds = time.perf_counter() - run_started
            checkpoint_dir = output_dir / "checkpoints" / architecture
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "architecture": architecture,
                    "seed": seed,
                    "best_epoch": best_epoch,
                    "model_state_dict": best_state,
                    "parameter_count": parameter_count,
                    "training_scope": scope,
                },
                checkpoint_dir / f"seed_{seed}.pt",
            )

            for partition, indices in (("train", train_indices), ("validation", validation_indices)):
                predictions = _predict(model, inputs, indices, device)
                selected_targets = targets[indices]
                selected_rows = [rows[index] for index in indices]
                per_cell, macro = _prediction_metrics_by_cell(
                    selected_rows,
                    selected_targets,
                    predictions,
                    architecture=architecture,
                    seed=seed,
                )
                cell_metrics.extend(per_cell)
                cell_macro = macro[partition]
                metrics.append(
                    {
                        "architecture": architecture,
                        "seed": seed,
                        "partition": partition,
                        **_metric_values(
                            selected_targets,
                            predictions,
                            physical_cell_count=cell_macro["cells"],
                        ),
                        "macro_cell_MAE": cell_macro["macro_MAE"],
                        "macro_cell_RMSE": cell_macro["macro_RMSE"],
                        "macro_cell_R2": cell_macro["macro_R2"],
                        "macro_cell_R2_cells": cell_macro["macro_R2_cells"],
                        "parameter_count": parameter_count,
                        "best_epoch": best_epoch,
                        "runtime_seconds": run_seconds,
                    }
                )
                for local_index, row in enumerate(selected_rows):
                    prediction_rows.append(
                        {
                            "architecture": architecture,
                            "seed": seed,
                            "dataset_part": row["dataset_part"],
                            "physical_cell_id": row["physical_cell_id"],
                            "rpt_index": row["rpt_index"],
                            "partition": partition,
                            "target_soh_percent": float(selected_targets[local_index]),
                            "predicted_soh_percent": float(predictions[local_index]),
                        }
                    )

    _write_csv(output_dir / "run_metrics.csv", metrics)
    _write_csv(output_dir / "per_cell_metrics.csv", cell_metrics)
    _write_csv(output_dir / "predictions.csv", prediction_rows)
    summary = _summarize_runs(metrics, seed_values, started_all)
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_validation_plot(output_dir / "validation_metrics.png", summary["architecture_summary"])
    return summary


def _summarize_partition_seed_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "N": rows[0]["N"] if rows else 0,
        "N_cells": rows[0]["N_cells"] if rows else 0,
    }
    for metric in (
        "MAE",
        "RMSE",
        "R2",
        "macro_cell_MAE",
        "macro_cell_RMSE",
        "macro_cell_R2",
    ):
        values = [
            float(row[metric]) for row in rows if row.get(metric) is not None
        ]
        result[f"{metric}_mean"] = statistics.mean(values) if values else None
        result[f"{metric}_stdev"] = (
            statistics.stdev(values) if len(values) > 1 else 0.0
        )
        result[f"{metric}_seed_count"] = len(values)
    return result


def _summarize_runs(
    metrics: list[dict[str, Any]], seeds: list[int], started_at: float
) -> dict[str, Any]:
    architecture_summary: list[dict[str, Any]] = []
    for architecture in ARCHITECTURE_NAMES:
        train = [
            row for row in metrics
            if row["architecture"] == architecture and row["partition"] == "train"
        ]
        validation = [
            row for row in metrics
            if row["architecture"] == architecture and row["partition"] == "validation"
        ]
        train_summary = _summarize_partition_seed_metrics(train)
        validation_summary = _summarize_partition_seed_metrics(validation)
        architecture_summary.append(
            {
                "architecture": architecture,
                "seed_count": len(validation),
                "N": validation_summary["N"],
                "N_cells": validation_summary["N_cells"],
                **{
                    f"train_{name}": value
                    for name, value in train_summary.items()
                },
                **{
                    f"validation_{name}": value
                    for name, value in validation_summary.items()
                },
            }
        )
    selected = min(
        architecture_summary,
        key=lambda row: (row["validation_MAE_mean"], row["validation_RMSE_mean"]),
    )
    return {
        "schema_version": 1,
        "training_authorized": True,
        "external_evaluation_performed": False,
        "seeds": seeds,
        "training_wall_seconds": time.perf_counter() - started_at,
        "architecture_summary": architecture_summary,
        "validation_selected_architecture": selected["architecture"],
        "seed_aggregation_rule": (
            "arithmetic mean and sample standard deviation across fixed seeds; "
            "undefined R2 values excluded and reported with their seed count"
        ),
        "selection_rule": "lowest mean Validation overall MAE across fixed seeds; RMSE tie-break",
    }


def _write_validation_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "Training completed but the required validation-plot dependency is unavailable."
        ) from error
    names = [row["architecture"] for row in rows]
    metrics = (
        ("validation_MAE_mean", "Validation MAE (percentage points)"),
        ("validation_RMSE_mean", "Validation RMSE (percentage points)"),
        ("validation_R2_mean", "Validation R2"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (key, title) in zip(axes, metrics):
        values = [row[key] for row in rows]
        axis.bar(names, values)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train the authorized BaFuse architecture matrix; fails closed without exact scope approval."
    )
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--authorization-file", type=Path)
    args = parser.parse_args(argv)
    result = run_training(args.config, args.authorization_file)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
