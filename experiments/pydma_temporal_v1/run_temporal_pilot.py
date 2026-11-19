from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pydma import DMAAnalyzer, DMAConfig
from pydma.analysis.degradation import calculate_degradation_modes
from pydma.core.objectives import fit_ocv
from pydma.core.optimizer import DMAOptimizer
from pydma.utils.roi import build_roi_mask


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bafuse.data.pydma_stage2 import _load_references
from bafuse.data.pydma_stage2 import _read_verified_curve
from bafuse.labeling.torch_pydma import TorchOCVEvaluator


SPLIT_FILE = ROOT / "data/manifests/experiment_manifests/part1_locked_split/split_manifest.csv"
RPT_MAP_FILE = ROOT / "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv"
BOL_FILE = ROOT / "reports/bafuse_stage_validation_v2/stage1_bol_reference_audit.csv"
CELLS = ("G1-Cell20", "G2-Cell8")
RPTS = (4, 5, 6)
SEEDS = (211, 212, 213, 214, 215)
LAMBDAS = (0.01, 0.1, 1.0)
METRICS = ("lli", "lam_anode", "lam_cathode")
MODEL_NAMES = ("M0", "M1", "M2", "M3", "M4")
RMSE_THRESHOLD_MV = 15.0
MIN_ACCEPTED_RUNS = 3
MAXITER = 50
POPSIZE = 30
DATA_LENGTH = 500
WEIGHT_OCV = 100.0
FIT_PARAMETER_NAMES = ("alpha_an", "beta_an", "alpha_ca", "beta_ca")
LOWER_BOUNDS = (1.0, -1.0, 1.0, -1.0)
UPPER_BOUNDS = (2.0, 0.0, 2.1, 0.0)
STRICT_MAPPING = {"verified", "verified_from_ordered_raw_blocks"}
OUTPUT_FILES = (
    "training_manifest.csv",
    "bol_fit_attempts.csv",
    "bol_fit_summary.csv",
    "fit_attempts.csv",
    "degradation_trajectories.csv",
    "temporal_fit_summary.csv",
    "parameter_stability.csv",
    "fit_checkpoint.jsonl",
    "temporal_physical_plausibility.md",
    "combined_findings.md",
    "run_manifest.json",
    "figures",
)


@dataclass(frozen=True)
class Curve:
    cell: str
    direction: str
    rpt: int
    source_file: str
    source_reference: str
    capacity: np.ndarray
    voltage: np.ndarray
    audit: dict[str, Any]

    @property
    def capacity_span(self) -> float:
        return float(np.ptp(self.capacity))


@dataclass(frozen=True)
class FitOutcome:
    params: np.ndarray
    rmse: float
    n_accepted_runs: int
    n_total_runs: int
    cost: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _source_stat(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.resolve().relative_to(ROOT.resolve()).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _make_config(direction: str, seed: int) -> DMAConfig:
    return DMAConfig(
        direction=direction,
        data_length=DATA_LENGTH,
        weight_ocv=WEIGHT_OCV,
        weight_dva=0.0,
        weight_ica=0.0,
        speed_preset="fast",
        req_accepted=1,
        max_tries_overall=1,
        rmse_threshold=RMSE_THRESHOLD_MV / 1000.0,
        random_seed=seed,
        print_progress=False,
        allow_first_cycle_inhomogeneity=False,
    )


def _locked_train_cells() -> set[str]:
    splits = pd.read_csv(SPLIT_FILE, dtype=str, keep_default_na=False)
    if splits["physical_cell_id"].duplicated().any():
        raise ValueError("Locked split has duplicate physical-cell rows.")
    selected = splits[
        splits["physical_cell_id"].isin(CELLS)
        & splits["split"].eq("train")
        & splits["status"].eq("locked")
    ]
    if set(selected["physical_cell_id"]) != set(CELLS):
        raise ValueError("Pilot cells are not all locked Train cells.")
    if not selected["all_rpts_same_split"].str.casefold().eq("true").all():
        raise ValueError("Locked split does not keep every RPT of each pilot cell together.")
    return set(selected["physical_cell_id"])


def _load_curves() -> tuple[list[Curve], list[dict[str, Any]], dict[str, Path]]:
    _locked_train_cells()
    mapping = pd.read_csv(RPT_MAP_FILE, dtype=str, keep_default_na=False)
    bol_rows = pd.read_csv(BOL_FILE, dtype=str, keep_default_na=False)
    curves: list[Curve] = []
    manifest: list[dict[str, Any]] = []
    sources: dict[str, Path] = {}

    for cell in CELLS:
        bol_matches = bol_rows[
            bol_rows["physical_cell_id"].eq(cell)
            & bol_rows["dataset_part"].eq("1")
        ]
        if len(bol_matches) != 1:
            raise ValueError(f"{cell}: expected one Part 1 BoL reference row.")
        bol_row = bol_matches.iloc[0]
        if bol_row["bol_verified"].casefold() != "true":
            raise ValueError(f"{cell}: BoL source identity is not verified.")
        bol_path = ROOT / "data/raw/oxford/part1" / bol_row["primary_bol_reference_file"]
        if not bol_path.is_file():
            raise FileNotFoundError(f"{cell}: missing BoL source {bol_path}")
        sources[f"{cell}:BOL"] = bol_path

        for direction, step in (("discharge", 8), ("charge", 10)):
            capacity, voltage, audit = _read_verified_curve(ROOT, bol_path, step, direction)
            curve = Curve(
                cell=cell,
                direction=direction,
                rpt=0,
                source_file=bol_path.resolve().relative_to(ROOT.resolve()).as_posix(),
                source_reference="",
                capacity=capacity,
                voltage=voltage,
                audit=audit,
            )
            curves.append(curve)
            manifest.append(_manifest_row(curve, "BOL", "unique_bol_curve_candidate"))

        for rpt in RPTS:
            rows = mapping[
                mapping["physical_cell_id"].eq(cell)
                & mapping["part"].eq("1")
                & mapping["rpt_index"].eq(str(rpt))
            ]
            if len(rows) != 1:
                raise ValueError(f"{cell} RPT{rpt}: expected one exact mapping, found {len(rows)}.")
            row = rows.iloc[0]
            if row["mapping_status_v2"] not in STRICT_MAPPING:
                raise ValueError(
                    f"{cell} RPT{rpt}: mapping is not strict-verified "
                    f"({row['mapping_status_v2']})."
                )
            source = ROOT / row["source_file"]
            if not source.is_file():
                raise FileNotFoundError(f"{cell} RPT{rpt}: missing source {source}")
            sources[f"{cell}:RPT{rpt}"] = source

            for direction, step in (("discharge", 8), ("charge", 10)):
                if row[f"step{step}_valid_bool"].casefold() != "true":
                    raise ValueError(f"{cell} RPT{rpt}: Step {step} is not valid.")
                reference = row[f"step{step}_segment_ref_exact"]
                if not reference:
                    raise ValueError(f"{cell} RPT{rpt}: missing exact Step {step} boundary.")
                capacity, voltage, audit = _read_verified_curve(
                    ROOT, source, step, direction, reference
                )
                curve = Curve(
                    cell=cell,
                    direction=direction,
                    rpt=rpt,
                    source_file=source.resolve().relative_to(ROOT.resolve()).as_posix(),
                    source_reference=reference,
                    capacity=capacity,
                    voltage=voltage,
                    audit=audit,
                )
                curves.append(curve)
                manifest.append(_manifest_row(curve, row["mapping_status_v2"], row[f"step{step}_status_exact"]))

    return curves, manifest, sources


def _manifest_row(curve: Curve, mapping_status: str, boundary_status: str) -> dict[str, Any]:
    return {
        "part": 1,
        "physical_cell_id": curve.cell,
        "partition": "train",
        "rpt_index": curve.rpt,
        "direction": curve.direction,
        "step": 8 if curve.direction == "discharge" else 10,
        "source_file": curve.source_file,
        "source_segment_reference": curve.source_reference,
        "mapping_status": mapping_status,
        "boundary_status": boundary_status,
        "curve_status": curve.audit["status"],
        "sample_count": curve.audit["sample_count"],
        "capacity_min_native": curve.audit["capacity_min_native"],
        "capacity_max_native": curve.audit["capacity_max_native"],
        "capacity_span_native": curve.audit["capacity_span_native"],
        "voltage_min_v": curve.audit["voltage_min_v"],
        "voltage_max_v": curve.audit["voltage_max_v"],
        "candidate_segments_json": json.dumps(
            _jsonable(curve.audit["candidate_segments"]), sort_keys=True
        ),
    }


def _params(result: Any) -> np.ndarray:
    values = result.params
    if len(values) != 9:
        raise ValueError(f"Unexpected PyDMA parameter vector length {len(values)}.")
    return np.asarray(values, dtype=np.float64)


def _passes_quality_gate(accepted_seed_runs: int) -> bool:
    return accepted_seed_runs >= MIN_ACCEPTED_RUNS


def _degradation(params: np.ndarray, capacity: float, bol_params: np.ndarray, bol_capacity: float) -> dict[str, float]:
    return {
        "lli": float(
            calculate_degradation_modes(
                params,
                capacity,
                bol_params[0] * bol_capacity,
                bol_params[2] * bol_capacity,
                (bol_params[2] + bol_params[3] - bol_params[1]) * bol_capacity,
            ).lli
        ),
        "lam_anode": float(
            calculate_degradation_modes(
                params,
                capacity,
                bol_params[0] * bol_capacity,
                bol_params[2] * bol_capacity,
                (bol_params[2] + bol_params[3] - bol_params[1]) * bol_capacity,
            ).lam_anode
        ),
        "lam_cathode": float(
            calculate_degradation_modes(
                params,
                capacity,
                bol_params[0] * bol_capacity,
                bol_params[2] * bol_capacity,
                (bol_params[2] + bol_params[3] - bol_params[1]) * bol_capacity,
            ).lam_cathode
        ),
    }


def _fit_independent(context: dict[str, Any], seed: int) -> FitOutcome:
    if not torch.cuda.is_available():
        raise RuntimeError("The temporal pilot requires the validated CUDA PyDMA backend.")
    from bafuse.labeling.torch_pydma import use_cuda_optimizer

    config = replace(context["config"], random_seed=seed)
    optimizer = DMAOptimizer(
        config,
        context["objective"],
        context["bounds"],
        rmse_fn=context["rmse"],
    )
    with use_cuda_optimizer():
        result = optimizer.run(
            req_accepted=1,
            max_tries=1,
            rmse_threshold=RMSE_THRESHOLD_MV / 1000.0,
            x0=context["initial_guess"],
            maxiter=MAXITER,
            popsize=POPSIZE,
            polish=False,
        )
    return FitOutcome(
        params=np.asarray(result.best_params, dtype=np.float64),
        rmse=float(result.best_rmse),
        n_accepted_runs=result.n_accepted,
        n_total_runs=result.n_total,
        cost=float(result.best_cost),
    )


def _bounds_for(config: DMAConfig, capacity: float) -> list[tuple[float, float]]:
    lower, upper = config.get_full_bounds(
        inhom_an_prev=0.0,
        inhom_ca_prev=0.0,
        capa_actual=capacity,
    )
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    lower[6:8] = 0.0
    upper[6:8] = 0.0
    return list(zip(lower.tolist(), upper.tolist()))


def _objective_context(curve: Curve, references: tuple[Any, Any], seed: int) -> dict[str, Any]:
    config = _make_config(curve.direction, seed)
    analyzer = DMAAnalyzer(config=config, anode=references[0], cathode=references[1])
    q, measured, q0, capacity_span = analyzer._prepare_measured_data(
        curve.capacity, curve.voltage
    )
    anode = analyzer._prepare_electrode(references[0])
    cathode = analyzer._prepare_electrode(references[1])
    zeros = np.zeros_like(q)
    dva_mask = build_roi_mask(q, config.roi_dva_min, config.roi_dva_max)
    ica_mask = build_roi_mask(q, config.roi_ica_min, config.roi_ica_max)
    objective = analyzer._create_objective(
        q,
        measured,
        zeros,
        zeros,
        dva_mask,
        ica_mask,
        q0,
        actual_capacity=capacity_span,
        anode=anode,
        cathode=cathode,
        anode_is_blend=False,
        cathode_is_blend=False,
    )
    r_offset_sign, current = analyzer._resistance_offset_terms()
    def rmse(params: np.ndarray) -> float:
        mse = fit_ocv(
            params,
            anode=anode,
            cathode=cathode,
            meas_voltage=measured,
            q=q,
            roi_ocv_min=config.roi_ocv_min,
            roi_ocv_max=config.roi_ocv_max,
            anode_is_blend=False,
            cathode_is_blend=False,
            inhom_anode_offset=config.inhom_anode_offset,
            inhom_cathode_offset=config.inhom_cathode_offset,
            r_offset_sign=r_offset_sign,
            pocv_current_a=current,
        )
        return float(np.sqrt(mse))
    return {
        "curve": curve,
        "config": config,
        "objective": objective,
        "rmse": rmse,
        "capacity_span": capacity_span,
        "bounds": _bounds_for(config, capacity_span),
        "initial_guess": config.get_initial_guess(
            inhom_an_prev=0.0,
            inhom_ca_prev=0.0,
            capa_actual=capacity_span,
        ),
    }


def _scale_penalties(rows: list[dict[str, Any]]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for metric in METRICS:
        values = np.asarray(
            [float(row[metric]) for row in rows if np.isfinite(float(row[metric]))],
            dtype=np.float64,
        )
        if not values.size:
            raise ValueError(f"Cannot derive training-only penalty scale for {metric}.")
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        scales[metric] = max(1.4826 * mad, 0.01)
    return scales


def _temporal_penalties(
    params: np.ndarray,
    capacities: list[float],
    bol_params: np.ndarray,
    bol_capacity: float,
    scales: dict[str, float],
) -> tuple[float, float]:
    values = {
        metric: np.asarray(
            [
                _degradation(params[index], capacities[index], bol_params, bol_capacity)[metric]
                for index in range(len(capacities))
            ],
            dtype=np.float64,
        )
        for metric in METRICS
    }
    smoothness = float(
        np.mean(
            [
                np.mean(np.square(np.diff(series, n=2) / scales[metric]))
                for metric, series in values.items()
            ]
        )
    )
    monotonicity = float(
        np.mean(
            [
                np.mean(np.square(np.minimum(np.diff(series), 0.0) / scales[metric]))
                for metric, series in values.items()
            ]
        )
    )
    return smoothness, monotonicity


def _run_joint_fit(
    cell_contexts: list[dict[str, Any]],
    seed: int,
    model: str,
    penalty_lambda: float,
    bol_params: np.ndarray,
    bol_capacity: float,
    scales: dict[str, float],
) -> tuple[np.ndarray, list[float], bool, int, int, float, float, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("The temporal pilot requires the validated CUDA PyDMA backend.")
    contexts = sorted(cell_contexts, key=lambda context: context["curve"].rpt)
    bounds = [bound for context in contexts for bound in context["bounds"]]
    x0 = np.concatenate([context["initial_guess"] for context in contexts])
    capacity_spans = [context["capacity_span"] for context in contexts]
    device = torch.device("cuda:0")
    evaluators = [
        TorchOCVEvaluator(context["objective"], device) for context in contexts
    ]
    scale_tensor = torch.tensor(
        [scales[metric] for metric in METRICS], dtype=torch.float64, device=device
    )
    bol = torch.tensor(bol_params, dtype=torch.float64, device=device)
    capacities = torch.tensor(capacity_spans, dtype=torch.float64, device=device)
    initial_anode = bol[0] * bol_capacity
    initial_cathode = bol[2] * bol_capacity
    initial_inventory = (bol[2] + bol[3] - bol[1]) * bol_capacity

    def split_params(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float64).reshape(len(contexts), 9)

    def degradation_tensor(parameters: torch.Tensor) -> torch.Tensor:
        anode = parameters[:, :, 0] * capacities
        cathode = parameters[:, :, 2] * capacities
        inventory = (
            parameters[:, :, 2] + parameters[:, :, 3] - parameters[:, :, 1]
        ) * capacities
        return torch.stack(
            (
                (initial_inventory - inventory) / initial_inventory,
                (initial_anode - anode) / initial_anode,
                (initial_cathode - cathode) / initial_cathode,
            ),
            dim=2,
        )

    def objective(candidates: np.ndarray) -> np.ndarray | float:
        array = np.asarray(candidates, dtype=np.float64)
        scalar = array.ndim == 1
        if scalar:
            array = array[:, None]
        batch = torch.as_tensor(array.T, dtype=torch.float64, device=device).reshape(
            -1, len(contexts), 9
        )
        data_cost = torch.stack(
            [
                evaluators[index].evaluate(batch[:, index, :])[0]
                for index in range(len(contexts))
            ],
            dim=1,
        ).mean(dim=1) / (WEIGHT_OCV * (RMSE_THRESHOLD_MV / 1000.0) ** 2)
        degradation = degradation_tensor(batch)
        deltas = torch.diff(degradation, dim=1)
        second_deltas = torch.diff(degradation, n=2, dim=1)
        smoothness = torch.square(second_deltas / scale_tensor).mean(dim=(1, 2))
        monotonicity = torch.square(torch.relu(-deltas / scale_tensor)).mean(dim=(1, 2))
        if model in {"M2", "M4"}:
            data_cost = data_cost + penalty_lambda * smoothness
        if model in {"M3", "M4"}:
            data_cost = data_cost + penalty_lambda * monotonicity
        values = data_cost.detach().cpu().numpy()
        return float(values[0]) if scalar else values

    def max_rmse(vector: np.ndarray) -> float:
        parameters = split_params(vector)
        return max(
            float(
                evaluators[index].evaluate(
                    torch.as_tensor(parameters[index][None, :], dtype=torch.float64, device=device)
                )[1][0].item()
            )
            for index in range(len(contexts))
        )

    config = replace(contexts[0]["config"], random_seed=seed)
    optimizer = DMAOptimizer(config, objective, bounds, rmse_fn=max_rmse)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="differential_evolution: the 'vectorized' keyword has overridden updating='immediate' to updating='deferred'",
            category=UserWarning,
        )
        result = optimizer.run(
            req_accepted=1,
            max_tries=1,
            rmse_threshold=RMSE_THRESHOLD_MV / 1000.0,
            x0=x0,
            maxiter=MAXITER,
            popsize=POPSIZE,
            polish=False,
            vectorized=True,
        )
    best = split_params(result.best_params)
    rmse_values = [context["rmse"](best[index]) * 1000 for index, context in enumerate(contexts)]
    smoothness, monotonicity = _temporal_penalties(
        best, capacity_spans, bol_params, bol_capacity, scales
    )
    accepted = bool(result.n_accepted == 1 and max(rmse_values) < RMSE_THRESHOLD_MV)
    return (
        best,
        rmse_values,
        accepted,
        result.n_accepted,
        result.n_total,
        float(result.best_cost),
        smoothness,
        monotonicity,
    )


def _attempt_row(
    curve: Curve,
    model: str,
    penalty_lambda: float,
    seed: int,
    params: np.ndarray,
    rmse_mv: float,
    bol_rmse_mv: float,
    accepted_run_count: int,
    total_optimizer_attempts: int,
    baseline_accepted: bool,
    fit_accepted: bool,
    optimization_cost: float,
    smoothness_penalty: float,
    monotonicity_penalty: float,
    reused: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "physical_cell_id": curve.cell,
        "partition": "train",
        "part": 1,
        "rpt_index": curve.rpt,
        "direction": curve.direction,
        "model": model,
        "penalty_lambda": penalty_lambda,
        "seed": seed,
        "source_file": curve.source_file,
        "source_segment_reference": curve.source_reference,
        "fit_rmse_mV": rmse_mv,
        "rmse_threshold_mV": RMSE_THRESHOLD_MV,
        "accepted_optimizer_runs": accepted_run_count,
        "total_optimizer_attempts": total_optimizer_attempts,
        "fit_gate_pass": fit_accepted,
        "bol_rmse_mV": bol_rmse_mv,
        "bol_reference_gate_pass": baseline_accepted,
        "degradation_reference_status": "accepted_bol_fit" if baseline_accepted else "raw_estimate_bol_fit_failed_gate",
        "optimization_cost": optimization_cost,
        "normalized_smoothness_penalty": smoothness_penalty,
        "normalized_monotonicity_penalty": monotonicity_penalty,
        "reused_m0_solution": reused,
    }
    row.update(
        {name: float(params[index]) for index, name in enumerate(FIT_PARAMETER_NAMES)}
    )
    row.update(_bound_flags(params))
    return row


def _bound_flags(params: np.ndarray) -> dict[str, Any]:
    flags: dict[str, Any] = {}
    for index, name in enumerate(FIT_PARAMETER_NAMES):
        low, high = LOWER_BOUNDS[index], UPPER_BOUNDS[index]
        value = float(params[index])
        flags[f"{name}_at_bound"] = bool(
            abs(value - low) <= 1e-6 or abs(value - high) <= 1e-6
        )
        flags[f"{name}_normalized_bound_distance"] = float(
            min((value - low) / (high - low), (high - value) / (high - low))
        )
    return flags


def _append_checkpoint(out_dir: Path, record: dict[str, Any]) -> None:
    with (out_dir / "fit_checkpoint.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(record), sort_keys=True, allow_nan=False) + "\n")


def _save_figure(
    out_dir: Path,
    rows: list[dict[str, Any]],
    cell: str,
    direction: str,
) -> None:
    selected = [
        row
        for row in rows
        if row["physical_cell_id"] == cell and row["direction"] == direction
    ]
    if not selected:
        return
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for axis, metric in zip(axes, METRICS):
        groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for row in selected:
            groups.setdefault((row["model"], float(row["penalty_lambda"])), []).append(row)
        for (model, strength), values in sorted(groups.items()):
            grouped: dict[int, list[float]] = {}
            for row in values:
                grouped.setdefault(int(row["rpt_index"]), []).append(
                    float(row[f"{metric}_percent"])
                )
            xs = sorted(grouped)
            means = [float(np.mean(grouped[x])) for x in xs]
            stds = [
                float(np.std(grouped[x], ddof=1)) if len(grouped[x]) > 1 else 0.0
                for x in xs
            ]
            label = model if model in {"M0", "M1"} else f"{model}, λ={strength:g}"
            axis.errorbar(xs, means, yerr=stds, marker="o", capsize=2, label=label)
        axis.set_title(metric.replace("_", " "))
        axis.set_xlabel("Verified RPT order")
        axis.set_ylabel("Estimated loss (%)")
        axis.set_xticks(RPTS)
        axis.grid(True, alpha=0.25)
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle(f"{cell} — {direction} temporal pilot (Train only)")
    fig.tight_layout()
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{cell}_{direction}_trajectories.png", dpi=160)
    plt.close(fig)


def _write_reports(out_dir: Path, attempts: pd.DataFrame, scales: dict[str, float]) -> None:
    accepted = attempts[attempts["fit_gate_pass"]]
    gate_groups = attempts.groupby(
        ["model", "penalty_lambda", "physical_cell_id", "direction", "rpt_index"],
        dropna=False,
    )["fit_gate_pass"].sum()
    gate_pass_count = int(gate_groups.ge(MIN_ACCEPTED_RUNS).sum())
    report_lines = [
        "# Temporal PyDMA Physical-Plausibility Pilot",
        "",
        "This is an isolated Train-only experiment. It does not modify production PyDMA outputs,",
        "labels, manifests, the locked split, the frozen SoH checkpoint, or the dashboard.",
        "",
        "## Design",
        "",
        "- Cells: G1-Cell20 and G2-Cell8, both locked Train cells.",
        "- Part 1 RPT4–RPT6; Step 8 discharge and Step 10 charge, using verified exact boundaries.",
        "- M0: independent PyDMA fits. M1: the identical M0 solutions, explicitly a separable no-penalty control.",
        "- M2: smoothness penalty; M3: non-decreasing-degradation penalty; M4: both penalties.",
        "- Penalties act on reference-relative LLI/LAM from the package degradation equations.",
        "- Scale for each degradation metric is `max(1.4826 × Train-only MAD, 0.01)` as a fraction.",
        "- Lambda grid: 0.01, 0.1, 1.0. No coefficient is selected using Validation or External data.",
        f"- Common bounded optimizer budget: maxiter={MAXITER}, population={POPSIZE}, polish disabled, seeds={','.join(map(str, SEEDS))}.",
        f"- Each seed is one optimizer run; a seed passes when RMSE < {RMSE_THRESHOLD_MV:g} mV and its BoL reference also passes.",
        f"- The cohort gate requires at least {MIN_ACCEPTED_RUNS} of {len(SEEDS)} seed runs to pass for a given curve/model.",
        "",
        "## Results",
        "",
        f"- Fit attempts recorded: {len(attempts)}; individual fits passing the RMSE gate: {len(accepted)}.",
        f"- Accepted fit rate: {len(accepted) / len(attempts):.1%}." if len(attempts) else "- No fit attempts recorded.",
        f"- Model/cell/direction/RPT strata passing the three-of-five gate: {gate_pass_count}/{len(gate_groups)}.",
        "- Counts include repeated RPT estimates for each seed/model and must not be interpreted as independent cells.",
        "- Per-model/seed metrics and all attempted parameters are in `fit_attempts.csv`.",
        "- Degradation trajectories and per-seed summaries are in their corresponding CSVs.",
        "",
        "## Training-only normalization scales",
        "",
    ]
    report_lines.extend(f"- {metric}: {scale:.8g} fractional loss" for metric, scale in scales.items())
    report_lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "- The coefficient grid is exploratory; no best lambda is declared.",
            "- Part 3 and all Validation/Test cells were excluded from fitting and coefficient comparisons.",
            "- Negative estimates are preserved. No clipping, absolute values, or positivity-based selection is used.",
            "- PyDMA LLI/LAM are model-derived diagnostics, not independent ground truth.",
            "- The small cohort, one search per seed, and disabled polishing limit physical conclusions; a lower objective is not proof of better physical identifiability.",
            "- The table uses native `Amphr` capacity spans; absolute capacity units are not asserted.",
            "",
        ]
    )
    (out_dir / "temporal_physical_plausibility.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    metric_lines = [
        "# Combined Findings — Temporal PyDMA Pilot",
        "",
        "The pilot estimates direction-specific degradation trajectories for two locked Train cells.",
        "M1 duplicates M0 and is not an independent fitted model. M2–M4 use the listed penalty components",
        "and lambda grid; all fit attempts, including those failing the RMSE gate, are retained.",
        "",
        "No penalty coefficient is selected from non-Train data. The pilot does not establish that temporal",
        "regularization improves physical validity or generalization. Compare fit RMSE, parameter bounds,",
        "seed dispersion, and negative estimates together; do not use positivity as a selection criterion.",
        "",
        "See `temporal_physical_plausibility.md`, `temporal_fit_summary.csv`, `parameter_stability.csv`,",
        "and the unsmoothed trajectory figures for quantitative results.",
        "",
    ]
    (out_dir / "combined_findings.md").write_text("\n".join(metric_lines), encoding="utf-8")


def _summarize(attempts: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    attempt_frame = pd.DataFrame(attempts)
    trajectory_rows: list[dict[str, Any]] = []
    for row in attempt_frame.to_dict("records"):
        curve = _curve_from_attempt(row)
        bol = row["_bol_params"]
        degradation = _degradation(
            np.asarray([row[name] for name in FIT_PARAMETER_NAMES] + [0.0] * 5),
            curve.capacity_span,
            bol,
            float(row["_bol_capacity"]),
        )
        trajectory_rows.append(
            {
                key: row[key]
                for key in ("physical_cell_id", "direction", "rpt_index", "model", "penalty_lambda", "seed", "fit_rmse_mV", "fit_gate_pass", "bol_reference_gate_pass")
            }
            | {f"{metric}_percent": 100.0 * value for metric, value in degradation.items()}
        )
    trajectories = pd.DataFrame(trajectory_rows)
    group_columns = ["model", "penalty_lambda", "physical_cell_id", "direction", "rpt_index"]
    summary = (
        trajectories.groupby(group_columns, dropna=False)
        .agg(
            observation_count=("seed", "count"),
            accepted_seed_count=("fit_gate_pass", "sum"),
            mean_fit_rmse_mV=("fit_rmse_mV", "mean"),
            std_fit_rmse_mV=("fit_rmse_mV", "std"),
            **{
                f"{metric}_mean_percent": (f"{metric}_percent", "mean")
                for metric in METRICS
            },
            **{
                f"{metric}_std_percent": (f"{metric}_percent", "std")
                for metric in METRICS
            },
        )
        .reset_index()
    )
    summary["quality_gate_pass"] = summary["accepted_seed_count"].map(
        _passes_quality_gate
    )
    parameter_rows = []
    for keys, group in attempt_frame.groupby(["model", "penalty_lambda", "physical_cell_id", "direction", "rpt_index"], dropna=False):
        record = dict(zip(["model", "penalty_lambda", "physical_cell_id", "direction", "rpt_index"], keys))
        record["fit_count"] = len(group)
        record["accepted_fit_count"] = int(group["fit_gate_pass"].sum())
        for name in FIT_PARAMETER_NAMES:
            record[f"{name}_mean"] = float(group[name].mean())
            record[f"{name}_std"] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.0
            record[f"{name}_at_bound_count"] = int(group[f"{name}_at_bound"].sum())
        parameter_rows.append(record)
    return trajectories, summary, pd.DataFrame(parameter_rows)


def _curve_from_attempt(row: dict[str, Any]) -> Curve:
    return CURVE_LOOKUP[(row["physical_cell_id"], row["direction"], int(row["rpt_index"]))]


CURVE_LOOKUP: dict[tuple[str, str, int], Curve] = {}


def run(run_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run-id may contain only letters, digits, underscore, and hyphen.")
    out_dir = EXPERIMENT_DIR / "runs" / run_id
    if out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing pilot run: {out_dir}")
    out_dir.mkdir(parents=True)

    curves, manifest_rows, sources = _load_curves()
    CURVE_LOOKUP.clear()
    for curve in curves:
        CURVE_LOOKUP[(curve.cell, curve.direction, curve.rpt)] = curve
    pd.DataFrame(manifest_rows).to_csv(out_dir / "training_manifest.csv", index=False)
    curve_index = {(curve.cell, curve.direction, curve.rpt): curve for curve in curves}

    references: dict[str, tuple[Any, Any]] = {}
    reference_metadata: dict[str, dict[str, Any]] = {}
    for direction in ("discharge", "charge"):
        anode, cathode, metadata = _load_references(ROOT, direction)
        references[direction] = (anode, cathode)
        reference_metadata[direction] = metadata

    start = time.perf_counter()
    context_index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for curve in curves:
        key = (curve.cell, curve.direction, curve.rpt)
        context_index[key] = _objective_context(
            curve, references[curve.direction], SEEDS[0]
        )
        print(
            f"Prepared {curve.cell} {curve.direction} "
            f"{'BOL' if curve.rpt == 0 else f'RPT{curve.rpt}'}",
            flush=True,
        )

    bol_records: dict[tuple[str, str, int], dict[str, Any]] = {}
    attempts: list[dict[str, Any]] = []
    bol_attempts: list[dict[str, Any]] = []
    for cell in CELLS:
        for direction in ("discharge", "charge"):
            bol_curve = curve_index[(cell, direction, 0)]
            for seed in SEEDS:
                result = _fit_independent(context_index[(cell, direction, 0)], seed)
                key = (cell, direction, seed)
                bol_records[key] = {
                    "params": _params(result),
                    "capacity": bol_curve.capacity_span,
                    "rmse_mV": float(result.rmse * 1000),
                    "accepted": bool(result.rmse * 1000 < RMSE_THRESHOLD_MV),
                }
                bol_row: dict[str, Any] = {
                    "physical_cell_id": cell,
                    "partition": "train",
                    "part": 1,
                    "observation_type": "BOL_REFERENCE",
                    "direction": direction,
                    "seed": seed,
                    "source_file": bol_curve.source_file,
                    "fit_rmse_mV": float(result.rmse * 1000),
                    "rmse_threshold_mV": RMSE_THRESHOLD_MV,
                    "accepted_optimizer_runs": result.n_accepted_runs,
                    "total_optimizer_attempts": result.n_total_runs,
                    "fit_gate_pass": bool(result.rmse * 1000 < RMSE_THRESHOLD_MV),
                }
                bol_row.update(
                    {
                        name: float(bol_records[key]["params"][index])
                        for index, name in enumerate(FIT_PARAMETER_NAMES)
                    }
                )
                bol_row.update(_bound_flags(bol_records[key]["params"]))
                bol_attempts.append(bol_row)
                _append_checkpoint(
                    out_dir,
                    {
                        "attempt_type": "BOL_REFERENCE",
                        **bol_row,
                    },
                )
                print(
                    f"Completed BoL {cell} {direction} seed={seed} "
                    f"RMSE={result.rmse * 1000:.2f}mV",
                    flush=True,
                )

    m0_rows: list[dict[str, Any]] = []
    for cell in CELLS:
        for direction in ("discharge", "charge"):
            for rpt in RPTS:
                curve = curve_index[(cell, direction, rpt)]
                for seed in SEEDS:
                    bol = bol_records[(cell, direction, seed)]
                    result = _fit_independent(
                        context_index[(cell, direction, rpt)], seed
                    )
                    params = _params(result)
                    degradation = _degradation(
                        params, curve.capacity_span, bol["params"], bol["capacity"]
                    )
                    row = _attempt_row(
                        curve,
                        "M0",
                        0.0,
                        seed,
                        params,
                        float(result.rmse * 1000),
                        bol["rmse_mV"],
                        result.n_accepted_runs,
                        result.n_total_runs,
                        bol["accepted"],
                        bool(
                            result.rmse * 1000 < RMSE_THRESHOLD_MV
                            and bol["accepted"]
                        ),
                        float(result.rmse**2 * WEIGHT_OCV),
                        0.0,
                        0.0,
                        False,
                    )
                    row.update(degradation)
                    row["_bol_params"] = bol["params"]
                    row["_bol_capacity"] = bol["capacity"]
                    m0_rows.append(row)
                    _append_checkpoint(
                        out_dir,
                        {
                            "attempt_type": "RPT_FIT",
                            **row,
                        },
                    )
                    print(
                        f"Completed M0 {cell} {direction} RPT{rpt} seed={seed} "
                        f"RMSE={result.rmse * 1000:.2f}mV",
                        flush=True,
                    )
    scales = _scale_penalties(m0_rows)

    for row in m0_rows:
        attempts.append(row)
        copied = dict(row)
        copied["model"] = "M1"
        copied["reused_m0_solution"] = True
        attempts.append(copied)
        _append_checkpoint(
            out_dir,
            {
                "attempt_type": "RPT_FIT",
                **copied,
            },
        )

    for cell in CELLS:
        for direction in ("discharge", "charge"):
            target_contexts = [
                context_index[(cell, direction, rpt)] for rpt in RPTS
            ]
            target_curves = [context["curve"] for context in target_contexts]
            for seed in SEEDS:
                bol = bol_records[(cell, direction, seed)]
                for model in ("M2", "M3", "M4"):
                    for strength in LAMBDAS:
                        (
                            parameters,
                            rmses,
                            accepted,
                            accepted_runs,
                            total_attempts,
                            cost,
                            smoothness,
                            monotonicity,
                        ) = _run_joint_fit(
                            target_contexts,
                            seed,
                            model,
                            strength,
                            bol["params"],
                            bol["capacity"],
                            scales,
                        )
                        for index, curve in enumerate(target_curves):
                            degradation = _degradation(
                                parameters[index],
                                curve.capacity_span,
                                bol["params"],
                                bol["capacity"],
                            )
                            row = _attempt_row(
                                curve,
                                model,
                                strength,
                                seed,
                                parameters[index],
                                rmses[index],
                                bol["rmse_mV"],
                                accepted_runs,
                                total_attempts,
                                bol["accepted"],
                                bool(
                                    accepted
                                    and rmses[index] < RMSE_THRESHOLD_MV
                                    and bol["accepted"]
                                ),
                                cost,
                                smoothness,
                                monotonicity,
                                False,
                            )
                            row["joint_seed_gate_pass"] = accepted
                            row.update(degradation)
                            row["_bol_params"] = bol["params"]
                            row["_bol_capacity"] = bol["capacity"]
                            attempts.append(row)
                            _append_checkpoint(
                                out_dir,
                                {
                                    "attempt_type": "RPT_FIT",
                                    **row,
                                },
                            )
                print(
                    f"Completed {cell} {direction} seed={seed}; "
                    f"elapsed={time.perf_counter() - start:.1f}s",
                    flush=True,
                )

    internal_attempts = pd.DataFrame(attempts)
    trajectories, summary, parameter_stability = _summarize(attempts)
    public_attempts = internal_attempts.drop(columns=["_bol_params", "_bol_capacity"])
    pd.DataFrame(bol_attempts).to_csv(out_dir / "bol_fit_attempts.csv", index=False)
    bol_summary = (
        pd.DataFrame(bol_attempts)
        .groupby(["physical_cell_id", "direction"], dropna=False)
        .agg(
            seed_count=("seed", "count"),
            accepted_seed_count=("fit_gate_pass", "sum"),
            mean_fit_rmse_mV=("fit_rmse_mV", "mean"),
            std_fit_rmse_mV=("fit_rmse_mV", "std"),
        )
        .reset_index()
    )
    bol_summary["quality_gate_pass"] = bol_summary["accepted_seed_count"].map(
        _passes_quality_gate
    )
    bol_summary.to_csv(out_dir / "bol_fit_summary.csv", index=False)
    public_attempts.to_csv(out_dir / "fit_attempts.csv", index=False)
    trajectories.to_csv(out_dir / "degradation_trajectories.csv", index=False)
    summary.to_csv(out_dir / "temporal_fit_summary.csv", index=False)
    parameter_stability.to_csv(out_dir / "parameter_stability.csv", index=False)
    for cell in CELLS:
        for direction in ("discharge", "charge"):
            _save_figure(out_dir, trajectories.to_dict("records"), cell, direction)
    _write_reports(out_dir, public_attempts, scales)

    manifest = {
        "run_id": run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "pydma_temporal_v1",
        "cells": list(CELLS),
        "partition": "locked_train_only",
        "parts": [1],
        "rpts": list(RPTS),
        "directions": ["discharge", "charge"],
        "seeds": list(SEEDS),
        "models": {
            "M0": "independent PyDMA fits",
            "M1": "identical reused M0 fits; separable no-penalty control",
            "M2": "second-difference smoothness penalty",
            "M3": "non-decreasing degradation hinge penalty",
            "M4": "both temporal penalties",
        },
        "lambda_grid": list(LAMBDAS),
        "penalty_scales_training_only": scales,
        "penalty_normalization": "1.4826*MAD over M0 Train-only degradation fractions, floor 0.01",
        "data_objective_normalization": f"mean PyDMA weighted objective divided by {WEIGHT_OCV:g}*({RMSE_THRESHOLD_MV:g}mV)^2",
        "rmse_threshold_mV": RMSE_THRESHOLD_MV,
        "maxiter": MAXITER,
        "population": POPSIZE,
        "polish": False,
        "optimizer_backend": "PyDMA-compatible CUDA batched OCV evaluator on cuda:0",
        "data_length": DATA_LENGTH,
        "weight_ocv": WEIGHT_OCV,
        "minimum_accepted_seed_runs": MIN_ACCEPTED_RUNS,
        "fit_gate": "seed pass requires OCV RMSE < 15 mV and passing same-seed BoL; curve/model cohort requires at least 3 of 5 seed runs",
        "input_files": [_source_stat(path) for path in [SPLIT_FILE, RPT_MAP_FILE, BOL_FILE, *sources.values()]],
        "half_cell_references": reference_metadata,
        "outputs": list(OUTPUT_FILES),
        "elapsed_seconds": time.perf_counter() - start,
    }
    (out_dir / "run_manifest.json").write_text(
        json.dumps(_jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="pilot_001")
    args = parser.parse_args()
    print(run(args.run_id))


if __name__ == "__main__":
    main()
