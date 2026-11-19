from __future__ import annotations

import argparse
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pydma import DMAAnalyzer, DMAConfig
from pydma.core import analyzer as analyzer_module
from pydma.core.optimizer import DMAOptimizer, MultiRunResult

from bafuse.data.eol_coverage_v1 import _read_csv, _read_pocv_components
from bafuse.data.pydma_stage2 import _load_references
from bafuse.labeling.pydma_policy import pydma_fit_passes_gate

SEEDS = (211, 212, 213, 214, 215)
CELLS = ("G1-Cell20", "G3-Cell14")
RMSE_THRESHOLD_MV = 15.0
PARAMETERS = ("alpha_an", "beta_an", "alpha_ca", "beta_ca")
LOWER_BOUNDS = {"alpha_an": 1.0, "beta_an": -1.0, "alpha_ca": 1.0, "beta_ca": -1.0}
UPPER_BOUNDS = {"alpha_an": 2.0, "beta_an": 0.0, "alpha_ca": 2.1, "beta_ca": 0.0}
OUTPUT_FILES = (
    "negative_fit_inventory.csv",
    "eol_seed_fits.csv",
    "eol_multiseed_comparison.csv",
    "optimizer_attempts.csv",
    "parameter_bound_analysis.csv",
    "charge_discharge_comparison.csv",
    "run_manifest.json",
    "fit_checkpoint.jsonl",
    "plots/G1-Cell20_discharge_seed_reconstructions.png",
    "plots/G3-Cell14_discharge_seed_reconstructions.png",
)


class _CapturingDMAOptimizer(DMAOptimizer):
    results: list[MultiRunResult] = []
    attempts: list[tuple[int | None, Any]] = []

    def run(self, *args: Any, **kwargs: Any) -> MultiRunResult:
        result = super().run(*args, **kwargs)
        self.results.append(result)
        return result

    def _run_single(self, seed: int | None = None, **kwargs: Any) -> Any:
        result = super()._run_single(seed=seed, **kwargs)
        type(self).attempts.append((seed, result))
        return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_value(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parameters(result: Any) -> dict[str, float]:
    values = result.fitted_params.to_dict()
    return {name: float(values[name]) for name in PARAMETERS}


def _bound_flags(parameters: dict[str, float]) -> dict[str, dict[str, Any]]:
    flags: dict[str, dict[str, Any]] = {}
    for name in PARAMETERS:
        low = LOWER_BOUNDS[name]
        high = UPPER_BOUNDS[name]
        value = float(parameters[name])
        distance = min(value - low, high - value) / (high - low)
        flags[name] = {
            "value": value,
            "lower_bound": low,
            "upper_bound": high,
            "normalized_distance_to_nearest_bound": distance,
            "within_0_1_percent": distance <= 0.001,
            "within_1_percent": distance <= 0.01,
        }
    return flags


def _attempt_parameters(values: np.ndarray) -> dict[str, float]:
    return {name: float(values[index]) for index, name in enumerate(PARAMETERS)}


def _historical_negative_inventory(path: Path) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    estimates = {
        "LLI_percent_vs_BoL": "LLI",
        "LAM_anode_percent_vs_BoL": "Anode LAM",
        "LAM_cathode_percent_vs_BoL": "Cathode LAM",
    }
    rows: list[dict[str, Any]] = []
    for _, record in frame.iterrows():
        for column, metric in estimates.items():
            value = pd.to_numeric(pd.Series([record[column]]), errors="coerce").iloc[0]
            if pd.isna(value) or value >= 0:
                continue
            rmse = float(record["fit_rmse_mV"])
            accepted = int(record["accepted_runs"])
            threshold = float(record["rmse_threshold_mV"])
            rows.append(
                {
                    "part": record["part"],
                    "group": record["group"],
                    "physical_cell_id": record["physical_cell_id"],
                    "rpt": record["rpt"],
                    "direction": record["direction"],
                    "test_step": record["test_step"],
                    "source_file": record["source_file"],
                    "negative_metric": metric,
                    "estimate_percent": float(value),
                    "fit_rmse_mV": rmse,
                    "rmse_threshold_mV": threshold,
                    "accepted_runs": accepted,
                    "total_runs": int(record["total_runs"]),
                    "fit_gate_passed": bool(rmse < threshold and accepted >= 3),
                    "diagnostic_label_status": record["diagnostic_label_status"],
                }
            )
    return rows


def _fit_once(
    *,
    direction: str,
    capacity: np.ndarray,
    voltage: np.ndarray,
    references: tuple[Any, Any, dict[str, Any]],
    seed: int,
    use_cuda: bool,
) -> tuple[Any, MultiRunResult, list[tuple[int | None, Any]], float]:
    _CapturingDMAOptimizer.results = []
    _CapturingDMAOptimizer.attempts = []
    original_optimizer = analyzer_module.DMAOptimizer
    analyzer_module.DMAOptimizer = _CapturingDMAOptimizer
    backend = nullcontext()
    if use_cuda:
        from bafuse.labeling.torch_pydma import use_cuda_optimizer

        backend = use_cuda_optimizer()
    started = time.perf_counter()
    try:
        with backend:
            result = DMAAnalyzer(
                config=DMAConfig(
                    direction=direction,
                    data_length=500,
                    weight_ocv=100.0,
                    weight_dva=0.0,
                    weight_ica=0.0,
                    speed_preset="fast",
                    req_accepted=3,
                    max_tries_overall=10,
                    rmse_threshold=RMSE_THRESHOLD_MV / 1000.0,
                    random_seed=seed,
                    print_progress=False,
                    allow_first_cycle_inhomogeneity=False,
                ),
                anode=references[0],
                cathode=references[1],
            ).analyze(
                measured_capacity=capacity,
                measured_voltage=voltage,
                maxiter=50,
                popsize=30,
                polish=True,
            )
    finally:
        analyzer_module.DMAOptimizer = original_optimizer
    elapsed = time.perf_counter() - started
    if len(_CapturingDMAOptimizer.results) != 1:
        raise RuntimeError(
            "Expected one captured PyDMA multirun result, got "
            f"{len(_CapturingDMAOptimizer.results)}."
        )
    return (
        result,
        _CapturingDMAOptimizer.results[0],
        list(_CapturingDMAOptimizer.attempts),
        elapsed,
    )


def _fit_record(
    *,
    cell: str,
    role: str,
    seed: int,
    source_file: str,
    capacity: np.ndarray,
    result: Any,
    elapsed: float,
) -> dict[str, Any]:
    parameters = _parameters(result)
    rmse_mV = float(result.rmse * 1000.0)
    gate_passed = pydma_fit_passes_gate(
        rmse_mV=rmse_mV,
        accepted_runs=int(result.n_accepted_runs),
        is_accepted=bool(result.is_accepted),
    )
    return {
        "physical_cell_id": cell,
        "fit_role": role,
        "seed": seed,
        "source_file": source_file,
        "capacity_span_native": float(np.ptp(capacity)),
        "rmse_mV": rmse_mV,
        "accepted_runs": int(result.n_accepted_runs),
        "total_runs": int(result.n_total_runs),
        "rejected_runs": int(result.n_total_runs - result.n_accepted_runs),
        "fit_gate_passed": bool(gate_passed),
        "optimizer_status": result.status,
        "runtime_seconds": elapsed,
        "fitted_parameters_json": _json_value(parameters),
        "parameter_bound_flags_json": _json_value(_bound_flags(parameters)),
        "seed_parameter_std_json": _json_value(
            {
                name: float(value)
                for name, value in zip(PARAMETERS, result.param_std[:4])
            }
        )
        if result.param_std is not None
        else "null",
        "result": result,
    }


def _attempt_records(
    *,
    cell: str,
    role: str,
    seed: int,
    source_file: str,
    attempts: list[tuple[int | None, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for attempt_number, (optimizer_seed, run) in enumerate(attempts, start=1):
        parameters = _attempt_parameters(run.params)
        accepted = run.rmse < RMSE_THRESHOLD_MV / 1000.0
        status = "accepted" if accepted else "rejected"
        rows.append(
            {
                "physical_cell_id": cell,
                "fit_role": role,
                "seed": seed,
                "attempt_number": attempt_number,
                "optimizer_seed": optimizer_seed,
                "source_file": source_file,
                "attempt_status": status,
                "rmse_mV": float(run.rmse * 1000.0),
                "rmse_threshold_mV": RMSE_THRESHOLD_MV,
                "optimizer_success": bool(run.success),
                "n_iterations": int(run.n_iterations),
                "n_function_evals": int(run.n_function_evals),
                "cost": float(run.cost),
                "fitted_parameters_json": _json_value(parameters),
                "parameter_bound_flags_json": _json_value(_bound_flags(parameters)),
            }
        )
    return rows


def _aggregate_seed_results(seed_fits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    metrics = ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
    for cell in CELLS:
        targets = [
            row
            for row in seed_fits
            if row["physical_cell_id"] == cell
        ]
        for metric in metrics:
            values = [float(row[metric]) for row in targets if row[metric] is not None]
            accepted = [
                float(row[metric])
                for row in targets
                if row[metric] is not None
                and row["bol_fit_gate_passed"]
                and row["eol_fit_gate_passed"]
            ]
            signs = {int(np.sign(value)) for value in values}
            params = [
                json.loads(row["eol_parameters_json"])
                for row in targets
            ]
            rows.append(
                {
                    "physical_cell_id": cell,
                    "metric": metric,
                    "seed_count": len(values),
                    "both_fits_gate_passed_seed_count": len(accepted),
                    "mean_percent": float(np.mean(values)) if values else None,
                    "std_percent_population": (
                        float(np.std(values, ddof=0)) if values else None
                    ),
                    "min_percent": float(np.min(values)) if values else None,
                    "max_percent": float(np.max(values)) if values else None,
                    "sign_consistency_all_seeds": (
                        "negative" if signs == {-1} else
                        "positive" if signs == {1} else
                        "zero" if signs == {0} else
                        "mixed" if signs else "unavailable"
                    ),
                    "accepted_pair_mean_percent": (
                        float(np.mean(accepted)) if accepted else None
                    ),
                    "accepted_pair_std_percent_population": (
                        float(np.std(accepted, ddof=0)) if accepted else None
                    ),
                    "seed_rmse_mean_mV": float(
                        np.mean([row["eol_rmse_mV"] for row in targets])
                    ),
                    "seed_rmse_std_mV_population": float(
                        np.std([row["eol_rmse_mV"] for row in targets], ddof=0)
                    ),
                    "alpha_an_mean": float(
                        np.mean([value["alpha_an"] for value in params])
                    ),
                    "alpha_an_std_population": float(
                        np.std([value["alpha_an"] for value in params], ddof=0)
                    ),
                    "alpha_an_within_0_1_percent_of_bound_count": sum(
                        json.loads(row["parameter_bound_flags_json"])["alpha_an"][
                            "within_0_1_percent"
                        ]
                        for row in targets
                    ),
                    "seed_estimates_percent_json": _json_value(
                        [
                            {
                                "seed": row["seed"],
                                "value": row[metric],
                                "bol_gate_passed": row["bol_fit_gate_passed"],
                                "eol_gate_passed": row["eol_fit_gate_passed"],
                            }
                            for row in targets
                        ]
                    ),
                }
            )
    return rows


def _plot_seed_fits(cell: str, fits: list[dict[str, Any]], output: Path) -> None:
    target_fits = [
        row
        for row in fits
        if row["physical_cell_id"] == cell and row["fit_role"] == "eol"
    ]
    if len(target_fits) != len(SEEDS):
        raise ValueError(f"{cell} has {len(target_fits)} target fits; expected {len(SEEDS)}.")
    fig, (ax_fit, ax_residual) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    q = np.asarray(target_fits[0]["plot_arrays"]["measured_q"])
    measured_voltage = np.asarray(target_fits[0]["plot_arrays"]["measured_voltage"])
    ax_fit.plot(q, measured_voltage, color="black", linewidth=1.5, label="Measured")
    palette = plt.get_cmap("viridis")
    for index, row in enumerate(target_fits):
        arrays = row["plot_arrays"]
        reconstructed_q = np.asarray(arrays["reconstructed_q"])
        reconstructed_voltage = np.asarray(arrays["reconstructed_voltage"])
        reconstructed = np.interp(q, reconstructed_q, reconstructed_voltage)
        color = palette(index / max(1, len(target_fits) - 1))
        label = f"seed {row['seed']} ({row['rmse_mV']:.2f} mV)"
        ax_fit.plot(reconstructed_q, reconstructed_voltage, color=color, label=label)
        ax_residual.plot(q, measured_voltage - reconstructed, color=color, label=f"seed {row['seed']}")
    ax_fit.set_ylabel("Voltage (V)")
    ax_fit.set_title(f"{cell} discharge pOCV: best accepted fit per seed")
    ax_fit.legend(fontsize=8, ncol=2)
    ax_fit.grid(alpha=0.25)
    ax_residual.axhline(0.0, color="black", linewidth=0.8)
    ax_residual.set_xlabel("Normalized fitting-axis capacity")
    ax_residual.set_ylabel("Measured - fit (V)")
    ax_residual.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _saved_charge_discharge_comparison(
    root: Path,
    manifest: list[dict[str, Any]],
    target_fits: list[dict[str, Any]],
    baseline_fits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in CELLS:
        checkpoint = next(
            row
            for row in manifest
            if row["physical_cell_id"] == cell and row["part"] == "2"
        )
        for direction in ("charge", "discharge"):
            target = next(
                row
                for row in target_fits
                if row["checkpoint_id"] == checkpoint["checkpoint_id"]
                and row["direction"] == direction
            )
            baseline = next(
                row
                for row in baseline_fits
                if row["physical_cell_id"] == cell and row["direction"] == direction
            )
            references = _load_references(root, direction)[2]
            target_rmse_mV = target.get("rmse_mV") or target.get("fit_rmse_mV")
            if not target_rmse_mV:
                raise ValueError(
                    f"Saved target fit has no RMSE for {cell} {direction}."
                )
            rows.append(
                {
                    "physical_cell_id": cell,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "direction": direction,
                    "target_step": target["step"],
                    "target_file": target["source_file"],
                    "target_rmse_mV": target_rmse_mV,
                    "target_accepted_runs": target["accepted_runs"],
                    "target_total_runs": target["total_runs"],
                    "target_fit_gate_passed": target["fit_gate_passed"],
                    "target_lli_percent": target["lli_percent"],
                    "target_lam_anode_percent": target["lam_anode_percent"],
                    "target_lam_cathode_percent": target["lam_cathode_percent"],
                    "target_parameters_json": target["fitted_parameters_json"],
                    "target_parameter_bounds_json": target["parameter_bounds_json"],
                    "bol_file": baseline["source_file"],
                    "bol_rmse_mV": baseline["rmse_mV"],
                    "bol_accepted_runs": baseline["accepted_runs"],
                    "bol_total_runs": baseline["total_runs"],
                    "bol_fit_gate_passed": baseline["fit_gate_passed"],
                    "bol_parameters_json": baseline["fitted_parameters_json"],
                    "direction_matched_half_cell_anode": references["anode_reference"],
                    "direction_matched_half_cell_cathode": references["cathode_reference"],
                    "capacity_unit_note": "native Amphr; absolute unit unverified",
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/data_audits/pydma_negative_root_cause_v1"),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if tuple(args.seeds) != SEEDS:
        raise ValueError(f"This experiment requires exactly seeds {SEEDS}, got {args.seeds}.")
    existing = [output_dir / name for name in OUTPUT_FILES if (output_dir / name).exists()]
    checkpoint_path = output_dir / "fit_checkpoint.jsonl"
    if args.resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f"Cannot resume without {checkpoint_path}.")
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            "Refusing to overwrite existing investigation outputs; pass --overwrite "
            "for this versioned directory."
        )
    if args.overwrite and not args.resume:
        checkpoint_path.write_text("", encoding="utf-8")
    checkpoint_rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if args.resume:
        with checkpoint_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                entry = json.loads(line)
                key = (
                    entry["physical_cell_id"],
                    entry["fit_role"],
                    int(entry["seed"]),
                )
                if key in checkpoint_rows:
                    raise ValueError(f"Duplicate checkpoint key at line {line_number}: {key}.")
                checkpoint_rows[key] = entry

    historical_path = (
        root
        / "experiments"
        / "results"
        / "data_audits"
        / "pydma_eol_audit_v1"
        / "pydma_negative_estimate_audit.csv"
    )
    eol_dir = root / "experiments" / "results" / "eol_coverage_v1"
    manifest = _read_csv(eol_dir / "eol_checkpoint_manifest.csv")
    saved_target_fits = _read_csv(eol_dir / "eol_pydma_fits.csv")
    saved_baseline_fits = _read_csv(eol_dir / "eol_pydma_baseline_fits.csv")
    references = _load_references(root, "discharge")
    use_cuda = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    print(f"Starting targeted refits; device={device}; seeds={SEEDS}", flush=True)

    seed_fits: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    plot_fit_rows: list[dict[str, Any]] = []
    attempt_rows.extend(
        attempt
        for entry in checkpoint_rows.values()
        for attempt in entry["attempt_rows"]
    )
    for cell in CELLS:
        checkpoint = next(
            row for row in manifest
            if row["physical_cell_id"] == cell and row["part"] == "2"
        )
        if not checkpoint["step8_status"].startswith("verified"):
            raise ValueError(f"Unverified EoL Step 8 curve for {cell}.")
        if not checkpoint["baseline_step8_status"].startswith("verified"):
            raise ValueError(f"Unverified BoL Step 8 curve for {cell}.")
        target_capacity, target_voltage, _ = _read_pocv_components(
            root,
            checkpoint["source_file"],
            checkpoint["step8_components_json"],
            8,
            "discharge",
        )
        bol_capacity, bol_voltage, _ = _read_pocv_components(
            root,
            checkpoint["baseline_source_file"],
            checkpoint["baseline_step8_components_json"],
            8,
            "discharge",
        )
        paired: dict[int, dict[str, dict[str, Any]]] = {}
        for seed in SEEDS:
            for role, capacity, voltage, source_file in (
                ("bol", bol_capacity, bol_voltage, checkpoint["baseline_source_file"]),
                ("eol", target_capacity, target_voltage, checkpoint["source_file"]),
            ):
                key = (cell, role, seed)
                cached = checkpoint_rows.get(key)
                if cached:
                    record = cached["fit_record"]
                    attempt_rows_current = cached["attempt_rows"]
                    plot_arrays = cached.get("plot_arrays")
                    print(f"Using checkpointed {cell} {role} seed={seed}", flush=True)
                else:
                    print(f"Fitting {cell} {role} seed={seed}", flush=True)
                    result, multi, attempts, elapsed = _fit_once(
                        direction="discharge",
                        capacity=capacity,
                        voltage=voltage,
                        references=references,
                        seed=seed,
                        use_cuda=use_cuda,
                    )
                    record = _fit_record(
                        cell=cell,
                        role=role,
                        seed=seed,
                        source_file=source_file,
                        capacity=capacity,
                        result=result,
                        elapsed=elapsed,
                    )
                    attempt_rows_current = _attempt_records(
                        cell=cell,
                        role=role,
                        seed=seed,
                        source_file=source_file,
                        attempts=attempts,
                    )
                    plot_arrays = None
                    if role == "eol":
                        plot_arrays = {
                            "measured_q": np.asarray(result.soc_measured).tolist(),
                            "measured_voltage": np.asarray(result.ocv_measured).tolist(),
                            "reconstructed_q": np.asarray(
                                result.soc_reconstructed
                            ).tolist(),
                            "reconstructed_voltage": np.asarray(
                                result.ocv_reconstructed
                            ).tolist(),
                        }
                    record = {name: value for name, value in record.items() if name != "result"}
                    entry = {
                        "physical_cell_id": cell,
                        "fit_role": role,
                        "seed": seed,
                        "fit_record": record,
                        "attempt_rows": attempt_rows_current,
                        "plot_arrays": plot_arrays,
                    }
                    with checkpoint_path.open("a", encoding="utf-8") as stream:
                        stream.write(_json_value(entry) + "\n")
                        stream.flush()
                    checkpoint_rows[key] = entry
                    attempt_rows.extend(attempt_rows_current)
                paired.setdefault(seed, {})[role] = record
                if role == "eol":
                    record["plot_arrays"] = plot_arrays
                    plot_fit_rows.append(record)
                print(
                    f"Completed {cell} {role} seed={seed}: "
                    f"RMSE={record['rmse_mV']:.3f}mV, "
                    f"accepted={record['accepted_runs']}/{record['total_runs']}, "
                    f"gate={record['fit_gate_passed']}, "
                    f"elapsed={record['runtime_seconds']:.1f}s",
                    flush=True,
                )
        for seed in SEEDS:
            bol = paired[seed]["bol"]
            eol = paired[seed]["eol"]
            bol_params = json.loads(bol["fitted_parameters_json"])
            eol_params = json.loads(eol["fitted_parameters_json"])
            from pydma.utils.results import FittedParams

            comparison = DMAAnalyzer(
                config=DMAConfig(direction="discharge"),
                anode=references[0],
                cathode=references[1],
            ).compare_with_reference(
                reference_params=FittedParams(**bol_params),
                current_params=FittedParams(**eol_params),
                reference_capacity=bol["capacity_span_native"],
                current_capacity=eol["capacity_span_native"],
            )
            seed_fits.append(
                {
                    "physical_cell_id": cell,
                    "seed": seed,
                    "bol_source_file": bol["source_file"],
                    "eol_source_file": eol["source_file"],
                    "bol_rmse_mV": bol["rmse_mV"],
                    "eol_rmse_mV": eol["rmse_mV"],
                    "bol_accepted_runs": bol["accepted_runs"],
                    "eol_accepted_runs": eol["accepted_runs"],
                    "bol_total_runs": bol["total_runs"],
                    "eol_total_runs": eol["total_runs"],
                    "bol_fit_gate_passed": bol["fit_gate_passed"],
                    "eol_fit_gate_passed": eol["fit_gate_passed"],
                    "lli_percent": float(comparison.lli * 100.0),
                    "lam_anode_percent": float(comparison.lam_anode * 100.0),
                    "lam_cathode_percent": float(comparison.lam_cathode * 100.0),
                    "bol_capacity_span_native": bol["capacity_span_native"],
                    "eol_capacity_span_native": eol["capacity_span_native"],
                    "capacity_unit_note": "native Amphr; absolute unit unverified",
                    "bol_parameters_json": bol["fitted_parameters_json"],
                    "eol_parameters_json": eol["fitted_parameters_json"],
                    "parameter_bound_flags_json": eol["parameter_bound_flags_json"],
                    "eol_seed_parameter_std_json": eol["seed_parameter_std_json"],
                }
            )

    output_dir.joinpath("plots").mkdir(exist_ok=True)
    for cell in CELLS:
        _plot_seed_fits(
            cell,
            plot_fit_rows,
            output_dir / "plots" / f"{cell}_discharge_seed_reconstructions.png",
        )

    negative_inventory = _historical_negative_inventory(historical_path)
    comparisons = _saved_charge_discharge_comparison(
        root, manifest, saved_target_fits, saved_baseline_fits
    )
    parameter_rows: list[dict[str, Any]] = []
    for attempt in attempt_rows:
        values = json.loads(attempt["fitted_parameters_json"])
        for parameter, flags in _bound_flags(values).items():
            parameter_rows.append(
                {
                    "physical_cell_id": attempt["physical_cell_id"],
                    "fit_role": attempt["fit_role"],
                    "seed": attempt["seed"],
                    "attempt_number": attempt["attempt_number"],
                    "optimizer_seed": attempt["optimizer_seed"],
                    "attempt_status": attempt["attempt_status"],
                    "rmse_mV": attempt["rmse_mV"],
                    "parameter": parameter,
                    **flags,
                }
            )

    pd.DataFrame(negative_inventory).to_csv(
        output_dir / "negative_fit_inventory.csv", index=False
    )
    pd.DataFrame(seed_fits).to_csv(output_dir / "eol_seed_fits.csv", index=False)
    pd.DataFrame(_aggregate_seed_results(seed_fits)).to_csv(
        output_dir / "eol_multiseed_comparison.csv", index=False
    )
    pd.DataFrame(attempt_rows).to_csv(output_dir / "optimizer_attempts.csv", index=False)
    pd.DataFrame(parameter_rows).to_csv(
        output_dir / "parameter_bound_analysis.csv", index=False
    )
    pd.DataFrame(comparisons).to_csv(
        output_dir / "charge_discharge_comparison.csv", index=False
    )

    raw_files = {
        "historical_negative_fit_inventory": historical_path,
        "eol_manifest": eol_dir / "eol_checkpoint_manifest.csv",
        "eol_target_fits": eol_dir / "eol_pydma_fits.csv",
        "eol_baseline_fits": eol_dir / "eol_pydma_baseline_fits.csv",
    }
    for row in manifest:
        if row["physical_cell_id"] not in CELLS or row["part"] != "2":
            continue
        for key in ("source_file", "baseline_source_file"):
            rel = row[key].replace("/", "\\")
            raw_files[f"{row['physical_cell_id']}_{key}"] = root / rel
    for direction, refs in (
        ("discharge", references[2]),
        ("charge", _load_references(root, "charge")[2]),
    ):
        for label in ("anode_path", "cathode_path"):
            source_key = "anode_reference" if label == "anode_path" else "cathode_reference"
            raw_files[f"{direction}_{label}"] = root / refs[source_key].replace("/", "\\")
    manifest_data = {
        "experiment": "PyDMA negative degradation root-cause targeted multiseed refit",
        "device": device,
        "cuda_available": use_cuda,
        "seeds": list(SEEDS),
        "cells": list(CELLS),
        "rmse_gate_mV": RMSE_THRESHOLD_MV,
        "required_accepted_runs": 3,
        "max_attempts": 10,
        "optimizer": {
            "data_length": 500,
            "weight_ocv": 100.0,
            "weight_dva": 0.0,
            "weight_ica": 0.0,
            "speed_preset": "fast",
            "maxiter": 50,
            "popsize": 30,
            "polish": True,
            "inner_attempt_seed_rule": "configured seed + zero-based attempt index",
        },
        "input_sha256": {
            name: _sha256(path) for name, path in raw_files.items() if path.is_file()
        },
        "output_files": list(OUTPUT_FILES),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest_data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Investigation outputs written to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
