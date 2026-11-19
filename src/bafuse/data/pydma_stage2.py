from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_STAGE1_DIR = Path("reports/bafuse_stage_validation_v1")
_AUDIT_DIR = Path("reports/final_mapping_label_audit")
_DEFAULT_RUN_ID = "stage2_pydma_pilot_v1"
_STRICT_MAPPING = {"verified", "verified_from_ordered_raw_blocks"}
_REF_PATTERN = re.compile(
    r"^(?P<source>.+)#Step(?P<step>\d+):segment(?P<segment>\d+):"
    r"rows(?P<start>\d+)-(?P<end>\d+):sub(?P<subsegment>\d+)$",
    re.IGNORECASE,
)
_PILOT_TARGETS = (
    ("1", "G1-Cell9", "2", "early_negative_high_rmse", 211),
    ("1", "G1-Cell9", "8", "middle_part1", 211),
    ("2", "G1-Cell9", "14", "late_part2_continuation", 211),
)
_SEEDS = (211, 212)
_NEGATIVE_AUDIT_COLUMNS = [
    "dataset_part",
    "group",
    "physical_cell_id",
    "rpt_index",
    "role",
    "direction",
    "step",
    "source_file",
    "source_segment_ref",
    "baseline_reference_id",
    "baseline_fit_accepted",
    "rpt_fit_id",
    "rpt_fit_accepted",
    "label_fit_accepted",
    "label_origin",
    "target_verified_as_independent_ground_truth",
    "pydma_formula_parity",
    "lli_percent",
    "lam_anode_percent",
    "lam_cathode_percent",
    "label_status",
    "status",
    "historical_lli_percent",
    "historical_lam_anode_percent",
    "historical_lam_cathode_percent",
    "comparison_error",
    "negative_targets",
    "historical_negative_present",
    "negative_estimate_classification",
    "physical_plausibility_status",
]


class CurvePreflightError(ValueError):
    pass


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required PyDMA pilot input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _resolve_source(root: Path, part: str, relative: str) -> Path:
    raw = str(relative).strip().replace("\\", "/")
    if not raw:
        raise FileNotFoundError("Source path is empty.")
    path = Path(raw)
    if path.is_absolute():
        resolved = path
    elif raw.casefold().startswith("data/raw/oxford/"):
        resolved = root / path
    else:
        resolved = root / "data" / "raw" / "oxford" / f"part{part}" / path
    if not resolved.is_file():
        raise FileNotFoundError(f"PyDMA pilot source is absent: {resolved}")
    return resolved


def _source_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _curve_candidates(
    path: Path,
    step_number: int,
    direction: str,
) -> list[dict[str, Any]]:
    if step_number not in {8, 10}:
        raise ValueError(f"Unsupported pOCV Step: {step_number}")
    if direction != ("discharge" if step_number == 8 else "charge"):
        raise ValueError(f"Step {step_number} does not match direction {direction!r}.")

    raw_runs: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    segment_index = 0
    row_offset = 0

    def finish_run() -> None:
        nonlocal active
        if active is not None:
            raw_runs.append(active)
        active = None

    for chunk in pd.read_csv(
        path,
        usecols=["Step", "Amphr", "Amps", "Volts"],
        chunksize=250_000,
        low_memory=False,
    ):
        steps = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(dtype=np.float64)
        capacity = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy(dtype=np.float64)
        current = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(dtype=np.float64)
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(dtype=np.float64)
        change_points = np.flatnonzero(
            (steps[1:] != steps[:-1]) | (np.isnan(steps[1:]) != np.isnan(steps[:-1]))
        ) + 1
        boundaries = np.r_[0, change_points, len(steps)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            value = steps[start]
            is_target_step = np.isfinite(value) and int(value) == step_number
            if active is not None and not is_target_step:
                finish_run()
            if not is_target_step:
                continue
            if active is None:
                segment_index += 1
                active = {
                    "segment_index": segment_index,
                    "capacity": [],
                    "voltage": [],
                    "source_rows": [],
                }
            cap = capacity[start:end]
            amps = current[start:end]
            volts = voltage[start:end]
            finite = np.isfinite(cap) & np.isfinite(volts) & np.isfinite(amps)
            sign = amps < 0 if direction == "discharge" else amps > 0
            mask = finite & sign & (np.abs(amps) <= 0.2)
            selected_capacity = cap[mask]
            active["capacity"].append(selected_capacity)
            active["voltage"].append(volts[mask])
            active["source_rows"].append(
                row_offset + np.flatnonzero(mask) + start + 2
            )
        row_offset += len(chunk)
    finish_run()

    candidates: list[dict[str, Any]] = []
    for run in raw_runs:
        capacity = np.concatenate(run["capacity"]) if run["capacity"] else np.asarray([])
        voltage = np.concatenate(run["voltage"]) if run["voltage"] else np.asarray([])
        source_rows = (
            np.concatenate(run["source_rows"])
            if run["source_rows"]
            else np.asarray([], dtype=np.int64)
        )
        if not len(capacity):
            continue
        starts = np.r_[0, np.flatnonzero(np.diff(capacity) <= 0) + 1]
        ends = np.r_[starts[1:], len(capacity)]
        for subsegment_index, (start, end) in enumerate(zip(starts, ends), start=1):
            segment_capacity = capacity[start:end]
            segment_voltage = voltage[start:end]
            segment_rows = source_rows[start:end]
            span = float(np.ptp(segment_capacity))
            valid = bool(
                len(segment_capacity) >= 10
                and np.all(np.diff(segment_capacity) > 0)
                and span > 0
            )
            candidates.append(
                {
                    "step_segment_index": int(run["segment_index"]),
                    "subsegment_index": subsegment_index,
                    "source_row_start": int(segment_rows[0]),
                    "source_row_end": int(segment_rows[-1]),
                    "sample_count": int(len(segment_capacity)),
                    "capacity_min": float(segment_capacity.min()),
                    "capacity_max": float(segment_capacity.max()),
                    "capacity_span": span,
                    "voltage_min": float(segment_voltage.min()),
                    "voltage_max": float(segment_voltage.max()),
                    "valid": valid,
                    "capacity": segment_capacity,
                    "voltage": segment_voltage,
                }
            )
    return candidates


def _read_verified_curve(
    root: Path,
    path: Path,
    step_number: int,
    direction: str,
    source_reference: str = "",
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    candidates = _curve_candidates(path, step_number, direction)
    reference = None
    if source_reference:
        match = _REF_PATTERN.fullmatch(source_reference.replace("\\", "/"))
        if match is None:
            raise CurvePreflightError(f"Unparseable source segment reference: {source_reference}")
        source_reference_path = match.group("source")
        if source_reference_path.casefold() != _source_relative(root, path).casefold():
            raise CurvePreflightError(
                f"Source reference points to {source_reference_path}, not {_source_relative(root, path)}."
            )
        if int(match.group("step")) != step_number:
            raise CurvePreflightError("Source segment Step does not match the requested Step.")
        reference = {
            "step_segment_index": int(match.group("segment")),
            "source_row_start": int(match.group("start")),
            "source_row_end": int(match.group("end")),
            "subsegment_index": int(match.group("subsegment")),
        }
        selected = [
            item
            for item in candidates
            if item["step_segment_index"] == reference["step_segment_index"]
            and item["subsegment_index"] == reference["subsegment_index"]
        ]
        if len(selected) != 1:
            raise CurvePreflightError(
                f"Expected one curve for the exact segment reference; found {len(selected)}."
            )
        candidate = selected[0]
        if (
            candidate["source_row_start"] != reference["source_row_start"]
            or candidate["source_row_end"] != reference["source_row_end"]
        ):
            raise CurvePreflightError(
                "Extracted source-row boundary differs from the verified manifest reference."
            )
        if not candidate["valid"]:
            raise CurvePreflightError("The exact source segment fails the monotone-curve checks.")
    else:
        valid = [item for item in candidates if item["valid"]]
        if len(valid) != 1:
            raise CurvePreflightError(
                f"BoL Step {step_number} requires one unique valid segment; found {len(valid)}."
            )
        candidate = valid[0]
    audit = {
        "status": "verified_exact_manifest_boundary" if reference else "unique_bol_curve_candidate",
        "source_file": _source_relative(root, path),
        "step": step_number,
        "direction": direction,
        "source_reference": source_reference,
        "selected_segment": candidate["step_segment_index"],
        "selected_subsegment": candidate["subsegment_index"],
        "source_row_start": candidate["source_row_start"],
        "source_row_end": candidate["source_row_end"],
        "source_row_indexing": "one-based CSV line numbers; header is line 1",
        "sample_count": candidate["sample_count"],
        "capacity_min_native": candidate["capacity_min"],
        "capacity_max_native": candidate["capacity_max"],
        "capacity_span_native": candidate["capacity_span"],
        "voltage_min_v": candidate["voltage_min"],
        "voltage_max_v": candidate["voltage_max"],
        "candidate_segments": [
            {key: value for key, value in item.items() if key not in {"capacity", "voltage"}}
            for item in candidates
        ],
    }
    return candidate["capacity"], candidate["voltage"], audit


def _load_references(root: Path, direction: str) -> tuple[Any, Any, dict[str, Any]]:
    from pydma.electrodes.electrode import ElectrodeOCP
    from pydma.preprocessing.loader import load_ocp

    half_cell = root / "data/raw/oxford/part1/Half_Cell/Half Cell"
    if direction == "charge":
        paths = (half_cell / "CH_HC_anode.csv", half_cell / "CH_HC_cathode.csv")
        refs = (
            load_ocp(paths[0], electrode_type="anode", soc_col="SoC", voltage_col="Volts"),
            load_ocp(paths[1], electrode_type="cathode", soc_col="SoC", voltage_col="Volts"),
        )
    else:
        paths = (half_cell / "DCH_HC_anode.csv", half_cell / "DHC_HC_cathode.csv")
        anode_frame = pd.read_csv(paths[0], usecols=["SoC", "Volts"])
        soc = pd.to_numeric(anode_frame["SoC"], errors="coerce").to_numpy(dtype=np.float64)
        voltage = pd.to_numeric(anode_frame["Volts"], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(soc) & np.isfinite(voltage)
        if not finite.any():
            raise ValueError(f"No finite anode reference values in {paths[0]}")
        refs = (
            ElectrodeOCP(
                soc=soc[finite],
                voltage=voltage[finite],
                name="DCH_HC_anode_valid_pairs",
                electrode_type="anode",
            ),
            load_ocp(paths[1], electrode_type="cathode", soc_col="SoC", voltage_col="Volts"),
        )
    metadata = {
        "anode_reference": _source_relative(root, paths[0]),
        "cathode_reference": _source_relative(root, paths[1]),
        "soc_range_normalized": {
            "anode": [float(np.min(refs[0].soc)), float(np.max(refs[0].soc))],
            "cathode": [float(np.min(refs[1].soc)), float(np.max(refs[1].soc))],
        },
        "voltage_range_v": {
            "anode": [float(np.min(refs[0].voltage)), float(np.max(refs[0].voltage))],
            "cathode": [float(np.min(refs[1].voltage)), float(np.max(refs[1].voltage))],
        },
    }
    return refs[0], refs[1], metadata


def _fit_record(
    *,
    fit_id: str,
    observation_type: str,
    part: str,
    cell: str,
    rpt: str,
    direction: str,
    seed: int,
    curve: np.ndarray,
    voltage: np.ndarray,
    curve_audit: dict[str, Any],
    references: dict[str, Any],
    source_file: str,
    baseline_reference_id: str,
    result: Any | None,
    runtime_seconds: float,
    error: str = "",
) -> dict[str, Any]:
    step = 10 if direction == "charge" else 8
    params = result.fitted_params.to_dict() if result is not None else {}
    config = result.config_snapshot if result is not None else {}
    lower = tuple(config.get("lower_bounds", (1.0, -1.0, 1.0, -1.0)))
    upper = tuple(config.get("upper_bounds", (2.0, 0.0, 2.1, 0.0)))
    bound_checks = []
    for name, value, low, high in zip(
        ("alpha_an", "beta_an", "alpha_ca", "beta_ca"),
        (params.get("alpha_an"), params.get("beta_an"), params.get("alpha_ca"), params.get("beta_ca")),
        lower,
        upper,
    ):
        if value is not None:
            bound_checks.append(
                {
                    "parameter": name,
                    "value": value,
                    "lower_bound": low,
                    "upper_bound": high,
                    "exact_or_near_bound": bool(
                        abs(float(value) - float(low)) <= 1e-6
                        or abs(float(value) - float(high)) <= 1e-6
                    ),
                    "normalized_distance_to_nearest_bound": min(
                        (float(value) - float(low)) / (float(high) - float(low)),
                        (float(high) - float(value)) / (float(high) - float(low)),
                    ),
                }
            )
    rmse_mv = float(result.rmse * 1000) if result is not None else None
    accepted_runs = int(result.n_accepted_runs) if result is not None else 0
    total_runs = int(result.n_total_runs) if result is not None else 0
    passes_15 = bool(
        result is not None
        and rmse_mv < 15.0
        and accepted_runs >= 3
        and bool(result.is_accepted)
    )
    passes_10 = bool(
        result is not None
        and rmse_mv < 10.0
        and accepted_runs >= 3
        and bool(result.is_accepted)
    )
    return {
        "fit_id": fit_id,
        "dataset_part": part,
        "physical_cell_id": cell,
        "observation_type": observation_type,
        "rpt_index": "0" if observation_type == "BOL_REFERENCE" else rpt,
        "direction": direction,
        "step": step,
        "seed": seed,
        "source_file": source_file,
        "source_row_start": curve_audit["source_row_start"],
        "source_row_end": curve_audit["source_row_end"],
        "source_row_indexing": curve_audit["source_row_indexing"],
        "baseline_reference_id": baseline_reference_id,
        "half_cell_anode": references["anode_reference"],
        "half_cell_cathode": references["cathode_reference"],
        "sample_count": int(len(curve)),
        "capacity_unit": "native Amphr; absolute unit unverified",
        "capacity_span_native": float(np.ptp(curve)),
        "voltage_min_v": float(np.min(voltage)),
        "voltage_max_v": float(np.max(voltage)),
        "curve_preflight_status": curve_audit["status"],
        "curve_audit_json": json.dumps(curve_audit, sort_keys=True),
        "rmse_mV": rmse_mv if rmse_mv is not None else "",
        "accepted_runs": accepted_runs,
        "total_runs": total_runs,
        "rejected_runs": total_runs - accepted_runs,
        "optimizer_accepted": bool(result is not None and result.is_accepted),
        "passes_primary_gate_15mV_3runs": passes_15,
        "passes_sensitivity_gate_10mV_3runs": passes_10,
        "fit_status": (
            str(result.status)
            if result is not None
            else "fit_exception"
        ),
        "runtime_seconds": runtime_seconds,
        "optimizer_config_json": json.dumps(config, sort_keys=True, default=str),
        "parameter_bounds_json": json.dumps(bound_checks, sort_keys=True),
        "parameter_std_json": json.dumps(
            None if result is None or result.param_std is None
            else np.asarray(result.param_std).tolist()
        ),
        "fitted_parameters_json": json.dumps(params, sort_keys=True),
        "error": error,
    }


def _independent_degradation_labels(
    reference_params: dict[str, Any],
    current_params: dict[str, Any],
    reference_capacity: float,
    current_capacity: float,
) -> dict[str, float]:
    anode_init = float(reference_params["alpha_an"]) * reference_capacity
    cathode_init = float(reference_params["alpha_ca"]) * reference_capacity
    inventory_init = (
        float(reference_params["alpha_ca"])
        + float(reference_params["beta_ca"])
        - float(reference_params["beta_an"])
    ) * reference_capacity
    if min(abs(anode_init), abs(cathode_init), abs(inventory_init)) <= 1e-12:
        raise ValueError("A baseline electrode capacity or lithium inventory is zero.")
    anode_current = float(current_params["alpha_an"]) * current_capacity
    cathode_current = float(current_params["alpha_ca"]) * current_capacity
    inventory_current = (
        float(current_params["alpha_ca"])
        + float(current_params["beta_ca"])
        - float(current_params["beta_an"])
    ) * current_capacity
    return {
        "lli_percent": 100.0 * (inventory_init - inventory_current) / inventory_init,
        "lam_anode_percent": 100.0 * (anode_init - anode_current) / anode_init,
        "lam_cathode_percent": 100.0 * (cathode_init - cathode_current) / cathode_init,
    }


def _historical_label(
    audit: pd.DataFrame,
    part: str,
    cell: str,
    rpt: str,
    direction: str,
) -> dict[str, Any]:
    rows = audit.loc[
        audit.result_table.eq("part1_part2_rmse15")
        & audit.dataset_part.eq(part)
        & audit.physical_cell_id.eq(cell)
        & audit.checkup.eq(f"RPT{rpt}")
        & audit.direction.eq(direction)
        & audit.diagnostic_label_accepted.str.casefold().isin({"true", "1", "yes"})
    ]
    columns = ["lli_percent", "lam_anode_percent", "lam_cathode_percent"]
    values = rows[columns].drop_duplicates()
    if len(values) == 1:
        return {
            "status": "one_unique_accepted_historical_value",
            **{f"historical_{column}": values.iloc[0][column] for column in columns},
        }
    return {
        "status": "no_historical_value" if rows.empty else "multiple_historical_values",
        **{f"historical_{column}": "" for column in columns},
    }


def _build_negative_label_audit(
    label_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    negative_rows = []
    for row in label_rows:
        negative_targets = [
            name
            for name in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
            if row[name] != "" and float(row[name]) < 0
        ]
        historical_negative = any(
            row.get(f"historical_{name}", "") != ""
            and float(row[f"historical_{name}"]) < 0
            for name in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
        )
        if not negative_targets and not historical_negative:
            continue
        partner = next(
            (
                other
                for other in label_rows
                if other["rpt_index"] == row["rpt_index"]
                and other["direction"] == row["direction"]
                and other["physical_cell_id"] == row["physical_cell_id"]
                and other["dataset_part"] == row["dataset_part"]
                and other["rpt_fit_id"] != row["rpt_fit_id"]
            ),
            None,
        )
        sign_changed = False
        if partner is not None:
            for name in ("lli_percent", "lam_anode_percent", "lam_cathode_percent"):
                if row[name] != "" and partner[name] != "":
                    sign_changed |= (float(row[name]) < 0) != (float(partner[name]) < 0)
        negative_rows.append(
            {
                **row,
                "negative_targets": ";".join(negative_targets),
                "historical_negative_present": historical_negative,
                "negative_estimate_classification": (
                    "OPTIMIZER_UNSTABLE" if sign_changed else "UNRESOLVED"
                ),
                "physical_plausibility_status": (
                    "not_ground_truth; requires baseline, optimizer, curve, and identifiability review"
                ),
            }
        )
    return negative_rows


def _write_csv(
    output: Path,
    name: str,
    rows: list[dict[str, Any]],
    columns: list[str] | None = None,
) -> None:
    path = output / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows, columns=columns).to_csv(
        temporary, index=False, encoding="utf-8-sig"
    )
    temporary.replace(path)


def run_pilot(
    root: Path | None = None,
    run_id: str = _DEFAULT_RUN_ID,
    stage1_dir: Path = _STAGE1_DIR,
) -> Path:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(f"Invalid PyDMA pilot run ID: {run_id!r}")
    stage1_dir = Path(stage1_dir)
    if stage1_dir.is_absolute():
        try:
            stage1_dir = stage1_dir.resolve().relative_to(repo)
        except ValueError as error:
            raise ValueError("Stage 1 outputs must be inside the repository.") from error
    if ".." in stage1_dir.parts:
        raise ValueError("Stage 1 output path cannot escape the repository.")
    output = repo / stage1_dir / run_id
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite previous PyDMA pilot artifacts: {output}"
        )

    from pydma import DMAAnalyzer, DMAConfig
    import torch

    from bafuse.data.final_data_validation_v2 import _read_csv, _truth
    from bafuse.labeling.pydma_runner import _fit_one

    mapping = _read_csv(repo, "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv")
    bol = _read_csv(repo, str(stage1_dir / "stage1_bol_reference_audit.csv"))
    historical = _read_csv(repo, str(_AUDIT_DIR / "pydma_fit_audit.csv"))
    bol_row = bol.loc[bol.physical_cell_id.eq("G1-Cell9")].iloc[0]
    if not _truth(bol_row.bol_verified):
        raise ValueError("Pilot cell does not have a verified Part 1 Step 4 BoL reference.")
    bol_path = _resolve_source(
        repo, "1", str(bol_row.primary_bol_reference_file)
    )

    target_curves: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    selection_notes = []
    for part, cell, rpt, role, _ in _PILOT_TARGETS:
        match = mapping.loc[
            mapping.part.eq(part)
            & mapping.physical_cell_id.eq(cell)
            & mapping.rpt_index.eq(rpt)
        ]
        if len(match) != 1:
            raise ValueError(f"Pilot key is not unique: Part {part} {cell} RPT{rpt}.")
        row = match.iloc[0]
        if (
            row.mapping_status_v2 not in _STRICT_MAPPING
            or row.step8_status_exact != "verified"
            or row.step10_status_exact != "verified"
        ):
            raise ValueError(
                f"Pilot target lacks strict exact-boundary verification: {cell} RPT{rpt}."
            )
        source = _resolve_source(repo, part, row.source_file)
        if str(row.partition_v2) != "train":
            raise ValueError(f"Pilot cell unexpectedly assigned to {row.partition_v2}.")
        selection_notes.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "role": role,
                "source_file": _source_relative(repo, source),
                "mapping_status": row.mapping_status_v2,
                "partition": row.partition_v2,
            }
        )
        for direction, step, ref_column in (
            ("discharge", 8, "step8_segment_ref_exact"),
            ("charge", 10, "step10_segment_ref_exact"),
        ):
            curve, voltage, audit = _read_verified_curve(
                repo, source, step, direction, str(row[ref_column])
            )
            target_curves[(part, rpt, direction)] = (curve, voltage, audit)

    baseline_curves: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for direction, step in (("discharge", 8), ("charge", 10)):
        curve, voltage, audit = _read_verified_curve(
            repo, bol_path, step, direction
        )
        baseline_curves[direction] = (curve, voltage, audit)

    curve_plan: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {
        ("1", "0", direction): curve
        for direction, curve in baseline_curves.items()
    }
    curve_plan.update(target_curves)
    references_by_direction = {
        direction: _load_references(repo, direction)
        for direction in ("charge", "discharge")
    }
    use_cuda = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"
    fit_tasks: list[tuple[str, str, str, str, int]] = []
    for direction in ("discharge", "charge"):
        for seed in _SEEDS:
            fit_tasks.append(("1", "G1-Cell9", "0", direction, seed))
    for part, cell, rpt, _, primary_seed in _PILOT_TARGETS:
        for direction in ("discharge", "charge"):
            fit_tasks.append((part, cell, rpt, direction, primary_seed))
            if rpt == "2":
                fit_tasks.append((part, cell, rpt, direction, 212))

    output.mkdir(parents=True)
    fit_rows: list[dict[str, Any]] = []
    fit_results: dict[str, Any] = {}
    fitted_parameters: dict[str, dict[str, Any]] = {}
    curve_arrays: dict[str, np.ndarray] = {}
    reference_meta = {direction: item[2] for direction, item in references_by_direction.items()}
    for part, cell, rpt, direction, seed in fit_tasks:
        is_bol = rpt == "0"
        curve_key = ("1", "0", direction) if is_bol else (part, rpt, direction)
        curve, voltage, curve_audit = curve_plan[curve_key]
        source_path = bol_path if is_bol else _resolve_source(
            repo,
            part,
            mapping.loc[
                mapping.part.eq(part)
                & mapping.physical_cell_id.eq(cell)
                & mapping.rpt_index.eq(rpt),
                "source_file",
            ].iloc[0],
        )
        fit_id = f"{'bol' if is_bol else 'rpt'}-p{part}-{cell.lower()}-rpt{rpt}-{direction}-seed{seed}"
        started = time.perf_counter()
        result = None
        error_text = ""
        try:
            result = _fit_one(
                direction,
                curve,
                voltage,
                references_by_direction[direction][0],
                references_by_direction[direction][1],
                seed=seed,
                context=f"Stage 2 pilot {cell} RPT{rpt} {direction}",
                use_cuda=use_cuda,
                maxiter=50,
                popsize=30,
                max_tries=10,
            )
            fit_results[fit_id] = result
            fitted_parameters[fit_id] = result.fitted_params.to_dict()
            curve_arrays[f"{fit_id}__measured_capacity"] = curve
            curve_arrays[f"{fit_id}__measured_voltage"] = voltage
            for field in (
                "soc_measured",
                "ocv_measured",
                "soc_reconstructed",
                "ocv_reconstructed",
            ):
                value = getattr(result, field, None)
                if value is not None:
                    curve_arrays[f"{fit_id}__{field}"] = np.asarray(value)
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            print(f"Stage 2 pilot fit failed {fit_id}: {error_text}", flush=True)
        runtime = time.perf_counter() - started
        row = _fit_record(
            fit_id=fit_id,
            observation_type="BOL_REFERENCE" if is_bol else "RPT",
            part=part,
            cell=cell,
            rpt=rpt,
            direction=direction,
            seed=seed,
            curve=curve,
            voltage=voltage,
            curve_audit=curve_audit,
            references=reference_meta[direction],
            source_file=_source_relative(repo, source_path),
            baseline_reference_id=(
                f"p1-{cell}-bol-{direction}-seed{seed}" if not is_bol else ""
            ),
            result=result,
            runtime_seconds=runtime,
            error=error_text,
        )
        fit_rows.append(row)

    attempt_by_id = {row["fit_id"]: row for row in fit_rows}
    label_rows: list[dict[str, Any]] = []
    for part, cell, rpt, role, primary_seed in _PILOT_TARGETS:
        target_map = mapping.loc[
            mapping.part.eq(part)
            & mapping.physical_cell_id.eq(cell)
            & mapping.rpt_index.eq(rpt)
        ].iloc[0]
        for direction in ("discharge", "charge"):
            seed_values = (211, 212) if rpt == "2" else (primary_seed,)
            for seed in seed_values:
                fit_id = f"rpt-p{part}-{cell.lower()}-rpt{rpt}-{direction}-seed{seed}"
                baseline_id = f"bol-p1-{cell.lower()}-rpt0-{direction}-seed{seed}"
                current_result = fit_results.get(fit_id)
                baseline_result = fit_results.get(baseline_id)
                values = {"lli_percent": "", "lam_anode_percent": "", "lam_cathode_percent": ""}
                formula_parity = False
                comparison_error = ""
                if current_result is not None and baseline_result is not None:
                    try:
                        reference_capacity = float(
                            np.ptp(baseline_curves[direction][0])
                        )
                        current_capacity = float(
                            np.ptp(target_curves[(part, rpt, direction)][0])
                        )
                        params_ref = baseline_result.fitted_params.to_dict()
                        params_current = current_result.fitted_params.to_dict()
                        values = _independent_degradation_labels(
                            params_ref,
                            params_current,
                            reference_capacity,
                            current_capacity,
                        )
                        library = DMAAnalyzer(
                            config=DMAConfig(direction=direction),
                            anode=references_by_direction[direction][0],
                            cathode=references_by_direction[direction][1],
                        ).compare_with_reference(
                            reference_params=baseline_result.fitted_params,
                            current_params=current_result.fitted_params,
                            reference_capacity=reference_capacity,
                            current_capacity=current_capacity,
                        )
                        library_values = {
                            "lli_percent": float(library.lli * 100),
                            "lam_anode_percent": float(library.lam_anode * 100),
                            "lam_cathode_percent": float(library.lam_cathode * 100),
                        }
                        formula_parity = all(
                            np.isclose(values[name], library_values[name], rtol=1e-10, atol=1e-12)
                            for name in values
                        )
                        if not formula_parity:
                            raise ArithmeticError(
                                "Independent saved-parameter formula differs from PyDMA comparison."
                            )
                    except Exception as error:
                        comparison_error = f"{type(error).__name__}: {error}"
                        values = {"lli_percent": "", "lam_anode_percent": "", "lam_cathode_percent": ""}
                historical_values = _historical_label(
                    historical, part, cell, rpt, direction
                )
                baseline_attempt = attempt_by_id.get(baseline_id, {})
                current_attempt = attempt_by_id.get(fit_id, {})
                fit_accepted = bool(
                    baseline_attempt.get("passes_primary_gate_15mV_3runs", False)
                    and current_attempt.get("passes_primary_gate_15mV_3runs", False)
                )
                label_rows.append(
                    {
                        "dataset_part": part,
                        "group": str(cell).split("-")[0].removeprefix("G"),
                        "physical_cell_id": cell,
                        "rpt_index": rpt,
                        "role": role,
                        "direction": direction,
                        "step": 8 if direction == "discharge" else 10,
                        "source_file": target_map.source_file,
                        "source_segment_ref": (
                            target_map.step8_segment_ref_exact
                            if direction == "discharge"
                            else target_map.step10_segment_ref_exact
                        ),
                        "baseline_reference_id": baseline_id,
                        "baseline_fit_accepted": bool(
                            baseline_attempt.get("passes_primary_gate_15mV_3runs", False)
                        ),
                        "rpt_fit_id": fit_id,
                        "rpt_fit_accepted": bool(
                            current_attempt.get("passes_primary_gate_15mV_3runs", False)
                        ),
                        "label_fit_accepted": fit_accepted,
                        "label_origin": "PYDMA_FITTED_DIAGNOSTIC",
                        "target_verified_as_independent_ground_truth": False,
                        "pydma_formula_parity": formula_parity,
                        "lli_percent": values["lli_percent"],
                        "lam_anode_percent": values["lam_anode_percent"],
                        "lam_cathode_percent": values["lam_cathode_percent"],
                        "label_status": (
                            "accepted_diagnostic_only"
                            if fit_accepted and formula_parity
                            else "withheld_or_unverified_diagnostic"
                        ),
                        **historical_values,
                        "comparison_error": comparison_error,
                    }
                )

    negative_rows = _build_negative_label_audit(label_rows)

    bol_parameters = [row for row in fit_rows if row["observation_type"] == "BOL_REFERENCE"]
    rpt_parameters = [row for row in fit_rows if row["observation_type"] == "RPT"]
    fit_quality = [
        {
            key: row[key]
            for key in (
                "fit_id",
                "dataset_part",
                "physical_cell_id",
                "observation_type",
                "rpt_index",
                "direction",
                "seed",
                "curve_preflight_status",
                "source_row_start",
                "source_row_end",
                "sample_count",
                "rmse_mV",
                "accepted_runs",
                "total_runs",
                "rejected_runs",
                "optimizer_accepted",
                "passes_primary_gate_15mV_3runs",
                "passes_sensitivity_gate_10mV_3runs",
                "parameter_bounds_json",
                "parameter_std_json",
                "runtime_seconds",
                "fit_status",
                "error",
            )
        }
        for row in fit_rows
    ]

    _write_csv(output, "stage2_pydma_bol_parameters.csv", bol_parameters)
    _write_csv(output, "stage2_pydma_rpt_parameters.csv", rpt_parameters)
    _write_csv(output, "stage2_pydma_fit_attempts.csv", fit_rows)
    _write_csv(output, "stage2_pydma_fit_quality.csv", fit_quality)
    _write_csv(output, "stage2_pydma_label_comparison.csv", label_rows)
    _write_csv(
        output,
        "stage2_pydma_negative_label_audit.csv",
        negative_rows,
        columns=_NEGATIVE_AUDIT_COLUMNS,
    )
    curve_path = output / "stage2_pydma_reconstructed_curves.npz"
    temporary = output / "stage2_pydma_reconstructed_curves.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **curve_arrays)
    temporary.replace(curve_path)

    rmses = [
        float(row["rmse_mV"])
        for row in fit_rows
        if row["rmse_mV"] != ""
    ]
    accepted_count = sum(bool(row["passes_primary_gate_15mV_3runs"]) for row in fit_rows)
    rejected_count = sum(
        row["fit_status"] != "fit_exception"
        and not bool(row["passes_primary_gate_15mV_3runs"])
        for row in fit_rows
    )
    exception_count = sum(row["fit_status"] == "fit_exception" for row in fit_rows)
    stability_rows = []
    for direction in ("discharge", "charge"):
        first_id = f"rpt-p1-g1-cell9-rpt2-{direction}-seed211"
        second_id = f"rpt-p1-g1-cell9-rpt2-{direction}-seed212"
        first = attempt_by_id.get(first_id, {})
        second = attempt_by_id.get(second_id, {})
        first_result = fit_results.get(first_id)
        second_result = fit_results.get(second_id)
        parameter_deltas = {}
        if first_result is not None and second_result is not None:
            params_a = first_result.fitted_params.to_dict()
            params_b = second_result.fitted_params.to_dict()
            parameter_deltas = {
                key: abs(float(params_a[key]) - float(params_b[key]))
                for key in ("alpha_an", "beta_an", "alpha_ca", "beta_ca")
            }
        labels_a = next(
            (row for row in label_rows if row["rpt_index"] == "2" and row["direction"] == direction and "seed211" in row["rpt_fit_id"]),
            None,
        )
        labels_b = next(
            (row for row in label_rows if row["rpt_index"] == "2" and row["direction"] == direction and "seed212" in row["rpt_fit_id"]),
            None,
        )
        label_deltas = {}
        if labels_a and labels_b:
            label_deltas = {
                name: (
                    abs(float(labels_a[name]) - float(labels_b[name]))
                    if labels_a[name] != "" and labels_b[name] != ""
                    else ""
                )
                for name in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
            }
        stability_rows.append(
            {
                "direction": direction,
                "seed211_rmse_mV": first.get("rmse_mV", ""),
                "seed212_rmse_mV": second.get("rmse_mV", ""),
                "seed211_primary_gate": first.get("passes_primary_gate_15mV_3runs", False),
                "seed212_primary_gate": second.get("passes_primary_gate_15mV_3runs", False),
                "absolute_parameter_deltas": json.dumps(parameter_deltas, sort_keys=True),
                "absolute_label_deltas_percentage_points": json.dumps(label_deltas, sort_keys=True),
                "stability_gate": "NOT_PREDEFINED; descriptive only",
            }
        )

    run_elapsed = sum(float(row["runtime_seconds"]) for row in fit_rows)
    report_lines = [
        "# Stage 2 PyDMA pilot batch",
        "",
        "Status: **pilot only; no full refit or model training**.",
        "",
        f"- Pilot cell: G1-Cell9, assigned to Train; P1 RPT2 (early/high-RMSE negative historical example), P1 RPT8 (middle), and P2 RPT14 (later continuation).",
        f"- Optimizer device: {device}. Deterministic seeds: 211 and 212 for BoL and repeated RPT2 fits; other targets use seed 211.",
        f"- Optimizer calls recorded: {len(fit_rows)}; passed primary gate: {accepted_count}; rejected: {rejected_count}; exceptions: {exception_count}.",
        f"- Summed optimizer runtime: {run_elapsed:.1f} seconds; observed RMSE range: {min(rmses):.3f}–{max(rmses):.3f} mV." if rmses else "- No optimizer fit produced an RMSE.",
        "- Primary acceptance rule unchanged: RMSE <15 mV and at least 3 accepted runs. The <10 mV rule is sensitivity reporting only.",
        "- Curves were extracted using exact manifest source-row ranges for RPTs. Each selected pOCV segment passed current-direction, |I|<=0.2 A, finite-pair, monotonic-capacity, and minimum-sample checks. BoL Step 8/10 had one unique valid candidate per direction.",
        "- Charge used CH_HC_anode/CH_HC_cathode; discharge used DCH_HC_anode/DHC_HC_cathode. Reference SoC ranges were normalized to [0,1] by PyDMA; voltage values were in approximately 0–4.3 V. Capacity remains native Amphr, not asserted as Ah.",
        "- LLI/LAM were recomputed directly from saved reference/current parameter dictionaries and measured capacity spans, then checked against `DMAAnalyzer.compare_with_reference`; see `pydma_formula_parity`.",
        "- Direction-specific estimates are not averaged. Accepted fits remain `PYDMA_FITTED_DIAGNOSTIC`, not independent ground truth; negative values are retained.",
        "",
        "## Deterministic pilot selection",
        "",
        "| Part | Cell | RPT | Role | Source | Status |",
        "|---:|---|---:|---|---|---|",
    ]
    report_lines.extend(
        f"| {row['dataset_part']} | {row['physical_cell_id']} | {row['rpt_index']} | {row['role']} | `{row['source_file']}` | {row['mapping_status']} |"
        for row in selection_notes
    )
    report_lines.extend(
        [
            "",
            "Part 3 was not fitted: the Guide-listed Part 3 BoL sources do not provide verified Step 4 baselines, and G10-Cell2 identity remains user-confirmed. Part 2 Groups 5–6 were excluded because their primary BoL policy or RPT/curve mapping is unresolved.",
            "",
            "## Initialization sensitivity",
            "",
            "| Direction | Seed 211 RMSE | Seed 212 RMSE | Seed 211 gate | Seed 212 gate | Parameter deltas | Label deltas (pp) |",
            "|---|---:|---:|---|---|---|---|",
        ]
    )
    report_lines.extend(
        f"| {row['direction']} | {row['seed211_rmse_mV']} | {row['seed212_rmse_mV']} | {row['seed211_primary_gate']} | {row['seed212_primary_gate']} | `{row['absolute_parameter_deltas']}` | `{row['absolute_label_deltas_percentage_points']}` |"
        for row in stability_rows
    )
    report_lines.extend(
        [
            "",
            "No numeric stability tolerance was specified in the task or existing repository policy. The report therefore provides observed seed-to-seed deltas without declaring the stability gate passed. No full batch was launched.",
            "",
            "## Negative diagnostics and limitations",
            "",
            f"- Negative audit rows in this pilot: {len(negative_rows)}. Classification is `UNRESOLVED` unless seed repeats change target sign, in which case it is `OPTIMIZER_UNSTABLE`.",
            "- This subset does not establish Part 3 coverage, full-cell baseline readiness, or all negative-estimate classifications.",
            "- Reconstructed arrays, source row ranges, optimizer parameters, bounds, accepted/rejected run counts, and per-fit diagnostics are saved in the versioned outputs.",
            "- Full refit remains blocked pending verified Part 2 Groups 5–6 / Part 3 baselines, resolution of exact-boundary mapping rows, and user-approved stability criteria.",
        ]
    )
    report = "\n".join(report_lines) + "\n"
    (output / "stage2_pydma_stability_report.md").write_text(report, encoding="utf-8")
    (output / "stage2_pydma_batch_report.md").write_text(report, encoding="utf-8")

    raw_sources = {
        _source_relative(repo, bol_path),
        *[str(row["source_file"]) for row in selection_notes],
        *[
            str(item)
            for item in (
                "data/raw/oxford/part1/Half_Cell/Half Cell/CH_HC_anode.csv",
                "data/raw/oxford/part1/Half_Cell/Half Cell/CH_HC_cathode.csv",
                "data/raw/oxford/part1/Half_Cell/Half Cell/DCH_HC_anode.csv",
                "data/raw/oxford/part1/Half_Cell/Half Cell/DHC_HC_cathode.csv",
            )
        ],
    }
    input_files = [
        "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv",
        str(stage1_dir / "stage1_bol_reference_audit.csv"),
        str(_AUDIT_DIR / "pydma_fit_audit.csv"),
        "src/bafuse/data/pydma_stage2.py",
        "src/bafuse/labeling/pydma_runner.py",
    ]
    input_hashes = {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in input_files
        if (repo / name).is_file()
    }
    output_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "stage2_run_manifest.json"
    }
    manifest = {
        "status": "pilot_completed_full_batch_not_run",
        "run_id": run_id,
        "optimizer_device": device,
        "deterministic_seeds": list(_SEEDS),
        "fit_attempts": len(fit_rows),
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "raw_sources": [
            {
                "path": name,
                "size_bytes": (repo / name).stat().st_size,
                "mtime_ns": (repo / name).stat().st_mtime_ns,
            }
            for name in sorted(raw_sources)
        ],
        "full_batch_launched": False,
        "limitations": [
            "No Part 3 fit because a verified same-cell BoL Step 4 baseline is unavailable.",
            "No numeric optimizer stability tolerance is pre-approved.",
            "PyDMA outputs are fitted diagnostics, not independent ground truth.",
        ],
    }
    (output / "stage2_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the versioned BaFuse PyDMA pilot only.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run-id", default=_DEFAULT_RUN_ID)
    parser.add_argument("--stage1-dir", type=Path, default=_STAGE1_DIR)
    args = parser.parse_args()
    print(run_pilot(args.root, args.run_id, args.stage1_dir))


if __name__ == "__main__":
    main()
