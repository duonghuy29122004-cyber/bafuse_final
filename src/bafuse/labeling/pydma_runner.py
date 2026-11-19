from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydma import DMAAnalyzer, DMAConfig
from pydma.electrodes.electrode import ElectrodeOCP
from pydma.preprocessing.loader import load_ocp

from .audit import FileRecord, GuideEntry
from .pydma_policy import (
    PYDMA_RMSE_THRESHOLD_MV,
    pydma_fit_passes_gate,
)


def _read_pocv_with_audit(
    path: Path,
    step_number: int,
    direction: str,
) -> tuple[tuple[np.ndarray, np.ndarray] | None, dict[str, Any]]:
    capacity_chunks: list[np.ndarray] = []
    voltage_chunks: list[np.ndarray] = []
    testtime_chunks: list[np.ndarray] = []
    source_row_chunks: list[np.ndarray] = []
    try:
        available_columns = set(pd.read_csv(path, nrows=0).columns)
        has_testtime = "TestTime" in available_columns
        columns = ["Step", "Amphr", "Amps", "Volts"]
        if has_testtime:
            columns.append("TestTime")
        source_row_offset = 0
        for chunk in pd.read_csv(
            path,
            usecols=columns,
            chunksize=250_000,
        ):
            source_rows = np.arange(
                source_row_offset + 2,
                source_row_offset + len(chunk) + 2,
                dtype=np.int64,
            )
            source_row_offset += len(chunk)
            if direction == "charge":
                mask = (
                    chunk["Step"].eq(step_number)
                    & chunk["Amps"].gt(0)
                    & chunk["Amps"].abs().le(0.2)
                )
            else:
                mask = (
                    chunk["Step"].eq(step_number)
                    & chunk["Amps"].lt(0)
                    & chunk["Amps"].abs().le(0.2)
                )
            selected = chunk.loc[mask, ["Amphr", "Volts"]]
            if selected.empty:
                continue
            capacity_chunks.append(selected["Amphr"].to_numpy(dtype=np.float64))
            voltage_chunks.append(selected["Volts"].to_numpy(dtype=np.float64))
            source_row_chunks.append(source_rows[np.flatnonzero(mask.to_numpy())])
            if has_testtime:
                testtime_chunks.append(
                    chunk.loc[mask, "TestTime"].to_numpy(dtype=np.float64)
                )
    except Exception as error:
        return None, {
            "status": "read_error",
            "source_file": str(path),
            "step": step_number,
            "direction": direction,
            "error": f"{type(error).__name__}: {error}",
            "segments": [],
            "selected_segment": None,
        }
    if not capacity_chunks:
        return None, {
            "status": "no_direction_filtered_rows",
            "source_file": str(path),
            "step": step_number,
            "direction": direction,
            "error": "No rows matched the Step/current direction filter.",
            "segments": [],
            "selected_segment": None,
        }
    capacity = np.concatenate(capacity_chunks)
    voltage = np.concatenate(voltage_chunks)
    source_rows = np.concatenate(source_row_chunks)
    testtime = np.concatenate(testtime_chunks) if testtime_chunks else None
    valid = np.isfinite(capacity) & np.isfinite(voltage)
    capacity, voltage = capacity[valid], voltage[valid]
    source_rows = source_rows[valid]
    if testtime is not None:
        testtime = testtime[valid]
    boundaries = np.flatnonzero(np.diff(capacity) <= 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, capacity.size]
    segments = []
    for index, (start, end) in enumerate(zip(starts, ends), 1):
        segment_capacity = capacity[start:end]
        segment_voltage = voltage[start:end]
        is_valid = bool(
            segment_capacity.size >= 10
            and np.all(np.diff(segment_capacity) > 0)
            and np.ptp(segment_capacity) > 0
        )
        segments.append(
            {
                "segment_index": index,
                "selected_order_start": int(start),
                "selected_order_end": int(end - 1),
                "sample_count": int(segment_capacity.size),
                "source_row_start": int(source_rows[start]),
                "source_row_end": int(source_rows[end - 1]),
                "capacity_min": float(segment_capacity.min()),
                "capacity_max": float(segment_capacity.max()),
                "voltage_min": float(segment_voltage.min()),
                "voltage_max": float(segment_voltage.max()),
                "capacity_span": float(np.ptp(segment_capacity)),
                "valid": is_valid,
            }
        )
    valid_segments = [segment for segment in segments if segment["valid"]]
    if not valid_segments:
        return None, {
            "status": "no_valid_monotone_segment",
            "source_file": str(path),
            "step": step_number,
            "direction": direction,
            "filtered_finite_samples": int(capacity.size),
            "segments": segments,
            "selected_segment": None,
            "selection_rule": "Require at least 10 finite pairs and strictly increasing Amphr within one contiguous segment.",
        }

    longest = max(segment["sample_count"] for segment in valid_segments)
    # Near-equal full-length segments are repeated curves; keep the first in file order.
    selected_segment = next(
        segment
        for segment in valid_segments
        if segment["sample_count"] >= 0.99 * longest
    )
    start = selected_segment["selected_order_start"]
    end = selected_segment["selected_order_end"] + 1
    selected_time = testtime[start:end] if testtime is not None else None
    if selected_time is None:
        time_audit: dict[str, Any] = {
            "status": "unavailable",
            "nonfinite_count": None,
            "nonincreasing_interval_count": None,
            "time_start": None,
            "time_end": None,
            "min_positive_interval": None,
            "max_interval": None,
        }
    else:
        finite_time = np.isfinite(selected_time)
        intervals = np.diff(selected_time)
        finite_intervals = np.isfinite(intervals)
        nonincreasing_count = int(
            np.count_nonzero(finite_intervals & (intervals <= 0))
            + np.count_nonzero(~finite_intervals)
        )
        positive_intervals = intervals[finite_intervals & (intervals > 0)]
        time_audit = {
            "status": (
                "verified_strictly_increasing"
                if finite_time.all() and nonincreasing_count == 0
                else "invalid_or_nonfinite"
            ),
            "nonfinite_count": int(np.count_nonzero(~finite_time)),
            "nonincreasing_interval_count": nonincreasing_count,
            "time_start": float(selected_time[0]) if finite_time[0] else None,
            "time_end": float(selected_time[-1]) if finite_time[-1] else None,
            "min_positive_interval": (
                float(positive_intervals.min()) if positive_intervals.size else None
            ),
            "max_interval": (
                float(intervals[finite_intervals].max())
                if finite_intervals.any()
                else None
            ),
        }
    return (capacity[start:end], voltage[start:end]), {
        "status": "selected_monotone_segment",
        "source_file": str(path),
        "step": step_number,
        "direction": direction,
        "filtered_finite_samples": int(capacity.size),
        "segments": segments,
        "selected_segment": selected_segment["segment_index"],
        "selected_sample_count": int(end - start),
        "selected_capacity_span": float(np.ptp(capacity[start:end])),
        "selected_source_row_start": int(source_rows[start]),
        "selected_source_row_end": int(source_rows[end - 1]),
        "selected_testtime_audit": time_audit,
        "selection_rule": "Select the longest valid monotone segment; if multiple are within 1% of longest, select the earliest in file order. Never concatenate across an Amphr reset.",
    }


def _read_pocv(path: Path, step_number: int, direction: str) -> tuple[np.ndarray, np.ndarray] | None:
    curve, _ = _read_pocv_with_audit(path, step_number, direction)
    return curve


def _select_curve_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the candidate with the most valid directions and points; preserve input order on ties."""
    return max(
        candidates,
        key=lambda item: (
            item["valid_direction_count"],
            item["selected_point_count"],
        ),
        default=None,
    )


def _stage_number(stage: str) -> tuple[int, int]:
    if stage.casefold().startswith("bol"):
        return 0, 0
    match = re.search(r"rpt\s*(\d+)", stage, re.IGNORECASE)
    return (int(match.group(1)), 1) if match else (10_000, 1)


def _record_source_path(dataset_root: Path, record: FileRecord) -> Path:
    source_path = Path(record.relative_path)
    return source_path if source_path.is_absolute() else dataset_root / source_path


def _fit_one(
    direction: str,
    capacity: np.ndarray,
    voltage: np.ndarray,
    anode: ElectrodeOCP,
    cathode: ElectrodeOCP,
    seed: int,
    context: str = "",
    use_cuda: bool = False,
    maxiter: int = 50,
    popsize: int = 30,
    max_tries: int = 10,
) -> Any:
    config = DMAConfig(
        direction=direction,
        data_length=500,
        weight_ocv=100.0,
        weight_dva=0.0,
        weight_ica=0.0,
        speed_preset="fast",
        req_accepted=3,
        max_tries_overall=max_tries,
        rmse_threshold=PYDMA_RMSE_THRESHOLD_MV / 1000.0,
        random_seed=seed,
        print_progress=False,
        allow_first_cycle_inhomogeneity=False,
    )
    started = time.perf_counter()
    print(
        f"PyDMA fit start {context} direction={direction} points={capacity.size}",
        flush=True,
    )
    try:
        backend = nullcontext()
        if use_cuda:
            from .torch_pydma import use_cuda_optimizer

            backend = use_cuda_optimizer()
        with backend:
            result = DMAAnalyzer(config=config, anode=anode, cathode=cathode).analyze(
                measured_capacity=capacity,
                measured_voltage=voltage,
                maxiter=maxiter,
                popsize=popsize,
                polish=True,
            )
    except Exception as error:
        print(
            f"PyDMA fit error {context} direction={direction} "
            f"elapsed_s={time.perf_counter() - started:.1f}: {type(error).__name__}: {error}",
            flush=True,
        )
        raise
    print(
        f"PyDMA fit done {context} direction={direction} "
        f"elapsed_s={time.perf_counter() - started:.1f} "
        f"rmse_mV={result.rmse * 1000:.3f} accepted_runs={result.n_accepted_runs}/"
        f"{result.n_total_runs}",
        flush=True,
    )
    return result


def run_pydma_for_cell(
    dataset_root: Path,
    records: list[FileRecord],
    guide_entries: list[GuideEntry],
    group_id: str,
    cell_id: str,
    use_cuda: bool = False,
    fit_targets: set[tuple[int, str]] | None = None,
    optimizer_options: dict[str, int] | None = None,
    allow_raw_comparison: bool = False,
    half_cell_reference_root: Path | None = None,
) -> list[dict[str, Any]]:
    half_cell = (half_cell_reference_root or dataset_root) / "Half_Cell" / "Half Cell"
    if not half_cell.is_dir():
        raise FileNotFoundError(f"Half-cell reference directory is missing: {half_cell}")
    charge_refs = (
        load_ocp(
            half_cell / "CH_HC_anode.csv",
            electrode_type="anode",
            soc_col="SoC",
            voltage_col="Volts",
        ),
        load_ocp(
            half_cell / "CH_HC_cathode.csv",
            electrode_type="cathode",
            soc_col="SoC",
            voltage_col="Volts",
        ),
    )
    discharge_anode_frame = pd.read_csv(half_cell / "DCH_HC_anode.csv", usecols=["SoC", "Volts"])
    finite = np.isfinite(discharge_anode_frame["SoC"].to_numpy(dtype=float)) & np.isfinite(
        discharge_anode_frame["Volts"].to_numpy(dtype=float)
    )
    if not finite.any():
        raise ValueError("No finite discharge-anode reference pairs are available.")
    discharge_refs = (
        ElectrodeOCP(
            soc=discharge_anode_frame.loc[finite, "SoC"].to_numpy(dtype=float),
            voltage=discharge_anode_frame.loc[finite, "Volts"].to_numpy(dtype=float),
            name="DCH_HC_anode_valid_pairs",
            electrode_type="anode",
        ),
        load_ocp(
            half_cell / "DHC_HC_cathode.csv",
            electrode_type="cathode",
            soc_col="SoC",
            voltage_col="Volts",
        ),
    )
    file_records: defaultdict[int, list[FileRecord]] = defaultdict(list)
    all_file_records: defaultdict[int, list[FileRecord]] = defaultdict(list)
    for record in records:
        if (
            record.group_id == group_id
            and record.cell_id == cell_id
            and record.file_id is not None
        ):
            all_file_records[record.file_id].append(record)
            if record.file_type == "csv":
                file_records[record.file_id].append(record)
    entries = sorted(
        (
            entry
            for entry in guide_entries
            if entry.group_id == group_id
            and entry.cell_id == cell_id
            and (entry.stage.casefold().startswith("bol") or re.search(r"rpt", entry.stage, re.I))
        ),
        key=lambda entry: (_stage_number(entry.stage), entry.file_id),
    )
    if fit_targets is not None:
        missing_targets = {
            (file_id, direction)
            for file_id, direction in fit_targets
            if direction not in {"charge", "discharge"}
            or not any(entry.file_id == file_id and not entry.stage.casefold().startswith("bol") for entry in entries)
        }
        if missing_targets:
            raise ValueError(f"Fit targets are not Guide-listed RPT directions: {sorted(missing_targets)}")
        requested_directions = {direction for _, direction in fit_targets}
        reference_ids = {
            entry.file_id
            for entry in entries
            if entry.stage.casefold().startswith("bol") and entry.file_id is not None
        }
        scan_file_ids = {file_id for file_id, _ in fit_targets} | reference_ids
    else:
        requested_directions = {"charge", "discharge"}
        scan_file_ids = {entry.file_id for entry in entries}
    fit_options = optimizer_options or {}
    curves: dict[tuple[int, str], dict[str, Any]] = {}
    curve_selection: dict[int, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for file_id in sorted({entry.file_id for entry in entries}):
        if file_id not in scan_file_ids:
            continue
        candidates = []
        for record in sorted(
            file_records.get(file_id, []),
            key=lambda item: item.relative_path.casefold(),
        ):
            print(
                f"PyDMA curve preflight Group {group_id} Cell {cell_id} "
                f"file_id={file_id} source={record.relative_path}",
                flush=True,
            )
            candidate_curves = {}
            candidate_audits = {}
            for direction, step in (("charge", 10), ("discharge", 8)):
                curve, audit = _read_pocv_with_audit(
                    _record_source_path(dataset_root, record),
                    step,
                    direction,
                )
                candidate_curves[direction] = curve
                candidate_audits[direction] = audit
            valid_count = sum(curve is not None for curve in candidate_curves.values())
            point_count = sum(
                len(curve[0]) for curve in candidate_curves.values() if curve is not None
            )
            candidates.append(
                {
                    "record": record,
                    "curves": candidate_curves,
                    "audits": candidate_audits,
                    "valid_direction_count": valid_count,
                    "selected_point_count": point_count,
                }
            )
        # Candidates are path-sorted; the selector keeps the first on ties.
        selected = _select_curve_candidate(candidates)
        curve_selection[file_id] = {
            "selection_policy": "Choose the duplicate CSV with the most valid directions, then the most selected curve points; ties use case-insensitive relative-path order.",
            "selected_source_file": (
                selected["record"].relative_path if selected is not None else ""
            ),
            "candidates": [
                {
                    "source_file": item["record"].relative_path,
                    "valid_direction_count": item["valid_direction_count"],
                    "selected_point_count": item["selected_point_count"],
                    "direction_audits": item["audits"],
                }
                for item in candidates
            ],
        }
        if selected is None:
            continue
        print(
            f"PyDMA duplicate selection Group {group_id} Cell {cell_id} file_id={file_id} "
            f"source={selected['record'].relative_path} "
            f"valid_directions={selected['valid_direction_count']} "
            f"selected_points={selected['selected_point_count']}",
            flush=True,
        )
        for direction, curve in selected["curves"].items():
            if curve is not None:
                curves[(file_id, direction)] = {
                    "record": selected["record"],
                    "capacity": curve[0],
                    "voltage": curve[1],
                    "audit": selected["audits"][direction],
                    "duplicate_selection": curve_selection[file_id],
                }

    # Record unavailable sources/directions rather than silently omitting them.
    for entry in entries:
        if fit_targets is not None and not any(
            target_file_id == entry.file_id and target_direction == direction
            for target_file_id, target_direction in fit_targets
            for direction in ("charge", "discharge")
        ):
            continue
        source_record = next(
            iter(file_records.get(entry.file_id, [])),
            next(iter(all_file_records.get(entry.file_id, [])), None),
        )
        for direction, step in (("charge", 10), ("discharge", 8)):
            if (entry.file_id, direction) in curves:
                continue
            audit = curve_selection.get(entry.file_id, {})
            source_candidates = audit.get("candidates", [])
            direction_audits = [
                candidate.get("direction_audits", {}).get(direction, {})
                for candidate in source_candidates
            ]
            error = (
                "; ".join(
                    f"{item.get('source_file', '')}: {item.get('status', 'unavailable')}"
                    + (f" ({item['error']})" if item.get("error") else "")
                    for item in direction_audits
                )
                or "No CSV source is available; matching MAT content is not decoded."
            )
            if source_record is not None:
                output.append(
                    _result_row(
                        group_id,
                        cell_id,
                        entry,
                        source_record,
                        direction,
                        step,
                        None,
                        "not_run_curve_unavailable",
                        None,
                        None,
                        None,
                        curve_audit={
                            "file_id": entry.file_id,
                            "direction_audits": direction_audits,
                            "duplicate_selection": audit,
                        },
                        error=error,
                    )
                )

    for direction in ("charge", "discharge"):
        if direction not in requested_directions:
            continue
        references = charge_refs if direction == "charge" else discharge_refs
        target_entries = [
            entry
            for entry in entries
            if (fit_targets is None or (entry.file_id, direction) in fit_targets)
            and (entry.file_id, direction) in curves
        ]
        bol = [
            entry
            for entry in entries
            if entry.stage.casefold().startswith("bol")
            and (entry.file_id, direction) in curves
        ]
        ordered = bol + [entry for entry in target_entries if entry not in bol]
        if not bol:
            for entry in ordered:
                curve = curves[(entry.file_id, direction)]
                output.append(
                    _result_row(
                        group_id,
                        cell_id,
                        entry,
                        curve["record"],
                        direction,
                        10 if direction == "charge" else 8,
                        None,
                        "not_run_no_valid_bol_reference",
                        None,
                        None,
                        None,
                        curve_audit=curve["audit"],
                        error="RPT curve passed extraction, but no same-direction BoL curve is available for the reference fit.",
                    )
                )
            continue
        reference_entry = bol[0]
        reference_curve = curves[(reference_entry.file_id, direction)]
        reference_record = reference_curve["record"]
        capacity = reference_curve["capacity"]
        reference_error = ""
        try:
            reference_fit = _fit_one(
                direction,
                capacity,
                reference_curve["voltage"],
                *references,
                seed=211,
                context=f"Group {group_id} Cell {cell_id} BoL file_id={reference_entry.file_id}",
                use_cuda=use_cuda,
                **fit_options,
            )
        except Exception as error:
            reference_fit = None
            reference_error = f"{type(error).__name__}: {error}"
        reference_accepted = pydma_fit_passes_gate(
            rmse_mV=(
                reference_fit.rmse * 1000
                if reference_fit is not None
                else None
            ),
            accepted_runs=(
                reference_fit.n_accepted_runs
                if reference_fit is not None
                else None
            ),
            is_accepted=(
                reference_fit.is_accepted if reference_fit is not None else False
            ),
        )
        output.append(
            _result_row(
                group_id,
                cell_id,
                reference_entry,
                reference_record,
                direction,
                10 if direction == "charge" else 8,
                reference_fit,
                "reference_baseline" if reference_accepted else (
                    "withheld_baseline_fit_exception" if reference_error else "withheld_baseline_fit_failed"
                ),
                0.0 if reference_accepted else None,
                0.0 if reference_accepted else None,
                0.0 if reference_accepted else None,
                curve_audit=reference_curve["audit"],
                error=reference_error,
            )
        )
        for entry in ordered:
            if entry.file_id == reference_entry.file_id or entry.stage.casefold().startswith("bol"):
                continue
            curve = curves[(entry.file_id, direction)]
            capacity = curve["capacity"]
            try:
                fit = _fit_one(
                    direction,
                    capacity,
                    curve["voltage"],
                    *references,
                    seed=211,
                    context=f"Group {group_id} Cell {cell_id} {entry.stage} file_id={entry.file_id}",
                    use_cuda=use_cuda,
                    **fit_options,
                )
            except Exception as error:
                output.append(
                    _result_row(
                        group_id,
                        cell_id,
                        entry,
                        curve["record"],
                        direction,
                        10 if direction == "charge" else 8,
                        None,
                        "withheld_fit_exception",
                        None,
                        None,
                        None,
                        curve_audit=curve["audit"],
                        error=f"{type(error).__name__}: {error}",
                    )
                )
                continue
            accepted = pydma_fit_passes_gate(
                rmse_mV=fit.rmse * 1000,
                accepted_runs=fit.n_accepted_runs,
                is_accepted=fit.is_accepted,
            )
            comparison_error = ""
            comparison_allowed = (
                accepted and reference_accepted
            ) or (
                allow_raw_comparison
                and fit is not None
                and reference_fit is not None
            )
            if comparison_allowed:
                try:
                    comparison = DMAAnalyzer(
                        config=DMAConfig(direction=direction),
                        anode=references[0],
                        cathode=references[1],
                    ).compare_with_reference(
                        reference_params=reference_fit.fitted_params,
                        current_params=fit.fitted_params,
                        reference_capacity=float(np.ptp(reference_curve["capacity"])),
                        current_capacity=float(np.ptp(capacity)),
                    )
                    values = (
                        comparison.lli * 100,
                        comparison.lam_anode * 100,
                        comparison.lam_cathode * 100,
                    )
                    status = (
                        "diagnostic_estimate_accepted"
                        if accepted and reference_accepted
                        else "raw_diagnostic_estimate"
                    )
                except Exception as error:
                    values = (None, None, None)
                    status = "withheld_comparison_exception"
                    comparison_error = f"{type(error).__name__}: {error}"
            else:
                values = (None, None, None)
                status = (
                    "withheld_fit_rejected"
                    if not accepted
                    else "withheld_baseline_fit_failed"
                )
            output.append(
                _result_row(
                    group_id,
                    cell_id,
                    entry,
                    curve["record"],
                    direction,
                    10 if direction == "charge" else 8,
                    fit,
                    status,
                    *values,
                    curve_audit=curve["audit"],
                    error=comparison_error,
                )
            )
    return output


def run_pydma_for_available_cells(
    dataset_root: Path,
    records: list[FileRecord],
    guide_entries: list[GuideEntry],
) -> list[dict[str, Any]]:
    csv_cell_ids = {
        (record.group_id, record.cell_id)
        for record in records
        if record.file_type == "csv"
        and record.file_id is not None
        and record.group_id
        and record.cell_id
    }
    guide_cell_ids = sorted(
        {
            (entry.group_id, entry.cell_id)
            for entry in guide_entries
            if (entry.group_id, entry.cell_id) in csv_cell_ids
            and (
                entry.stage.casefold().startswith("bol")
                or re.search(r"rpt", entry.stage, re.IGNORECASE)
            )
        },
        key=lambda item: (int(item[0]), int(item[1])),
    )
    output: list[dict[str, Any]] = []
    for group_id, cell_id in guide_cell_ids:
        print(f"Running PyDMA for documented Group {group_id}, Cell {cell_id}", flush=True)
        output.extend(
            run_pydma_for_cell(
                dataset_root,
                records,
                guide_entries,
                group_id,
                cell_id,
            )
        )
    return output


def _result_row(
    group_id: str,
    cell_id: str,
    entry: GuideEntry,
    record: FileRecord,
    direction: str,
    step: int,
    result: Any,
    status: str,
    lli: float | None,
    lam_anode: float | None,
    lam_cathode: float | None,
    *,
    curve_audit: dict[str, Any] | None = None,
    error: str = "",
) -> dict[str, Any]:
    return {
        "group_id": f"Group {group_id}",
        "cell_id": f"Cell {cell_id}",
        "checkup": "BoL" if entry.stage.casefold().startswith("bol") else f"RPT{_stage_number(entry.stage)[0]}",
        "direction": direction,
        "source_file": Path(record.relative_path).name,
        "test_step": step,
        "fit_rmse_mV": "" if result is None else float(result.rmse * 1000),
        "fit_accepted_by_pydma_15mV": (
            False
            if result is None
            else pydma_fit_passes_gate(
                rmse_mV=result.rmse * 1000,
                accepted_runs=result.n_accepted_runs,
                is_accepted=result.is_accepted,
            )
        ),
        "fit_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
        "accepted_runs": "" if result is None else int(result.n_accepted_runs),
        "total_runs": "" if result is None else int(result.n_total_runs),
        "diagnostic_label_status": status,
        "LLI_percent_vs_BoL": "" if lli is None else float(lli),
        "LAM_anode_percent_vs_BoL": "" if lam_anode is None else float(lam_anode),
        "LAM_cathode_percent_vs_BoL": "" if lam_cathode is None else float(lam_cathode),
        "curve_audit_json": json.dumps(curve_audit or {}, ensure_ascii=False, allow_nan=False),
        "error": error,
        "quality_note": (
            "Model-based PyDMA OCV diagnostic estimate, not ground truth."
            if status == "diagnostic_estimate_accepted"
            else (
                f"Diagnostic estimate withheld: {status}; {error}"
                if result is None or error
                else f"Diagnostic estimate withheld: {status}; candidate fit RMSE={result.rmse * 1000:.3f} mV."
            )
        ),
    }
