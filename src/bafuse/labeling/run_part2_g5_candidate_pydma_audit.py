from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .pydma_crosswalk import is_unique_checkpoint, manifest_path, source_order
from .pydma_policy import pydma_fit_passes_gate
from .pydma_runner import _fit_one, _read_pocv_with_audit


_CELLS = ("G5-Cell5", "G5-Cell6", "G5-Cell16")
_DIRECTIONS = {
    "discharge": {"step": 8, "current": "negative"},
    "charge": {"step": 10, "current": "positive"},
}
_CANDIDATE_RPTS = (1, 2, 3, 4)
_OPTIMIZER_OPTIONS = {"maxiter": 50, "popsize": 30, "max_tries": 10}
_MAPPING_DISCLAIMER = (
    "Audit-only candidate mapping: TPG5.3 is Guide-listed for RPT1-RPT4; "
    "the four valid Step 8/10 curves are assigned in file order. This mapping "
    "is not confirmed by original Maccor logs and must not be used as a "
    "training label or merged into canonical PyDMA tables."
)


@dataclass(frozen=True)
class CurveSegment:
    capacity: np.ndarray
    voltage: np.ndarray
    audit: dict[str, Any]


def _split_monotone_segments(
    capacity: np.ndarray,
    voltage: np.ndarray,
    source_rows: np.ndarray,
    direction: str,
    step: int,
) -> tuple[list[CurveSegment], list[dict[str, Any]]]:
    if capacity.ndim != 1 or voltage.shape != capacity.shape or source_rows.shape != capacity.shape:
        raise ValueError("Capacity, voltage, and source row arrays must be matching 1D arrays.")
    finite = np.isfinite(capacity) & np.isfinite(voltage)
    capacity = capacity[finite].astype(np.float64, copy=False)
    voltage = voltage[finite].astype(np.float64, copy=False)
    source_rows = source_rows[finite].astype(np.int64, copy=False)
    if capacity.size == 0:
        return [], []

    boundaries = np.flatnonzero(np.diff(capacity) <= 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, capacity.size]
    valid_segments: list[CurveSegment] = []
    audits: list[dict[str, Any]] = []
    for segment_index, (start, end) in enumerate(zip(starts, ends), start=1):
        segment_capacity = capacity[start:end]
        segment_voltage = voltage[start:end]
        segment_rows = source_rows[start:end]
        is_valid = bool(
            segment_capacity.size >= 10
            and np.all(np.diff(segment_capacity) > 0)
            and np.ptp(segment_capacity) > 0
        )
        audit = {
            "direction": direction,
            "step": step,
            "segment_index": segment_index,
            "source_row_start": int(segment_rows[0]),
            "source_row_end": int(segment_rows[-1]),
            "sample_count": int(segment_capacity.size),
            "capacity_min": float(segment_capacity.min()),
            "capacity_max": float(segment_capacity.max()),
            "voltage_min": float(segment_voltage.min()),
            "voltage_max": float(segment_voltage.max()),
            "valid_monotone_curve": is_valid,
        }
        audits.append(audit)
        if is_valid:
            valid_segments.append(
                CurveSegment(segment_capacity, segment_voltage, audit)
            )
    return valid_segments, audits


def _read_candidate_curves(
    path: Path,
) -> tuple[dict[str, list[CurveSegment]], list[dict[str, Any]], int]:
    selected: dict[str, dict[str, list[np.ndarray]]] = {
        direction: {"capacity": [], "voltage": [], "source_rows": []}
        for direction in _DIRECTIONS
    }
    source_rows_read = 0
    for chunk in pd.read_csv(
        path,
        usecols=["Step", "Amphr", "Amps", "Volts"],
        chunksize=250_000,
    ):
        step = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(dtype=np.float64)
        capacity = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        current = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for direction, spec in _DIRECTIONS.items():
            if spec["current"] == "positive":
                current_mask = (current > 0) & (np.abs(current) <= 0.2)
            else:
                current_mask = (current < 0) & (np.abs(current) <= 0.2)
            mask = (
                (step == spec["step"])
                & current_mask
                & np.isfinite(capacity)
                & np.isfinite(voltage)
            )
            indices = np.flatnonzero(mask)
            if indices.size == 0:
                continue
            selected[direction]["capacity"].append(capacity[indices])
            selected[direction]["voltage"].append(voltage[indices])
            selected[direction]["source_rows"].append(
                indices.astype(np.int64) + source_rows_read
            )
        source_rows_read += len(chunk)

    curves: dict[str, list[CurveSegment]] = {}
    audit_rows: list[dict[str, Any]] = []
    for direction, spec in _DIRECTIONS.items():
        chunks = selected[direction]
        if chunks["capacity"]:
            capacity = np.concatenate(chunks["capacity"])
            voltage = np.concatenate(chunks["voltage"])
            source_rows = np.concatenate(chunks["source_rows"])
        else:
            capacity = np.asarray([], dtype=np.float64)
            voltage = np.asarray([], dtype=np.float64)
            source_rows = np.asarray([], dtype=np.int64)
        valid, audits = _split_monotone_segments(
            capacity, voltage, source_rows, direction, spec["step"]
        )
        curves[direction] = valid
        audit_rows.extend(audits)
    return curves, audit_rows, source_rows_read


def _require_four_candidate_curves(
    curves: dict[str, list[CurveSegment]], source_file: str
) -> None:
    for direction in _DIRECTIONS:
        count = len(curves.get(direction, []))
        if count != len(_CANDIDATE_RPTS):
            raise ValueError(
                f"{source_file} has {count} valid {direction} Step "
                f"{_DIRECTIONS[direction]['step']} segments; expected exactly "
                f"{len(_CANDIDATE_RPTS)} for the audit-only RPT1-RPT4 mapping."
            )


def _map_candidate_rpts(
    curves: dict[str, list[CurveSegment]], source_file: str
) -> dict[str, dict[int, CurveSegment]]:
    _require_four_candidate_curves(curves, source_file)
    return {
        direction: dict(zip(_CANDIDATE_RPTS, curves[direction], strict=True))
        for direction in _DIRECTIONS
    }


def _candidate_plan(
    manifest: pd.DataFrame, part2_root: Path
) -> dict[str, dict[str, Any]]:
    plans: dict[str, dict[str, Any]] = {}
    roots = {"2": part2_root}
    for physical_cell_id in _CELLS:
        cell_rows = manifest[
            (manifest["dataset_part"] == "2")
            & (manifest["physical_cell_id"] == physical_cell_id)
            & (manifest["group_id"] == "5")
        ].to_dict(orient="records")
        baselines = [
            row
            for row in cell_rows
            if row["measurement_id"].casefold() == "bol"
            and is_unique_checkpoint(row)
        ]
        if not baselines:
            raise ValueError(f"No unique Guide-mapped Group 5 BoL for {physical_cell_id}.")
        baseline = min(baselines, key=source_order)
        if Path(baseline["source_file"]).name.casefold() != (
            f"tpg5.1-{physical_cell_id.casefold().replace('g5-', '')}.csv"
        ):
            raise ValueError(
                f"Expected the earliest local BoL source to be TPG5.1 for "
                f"{physical_cell_id}, got {baseline['source_file']}."
            )

        candidates = []
        for row in cell_rows:
            normalized_stage = re.sub(r"\s+", "", row["guide_stage"]).casefold()
            normalized_stage = normalized_stage.replace("–", "-").replace("—", "-")
            if (
                row["procedure_file_id"] == "3"
                and row["measurement_mapping_status"] == "ambiguous_compound_stage"
                and "rpt1-rpt4" in normalized_stage
                and row["source_mapping_status"] == "unique_source_file"
                and row["source_file_count"] == "1"
            ):
                candidates.append(row)
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one Guide-listed TPG5.3 RPT1-RPT4 compound entry for "
                f"{physical_cell_id}; found {len(candidates)}."
            )
        target = candidates[0]
        plans[physical_cell_id] = {
            "baseline_row": baseline,
            "target_row": target,
            "baseline_path": manifest_path(baseline, roots),
            "target_path": manifest_path(target, roots),
        }
    return plans


def _load_half_cell_references(part1_root: Path) -> dict[str, tuple[Any, Any]]:
    from pydma.electrodes.electrode import ElectrodeOCP
    from pydma.preprocessing.loader import load_ocp

    half_cell = part1_root / "Half_Cell" / "Half Cell"
    if not half_cell.is_dir():
        raise FileNotFoundError(f"Half-cell reference directory is missing: {half_cell}")
    charge = (
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
    discharge_anode = pd.read_csv(
        half_cell / "DCH_HC_anode.csv", usecols=["SoC", "Volts"]
    )
    finite = np.isfinite(discharge_anode["SoC"].to_numpy(dtype=float)) & np.isfinite(
        discharge_anode["Volts"].to_numpy(dtype=float)
    )
    if not finite.any():
        raise ValueError("No finite discharge-anode reference pairs are available.")
    discharge = (
        ElectrodeOCP(
            soc=discharge_anode.loc[finite, "SoC"].to_numpy(dtype=float),
            voltage=discharge_anode.loc[finite, "Volts"].to_numpy(dtype=float),
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
    return {"charge": charge, "discharge": discharge}


def _fit_record(
    *,
    physical_cell_id: str,
    candidate_rpt: int,
    source_file: str,
    baseline_source_file: str,
    direction: str,
    segment: CurveSegment,
    references: tuple[Any, Any],
    baseline_result: Any,
    baseline_curve: tuple[np.ndarray, np.ndarray] | None,
    baseline_error: str,
) -> dict[str, Any]:
    from pydma import DMAAnalyzer, DMAConfig

    result = None
    fit_error = ""
    try:
        result = _fit_one(
            direction,
            segment.capacity,
            segment.voltage,
            *references,
            seed=211,
            context=f"Group 5 {physical_cell_id} candidate RPT{candidate_rpt}",
            use_cuda=True,
            **_OPTIMIZER_OPTIONS,
        )
    except Exception as error:
        fit_error = f"{type(error).__name__}: {error}"

    accepted = bool(
        result is not None
        and pydma_fit_passes_gate(
            rmse_mV=float(result.rmse * 1000),
            accepted_runs=int(result.n_accepted_runs),
            is_accepted=bool(result.is_accepted),
        )
    )
    baseline_accepted = bool(
        baseline_result is not None
        and pydma_fit_passes_gate(
            rmse_mV=float(baseline_result.rmse * 1000),
            accepted_runs=int(baseline_result.n_accepted_runs),
            is_accepted=bool(baseline_result.is_accepted),
        )
    )
    values: tuple[float | None, float | None, float | None] = (None, None, None)
    comparison_error = ""
    if result is not None and baseline_accepted and baseline_curve is not None:
        try:
            comparison = DMAAnalyzer(
                config=DMAConfig(direction=direction),
                anode=references[0],
                cathode=references[1],
            ).compare_with_reference(
                reference_params=baseline_result.fitted_params,
                current_params=result.fitted_params,
                reference_capacity=float(np.ptp(baseline_curve[0])),
                current_capacity=float(np.ptp(segment.capacity)),
            )
            values = (
                float(comparison.lli * 100),
                float(comparison.lam_anode * 100),
                float(comparison.lam_cathode * 100),
            )
        except Exception as error:
            comparison_error = f"{type(error).__name__}: {error}"

    if fit_error:
        status = "candidate_fit_exception"
    elif comparison_error:
        status = "candidate_comparison_exception"
    elif not baseline_accepted:
        status = "withheld_baseline_fit_not_accepted"
    elif accepted:
        status = "audit_only_candidate_accepted"
    elif result is not None and values[0] is not None:
        status = "audit_only_raw_candidate"
    else:
        status = "candidate_fit_not_accepted"

    audit = segment.audit
    return {
        "dataset_part": "2",
        "group_id": "5",
        "physical_cell_id": physical_cell_id,
        "candidate_rpt": candidate_rpt,
        "identity_status": "provisional_guide_and_step_order_not_log_verified",
        "mapping_basis": _MAPPING_DISCLAIMER,
        "source_file": source_file,
        "baseline_source_file": baseline_source_file,
        "direction": direction,
        "test_step": audit["step"],
        "candidate_segment_index": audit["segment_index"],
        "source_row_start": audit["source_row_start"],
        "source_row_end": audit["source_row_end"],
        "curve_sample_count": audit["sample_count"],
        "curve_capacity_span_native": audit["capacity_max"] - audit["capacity_min"],
        "fit_rmse_mV": float(result.rmse * 1000) if result is not None else None,
        "fit_is_accepted": bool(result.is_accepted) if result is not None else False,
        "accepted_runs": int(result.n_accepted_runs) if result is not None else None,
        "total_runs": int(result.n_total_runs) if result is not None else None,
        "passes_strict_rmse15_gate": accepted,
        "baseline_rmse_mV": (
            float(baseline_result.rmse * 1000)
            if baseline_result is not None
            else None
        ),
        "baseline_accepted_runs": (
            int(baseline_result.n_accepted_runs)
            if baseline_result is not None
            else None
        ),
        "baseline_passes_strict_rmse15_gate": baseline_accepted,
        "LLI_percent_vs_BoL": values[0],
        "LAM_anode_percent_vs_BoL": values[1],
        "LAM_cathode_percent_vs_BoL": values[2],
        "audit_status": status,
        "is_training_label": False,
        "error": fit_error or comparison_error or baseline_error,
    }


def run_candidate_audit(
    part1_root: Path,
    part2_root: Path,
    manifest_path_arg: Path,
    output_dir: Path,
    expected_gpu: str | None = None,
) -> dict[str, Any]:
    import torch

    part1_root = part1_root.resolve()
    part2_root = part2_root.resolve()
    manifest_path_arg = manifest_path_arg.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("The Group 5 candidate PyDMA audit requires CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if expected_gpu and expected_gpu.casefold() not in device_name.casefold():
        raise RuntimeError(
            f"Expected GPU containing {expected_gpu!r}, found {device_name!r}."
        )

    manifest = pd.read_csv(
        manifest_path_arg, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    required_columns = {
        "dataset_part",
        "group_id",
        "cell_id",
        "physical_cell_id",
        "procedure_file_id",
        "guide_stage",
        "measurement_id",
        "measurement_mapping_status",
        "source_file",
        "source_file_count",
        "source_mapping_status",
        "source_file_size_bytes",
    }
    missing_columns = required_columns - set(manifest.columns)
    if missing_columns:
        raise ValueError(
            f"Guide manifest lacks required fields: {sorted(missing_columns)}"
        )

    plans = _candidate_plan(manifest, part2_root)
    target_curves: dict[str, dict[str, dict[int, CurveSegment]]] = {}
    boundary_audit_rows: list[dict[str, Any]] = []
    source_stats_before: dict[str, dict[str, int]] = {}
    for physical_cell_id, plan in plans.items():
        for source in (plan["baseline_path"], plan["target_path"]):
            source_stats_before[str(source)] = {
                "size_bytes": source.stat().st_size,
                "modified_ns": source.stat().st_mtime_ns,
            }
        source = plan["target_path"]
        curves, audit_rows, source_rows = _read_candidate_curves(source)
        candidate_segments = _map_candidate_rpts(
            curves, plan["target_row"]["source_file"]
        )
        target_curves[physical_cell_id] = candidate_segments
        for direction in _DIRECTIONS:
            candidate_rpts_by_segment = {
                segment.audit["segment_index"]: candidate_rpt
                for candidate_rpt, segment in candidate_segments[direction].items()
            }
            for audit in audit_rows:
                if audit["direction"] != direction:
                    continue
                candidate_rpt = candidate_rpts_by_segment.get(
                    audit["segment_index"]
                )
                boundary_audit_rows.append(
                    {
                        "physical_cell_id": physical_cell_id,
                        "candidate_rpt": candidate_rpt,
                        **audit,
                        "mapping_status": (
                            "audit_only_candidate_order_not_maccor_log_verified"
                            if candidate_rpt is not None
                            else "not_mapped_invalid_curve_segment"
                        ),
                        "guide_stage": plan["target_row"]["guide_stage"],
                        "source_file": plan["target_row"]["source_file"],
                        "source_rows_read": source_rows,
                    }
                )

    references = _load_half_cell_references(part1_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(boundary_audit_rows).to_csv(
        output_dir / "candidate_curve_boundaries.csv", index=False
    )

    baseline_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for physical_cell_id, plan in plans.items():
        baseline_curves: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        baseline_results: dict[str, Any] = {}
        baseline_errors: dict[str, str] = {}
        for direction, spec in _DIRECTIONS.items():
            curve, curve_audit = _read_pocv_with_audit(
                plan["baseline_path"], spec["step"], direction
            )
            baseline_curves[direction] = curve
            result = None
            error_text = ""
            if curve is None:
                error_text = "No valid local BoL curve passed the standard Step/current filter."
            else:
                try:
                    result = _fit_one(
                        direction,
                        curve[0],
                        curve[1],
                        *references[direction],
                        seed=211,
                        context=f"Group 5 {physical_cell_id} local BoL",
                        use_cuda=True,
                        **_OPTIMIZER_OPTIONS,
                    )
                except Exception as error:
                    error_text = f"{type(error).__name__}: {error}"
            baseline_results[direction] = result
            baseline_errors[direction] = error_text
            gate_passed = bool(
                result is not None
                and pydma_fit_passes_gate(
                    rmse_mV=float(result.rmse * 1000),
                    accepted_runs=int(result.n_accepted_runs),
                    is_accepted=bool(result.is_accepted),
                )
            )
            baseline_rows.append(
                {
                    "dataset_part": "2",
                    "group_id": "5",
                    "physical_cell_id": physical_cell_id,
                    "source_file": plan["baseline_row"]["source_file"],
                    "guide_stage": plan["baseline_row"]["guide_stage"],
                    "direction": direction,
                    "test_step": spec["step"],
                    "curve_audit": json.dumps(curve_audit, ensure_ascii=False),
                    "fit_rmse_mV": (
                        float(result.rmse * 1000) if result is not None else None
                    ),
                    "fit_is_accepted": (
                        bool(result.is_accepted) if result is not None else False
                    ),
                    "accepted_runs": (
                        int(result.n_accepted_runs) if result is not None else None
                    ),
                    "total_runs": (
                        int(result.n_total_runs) if result is not None else None
                    ),
                    "passes_strict_rmse15_gate": gate_passed,
                    "audit_status": (
                        "reference_baseline_accepted"
                        if gate_passed
                        else "reference_baseline_not_accepted"
                    ),
                    "is_training_label": False,
                    "error": error_text,
                }
            )

        for direction in _DIRECTIONS:
            for candidate_rpt, segment in target_curves[physical_cell_id][
                direction
            ].items():
                candidate_rows.append(
                    _fit_record(
                        physical_cell_id=physical_cell_id,
                        candidate_rpt=candidate_rpt,
                        source_file=plan["target_row"]["source_file"],
                        baseline_source_file=plan["baseline_row"]["source_file"],
                        direction=direction,
                        segment=segment,
                        references=references[direction],
                        baseline_result=baseline_results[direction],
                        baseline_curve=baseline_curves[direction],
                        baseline_error=baseline_errors[direction],
                    )
                )

    source_stats_after: dict[str, dict[str, int]] = {}
    for physical_cell_id, plan in plans.items():
        for source in (plan["baseline_path"], plan["target_path"]):
            source_stats_after[str(source)] = {
                "size_bytes": source.stat().st_size,
                "modified_ns": source.stat().st_mtime_ns,
            }
    if source_stats_before != source_stats_after:
        raise RuntimeError("A TPG5.3 source file changed during the audit.")

    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline_fits.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(
        output_dir / "candidate_pydma_fits.csv", index=False
    )
    fit_status_counts = Counter(row["audit_status"] for row in candidate_rows)
    accepted_count = sum(row["passes_strict_rmse15_gate"] for row in candidate_rows)
    summary = {
        "experiment": "Part 2 Group 5 audit-only candidate PyDMA fits",
        "dataset_scope": "G5-Cell5, G5-Cell6, G5-Cell16; candidate RPT1-RPT4 from TPG5.3",
        "identity_status": (
            "Guide/step-order candidate mapping; not verified by original Maccor logs."
        ),
        "mapping_disclaimer": _MAPPING_DISCLAIMER,
        "training_integration": False,
        "canonical_label_tables_modified": False,
        "candidate_cell_count": len(plans),
        "candidate_rpt_count_per_cell": len(_CANDIDATE_RPTS),
        "directional_candidate_fit_count": len(candidate_rows),
        "strict_gate_accepted_fit_count": int(accepted_count),
        "candidate_status_counts": dict(sorted(fit_status_counts.items())),
        "baseline_fit_count": len(baseline_rows),
        "source_files_modified": False,
        "source_files_before": source_stats_before,
        "source_files_after": source_stats_after,
        "runtime": {
            "device": "cuda:0",
            "gpu_name": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_options": _OPTIMIZER_OPTIONS,
        },
        "fit_policy": {
            "local_bol": "Earliest unique Guide-mapped Part 2 BoL source (TPG5.1) per cell.",
            "candidate_targets": "Four valid monotone Step 8 discharge and Step 10 charge segments from TPG5.3, ordered in file order and provisionally mapped to RPT1-RPT4.",
            "current_filter": "Step 8 discharge: negative current, |Amps| <= 0.2 A; Step 10 charge: positive current, |Amps| <= 0.2 A.",
            "segment_rule": "Finite Amphr/Volts pairs split at Amphr non-increase; each candidate curve must have at least 10 strictly increasing capacity samples.",
            "accepted_gate": "Fit is accepted, RMSE <15 mV, and at least 3 accepted runs.",
        },
        "input_manifest": str(manifest_path_arg),
        "part1_root": str(part1_root),
        "part2_root": str(part2_root),
        "limitations": [
            "The Guide maps TPG5.3 to RPT1-RPT4 as a compound stage; file-order segmentation is candidate, not an original-Maccor-log-confirmed identity.",
            "The Guide overlap for G5-Cell6 RPT24 is outside this RPT1-RPT4 candidate audit and remains unresolved.",
            "Candidate values are not accepted as SOH or degradation training labels and are not merged into canonical PyDMA result files.",
            "A passing optimizer gate does not validate the candidate RPT identity.",
        ],
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Part 2 Group 5 audit-only candidate PyDMA fits\n\n"
        f"{_MAPPING_DISCLAIMER}\n\n"
        "This run is separate from all training labels. It uses each cell's "
        "earliest unique local BoL (`TPG5.1`) and the four valid monotone "
        "Step 8/10 segments in `TPG5.3`, provisionally ordered as RPT1-RPT4. "
        "The original Maccor logs were not available to confirm these "
        "within-file RPT boundaries. The Guide's Cell 6 RPT24 overlap remains "
        "outside this audit. A passing PyDMA RMSE/run-count gate does not "
        "validate the candidate identity.\n\n"
        "Files: `candidate_curve_boundaries.csv`, `baseline_fits.csv`, "
        "`candidate_pydma_fits.csv`, and `run_summary.json`. Candidate values "
        "must not be merged into canonical PyDMA tables or model training "
        "without later identity validation.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate audit-only candidate PyDMA fits for Part 2 Group 5 RPT1-RPT4."
    )
    parser.add_argument("--part1-root", type=Path, required=True)
    parser.add_argument("--part2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", default=None)
    args = parser.parse_args()
    summary = run_candidate_audit(
        args.part1_root,
        args.part2_root,
        args.manifest,
        args.output,
        args.expected_gpu,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
