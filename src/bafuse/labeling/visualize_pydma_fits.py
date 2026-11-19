from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from contextlib import redirect_stdout
from importlib.metadata import version
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from bafuse.config import load_project_paths

_LABEL_COLUMNS = {
    "lli_pct_label_percent": "LLI_percent_vs_BoL",
    "lam_anode_pct_label_percent": "LAM_anode_percent_vs_BoL",
    "lam_cathode_pct_label_percent": "LAM_cathode_percent_vs_BoL",
}
_PREDICTION_COLUMNS = {
    "lli_pct_label_percent": "lli_pct_prediction_percent",
    "lam_anode_pct_label_percent": "lam_anode_pct_prediction_percent",
    "lam_cathode_pct_label_percent": "lam_cathode_pct_prediction_percent",
}


def _canonical_rpt(value: Any) -> str:
    return "".join(str(value).split()).casefold()


def _part_number(value: Any) -> str:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise ValueError(f"Cannot resolve dataset part from {value!r}.")
    return str(int(match.group()))


def _identity_number(value: Any, field: str) -> int:
    matches = re.findall(r"\d+", str(value))
    if len(matches) != 1:
        raise ValueError(f"Cannot resolve {field} identity from {value!r}.")
    return int(matches[0])


def _fit_key(
    part: Any,
    physical_cell_id: Any,
    source_file: Any,
    measurement_id: Any,
    direction: Any,
) -> tuple[str, str, str, str, str]:
    return (
        _part_number(part),
        str(physical_cell_id).strip().casefold(),
        Path(str(source_file).replace("\\", "/")).name.casefold(),
        _canonical_rpt(measurement_id),
        str(direction).strip().casefold(),
    )


def _load_used_fits(
    part12_labels: Path,
    part3_labels: Path,
    stage2_predictions: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    labels: dict[tuple[str, str, str, str, str], dict[str, Any]] = {}
    accepted_count = 0
    for path, fixed_part in ((part12_labels, None), (part3_labels, "3")):
        if not path.is_file():
            raise FileNotFoundError(f"PyDMA label table was not found: {path}")
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        for row in frame.to_dict(orient="records"):
            if row.get("checkup", "").strip().casefold() == "bol":
                continue
            if row.get("diagnostic_label_status", "").strip() != "diagnostic_estimate_accepted":
                continue
            accepted_count += 1
            try:
                rmse = float(row["fit_rmse_mV"])
                accepted_runs = int(row["accepted_runs"])
                targets = {
                    pred_column: float(row[label_column])
                    for pred_column, label_column in _LABEL_COLUMNS.items()
                }
                audit = json.loads(row["curve_audit_json"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Accepted PyDMA fit row in {path} is incomplete.") from error
            if (
                row.get("fit_accepted_by_pydma_15mV", "").strip().casefold() != "true"
                or not math.isfinite(rmse)
                or rmse >= 15.0
                or accepted_runs < 3
                or not all(math.isfinite(value) for value in targets.values())
            ):
                raise ValueError(f"Accepted PyDMA fit row in {path} fails its recorded gate.")
            part = fixed_part or _part_number(row["source_part"])
            cell = (
                f"G{_identity_number(row['group_id'], 'group')}-"
                f"Cell{_identity_number(row['cell_id'], 'cell')}"
            )
            key = _fit_key(part, cell, row["source_file"], row["checkup"], row["direction"])
            if key in labels:
                raise ValueError(f"Duplicate accepted label-fit identity: {key}")
            labels[key] = {
                **row,
                "part": part,
                "physical_cell_id": cell,
                "historical_rmse_mV": rmse,
                "historical_accepted_runs": accepted_runs,
                "targets": targets,
                "audit": audit,
                "key": key,
            }

    if not stage2_predictions.is_file():
        raise FileNotFoundError(f"Stage 2 predictions were not found: {stage2_predictions}")
    predictions = pd.read_csv(
        stage2_predictions, dtype=str, keep_default_na=False, encoding="utf-8-sig"
    )
    required = {
        "dataset_part",
        "physical_cell_id",
        "measurement_id",
        "direction",
        "source_file",
        "fit_rmse_mV",
        "accepted_runs",
        *_LABEL_COLUMNS,
        *_PREDICTION_COLUMNS.values(),
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Stage 2 prediction table lacks columns: {sorted(missing)}")

    used: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str, str]] = set()
    for prediction in predictions.to_dict(orient="records"):
        key = _fit_key(
            prediction["dataset_part"],
            prediction["physical_cell_id"],
            prediction["source_file"],
            prediction["measurement_id"],
            prediction["direction"],
        )
        if key in seen:
            raise ValueError(f"Duplicate Stage 2 fit identity: {key}")
        seen.add(key)
        label = labels.get(key)
        if label is None:
            raise ValueError(f"Stage 2 fit has no exact accepted PyDMA label match: {key}")
        if not math.isclose(
            float(prediction["fit_rmse_mV"]),
            label["historical_rmse_mV"],
            rel_tol=1e-8,
            abs_tol=1e-8,
        ) or int(prediction["accepted_runs"]) != label["historical_accepted_runs"]:
            raise ValueError(f"Stage 2 fit metadata disagrees with PyDMA labels: {key}")
        for column, value in label["targets"].items():
            if not math.isclose(
                float(prediction[column]), value, rel_tol=1e-8, abs_tol=1e-8
            ):
                raise ValueError(f"Stage 2 label disagrees with PyDMA source: {key} {column}")
        used.append({**label, "prediction": prediction})

    not_used = [
        {
            "dataset_part": row["part"],
            "physical_cell_id": row["physical_cell_id"],
            "measurement_id": row["checkup"],
            "direction": row["direction"],
            "source_file": row["source_file"],
            "historical_rmse_mV": row["historical_rmse_mV"],
            "reason": "Not present in the selected Stage 2 prediction table.",
        }
        for key, row in labels.items()
        if key not in seen
    ]
    return used, not_used, accepted_count


def _segment_ranges(capacity: np.ndarray) -> list[tuple[int, int]]:
    boundaries = np.flatnonzero(np.diff(capacity) <= 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, capacity.size]
    return [(int(start), int(end)) for start, end in zip(starts, ends)]


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-7)


def _read_source_curves(
    path: Path, records: list[dict[str, Any]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    requested = {
        (int(record["test_step"]), str(record["direction"]).casefold())
        for record in records
    }
    chunks: dict[tuple[int, str], list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        key: [] for key in requested
    }
    row_offset = 0
    for chunk in pd.read_csv(
        path,
        usecols=["Step", "Amphr", "Amps", "Volts"],
        chunksize=250_000,
    ):
        step = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(dtype=np.float64)
        capacity = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy(dtype=np.float64)
        current = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(dtype=np.float64)
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(dtype=np.float64)
        source_rows = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
        finite = np.isfinite(capacity) & np.isfinite(voltage)
        for step_number, direction in requested:
            current_mask = (
                (current > 0) & (np.abs(current) <= 0.2)
                if direction == "charge"
                else (current < 0) & (np.abs(current) <= 0.2)
            )
            mask = (step == step_number) & current_mask & finite
            if np.any(mask):
                chunks[(step_number, direction)].append(
                    (capacity[mask], voltage[mask], source_rows[mask])
                )
        row_offset += len(chunk)

    selected: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for record in records:
        fit_id = int(record["fit_id"])
        audit = record["audit"]
        step_direction = (
            int(record["test_step"]),
            str(record["direction"]).casefold(),
        )
        if audit.get("step") is not None and int(audit["step"]) != step_direction[0]:
            raise ValueError(f"Audit step disagrees with label step for fit {fit_id}.")
        if audit.get("direction") is not None and audit["direction"] != step_direction[1]:
            raise ValueError(f"Audit direction disagrees with label direction for fit {fit_id}.")
        parts = chunks[step_direction]
        if not parts:
            raise ValueError(f"No source rows found for accepted fit {fit_id}: {path}")
        capacity = np.concatenate([part[0] for part in parts])
        voltage = np.concatenate([part[1] for part in parts])
        source_rows = np.concatenate([part[2] for part in parts])

        if "candidate_segment_index" in audit:
            ranges = _segment_ranges(capacity)
            segment_index = int(audit["candidate_segment_index"])
            if segment_index < 1 or segment_index > len(ranges):
                raise ValueError(f"Candidate segment index is invalid for fit {fit_id}.")
            start, end = ranges[segment_index - 1]
            curve_capacity = capacity[start:end]
            curve_voltage = voltage[start:end]
            curve_rows = source_rows[start:end]
            expected_start = int(audit["source_row_start"])
            expected_end = int(audit["source_row_end"])
            if (
                curve_rows.size != int(audit["curve_sample_count"])
                or int(curve_rows[0]) != expected_start
                or int(curve_rows[-1]) != expected_end
                or not np.all(np.diff(curve_capacity) > 0)
            ):
                raise ValueError(f"Candidate segment boundaries changed for fit {fit_id}.")
        else:
            ranges = _segment_ranges(capacity)
            segment_index = int(audit["selected_segment"])
            if segment_index < 1 or segment_index > len(ranges):
                raise ValueError(f"Selected segment index is invalid for fit {fit_id}.")
            start, end = ranges[segment_index - 1]
            curve_capacity = capacity[start:end]
            curve_voltage = voltage[start:end]
            segment = audit["segments"][segment_index - 1]
            if (
                curve_capacity.size != int(segment["sample_count"])
                or not np.all(np.diff(curve_capacity) > 0)
                or not _close(float(curve_capacity.min()), float(segment["capacity_min"]))
                or not _close(float(curve_capacity.max()), float(segment["capacity_max"]))
                or not _close(float(curve_voltage.min()), float(segment["voltage_min"]))
                or not _close(float(curve_voltage.max()), float(segment["voltage_max"]))
            ):
                raise ValueError(f"Source curve no longer matches its saved audit for fit {fit_id}.")
        if curve_capacity.size < 10:
            raise ValueError(f"Accepted fit {fit_id} has fewer than 10 measured curve points.")
        selected[fit_id] = (curve_capacity, curve_voltage)
    return selected


def _load_references(part1_root: Path) -> dict[str, tuple[Any, Any]]:
    from pydma.electrodes.electrode import ElectrodeOCP
    from pydma.preprocessing.loader import load_ocp

    half_cell = part1_root / "Half_Cell" / "Half Cell"
    if not half_cell.is_dir():
        raise FileNotFoundError(f"Part 1 half-cell references are missing: {half_cell}")
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
    anode_frame = pd.read_csv(half_cell / "DCH_HC_anode.csv", usecols=["SoC", "Volts"])
    soc = anode_frame["SoC"].to_numpy(dtype=np.float64)
    voltage = anode_frame["Volts"].to_numpy(dtype=np.float64)
    finite = np.isfinite(soc) & np.isfinite(voltage)
    if not finite.any():
        raise ValueError("Part 1 discharge-anode reference has no finite samples.")
    discharge = (
        ElectrodeOCP(
            soc=soc[finite],
            voltage=voltage[finite],
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


def _resolve_source_path(
    record: dict[str, Any], roots: dict[str, Path]
) -> Path:
    audit = record["audit"]
    if "candidate_segment_index" in audit:
        path = roots["2"] / str(record["source_file"])
    else:
        path = Path(str(audit["source_file"]))
    if path.name.casefold() != Path(str(record["source_file"]).replace("\\", "/")).name.casefold():
        raise ValueError(f"Audit source path disagrees with label row for fit {record['fit_id']}.")
    if not path.is_file():
        raise FileNotFoundError(f"Fit source file was not found: {path}")
    return path.resolve()


def _raw_curve_soc(capacity: np.ndarray, voltage: np.ndarray) -> np.ndarray:
    span = float(np.ptp(capacity))
    if span <= 0:
        raise ValueError("Measured curve capacity has no span.")
    soc = (capacity - float(capacity.min())) / span
    if np.polyfit(soc, voltage, 1)[0] <= 0:
        soc = 1.0 - soc
    return soc


def _save_fit_page(
    pdf: PdfPages,
    record: dict[str, Any],
    capacity: np.ndarray,
    voltage: np.ndarray,
    result: Any,
    simulated: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    import matplotlib.pyplot as plt

    prediction = record["prediction"]
    measured_soc = np.asarray(result.soc_measured, dtype=np.float64)
    measured_voltage = np.asarray(result.ocv_measured, dtype=np.float64)
    simulated_soc = np.asarray(simulated["soc"], dtype=np.float64)
    simulated_voltage = np.asarray(simulated["voltage"], dtype=np.float64)
    residual_mV = (
        measured_voltage
        - np.interp(measured_soc, simulated_soc, simulated_voltage)
    ) * 1000.0
    raw_soc = _raw_curve_soc(capacity, voltage)
    stride = max(1, int(math.ceil(capacity.size / 2200)))
    provisional = "candidate_segment_index" in record["audit"]
    figure, (curve_axis, residual_axis) = plt.subplots(
        2, 1, figsize=(11.2, 8.0), sharex=True,
        gridspec_kw={"height_ratios": [2.4, 1.0]},
    )
    figure.subplots_adjust(top=0.86, bottom=0.09, hspace=0.1)
    curve_axis.plot(
        raw_soc[::stride], voltage[::stride], ".", color="#b5bdc5",
        markersize=1.5, rasterized=True, label="Measured source samples",
    )
    curve_axis.plot(
        measured_soc, measured_voltage, color="#1261a0", linewidth=1.7,
        label="PyDMA-preprocessed measured curve",
    )
    curve_axis.plot(
        simulated_soc, simulated_voltage, "--", color="#e87500",
        linewidth=1.8, label="Reproduced PyDMA fit",
    )
    curve_axis.set_ylabel("Voltage (V)")
    curve_axis.grid(True, alpha=0.25)
    curve_axis.legend(loc="best", fontsize=8)
    residual_axis.axhline(0.0, color="black", linewidth=0.8)
    residual_axis.plot(measured_soc, residual_mV, color="#6a3d9a", linewidth=1.0)
    residual_axis.set_xlabel("Normalized capacity axis (PyDMA orientation)")
    residual_axis.set_ylabel("Residual (mV)")
    residual_axis.grid(True, alpha=0.25)
    part = record["part"]
    cell = record["physical_cell_id"]
    rpt = record["checkup"]
    direction = str(record["direction"]).capitalize()
    mark = "  |  PROVISIONAL G5 RPT mapping" if provisional else ""
    figure.suptitle(
        f"Part {part} | {cell} | {rpt} | {direction}{mark}\n"
        f"Historical RMSE {record['historical_rmse_mV']:.2f} mV  |  "
        f"Reproduced RMSE {float(result.rmse) * 1000:.2f} mV  |  "
        f"Stage 2 labels: LLI {float(prediction['lli_pct_label_percent']):.2f}, "
        f"anode LAM {float(prediction['lam_anode_pct_label_percent']):.2f}, "
        f"cathode LAM {float(prediction['lam_cathode_pct_label_percent']):.2f}%",
        fontsize=10,
    )
    pdf.savefig(figure, dpi=120)
    plt.close(figure)
    return {
        "raw_soc": raw_soc[::stride],
        "raw_voltage": voltage[::stride],
        "measured_soc": measured_soc,
        "measured_voltage": measured_voltage,
        "simulated_soc": simulated_soc,
        "simulated_voltage": simulated_voltage,
        "residual_mV": residual_mV,
    }


def _save_measured_page(
    pdf: PdfPages,
    record: dict[str, Any],
    capacity: np.ndarray,
    voltage: np.ndarray,
) -> None:
    import matplotlib.pyplot as plt

    prediction = record["prediction"]
    soc = _raw_curve_soc(capacity, voltage)
    stride = max(1, int(math.ceil(capacity.size / 3000)))
    provisional = "candidate_segment_index" in record["audit"]
    mean_error = np.mean(
        [
            abs(
                float(prediction[label])
                - float(prediction[prediction_column])
            )
            for label, prediction_column in _PREDICTION_COLUMNS.items()
        ]
    )
    figure, axis = plt.subplots(figsize=(11.2, 6.0))
    figure.subplots_adjust(top=0.82, bottom=0.12)
    axis.plot(
        soc[::stride],
        voltage[::stride],
        ".",
        color="#1261a0",
        markersize=1.6,
        rasterized=True,
    )
    axis.set_xlabel("Normalized capacity axis (PyDMA orientation)")
    axis.set_ylabel("Measured voltage (V)")
    axis.grid(True, alpha=0.25)
    mark = "  |  PROVISIONAL G5 RPT mapping" if provisional else ""
    figure.suptitle(
        f"Part {record['part']} | {record['physical_cell_id']} | "
        f"{record['checkup']} | {str(record['direction']).capitalize()}{mark}\n"
        f"Measured source curve; historical accepted-fit RMSE "
        f"{record['historical_rmse_mV']:.2f} mV; "
        f"Stage 2 mean absolute degradation error {mean_error:.2f} pp",
        fontsize=10,
    )
    pdf.savefig(figure, dpi=120)
    plt.close(figure)


def _save_contact_sheet(
    path: Path,
    summaries: list[dict[str, Any]],
    quicklooks: dict[int, dict[str, np.ndarray]],
) -> None:
    import matplotlib.pyplot as plt

    worst = sorted(
        summaries,
        key=lambda row: row["mean_abs_stage2_degradation_error_pctpoints"],
        reverse=True,
    )[:12]
    figure, axes = plt.subplots(4, 3, figsize=(15, 15), squeeze=False)
    figure.suptitle(
        "12 fits with largest mean absolute Stage 2 degradation error\n"
        "Selection is for inspection only; it is not a new quality threshold.",
        fontsize=13,
    )
    for axis, row in zip(axes.flat, worst):
        plot = quicklooks[row["fit_id"]]
        axis.plot(plot["raw_soc"], plot["raw_voltage"], ".", color="#b5bdc5", markersize=1)
        axis.plot(plot["measured_soc"], plot["measured_voltage"], color="#1261a0", linewidth=1)
        axis.plot(plot["simulated_soc"], plot["simulated_voltage"], "--", color="#e87500", linewidth=1)
        axis.set_title(
            f"P{row['dataset_part']} {row['physical_cell_id']} {row['measurement_id']} "
            f"{row['direction']}\nfit {row['refit_rmse_mV']:.1f} mV; "
            f"model MAE {row['mean_abs_stage2_degradation_error_pctpoints']:.1f} pp",
            fontsize=8,
        )
        axis.grid(True, alpha=0.2)
        axis.set_xlabel("Normalized capacity")
        axis.set_ylabel("Voltage (V)")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _summary_row(
    record: dict[str, Any],
    result: Any | None,
    residual_mV: np.ndarray | None,
    source_path: Path,
) -> dict[str, Any]:
    prediction = record["prediction"]
    stage2_errors = [
        abs(float(prediction[label]) - float(prediction[prediction_column]))
        for label, prediction_column in _PREDICTION_COLUMNS.items()
    ]
    return {
        "fit_id": record["fit_id"],
        "dataset_part": record["part"],
        "physical_cell_id": record["physical_cell_id"],
        "measurement_id": record["checkup"],
        "direction": record["direction"],
        "test_step": record["test_step"],
        "source_file": record["source_file"],
        "source_run": record.get("source_run", "Part 3 run_009"),
        "mapping_status": (
            "provisional_G5_candidate_order"
            if "candidate_segment_index" in record["audit"]
            else "saved_curve_audit"
        ),
        "historical_rmse_mV": record["historical_rmse_mV"],
        "refit_rmse_mV": None if result is None else float(result.rmse) * 1000,
        "refit_rmse_full_range_mV": (
            None if result is None else float(result.rmse_full_range) * 1000
        ),
        "refit_accepted_runs": None if result is None else int(result.n_accepted_runs),
        "refit_total_runs": None if result is None else int(result.n_total_runs),
        "refit_passes_strict_gate": (
            None
            if result is None
            else bool(
                result.is_accepted
                and result.rmse * 1000 < 15
                and result.n_accepted_runs >= 3
            )
        ),
        "historical_accepted_runs": record["historical_accepted_runs"],
        "curve_sample_count": int(len(record["curve_capacity"])),
        "source_path": str(source_path),
        "mean_abs_stage2_degradation_error_pctpoints": float(np.mean(stage2_errors)),
        "lli_label_percent": float(prediction["lli_pct_label_percent"]),
        "lli_prediction_percent": float(prediction["lli_pct_prediction_percent"]),
        "lam_anode_label_percent": float(prediction["lam_anode_pct_label_percent"]),
        "lam_anode_prediction_percent": float(prediction["lam_anode_pct_prediction_percent"]),
        "lam_cathode_label_percent": float(prediction["lam_cathode_pct_label_percent"]),
        "lam_cathode_prediction_percent": float(prediction["lam_cathode_pct_prediction_percent"]),
        "max_abs_refit_residual_mV": (
            None if residual_mV is None else float(np.max(np.abs(residual_mV)))
        ),
        "rms_refit_residual_mV": (
            None
            if residual_mV is None
            else float(np.sqrt(np.mean(np.square(residual_mV))))
        ),
    }


def run_visualization(
    *,
    part12_labels: Path,
    part3_labels: Path,
    stage2_predictions: Path,
    roots: dict[str, Path],
    output: Path,
) -> dict[str, Any]:
    import matplotlib
    import torch
    from pydma import DMAAnalyzer, DMAConfig
    from bafuse.labeling.torch_pydma import use_cuda_optimizer

    matplotlib.use("Agg")
    if not torch.cuda.is_available():
        raise RuntimeError("PyDMA fit visualization requires the CUDA backend.")
    device_name = torch.cuda.get_device_name(0)
    used, not_used, accepted_count = _load_used_fits(
        part12_labels, part3_labels, stage2_predictions
    )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    for fit_id, record in enumerate(used, start=1):
        record["fit_id"] = fit_id
        record["test_step"] = int(record["test_step"])
        record["direction"] = str(record["direction"]).casefold()
        record["source_path"] = _resolve_source_path(record, roots)

    by_path: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for record in used:
        by_path[record["source_path"]].append(record)
    curves: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for source_index, (path, records) in enumerate(by_path.items(), start=1):
        curves.update(_read_source_curves(path, records))
        if source_index % 10 == 0 or source_index == len(by_path):
            print(
                f"Validated curves from {source_index}/{len(by_path)} source files.",
                flush=True,
            )

    references = _load_references(roots["1"])
    pdf_partial = output / "pydma_fit_overlays.partial.pdf"
    summaries: list[dict[str, Any]] = []
    quicklooks: dict[int, dict[str, np.ndarray]] = {}
    device = {"gpu_name": device_name, "pydma_version": version("pydma")}
    overlay_records = sorted(
        used,
        key=lambda record: np.mean(
            [
                abs(
                    float(record["prediction"][label])
                    - float(record["prediction"][prediction_column])
                )
                for label, prediction_column in _PREDICTION_COLUMNS.items()
            ]
        ),
        reverse=True,
    )[:12]
    overlay_ids = {record["fit_id"] for record in overlay_records}
    with PdfPages(pdf_partial, metadata={"Title": "Reproduced PyDMA fit overlays"}) as pdf:
        import matplotlib.pyplot as plt

        cover, axis = plt.subplots(figsize=(11.2, 8.0))
        axis.axis("off")
        provisional_count = sum(
            "candidate_segment_index" in record["audit"] for record in used
        )
        cover_text = (
            "PyDMA fit-curve visualization for the accepted labels used by the "
            "128+128 Stage 2 benchmark\n\n"
            f"Fits plotted: {len(used)}; accepted RPT-direction rows in source label tables: "
            f"{accepted_count}; accepted rows not used by Stage 2: {len(not_used)}.\n"
            f"Provisional Group 5 candidate-order mappings among plotted fits: {provisional_count}.\n\n"
            f"All pages show the exact measured curve used by Stage 2. "
            f"{len(overlay_ids)} overlays were reproduced for the fits with the highest "
            "mean absolute Stage 2 LLI/LAM prediction error; other pages show historical "
            "fit RMSE without a simulated overlay. The original tables do not store fitted "
            "parameters; these are deterministic refits using seed 211, data_length 500, "
            "RMSE threshold 15 mV, three required accepted runs, maxiter 50, popsize 30, "
            "max_tries 10, and CUDA. They are not guaranteed to be the exact historical "
            "optimizer outputs.\n\n"
            "The normalized capacity axis preserves PyDMA's voltage-increasing orientation; "
            "it is not labeled in Ah because Amphr units have not been independently "
            "validated. A passing fit gate does not validate RPT identity or make "
            "PyDMA-derived LLI/LAM independent ground truth.\n\n"
            "Group 5 RPT identities are provisional file-order mappings from TPG5.3 and "
            "have not been confirmed against original Maccor logs. Part 3 is shown as "
            "external/domain-shift data and uses the Part 1 half-cell references, matching "
            "the recorded label run."
        )
        axis.text(0.04, 0.96, cover_text, va="top", fontsize=12, wrap=True)
        pdf.savefig(cover)
        plt.close(cover)

        for index, record in enumerate(used, start=1):
            capacity, voltage = curves[record["fit_id"]]
            record["curve_capacity"] = capacity
            result = None
            residual = None
            if record["fit_id"] in overlay_ids:
                config = DMAConfig(
                    direction=record["direction"],
                    data_length=500,
                    weight_ocv=100.0,
                    weight_dva=0.0,
                    weight_ica=0.0,
                    speed_preset="fast",
                    req_accepted=3,
                    max_tries_overall=10,
                    rmse_threshold=0.015,
                    random_seed=211,
                    print_progress=False,
                    allow_first_cycle_inhomogeneity=False,
                )
                analyzer = DMAAnalyzer(
                    config=config,
                    anode=references[record["direction"]][0],
                    cathode=references[record["direction"]][1],
                )
                try:
                    with redirect_stdout(StringIO()):
                        with use_cuda_optimizer():
                            result = analyzer.analyze(
                                measured_capacity=capacity,
                                measured_voltage=voltage,
                                maxiter=50,
                                popsize=30,
                                polish=True,
                            )
                    simulated = analyzer.compute_simulated_curves(
                        result.fitted_params, n_points=1000
                    )
                except Exception as error:
                    raise RuntimeError(
                        f"PyDMA refit failed for {record['key']}: "
                        f"{type(error).__name__}: {error}"
                    ) from error
                plot = _save_fit_page(pdf, record, capacity, voltage, result, simulated)
                quicklooks[record["fit_id"]] = plot
                residual = (
                    np.asarray(result.ocv_measured, dtype=np.float64)
                    - np.interp(
                        np.asarray(result.soc_measured, dtype=np.float64),
                        np.asarray(result.soc_reconstructed, dtype=np.float64),
                        np.asarray(result.ocv_reconstructed, dtype=np.float64),
                    )
                ) * 1000
            else:
                _save_measured_page(pdf, record, capacity, voltage)
            summaries.append(
                _summary_row(record, result, residual, record["source_path"])
            )
            if index % 10 == 0 or index == len(used):
                pd.DataFrame(summaries).to_csv(
                    output / "pydma_fit_summary.partial.csv", index=False
                )
                print(
                    f"Plotted {index}/{len(used)} source curves; refitted overlays "
                    f"{sum(row['refit_rmse_mV'] is not None for row in summaries)}/"
                    f"{len(overlay_ids)}.",
                    flush=True,
                )

    (output / "pydma_fit_overlays.partial.pdf").replace(output / "pydma_fit_overlays.pdf")
    pd.DataFrame(summaries).to_csv(output / "pydma_fit_summary.csv", index=False)
    partial_summary = output / "pydma_fit_summary.partial.csv"
    if partial_summary.exists():
        partial_summary.unlink()
    pd.DataFrame(not_used).to_csv(output / "accepted_fits_not_used_by_stage2.csv", index=False)
    _save_contact_sheet(
        output / "highest_stage2_error_fit_contact_sheet.png", summaries, quicklooks
    )
    metadata = {
        "experiment": "Reproduced PyDMA overlays for accepted labels used by the 128+128 Stage 2 run",
        "stage2_predictions": str(stage2_predictions.resolve()),
        "accepted_fit_table_rows": accepted_count,
        "stage2_used_fit_count": len(used),
        "accepted_fit_rows_not_used_by_stage2": len(not_used),
        "reproduced_overlay_count": len(overlay_ids),
        "measured_only_curve_count": len(used) - len(overlay_ids),
        "overlay_selection": (
            "12 fits with highest mean absolute Stage 2 LLI/LAM prediction error"
        ),
        "provisional_group5_candidate_order_fit_count": provisional_count,
        "part3_external_fit_count": sum(record["part"] == "3" for record in used),
        "fit_parameters": {
            "random_seed": 211,
            "data_length": 500,
            "rmse_threshold_mV": 15,
            "required_accepted_runs": 3,
            "max_tries": 10,
            "maxiter": 50,
            "popsize": 30,
            "polish": True,
            "device": device,
            "references": "Part 1 half-cell references for Parts 1, 2, and 3",
        },
        "caveats": [
            "Historical fitted parameters were not persisted; the overlays are deterministic refits, not exact historical optimizer outputs.",
            "A passing PyDMA fit gate does not validate RPT identity or establish independent physical ground truth.",
            "Group 5 TPG5.3 RPT identities are provisional file-order mappings not verified by original Maccor logs.",
            "Part 3 is external/domain-shift data and uses Part 1 half-cell references.",
            "Amphr units are not independently validated; plots use normalized capacity, not Ah.",
        ],
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=load_project_paths().root)
    parser.add_argument("--part12-labels", type=Path)
    parser.add_argument("--part3-labels", type=Path)
    parser.add_argument("--stage2-predictions", type=Path)
    parser.add_argument("--part1-root", type=Path)
    parser.add_argument("--part2-root", type=Path)
    parser.add_argument("--part3-root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    paths = load_project_paths(args.root)

    def resolve(value: Path | None, default: Path) -> Path:
        selected = value or default
        return selected if selected.is_absolute() else paths.root / selected

    metadata = run_visualization(
        part12_labels=resolve(args.part12_labels, paths.visualization_part12_labels),
        part3_labels=resolve(args.part3_labels, paths.visualization_part3_labels),
        stage2_predictions=resolve(
            args.stage2_predictions, paths.visualization_stage2_predictions
        ),
        roots={
            part: resolve(getattr(args, f"part{part}_root"), paths.dataset_parts[part])
            for part in ("1", "2", "3")
        },
        output=resolve(args.output, paths.pydma_visualization),
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
