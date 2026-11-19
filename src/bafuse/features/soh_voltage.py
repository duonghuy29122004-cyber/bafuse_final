from __future__ import annotations

import csv
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


SPLIT_VERSION = "parts12-10-3-3-plus-part3-external-v1"
EXPECTED_PARTITION_SIZES = {
    "train": 10,
    "validation": 3,
    "internal_test": 3,
    "external_test": 11,
}
EXPLORATORY_SPLIT_VERSION = "parts12-12-4-plus-part3-external-v1"
EXPLORATORY_PARTITION_SIZES = {
    "train": 12,
    "validation": 4,
    "external_test": 11,
}
FINAL_SPLIT_VERSION = "parts12-12-4-plus-part3-12-external-v2"
FINAL_PARTITION_SIZES = {
    "train": 12,
    "validation": 4,
    "external_test": 12,
}
_SPLIT_PARTITION_SIZES = {
    SPLIT_VERSION: EXPECTED_PARTITION_SIZES,
    EXPLORATORY_SPLIT_VERSION: EXPLORATORY_PARTITION_SIZES,
    FINAL_SPLIT_VERSION: FINAL_PARTITION_SIZES,
}
N_VOLTAGE_POINTS = 128
N_PAIRED_VOLTAGE_POINTS = 128
CSV_CHUNK_ROWS = 250_000
FEATURE_NAMES = [f"voltage_progress_{i:03d}" for i in range(N_VOLTAGE_POINTS)]
DISCHARGE_FEATURE_NAMES = [
    f"discharge_voltage_progress_{i:03d}"
    for i in range(N_PAIRED_VOLTAGE_POINTS)
]
CHARGE_FEATURE_NAMES = [
    f"charge_voltage_progress_{i:03d}"
    for i in range(N_PAIRED_VOLTAGE_POINTS)
]


@dataclass(frozen=True)
class SplitManifest:
    version: str
    cell_to_partition: dict[str, str]
    partition_sizes: dict[str, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(path: Path) -> SplitManifest:
    cell_to_partition: dict[str, str] = {}
    version: str | None = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            row_version = row["split_version"]
            if version is None:
                version = row_version
            elif row_version != version:
                raise ValueError("Split manifest contains multiple split versions.")
            if row_version not in _SPLIT_PARTITION_SIZES:
                raise ValueError(f"Unexpected split version: {row_version}")
            cell = row["physical_cell_id"]
            partition = row["partition"]
            if not cell or partition not in _SPLIT_PARTITION_SIZES[row_version]:
                raise ValueError(f"Invalid cell or partition in split row: {row}")
            if cell in cell_to_partition:
                raise ValueError(f"Cell appears twice in the split: {cell}")
            cell_to_partition[cell] = partition
    if version is None:
        raise ValueError("Split manifest is empty.")
    sizes: dict[str, int] = defaultdict(int)
    for partition in cell_to_partition.values():
        sizes[partition] += 1
    expected_sizes = _SPLIT_PARTITION_SIZES[version]
    if dict(sizes) != expected_sizes:
        raise ValueError(
            f"Split partition sizes differ from {version}: {dict(sizes)}"
        )
    return SplitManifest(version, cell_to_partition, dict(sizes))


def load_split(path: Path) -> dict[str, str]:
    manifest = load_split_manifest(path)
    if manifest.version != SPLIT_VERSION:
        raise ValueError(f"Expected locked split version {SPLIT_VERSION}.")
    return manifest.cell_to_partition


def resolve_source_file(dataset_root: Path, relative_source: str) -> Path:
    normalized = str(relative_source).replace("\\", "/").strip()
    rel = PurePosixPath(normalized)
    if not normalized or rel.is_absolute() or ".." in rel.parts:
        raise ValueError(f"Unsafe or empty manifest source path: {relative_source!r}")
    candidate = dataset_root.joinpath(*rel.parts).resolve()
    if not candidate.is_relative_to(dataset_root):
        raise ValueError(f"Manifest source escapes dataset root: {relative_source}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Mapped CSV was not found: {relative_source}")
    return candidate


def extract_step4_voltage(
    csv_path: Path,
    segment_index: int,
    n_points: int = N_VOLTAGE_POINTS,
    chunk_rows: int = CSV_CHUNK_ROWS,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Resample one Step 4 voltage segment onto normalized sample progress."""
    if segment_index < 1:
        raise ValueError(f"Step 4 segment index must be positive; got {segment_index}.")
    voltage_parts: list[np.ndarray] = []
    segment_count = 0
    step4_rows = 0
    previous_was_step4 = False
    for chunk in pd.read_csv(csv_path, usecols=["Step", "Volts"], chunksize=chunk_rows):
        steps = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(dtype=np.float64)
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(dtype=np.float64)
        is_step4 = np.isfinite(steps) & (steps == 4)
        starts = is_step4 & ~np.r_[previous_was_step4, is_step4[:-1]]
        segment_ids = segment_count + np.cumsum(starts)
        selected = is_step4 & (segment_ids == segment_index)
        step4_rows += int(selected.sum())
        finite = selected & np.isfinite(voltage)
        if finite.any():
            voltage_parts.append(voltage[finite])
        segment_count += int(starts.sum())
        if is_step4.size:
            previous_was_step4 = bool(is_step4[-1])

    if segment_count < segment_index:
        raise ValueError(
            f"Step 4 segment {segment_index} not found; source has {segment_count}."
        )
    if not voltage_parts:
        raise ValueError(f"Step 4 segment {segment_index} has no finite voltage values.")
    voltage = np.concatenate(voltage_parts)
    if voltage.size < n_points:
        raise ValueError(f"Segment has {voltage.size} finite samples; need {n_points}.")
    v_min, v_max = float(voltage.min()), float(voltage.max())
    if v_min > 2.6:
        raise ValueError(f"Curve minimum {v_min:.3f} V does not approach the 2.5 V cutoff.")
    if v_max > 4.3:
        raise ValueError(f"Curve maximum {v_max:.3f} V exceeds the 4.2 V tolerance.")
    features = np.interp(
        np.linspace(0.0, 1.0, n_points),
        np.linspace(0.0, 1.0, voltage.size),
        voltage,
    )
    metadata = {
        "step4_rows": step4_rows,
        "finite_voltage_samples": int(voltage.size),
        "step4_segment_count": segment_count,
        "voltage_min_v": v_min,
        "voltage_max_v": v_max,
    }
    return features, metadata


def extract_rpt_charge_voltages(
    csv_path: Path,
    segment_indices: list[int],
    dataset_part: str,
    n_points: int = N_PAIRED_VOLTAGE_POINTS,
    chunk_rows: int = CSV_CHUNK_ROWS,
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    if dataset_part not in {"1", "2", "3"}:
        raise ValueError(f"Unsupported dataset part for charge extraction: {dataset_part}.")
    if n_points < 2 or chunk_rows < 1:
        raise ValueError("n_points must be at least 2 and chunk_rows must be positive.")
    if not segment_indices or any(index < 1 for index in segment_indices):
        raise ValueError("At least one positive Step 4 segment index is required.")
    if len(set(segment_indices)) != len(segment_indices):
        raise ValueError("Step 4 segment indexes must be unique.")

    requested = set(segment_indices)
    features: dict[int, np.ndarray] = {}
    metadata: dict[int, dict[str, Any]] = {}
    step4_segment_index = 0
    active: dict[str, Any] | None = None
    run_step: int | None = None
    run_amp_parts: list[np.ndarray] = []
    run_voltage_parts: list[np.ndarray] = []

    def finish_active(boundary_seen: bool) -> None:
        nonlocal active
        if active is None:
            return
        index = int(active["segment_index"])
        if index in requested:
            charge_voltage = (
                np.concatenate(active["voltage_parts"])
                if active["voltage_parts"]
                else np.empty(0, dtype=np.float64)
            )
            total_current_rows = int(active["total_current_rows"])
            negative_fraction = (
                int(active["negative_current_rows"]) / total_current_rows
                if total_current_rows
                else 0.0
            )
            reason = active["error"]
            status = "ok"
            if not boundary_seen:
                status, reason = "missing_step8_boundary", (
                    "The mapped Step 4 segment was not followed by its Step 8 boundary."
                )
            elif reason:
                status = "ambiguous_charge_window"
            elif active["positive_step_runs"].get(6, 0) != 1:
                status, reason = "invalid_step6_charge", (
                    "Expected exactly one positive-current Step 6 segment in the mapped RPT."
                )
            elif dataset_part in {"1", "2"} and active["positive_step_runs"].get(5, 0):
                status, reason = "unexpected_step5_charge", (
                    "Parts 1–2 charge features use Step 6; Step 5 unexpectedly had positive current."
                )
            elif dataset_part == "3" and active["positive_step_runs"].get(5, 0) > 1:
                status, reason = "ambiguous_step5_charge", (
                    "Part 3 has multiple positive-current Step 5 segments in the mapped RPT."
                )
            elif negative_fraction > 0.01:
                status, reason = "invalid_charge_current_direction", (
                    "More than 1% of candidate charge-step current samples were negative."
                )
            elif charge_voltage.size < n_points:
                status, reason = "insufficient_charge_samples", (
                    f"Only {charge_voltage.size} positive-current voltage samples; "
                    f"need {n_points}."
                )
            else:
                v_min = float(charge_voltage.min())
                v_max = float(charge_voltage.max())
                if v_min < 2.3 or v_max > 4.3 or v_max < 4.0 or v_min >= 4.0:
                    status, reason = "invalid_charge_voltage_range", (
                        f"Unexpected charge voltage range [{v_min:.3f}, {v_max:.3f}] V."
                    )
                else:
                    features[index] = np.interp(
                        np.linspace(0.0, 1.0, n_points),
                        np.linspace(0.0, 1.0, charge_voltage.size),
                        charge_voltage,
                    )
            metadata[index] = {
                "status": status,
                "reason": reason or "",
                "step4_segment_index": index,
                "step4_segment_count": step4_segment_index,
                "charge_protocol_steps": "6" if dataset_part in {"1", "2"} else "5,6",
                "charge_steps_used": ",".join(
                    str(step) for step in active["steps_used"]
                ),
                "positive_current_samples": int(charge_voltage.size),
                "negative_current_fraction": float(negative_fraction),
                "voltage_min_v": (
                    float(charge_voltage.min()) if charge_voltage.size else None
                ),
                "voltage_max_v": (
                    float(charge_voltage.max()) if charge_voltage.size else None
                ),
            }
        active = None

    def consume_run() -> None:
        nonlocal active, step4_segment_index
        if run_step is None:
            return
        if run_step == 4:
            if active is not None:
                finish_active(boundary_seen=False)
            step4_segment_index += 1
            active = {
                "segment_index": step4_segment_index,
                "capture": step4_segment_index in requested,
                "voltage_parts": [],
                "positive_step_runs": {},
                "steps_used": [],
                "total_current_rows": 0,
                "negative_current_rows": 0,
                "step6_seen": False,
                "step7_seen": False,
                "error": "",
            }
            return
        if active is None:
            return
        if run_step == 8:
            finish_active(boundary_seen=True)
            return
        if run_step == 5:
            if active["step6_seen"] or active["step7_seen"]:
                active["error"] = "Step 5 occurred after the mapped charge sequence began."
            if dataset_part in {"1", "2"}:
                finite_current = (
                    np.concatenate(run_amp_parts)
                    if run_amp_parts
                    else np.empty(0, dtype=np.float64)
                )
                if active["capture"] and np.any(finite_current > 0.02):
                    active["error"] = (
                        "Positive current occurred in Step 5 although the Parts 1–2 "
                        "RPT protocol specifies charge at Step 6."
                    )
                return
        elif run_step == 6:
            if active["step7_seen"]:
                active["error"] = "Step 6 occurred after the mapped Step 7 rest."
            active["step6_seen"] = True
        elif run_step == 7:
            if not active["step6_seen"]:
                active["error"] = "Step 7 occurred before the mapped Step 6 charge."
            active["step7_seen"] = True
            return
        else:
            active["error"] = (
                f"Unexpected Step {run_step} between the mapped Step 4 and Step 8."
            )
            return

        if run_step not in {5, 6} or not active["capture"]:
            return
        amps = (
            np.concatenate(run_amp_parts)
            if run_amp_parts
            else np.empty(0, dtype=np.float64)
        )
        volts = (
            np.concatenate(run_voltage_parts)
            if run_voltage_parts
            else np.empty(0, dtype=np.float64)
        )
        finite_current = np.isfinite(amps)
        active["total_current_rows"] += int(finite_current.sum())
        active["negative_current_rows"] += int((amps[finite_current] < -0.01).sum())
        selected = np.isfinite(amps) & np.isfinite(volts) & (amps > 0.0)
        if selected.any():
            active["positive_step_runs"][run_step] = (
                active["positive_step_runs"].get(run_step, 0) + 1
            )
            active["steps_used"].append(run_step)
            active["voltage_parts"].append(volts[selected])

    for chunk in pd.read_csv(
        csv_path,
        usecols=["Step", "Amps", "Volts"],
        chunksize=chunk_rows,
    ):
        steps = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        amps = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        volts = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        step_ids = np.where(
            np.isfinite(steps) & (steps == np.floor(steps)), steps, -1
        ).astype(np.int64)
        starts = np.r_[0, np.flatnonzero(step_ids[1:] != step_ids[:-1]) + 1]
        ends = np.r_[starts[1:], step_ids.size]
        for start, end in zip(starts, ends):
            step = int(step_ids[start])
            if run_step != step:
                consume_run()
                run_step = step
                run_amp_parts = []
                run_voltage_parts = []
            if (
                step in {5, 6}
                and active is not None
                and active["capture"]
            ):
                run_amp_parts.append(amps[start:end])
                run_voltage_parts.append(volts[start:end])
    consume_run()
    if active is not None:
        finish_active(boundary_seen=False)

    for index in requested:
        if index not in metadata:
            metadata[index] = {
                "status": "missing_step4_segment",
                "reason": (
                    f"Requested Step 4 segment {index}; source has "
                    f"{step4_segment_index} segments."
                ),
                "step4_segment_index": index,
                "step4_segment_count": step4_segment_index,
                "charge_protocol_steps": "6" if dataset_part in {"1", "2"} else "5,6",
                "charge_steps_used": "",
                "positive_current_samples": 0,
                "negative_current_fraction": 0.0,
                "voltage_min_v": None,
                "voltage_max_v": None,
            }
    for item in metadata.values():
        item["step4_segment_count"] = step4_segment_index
    return features, metadata
