from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_OBSERVATION_FIELDS = [
    "part",
    "group",
    "physical_cell_id",
    "rpt_index",
    "source_tpg",
    "source_file",
    "source_record_id",
    "guide_stage",
    "split",
    "bol_reference_file",
    "bol_step4_segment_ref",
    "step4_segment_ref",
    "step8_segment_ref",
    "step10_segment_ref",
    "mapping_status",
    "step4_status",
    "step8_status",
    "step10_status",
    "ready_soh",
    "ready_degradation",
    "needs_audit",
    "audit_notes",
]
_ISSUE_FIELDS = [
    "part",
    "group",
    "physical_cell_id",
    "source_tpg",
    "source_file",
    "candidate_rpt_index",
    "issue_type",
    "evidence",
    "recommended_action",
    "severity",
]
_STEPS = (4, 8, 10)
_MIN_CURVE_ROWS = 10
_POCV_CURRENT_LIMIT_A = 0.2
_MAX_REFERENCE_SOC = 100.0
_RPT_RANGE_RE = re.compile(
    r"\bRPT\s*(\d+)\s*(?:-|–|—|to)\s*(?:RPT\s*)?(\d+)\b", re.IGNORECASE
)
_RPT_SINGLE_RE = re.compile(r"\bRPT\s*(\d+)\b", re.IGNORECASE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_rpt_stage(stage: str) -> tuple[list[int], bool]:
    """Return only RPT indices explicitly present in Guide stage text."""
    normalized = (stage or "").replace("–", "-").replace("—", "-").split("(", 1)[0]
    numbers: list[int] = []
    spans: list[tuple[int, int]] = []
    for match in _RPT_RANGE_RE.finditer(normalized):
        first, last = map(int, match.groups())
        spans.append(match.span())
        if last >= first:
            numbers.extend(range(first, last + 1))
    remaining = list(normalized)
    for start, end in spans:
        remaining[start:end] = " " * (end - start)
    numbers.extend(int(value) for value in _RPT_SINGLE_RE.findall("".join(remaining)))
    duplicate = len(numbers) != len(set(numbers))
    return numbers, duplicate


def _source_paths(value: str) -> list[str]:
    return [item.strip() for item in re.split(r"\s*\|\|\s*", value or "") if item.strip()]


def _resolve_source(root: Path, part: str, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    expected_root = (root / "data" / "raw" / "oxford" / f"part{part}").resolve()
    part_prefix = f"data/raw/oxford/part{part}/"
    if normalized.casefold().startswith(part_prefix.casefold()):
        normalized = normalized[len(part_prefix) :]
    path = (expected_root / normalized).resolve()
    try:
        path.relative_to(expected_root)
    except ValueError as error:
        raise ValueError(f"Manifest source path escapes Part {part}: {relative_path}") from error
    return path


def _new_segment(step: int, index: int, start_row: int) -> dict[str, Any]:
    return {
        "step": step,
        "segment_index": index,
        "source_row_start": start_row,
        "source_row_end": start_row,
        "source_rows": 0,
        "capacity_parts": [],
        "voltage_parts": [],
        "current_parts": [],
    }


def _finalize_raw_segment(raw: dict[str, Any]) -> list[dict[str, Any]]:
    step = int(raw["step"])
    capacity = (
        np.concatenate(raw["capacity_parts"]).astype(np.float64, copy=False)
        if raw["capacity_parts"]
        else np.asarray([], dtype=np.float64)
    )
    voltage = (
        np.concatenate(raw["voltage_parts"]).astype(np.float64, copy=False)
        if raw["voltage_parts"]
        else np.asarray([], dtype=np.float64)
    )
    current = (
        np.concatenate(raw["current_parts"]).astype(np.float64, copy=False)
        if raw["current_parts"]
        else np.asarray([], dtype=np.float64)
    )
    finite = np.isfinite(capacity) & np.isfinite(voltage)
    capacity, voltage, current = capacity[finite], voltage[finite], current[finite]
    source_rows = np.arange(
        int(raw["source_row_start"]),
        int(raw["source_row_start"]) + int(raw["source_rows"]),
        dtype=np.int64,
    )[finite]

    if step == 4:
        current_finite = current[np.isfinite(current)]
        signs = set()
        if np.any(current_finite < 0):
            signs.add("negative")
        if np.any(current_finite > 0):
            signs.add("positive")
        segment = _summarize_curve(
            step=step,
            index=int(raw["segment_index"]),
            capacity=capacity,
            voltage=voltage,
            source_rows=source_rows,
        )
        segment["raw_step_rows"] = int(raw["source_rows"])
        segment["current_signs"] = ";".join(sorted(signs))
        segment["valid"] = bool(
            segment["sample_count"] >= _MIN_CURVE_ROWS
            and segment["capacity_span"] > 0
            and "negative" in signs
            and "positive" not in signs
        )
        segment["status"] = "valid_discharge_curve" if segment["valid"] else "invalid_or_partial"
        return [segment]

    direction = "discharge" if step == 8 else "charge"
    if direction == "discharge":
        mask = np.isfinite(current) & (current < 0) & (np.abs(current) <= _POCV_CURRENT_LIMIT_A)
    else:
        mask = np.isfinite(current) & (current > 0) & (np.abs(current) <= _POCV_CURRENT_LIMIT_A)
    capacity, voltage, source_rows = capacity[mask], voltage[mask], source_rows[mask]
    if not capacity.size:
        return []
    boundaries = np.flatnonzero(np.diff(capacity) <= 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, capacity.size]
    output = []
    for sub_index, (start, end) in enumerate(zip(starts, ends), start=1):
        segment = _summarize_curve(
            step=step,
            index=int(raw["segment_index"]),
            capacity=capacity[start:end],
            voltage=voltage[start:end],
            source_rows=source_rows[start:end],
        )
        segment["subsegment_index"] = sub_index
        segment["raw_step_rows"] = int(raw["source_rows"])
        segment["direction"] = direction
        segment["valid"] = bool(
            segment["sample_count"] >= _MIN_CURVE_ROWS
            and segment["capacity_span"] > 0
        )
        segment["status"] = "valid_monotone_curve" if segment["valid"] else "invalid_or_partial"
        output.append(segment)
    return output


def _summarize_curve(
    step: int,
    index: int,
    capacity: np.ndarray,
    voltage: np.ndarray,
    source_rows: np.ndarray,
) -> dict[str, Any]:
    if capacity.size:
        capacity_span = float(np.ptp(capacity))
        voltage_min = float(np.min(voltage))
        voltage_max = float(np.max(voltage))
        row_start = int(source_rows[0])
        row_end = int(source_rows[-1])
        monotone = bool(np.all(np.diff(capacity) > 0))
    else:
        capacity_span = 0.0
        voltage_min = float("nan")
        voltage_max = float("nan")
        row_start = -1
        row_end = -1
        monotone = False
    return {
        "step": step,
        "segment_index": index,
        "source_row_start": row_start,
        "source_row_end": row_end,
        "sample_count": int(capacity.size),
        "capacity_min": float(np.min(capacity)) if capacity.size else float("nan"),
        "capacity_max": float(np.max(capacity)) if capacity.size else float("nan"),
        "capacity_span": capacity_span,
        "voltage_min": voltage_min,
        "voltage_max": voltage_max,
        "strictly_increasing_capacity": monotone,
    }


def _append_chunk_to_segment(
    segment: dict[str, Any],
    capacity: np.ndarray,
    voltage: np.ndarray,
    current: np.ndarray,
    row_indices: np.ndarray,
) -> None:
    segment["capacity_parts"].append(capacity)
    segment["voltage_parts"].append(voltage)
    segment["current_parts"].append(current)
    segment["source_rows"] += int(len(capacity))
    if len(row_indices):
        segment["source_row_end"] = int(row_indices[-1])


def scan_tpg_file(path: Path, chunk_rows: int = 100_000) -> dict[str, Any]:
    """Read Step 4/8/10 rows and retain their observed source-row boundaries."""
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    stat = path.stat()
    result: dict[str, Any] = {
        "source_file": str(path),
        "source_size_bytes": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "rows_scanned": 0,
        "scan_status": "scanning",
        "scan_error": "",
        "steps": {str(step): [] for step in _STEPS},
    }
    active: dict[str, Any] | None = None
    next_segment_index = {step: 0 for step in _STEPS}
    row_offset = 0
    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        required = {"Step", "Amphr", "Volts", "Amps"}
        missing = required.difference(columns)
        if missing:
            result.update(
                scan_status="missing_required_columns",
                scan_error="Missing required columns: " + ", ".join(sorted(missing)),
            )
            return result
        for chunk in pd.read_csv(
            path,
            usecols=["Step", "Amphr", "Volts", "Amps"],
            chunksize=chunk_rows,
            low_memory=False,
        ):
            step_values = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(dtype=float)
            capacity_values = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy(dtype=float)
            voltage_values = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(dtype=float)
            current_values = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(dtype=float)
            integer_steps = np.where(np.isfinite(step_values), step_values, -1).astype(np.int32)
            change_points = np.flatnonzero(np.diff(integer_steps) != 0) + 1
            starts = np.r_[0, change_points]
            ends = np.r_[change_points, len(chunk)]
            for start, end in zip(starts, ends):
                step = int(integer_steps[start])
                is_target = step in _STEPS
                if active is not None and (not is_target or active["step"] != step):
                    finalized = _finalize_raw_segment(active)
                    result["steps"][str(active["step"])].extend(finalized)
                    active = None
                if not is_target:
                    continue
                if active is None:
                    next_segment_index[step] += 1
                    active = _new_segment(
                        step, next_segment_index[step], row_offset + int(start) + 2
                    )
                indices = np.arange(
                    row_offset + int(start) + 2,
                    row_offset + int(end) + 2,
                    dtype=np.int64,
                )
                _append_chunk_to_segment(
                    active,
                    capacity_values[start:end],
                    voltage_values[start:end],
                    current_values[start:end],
                    indices,
                )
            row_offset += len(chunk)
            result["rows_scanned"] += len(chunk)
        if active is not None:
            result["steps"][str(active["step"])].extend(_finalize_raw_segment(active))
        result["scan_status"] = "scanned"
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        result["scan_status"] = "scan_failed"
        result["scan_error"] = f"{type(error).__name__}: {error}"
    return result


def _rpt_indices(row: dict[str, str]) -> tuple[list[int], bool]:
    return parse_rpt_stage(row.get("experimental_stage", ""))


def _is_bol(row: dict[str, str]) -> bool:
    return row.get("experimental_stage", "").strip().casefold().startswith("bol")


def _baseline_catalog(
    root: Path, manifest: list[dict[str, str]] | None = None
) -> dict[str, dict[str, str]]:
    catalog: dict[str, dict[str, str]] = {}
    p1_labels = root / "data/processed/label_soh_part1_2/run_006_part1_soh_labels_earliest_bol/labels_soh.csv"
    if p1_labels.is_file():
        for row in _read_csv(p1_labels):
            if row.get("validation_status") == "baseline_reference":
                catalog[row["physical_cell_id"]] = {
                    "part": "1",
                    "source_file": row["baseline_source_file"],
                    "segment_index": row["baseline_segment_index"] or "1",
                    "selection_status": "selected_existing_part1_policy",
                }
    p2_bol = root / "experiments/results/data_audits/part2_g5_g6_soh_extension/bol_denominator_audit.csv"
    if p2_bol.is_file():
        for row in _read_csv(p2_bol):
            catalog[row["physical_cell_id"]] = {
                "part": "2",
                "source_file": row["source_file"],
                "segment_index": row["segment_index"] or "1",
                "selection_status": "selected_local_bol_provisional_policy",
            }
    p3_labels = root / "experiments/results/data_audits/part3_soh_candidates/labels_soh_candidates.csv"
    if p3_labels.is_file():
        for row in _read_csv(p3_labels):
            if row.get("validation_status") == "baseline_reference":
                catalog[row["physical_cell_id"]] = {
                    "part": "3",
                    "source_file": row["baseline_source_file"],
                    "segment_index": row["baseline_segment_index"] or "1",
                    "selection_status": "selected_part3_candidate_baseline",
                }
    if manifest:
        bol_by_cell: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        for row in manifest:
            if (
                row.get("record_type") == "cycling_guide_record"
                and _is_bol(row)
                and row.get("physical_cell_id")
                and row.get("mapping_status")
                not in {"ambiguous_source_identity", "missing_source", "no_source_per_user_confirmation"}
            ):
                paths = _source_paths(row.get("source_paths", ""))
                if len(paths) == 1:
                    bol_by_cell[row["physical_cell_id"]].append(row)
        for cell, rows in bol_by_cell.items():
            if cell not in catalog and len(rows) == 1:
                row = rows[0]
                catalog[cell] = {
                    "part": row["dataset_part"],
                    "source_file": row["source_paths"].replace(
                        f"data/raw/oxford/part{row['dataset_part']}/", ""
                    ),
                    "segment_index": "1",
                    "selection_status": "single_manifest_bol_requires_qc",
                }
    return catalog


def _half_cell_reference_audit(root: Path) -> dict[str, dict[str, Any]]:
    directory = root / "data/raw/oxford/part1/Half_Cell/Half Cell"
    names = {
        "charge_anode": "CH_HC_anode.csv",
        "charge_cathode": "CH_HC_cathode.csv",
        "discharge_anode": "DCH_HC_anode.csv",
        "discharge_cathode": "DHC_HC_cathode.csv",
    }
    results: dict[str, dict[str, Any]] = {}
    for key, name in names.items():
        path = directory / name
        record: dict[str, Any] = {"source_file": str(path), "status": "missing_source", "finite_pairs": 0}
        if path.is_file():
            try:
                frame = pd.read_csv(path, usecols=["SoC", "Volts"])
                soc = pd.to_numeric(frame["SoC"], errors="coerce").to_numpy(dtype=float)
                voltage = pd.to_numeric(frame["Volts"], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(soc) & np.isfinite(voltage)
                record["finite_pairs"] = int(np.count_nonzero(finite))
                record["soc_min"] = float(np.min(soc[finite])) if finite.any() else ""
                record["soc_max"] = float(np.max(soc[finite])) if finite.any() else ""
                record["voltage_min"] = float(np.min(voltage[finite])) if finite.any() else ""
                record["voltage_max"] = float(np.max(voltage[finite])) if finite.any() else ""
                record["status"] = (
                    "verified_reference_columns_and_finite_pairs"
                    if record["finite_pairs"] >= _MIN_CURVE_ROWS
                    and 0 <= record["soc_min"]
                    and record["soc_max"] <= _MAX_REFERENCE_SOC
                    else "needs_curve_audit"
                )
            except (OSError, ValueError, pd.errors.ParserError) as error:
                record.update(status="read_error", error=f"{type(error).__name__}: {error}")
        results[key] = record
    return results


def _segment_ref(path: str, segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    return (
        f"{path}#Step{segment['step']}:segment{segment['segment_index']}"
        f":rows{segment['source_row_start']}-{segment['source_row_end']}"
        + (
            f":sub{segment['subsegment_index']}"
            if segment.get("subsegment_index")
            else ""
        )
    )


def _source_stage_records(
    manifest: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    observations: list[dict[str, str]] = []
    bol_rows: list[dict[str, str]] = []
    for row in manifest:
        if row.get("record_type") != "cycling_guide_record":
            continue
        if not _is_bol(row) and not _rpt_indices(row)[0]:
            continue
        if _is_bol(row):
            bol_rows.append(row)
            continue
        indices, _ = _rpt_indices(row)
        if not indices:
            continue
        if row.get("experimental_stage", "").strip().casefold() == "profile":
            continue
        for rpt in indices:
            expanded = dict(row)
            expanded["candidate_rpt_index"] = str(rpt)
            expanded["candidate_rpt_count_for_source"] = str(len(indices))
            observations.append(expanded)
    return observations, bol_rows


def _issue(
    issues: list[dict[str, Any]],
    row: dict[str, str],
    issue_type: str,
    evidence: str,
    action: str,
    severity: str = "medium",
    rpt: str = "",
) -> None:
    issues.append(
        {
            "part": row.get("dataset_part", ""),
            "group": row.get("group_id", ""),
            "physical_cell_id": row.get("physical_cell_id", ""),
            "source_tpg": row.get("procedure", ""),
            "source_file": row.get("source_paths", "") or row.get("candidate_source_paths", ""),
            "candidate_rpt_index": rpt,
            "issue_type": issue_type,
            "evidence": evidence,
            "recommended_action": action,
            "severity": severity,
        }
    )


def _map_step_segments(
    segments: list[dict[str, Any]],
    expected_count: int,
    *,
    exact_single_rpt: bool,
    preferred_segment_index: int | None = None,
) -> tuple[dict[int, dict[str, Any]], str, str]:
    valid = [segment for segment in segments if segment.get("valid")]
    if preferred_segment_index is not None:
        selected = [
            segment
            for segment in valid
            if segment["segment_index"] == preferred_segment_index
        ]
        if len(selected) == 1:
            return {0: selected[0]}, "verified", ""
        return {}, "needs_curve_audit", "Selected baseline segment is absent or invalid."
    if expected_count == 1 and exact_single_rpt and len(valid) == 1:
        return {0: valid[0]}, "verified", ""
    if expected_count > 1 and len(valid) == expected_count:
        return (
            {index: segment for index, segment in enumerate(valid)},
            "verified",
            "Segments mapped to explicitly listed Guide RPTs in observed source-row order.",
        )
    if not valid:
        return {}, "missing_step", "No complete valid segment was found."
    return (
        {},
        "needs_curve_audit",
        f"Found {len(valid)} valid segment(s), expected {expected_count}; no ordinal assignment made.",
    )


def _scan_sources(
    root: Path,
    rows: list[dict[str, str]],
    baseline_catalog: dict[str, dict[str, str]],
    chunk_rows: int,
) -> tuple[dict[tuple[str, str], dict[str, Any]], list[dict[str, Any]]]:
    path_records: dict[tuple[str, str], dict[str, Any]] = {}
    identity_by_path: defaultdict[Path, set[str]] = defaultdict(set)
    aliases_by_path: defaultdict[Path, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if row.get("record_type") != "cycling_guide_record":
            continue
        if not _is_bol(row) and not _rpt_indices(row)[0]:
            continue
        part = row.get("dataset_part", "")
        for relative in _source_paths(row.get("source_paths", "")):
            key = (part, relative)
            path = _resolve_source(root, part, relative)
            aliases_by_path[path].add(key)
            if row.get("physical_cell_id"):
                identity_by_path[path].add(row["physical_cell_id"])
    for physical_cell_id, baseline in baseline_catalog.items():
        key = (baseline["part"], baseline["source_file"])
        path = _resolve_source(root, baseline["part"], baseline["source_file"])
        aliases_by_path[path].add(key)
        identity_by_path[path].add(physical_cell_id)

    issues: list[dict[str, Any]] = []
    work = sorted(aliases_by_path.items(), key=lambda item: str(item[0]))

    def scan_source(path: Path, aliases: set[tuple[str, str]]) -> tuple[Path, set[tuple[str, str]], dict[str, Any]]:
        primary_key = sorted(aliases)[0]
        part, relative = primary_key
        if path.is_file():
            scan = scan_tpg_file(path, chunk_rows)
        else:
            scan = {
                "source_file": str(path),
                "source_size_bytes": "",
                "source_mtime_ns": "",
                "rows_scanned": 0,
                "scan_status": "missing_source",
                "scan_error": "Mapped source file does not exist.",
                "steps": {str(step): [] for step in _STEPS},
            }
        scan["relative_path"] = relative
        scan["dataset_part"] = part
        scan["physical_cells"] = ";".join(sorted(identity_by_path[path]))
        return path, aliases, scan

    with ThreadPoolExecutor(max_workers=min(4, max(1, len(work)))) as executor:
        futures = {
            executor.submit(scan_source, path, aliases): index
            for index, (path, aliases) in enumerate(work, 1)
        }
        for future in as_completed(futures):
            index = futures[future]
            path, aliases, scan = future.result()
            part = scan["dataset_part"]
            relative = scan["relative_path"]
            for alias in aliases:
                path_records[alias] = scan
            if len(identity_by_path[path]) > 1:
                issue_row = {
                    "dataset_part": part,
                    "group_id": "",
                    "physical_cell_id": ";".join(sorted(identity_by_path[path])),
                    "procedure": Path(relative).stem,
                    "source_paths": relative,
                }
                _issue(
                    issues,
                    issue_row,
                    "source_identity_conflict",
                    f"One source path is linked to multiple physical cells: {sorted(identity_by_path[path])}.",
                    "Resolve source identity before using this file for any target.",
                    "high",
                )
            print(
                f"Scanned {index}/{len(work)} P{part} {relative}: "
                f"{scan['scan_status']} ({scan['rows_scanned']} rows)",
                flush=True,
            )
    return path_records, issues


def _baseline_for_cell(
    physical_cell_id: str,
    catalog: dict[str, dict[str, str]],
    source_records: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, str] | None, dict[str, Any] | None, str]:
    baseline = catalog.get(physical_cell_id)
    if not baseline:
        return None, None, "No selected BoL reference is available."
    scan = source_records.get((baseline["part"], baseline["source_file"]))
    if scan is None or scan["scan_status"] != "scanned":
        return baseline, None, "Selected BoL source was not scanned or is missing."
    segments = scan["steps"]["4"]
    try:
        segment_index = int(baseline["segment_index"])
    except ValueError:
        segment_index = 1
    selected = next(
        (segment for segment in segments if segment["segment_index"] == segment_index),
        None,
    )
    if not selected or not selected.get("valid"):
        return baseline, selected, "Selected BoL Step 4 segment failed curve QC."
    return baseline, selected, ""


def _build_observations(
    manifest: list[dict[str, str]],
    source_records: dict[tuple[str, str], dict[str, Any]],
    baseline_catalog: dict[str, dict[str, str]],
    half_cell_refs: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    candidate_rows, bol_rows = _source_stage_records(manifest)
    guide_by_record_id = {row["record_id"]: row for row in manifest}

    output: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_keys: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidate_rows:
        part = row["dataset_part"]
        source_paths = _source_paths(row.get("source_paths", ""))
        rpt = row["candidate_rpt_index"]
        expected_count = int(row["candidate_rpt_count_for_source"])
        identity_status = row.get("mapping_status", "")
        stage_indices, duplicate_stage_indices = _rpt_indices(row)
        has_provisional_mapping = "provisional" in identity_status
        source_file = source_paths[0] if len(source_paths) == 1 else ""
        source_scan = source_records.get((part, source_file)) if source_file else None
        source_missing = (
            not source_file
            or source_scan is None
            or source_scan["scan_status"] != "scanned"
            or identity_status in {
                "ambiguous_source_identity",
                "missing_source",
                "no_source_per_user_confirmation",
            }
        )
        cell = row.get("physical_cell_id", "")
        baseline, baseline_segment, baseline_error = _baseline_for_cell(
            cell, baseline_catalog, source_records
        )
        notes: list[str] = []
        if duplicate_stage_indices:
            notes.append("Guide stage repeats an RPT index.")
            _issue(
                issues,
                row,
                "duplicate_rpt_in_guide_stage",
                f"Guide stage {row.get('experimental_stage', '')!r} repeats RPT{rpt}.",
                "Resolve the Guide entry; do not deduplicate automatically.",
                "high",
                rpt,
            )

        if source_missing:
            step4_status = step8_status = step10_status = "missing_source"
            step4_map: dict[int, dict[str, Any]] = {}
            step8_map: dict[int, dict[str, Any]] = {}
            step10_map: dict[int, dict[str, Any]] = {}
            notes.append("Unique source file is unavailable or not mapped.")
            _issue(
                issues,
                row,
                "missing_source",
                f"Guide stage {row.get('experimental_stage', '')}; source candidates: {row.get('candidate_source_paths', '') or row.get('source_paths', '')}.",
                "Obtain or confirm the matching source file; do not substitute another cell.",
                "high",
                rpt,
            )
        else:
            assert source_scan is not None
            exact_single = expected_count == 1 and identity_status in {
                "verified_unique_source",
                "user_confirmed_cell_identity",
                "user_confirmed_identity_by_group_membership",
                "verified_guide_eol_source",
            }
            step4_map, step4_status, step4_note = _map_step_segments(
                source_scan["steps"]["4"],
                expected_count,
                exact_single_rpt=exact_single,
            )
            step8_map, step8_status, step8_note = _map_step_segments(
                source_scan["steps"]["8"],
                expected_count,
                exact_single_rpt=exact_single,
            )
            step10_map, step10_status, step10_note = _map_step_segments(
                source_scan["steps"]["10"],
                expected_count,
                exact_single_rpt=exact_single,
            )
            if has_provisional_mapping:
                step4_count = sum(segment.get("valid", False) for segment in source_scan["steps"]["4"])
                step8_count = sum(segment.get("valid", False) for segment in source_scan["steps"]["8"])
                step10_count = sum(segment.get("valid", False) for segment in source_scan["steps"]["10"])
                step4_map, step8_map, step10_map = {}, {}, {}
                step4_status = step8_status = step10_status = "needs_curve_audit"
                step4_note = (
                    f"Provisional RPT mapping: {step4_count} valid Step 4 segment(s) "
                    f"for {expected_count} candidate RPTs; no segment assigned."
                )
                step8_note = (
                    f"Provisional RPT mapping: {step8_count} valid Step 8 segment(s) "
                    f"for {expected_count} candidate RPTs; no segment assigned."
                )
                step10_note = (
                    f"Provisional RPT mapping: {step10_count} valid Step 10 segment(s) "
                    f"for {expected_count} candidate RPTs; no segment assigned."
                )
            segment_position = stage_indices.index(int(rpt))
            step4_segment = step4_map.get(segment_position)
            step8_segment = step8_map.get(segment_position)
            step10_segment = step10_map.get(segment_position)
            for status, detail in (
                (step4_status, step4_note),
                (step8_status, step8_note),
                (step10_status, step10_note),
            ):
                if detail:
                    notes.append(detail)
            if step4_status == "missing_step":
                _issue(
                    issues,
                    row,
                    "missing_step4",
                    f"No complete Step 4 discharge segment in {source_file}.",
                    "Keep this RPT out of SoH estimation until its Step 4 curve is resolved.",
                    "high",
                    rpt,
                )
            elif step4_status == "needs_curve_audit":
                _issue(
                    issues,
                    row,
                    "needs_curve_audit",
                    step4_note,
                    "Inspect raw Step 4 boundaries and assign a segment only with evidence.",
                    "high",
                    rpt,
                )
            if step8_status != "verified":
                _issue(
                    issues,
                    row,
                    "missing_step8" if step8_status == "missing_step" else "needs_curve_audit",
                    step8_note,
                    "Inspect Step 8 discharge segment boundaries and preserve discharge direction.",
                    "high",
                    rpt,
                )
            if step10_status != "verified":
                _issue(
                    issues,
                    row,
                    "missing_step10" if step10_status == "missing_step" else "needs_curve_audit",
                    step10_note,
                    "Inspect Step 10 charge segment boundaries and preserve charge direction.",
                    "high",
                    rpt,
                )

        if source_missing:
            step4_segment = step8_segment = step10_segment = None
        if has_provisional_mapping:
            notes.append("Manifest mapping is explicitly provisional; observation is not marked verified.")
        if baseline_error:
            notes.append(baseline_error)
            _issue(
                issues,
                row,
                "bol_reference_unverified",
                baseline_error,
                "Validate the selected same-cell BoL Step 4 segment before using this RPT for SoH.",
                "high",
                rpt,
            )

        pydma_direction_refs_ok = all(
            half_cell_refs[key]["status"] == "verified_reference_columns_and_finite_pairs"
            for key in ("charge_anode", "charge_cathode", "discharge_anode", "discharge_cathode")
        )
        step_mapping_evidence = (
            step4_status == "verified"
            or (step8_status == "verified" and step10_status == "verified")
        )
        compound_supported_by_curve_order = (
            len(stage_indices) > 1
            and identity_status == "ambiguous_compound_guide_stage"
            and step_mapping_evidence
        )
        mapping_verified = (
            not source_missing
            and not has_provisional_mapping
            and (
                identity_status in {
                    "verified_unique_source",
                    "user_confirmed_cell_identity",
                    "user_confirmed_identity_by_group_membership",
                    "verified_guide_eol_source",
                    "provisional_baseline_primary",
                }
                or compound_supported_by_curve_order
            )
        )
        overall_status = (
            "missing_source"
            if source_missing
            else "ambiguous_rpt"
            if identity_status == "ambiguous_source_identity"
            or (
                identity_status == "ambiguous_compound_guide_stage"
                and not compound_supported_by_curve_order
            )
            else "needs_curve_audit"
            if has_provisional_mapping or "needs_curve_audit" in (step4_status, step8_status, step10_status)
            else "missing_step"
            if "missing_step" in (step4_status, step8_status, step10_status)
            else "verified"
        )
        ready_soh = bool(
            mapping_verified
            and step4_status == "verified"
            and step4_segment
            and baseline_segment
        )
        ready_degradation = bool(
            mapping_verified
            and step8_status == "verified"
            and step10_status == "verified"
            and step8_segment
            and step10_segment
            and pydma_direction_refs_ok
        )
        if step8_segment and step10_segment:
            if step8_segment["source_row_start"] >= step10_segment["source_row_start"]:
                notes.append("Step 8 discharge does not precede its Step 10 charge segment in source-row order.")
                ready_degradation = False
                _issue(
                    issues,
                    row,
                    "step_order_anomaly",
                    "Mapped Step 8 source rows do not precede mapped Step 10 source rows.",
                    "Review the RPT boundary and charge/discharge pair before PyDMA use.",
                    "high",
                    rpt,
                )
        if not mapping_verified:
            notes.append(f"Source/RPT attribution remains {overall_status}.")
        if not pydma_direction_refs_ok:
            notes.append("One or more direction-specific half-cell references failed basic file/curve checks.")

        record = {
            "part": part,
            "group": row.get("group_id", ""),
            "physical_cell_id": cell,
            "rpt_index": rpt,
            "source_tpg": row.get("procedure", ""),
            "source_file": source_file or row.get("source_paths", ""),
            "source_record_id": row.get("record_id", ""),
            "guide_stage": row.get("experimental_stage", ""),
            "split": row.get("split", ""),
            "bol_reference_file": baseline["source_file"] if baseline else "",
            "bol_step4_segment_ref": _segment_ref(
                baseline["source_file"] if baseline else "", baseline_segment
            ),
            "step4_segment_ref": _segment_ref(source_file, step4_segment),
            "step8_segment_ref": _segment_ref(source_file, step8_segment),
            "step10_segment_ref": _segment_ref(source_file, step10_segment),
            "mapping_status": overall_status,
            "step4_status": step4_status,
            "step8_status": step8_status,
            "step10_status": step10_status,
            "ready_soh": str(ready_soh).lower(),
            "ready_degradation": str(ready_degradation).lower(),
            "needs_audit": str(overall_status != "verified").lower(),
            "audit_notes": " ".join(dict.fromkeys(notes)),
        }
        output.append(record)
        seen_keys[(cell, rpt)].append(record)

    for (cell, rpt), records in seen_keys.items():
        if len(records) > 1:
            for record in records:
                record["mapping_status"] = "ambiguous_rpt"
                record["ready_soh"] = "false"
                record["ready_degradation"] = "false"
                record["needs_audit"] = "true"
                record["audit_notes"] = (
                    record["audit_notes"] + " Duplicate physical-cell/RPT key."
                ).strip()
            source_row = guide_by_record_id.get(records[0]["source_record_id"], {})
            _issue(
                issues,
                source_row,
                "duplicate_rpt_assignment",
                f"{len(records)} source observations map to {cell} RPT{rpt}.",
                "Resolve duplicate/overlapping Guide mappings; do not score both rows.",
                "high",
                rpt,
            )
    for row in bol_rows:
        if not row.get("physical_cell_id"):
            continue
        if row.get("mapping_status") in {"ambiguous_source_identity", "missing_source"}:
            _issue(
                issues,
                row,
                "ambiguous_bol_reference",
                f"BoL source mapping status is {row.get('mapping_status')}: {row.get('source_paths', '')}",
                "Confirm the same-cell reference source and selected complete Step 4 segment.",
                "high",
            )
    return output, issues


def _report(
    observations: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    baseline_catalog: dict[str, dict[str, str]],
    source_records: dict[tuple[str, str], dict[str, Any]],
    half_cell_refs: dict[str, dict[str, Any]],
) -> str:
    unique_scans = {id(record): record for record in source_records.values()}
    lines = [
        "# RPT mapping and BaFuse input audit",
        "",
        "This report was generated by `python -m bafuse.data.rpt_mapping_audit` from canonical manifest v3 and the current raw CSV sources. It does not read or infer cycle-age.",
        "",
        "## Overall counts",
        "",
        f"- Physical cells represented by RPT candidates: {len({row['physical_cell_id'] for row in observations if row['physical_cell_id']})}.",
        f"- RPT candidate rows: {len(observations)}.",
        f"- Verified observations: {sum(row['mapping_status'] == 'verified' for row in observations)}.",
        f"- Ambiguous RPT observations: {sum(row['mapping_status'] == 'ambiguous_rpt' for row in observations)}.",
        f"- Missing-source observations: {sum(row['mapping_status'] == 'missing_source' for row in observations)}.",
        f"- Guide entries with no source file: {sum(issue['issue_type'] == 'missing_source' for issue in issues)}.",
        f"- Duplicate physical-cell/RPT assignments: {sum(issue['issue_type'] == 'duplicate_rpt_assignment' for issue in issues)}.",
        f"- Observations with a verified Step 4 curve: {sum(row['step4_status'] == 'verified' for row in observations)}.",
        f"- Observations with verified Step 8 and Step 10 curves: {sum(row['step8_status'] == 'verified' and row['step10_status'] == 'verified' for row in observations)}.",
        f"- Ready for SoH: {sum(row['ready_soh'] == 'true' for row in observations)}.",
        f"- Ready for degradation: {sum(row['ready_degradation'] == 'true' for row in observations)}.",
        f"- Selected BoL references: {len(baseline_catalog)}; scanned source files: {len(unique_scans)}.",
        "",
        "Eligibility is independent: `ready_soh` requires a verified same-cell Step 4 observation and a validated selected BoL Step 4 reference. `ready_degradation` requires verified, ordered Step 8 discharge and Step 10 charge curves plus finite direction-specific half-cell reference inputs. The row/span checks are a base structural gate, not proof of full protocol completeness or an accepted PyDMA fit; these flags are not model outputs or ground-truth labels.",
        "",
        "## Counts by dataset part",
        "",
        "| Part | Cells | RPT candidates | Verified | Ambiguous | Missing source | Step 4 valid | Step 8+10 valid | Ready SoH | Ready degradation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for part in ("1", "2", "3"):
        rows = [row for row in observations if row["part"] == part]
        lines.append(
            "| "
            + " | ".join(
                str(value)
                for value in (
                    part,
                    len({row["physical_cell_id"] for row in rows if row["physical_cell_id"]}),
                    len(rows),
                    sum(row["mapping_status"] == "verified" for row in rows),
                    sum(row["mapping_status"] == "ambiguous_rpt" for row in rows),
                    sum(row["mapping_status"] == "missing_source" for row in rows),
                    sum(row["step4_status"] == "verified" for row in rows),
                    sum(row["step8_status"] == "verified" and row["step10_status"] == "verified" for row in rows),
                    sum(row["ready_soh"] == "true" for row in rows),
                    sum(row["ready_degradation"] == "true" for row in rows),
                )
            )
            + " |"
        )
    lines.extend(["", "## Step and half-cell criteria", ""])
    lines.extend(
        [
            f"- Step 4 requires at least {_MIN_CURVE_ROWS} finite capacity/voltage pairs, positive capacity span, negative discharge current, and no positive-current rows. This follows the existing Step 4 audit's 10-row/span checks and adds the protocol's discharge-direction requirement.",
            f"- Step 8/10 require at least {_MIN_CURVE_ROWS} finite pairs and strictly increasing capacity with positive span, using the existing PyDMA direction filter (negative current for Step 8; positive current for Step 10; |current| ≤ {_POCV_CURRENT_LIMIT_A} A).",
            "- Half-cell references are checked separately for expected SoC/Volts columns and finite pairs, with observed SoC accepted over its source range 0–100. Charge uses `CH_HC_anode.csv` + `CH_HC_cathode.csv`; discharge uses `DCH_HC_anode.csv` + `DHC_HC_cathode.csv` (the cathode filename spelling in the dataset is `DHC`).",
            "- A compound source is expanded only from explicit RPT numbers/ranges in the Guide stage. Segment assignment is by observed source-row order only when the valid segment count equals the explicit RPT count. Mismatches remain unassigned and flagged.",
            "- Step 4 and Step 8/10 are assessed independently; absence of one modality does not delete a usable observation for the other target.",
            "",
            "### Half-cell reference status",
            "",
            "| Direction/electrode | Source | Status | Finite pairs |",
            "|---|---|---|---:|",
        ]
    )
    for key, result in half_cell_refs.items():
        lines.append(
            f"| {key} | `{result['source_file']}` | {result['status']} | {result['finite_pairs']} |"
        )
    lines.extend(["", "## Group 5 Cell6 range audit", ""])
    g5 = [
        row
        for row in observations
        if row["part"] == "2"
        and row["physical_cell_id"] == "G5-Cell6"
        and row["source_tpg"] in {"TPG5.9", "TPG5.10"}
    ]
    for tpg in ("TPG5.9", "TPG5.10"):
        rows = [row for row in g5 if row["source_tpg"] == tpg]
        source = rows[0]["source_file"] if rows else ""
        scan = source_records.get(("2", source), {})
        step_data = scan.get("steps", {})
        lines.append(
            f"- `{tpg}`: {len(rows)} candidate RPT rows; structurally valid monotone segment counts "
            f"(not RPT assignments) are "
            f"Step 4={sum(s['valid'] for s in step_data.get('4', []))}, "
            f"Step 8={sum(s['valid'] for s in step_data.get('8', []))}, "
            f"Step 10={sum(s['valid'] for s in step_data.get('10', []))}; "
            f"manifest mapping statuses: {', '.join(sorted({row['mapping_status'] for row in rows})) or 'not present'}. "
            "Cyc and cycle-age are not read."
        )
    lines.extend(["", "## Unresolved issues", ""])
    if not issues:
        lines.append("No mapping issues were detected by the configured checks.")
    else:
        grouped_issues: defaultdict[tuple[str, ...], set[str]] = defaultdict(set)
        for issue in issues:
            key = (
                issue["severity"],
                issue["issue_type"],
                issue["part"],
                issue["group"],
                issue["physical_cell_id"],
                issue["source_tpg"],
                issue["source_file"],
                issue["evidence"],
                issue["recommended_action"],
            )
            for value in (issue["candidate_rpt_index"] or "").split(","):
                if value.strip():
                    grouped_issues[key].add(value.strip())
            if not issue["candidate_rpt_index"]:
                grouped_issues[key].add("")
        lines.append(
            f"{len(issues)} candidate-level issue rows are grouped below by source and evidence; "
            "see `rpt_mapping_issues.csv` for the detailed rows."
        )
        for key in sorted(grouped_issues):
            severity, issue_type, part, group, cell, tpg, source, evidence, action = key
            rpt_values = sorted(
                (value for value in grouped_issues[key] if value),
                key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
            )
            rpt_label = (
                "RPT" + ", ".join(rpt_values)
                if rpt_values
                else "Guide entry"
            )
            lines.append(
                f"- **{severity.upper()} {issue_type}** — P{part} G{group} {cell} "
                f"`{tpg}` {rpt_label} (`{source}`): {evidence} Action: {action}"
            )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- Segment counts and source-row boundaries are read from CSV data; they do not prove original tester-log cycle boundaries. Any Guide/curve ordinal match is stated as such and is not upgraded where the manifest identifies a provisional or ambiguous mapping.",
            "- Capacity uses the source `Amphr` column; this audit does not assert absolute units.",
            "- PyDMA-derived LLI/LAM values are diagnostic estimates, not measured ground truth. This audit does not run PyDMA or train models.",
            "- Part 3 remains external test. Its records are audited for mapping/readiness but are not used to fit or tune model parameters.",
            "- Input artifacts: `data/manifests/canonical_observation_manifest_v3.csv`; existing SOH baseline-selection audits; raw TPG CSVs; half-cell reference CSVs.",
            "",
        ]
    )
    return "\n".join(lines)


def run_audit(
    root: Path,
    manifest_path: Path,
    observations_path: Path,
    report_path: Path,
    issues_path: Path,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Canonical manifest does not exist: {manifest_path}")
    manifest = _read_csv(manifest_path)
    required = {
        "record_id",
        "record_type",
        "dataset_part",
        "physical_cell_id",
        "procedure",
        "experimental_stage",
        "source_paths",
        "mapping_status",
    }
    missing = required.difference(manifest[0] if manifest else {})
    if missing:
        raise ValueError(f"Canonical manifest is missing required columns: {sorted(missing)}")
    if manifest_path.name != "canonical_observation_manifest_v3.csv":
        raise ValueError("This audit is pinned to canonical_observation_manifest_v3.csv.")
    baseline_catalog = _baseline_catalog(root, manifest)
    source_records, source_issues = _scan_sources(
        root, manifest, baseline_catalog, chunk_rows
    )
    half_cell_refs = _half_cell_reference_audit(root)
    observations, issues = _build_observations(
        manifest, source_records, baseline_catalog, half_cell_refs
    )
    issues = source_issues + issues
    issues.sort(
        key=lambda row: (
            row.get("part", ""),
            row.get("group", ""),
            row.get("physical_cell_id", ""),
            row.get("source_tpg", ""),
            row.get("candidate_rpt_index", ""),
            row.get("issue_type", ""),
        )
    )
    # Rows with an explicit Guide RPT but no matching manifest source are retained as issues.
    for row in manifest:
        if row.get("record_type") != "unresolved_guide_entry":
            continue
        rpt_indices, _ = parse_rpt_stage(row.get("experimental_stage", ""))
        _issue(
            issues,
            row,
            "missing_source",
            (
                f"Unresolved Guide entry with explicit RPT metadata: {row.get('experimental_stage', '')}."
                if rpt_indices or row.get("measurement_id", "").upper().startswith("RPT")
                else f"Guide marks this source unavailable ({row.get('experimental_stage', '') or 'no stage'})."
            ),
            "Retain the no-source record; do not substitute another procedure or cell.",
            "high",
            ",".join(map(str, rpt_indices)),
        )
    for row in manifest:
        if (
            row.get("record_type") != "cycling_guide_record"
            or "provisional" not in row.get("mapping_status", "")
            or "guide lists" not in row.get("experimental_stage", "").casefold()
        ):
            continue
        primary_stage, _, annotation = row["experimental_stage"].partition("(")
        candidate_indices, _ = parse_rpt_stage(primary_stage)
        guide_text = annotation.partition("Guide lists")[2]
        guide_indices, _ = parse_rpt_stage(guide_text)
        if candidate_indices and guide_indices and candidate_indices != guide_indices:
            _issue(
                issues,
                row,
                "guide_candidate_range_conflict",
                f"Provisional mapping lists RPT{candidate_indices[0]}-RPT{candidate_indices[-1]}; "
                f"Guide annotation lists RPT{guide_indices[0]}-RPT{guide_indices[-1]}.",
                "Resolve the overlapping RPT24 assignment between TPG5.9 and TPG5.10 from independent curve-boundary evidence.",
                "high",
                ",".join(map(str, sorted(set(candidate_indices) | set(guide_indices)))),
            )
    duplicate_keys = Counter(
        (row["physical_cell_id"], row["rpt_index"]) for row in observations
    )
    summary = {
        "observation_rows": len(observations),
        "issue_rows": len(issues),
        "source_files_scanned": len({id(record) for record in source_records.values()}),
        "scan_statuses": dict(
            Counter(record["scan_status"] for record in {id(r): r for r in source_records.values()}.values())
        ),
        "mapping_statuses": dict(Counter(row["mapping_status"] for row in observations)),
        "duplicate_rpt_keys": sum(count > 1 for count in duplicate_keys.values()),
        "ready_soh": sum(row["ready_soh"] == "true" for row in observations),
        "ready_degradation": sum(row["ready_degradation"] == "true" for row in observations),
        "part3_rows_external_only": all(
            row["split"] == "external_test" for row in observations if row["part"] == "3"
        ),
    }
    report = _report(
        observations, issues, baseline_catalog, source_records, half_cell_refs
    )
    outputs = [
        (observations_path, observations, _OBSERVATION_FIELDS),
        (issues_path, issues, _ISSUE_FIELDS),
    ]
    for path, _, _ in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_paths: list[Path] = []
    try:
        for path, rows, fields in outputs:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent, suffix=".tmp"
            ) as stream:
                temp = Path(stream.name)
                temp_paths.append(temp)
                _write_csv(temp, rows, fields)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", delete=False, dir=report_path.parent, suffix=".tmp"
        ) as stream:
            report_temp = Path(stream.name)
            temp_paths.append(report_temp)
            stream.write(report)
        for temp, (path, _, _) in zip(temp_paths[:2], outputs):
            temp.replace(path)
        report_temp.replace(report_path)
    except Exception:
        for temp in temp_paths:
            if temp.exists():
                temp.unlink()
        raise
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Guide-to-RPT mappings and Step 4/8/10 curve boundaries from source CSVs."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/canonical_observation_manifest_v3.csv"),
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/manifests/bafuse_rpt_observations.csv"),
    )
    parser.add_argument("--report", type=Path, default=Path("reports/rpt_mapping_audit.md"))
    parser.add_argument("--issues", type=Path, default=Path("reports/rpt_mapping_issues.csv"))
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else root / args.manifest
    observations = args.observations if args.observations.is_absolute() else root / args.observations
    report = args.report if args.report.is_absolute() else root / args.report
    issues = args.issues if args.issues.is_absolute() else root / args.issues
    summary = run_audit(root, manifest, observations, report, issues, args.chunk_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
