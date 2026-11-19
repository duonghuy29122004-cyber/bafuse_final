from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .rpt_mapping_audit import (
    _finalize_raw_segment,
    _half_cell_reference_audit,
    _read_csv,
    _resolve_source,
    parse_rpt_stage,
)


_REVIEW_FIELDS = [
    "review_scope",
    "part",
    "group",
    "physical_cell_id",
    "rpt_index",
    "source_tpg",
    "source_file",
    "source_record_id",
    "guide_stage",
    "split",
    "original_mapping_status",
    "proposed_mapping_status",
    "block_index",
    "block_row_start",
    "block_row_end",
    "step4_segment_ref",
    "step4_status",
    "step4_rows",
    "step4_capacity_span_native",
    "step4_current_min_a",
    "step4_current_max_a",
    "step4_voltage_min_v",
    "step4_voltage_max_v",
    "step4_valid_subsegment_count",
    "step6_segment_ref",
    "step6_status",
    "step6_current_min_a",
    "step6_current_max_a",
    "step6_voltage_min_v",
    "step6_voltage_max_v",
    "step7_rest_ref",
    "step8_segment_ref",
    "step8_status",
    "step8_rows",
    "step8_capacity_span_native",
    "step8_current_min_a",
    "step8_current_max_a",
    "step8_voltage_min_v",
    "step8_voltage_max_v",
    "step8_valid_subsegment_count",
    "step8_direction_fraction",
    "step9_rest_ref",
    "step10_segment_ref",
    "step10_status",
    "step10_rows",
    "step10_capacity_span_native",
    "step10_current_min_a",
    "step10_current_max_a",
    "step10_voltage_min_v",
    "step10_voltage_max_v",
    "step10_valid_subsegment_count",
    "step10_direction_fraction",
    "step11_rest_ref",
    "block_test_time_start",
    "block_test_time_end",
    "pulse_signature_status",
    "step4_mapping_candidate",
    "ready_soh_candidate",
    "soh_block_reason",
    "ready_degradation_candidate",
    "evidence",
    "candidate_block_evidence",
]
_UPDATE_FIELDS = [
    "part",
    "group",
    "physical_cell_id",
    "rpt_index",
    "source_tpg",
    "source_file",
    "source_record_id",
    "original_mapping_status",
    "proposed_mapping_status",
    "step4_segment_ref",
    "step4_status",
    "step6_segment_ref",
    "step6_status",
    "step8_segment_ref",
    "step8_status",
    "step10_segment_ref",
    "step10_status",
    "step4_mapping_candidate",
    "ready_soh_candidate",
    "soh_block_reason",
    "ready_degradation_candidate",
    "evidence",
]
_CURVE_STEPS = {4, 8, 10}
_PULSE_SEQUENCE = (14, 16, 18, 22, 24, 26, 30, 32, 34)
_MIN_CURVE_ROWS = 10
_POCV_CURRENT_LIMIT_A = 0.2
_STEP_COLUMNS = ("Step", "TestTime", "StepTime", "Amphr", "Amps", "Volts")
_GUIDE_LIST_RE = re.compile(r"Guide lists\s+([^)]*)", re.IGNORECASE)
_G5_CELL6_CONFIRMED_RPT_RANGES = {
    ("G5-Cell6", "TPG5.9"): tuple(range(17, 24)),
    ("G5-Cell6", "TPG5.10"): tuple(range(24, 35)),
}


@dataclass
class StepRun:
    step: int
    source_row_start: int
    source_row_end: int
    row_count: int = 0
    test_time_start: float | None = None
    test_time_end: float | None = None
    step_time_start: float | None = None
    step_time_end: float | None = None
    capacity_min: float | None = None
    capacity_max: float | None = None
    voltage_min: float | None = None
    voltage_max: float | None = None
    current_min: float | None = None
    current_max: float | None = None
    finite_current_rows: int = 0
    positive_current_rows: int = 0
    negative_current_rows: int = 0
    pocv_direction_rows: int = 0
    curve_segments: list[dict[str, Any]] = field(default_factory=list)
    _capacity_parts: list[np.ndarray] = field(default_factory=list, repr=False)
    _voltage_parts: list[np.ndarray] = field(default_factory=list, repr=False)
    _current_parts: list[np.ndarray] = field(default_factory=list, repr=False)


@dataclass
class SourceScan:
    status: str
    error: str
    rows_scanned: int
    runs: list[StepRun]


def _update_run(
    run: StepRun,
    arrays: dict[str, np.ndarray],
    start: int,
    end: int,
    row_end: int,
) -> None:
    run.row_count += end - start
    run.source_row_end = row_end
    for column, start_name, end_name in (
        ("TestTime", "test_time_start", "test_time_end"),
        ("StepTime", "step_time_start", "step_time_end"),
    ):
        values = arrays.get(column)
        if values is not None:
            finite = values[start:end][np.isfinite(values[start:end])]
            if finite.size:
                if getattr(run, start_name) is None:
                    setattr(run, start_name, float(finite[0]))
                setattr(run, end_name, float(finite[-1]))
    for column, low_name, high_name in (
        ("Amphr", "capacity_min", "capacity_max"),
        ("Volts", "voltage_min", "voltage_max"),
        ("Amps", "current_min", "current_max"),
    ):
        values = arrays[column][start:end]
        finite = values[np.isfinite(values)]
        if finite.size:
            low, high = float(finite.min()), float(finite.max())
            existing_low, existing_high = getattr(run, low_name), getattr(run, high_name)
            setattr(run, low_name, low if existing_low is None else min(existing_low, low))
            setattr(run, high_name, high if existing_high is None else max(existing_high, high))
    current = arrays["Amps"][start:end]
    finite_current = current[np.isfinite(current)]
    run.finite_current_rows += int(finite_current.size)
    run.positive_current_rows += int((finite_current > 0.01).sum())
    run.negative_current_rows += int((finite_current < -0.01).sum())
    if run.step in {8, 10}:
        if run.step == 8:
            run.pocv_direction_rows += int(
                (np.isfinite(current) & (current < 0) & (np.abs(current) <= _POCV_CURRENT_LIMIT_A)).sum()
            )
        else:
            run.pocv_direction_rows += int(
                (np.isfinite(current) & (current > 0) & (np.abs(current) <= _POCV_CURRENT_LIMIT_A)).sum()
            )
    if run.step in _CURVE_STEPS:
        run._capacity_parts.append(arrays["Amphr"][start:end])
        run._voltage_parts.append(arrays["Volts"][start:end])
        run._current_parts.append(current)


def _finish_run(run: StepRun, segment_index: int) -> StepRun:
    if run.step in _CURVE_STEPS:
        raw = {
            "step": run.step,
            "segment_index": segment_index,
            "source_row_start": run.source_row_start,
            "source_rows": run.row_count,
            "capacity_parts": run._capacity_parts,
            "voltage_parts": run._voltage_parts,
            "current_parts": run._current_parts,
        }
        run.curve_segments = _finalize_raw_segment(raw)
        run._capacity_parts.clear()
        run._voltage_parts.clear()
        run._current_parts.clear()
    return run


def scan_raw_tpg(path: Path, chunk_rows: int = 100_000) -> SourceScan:
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive.")
    columns = list(pd.read_csv(path, nrows=0).columns)
    required = {"Step", "Amphr", "Volts", "Amps"}
    missing = required.difference(columns)
    if missing:
        return SourceScan(
            "missing_required_columns",
            "Missing required columns: " + ", ".join(sorted(missing)),
            0,
            [],
        )
    usecols = [column for column in _STEP_COLUMNS if column in columns]
    runs: list[StepRun] = []
    active: StepRun | None = None
    row_offset = 0
    try:
        for chunk in pd.read_csv(
            path, usecols=usecols, chunksize=chunk_rows, low_memory=False
        ):
            arrays = {
                column: pd.to_numeric(chunk[column], errors="coerce").to_numpy(float)
                for column in usecols
            }
            step_values = np.where(
                np.isfinite(arrays["Step"]), arrays["Step"], -1
            ).astype(np.int32)
            boundaries = np.r_[0, np.flatnonzero(np.diff(step_values) != 0) + 1, len(chunk)]
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                step = int(step_values[start])
                source_start = row_offset + int(start) + 2
                source_end = row_offset + int(end) + 1
                if active is not None and active.step != step:
                    runs.append(_finish_run(active, len(runs) + 1))
                    active = None
                if active is None:
                    active = StepRun(step, source_start, source_end)
                _update_run(active, arrays, int(start), int(end), source_end)
            row_offset += len(chunk)
    except (OSError, UnicodeError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as error:
        return SourceScan(
            "scan_failed",
            f"{type(error).__name__}: {error}",
            row_offset,
            runs,
        )
    if active is not None:
        runs.append(_finish_run(active, len(runs) + 1))
    return SourceScan("scanned", "", row_offset, runs)


def _curve_quality(run: StepRun | None, source_file: str) -> tuple[str, str, str]:
    if run is None:
        return "", "missing_step", ""
    valid = [segment for segment in run.curve_segments if segment.get("valid")]
    if not valid:
        return _run_ref(source_file, run), "needs_curve_audit", ""
    if len(valid) != 1:
        refs = ";".join(_curve_ref(source_file, segment) for segment in valid)
        return refs, "needs_curve_audit", ""
    direction_fraction = (
        run.pocv_direction_rows / run.finite_current_rows
        if run.step in {8, 10} and run.finite_current_rows
        else 1.0
    )
    if direction_fraction < 0.99:
        return _curve_ref(source_file, valid[0]), "needs_curve_audit", ""
    return _curve_ref(source_file, valid[0]), "verified", ""


def _run_ref(source_file: str, run: StepRun | None) -> str:
    if run is None:
        return ""
    return f"{source_file}#Step{run.step}:rows{run.source_row_start}-{run.source_row_end}"


def _metric(run: StepRun | None, name: str) -> str:
    value = getattr(run, name) if run is not None else None
    return "" if value is None else f"{value:.8g}"


def _capacity_span(run: StepRun | None) -> str:
    if run is None or run.capacity_min is None or run.capacity_max is None:
        return ""
    return f"{run.capacity_max - run.capacity_min:.8g}"


def _valid_curve_count(run: StepRun | None) -> int:
    if run is None:
        return 0
    return sum(bool(segment.get("valid")) for segment in run.curve_segments)


def _direction_fraction(run: StepRun | None) -> str:
    if run is None or run.step not in {8, 10} or not run.finite_current_rows:
        return ""
    return f"{run.pocv_direction_rows / run.finite_current_rows:.8g}"


def _block_evidence(block: dict[str, Any], source_file: str) -> dict[str, Any]:
    steps: dict[str, Any] = {}
    for step in (4, 6, 7, 8, 9, 10, 11):
        run = block.get(f"step{step}")
        if run is None:
            steps[str(step)] = None
            continue
        references = (
            [_curve_ref(source_file, segment) for segment in run.curve_segments if segment.get("valid")]
            if step in _CURVE_STEPS
            else [_run_ref(source_file, run)]
        )
        steps[str(step)] = {
            "row_ref": _run_ref(source_file, run),
            "curve_refs": references,
            "rows": run.row_count,
            "test_time_start_native": run.test_time_start,
            "test_time_end_native": run.test_time_end,
            "capacity_span_native": (
                run.capacity_max - run.capacity_min
                if run.capacity_min is not None and run.capacity_max is not None
                else None
            ),
            "current_min_a": run.current_min,
            "current_max_a": run.current_max,
            "voltage_min_v": run.voltage_min,
            "voltage_max_v": run.voltage_max,
            "valid_curve_segment_count": _valid_curve_count(run),
        }
    return {
        "block_index": block["block_index"],
        "row_start": block["source_row_start"],
        "row_end": block["source_row_end"],
        "core_sequence_verified": block["core_sequence_verified"],
        "pulse_signature_status": block["pulse_signature_status"],
        "steps": steps,
    }


def _curve_ref(source_file: str, segment: dict[str, Any]) -> str:
    subsegment = segment.get("subsegment_index")
    suffix = f":sub{subsegment}" if subsegment is not None else ""
    return (
        f"{source_file}#Step{segment['step']}:segment{segment['segment_index']}:"
        f"rows{segment['source_row_start']}-{segment['source_row_end']}{suffix}"
    )


def _current_direction_status(run: StepRun | None) -> str:
    if run is None or not run.finite_current_rows:
        return "missing_step"
    negative_fraction = run.negative_current_rows / run.finite_current_rows
    if (
        run.positive_current_rows > 0
        and negative_fraction <= 0.01
        and (run.voltage_max or 0) >= 4.0
    ):
        return "verified_charge_direction"
    return "needs_curve_audit"


def _pulse_status(section: list[StepRun], step11_position: int | None) -> str:
    if step11_position is None:
        return "not_evaluable"
    candidates = [
        (index, run.step)
        for index, run in enumerate(section)
        if index > step11_position and run.step in _PULSE_SEQUENCE
    ]
    positions = [index for index, _ in candidates]
    steps = [step for _, step in candidates]
    if (
        all(steps.count(step) == 1 for step in _PULSE_SEQUENCE)
        and [step for step in steps if step in _PULSE_SEQUENCE] == list(_PULSE_SEQUENCE)
        and positions == sorted(positions)
    ):
        return "complete_ordered_pulse_signature"
    return "partial_or_nonunique_pulse_signature" if candidates else "not_observed"


def _is_rest_run(run: StepRun) -> bool:
    return bool(
        run.row_count >= _MIN_CURVE_ROWS
        and run.finite_current_rows
        and (run.current_min or 0) >= -0.02
        and (run.current_max or 0) <= 0.02
    )


def _has_valid_curve(run: StepRun) -> bool:
    return any(segment.get("valid") for segment in run.curve_segments)


def analyze_rpt_blocks(runs: list[StepRun]) -> list[dict[str, Any]]:
    step4_positions = [index for index, run in enumerate(runs) if run.step == 4]
    blocks: list[dict[str, Any]] = []
    for block_index, begin in enumerate(step4_positions, start=1):
        end = (
            step4_positions[block_index]
            if block_index < len(step4_positions)
            else len(runs)
        )
        section = runs[begin:end]
        position_by_step: dict[int, list[int]] = defaultdict(list)
        for index, run in enumerate(section):
            position_by_step[run.step].append(index)

        sequences: set[tuple[int, int, int, int, int, int]] = set()
        for step8_position in position_by_step[8]:
            step8_run = section[step8_position]
            if not _has_valid_curve(step8_run):
                continue
            prior_step7 = [
                index
                for index in position_by_step[7]
                if index < step8_position and _is_rest_run(section[index])
            ]
            if not prior_step7:
                continue
            step7_position = prior_step7[-1]
            next_step9 = [
                index
                for index in position_by_step[9]
                if index > step8_position and _is_rest_run(section[index])
            ]
            if not next_step9:
                continue
            step9_position = next_step9[0]
            if any(
                section[index].step == 8 and _has_valid_curve(section[index])
                for index in range(step7_position + 1, step9_position)
                if index != step8_position
            ):
                continue
            step10_positions = [
                index
                for index in position_by_step[10]
                if index > step9_position and _has_valid_curve(section[index])
            ]
            if len(step10_positions) != 1:
                continue
            step10_position = step10_positions[0]
            next_step11 = [
                index
                for index in position_by_step[11]
                if index > step10_position and _is_rest_run(section[index])
            ]
            if not next_step11:
                continue
            step11_position = next_step11[0]
            if any(
                _has_valid_curve(section[index])
                for index in position_by_step[10]
                if step10_position < index < step11_position
            ):
                continue
            step6_position = step7_position - 1
            if step6_position < 1 or section[step6_position].step != 6:
                continue
            sequences.add(
                (
                    step6_position,
                    step7_position,
                    step8_position,
                    step9_position,
                    step10_position,
                    step11_position,
                )
            )

        selected = next(iter(sequences)) if len(sequences) == 1 else None
        step6, step7, step8, step9, step10, step11 = (
            (section[index] for index in selected)
            if selected
            else (None, None, None, None, None, None)
        )
        step11_position = selected[-1] if selected else None
        blocks.append(
            {
                "block_index": block_index,
                "source_row_start": section[0].source_row_start,
                "source_row_end": section[-1].source_row_end,
                "step4": section[0],
                "step6": step6,
                "step7": step7,
                "step8": step8,
                "step9": step9,
                "step10": step10,
                "step11": step11,
                "core_sequence_verified": selected is not None,
                "pulse_signature_status": _pulse_status(section, step11_position),
                "section": section,
            }
        )
    return blocks


def map_ordered_blocks(
    blocks: list[dict[str, Any]], candidate_indices: list[int]
) -> list[tuple[int, dict[str, Any]]] | None:
    core_blocks = [block for block in blocks if block["core_sequence_verified"]]
    if (
        len(core_blocks) != len(candidate_indices)
        or len(candidate_indices) != len(set(candidate_indices))
        or any(
            left["source_row_end"] >= right["source_row_start"]
            for left, right in zip(core_blocks, core_blocks[1:])
        )
    ):
        return None
    return list(zip(candidate_indices, core_blocks))


def _stage_claims(stage: str) -> tuple[list[int], list[int], bool]:
    primary_text, _, annotation = (stage or "").partition("(")
    primary, primary_duplicate = parse_rpt_stage(primary_text)
    guide_match = _GUIDE_LIST_RE.search(annotation)
    guide_claims, guide_duplicate = (
        parse_rpt_stage(guide_match.group(1))
        if guide_match
        else (primary.copy(), False)
    )
    return primary, guide_claims, primary_duplicate or guide_duplicate


def _effective_stage_claims(
    cell: str, tpg: str, stage: str
) -> tuple[list[int], list[int], bool, bool]:
    primary, guide, duplicate = _stage_claims(stage)
    confirmed_range = _G5_CELL6_CONFIRMED_RPT_RANGES.get((cell, tpg))
    if confirmed_range is None:
        return primary, guide, duplicate, False
    annotation_differs = bool(guide and guide != list(confirmed_range))
    resolved_range = list(confirmed_range)
    return resolved_range, resolved_range, duplicate, annotation_differs


def _curve_source_status(run: StepRun | None, step: int, source_file: str) -> tuple[str, str]:
    reference, status, _ = _curve_quality(run, source_file)
    if status == "missing_step":
        return reference, status
    if step == 4:
        valid = run is not None and any(segment.get("valid") for segment in run.curve_segments)
        return reference, "verified" if valid else "needs_curve_audit"
    return reference, status


def _row_for_candidate(
    candidate: dict[str, str],
    review_scope: str,
    proposed_status: str,
    block: dict[str, Any] | None,
    evidence: str,
    half_cell_refs_ok: bool,
) -> dict[str, Any]:
    source_file = candidate["source_file"]
    block = block or {}
    step4 = block.get("step4")
    step6 = block.get("step6")
    step7 = block.get("step7")
    step8 = block.get("step8")
    step9 = block.get("step9")
    step10 = block.get("step10")
    step11 = block.get("step11")
    step4_ref, step4_status = _curve_source_status(step4, 4, source_file)
    step8_ref, step8_status = _curve_source_status(step8, 8, source_file)
    step10_ref, step10_status = _curve_source_status(step10, 10, source_file)
    step6_status = _current_direction_status(step6)
    baseline_ok = bool(candidate.get("bol_reference_file") and candidate.get("bol_step4_segment_ref"))
    step4_mapping_candidate = (
        proposed_status == "verified" and step4_status == "verified"
    )
    ready_soh = False
    soh_block_reason = (
        "Part 2 same-cell BoL denominator remains provisional; this audit does not validate or change it."
        if step4_mapping_candidate and baseline_ok
        else "RPT mapping or Step 4 curve is not verified."
    )
    ready_degradation = (
        proposed_status == "verified"
        and step8_status == "verified"
        and step10_status == "verified"
        and half_cell_refs_ok
    )
    return {
        "review_scope": review_scope,
        "part": candidate["part"],
        "group": candidate["group"],
        "physical_cell_id": candidate["physical_cell_id"],
        "rpt_index": candidate["rpt_index"],
        "source_tpg": candidate["source_tpg"],
        "source_file": source_file,
        "source_record_id": candidate["source_record_id"],
        "guide_stage": candidate["guide_stage"],
        "split": candidate["split"],
        "original_mapping_status": candidate["mapping_status"],
        "proposed_mapping_status": proposed_status,
        "block_index": block.get("block_index", ""),
        "block_row_start": block.get("source_row_start", ""),
        "block_row_end": block.get("source_row_end", ""),
        "step4_segment_ref": step4_ref,
        "step4_status": step4_status,
        "step4_rows": step4.row_count if step4 else "",
        "step4_capacity_span_native": _capacity_span(step4),
        "step4_current_min_a": _metric(step4, "current_min"),
        "step4_current_max_a": _metric(step4, "current_max"),
        "step4_voltage_min_v": _metric(step4, "voltage_min"),
        "step4_voltage_max_v": _metric(step4, "voltage_max"),
        "step4_valid_subsegment_count": _valid_curve_count(step4),
        "step6_segment_ref": _run_ref(source_file, step6),
        "step6_status": step6_status,
        "step6_current_min_a": _metric(step6, "current_min"),
        "step6_current_max_a": _metric(step6, "current_max"),
        "step6_voltage_min_v": _metric(step6, "voltage_min"),
        "step6_voltage_max_v": _metric(step6, "voltage_max"),
        "step7_rest_ref": _run_ref(source_file, step7),
        "step8_segment_ref": step8_ref,
        "step8_status": step8_status,
        "step8_rows": step8.row_count if step8 else "",
        "step8_capacity_span_native": _capacity_span(step8),
        "step8_current_min_a": _metric(step8, "current_min"),
        "step8_current_max_a": _metric(step8, "current_max"),
        "step8_voltage_min_v": _metric(step8, "voltage_min"),
        "step8_voltage_max_v": _metric(step8, "voltage_max"),
        "step8_valid_subsegment_count": _valid_curve_count(step8),
        "step8_direction_fraction": _direction_fraction(step8),
        "step9_rest_ref": _run_ref(source_file, step9),
        "step10_segment_ref": step10_ref,
        "step10_status": step10_status,
        "step10_rows": step10.row_count if step10 else "",
        "step10_capacity_span_native": _capacity_span(step10),
        "step10_current_min_a": _metric(step10, "current_min"),
        "step10_current_max_a": _metric(step10, "current_max"),
        "step10_voltage_min_v": _metric(step10, "voltage_min"),
        "step10_voltage_max_v": _metric(step10, "voltage_max"),
        "step10_valid_subsegment_count": _valid_curve_count(step10),
        "step10_direction_fraction": _direction_fraction(step10),
        "step11_rest_ref": _run_ref(source_file, step11),
        "block_test_time_start": _metric(step4, "test_time_start"),
        "block_test_time_end": _metric(step11, "test_time_end"),
        "pulse_signature_status": block.get("pulse_signature_status", "not_mapped"),
        "step4_mapping_candidate": str(step4_mapping_candidate).lower(),
        "ready_soh_candidate": str(ready_soh).lower(),
        "soh_block_reason": soh_block_reason,
        "ready_degradation_candidate": str(ready_degradation).lower(),
        "evidence": evidence,
        "candidate_block_evidence": (
            json.dumps(
                [_block_evidence(block, source_file)],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if block
            else "[]"
        ),
    }


def _group_source_rows(
    observations: list[dict[str, str]],
) -> dict[tuple[str, str, str, str], list[dict[str, str]]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        if row.get("part") != "2":
            continue
        highlighted = (
            row.get("physical_cell_id") == "G5-Cell6"
            and row.get("source_tpg") in {"TPG5.9", "TPG5.10"}
        )
        if row.get("mapping_status") != "ambiguous_rpt" and not highlighted:
            continue
        grouped[
            (
                row["part"],
                row["physical_cell_id"],
                row["source_tpg"],
                row["source_file"],
            )
        ].append(row)
    return grouped


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8-sig", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        return Path(stream.name)


def _report(rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> str:
    status_counts = Counter(row["proposed_mapping_status"] for row in rows)
    original_ambiguous = sum(row["original_mapping_status"] == "ambiguous_rpt" for row in rows)
    extra_reviewed = len(rows) - original_ambiguous
    step4_candidates = sum(row["step4_mapping_candidate"] == "true" for row in rows)
    ready_soh = sum(row["ready_soh_candidate"] == "true" for row in rows)
    ready_degradation = sum(row["ready_degradation_candidate"] == "true" for row in rows)
    lines = [
        "# Ambiguous RPT boundary resolution",
        "",
        "This report reads the existing RPT observation table and the mapped Part 2 raw TPG CSVs. It does not modify canonical manifest v3, labels, splits, checkpoints, scalers, or PyDMA results.",
        "",
        "## Scope and method",
        "",
        f"- Reviewed {original_ambiguous} observations previously marked `ambiguous_rpt` and {extra_reviewed} additional G5-Cell6 TPG5.9/TPG5.10 observations to audit the RPT24 Guide overlap.",
        f"- Read {len(sources)} distinct raw source files; {sum(source['scan_status'] == 'scanned' for source in sources)} scanned successfully.",
        "- Reconstructed candidate blocks from observed Step 4, the immediately preceding Step 6 charge, Step 7 rest, ordered Step 8/9/10/11 sequence, current direction, voltage/capacity ranges, source-row spans, and the observed pulse-step signature where present.",
        "- An ordinal mapping is proposed only when the number of ordered core protocol blocks equals the Guide candidate count. A count of monotonic subcurves alone is not used to assign an RPT.",
        "- Step 8/10 references are individually marked for curve QC; multiple valid monotone subsegments are retained as audit evidence rather than selecting one silently. Step 4 references are not used unless their discharge curve passes the existing basic QC.",
        "- `TestTime` is reported only within each source file. No file clocks are concatenated, and no cycle-age proxy is inferred.",
        "",
        "## Resolution counts",
        "",
        "| Proposed status | Observations |",
        "|---|---:|",
    ]
    for status in ("verified", "partially_verified", "provisional", "unresolved", "conflict"):
        lines.append(f"| {status} | {status_counts.get(status, 0)} |")
    lines.extend(
        [
            "",
            f"- Verified RPT/Step 4 mapping candidates: {step4_candidates}.",
            f"- Ready for SOH labeling: {ready_soh}; Part 2's same-cell BoL denominator remains provisional and unresolved, so this audit does not mark any observation target-ready.",
            f"- Candidate-ready for degradation with individually verified Step 8/10 curves and valid half-cell references: {ready_degradation}.",
            "",
            "The Step 4 count is structural mapping/curve availability only. Degradation counts indicate curve/reference eligibility, not model results, accepted PyDMA fits, or ground-truth labels.",
            "",
            "## Source-level findings",
            "",
            "| Cell | TPG | Guide candidates | Core blocks | Pulse signatures | Proposed outcome |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for source in sources:
        lines.append(
            "| {cell} | {tpg} | {candidates} | {blocks} | {pulses} | {outcome} |".format(
                cell=source["physical_cell_id"],
                tpg=source["source_tpg"],
                candidates=source["candidate_count"],
                blocks=source["core_block_count"],
                pulses=source["complete_pulse_count"],
                outcome=source["source_outcome"],
            )
        )
    lines.extend(
        [
            "",
            "### Part 2 G5-Cell6 RPT24 overlap",
            "",
            "The confirmed split is TPG5.9 = RPT17–RPT23 and TPG5.10 = RPT24–RPT34. Raw scans found seven ordered core blocks in TPG5.9 and eleven in TPG5.10, matching those ranges. The Guide annotation for TPG5.9 also lists RPT24; this is retained as a documentation discrepancy, not treated as an unresolved ownership conflict. The source-local blocks are mapped in order, and the original input manifest is not modified.",
            "",
            "## Unresolved cases and impact",
            "",
            "Rows without a one-to-one ordered core-block mapping remain `unresolved`; their candidate block spans, per-Step source rows, native elapsed-time bounds, current/voltage ranges, and valid subsegment counts are serialized in the detailed CSV. Extra or missing RPT-like blocks, incomplete Step chains, and single-RPT Guide entries with multiple plausible blocks are not resolved by assigning the nearest curve.",
            "",
            "### Manual-review list",
            "",
        ]
    )
    grouped_rows: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_rows[(row["physical_cell_id"], row["source_tpg"])].append(row)
    for (cell, tpg), source_rows in sorted(grouped_rows.items()):
        statuses = sorted({row["proposed_mapping_status"] for row in source_rows})
        if all(status == "verified" for status in statuses):
            continue
        sample = source_rows[0]
        rpts = ",".join(sorted({row["rpt_index"] for row in source_rows}, key=int))
        lines.append(
            f"- `{cell}` `{tpg}` RPT candidates {rpts}: {', '.join(statuses)}; "
            f"source `{sample['source_file']}`."
        )
    lines.extend(
        [
            "",
            "No downstream artifact was changed. The proposed updates are recommendations only; any later application must preserve the locked cell split and keep Part 3 external.",
            "",
            "## Reproduction",
            "",
            "Run `python -m bafuse.data.ambiguous_rpt_resolution` from the repository root. The script rereads the raw source CSVs and writes the detailed resolution CSV, proposed-update CSV, and this report.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_resolution(
    root: Path,
    observations_path: Path,
    detail_path: Path,
    updates_path: Path,
    report_path: Path,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    observations = _read_csv(observations_path)
    grouped = _group_source_rows(observations)
    half_cell_refs = _half_cell_reference_audit(root)
    half_cell_refs_ok = bool(half_cell_refs) and all(
        value.get("status") == "verified_reference_columns_and_finite_pairs"
        for value in half_cell_refs.values()
    )
    result_rows: list[dict[str, Any]] = []
    source_summaries: list[dict[str, Any]] = []
    guide_claims: dict[tuple[str, str], list[tuple[tuple[str, str, str, str], str]]] = defaultdict(list)

    for key, candidates in sorted(grouped.items()):
        part, cell, tpg, relative_path = key
        candidates.sort(key=lambda row: int(row["rpt_index"]))
        for candidate in candidates:
            primary, guide, _, _ = _effective_stage_claims(
                cell, tpg, candidate.get("guide_stage", "")
            )
            for rpt in sorted(set(primary) | set(guide)):
                guide_claims[(cell, str(rpt))].append((key, candidate.get("guide_stage", "")))

        path = _resolve_source(root, part, relative_path)
        scan = (
            scan_raw_tpg(path, chunk_rows)
            if path.is_file()
            else SourceScan("missing_source", "Mapped raw source does not exist.", 0, [])
        )
        blocks = analyze_rpt_blocks(scan.runs) if scan.status == "scanned" else []
        core_blocks = [block for block in blocks if block["core_sequence_verified"]]
        candidate_indices = [int(row["rpt_index"]) for row in candidates]
        ordered_mapping = (
            map_ordered_blocks(blocks, candidate_indices)
            if scan.status == "scanned"
            else None
        )
        exact_ordered_mapping = ordered_mapping is not None
        primary, guide, duplicate_stage, guide_annotation_exception = _effective_stage_claims(
            cell, tpg, candidates[0].get("guide_stage", "")
        )
        guide_range_conflict = bool(guide and primary and guide != primary)

        if exact_ordered_mapping:
            mapped_blocks = list(zip(candidates, core_blocks))
            outcome = "ordered core blocks match Guide count"
            if guide_annotation_exception:
                outcome += (
                    "; user-confirmed RPT17-RPT23 split resolves the Guide annotation "
                    "that also lists RPT24"
                )
        else:
            mapped_blocks = []
            candidate_spans = ";".join(
                f"B{block['block_index']}:rows{block['source_row_start']}-{block['source_row_end']}"
                for block in core_blocks
            ) or "none"
            outcome = (
                scan.error or scan.status
                if scan.status != "scanned"
                else (
                    f"{len(core_blocks)} core blocks for {len(candidates)} Guide candidates; "
                    f"no ordinal assignment; candidate spans={candidate_spans}"
                )
            )

        source_rows: list[dict[str, Any]] = []
        for candidate, block in mapped_blocks:
            proposed_status = "provisional" if guide_range_conflict else "verified"
            source_evidence = (
                "Unique ordered Step 4/6/7/8/9/10/11 protocol block; "
                f"Step 6 current/voltage status={_current_direction_status(block['step6'])}; "
                f"pulse signature={block['pulse_signature_status']}."
            )
            if guide_annotation_exception:
                source_evidence += (
                    " User-confirmed G5-Cell6 split assigns TPG5.9 to RPT17-RPT23 "
                    "and TPG5.10 to RPT24-RPT34; the raw Guide annotation for TPG5.9 "
                    "also lists RPT24."
                )
            if duplicate_stage:
                proposed_status = "conflict"
                source_evidence += " Guide stage repeats an RPT index."
            detail = _row_for_candidate(
                candidate,
                "initial_ambiguous"
                if candidate["mapping_status"] == "ambiguous_rpt"
                else "highlighted_G5_Cell6_overlap",
                proposed_status,
                block,
                source_evidence,
                half_cell_refs_ok,
            )
            source_rows.append(detail)
        if not exact_ordered_mapping:
            for candidate in candidates:
                review_scope = (
                    "initial_ambiguous"
                    if candidate["mapping_status"] == "ambiguous_rpt"
                    else "highlighted_G5_Cell6_overlap"
                )
                stage_claims = _effective_stage_claims(
                    cell, tpg, candidate.get("guide_stage", "")
                )
                stage_conflict = stage_claims[1] != stage_claims[0]
                proposed_status = "provisional" if stage_conflict else "unresolved"
                detail = _row_for_candidate(
                    candidate,
                    review_scope,
                    proposed_status,
                    None,
                    outcome
                    + (
                        "; Guide RPT range conflicts with the provisional candidate range."
                        if stage_conflict
                        else ""
                    ),
                    half_cell_refs_ok,
                )
                detail["candidate_block_evidence"] = json.dumps(
                    [_block_evidence(block, relative_path) for block in core_blocks],
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                source_rows.append(detail)
        source_rows.sort(key=lambda row: int(row["rpt_index"]))
        source_summaries.append(
            {
                "part": part,
                "physical_cell_id": cell,
                "source_tpg": tpg,
                "source_file": relative_path,
                "candidate_count": len(candidates),
                "core_block_count": len(core_blocks),
                "complete_pulse_count": sum(
                    block["pulse_signature_status"] == "complete_ordered_pulse_signature"
                    for block in core_blocks
                ),
                "scan_status": scan.status,
                "scan_error": scan.error,
                "rows_scanned": scan.rows_scanned,
                "source_outcome": outcome
                + ("; Guide RPT range conflict" if guide_range_conflict else ""),
            }
        )
        result_rows.extend(source_rows)

    conflicted_guide_claims = {
        key: entries
        for key, entries in guide_claims.items()
        if len({entry[0] for entry in entries}) > 1
    }
    for row in result_rows:
        claim_key = (row["physical_cell_id"], row["rpt_index"])
        if claim_key in conflicted_guide_claims:
            row["proposed_mapping_status"] = "conflict"
            row["step4_mapping_candidate"] = "false"
            row["ready_soh_candidate"] = "false"
            row["soh_block_reason"] = "Guide claims this cell/RPT in multiple source procedures."
            row["ready_degradation_candidate"] = "false"
            row["evidence"] = (
                row["evidence"]
                + " The Guide claims this same cell/RPT in multiple source procedures."
            ).strip()

    detail_temp = _atomic_write_csv(detail_path, result_rows, _REVIEW_FIELDS)
    update_temp = _atomic_write_csv(updates_path, result_rows, _UPDATE_FIELDS)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", delete=False, dir=report_path.parent, suffix=".tmp"
    ) as stream:
        stream.write(_report(result_rows, source_summaries))
        report_temp = Path(stream.name)
    try:
        detail_temp.replace(detail_path)
        update_temp.replace(updates_path)
        report_temp.replace(report_path)
    except Exception:
        for temporary in (detail_temp, update_temp, report_temp):
            if temporary.exists():
                temporary.unlink()
        raise

    status_counts = Counter(row["proposed_mapping_status"] for row in result_rows)
    return {
        "initial_ambiguous_observations": sum(
            row["review_scope"] == "initial_ambiguous" for row in result_rows
        ),
        "additional_highlighted_observations": sum(
            row["review_scope"] != "initial_ambiguous" for row in result_rows
        ),
        "raw_source_files": len(source_summaries),
        "source_scan_statuses": dict(Counter(source["scan_status"] for source in source_summaries)),
        "proposed_mapping_statuses": dict(status_counts),
        "step4_mapping_candidates": sum(
            row["step4_mapping_candidate"] == "true" for row in result_rows
        ),
        "ready_soh_candidates": sum(row["ready_soh_candidate"] == "true" for row in result_rows),
        "ready_degradation_candidates": sum(
            row["ready_degradation_candidate"] == "true" for row in result_rows
        ),
        "guide_rpt_conflicts": len(conflicted_guide_claims),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resolve ambiguous Part 2 RPT blocks from raw Step 4/6/8/10 evidence."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--observations",
        type=Path,
        default=Path("data/manifests/bafuse_rpt_observations.csv"),
    )
    parser.add_argument(
        "--detail",
        type=Path,
        default=Path("reports/ambiguous_rpt_resolution.csv"),
    )
    parser.add_argument(
        "--updates",
        type=Path,
        default=Path("reports/rpt_mapping_proposed_updates.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("reports/ambiguous_rpt_resolution.md"),
    )
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    root = args.root.resolve()
    paths = [
        path if path.is_absolute() else root / path
        for path in (args.observations, args.detail, args.updates, args.report)
    ]
    summary = run_resolution(root, *paths, chunk_rows=args.chunk_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
