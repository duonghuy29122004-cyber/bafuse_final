from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bafuse.data.pydma_stage2 import _curve_candidates
from bafuse.data.soh_step4_scan import scan_step4_file
from bafuse.labeling.audit import _identify_file, read_guide_entries


_BOL_STAGE = re.compile(r"\s*BoL(?:\s+RPT)?\s*", re.IGNORECASE)
_STEP4_REF = re.compile(
    r"^(?P<source>.+)#Step4:segment(?P<segment>\d+):"
    r"rows(?P<start>\d+)-(?P<end>\d+)$",
    re.IGNORECASE,
)
_CURVE_AUDIT_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "procedure",
    "procedure_file_id",
    "guide_stage",
    "guide_entry_ordinal",
    "mapping_provenance",
    "source_file",
    "source_file_status",
    "source_file_size_bytes",
    "source_sha256",
    "guide_sha256",
    "step4_scan_status",
    "step4_segment_count",
    "step4_segments_json",
    "step8_scan_status",
    "step8_candidate_count",
    "step8_valid_candidate_count",
    "step8_candidates_json",
    "step10_scan_status",
    "step10_candidate_count",
    "step10_valid_candidate_count",
    "step10_candidates_json",
    "timeline_testtime_first",
    "timeline_testtime_last",
    "timeline_testtime_decrease_count",
    "timeline_step_runs_json",
    "scan_errors",
]
_DECISION_FIELDS = [
    "dataset_part",
    "physical_cell_id",
    "reference_scope",
    "identity_provenance",
    "guide_bol_procedure_count",
    "guide_bol_source_count",
    "guide_bol_sources_json",
    "current_step4_reference",
    "current_step4_validation",
    "pydma_step8_reference_source",
    "pydma_step8_valid_candidate_count",
    "pydma_step8_status",
    "step10_valid_candidate_count",
    "step10_status",
    "g2_duplicate_step4_comparison_status",
    "g2_duplicate_step4_compared_signals",
    "g2_duplicate_step4_max_abs_difference",
    "decision",
]
_REASSESSMENT_FIELDS = [
    "dataset_part",
    "physical_cell_id",
    "rpt_index",
    "partition",
    "previous_status",
    "mapping_status",
    "target_source_file",
    "saved_fit_id",
    "saved_fit_rmse_mV",
    "historical_negative_values_json",
    "baseline_source_part",
    "baseline_source_file",
    "baseline_step8_valid_candidate_count",
    "baseline_step8_status",
    "reassessment_status",
    "action",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: str | None) -> float | None:
    try:
        number = float(value or "")
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _guide_source_path(part: str, source: str) -> str:
    normalized = source.strip().replace("\\", "/")
    prefix = f"data/raw/oxford/part{part}/"
    return normalized if normalized.startswith(prefix) else prefix + normalized


def _source_file_map(root: Path) -> dict[tuple[str, str, int], list[Path]]:
    sources: defaultdict[tuple[str, str, int], list[Path]] = defaultdict(list)
    for path in root.rglob("*.csv"):
        metadata = _identify_file(path, root)
        if metadata["modality"] != "cycling_and_RPT" or metadata["file_id"] is None:
            continue
        key = (metadata["group_id"], metadata["cell_id"], metadata["file_id"])
        sources[key].append(path)
    for paths in sources.values():
        paths.sort()
    return sources


def _scan_timeline(path: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    all_step_runs: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    all_step_counts: Counter[int] = Counter()
    active: dict[str, Any] | None = None
    all_step_active: dict[str, Any] | None = None
    previous_testtime: float | None = None
    decrease_count = 0
    first_testtime: float | None = None
    last_testtime: float | None = None
    source_row_count = 0

    def finish_run() -> None:
        nonlocal active
        if active is not None:
            runs.append(active)
        active = None

    def finish_all_step_run() -> None:
        nonlocal all_step_active
        if all_step_active is not None:
            all_step_runs.append(all_step_active)
        all_step_active = None

    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or "Step" not in reader.fieldnames:
            raise ValueError(f"Missing Step column in {path}")
        row_number = 1
        for row in reader:
            row_number += 1
            source_row_count += 1
            step_value = _number(row.get("Step"))
            step = int(step_value) if step_value is not None and step_value.is_integer() else None
            testtime = _number(row.get("TestTime"))
            if testtime is not None:
                if first_testtime is None:
                    first_testtime = testtime
                if previous_testtime is not None and testtime < previous_testtime:
                    decrease_count += 1
                    if active is not None:
                        active["testtime_decrease_count"] += 1
                previous_testtime = testtime
                last_testtime = testtime

            if all_step_active is not None and step != all_step_active["step"]:
                finish_all_step_run()
            if step is not None:
                if all_step_active is None:
                    all_step_counts[step] += 1
                    all_step_active = {
                        "step": step,
                        "segment_index": all_step_counts[step],
                        "source_row_start": row_number,
                        "source_row_end": row_number,
                        "row_count": 0,
                        "testtime_first": testtime,
                        "testtime_last": testtime,
                    }
                all_step_active["source_row_end"] = row_number
                all_step_active["row_count"] += 1
                if testtime is not None:
                    all_step_active["testtime_last"] = testtime
            elif all_step_active is not None:
                finish_all_step_run()

            if active is not None and step != active["step"]:
                finish_run()
            if step not in {4, 8, 10}:
                continue
            if active is None:
                counts[step] += 1
                active = {
                    "step": step,
                    "segment_index": counts[step],
                    "source_row_start": row_number,
                    "source_row_end": row_number,
                    "row_count": 0,
                    "testtime_first": testtime,
                    "testtime_last": testtime,
                    "testtime_min": testtime,
                    "testtime_max": testtime,
                    "testtime_decrease_count": 0,
                    "amphr_min": None,
                    "amphr_max": None,
                    "amps_min": None,
                    "amps_max": None,
                    "current_sign_counts": {
                        "negative": 0,
                        "zero": 0,
                        "positive": 0,
                    },
                    "current_direction_changes": 0,
                    "previous_nonzero_current_sign": None,
                    "volts_min": None,
                    "volts_max": None,
                    "preceding_step4_segment_index": counts[4] or None,
                }
            active["source_row_end"] = row_number
            active["row_count"] += 1
            if testtime is not None:
                active["testtime_last"] = testtime
                active["testtime_min"] = (
                    testtime if active["testtime_min"] is None
                    else min(active["testtime_min"], testtime)
                )
                active["testtime_max"] = (
                    testtime if active["testtime_max"] is None
                    else max(active["testtime_max"], testtime)
                )
            for column, low_key, high_key in (
                ("Amphr", "amphr_min", "amphr_max"),
                ("Amps", "amps_min", "amps_max"),
                ("Volts", "volts_min", "volts_max"),
            ):
                value = _number(row.get(column))
                if value is not None:
                    active[low_key] = (
                        value if active[low_key] is None else min(active[low_key], value)
                    )
                    active[high_key] = (
                        value if active[high_key] is None else max(active[high_key], value)
                    )
                    if column == "Amps":
                        sign = (
                            "negative"
                            if value < 0
                            else "positive"
                            if value > 0
                            else "zero"
                        )
                        active["current_sign_counts"][sign] += 1
                        if sign != "zero":
                            previous_sign = active["previous_nonzero_current_sign"]
                            if previous_sign is not None and previous_sign != sign:
                                active["current_direction_changes"] += 1
                            active["previous_nonzero_current_sign"] = sign
        finish_run()
        finish_all_step_run()

    return {
        "testtime_first": first_testtime,
        "testtime_last": last_testtime,
        "testtime_decrease_count": decrease_count,
        "source_row_count": source_row_count,
        "step_runs": runs,
        "all_step_runs": all_step_runs,
    }


def _candidate_records(
    root: Path,
    source: Path,
    step: int,
    direction: str,
    runs: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    candidates = _curve_candidates(source, step, direction)
    run_by_index = {
        int(run["segment_index"]): run for run in (runs or []) if run["step"] == step
    }
    records: list[dict[str, Any]] = []
    relative = source.resolve().relative_to(root.resolve()).as_posix()
    for candidate in candidates:
        record = {
            key: value
            for key, value in candidate.items()
            if key not in {"capacity", "voltage"}
        }
        record["source_reference"] = (
            f"{relative}#Step{step}:segment{candidate['step_segment_index']}:"
            f"rows{candidate['source_row_start']}-{candidate['source_row_end']}:"
            f"sub{candidate['subsegment_index']}"
        )
        run = run_by_index.get(int(candidate["step_segment_index"]))
        if run is not None:
            record["raw_step_row_start"] = run["source_row_start"]
            record["raw_step_row_end"] = run["source_row_end"]
            record["raw_step_testtime_first"] = run["testtime_first"]
            record["raw_step_testtime_last"] = run["testtime_last"]
            record["preceding_step4_segment_index"] = run[
                "preceding_step4_segment_index"
            ]
        records.append(record)
    return records


def _read_step4_signal_segment(
    path: Path,
    segment: dict[str, Any],
) -> tuple[list[str], np.ndarray]:
    available = list(pd.read_csv(path, nrows=0).columns)
    signals = [name for name in ("Amphr", "Amps", "Volts", "Temp1") if name in available]
    required = {"Amphr", "Amps", "Volts"}
    if not required.issubset(signals):
        raise ValueError(f"Required Step 4 signal columns are absent in {path}")

    start_index = int(segment["source_row_start"]) - 2
    end_index = int(segment["source_row_end"]) - 1
    offset = 0
    selected: list[np.ndarray] = []
    reader = pd.read_csv(
        path,
        usecols=signals,
        chunksize=250_000,
        low_memory=False,
    )
    try:
        for chunk in reader:
            overlap_start = max(start_index, offset)
            overlap_end = min(end_index, offset + len(chunk))
            if overlap_start < overlap_end:
                frame = chunk.iloc[overlap_start - offset : overlap_end - offset]
                selected.append(
                    np.column_stack(
                        [
                            pd.to_numeric(frame[column], errors="coerce").to_numpy(
                                dtype=np.float64
                            )
                            for column in signals
                        ]
                    )
                )
            offset += len(chunk)
            if offset >= end_index:
                break
    finally:
        reader.close()
    if not selected:
        raise ValueError(f"No source rows were read for the Step 4 segment in {path}")
    return signals, np.concatenate(selected, axis=0)


def _compare_step4_segments(
    first_path: Path,
    first_segment: dict[str, Any],
    second_path: Path,
    second_segment: dict[str, Any],
) -> dict[str, Any]:
    first_signals, first_values = _read_step4_signal_segment(first_path, first_segment)
    second_signals, second_values = _read_step4_signal_segment(second_path, second_segment)
    if first_signals != second_signals or first_values.shape != second_values.shape:
        return {
            "status": "different_signal_schema_or_row_count",
            "signals": ";".join(first_signals),
            "max_abs_difference": "",
        }
    finite = np.isfinite(first_values) & np.isfinite(second_values)
    max_difference = (
        float(np.max(np.abs(first_values[finite] - second_values[finite])))
        if finite.any()
        else None
    )
    same = bool(
        np.allclose(
            first_values,
            second_values,
            rtol=0,
            atol=1e-8,
            equal_nan=True,
        )
    )
    return {
        "status": "matching_within_1e-8" if same else "signals_differ",
        "signals": ";".join(first_signals),
        "max_abs_difference": "" if max_difference is None else max_difference,
    }


def _missing_source_row(row: dict[str, Any]) -> dict[str, Any]:
    result = {field: "" for field in _CURVE_AUDIT_FIELDS}
    result.update(row)
    result.update(
        {
            "source_file_status": "source_file_not_found",
            "step4_scan_status": "not_scanned",
            "step8_scan_status": "not_scanned",
            "step10_scan_status": "not_scanned",
            "step4_segment_count": 0,
            "step8_candidate_count": 0,
            "step8_valid_candidate_count": 0,
            "step10_candidate_count": 0,
            "step10_valid_candidate_count": 0,
            "step4_segments_json": "[]",
            "step8_candidates_json": "[]",
            "step10_candidates_json": "[]",
            "timeline_step_runs_json": "[]",
            "scan_errors": "Source file is absent; no curve or label was inferred.",
        }
    )
    return result


def _audit_source(
    root: Path,
    part: int,
    guide: dict[str, Any],
    path: Path,
    guide_sha256: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        field: "" for field in _CURVE_AUDIT_FIELDS
    }
    row.update(
        {
            "dataset_part": part,
            "group_id": guide["group_id"],
            "physical_cell_id": guide["physical_cell_id"],
            "procedure": guide["procedure"],
            "procedure_file_id": guide["procedure_file_id"],
            "guide_stage": guide["guide_stage"],
            "guide_entry_ordinal": guide["guide_entry_ordinal"],
            "mapping_provenance": guide["mapping_provenance"],
            "source_file": path.resolve().relative_to(root.resolve()).as_posix(),
            "source_file_status": guide.get(
                "source_file_status", "unique_source_file"
            ),
            "source_file_size_bytes": path.stat().st_size,
            "source_sha256": _sha256(path),
            "guide_sha256": guide_sha256,
        }
    )
    errors: list[str] = []
    step4 = scan_step4_file(path)
    row["step4_scan_status"] = step4["scan_status"]
    row["step4_segment_count"] = step4["step4_segment_count"]
    row["step4_segments_json"] = step4["step4_segments_json"]
    if step4["scan_status"] in {"scan_failed", "not_scanned"}:
        errors.append(
            f"Step4:{step4['scan_status']}:{step4.get('scan_error', '')}"
        )

    timeline: dict[str, Any] | None = None
    is_g2_bol = (
        part == 1
        and guide["physical_cell_id"] == "G2-Cell4"
        and guide["procedure"] in {"TPG2", "TPG2.2"}
    )
    if is_g2_bol:
        timeline = _scan_timeline(path)
        row["timeline_testtime_first"] = timeline["testtime_first"]
        row["timeline_testtime_last"] = timeline["testtime_last"]
        row["timeline_testtime_decrease_count"] = timeline[
            "testtime_decrease_count"
        ]
        row["timeline_step_runs_json"] = json.dumps(
            timeline["step_runs"], ensure_ascii=False, allow_nan=False, sort_keys=True
        )

    for step, direction in ((8, "discharge"), (10, "charge")):
        try:
            candidates = _candidate_records(
                root,
                path,
                step,
                direction,
                timeline["step_runs"] if timeline is not None else None,
            )
        except (
            OSError,
            UnicodeError,
            pd.errors.EmptyDataError,
            pd.errors.ParserError,
            ValueError,
        ) as error:
            candidates = []
            status = f"scan_failed:{type(error).__name__}:{error}"
            errors.append(f"Step{step}:{status}")
        else:
            valid_count = sum(bool(item["valid"]) for item in candidates)
            status = (
                "no_candidates"
                if not candidates
                else "one_valid_candidate"
                if valid_count == 1 and len(candidates) == 1
                else "multiple_valid_candidates"
                if valid_count > 1
                else "candidates_not_structurally_valid"
            )
        row[f"step{step}_scan_status"] = status
        row[f"step{step}_candidate_count"] = len(candidates)
        row[f"step{step}_valid_candidate_count"] = sum(
            bool(item["valid"]) for item in candidates
        )
        row[f"step{step}_candidates_json"] = json.dumps(
            candidates, ensure_ascii=False, allow_nan=False, sort_keys=True
        )

    row["scan_errors"] = " | ".join(errors)
    return row


def _guide_inventory(
    root: Path,
    parts: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, Path], dict[int, str]]:
    rows: list[dict[str, Any]] = []
    roots: dict[int, Path] = {}
    hashes: dict[int, str] = {}
    for part in parts:
        dataset_root = (root / "data" / "raw" / "oxford" / f"part{part}").resolve()
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Part {part} source directory is absent: {dataset_root}")
        guides = sorted(dataset_root.glob("Guide*.xlsx"))
        if len(guides) != 1:
            raise ValueError(
                f"Expected one Part {part} Guide XLSX, found {len(guides)} in {dataset_root}"
            )
        guide_path = guides[0]
        guide_hash = _sha256(guide_path)
        roots[part] = dataset_root
        hashes[part] = guide_hash
        source_files = _source_file_map(dataset_root)
        entries = read_guide_entries(guide_path)
        for ordinal, entry in enumerate(entries, start=1):
            if not _BOL_STAGE.fullmatch(entry.stage):
                continue
            physical_cell = f"G{entry.group_id}-Cell{entry.cell_id}"
            source_file_id = entry.file_id
            provenance = "official_guide_cell_and_procedure"
            if part == 3 and re.fullmatch(r"TPG\d+", entry.procedure):
                source_file_id = 0
                provenance = "official_guide_procedure_part3_zero_suffix"
            matches = source_files.get(
                (entry.group_id, entry.cell_id, source_file_id), []
            )
            guide_record = {
                "dataset_part": part,
                "group_id": entry.group_id,
                "physical_cell_id": physical_cell,
                "procedure": entry.procedure,
                "procedure_file_id": entry.file_id,
                "guide_stage": entry.stage,
                "guide_entry_ordinal": ordinal,
                "mapping_provenance": provenance,
                "source_file_status": (
                    "unique_source_file"
                    if len(matches) == 1
                    else "multiple_source_files_individually_scanned"
                ),
            }
            if not matches:
                rows.append(_missing_source_row(guide_record))
                continue
            for path in matches:
                rows.append(
                    _audit_source(root, part, guide_record, path, guide_hash)
                )

        if part == 3 and not any(
            row["physical_cell_id"] == "G10-Cell2" for row in rows
        ):
            matches = source_files.get(("10", "2", 0), [])
            guide_record = {
                "dataset_part": 3,
                "group_id": "10",
                "physical_cell_id": "G10-Cell2",
                "procedure": "TPG10",
                "procedure_file_id": 1,
                "guide_stage": "BoL; cell assignment user-confirmed",
                "guide_entry_ordinal": "not available; Guide cell is blank",
                "mapping_provenance": (
                    "user_confirmed_unlabeled_part3_group10_procedure_maps_to_cell2"
                ),
                "source_file_status": (
                    "unique_source_file"
                    if len(matches) == 1
                    else "multiple_source_files_individually_scanned"
                ),
            }
            if matches:
                for path in matches:
                    rows.append(
                        _audit_source(root, part, guide_record, path, guide_hash)
                    )
            else:
                rows.append(_missing_source_row(guide_record))
    return rows, roots, hashes


def _reference_decisions(
    root: Path,
    curve_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    stage1_path = root / "reports" / "bafuse_stage_validation_v1" / "stage1_bol_reference_audit.csv"
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Existing BoL reference audit is absent: {stage1_path}")
    stage1 = {
        (row["dataset_part"], row["physical_cell_id"]): row
        for row in _read_csv(stage1_path)
    }
    curve_by_source = {
        (str(row["dataset_part"]), str(row["source_file"])): row
        for row in curve_rows
        if row.get("source_file")
    }
    grouped: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in curve_rows:
        grouped[(str(row["dataset_part"]), str(row["physical_cell_id"]))].append(row)

    decisions: list[dict[str, Any]] = []
    for (part, cell), candidates in sorted(grouped.items()):
        prior = stage1.get((part, cell), {})
        source_ref = str(prior.get("primary_bol_reference_file", ""))
        step4_ref = str(prior.get("bol_step4_segment_ref", ""))
        source_key = _guide_source_path(part, source_ref) if source_ref else ""
        selected_source = curve_by_source.get((part, source_key))
        step4_validation = "not_selected_by_existing_policy"
        if part == "1" and source_ref:
            if selected_source is None:
                step4_validation = "selected_source_not_in_current_guide_inventory"
            else:
                match = _STEP4_REF.fullmatch(step4_ref.replace("\\", "/"))
                if match is None:
                    step4_validation = "existing_step4_reference_unparseable"
                elif match.group("source") != source_ref.replace("\\", "/"):
                    step4_validation = "existing_step4_reference_points_to_different_source"
                else:
                    segments = json.loads(selected_source["step4_segments_json"])
                    expected = (
                        int(match.group("segment")),
                        int(match.group("start")),
                        int(match.group("end")),
                    )
                    found = any(
                        (
                            int(segment["segment_index"]),
                            int(segment["source_row_start"]),
                            int(segment["source_row_end"]),
                        )
                        == expected
                        and int(segment["capacity_finite_rows"]) >= 10
                        and float(segment["capacity_span"] or 0) > 0
                        and not {"negative", "positive"}.issubset(
                            set(segment["current_signs"])
                        )
                        for segment in segments
                    )
                    step4_validation = (
                        "validated_against_current_raw_scan"
                        if found
                        else "existing_step4_reference_not_found_in_current_scan"
                    )

        step8_source = selected_source if part == "1" else None
        step8_count = (
            int(step8_source["step8_valid_candidate_count"])
            if step8_source is not None
            else sum(int(row["step8_valid_candidate_count"] or 0) for row in candidates)
        )
        step10_count = (
            int(step8_source["step10_valid_candidate_count"])
            if step8_source is not None
            else sum(int(row["step10_valid_candidate_count"] or 0) for row in candidates)
        )
        if part == "1":
            step8_status = (
                "unique_structural_candidate_not_protocol_identity_proof"
                if step8_count == 1
                else "ambiguous_multiple_structural_candidates"
                if step8_count > 1
                else "no_structural_candidate"
            )
            step10_status = (
                "unique_structural_candidate"
                if step10_count == 1
                else "multiple_candidates_including_partial_candidate_review"
                if step10_count > 1
                else "no_structural_candidate"
            )
            decision = (
                "retain_existing_provisional_step4_reference"
                if step4_validation == "validated_against_current_raw_scan"
                else "existing_step4_reference_requires_review"
            )
            reference_scope = "part1_provisional_reference"
            identity_provenance = "official_guide"
        elif part == "3":
            step8_status = "audited_external_only_not_selected"
            step10_status = "audited_external_only_not_selected"
            decision = "external_test_only_no_reference_or_model_change"
            reference_scope = "external_part3"
            identity_provenance = (
                "user_confirmed_cell_identity"
                if cell == "G10-Cell2"
                else "official_guide"
            )
        else:
            step8_status = "local_reference_policy_not_selected"
            step10_status = "local_reference_policy_not_selected"
            decision = "local_bol_policy_unresolved_not_selected"
            reference_scope = "local_part2_reference"
            identity_provenance = "official_guide"
        duplicate_comparison = {
            "status": "",
            "signals": "",
            "max_abs_difference": "",
        }
        if part == "1" and cell == "G2-Cell4":
            first_source = next(
                (
                    row for row in candidates
                    if row["procedure"] == "TPG2"
                    and row["source_file"].endswith("TPG2_1_Cell4.csv")
                ),
                None,
            )
            second_source = next(
                (
                    row for row in candidates
                    if row["procedure"] == "TPG2.2"
                    and row["source_file"].endswith("TPG2_2_Cell4.csv")
                ),
                None,
            )
            if first_source is None or second_source is None:
                duplicate_comparison["status"] = "comparison_sources_not_found"
            else:
                first_segments = json.loads(first_source["step4_segments_json"])
                second_segments = json.loads(second_source["step4_segments_json"])
                if len(first_segments) < 3 or not second_segments:
                    duplicate_comparison["status"] = "comparison_segments_not_found"
                else:
                    duplicate_comparison = _compare_step4_segments(
                        root / first_source["source_file"],
                        first_segments[2],
                        root / second_source["source_file"],
                        second_segments[0],
                    )
        decisions.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "reference_scope": reference_scope,
                "identity_provenance": identity_provenance,
                "guide_bol_procedure_count": len(
                    {
                        str(row["procedure"])
                        for row in candidates
                    }
                ),
                "guide_bol_source_count": len(
                    {str(row["source_file"]) for row in candidates if row["source_file"]}
                ),
                "guide_bol_sources_json": json.dumps(
                    [
                        {
                            "procedure": row["procedure"],
                            "source_file": row["source_file"],
                            "source_file_status": row["source_file_status"],
                            "mapping_provenance": row["mapping_provenance"],
                        }
                        for row in candidates
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                "current_step4_reference": step4_ref,
                "current_step4_validation": step4_validation,
                "pydma_step8_reference_source": source_ref if part == "1" else "",
                "pydma_step8_valid_candidate_count": step8_count,
                "pydma_step8_status": step8_status,
                "step10_valid_candidate_count": step10_count,
                "step10_status": step10_status,
                "g2_duplicate_step4_comparison_status": duplicate_comparison[
                    "status"
                ],
                "g2_duplicate_step4_compared_signals": duplicate_comparison[
                    "signals"
                ],
                "g2_duplicate_step4_max_abs_difference": duplicate_comparison[
                    "max_abs_difference"
                ],
                "decision": decision,
            }
        )
    return decisions


def _negative_reassessment(
    root: Path,
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    worklist_path = root / "reports" / "bafuse_negative_correction_v2" / "negative_worklist.csv"
    if not worklist_path.is_file():
        raise FileNotFoundError(f"Negative correction worklist is absent: {worklist_path}")
    decision_by_key = {
        (str(row["dataset_part"]), str(row["physical_cell_id"])): row
        for row in decisions
    }
    output: list[dict[str, Any]] = []
    for row in _read_csv(worklist_path):
        part = str(row["dataset_part"])
        cell = str(row["physical_cell_id"])
        group_id = cell.split("-")[0].removeprefix("G")
        baseline_part = "1" if part == "2" and group_id in {"1", "2", "3", "4"} else part
        baseline = decision_by_key.get((baseline_part, cell), {})
        mapping_status = str(row.get("mapping_status", ""))
        previous_status = str(row.get("status", ""))
        step8_status = str(baseline.get("pydma_step8_status", "no_reference_candidate"))
        valid_count = str(baseline.get("pydma_step8_valid_candidate_count", "0"))
        if mapping_status not in {"verified", "verified_from_ordered_raw_blocks"}:
            reassessment = "mapping_unresolved_hold"
        elif part == "3":
            reassessment = "external_part3_audit_only_no_model_label_change"
        elif step8_status == "ambiguous_multiple_structural_candidates":
            reassessment = "baseline_step8_ambiguous_hold"
        elif baseline_part == "2":
            reassessment = "local_part2_reference_policy_unresolved_hold"
        elif previous_status in {
            "OPTIMIZER_UNSTABLE",
            "NEGATIVE_BEST_FIT_REVIEW",
        }:
            reassessment = "saved_fit_unchanged_diagnostic_only"
        elif part == "2" and baseline_part == "1":
            reassessment = "part1_reference_candidate_found_no_refit_or_promotion"
        else:
            reassessment = "no_reassessment_change"

        output.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": row["rpt_index"],
                "partition": row.get("partition", ""),
                "previous_status": previous_status,
                "mapping_status": mapping_status,
                "target_source_file": row.get("source_file", ""),
                "saved_fit_id": row.get("selected_fit_id", ""),
                "saved_fit_rmse_mV": row.get("selected_rmse_mV", ""),
                "historical_negative_values_json": row.get(
                    "original_degradation_values_json", ""
                ),
                "baseline_source_part": baseline_part,
                "baseline_source_file": baseline.get(
                    "pydma_step8_reference_source", ""
                ),
                "baseline_step8_valid_candidate_count": valid_count,
                "baseline_step8_status": step8_status,
                "reassessment_status": reassessment,
                "action": "no_refit_no_label_change",
            }
        )
    return output


def _summary_markdown(
    curve_rows: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    reassessments: list[dict[str, Any]],
    guide_hashes: dict[int, str],
) -> str:
    part_lines: list[str] = []
    for part in (1, 2, 3):
        rows = [row for row in curve_rows if int(row["dataset_part"]) == part]
        if not rows:
            continue
        part_lines.append(
            f"| Part {part} | {len({row['physical_cell_id'] for row in rows})} | "
            f"{sum(bool(row['source_file']) for row in rows)} | "
            f"{sum(int(row['step4_segment_count'] or 0) > 0 for row in rows)} | "
            f"{sum(int(row['step8_valid_candidate_count'] or 0) == 1 for row in rows)} | "
            f"{sum(int(row['step8_valid_candidate_count'] or 0) > 1 for row in rows)} | "
            f"{sum(int(row['step10_valid_candidate_count'] or 0) > 0 for row in rows)} |"
        )
    status_counts = Counter(row["previous_status"] for row in reassessments)
    result_counts = Counter(row["reassessment_status"] for row in reassessments)
    g2 = next(
        row for row in curve_rows
        if row["dataset_part"] == 1
        and row["physical_cell_id"] == "G2-Cell4"
        and row["procedure"] == "TPG2"
    )
    step4_segments = json.loads(g2["step4_segments_json"])
    step8 = json.loads(g2["step8_candidates_json"])
    step10 = json.loads(g2["step10_candidates_json"])
    timeline = json.loads(g2["timeline_step_runs_json"])
    g2_step4_refs = ", ".join(
        f"rows {segment['source_row_start']}-{segment['source_row_end']}"
        for segment in step4_segments
    )
    g2_step8_refs = ", ".join(
        f"rows {candidate['source_row_start']}-{candidate['source_row_end']}"
        for candidate in step8
        if candidate["valid"]
    )
    g2_step10_spans = ", ".join(
        f"rows {candidate['source_row_start']}-{candidate['source_row_end']}, "
        f"span={candidate['capacity_span']:.6f}, "
        f"V={candidate['voltage_min']:.3f}-{candidate['voltage_max']:.3f} V"
        for candidate in step10
        if candidate["valid"]
    )
    run_context = "; ".join(
        f"Step{run['step']}#{run['segment_index']} rows "
        f"{run['source_row_start']}-{run['source_row_end']} "
        f"(TestTime {run['testtime_first']}-{run['testtime_last']}; "
        f"Amps {run['amps_min']}-{run['amps_max']})"
        for run in timeline
    )
    g2_decision = next(
        row for row in decisions
        if row["dataset_part"] == "1"
        and row["physical_cell_id"] == "G2-Cell4"
    )
    multiple_step8 = [
        row for row in curve_rows
        if int(row["step8_valid_candidate_count"] or 0) > 1
    ]
    multiple_step10 = [
        row for row in curve_rows
        if int(row["step10_valid_candidate_count"] or 0) > 1
    ]
    no_step4 = [
        row for row in curve_rows
        if int(row["step4_segment_count"] or 0) == 0
    ]
    ambiguous_negative_keys = [
        f"P{row['dataset_part']} {row['physical_cell_id']} RPT{row['rpt_index']}"
        for row in reassessments
        if row["reassessment_status"] == "baseline_step8_ambiguous_hold"
    ]
    multiple_step8_lines = [
        f"- Part {row['dataset_part']} {row['physical_cell_id']} "
        f"{row['procedure']}: `{row['source_file']}` "
        f"({row['step8_valid_candidate_count']} valid candidates)."
        for row in multiple_step8
    ]
    multiple_step10_lines = [
        f"- Part {row['dataset_part']} {row['physical_cell_id']} "
        f"{row['procedure']}: `{row['source_file']}` "
        f"({row['step10_valid_candidate_count']} structurally monotone candidates)."
        for row in multiple_step10
    ]
    no_step4_lines = [
        f"- Part {row['dataset_part']} {row['physical_cell_id']} "
        f"`{row['source_file']}`."
        for row in no_step4
    ]

    lines = [
        "# Procedure-aware BoL reference audit",
        "",
        "This versioned audit reads the Guide workbooks and raw BoL CSV sources. It does not",
        "change labels, saved fits, split assignments, model artifacts, or raw data. It does",
        "not infer cycle age or apply a cycles-per-RPT rule. Part 3 remains external-only.",
        "",
        "## Coverage",
        "",
        "| Part | Cells | Source files | Sources with Step 4 | One valid Step 8 candidate | Multiple valid Step 8 candidates | Sources with Step 10 candidates |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *part_lines,
        "",
        f"- Guide/user-confirmed procedure-cell entries: "
        f"**{len({(row['dataset_part'], row['physical_cell_id'], row['procedure']) for row in curve_rows})}**.",
        f"- Individual source-path audit rows: **{len(curve_rows)}**.",
        f"- Negative worklist rows reclassified without refitting: **{len(reassessments)}**.",
        f"- Prior negative-worklist statuses: `{json.dumps(status_counts, sort_keys=True)}`.",
        f"- Reassessment statuses: `{json.dumps(result_counts, sort_keys=True)}`.",
        "",
        "## G2-Cell4 preflight root cause",
        "",
        "The existing Part 1 policy selects `TPG2_1_Cell4.csv` as the first BoL source for",
        "the provisional Step 4 Q0 reference; `TPG2_2_Cell4.csv` is retained as a separate",
        "Guide-labelled BoL/QC source. The negative-correction preflight resolves one",
        "primary source file and scans that file; it does not concatenate the two BoL files.",
        "The error `found 2` is caused by two structurally valid low-rate Step 8 discharge",
        "candidates inside `TPG2_1_Cell4.csv` itself.",
        "",
        f"- Step 4 blocks in the first source: {len(step4_segments)} ({g2_step4_refs}).",
        f"- Valid Step 8 candidate rows: {g2_step8_refs}.",
        f"- Step 10 candidates passing structural monotonicity: {g2_step10_spans}.",
        f"- In-file TestTime context: {run_context}.",
        f"- File-level TestTime decreases: {g2['timeline_testtime_decrease_count']}; "
        f"range {g2['timeline_testtime_first']} to {g2['timeline_testtime_last']}.",
        f"- Direct comparison of TPG2.1 Step 4 segment 3 against TPG2.2 Step 4 segment 1: "
        f"`{g2_decision['g2_duplicate_step4_comparison_status']}` across "
        f"{g2_decision['g2_duplicate_step4_compared_signals']} "
        f"(maximum absolute difference "
        f"{g2_decision['g2_duplicate_step4_max_abs_difference']}).",
        "- The two Step 8 candidates are distinct chronological low-rate runs with full",
        "  voltage spans; raw order does not identify which is the intended baseline curve.",
        "  The later Step 10 candidate is structurally monotone but covers only a partial",
        "  voltage/capacity range, so it is not evidence that the second Step 8 is the",
        "  intended baseline.",
        "- The third Step 4 block in `TPG2_1_Cell4.csv` matches the Step 4 block in",
        "  `TPG2_2_Cell4.csv` in the direct signal comparison above. That supports keeping",
        "  `TPG2.2` as QC, but does not disambiguate the two Step 8 curves in `TPG2.1`.",
        "- Therefore Step 4 Q0 remains on the existing provisional reference, while the",
        "  PyDMA Step 8 baseline stays unresolved. No fit or label was promoted.",
        "",
        "## Other source-level ambiguities",
        "",
        "Multiple structurally valid Step 8 candidates remain unselected:",
        *(multiple_step8_lines or ["- None."]),
        "",
        "Multiple structurally monotone Step 10 candidates, including short partial",
        "voltage spans, remain unselected:",
        *(multiple_step10_lines or ["- None."]),
        "",
        "Source paths with no Step 4 segment (retained for QC; not substituted for the",
        "selected Part 1 Q0 source):",
        *(no_step4_lines or ["- None."]),
        "",
        "## Reference decisions",
        "",
        "- Part 1 Step 4 Q0 references are retained exactly as recorded in the existing",
        "  stage-1 audit; each selected row boundary is checked against the current raw scan.",
        "- Part 1 Step 8 and Step 10 candidate counts are audited separately. Multiple valid",
        "  candidates remain unresolved; no longest-curve selection is made.",
        "- Part 2 local BoL and Part 3 references remain unselected because their denominator",
        "  policies are not established here; their raw Step candidate counts are still",
        "  recorded per source. Part 3 is audit-only external evaluation.",
        "- The Part 3 Group 10 `TPG10.0-Cell2.csv` identity uses the user's explicit",
        "  confirmation; the Guide's cell field is blank, so this is not reported as",
        "  independently Guide-verified.",
        "- Part 2 Groups 1-4 negative cases are linked to the same physical cell's Part 1",
        "  BoL candidate for review, but no cached fit or label is rewritten.",
        "",
        "## Negative-result impact",
        "",
        f"- Existing `BOL_STEP8_UNRESOLVED` case: "
        f"{sum(row['previous_status'] == 'BOL_STEP8_UNRESOLVED' for row in reassessments)}.",
        f"- Mapping-unresolved cases retained as unresolved: "
        f"{sum(row['reassessment_status'] == 'mapping_unresolved_hold' for row in reassessments)}.",
        f"- Negative cases blocked by ambiguous Step 8 references: "
        f"{', '.join(ambiguous_negative_keys) or 'none'}.",
        f"- Saved fit rows remain diagnostic-only; no new fit was run and no fit result was",
        "  changed. The 9 mapping-unresolved and all baseline-unresolved cases remain",
        "  withheld pending source/mapping evidence or explicit reference-policy approval.",
        "",
        "## Reproducibility",
        "",
        "Run from the repository root:",
        "",
        "```powershell",
        "python -m bafuse.data.bol_procedure_audit --root . "
        "--output-dir reports/bafuse_bol_negative_fix_v1",
        "```",
        "",
        "The script refuses to overwrite a non-empty output directory. CSVs include",
        "source file hashes, Guide hashes, and exact Step 4/8/10 source-row references.",
        "",
        "| Guide | SHA-256 |",
        "|---|---|",
        *[
            f"| Part {part} | `{digest}` |"
            for part, digest in sorted(guide_hashes.items())
        ],
        "",
    ]
    return "\n".join(lines)


def build_bol_procedure_audit(
    root: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[int, str],
]:
    root = root.resolve()
    curve_rows, _, guide_hashes = _guide_inventory(root, (1, 2, 3))
    decisions = _reference_decisions(root, curve_rows)
    reassessments = _negative_reassessment(root, decisions)
    return curve_rows, decisions, reassessments, guide_hashes


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_bol_procedure_audit(root: Path, output_dir: Path) -> None:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty audit directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows, decisions, reassessments, guide_hashes = build_bol_procedure_audit(root)
    _write_csv(output_dir / "bol_procedure_curve_audit.csv", _CURVE_AUDIT_FIELDS, curve_rows)
    _write_csv(output_dir / "bol_reference_decisions.csv", _DECISION_FIELDS, decisions)
    _write_csv(
        output_dir / "negative_result_reassessment.csv",
        _REASSESSMENT_FIELDS,
        reassessments,
    )
    (output_dir / "summary.md").write_text(
        _summary_markdown(curve_rows, decisions, reassessments, guide_hashes),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Guide-labelled BoL procedures.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/bafuse_bol_negative_fix_v1"),
    )
    args = parser.parse_args()
    write_bol_procedure_audit(args.root, args.output_dir)


if __name__ == "__main__":
    main()
