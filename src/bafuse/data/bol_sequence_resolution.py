from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bafuse.data.bol_procedure_audit import (
    _candidate_records,
    _compare_step4_segments,
    _scan_timeline,
    _source_file_map,
)
from bafuse.data.soh_step4_scan import scan_step4_file
from bafuse.labeling.audit import read_guide_entries


_CELLS = (
    (1, "2", "3"),
    (1, "2", "4"),
    (2, "5", "5"),
    (2, "5", "16"),
)
_STEP4_LIMIT_V = 2.6
_POCV_LOW_LIMIT_V = 2.6
_POCV_HIGH_LIMIT_V = 4.1
_TARGET_KEYS = (
    ("1", "G2-Cell4", "6"),
    ("2", "G2-Cell3", "9"),
    ("2", "G2-Cell4", "7"),
    ("2", "G2-Cell4", "9"),
)
_INVENTORY_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "procedure",
    "procedure_file_id",
    "guide_stage",
    "guide_entry_ordinal",
    "guide_file",
    "guide_sha256",
    "readme_file",
    "readme_sha256",
    "source_file",
    "source_sha256",
    "source_size_bytes",
    "source_row_count",
    "testtime_first",
    "testtime_last",
    "testtime_decrease_count",
    "step4_raw_run_count",
    "step8_raw_run_count",
    "step10_raw_run_count",
    "step4_segment_count",
    "step8_candidate_count",
    "step10_candidate_count",
    "cross_source_step4_comparison_status",
    "cross_source_step4_max_abs_difference",
]
_SEQUENCE_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "procedure",
    "procedure_file_id",
    "sequence_id",
    "sequence_order",
    "guide_stage",
    "source_file",
    "source_sha256",
    "sequence_start_row",
    "sequence_end_row",
    "sequence_boundary_end_row",
    "testtime_first",
    "testtime_last",
    "testtime_decrease_count",
    "step4_reference",
    "step4_row_count",
    "step4_testtime_first",
    "step4_testtime_last",
    "step4_capacity_finite_rows",
    "step4_capacity_span_native",
    "step4_current_min",
    "step4_current_max",
    "step4_current_sign_counts_json",
    "step4_current_direction_changes",
    "step4_voltage_min",
    "step4_voltage_max",
    "step4_complete",
    "step8_raw_runs_json",
    "step8_candidates_json",
    "step8_reference",
    "step8_status",
    "step8_testtime_first",
    "step8_testtime_last",
    "step8_current_min",
    "step8_current_max",
    "step8_voltage_min",
    "step8_voltage_max",
    "step8_capacity_span_native",
    "step10_raw_runs_json",
    "step10_candidates_json",
    "step10_reference",
    "step10_status",
    "step10_testtime_first",
    "step10_testtime_last",
    "step10_current_min",
    "step10_current_max",
    "step10_voltage_min",
    "step10_voltage_max",
    "step10_capacity_span_native",
    "intervening_steps_json",
    "protocol_order_status",
    "sequence_status",
    "protocol_consistent",
    "unresolved_reasons_json",
]
_CANDIDATE_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "procedure",
    "procedure_file_id",
    "sequence_id",
    "source_file",
    "sequence_status",
    "candidate_step4_reference",
    "candidate_step8_reference",
    "candidate_step10_reference",
    "is_complete_candidate",
    "eligibility_status",
    "selection_status",
    "selection_basis",
]
_DECISION_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "guide_primary_candidate_source",
    "original_selected_source",
    "original_step4_reference",
    "selected_sequence_id",
    "candidate_sequence_id",
    "selected_step4_reference",
    "selected_step8_reference",
    "selected_step10_reference",
    "candidate_step4_reference",
    "candidate_step8_reference",
    "candidate_step10_reference",
    "alternative_sequence_ids_json",
    "step4_reference_preserved",
    "selection_status",
    "confidence",
    "selection_basis",
    "existing_fit_impact",
]
_AFFECTED_FIELDS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "rpt_index",
    "split",
    "source_tpg",
    "source_file",
    "mapping_status",
    "step4_reference",
    "step4_valid",
    "step8_reference",
    "step8_valid",
    "step10_reference",
    "step10_valid",
    "source_reference_status",
    "duplicate_observation_key_count",
    "historical_fit_rows",
    "historical_accepted_fit_rows",
    "historical_negative_output_rows",
    "historical_positive_output_rows",
    "historical_best_rmse_mV",
    "historical_outputs_json",
    "parameter_vector_status",
    "impact_status",
]
_REASSESSMENT_FIELDS = [
    "dataset_part",
    "physical_cell_id",
    "rpt_index",
    "previous_worklist_status",
    "target_source_file",
    "target_step8_reference",
    "target_step10_reference",
    "historical_negative_outputs_json",
    "baseline_source_file",
    "baseline_sequence_id",
    "baseline_step8_reference",
    "baseline_step10_reference",
    "reference_decision_status",
    "saved_parameter_reuse_status",
    "pilot_status",
    "pilot_selected_fit_id",
    "pilot_selected_rmse_mV",
    "pilot_selected_labels_json",
    "action",
]
_ISSUE_FIELDS = [
    "dataset_part",
    "physical_cell_id",
    "source_file",
    "candidate_sequence_id",
    "issue_type",
    "evidence",
    "recommended_action",
    "severity",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _source_reference(source: str, step: int, segment: dict[str, Any]) -> str:
    return (
        f"{source}#Step{step}:segment{segment['segment_index']}:"
        f"rows{segment['source_row_start']}-{segment['source_row_end']}"
    )


def _candidate_is_complete(
    candidate: dict[str, Any],
    raw_run: dict[str, Any] | None,
    step: int,
) -> tuple[bool, str]:
    if not candidate.get("valid"):
        return False, "candidate_fails_monotone_curve_checks"
    if raw_run is None:
        return False, "raw_step_run_missing"
    if int(raw_run.get("testtime_decrease_count", 0)):
        return False, "testtime_decreased_within_raw_step"
    if int(raw_run.get("current_direction_changes", 0)):
        return False, "current_direction_changed_within_raw_step"

    counts = raw_run.get("current_sign_counts", {})
    if step == 8:
        if int(counts.get("negative", 0)) == 0 or int(counts.get("positive", 0)):
            return False, "step8_current_direction_not_discharge_only"
        if float(candidate["voltage_min"]) > _POCV_LOW_LIMIT_V:
            return False, "step8_did_not_reach_protocol_lower_voltage"
        if float(candidate["voltage_max"]) < _POCV_HIGH_LIMIT_V:
            return False, "step8_did_not_cover_upper_voltage"
    elif step == 10:
        if int(counts.get("positive", 0)) == 0 or int(counts.get("negative", 0)):
            return False, "step10_current_direction_not_charge_only"
        if float(candidate["voltage_max"]) < _POCV_HIGH_LIMIT_V:
            return False, "step10_did_not_reach_protocol_upper_voltage"
    else:
        raise ValueError(f"Unsupported pOCV Step: {step}")
    return True, "complete_protocol_voltage_and_current_span"


def reconstruct_step_sequences(
    *,
    part: int,
    group_id: str,
    physical_cell_id: str,
    procedure: str,
    procedure_file_id: int,
    source_file: str,
    source_sha256: str,
    guide_stage: str,
    timeline: dict[str, Any],
    step4_segments: list[dict[str, Any]],
    step8_candidates: list[dict[str, Any]],
    step10_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    source_row_count = int(timeline["source_row_count"])
    step_runs = timeline["step_runs"]
    step4_runs = {
        int(run["segment_index"]): run
        for run in step_runs
        if int(run["step"]) == 4
    }
    step8_runs = [run for run in step_runs if int(run["step"]) == 8]
    step10_runs = [run for run in step_runs if int(run["step"]) == 10]
    all_step_runs = timeline["all_step_runs"]

    if len(step4_segments) != len(step4_runs):
        raise ValueError(
            f"Step 4 scanner/timeline segment counts differ for {source_file}."
        )
    for segment in step4_segments:
        raw_run = step4_runs.get(int(segment["segment_index"]))
        if raw_run is None or (
            int(raw_run["source_row_start"]) != int(segment["source_row_start"])
            or int(raw_run["source_row_end"]) != int(segment["source_row_end"])
        ):
            raise ValueError(
                f"Step 4 scanner/timeline boundaries differ for {source_file}."
            )

    rows: list[dict[str, Any]] = []
    ordered_step4 = sorted(step4_segments, key=lambda item: int(item["source_row_start"]))
    for position, segment in enumerate(ordered_step4):
        step4_index = int(segment["segment_index"])
        step4_run = step4_runs[step4_index]
        start_row = int(segment["source_row_start"])
        end_row = (
            int(ordered_step4[position + 1]["source_row_start"]) - 1
            if position + 1 < len(ordered_step4)
            else source_row_count + 1
        )
        raw8_overlaps = [
            run
            for run in step8_runs
            if int(run["source_row_start"]) < end_row
            and int(run["source_row_end"]) >= end_row
        ]
        raw10_overlaps = [
            run
            for run in step10_runs
            if int(run["source_row_start"]) < end_row
            and int(run["source_row_end"]) >= end_row
        ]
        raw8 = [
            run
            for run in step8_runs
            if int(run["source_row_start"]) > int(segment["source_row_end"])
            and int(run["source_row_start"]) < end_row
            and int(run["source_row_end"]) < end_row
        ]
        raw10 = [
            run
            for run in step10_runs
            if int(run["source_row_start"]) > int(segment["source_row_end"])
            and int(run["source_row_start"]) < end_row
            and int(run["source_row_end"]) < end_row
        ]
        candidates8 = [
            item
            for item in step8_candidates
            if int(item["source_row_start"]) > int(segment["source_row_end"])
            and int(item["source_row_start"]) < end_row
            and int(item["source_row_end"]) < end_row
        ]
        candidates10 = [
            item
            for item in step10_candidates
            if int(item["source_row_start"]) > int(segment["source_row_end"])
            and int(item["source_row_start"]) < end_row
            and int(item["source_row_end"]) < end_row
        ]
        raw8_by_index = {int(run["segment_index"]): run for run in raw8}
        raw10_by_index = {int(run["segment_index"]): run for run in raw10}
        complete8 = [
            item
            for item in candidates8
            if _candidate_is_complete(
                item, raw8_by_index.get(int(item["step_segment_index"])), 8
            )[0]
        ]
        complete10 = [
            item
            for item in candidates10
            if _candidate_is_complete(
                item, raw10_by_index.get(int(item["step_segment_index"])), 10
            )[0]
        ]
        step4_signs = set(segment.get("current_signs", []))
        step4_complete = bool(
            int(segment.get("capacity_finite_rows", 0)) >= 10
            and float(segment.get("capacity_span") or 0) > 0
            and "negative" in step4_signs
            and "positive" not in step4_signs
            and float(step4_run.get("volts_min") or float("inf")) <= _STEP4_LIMIT_V
            and int(step4_run.get("current_direction_changes", 0)) == 0
            and int(step4_run.get("testtime_decrease_count", 0)) == 0
        )
        reasons: list[str] = []
        if raw8_overlaps:
            reasons.append("step8_raw_run_overlaps_next_step4_boundary")
        if raw10_overlaps:
            reasons.append("step10_raw_run_overlaps_next_step4_boundary")
        if not step4_complete:
            reasons.append("step4_fails_discharge_or_lower_cutoff_checks")
        if not candidates8:
            reasons.append("no_low_rate_step8_candidate_in_sequence")
        elif len(candidates8) > 1:
            reasons.append("multiple_step8_candidates_in_sequence")
        elif not complete8:
            reasons.append(
                _candidate_is_complete(
                    candidates8[0],
                    raw8_by_index.get(int(candidates8[0]["step_segment_index"])),
                    8,
                )[1]
            )
        if not candidates10:
            reasons.append("no_low_rate_step10_candidate_in_sequence")
        elif len(candidates10) > 1:
            reasons.append("multiple_step10_candidates_in_sequence")
        elif not complete10:
            reasons.append(
                _candidate_is_complete(
                    candidates10[0],
                    raw10_by_index.get(int(candidates10[0]["step_segment_index"])),
                    10,
                )[1]
            )

        chosen8 = candidates8[0] if len(candidates8) == 1 and len(complete8) == 1 else None
        chosen10 = candidates10[0] if len(candidates10) == 1 and len(complete10) == 1 else None
        order_status = "incomplete"
        if chosen8 and chosen10:
            if (
                int(segment["source_row_end"]) < int(chosen8["source_row_start"])
                and int(chosen8["source_row_end"]) < int(chosen10["source_row_start"])
                and int(chosen10["source_row_end"]) < end_row
            ):
                order_status = "step4_then_step8_then_step10"
            else:
                order_status = "out_of_order_or_overlapping"
                reasons.append("step_blocks_not_in_protocol_order")
        elif len(candidates8) > 1 or len(candidates10) > 1:
            order_status = "ambiguous_multiple_candidates"

        ambiguous = (
            len(candidates8) > 1
            or len(candidates10) > 1
            or bool(raw8_overlaps)
            or bool(raw10_overlaps)
            or int(step4_run.get("testtime_decrease_count", 0)) > 0
            or int(step4_run.get("current_direction_changes", 0)) > 0
            or order_status == "out_of_order_or_overlapping"
        )
        complete = bool(step4_complete and chosen8 and chosen10 and order_status == "step4_then_step8_then_step10")
        sequence_status = (
            "ambiguous"
            if ambiguous
            else "complete"
            if complete
            else "partial"
        )
        intervening_steps = [
            {
                "step": run["step"],
                "segment_index": run["segment_index"],
                "source_row_start": run["source_row_start"],
                "source_row_end": run["source_row_end"],
                "row_count": run["row_count"],
            }
            for run in all_step_runs
            if int(run["source_row_start"]) > int(segment["source_row_end"])
            and int(run["source_row_end"]) < end_row
            and int(run["step"]) not in {4, 8, 10}
        ]
        prefix = f"TPG{group_id}.{procedure_file_id}"
        sequence_id = f"{prefix}-SEQ{position + 1}"

        def summarize_candidates(
            candidates: list[dict[str, Any]],
            raw_by_index: dict[int, dict[str, Any]],
            step: int,
        ) -> list[dict[str, Any]]:
            summaries: list[dict[str, Any]] = []
            for candidate in candidates:
                raw_run = raw_by_index.get(int(candidate["step_segment_index"]))
                is_complete, reason = _candidate_is_complete(candidate, raw_run, step)
                summaries.append(
                    {
                        **{
                            key: value
                            for key, value in candidate.items()
                            if key not in {"capacity", "voltage"}
                        },
                        "raw_current_sign_counts": (
                            raw_run.get("current_sign_counts", {}) if raw_run else {}
                        ),
                        "raw_current_direction_changes": (
                            raw_run.get("current_direction_changes", "") if raw_run else ""
                        ),
                        "complete_protocol_candidate": is_complete,
                        "candidate_status": reason,
                    }
                )
            return summaries

        summaries8 = summarize_candidates(candidates8, raw8_by_index, 8)
        summaries10 = summarize_candidates(candidates10, raw10_by_index, 10)

        def selected_reference(candidate: dict[str, Any] | None) -> str:
            return str(candidate["source_reference"]) if candidate else ""

        def one_raw_value(runs: list[dict[str, Any]], field: str) -> Any:
            if len(runs) != 1:
                return ""
            return runs[0].get(field, "")

        observed_raw_runs = sorted(
            raw8 + raw10, key=lambda run: int(run["source_row_start"])
        )
        last_raw_run = observed_raw_runs[-1] if observed_raw_runs else step4_run
        observed8 = candidates8[0] if len(candidates8) == 1 else None
        observed10 = candidates10[0] if len(candidates10) == 1 else None
        sequence_end_row = max(
            [int(segment["source_row_end"])]
            + [int(run["source_row_end"]) for run in raw8 + raw10]
        )
        row = {
            "dataset_part": part,
            "group_id": group_id,
            "physical_cell_id": physical_cell_id,
            "procedure": procedure,
            "procedure_file_id": procedure_file_id,
            "sequence_id": sequence_id,
            "sequence_order": position + 1,
            "guide_stage": guide_stage,
            "source_file": source_file,
            "source_sha256": source_sha256,
            "sequence_start_row": start_row,
            "sequence_end_row": sequence_end_row,
            "sequence_boundary_end_row": end_row,
            "testtime_first": step4_run.get("testtime_first"),
            "testtime_last": last_raw_run.get("testtime_last"),
            "testtime_decrease_count": timeline["testtime_decrease_count"],
            "step4_reference": _source_reference(source_file, 4, segment),
            "step4_row_count": segment.get("step4_rows", ""),
            "step4_testtime_first": step4_run.get("testtime_first"),
            "step4_testtime_last": step4_run.get("testtime_last"),
            "step4_capacity_finite_rows": segment.get("capacity_finite_rows", ""),
            "step4_capacity_span_native": segment.get("capacity_span", ""),
            "step4_current_min": step4_run.get("amps_min", ""),
            "step4_current_max": step4_run.get("amps_max", ""),
            "step4_current_sign_counts_json": _json(
                step4_run.get("current_sign_counts", {})
            ),
            "step4_current_direction_changes": step4_run.get(
                "current_direction_changes", 0
            ),
            "step4_voltage_min": step4_run.get("volts_min", ""),
            "step4_voltage_max": step4_run.get("volts_max", ""),
            "step4_complete": step4_complete,
            "step8_raw_runs_json": _json(raw8),
            "step8_candidates_json": _json(summaries8),
            "step8_reference": selected_reference(chosen8),
            "step8_status": (
                "complete"
                if chosen8
                else "ambiguous"
                if len(candidates8) > 1
                else "partial"
                if candidates8
                else "missing_candidate"
            ),
            "step8_testtime_first": (
                raw8[0].get("testtime_first")
                if len(raw8) == 1
                else ""
            ),
            "step8_testtime_last": (
                raw8[0].get("testtime_last")
                if len(raw8) == 1
                else ""
            ),
            "step8_current_min": one_raw_value(raw8, "amps_min"),
            "step8_current_max": one_raw_value(raw8, "amps_max"),
            "step8_voltage_min": (
                observed8.get("voltage_min", "") if observed8 else ""
            ),
            "step8_voltage_max": (
                observed8.get("voltage_max", "") if observed8 else ""
            ),
            "step8_capacity_span_native": (
                observed8.get("capacity_span", "") if observed8 else ""
            ),
            "step10_raw_runs_json": _json(raw10),
            "step10_candidates_json": _json(summaries10),
            "step10_reference": selected_reference(chosen10),
            "step10_status": (
                "complete"
                if chosen10
                else "ambiguous"
                if len(candidates10) > 1
                else "partial"
                if candidates10
                else "missing_candidate"
            ),
            "step10_testtime_first": (
                raw10[0].get("testtime_first")
                if len(raw10) == 1
                else ""
            ),
            "step10_testtime_last": (
                raw10[0].get("testtime_last")
                if len(raw10) == 1
                else ""
            ),
            "step10_current_min": one_raw_value(raw10, "amps_min"),
            "step10_current_max": one_raw_value(raw10, "amps_max"),
            "step10_voltage_min": (
                observed10.get("voltage_min", "") if observed10 else ""
            ),
            "step10_voltage_max": (
                observed10.get("voltage_max", "") if observed10 else ""
            ),
            "step10_capacity_span_native": (
                observed10.get("capacity_span", "") if observed10 else ""
            ),
            "intervening_steps_json": _json(intervening_steps),
            "protocol_order_status": order_status,
            "sequence_status": sequence_status,
            "protocol_consistent": complete,
            "unresolved_reasons_json": _json(reasons),
            "guide_stage": guide_stage,
        }
        rows.append(row)
    return rows


def _guide_sources(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for part, group_id, cell_id in _CELLS:
        dataset_root = root / "data" / "raw" / "oxford" / f"part{part}"
        guide_paths = sorted(dataset_root.glob("Guide*.xlsx"))
        readme_path = dataset_root / "Readme.txt"
        if len(guide_paths) != 1 or not readme_path.is_file():
            raise FileNotFoundError(
                f"Expected one Guide XLSX and Readme.txt for Part {part}."
            )
        guide_path = guide_paths[0]
        entries = read_guide_entries(guide_path)
        source_map = _source_file_map(dataset_root)
        for file_id in (1, 2):
            matching_entries = [
                entry
                for entry in entries
                if entry.group_id == group_id
                and entry.cell_id == cell_id
                and entry.file_id == file_id
                and entry.stage.strip().casefold().startswith("bol")
            ]
            if len(matching_entries) != 1:
                raise ValueError(
                    f"Expected one Guide BoL entry for Part {part} G{group_id}-Cell{cell_id} "
                    f"file {file_id}; found {len(matching_entries)}."
                )
            entry = matching_entries[0]
            matches = source_map.get((group_id, cell_id, file_id), [])
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one source file for Part {part} G{group_id}-Cell{cell_id} "
                    f"file {file_id}; found {len(matches)}."
                )
            source = matches[0].resolve()
            source_relative = source.relative_to(root).as_posix()
            inventory.append(
                {
                    "part": part,
                    "group_id": group_id,
                    "physical_cell_id": f"G{group_id}-Cell{cell_id}",
                    "procedure": entry.procedure,
                    "procedure_file_id": file_id,
                    "guide_stage": entry.stage,
                    "guide_entry_ordinal": entries.index(entry) + 1,
                    "guide_path": guide_path,
                    "readme_path": readme_path,
                    "dataset_root": dataset_root,
                    "source_path": source,
                    "source_file": source_relative,
                }
            )
    return inventory


def _sequence_data(
    root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    source_inventory = _guide_sources(root)
    inventories: list[dict[str, Any]] = []
    sequences: list[dict[str, Any]] = []
    comparisons: dict[str, Any] = {}
    intermediate: dict[str, dict[str, Any]] = {}

    for source in source_inventory:
        timeline = _scan_timeline(source["source_path"])
        step4_scan = scan_step4_file(source["source_path"])
        if step4_scan["scan_status"] in {
            "scan_failed",
            "not_scanned",
            "missing_required_columns",
        }:
            raise ValueError(
                f"Step 4 scan failed for {source['source_file']}: "
                f"{step4_scan['scan_status']} {step4_scan['scan_error']}"
            )
        step4_segments = json.loads(step4_scan["step4_segments_json"])
        has_step8 = any(run["step"] == 8 for run in timeline["step_runs"])
        has_step10 = any(run["step"] == 10 for run in timeline["step_runs"])
        step8_candidates = (
            _candidate_records(
                root, source["source_path"], 8, "discharge", timeline["step_runs"]
            )
            if has_step8
            else []
        )
        step10_candidates = (
            _candidate_records(
                root, source["source_path"], 10, "charge", timeline["step_runs"]
            )
            if has_step10
            else []
        )
        source_sha256 = _sha256(source["source_path"])
        source_sequences = reconstruct_step_sequences(
            part=source["part"],
            group_id=source["group_id"],
            physical_cell_id=source["physical_cell_id"],
            procedure=source["procedure"],
            procedure_file_id=source["procedure_file_id"],
            source_file=source["source_file"],
            source_sha256=source_sha256,
            guide_stage=source["guide_stage"],
            timeline=timeline,
            step4_segments=step4_segments,
            step8_candidates=step8_candidates,
            step10_candidates=step10_candidates,
        )
        sequences.extend(source_sequences)
        intermediate[source["source_file"]] = {
            "source": source,
            "timeline": timeline,
            "step4_segments": step4_segments,
            "step8_candidates": step8_candidates,
            "step10_candidates": step10_candidates,
            "sequences": source_sequences,
            "source_sha256": source_sha256,
        }
        stat = source["source_path"].stat()
        inventories.append(
            {
                "dataset_part": source["part"],
                "group_id": source["group_id"],
                "physical_cell_id": source["physical_cell_id"],
                "procedure": source["procedure"],
                "procedure_file_id": source["procedure_file_id"],
                "guide_stage": source["guide_stage"],
                "guide_entry_ordinal": source["guide_entry_ordinal"],
                "guide_file": source["guide_path"].relative_to(root).as_posix(),
                "guide_sha256": _sha256(source["guide_path"]),
                "readme_file": source["readme_path"].relative_to(root).as_posix(),
                "readme_sha256": _sha256(source["readme_path"]),
                "source_file": source["source_file"],
                "source_sha256": source_sha256,
                "source_size_bytes": stat.st_size,
                "source_row_count": timeline["source_row_count"],
                "testtime_first": timeline["testtime_first"],
                "testtime_last": timeline["testtime_last"],
                "testtime_decrease_count": timeline["testtime_decrease_count"],
                "step4_raw_run_count": sum(
                    run["step"] == 4 for run in timeline["step_runs"]
                ),
                "step8_raw_run_count": sum(
                    run["step"] == 8 for run in timeline["step_runs"]
                ),
                "step10_raw_run_count": sum(
                    run["step"] == 10 for run in timeline["step_runs"]
                ),
                "step4_segment_count": len(step4_segments),
                "step8_candidate_count": len(step8_candidates),
                "step10_candidate_count": len(step10_candidates),
                "cross_source_step4_comparison_status": "",
                "cross_source_step4_max_abs_difference": "",
            }
        )

    inventory_by_key = {
        (row["dataset_part"], row["group_id"], row["physical_cell_id"], row["procedure_file_id"]): row
        for row in inventories
    }
    for part, group_id, cell_id in _CELLS:
        first_key = (part, group_id, f"G{group_id}-Cell{cell_id}", 1)
        second_key = (part, group_id, f"G{group_id}-Cell{cell_id}", 2)
        first = intermediate[inventory_by_key[first_key]["source_file"]]
        second = intermediate[inventory_by_key[second_key]["source_file"]]
        first_segments = first["step4_segments"]
        second_segments = second["step4_segments"]
        if not first_segments or not second_segments:
            comparison = {
                "status": "step4_segment_missing_in_one_source",
                "signals": "",
                "max_abs_difference": "",
            }
        else:
            comparison = _compare_step4_segments(
                first["source"]["source_path"],
                first_segments[-1],
                second["source"]["source_path"],
                second_segments[0],
            )
        comparisons[f"G{group_id}-Cell{cell_id}"] = comparison
        for key in (first_key, second_key):
            row = inventory_by_key[key]
            row["cross_source_step4_comparison_status"] = comparison["status"]
            row["cross_source_step4_max_abs_difference"] = comparison[
                "max_abs_difference"
            ]

    return inventories, sequences, {
        "intermediate": intermediate,
        "comparisons": comparisons,
        "source_inventory": source_inventory,
    }


def _reference_decisions(
    root: Path, sequences: list[dict[str, Any]], intermediate: dict[str, Any]
) -> list[dict[str, Any]]:
    stage1_path = root / "reports" / "bafuse_stage_validation_v1" / "stage1_bol_reference_audit.csv"
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Required Part 1/2 BoL audit is missing: {stage1_path}")
    with stage1_path.open("r", encoding="utf-8-sig", newline="") as stream:
        stage1_rows = list(csv.DictReader(stream))
    decisions: list[dict[str, Any]] = []
    for part, group_id, cell_id in _CELLS:
        physical_cell = f"G{group_id}-Cell{cell_id}"
        primary = [
            item
            for item in intermediate["source_inventory"]
            if item["part"] == part
            and item["physical_cell_id"] == physical_cell
            and item["procedure_file_id"] == 1
        ]
        if len(primary) != 1:
            raise ValueError(f"Expected one primary Guide BoL source for {physical_cell}.")
        source = primary[0]
        related = [
            row
            for row in sequences
            if row["dataset_part"] == part
            and row["physical_cell_id"] == physical_cell
            and row["source_file"] == source["source_file"]
        ]
        complete = sorted(
            (row for row in related if row["sequence_status"] == "complete"),
            key=lambda row: int(row["sequence_order"]),
        )
        prior = [
            row
            for row in stage1_rows
            if row["dataset_part"] == str(part)
            and row["physical_cell_id"] == physical_cell
            and row["observation_type"] == "BOL_REFERENCE"
        ]
        if len(prior) != 1:
            raise ValueError(f"Expected one historical BoL row for {physical_cell}.")
        prior_row = prior[0]
        original_source = str(prior_row.get("primary_bol_reference_file", ""))
        original_step4 = str(prior_row.get("bol_step4_segment_ref", ""))
        selected = complete[0] if complete and part == 1 else None
        candidate = complete[0] if complete else None
        rel_source = source["source_path"].relative_to(source["dataset_root"]).as_posix()
        original_source_relative = (
            f"data/raw/oxford/part{part}/{original_source}".replace("\\", "/")
            if original_source
            else ""
        )
        guide_candidate_source = source["source_path"].relative_to(root).as_posix()
        expected_step4_relative = (
            f"{rel_source}#Step4:"
            f"segment{selected['step4_reference'].split('#Step4:segment', 1)[1]}"
            if selected
            else ""
        )
        preserved = bool(
            selected
            and original_step4
            and original_step4.replace("\\", "/") == expected_step4_relative
        )
        if part == 1 and selected:
            status = "selected_provisional"
            confidence = "medium"
            basis = (
                "The official Guide identifies the BoL source; the official protocol orders "
                "Step4, Step8 discharge and Step10 charge. This is the earliest complete "
                "sequence in that source and its Step4 boundary matches the existing Q0 "
                "reference. No protocol text designates among the repeated sequences, so "
                "the choice remains provisional."
            )
            impact = (
                "Prior PyDMA results tied to this BoL source may be affected because the "
                "historical summaries do not preserve the exact Step8 boundary or fit "
                "parameters. Preserve historical outputs; only the explicitly requested "
                "G2-Cell4 RPT6 pilot is recomputed."
            )
        elif part == 2 and candidate:
            status = "not_promoted_policy_unresolved"
            confidence = "candidate_only"
            basis = (
                "A complete earliest sequence is reconstructed, but no verified Part 2 "
                "BoL reference-selection policy is established. The sequence is retained "
                "as a candidate only and is not passed to PyDMA."
            )
            impact = "No existing Part 2 reference or result is changed."
        else:
            status = "ambiguous_no_complete_sequence"
            confidence = "unresolved"
            basis = "No complete unambiguous Step4/Step8/Step10 sequence is available."
            impact = "No existing fit is changed."
        all_alternatives = [
            row for row in related if not selected or row["sequence_id"] != selected["sequence_id"]
        ]
        candidate_sequence = selected or candidate
        selected_path = selected or {}
        decisions.append(
            {
                "dataset_part": part,
                "group_id": group_id,
                "physical_cell_id": physical_cell,
                "guide_primary_candidate_source": guide_candidate_source,
                "original_selected_source": original_source_relative,
                "original_step4_reference": original_step4,
                "selected_sequence_id": selected_path.get("sequence_id", ""),
                "candidate_sequence_id": (
                    candidate_sequence.get("sequence_id", "")
                    if status == "not_promoted_policy_unresolved"
                    else ""
                ),
                "selected_step4_reference": selected_path.get("step4_reference", ""),
                "selected_step8_reference": selected_path.get("step8_reference", ""),
                "selected_step10_reference": selected_path.get("step10_reference", ""),
                "candidate_step4_reference": (
                    candidate_sequence.get("step4_reference", "")
                    if status == "not_promoted_policy_unresolved"
                    else ""
                ),
                "candidate_step8_reference": (
                    candidate_sequence.get("step8_reference", "")
                    if status == "not_promoted_policy_unresolved"
                    else ""
                ),
                "candidate_step10_reference": (
                    candidate_sequence.get("step10_reference", "")
                    if status == "not_promoted_policy_unresolved"
                    else ""
                ),
                "alternative_sequence_ids_json": _json(
                    [row["sequence_id"] for row in all_alternatives]
                ),
                "step4_reference_preserved": preserved,
                "selection_status": status,
                "confidence": confidence,
                "selection_basis": basis,
                "existing_fit_impact": impact,
            }
        )
    return decisions


def _normal_rpt(value: str) -> str:
    match = re.fullmatch(r"\s*RPT\s*(\d+)\s*", value, re.IGNORECASE)
    return match.group(1) if match else ""


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _is_negative(value: Any) -> bool:
    try:
        return float(value) < 0
    except (TypeError, ValueError):
        return False


def _is_positive(value: Any) -> bool:
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


def _mapped_references_valid(row: dict[str, str], data_row_count: str) -> bool:
    try:
        max_line = int(data_row_count) + 1
    except (TypeError, ValueError):
        return False
    for field in (
        "step4_segment_reference",
        "step8_segment_reference",
        "step10_segment_reference",
    ):
        reference = str(row.get(field, "")).strip()
        if not reference:
            continue
        match = re.search(r":rows(?P<start>\d+)-(?P<end>\d+)$", reference)
        if match is None:
            return False
        start, end = int(match.group("start")), int(match.group("end"))
        if start < 2 or end < start or end > max_line:
            return False
    return True


def _affected_observations(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mapping_path = root / "reports" / "bafuse_final_data_validation_v2" / "rpt_mapping_validation.csv"
    labels_path = root / "reports" / "final_mapping_label_audit" / "pydma_fit_audit.csv"
    inventory_path = root / "reports" / "bafuse_negative_root_cause_v1" / "negative_inventory.csv"
    for path in (mapping_path, labels_path, inventory_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required impact-audit input is missing: {path}")
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as stream:
        mappings = list(csv.DictReader(stream))
    with labels_path.open("r", encoding="utf-8-sig", newline="") as stream:
        label_rows = list(csv.DictReader(stream))
    with inventory_path.open("r", encoding="utf-8-sig", newline="") as stream:
        negative_rows = list(csv.DictReader(stream))

    key_counts = Counter(
        (row["part"], row["physical_cell_id"], row["rpt_index"])
        for row in mappings
    )
    all_duplicate_keys = [
        key for key, count in key_counts.items() if count > 1
    ]
    relevant = []
    for row in mappings:
        part = str(row["part"])
        cell = str(row["physical_cell_id"])
        if cell in {"G2-Cell3", "G2-Cell4"} and part in {"1", "2"}:
            relevant.append(row)
        elif cell in {"G5-Cell5", "G5-Cell16"} and part == "2":
            relevant.append(row)

    negative_by_key: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in negative_rows:
        negative_by_key[
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        ].append(row)

    result: list[dict[str, Any]] = []
    for row in relevant:
        part = str(row["part"])
        cell = str(row["physical_cell_id"])
        rpt = str(row["rpt_index"])
        key = (part, cell, rpt)
        fits = [
            fit
            for fit in label_rows
            if fit["dataset_part"] == part
            and fit["physical_cell_id"] == cell
            and _normal_rpt(fit["checkup"]) == rpt
        ]
        accepted = [fit for fit in fits if _truth(fit["fit_accepted_flag"])]
        negative_metrics = [
            {
                "checkup": fit["checkup"],
                "direction": fit["direction"],
                "rmse_mV": fit["fit_rmse_mV"],
                "accepted": fit["fit_accepted_flag"],
                "lli_percent": fit["lli_percent"],
                "lam_anode_percent": fit["lam_anode_percent"],
                "lam_cathode_percent": fit["lam_cathode_percent"],
                "source_file": fit["source_file"],
            }
            for fit in fits
        ]
        negative_outputs = negative_by_key.get(key, [])
        parameter_saved = [
            item
            for item in negative_outputs
            if _truth(item.get("parameter_vector_saved", ""))
        ]
        source_file = str(row["source_file"])
        source_rel = source_file
        if ":" in source_rel:
            source_rel = source_rel.split(":rows", 1)[0].split("#Step", 1)[0]
        source_path = root / source_rel
        source_reference_status = "source_missing"
        if source_path.is_file():
            source_reference_status = (
                "source_exists_and_row_boundaries_valid"
                if _mapped_references_valid(row, row["source_rows_scanned"])
                else "source_exists_but_row_reference_invalid"
            )
        if cell in {"G2-Cell3", "G2-Cell4"} and part == "1":
            impact = "part1_bol_step8_sequence_resolution_applies; historical outputs preserved"
        elif cell in {"G2-Cell3", "G2-Cell4"} and part == "2":
            impact = "same_physical_cell_continuation; part2_reference_policy_unresolved; no refit"
        else:
            impact = "part2_candidate_bol_not_promoted; no downstream reference changed"
        rmse_values = []
        for fit in accepted:
            try:
                rmse_values.append(float(fit["fit_rmse_mV"]))
            except (TypeError, ValueError):
                continue
        result.append(
            {
                "dataset_part": part,
                "group_id": row["group"],
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "split": row["split"],
                "source_tpg": row["source_tpg"],
                "source_file": row["source_file"],
                "mapping_status": row["final_mapping_status"],
                "step4_reference": row["step4_segment_reference"],
                "step4_valid": row["step4_valid"],
                "step8_reference": row["step8_segment_reference"],
                "step8_valid": row["step8_valid"],
                "step10_reference": row["step10_segment_reference"],
                "step10_valid": row["step10_valid"],
                "source_reference_status": source_reference_status,
                "duplicate_observation_key_count": key_counts[key],
                "historical_fit_rows": len(fits),
                "historical_accepted_fit_rows": len(accepted),
                "historical_negative_output_rows": sum(
                    1
                    for fit in accepted
                    if any(
                        _is_negative(fit[metric])
                        for metric in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
                        if str(fit[metric]).strip()
                    )
                ),
                "historical_positive_output_rows": sum(
                    1
                    for fit in accepted
                    if any(
                        _is_positive(fit[metric])
                        for metric in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
                        if str(fit[metric]).strip()
                    )
                ),
                "historical_best_rmse_mV": min(rmse_values) if rmse_values else "",
                "historical_outputs_json": _json(negative_metrics),
                "parameter_vector_status": (
                    f"{len(parameter_saved)} saved in negative inventory"
                    if parameter_saved
                    else "no saved parameter vector documented for reviewed negative rows"
                ),
                "impact_status": impact,
            }
        )
    return result, {
        "mapping_path": mapping_path,
        "labels_path": labels_path,
        "negative_inventory_path": inventory_path,
        "mapping_rows": len(mappings),
        "duplicate_observation_keys": all_duplicate_keys,
        "affected_mapping_rows": len(relevant),
        "label_rows": len(label_rows),
        "negative_inventory_rows": len(negative_rows),
    }


def _negative_reassessment(
    root: Path,
    decisions: list[dict[str, Any]],
    affected: list[dict[str, Any]],
    pilot_selected: list[dict[str, str]],
) -> list[dict[str, Any]]:
    with (root / "reports" / "bafuse_negative_correction_v2" / "negative_worklist.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as stream:
        worklist = list(csv.DictReader(stream))
    decisions_by_cell = {row["physical_cell_id"]: row for row in decisions}
    affected_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in affected
    }
    worklist_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in worklist
    }
    selected_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in pilot_selected
    }
    result = []
    for part, cell, rpt in _TARGET_KEYS:
        key = (part, cell, rpt)
        mapping = affected_by_key.get(key, {})
        decision = decisions_by_cell[cell]
        old = worklist_by_key.get(key, {})
        selected = selected_by_key.get(key, {})
        history = json.loads(mapping.get("historical_outputs_json", "[]"))
        negatives = [
            row
            for row in history
            if row["direction"] == "discharge"
            and _truth(row["accepted"])
            and any(
                str(row[metric]).strip() and _is_negative(row[metric])
                for metric in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
            )
        ]
        pilot_was_run = bool(selected)
        baseline_reference = (
            decision.get("selected_step8_reference", "")
            if decision["selection_status"] == "selected_provisional"
            else ""
        )
        if part == "1" and pilot_was_run:
            action = "Targeted G2-Cell4 RPT6 pilot completed; historical values retained."
            pilot_status = selected.get("status", "")
        elif part == "2":
            action = (
                "Held without refit: Part 2 reference policy is unresolved; "
                "candidate sequences were not promoted."
            )
            pilot_status = "not_run_policy_unresolved"
        else:
            action = "No compatible saved parameters; targeted pilot not run."
            pilot_status = "not_run"
        result.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "previous_worklist_status": old.get("status", ""),
                "target_source_file": mapping.get("source_file", ""),
                "target_step8_reference": mapping.get("step8_reference", ""),
                "target_step10_reference": mapping.get("step10_reference", ""),
                "historical_negative_outputs_json": _json(negatives),
                "baseline_source_file": (
                    decision.get("original_selected_source", "")
                    or (
                        "not selected; Guide candidate "
                        + decision.get("guide_primary_candidate_source", "")
                    )
                ),
                "baseline_sequence_id": decision.get("selected_sequence_id", ""),
                "baseline_step8_reference": baseline_reference,
                "baseline_step10_reference": (
                    decision.get("selected_step10_reference", "")
                    if decision["selection_status"] == "selected_provisional"
                    else ""
                ),
                "reference_decision_status": decision["selection_status"],
                "saved_parameter_reuse_status": (
                    "not_available_for_reuse; no saved accepted parameter vector in prior audit"
                ),
                "pilot_status": pilot_status,
                "pilot_selected_fit_id": selected.get("selected_fit_id", ""),
                "pilot_selected_rmse_mV": selected.get("selected_rmse_mV", ""),
                "pilot_selected_labels_json": selected.get("selected_labels_json", ""),
                "action": action,
            }
        )
    return result


def _remaining_issues(
    inventories: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    impact_context: dict[str, Any],
    affected: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for decision in decisions:
        if decision["selection_status"] == "not_promoted_policy_unresolved":
            issues.append(
                {
                    "dataset_part": decision["dataset_part"],
                    "physical_cell_id": decision["physical_cell_id"],
                    "source_file": decision["original_selected_source"],
                    "candidate_sequence_id": decision["candidate_sequence_id"],
                    "issue_type": "part2_bol_reference_policy_unresolved",
                    "evidence": (
                        "A complete candidate sequence exists, but no verified Part 2 "
                        "reference-selection policy authorizes promotion."
                    ),
                    "recommended_action": (
                        "Keep this sequence as a candidate; define and validate a Part 2 "
                        "reference policy before using it for PyDMA."
                    ),
                    "severity": "high",
                }
            )
    for cell, comparison in impact_context["comparisons"].items():
        if comparison["status"] == "matching_within_1e-8":
            source_rows = [
                row
                for row in inventories
                if row["physical_cell_id"] == cell
            ]
            issues.append(
                {
                    "dataset_part": source_rows[0]["dataset_part"] if source_rows else "",
                    "physical_cell_id": cell,
                    "source_file": ";".join(row["source_file"] for row in source_rows),
                    "candidate_sequence_id": "",
                    "issue_type": "matching_step4_signal_source_relationship_unresolved",
                    "evidence": (
                        "The last Step4 signal block in the primary Guide BoL source "
                        "matches the first Step4 block in the second Guide BoL source "
                        "within 1e-8. This does not establish duplicate acquisition, "
                        "re-export, or continuation."
                    ),
                    "recommended_action": (
                        "Retain both Guide source records; do not merge them without "
                        "independent acquisition/export provenance."
                    ),
                    "severity": "medium",
                }
            )
    for sequence in sequences:
        if sequence["sequence_status"] == "partial":
            issues.append(
                {
                    "dataset_part": sequence["dataset_part"],
                    "physical_cell_id": sequence["physical_cell_id"],
                    "source_file": sequence["source_file"],
                    "candidate_sequence_id": sequence["sequence_id"],
                    "issue_type": "incomplete_or_partial_bol_sequence",
                    "evidence": sequence["unresolved_reasons_json"],
                    "recommended_action": (
                        "Keep as a file-local sequence candidate; do not infer an RPT "
                        "index or use incomplete Step data as a complete degradation reference."
                    ),
                    "severity": "medium",
                }
            )
    if impact_context["duplicate_observation_keys"]:
        issues.append(
            {
                "dataset_part": "",
                "physical_cell_id": "",
                "source_file": str(impact_context["mapping_path"].relative_to(
                    impact_context["mapping_path"].parents[2]
                )),
                "candidate_sequence_id": "",
                "issue_type": "duplicate_observation_keys_in_existing_mapping",
                "evidence": _json(impact_context["duplicate_observation_keys"]),
                "recommended_action": (
                    "Resolve duplicates in a separate mapping change; this sequence audit "
                    "does not rewrite observation identities."
                ),
                "severity": "high",
            }
        )
    for observation in affected:
        if observation["source_reference_status"] != "source_exists_and_row_boundaries_valid":
            issues.append(
                {
                    "dataset_part": observation["dataset_part"],
                    "physical_cell_id": observation["physical_cell_id"],
                    "source_file": observation["source_file"],
                    "candidate_sequence_id": "",
                    "issue_type": "mapped_source_or_row_reference_invalid",
                    "evidence": observation["source_reference_status"],
                    "recommended_action": (
                        "Keep this observation out of sequence-based refits until its source "
                        "and exact row references are corrected."
                    ),
                    "severity": "high",
                }
            )
    return issues


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    headers = [label for _, label in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = str(row.get(key, "")).replace("|", "\\|").replace("\n", " ")
            values.append(value)
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _cell4_markdown(
    sequences: list[dict[str, Any]],
    comparison: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    rows = [
        row
        for row in sequences
        if row["physical_cell_id"] == "G2-Cell4"
        and row["dataset_part"] == 1
        and row["procedure_file_id"] == 1
    ]
    sequence_table = _markdown_table(
        rows,
        [
            ("sequence_id", "Sequence"),
            ("sequence_start_row", "Source-row start"),
            ("sequence_end_row", "Last observed row"),
            ("sequence_boundary_end_row", "Next-anchor boundary"),
            ("step4_reference", "Step 4"),
            ("step8_reference", "Step 8"),
            ("step10_reference", "Step 10"),
            ("sequence_status", "Sequence status"),
            ("unresolved_reasons_json", "Unresolved reasons"),
        ],
    )
    signal_table = _markdown_table(
        rows,
        [
            ("sequence_id", "Sequence"),
            ("testtime_first", "First TestTime"),
            ("testtime_last", "Last TestTime"),
            ("step4_current_min", "Step 4 Amps min"),
            ("step4_current_max", "Step 4 Amps max"),
            ("step4_voltage_min", "Step 4 Volts min"),
            ("step4_voltage_max", "Step 4 Volts max"),
            ("step4_capacity_span_native", "Step 4 Amphr span"),
            ("step8_current_min", "Step 8 Amps min"),
            ("step8_current_max", "Step 8 Amps max"),
            ("step8_voltage_min", "Step 8 Volts min"),
            ("step8_voltage_max", "Step 8 Volts max"),
            ("step8_capacity_span_native", "Step 8 Amphr span"),
            ("step10_current_min", "Step 10 Amps min"),
            ("step10_current_max", "Step 10 Amps max"),
            ("step10_voltage_min", "Step 10 Volts min"),
            ("step10_voltage_max", "Step 10 Volts max"),
            ("step10_capacity_span_native", "Step 10 Amphr span"),
        ],
    )
    return "\n".join(
        [
            "# Part 1 G2-Cell4 BoL sequence analysis",
            "",
            "The Guide maps `TPG2` file 1 to G2-Cell4 BoL. The source was scanned directly; "
            "sequence IDs below are file-local and are not RPT indices or cycle ages.",
            "",
            sequence_table,
            "",
            "## Timing and signal ranges",
            "",
            signal_table,
            "",
            "",
            "The official Part 1 Readme describes the RPT order as Step 4 discharge to 2.5 V, "
            "Step 8 pseudo-OCV discharge to 2.5 V, and Step 10 pseudo-OCV charge to 4.2 V. "
            "The audit uses chronological Step 4 anchors and assigns Step 8/10 blocks only "
            "inside the interval before the next Step 4; it does not pair by nearest timestamp.",
            "",
            f"Cross-source Step 4 signal comparison: `{comparison['status']}`, maximum "
            f"absolute difference `{comparison['max_abs_difference']}` across the compared "
            "signals. Signal equality alone does not establish a duplicate acquisition or export.",
            "",
            f"Decision: `{decision['selection_status']}` — sequence "
            f"`{decision['selected_sequence_id']}` is provisionally selected because it is the "
            "earliest complete sequence and its Step 4 boundary equals the existing Q0 "
            "reference. The existing Step 4 Q0 reference was not changed. The second sequence "
            "has full Step 8 voltage coverage but a partial Step 10 charge curve; the third "
            "Step 4 anchor has no qualifying low-rate Step 8/10 pair.",
            "",
            "The raw `TestTime` is reported only as within-file chronology; it is not used to "
            "infer RPT identity or cycle age. `Amphr` spans remain in native column units.",
            "",
        ]
    )


def _final_report(
    inventories: list[dict[str, Any]],
    sequences: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    affected: list[dict[str, Any]],
    reassessment: list[dict[str, Any]],
    issues: list[dict[str, Any]],
    impact_context: dict[str, Any],
    pilot_dir: Path | None,
) -> str:
    selected = [row for row in decisions if row["selection_status"] == "selected_provisional"]
    not_promoted = [
        row for row in decisions if row["selection_status"] == "not_promoted_policy_unresolved"
    ]
    complete = [row for row in sequences if row["sequence_status"] == "complete"]
    partial = [row for row in sequences if row["sequence_status"] == "partial"]
    ambiguous = [row for row in sequences if row["sequence_status"] == "ambiguous"]
    p1_bol_cell_keys = {
        ("1", "G2-Cell3"),
        ("1", "G2-Cell4"),
    }
    potentially_affected = sum(
        1
        for row in affected
        if (row["dataset_part"], row["physical_cell_id"]) in p1_bol_cell_keys
    )
    impact_groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in affected:
        impact_groups[(str(row["dataset_part"]), str(row["physical_cell_id"]))].append(row)
    impact_summary = []
    for (part, cell), rows in sorted(impact_groups.items()):
        impact_summary.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "mapped_observations": len(rows),
                "historical_fit_rows": sum(int(row["historical_fit_rows"]) for row in rows),
                "historical_accepted_fit_rows": sum(
                    int(row["historical_accepted_fit_rows"]) for row in rows
                ),
                "historical_negative_output_rows": sum(
                    int(row["historical_negative_output_rows"]) for row in rows
                ),
                "historical_positive_output_rows": sum(
                    int(row["historical_positive_output_rows"]) for row in rows
                ),
            }
        )
    return "\n".join(
        [
            "# BoL sequence resolution and PyDMA reference impact",
            "",
            "## Scope and evidence",
            "",
            "This versioned audit directly read the four primary BoL CSVs, their four "
            "second Guide-labelled BoL CSVs, both corresponding Guide workbooks, and the "
            "Part 1/2 Readmes. It did not modify raw data, the locked split, labels, model "
            "checkpoints, scalers, or historical PyDMA outputs. Part 3 was not used.",
            "",
            "The Guide establishes physical-cell/procedure association. The Readmes establish "
            "Step 4/8/10 protocol order and voltage endpoints. Within-file Step 4 anchors and "
            "the next-anchor boundary are used to reconstruct local sequences. No RPT index, "
            "cycle age, or cross-file acquisition relationship is inferred.",
            "",
            "Completeness uses the documented 2.5 V discharge and 4.2 V charge endpoints with "
            "a 0.1 V tolerance for recorded voltage coverage. `Amphr` spans are reported in "
            "native units; this audit does not assert Ah semantics.",
            "",
            "## Results",
            "",
            f"- Guide-mapped source files scanned: **{len(inventories)}** (four primary and four secondary).",
            f"- File-local sequences reconstructed: **{len(sequences)}**; complete **{len(complete)}**, "
            f"partial **{len(partial)}**, ambiguous **{len(ambiguous)}**.",
            f"- Part 1 provisional sequence references selected: **{len(selected)}** "
            "(G2-Cell3 and G2-Cell4).",
            f"- Part 2 complete candidates not promoted: **{len(not_promoted)}** "
            "(G5-Cell5 and G5-Cell16; policy remains unresolved).",
            f"- Existing mapped observations for affected physical cells audited: **{len(affected)}** "
            f"across **{potentially_affected}** Part 1 G2-Cell3/4 observations and linked "
            "Part 2 continuations / Part 2 G5 candidate cells.",
            f"- Duplicate observation keys in the existing mapping: **{len(impact_context['duplicate_observation_keys'])}**.",
            f"- Prior negative keys reassessed: **{len(reassessment)}**; no historical row was overwritten.",
            "",
            "## Reference decisions",
            "",
            _markdown_table(
                decisions,
                [
                    ("physical_cell_id", "Cell"),
                    ("guide_primary_candidate_source", "Guide primary candidate"),
                    ("original_selected_source", "Original source"),
                    ("selected_sequence_id", "Selected sequence"),
                    ("candidate_sequence_id", "Candidate only"),
                    ("selection_status", "Status"),
                    ("step4_reference_preserved", "Q0 Step 4 preserved"),
                    ("confidence", "Confidence"),
                ],
            ),
            "",
            "Part 1 selections are provisional, not new official protocol rules. Part 2 "
            "sequences are not used as references pending an explicit policy. The third "
            "Step 4 block in each primary file matches the first Step 4 block in the paired "
            "secondary file within the reported numerical tolerance; no duplication claim is made.",
            "",
            "## Negative-case reassessment",
            "",
            _markdown_table(
                reassessment,
                [
                    ("dataset_part", "Part"),
                    ("physical_cell_id", "Cell"),
                    ("rpt_index", "RPT"),
                    ("reference_decision_status", "Reference status"),
                    ("pilot_status", "Pilot status"),
                    ("pilot_selected_rmse_mV", "Pilot RMSE (mV)"),
                    ("action", "Action"),
                ],
            ),
            "",
            f"Targeted run artifacts: `{pilot_dir.as_posix() if pilot_dir else 'not run'}`. "
            "The only targeted observation is Part 1 G2-Cell4 RPT6. Part 2 G2-Cell3 RPT9 "
            "and G2-Cell4 RPT7/RPT9 remain held because a Part 2 reference policy is not verified.",
            "",
            "## Existing result impact",
            "",
            "The existing mapping and historical PyDMA output tables remain unchanged. The "
            "historical fit summaries do not provide reusable parameter vectors/reference "
            "normalization provenance for the affected keys. Therefore the report identifies "
            "potentially affected historical observations but does not relabel or claim that "
            "their old values changed. Only the single requested target pilot is newly fitted.",
            "",
            _markdown_table(
                impact_summary,
                [
                    ("dataset_part", "Part"),
                    ("physical_cell_id", "Cell"),
                    ("mapped_observations", "Mapped observations"),
                    ("historical_fit_rows", "Historical fit rows"),
                    ("historical_accepted_fit_rows", "Accepted fit rows"),
                    ("historical_negative_output_rows", "Rows with negative metric"),
                    ("historical_positive_output_rows", "Rows with positive metric"),
                ],
            ),
            "",
            "The locked 12/4/12 split is read from the existing mapping and left unchanged. "
            "All Part 3 observations remain external-only and outside this audit.",
            "",
            "## Reproduction",
            "",
            "From the repository root, use a fresh run ID so existing outputs are never "
            "overwritten:",
            "",
            "```powershell",
            "$env:PYTHONPATH='src'",
            ".\\.venv-torch-cuda\\Scripts\\python.exe -m bafuse.data.bol_sequence_resolution --run-id <new-id>",
            ".\\.venv\\Scripts\\python.exe -m unittest tests.test_bol_procedure_audit tests.test_bol_sequence_resolution tests.test_negative_correction",
            "```",
            "",
            "## Remaining issues",
            "",
            _markdown_table(
                issues,
                [
                    ("dataset_part", "Part"),
                    ("physical_cell_id", "Cell"),
                    ("candidate_sequence_id", "Sequence"),
                    ("issue_type", "Issue"),
                    ("evidence", "Evidence"),
                    ("recommended_action", "Recommended action"),
                ],
            ),
            "",
            "## Integration status",
            "",
            "Not ready for full BaFuse degradation integration: Part 2 reference selection "
            "remains unresolved, and the earlier BoL sequence's protocol primacy is provisional. "
            "The Part 1 G2-Cell4 RPT6 pilot is a bounded diagnostic, not a bulk relabeling or "
            "ground-truth update.",
            "",
        ]
    )


def finalize_sequence_resolution(
    root: Path,
    output_dir: Path,
    pilot_dir: Path,
    regression_result: str,
) -> Path:
    repo = root.resolve()
    output = output_dir if output_dir.is_absolute() else repo / output_dir
    pilot = pilot_dir if pilot_dir.is_absolute() else repo / pilot_dir
    output = output.resolve()
    pilot = pilot.resolve()
    required_outputs = (
        "bol_sequence_inventory.csv",
        "step_sequence_reconstruction.csv",
        "bol_reference_decisions.csv",
        "affected_observations.csv",
        "remaining_ambiguities.csv",
    )
    for filename in required_outputs:
        if not (output / filename).is_file():
            raise FileNotFoundError(f"Sequence audit output is missing: {output / filename}")

    def read_rows(path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    inventory = read_rows(output / "bol_sequence_inventory.csv")
    sequences = read_rows(output / "step_sequence_reconstruction.csv")
    decisions = read_rows(output / "bol_reference_decisions.csv")
    affected = read_rows(output / "affected_observations.csv")
    issues = read_rows(output / "remaining_ambiguities.csv")
    attempts_path = pilot / "targeted_refit_attempts.csv"
    selected_path = pilot / "selected_degradation_results.csv"
    if not attempts_path.is_file() or not selected_path.is_file():
        raise FileNotFoundError(f"Targeted PyDMA outputs are incomplete in {pilot}.")
    attempts = read_rows(attempts_path)
    selected = read_rows(selected_path)
    sequences_by_id = {row["sequence_id"]: row for row in sequences}
    for decision in decisions:
        candidate_id = decision.get("candidate_sequence_id", "")
        if candidate_id and not decision.get("candidate_step4_reference"):
            decision["candidate_step4_reference"] = sequences_by_id.get(
                candidate_id, {}
            ).get("step4_reference", "")
    _write_csv(output / "bol_reference_decisions.csv", _DECISION_FIELDS, decisions)
    _write_csv(
        output / "targeted_refit_attempts.csv",
        list(attempts[0]) if attempts else [],
        attempts,
    )
    _write_csv(
        output / "selected_degradation_results.csv",
        list(selected[0]) if selected else [],
        selected,
    )
    reassessment = _negative_reassessment(repo, decisions, affected, selected)
    _write_csv(output / "negative_reassessment.csv", _REASSESSMENT_FIELDS, reassessment)
    comparison_rows = [
        row
        for row in inventory
        if row["dataset_part"] == "1"
        and row["physical_cell_id"] == "G2-Cell4"
        and row["procedure_file_id"] == "1"
    ]
    secondary_rows = [
        row
        for row in inventory
        if row["dataset_part"] == "1"
        and row["physical_cell_id"] == "G2-Cell4"
        and row["procedure_file_id"] == "2"
    ]
    if len(comparison_rows) != 1 or len(secondary_rows) != 1:
        raise ValueError("G2-Cell4 primary/secondary BoL inventory rows are not unique.")
    comparison = {
        "status": comparison_rows[0]["cross_source_step4_comparison_status"],
        "max_abs_difference": comparison_rows[0][
            "cross_source_step4_max_abs_difference"
        ],
    }
    decisions_by_cell = {row["physical_cell_id"]: row for row in decisions}
    (output / "g2_cell4_sequence_analysis.md").write_text(
        _cell4_markdown(sequences, comparison, decisions_by_cell["G2-Cell4"]),
        encoding="utf-8",
    )
    _, impact_context = _affected_observations(repo)
    (output / "regression_test_results.md").write_text(
        "# Regression test results\n\n" + regression_result.strip() + "\n",
        encoding="utf-8",
    )
    (output / "final_bol_sequence_resolution_report.md").write_text(
        _final_report(
            inventory,
            sequences,
            decisions,
            affected,
            reassessment,
            issues,
            impact_context,
            pilot,
        ),
        encoding="utf-8",
    )
    return output


def run_targeted_pilot(
    root: Path, output_dir: Path, regression_result: str
) -> Path:
    repo = root.resolve()
    output = output_dir if output_dir.is_absolute() else repo / output_dir
    output = output.resolve()
    if not (output / "bol_reference_decisions.csv").is_file():
        raise FileNotFoundError(
            f"Run the raw-data sequence audit before the PyDMA pilot: {output}"
        )
    from bafuse.data.negative_correction import run_negative_correction

    pilot = output / "pilot_run"
    run_negative_correction(
        root=repo,
        run_id=f"sequence_{output.name}",
        scope="pilot",
        target_keys=(("1", "G2-Cell4", "6"),),
        bol_reference_manifest=output / "bol_reference_decisions.csv",
        output_dir=pilot,
    )
    return finalize_sequence_resolution(repo, output, pilot, regression_result)


def run_sequence_resolution(
    root: Path | None = None,
    run_id: str = "v1",
    run_pilot: bool = True,
    regression_result: str = "",
) -> Path:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(f"Invalid sequence-resolution run ID: {run_id!r}")
    output = repo / "reports" / f"bafuse_bol_sequence_resolution_{run_id}"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite sequence-resolution artifacts: {output}")

    inventories, sequences, context = _sequence_data(repo)
    decisions = _reference_decisions(repo, sequences, context)
    affected, impact_context = _affected_observations(repo)
    issue_context = {**impact_context, "comparisons": context["comparisons"]}
    issues = _remaining_issues(
        inventories, sequences, decisions, issue_context, affected
    )

    candidate_rows = []
    decision_by_cell = {row["physical_cell_id"]: row for row in decisions}
    for sequence in sequences:
        decision = decision_by_cell[sequence["physical_cell_id"]]
        chosen_id = decision["selected_sequence_id"]
        candidate_id = decision["candidate_sequence_id"]
        if sequence["sequence_id"] == chosen_id:
            selection_status = "selected_provisional"
        elif sequence["sequence_id"] == candidate_id:
            selection_status = "not_promoted_policy_unresolved"
        else:
            selection_status = "alternative_or_incomplete"
        candidate_rows.append(
            {
                "dataset_part": sequence["dataset_part"],
                "group_id": sequence["group_id"],
                "physical_cell_id": sequence["physical_cell_id"],
                "procedure": sequence["procedure"],
                "procedure_file_id": sequence["procedure_file_id"],
                "sequence_id": sequence["sequence_id"],
                "source_file": sequence["source_file"],
                "sequence_status": sequence["sequence_status"],
                "candidate_step4_reference": sequence["step4_reference"],
                "candidate_step8_reference": sequence["step8_reference"],
                "candidate_step10_reference": sequence["step10_reference"],
                "is_complete_candidate": sequence["sequence_status"] == "complete",
                "eligibility_status": (
                    "complete_sequence_candidate"
                    if sequence["sequence_status"] == "complete"
                    else sequence["sequence_status"]
                ),
                "selection_status": selection_status,
                "selection_basis": decision["selection_basis"],
            }
        )

    output.mkdir(parents=True)
    _write_csv(output / "bol_sequence_inventory.csv", _INVENTORY_FIELDS, inventories)
    _write_csv(output / "step_sequence_reconstruction.csv", _SEQUENCE_FIELDS, sequences)
    _write_csv(output / "bol_reference_candidates.csv", _CANDIDATE_FIELDS, candidate_rows)
    _write_csv(output / "bol_reference_decisions.csv", _DECISION_FIELDS, decisions)
    _write_csv(output / "affected_observations.csv", _AFFECTED_FIELDS, affected)
    _write_csv(output / "remaining_ambiguities.csv", _ISSUE_FIELDS, issues)

    pilot_dir: Path | None = None
    pilot_attempts: list[dict[str, str]] = []
    pilot_selected: list[dict[str, str]] = []
    if run_pilot:
        from bafuse.data.negative_correction import run_negative_correction

        pilot_dir = output / "pilot_run"
        run_negative_correction(
            root=repo,
            run_id=f"sequence_{run_id}",
            scope="pilot",
            target_keys=(("1", "G2-Cell4", "6"),),
            bol_reference_manifest=output / "bol_reference_decisions.csv",
            output_dir=pilot_dir,
        )
        for filename in (
            "targeted_refit_attempts.csv",
            "selected_degradation_results.csv",
        ):
            with (pilot_dir / filename).open("r", encoding="utf-8-sig", newline="") as stream:
                loaded = list(csv.DictReader(stream))
            if filename == "targeted_refit_attempts.csv":
                pilot_attempts.extend(loaded)
            else:
                pilot_selected.extend(loaded)
    else:
        _write_csv(output / "targeted_refit_attempts.csv", [], [])
        _write_csv(output / "selected_degradation_results.csv", [], [])
    if run_pilot:
        with (output / "targeted_refit_attempts.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            fields = list(pilot_attempts[0]) if pilot_attempts else []
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pilot_attempts)
        with (output / "selected_degradation_results.csv").open(
            "w", encoding="utf-8-sig", newline=""
        ) as stream:
            fields = list(pilot_selected[0]) if pilot_selected else []
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(pilot_selected)

    reassessment = _negative_reassessment(repo, decisions, affected, pilot_selected)
    _write_csv(output / "negative_reassessment.csv", _REASSESSMENT_FIELDS, reassessment)
    g2_comparison = context["comparisons"]["G2-Cell4"]
    (output / "g2_cell4_sequence_analysis.md").write_text(
        _cell4_markdown(sequences, g2_comparison, decision_by_cell["G2-Cell4"]),
        encoding="utf-8",
    )
    test_report = regression_result or (
        "Focused regression tests were not recorded by this invocation. "
        "Run `python -m unittest tests.test_bol_procedure_audit "
        "tests.test_bol_sequence_resolution tests.test_negative_correction`."
    )
    (output / "regression_test_results.md").write_text(
        "# Regression test results\n\n" + test_report.strip() + "\n",
        encoding="utf-8",
    )
    (output / "final_bol_sequence_resolution_report.md").write_text(
        _final_report(
            inventories,
            sequences,
            decisions,
            affected,
            reassessment,
            issues,
            impact_context,
            pilot_dir,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct Guide-labeled BoL Step4/8/10 sequences and audit PyDMA impact."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--skip-pilot", action="store_true")
    parser.add_argument("--regression-result", default="")
    args = parser.parse_args()
    print(
        run_sequence_resolution(
            root=args.root,
            run_id=args.run_id,
            run_pilot=not args.skip_pilot,
            regression_result=args.regression_result,
        )
    )


if __name__ == "__main__":
    main()
