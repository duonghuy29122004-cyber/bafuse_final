from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .audit import FileRecord, GuideEntry
from .pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
    pydma_fit_passes_gate,
)


_RPT_PATTERN = re.compile(r"BoL|RPT", re.IGNORECASE)


def _normalize_stage(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]", "", value.casefold())
    if normalized.startswith("bol"):
        return "bol"
    match = re.search(r"rpt(\d+)", normalized)
    return f"rpt{int(match.group(1))}" if match else normalized


def _normalize_identifier(value: str, prefix: str) -> str:
    return re.sub(r"[^0-9]", "", value.casefold().removeprefix(prefix.casefold()))


def _record_step4(record: FileRecord) -> dict[str, Any]:
    if record.file_type != "csv" or not _RPT_PATTERN.search(record.stage):
        return {}
    try:
        summary = json.loads(record.measurement_summary_json or "{}")
    except json.JSONDecodeError:
        return {}
    if not summary.get("step4_row_count"):
        return {}
    return summary


def _usable_step4_summary(record: FileRecord) -> dict[str, Any]:
    summary = _record_step4(record)
    if (
        summary.get("step4_capacity_raw_span") is None
        or summary.get("step4_row_count", 0) < 10
    ):
        return {}
    return summary


def _stage_order(stage: str, file_id: int) -> tuple[int, int]:
    if re.fullmatch(r"\s*BoL(?:\s+RPT)?\s*", stage, re.IGNORECASE):
        return (0, file_id)
    match = re.fullmatch(r"\s*RPT\s*(\d+)\s*", stage, re.IGNORECASE)
    if match:
        return (int(match.group(1)), file_id)
    return (10_000, file_id)


def _is_unambiguous_rpt_stage(stage: str) -> bool:
    return bool(
        re.fullmatch(r"\s*(?:BoL(?:\s+RPT)?|RPT\s*\d+)\s*", stage, re.IGNORECASE)
    )


def build_soh_labels(records: list[FileRecord], guide_entries: list[GuideEntry]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    csv_by_key: defaultdict[tuple[str, str, int], list[FileRecord]] = defaultdict(list)
    for record in records:
        if record.file_type == "csv" and record.file_id is not None and record.cell_id:
            csv_by_key[(record.group_id, record.cell_id, record.file_id)].append(record)
    candidates: list[dict[str, Any]] = []
    by_cell: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in guide_entries:
        if not _RPT_PATTERN.search(entry.stage):
            continue
        physical = f"G{entry.group_id}-Cell{entry.cell_id}"
        matching_records = csv_by_key.get(
            (entry.group_id, entry.cell_id, entry.file_id), []
        )
        stage_is_unambiguous = _is_unambiguous_rpt_stage(entry.stage)
        step4_matches = (
            [
                (candidate, summary)
                for candidate in matching_records
                if (summary := _usable_step4_summary(candidate))
            ]
            if stage_is_unambiguous
            else []
        )
        source_mapping_note = ""
        if len(matching_records) == 1:
            record = matching_records[0]
            summary = (
                _record_step4(record) if stage_is_unambiguous else {}
            )
        elif len(step4_matches) == 1:
            record, summary = step4_matches[0]
            source_mapping_note = (
                f" {len(matching_records)} CSVs match this Guide procedure id; "
                "selected the unique CSV with at least 10 Step 4 rows and a capacity span."
            )
        elif matching_records:
            record = None
            summary = {}
            source_mapping_note = (
                " Multiple CSVs match this Guide procedure id and the source cannot "
                "be resolved to one eligible Step 4 curve: "
                + ", ".join(candidate.relative_path for candidate in matching_records)
                + "."
            )
        else:
            record = None
            summary = {}

        if record and summary.get("step4_capacity_raw_span") is not None and summary.get("step4_row_count", 0) >= 10:
            capacity = float(summary["step4_capacity_raw_span"])
            current_signs = summary.get("step4_current_signs", [])
            status = "capacity_observed"
            notes = "Capacity is Step 4 Amphr span for documented CC-CV discharge to 2.5 V; absolute unit is not explicitly stated in Guide."
            if "positive" in current_signs and "negative" in current_signs:
                status = "review_mixed_current_signs"
                notes += " Step 4 includes both current signs; verify protocol segmentation."
        elif not stage_is_unambiguous:
            capacity = None
            status = "unavailable_ambiguous_rpt_mapping"
            notes = (
                "Guide stage combines multiple measurements or does not identify one "
                "BoL/RPT checkpoint; file-level Step 4 capacity cannot be assigned "
                "to a single SOH observation."
            )
        elif matching_records and record is None:
            capacity = None
            status = "unavailable_ambiguous_source_mapping"
            notes = (
                "Multiple CSVs map to the Guide procedure and no unique Step 4 "
                "source could be selected."
            )
        else:
            capacity = None
            status = "unavailable_no_valid_step4_csv"
            notes = "No readable non-empty CSV with at least 10 Step 4 capacity samples; MAT source not decoded."
        notes += source_mapping_note
        candidate = {
            "group_id": entry.group_id,
            "cell_id": entry.cell_id,
            "physical_cell_id": physical,
            "condition": "RPT",
            "soc": "",
            "measurement_id": entry.stage,
            "measurement_order": entry.file_id,
            "procedure": entry.procedure,
            "capacity_raw": capacity,
            "capacity_unit": "unknown_native_Amphr",
            "capacity_ah": None,
            "reference_capacity_raw": None,
            "reference_capacity_ah": None,
            "soh_percent": None,
            "label_source": "Oxford Part 1 RPT Step 4 CSV",
            "label_method": "Step 4 discharge capacity span divided by first documented BoL Step 4 capacity from the same group/cell, times 100",
            "label_confidence": "unavailable",
            "source_file": (
                record.relative_path
                if record
                else ";".join(item.relative_path for item in matching_records)
            ),
            "validation_status": status,
            "validation_notes": notes,
            "_sort": _stage_order(entry.stage, entry.file_id),
        }
        candidates.append(candidate)
        if stage_is_unambiguous and status != "unavailable_ambiguous_source_mapping":
            by_cell[physical].append(candidate)

    for physical, observations in by_cell.items():
        observations.sort(key=lambda row: row["_sort"])
        bol = [row for row in observations if row["measurement_id"].lower().startswith("bol") and row["capacity_raw"]]
        baseline = bol[0] if bol else None
        if baseline is None:
            for row in observations:
                row["validation_status"] = "unavailable_no_measured_bol_reference"
                row["validation_notes"] += " No valid same-cell BoL Step 4 discharge capacity is available."
            continue
        baseline_value = float(baseline["capacity_raw"])
        baseline["reference_capacity_raw"] = baseline_value
        baseline["soh_percent"] = 100.0
        baseline["validation_status"] = "baseline_reference"
        baseline["label_confidence"] = "medium_ratio_only"
        baseline["validation_notes"] += " This measured BoL value defines the 100% baseline; it is not an official SOH label."
        for row in observations:
            if row is baseline or row["capacity_raw"] is None:
                if row is not baseline:
                    row["reference_capacity_raw"] = baseline_value
                continue
            row["reference_capacity_raw"] = baseline_value
            row["soh_percent"] = 100.0 * float(row["capacity_raw"]) / baseline_value
            row["label_confidence"] = "medium_ratio_only"
            row["validation_status"] = (
                "review_mixed_current_signs"
                if row["validation_status"] == "review_mixed_current_signs"
                else "calculated_from_measured_capacity"
            )
            if row["soh_percent"] < 0 or row["soh_percent"] > 120:
                row["validation_status"] = "physically_implausible_review"
                row["validation_notes"] += " SOH outside 0-120%; retained without clipping and requires review."
        bol_values = [float(row["capacity_raw"]) for row in bol if row["capacity_raw"] is not None]
        if len(bol_values) > 1:
            relative_spread = (max(bol_values) - min(bol_values)) / baseline_value
            for duplicate in bol[1:]:
                duplicate["validation_status"] = "review_duplicate_bol_measurement"
                duplicate["validation_notes"] += " Additional Guide-mapped BoL RPT from the same cell; retained and compared with the selected baseline."
            if relative_spread > 0.02:
                for row in observations:
                    if row["soh_percent"] is not None:
                        row["label_confidence"] = "low_baseline_conflict"
                        row["validation_notes"] += f" Multiple BoL files differ by {relative_spread:.1%}; baseline conflict flagged."
        previous: dict[str, Any] | None = None
        for row in observations:
            if row["soh_percent"] is None:
                continue
            if previous and float(row["soh_percent"]) > float(previous["soh_percent"]) + 2.0:
                row["validation_notes"] += " SOH increased by >2 percentage points versus prior RPT; retained as a review flag."
                if row["validation_status"] == "calculated_from_measured_capacity":
                    row["validation_status"] = "review_soh_increase"
            previous = row

    output_rows = []
    for row in candidates:
        output_rows.append({key: value for key, value in row.items() if not key.startswith("_")})
    validation_rows = [
        {
            "label_type": "SOH",
            "group_id": row["group_id"],
            "physical_cell_id": row["physical_cell_id"],
            "measurement_id": row["measurement_id"],
            "source_file": row["source_file"],
            "status": row["validation_status"],
            "value_percent": row["soh_percent"],
            "notes": row["validation_notes"],
        }
        for row in output_rows
    ]
    return output_rows, validation_rows


def build_degradation_labels(
    records: list[FileRecord],
    guide_entries: list[GuideEntry],
    imported_results: Path | None,
    generated_results: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    imported: dict[tuple[str, str, str, str], dict[str, str]] = {}
    if imported_results is not None:
        if not imported_results.is_file():
            raise FileNotFoundError(f"PyDMA result file does not exist: {imported_results}")
        with imported_results.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                imported[
                    (
                        _normalize_identifier(row.get("group_id", ""), "group"),
                        _normalize_identifier(row.get("cell_id", ""), "cell"),
                        _normalize_stage(row.get("checkup", "")),
                        row.get("direction", "").strip().casefold(),
                    )
                ] = row
    for row in generated_results or []:
        imported[
            (
                _normalize_identifier(str(row.get("group_id", "")), "group"),
                _normalize_identifier(str(row.get("cell_id", "")), "cell"),
                _normalize_stage(str(row.get("checkup", ""))),
                str(row.get("direction", "")).strip().casefold(),
            )
        ] = {key: str(value) for key, value in row.items()}
    csv_keys = {
        (record.group_id, record.cell_id, record.file_id): record
        for record in records
        if record.file_type == "csv" and record.file_id is not None and record.cell_id
    }
    rows: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for entry in guide_entries:
        if not _RPT_PATTERN.search(entry.stage):
            continue
        record = csv_keys.get((entry.group_id, entry.cell_id, entry.file_id))
        source = record.relative_path if record else ""
        for direction, step in (("charge", 10), ("discharge", 8)):
            result = imported.get(
                (
                    _normalize_identifier(entry.group_id, "group"),
                    _normalize_identifier(entry.cell_id, "cell"),
                    _normalize_stage(entry.stage),
                    direction,
                )
            )
            result_status = result.get("diagnostic_label_status", "") if result else "not_run"
            accepted = result_status == "diagnostic_estimate_accepted"
            rmse = _optional_float(result.get("fit_rmse_mV")) if result else None
            accepted_runs = _optional_int(result.get("accepted_runs")) if result else 0
            total_runs = _optional_int(result.get("total_runs")) if result else 0
            source_matches = bool(result and Path(result.get("source_file", "")).name.casefold() == Path(source).name.casefold())
            if accepted and (
                not pydma_fit_passes_gate(
                    rmse_mV=rmse,
                    accepted_runs=accepted_runs,
                    is_accepted=True,
                )
                or not source_matches
            ):
                accepted = False
                result_status = "rejected_import_validation"
            values = {
                "lli_percent": _optional_float(result.get("LLI_percent_vs_BoL")) if accepted and result else None,
                "lam_anode_percent": _optional_float(result.get("LAM_anode_percent_vs_BoL")) if accepted and result else None,
                "lam_cathode_percent": _optional_float(result.get("LAM_cathode_percent_vs_BoL")) if accepted and result else None,
            }
            notes = (
                f"Imported prior PyDMA diagnostic fit; matching source filename, RMSE <{PYDMA_RMSE_THRESHOLD_MV:g} mV, and at least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs verified. Model-based estimate, not ground truth."
                if accepted
                else (
                    result.get("quality_note", "PyDMA was not run for this observation.")
                    if result
                    else "PyDMA not run for this observation; no estimate generated."
                )
            )
            row = {
                "group_id": entry.group_id,
                "cell_id": entry.cell_id,
                "physical_cell_id": f"G{entry.group_id}-Cell{entry.cell_id}",
                "measurement_id": entry.stage,
                "direction": direction,
                "test_step": step,
                "lam_anode_percent": values["lam_anode_percent"],
                "lam_cathode_percent": values["lam_cathode_percent"],
                "lli_percent": values["lli_percent"],
                "label_source": "PyDMA OCV-only fit" if accepted else "Unavailable",
                "label_method": "OCV_cell(SOC)=U_cathode(alpha_ca*SOC+beta_ca)-U_anode(alpha_an*SOC+beta_an); relative to same-cell BoL fit",
                "fit_success": accepted,
                "fit_residual_mV": rmse,
                "fit_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
                "accepted_runs": accepted_runs,
                "total_runs": total_runs,
                "identifiability_status": "diagnostic_only_not_independently_validated" if accepted else result_status,
                "label_confidence": "low_model_based" if accepted else "unavailable",
                "source_file": source,
                "source_result_file": str(imported_results) if imported_results else "",
                "validation_notes": notes,
            }
            rows.append(row)
            validation.append(
                {
                    "label_type": "LLI/LAM",
                    "group_id": entry.group_id,
                    "physical_cell_id": row["physical_cell_id"],
                    "measurement_id": entry.stage,
                    "source_file": source,
                    "status": row["identifiability_status"],
                    "value_percent": json.dumps(values, sort_keys=True),
                    "notes": notes,
                }
            )
    return rows, validation


def _optional_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
