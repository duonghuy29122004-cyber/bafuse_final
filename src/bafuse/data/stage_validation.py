from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_INPUT_DIR = Path("reports/bafuse_final_data_validation_v2")
_OUTPUT_DIR = Path("reports/bafuse_stage_validation_v1")
_SPLIT_COUNTS = {"train": 12, "validation": 4, "external_test": 12}
_STRICT_MAPPING = {"verified", "verified_from_ordered_raw_blocks"}
_RPT_PATTERN = re.compile(r"\bRPT\s*(\d+)\b", re.IGNORECASE)
_PRIORITY_FILES = (
    ("2", "5", "G5-Cell5", "TPG5.5", "Group_5/Group 5/TPG5.5-Cell5.csv"),
    ("2", "5", "G5-Cell6", "TPG5.5", "Group_5/Group 5/TPG5.5-Cell6.csv"),
    ("2", "5", "G5-Cell6", "TPG5.7", "Group_5/Group 5/TPG5.7-Cell6.csv"),
    ("2", "5", "G5-Cell16", "TPG5.8", "Group_5/Group 5/TPG5.8-Cell16.csv"),
    ("2", "5", "G5-Cell6", "TPG5.9", "Group_5/Group 5/TPG5.9-Cell6.csv"),
    ("2", "5", "G5-Cell6", "TPG5.10", "Group_5/Group 5/TPG5.10-Cell6.csv"),
    ("2", "5", "G5-Cell6", "TPG5.13", "Group_5/Group 5/TPG5.13-Cell6.csv"),
    ("2", "5", "G5-Cell16", "TPG5.13EoL", "Group_5/Group 5/TPG5.13EoL-Cell16.csv"),
    ("2", "6", "G6-Cell1", "TPG6.3", "Group_6/Group 6/TPG6.3-Cell1.csv"),
    ("2", "6", "G6-Cell1", "TPG6.8", "Group_6/Group 6/TPG6.8-Cell1.csv"),
)


def _read_csv(root: Path, relative: str) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required Stage 1 input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _source_path(root: Path, part: str, value: str) -> Path | None:
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    candidates = [root / path]
    if not raw.casefold().startswith("data/raw/oxford/"):
        candidates.append(root / "data" / "raw" / "oxford" / f"part{part}" / path)
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


def _source_exists(root: Path, part: str, value: str) -> bool:
    path = _source_path(root, part, value)
    return bool(path and path.is_file())


def _new_raw_step(step: int, index: int, start_row: int) -> dict[str, Any]:
    return {
        "step": step,
        "segment_index": index,
        "source_row_start": start_row,
        "source_row_end": start_row - 1,
        "raw_rows": 0,
        "finite_capacity_voltage_rows": 0,
        "current_negative_rows": 0,
        "current_positive_rows": 0,
        "current_zero_rows": 0,
        "capacity_min": float("inf"),
        "capacity_max": float("-inf"),
        "voltage_min": float("inf"),
        "voltage_max": float("-inf"),
        "low_rate_rows": 0,
        "low_valid_subsegments": 0,
        "low_subsegments": [],
        "low_segment_length": 0,
        "low_segment_start_row": None,
        "low_segment_end_row": None,
        "low_segment_capacity_min": float("inf"),
        "low_segment_capacity_max": float("-inf"),
        "low_segment_voltage_min": float("inf"),
        "low_segment_voltage_max": float("-inf"),
        "previous_low_capacity": None,
        "capacity_resets": 0,
        "previous_capacity": None,
    }


def _finish_low_subsegment(segment: dict[str, Any]) -> None:
    if not segment["low_segment_length"]:
        return
    valid = bool(
        segment["low_segment_length"] >= 10
        and segment["low_segment_capacity_max"]
        > segment["low_segment_capacity_min"]
    )
    if valid:
        segment["low_valid_subsegments"] += 1
    segment["low_subsegments"].append(
        {
            "source_row_start": segment["low_segment_start_row"],
            "source_row_end": segment["low_segment_end_row"],
            "sample_count": int(segment["low_segment_length"]),
            "capacity_min": segment["low_segment_capacity_min"],
            "capacity_max": segment["low_segment_capacity_max"],
            "voltage_min": segment["low_segment_voltage_min"],
            "voltage_max": segment["low_segment_voltage_max"],
            "valid_candidate": valid,
        }
    )
    segment["low_segment_length"] = 0
    segment["low_segment_start_row"] = None
    segment["low_segment_end_row"] = None
    segment["low_segment_capacity_min"] = float("inf")
    segment["low_segment_capacity_max"] = float("-inf")
    segment["low_segment_voltage_min"] = float("inf")
    segment["low_segment_voltage_max"] = float("-inf")


def _update_low_rate_capacity(
    segment: dict[str, Any],
    values: Any,
    source_rows: Any,
    voltages: Any,
) -> None:
    if not len(values):
        return
    values = values.astype("float64", copy=False)
    source_rows = source_rows.astype("int64", copy=False)
    voltages = voltages.astype("float64", copy=False)
    boundaries = list((values[1:] <= values[:-1]).nonzero()[0] + 1)
    previous = segment["previous_low_capacity"]
    if previous is not None and values[0] <= previous:
        _finish_low_subsegment(segment)
    starts = [0, *boundaries]
    ends = [*boundaries, len(values)]
    for sub_index, (start, end) in enumerate(zip(starts, ends)):
        if sub_index:
            _finish_low_subsegment(segment)
        if not segment["low_segment_length"]:
            segment["low_segment_start_row"] = int(source_rows[start])
        segment["low_segment_length"] += end - start
        segment["low_segment_end_row"] = int(source_rows[end - 1])
        segment["low_segment_capacity_min"] = min(
            segment["low_segment_capacity_min"], float(values[start:end].min())
        )
        segment["low_segment_capacity_max"] = max(
            segment["low_segment_capacity_max"], float(values[start:end].max())
        )
        finite_voltage = voltages[start:end]
        if len(finite_voltage):
            segment["low_segment_voltage_min"] = min(
                segment["low_segment_voltage_min"], float(finite_voltage.min())
            )
            segment["low_segment_voltage_max"] = max(
                segment["low_segment_voltage_max"], float(finite_voltage.max())
            )
    segment["previous_low_capacity"] = float(values[-1])


def _scan_raw_step_file(
    path: Path,
    part: str,
    group: str,
    cell: str,
    source_tpg: str,
    relative_path: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    row_offset = 0
    step_counts: dict[int, int] = {}

    def finish_active() -> None:
        nonlocal active
        if active is None:
            return
        if active["step"] in {4, 6, 8, 10}:
            _finish_low_subsegment(active)
            cap_min = active["capacity_min"]
            cap_max = active["capacity_max"]
            voltage_min = active["voltage_min"]
            voltage_max = active["voltage_max"]
            capacity_span = cap_max - cap_min if cap_min != float("inf") else 0.0
            step = active["step"]
            if step in {4, 6}:
                candidate = (
                    active["finite_capacity_voltage_rows"] >= 10
                    and capacity_span > 0
                    and (
                        active["current_negative_rows"] > 0
                        and active["current_positive_rows"] == 0
                        if step == 4
                        else active["current_positive_rows"] > 0
                        and active["current_negative_rows"] == 0
                    )
                )
            else:
                candidate = (
                    active["low_rate_rows"] >= 10
                    and active["low_valid_subsegments"] > 0
                )
            output.append(
                {
                    "part": part,
                    "group": group,
                    "physical_cell_id": cell,
                    "source_tpg": source_tpg,
                    "source_file": f"data/raw/oxford/part{part}/{relative_path}",
                    "step": step,
                    "segment_index": active["segment_index"],
                    "source_row_start": active["source_row_start"],
                    "source_row_end": active["source_row_end"],
                    "source_row_indexing": "one-based CSV line numbers; header is line 1",
                    "raw_rows": active["raw_rows"],
                    "finite_capacity_voltage_rows": active["finite_capacity_voltage_rows"],
                    "current_negative_rows": active["current_negative_rows"],
                    "current_positive_rows": active["current_positive_rows"],
                    "current_zero_rows": active["current_zero_rows"],
                    "capacity_min": cap_min if cap_min != float("inf") else "",
                    "capacity_max": cap_max if cap_max != float("-inf") else "",
                    "capacity_span": capacity_span,
                    "capacity_resets": active["capacity_resets"],
                    "voltage_min": voltage_min if voltage_min != float("inf") else "",
                    "voltage_max": voltage_max if voltage_max != float("-inf") else "",
                    "low_rate_rows": active["low_rate_rows"],
                    "low_valid_subsegments": active["low_valid_subsegments"],
                    "low_rate_subsegments": json.dumps(
                        active["low_subsegments"], separators=(",", ":")
                    ),
                    "curve_candidate": candidate,
                    "mapping_decision": "not_assigned_to_an_RPT_by_step_count",
                }
            )
        active = None

    for chunk in pd.read_csv(
        path,
        usecols=["Step", "Amphr", "Amps", "Volts"],
        chunksize=250_000,
    ):
        steps = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy()
        capacity = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy()
        current = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy()
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy()
        changes = (steps[1:] != steps[:-1]).nonzero()[0] + 1
        boundaries = [0, *changes.tolist(), len(steps)]
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            step_value = steps[start]
            if active is not None and (
                not pd.notna(step_value) or int(step_value) != active["step"]
            ):
                finish_active()
            if active is None and pd.notna(step_value):
                step = int(step_value)
                step_counts[step] = step_counts.get(step, 0) + 1
                active = _new_raw_step(step, step_counts[step], row_offset + start + 2)
            if active is None:
                continue
            segment = active
            segment["source_row_end"] = row_offset + end + 1
            segment["raw_rows"] += end - start
            cap = capacity[start:end]
            amps = current[start:end]
            volts = voltage[start:end]
            finite = pd.notna(cap) & pd.notna(volts)
            finite_cap = cap[finite]
            if len(finite_cap):
                segment["finite_capacity_voltage_rows"] += len(finite_cap)
                segment["capacity_min"] = min(segment["capacity_min"], float(finite_cap.min()))
                segment["capacity_max"] = max(segment["capacity_max"], float(finite_cap.max()))
                for value in finite_cap:
                    value = float(value)
                    if segment["previous_capacity"] is not None and value <= segment["previous_capacity"]:
                        segment["capacity_resets"] += 1
                    segment["previous_capacity"] = value
            finite_volts = volts[pd.notna(volts)]
            if len(finite_volts):
                segment["voltage_min"] = min(segment["voltage_min"], float(finite_volts.min()))
                segment["voltage_max"] = max(segment["voltage_max"], float(finite_volts.max()))
            finite_current = pd.notna(amps)
            amps_finite = amps[finite_current]
            segment["current_negative_rows"] += int((amps_finite < 0).sum())
            segment["current_positive_rows"] += int((amps_finite > 0).sum())
            segment["current_zero_rows"] += int((amps_finite == 0).sum())
            if segment["step"] in {8, 10}:
                direction = amps < 0 if segment["step"] == 8 else amps > 0
                selected = finite & finite_current & direction & (abs(amps) <= 0.2)
                selected_capacity = cap[selected]
                selected_voltage = volts[selected]
                selected_rows = row_offset + np.flatnonzero(selected) + start + 2
                segment["low_rate_rows"] += len(selected_capacity)
                _update_low_rate_capacity(
                    segment, selected_capacity, selected_rows, selected_voltage
                )
        row_offset += len(chunk)
    finish_active()
    return output


def _scan_priority_sources(root: Path) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for part, group, cell, source_tpg, relative_path in _PRIORITY_FILES:
        full_relative = f"data/raw/oxford/part{part}/{relative_path}"
        path = root / full_relative
        if not path.is_file():
            records.append(
                {
                    "part": part,
                    "group": group,
                    "physical_cell_id": cell,
                    "source_tpg": source_tpg,
                    "source_file": full_relative,
                    "scan_status": "missing_source",
                    "step": "",
                    "segment_index": "",
                    "source_row_start": "",
                    "source_row_end": "",
                    "source_row_indexing": "",
                    "raw_rows": "",
                    "finite_capacity_voltage_rows": "",
                    "current_negative_rows": "",
                    "current_positive_rows": "",
                    "current_zero_rows": "",
                    "capacity_min": "",
                    "capacity_max": "",
                    "capacity_span": "",
                    "capacity_resets": "",
                    "voltage_min": "",
                    "voltage_max": "",
                    "low_rate_rows": "",
                    "low_valid_subsegments": "",
                    "curve_candidate": False,
                    "mapping_decision": "source_missing",
                }
            )
            continue
        for record in _scan_raw_step_file(
            path, part, group, cell, source_tpg, relative_path
        ):
            record["scan_status"] = "scanned"
            records.append(record)
    return pd.DataFrame(records)


def _attach_priority_counts(
    frame: pd.DataFrame,
    priority_scan: pd.DataFrame,
) -> pd.DataFrame:
    result = frame.copy()
    summaries: dict[str, str] = {}
    for source_file, group in priority_scan.loc[
        priority_scan.scan_status.eq("scanned")
    ].groupby("source_file"):
        counts = group.groupby("step").curve_candidate.sum().to_dict()
        summaries[source_file.replace("\\", "/").casefold()] = "; ".join(
            f"Step{step} candidate blocks={int(counts.get(step, 0))}"
            for step in (4, 6, 8, 10)
        )
    result["priority_raw_step_candidates"] = [
        summaries.get(str(source).replace("\\", "/").casefold(), "")
        for source in result.source_file
    ]
    return result


def _part_for_cell(cell: str, mapping: pd.DataFrame, bol: pd.DataFrame) -> str:
    mapped = mapping.loc[mapping.physical_cell_id.eq(cell), "part"]
    if not mapped.empty:
        return str(sorted(mapped.unique(), key=lambda item: int(item))[0])
    candidates = bol.loc[bol.physical_cell_id.eq(cell), "dataset_part"]
    return str(candidates.iloc[0]) if not candidates.empty else ""


def _prepare_mapping(root: Path, mapping: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = mapping.copy()
    result["source_exists_actual"] = [
        _source_exists(root, str(row.part), str(row.source_file))
        for row in result.itertuples()
    ]
    result["stage1_mapping_status"] = result["mapping_status_v2"]
    missing = ~result.source_exists_actual
    result.loc[missing, "stage1_mapping_status"] = "missing_source"
    result["mapping_verified"] = result.stage1_mapping_status.isin(_STRICT_MAPPING)
    result["stage1_unresolved_reason"] = ""
    for index, row in result.loc[~result.mapping_verified].iterrows():
        status = row["stage1_mapping_status"]
        if status == "ambiguous_rpt":
            reason = "RPT identity or segment assignment remains ambiguous."
        elif status == "needs_curve_audit":
            reason = "Raw block or exact curve boundary is not verified."
        elif status == "missing_step":
            reason = "One or more required measurement steps are missing."
        elif status == "user_confirmed_identity_not_independently_verified":
            reason = "Physical-cell identity is user-confirmed but not independently Guide-verified."
        elif status == "missing_source":
            reason = "The mapped source path is absent from the current workspace."
        else:
            reason = f"Mapping status is not in the strict verified set: {status}."
        result.at[index, "stage1_unresolved_reason"] = reason
    duplicate = result.duplicated(["part", "physical_cell_id", "rpt_index"], keep=False)
    if duplicate.any():
        keys = result.loc[duplicate, ["part", "physical_cell_id", "rpt_index"]]
        raise ValueError(f"Duplicate observation keys in Stage 1 mapping input: {keys.to_dict('records')[:10]}")
    verified = result.loc[result.mapping_verified].copy()
    unresolved = result.loc[~result.mapping_verified].copy()
    return verified, unresolved


def _prepare_bol(
    root: Path,
    bol_candidates: pd.DataFrame,
    split: pd.DataFrame,
    mapping: pd.DataFrame,
    observations: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cells = split.drop_duplicates("physical_cell_id")
    for cell in cells.physical_cell_id:
        part = _part_for_cell(cell, mapping, bol_candidates)
        candidates = bol_candidates.loc[
            bol_candidates.physical_cell_id.eq(cell)
            & bol_candidates.dataset_part.eq(part)
        ].copy()
        if candidates.empty:
            candidates = bol_candidates.loc[bol_candidates.physical_cell_id.eq(cell)].copy()

        primary = candidates.loc[
            candidates.reference_selection_status.eq("part1_provisional_earliest_complete_bol_policy")
            & candidates.validation_status_original.eq("baseline_reference")
        ]
        if not primary.empty:
            source_match = primary.source_file.eq(primary.baseline_source_file)
            if source_match.any():
                primary = primary.loc[source_match]
            primary = primary.iloc[[0]]
        selected = primary.iloc[0] if not primary.empty else None
        primary_file = str(selected.source_file) if selected is not None else ""
        primary_actual = _source_exists(root, part, primary_file)
        step4_verified = bool(
            selected is not None
            and _truth(selected.get("source_exists", ""))
            and _truth(selected.get("baseline_source_exists", ""))
            and selected.get("audit_status") == "verified_source_and_arithmetic"
            and primary_actual
            and str(selected.get("step4_segment_index", "")).strip()
        )
        identity = "guide_crosswalk"
        if cell == "G10-Cell2":
            identity = "user_confirmed_not_independently_guide_verified"
        elif not candidates.empty and candidates.identity_provenance.astype(str).str.contains(
            "user_confirmed", case=False, na=False
        ).any():
            identity = "user_confirmed_not_independently_guide_verified"

        refs = observations.loc[
            observations.physical_cell_id.eq(cell), "bol_step4_segment_ref"
        ].drop_duplicates()
        bol_step4_ref = next((value for value in refs if value), "")
        candidate_files = sorted(
            {
                str(value)
                for value in candidates.source_file
                if str(value).strip()
            }
        )
        policy_status = (
            "part1_provisional_earliest_complete_bol_selected"
            if selected is not None and step4_verified
            else "primary_bol_policy_unresolved"
            if candidate_files
            else "missing_bol_source"
        )
        rows.append(
            {
                "dataset_part": part,
                "group": str(cell).split("-")[0].removeprefix("G"),
                "physical_cell_id": cell,
                "observation_type": "BOL_REFERENCE",
                "normalized_rpt_index": 0,
                "raw_bol_source_annotations": " || ".join(candidate_files),
                "primary_bol_reference_file": primary_file,
                "bol_candidate_files": " || ".join(candidate_files),
                "bol_source_exists": primary_actual if primary_file else False,
                "bol_step4_segment_ref": bol_step4_ref,
                "bol_step4_status": "verified" if step4_verified else "not_verified",
                "bol_step8_status": "not_audited_for_bol",
                "bol_step10_status": "not_audited_for_bol",
                "bol_verified": step4_verified,
                "bol_policy_status": policy_status,
                "reference_policy": (
                    "part1_earliest_complete_bol_provisional"
                    if part == "1"
                    else "unresolved"
                ),
                "identity_provenance": identity,
                "capacity_unit": (
                    str(selected.get("capacity_unit", "native Amphr; absolute unit unverified"))
                    if selected is not None
                    else "native Amphr; absolute unit unverified"
                ),
                "source_row_provenance": bol_step4_ref,
                "bol_step8_fit_status": "not_run",
                "bol_step10_fit_status": (
                    "existing_single_pilot_only"
                    if cell == "G1-Cell20"
                    else "not_run"
                ),
                "partition": str(
                    split.loc[split.physical_cell_id.eq(cell), "partition"].iloc[0]
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.physical_cell_id.duplicated().any():
        raise ValueError("Stage 1 BoL audit must contain one row per physical cell.")
    return result


def _rpt_number(value: str) -> str:
    match = _RPT_PATTERN.search(str(value))
    return match.group(1) if match else ""


def _soh_by_key(soh: pd.DataFrame) -> dict[tuple[str, str, str], pd.Series]:
    result: dict[tuple[str, str, str], pd.Series] = {}
    ordered = soh.sort_values(
        by="validation_status_v2",
        key=lambda values: values.ne("VERIFIED_DERIVED_LABEL"),
        ascending=False,
    )
    for _, row in ordered.iterrows():
        rpt = _rpt_number(row.get("measurement_id", ""))
        if rpt:
            result[(str(row.dataset_part), str(row.physical_cell_id), rpt)] = row
    return result


def _prepare_target_masks(
    availability: pd.DataFrame,
    soh: pd.DataFrame,
    bol: pd.DataFrame,
) -> pd.DataFrame:
    soh_index = _soh_by_key(soh)
    bol_index = {str(row.physical_cell_id): row for row in bol.itertuples()}
    rows: list[dict[str, Any]] = []
    for row in availability.itertuples():
        key = (str(row.part), str(row.physical_cell_id), str(row.rpt_index))
        label = soh_index.get(key)
        soh_status = str(label.validation_status_v2) if label is not None else str(row.soh_label_status)
        verified_soh = (
            soh_status == "VERIFIED_DERIVED_LABEL"
            and str(row.mapping_status) in _STRICT_MAPPING
            and str(row.step4_boundary_status) == "verified"
            and _truth(row.step4_available)
        )
        has_historical_pydma = int(row.historical_pydma_accepted_direction_records or 0) > 0
        bol_row = bol_index.get(str(row.physical_cell_id))
        bol_verified = bool(bol_row and bol_row.bol_verified)
        source_exists = _truth(row.step4_available) or _truth(row.step8_available) or _truth(row.step10_available)
        rows.append(
            {
                "dataset_part": str(row.part),
                "group": str(row.group),
                "physical_cell_id": str(row.physical_cell_id),
                "observation_type": "RPT",
                "raw_rpt_index": str(row.rpt_index),
                "normalized_rpt_index": str(row.rpt_index),
                "source_file": str(row.source_file),
                "partition": str(row.partition),
                "mapping_status": str(row.mapping_status),
                "mapping_verified": str(row.mapping_status) in _STRICT_MAPPING,
                "step_curve_candidate_available": source_exists,
                "bol_verified": bol_verified,
                "bol_reference_file": str(bol_row.primary_bol_reference_file) if bol_row else "",
                "bol_policy_status": str(bol_row.bol_policy_status) if bol_row else "missing_bol_audit",
                "step4_verified": str(row.step4_boundary_status) == "verified",
                "step6_status": str(row.step6_charge_status),
                "step8_verified": str(row.step8_boundary_status) == "verified",
                "step10_verified": str(row.step10_boundary_status) == "verified",
                "pydma_fit_verified": False,
                "fit_status": (
                    "historical_diagnostic_not_reproduced"
                    if has_historical_pydma
                    else "UNAVAILABLE"
                ),
                "fit_rmse_mv": "",
                "fit_reproduced": False,
                "has_verified_soh": verified_soh,
                "has_verified_lli": False,
                "has_verified_lam_anode": False,
                "has_verified_lam_cathode": False,
                "soh_percent": (
                    str(label.recomputed_soh_percent)
                    if verified_soh and label is not None
                    else ""
                ),
                "lli_percent": "",
                "lam_anode_percent": "",
                "lam_cathode_percent": "",
                "soh_label_origin": (
                    "VERIFIED_DERIVED_SOH" if verified_soh else "UNAVAILABLE"
                ),
                "label_origin": (
                    "VERIFIED_DERIVED_SOH"
                    if verified_soh
                    else "PYDMA_FITTED_DIAGNOSTIC"
                    if has_historical_pydma
                    else "UNAVAILABLE"
                ),
                "target_verified": verified_soh,
                "soh_label_status": soh_status,
                "reference_training_eligible": False,
                "historical_pydma_accepted_direction_records": str(
                    row.historical_pydma_accepted_direction_records
                ),
            }
        )

    for row in bol.itertuples():
        rows.append(
            {
                "dataset_part": str(row.dataset_part),
                "group": str(row.group),
                "physical_cell_id": str(row.physical_cell_id),
                "observation_type": "BOL_REFERENCE",
                "raw_rpt_index": "",
                "normalized_rpt_index": "0",
                "source_file": str(row.primary_bol_reference_file),
                "partition": str(row.partition),
                "mapping_status": "reference_row",
                "mapping_verified": False,
                "source_measurement_available": bool(row.bol_source_exists),
                "bol_verified": bool(row.bol_verified),
                "bol_reference_file": str(row.primary_bol_reference_file),
                "bol_policy_status": str(row.bol_policy_status),
                "step4_verified": str(row.bol_step4_status) == "verified",
                "step6_status": "not_audited_for_bol",
                "step8_verified": False,
                "step10_verified": False,
                "pydma_fit_verified": False,
                "fit_status": "reference_fit_not_validated",
                "fit_rmse_mv": "",
                "fit_reproduced": False,
                "has_verified_soh": False,
                "has_verified_lli": False,
                "has_verified_lam_anode": False,
                "has_verified_lam_cathode": False,
                "soh_percent": "100.0",
                "lli_percent": "0.0",
                "lam_anode_percent": "0.0",
                "lam_cathode_percent": "0.0",
                "soh_label_origin": "REFERENCE_DEFINITION",
                "label_origin": "REFERENCE_DEFINITION",
                "target_verified": False,
                "soh_label_status": str(row.bol_policy_status),
                "reference_training_eligible": False,
                "historical_pydma_accepted_direction_records": "0",
            }
        )
    result = pd.DataFrame(rows)
    key_columns = ["dataset_part", "group", "physical_cell_id", "normalized_rpt_index"]
    duplicate = result.duplicated(key_columns, keep=False)
    if duplicate.any():
        keys = result.loc[duplicate, key_columns]
        raise ValueError(f"Duplicate Stage 1 target-mask keys: {keys.to_dict('records')[:10]}")
    return result


def _prepare_split_integrity(
    split: pd.DataFrame,
    mapping: pd.DataFrame,
    target_masks: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    cells_unique = not split.physical_cell_id.duplicated().any()
    rows = split.drop_duplicates("physical_cell_id").copy()
    counts = rows.partition.value_counts().to_dict()
    part_by_cell = {
        cell: _part_for_cell(cell, mapping, pd.DataFrame(columns=["physical_cell_id", "dataset_part"]))
        for cell in rows.physical_cell_id
    }
    part3_external = all(
        str(part_by_cell[cell]) != "3" or str(partition) == "external_test"
        for cell, partition in zip(rows.physical_cell_id, rows.partition)
    )
    no_part12_external = all(
        str(part_by_cell[cell]) == "3" or str(partition) != "external_test"
        for cell, partition in zip(rows.physical_cell_id, rows.partition)
    )
    assignment = dict(zip(rows.physical_cell_id, rows.partition))
    rpt_targets = target_masks.loc[target_masks.observation_type.eq("RPT")]
    complete_assignment = set(rpt_targets.physical_cell_id) == set(rows.physical_cell_id)
    observation_partitions_match = all(
        assignment.get(row.physical_cell_id) == row.partition
        for row in target_masks.itertuples()
    )
    no_internal_test = not rows.partition.str.casefold().eq("internal_test").any()
    global_pass = (
        counts == _SPLIT_COUNTS
        and cells_unique
        and part3_external
        and no_part12_external
        and complete_assignment
        and observation_partitions_match
        and no_internal_test
        and not target_masks.duplicated(
            ["dataset_part", "group", "physical_cell_id", "normalized_rpt_index"]
        ).any()
    )
    output = rows[["physical_cell_id", "partition"]].copy()
    output["dataset_part"] = output.physical_cell_id.map(part_by_cell)
    output["cell_unique"] = cells_unique
    output["part3_external"] = output.apply(
        lambda row: row.dataset_part != "3" or row.partition == "external_test",
        axis=1,
    )
    output["part12_not_external"] = output.apply(
        lambda row: row.dataset_part == "3" or row.partition != "external_test",
        axis=1,
    )
    output["rpt_rows_assigned"] = output.physical_cell_id.isin(
        set(rpt_targets.physical_cell_id)
    )
    output["observation_partitions_match"] = output.physical_cell_id.map(
        lambda cell: all(
            assignment.get(row.physical_cell_id) == row.partition
            for row in target_masks.loc[
                target_masks.physical_cell_id.eq(cell)
            ].itertuples()
        )
    )
    output["no_internal_test"] = no_internal_test
    output["integrity_status"] = "PASS" if global_pass else "FAIL"
    return output, global_pass


def _probe_mat(root: Path, availability: pd.DataFrame) -> list[str]:
    samples = availability.loc[
        availability.extension.eq(".mat")
        & availability.availability_status.eq("present")
    ]
    findings: list[str] = []
    try:
        from scipy.io import whosmat
    except ImportError as error:
        return [f"MAT inspection unavailable: {type(error).__name__}: {error}"]
    for part in ("1", "2", "3"):
        candidates = samples.loc[samples.part.eq(part)]
        if candidates.empty:
            findings.append(f"Part {part}: no present procedure MAT sample in the source inventory.")
            continue
        procedure_candidates = candidates.loc[
            candidates.relative_path.str.contains("TPG", case=False, na=False)
        ]
        row = (
            procedure_candidates.iloc[0]
            if not procedure_candidates.empty
            else candidates.iloc[0]
        )
        path = _source_path(root, part, row.repository_path)
        source_kind = (
            "procedure" if "TPG" in str(row.relative_path).casefold() else "non-procedure"
        )
        if path is None or not path.is_file():
            findings.append(f"Part {part}: {source_kind} MAT source missing: {row.repository_path}")
            continue
        try:
            variables = whosmat(path)
        except (TypeError, NotImplementedError, OSError, ValueError) as error:
            findings.append(
                f"Part {part}: {source_kind} MAT {row.repository_path} could not be enumerated by scipy.io.whosmat "
                f"({type(error).__name__}: {error})."
            )
        else:
            findings.append(
                f"Part {part}: {source_kind} MAT {row.repository_path} enumerated {len(variables)} MAT variables."
            )
    return findings


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_stage1(
    root: Path | None = None, output_dir: Path | None = None
) -> Path:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    output = (output_dir or _OUTPUT_DIR)
    if not output.is_absolute():
        output = repo / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    mapping = _read_csv(repo, str(_INPUT_DIR / "rpt_mapping_validation.csv"))
    bol_candidates = _read_csv(repo, str(_INPUT_DIR / "bol_reference_verified.csv"))
    soh = _read_csv(repo, str(_INPUT_DIR / "soh_label_comparison.csv"))
    availability = _read_csv(repo, str(_INPUT_DIR / "bafuse_target_availability.csv"))
    split = _read_csv(repo, str(_INPUT_DIR / "cell_split_12train_4val_12external.csv"))
    observations = _read_csv(repo, "data/manifests/bafuse_rpt_observations.csv")
    source_inventory = _read_csv(repo, "data/manifests/source_file_availability.csv")

    priority_scan = _scan_priority_sources(repo)
    mapping = _attach_priority_counts(mapping, priority_scan)
    verified, unresolved = _prepare_mapping(repo, mapping)
    bol = _prepare_bol(repo, bol_candidates, split, mapping, observations)
    target_masks = _prepare_target_masks(availability, soh, bol)
    split_integrity, split_pass = _prepare_split_integrity(split, mapping, target_masks)
    if not split_pass:
        raise ValueError("Stage 1 split integrity failed; no Stage 1 outputs were promoted.")

    soh_comparison = soh.copy()
    soh_comparison["source_exists_actual"] = [
        _source_exists(repo, str(row.dataset_part), str(row.source_file))
        for row in soh_comparison.itertuples()
    ]
    verified_soh = soh_comparison.loc[
        soh_comparison.validation_status_v2.eq("VERIFIED_DERIVED_LABEL")
    ].copy()

    outputs = {
        "stage1_mapping_verified.csv": verified,
        "stage1_mapping_unresolved.csv": unresolved,
        "stage1_bol_reference_audit.csv": bol,
        "stage1_soh_label_comparison.csv": soh_comparison,
        "stage1_soh_verified_candidates.csv": verified_soh,
        "stage1_target_masks.csv": target_masks,
        "stage1_split_integrity.csv": split_integrity,
        "stage1_priority_step_audit.csv": priority_scan,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False, encoding="utf-8-sig")

    mapping_counts = mapping.groupby(["part", "mapping_status_v2"]).size().to_dict()
    soh_counts = soh_comparison.validation_status_v2.value_counts().to_dict()
    soh_audit_counts = soh_comparison.audit_status.value_counts().to_dict()
    soh_review_flags = soh_comparison.validation_status_original.value_counts().to_dict()
    soh_deltas = pd.to_numeric(soh_comparison.soh_abs_delta_pp, errors="coerce").dropna()
    rpt_targets = target_masks.loc[target_masks.observation_type.eq("RPT")]
    target_rows = []
    for partition, group in rpt_targets.groupby("partition"):
        target_rows.append(
            f"| {partition} | {len(group)} | "
            f"{int(group.has_verified_soh.sum())} | "
            f"{int(group.has_verified_lli.sum())} | "
            f"{int(group.has_verified_lam_anode.sum())} | "
            f"{int(group.has_verified_lam_cathode.sum())} |"
        )
    source_status_counts = source_inventory.availability_status.value_counts().to_dict()
    present_mat_counts = source_inventory.loc[
        source_inventory.extension.eq(".mat")
        & source_inventory.availability_status.eq("present")
    ].groupby("part").size().to_dict()
    missing_mat_counts = source_inventory.loc[
        source_inventory.extension.eq(".mat")
        & source_inventory.availability_status.ne("present")
    ].groupby("part").size().to_dict()
    mat_findings = _probe_mat(repo, source_inventory)
    priority_summary = []
    for source_file, group in priority_scan.groupby("source_file", sort=True):
        candidate_counts = group.loc[group.curve_candidate.astype(str).str.casefold().isin(
            {"true", "1"}
        )].groupby("step").size().to_dict()
        all_blocks = group.groupby("step").size().to_dict()
        priority_summary.append(
            f"| `{source_file}` | "
            f"{all_blocks.get(4, 0)}/{candidate_counts.get(4, 0)} | "
            f"{all_blocks.get(6, 0)}/{candidate_counts.get(6, 0)} | "
            f"{all_blocks.get(8, 0)}/{candidate_counts.get(8, 0)} | "
            f"{all_blocks.get(10, 0)}/{candidate_counts.get(10, 0)} |"
        )
    report_lines = [
        "# Stage 1 data quality and SoH validation",
        "",
        "This versioned audit reuses the existing exact-boundary scan and label audit, validates source paths against the current workspace, and keeps unresolved observations quarantined. It does not modify raw files, canonical manifest v3, historical labels, PyDMA outputs, checkpoints, or scalers.",
        "",
        "## Mapping",
        "",
        f"- Candidate RPT observations: {len(mapping)}.",
        f"- Strict verified mapping rows: {len(verified)}.",
        f"- Explicitly unresolved/quarantined rows: {len(unresolved)}.",
        f"- Duplicate `(part, physical_cell_id, RPT)` keys: {int(mapping.duplicated(['part','physical_cell_id','rpt_index']).sum())}.",
        "- Mapping status counts by Part:",
        "",
        "| Part | Status | Rows |",
        "|---:|---|---:|",
    ]
    report_lines.extend(
        f"| {part} | {status} | {count} |"
        for (part, status), count in sorted(mapping_counts.items())
    )
    report_lines.extend(
        [
            "",
            "Part 2 retains 198 non-strictly-verified rows; curve candidates are not promoted to verified RPT identities. The Guide's Group 5 entries differ by physical cell: Cell5 TPG5.9/5.10 is RPT17–24 / 25–41; Cell6 is RPT17–24 / 24–34; Cell16 is RPT17–28 / 29–51. The user-confirmed Cell6 policy uses TPG5.9 RPT17–23 and TPG5.10 RPT24–34, preserving the Guide's RPT24 discrepancy rather than claiming independent verification.",
            "- Guide maps G5-Cell16 TPG5.13 as EoL, while G5-Cell6 TPG5.13 contains Profile + RPT48–66; these files remain distinct sources. G6-Cell1 TPG6.3 is Profile + RPT1 and TPG6.8 contains RPT4 + RPT5.",
            "",
            "Direct raw scans below count structurally eligible Step blocks, not RPT labels. Step 4 candidates require finite capacity/voltage, nonzero capacity span, and negative current; Step 6 candidates require positive current; Step 8/10 candidates additionally require low-rate direction-filtered monotone capacity. No segment is assigned to an RPT by ordinal count alone.",
            "",
            "| Source | Step 4 blocks/candidates | Step 6 blocks/candidates | Step 8 blocks/candidates | Step 10 blocks/candidates |",
            "|---|---:|---:|---:|---:|",
            *priority_summary,
            "",
            "Exact source-row ranges and curve statistics are in `stage1_priority_step_audit.csv`; row references are one-based CSV line numbers including the header (the first data row is line 2). The count mismatches in G5-Cell6 TPG5.9/5.10 and TPG5.13, and the G6 multi-RPT procedures, remain quarantined.",
            "",
            "## BoL references",
            "",
            f"- Physical cells audited: {len(bol)}.",
            f"- Part 1 provisional BoL references with verified Step 4 source/capacity: {int(((bol.dataset_part == '1') & bol.bol_verified).sum())}.",
            f"- Part 2/3 cells with unresolved primary BoL policy: {int(((bol.dataset_part != '1') & bol.bol_policy_status.eq('primary_bol_policy_unresolved')).sum())}.",
            "- Part 1 uses the user-selected earliest complete Step 4 segment in the earliest Guide-mapped BoL procedure; secondary BoL files are retained as QC candidates.",
            "- Part 2 Groups 1–4 continue the same physical cells and use their Part 1 BoL references. The four new Part 2 Group 5–6 cells and all 12 Part 3 cells have unresolved primary BoL policies.",
            "- G10-Cell2 identity is user-confirmed, not independently Guide-verified.",
            "- BoL Step 8/10 availability is explicitly marked `not_audited_for_bol`; the existing BoL audit only verifies Step 4.",
            "- BoL target values 100/0/0/0 in `stage1_target_masks.csv` are reference conventions only, excluded from verified-target masks and training by default.",
            "",
            "## SoH label comparison",
            "",
            f"- Historical audit/comparison rows: {len(soh_comparison)}.",
            f"- Verified-derived Part 1 SoH candidates: {len(verified_soh)}.",
            f"- Source/arithmetic audit states: `{soh_audit_counts}`.",
            f"- Review annotations retained from the label audit: `{ {key: value for key, value in soh_review_flags.items() if str(key).startswith('review_')} }`.",
            f"- Numeric stored-vs-recomputed deltas: {len(soh_deltas)}; nonzero values: {int(soh_deltas.ne(0).sum())}; maximum absolute delta: {float(soh_deltas.abs().max()) if not soh_deltas.empty else 'not available'} percentage points.",
            "- No mismatch is inferred from policy-unresolved, QC/legacy, or unavailable rows; only the original audit's explicit review states are retained.",
            "",
            "| Status | Rows |",
            "|---|---:|",
        ]
    )
    report_lines.extend(f"| {status} | {count} |" for status, count in sorted(soh_counts.items()))
    report_lines.extend(
        [
            "",
            "No dataset-defined official SoH target is established. `Amphr` remains native/unknown absolute units. The 94 promoted candidates are derived same-cell capacity ratios and remain Part 1 only.",
            "",
            "## Independent target masks",
            "",
            "| Partition | RPT rows | Verified SoH | Verified LLI | Verified anode LAM | Verified cathode LAM |",
            "|---|---:|---:|---:|---:|---:|",
            *target_rows,
            "",
            "Historical accepted PyDMA diagnostics are marked as diagnostics, not verified labels. No RPT-level PyDMA fit was reproduced in Stage 1.",
            "",
            "## Split integrity",
            "",
            f"- Version: `{split.split_version.iloc[0] if 'split_version' in split.columns else 'parts12-12-4-plus-part3-12-external-v2'}`.",
            f"- Cell counts: {split.partition.value_counts().to_dict()}.",
            f"- Physical-cell/RPT key overlap: none; Stage 1 split check: {'PASS' if split_pass else 'FAIL'}.",
            "- All 12 Part 3 cells remain External Test; no Internal Test; no split was regenerated.",
            "",
            "## Raw source and MAT scope",
            "",
            f"- Current source inventory availability: {source_status_counts}.",
            f"- MAT files listed present by Part: {present_mat_counts}; absent from the current workspace: {missing_mat_counts}.",
            "- The repository dataset review records 957 CSV headers checked, but only 180 CSVs value-parsed in the prior chunk audit; this report does not claim complete row-level validation.",
            *[f"- {finding}" for finding in mat_findings],
            "- Guide evidence was read from the original Part 1–3 XLSX/PDF files. Procedure MAT samples could not be fully enumerated by `scipy.io.whosmat`; MAT contents are not used to promote labels.",
            "",
            "## Remaining blockers",
            "",
            "- 227 mapping/curve rows remain unresolved or user-confirmed-only and are listed in `stage1_mapping_unresolved.csv`.",
            "- Part 2 Groups 5–6 and Part 3 primary BoL policies remain unresolved; Step 8/10 BoL curves were not audited across all cells.",
            "- Historical PyDMA negative estimates remain unclassified and are not clipped.",
            "- Multi-target degradation training and charge/physics fusion are not ready; no model training was run.",
            "",
            "Stage 1 status: **audited with unresolved cases quarantined; not all data modalities or labels are verified**.",
        ]
    )
    report_path = output / "stage1_quality_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    inputs = [
        str(_INPUT_DIR / name)
        for name in (
            "rpt_mapping_validation.csv",
            "bol_reference_verified.csv",
            "soh_label_comparison.csv",
            "bafuse_target_availability.csv",
            "cell_split_12train_4val_12external.csv",
        )
    ] + [
        "data/manifests/bafuse_rpt_observations.csv",
        "data/manifests/source_file_availability.csv",
        "data/manifests/canonical_observation_manifest_v3.csv",
        "src/bafuse/data/stage_validation.py",
        "experiments/dataset_review/README.md",
        "data/raw/oxford/part1/Guide_to_Datafiles.xlsx",
        "data/raw/oxford/part1/Guide_to_Datafiles (1).pdf",
        "data/raw/oxford/part2/Guide_to_Datafiles_2.xlsx",
        "data/raw/oxford/part2/Guide_to_Datafiles_2.pdf",
        "data/raw/oxford/part3/Guide_to_Datafiles_3.xlsx",
        "data/raw/oxford/part3/Guide_to_Datafiles_3.pdf",
    ]
    input_hashes = {
        name: _sha256(repo / name)
        for name in inputs
        if (repo / name).is_file()
    }
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "stage1_run_manifest.json"
    }
    manifest = {
        "status": "stage1_audited_unresolved_rows_quarantined",
        "split_version": "parts12-12-4-plus-part3-12-external-v2",
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "source_path_checks": {
            "mapping_verified_rows": len(verified),
            "mapping_unresolved_rows": len(unresolved),
            "bol_physical_cells": len(bol),
        },
        "priority_raw_scans": [
            {
                "source_file": source_file,
                "scan_status": str(group.scan_status.iloc[0]),
                "size_bytes": (
                    (repo / source_file).stat().st_size
                    if (repo / source_file).is_file()
                    else None
                ),
                "step_blocks": {
                    str(step): int(count)
                    for step, count in group.groupby("step").size().items()
                    if str(step).strip()
                },
            }
            for source_file, group in priority_scan.groupby("source_file", sort=True)
        ],
        "limitations": [
            "No full-row audit of all 957 CSVs is claimed.",
            "BoL Step 8/10 and the Part 2 Groups 5-6 / Part 3 baseline policies remain unresolved.",
            "Historical PyDMA values are diagnostics, not verified degradation targets.",
            "No machine-learning model was trained.",
        ],
    }
    (output / "stage1_run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate versioned BaFuse Stage 1 validation outputs.")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    print(generate_stage1(args.root, args.output_dir))


if __name__ == "__main__":
    main()
