from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import pandas as pd

from .ambiguous_rpt_resolution import analyze_rpt_blocks, map_ordered_blocks, scan_raw_tpg
from .rpt_mapping_audit import _half_cell_reference_audit, _source_paths, parse_rpt_stage


_OBSERVATIONS = Path("data/manifests/bafuse_rpt_observations.csv")
_CANONICAL = Path("data/manifests/canonical_observation_manifest_v3.csv")
_LOCKED_SPLIT = Path("data/manifests/experiment_manifests/part1_locked_split/split_manifest.csv")
_SOH_TABLES = (
    ("part1_earliest_bol", Path("data/processed/label_soh_part1_2/run_006_part1_soh_labels_earliest_bol/labels_soh.csv"), "1"),
    ("part1_capacity_qc", Path("data/processed/label_soh_part1_2/run_007_part1_capacity_qc/labels_soh.csv"), "1"),
    ("part1_legacy", Path("data/processed/label_soh_part1_2/labeling_output_part1_rmse15/labels_soh.csv"), "1"),
    ("part2_candidates", Path("data/processed/label_soh_part1_2/run_010_part2_soh_candidates/labels_soh_part2.csv"), "2"),
    ("part3_candidates", Path("experiments/results/data_audits/part3_soh_candidates/labels_soh_candidates.csv"), "3"),
)
_PYDMA_TABLES = (
    ("part1_part2_rmse15", Path("data/processed/label_pydma_fit_part1_2/pydma_results_rmse15.csv"), "", 15.0),
    ("part1_part2_pre_append_snapshot", Path("data/processed/label_pydma_fit_part1_2/pydma_results_rmse15_pre_g5_candidate_append_20261105.csv"), "", 15.0),
    ("part2_rmse10_review", Path("experiments/results/data_audits/part2_pydma_review/pydma_results.csv"), "2", 10.0),
    ("part2_g5_audit_candidates", Path("experiments/results/data_audits/part2_g5_candidate_pydma_audit_20261105_193440/candidate_pydma_fits.csv"), "2", 15.0),
    ("part3_rmse15", Path("data/processed/label_pydma_fit_part3/run_009_part3_pydma/pydma_results.csv"), "3", 15.0),
)
_FEATURES = Path("data/processed/features/part1_expanded_same_time_baselines/features.csv")
_SUPPLEMENTAL_FEATURES = Path("experiments/results/data_audits/part2_g5_g6_soh_extension/feature_run/features.csv")
_PREDICTIONS = Path("experiments/results/benchmarks/part1_cnn1d_baseline/predictions.csv")
_CHECKPOINT = Path("models/checkpoints/part1_cnn1d/cnn1d_checkpoint.pt")
_MODEL_CONFIG = Path("experiments/results/benchmarks/part1_cnn1d_baseline/run_config.json")
_MODEL_INPUT_FILES = {
    "split": Path("data/manifests/splits/parts12-10-3-3-plus-part3-external-v1.csv"),
    "part1_labels": Path("data/processed/label_soh_part1_2/run_007_part1_capacity_qc/labels_soh.csv"),
    "part2_labels": Path("data/processed/label_soh_part1_2/run_010_part2_soh_candidates/labels_soh_part2.csv"),
    "part3_labels": Path("experiments/results/data_audits/part3_soh_candidates/labels_soh_candidates.csv"),
    "features.csv": Path("data/processed/features/part1_expanded_same_time_baselines/features.csv"),
    "sample_manifest.csv": Path("data/processed/features/part1_expanded_same_time_baselines/sample_manifest.csv"),
    "feature_run_metrics.json": Path("data/processed/features/part1_expanded_same_time_baselines/metrics.json"),
}
_REPORT_DIR = Path("reports/final_mapping_label_audit")
_DEGRADATION_COLUMNS = (
    "LLI_percent_vs_BoL",
    "LAM_anode_percent_vs_BoL",
    "LAM_cathode_percent_vs_BoL",
)


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    number = _number(value)
    return int(number) if number is not None and number.is_integer() else None


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _part_number(value: Any) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def _physical_cell(group: Any, cell: Any) -> str:
    group_number = _part_number(group)
    cell_number = _part_number(cell)
    return f"G{group_number}-Cell{cell_number}" if group_number and cell_number else ""


def expected_rpts(part: str, cell: str, tpg: str, stage: str) -> list[int]:
    if part == "2" and cell == "G5-Cell6" and tpg == "TPG5.9":
        return list(range(17, 24))
    if part == "2" and cell == "G5-Cell6" and tpg == "TPG5.10":
        return list(range(24, 35))
    return parse_rpt_stage(stage)[0]


def is_user_policy_mapping(part: str, cell: str, tpg: str, rpt: int) -> bool:
    return (
        part == "2"
        and cell == "G5-Cell6"
        and ((tpg == "TPG5.9" and 17 <= rpt <= 23) or (tpg == "TPG5.10" and rpt == 24))
    )


def passes_fit_gate(
    accepted: bool,
    rmse_mV: float | None,
    threshold_mV: float,
    accepted_runs: int | None,
) -> bool:
    return bool(
        accepted
        and rmse_mV is not None
        and rmse_mV < threshold_mV
        and accepted_runs is not None
        and accepted_runs >= 3
    )


def _resolve_source(root: Path, part: str, value: str, search_all_parts: bool = False) -> Path | None:
    normalized = str(value or "").strip().replace("\\", "/")
    if not normalized:
        return None
    win_path = PureWindowsPath(normalized)
    if win_path.is_absolute():
        candidate = Path(normalized)
        return candidate.resolve() if candidate.is_file() else None
    prefix = re.match(r"^data/raw/oxford/part([123])/(.*)$", normalized, re.IGNORECASE)
    if prefix:
        part, normalized = prefix.groups()
    normalized = normalized.lstrip("/")
    parts = ("1", "2", "3") if search_all_parts else (part,)
    for candidate_part in parts:
        candidate = root / "data" / "raw" / "oxford" / f"part{candidate_part}" / Path(normalized)
        if candidate.is_file():
            return candidate.resolve()
    basename = Path(normalized).name.casefold()
    matches: list[Path] = []
    for candidate_part in parts:
        part_root = root / "data" / "raw" / "oxford" / f"part{candidate_part}"
        if part_root.is_dir():
            matches.extend(
                path.resolve()
                for path in part_root.rglob(Path(normalized).name)
                if path.is_file() and path.name.casefold() == basename
            )
    unique = sorted(set(matches))
    return unique[0] if len(unique) == 1 else None


def _run_is_valid(run: Any) -> bool:
    return bool(
        run is not None
        and any(bool(segment.get("valid")) for segment in run.curve_segments)
    )


def _run_ref(path: str, run: Any) -> str:
    if run is None:
        return ""
    return f"{path}:rows{run.source_row_start}-{run.source_row_end}"


def _step_runs(scan: Any, step: int) -> list[Any]:
    return [run for run in scan.runs if run.step == step] if scan else []


def _source_inventory(
    root: Path,
    observations: pd.DataFrame,
    soh_tables: list[tuple[str, pd.DataFrame, str]],
) -> tuple[dict[str, Path], dict[str, Any]]:
    source_paths: dict[str, Path] = {}
    source_errors: dict[str, Any] = {}

    def add(part: str, value: str, label: str) -> None:
        if not str(value or "").strip():
            return
        path = _resolve_source(root, part, value, search_all_parts=label.startswith("baseline"))
        if path is None:
            source_errors[f"{label}|{part}|{value}"] = "missing_or_ambiguous_source"
            return
        source_paths[str(path).casefold()] = path

    for row in observations.to_dict("records"):
        add(str(row.get("part", "")), str(row.get("source_file", "")), "observation")
        add(str(row.get("part", "")), str(row.get("bol_reference_file", "")), "baseline")
    for _, frame, default_part in soh_tables:
        for row in frame.to_dict("records"):
            part = _part_number(row.get("dataset_part")) or default_part
            add(part, str(row.get("source_file", "")), "label_source")
            add(part, str(row.get("baseline_source_file", "")), "baseline_label_source")
    for _, path, default_part, _ in _PYDMA_TABLES:
        if not (root / path).is_file():
            continue
        for row in _read_csv(root / path).to_dict("records"):
            part = _part_number(row.get("source_part")) or default_part
            group = _part_number(row.get("group_id"))
            cell = _part_number(row.get("cell_id")) or _part_number(row.get("physical_cell_id"))
            source = str(row.get("source_file", ""))
            is_part2_bol = (
                part == "2" and str(row.get("checkup", "")).strip().casefold() == "bol"
            )
            resolved = _fit_source_path(
                root, part, source, group, cell, search_all_parts=is_part2_bol
            )
            if resolved is None:
                source_errors[f"pydma|{part}|{source}"] = "missing_or_ambiguous_source"
            else:
                source_paths[str(resolved).casefold()] = resolved
    canonical = _read_csv(root / _CANONICAL)
    for row in canonical.to_dict("records"):
        if row.get("record_type") != "cycling_guide_record":
            continue
        part = str(row.get("dataset_part", "")).split(";")[0]
        for source in _source_paths(str(row.get("source_paths", ""))):
            if source.casefold().endswith(".csv"):
                add(part, source, "canonical_source")
    return source_paths, source_errors


def _scan_sources(paths: dict[str, Path], chunk_rows: int) -> dict[str, Any]:
    scans: dict[str, Any] = {}
    for index, (key, path) in enumerate(sorted(paths.items()), start=1):
        before = path.stat()
        scan = scan_raw_tpg(path, chunk_rows=chunk_rows)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Raw source changed while being audited: {path}")
        scans[key] = scan
        if index % 20 == 0 or index == len(paths):
            print(f"Read {index}/{len(paths)} unique raw source files", flush=True)
    return scans


def _scan_for(path: Path | None, scans: dict[str, Any]) -> Any:
    return scans.get(str(path).casefold()) if path is not None else None


def _mapping_audit(root: Path, observations: pd.DataFrame, scans: dict[str, Any]) -> pd.DataFrame:
    grouped: dict[tuple[str, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in observations.to_dict("records"):
        grouped[
            (
                str(row.get("part", "")),
                str(row.get("physical_cell_id", "")),
                str(row.get("source_tpg", "")),
                str(row.get("source_file", "")),
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for (part, cell, tpg, source), rows in sorted(grouped.items()):
        rows.sort(key=lambda row: _integer(row.get("rpt_index")) or 0)
        stage = rows[0].get("guide_stage", "")
        candidates = expected_rpts(part, cell, tpg, stage)
        if not candidates:
            candidates = sorted({_integer(row.get("rpt_index")) for row in rows if _integer(row.get("rpt_index")) is not None})
        path = _resolve_source(root, part, source)
        scan = _scan_for(path, scans)
        blocks = analyze_rpt_blocks(scan.runs) if scan and scan.status == "scanned" else []
        core_blocks = [block for block in blocks if block["core_sequence_verified"]]
        ordered = map_ordered_blocks(blocks, candidates) if candidates else None
        block_by_rpt = dict(ordered or [])
        by_rpt: dict[int, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            rpt = _integer(row.get("rpt_index"))
            if rpt is not None:
                by_rpt[rpt].append(row)
        row_numbers = sorted(set(candidates) | set(by_rpt))
        for rpt in row_numbers:
            original = by_rpt.get(rpt, [{}])[0]
            if not original:
                template = rows[0]
                original = {
                    "group": template.get("group", ""),
                    "split": template.get("split", ""),
                    "mapping_status": "not_listed_in_existing_observation_table",
                }
            block = block_by_rpt.get(rpt)
            original_status = original.get("mapping_status", "")
            if path is None:
                final_status = "missing_source"
            elif (
                original_status == "GUIDE_VERIFIED"
                and part == "3"
                and cell == "G10-Cell2"
                and block is not None
            ):
                final_status = "GUIDE_VERIFIED"
            elif original_status == "verified":
                final_status = "verified"
            elif original_status == "user_confirmed_cell_identity":
                final_status = "user_confirmed_identity"
            elif is_user_policy_mapping(part, cell, tpg, rpt) and block is not None:
                final_status = "user_policy_confirmed"
            elif block is not None:
                final_status = "verified_from_ordered_raw_blocks"
            elif original_status == "missing_step":
                final_status = "missing_step"
            elif original_status == "needs_curve_audit":
                final_status = "needs_curve_audit"
            else:
                final_status = "ambiguous_rpt"

            if block is None and len(candidates) == 1:
                step4_runs = _step_runs(scan, 4)
                if len(step4_runs) == 1 and final_status in {
                    "verified",
                    "user_confirmed_identity",
                    "needs_curve_audit",
                }:
                    block = {
                        "block_index": 1,
                        "source_row_start": step4_runs[0].source_row_start,
                        "source_row_end": step4_runs[0].source_row_end,
                        "step4": step4_runs[0],
                        "step8": None,
                        "step10": None,
                    }

            step4 = block.get("step4") if block else None
            step8 = block.get("step8") if block else None
            step10 = block.get("step10") if block else None
            step4_valid = _run_is_valid(step4)
            step8_valid = _run_is_valid(step8)
            step10_valid = _run_is_valid(step10)
            user_identity = ""
            if part == "1" and cell == "G3-Cell14" and tpg in {"TPG3.17", "TPG3.18"}:
                user_identity = "user_confirmed_G3_Cell14"
            elif part == "3" and cell == "G10-Cell2":
                user_identity = (
                    "Group 10 Cell 2 identity confirmed by project owner; "
                    "explicit Cell 2 label omitted from printed Guide."
                )
            notes = []
            if final_status == "GUIDE_VERIFIED":
                notes.append(
                    "Group 10 Guide sequence and source-file consistency were verified; "
                    "the Guide omits the explicit Cell 2 label."
                )
            if final_status == "user_policy_confirmed":
                notes.append("RPT17-RPT23 / RPT24-RPT34 split is user-confirmed; TPG5.9 Guide also lists RPT24.")
            if rpt not in by_rpt and block is not None:
                notes.append(
                    "Audit-only row expanded from explicit Guide range and ordered raw blocks; "
                    "not added to the existing observation table."
                )
            if not candidates:
                notes.append("Guide stage has no explicit RPT index; retained existing candidate only.")
            if candidates and len(core_blocks) != len(candidates):
                notes.append(f"Raw complete protocol blocks={len(core_blocks)}; Guide candidates={len(candidates)}; no ordinal mapping assigned.")
            if scan is None:
                notes.append("Raw source was not scanned.")
            elif scan.status != "scanned":
                notes.append(f"Raw scan status={scan.status}: {scan.error}")
            output.append(
                {
                    "part": part,
                    "group": str(original.get("group", "")),
                    "physical_cell_id": cell,
                    "rpt_index": rpt,
                    "source_tpg": tpg,
                    "source_file": source,
                    "guide_stage": stage,
                    "existing_mapping_status": original_status,
                    "final_mapping_status": final_status,
                    "split": original.get("split", ""),
                    "source_exists": path is not None,
                    "source_scan_status": scan.status if scan else "not_scanned",
                    "source_rows_scanned": scan.rows_scanned if scan else "",
                    "raw_core_block_count": len(core_blocks),
                    "guide_rpt_candidate_count": len(candidates),
                    "step4_segment_reference": _run_ref(source, step4),
                    "step4_valid": step4_valid,
                    "step8_segment_reference": _run_ref(source, step8),
                    "step8_valid": step8_valid,
                    "step10_segment_reference": _run_ref(source, step10),
                    "step10_valid": step10_valid,
                    "identity_provenance": user_identity,
                    "ready_soh_candidate": bool(
                        step4_valid
                        and final_status
                        in {
                            "verified",
                            "verified_from_ordered_raw_blocks",
                            "user_policy_confirmed",
                            "user_confirmed_identity",
                            "GUIDE_VERIFIED",
                        }
                    ),
                    "ready_degradation_candidate": bool(
                        step8_valid
                        and step10_valid
                        and final_status
                        in {
                            "verified",
                            "verified_from_ordered_raw_blocks",
                            "user_policy_confirmed",
                            "user_confirmed_identity",
                            "GUIDE_VERIFIED",
                        }
                    ),
                    "audit_notes": " ".join(notes),
                }
            )
    frame = pd.DataFrame(output)
    duplicate = frame.duplicated(["part", "physical_cell_id", "rpt_index"], keep=False)
    frame.loc[duplicate, "audit_notes"] = frame.loc[duplicate, "audit_notes"].map(
        lambda value: (str(value) + " Duplicate physical-cell/part/RPT key.").strip()
    )
    return frame


def _step4_capacity(scan: Any, segment_index: int | None) -> float | None:
    if scan is None or scan.status != "scanned":
        return None
    segments = _step_runs(scan, 4)
    index = (segment_index or 1) - 1
    if index < 0 or index >= len(segments):
        return None
    run = segments[index]
    if run.capacity_min is None or run.capacity_max is None:
        return None
    return float(run.capacity_max - run.capacity_min)


def _soh_audit(
    root: Path,
    tables: list[tuple[str, pd.DataFrame, str]],
    scans: dict[str, Any],
    observations: pd.DataFrame,
) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    observation_sources: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for observation in observations.to_dict("records"):
        rpt = _integer(observation.get("rpt_index"))
        source = str(observation.get("source_file", "")).strip()
        if rpt is not None and source:
            observation_sources[
                (
                    str(observation.get("part", "")),
                    str(observation.get("physical_cell_id", "")),
                    rpt,
                )
            ].add(source)
    for table_name, frame, default_part in tables:
        for row in frame.to_dict("records"):
            part = _part_number(row.get("dataset_part")) or default_part
            source = str(row.get("source_file", "")).strip()
            source_resolution = "label_table_source" if source else "unresolved"
            measurement = str(row.get("measurement_id", ""))
            rpt_match = re.search(r"\bRPT\s*0*(\d+)\b", measurement, flags=re.IGNORECASE)
            rpt = int(rpt_match.group(1)) if rpt_match else _integer(measurement)
            if not source and rpt is not None:
                candidates = observation_sources.get(
                    (part, str(row.get("physical_cell_id", "")), rpt),
                    set(),
                )
                if len(candidates) == 1:
                    source = next(iter(candidates))
                    source_resolution = "unique_cell_rpt_observation_match"
                elif len(candidates) > 1:
                    source_resolution = "ambiguous_cell_rpt_observation_match"
            baseline = str(row.get("baseline_source_file", ""))
            source_path = _resolve_source(root, part, source)
            baseline_part = part
            if baseline.startswith("data/raw/oxford/part"):
                match = re.match(r"data/raw/oxford/part([123])/", baseline.replace("\\", "/"))
                if match:
                    baseline_part = match.group(1)
            elif part == "2" and _integer(row.get("group_id")) in {1, 2, 3, 4}:
                baseline_part = "1"
            baseline_path = _resolve_source(root, baseline_part, baseline, search_all_parts=True)
            segment_index = _integer(row.get("step4_segment_index")) or 1
            baseline_segment_index = _integer(row.get("baseline_segment_index")) or 1
            capacity = _step4_capacity(_scan_for(source_path, scans), segment_index)
            baseline_capacity = _step4_capacity(_scan_for(baseline_path, scans), baseline_segment_index)
            stored_capacity = _number(row.get("capacity_raw"))
            stored_reference = _number(row.get("reference_capacity_raw"))
            stored_soh = _number(row.get("soh_percent"))
            recalculated = (
                100.0 * capacity / baseline_capacity
                if capacity is not None and baseline_capacity not in (None, 0)
                else None
            )
            status = "verified_source_and_arithmetic"
            if source_path is None:
                status = "source_missing_or_ambiguous"
            elif capacity is None:
                status = "step4_segment_unavailable"
            elif stored_capacity is not None and abs(stored_capacity - capacity) > 0.02:
                status = "capacity_span_mismatch"
            elif baseline and baseline_path is None:
                status = "baseline_source_missing_or_ambiguous"
            elif baseline_capacity is None and stored_soh is not None:
                status = "baseline_step4_unavailable"
            elif stored_reference is not None and baseline_capacity is not None and abs(stored_reference - baseline_capacity) > 0.02:
                status = "baseline_capacity_mismatch"
            elif stored_soh is not None and recalculated is not None and abs(stored_soh - recalculated) > 0.02:
                status = "soh_arithmetic_mismatch"
            elif stored_soh is not None and not str(row.get("capacity_unit", "")).casefold().find("unverified") >= 0:
                status = "unit_semantics_require_review"
            output.append(
                {
                    "label_table": table_name,
                    "dataset_part": part,
                    "physical_cell_id": row.get("physical_cell_id", ""),
                    "measurement_id": row.get("measurement_id", ""),
                    "validation_status_original": row.get("validation_status", ""),
                    "source_file": source,
                    "source_resolution": source_resolution,
                    "source_exists": source_path is not None,
                    "step4_segment_index": segment_index,
                    "raw_capacity_span_native": capacity,
                    "stored_capacity_raw": stored_capacity,
                    "capacity_abs_delta": abs(stored_capacity - capacity) if stored_capacity is not None and capacity is not None else "",
                    "capacity_unit": row.get("capacity_unit", ""),
                    "baseline_source_file": baseline,
                    "baseline_source_exists": baseline_path is not None if baseline else "",
                    "baseline_step4_segment_index": baseline_segment_index,
                    "raw_baseline_capacity_native": baseline_capacity,
                    "stored_reference_capacity_raw": stored_reference,
                    "recomputed_soh_percent": recalculated,
                    "stored_soh_percent": stored_soh,
                    "soh_abs_delta_pp": abs(stored_soh - recalculated) if stored_soh is not None and recalculated is not None else "",
                    "audit_status": status,
                    "label_is_official_ground_truth": False,
                    "audit_note": (
                        "Capacity is native Amphr with unverified absolute units; same-cell ratio does not establish an official SOH label."
                        + (
                            " Source file resolved from a unique cell/RPT observation mapping."
                            if source_resolution == "unique_cell_rpt_observation_match"
                            else ""
                        )
                    ),
                }
            )
    return pd.DataFrame(output)


def _fit_source_path(
    root: Path,
    part: str,
    source: str,
    group: str,
    cell: str,
    *,
    search_all_parts: bool = False,
) -> Path | None:
    if not search_all_parts:
        path = _resolve_source(root, part, source)
        if path is not None:
            return path
    filename = Path(source.replace("\\", "/")).name
    if not filename:
        return None
    parts = (
        ("1", "2", "3")
        if search_all_parts or part not in {"1", "2", "3"}
        else (part,)
    )
    candidates: list[Path] = []
    for candidate_part in parts:
        part_root = root / "data" / "raw" / "oxford" / f"part{candidate_part}"
        if not part_root.is_dir():
            continue
        candidates.extend(
            item.resolve()
            for item in part_root.rglob(filename)
            if item.is_file()
            and (
                not group
                or f"Group {group}".casefold() in str(item.parent).casefold()
                or f"Group_{group}".casefold() in str(item.parent).casefold()
            )
            and (
                not cell
                or re.search(rf"Cell\s*{re.escape(cell)}(?!\d)", item.name, re.IGNORECASE)
            )
        )
    return candidates[0] if len(set(candidates)) == 1 else None


def _curve_segment_count(value: str) -> int:
    try:
        audit = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return 0
    if not isinstance(audit, dict):
        return 0
    segments = audit.get("segments", [])
    return len(segments) if isinstance(segments, list) else 0


def _pydma_audit(root: Path, scans: dict[str, Any]) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for table_name, path, default_part, default_threshold in _PYDMA_TABLES:
        if not path.is_file():
            continue
        frame = _read_csv(root / path)
        raw_rows = frame.to_dict("records")
        key_counts: Counter[tuple[str, str, str, str, str]] = Counter()
        normalized: list[dict[str, Any]] = []
        for row in raw_rows:
            part = _part_number(row.get("source_part")) or default_part
            group = _part_number(row.get("group_id"))
            cell_id = _part_number(row.get("cell_id")) or _part_number(row.get("physical_cell_id"))
            physical = row.get("physical_cell_id") or _physical_cell(group, cell_id)
            checkup = row.get("checkup") or f"RPT{row.get('candidate_rpt', '')}"
            direction = str(row.get("direction", "")).strip().casefold()
            source = str(row.get("source_file", ""))
            key_counts[(part, physical, checkup.casefold(), direction, source.casefold())] += 1
            normalized.append((row, part, group, cell_id, physical, checkup, direction, source))
        for row, part, group, cell_id, physical, checkup, direction, source in normalized:
            threshold = _number(row.get("fit_threshold_mV")) or default_threshold
            if table_name == "part2_rmse10_review":
                threshold = 10.0
            rmse = _number(row.get("fit_rmse_mV"))
            runs = _integer(row.get("accepted_runs"))
            accepted_flag = _truth(
                row.get("fit_accepted_by_pydma_15mV")
                or row.get("fit_accepted_by_pydma_10mV")
                or row.get("fit_is_accepted")
                or row.get("passes_strict_rmse15_gate")
            )
            fit_status = str(row.get("diagnostic_label_status") or row.get("audit_status") or row.get("candidate_status") or "")
            if "accepted" in fit_status.casefold() and not accepted_flag:
                accepted_flag = True
            gate_pass = passes_fit_gate(accepted_flag, rmse, threshold, runs)
            expected_step = {"charge": 10, "discharge": 8}.get(direction)
            test_step = _integer(row.get("test_step"))
            if test_step is None and table_name == "part2_g5_audit_candidates":
                test_step = _integer(row.get("test_step"))
            is_part2_bol = part == "2" and checkup.casefold() == "bol"
            source_path = _fit_source_path(
                root, part, source, group, cell_id, search_all_parts=is_part2_bol
            )
            scan = _scan_for(source_path, scans)
            values = [_number(row.get(column)) for column in _DEGRADATION_COLUMNS]
            duplicate_key = (part, physical, checkup.casefold(), direction, source.casefold())
            output.append(
                {
                    "result_table": table_name,
                    "dataset_part": part,
                    "physical_cell_id": physical,
                    "checkup": checkup,
                    "direction": direction,
                    "source_file": source,
                    "source_exists": source_path is not None,
                    "test_step": test_step,
                    "expected_test_step": expected_step or "",
                    "step_direction_consistent": expected_step is not None and test_step == expected_step,
                    "fit_rmse_mV": rmse,
                    "rmse_threshold_mV": threshold,
                    "accepted_runs": runs,
                    "total_runs": _integer(row.get("total_runs")),
                    "fit_accepted_flag": accepted_flag,
                    "passes_recorded_gate": gate_pass,
                    "fit_status_original": fit_status,
                    "diagnostic_label_accepted": fit_status == "diagnostic_estimate_accepted",
                    "has_three_numeric_outputs": all(value is not None for value in values),
                    "lli_percent": values[0],
                    "lam_anode_percent": values[1],
                    "lam_cathode_percent": values[2],
                    "curve_segment_count_in_audit": _curve_segment_count(str(row.get("curve_audit_json", ""))),
                    "source_scan_status": scan.status if scan else "not_scanned",
                    "duplicate_key_count_in_table": key_counts[duplicate_key],
                    "candidate_identity_status": row.get("identity_status", ""),
                    "is_training_label": row.get("is_training_label", ""),
                    "audit_note": (
                        "PyDMA estimates are model-derived diagnostics, not independent ground truth. "
                        + ("Audit-only candidate; not a training label. " if table_name == "part2_g5_audit_candidates" else "")
                        + ("Multiple curve segments require RPT-alignment review. " if _curve_segment_count(str(row.get("curve_audit_json", ""))) > 1 else "")
                    ),
                }
            )
    return pd.DataFrame(output)


def _compare_locked_split(joined: pd.DataFrame, locked_path: Path) -> dict[str, Any]:
    if not locked_path.is_file():
        return {"status": "missing", "path": str(locked_path)}
    locked = _read_csv(locked_path)
    locked_part1 = locked[["physical_cell_id", "split"]].copy()
    locked_part1["expected_partition"] = locked_part1["split"].replace({"test": "internal_test"})
    current_part1 = joined.loc[joined["dataset_part"] == "1"].groupby(
        "physical_cell_id", as_index=False
    )["partition"].first()
    comparison = locked_part1.merge(
        current_part1, on="physical_cell_id", how="left", validate="one_to_one"
    )
    comparison["matches_locked_split"] = (
        comparison["partition"] == comparison["expected_partition"]
    )
    return {
        "status": "compared",
        "split_version": str(locked["split_version"].iloc[0]) if not locked.empty else "",
        "locked_cell_count": len(comparison),
        "matched_cell_count": int(comparison["matches_locked_split"].sum()),
        "mismatched_cell_count": int((~comparison["matches_locked_split"]).sum()),
        "mismatches": [
            {
                "physical_cell_id": row["physical_cell_id"],
                "locked_partition": row["expected_partition"],
                "baseline_partition": (
                    str(row["partition"]) if pd.notna(row["partition"]) else "not_present"
                ),
            }
            for row in comparison.loc[~comparison["matches_locked_split"]].to_dict("records")
        ],
    }


def _partition_split_audit(joined: pd.DataFrame) -> dict[str, Any]:
    scoped = joined[["dataset_part", "physical_cell_id", "partition"]].copy()
    scoped["split_scope_cell"] = scoped["dataset_part"].map(
        lambda part: "parts12" if str(part) in {"1", "2"} else f"part{part}"
    ) + ":" + scoped["physical_cell_id"].astype(str)
    partitions = scoped.groupby("split_scope_cell")["partition"].agg(
        lambda values: sorted(set(values.astype(str)))
    )
    mismatches = [
        {"split_scope_cell": key, "partitions": values}
        for key, values in partitions.items()
        if len(values) > 1
    ]
    return {"consistent": not mismatches, "mismatches": mismatches}


def _model_audit(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    features_path, predictions_path = root / _FEATURES, root / _PREDICTIONS
    if not features_path.is_file() or not predictions_path.is_file():
        return pd.DataFrame(), {"status": "missing_feature_or_prediction_artifact"}
    features = _read_csv(features_path)
    predictions = _read_csv(predictions_path)
    key = ["dataset_part", "physical_cell_id", "measurement_id"]
    supplemental_path = root / _SUPPLEMENTAL_FEATURES
    supplemental = _read_csv(supplemental_path) if supplemental_path.is_file() else pd.DataFrame()
    for frame in (features, predictions, supplemental):
        if frame.empty:
            continue
        frame["dataset_part"] = frame["dataset_part"].astype(str)
    if (
        features.duplicated(key).any()
        or predictions.duplicated(key).any()
        or (not supplemental.empty and supplemental.duplicated(key).any())
    ):
        raise ValueError("CNN feature or prediction table has duplicate cell/RPT keys.")
    joined = features.merge(predictions[key + ["partition", "soh_percent", "cnn1d_prediction_percent", "train_cell_mean_prediction_percent"]], on=key, how="inner", validate="one_to_one")
    if supplemental.empty:
        supplemental_summary: dict[str, Any] = {"status": "unavailable"}
    else:
        supplemental_only = supplemental[key].merge(
            features[key], on=key, how="left", indicator=True, validate="one_to_one"
        )
        supplemental_only = supplemental_only.loc[
            supplemental_only["_merge"] == "left_only"
        ].drop(columns="_merge")
        supplemental_summary = {
            "status": "audited",
            "feature_rows": len(supplemental),
            "rows_not_in_saved_baseline_features": len(supplemental_only),
            "cells_not_in_saved_baseline_features": sorted(
                supplemental_only["physical_cell_id"].unique().tolist()
            ),
            "parts_not_in_saved_baseline_features": sorted(
                supplemental_only["dataset_part"].unique().tolist()
            ),
        }
    feature_columns = [column for column in features.columns if column.startswith("voltage_progress_")]
    unexpected_inputs = [column for column in features.columns if column not in feature_columns and column not in key]
    partition_split_audit = _partition_split_audit(joined)
    split_consistent = partition_split_audit["consistent"]
    external_only = bool((joined.loc[joined["dataset_part"] == "3", "partition"] == "external_test").all())
    train_rows = joined.loc[joined["partition"] == "train"].copy()
    train_rows["soh_percent"] = pd.to_numeric(train_rows["soh_percent"], errors="raise")
    train_cell_macro_mean = float(
        train_rows.groupby("physical_cell_id")["soh_percent"].mean().mean()
    )
    saved_baseline_values = sorted(
        pd.to_numeric(joined["train_cell_mean_prediction_percent"], errors="raise").unique()
    )
    baseline_matches_train = bool(saved_baseline_values) and all(
        np.isclose(value, train_cell_macro_mean, rtol=0.0, atol=1e-4)
        for value in saved_baseline_values
    )
    locked_split_comparison = _compare_locked_split(joined, root / _LOCKED_SPLIT)
    historical_locked_split_used = (
        locked_split_comparison.get("status") == "compared"
        and locked_split_comparison["mismatched_cell_count"] == 0
    )
    model_input_hash_audit = _model_input_hash_audit(root)
    split_warning = (
        "Saved CNN baseline does not exactly match the historical locked split. "
        + (
            f"{locked_split_comparison['mismatched_cell_count']} of "
            f"{locked_split_comparison['locked_cell_count']} locked cells differ."
            if locked_split_comparison.get("status") == "compared"
            else "The locked split could not be compared."
        )
        + " Part 3 metrics are candidate-label external results."
    )
    checkpoint_result: dict[str, Any] = {"status": "unavailable"}
    try:
        import torch

        payload = torch.load(root / _CHECKPOINT, map_location="cpu", weights_only=True)
        train = joined.loc[joined["partition"] == "train"]
        x = train[feature_columns].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
        y = pd.to_numeric(train["soh_percent"], errors="raise").to_numpy(dtype=float)
        x_mean_delta = float(np.max(np.abs(x.mean(axis=0) - payload["x_mean"].cpu().numpy())))
        x_scale_delta = float(np.max(np.abs(x.std(axis=0) - payload["x_scale"].cpu().numpy())))
        y_mean_delta = abs(float(y.mean()) - float(payload["y_mean"]))
        y_scale_delta = abs(float(y.std()) - float(payload["y_scale"]))
        checkpoint_result = {
            "status": "verified",
            "x_mean_max_abs_delta_vs_train": x_mean_delta,
            "x_scale_max_abs_delta_vs_train": x_scale_delta,
            "y_mean_abs_delta_vs_train": y_mean_delta,
            "y_scale_abs_delta_vs_train": y_scale_delta,
            "scalers_match_train_only": max(x_mean_delta, x_scale_delta, y_mean_delta, y_scale_delta) < 1e-4,
        }
    except ImportError:
        checkpoint_result = {"status": "torch_unavailable"}
    except (OSError, KeyError, ValueError, RuntimeError) as error:
        checkpoint_result = {"status": "checkpoint_read_error", "error": f"{type(error).__name__}: {error}"}

    metrics: list[dict[str, Any]] = []
    for partition, partition_frame in joined.groupby("partition"):
        cells = partition_frame.groupby("physical_cell_id")
        cell_mae = cells.apply(
            lambda cell: float(
                np.mean(
                    np.abs(
                        pd.to_numeric(cell["cnn1d_prediction_percent"]).to_numpy()
                        - pd.to_numeric(cell["soh_percent"]).to_numpy()
                    )
                )
            ),
            include_groups=False,
        )
        for cell, cell_frame in [("ALL", partition_frame), *list(cells)]:
            error = (
                pd.to_numeric(cell_frame["cnn1d_prediction_percent"]).to_numpy()
                - pd.to_numeric(cell_frame["soh_percent"]).to_numpy()
            )
            baseline_error = (
                pd.to_numeric(cell_frame["train_cell_mean_prediction_percent"]).to_numpy()
                - pd.to_numeric(cell_frame["soh_percent"]).to_numpy()
            )
            metrics.append(
                {
                    "partition": partition,
                    "physical_cell_id": cell,
                    "rpt_count": len(cell_frame),
                    "cell_count": partition_frame["physical_cell_id"].nunique(),
                    "cnn_mae_pp": float(np.mean(np.abs(error))),
                    "cnn_rmse_pp": float(np.sqrt(np.mean(error**2))),
                    "train_mean_baseline_mae_pp": float(np.mean(np.abs(baseline_error))),
                    "macro_cell_mae_pp": float(cell_mae.mean()),
                    "feature_count": len(feature_columns),
                    "feature_columns_only_voltage_progress": bool(len(feature_columns) == 128 and not unexpected_inputs),
                    "partition_split_by_physical_cell": split_consistent,
                    "part3_external_only": external_only,
                    "matches_historical_locked_split": historical_locked_split_used,
                    "cell_macro_train_mean_baseline_matches": baseline_matches_train,
                    "checkpoint_scaler_audit": json.dumps(checkpoint_result, sort_keys=True),
                }
            )
    return pd.DataFrame(metrics), {
        "status": "audited",
        "feature_rows": len(features),
        "supplemental_features": supplemental_summary,
        "prediction_rows": len(predictions),
        "matched_rows": len(joined),
        "feature_count": len(feature_columns),
        "unexpected_feature_columns": unexpected_inputs,
        "physical_cell_split_consistent": split_consistent,
        "physical_cell_split_audit": partition_split_audit,
        "part3_external_only": external_only,
        "checkpoint_scalers": checkpoint_result,
        "train_cell_macro_mean_soh": train_cell_macro_mean,
        "saved_baseline_values": saved_baseline_values,
        "baseline_matches_train_cell_macro_mean": baseline_matches_train,
        "input_hash_audit": model_input_hash_audit,
        "locked_split_comparison": locked_split_comparison,
        "locked_split_version": locked_split_comparison.get("split_version", ""),
        "baseline_split_version": "parts12-10-3-3-plus-part3-external-v1",
        "historical_locked_split_used_by_baseline": historical_locked_split_used,
        "warning": split_warning,
    }


def _source_integrity(root: Path, canonical: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for row in canonical.to_dict("records"):
        part = str(row.get("dataset_part", ""))
        sources = _source_paths(str(row.get("source_paths", "")))
        if not sources:
            output.append(
                {
                    "record_id": row.get("record_id", ""),
                    "record_type": row.get("record_type", ""),
                    "dataset_part": part,
                    "physical_cell_id": row.get("physical_cell_id", ""),
                    "source_path": "",
                    "source_exists": False,
                    "mapping_status": row.get("mapping_status", ""),
                    "audit_status": "no_source_expected" if "no_source" in str(row.get("mapping_status", "")) else "missing_source_reference",
                }
            )
            continue
        for source in sources:
            path = _resolve_source(root, part.split(";")[0], source)
            mapping_status = str(row.get("mapping_status", ""))
            expected_missing = any(
                marker in mapping_status.casefold()
                for marker in ("missing_source", "no_source")
            )
            output.append(
                {
                    "record_id": row.get("record_id", ""),
                    "record_type": row.get("record_type", ""),
                    "dataset_part": part,
                    "physical_cell_id": row.get("physical_cell_id", ""),
                    "source_path": source,
                    "source_exists": path is not None,
                    "mapping_status": row.get("mapping_status", ""),
                    "audit_status": (
                        "source_exists"
                        if path is not None
                        else "missing_source_expected"
                        if expected_missing
                        else "source_missing_or_ambiguous"
                    ),
                }
            )
    return pd.DataFrame(output)


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_input_hash_audit(root: Path) -> dict[str, Any]:
    config_path = root / _MODEL_CONFIG
    if not config_path.is_file():
        return {"status": "missing_model_config"}
    expected_hashes = json.loads(config_path.read_text(encoding="utf-8")).get(
        "input_sha256", {}
    )
    inputs: dict[str, Any] = {}
    for name, relative_path in _MODEL_INPUT_FILES.items():
        path = root / relative_path
        actual_hash = _sha256(path) if path.is_file() else None
        expected_hash = expected_hashes.get(name)
        inputs[name] = {
            "path": str(relative_path),
            "exists": path.is_file(),
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "matches": expected_hash == actual_hash and actual_hash is not None,
        }
    return {
        "status": "verified" if all(item["matches"] for item in inputs.values()) else "mismatch",
        "inputs": inputs,
    }


def run_audit(root: Path, *, chunk_rows: int = 100_000, skip_raw_scan: bool = False) -> dict[str, Any]:
    observations = _read_csv(root / _OBSERVATIONS)
    canonical = _read_csv(root / _CANONICAL)
    soh_tables = [
        (name, _read_csv(root / path), part)
        for name, path, part in _SOH_TABLES
        if (root / path).is_file()
    ]
    source_paths, source_errors = _source_inventory(root, observations, soh_tables)
    scans = {} if skip_raw_scan else _scan_sources(source_paths, chunk_rows)
    mapping = _mapping_audit(root, observations, scans)
    soh = _soh_audit(root, soh_tables, scans, observations)
    pydma = _pydma_audit(root, scans)
    model_metrics, model_summary = _model_audit(root)
    source_integrity = _source_integrity(root, canonical)
    half_cell = _half_cell_reference_audit(root)

    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(root / _REPORT_DIR / "mapping_observation_audit.csv", mapping)
    _write_csv(root / _REPORT_DIR / "soh_label_audit.csv", soh)
    _write_csv(root / _REPORT_DIR / "pydma_fit_audit.csv", pydma)
    _write_csv(root / _REPORT_DIR / "model_integrity_audit.csv", model_metrics)
    _write_csv(root / _REPORT_DIR / "source_integrity_audit.csv", source_integrity)

    mapping_counts = mapping.groupby(["part", "final_mapping_status"], dropna=False).size().to_dict()
    soh_counts = soh.groupby(["dataset_part", "label_table", "audit_status"], dropna=False).size().to_dict()
    pydma_counts = pydma.groupby(["result_table", "fit_status_original"], dropna=False).size().to_dict()
    ready_soh = int(mapping["ready_soh_candidate"].astype(bool).sum()) if not mapping.empty else 0
    ready_deg = int(mapping["ready_degradation_candidate"].astype(bool).sum()) if not mapping.empty else 0
    primary_pydma = pydma.loc[pydma["result_table"] == "part1_part2_rmse15"]
    primary_accepted = primary_pydma.loc[
        (primary_pydma["fit_status_original"] == "diagnostic_estimate_accepted")
        & primary_pydma["passes_recorded_gate"]
    ]
    primary_strict10_count = sum(
        passes_fit_gate(
            bool(row.fit_accepted_flag),
            row.fit_rmse_mV,
            10.0,
            row.accepted_runs,
        )
        for row in primary_accepted.itertuples()
    )
    primary_accepted_ge10_count = int(
        (pd.to_numeric(primary_accepted["fit_rmse_mV"], errors="coerce") >= 10.0).sum()
    )
    primary_raw_flagged_count = int(
        (
            (primary_pydma["fit_status_original"] == "raw_diagnostic_estimate")
            & primary_pydma["fit_accepted_flag"]
        ).sum()
    )
    negative_pydma_outputs: dict[str, dict[str, int]] = {}
    for table_name, table_rows in pydma.groupby("result_table"):
        accepted_rows = table_rows.loc[
            table_rows["fit_status_original"] == "diagnostic_estimate_accepted"
        ]
        negative_pydma_outputs[str(table_name)] = {
            column: int(
                pd.to_numeric(accepted_rows[column], errors="coerce").lt(0).sum()
            )
            for column in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
        }
    ambiguous_rows = mapping.loc[mapping["final_mapping_status"] == "ambiguous_rpt"]
    duplicate_keys = mapping.duplicated(["part", "physical_cell_id", "rpt_index"], keep=False)
    failed_sources = int((source_integrity["audit_status"] == "source_missing_or_ambiguous").sum())
    scanned_status = Counter(str(getattr(scan, "status", "")) for scan in scans.values())
    issue_rows: list[dict[str, Any]] = []
    for reference, status in source_errors.items():
        issue_rows.append(
            {
                "issue_type": "SOURCE_PATH_UNRESOLVED",
                "part": "",
                "physical_cell_id": "",
                "source_file": reference.split("|", 2)[-1],
                "candidate_rpt": "",
                "evidence": status,
                "recommended_action": "Verify source provenance or retain as explicitly missing; do not substitute another cell's source.",
                "severity": "medium",
            }
        )
    for row in ambiguous_rows.to_dict("records"):
        issue_rows.append(
            {
                "issue_type": "AMBIGUOUS_RPT_MAPPING",
                "part": row["part"],
                "physical_cell_id": row["physical_cell_id"],
                "source_file": row["source_file"],
                "candidate_rpt": row["rpt_index"],
                "evidence": row["audit_notes"],
                "recommended_action": "Keep quarantined until procedure/RPT boundaries are independently resolved.",
                "severity": "high",
            }
        )
    for row in mapping.loc[duplicate_keys].to_dict("records"):
        issue_rows.append(
            {
                "issue_type": "DUPLICATE_OBSERVATION_KEY",
                "part": row["part"],
                "physical_cell_id": row["physical_cell_id"],
                "source_file": row["source_file"],
                "candidate_rpt": row["rpt_index"],
                "evidence": "Multiple rows share (part, physical_cell_id, RPT index).",
                "recommended_action": "Resolve duplicate source records before training or scoring.",
                "severity": "high",
            }
        )
    for row in soh.loc[soh["audit_status"] != "verified_source_and_arithmetic"].to_dict("records"):
        issue_rows.append(
            {
                "issue_type": row["audit_status"].upper(),
                "part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "source_file": row["source_file"],
                "candidate_rpt": row["measurement_id"],
                "evidence": row["audit_note"],
                "recommended_action": "Retain as a candidate or QC record; do not promote to official ground truth.",
                "severity": "medium",
            }
        )
    locked_comparison = model_summary.get("locked_split_comparison", {})
    for row in locked_comparison.get("mismatches", []):
        issue_rows.append(
            {
                "issue_type": "LOCKED_SPLIT_MISMATCH",
                "part": "1",
                "physical_cell_id": row["physical_cell_id"],
                "source_file": "",
                "candidate_rpt": "",
                "evidence": (
                    f"Locked split: {row['locked_partition']}; "
                    f"saved baseline: {row['baseline_partition']}."
                ),
                "recommended_action": (
                    "Preserve the locked manifest; report the saved baseline as a distinct split "
                    "and do not silently relabel its metrics."
                ),
                "severity": "high",
            }
        )
    split_audit = model_summary.get("physical_cell_split_audit", {})
    for row in split_audit.get("mismatches", []):
        scope, physical_cell_id = row["split_scope_cell"].split(":", 1)
        issue_rows.append(
            {
                "issue_type": "CELL_SPLIT_INTEGRITY_VIOLATION",
                "part": "1;2" if scope == "parts12" else scope.removeprefix("part"),
                "physical_cell_id": physical_cell_id,
                "source_file": "",
                "candidate_rpt": "",
                "evidence": f"Cell appears in multiple partitions: {row['partitions']}.",
                "recommended_action": (
                    "Keep all observations for the same physical cell in one partition; "
                    "keep Part 3 external and separate from Parts 1–2 cell identifiers."
                ),
                "severity": "critical",
            }
        )
    if model_summary.get("status") == "audited" and not model_summary.get(
        "baseline_matches_train_cell_macro_mean", False
    ):
        issue_rows.append(
            {
                "issue_type": "TRAIN_MEAN_BASELINE_MISMATCH",
                "part": "1;2",
                "physical_cell_id": "",
                "source_file": "",
                "candidate_rpt": "",
                "evidence": "Saved baseline values do not match the macro mean across training-cell SOH means.",
                "recommended_action": "Recompute the baseline from the training partition only before interpreting baseline comparisons.",
                "severity": "medium",
            }
        )
    if model_summary.get("input_hash_audit", {}).get("status") != "verified":
        issue_rows.append(
            {
                "issue_type": "CNN_INPUT_HASH_MISMATCH",
                "part": "",
                "physical_cell_id": "",
                "source_file": "",
                "candidate_rpt": "",
                "evidence": json.dumps(
                    model_summary.get("input_hash_audit", {}), sort_keys=True
                ),
                "recommended_action": (
                    "Preserve the current checkpoint and results, but do not claim exact "
                    "input reproducibility until the recorded source hashes match."
                ),
                "severity": "high",
            }
        )
    supplemental_summary = model_summary.get("supplemental_features", {})
    if supplemental_summary.get("rows_not_in_saved_baseline_features", 0):
        issue_rows.append(
            {
                "issue_type": "SUPPLEMENTAL_ROWS_NOT_IN_SAVED_BASELINE",
                "part": ";".join(supplemental_summary.get("parts_not_in_saved_baseline_features", [])),
                "physical_cell_id": ";".join(supplemental_summary.get("cells_not_in_saved_baseline_features", [])),
                "source_file": "",
                "candidate_rpt": "",
                "evidence": (
                    f"{supplemental_summary['rows_not_in_saved_baseline_features']} supplemental "
                    "feature rows are absent from the hash-verified baseline feature input."
                ),
                "recommended_action": (
                    "Do not claim the saved checkpoint trained on or evaluated these supplemental "
                    "rows; keep their labels separate pending the planned input integration."
                ),
                "severity": "medium",
            }
        )
    _write_csv(root / _REPORT_DIR / "rpt_mapping_issues.csv", pd.DataFrame(issue_rows))

    summary = {
        "raw_sources_scanned": len(scans),
        "raw_scan_status_counts": dict(scanned_status),
        "unresolved_source_references": len(source_errors),
        "canonical_source_reference_failures": failed_sources,
        "expected_missing_source_references": int(
            (source_integrity["audit_status"] == "missing_source_expected").sum()
        ),
        "mapping_rows": len(mapping),
        "mapping_counts_by_part_status": {f"{key[0]}|{key[1]}": value for key, value in mapping_counts.items()},
        "duplicate_observation_rows": int(duplicate_keys.sum()),
        "soh_label_rows": len(soh),
        "soh_audit_counts": {f"{key[0]}|{key[1]}|{key[2]}": value for key, value in soh_counts.items()},
        "pydma_fit_rows": len(pydma),
        "pydma_status_counts": {f"{key[0]}|{key[1]}": value for key, value in pydma_counts.items()},
        "part12_rmse15_accepted_count": len(primary_accepted),
        "part12_rmse15_accepted_also_passing_strict10": primary_strict10_count,
        "part12_rmse15_accepted_with_rmse_ge10": primary_accepted_ge10_count,
        "part12_raw_diagnostics_with_acceptance_flag": primary_raw_flagged_count,
        "negative_pydma_outputs_by_table": negative_pydma_outputs,
        "mapping_step4_ready_candidates": ready_soh,
        "mapping_step8_step10_ready_candidates": ready_deg,
        "part3_half_cell_reference_audit": half_cell,
        "model_audit": model_summary,
        "canonical_manifest_sha256": _sha256(root / _CANONICAL),
        "canonical_manifest_modified": False,
        "labels_or_model_artifacts_modified": False,
        "optimizer_rerun": False,
        "limitations": [
            "SOH is a same-cell capacity ratio candidate; the dataset does not define an official SOH label.",
            "Amphr absolute units are unverified. Ratio arithmetic does not validate the capacity protocol.",
            "PyDMA outputs are fitted diagnostic estimates, not independent ground truth. Saved fit parameters are unavailable for formula-level refitting.",
            "The historical CNN training source code is not present; checkpoint scalers were compared directly with saved training-partition rows.",
            "Part 3 remains external and its candidate SOH labels are not suitable for model selection.",
            "MAT variable contents and full row-level validation outside scanned CSV sources remain unsupported.",
        ],
    }
    (root / _REPORT_DIR / "run_manifest.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    report_lines = [
        "# Final mapping and label audit",
        "",
        "This is a read-only audit of canonical manifest v3, the RPT observation table, source CSV values, existing SOH/PyDMA outputs, and the saved CNN checkpoint. It does not retrain, refit PyDMA, modify labels, or overwrite any earlier manifest.",
        "",
        "## Mapping",
        "",
        f"- Raw source CSVs read: {len(scans)}; scan outcomes: `{dict(scanned_status)}`.",
        f"- RPT candidate rows in this audit: {len(mapping)}; duplicate `(part, cell, RPT)` rows: {int(duplicate_keys.sum())}.",
        f"- Source mapping status by part: `{summary['mapping_counts_by_part_status']}`.",
        f"- Curve-level Step 4 candidates: {ready_soh}; Step 8+10 candidates: {ready_deg}. These are source/curve candidates, not validated labels.",
        "- G5-Cell6 RPT17–23 / RPT24–34 is retained as a user-confirmed policy. The Guide discrepancy at RPT24 is not presented as independently resolved.",
        "- G5-Cell6 TPG5.7 is expanded from its Guide RPT10–15 range only when the six ordered raw protocol blocks match; otherwise its rows remain ambiguous.",
        "- Part 1 G3 TPG3.17/3.18 → Cell14 remains user-confirmed; Part 3 Group 10 TPG/RPT sequence is GUIDE_VERIFIED for Cell2, with the project-owner identity note retained because the printed Guide omits the cell label.",
        "- `TPG5.13EoL-Cell16.csv` and `TPG5.13-Cell6.csv` are separate canonical sources; no identity or file is merged.",
        "",
        "## SOH labels",
        "",
        f"- Audited label rows: {len(soh)}. Per-table source/arithmetic outcomes: `{summary['soh_audit_counts']}`.",
        "- Direct raw Step 4 spans and stored ratio arithmetic are checked where source and segment references are available.",
        "- No value is certified as official SOH ground truth. The Part 1 denominator is a provisional user-selected policy; Part 2/3 baseline policies remain unresolved.",
        "- `Amphr` remains native/unknown-unit; do not report the raw capacity as Ah.",
        "",
        "## PyDMA",
        "",
        f"- Audited fit records: {len(pydma)}. Status counts: `{summary['pydma_status_counts']}`.",
        "- Fit gates are checked against each table's recorded RMSE threshold and the minimum of three accepted optimizer runs. Charge must use Step 10; discharge must use Step 8.",
        f"- Part 1/2's saved 15 mV table has {len(primary_accepted)} recorded-gate accepts; {primary_strict10_count} also pass a strict <10 mV gate, while {primary_accepted_ge10_count} are 10–15 mV and are not accepted under that stricter policy.",
        f"- {primary_raw_flagged_count} raw-diagnostic rows have an acceptance boolean set, but their withheld diagnostic status remains authoritative; they are not counted as accepted labels.",
        f"- Negative accepted PyDMA estimates are retained, not clipped: `{negative_pydma_outputs}`. Treat them as model-derived diagnostics requiring interpretation, not measured negative degradation.",
        "- Fit outputs remain model-derived diagnostics. The saved files do not include parameter vectors needed to independently recompute all LLI/LAM formulas.",
        "- Group 5 candidate fits are audit-only and are not promoted to labels. Part 3 is external-only.",
        "",
        "## CNN1D, features, and split",
        "",
        f"- Checkpoint/scaler audit: `{model_summary.get('checkpoint_scalers', {})}`.",
        f"- Saved baseline partition: {model_summary.get('baseline_split_version')}; historical locked split used: {model_summary.get('historical_locked_split_used_by_baseline')}.",
        f"- Locked split comparison: `{model_summary.get('locked_split_comparison', {})}`.",
        f"- Saved baseline matches train-cell macro mean: {model_summary.get('baseline_matches_train_cell_macro_mean')}; value: {model_summary.get('train_cell_macro_mean_soh')}.",
        f"- Saved model input hashes: `{model_summary.get('input_hash_audit', {})}`.",
        f"- Part 3 external-only check: {model_summary.get('part3_external_only')}; cell partition consistency: {model_summary.get('physical_cell_split_consistent')}.",
        "- The CNN input table contains 128 Step 4 voltage-progress features plus identifiers; no same-RPT capacity, capacity-derived target, cell ID, or part ID is a CNN input.",
        f"- Supplemental extension feature audit: `{model_summary.get('supplemental_features', {})}`. These rows are separate from the saved checkpoint's hash-verified input feature table.",
        "- Checkpoint scaling is compared against the observed training rows. The saved baseline used the expanded 10/3/3 split, not the historical locked 8/2/2 split; do not label its internal metrics as the locked benchmark.",
        "- Part 3 metrics are candidate-label external results, not a basis for model selection.",
        "",
        "## Outstanding items",
        "",
        f"- See `rpt_mapping_issues.csv` for {len(issue_rows)} unresolved or review items.",
        "- No model retraining, PyDMA optimizer rerun, or downstream label/checkpoint/split changes were performed.",
        "- Full MAT decoding and independent validation of the protocol-level SOH definition remain outstanding.",
        "",
        "## Reproduction",
        "",
        "Run `python -m bafuse.data.final_mapping_label_audit` from the repository root. The script rereads the source CSVs used by the mapping and label tables and writes only to this versioned report directory.",
    ]
    (root / _REPORT_DIR / "summary.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit RPT mapping, SOH/PyDMA labels, and CNN split integrity.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--skip-raw-scan", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows < 1:
        parser.error("--chunk-rows must be positive.")
    result = run_audit(args.root.resolve(), chunk_rows=args.chunk_rows, skip_raw_scan=args.skip_raw_scan)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
