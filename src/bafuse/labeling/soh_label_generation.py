from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bafuse.data.soh_step4_scan import write_step4_scan


_LABEL_METHOD = (
    "Step 4 discharge capacity span / earliest complete segment in the first "
    "Guide-mapped BoL procedure for the same cell * 100"
)
_MINIMUM_REVIEW_SOH_PERCENT = 50.0
_LABEL_FIELDS = [
    "dataset_part",
    "group_id",
    "cell_id",
    "physical_cell_id",
    "condition",
    "measurement_id",
    "measurement_order",
    "rpt_order",
    "nominal_cycle_age",
    "cycle_age_status",
    "procedure",
    "capacity_raw",
    "capacity_unit",
    "reference_capacity_raw",
    "sensitivity_reference_capacity_raw",
    "soh_percent",
    "soh_percent_using_secondary_bol",
    "label_source",
    "label_method",
    "label_confidence",
    "source_file",
    "source_candidates",
    "step4_segment_index",
    "step4_row_count",
    "capacity_finite_rows",
    "positive_capacity_deltas",
    "negative_capacity_deltas",
    "step4_current_signs",
    "baseline_source_file",
    "baseline_procedure",
    "baseline_segment_index",
    "validation_status",
    "validation_notes",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _measurement_file_id(row: dict[str, str]) -> int | None:
    try:
        return int(row.get("procedure_file_id", ""))
    except (TypeError, ValueError):
        return None


def _is_bol(row: dict[str, str]) -> bool:
    return (row.get("measurement_id") or "").strip().casefold() == "bol"


def _source_paths(observation: dict[str, str]) -> list[str]:
    return [
        path.strip()
        for path in observation.get("source_file", "").split(";")
        if path.strip()
    ]


def _source_segments(
    observation: dict[str, str],
    scan_by_path: dict[str, dict[str, str]],
) -> tuple[dict[str, str] | None, list[dict[str, Any]], str]:
    source_paths = _source_paths(observation)
    if (
        observation.get("source_mapping_status") == "source_file_not_found"
        or (
            observation.get("source_file_count") == "0"
            and not source_paths
        )
    ):
        return None, [], "unavailable_source_file_not_found"
    if (
        observation.get("source_mapping_status") == "unique_source_file"
        and observation.get("source_file_count") == "1"
        and len(source_paths) == 1
    ):
        relative_path = source_paths[0]
    else:
        selected_candidate = (
            observation.get("step4_candidate_source_file") or ""
        ).strip()
        if selected_candidate and selected_candidate in source_paths:
            relative_path = selected_candidate
        else:
            eligible_paths = []
            for candidate_path in source_paths:
                candidate_scan = scan_by_path.get(candidate_path)
                if candidate_scan is None:
                    continue
                try:
                    candidate_segments = json.loads(
                        candidate_scan.get("step4_segments_json") or "[]"
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid Step 4 segment JSON for {candidate_path}: {error}"
                    ) from error
                if any(
                    isinstance(segment, dict) and _segment_is_complete(segment)
                    for segment in candidate_segments
                ):
                    eligible_paths.append(candidate_path)
            if len(eligible_paths) != 1:
                return None, [], "unavailable_ambiguous_source_mapping"
            relative_path = eligible_paths[0]

    scan_row = scan_by_path.get(relative_path)
    if not relative_path or scan_row is None:
        return None, [], "unavailable_no_scanned_source"
    if observation.get("dataset_part", "1") not in {
        part.strip() for part in scan_row.get("dataset_parts", "").split(";")
    }:
        return None, [], "unavailable_source_part_mismatch"
    if scan_row.get("scan_status") == "scan_failed":
        return scan_row, [], "unavailable_step4_scan_failed"

    try:
        segments = json.loads(scan_row.get("step4_segments_json") or "[]")
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid Step 4 segment JSON for {relative_path}: {error}"
        ) from error
    if not isinstance(segments, list) or any(
        not isinstance(segment, dict) for segment in segments
    ):
        raise ValueError(f"Invalid Step 4 segment data for {relative_path}.")
    return scan_row, segments, ""


def _segment_is_complete(segment: dict[str, Any]) -> bool:
    if not _segment_has_capacity(segment):
        return False
    return not _has_capacity_reversals(segment)


def _segment_has_capacity(segment: dict[str, Any]) -> bool:
    try:
        finite_rows = int(segment.get("capacity_finite_rows", 0))
        span = float(segment["capacity_span"])
    except (KeyError, TypeError, ValueError):
        return False
    return finite_rows >= 10 and math.isfinite(span) and span > 0


def _has_capacity_reversals(segment: dict[str, Any]) -> bool:
    try:
        positive = int(segment.get("positive_capacity_deltas", 0))
        negative = int(segment.get("negative_capacity_deltas", 0))
    except (TypeError, ValueError):
        return True
    return positive > 0 and negative > 0


def _has_mixed_current_signs(segment: dict[str, Any]) -> bool:
    signs = set(segment.get("current_signs") or [])
    return {"negative", "positive"}.issubset(signs)


def _resolve_curve(
    observation: dict[str, str],
    scan_by_path: dict[str, dict[str, str]],
    *,
    baseline: bool,
) -> tuple[dict[str, Any] | None, str]:
    scan_row, segments, error_status = _source_segments(observation, scan_by_path)
    if error_status:
        return None, error_status
    if scan_row is None:
        return None, "unavailable_no_scanned_source"

    if baseline:
        complete = [
            segment for segment in segments if _segment_has_capacity(segment)
        ]
        if not complete:
            return None, "unavailable_no_complete_step4_segment"
        selected = min(complete, key=lambda segment: int(segment["segment_index"]))
        if _has_capacity_reversals(selected):
            return None, "review_capacity_direction_reversal"
    else:
        if scan_row.get("scan_status") == "multiple_step4_segments":
            return None, "unavailable_multiple_step4_segments"
        if scan_row.get("scan_status") != "one_contiguous_step4_candidate":
            return None, f"unavailable_{scan_row.get('scan_status', 'step4_scan_status')}"
        if len(segments) != 1 or not _segment_has_capacity(segments[0]):
            return None, "unavailable_invalid_step4_segment"
        selected = segments[0]
        if _has_capacity_reversals(selected):
            return None, "review_capacity_direction_reversal"

    if _has_mixed_current_signs(selected):
        return None, "review_mixed_current_signs"
    resolved_segment = dict(selected)
    resolved_segment["_source_file"] = scan_row.get("relative_path", "")
    return resolved_segment, ""


def _empty_label(
    observation: dict[str, str],
    *,
    status: str,
    notes: str,
    baseline: dict[str, Any] | None = None,
    capacity: float | None = None,
    segment: dict[str, Any] | None = None,
    alternate_capacity: float | None = None,
    soh: float | None = None,
    sensitivity_soh: float | None = None,
) -> dict[str, Any]:
    return {
        "dataset_part": observation.get("dataset_part", "1"),
        "group_id": observation.get("group_id", ""),
        "cell_id": observation.get("cell_id", ""),
        "physical_cell_id": observation.get("physical_cell_id", ""),
        "condition": "RPT",
        "measurement_id": observation.get("measurement_id", ""),
        "measurement_order": observation.get("procedure_file_id", ""),
        "rpt_order": observation.get("rpt_order", ""),
        "nominal_cycle_age": observation.get("nominal_cycle_age", ""),
        "cycle_age_status": observation.get("cycle_age_status", ""),
        "procedure": observation.get("procedure", ""),
        "capacity_raw": capacity,
        "capacity_unit": "native Amphr; absolute unit unverified",
        "reference_capacity_raw": (
            baseline["capacity"] if baseline is not None else None
        ),
        "sensitivity_reference_capacity_raw": alternate_capacity,
        "soh_percent": soh,
        "soh_percent_using_secondary_bol": sensitivity_soh,
        "label_source": (
            f"Oxford Part {observation.get('dataset_part', '1')} RPT Step 4 CSV"
        ),
        "label_method": _LABEL_METHOD,
        "label_confidence": (
            "medium_ratio_only" if soh is not None else "unavailable"
        ),
        "source_file": (
            segment.get("_source_file")
            if segment is not None
            else (
                observation.get("step4_candidate_source_file", "")
                if observation.get("step4_candidate_source_file")
                in _source_paths(observation)
                else (
                    observation.get("source_file", "")
                    if observation.get("source_mapping_status")
                    == "unique_source_file"
                    else ""
                )
            )
        ),
        "source_candidates": observation.get("source_file", ""),
        "step4_segment_index": segment.get("segment_index") if segment else "",
        "step4_row_count": segment.get("step4_rows") if segment else "",
        "capacity_finite_rows": (
            segment.get("capacity_finite_rows") if segment else ""
        ),
        "positive_capacity_deltas": (
            segment.get("positive_capacity_deltas") if segment else ""
        ),
        "negative_capacity_deltas": (
            segment.get("negative_capacity_deltas") if segment else ""
        ),
        "step4_current_signs": (
            ";".join(segment.get("current_signs", [])) if segment else ""
        ),
        "baseline_source_file": (
            baseline["source_file"] if baseline is not None else ""
        ),
        "baseline_procedure": (
            baseline["procedure"] if baseline is not None else ""
        ),
        "baseline_segment_index": (
            baseline["segment"].get("segment_index")
            if baseline is not None
            else ""
        ),
        "validation_status": status,
        "validation_notes": notes,
    }


def build_soh_labels_for_part(
    observations: list[dict[str, str]],
    source_scan_rows: list[dict[str, str]],
    *,
    dataset_part: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    part_observations = [
        row
        for row in observations
        if row.get("dataset_part") == dataset_part
        and row.get("physical_cell_id", "").strip()
    ]
    if not part_observations:
        raise ValueError(
            f"The Step 4 observation manifest has no Part {dataset_part} rows "
            "with a physical cell identity."
        )

    scan_by_path: dict[str, dict[str, str]] = {}
    for row in source_scan_rows:
        if dataset_part not in {
            part.strip() for part in row.get("dataset_parts", "").split(";")
        }:
            continue
        relative_path = row.get("relative_path", "")
        if relative_path in scan_by_path:
            raise ValueError(
                f"Duplicate Part {dataset_part} source scan row: {relative_path}"
            )
        scan_by_path[relative_path] = row

    by_cell: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in part_observations:
        by_cell[row.get("physical_cell_id", "")].append(row)

    labels: list[dict[str, Any]] = []
    sensitivity_deltas: list[float] = []
    status_counts: Counter[str] = Counter()
    baseline_cells = 0
    secondary_bol_qc_rows = 0
    numeric_rpt_rows = 0
    sensitivity_cells: set[str] = set()
    for physical_cell_id, cell_rows in sorted(by_cell.items()):
        bol_rows = sorted(
            (row for row in cell_rows if _is_bol(row)),
            key=lambda row: (
                _measurement_file_id(row)
                if _measurement_file_id(row) is not None
                else 10**9,
                row.get("procedure", ""),
            ),
        )
        selected_bol = bol_rows[0] if bol_rows else None
        baseline_segment: dict[str, Any] | None = None
        baseline_capacity: float | None = None
        baseline_record: dict[str, Any] | None = None
        baseline_error = "unavailable_no_guide_bol_observation"
        if selected_bol is not None:
            if (
                len(bol_rows) > 1
                and _measurement_file_id(bol_rows[0])
                == _measurement_file_id(bol_rows[1])
            ):
                baseline_error = "unavailable_ambiguous_first_bol_procedure"
            else:
                segment, curve_error = _resolve_curve(
                    selected_bol, scan_by_path, baseline=True
                )
                if segment is not None:
                    baseline_segment = segment
                    baseline_capacity = float(segment["capacity_span"])
                    baseline_record = {
                        "capacity": baseline_capacity,
                        "source_file": baseline_segment.get("_source_file", ""),
                        "procedure": selected_bol.get("procedure", ""),
                        "segment": segment,
                    }
                    baseline_cells += 1
                else:
                    baseline_error = curve_error

        alternate_capacity: float | None = None
        if baseline_record is not None and len(bol_rows) == 2:
            alternate_segment, alternate_error = _resolve_curve(
                bol_rows[1], scan_by_path, baseline=False
            )
            if alternate_segment is not None:
                alternate_capacity = float(alternate_segment["capacity_span"])
                baseline_record["alternate_source_file"] = bol_rows[1].get(
                    "source_file", ""
                )
                baseline_record["alternate_procedure"] = bol_rows[1].get(
                    "procedure", ""
                )
                baseline_record["alternate_segment"] = alternate_segment
                baseline_record["alternate_capacity"] = alternate_capacity
            else:
                baseline_record["alternate_error"] = alternate_error

        rpt_counts = Counter(
            row.get("measurement_id", "")
            for row in cell_rows
            if not _is_bol(row)
            and (row.get("measurement_id") or "").strip().casefold().startswith("rpt")
        )
        for observation in cell_rows:
            if observation is selected_bol:
                if baseline_record is None:
                    label = _empty_label(
                        observation,
                        status=baseline_error,
                        notes=(
                            "The earliest Guide-mapped BoL procedure could not "
                            "provide a complete, unambiguous Step 4 baseline; later "
                            "BoL procedures are not used as fallback."
                        ),
                    )
                else:
                    sensitivity_soh = (
                        100.0 * baseline_capacity / alternate_capacity
                        if alternate_capacity is not None
                        else None
                    )
                    label = _empty_label(
                        observation,
                        status="baseline_reference",
                        notes=(
                            "Selected by the user-approved provisional policy: "
                            "earliest complete segment in the first Guide-mapped "
                            "BoL procedure. The later BoL procedure is QC/sensitivity "
                            "only; protocol identity remains unverified."
                        ),
                        baseline=baseline_record,
                        capacity=baseline_capacity,
                        segment=baseline_segment,
                        alternate_capacity=alternate_capacity,
                        soh=100.0,
                        sensitivity_soh=sensitivity_soh,
                    )
                labels.append(label)
                status_counts[label["validation_status"]] += 1
                continue

            if _is_bol(observation):
                secondary_bol_qc_rows += 1
                secondary_segment, secondary_error = _resolve_curve(
                    observation, scan_by_path, baseline=False
                )
                secondary_capacity = (
                    float(secondary_segment["capacity_span"])
                    if secondary_segment is not None
                    else None
                )
                sensitivity_soh = (
                    100.0 if secondary_capacity is not None else None
                )
                label = _empty_label(
                    observation,
                    status="secondary_bol_qc_only",
                    notes=(
                        "Retained for QC/sensitivity, not as an independent RPT "
                        "target or chronological SOH observation."
                        if secondary_segment is not None
                        else (
                            "Retained as a secondary BoL record but not used as an "
                            f"independent target: {secondary_error}."
                        )
                    ),
                    baseline=baseline_record,
                    capacity=secondary_capacity,
                    segment=secondary_segment,
                    alternate_capacity=alternate_capacity,
                    sensitivity_soh=sensitivity_soh,
                )
                labels.append(label)
                status_counts[label["validation_status"]] += 1
                continue

            measurement_id = observation.get("measurement_id", "")
            if rpt_counts[measurement_id] != 1:
                status = "unavailable_duplicate_rpt_mapping"
                segment = None
            elif baseline_record is None:
                status = "unavailable_no_measured_bol_reference"
                segment = None
            elif observation.get("measurement_mapping_status") != "single_checkpoint":
                status = "unavailable_ambiguous_rpt_mapping"
                segment = None
            else:
                segment, status = _resolve_curve(
                    observation, scan_by_path, baseline=False
                )
                if segment is not None:
                    status = ""

            if status:
                label = _empty_label(
                    observation,
                    status=status,
                    notes=(
                        "No SOH target was assigned because this observation did "
                        "not pass the unique-checkpoint, source-mapping, Step 4 "
                        "segment, and capacity QC gates."
                    ),
                    baseline=baseline_record,
                    alternate_capacity=alternate_capacity,
                )
            else:
                if segment is None or baseline_capacity is None:
                    raise RuntimeError("Validated RPT curve has no baseline capacity.")
                capacity = float(segment["capacity_span"])
                soh = 100.0 * capacity / baseline_capacity
                sensitivity_soh = (
                    100.0 * capacity / alternate_capacity
                    if alternate_capacity is not None
                    else None
                )
                validation_status = "calculated_from_measured_capacity"
                notes = (
                    "SOH is a same-cell capacity ratio, not an official dataset "
                    "label. Capacity is native Amphr; its absolute unit is "
                    "unverified."
                )
                if soh < 0 or soh > 120:
                    validation_status = "physically_implausible_review"
                    notes += " SOH is outside 0-120%; retained without clipping."
                elif soh < _MINIMUM_REVIEW_SOH_PERCENT:
                    validation_status = "review_capacity_span_outlier"
                    notes += (
                        " SOH is below the conservative 50% same-cell baseline "
                        "review floor; retain the measured value for audit but "
                        "exclude it from training until the source curve is verified."
                    )
                else:
                    numeric_rpt_rows += 1
                label = _empty_label(
                    observation,
                    status=validation_status,
                    notes=notes,
                    baseline=baseline_record,
                    capacity=capacity,
                    segment=segment,
                    alternate_capacity=alternate_capacity,
                    soh=soh,
                    sensitivity_soh=sensitivity_soh,
                )
                if sensitivity_soh is not None:
                    sensitivity_deltas.append(abs(soh - sensitivity_soh))
                    sensitivity_cells.add(physical_cell_id)
            labels.append(label)
            status_counts[label["validation_status"]] += 1

    labels.sort(
        key=lambda row: (
            int(row["group_id"]) if str(row["group_id"]).isdigit() else 10**9,
            int(row["cell_id"]) if str(row["cell_id"]).isdigit() else 10**9,
            int(row["rpt_order"]) if str(row["rpt_order"]).isdigit() else 10**9,
            int(row["measurement_order"])
            if str(row["measurement_order"]).isdigit()
            else 10**9,
        )
    )
    part_observation_count = sum(
        row.get("dataset_part") == dataset_part for row in observations
    )
    summary = {
        "operation": (
            f"Part {dataset_part} SOH label generation from the Step 4 screening manifest"
        ),
        "dataset_part": dataset_part,
        "dataset_scope": (
            "Oxford Path Dependent Battery Degradation Dataset "
            f"Part {dataset_part}"
        ),
        "baseline_policy": (
            "earliest complete Step 4 segment in the earliest Guide-mapped BoL "
            "procedure; later BoL procedure is QC/sensitivity only"
        ),
        "baseline_policy_status": "user_selected_provisional_protocol_unverified",
        "capacity_span_review_floor_soh_percent": _MINIMUM_REVIEW_SOH_PERCENT,
        "label_method": _LABEL_METHOD,
        "label_count": len(labels),
        "baseline_cell_count": baseline_cells,
        "secondary_bol_qc_rows": secondary_bol_qc_rows,
        "numeric_rpt_label_count": numeric_rpt_rows,
        "capacity_direction_checked_segments": sum(
            row.get("capacity_raw") is not None
            and row.get("step4_segment_index") not in {"", None}
            for row in labels
        ),
        "capacity_direction_reversal_observations": status_counts.get(
            "review_capacity_direction_reversal", 0
        ),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "secondary_bol_sensitivity_available_cells": len(sensitivity_cells),
        "mean_absolute_rpt_soh_difference_using_secondary_bol_percentage_points": (
            sum(sensitivity_deltas) / len(sensitivity_deltas)
            if sensitivity_deltas
            else None
        ),
        "max_absolute_rpt_soh_difference_using_secondary_bol_percentage_points": (
            max(sensitivity_deltas) if sensitivity_deltas else None
        ),
        "official_soh_labels_found": False,
        "parts_2_and_3_included": False,
        "included_dataset_part": dataset_part,
        "source_files_modified": False,
        "limitations": [
            "The baseline policy is user-selected and provisional; signal equality does not prove which Guide procedure the original protocol intended as the official denominator.",
            "Amphr is retained in its native unit; the absolute unit is not verified by the local Guide.",
            f"Only single-checkpoint Part {dataset_part} RPT observations with one QC-eligible Step 4 segment receive numeric targets; ambiguous or multi-segment RPTs are withheld.",
            "Capacity-derived SOH below 50% of the same-cell BoL is retained for audit but withheld from training pending source-curve review.",
            "The later BoL procedure is retained for QC and sensitivity analysis, not counted as another chronological RPT.",
            f"Other dataset parts are excluded from this Part {dataset_part} output.",
            "Nominal cycle age remains a documented cadence proxy, not reconstructed exposure.",
        ],
        "quarantined_missing_physical_cell_id_observation_count": (
            part_observation_count - len(part_observations)
        ),
    }
    return labels, summary


def build_part1_soh_labels(
    observations: list[dict[str, str]],
    source_scan_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_soh_labels_for_part(
        observations, source_scan_rows, dataset_part="1"
    )


def build_part3_soh_labels(
    observations: list[dict[str, str]],
    source_scan_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_soh_labels_for_part(
        observations, source_scan_rows, dataset_part="3"
    )


def _verify_scan_inputs(scan_dir: Path, dataset_root: Path) -> tuple[Path, Path, dict[str, Any]]:
    summary_path = scan_dir / "summary.json"
    observation_path = scan_dir / "guide_observation_manifest_step4.csv"
    source_scan_path = scan_dir / "source_file_step4_scan.csv"
    for path in (summary_path, observation_path, source_scan_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Step 4 scan input is missing: {path}")

    scan_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected_root = scan_summary.get("dataset_roots", {}).get("1")
    if expected_root != str(dataset_root.resolve()):
        raise ValueError(
            "Part 1 dataset root differs from the one recorded in the Step 4 scan."
        )
    guide_paths = list(dataset_root.glob("Guide*.xlsx"))
    if len(guide_paths) != 1:
        raise ValueError(
            f"Expected exactly one Guide workbook in {dataset_root}, found {len(guide_paths)}."
        )
    expected_guide_hash = scan_summary.get("guide_sha256", {}).get("1")
    if _sha256(guide_paths[0]) != expected_guide_hash:
        raise ValueError("Part 1 Guide changed since the Step 4 scan.")

    integrity = scan_summary.get("post_scan_integrity_verification", {})
    if (
        integrity.get("source_paths_missing") != 0
        or integrity.get("source_size_mismatches") != 0
        or integrity.get("source_paths_modified_after_scan") != 0
    ):
        raise ValueError("The Step 4 scan did not pass its source integrity checks.")
    return observation_path, source_scan_path, scan_summary


def run_part1_soh_label_generation(
    scan_dir: Path,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must not be inside the source dataset.")
    try:
        output_dir.relative_to(scan_dir)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must not be inside the Step 4 scan directory.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output_dir}"
        )

    observation_path, source_scan_path, scan_summary = _verify_scan_inputs(
        scan_dir, dataset_root
    )
    observations = _read_csv(observation_path)
    source_scan_rows = _read_csv(source_scan_path)
    source_by_path = {
        row.get("relative_path", ""): row
        for row in source_scan_rows
        if "1" in {
            part.strip() for part in row.get("dataset_parts", "").split(";")
        }
    }
    for observation in observations:
        if observation.get("dataset_part") != "1":
            continue
        relative_paths = [
            item.strip()
            for item in observation.get("source_file", "").split(";")
            if item.strip()
        ]
        for relative_path in relative_paths:
            scan_row = source_by_path.get(relative_path)
            source_path = (dataset_root / relative_path).resolve()
            try:
                source_path.relative_to(dataset_root)
            except ValueError as error:
                raise ValueError(
                    f"Source path escapes the Part 1 dataset root: {relative_path}"
                ) from error
            if scan_row is None:
                if observation.get("measurement_mapping_status") == "single_checkpoint":
                    raise ValueError(
                        f"Mapped Part 1 source was not included in the Step 4 scan: {relative_path}"
                    )
                continue
            if not source_path.is_file():
                raise FileNotFoundError(
                    f"Source file recorded by the Step 4 scan is missing: {source_path}"
                )
            stat = source_path.stat()
            if (
                str(stat.st_size) != scan_row.get("source_size_bytes")
                or str(stat.st_mtime_ns) != scan_row.get("source_modified_ns")
            ):
                raise ValueError(
                    f"Source file changed since the Step 4 scan: {source_path}"
                )

    labels, summary = build_part1_soh_labels(observations, source_scan_rows)
    summary.update(
        {
            "step4_scan_directory": str(scan_dir),
            "dataset_root": str(dataset_root),
            "guide_sha256": scan_summary["guide_sha256"]["1"],
            "observation_manifest_sha256": _sha256(observation_path),
            "source_scan_sha256": _sha256(source_scan_path),
            "source_integrity_rechecked": True,
            "source_files_modified": False,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "labels_soh.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_LABEL_FIELDS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(labels)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def run_part3_soh_candidate_generation(
    corrected_manifest_path: Path,
    rpt_scan_dir: Path,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    corrected_manifest_path = corrected_manifest_path.resolve()
    rpt_scan_dir = rpt_scan_dir.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()

    for protected_path in (dataset_root, corrected_manifest_path.parent, rpt_scan_dir):
        try:
            output_dir.relative_to(protected_path)
        except ValueError:
            continue
        raise ValueError(f"Output directory must not be inside {protected_path}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty; refusing to overwrite: {output_dir}"
        )

    manifest_summary_path = corrected_manifest_path.with_name("summary.json")
    rpt_summary_path = rpt_scan_dir / "summary.json"
    rpt_source_scan_path = rpt_scan_dir / "source_file_step4_scan.csv"
    for path in (
        corrected_manifest_path,
        manifest_summary_path,
        rpt_summary_path,
        rpt_source_scan_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"Required Part 3 input is missing: {path}")

    manifest_summary = json.loads(
        manifest_summary_path.read_text(encoding="utf-8")
    )
    rpt_summary = json.loads(rpt_summary_path.read_text(encoding="utf-8"))
    expected_root = str(dataset_root)
    guide_paths = list(dataset_root.glob("Guide*.xlsx"))
    if len(guide_paths) != 1:
        raise ValueError(
            f"Expected exactly one Guide workbook in {dataset_root}, found {len(guide_paths)}."
        )
    guide_hash = _sha256(guide_paths[0])
    if manifest_summary.get("dataset_roots", {}).get("3") != expected_root:
        raise ValueError("Part 3 dataset root differs from the corrected manifest.")
    if manifest_summary.get("guide_sha256", {}).get("3") != guide_hash:
        raise ValueError("Part 3 Guide changed since corrected manifest creation.")
    if rpt_summary.get("dataset_roots", {}).get("3") != expected_root:
        raise ValueError("Part 3 dataset root differs from the RPT Step 4 scan.")
    if rpt_summary.get("guide_sha256", {}).get("3") != guide_hash:
        raise ValueError("Part 3 Guide changed since the RPT Step 4 scan.")

    observations = _read_csv(corrected_manifest_path)
    part3_observations = [
        row for row in observations if row.get("dataset_part") == "3"
    ]
    identified_observations = [
        row for row in part3_observations
        if row.get("physical_cell_id", "").strip()
    ]
    quarantined_observations = (
        len(part3_observations) - len(identified_observations)
    )
    baseline_observations = [
        row for row in identified_observations if _is_bol(row)
    ]
    if not baseline_observations:
        raise ValueError("Corrected Part 3 manifest contains no identified BoL rows.")
    baseline_paths: set[str] = set()
    for observation in baseline_observations:
        paths = _source_paths(observation)
        if (
            observation.get("source_mapping_status") != "unique_source_file"
            or observation.get("source_file_count") != "1"
            or observation.get("source_file_id") != "0"
            or observation.get("source_mapping_rule")
            != "part3_unsuffixed_bol_to_zero_suffix"
            or len(paths) != 1
        ):
            raise ValueError(
                "Each identified Part 3 BoL must map uniquely to its corrected `.0` source."
            )
        baseline_paths.add(paths[0])
    if len(baseline_paths) != len(baseline_observations):
        raise ValueError("Corrected Part 3 BoL rows do not map to distinct sources.")

    output_dir.mkdir(parents=True, exist_ok=True)
    with corrected_manifest_path.open("rb") as source, (
        output_dir / "guide_observation_manifest.csv"
    ).open("wb") as target:
        target.write(source.read())
    (output_dir / "guide_manifest_summary.json").write_bytes(
        manifest_summary_path.read_bytes()
    )

    baseline_input_dir = output_dir / "part3_bol_step4_input"
    baseline_input_dir.mkdir()
    baseline_manifest_path = baseline_input_dir / "guide_observation_manifest.csv"
    baseline_fields = list(
        dict.fromkeys(key for row in observations for key in row)
    )
    with baseline_manifest_path.open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=baseline_fields)
        writer.writeheader()
        writer.writerows(baseline_observations)
    (baseline_input_dir / "summary.json").write_bytes(
        manifest_summary_path.read_bytes()
    )

    baseline_scan_dir = output_dir / "part3_bol_step4_scan"
    baseline_scan_summary = write_step4_scan(
        baseline_manifest_path,
        {3: dataset_root},
        baseline_scan_dir,
    )
    baseline_scan_path = baseline_scan_dir / "source_file_step4_scan.csv"
    baseline_scan_rows = _read_csv(baseline_scan_path)
    rpt_scan_rows = _read_csv(rpt_source_scan_path)

    source_by_path: dict[str, dict[str, str]] = {}
    for scan_row in rpt_scan_rows + baseline_scan_rows:
        if "3" not in {
            part.strip() for part in scan_row.get("dataset_parts", "").split(";")
        }:
            continue
        relative_path = scan_row.get("relative_path", "")
        if relative_path in source_by_path:
            raise ValueError(f"Duplicate Part 3 source scan row: {relative_path}")
        source_by_path[relative_path] = scan_row

    required_paths = {
        path
        for observation in identified_observations
        for path in _source_paths(observation)
    }
    if not baseline_paths.issubset({row.get("relative_path", "") for row in baseline_scan_rows}):
        raise ValueError("The corrected BoL sources were not all included in the Step 4 scan.")
    combined_scan_rows = [
        source_by_path[path] for path in sorted(required_paths) if path in source_by_path
    ]
    missing_scan_paths = required_paths - {
        row.get("relative_path", "") for row in combined_scan_rows
    }
    if missing_scan_paths:
        raise ValueError(
            "Part 3 mapped sources are missing from the Step 4 scans: "
            + ", ".join(sorted(missing_scan_paths))
        )

    for relative_path, scan_row in source_by_path.items():
        if relative_path not in required_paths:
            continue
        source_path = (dataset_root / relative_path).resolve()
        try:
            source_path.relative_to(dataset_root)
        except ValueError as error:
            raise ValueError(
                f"Source path escapes the Part 3 dataset root: {relative_path}"
            ) from error
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Source file recorded by the Step 4 scan is missing: {source_path}"
            )
        stat = source_path.stat()
        if (
            str(stat.st_size) != scan_row.get("source_size_bytes")
            or str(stat.st_mtime_ns) != scan_row.get("source_modified_ns")
        ):
            raise ValueError(f"Source file changed since its Step 4 scan: {source_path}")

    labels, summary = build_part3_soh_labels(
        observations, combined_scan_rows
    )
    summary.update(
        {
            "operation": "Part 3 external-cohort candidate SOH generation",
            "candidate_only": True,
            "external_cohort": True,
            "baseline_policy_status": "user_selected_provisional_protocol_unverified",
            "baseline_mapping_policy": (
                "Corrected Guide base-BoL procedure mapped to source `.0`; use "
                "the earliest complete Step 4 segment for each physical cell."
            ),
            "dataset_root": expected_root,
            "guide_sha256": guide_hash,
            "corrected_manifest": str(corrected_manifest_path),
            "corrected_manifest_sha256": _sha256(corrected_manifest_path),
            "rpt_step4_scan_directory": str(rpt_scan_dir),
            "rpt_step4_scan_sha256": _sha256(rpt_source_scan_path),
            "bol_step4_scan_directory": str(baseline_scan_dir),
            "bol_step4_scan_sha256": _sha256(baseline_scan_path),
            "bol_step4_scan_summary": baseline_scan_summary,
            "combined_source_scan_sha256": None,
            "source_integrity_rechecked": True,
            "source_files_modified": False,
            "quarantined_missing_physical_cell_id_observation_count": (
                quarantined_observations
            ),
            "limitations": [
                "These are capacity-derived external-cohort candidate labels, not official SOH labels or independent physical ground truth.",
                "The per-cell `.0` BoL denominator is a provisional policy; official protocol identity remains unverified.",
                "Part 3 is held out from model selection; do not use these candidates to choose features, thresholds, or hyperparameters.",
                "Capacity is retained in native Amphr units; its absolute unit is not verified by the local Guide.",
                "Capacity-derived SOH below 50% is retained for audit but withheld from training pending source-curve review.",
                "The cell-identity-free Group 10 Guide chain is quarantined and excluded.",
                "Nominal cycle age is a cadence proxy, not reconstructed exposure.",
            ],
        }
    )

    combined_scan_path = output_dir / "combined_source_file_step4_scan.csv"
    scan_fields = list(
        dict.fromkeys(key for row in combined_scan_rows for key in row)
    )
    with combined_scan_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scan_fields)
        writer.writeheader()
        writer.writerows(combined_scan_rows)
    summary["combined_source_scan_sha256"] = _sha256(combined_scan_path)

    with (output_dir / "labels_soh_candidates.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_LABEL_FIELDS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(labels)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# Part 3 external-cohort SOH candidates\n\n"
        "This output uses capacity ratios from Step 4 discharge segments. "
        "Each identified physical cell uses the corrected Guide base-BoL "
        "source `.0` as a provisional same-cell denominator. The BoL files "
        "were rescanned with the shared Step 4 scanner; RPT scans were reused "
        "only after current source size and modification-time checks passed.\n\n"
        "These records are candidate labels, not official SOH targets or "
        "independent physical ground truth. Part 3 is external to model "
        "selection and must not be used for feature/model/hyperparameter "
        "selection. The Group 10 chain without a cell ID remains quarantined.\n\n"
        "See `summary.json`, `labels_soh_candidates.csv`, "
        "`combined_source_file_step4_scan.csv`, and `part3_bol_step4_scan/` "
        "for provenance and QC details.\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a new Part 1 SOH label table using the selected earliest-BoL "
            "baseline policy and the verified Step 4 scan."
        )
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_part1_soh_label_generation(
        args.scan_dir, args.dataset_root, args.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
