from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_adaptive_v1"
    / "phase_a_b_label_audit_v1"
)
INPUTS = {
    "observation_audit": ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_missing_modality_audit_v1"
    / "all_rpts_modality_availability.csv",
    "part1_labels": ROOT
    / "data"
    / "processed"
    / "label_soh_part1_2"
    / "run_007_part1_capacity_qc"
    / "labels_soh.csv",
    "supplemental_labels": ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part2_g5_g6_soh_extension"
    / "labels_soh_part2_supplemental_g5_g6.csv",
    "step4_segments": ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part2_g5_g6_soh_extension"
    / "step4_segment_audit.csv",
    "step4_boundaries": ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "step4_boundary_scan"
    / "guide_observation_manifest_step4.csv",
    "bol_references": ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part2_g5_g6_soh_extension"
    / "bol_denominator_audit.csv",
    "guide": ROOT
    / "data"
    / "manifests"
    / "experiment_manifests"
    / "guide_observation_manifest"
    / "guide_observation_manifest.csv",
}

AUDIT_FIELDS = (
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "partition",
    "rpt_index",
    "measurement_id",
    "nominal_cycle_age",
    "cycle_age_status",
    "label_status_before_audit",
    "guide_mapping_confidence",
    "guide_stage",
    "guide_source_mapping_status",
    "source_file",
    "source_file_exists",
    "source_mapping_provenance",
    "step4_availability_status",
    "step4_boundary_status",
    "step4_rpt_alignment_status",
    "step4_segment_ref",
    "step4_rows_observed",
    "step4_capacity_finite_rows",
    "step4_native_capacity_span",
    "step4_current_signs",
    "step4_step_ids",
    "procedure_step4_segment_count",
    "procedure_step4_capacity_spans",
    "procedure_step4_current_signs",
    "bol_capacity_reference_raw",
    "bol_capacity_source_file",
    "bol_reference_status",
    "capacity_ratio_reliable",
    "candidate_soh_percent",
    "candidate_calculation",
    "candidate_decision",
    "candidate_reason",
    "existing_supplemental_soh_percent",
    "existing_supplemental_capacity_raw",
    "supplemental_step4_segment_index",
    "supplemental_validation_status",
    "supplemental_label_provenance",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _number(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _normalized_path(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


def _source_suffix(value: str) -> str:
    normalized = _normalized_path(value)
    marker = "data/raw/oxford/part2/"
    return normalized.split(marker, 1)[-1]


def _procedure_file_id(source_file: str) -> str | None:
    match = re.search(r"TPG[1-6][._](\d+)", Path(source_file).name)
    return match.group(1) if match else None


def _scan_step4_capacity(
    source: Path, start_line: int, end_line: int
) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"Mapped Step 4 source is missing: {source}")
    if start_line < 2 or end_line < start_line:
        raise ValueError(f"Invalid Step 4 line range: {start_line}-{end_line}")

    capacities: list[float] = []
    current_signs: set[str] = set()
    step_ids: set[str] = set()
    missing_voltage = 0
    missing_capacity = 0
    observed_rows = 0
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Mapped source has no CSV header: {source}")
        indices = {name: index for index, name in enumerate(header)}
        required = {"Step", "Volts", "Amps", "Amphr"}
        missing_columns = required - set(indices)
        if missing_columns:
            raise ValueError(
                f"Mapped source missing required columns: {sorted(missing_columns)}"
            )
        for values in reader:
            line_number = reader.line_num
            if line_number < start_line:
                continue
            if line_number > end_line:
                break
            observed_rows += 1

            def value(name: str) -> float | None:
                index = indices[name]
                if index >= len(values):
                    return None
                return _number(values[index])

            step = value("Step")
            current = value("Amps")
            voltage = value("Volts")
            capacity = value("Amphr")
            step_ids.add(str(int(step)) if step is not None and step.is_integer() else "invalid")
            if current is None:
                current_signs.add("missing")
            elif current < 0:
                current_signs.add("negative")
            elif current > 0:
                current_signs.add("positive")
            else:
                current_signs.add("zero")
            missing_voltage += voltage is None
            missing_capacity += capacity is None
            if capacity is not None:
                capacities.append(capacity)

    expected_rows = end_line - start_line + 1
    if observed_rows != expected_rows:
        raise ValueError(
            f"Step 4 row count mismatch for {source}: "
            f"expected {expected_rows}, observed {observed_rows}"
        )
    if (
        step_ids != {"4"}
        or current_signs != {"negative"}
        or missing_voltage
        or missing_capacity
        or len(capacities) != expected_rows
    ):
        raise ValueError(
            f"Step 4 validation failed for {source} rows "
            f"{start_line}-{end_line}: steps={step_ids}, "
            f"current={current_signs}, missing_voltage={missing_voltage}, "
            f"missing_capacity={missing_capacity}"
        )

    deltas = [second - first for first, second in zip(capacities, capacities[1:])]
    if any(delta < 0 for delta in deltas):
        raise ValueError(f"Amphr is not monotonic in mapped Step 4 segment: {source}")
    return {
        "rows": observed_rows,
        "capacity_finite_rows": len(capacities),
        "capacity_span": max(capacities) - min(capacities),
        "current_signs": "negative",
        "step_ids": "4",
    }


def _index_unique(
    rows: list[dict[str, str]], key_fn: Any, description: str
) -> dict[Any, dict[str, str]]:
    indexed: dict[Any, dict[str, str]] = {}
    for row in rows:
        key = key_fn(row)
        if key in indexed:
            raise ValueError(f"Duplicate {description} key: {key}")
        indexed[key] = row
    return indexed


def _build_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    observations = _read_csv(INPUTS["observation_audit"])
    part1_labels = _read_csv(INPUTS["part1_labels"])
    supplemental = _read_csv(INPUTS["supplemental_labels"])
    step4_segments = _read_csv(INPUTS["step4_segments"])
    step4_boundaries = _read_csv(INPUTS["step4_boundaries"])
    bol_references = _read_csv(INPUTS["bol_references"])
    guide = _read_csv(INPUTS["guide"])

    label_by_rpt = _index_unique(
        [row for row in part1_labels if row["rpt_order"] != "0"],
        lambda row: (row["physical_cell_id"], row["rpt_order"]),
        "Part 1 label",
    )
    supplement_by_rpt = _index_unique(
        supplemental,
        lambda row: (row["physical_cell_id"], row["rpt_order"]),
        "supplemental label",
    )
    bol_by_cell = _index_unique(
        bol_references,
        lambda row: row["physical_cell_id"],
        "Part 2 BoL reference",
    )
    segment_by_source: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in step4_segments:
        key = (row["physical_cell_id"], _normalized_path(row["source_file"]))
        segment_by_source.setdefault(key, []).append(row)

    guide_by_file: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in guide:
        key = (
            row["dataset_part"],
            row["group_id"],
            row["physical_cell_id"],
            row["procedure_file_id"],
        )
        if key in guide_by_file:
            raise ValueError(f"Duplicate Guide procedure key: {key}")
        guide_by_file[key] = row
    boundary_by_file = _index_unique(
        step4_boundaries,
        lambda row: (
            row["dataset_part"],
            row["group_id"],
            row["physical_cell_id"],
            row["procedure_file_id"],
        ),
        "Step 4 boundary",
    )

    missing = [
        row
        for row in observations
        if row["partition"] in {"train", "validation"}
        and row["valid_soh_label"].strip().casefold() == "false"
    ]
    if len(missing) != 228:
        raise ValueError(f"Expected 228 missing Train/Validation labels, got {len(missing)}")

    guide_by_rpt = {
        (row["physical_cell_id"], row["rpt_order"]): row
        for row in guide
        if row["dataset_part"] == "2"
        and row["measurement_mapping_status"] == "single_checkpoint"
        and row["rpt_order"]
    }
    part1_baselines: dict[str, dict[str, str]] = {}
    for row in part1_labels:
        if row["validation_status"] == "baseline_reference":
            if row["physical_cell_id"] in part1_baselines:
                raise ValueError(
                    f"Duplicate selected Part 1 BoL reference: {row['physical_cell_id']}"
                )
            part1_baselines[row["physical_cell_id"]] = row

    result: list[dict[str, Any]] = []
    for item in missing:
        part = item["dataset_part"]
        group_id = item["group"]
        cell = item["physical_cell_id"]
        rpt = item["rpt_index"]
        measurement_id = item["measurement_id"]
        source_file = item["source_file"]
        part1_label = label_by_rpt.get((cell, rpt)) if part == "1" else None
        supplemental_row = supplement_by_rpt.get((cell, rpt)) if part == "2" else None
        bol_row = bol_by_cell.get(cell) if part == "2" else None
        guide_row: dict[str, str] | None = None
        boundary_row: dict[str, str] | None = None

        file_id = _procedure_file_id(source_file)
        if file_id is not None:
            guide_row = guide_by_file.get((part, group_id, cell, file_id))
            boundary_row = boundary_by_file.get((part, group_id, cell, file_id))

        mapping_confidence = (
            guide_row["measurement_mapping_status"] if guide_row else "unresolved"
        )
        guide_stage = guide_row["guide_stage"] if guide_row else ""
        guide_source_status = guide_row["source_mapping_status"] if guide_row else ""
        source_path = ROOT / source_file
        source_exists = source_path.is_file()
        source_mapping_provenance = (
            f"Guide procedure {file_id}: {guide_stage}; "
            f"source {guide_source_status}; canonical observation manifest "
            f"source_file={source_file}"
            if guide_row
            else f"Canonical observation manifest source_file={source_file}"
        )

        bol_capacity: str = ""
        bol_source: str = ""
        bol_status = "not established"
        step4_status = "not verified"
        step4_boundary_status = (
            boundary_row["step4_boundary_status"] if boundary_row else "not mapped"
        )
        step4_alignment = "not verified"
        step4_ref = item.get("step4_segment_ref", "")
        step4_rows = ""
        step4_capacity_rows = ""
        step4_span = ""
        step4_current_signs = ""
        step4_step_ids = ""
        procedure_step4_segment_count = ""
        procedure_step4_capacity_spans = ""
        procedure_step4_current_signs = ""
        capacity_ratio_reliable = "FALSE"
        candidate_soh = ""
        calculation = ""
        decision = "requires_manual_review"
        reason = ""
        existing_supplemental_soh = ""
        supplemental_provenance = ""

        if part == "1":
            if part1_label is None:
                raise ValueError(f"Missing Part 1 label audit row for {cell} {measurement_id}")
            bol_capacity = part1_label["reference_capacity_raw"]
            bol_source = part1_label["baseline_source_file"]
            bol_status = (
                "selected Part 1 baseline under user-approved provisional policy"
                if bol_capacity
                else "no verified cell-specific BoL capacity"
            )
            step4_status = part1_label["validation_status"]
            if part1_label["step4_row_count"]:
                step4_rows = part1_label["step4_row_count"]
                step4_capacity_rows = part1_label["capacity_finite_rows"]
                step4_span = part1_label["capacity_raw"]
                step4_current_signs = part1_label["step4_current_signs"]
                step4_step_ids = "4"
                step4_alignment = "existing Part 1 Step 4 audit"
            if cell == "G3-Cell14" and rpt in {"11", "12"}:
                match = re.search(r"#Step4:segment\d+:rows(\d+)-(\d+)", step4_ref)
                if match is None:
                    raise ValueError(f"Missing exact Step 4 range for {cell} {measurement_id}")
                start_line, end_line = map(int, match.groups())
                evidence = _scan_step4_capacity(
                    source_path, start_line, end_line
                )
                if not guide_row:
                    raise ValueError(f"Missing Guide mapping for {cell} {measurement_id}")
                cell_number = re.search(r"Cell(\d+)", cell)
                if (
                    guide_row["measurement_mapping_status"] != "single_checkpoint"
                    or guide_row["measurement_id"] != measurement_id
                    or cell_number is None
                    or source_path.name.casefold()
                    != (
                        f"TPG{group_id}_{file_id}_Cell{cell_number.group(1)}.csv"
                    ).casefold()
                ):
                    raise ValueError(
                        f"Guide/source identity mismatch for {cell} {measurement_id}"
                    )
                baseline = part1_baselines.get(cell)
                if baseline is None:
                    raise ValueError(f"Missing selected BoL reference for {cell}")
                bol_source_path = (
                    ROOT
                    / "data"
                    / "raw"
                    / "oxford"
                    / "part1"
                    / baseline["source_file"]
                )
                denominator = _number(baseline["capacity_raw"])
                if not bol_source_path.is_file() or denominator is None or denominator <= 0:
                    raise ValueError(f"Invalid selected BoL reference for {cell}")
                if _number(part1_label["reference_capacity_raw"]) != denominator:
                    raise ValueError(f"BoL denominator mismatch for {cell} {measurement_id}")
                step4_status = "verified_raw_segment"
                step4_alignment = (
                    "Guide single checkpoint and exact Step 4 segment verified"
                )
                step4_rows = str(evidence["rows"])
                step4_capacity_rows = str(evidence["capacity_finite_rows"])
                step4_span_value = float(evidence["capacity_span"])
                step4_span = f"{step4_span_value:.12f}"
                step4_current_signs = evidence["current_signs"]
                step4_step_ids = evidence["step_ids"]
                bol_capacity = f"{denominator:.12f}"
                bol_source = baseline["source_file"]
                bol_status = (
                    "verified selected Part 1 baseline; provisional policy, "
                    "protocol identity not independently established"
                )
                ratio = 100.0 * step4_span_value / denominator
                if not math.isfinite(ratio) or not 0 < ratio <= 120:
                    raise ValueError(
                        f"Out-of-range recovered SOH for {cell} {measurement_id}: {ratio}"
                    )
                capacity_ratio_reliable = "TRUE"
                candidate_soh = f"{ratio:.12f}"
                calculation = (
                    f"100 * {step4_span_value:.12f} / {denominator:.12f} "
                    f"= {candidate_soh}"
                )
                decision = "accepted"
                reason = (
                    "Accepted as a provisional Part 1 recovery: the Guide maps this "
                    "single checkpoint to the RPT, the exact Step 4 range contains "
                    "only Step 4 rows with negative current and finite monotonic "
                    "Amphr, and the denominator is the existing selected Part 1 "
                    "BoL reference. The source protocol still does not define this "
                    "as an official SOH label."
                )
            elif cell == "G1-Cell9" and rpt == "8":
                decision = "rejected"
                reason = (
                    "Rejected for label recovery/training: the existing 20-row "
                    "Step 4 segment has native Amphr span 0.007603714301 and "
                    "produces a 0.255867760941% ratio, below the established "
                    "50% review floor; the curve requires source verification."
                )
            elif cell == "G1-Cell15" and rpt == "10":
                decision = "rejected"
                reason = (
                    "Rejected: the existing audit found too few valid Step 4 "
                    "capacity rows to define a discharge-capacity span."
                )
            elif cell == "G1-Cell20" and rpt == "3":
                decision = "rejected"
                reason = (
                    "Rejected: the mapped source has no Step 4 discharge rows, "
                    "so no capacity-based SOH ratio can be calculated."
                )
            else:
                raise ValueError(
                    f"Unexpected missing Part 1 label candidate: {cell} {measurement_id}"
                )
        elif group_id == "5":
            if guide_row is None:
                raise ValueError(f"Missing Part 2 Guide procedure for {cell} {source_file}")
            segment_key = (cell, _normalized_path(_source_suffix(source_file)))
            procedure_segments = segment_by_source.get(segment_key, [])
            if procedure_segments:
                step4_status = "procedure-level Step 4 segments observed"
                procedure_step4_segment_count = str(len(procedure_segments))
                procedure_step4_capacity_spans = ";".join(
                    row["capacity_span"] for row in procedure_segments
                )
                procedure_step4_current_signs = ";".join(
                    sorted({row["current_signs"] for row in procedure_segments})
                )
            step4_status = (
                "RPT-level Step 4 not verified; "
                f"boundary audit={step4_boundary_status}"
            )
            step4_alignment = "unresolved compound Guide stage; no RPT-level assignment"
            if supplemental_row:
                existing_supplemental_soh = supplemental_row["soh_percent"]
                supplemental_provenance = (
                    "separate Group 5 supplemental extension; candidate assignment "
                    "within a bundled Profile+RPT procedure; not canonical"
                )
            bol_capacity = bol_row["reference_capacity_raw"] if bol_row else ""
            bol_source = bol_row["source_file"] if bol_row else ""
            bol_status = (
                "local BoL candidate recorded; Group 5 procedure/RPT mapping and "
                "denominator policy unresolved"
                if bol_row
                else "no established cell-specific BoL reference"
            )
            reason = (
                "Manual review required: the Guide stage bundles profile and one "
                "or more RPTs, so raw Step 4 segments cannot be assigned to this "
                "RPT without resolving the compound-stage mapping. Existing "
                "supplemental labels for this group are provisional and are not "
                "promoted."
            )
        elif group_id == "6":
            if guide_row is None:
                raise ValueError(f"Missing Part 2 Guide procedure for {cell} {source_file}")
            segment_key = (cell, _normalized_path(_source_suffix(source_file)))
            procedure_segments = segment_by_source.get(segment_key, [])
            if procedure_segments:
                step4_status = "procedure-level Step 4 segments observed"
                procedure_step4_segment_count = str(len(procedure_segments))
                procedure_step4_capacity_spans = ";".join(
                    row["capacity_span"] for row in procedure_segments
                )
                procedure_step4_current_signs = ";".join(
                    sorted({row["current_signs"] for row in procedure_segments})
                )
            step4_status = (
                "RPT-level Step 4 not verified; "
                f"boundary audit={step4_boundary_status}"
            )
            if supplemental_row:
                existing_supplemental_soh = supplemental_row["soh_percent"]
                supplemental_provenance = (
                    "separate Group 6 supplemental extension using a local BoL "
                    "candidate; not canonical"
                )
            bol_capacity = bol_row["reference_capacity_raw"] if bol_row else ""
            bol_source = bol_row["source_file"] if bol_row else ""
            bol_status = (
                "local BoL candidate recorded; Group 6 exclusion/reference policy "
                "not independently resolved"
                if bol_row
                else "no established cell-specific BoL reference"
            )
            if guide_row["measurement_mapping_status"] == "ambiguous_compound_stage":
                step4_alignment = "ambiguous compound Guide stage; RPT alignment unresolved"
                reason = (
                    "Manual review required: the Guide bundles multiple RPTs or "
                    "CCCV activity in this stage, and the individual discharge "
                    "curve cannot be assigned to this RPT without resolving that "
                    "mapping. The Group 6 exclusion policy remains in force."
                )
            else:
                step4_alignment = (
                    "single Guide checkpoint but multiple/uncertain Step 4 "
                    "segments and local denominator policy unresolved"
                )
                reason = (
                    "Manual review required: the Guide maps a single checkpoint, "
                    "but the boundary audit does not establish one unique Step 4 "
                    "discharge curve for the RPT. Group 6 is calendar-aging-only, "
                    "and its local BoL candidate and capacity-ratio labeling policy "
                    "have not been independently validated. Existing Group 6 "
                    "exclusion is preserved."
                )
        else:
            raise ValueError(f"Unexpected missing-label group: {group_id}")

        result.append(
            {
                "dataset_part": part,
                "group_id": group_id,
                "physical_cell_id": cell,
                "partition": item["partition"],
                "rpt_index": rpt,
                "measurement_id": measurement_id,
                "nominal_cycle_age": (
                    guide_row["nominal_cycle_age"] if guide_row else ""
                ),
                "cycle_age_status": (
                    guide_row["cycle_age_status"] if guide_row else ""
                ),
                "label_status_before_audit": item["label_status"],
                "guide_mapping_confidence": mapping_confidence,
                "guide_stage": guide_stage,
                "guide_source_mapping_status": guide_source_status,
                "source_file": source_file,
                "source_file_exists": "TRUE" if source_exists else "FALSE",
                "source_mapping_provenance": source_mapping_provenance,
                "step4_availability_status": step4_status,
                "step4_boundary_status": step4_boundary_status,
                "step4_rpt_alignment_status": step4_alignment,
                "step4_segment_ref": step4_ref,
                "step4_rows_observed": step4_rows,
                "step4_capacity_finite_rows": step4_capacity_rows,
                "step4_native_capacity_span": step4_span,
                "step4_current_signs": step4_current_signs,
                "step4_step_ids": step4_step_ids,
                "procedure_step4_segment_count": procedure_step4_segment_count,
                "procedure_step4_capacity_spans": procedure_step4_capacity_spans,
                "procedure_step4_current_signs": procedure_step4_current_signs,
                "bol_capacity_reference_raw": bol_capacity,
                "bol_capacity_source_file": bol_source,
                "bol_reference_status": bol_status,
                "capacity_ratio_reliable": capacity_ratio_reliable,
                "candidate_soh_percent": candidate_soh,
                "candidate_calculation": calculation,
                "candidate_decision": decision,
                "candidate_reason": reason,
                "existing_supplemental_soh_percent": existing_supplemental_soh,
                "existing_supplemental_capacity_raw": (
                    supplemental_row["capacity_raw"] if supplemental_row else ""
                ),
                "supplemental_step4_segment_index": (
                    supplemental_row["step4_segment_index"]
                    if supplemental_row
                    else ""
                ),
                "supplemental_validation_status": (
                    supplemental_row["validation_status"] if supplemental_row else ""
                ),
                "supplemental_label_provenance": supplemental_provenance,
            }
        )
    result.sort(
        key=lambda row: (
            row["partition"],
            row["physical_cell_id"],
            int(row["rpt_index"]),
        )
    )
    return result, observations


def _build_report(
    rows: list[dict[str, Any]], observations: list[dict[str, str]]
) -> str:
    decisions = Counter(row["candidate_decision"] for row in rows)
    by_partition: dict[str, Counter[str]] = {}
    by_group: dict[tuple[str, str], Counter[str]] = {}
    by_cell: dict[tuple[str, str, str], Counter[str]] = {}
    for row in rows:
        by_partition.setdefault(row["partition"], Counter())[
            row["candidate_decision"]
        ] += 1
        by_group.setdefault((row["dataset_part"], row["group_id"]), Counter())[
            row["candidate_decision"]
        ] += 1
        by_cell.setdefault(
            (row["partition"], row["group_id"], row["physical_cell_id"]), Counter()
        )[row["candidate_decision"]] += 1

    eligible_cells = {
        row["physical_cell_id"]
        for row in observations
        if row["partition"] in {"train", "validation"}
        and row["valid_soh_label"].strip().casefold() == "true"
    }
    recovered_cells = {
        row["physical_cell_id"]
        for row in rows
        if row["candidate_decision"] == "accepted"
    }
    new_cells = recovered_cells - eligible_cells
    accepted = [row for row in rows if row["candidate_decision"] == "accepted"]
    lines = [
        "# Phase A/B: SOH label recovery audit",
        "",
        "## Gate result",
        "",
        "The audit covers all 228 missing/invalid Train and Validation RPT labels. "
        f"It accepted {decisions['accepted']} provisional Part 1 labels, rejected "
        f"{decisions['rejected']} Part 1 candidates, and retained "
        f"{decisions['requires_manual_review']} Part 2 candidates for manual review.",
        "Phase C and all model training/routing are **not authorized by this audit**. "
        "Group 5 compound-stage identities and Group 6 BoL/exclusion policy remain "
        "unresolved; no Part 2 candidate was promoted.",
        "",
        "## Decision counts",
        "",
        "| Partition | Missing/invalid RPTs | Accepted | Rejected | Manual review |",
        "|---|---:|---:|---:|---:|",
    ]
    for partition in ("train", "validation"):
        counts = by_partition.get(partition, Counter())
        lines.append(
            f"| {partition} | {sum(counts.values())} | {counts['accepted']} | "
            f"{counts['rejected']} | {counts['requires_manual_review']} |"
        )
    lines += [
        "",
        "## Candidate disposition by group",
        "",
        "| Part | Group | RPTs | Accepted | Rejected | Manual review |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for (part, group), counts in sorted(by_group.items()):
        lines.append(
            f"| {part} | {group} | {sum(counts.values())} | "
            f"{counts['accepted']} | {counts['rejected']} | "
            f"{counts['requires_manual_review']} |"
        )
    lines += [
        "",
        "## Per-cell coverage",
        "",
        "| Partition | Group | Physical cell | RPTs | Accepted | Rejected | Manual review |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for (partition, group, cell), counts in sorted(by_cell.items()):
        lines.append(
            f"| {partition} | {group} | {cell} | {sum(counts.values())} | "
            f"{counts['accepted']} | {counts['rejected']} | "
            f"{counts['requires_manual_review']} |"
        )

    lines += [
        "",
        "## Accepted candidates",
        "",
        "| Cell | RPT | Partition | Step 4 rows | Native capacity span | BoL capacity | Candidate SoH (%) |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in accepted:
        lines.append(
            f"| {row['physical_cell_id']} | {row['rpt_index']} | "
            f"{row['partition']} | {row['step4_rows_observed']} | "
            f"{row['step4_native_capacity_span']} | "
            f"{row['bol_capacity_reference_raw']} | {row['candidate_soh_percent']} |"
        )
    lines += [
        "",
        "Both accepted candidates are G3-Cell14 Train observations. The Guide maps "
        "TPG3.17 to RPT11 and TPG3.18 to RPT12 as single checkpoints. The exact "
        "header-inclusive CSV ranges were re-read: RPT11 rows 518128-523985 "
        "(5,858 rows) and RPT12 rows 1329687-1335484 (5,798 rows). Each range "
        "contains only Step 4, negative current, finite voltage and Amphr, and "
        "monotonically increasing Amphr. The capacity ratio uses the existing "
        "selected G3-Cell14 BoL capacity 2.957566371659 from "
        "`Group_3/Group 3/TPG3_1_Cell14.csv`.",
        "",
        "The precise calculation is `100 × (max(Amphr) - min(Amphr)) / Q_BoL`: "
        "RPT11 uses 100 × 2.439932374134 / 2.957566371659 = "
        "82.497975278416%; RPT12 uses 100 × 2.414883866976 / "
        "2.957566371659 = 81.651045606845%. Native Amphr units remain unverified; "
        "the within-cell ratio cancels a shared scale but does not make the dataset "
        "label official. The denominator follows the existing user-approved "
        "provisional Part 1 policy.",
        "",
        "The Guide manifest retains `source_file_not_found` for these entries. "
        "The raw files are present at the exact paths in the canonical observation "
        "manifest under `data/raw/oxford/part1/`; their filenames match the Guide "
        "procedure IDs and physical-cell IDs, and the Guide single-checkpoint RPT "
        "IDs agree with the exact Step 4 segment references. The prior source "
        "lookup status is preserved rather than silently rewritten.",
        "",
        "## Rejected and withheld candidates",
        "",
        "- **G1-Cell9 RPT8 (Train):** 20 Step 4 rows yield a native capacity span "
        "of 0.007603714301 and a 0.255867760941% ratio. It remains below the "
        "existing 50% review floor and is not accepted.",
        "- **G1-Cell15 RPT10 (Validation):** too few valid Step 4 capacity rows.",
        "- **G1-Cell20 RPT3 (Train):** no Step 4 discharge rows.",
        "- **Group 5 (197 RPTs):** Guide entries bundle profile and RPT ranges. "
        "Procedure-level Step 4 segments and 12 supplemental candidate labels do "
        "not establish exact RPT-to-curve boundaries; all remain manual review.",
        "- **Group 6 (26 RPTs):** 13 Guide entries are single checkpoints but "
        "their boundary audit reports multiple Step 4 segments; 13 entries are "
        "compound stages. The supplemental 13 capacity-ratio labels rely on a "
        "local BoL candidate; Group 6 is calendar-aging-only and the existing "
        "exclusion/reference policy is unresolved. None are promoted.",
        "",
        "## Phase B gate and label comparison",
        "",
        f"- Newly eligible physical cells: **{len(new_cells)}**. G3-Cell14 already "
        "has other validated labels in the Train partition.",
        f"- Ambiguous RPT mappings requiring manual review: **{sum(1 for row in rows if row['guide_mapping_confidence'] == 'ambiguous_compound_stage')}**.",
        "- The two accepted candidates use the same Step 4 native Amphr-span ratio "
        "and the same selected BoL baseline policy as existing Part 1 labels. "
        "They are new rows only; no existing label, manifest, or artifact changed.",
        "- Part 2 Group 5/6 supplemental values remain separate, provisional "
        "evidence, not validated ground truth.",
        "- No model, checkpoint, split, PyDMA output, canonical label, or dashboard "
        "was modified; no training or external evaluation was run.",
        "",
        "## Reproducibility",
        "",
        "From the repository root, run:",
        "",
        "```powershell",
        "python experiments\\soh_forecast\\run_label_recovery_audit.py",
        "```",
        "",
        "Outputs are written only under "
        "`experiments/soh_forecast/bafuse_adaptive_v1/phase_a_b_label_audit_v1/`. "
        "The per-RPT reasons, evidence, and source records are in "
        "`missing_soh_label_audit.csv` and "
        "`recovered_soh_label_candidates.csv`.",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path = DEFAULT_OUTPUT) -> None:
    rows, observations = _build_rows()
    audit_rows = [{field: row[field] for field in AUDIT_FIELDS} for row in rows]
    _write_csv(output_dir / "missing_soh_label_audit.csv", audit_rows)
    _write_csv(output_dir / "recovered_soh_label_candidates.csv", audit_rows)
    report = _build_report(rows, observations)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "recovered_label_quality_report.md").write_text(
        report, encoding="utf-8"
    )
    decisions = Counter(row["candidate_decision"] for row in rows)
    print(
        f"Wrote {len(rows)} candidates: {decisions['accepted']} accepted, "
        f"{decisions['rejected']} rejected, "
        f"{decisions['requires_manual_review']} manual review."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit missing SOH labels (Phase A/B).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Versioned output directory (must not replace canonical labels).",
    )
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
