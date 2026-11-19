from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


_OUTPUT = Path("reports/bafuse_part2_bol_resolution_v1")
_MAPPING = Path("reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv")
_BOL_AUDIT = Path("reports/bafuse_bol_negative_fix_v1/bol_procedure_curve_audit.csv")
_BOL_DECISIONS = Path("reports/bafuse_bol_sequence_resolution_v1/bol_reference_decisions.csv")
_SEQUENCES = Path("reports/bafuse_bol_sequence_resolution_v1/step_sequence_reconstruction.csv")
_STAGE1_BOL = Path("reports/bafuse_stage_validation_v1/stage1_bol_reference_audit.csv")
_SOH_AUDIT = Path("reports/bafuse_final_data_validation_v2/soh_labels_verified_candidates.csv")
_WORKLIST = Path("reports/bafuse_negative_correction_v2/negative_worklist.csv")
_P1_PILOT = Path("reports/bafuse_bol_sequence_resolution_v1/targeted_refit_attempts.csv")
_P2_RESULTS = Path("experiments/results/data_audits/part2_pydma_review/pydma_results.csv")
_P2_SUMMARY = Path("experiments/results/data_audits/part2_pydma_review/run_summary.json")


def _read_csv(root: Path, relative: Path) -> list[dict[str, str]]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(root: Path, relative: Path) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write a headerless audit table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _int(value: Any) -> int | None:
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _cell_group(cell: str) -> str:
    match = re.fullmatch(r"G(\d+)-Cell.+", cell)
    if not match:
        raise ValueError(f"Unexpected physical cell identifier: {cell!r}")
    return f"Group {match.group(1)}"


def _source_path(part: str, source: str) -> str:
    path = source.strip().replace("\\", "/")
    if path.casefold().startswith("data/raw/oxford/"):
        return path
    return f"data/raw/oxford/part{part}/{path}"


def _unique(
    rows: list[dict[str, str]], **criteria: str
) -> dict[str, str] | None:
    matches = [
        row
        for row in rows
        if all(str(row.get(key, "")) == value for key, value in criteria.items())
    ]
    if len(matches) > 1:
        raise ValueError(f"Non-unique audit key {criteria}: {len(matches)} rows")
    return matches[0] if matches else None


def _pydma_cell_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    group_number = _int(row.get("group_id"))
    cell_number = _int(row.get("cell_id"))
    checkup = row.get("checkup", "")
    rpt_number = 0 if checkup.strip().casefold() == "bol" else _int(checkup)
    if group_number is None or cell_number is None or rpt_number is None:
        raise ValueError(f"Cannot parse saved Part 2 fit key: {row}")
    return (
        f"G{group_number}-Cell{cell_number}",
        str(rpt_number),
        row.get("direction", ""),
        checkup,
    )


def _mapping_for(
    rows: list[dict[str, str]], part: str, cell: str, rpt: str
) -> dict[str, str]:
    row = _unique(rows, part=part, physical_cell_id=cell, rpt_index=rpt)
    if row is None:
        raise ValueError(f"Missing exact mapping row: Part {part} {cell} RPT{rpt}")
    return row


def _candidate_summary(value: str, step: int) -> list[dict[str, Any]]:
    candidates = json.loads(value or "[]")
    result = []
    for candidate in candidates:
        result.append(
            {
                "step": step,
                "segment_index": candidate.get(
                    "segment_index", candidate.get("step_segment_index")
                ),
                "subsegment_index": candidate.get("subsegment_index"),
                "source_row_start": candidate.get("source_row_start"),
                "source_row_end": candidate.get("source_row_end"),
                "sample_count": candidate.get("sample_count"),
                "capacity_span_native": candidate.get(
                    "capacity_span", candidate.get("capacity_span_native")
                ),
                "voltage_min_v": candidate.get("voltage_min"),
                "voltage_max_v": candidate.get("voltage_max"),
                "valid": candidate.get("valid"),
            }
        )
    return result


def _verify_curve(
    root: Path,
    mapping: dict[str, str],
    step: int,
    direction: str,
) -> dict[str, Any]:
    from bafuse.data.pydma_stage2 import _read_verified_curve, _resolve_source

    ref_column = f"step{step}_segment_ref_exact"
    status_column = f"step{step}_status_exact"
    if mapping.get(status_column) != "verified":
        raise ValueError(
            f"Exact Step {step} mapping is not verified for "
            f"{mapping.get('physical_cell_id')} RPT{mapping.get('rpt_index')}: "
            f"{mapping.get(status_column)}"
        )
    path = _resolve_source(root, mapping["part"], mapping["source_file"])
    _, _, audit = _read_verified_curve(
        root, path, step, direction, mapping[ref_column]
    )
    return audit


def _write_plot(path: Path, accepted_rows: list[dict[str, str]]) -> None:
    import matplotlib.pyplot as plt

    cells = sorted({_pydma_cell_key(row)[0] for row in accepted_rows})
    if not cells:
        raise ValueError("No accepted Part 2 diagnostic fits are available to plot.")
    figure, axes = plt.subplots(
        len(cells), 3, figsize=(15, max(3.2, 2.6 * len(cells))), squeeze=False
    )
    metrics = (
        ("LLI_percent_vs_BoL", "LLI (%)"),
        ("LAM_anode_percent_vs_BoL", "Anode LAM (%)"),
        ("LAM_cathode_percent_vs_BoL", "Cathode LAM (%)"),
    )
    for row_index, cell in enumerate(cells):
        cell_rows = [row for row in accepted_rows if _pydma_cell_key(row)[0] == cell]
        for col_index, (field, label) in enumerate(metrics):
            axis = axes[row_index][col_index]
            for direction in sorted({row["direction"] for row in cell_rows}):
                points = [
                    row
                    for row in cell_rows
                    if row["direction"] == direction and _float(row.get(field)) is not None
                ]
                points.sort(key=lambda row: _int(row["checkup"]) or -1)
                x = [_int(row["checkup"]) for row in points]
                y = [_float(row[field]) for row in points]
                if points:
                    axis.plot(x, y, marker="o", label=direction)
            axis.set_title(f"{cell} — {label}")
            axis.set_xlabel("Verified RPT order")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
            if row_index == 0 and col_index == 0:
                axis.legend()
    figure.suptitle(
        "Saved accepted Part 2 PyDMA diagnostics; directions remain separate",
        y=1.002,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def generate(
    root: Path | None = None,
    output: Path = _OUTPUT,
    overwrite_generated: bool = False,
) -> Path:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    output = Path(output)
    if output.is_absolute():
        output_path = output
    else:
        output_path = repo / output
    if output_path.exists() and not overwrite_generated:
        raise FileExistsError(f"Refusing to overwrite versioned report: {output_path}")
    expected_outputs = {
        "part2_cell_continuity_audit.csv",
        "part2_bol_reference_candidates.csv",
        "part2_bol_policy_decisions.csv",
        "cross_part_reference_comparison.csv",
        "blocked_negative_case_resolution.csv",
        "negative_fit_reassessment.csv",
        "targeted_pydma_attempts.csv",
        "selected_degradation_diagnostics.csv",
        "degradation_trend_audit.csv",
        "remaining_unresolved_cases.csv",
        "bafuse_integration_readiness.md",
        "final_part2_bol_resolution_report.md",
        "degradation_trends.png",
    }
    if output_path.exists():
        existing = {path.name for path in output_path.iterdir()}
        if not overwrite_generated or not existing.issubset(expected_outputs):
            raise FileExistsError(
                f"Refusing to modify report contents outside this generator: {output_path}"
            )

    mapping = _read_csv(repo, _MAPPING)
    bol_audit = _read_csv(repo, _BOL_AUDIT)
    bol_decisions = _read_csv(repo, _BOL_DECISIONS)
    sequences = _read_csv(repo, _SEQUENCES)
    stage1_bol = _read_csv(repo, _STAGE1_BOL)
    soh_audit = _read_csv(repo, _SOH_AUDIT)
    worklist = _read_csv(repo, _WORKLIST)
    p1_pilot = _read_csv(repo, _P1_PILOT)
    p2_results = _read_csv(repo, _P2_RESULTS)
    p2_summary = _read_json(repo, _P2_SUMMARY)

    mapping2 = [row for row in mapping if row.get("part") == "2"]
    mapping1 = [row for row in mapping if row.get("part") == "1"]
    cells2 = sorted({row["physical_cell_id"] for row in mapping2})
    cells1 = {row["physical_cell_id"] for row in mapping1}
    for row in mapping2:
        if not _bool(row.get("source_exists_bool", row.get("source_exists"))):
            raise FileNotFoundError(
                f"Part 2 manifest source is missing: {row.get('source_file')}"
            )
        path = repo / _source_path("2", row["source_file"])
        if not path.is_file():
            raise FileNotFoundError(f"Part 2 manifest source is missing: {path}")

    key_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in mapping2:
        key_counts[(row["part"], row["physical_cell_id"], row["rpt_index"])] += 1
    duplicate_keys = [key for key, count in key_counts.items() if count > 1]

    continuity_rows: list[dict[str, Any]] = []
    for cell in cells2:
        cell_rows1 = [row for row in mapping1 if row["physical_cell_id"] == cell]
        cell_rows2 = [row for row in mapping2 if row["physical_cell_id"] == cell]
        p1_rpts = [_int(row["rpt_index"]) for row in cell_rows1]
        p2_rpts = [_int(row["rpt_index"]) for row in cell_rows2]
        group = _cell_group(cell)
        continuation = group in {"Group 1", "Group 2", "Group 3", "Group 4"}
        order_status = (
            "documented_part1_part2_rpt_order"
            if continuation
            and p1_rpts
            and p2_rpts
            and max(p1_rpts) < min(p2_rpts)
            else "part2_only_cohort"
            if not continuation
            else "continuity_order_requires_review"
        )
        continuity_rows.append(
            {
                "dataset_part": "2",
                "group": group,
                "physical_cell_id": cell,
                "part1_cell_mapping_present": cell in cells1,
                "part1_last_mapped_rpt": max(p1_rpts) if p1_rpts else "",
                "part2_first_mapped_rpt": min(p2_rpts) if p2_rpts else "",
                "guide_readme_evidence": (
                    "Part 2 README describes Groups 1-4 as continuation after MoL; "
                    "per-cell Guides continue the RPT sequence."
                    if continuation
                    else "Part 2 Guide identifies a separate Group 5/6 local cohort."
                ),
                "continuity_status": order_status,
                "identity_limit": (
                    "Documented continuity; no independent serial-number record "
                    "was found, so this is not serial-level proof."
                    if continuation
                    else "No Part 1 physical-cell match; retain as a Part 2 cohort."
                ),
                "source_readme": "data/raw/oxford/part2/Readme.txt",
                "part1_guide": "data/raw/oxford/part1/Guide_to_Datafiles.xlsx",
                "part2_guide": "data/raw/oxford/part2/Guide_to_Datafiles_2.xlsx",
            }
        )

    p1_bol_by_cell: dict[str, dict[str, str]] = {}
    for cell in cells2:
        candidates = [
            row
            for row in stage1_bol
            if row.get("physical_cell_id") == cell
            and row.get("dataset_part") == "1"
            and _bool(row.get("bol_verified"))
        ]
        if candidates:
            p1_bol_by_cell[cell] = candidates[0]

    candidate_rows: list[dict[str, Any]] = []
    for cell, bol in p1_bol_by_cell.items():
        source = _source_path("1", bol["primary_bol_reference_file"])
        soh_rows = [
            row
            for row in soh_audit
            if row.get("dataset_part") == "1"
            and row.get("physical_cell_id") == cell
            and row.get("baseline_source_file")
        ]
        q0 = _float(soh_rows[0].get("raw_baseline_capacity_native")) if soh_rows else None
        candidate_rows.append(
            {
                "physical_cell_id": cell,
                "candidate_part": "1",
                "candidate_role": "original_lifetime_bol",
                "source_file": source,
                "step4_reference": bol.get("bol_step4_segment_ref", ""),
                "step4_capacity_span_native": q0 if q0 is not None else "",
                "step8_status": bol.get("bol_step8_status", ""),
                "step10_status": bol.get("bol_step10_status", ""),
                "reference_status": bol.get("bol_policy_status", ""),
                "source_evidence": (
                    "Stage 1 Part 1 BoL audit; earliest Guide-mapped source. "
                    "Step 4 Q0 is kept separate from PyDMA Step 8/10 reference."
                ),
            }
        )

    for row in bol_audit:
        if row.get("dataset_part") != "2":
            continue
        cell = row.get("physical_cell_id", "")
        step4 = _candidate_summary(row.get("step4_segments_json", ""), 4)
        step8 = _candidate_summary(row.get("step8_candidates_json", ""), 8)
        step10 = _candidate_summary(row.get("step10_candidates_json", ""), 10)
        source = row.get("source_file", "")
        complete = any(item.get("valid") for item in step8) and any(
            item.get("valid") for item in step10
        )
        candidate_rows.append(
            {
                "physical_cell_id": cell,
                "candidate_part": "2",
                "candidate_role": (
                    "local_bol_curve_candidate" if complete else "step4_only_qc_candidate"
                ),
                "source_file": source,
                "step4_reference": json.dumps(step4, sort_keys=True),
                "step4_capacity_span_native": (
                    step4[0].get("capacity_span_native", "") if step4 else ""
                ),
                "step8_status": row.get("step8_scan_status", ""),
                "step8_candidates": json.dumps(step8, sort_keys=True),
                "step10_status": row.get("step10_scan_status", ""),
                "step10_candidates": json.dumps(step10, sort_keys=True),
                "reference_status": (
                    (
                        "guide_labeled_unique_complete_local_bol"
                        if cell.startswith("G6-")
                        else "complete_curve_candidate_not_promoted"
                    )
                    if complete
                    else "Step4_only_QC_not_pOCV_reference"
                ),
                "source_evidence": (
                    "Direct raw-derived BoL procedure audit; repeated segments are "
                    "preserved and not treated as separate RPTs."
                ),
            }
        )

    for decision in bol_decisions:
        if decision.get("physical_cell_id") not in {"G2-Cell3", "G2-Cell4"}:
            continue
        cell = decision["physical_cell_id"]
        selected_id = decision.get("selected_sequence_id", "")
        alternatives = json.loads(decision.get("alternative_sequence_ids_json") or "[]")
        for sequence in sequences:
            if (
                sequence.get("dataset_part") != "1"
                or sequence.get("physical_cell_id") != cell
                or sequence.get("sequence_id") not in {selected_id, *alternatives}
            ):
                continue
            candidate_rows.append(
                {
                    "physical_cell_id": cell,
                    "candidate_part": "1",
                    "candidate_role": (
                        "selected_provisional_file_local_bol_sequence"
                        if sequence.get("sequence_id") == selected_id
                        else "alternative_file_local_bol_sequence"
                    ),
                    "source_file": sequence.get("source_file", ""),
                    "sequence_id": sequence.get("sequence_id", ""),
                    "step4_reference": sequence.get("step4_reference", ""),
                    "step4_capacity_span_native": sequence.get(
                        "step4_capacity_span_native", ""
                    ),
                    "step8_status": sequence.get("step8_status", ""),
                    "step8_reference": sequence.get("step8_reference", ""),
                    "step8_capacity_span_native": sequence.get(
                        "step8_capacity_span_native", ""
                    ),
                    "step10_status": sequence.get("step10_status", ""),
                    "step10_reference": sequence.get("step10_reference", ""),
                    "step10_capacity_span_native": sequence.get(
                        "step10_capacity_span_native", ""
                    ),
                    "reference_status": decision.get("selection_status", ""),
                    "source_evidence": decision.get("selection_basis", ""),
                }
            )

    policy_rows: list[dict[str, Any]] = []
    for cell in cells2:
        group = _cell_group(cell)
        bol = p1_bol_by_cell.get(cell)
        local_candidates = [row for row in bol_audit if row.get("physical_cell_id") == cell]
        local_full = [
            row
            for row in local_candidates
            if row.get("dataset_part") == "2"
            and int(row.get("step8_valid_candidate_count", "0") or 0) > 0
            and int(row.get("step10_valid_candidate_count", "0") or 0) > 0
        ]
        if group in {"Group 1", "Group 2", "Group 3", "Group 4"}:
            selected = (
                _source_path("1", bol["primary_bol_reference_file"]) if bol else ""
            )
            status = "documented_continuation_use_original_part1_bol"
            if cell in {"G2-Cell3", "G2-Cell4"}:
                status = "part1_origin_selected_step8_10_sequence_remains_provisional"
            rationale = (
                "Use the same physical cell's Part 1 BoL as the lifetime degradation "
                "origin. Part 2 README and Guide support continuation; no Part 2 local "
                "BoL is Guide-labeled for Groups 1-4."
            )
        elif group == "Group 5":
            selected = ""
            status = "local_part2_candidate_unresolved_not_promoted"
            rationale = (
                "The Guide labels local BoL procedures, but TPG5.1 has repeated "
                "curve candidates and later Step 10 segments are partial; TPG5.2 "
                "contains Step 4 only. Keep candidate sequences separate until "
                "protocol evidence selects one."
            )
        else:
            selected = local_full[0]["source_file"] if local_full else ""
            status = (
                "local_part2_bol_curve_verified_calendar_aging"
                if local_full
                else "local_part2_bol_unresolved"
            )
            rationale = (
                "Use the Guide-labeled local Part 2 BoL for this calendar-aging "
                "cohort only; do not infer cycle age or Part 1 continuity."
            )
        policy_rows.append(
            {
                "dataset_part": "2",
                "group": group,
                "physical_cell_id": cell,
                "reference_policy": (
                    "original_part1_bol" if group in {
                        "Group 1", "Group 2", "Group 3", "Group 4"
                    } else "part2_local_bol_candidate"
                ),
                "selected_reference_file": selected,
                "decision_status": status,
                "part1_reference_policy_status": (
                    bol.get("bol_policy_status", "") if bol else ""
                ),
                "evidence": rationale,
                "evidence_sources": (
                    "data/raw/oxford/part1/Readme.txt; "
                    "data/raw/oxford/part1/Guide_to_Datafiles.xlsx; "
                    "data/raw/oxford/part2/Readme.txt; "
                    "data/raw/oxford/part2/Guide_to_Datafiles_2.xlsx; "
                    "reports/bafuse_stage_validation_v1/stage1_bol_reference_audit.csv"
                ),
                "limitations": (
                    "Guide/README continuity is not serial-number evidence."
                    if group in {"Group 1", "Group 2", "Group 3", "Group 4"}
                    else "Do not promote a candidate without complete reference-policy evidence."
                ),
            }
        )

    cross_rows: list[dict[str, Any]] = []
    for cell in cells2:
        group = _cell_group(cell)
        p1 = p1_bol_by_cell.get(cell)
        local = [row for row in bol_audit if row.get("physical_cell_id") == cell]
        local = [row for row in local if row.get("dataset_part") == "2"]
        soh = [
            row
            for row in soh_audit
            if row.get("physical_cell_id") == cell and row.get("dataset_part") == "1"
        ]
        q0 = _float(soh[0].get("raw_baseline_capacity_native")) if soh else None
        cross_rows.append(
            {
                "physical_cell_id": cell,
                "group": group,
                "part1_original_bol_source": (
                    _source_path("1", p1["primary_bol_reference_file"]) if p1 else ""
                ),
                "part1_q0_capacity_span_native": q0 if q0 is not None else "",
                "part1_step4_status": p1.get("bol_step4_status", "") if p1 else "",
                "part1_pocv_reference_status": p1.get("bol_policy_status", "") if p1 else "",
                "part2_local_bol_candidates": "; ".join(
                    row.get("source_file", "") for row in local
                ),
                "part2_local_step4_spans_native": "; ".join(
                    str(
                        _candidate_summary(row.get("step4_segments_json", ""), 4)[0]
                        .get("capacity_span_native", "")
                    )
                    for row in local
                    if _candidate_summary(row.get("step4_segments_json", ""), 4)
                ),
                "step8_step10_completeness": "; ".join(
                    f"{row.get('source_file')}: "
                    f"Step8={row.get('step8_valid_candidate_count')},"
                    f"Step10={row.get('step10_valid_candidate_count')}"
                    for row in local
                ),
                "physical_interpretation": (
                    "Part 1 original-life origin; Part 2 observations are continuation."
                    if group in {"Group 1", "Group 2", "Group 3", "Group 4"}
                    else "Separate Part 2 cohort; local reference is not equivalent to original-life BoL."
                ),
                "comparison_limit": (
                    "Capacity spans are native Amphr values; do not call them Ah. "
                    "Do not mix original-BoL and later-local-reference degradation."
                ),
            }
        )

    p2_keyed: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in p2_results:
        cell, rpt, direction, _ = _pydma_cell_key(row)
        p2_keyed[(cell, rpt, direction)] = row
    current_policy_by_cell = {
        row["physical_cell_id"]: row for row in policy_rows
    }
    negative_rows: list[dict[str, Any]] = []
    for entry in worklist:
        if entry.get("dataset_part") != "2":
            continue
        cell = entry["physical_cell_id"]
        rpt = entry["rpt_index"]
        direction = entry["direction"]
        fit = p2_keyed.get((cell, rpt, direction), {})
        opposite = "charge" if direction == "discharge" else "discharge"
        counterpart = p2_keyed.get((cell, rpt, opposite), {})
        number = _int(rpt)
        previous = p2_keyed.get((cell, str(number - 1), direction), {}) if number else {}
        following = p2_keyed.get((cell, str(number + 1), direction), {}) if number else {}
        historical = json.loads(entry.get("original_degradation_values_json") or "[]")
        current_policy = current_policy_by_cell.get(cell, {})
        current_policy_status = current_policy.get("decision_status", "")
        part1_curve_status = current_policy.get("part1_reference_policy_status", "")
        provisional_reference = (
            "provisional" in current_policy_status.casefold()
            or "provisional" in part1_curve_status.casefold()
        )
        if entry.get("mapping_status") != "verified":
            classification = "mapping_unresolved"
        elif "candidate_unresolved" in current_policy_status:
            classification = "local_reference_policy_unresolved"
        elif provisional_reference:
            classification = "provisional_reference_fit_diagnostic"
        elif fit.get("diagnostic_label_status") == "diagnostic_estimate_accepted":
            classification = "accepted_fit_but_parameter_and_residual_audit_missing"
        else:
            classification = "raw_fit_not_accepted_under_current_reference_policy"
        p1_reference = p1_bol_by_cell.get(cell, {})
        baseline_source = (
            _source_path("1", p1_reference["primary_bol_reference_file"])
            if p1_reference
            else ""
        )
        recommendation = (
            "Retain only as diagnostic evidence; do not select by sign. Recover fit "
            "vectors/residuals and keep the baseline's provisional status explicit."
            if current_policy_status.startswith("documented_continuation")
            or "provisional" in current_policy_status.casefold()
            else "Resolve the candidate reference and source mapping before relabeling."
        )
        negative_rows.append(
            {
                "dataset_part": "2",
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "direction": direction,
                "source_file": entry.get("source_file", ""),
                "mapping_status": entry.get("mapping_status", ""),
                "historical_bol_policy_status": entry.get("bol_policy_status", ""),
                "current_reference_policy_status": current_policy_status,
                "part1_curve_policy_status": part1_curve_status,
                "selected_baseline_source_file": baseline_source,
                "step8_segment_ref": entry.get("step8_segment_ref", ""),
                "step10_segment_ref": entry.get("step10_segment_ref", ""),
                "saved_fit_status": fit.get("diagnostic_label_status", "not_found"),
                "saved_rmse_mV": fit.get("fit_rmse_mV", ""),
                "saved_accepted_runs": fit.get("accepted_runs", ""),
                "saved_total_runs": fit.get("total_runs", ""),
                "saved_lli_percent": fit.get("LLI_percent_vs_BoL", ""),
                "saved_lam_anode_percent": fit.get("LAM_anode_percent_vs_BoL", ""),
                "saved_lam_cathode_percent": fit.get("LAM_cathode_percent_vs_BoL", ""),
                "opposite_direction_status": counterpart.get("diagnostic_label_status", ""),
                "opposite_direction_rmse_mV": counterpart.get("fit_rmse_mV", ""),
                "opposite_direction_lam_anode_percent": counterpart.get(
                    "LAM_anode_percent_vs_BoL", ""
                ),
                "previous_saved_rpt": str(number - 1) if previous else "",
                "previous_saved_status": previous.get("diagnostic_label_status", ""),
                "previous_saved_lam_anode_percent": previous.get(
                    "LAM_anode_percent_vs_BoL", ""
                ),
                "next_saved_rpt": str(number + 1) if following else "",
                "next_saved_status": following.get("diagnostic_label_status", ""),
                "next_saved_lam_anode_percent": following.get(
                    "LAM_anode_percent_vs_BoL", ""
                ),
                "historical_attempt_signatures_json": json.dumps(
                    historical, sort_keys=True
                ),
                "parameter_vectors_saved": False,
                "reconstructed_residuals_saved": False,
                "reassessment": classification,
                "recommended_action": recommendation,
            }
        )

    p1_target_attempts = [
        row
        for row in p1_pilot
        if row.get("physical_cell_id") == "G2-Cell4"
        and row.get("rpt_index") == "6"
        and row.get("direction") == "discharge"
        and row.get("observation_type") == "RPT"
    ]
    p1_target_attempts.sort(key=lambda row: _int(row.get("seed")) or 0)
    if len(p1_target_attempts) != 3:
        raise ValueError("Expected the three saved G2-Cell4 Part 1 RPT6 pilot runs.")
    selected_p1 = min(p1_target_attempts, key=lambda row: float(row["rmse_mV"]))
    bounds = [json.loads(row["parameter_bounds_json"]) for row in p1_target_attempts]
    residuals = [json.loads(row["residual_summary_json"]) for row in p1_target_attempts]
    metric_spreads = {
        key: max(float(row[key]) for row in p1_target_attempts)
        - min(float(row[key]) for row in p1_target_attempts)
        for key in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
    }

    blocked_specs = [
        ("1", "G2-Cell4", "6"),
        ("2", "G2-Cell3", "9"),
        ("2", "G2-Cell4", "7"),
        ("2", "G2-Cell4", "9"),
    ]
    blocked_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    for part, cell, rpt in blocked_specs:
        mapped = _mapping_for(mapping, part, cell, rpt)
        curve_audits = {
            step: _verify_curve(
                repo, mapped, step, "discharge" if step == 8 else "charge"
            )
            for step in (8, 10)
        }
        p2_fit = (
            p2_keyed.get((cell, rpt, "discharge"), {}) if part == "2" else {}
        )
        if part == "1":
            status = "existing_pilot_reused_no_refit"
            fit_details = {
                "selected_seed": selected_p1.get("seed", ""),
                "selected_rmse_mV": selected_p1.get("rmse_mV", ""),
                "selected_accepted_runs": selected_p1.get("accepted_runs", ""),
                "selected_total_runs": selected_p1.get("total_runs", ""),
                "selected_lli_percent": selected_p1.get("lli_percent", ""),
                "selected_lam_anode_percent": selected_p1.get("lam_anode_percent", ""),
                "selected_lam_cathode_percent": selected_p1.get("lam_cathode_percent", ""),
                "selected_parameters_json": selected_p1.get(
                    "fitted_parameters_json", ""
                ),
                "selected_parameter_bounds_json": selected_p1.get(
                    "parameter_bounds_json", ""
                ),
                "selected_residual_summary_json": selected_p1.get(
                    "residual_summary_json", ""
                ),
                "seed_rmse_min_mV": min(float(row["rmse_mV"]) for row in p1_target_attempts),
                "seed_rmse_max_mV": max(float(row["rmse_mV"]) for row in p1_target_attempts),
                "label_seed_spreads_json": json.dumps(metric_spreads, sort_keys=True),
                "max_abs_residual_across_seeds_mV": max(
                    item["max_abs_residual_mV"] for item in residuals
                ),
                "all_parameters_within_bounds": all(
                    not item["exact_or_near_bound"] for group in bounds for item in group
                ),
            }
        else:
            status = "not_refit_reference_or_historical_fit_unresolved"
            fit_details = {
                "selected_seed": "",
                "selected_rmse_mV": p2_fit.get("fit_rmse_mV", ""),
                "selected_accepted_runs": p2_fit.get("accepted_runs", ""),
                "selected_total_runs": p2_fit.get("total_runs", ""),
                "selected_lli_percent": p2_fit.get("LLI_percent_vs_BoL", ""),
                "selected_lam_anode_percent": p2_fit.get(
                    "LAM_anode_percent_vs_BoL", ""
                ),
                "selected_lam_cathode_percent": p2_fit.get(
                    "LAM_cathode_percent_vs_BoL", ""
                ),
                "selected_parameters_json": "",
                "selected_parameter_bounds_json": "",
                "selected_residual_summary_json": "",
                "seed_rmse_min_mV": "",
                "seed_rmse_max_mV": "",
                "label_seed_spreads_json": "",
                "max_abs_residual_across_seeds_mV": "",
                "all_parameters_within_bounds": "",
            }
        blocked_rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "source_file": mapped["source_file"],
                "mapping_status": mapped.get("mapping_status_v2", ""),
                "step8_source_reference": mapped.get("step8_segment_ref_exact", ""),
                "step8_source_row_start": curve_audits[8]["source_row_start"],
                "step8_source_row_end": curve_audits[8]["source_row_end"],
                "step8_samples": curve_audits[8]["sample_count"],
                "step8_capacity_span_native": curve_audits[8]["capacity_span_native"],
                "step10_source_reference": mapped.get("step10_segment_ref_exact", ""),
                "step10_source_row_start": curve_audits[10]["source_row_start"],
                "step10_source_row_end": curve_audits[10]["source_row_end"],
                "step10_samples": curve_audits[10]["sample_count"],
                "step10_capacity_span_native": curve_audits[10]["capacity_span_native"],
                "baseline_reference_file": (
                    _source_path("1", p1_bol_by_cell[cell]["primary_bol_reference_file"])
                    if cell in p1_bol_by_cell
                    else ""
                ),
                "baseline_reference_status": (
                    p1_bol_by_cell[cell].get("bol_policy_status", "")
                    if cell in p1_bol_by_cell
                    else "unverified_part2_reference"
                ),
                "saved_fit_status": p2_fit.get("diagnostic_label_status", ""),
                "resolution_status": status,
                "evidence_note": (
                    "Saved pilot is reused; no automatic refit."
                    if part == "1"
                    else "Exact target Step 8/10 boundaries pass raw preflight, but historical "
                    "parameter/residual artifacts are absent and the Part 1 Group 2 baseline "
                    "curve choice remains provisional."
                ),
                **fit_details,
            }
        )
        attempt_rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "direction": "discharge",
                "status": status,
                "new_optimizer_attempts": 0,
                "existing_saved_rmse_mV": fit_details["selected_rmse_mV"],
                "existing_saved_accepted_runs": fit_details["selected_accepted_runs"],
                "existing_saved_total_runs": fit_details["selected_total_runs"],
                "reason": (
                    "Existing 3-seed pilot is reused; explicit instruction forbids automatic rerun."
                    if part == "1"
                    else "No new fit: the exact baseline Step 8 sequence is still provisional, "
                    "and historical fit vectors/residuals are unavailable for a valid comparison."
                ),
                "source_row_evidence": json.dumps(
                    {
                        "step8": [
                            curve_audits[8]["source_row_start"],
                            curve_audits[8]["source_row_end"],
                        ],
                        "step10": [
                            curve_audits[10]["source_row_start"],
                            curve_audits[10]["source_row_end"],
                        ],
                    },
                    sort_keys=True,
                ),
            }
        )

    accepted_p2 = [
        row
        for row in p2_results
        if row.get("diagnostic_label_status") == "diagnostic_estimate_accepted"
    ]
    accepted_p2.sort(
        key=lambda row: (
            _pydma_cell_key(row)[0],
            _pydma_cell_key(row)[2],
            _int(row.get("checkup")) or 0,
        )
    )
    diagnostic_rows: list[dict[str, Any]] = []
    for row in accepted_p2:
        cell, rpt, direction, _ = _pydma_cell_key(row)
        diagnostic_rows.append(
            {
                "dataset_part": "2",
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "direction": direction,
                "source_file": row.get("source_file", ""),
                "fit_status": row.get("diagnostic_label_status", ""),
                "baseline_reference_status": (
                    p1_bol_by_cell.get(cell, {}).get("bol_policy_status", "")
                    or (
                        "local_part2_guide_bol"
                        if _cell_group(cell) == "Group 6"
                        else "local_part2_candidate_unresolved"
                    )
                ),
                "rmse_mV": row.get("fit_rmse_mV", ""),
                "accepted_runs": row.get("accepted_runs", ""),
                "total_runs": row.get("total_runs", ""),
                "lli_percent": row.get("LLI_percent_vs_BoL", ""),
                "lam_anode_percent": row.get("LAM_anode_percent_vs_BoL", ""),
                "lam_cathode_percent": row.get("LAM_cathode_percent_vs_BoL", ""),
                "fit_parameters_saved": False,
                "residual_arrays_saved": False,
                "interpretation": "Accepted PyDMA fit diagnostic; not measured ground truth.",
            }
        )
    for row in p1_target_attempts:
        diagnostic_rows.append(
            {
                "dataset_part": "1",
                "physical_cell_id": "G2-Cell4",
                "rpt_index": "6",
                "direction": "discharge",
                "source_file": row.get("source_file", ""),
                "fit_status": row.get("fit_status", ""),
                "seed": row.get("seed", ""),
                "rmse_mV": row.get("rmse_mV", ""),
                "accepted_runs": row.get("accepted_runs", ""),
                "total_runs": row.get("total_runs", ""),
                "lli_percent": row.get("lli_percent", ""),
                "lam_anode_percent": row.get("lam_anode_percent", ""),
                "lam_cathode_percent": row.get("lam_cathode_percent", ""),
                "fitted_parameters_json": row.get("fitted_parameters_json", ""),
                "parameter_bounds_json": row.get("parameter_bounds_json", ""),
                "residual_summary_json": row.get("residual_summary_json", ""),
                "fit_parameters_saved": True,
                "residual_arrays_saved": True,
                "interpretation": (
                    "Existing bounded pilot; passes the primary numerical gate, "
                    "but remains a diagnostic under a provisional baseline."
                ),
            }
        )

    trend_rows: list[dict[str, Any]] = []
    by_cell_direction: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in accepted_p2:
        cell, _, direction, _ = _pydma_cell_key(row)
        by_cell_direction[(cell, direction)].append(row)
    for (cell, direction), rows in by_cell_direction.items():
        rows.sort(key=lambda row: _int(row.get("checkup")) or -1)
        previous: dict[str, str] | None = None
        for row in rows:
            rpt = _int(row.get("checkup"))
            prev_rpt = _int(previous.get("checkup")) if previous else None
            trend_rows.append(
                {
                    "dataset_part": "2",
                    "physical_cell_id": cell,
                    "direction": direction,
                    "rpt_index": rpt if rpt is not None else "",
                    "fit_status": row.get("diagnostic_label_status", ""),
                    "baseline_reference_status": (
                        p1_bol_by_cell.get(cell, {}).get("bol_policy_status", "")
                        or (
                            "local_part2_guide_bol"
                            if _cell_group(cell) == "Group 6"
                            else "local_part2_candidate_unresolved"
                        )
                    ),
                    "rmse_mV": row.get("fit_rmse_mV", ""),
                    "lli_percent": row.get("LLI_percent_vs_BoL", ""),
                    "lam_anode_percent": row.get("LAM_anode_percent_vs_BoL", ""),
                    "lam_cathode_percent": row.get("LAM_cathode_percent_vs_BoL", ""),
                    "previous_accepted_rpt": prev_rpt if prev_rpt is not None else "",
                    "rpt_gap": (rpt - prev_rpt) if rpt is not None and prev_rpt is not None else "",
                    "delta_lli_percent": (
                        float(row["LLI_percent_vs_BoL"])
                        - float(previous["LLI_percent_vs_BoL"])
                        if previous and _float(row.get("LLI_percent_vs_BoL")) is not None
                        else ""
                    ),
                    "delta_lam_anode_percent": (
                        float(row["LAM_anode_percent_vs_BoL"])
                        - float(previous["LAM_anode_percent_vs_BoL"])
                        if previous and _float(row.get("LAM_anode_percent_vs_BoL")) is not None
                        else ""
                    ),
                    "delta_lam_cathode_percent": (
                        float(row["LAM_cathode_percent_vs_BoL"])
                        - float(previous["LAM_cathode_percent_vs_BoL"])
                        if previous and _float(row.get("LAM_cathode_percent_vs_BoL")) is not None
                        else ""
                    ),
                    "trend_note": (
                        "Saved-fit diagnostic only; delta is between accepted records "
                        "and may span missing RPTs."
                    ),
                }
            )
            previous = row

    unresolved_rows: list[dict[str, Any]] = []
    for row in negative_rows:
        unresolved_rows.append(
            {
                "case": f"{row['physical_cell_id']} RPT{row['rpt_index']} {row['direction']}",
                "issue_type": row["reassessment"],
                "status": "unresolved_for_labeling",
                "source_file": row["source_file"],
                "evidence": (
                    f"Mapping={row['mapping_status']}; "
                    f"BoL={row['historical_bol_policy_status']}; "
                    f"fit={row['saved_fit_status']}; RMSE={row['saved_rmse_mV']}; "
                    f"accepted={row['saved_accepted_runs']}/{row['saved_total_runs']}."
                ),
                "recommended_action": row["recommended_action"],
            }
        )
    for cell in ("G2-Cell3", "G2-Cell4"):
        decision = _unique(bol_decisions, physical_cell_id=cell)
        if decision:
            unresolved_rows.append(
                {
                    "case": f"{cell} Part 1 BoL Step 8/10 reference",
                    "issue_type": "provisional_reference_sequence",
                    "status": "unresolved_for_pydma_promotion",
                    "source_file": decision.get("original_selected_source", ""),
                    "evidence": decision.get("selection_basis", ""),
                    "recommended_action": "Keep TPG2.1-SEQ1 provisional until source protocol selects among repeated sequences.",
                }
            )
    for row in bol_audit:
        if row.get("dataset_part") == "2" and row.get("physical_cell_id", "").startswith("G5-"):
            unresolved_rows.append(
                {
                    "case": f"{row['physical_cell_id']} local BoL candidate",
                    "issue_type": "local_bol_policy_or_sequence_unresolved",
                    "status": "candidate_only",
                    "source_file": row.get("source_file", ""),
                    "evidence": (
                        f"Step4 segments={row.get('step4_segment_count')}; "
                        f"Step8 valid={row.get('step8_valid_candidate_count')}; "
                        f"Step10 valid={row.get('step10_valid_candidate_count')}."
                    ),
                    "recommended_action": "Resolve complete RPT sequence and reference selection from Guide/procedure before PyDMA labels.",
                }
            )
    unresolved_rows.append(
        {
            "case": "Part 1–2 physical-cell identity for Groups 1–4",
            "issue_type": "no_serial_number_confirmation",
            "status": "documented_continuity_not_serial_verified",
            "source_file": "data/raw/oxford/part2/Readme.txt",
            "evidence": "Part 2 README says continuation; Guide per-cell RPT order continues; no independent serial-number evidence.",
            "recommended_action": "Retain the documented continuation mapping and disclose the identity evidence limit.",
        }
    )
    terminal_sources = (
        "data/raw/oxford/part2/Group_5/Group 5/TPG5.13EoL-Cell16.csv",
        "data/raw/oxford/part2/Group_5/Group 5/TPG5.13-Cell6.csv",
    )
    terminal_hashes = {}
    for source in terminal_sources:
        path = repo / Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Expected separate Group 5 source is missing: {path}")
        terminal_hashes[source] = _sha256(path)
    if terminal_hashes[terminal_sources[0]] == terminal_hashes[terminal_sources[1]]:
        raise ValueError("The two Group 5 terminal source files unexpectedly have identical content.")
    unresolved_rows.append(
        {
            "case": "P2 Group 5 Cell16 EoL and Cell6 TPG5.13 sources",
            "issue_type": "independent_source_files_not_mergeable",
            "status": "verified_distinct_source_identity",
            "source_file": "; ".join(terminal_sources),
            "evidence": json.dumps(terminal_hashes, sort_keys=True),
            "recommended_action": "Keep TPG5.13EoL-Cell16.csv and TPG5.13-Cell6.csv separate; do not merge or relabel.",
        }
    )

    availability_rows = [row for row in mapping2 if row.get("mapping_status_v2") == "verified"]
    soh_ready = [
        row for row in availability_rows if row.get("step4_status_exact") == "verified"
    ]
    degradation_ready = [
        row
        for row in availability_rows
        if row.get("step8_status_exact") == "verified"
        and row.get("step10_status_exact") == "verified"
    ]

    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(output_path / "part2_cell_continuity_audit.csv", continuity_rows)
    _write_csv(output_path / "part2_bol_reference_candidates.csv", candidate_rows)
    _write_csv(output_path / "part2_bol_policy_decisions.csv", policy_rows)
    _write_csv(output_path / "cross_part_reference_comparison.csv", cross_rows)
    _write_csv(output_path / "blocked_negative_case_resolution.csv", blocked_rows)
    _write_csv(output_path / "negative_fit_reassessment.csv", negative_rows)
    _write_csv(output_path / "targeted_pydma_attempts.csv", attempt_rows)
    _write_csv(output_path / "selected_degradation_diagnostics.csv", diagnostic_rows)
    _write_csv(output_path / "degradation_trend_audit.csv", trend_rows)
    _write_csv(output_path / "remaining_unresolved_cases.csv", unresolved_rows)
    _write_plot(output_path / "degradation_trends.png", accepted_p2)

    p2_accepted_count = len(accepted_p2)
    p2_status_counts: dict[str, int] = defaultdict(int)
    for row in p2_results:
        p2_status_counts[row.get("diagnostic_label_status", "missing_status")] += 1
    summary = (
        f"# Part 2 BoL and PyDMA resolution\n\n"
        f"## Decision summary\n\n"
        f"- Part 2 mapped physical cells: **{len(cells2)}**; Groups 1–4 cells with "
        f"Part 1 mapping: **{sum(cell in cells1 for cell in cells2)}**.\n"
        f"- Part 1–2 physical continuity: documented for Groups 1–4 and consistent "
        f"with Guide RPT order; no serial-number proof was found.\n"
        f"- Part 2 Group 1–4 degradation origin: same-cell original Part 1 BoL. "
        f"The Stage 1 Part 1 baseline selection remains provisional; Group 2 Cell 3/4 "
        f"also has unresolved repeated Step 8/10 sequences.\n"
        f"- Part 2 Group 5: local BoL curves remain candidates; no baseline promoted. "
        f"Group 6: local Guide BoL with complete Step 4/8/10 candidate; calendar aging.\n"
        f"- `TPG5.13EoL-Cell16.csv` and `TPG5.13-Cell6.csv` were checked as distinct "
        f"source files by SHA-256 and were not merged.\n"
        f"- Exact raw Step 8 and Step 10 boundaries were re-read for the four blocked "
        f"cases (Part 1 G2-Cell4 RPT6 and Part 2 G2-Cell3 RPT9, G2-Cell4 RPT7/RPT9).\n"
        f"- Existing Part 2 PyDMA result statuses: "
        f"{json.dumps(dict(p2_status_counts), sort_keys=True)}; accepted diagnostic "
        f"records: **{p2_accepted_count}**. These are not measured ground truth.\n"
        f"- New optimizer runs: **0**. RPT6's existing pilot was reused; no Part 2 "
        f"repeat fit was run because the selected Part 1 baseline curve policy remains "
        f"provisional and saved historical fit vectors/residuals are unavailable.\n"
        f"- Verified Part 2 mapping rows with Step 4: **{len(soh_ready)}**; rows with "
        f"verified Step 8+10: **{len(degradation_ready)}**. These are raw-curve "
        f"availability counts, not labeled training observations.\n"
        f"- Duplicate Part 2 observation keys in the input mapping: **{len(duplicate_keys)}**.\n\n"
        f"- Historical Part 2 run baseline policy: "
        f"{json.dumps(p2_summary.get('baseline_policy', {}), sort_keys=True)}.\n\n"
        f"## Existing G2-Cell4 Part 1 RPT6 pilot\n\n"
        f"The saved 3-seed pilot (211/212/213) remains accepted under the numerical "
        f"15 mV / 3-run gate: LLI **{selected_p1['lli_percent']}%**, anode LAM "
        f"**{selected_p1['lam_anode_percent']}%**, cathode LAM "
        f"**{selected_p1['lam_cathode_percent']}%**, selected RMSE "
        f"**{selected_p1['rmse_mV']} mV**. Parameter vectors are within the configured "
        f"bounds and seed spreads are recorded in the CSV. However, max absolute "
        f"residual is **{max(item['max_abs_residual_mV'] for item in residuals):.3f} mV** "
        f"and the reference sequence remains provisional; retain the result as a "
        f"diagnostic, not a promoted label.\n\n"
        f"## Interpretation constraints\n\n"
        f"- Step 4 capacity spans are native `Amphr`; absolute units are unverified.\n"
        f"- SoH Step 4 Q0 and PyDMA Step 8/10 half-cell reference systems remain separate.\n"
        f"- No labels, splits, model checkpoints, scalers, raw data, or historical fits "
        f"were modified.\n"
        f"- See `degradation_trends.png` for accepted saved Part 2 diagnostic trends. "
        f"Lines never imply missing RPTs were observed.\n"
    )
    (output_path / "final_part2_bol_resolution_report.md").write_text(
        summary, encoding="utf-8"
    )
    readiness = (
        "# BaFuse integration readiness — Part 2\n\n"
        "## Ready as raw inputs\n\n"
        f"- **SoH curve candidates:** {len(soh_ready)} Part 2 mapping rows have a "
        f"verified source and exact Step 4 boundary. This does not create or certify "
        f"a Part 2 SOH label.\n"
        f"- **Degradation curve candidates:** {len(degradation_ready)} rows have exact "
        f"Step 8 discharge and Step 10 charge boundaries. Curve pairs remain subject "
        f"to reference-policy and source-sequence checks.\n"
        f"- **Saved PyDMA diagnostics:** {p2_accepted_count} accepted Part 2 directional "
        f"records are available for audit; they are fitted diagnostics, not independent "
        f"ground truth.\n\n"
        "## Not ready for supervised label integration\n\n"
        "- Same-cell Part 1 is the documented Part 2 lifetime origin for Groups 1–4, "
        "but the selected Part 1 baseline curve policy remains provisional. Group 2 "
        "has additional repeated Step 8/10 sequences; Part 2 G2-Cell3 RPT9 and "
        "G2-Cell4 RPT7/RPT9 remain blocked for label promotion.\n"
        "- Group 5 local BoL/target RPT boundaries and candidate sequence policy remain "
        "unresolved; retain all source files separately.\n"
        "- Historical negative fits do not consistently preserve parameter vectors "
        "and reconstructed residuals. Do not select fits by sign or use PyDMA labels "
        "as measured truth.\n"
        "- Group 6 is a calendar-aging cohort. Keep calendar/storage exposure distinct "
        "from cycle aging; no cycle age is inferred here.\n\n"
        "## Kept unchanged\n\n"
        "Part 1 SOH labels and Step 4 reference, the locked 12/4/12 split, all historical "
        "labels/fits, model checkpoints, scalers, raw data, and Part 3 external-test "
        "isolation were not changed.\n"
    )
    (output_path / "bafuse_integration_readiness.md").write_text(
        readiness, encoding="utf-8"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the versioned Part 2 BoL provenance and PyDMA audit."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=_OUTPUT)
    parser.add_argument("--overwrite-generated", action="store_true")
    args = parser.parse_args()
    print(generate(args.root, args.output, args.overwrite_generated))


if __name__ == "__main__":
    main()
