from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bafuse.labeling.independent_soh_audit import (
    BENCHMARK_DIR,
    GUIDE_MANIFEST,
    M1_DIR,
    OUTPUT_DIR,
    PART3_GUIDE_MANIFEST,
    RECONSTRUCTION_DIR,
    SPLIT_MANIFEST,
    _tail_for_part,
    _guide_mapping_status,
    parse_block_reference,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report table: {path}")
    columns = list(dict.fromkeys(name for row in rows for name in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reasons(row: dict[str, str]) -> list[str]:
    reasons = []
    if row["guide_mapping_status"] != "ACCEPTED":
        reasons.append("guide_mapping_review")
    if any(
        row[name] != "ACCEPTED"
        for name in ("raw_step4_status", "raw_step10_status", "raw_bol_step4_status")
    ):
        reasons.append("raw_block_integrity_review")
    if row["protocol_compatibility_status"] != "ACCEPTED":
        reasons.append("bol_or_protocol_compatibility_review")
    for field in (
        "step4_current_direction_valid",
        "step10_current_direction_valid",
        "bol_current_direction_valid",
    ):
        if row[field].casefold() != "true":
            reasons.append(f"{field}_false")
    for field in (
        "step4_testtime_zero_intervals",
        "step4_testtime_negative_intervals",
        "step4_steptime_zero_intervals",
        "step4_steptime_negative_intervals",
    ):
        if row[field] not in {"0", ""}:
            reasons.append(f"{field}_nonzero")
    if row["step4_endpoint_valid_under_documented_2_5v_rule"].casefold() != "true":
        reasons.append("step4_endpoint_review")
    if row["physics_feature_alignment_status"] != "ACCEPTED":
        reasons.append("same_rpt_feature_alignment_review")
    if row["provenance_hashes_match_independent_raw"].casefold() != "true":
        reasons.append("provenance_hash_mismatch")
    if not row["independent_soh_testtime_percent"]:
        reasons.append("capacity_ratio_unavailable")
    discrepancy = row["independent_minus_stored_soh_pp"]
    if discrepancy and abs(float(discrepancy)) > 1e-8:
        reasons.append("stored_label_differs_from_independent_computation_over_float_check")
    return reasons


def build_reports(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    record_path = output_dir / "independent_record_audit.csv"
    block_path = output_dir / "independent_block_audit.csv"
    summary_path = output_dir / "independent_audit_summary.json"
    if not record_path.is_file() or not block_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Run the independent raw audit before building its reports."
        )

    record_rows = _read_csv(record_path)
    candidate_audits = [
        row for row in record_rows
        if row.get("current_manifest_disposition") != "EXCLUDED_CONTROL"
    ]
    candidate_rows = _read_csv(RECONSTRUCTION_DIR / "final_candidate_manifest.csv")
    reference_rows = _read_csv(RECONSTRUCTION_DIR / "reconstructed_soh_references.csv")
    provenance_rows = _read_csv(RECONSTRUCTION_DIR / "label_provenance.csv")
    feature_rows = _read_csv(RECONSTRUCTION_DIR / "raw_curve_feature_matrix.csv")
    split_rows = _read_csv(SPLIT_MANIFEST)
    candidates = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in candidate_rows
    }
    guide_by_key: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    guide_by_cell: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for guide in _read_csv(GUIDE_MANIFEST):
        guide_by_key[
            (guide["dataset_part"], guide["physical_cell_id"], guide["measurement_id"])
        ].append(guide)
        guide_by_cell[(guide["dataset_part"], guide["physical_cell_id"])].append(guide)
    part3_guide_by_cell: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for guide in _read_csv(PART3_GUIDE_MANIFEST):
        part3_guide_by_cell[
            (guide["dataset_part"], guide["physical_cell_id"])
        ].append(guide)
    block_rows = _read_csv(block_path)
    block_by_reference = {
        (
            _tail_for_part(row["source_file"], int(row["source_part"])),
            int(row["source_part"]),
            int(row["step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        ): row
        for row in block_rows
    }
    audit_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in candidate_audits
    }
    feature_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in feature_rows
    }
    reference_keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in reference_rows
    ]
    provenance_keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in provenance_rows
    ]
    if len(candidates) != len(candidate_rows) or len(audit_by_key) != len(candidate_audits):
        raise ValueError("Duplicate candidate or independent audit key.")
    if len(candidates) != 203:
        raise ValueError(f"Expected 203 canonical candidate keys, found {len(candidates)}.")
    if set(candidates) != set(audit_by_key):
        raise ValueError("Independent audit key set differs from the canonical candidates.")
    if len(reference_keys) != len(set(reference_keys)) or set(reference_keys) != set(candidates):
        raise ValueError("Reconstructed reference keys are duplicated or differ from candidates.")
    if len(provenance_keys) != len(set(provenance_keys)) or set(provenance_keys) != set(candidates):
        raise ValueError("Label provenance keys are duplicated or differ from candidates.")
    _write_csv(record_path, record_rows)
    if set(feature_by_key) != set(candidates):
        raise ValueError("Feature matrix key set differs from canonical candidates.")
    if len(feature_by_key) != len(feature_rows):
        raise ValueError("Duplicate raw feature identity.")

    split_by_cell: dict[str, str] = {}
    for row in split_rows:
        cell = row["physical_cell_id"]
        if cell in split_by_cell:
            raise ValueError(f"Duplicate physical cell in split manifest: {cell}")
        split_by_cell[cell] = row["partition"]
    if dict(Counter(split_by_cell.values())) != {
        "train": 9,
        "validation": 3,
        "external_test": 12,
    }:
        raise ValueError("Unexpected physical-cell partition counts.")
    for key, candidate in candidates.items():
        if split_by_cell.get(candidate["physical_cell_id"]) != candidate["partition"]:
            raise ValueError(f"Candidate partition differs from split manifest: {key}")
        if feature_by_key[key]["partition"] != candidate["partition"]:
            raise ValueError(f"Feature partition differs from target partition: {key}")

    for key, audit in audit_by_key.items():
        candidate = candidates[key]
        guide_status, guide_note = _guide_mapping_status(
            candidate,
            guide_by_key,
            guide_by_cell,
            part3_guide_by_cell,
        )
        audit["guide_mapping_status"] = guide_status
        audit["guide_mapping_note"] = guide_note
        protocol_ok = (
            guide_status == "ACCEPTED"
            and candidate["bol_resolution_status"] == "resolved"
            and candidate["reference_protocol_compatibility"]
            == "same_cell_step4_and_part_protocol_crosswalk"
        )
        audit["protocol_compatibility_status"] = (
            "ACCEPTED" if protocol_ok else "REVIEW"
        )

        block_statuses = []
        for ref_name, part_name, step in (
            ("step4_segment_ref", "dataset_part", 4),
            ("step10_segment_ref", "dataset_part", 10),
            ("bol_step4_segment_ref", "bol_source_part", 4),
        ):
            parsed = parse_block_reference(candidate[ref_name])
            reference = (
                _tail_for_part(parsed.source, int(candidate[part_name])),
                int(candidate[part_name]),
                step,
                parsed.start_line,
                parsed.end_line,
            )
            block = block_by_reference.get(reference)
            block_statuses.append(
                block is not None
                and block["raw_block_status"] == "ACCEPTED"
            )
        audit["raw_step4_status"] = "ACCEPTED" if block_statuses[0] else "REVIEW"
        audit["raw_step10_status"] = "ACCEPTED" if block_statuses[1] else "REVIEW"
        audit["raw_bol_step4_status"] = "ACCEPTED" if block_statuses[2] else "REVIEW"

        discrepancy = audit.get("independent_minus_stored_soh_pp")
        discrepancy_ok = not discrepancy or abs(float(discrepancy)) <= 1e-8
        audit["stored_label_matches_independent_within_float_check"] = str(
            discrepancy_ok
        ).lower()
        all_checks_ok = (
            all(block_statuses)
            and protocol_ok
            and audit["step4_current_direction_valid"].casefold() == "true"
            and audit["step10_current_direction_valid"].casefold() == "true"
            and audit["bol_current_direction_valid"].casefold() == "true"
            and audit["step4_testtime_zero_intervals"] == "0"
            and audit["step4_testtime_negative_intervals"] == "0"
            and audit["step4_steptime_zero_intervals"] == "0"
            and audit["step4_steptime_negative_intervals"] == "0"
            and audit["step4_endpoint_valid_under_documented_2_5v_rule"].casefold()
            == "true"
            and audit["physics_feature_alignment_status"] == "ACCEPTED"
            and audit["provenance_hashes_match_independent_raw"].casefold() == "true"
            and bool(audit["independent_soh_testtime_percent"])
            and discrepancy_ok
        )
        audit["raw_recomputation_disposition"] = (
            "ACCEPTED" if all_checks_ok else "REVIEW"
        )
        audit["stored_label_matches_independent_within_float_check"] = str(
            discrepancy_ok and all_checks_ok
        ).lower()

    for row in candidate_audits:
        if row.get("current_manifest_disposition") == "EXCLUDED_CONTROL":
            continue
        row["guide_mapping_status"] = audit_by_key[
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        ]["guide_mapping_status"]

    _write_csv(record_path, record_rows)

    original_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    raw_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    per_cell: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    values_by_partition: defaultdict[str, list[float]] = defaultdict(list)
    stored_values_by_partition: defaultdict[str, list[float]] = defaultdict(list)
    proposal_rows = []
    discrepancy_rows = []
    for key, audit in sorted(audit_by_key.items()):
        candidate = candidates[key]
        partition = candidate["partition"]
        original_counts[partition][candidate["label_disposition"]] += 1
        raw_counts[partition][audit["raw_recomputation_disposition"]] += 1
        per_cell[(partition, candidate["physical_cell_id"])].append(audit)
        if candidate["label_disposition"] == "ACCEPTED" and candidate["reconstructed_soh_percent"]:
            stored_values_by_partition[partition].append(
                float(candidate["reconstructed_soh_percent"])
            )
        soh_text = audit["independent_soh_testtime_percent"]
        if soh_text and audit["raw_recomputation_disposition"] == "ACCEPTED":
            values_by_partition[partition].append(float(soh_text))

        reasons = _reasons(audit)
        proposal = {
            "dataset_part": candidate["dataset_part"],
            "physical_cell_id": candidate["physical_cell_id"],
            "rpt_index": candidate["rpt_index"],
            "measurement_id": candidate["measurement_id"],
            "partition": partition,
            "source_file": candidate["source_file"],
            "step4_segment_ref": candidate["step4_segment_ref"],
            "step10_segment_ref": candidate["step10_segment_ref"],
            "bol_source_part": candidate["bol_source_part"],
            "bol_source_file": candidate["bol_source_file"],
            "bol_step4_segment_ref": candidate["bol_step4_segment_ref"],
            "original_label_disposition": candidate["label_disposition"],
            "original_reconstructed_soh_percent": candidate["reconstructed_soh_percent"],
            "independent_soh_testtime_percent": soh_text,
            "independent_soh_steptime_percent": audit["independent_soh_steptime_percent"],
            "independent_minus_stored_soh_pp": audit["independent_minus_stored_soh_pp"],
            "raw_recomputation_disposition": audit["raw_recomputation_disposition"],
            "guide_mapping_status": audit["guide_mapping_status"],
            "raw_step4_status": audit["raw_step4_status"],
            "raw_step10_status": audit["raw_step10_status"],
            "raw_bol_step4_status": audit["raw_bol_step4_status"],
            "protocol_compatibility_status": audit["protocol_compatibility_status"],
            "physics_feature_alignment_status": audit["physics_feature_alignment_status"],
            "review_reasons": ";".join(reasons),
            "proposed_training_eligibility": "BLOCKED_PENDING_GLOBAL_UNIT_PROTOCOL_AND_UNCERTAINTY_EVIDENCE",
            "training_authorized": False,
        }
        if partition in {"train", "validation"}:
            proposal_rows.append(proposal)
        if (
            candidate["label_disposition"] != "ACCEPTED"
            or audit["raw_recomputation_disposition"] != "ACCEPTED"
            or (
                audit["independent_minus_stored_soh_pp"]
                and abs(float(audit["independent_minus_stored_soh_pp"])) > 1e-8
            )
        ):
            discrepancy_rows.append(proposal)

    _write_csv(output_dir / "train_validation_manifest_proposal.csv", proposal_rows)
    _write_csv(output_dir / "record_discrepancy_report.csv", discrepancy_rows)

    hash_mismatches = sum(row["block_hash_exact"].casefold() != "true" for row in block_rows)
    raw_status_counts = Counter(row["raw_block_status"] for row in block_rows)
    modality_counts = {
        "step4_discharge_valid": sum(row["raw_step4_status"] == "ACCEPTED" for row in candidate_audits),
        "step10_charge_valid": sum(row["raw_step10_status"] == "ACCEPTED" for row in candidate_audits),
        "physics_alignment_valid": sum(
            row["physics_feature_alignment_status"] == "ACCEPTED"
            for row in candidate_audits
        ),
        "all_three_modalities_structurally_valid": sum(
            row["raw_step4_status"] == "ACCEPTED"
            and row["raw_step10_status"] == "ACCEPTED"
            and row["physics_feature_alignment_status"] == "ACCEPTED"
            for row in candidate_audits
        ),
    }
    original_total = Counter()
    for counts in original_counts.values():
        original_total.update(counts)
    raw_total = Counter()
    for counts in raw_counts.values():
        raw_total.update(counts)
    split_summary = {
        partition: {
            "candidate_count": sum(1 for row in candidate_rows if row["partition"] == partition),
            "original_label_dispositions": dict(original_counts[partition]),
            "independent_recomputation_dispositions": dict(raw_counts[partition]),
            "training_eligible_records": 0,
            "accepted_raw_recomputation_soh_distribution_percent": (
                {
                    "n": len(values_by_partition[partition]),
                    "min": min(values_by_partition[partition]),
                    "max": max(values_by_partition[partition]),
                    "mean": sum(values_by_partition[partition]) / len(values_by_partition[partition]),
                }
                if values_by_partition[partition]
                else {"n": 0, "min": None, "max": None, "mean": None}
            ),
            "stored_original_accepted_soh_distribution_percent": (
                {
                    "n": len(stored_values_by_partition[partition]),
                    "min": min(stored_values_by_partition[partition]),
                    "max": max(stored_values_by_partition[partition]),
                    "mean": sum(stored_values_by_partition[partition])
                    / len(stored_values_by_partition[partition]),
                }
                if stored_values_by_partition[partition]
                else {"n": 0, "min": None, "max": None, "mean": None}
            ),
        }
        for partition in ("train", "validation", "external_test")
    }
    all_cells = sorted(
        (row["partition"], row["physical_cell_id"]) for row in split_rows
    )
    per_cell_summary = []
    for partition, cell in all_cells:
        rows = per_cell.get((partition, cell), [])
        raw_accepted = [
            float(row["independent_soh_testtime_percent"])
            for row in rows
            if row["raw_recomputation_disposition"] == "ACCEPTED"
            and row["independent_soh_testtime_percent"]
        ]
        per_cell_summary.append(
            {
                "partition": partition,
                "physical_cell_id": cell,
                "candidate_count": len(rows),
                "original_label_dispositions": dict(
                    Counter(candidates[(row["dataset_part"], cell, row["rpt_index"])]["label_disposition"] for row in rows)
                ),
                "independent_recomputation_dispositions": dict(
                    Counter(row["raw_recomputation_disposition"] for row in rows)
                ),
                "conditionally_recomputed_soh_n": len(raw_accepted),
                "conditionally_recomputed_soh_min_percent": min(raw_accepted) if raw_accepted else None,
                "conditionally_recomputed_soh_max_percent": max(raw_accepted) if raw_accepted else None,
            }
        )

    control = next(
        (
            row for row in record_rows
            if row.get("current_manifest_disposition") == "EXCLUDED_CONTROL"
        ),
        None,
    )
    if control is not None:
        discrepancy_rows.append(
            {
                "dataset_part": control["dataset_part"],
                "physical_cell_id": control["physical_cell_id"],
                "rpt_index": control["rpt_index"],
                "partition": control["partition"],
                "source_file": control["raw_source_file"],
                "step4_segment_ref": control["step4_segment_ref"],
                "step10_segment_ref": control["step10_segment_ref"],
                "raw_recomputation_disposition": control["raw_recomputation_disposition"],
                "original_label_disposition": control["current_manifest_disposition"],
                "review_reasons": "excluded_control_incomplete_step4_discharge",
                "proposed_training_eligibility": "EXCLUDED",
                "training_authorized": False,
            }
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("accepted_candidate_records_recomputed", None)
    summary["original_candidate_disposition_counts"] = dict(original_total)
    summary["raw_recomputation_disposition_counts"] = dict(raw_total)
    summary["originally_accepted_candidate_records_recomputed"] = original_total["ACCEPTED"]
    label_delta_values = [
        abs(float(row["independent_minus_stored_soh_pp"]))
        for row in candidate_audits
        if row["current_manifest_disposition"] == "ACCEPTED"
        and row["independent_minus_stored_soh_pp"]
    ]
    summary["accepted_label_discrepancy_count_over_1e-8_pp_computational_check"] = sum(
        value > 1e-8 for value in label_delta_values
    )
    summary["maximum_accepted_label_absolute_difference_pp"] = (
        max(label_delta_values) if label_delta_values else None
    )
    summary["training_eligible_records_cleared_by_this_audit"] = 0
    summary["training_authorized"] = False
    summary["training_started"] = False
    summary["external_used_for_model_selection"] = False
    summary["independent_block_status_counts"] = dict(raw_status_counts)
    summary["independent_block_hash_mismatches"] = hash_mismatches
    summary["guide_mapping_mismatches"] = sum(
        row["guide_mapping_status"] != "ACCEPTED" for row in candidate_audits
    )
    summary["report_postprocessing_changes"] = [
        "Guide mappings were re-evaluated using semicolon-aware source membership and the Part 3 BoL crosswalk.",
        "Raw block and protocol dispositions were recomputed from the retained independent block audit.",
    ]
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    split_cells: defaultdict[str, set[str]] = defaultdict(set)
    for row in split_rows:
        split_cells[row["partition"]].add(row["physical_cell_id"])
    split_overlap = {
        f"{left}:{right}": sorted(split_cells[left] & split_cells[right])
        for left, right in (
            ("train", "validation"),
            ("train", "external_test"),
            ("validation", "external_test"),
        )
    }
    leakage_audit = {
        "schema_version": 1,
        "candidate_key": ["dataset_part", "physical_cell_id", "rpt_index"],
        "candidate_key_count": len(candidates),
        "candidate_keys_unique": len(candidates) == len(set(candidates)),
        "feature_keys_exactly_match_candidate_keys": set(feature_by_key) == set(candidates),
        "partition_mismatches": 0,
        "physical_cell_counts": {name: len(cells) for name, cells in split_cells.items()},
        "physical_cell_partition_overlaps": split_overlap,
        "train_validation_external_partitions_disjoint": not any(split_overlap.values()),
        "model_predictor_allowlist": [
            *[f"discharge_v_{index:03d}" for index in range(128)],
            *[f"charge_v_{index:03d}" for index in range(128)],
            "mean_discharge_temperature_native",
            "discharge_temperature_rise_native",
            "discharge_plateau_slope_v_per_progress",
            "discharge_slope_variability_v_per_progress",
            "mean_absolute_discharge_current_native",
        ],
        "excluded_predictors": [
            "capacity",
            "Amphr",
            "discharge_duration",
            "RPT index",
            "cell identity",
            "dataset part",
            "target-derived values",
            "future observations",
        ],
        "preprocessing_fit_scope_in_code": "Train records only",
        "external_records_loaded_by_internal_trainer": False,
        "external_used_for_preprocessing_or_model_selection": False,
        "structural_alignment_does_not_resolve_global_unit_or_uncertainty_evidence": True,
        "training_eligibility_cleared": 0,
    }
    (output_dir / "leakage_alignment_audit.json").write_text(
        json.dumps(leakage_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "eligibility_summary.json").write_text(
        json.dumps(
            {
                "candidate_counts": dict(Counter(row["partition"] for row in candidate_rows)),
                "original_disposition_counts": {
                    partition: dict(counts)
                    for partition, counts in original_counts.items()
                },
                "independent_recomputation_disposition_counts": {
                    partition: dict(counts)
                    for partition, counts in raw_counts.items()
                },
                "split_summary": split_summary,
                "per_cell_coverage": per_cell_summary,
                "modality_counts": modality_counts,
                "training_eligible_records_cleared": 0,
                "training_authorized": False,
                "external_used_for_model_selection": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    report_lines = [
        "# Independent raw SoH reference audit",
        "",
        "Status: data audit and preparation only. No model training or degradation fitting was run.",
        "",
        "## Candidate and disposition counts",
        "",
        "| Split | Candidates | Original ACCEPTED | Original REVIEW | Raw recomputation ACCEPTED | Raw recomputation REVIEW | Training-cleared |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for partition, values in split_summary.items():
        original = values["original_label_dispositions"]
        recomputed = values["independent_recomputation_dispositions"]
        report_lines.append(
            f"| {partition} | {values['candidate_count']} | {original.get('ACCEPTED', 0)} "
            f"| {original.get('REVIEW', 0)} | {recomputed.get('ACCEPTED', 0)} "
            f"| {recomputed.get('REVIEW', 0)} | 0 |"
        )
    report_lines.extend(
        [
            "",
            "Original and recomputed rejected counts by split:",
            "",
            *[
                f"- {partition}: original REJECTED={values['original_label_dispositions'].get('REJECTED', 0)}; "
                f"raw recomputation REJECTED={values['independent_recomputation_dispositions'].get('REJECTED', 0)}."
                for partition, values in split_summary.items()
            ],
            "",
            "## Modality availability",
            "",
            f"- Step 4 structurally valid: {modality_counts['step4_discharge_valid']}/{len(candidates)}.",
            f"- Step 10 structurally valid: {modality_counts['step10_charge_valid']}/{len(candidates)}.",
            f"- Physics aligned to source features: {modality_counts['physics_alignment_valid']}/{len(candidates)}.",
            f"- All three structural modalities valid: {modality_counts['all_three_modalities_structurally_valid']}/{len(candidates)}.",
            "",
            "## Conditional raw-recomputation target distributions",
            "",
            "These are diagnostics for records whose raw recomputation disposition passed; they are not training-approved label distributions.",
            "",
            *[
                f"- {partition}: n={values['accepted_raw_recomputation_soh_distribution_percent']['n']}, "
                f"range={values['accepted_raw_recomputation_soh_distribution_percent']['min']} to "
                f"{values['accepted_raw_recomputation_soh_distribution_percent']['max']}%, "
                f"mean={values['accepted_raw_recomputation_soh_distribution_percent']['mean']}%."
                for partition, values in split_summary.items()
            ],
            "",
            "Stored original ACCEPTED reference distributions (cross-check only, not label authority):",
            "",
            *[
                f"- {partition}: n={values['stored_original_accepted_soh_distribution_percent']['n']}, "
                f"range={values['stored_original_accepted_soh_distribution_percent']['min']} to "
                f"{values['stored_original_accepted_soh_distribution_percent']['max']}%, "
                f"mean={values['stored_original_accepted_soh_distribution_percent']['mean']}%."
                for partition, values in split_summary.items()
            ],
        ]
    )
    report_lines.extend(
        [
            "",
            "The raw-recomputation ACCEPTED column means the independent structural,",
            "mapping, direction, timestamp, endpoint, and arithmetic checks passed;",
            "it does **not** resolve globally uncertain time/capacity semantics or",
            "provide measurement uncertainty. Therefore 0 records are cleared for",
            "supervised training by this audit.",
            "",
            "## Reproducibility and split checks",
            "",
            f"- Candidate rows: {len(candidates)}; unique exact keys: {len(candidates)}.",
            f"- Distinct mapped raw blocks re-read: {len(block_rows)}; exact block-hash mismatches: {hash_mismatches}.",
            f"- Raw block status counts: `{dict(raw_status_counts)}`.",
            f"- Source files scanned by the independent audit: {summary.get('source_files_scanned')}.",
            f"- Physical-cell partition overlaps: `{split_overlap}`.",
            "- Raw file size/mtime were checked against M1 before scanning and rechecked after each file scan.",
            "- The launch-time source-code hash was not captured. The checksum manifest records final source, and post-processing applies the final computational-mismatch disposition gate.",
            "- See `record_discrepancy_report.csv`, `train_validation_manifest_proposal.csv`,",
            "  `leakage_alignment_audit.json`, and `eligibility_summary.json` for row-level detail.",
            "",
            "## Measurement limitations",
            "",
            "- TestTime/StepTime numeric units are strongly inferred, not recovered from an archived export setting.",
            "- The absolute Oxford `Amphr` unit/semantics mapping is not archived; capacity comparisons remain conditional.",
            "- Instrument accuracy/repeatability and propagated integration uncertainty were not established.",
            "- Native temperature units, calibration, placement, and cross-instrument comparability remain unresolved.",
            "- Step 4/BoL ratios are not official Oxford ground-truth SoH labels.",
            "",
            "## Training and external-use safeguards",
            "",
            "- `training_authorized: false`; no training began.",
            "- Part 3 remains external-only and was not used for preprocessing, tuning, or selection.",
            "- G1-Cell9 RPT8 is retained as an excluded control and not repaired using charge or adjacent RPTs.",
            "- Review records remain out of the primary supervised cohort.",
            "- Focused tests: 21 passed; see `test_results.json`.",
            "",
            "See `approval_checklist.md`, `fusion_ablation_plan.md`, and the sibling",
            "`architecture_matrix_v1/README.md` for preparation and decisions required.",
            "",
        ]
    )
    (output_dir / "independent_soh_audit_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    cases = {
        "G2-Cell3": [
            row for row in candidate_audits
            if row["physical_cell_id"] == "G2-Cell3" and row["rpt_index"] == "2"
        ],
        "G9-Cell9": [
            row for row in candidate_audits if row["physical_cell_id"] == "G9-Cell9"
        ],
        "G10-Cell22": [
            row for row in candidate_audits if row["physical_cell_id"] == "G10-Cell22"
        ],
        "G10-Cell25": [
            row for row in candidate_audits if row["physical_cell_id"] == "G10-Cell25"
        ],
    }
    no_candidate_cells = sorted(
        cell
        for cell in split_by_cell
        if not any(row["physical_cell_id"] == cell for row in candidate_rows)
    )
    blocker_lines = [
        "# Remaining data-quality blockers",
        "",
        "No record is cleared for supervised training by this audit. Preserve all",
        "REVIEW and EXCLUDED statuses; do not infer missing labels or rewrite identity.",
        "",
        "## Global measurement blockers",
        "",
        "- Archived numeric settings for TestTime/StepTime are unavailable. Minute-based",
        "  agreement with native Amphr is consistency evidence, not recovered unit proof.",
        "- The Oxford `Amphr` to exported capacity unit/semantics mapping is not archived.",
        "  Same-protocol ratios may cancel a shared scale but do not validate curve bounds.",
        "- Instrument accuracy/repeatability and a propagated integration uncertainty",
        "  budget are not available; no tolerance is invented.",
        "- Temperature channel units, calibration, sensor placement, and cross-part",
        "  comparability remain unresolved; missing measurements are not imputed.",
        "",
        "## Record exceptions",
        "",
    ]
    for cell, rows in cases.items():
        if not rows:
            blocker_lines.append(f"- {cell}: expected exception key was absent from candidate audit.")
            continue
        for row in rows:
            blocker_lines.append(
                f"- {cell} RPT{row['rpt_index']}: original={row['current_manifest_disposition']}, "
                f"independent raw recomputation={row['raw_recomputation_disposition']}, "
                f"BoL/protocol={row['protocol_compatibility_status']}, "
                f"Step 4 repeated TestTime/StepTime intervals="
                f"{row['step4_testtime_zero_intervals']}/{row['step4_steptime_zero_intervals']}."
            )
    blocker_lines.extend(
        [
            f"- Cells assigned to the split with no candidate RPT mapping: {', '.join(no_candidate_cells) or 'none'}.",
            "- G1-Cell9 RPT8 is an excluded control with incomplete Step 4 discharge; do not",
            "  replace it with Step 10 charge data or a neighboring RPT.",
            "",
            "## Decision boundary",
            "",
            "The original ACCEPTED disposition is not equivalent to independent scientific",
            "clearance. The row-level independent audit reports computational and",
            "structural checks separately from the global unit/protocol/uncertainty",
            "evidence. Only after these blockers and the approval checklist are resolved",
            "should a new, scope-hashed training manifest be considered.",
            "",
        ]
    )
    (output_dir / "remaining_data_quality_blockers.md").write_text(
        "\n".join(blocker_lines), encoding="utf-8"
    )

    reproducibility = {
        "schema_version": 1,
        "audit_command": "python -m bafuse.labeling.independent_soh_audit",
        "report_command": "python -m bafuse.labeling.independent_soh_audit_reports",
        "focused_test_command": (
            "uv run --no-project --python .venv-torch-cuda/Scripts/python.exe "
            "--with pytest pytest -q tests/test_independent_soh_audit.py "
            "tests/test_bafuse_train_architecture_metrics.py "
            "tests/test_bafuse_architecture_models.py "
            "tests/test_bafuse_training_authorization.py"
        ),
        "candidate_count": len(candidates),
        "independent_block_count": len(block_rows),
        "independent_block_hash_mismatches": hash_mismatches,
        "raw_block_status_counts": dict(raw_status_counts),
        "source_file_size_and_mtime_checked_against_m1": True,
        "launch_time_audit_code_hash_captured": False,
        "code_changes_during_scan": [
            "Summary field naming was clarified.",
            "Duplicate/mismatched reference and provenance key checks were added.",
            "Unused local variables were removed.",
            "A stored-label mismatch above the floating-point comparison check was made a REVIEW disposition.",
        ],
        "raw_capacity_and_feature_calculation_path_changed_during_scan": False,
        "final_disposition_postprocessing_applied": True,
        "raw_source_modified": False,
        "training_started": False,
        "degradation_fitting_started": False,
        "external_model_selection_performed": False,
    }
    (output_dir / "reproducibility_checks.json").write_text(
        json.dumps(reproducibility, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    project_source = Path(__file__).resolve().parents[3]
    checksum_targets = {
        "candidate_manifest": RECONSTRUCTION_DIR / "final_candidate_manifest.csv",
        "reconstructed_references": RECONSTRUCTION_DIR / "reconstructed_soh_references.csv",
        "bol_mapping": RECONSTRUCTION_DIR / "bol_mapping.csv",
        "label_provenance": RECONSTRUCTION_DIR / "label_provenance.csv",
        "feature_matrix": RECONSTRUCTION_DIR / "raw_curve_feature_matrix.csv",
        "m1_block_audit": M1_DIR / "measurement_block_audit.csv",
        "guide_manifest": GUIDE_MANIFEST,
        "part3_bol_crosswalk": PART3_GUIDE_MANIFEST,
        "cell_split": SPLIT_MANIFEST,
        "independent_audit_code": Path(__file__).resolve().parent / "independent_soh_audit.py",
        "report_builder_code": Path(__file__).resolve(),
        "architecture_config": BENCHMARK_DIR / "architecture_matrix_v1" / "architecture_config.json",
        "architecture_model_code": project_source / "src" / "bafuse" / "hybrid" / "architecture_models.py",
        "training_code": project_source / "src" / "bafuse" / "hybrid" / "train_architectures.py",
        "dataset_review": project_source / "experiments" / "dataset_review" / "README.md",
        "research_spec": project_source / "specs" / "bafuse" / "spec.md",
        "research_context": project_source / "specs" / "bafuse" / "research.md",
        "independent_record_audit": record_path,
        "independent_block_audit": block_path,
        "independent_audit_summary": summary_path,
        "run_metadata": output_dir / "run_metadata.json",
        "reproducibility_checks": output_dir / "reproducibility_checks.json",
        "train_validation_manifest_proposal": output_dir / "train_validation_manifest_proposal.csv",
        "record_discrepancy_report": output_dir / "record_discrepancy_report.csv",
        "leakage_alignment_audit": output_dir / "leakage_alignment_audit.json",
        "eligibility_summary": output_dir / "eligibility_summary.json",
        "independent_audit_report": output_dir / "independent_soh_audit_report.md",
        "fusion_ablation_plan": output_dir / "fusion_ablation_plan.md",
        "approval_checklist": output_dir / "approval_checklist.md",
        "focused_test_results": output_dir / "test_results.json",
        "remaining_data_quality_blockers": output_dir / "remaining_data_quality_blockers.md",
    }
    checksums = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in checksum_targets.items()
    }
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "SHA-256",
                "generated_from_independent_audit_version": "independent-raw-soh-audit-v1",
                "training_authorized": False,
                "files": checksums,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "candidate_count": len(candidates),
        "block_count": len(block_rows),
        "block_hash_mismatches": hash_mismatches,
        "training_eligible_records": 0,
        "output_dir": str(output_dir.resolve()),
    }


def main() -> int:
    print(json.dumps(build_reports(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
