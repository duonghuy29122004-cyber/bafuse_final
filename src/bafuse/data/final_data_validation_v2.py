from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bafuse.features.soh_voltage import (
    FINAL_PARTITION_SIZES,
    FINAL_SPLIT_VERSION,
    load_split_manifest,
)


_AUDIT_DIR = Path("reports/final_mapping_label_audit")
_OUTPUT_DIR = Path("reports/bafuse_final_data_validation_v2")
_VERIFIED_MAPPING_STATUSES = {
    "verified",
    "verified_from_ordered_raw_blocks",
    "GUIDE_VERIFIED",
}
_PRIMARY_SOH_TABLES = {
    "1": "part1_earliest_bol",
    "2": "part2_candidates",
    "3": "part3_candidates",
}


def _read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _rpt_number(value: Any) -> str:
    match = re.search(r"\bRPT\s*(\d+)\b", str(value), re.IGNORECASE)
    return match.group(1) if match else ""


def _source_name(value: Any) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _normalized_mapping_status(row: pd.Series) -> str:
    provenance = str(row.get("identity_provenance", "")).casefold()
    status = str(row.get("final_mapping_status", ""))
    if "user_confirmed" in provenance:
        return "user_confirmed_identity_not_independently_verified"
    return status


def _overall_mapping_status(row: pd.Series) -> str:
    raw_status = row["raw_scan_mapping_status_v2"]
    manifest_status = row.get("observation_manifest_mapping_status", "")
    if pd.isna(manifest_status):
        manifest_status = ""
    step_statuses = (
        row.get("step4_status_exact", ""),
        row.get("step8_status_exact", ""),
        row.get("step10_status_exact", ""),
    )
    step_statuses = tuple("" if pd.isna(status) else status for status in step_statuses)
    if raw_status == "user_confirmed_identity_not_independently_verified":
        return raw_status
    if not manifest_status:
        return "needs_curve_audit"
    if "ambiguous_rpt" in {raw_status, manifest_status}:
        return "ambiguous_rpt"
    if (
        raw_status == "needs_curve_audit"
        or manifest_status == "needs_curve_audit"
        or "needs_curve_audit" in step_statuses
    ):
        return "needs_curve_audit"
    if "missing_step" in {raw_status, manifest_status, *step_statuses}:
        return "missing_step"
    if raw_status in _VERIFIED_MAPPING_STATUSES:
        if manifest_status and manifest_status not in _VERIFIED_MAPPING_STATUSES:
            return "needs_curve_audit"
        return raw_status
    return raw_status


def _status_for_soh(row: pd.Series, mapping_status: str) -> str:
    if not _truth(row.get("source_exists")):
        return "MISSING_SOURCE"
    if row.get("audit_status") == "step4_segment_unavailable":
        return "UNVERIFIABLE"
    if row.get("audit_status") != "verified_source_and_arithmetic":
        return "UNVERIFIABLE"
    if row.get("dataset_part") != "1":
        return "UNVERIFIABLE_BOL_POLICY_UNRESOLVED"
    if row.get("label_table") != _PRIMARY_SOH_TABLES["1"]:
        return "QC_OR_LEGACY_ONLY"
    if not _rpt_number(row.get("measurement_id")):
        return "BOL_REFERENCE_NOT_TRAINING_LABEL"
    if mapping_status not in _VERIFIED_MAPPING_STATUSES:
        return "UNVERIFIABLE_MAPPING"
    return "VERIFIED_DERIVED_LABEL"


def _write_report(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _make_mapping_outputs(
    root: Path,
    output: Path,
    mapping: pd.DataFrame,
    split: pd.DataFrame,
    rpt_observations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignment = dict(zip(split["physical_cell_id"], split["partition"]))
    validation = mapping.copy()
    exact_columns = [
        "part",
        "physical_cell_id",
        "rpt_index",
        "mapping_status",
        "step4_status",
        "step8_status",
        "step10_status",
        "step4_segment_ref",
        "step8_segment_ref",
        "step10_segment_ref",
        "ready_soh",
        "ready_degradation",
        "audit_notes",
    ]
    exact = rpt_observations[exact_columns].rename(
        columns={
            "mapping_status": "observation_manifest_mapping_status",
            "step4_status": "step4_status_exact",
            "step8_status": "step8_status_exact",
            "step10_status": "step10_status_exact",
            "step4_segment_ref": "step4_segment_ref_exact",
            "step8_segment_ref": "step8_segment_ref_exact",
            "step10_segment_ref": "step10_segment_ref_exact",
            "ready_soh": "observation_manifest_ready_soh",
            "ready_degradation": "observation_manifest_ready_degradation",
            "audit_notes": "observation_manifest_audit_notes",
        }
    )
    validation = validation.merge(
        exact,
        on=["part", "physical_cell_id", "rpt_index"],
        how="left",
        validate="many_to_one",
    )
    validation["partition_v2"] = validation["physical_cell_id"].map(assignment).fillna(
        "excluded_not_in_v2_inventory"
    )
    validation["split_version_v2"] = FINAL_SPLIT_VERSION
    validation["raw_scan_mapping_status_v2"] = validation.apply(
        _normalized_mapping_status, axis=1
    )
    validation["mapping_status_v2"] = validation.apply(_overall_mapping_status, axis=1)
    validation["source_exists_bool"] = validation["source_exists"].map(_truth)
    validation["step4_valid_bool"] = validation["step4_valid"].map(_truth)
    validation["step8_valid_bool"] = validation["step8_valid"].map(_truth)
    validation["step10_valid_bool"] = validation["step10_valid"].map(_truth)
    validation.to_csv(output / "rpt_mapping_validation.csv", index=False, encoding="utf-8-sig")

    issue_rows = []
    for row in validation.to_dict(orient="records"):
        strict_verified = row["mapping_status_v2"] in _VERIFIED_MAPPING_STATUSES
        missing_curves = [
            step
            for step in (4, 8, 10)
            if row.get(f"step{step}_status_exact", "") != "verified"
        ]
        if strict_verified and not missing_curves:
            continue
        status = row["mapping_status_v2"]
        if status == "ambiguous_rpt":
            action = "Resolve RPT boundary against Guide and raw block evidence; keep excluded until resolved."
            severity = "high"
        elif status == "user_confirmed_identity_not_independently_verified":
            action = "Retain the user-confirmed identity provenance; do not describe it as independently Guide-verified."
            severity = "medium"
        elif missing_curves:
            action = "Review the listed Step curve(s) for this observation; do not substitute another RPT or cell."
            severity = "medium"
        else:
            action = "Preserve as unresolved and gather source/Guide evidence before using as a verified label."
            severity = "medium"
        issue_rows.append(
            {
                "part": row["part"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "source_tpg": row["source_tpg"],
                "candidate_rpt": row["rpt_index"],
                "issue_type": status if not strict_verified else "missing_step_" + "_".join(map(str, missing_curves)),
                "source_file": row["source_file"],
                "evidence": row["audit_notes"],
                "recommended_action": action,
                "severity": severity,
            }
        )
    issues = pd.DataFrame(issue_rows)
    issues.to_csv(output / "rpt_mapping_remaining_issues.csv", index=False, encoding="utf-8-sig")

    duplicates = int(
        validation.duplicated(["part", "physical_cell_id", "rpt_index"], keep=False).sum()
    )
    lines = [
        "# RPT mapping validation",
        "",
        "This versioned report reuses the raw-source scan in `reports/final_mapping_label_audit` and preserves unresolved mappings rather than inferring identities or RPT boundaries.",
        "",
        f"- Candidate RPT rows: {len(validation)}",
        f"- Duplicate `(part, physical_cell_id, RPT)` rows: {duplicates}",
        f"- Raw-scan rows with a valid Step 4 candidate: {int(validation.step4_valid_bool.sum())}",
        f"- Raw-scan rows with a valid Step 8 candidate: {int(validation.step8_valid_bool.sum())}",
        f"- Raw-scan rows with a valid Step 10 candidate: {int(validation.step10_valid_bool.sum())}",
        f"- RPT-manifest rows with verified Step 4 boundaries: {int(validation.step4_status_exact.eq('verified').sum())}",
        f"- RPT-manifest rows with verified Step 8 boundaries: {int(validation.step8_status_exact.eq('verified').sum())}",
        f"- RPT-manifest rows with verified Step 10 boundaries: {int(validation.step10_status_exact.eq('verified').sum())}",
        f"- RPT-manifest rows marked ready for degradation: {int(validation.observation_manifest_ready_degradation.map(_truth).sum())}",
        f"- Rows with unresolved or user-confirmed-only mapping/curve issues: {len(issues)}",
        "",
        "## Mapping status by dataset part",
        "",
        "| Part | Candidates | Strict verified | User-confirmed identity only | Ambiguous RPT | Needs curve audit | Missing step |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for part in ("1", "2", "3"):
        subset = validation[validation.part.eq(part)]
        counts = subset.mapping_status_v2.value_counts().to_dict()
        lines.append(
            f"| {part} | {len(subset)} | "
            f"{sum(counts.get(s, 0) for s in _VERIFIED_MAPPING_STATUSES)} | "
            f"{counts.get('user_confirmed_identity_not_independently_verified', 0)} | "
            f"{counts.get('ambiguous_rpt', 0)} | "
            f"{counts.get('needs_curve_audit', 0)} | "
            f"{counts.get('missing_step', 0)} |"
        )
    lines.extend(
        [
            "",
            "The historical review reported 164 outstanding mapping/curve issues across multiple audit scopes; that number is not the same denominator as the unresolved RPT rows above.",
            "",
            "G10-Cell2 is GUIDE_VERIFIED for the Group 10 procedure sequence and source files. The printed Guide omits the explicit Cell 2 label; the project-owner provenance note is retained.",
            "",
            "Raw-scan candidate validity does not override `needs_curve_audit` or ambiguous exact-boundary statuses in the RPT observation manifest. The prior G5-Cell6 TPG5.9/TPG5.10 RPT24 discrepancy remains recorded; no RPT boundary or mapping was changed by this report.",
        ]
    )
    _write_report(output / "rpt_mapping_final_report.md", "\n".join(lines))
    return validation, issues


def _make_soh_outputs(
    root: Path,
    output: Path,
    validation: pd.DataFrame,
    soh_audit: pd.DataFrame,
    g10_scan: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    map_status = {
        (row.part, row.physical_cell_id, str(row.rpt_index)): row.mapping_status_v2
        for row in validation.itertuples()
    }
    mapping_key = {
        (
            row.part,
            row.physical_cell_id,
            str(row.rpt_index),
        ): row
        for row in validation.itertuples()
    }
    comparison = soh_audit.copy()
    comparison["partition_v2"] = comparison["physical_cell_id"].map(
        dict(zip(validation.physical_cell_id, validation.partition_v2))
    ).fillna("excluded_not_in_v2_inventory")
    statuses = []
    for row in comparison.itertuples():
        rpt = _rpt_number(row.measurement_id)
        map_value = map_status.get((row.dataset_part, row.physical_cell_id, rpt), "")
        statuses.append(_status_for_soh(pd.Series(row._asdict()), map_value))
    comparison["validation_status_v2"] = statuses

    g10_rows = []
    for row in g10_scan[g10_scan.measurement_id.ne("BoL")].to_dict(orient="records"):
        rpt = str(row["measurement_id"])
        g10_rows.append(
            {
                "label_table": "part3_g10_cell2_raw_step4_scan",
                "dataset_part": "3",
                "physical_cell_id": "G10-Cell2",
                "measurement_id": f"RPT{rpt}",
                "source_file": row["source_file"],
                "mapping_status": "GUIDE_VERIFIED",
                "source_exists": "True",
                "raw_capacity_span_native": row["capacity_span_native_amphr"],
                "capacity_unit": "native Amphr; absolute unit unverified",
                "baseline_source_file": row["bol_source_file"],
                "raw_baseline_capacity_native": row["bol_capacity_span_native_amphr"],
                "recomputed_soh_percent": row["soh_percent_candidate"],
                "stored_soh_percent": "",
                "soh_abs_delta_pp": "",
                "audit_status": "guide_verified_mapping_derived_candidate_not_official",
                "label_is_official_ground_truth": "False",
                "audit_note": (
                    "Derived from direct Step 4 scan; Group 10 procedure sequence and "
                    "source files are verified. Group 10 Cell 2 identity confirmed by "
                    "project owner; explicit Cell 2 label omitted from printed Guide."
                ),
                "partition_v2": "external_test",
                "validation_status_v2": "GUIDE_VERIFIED_MAPPING_DERIVED_CANDIDATE",
            }
        )
    if g10_rows:
        comparison = pd.concat([comparison, pd.DataFrame(g10_rows)], ignore_index=True, sort=False)
    comparison.to_csv(output / "soh_label_comparison.csv", index=False, encoding="utf-8-sig")

    primary = comparison[
        comparison.label_table.eq(_PRIMARY_SOH_TABLES["1"])
        & comparison.validation_status_v2.eq("VERIFIED_DERIVED_LABEL")
    ].copy()
    primary.to_csv(output / "soh_labels_verified_candidates.csv", index=False, encoding="utf-8-sig")

    bol = soh_audit[
        soh_audit.label_table.isin(_PRIMARY_SOH_TABLES.values())
        & soh_audit.measurement_id.str.contains("BoL", case=False)
    ].copy()
    bol["reference_selection_status"] = bol.apply(
        lambda row: (
            "part1_provisional_earliest_complete_bol_policy"
            if row.dataset_part == "1"
            else "candidate_only_bol_policy_unresolved"
        ),
        axis=1,
    )
    g10_bol = g10_scan[g10_scan.measurement_id.eq("BoL")].copy()
    if not g10_bol.empty:
        g10_bol["dataset_part"] = "3"
        g10_bol["group_id"] = "10"
        g10_bol["physical_cell_id"] = "G10-Cell2"
        g10_bol["reference_selection_status"] = "guide_verified_owner_confirmed_identity_bol_candidate"
    bol = pd.concat([bol, g10_bol], ignore_index=True, sort=False)
    bol.to_csv(output / "bol_reference_verified.csv", index=False, encoding="utf-8-sig")

    part1_primary = comparison[
        comparison.label_table.eq(_PRIMARY_SOH_TABLES["1"])
        & comparison.validation_status_v2.eq("VERIFIED_DERIVED_LABEL")
    ]
    status_counts = comparison.validation_status_v2.value_counts().to_dict()
    report = [
        "# SoH validation status",
        "",
        f"- Historical label audit rows preserved: {len(soh_audit)}",
        f"- Part 1 primary-table derived labels passing source/arithmetic and strict mapping checks: {len(part1_primary)}",
        f"- Additional G10-Cell2 Step 4-derived RPT candidates: {len(g10_rows)} (GUIDE_VERIFIED mapping with owner-confirmed Cell 2 identity, external only)",
        f"- Official SOH ground-truth labels: 0; the dataset does not define an official SOH target.",
        "- `Amphr` remains native/unknown; a same-cell ratio does not verify its absolute unit.",
        "- Part 2 and Part 3 baseline policies remain unresolved for this validation version; arithmetic passing in the historical audit does not promote those labels to verified targets.",
        "",
        "## Versioned status counts",
        "",
        "| Status | Rows |",
        "|---|---:|",
    ]
    report.extend(f"| {status} | {count} |" for status, count in sorted(status_counts.items()))
    report.extend(
        [
            "",
            "The comparison table preserves existing values and appends the directly scanned G10-Cell2 candidates. No historical label file was modified.",
        ]
    )
    _write_report(output / "soh_validation_report.md", "\n".join(report))
    return comparison, bol


def _make_target_outputs(
    root: Path,
    output: Path,
    validation: pd.DataFrame,
    comparison: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    resolution = _read_csv(root, "reports/ambiguous_rpt_resolution.csv")
    resolution_by_key = {
        (row.part, row.physical_cell_id, str(row.rpt_index)): row
        for row in resolution.itertuples()
    }
    physics = _read_csv(
        root, "experiments/results/benchmarks/part1_physics_features/physics_features.csv"
    )
    physics_by_key = {
        (row.physical_cell_id, _source_name(row.source_file)): row
        for row in physics.itertuples()
    }
    aggregate = _read_csv(root, "experiments/results/hybrid_cnn1d_pydma/pydma_rpt_aggregated.csv")
    aggregate_by_key = {}
    for row in aggregate.itertuples():
        aggregate_by_key[
            (row.dataset_part, row.physical_cell_id, _rpt_number(row.rpt_id), row.direction)
        ] = row

    pydma_audit = _read_csv(root, "reports/final_mapping_label_audit/pydma_fit_audit.csv")
    accepted_count: dict[tuple[str, str, str], int] = {}
    for row in pydma_audit.itertuples():
        if row.result_table not in {"part1_part2_rmse15", "part3_rmse15"}:
            continue
        rpt = _rpt_number(row.checkup)
        if not rpt:
            continue
        key = (row.dataset_part, row.physical_cell_id, rpt)
        accepted_count[key] = accepted_count.get(key, 0) + int(_truth(row.diagnostic_label_accepted))

    soh_by_key = {}
    primary_soh = comparison[
        comparison.label_table.isin(_PRIMARY_SOH_TABLES.values())
    ]
    for row in primary_soh.itertuples():
        rpt = _rpt_number(row.measurement_id)
        if rpt:
            soh_by_key[(row.dataset_part, row.physical_cell_id, rpt)] = row

    availability_rows = []
    combined_rows = []
    for row in validation.to_dict(orient="records"):
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        resolution_row = resolution_by_key.get(key)
        physics_row = physics_by_key.get((row["physical_cell_id"], _source_name(row["source_file"])))
        soh_row = soh_by_key.get(key)
        charge = aggregate_by_key.get((row["part"], row["physical_cell_id"], str(row["rpt_index"]), "charge"))
        discharge = aggregate_by_key.get((row["part"], row["physical_cell_id"], str(row["rpt_index"]), "discharge"))
        step6_status = (
            resolution_row.step6_status if resolution_row is not None else "not_audited_for_all_sources"
        )
        physics_status = (
            physics_row.physics_feature_status
            if physics_row is not None
            else "not_available_in_existing_part1_feature_export"
            if row["part"] == "1"
            else "not_audited_for_this_part"
        )
        pydma_count = accepted_count.get(key, 0)
        soh_status = soh_row.validation_status_v2 if soh_row is not None else "no_existing_primary_label_row"
        ready_soh = (
            soh_status == "VERIFIED_DERIVED_LABEL"
            and _truth(row["step4_valid"])
            and row.get("step4_status_exact", "") == "verified"
            and row["mapping_status_v2"] in _VERIFIED_MAPPING_STATUSES
        )
        ready_degradation = (
            _truth(row.get("observation_manifest_ready_degradation", False))
            and row.get("step8_status_exact", "") == "verified"
            and row.get("step10_status_exact", "") == "verified"
            and row["mapping_status_v2"] != "ambiguous_rpt"
        )
        availability_rows.append(
            {
                "part": row["part"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "source_file": row["source_file"],
                "partition": row["partition_v2"],
                "mapping_status": row["mapping_status_v2"],
                "step4_available": row["step4_valid_bool"],
                "step8_available": row["step8_valid_bool"],
                "step10_available": row["step10_valid_bool"],
                "step4_boundary_status": row.get("step4_status_exact", ""),
                "step8_boundary_status": row.get("step8_status_exact", ""),
                "step10_boundary_status": row.get("step10_status_exact", ""),
                "step6_charge_status": step6_status,
                "physics_temp1_status": physics_status,
                "physics_temp1_available": physics_status == "available",
                "soh_label_status": soh_status,
                "ready_soh_derived_candidate": ready_soh,
                "ready_degradation_curve_candidate": ready_degradation,
                "historical_pydma_accepted_direction_records": pydma_count,
                "new_pydma_refit_status": "not_refit_in_this_run",
            }
        )
        combined_rows.append(
            {
                "part": row["part"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "source_file": row["source_file"],
                "partition": row["partition_v2"],
                "soh_percent_candidate": (
                    soh_row.recomputed_soh_percent if soh_row is not None else ""
                ),
                "soh_verification_status": soh_status,
                "historical_charge_lli_percent": (
                    charge.lli_estimated_pct if charge is not None else ""
                ),
                "historical_charge_lam_anode_percent": (
                    charge.lam_anode_estimated_pct if charge is not None else ""
                ),
                "historical_charge_lam_cathode_percent": (
                    charge.lam_cathode_estimated_pct if charge is not None else ""
                ),
                "historical_discharge_lli_percent": (
                    discharge.lli_estimated_pct if discharge is not None else ""
                ),
                "historical_discharge_lam_anode_percent": (
                    discharge.lam_anode_estimated_pct if discharge is not None else ""
                ),
                "historical_discharge_lam_cathode_percent": (
                    discharge.lam_cathode_estimated_pct if discharge is not None else ""
                ),
                "historical_pydma_status": (
                    "historical_model_derived_diagnostic_not_independent_ground_truth"
                    if charge is not None or discharge is not None
                    else "no_aggregate_record"
                ),
                "new_refit_labels": "not_available_full_refit_not_run",
            }
        )
    availability = pd.DataFrame(availability_rows)
    combined = pd.DataFrame(combined_rows)
    availability.to_csv(output / "bafuse_target_availability.csv", index=False, encoding="utf-8-sig")
    g10_scan = _read_csv(root, str(_OUTPUT_DIR / "g10_cell2_step4_raw_scan.csv"))
    g10_targets = []
    for row in g10_scan[g10_scan.measurement_id.ne("BoL")].to_dict(orient="records"):
        g10_targets.append(
            {
                "part": "3",
                "group": "10",
                "physical_cell_id": "G10-Cell2",
                "rpt_index": row["measurement_id"],
                "source_file": row["source_file"],
                "partition": "external_test",
                "soh_percent_candidate": row["soh_percent_candidate"],
                "soh_verification_status": "GUIDE_VERIFIED_MAPPING_DERIVED_CANDIDATE",
                "historical_charge_lli_percent": "",
                "historical_charge_lam_anode_percent": "",
                "historical_charge_lam_cathode_percent": "",
                "historical_discharge_lli_percent": "",
                "historical_discharge_lam_anode_percent": "",
                "historical_discharge_lam_cathode_percent": "",
                "historical_pydma_status": "no_cell_specific_aggregate_record",
                "new_refit_labels": "not_available_full_refit_not_run",
            }
        )
    if g10_targets:
        combined = pd.concat([combined, pd.DataFrame(g10_targets)], ignore_index=True, sort=False)
    combined.to_csv(
        output / "bafuse_multitarget_labels_verified.csv", index=False, encoding="utf-8-sig"
    )

    historical_comparison = aggregate.copy()
    historical_comparison["new_refit_status"] = "not_refit_full_run_not_run"
    historical_comparison["new_refit_lli_percent"] = ""
    historical_comparison["new_refit_lam_anode_percent"] = ""
    historical_comparison["new_refit_lam_cathode_percent"] = ""
    historical_comparison["comparison_scope"] = (
        "Historical aggregate diagnostic; no new RPT fit is available in this version."
    )
    historical_comparison.to_csv(
        output / "pydma_label_comparison.csv", index=False, encoding="utf-8-sig"
    )

    negative = pydma_audit[pydma_audit.diagnostic_label_accepted.map(_truth)].copy()
    numeric = ["lli_percent", "lam_anode_percent", "lam_cathode_percent"]
    negative_values = {
        column: pd.to_numeric(negative[column], errors="coerce")
        for column in numeric
    }
    mask = pd.concat(
        [values.lt(0) for values in negative_values.values()], axis=1
    ).any(axis=1)
    negative = negative.loc[mask].copy()
    negative["negative_targets"] = [
        ";".join(
            name
            for name, values in negative_values.items()
            if pd.notna(values.loc[index]) and values.loc[index] < 0
        )
        for index in negative.index
    ]
    negative["physical_plausibility_status"] = (
        "historical_negative_model_estimate_requires_manual_review_not_clipped"
    )
    negative["duplicate_scope_note"] = (
        "Rows span multiple historical result tables; deduplicate by cell/RPT/direction before counting unique fits."
    )
    negative.to_csv(output / "pydma_negative_label_audit.csv", index=False, encoding="utf-8-sig")
    return availability, combined


def _make_split_reports(
    root: Path,
    output: Path,
    split: pd.DataFrame,
    observations: pd.DataFrame,
    availability: pd.DataFrame,
    combined: pd.DataFrame,
) -> None:
    manifest = load_split_manifest(output / "cell_split_12train_4val_12external.csv")
    if manifest.version != FINAL_SPLIT_VERSION or manifest.partition_sizes != FINAL_PARTITION_SIZES:
        raise ValueError(f"Unexpected split: {manifest}")
    p12_train = sorted(
        cell
        for cell, partition in manifest.cell_to_partition.items()
        if cell.startswith(("G1-", "G2-", "G3-", "G4-", "G5-", "G6-"))
        and partition == "train"
    )
    p12_validation = sorted(
        cell
        for cell, partition in manifest.cell_to_partition.items()
        if cell.startswith(("G1-", "G2-", "G3-", "G4-", "G5-", "G6-"))
        and partition == "validation"
    )
    p3_external = sorted(
        cell for cell, partition in manifest.cell_to_partition.items()
        if cell.startswith(("G7-", "G8-", "G9-", "G10-")) and partition == "external_test"
    )
    if len(p12_train) != 12 or len(p12_validation) != 4 or len(p3_external) != 12:
        raise ValueError("The exact 12/4/12 cell allocation failed integrity checks.")
    if observations.groupby("physical_cell_id").partition.nunique().gt(1).any():
        raise ValueError("RPT observations for a physical cell leak across partitions.")
    lines = [
        "# Split integrity report",
        "",
        f"- Version: `{manifest.version}`",
        f"- Cell counts: {manifest.partition_sizes}",
        f"- Unique assigned cells: {len(manifest.cell_to_partition)}",
        "- Part 3 external cells: 12/12",
        "- Part 1/2 external cells: 0",
        "- Internal test cells: 0",
        "- Physical-cell partition overlap: none",
        f"- RPT candidate rows: {len(observations)}",
        f"- RPT rows excluded from assigned cells: {int(observations.partition.eq('excluded_not_in_v2_inventory').sum())}",
        "",
        "## Observation and target counts",
        "",
        "| Partition | RPT candidates | Step 4 candidates | Verified-derived SoH candidates | Degradation-ready observations | Historical PyDMA direction labels | Temp1 feature rows |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for partition in ("train", "validation", "external_test"):
        subset = availability[availability.partition.eq(partition)]
        label_subset = combined[combined.partition.eq(partition)]
        hist_labels = sum(
            label_subset[column].astype(str).str.strip().ne("").sum()
            for column in (
                "historical_charge_lli_percent",
                "historical_discharge_lli_percent",
            )
        )
        lines.append(
            f"| {partition} | {len(subset)} | "
            f"{int(subset.step4_available.map(_truth).sum())} | "
            f"{int(subset.ready_soh_derived_candidate.map(_truth).sum())} | "
            f"{int(subset.ready_degradation_curve_candidate.map(_truth).sum())} | "
            f"{int(hist_labels)} | "
            f"{int(subset.physics_temp1_available.map(_truth).sum())} |"
        )
    lines.extend(
        [
            "",
        ]
    )
    lines.extend(
        [
        "## Train (12)",
        "",
        ", ".join(p12_train),
        "",
        "## Validation (4)",
        "",
        ", ".join(p12_validation),
        "",
        "## External Test (12)",
        "",
        ", ".join(p3_external),
        "",
        "The 12 Part 1/2 Train and 4 Validation assignments are preserved from the existing 12/4/11 exploratory split. All 12 Part 3 cells, including GUIDE_VERIFIED G10-Cell2, remain in External Test; this supersedes the attachment's earlier 11-Part-3-plus-1-Part-1/2 proposal.",
        "",
        "The existing CNN1D checkpoint was trained on the historical 10/3/3 split. Its metrics are not metrics for this split. No model was trained, and no preprocessing was fitted using Validation or External Test.",
        ]
    )
    _write_report(output / "split_integrity_report.md", "\n".join(lines))


def _write_run_manifest(root: Path, output: Path, inputs: list[str]) -> None:
    input_hashes = {
        relative: _sha256(root / relative)
        for relative in inputs
        if (root / relative).is_file()
    }
    output_hashes = {
        path.name: _sha256(path)
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "run_manifest.json"
    }
    payload = {
        "status": "mapping_soh_split_outputs_generated_pydma_full_refit_not_run",
        "split_version": FINAL_SPLIT_VERSION,
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "limitations": [
            "The mapping layer reuses the versioned raw-source scan; unresolved mappings remain unresolved.",
            "Part 2 and Part 3 BoL/SoH policies remain provisional or unresolved.",
            "Step 6 and Temp1 availability are not fully audited for every RPT.",
            "Historical PyDMA aggregates are diagnostics; the full versioned refit is not run.",
        ],
    }
    (output / "run_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_reports(root: Path | None = None) -> None:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    output = repo / _OUTPUT_DIR
    output.mkdir(parents=True, exist_ok=True)
    mapping = _read_csv(repo, str(_AUDIT_DIR / "mapping_observation_audit.csv"))
    soh_audit = _read_csv(repo, str(_AUDIT_DIR / "soh_label_audit.csv"))
    rpt_observations = _read_csv(repo, "data/manifests/bafuse_rpt_observations.csv")
    observations = _read_csv(repo, str(_OUTPUT_DIR / "observation_split_12train_4val_12external.csv"))
    split = _read_csv(repo, str(_OUTPUT_DIR / "cell_split_12train_4val_12external.csv"))
    g10_scan = _read_csv(repo, str(_OUTPUT_DIR / "g10_cell2_step4_raw_scan.csv"))

    validation, issues = _make_mapping_outputs(repo, output, mapping, split, rpt_observations)
    comparison, _ = _make_soh_outputs(repo, output, validation, soh_audit, g10_scan)
    availability, combined = _make_target_outputs(repo, output, validation, comparison)
    _make_split_reports(repo, output, split, observations, availability, combined)

    audit_stats = _read_csv(repo, str(_AUDIT_DIR / "pydma_fit_audit.csv"))
    negative_records = _read_csv(
        repo, str(_OUTPUT_DIR / "pydma_negative_label_audit.csv")
    )
    negative_unique_keys = int(
        negative_records[
            ["dataset_part", "physical_cell_id", "checkup", "direction"]
        ].drop_duplicates().shape[0]
    )
    pilot_path = repo / _OUTPUT_DIR / "pydma_refit_results.csv"
    if pilot_path.is_file():
        pilot = _read_csv(repo, str(_OUTPUT_DIR / "pydma_refit_results.csv"))
        eligible = availability[
            availability.ready_degradation_curve_candidate.map(_truth)
        ]
        fit_count = 2 * (
            len(eligible) + eligible.physical_cell_id.nunique()
        )
        runtime = float(pilot.iloc[0]["runtime_seconds"])
        estimate_hours = fit_count * runtime / 3600
        pilot_report = [
            "# PyDMA refit quality report",
            "",
            "Status: **pilot only; full refit not run**.",
            "",
            f"- Pilot fit: {pilot.iloc[0]['physical_cell_id']} BoL, Step {pilot.iloc[0]['step']} {pilot.iloc[0]['direction']}, {pilot.iloc[0]['sample_count']} low-rate points.",
            f"- Result: RMSE {float(pilot.iloc[0]['rmse_mV']):.3f} mV; {pilot.iloc[0]['accepted_runs']}/{pilot.iloc[0]['total_runs']} accepted runs; <15 mV gate: {pilot.iloc[0]['rmse_gate_15mV']}; <10 mV sensitivity gate: {pilot.iloc[0]['rmse_gate_10mV']}.",
            f"- Runtime for this single fit: {runtime:.1f} seconds.",
            f"- Current exact-boundary readiness table contains {len(eligible)} degradation-ready RPTs across {eligible.physical_cell_id.nunique()} cells. A two-direction refit plus two-direction BoL reference fits would require approximately {fit_count} fits if all references are available.",
            f"- At the observed pilot rate, the nominal work estimate is {estimate_hours:.1f} hours. This is a rough estimate; it is not a full-run benchmark.",
            f"- Historical accepted records with negative estimates: LLI={int((pd.to_numeric(audit_stats.loc[audit_stats.diagnostic_label_accepted.map(_truth), 'lli_percent'], errors='coerce') < 0).sum())}, LAM-anode={int((pd.to_numeric(audit_stats.loc[audit_stats.diagnostic_label_accepted.map(_truth), 'lam_anode_percent'], errors='coerce') < 0).sum())}, LAM-cathode={int((pd.to_numeric(audit_stats.loc[audit_stats.diagnostic_label_accepted.map(_truth), 'lam_cathode_percent'], errors='coerce') < 0).sum())}; {len(negative_records)} audit records cover {negative_unique_keys} unique cell/RPT/direction keys across overlapping historical tables.",
            "- Negative estimates are retained, not clipped. Their physical plausibility remains unresolved; low RMSE alone does not establish identifiability.",
            "- This BoL-only pilot produces no degraded-RPT LLI/LAM labels. Historical aggregates are preserved for comparison, but new RPT refit values are blank.",
            "- Optimized pilot parameters and measured/reconstructed curves are stored in the versioned parameter CSVs and `pydma_pilot_reconstructed_curves.npz`.",
        ]
        _write_report(
            repo / _OUTPUT_DIR / "pydma_fit_quality_report.md",
            "\n".join(pilot_report),
        )
    summary = [
        "# BaFuse final data validation v2 — status",
        "",
        "This is a versioned audit, not a claim that the full dataset is ready for model training. Historical manifests, labels, PyDMA outputs, checkpoint, and scalers were not modified.",
        "",
        "## Mapping",
        "",
        f"- RPT candidates: {len(validation)}.",
        f"- Strict verified mapping statuses: {int(validation.mapping_status_v2.isin(_VERIFIED_MAPPING_STATUSES).sum())}.",
        f"- Remaining issue rows, including unresolved mappings and missing curves: {len(issues)}.",
        f"- Duplicate `(part, cell, RPT)` rows: {int(validation.duplicated(['part','physical_cell_id','rpt_index']).sum())}.",
        "- No unresolved RPT was silently reassigned.",
        "",
        "## SoH",
        "",
        f"- Historical label rows reviewed: {len(soh_audit)}.",
        f"- Verified-derived Part 1 candidates in this version: {int(comparison.validation_status_v2.eq('VERIFIED_DERIVED_LABEL').sum())}.",
        "- Official SoH ground truth: none established by the dataset.",
        "- Part 2/3 baseline policy and absolute `Amphr` units remain unresolved.",
        "- G10-Cell2 has a directly scanned Step 4 BoL and ten Step 4/Step 10-eligible RPT candidates; its Group 10 Guide sequence and source files are verified, with the project-owner identity note retained.",
        "",
        "## PyDMA",
        "",
        f"- Historical audit records across versions/tables: {len(audit_stats)}; these are not unique refit attempts.",
        f"- Historical rows with accepted diagnostic labels: {int(audit_stats.diagnostic_label_accepted.map(_truth).sum())}.",
        f"- Historical accepted records with negative LLI/LAM estimates: {len(negative_records)} rows across {negative_unique_keys} unique cell/RPT/direction keys; estimates were not clipped.",
        f"- Current versioned PyDMA fit artifacts present: {int((repo / _OUTPUT_DIR / 'pydma_refit_results.csv').is_file())}; see the fit report for pilot/full-run status.",
        "- PyDMA outputs remain model-derived diagnostics, not independent ground truth.",
        "",
        "## Split and readiness",
        "",
        f"- Split: `{FINAL_SPLIT_VERSION}` — 12 Train / 4 Validation / 12 External Test.",
        "- All 12 Part 3 cells are External Test by the later user decision; no Part 1/2 cell is external.",
        "- No model training was run. Existing checkpoint metrics remain tied to the historical split.",
        "- CNN1D Step 4 data are the most mature modality but label eligibility remains cell/RPT-specific.",
        "- Charge extraction is protocol-aware (Steps 6 for Parts 1–2 and Steps 5+6 for Part 3), but charge-encoder readiness is not established for all mapped observations.",
        "- Thermal encoder training is not ready: the existing Temp1 feature export covers only a subset of Part 1 and Part 3 temperature availability is incomplete.",
        "- See `bafuse_target_availability.csv`; unknown availability is not treated as missing or present.",
    ]
    _write_report(output / "final_validation_summary.md", "\n".join(summary))

    corrections = issues.copy()
    split_change = pd.DataFrame(
        [
            {
                "part": "3",
                "group": "10",
                "physical_cell_id": "G10-Cell2",
                "source_tpg": "TPG10",
                "candidate_rpt": "",
                "issue_type": "split_policy_update",
                "source_file": "data/raw/oxford/part3/Group_10/Group 10/TPG10.0-Cell2.csv",
                "evidence": "All 12 Part 3 cells remain External Test. The Group 10 procedure sequence and source files were verified; the printed Guide omits the Cell 2 label, whose identity was confirmed by the project owner.",
                "recommended_action": "Keep external-only; preserve GUIDE_VERIFIED and the exact project-owner identity provenance note.",
                "severity": "medium",
            }
        ]
    )
    corrections = pd.concat([corrections, split_change], ignore_index=True, sort=False)
    corrections.to_csv(output / "proposed_corrections.csv", index=False, encoding="utf-8-sig")

    inputs = [
        str(_AUDIT_DIR / "mapping_observation_audit.csv"),
        str(_AUDIT_DIR / "soh_label_audit.csv"),
        str(_AUDIT_DIR / "pydma_fit_audit.csv"),
        "data/manifests/canonical_observation_manifest_v3.csv",
        "data/manifests/bafuse_rpt_observations.csv",
        "data/manifests/splits/parts12-12-4-plus-part3-external-v1.csv",
        str(_OUTPUT_DIR / "g10_cell2_step4_raw_scan.csv"),
        "reports/ambiguous_rpt_resolution.csv",
        "experiments/results/benchmarks/part1_physics_features/physics_features.csv",
        "experiments/results/hybrid_cnn1d_pydma/pydma_rpt_aggregated.csv",
        "src/bafuse/data/final_data_validation_v2.py",
        "data/raw/oxford/part1/Group_1/Group 1/TPG1 - Cell 20.csv",
        "data/raw/oxford/part1/Half_Cell/Half Cell/CH_HC_anode.csv",
        "data/raw/oxford/part1/Half_Cell/Half Cell/CH_HC_cathode.csv",
        "data/manifests/source_file_availability.csv",
        "data/processed/label_soh_part1_2/run_007_part1_capacity_qc/labels_soh.csv",
        "data/processed/label_soh_part1_2/run_010_part2_soh_candidates/labels_soh_part2.csv",
        "experiments/results/data_audits/part3_soh_candidates/labels_soh_candidates.csv",
        "data/processed/label_pydma_fit_part1_2/pydma_results_rmse15.csv",
        "data/processed/label_pydma_fit_part3/run_009_part3_pydma/pydma_results.csv",
        "models/checkpoints/part1_cnn1d/cnn1d_checkpoint.pt",
    ]
    inputs.extend(g10_scan.source_file.drop_duplicates().tolist())
    _write_run_manifest(repo, output, inputs)


def _run_pydma_pilot(root: Path) -> None:
    from pydma import DMAAnalyzer, DMAConfig
    from pydma.electrodes.electrode import ElectrodeOCP
    from pydma.preprocessing.loader import load_ocp

    from bafuse.labeling.pydma_runner import _fit_one, _read_pocv_with_audit

    output = root / _OUTPUT_DIR
    dataset_root = root / "data/raw/oxford/part1"
    half_cell = dataset_root / "Half_Cell/Half Cell"
    source = dataset_root / "Group_1/Group 1/TPG1 - Cell 20.csv"
    curve, curve_audit = _read_pocv_with_audit(source, 10, "charge")
    if curve is None or len(curve_audit["segments"]) != 1:
        raise ValueError(f"Pilot baseline curve is not uniquely resolved: {curve_audit}")

    anode_ref = load_ocp(
        half_cell / "CH_HC_anode.csv",
        electrode_type="anode",
        soc_col="SoC",
        voltage_col="Volts",
    )
    cathode_ref = load_ocp(
        half_cell / "CH_HC_cathode.csv",
        electrode_type="cathode",
        soc_col="SoC",
        voltage_col="Volts",
    )
    started = time.perf_counter()
    result = _fit_one(
        "charge",
        curve[0],
        curve[1],
        anode_ref,
        cathode_ref,
        seed=211,
        context="versioned pilot G1-Cell20 BoL",
        use_cuda=True,
        maxiter=50,
        popsize=30,
        max_tries=10,
    )
    elapsed = time.perf_counter() - started
    historical_audit = _read_csv(root, str(_AUDIT_DIR / "pydma_fit_audit.csv"))
    accepted_history = historical_audit[
        historical_audit.diagnostic_label_accepted.map(_truth)
    ].copy()
    historical_values = {
        name: pd.to_numeric(accepted_history[name], errors="coerce")
        for name in ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
    }
    historical_negative_counts = {
        name: int(values.lt(0).sum())
        for name, values in historical_values.items()
    }
    any_negative_mask = pd.concat(
        [values.lt(0) for values in historical_values.values()], axis=1
    ).any(axis=1)
    any_negative_records = int(any_negative_mask.sum())
    any_negative_unique_keys = int(
        accepted_history.loc[
            any_negative_mask,
            ["dataset_part", "physical_cell_id", "checkup", "direction"],
        ].drop_duplicates().shape[0]
    )

    params = result.fitted_params.to_dict()
    config = result.config_snapshot
    lower = tuple(config.get("lower_bounds", (1.0, -1.0, 1.0, -1.0)))
    upper = tuple(config.get("upper_bounds", (2.0, 0.0, 2.1, 0.0)))
    core_values = [
        params["alpha_an"],
        params["beta_an"],
        params["alpha_ca"],
        params["beta_ca"],
    ]
    bounds = []
    for name, value, low, high in zip(
        ("alpha_an", "beta_an", "alpha_ca", "beta_ca"),
        core_values,
        lower,
        upper,
    ):
        bounds.append(
            {
                "parameter": name,
                "value": value,
                "lower_bound": low,
                "upper_bound": high,
                "at_or_near_bound": bool(
                    abs(value - low) <= 1e-6 or abs(value - high) <= 1e-6
                ),
            }
        )

    common = {
        "fit_id": "pilot-p1-g1-cell20-bol-charge",
        "scope": "pilot_only",
        "dataset_part": "1",
        "physical_cell_id": "G1-Cell20",
        "measurement_id": "BoL",
        "direction": "charge",
        "step": 10,
        "source_file": str(source.relative_to(root)).replace("\\", "/"),
        "half_cell_anode": str((half_cell / "CH_HC_anode.csv").relative_to(root)).replace("\\", "/"),
        "half_cell_cathode": str((half_cell / "CH_HC_cathode.csv").relative_to(root)).replace("\\", "/"),
        "curve_selection_status": curve_audit["status"],
        "curve_segment_count": len(curve_audit["segments"]),
        "sample_count": len(curve[0]),
        "rmse_mV": result.rmse * 1000,
        "accepted_runs": result.n_accepted_runs,
        "total_runs": result.n_total_runs,
        "is_accepted": result.is_accepted,
        "fit_status": result.status,
        "runtime_seconds": elapsed,
        "rmse_gate_15mV": bool(result.rmse * 1000 < 15 and result.n_accepted_runs >= 3),
        "rmse_gate_10mV": bool(result.rmse * 1000 < 10 and result.n_accepted_runs >= 3),
        "optimizer_config": json.dumps(config, sort_keys=True, default=str),
        "parameter_bound_diagnostics": json.dumps(bounds, sort_keys=True),
        "parameter_std": json.dumps(
            None if result.param_std is None else np.asarray(result.param_std).tolist()
        ),
        "fitted_parameters": json.dumps(params, sort_keys=True),
    }
    pd.DataFrame([common]).to_csv(
        output / "pydma_refit_results.csv", index=False, encoding="utf-8-sig"
    )
    parameter_row = {
        **{key: common[key] for key in ("fit_id", "scope", "dataset_part", "physical_cell_id", "measurement_id", "direction", "source_file")},
        **params,
        "optimizer_config": json.dumps(config, sort_keys=True, default=str),
        "parameter_bound_diagnostics": json.dumps(bounds, sort_keys=True),
        "parameter_std": common["parameter_std"],
    }
    pd.DataFrame([parameter_row]).to_csv(
        output / "pydma_optimized_parameters.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame([parameter_row]).to_csv(
        output / "pydma_bol_parameters.csv", index=False, encoding="utf-8-sig"
    )
    np.savez_compressed(
        output / "pydma_pilot_reconstructed_curves.npz",
        measured_capacity=curve[0],
        measured_voltage=curve[1],
        soc_measured=result.soc_measured,
        ocv_measured=result.ocv_measured,
        soc_reconstructed=result.soc_reconstructed,
        ocv_reconstructed=result.ocv_reconstructed,
    )

    fit_count = 2 * (301 + 28)
    projected_hours = fit_count * elapsed / 3600
    report = [
        "# PyDMA refit quality report",
        "",
        "Status: **pilot only; full refit not run**.",
        "",
        f"- Pilot fit: G1-Cell20 earliest Guide-mapped BoL, Step 10 charge, {len(curve[0])} low-rate points.",
        f"- Fit result: RMSE {result.rmse * 1000:.3f} mV; {result.n_accepted_runs}/{result.n_total_runs} accepted runs; gate <15 mV: {common['rmse_gate_15mV']}; sensitivity <10 mV: {common['rmse_gate_10mV']}.",
        f"- Runtime for this single fit: {elapsed:.1f} seconds.",
        f"- Estimated work at the observed single-fit runtime for 301 degradation-ready RPT candidates plus 28 BoL references, both directions: about {projected_hours:.1f} hours. This is a rough planning estimate, not a runtime guarantee.",
        "- The previous audit's 1,209 records span multiple historical tables and are not unique new-fit attempts.",
        f"- Historical accepted rows with negative estimates: LLI={historical_negative_counts['lli_percent']}, LAM-anode={historical_negative_counts['lam_anode_percent']}, LAM-cathode={historical_negative_counts['lam_cathode_percent']}; {any_negative_records} records represent {any_negative_unique_keys} unique cell/RPT/direction keys across overlapping tables.",
        "- No degraded-RPT labels are produced by a BoL-only pilot. The full PyDMA label comparison remains unavailable until exact-boundary refits complete.",
        "- Reconstructed measured/fitted voltage arrays and optimized parameters for the pilot are saved in versioned files; no historical PyDMA output was changed.",
        "- Accepted PyDMA estimates remain model-derived diagnostics; a low RMSE does not establish physical identifiability.",
    ]
    _write_report(output / "pydma_fit_quality_report.md", "\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate BaFuse final validation v2 outputs.")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--pydma-pilot", action="store_true")
    args = parser.parse_args()
    root = (args.repo_root or Path(__file__).resolve().parents[3]).resolve()
    if args.pydma_pilot:
        _run_pydma_pilot(root)
    else:
        generate_reports(root)


if __name__ == "__main__":
    main()
