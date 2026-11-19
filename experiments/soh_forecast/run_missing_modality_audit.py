from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PHASE2_DIR = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
AUDIT_DIR = PHASE2_DIR / "evaluation_audit_v1"
DEFAULT_OUTPUT_DIR = (
    ROOT / "experiments" / "soh_forecast" / "bafuse_missing_modality_audit_v1"
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PHASE2_DIR))
sys.path.insert(0, str(AUDIT_DIR))

import run_phase2 as phase2
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, build_physics_features


PARTITIONS = ("train", "validation", "external_test")
COMBINATIONS = (
    ("D+C+P", ("D", "C", "P")),
    ("D+C only", ("D", "C")),
    ("D+P only", ("D", "P")),
    ("C+P only", ("C", "P")),
    ("D only", ("D",)),
    ("C only", ("C",)),
    ("P only", ("P",)),
    ("No valid modality", ()),
)
MODEL_MODALITIES = {
    "M1": ("D",),
    "M2": ("D", "C"),
    "M3": ("D", "P"),
    "M4": ("D", "C", "P"),
    "M5": ("C", "P"),
    "M6": ("C",),
    "M7": ("P",),
}
AVAILABILITY_PATH = PHASE2_DIR / "modality_availability.csv"
FEATURE_MANIFEST_PATH = PHASE2_DIR / "multimodal_feature_manifest.csv"
PHYSICS_FEATURES_PATH = PHASE2_DIR / "physics_features.csv"
SUPPLEMENTAL_LABELS_PATH = AUDIT_DIR / "supplemental_g10_cell2_labels.csv"
EXTERNAL_COVERAGE_PATH = AUDIT_DIR / "external_12_cell_coverage.csv"
EXTERNAL_EXCLUSIONS_PATH = AUDIT_DIR / "external_exclusion_reasons.csv"


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


def _key(part: Any, cell: Any, rpt: Any) -> tuple[str, str, str]:
    part_text = str(int(float(part)))
    cell_text = str(cell).strip()
    rpt_text = str(rpt).strip()
    if rpt_text.startswith("RPT"):
        rpt_text = rpt_text[3:]
    return part_text, cell_text, str(int(float(rpt_text)))


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _truth_text(value: bool | None) -> str:
    if value is None:
        return "UNKNOWN"
    return "TRUE" if value else "FALSE"


def _float_or_none(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed and abs(parsed) != float("inf") else None


def _modality_combination(row: dict[str, Any]) -> str:
    present = tuple(name for name in ("D", "C", "P") if row[name] is True)
    for label, modalities in COMBINATIONS:
        if present == modalities:
            return label
    raise ValueError(f"Unsupported modality combination: {present}")


def _read_physics_cache() -> dict[tuple[str, str, str], list[float]]:
    rows = _read_csv(PHYSICS_FEATURES_PATH)
    features: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = _key(
            row.get("dataset_part", row.get("part", "")),
            row["physical_cell_id"],
            row.get("rpt_index", row.get("measurement_id", "")),
        )
        if key in features:
            raise ValueError(f"Duplicate physics feature vector: {key}")
        values = [float(row[name]) for name in PHYSICS_FEATURE_NAMES]
        if any(not (value == value and abs(value) != float("inf")) for value in values):
            raise ValueError(f"Non-finite cached physics feature for {key}")
        features[key] = values
    return features


def _validate_uncached_physics(
    observations: list[dict[str, Any]],
    availability_by_key: dict[tuple[str, str, str], dict[str, str]],
    physics_cache: dict[tuple[str, str, str], list[float]],
) -> dict[tuple[str, str, str], tuple[bool, str]]:
    candidates = []
    for row in observations:
        key = _key(row["part"], row["physical_cell_id"], row["rpt_index"])
        availability = availability_by_key[key]
        if (
            row["soh_percent"] is not None
            and _truth(availability["step4_raw_curve_valid"])
            and _truth(availability["temperature_complete"])
            and key not in physics_cache
            and row["physical_cell_id"] != "G10-Cell2"
        ):
            candidates.append(row)
    if not candidates:
        return {}

    requests_by_source = phase2._segment_requests(candidates)
    step4_requests = {
        source: [request for request in requests if request["step"] == 4]
        for source, requests in requests_by_source.items()
    }
    for requests in step4_requests.values():
        phase2._scan_source_requests(ROOT, requests)

    results: dict[tuple[str, str, str], tuple[bool, str]] = {}
    for row in candidates:
        key = _key(row["part"], row["physical_cell_id"], row["rpt_index"])
        step4 = row.get("step4_result")
        if not step4 or not step4.get("valid"):
            status = step4.get("status", "step4_extraction_not_attempted") if step4 else "step4_extraction_not_attempted"
            results[key] = (False, f"Step 4 recheck failed: {status}")
            continue
        if not step4.get("temperature_complete", False):
            results[key] = (False, "Missing or non-finite Step 4 temperature")
            continue
        try:
            features = build_physics_features(
                step4["voltage_raw"],
                step4["current_raw"],
                step4["temperature_raw"],
            )
        except ValueError as error:
            results[key] = (False, f"Invalid physics feature calculation: {error}")
            continue
        if tuple(features) != PHYSICS_FEATURE_NAMES:
            raise ValueError(f"Unexpected physics feature schema for {key}")
        results[key] = (True, "Recomputed from verified Step 4 curve and complete temperature")
    return results


def _missing_reasons(row: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not row["D"]:
        if not _truth(row["step4_mapping_verified"]):
            reasons.append("D: missing or unverified Step 4 mapping")
        else:
            reasons.append(f"D: invalid Step 4 discharge ({row['step4_status']})")
    if not row["C"]:
        if not _truth(row["step10_mapping_verified"]):
            reasons.append("C: missing or unverified Step 10 mapping")
        elif row["step10_status"] == "unexpected_current_direction":
            reasons.append("C: invalid Step 10 charge current direction")
        else:
            reasons.append(f"C: invalid Step 10 charge ({row['step10_status']})")
    if not row["P"]:
        if not row["D"]:
            reasons.append("P: unavailable because valid Step 4 discharge is missing")
        elif not row["temperature_complete"]:
            reasons.append("P: missing or non-finite Step 4 temperature")
        else:
            reasons.append(f"P: {row['physics_status']}")
    return reasons


def _audit_rows() -> tuple[list[dict[str, Any]], dict[str, str]]:
    observations, split_version = phase2._load_observations()
    split_by_cell, loaded_split_version = phase2._load_split()
    if split_version != loaded_split_version:
        raise ValueError("Observation and locked-split manifests disagree.")

    availability_rows = _read_csv(AVAILABILITY_PATH)
    availability_by_key = {
        _key(
            row.get("dataset_part", row.get("part", "")),
            row["physical_cell_id"],
            row.get("rpt_index", row.get("measurement_id", "")),
        ): row
        for row in availability_rows
    }
    manifest_rows = _read_csv(FEATURE_MANIFEST_PATH)
    manifest_by_key = {
        _key(
            row.get("dataset_part", row.get("part", "")),
            row["physical_cell_id"],
            row.get("rpt_index", row.get("measurement_id", "")),
        ): row
        for row in manifest_rows
    }
    if len(availability_by_key) != len(availability_rows):
        raise ValueError("Duplicate observation key in modality availability audit.")
    if len(manifest_by_key) != len(manifest_rows):
        raise ValueError("Duplicate observation key in feature manifest.")
    if set(availability_by_key) != set(manifest_by_key):
        raise ValueError("Modality and feature manifests contain different mapped keys.")
    if len(observations) != len(availability_by_key):
        raise ValueError(
            f"Canonical mapping count differs: {len(observations)} vs {len(availability_by_key)}"
        )

    physics_cache = _read_physics_cache()
    uncached_physics = _validate_uncached_physics(
        observations, availability_by_key, physics_cache
    )
    supplemental_labels = _read_csv(SUPPLEMENTAL_LABELS_PATH)
    supplemental_exclusions = _read_csv(EXTERNAL_EXCLUSIONS_PATH)
    supplemental_by_key = {
        _key(
            item["dataset_part"],
            item["physical_cell_id"],
            item["measurement_id"],
        ): item
        for item in supplemental_labels
    }
    exclusion_by_key = {
        _key(
            item["dataset_part"],
            item["physical_cell_id"],
            item["measurement_id"],
        ): item
        for item in supplemental_exclusions
        if item["physical_cell_id"] == "G10-Cell2"
    }
    external_coverage = {
        row["physical_cell_id"]: row for row in _read_csv(EXTERNAL_COVERAGE_PATH)
    }
    if len(supplemental_by_key) != 10 or len(exclusion_by_key) != 10:
        raise ValueError("Expected 10 audited supplemental G10-Cell2 RPT records.")
    if external_coverage["G10-Cell2"]["valid_physics_feature_observations"] != "10":
        raise ValueError("G10-Cell2 physics feature count is not 10 in the external audit.")

    rows: list[dict[str, Any]] = []
    for observation in observations:
        key = _key(
            observation["part"],
            observation["physical_cell_id"],
            observation["rpt_index"],
        )
        availability = availability_by_key[key]
        manifest = manifest_by_key[key]
        if observation["physical_cell_id"] not in split_by_cell:
            raise ValueError(f"Mapped cell is absent from the locked split: {key}")
        if observation["partition"] != split_by_cell[observation["physical_cell_id"]]:
            raise ValueError(f"Partition mismatch for mapped key {key}")

        valid_label = observation["soh_percent"] is not None
        label_status = observation["label_status"]
        label_scope = observation["label_scope"]
        target = observation["soh_percent"]
        source_provenance = "canonical SOH label manifest"

        supplemental = supplemental_by_key.get(key)
        if supplemental is not None:
            exclusion = exclusion_by_key[key]
            if supplemental["validation_status"] != "calculated_from_measured_capacity":
                raise ValueError(f"Supplemental label is not validated for {key}")
            valid_label = True
            target = float(supplemental["soh_percent"])
            label_status = supplemental["validation_status"]
            label_scope = supplemental["label_scope"]
            source_provenance = supplemental["identity_provenance"]
            step4_status = supplemental["step4_raw_status"]
            step10_status = supplemental["step10_raw_status"]
            step4_mapping = exclusion["step4_mapping_status"] == "verified"
            step10_mapping = exclusion["step10_mapping_status"] == "verified"
            d_valid = (
                step4_mapping and step4_status == "verified_raw_segment"
            )
            c_valid = (
                step10_mapping and step10_status == "verified_raw_segment"
            )
            temp_complete = _truth(supplemental["temperature_complete"])
            p_valid = d_valid and temp_complete
            physics_status = (
                "verified by the saved supplemental Step 4/temperature feature audit"
                if p_valid
                else "not verified by the saved supplemental feature audit"
            )
        else:
            d_valid = (
                _truth(availability["step4_raw_curve_valid"])
                if valid_label
                else None
            )
            c_valid = (
                _truth(availability["step10_raw_curve_valid"])
                if valid_label
                else None
            )
            temp_complete = (
                _truth(availability["temperature_complete"])
                if valid_label and availability["temperature_complete"] != ""
                else False
                if valid_label
                else None
            )
            p_valid: bool | None = None
            if valid_label:
                if key in physics_cache:
                    p_valid = True
                    physics_status = "verified_existing_finite_physics_feature_vector"
                elif key in uncached_physics:
                    p_valid, physics_status = uncached_physics[key]
                elif d_valid and temp_complete:
                    p_valid = False
                    physics_status = "physics feature was not verified"
                else:
                    p_valid = False
                    physics_status = (
                        "missing or non-finite Step 4 temperature"
                        if d_valid
                        else "requires valid Step 4 discharge"
                    )
            else:
                physics_status = "not_checked_without_valid_soh_label"
            step4_status = manifest["step4_raw_status"]
            step10_status = manifest["step10_raw_status"]
            step4_mapping = availability["step4_mapping_verified"]
            step10_mapping = availability["step10_mapping_verified"]

        if valid_label and any(
            value is None for value in (d_valid, c_valid, p_valid)
        ):
            raise ValueError(f"Valid-label observation lacks audited modality status: {key}")
        if key in manifest_by_key and valid_label:
            if observation["physical_cell_id"] != "G10-Cell2":
                manifest_label = _float_or_none(manifest["soh_percent"])
                if manifest_label is None or abs(manifest_label - float(target)) > 1e-8:
                    raise ValueError(f"SOH label mismatch between manifests for {key}")

        row = {
            "key": key,
            "dataset_part": key[0],
            "group": str(observation["group"]),
            "physical_cell_id": key[1],
            "partition": observation["partition"],
            "rpt_index": int(key[2]),
            "measurement_id": f"RPT{key[2]}",
            "source_file": observation["source_file"],
            "step4_segment_ref": observation.get("step4_segment_ref", ""),
            "step10_segment_ref": observation.get("step10_segment_ref", ""),
            "valid_soh_label": valid_label,
            "soh_percent": target,
            "label_status": label_status,
            "label_scope": label_scope,
            "source_provenance": source_provenance,
            "step4_mapping_verified": step4_mapping,
            "step4_status": step4_status,
            "D": d_valid,
            "step10_mapping_verified": step10_mapping,
            "step10_status": step10_status,
            "C": c_valid,
            "temperature_complete": temp_complete,
            "physics_status": physics_status,
            "P": p_valid,
            "feature_manifest_dcp_eligible": _truth(manifest["fusion_eligible"]),
        }
        row["complete_DCP"] = bool(
            valid_label and row["D"] is True and row["C"] is True and row["P"] is True
        )
        row["modality_combination"] = (
            _modality_combination(row) if valid_label else "unassessed_no_valid_label"
        )
        row["missing_modalities"] = (
            ";".join(name for name in ("D", "C", "P") if row[name] is False)
            if valid_label
            else ""
        )
        row["missing_reasons"] = (
            _missing_reasons(row)
            if valid_label and not row["complete_DCP"]
            else []
        )
        rows.append(row)

    if len({row["key"] for row in rows}) != len(rows):
        raise ValueError("Duplicate keys after supplemental label merge.")
    if sum(
        row["complete_DCP"]
        for row in rows
        if row["physical_cell_id"] != "G10-Cell2"
    ) != 211:
        raise ValueError("Base complete-modality cohort does not reconcile to 211.")
    if sum(row["physical_cell_id"] == "G10-Cell2" and row["complete_DCP"] for row in rows) != 9:
        raise ValueError("G10-Cell2 complete-modality overlay does not equal 9.")
    return rows, split_by_cell


def _cell_coverage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_cell[(row["partition"], row["physical_cell_id"])].append(row)
    results: list[dict[str, Any]] = []
    for (partition, cell), selected in sorted(by_cell.items()):
        valid = [row for row in selected if row["valid_soh_label"]]
        parts = sorted({row["dataset_part"] for row in selected}, key=int)
        groups = sorted({row["group"] for row in selected}, key=int)
        out: dict[str, Any] = {
            "partition": partition,
            "dataset_part": ";".join(parts),
            "group": ";".join(groups),
            "physical_cell_id": cell,
            "total_mapped_rpts": len(selected),
            "valid_soh_reference_rpts": len(valid),
            "valid_step4_discharge_rpts": sum(row["D"] is True for row in valid),
            "valid_step10_charge_rpts": sum(row["C"] is True for row in valid),
            "valid_physics_feature_rpts": sum(row["P"] is True for row in valid),
            "complete_DCP_rpts": sum(row["complete_DCP"] for row in valid),
            "valid_label_rpts_excluded_from_DCP": sum(
                not row["complete_DCP"] for row in valid
            ),
            "mapped_rpts_without_valid_soh_label": len(selected) - len(valid),
            "physical_cell_eligible_for_DCP": bool(
                any(row["complete_DCP"] for row in valid)
            ),
            "status_for_unlabeled_rpts": (
                "not assessed; no validated SOH label"
                if len(valid) < len(selected)
                else "all mapped RPTs have valid SoH labels"
            ),
        }
        eligible_by_model: dict[str, list[dict[str, Any]]] = {}
        for model, modalities in MODEL_MODALITIES.items():
            eligible = [
                row
                for row in valid
                if all(row[name] is True for name in modalities)
            ]
            eligible_by_model[model] = eligible
            out[f"eligible_rpts_{model}"] = len(eligible)
            out[f"eligible_for_{model}"] = bool(eligible)
        results.append(out)
    return results


def _missing_cell_rows(
    cell_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "primary_exclusion_reason": (
                "No validated SoH reference labels; mapped RPT modality rows were not assessed."
                if row["valid_soh_reference_rpts"] == 0
                else "No valid D+C+P labeled observation; see all_rpts_modality_availability.csv."
            ),
        }
        for row in cell_rows
        if row["complete_DCP_rpts"] == 0
    ]


def _combination_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        partition_rows = [row for row in rows if row["partition"] == partition]
        valid_rows = [row for row in partition_rows if row["valid_soh_label"]]
        for combination, _ in COMBINATIONS:
            selected = [
                row
                for row in valid_rows
                if row["modality_combination"] == combination
            ]
            results.append(
                {
                    "partition": partition,
                    "row_type": "valid_label_modality_combination",
                    "modality_combination": combination,
                    "observation_count": len(selected),
                    "physical_cell_count": len(
                        {row["physical_cell_id"] for row in selected}
                    ),
                    "soh_label_status": "valid",
                    "included_in_valid_label_combination_total": True,
                }
            )
        unlabeled = [
            row for row in partition_rows if not row["valid_soh_label"]
        ]
        results.append(
            {
                "partition": partition,
                "row_type": "missing_or_invalid_soh_label",
                "modality_combination": "not_assessed",
                "observation_count": len(unlabeled),
                "physical_cell_count": len(
                    {row["physical_cell_id"] for row in unlabeled}
                ),
                "soh_label_status": "missing_or_invalid",
                "included_in_valid_label_combination_total": False,
            }
        )
        results.append(
            {
                "partition": partition,
                "row_type": "partition_total",
                "modality_combination": "all_mapped_rpts",
                "observation_count": len(partition_rows),
                "physical_cell_count": len(
                    {row["physical_cell_id"] for row in partition_rows}
                ),
                "soh_label_status": "valid_and_missing_or_invalid",
                "included_in_valid_label_combination_total": False,
            }
        )
        if sum(row["observation_count"] for row in results if row["partition"] == partition and row["row_type"] == "valid_label_modality_combination") != len(valid_rows):
            raise ValueError(f"Modality combinations do not reconcile for {partition}")
        if len(valid_rows) + len(unlabeled) != len(partition_rows):
            raise ValueError(f"Label and unlabeled counts do not reconcile for {partition}")
    return results


def _potential_coverage_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        selected = [
            row
            for row in rows
            if row["partition"] == partition and row["valid_soh_label"]
        ]
        full = [row for row in selected if row["complete_DCP"]]
        full_keys = {row["key"] for row in full}
        for model, modalities in MODEL_MODALITIES.items():
            eligible = [
                row
                for row in selected
                if all(row[name] is True for name in modalities)
            ]
            results.append(
                {
                    "partition": partition,
                    "model_configuration": model,
                    "required_modalities": "+".join(modalities),
                    "valid_soh_labeled_observations": len(selected),
                    "eligible_observations": len(eligible),
                    "eligible_physical_cells": len(
                        {row["physical_cell_id"] for row in eligible}
                    ),
                    "complete_DCP_observations": len(full),
                    "additional_observations_vs_M4": len(
                        {row["key"] for row in eligible} - full_keys
                    ),
                    "additional_physical_cells_vs_M4": len(
                        {
                            row["physical_cell_id"]
                            for row in eligible
                            if row["key"] not in full_keys
                        }
                        - {row["physical_cell_id"] for row in full}
                    ),
                    "interpretation": "Data eligibility only; no model was trained or evaluated on added keys.",
                }
            )
    return results


def _missing_rpt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    for row in rows:
        if not row["valid_soh_label"] or row["complete_DCP"]:
            continue
        detailed.append(
            {
                "partition": row["partition"],
                "dataset_part": row["dataset_part"],
                "group": row["group"],
                "cell_id": row["physical_cell_id"],
                "rpt": row["measurement_id"],
                "soh_label_percent": row["soh_percent"],
                "D": _truth_text(row["D"]),
                "C": _truth_text(row["C"]),
                "P": _truth_text(row["P"]),
                "missing_modalities": row["missing_modalities"],
                "exclusion_reason": "; ".join(row["missing_reasons"]),
                "step4_status": row["step4_status"],
                "step10_status": row["step10_status"],
                "step4_segment_ref": row["step4_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "source_file": row["source_file"],
                "label_scope": row["label_scope"],
                "source_provenance": row["source_provenance"],
            }
        )
    return detailed


def _all_rpt_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in rows:
        reasons = row["missing_reasons"]
        if not row["valid_soh_label"]:
            reason = (
                f"No validated SoH label ({row['label_status']}); "
                "modalities not assessed for unlabeled mapping."
            )
        elif row["complete_DCP"]:
            reason = ""
        else:
            reason = "; ".join(reasons)
        results.append(
            {
                "dataset_part": row["dataset_part"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "partition": row["partition"],
                "rpt_index": row["rpt_index"],
                "measurement_id": row["measurement_id"],
                "source_file": row["source_file"],
                "valid_soh_label": _truth_text(row["valid_soh_label"]),
                "soh_percent": row["soh_percent"],
                "label_status": row["label_status"],
                "label_scope": row["label_scope"],
                "D": _truth_text(row["D"]),
                "C": _truth_text(row["C"]),
                "P": _truth_text(row["P"]),
                "step4_mapping_verified": row["step4_mapping_verified"],
                "step4_status": row["step4_status"],
                "step10_mapping_verified": row["step10_mapping_verified"],
                "step10_status": row["step10_status"],
                "temperature_complete": _truth_text(row["temperature_complete"]),
                "physics_status": row["physics_status"],
                "complete_DCP": _truth_text(row["complete_DCP"]),
                "modality_combination": row["modality_combination"],
                "missing_modalities": row["missing_modalities"],
                "exclusion_reason": reason,
                "step4_segment_ref": row["step4_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "source_provenance": row["source_provenance"],
            }
        )
    return results


def _report(
    output_dir: Path,
    rows: list[dict[str, Any]],
    cell_rows: list[dict[str, Any]],
    combination_rows: list[dict[str, Any]],
    potential_rows: list[dict[str, Any]],
    missing_rows: list[dict[str, Any]],
    split_by_cell: dict[str, str],
) -> None:
    summary = []
    for partition in PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        valid = [row for row in selected if row["valid_soh_label"]]
        complete = [row for row in valid if row["complete_DCP"]]
        cells = {row["physical_cell_id"] for row in selected}
        summary.append(
            {
                "partition": partition,
                "locked_cells": sum(value == partition for value in split_by_cell.values()),
                "mapped": len(selected),
                "valid_labels": len(valid),
                "complete": len(complete),
                "excluded_labeled": len(valid) - len(complete),
                "missing_labels": len(selected) - len(valid),
                "complete_cells": len(
                    {row["physical_cell_id"] for row in complete}
                ),
                "mapped_cells": len(cells),
            }
        )

    lines = [
        "# BaFuse missing-modality coverage audit",
        "",
        "## Scope and method",
        "",
        "- This is a read-only availability audit. No models were trained, no checkpoint or label was changed, and the dashboard and PyDMA branch were not modified.",
        "- Locked split: 12 train / 4 validation / 12 external physical cells, read from the canonical split manifest.",
        "- Modality categories below are counted only for mapped RPTs with a valid SoH reference. Rows without a valid label are counted separately; their modalities are marked UNKNOWN/not assessed rather than FALSE.",
        "- D means a mapped Step 4 discharge passed the existing raw-segment checks. C means a mapped Step 10 charge passed raw-segment/current-direction checks. P requires valid Step 4 discharge, complete finite Step 4 temperature, and a finite approved physics feature vector.",
        "- The existing Phase 2 feature writer only materializes physics vectors after Step 10 succeeds. For labeled rows with valid Step 4 plus complete temperature but no cached P vector, this audit re-read only the mapped Step 4 segment and recomputed the approved physics feature function; it did not fit or train anything.",
        "- G10-Cell2 uses the existing supplemental 10-label audit output and verified raw-segment scan. Its identity is user-confirmed because the Guide omits the cell number; the labels remain provisional.",
        "",
        "## Partition summary",
        "",
        "| Partition | Locked cells | Mapped RPTs | Valid SoH labels | D+C+P | Labeled but excluded from D+C+P | Missing/invalid labels | Cells with D+C+P |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary:
        lines.append(
            f"| {item['partition']} | {item['locked_cells']} | {item['mapped']} | {item['valid_labels']} | "
            f"{item['complete']} | {item['excluded_labeled']} | {item['missing_labels']} | {item['complete_cells']} |"
        )
    lines.extend(
        [
            "",
            "Total mapped RPTs: 472. Total valid SoH-labeled observations including the 10 supplemental G10-Cell2 candidates: 241. Total complete-modality observations: 220. Of the 252 mapped observations not in D+C+P, 231 lack a validated label and were not modality-audited; 21 have valid labels but lack at least one modality.",
            "",
            "## Physical cells missing from the complete-modality cohort",
            "",
            "The 3 train cells and 1 validation cell absent from the complete-modality cohort have mapped RPTs but zero validated SoH labels. Their modality status is not assessed; do not infer missing discharge or charge from the missing training observations.",
            "",
        ]
    )
    missing_cells = _missing_cell_rows(cell_rows)
    if missing_cells:
        lines.extend(
            [
                "| Partition | Part(s) | Group | Physical cell | Mapped RPTs | Valid labels | Reason |",
                "|---|---|---:|---|---:|---:|---|",
            ]
        )
        for row in missing_cells:
            lines.append(
                f"| {row['partition']} | {row['dataset_part']} | {row['group']} | {row['physical_cell_id']} | "
                f"{row['total_mapped_rpts']} | {row['valid_soh_reference_rpts']} | {row['primary_exclusion_reason']} |"
            )
    lines.extend(
        [
            "",
            "## Modality combinations",
            "",
            "Combination categories are mutually exclusive and sum to valid-label observations. Missing/invalid SoH labels are a separate not-assessed row in `modality_combination_summary.csv`.",
            "",
            "| Partition | D+C+P | D+C only | D+P only | C+P only | D only | C only | P only | No valid modality | Missing/invalid SoH label |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for partition in PARTITIONS:
        counts = {
            row["modality_combination"]: row["observation_count"]
            for row in combination_rows
            if row["partition"] == partition
            and row["row_type"] == "valid_label_modality_combination"
        }
        missing_labels = next(
            row["observation_count"]
            for row in combination_rows
            if row["partition"] == partition
            and row["row_type"] == "missing_or_invalid_soh_label"
        )
        lines.append(
            f"| {partition} | "
            + " | ".join(
                str(counts[name])
                for name, _ in COMBINATIONS
            )
            + f" | {missing_labels} |"
        )

    lines.extend(
        [
            "",
            "## Potential data coverage by model configuration",
            "",
            "These counts are label-and-modality eligibility only. They are not validated performance estimates and do not imply that an existing checkpoint accepts missing modalities.",
            "",
            "| Partition | Model | Required modalities | Eligible observations | Eligible cells | Additional observations vs M4 |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in potential_rows:
        lines.append(
            f"| {row['partition']} | {row['model_configuration']} | {row['required_modalities']} | "
            f"{row['eligible_observations']} | {row['eligible_physical_cells']} | {row['additional_observations_vs_M4']} |"
        )

    lines.extend(
        [
            "",
            "## Labeled RPTs excluded from three-encoder cohort",
            "",
            "Every row below has a valid SoH reference but lacks at least one of D/C/P. Reasons are based on mapping and raw-segment status; missing charge, missing temperature, and missing discharge are distinguished.",
            "",
            "| Partition | Part | Group | Cell ID | RPT | SoH | D | C | P | Missing modalities | Exclusion reason |",
            "|---|---:|---:|---|---|---:|---|---|---|---|---|",
        ]
    )
    for row in missing_rows:
        lines.append(
            f"| {row['partition']} | {row['dataset_part']} | {row['group']} | {row['cell_id']} | {row['rpt']} | "
            f"{float(row['soh_label_percent']):.4f} | {row['D']} | {row['C']} | {row['P']} | "
            f"{row['missing_modalities']} | {row['exclusion_reason']} |"
        )

    reason_counts: Counter[str] = Counter()
    for row in missing_rows:
        if not row["C"] == "TRUE":
            reason_counts["missing_or_invalid_charge"] += 1
        if not row["D"] == "TRUE":
            reason_counts["missing_or_invalid_discharge"] += 1
        if row["D"] == "TRUE" and not row["P"] == "TRUE":
            reason_counts["physics_missing_with_valid_discharge"] += 1
    missing_soh_total = sum(item["missing_labels"] for item in summary)
    lines.extend(
        [
            "",
            "## Root-cause summary",
            "",
            f"- Missing/invalid SoH labels: {missing_soh_total} mapped RPTs. These are the largest exclusion category; their D/C/P status is unknown because the existing extraction pipeline did not evaluate unlabeled RPTs.",
            f"- Among the {len(missing_rows)} valid-label RPTs excluded from D+C+P: {reason_counts['missing_or_invalid_charge']} lack valid Charge, {reason_counts['physics_missing_with_valid_discharge']} have valid Discharge but invalid Physics due to incomplete temperature, and {reason_counts['missing_or_invalid_discharge']} lack valid Discharge (and consequently cannot provide derived Physics). These are disjoint root causes for the labeled exclusions.",
            "- Invalid Step 10 current direction and missing/unverified Step 10 mappings are distinguished in the RPT-level exclusion reasons; the 4 external D+C-only observations are excluded for incomplete Step 4 temperature/Physics, not Charge.",
            "- The G10-Cell2 RPT10 Step 10 current direction is unexpected. Its Step 4 discharge and Physics are valid, so it remains eligible for D-only and D+P data coverage, but not C, D+C, D+C+P, or C+P.",
            "",
            "## Reproducibility and boundaries",
            "",
            "Run from the repository root:",
            "",
            "```powershell",
            ".venv-torch-cuda\\Scripts\\python.exe experiments\\soh_forecast\\run_missing_modality_audit.py",
            "```",
            "",
            "Run the focused output checks after generating the bundle:",
            "",
            "```powershell",
            ".venv-torch-cuda\\Scripts\\python.exe -m pytest tests\\test_bafuse_missing_modality_audit.py -q",
            "```",
            "",
            "The bundle includes the six CSVs named above plus this report. No model was trained, no checkpoint was loaded for inference, no label was generated or edited, and no adaptive fusion or dashboard change was made.",
        ]
    )
    (output_dir / "missing_modality_audit_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty audit directory: {output_dir}"
        )
    rows, split_by_cell = _audit_rows()
    cell_rows = _cell_coverage(rows)
    missing_cells = _missing_cell_rows(cell_rows)
    combination_rows = _combination_rows(rows)
    potential_rows = _potential_coverage_rows(rows)
    missing_rows = _missing_rpt_rows(rows)
    all_rpt_rows = _all_rpt_rows(rows)
    expected_partitions = {"train": (274, 113, 106), "validation": (111, 44, 41), "external_test": (87, 84, 73)}
    for partition, (mapped, labels, complete) in expected_partitions.items():
        selected = [row for row in rows if row["partition"] == partition]
        valid_count = sum(row["valid_soh_label"] for row in selected)
        complete_count = sum(row["complete_DCP"] for row in selected)
        if (len(selected), valid_count, complete_count) != (mapped, labels, complete):
            raise ValueError(
                f"Coverage mismatch for {partition}: "
                f"{(len(selected), valid_count, complete_count)}"
            )
    if len(missing_cells) != 4 or len(missing_rows) != 21:
        raise ValueError(
            f"Expected 4 cells and 21 labeled RPT exclusions, got "
            f"{len(missing_cells)} and {len(missing_rows)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_cells_modality_coverage.csv", cell_rows)
    _write_csv(output_dir / "all_rpts_modality_availability.csv", all_rpt_rows)
    _write_csv(output_dir / "missing_rpts_detailed.csv", missing_rows)
    _write_csv(output_dir / "missing_cells_summary.csv", missing_cells)
    _write_csv(output_dir / "modality_combination_summary.csv", combination_rows)
    _write_csv(output_dir / "potential_encoder_coverage.csv", potential_rows)
    _report(
        output_dir,
        rows,
        cell_rows,
        combination_rows,
        potential_rows,
        missing_rows,
        split_by_cell,
    )
    print(
        f"Read-only coverage audit written to {output_dir}; mapped={len(rows)}, "
        f"valid_labels={sum(row['valid_soh_label'] for row in rows)}, "
        f"complete_DCP={sum(row['complete_DCP'] for row in rows)}, "
        f"missing_label_RPTs={len(missing_cells)} cells."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit BaFuse modality availability without training models."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
