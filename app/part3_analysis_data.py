from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bafuse.hybrid.final2_dataset import PHYSICS_FEATURE_NAMES
from app.model_runtime import (
    MODEL_NAMES,
    STEP4_BASELINE,
    Final2ModelRuntime,
    load_shared_charge_degradation,
)
from app.part3_review_preprocessing import (
    STEP10_REPAIR_KEYS,
    ReviewPreprocessingError,
    extract_rpt12_step10,
    load_review_resolution_index,
    verify_review_source,
)
from app.tpg_mat_ingestion import TpgIngestionError, analyze_tpg_file


ROOT = Path(__file__).resolve().parents[1]
DATASET_IDENTIFIER = "bafuse-final2-p5-g10-guide-verified-v2"
PREPROCESSING_CONTRACT = (
    "part3-p5-step4-step10-row-progress-128-v1-review-step10-v1"
)
AUDIT_PATH = ROOT / "app" / "part3_modality_eligibility.csv"
CANONICAL_MAPPING_PATH = (
    ROOT / "data" / "manifests" / "canonical_observation_manifest_v3.csv"
)
EOL_MAPPING_PATH = ROOT / "data" / "manifests" / "part3_demo_eol_mapping_v1.csv"
EOL_AUDIT_PATH = ROOT / "app" / "part3_demo_eol_eligibility.csv"
FROZEN_DIR = ROOT / "experiments" / "final2" / "training_6_models" / "frozen_dataset"
DATASET_HASH_PATH = FROZEN_DIR / "dataset_sha256_manifest.json"
EXTERNAL_NPZ_PATH = FROZEN_DIR / "external_test.npz"
GROUP_CELL_PATTERN = re.compile(r"^G(\d+)-Cell(\d+)$", re.IGNORECASE)


class DashboardDataError(ValueError):
    pass


class _ExtractionRuntime:
    def __init__(self, runtime: Final2ModelRuntime) -> None:
        self.final2_runtime = runtime

    def predict_for_model(
        self, rows: list[dict[str, Any]], model_name: str | None = None
    ) -> dict[str, Any]:
        return self.final2_runtime.preprocess_rows(rows, model_name)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not reader.fieldnames:
                raise DashboardDataError(f"CSV has no header: {path}")
            rows = list(reader)
            if any(None in row for row in rows):
                raise DashboardDataError(f"CSV row has extra fields: {path}")
            return rows
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise DashboardDataError(f"Could not read {path}: {error}") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise DashboardDataError(f"Could not hash {path}: {error}") from error
    return digest.hexdigest()


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes", "y"}


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rpt_number(value: Any) -> str:
    match = re.search(r"\d+", str(value).strip())
    if not match:
        raise DashboardDataError(f"Invalid RPT number: {value!r}")
    return str(int(match.group(0)))


def _identity_key(
    group: Any, cell: Any, rpt: Any
) -> tuple[str, str, str, str]:
    group_text = str(int(str(group).strip()))
    cell_text = str(cell).strip()
    match = GROUP_CELL_PATTERN.fullmatch(cell_text)
    if not match or match.group(1) != group_text:
        raise DashboardDataError(
            f"Group/cell identity is inconsistent: group={group_text}, cell={cell_text}"
        )
    return "3", group_text, cell_text, _rpt_number(rpt)


def _normalized_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _group_cell_from_npz(cell: str) -> tuple[str, str]:
    match = GROUP_CELL_PATTERN.fullmatch(str(cell).strip())
    if not match:
        raise DashboardDataError(f"Invalid physical cell ID in external tensor: {cell!r}")
    return str(int(match.group(1))), str(cell).strip()


class Part3AnalysisData:
    """Read-only Part 3 inventory, frozen inference and validated curve cache."""

    def __init__(
        self,
        runtime: Final2ModelRuntime | None = None,
    ) -> None:
        self.runtime = runtime or Final2ModelRuntime()
        self.dataset_identifier = DATASET_IDENTIFIER
        self._source_lock = threading.RLock()
        self._inference_lock = threading.RLock()
        self._source_cache: dict[
            tuple[str, str, str, str],
            tuple[dict[tuple[str, str, str, str], dict[str, Any]], str],
        ] = {}
        self._prediction_cache: dict[tuple[str, ...], dict[str, Any]] = {}
        self._degradation_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        self._review_resolutions = load_review_resolution_index()
        self._records, canonical_ids = self._load_inventory()
        self._apply_review_resolution()
        self._frozen_inputs, self._frozen_references = self._load_frozen_external()
        self._cells_by_group: dict[str, list[str]] = defaultdict(list)
        for key in self._records:
            group, cell = key[1], key[2]
            if cell not in self._cells_by_group[group]:
                self._cells_by_group[group].append(cell)
        for cells in self._cells_by_group.values():
            cells.sort(key=lambda value: int(GROUP_CELL_PATTERN.fullmatch(value).group(2)))
        self._canonical_record_ids = canonical_ids

    def _apply_review_resolution(self) -> None:
        for key, resolution in self._review_resolutions.items():
            record = self._records.get(key)
            if record is None:
                raise DashboardDataError(
                    f"Review-resolution observation is not in the Part 3 inventory: {key}"
                )
            if (
                record["status"] != resolution["original_status"]
                or _normalized_path(record["source_file"])
                != _normalized_path(resolution["raw_source_path"])
                or resolution["evaluation_membership"] != "EXCLUDED_FROZEN_FINAL2"
            ):
                raise DashboardDataError(
                    f"Review-resolution provenance conflicts with the inventory: {key}"
                )
            reference = _number(resolution["soh_reference"])
            if reference is None:
                raise DashboardDataError(
                    f"Review-resolution SoH reference is not finite: {key}"
                )
            step4_valid = _truth(resolution["step4_valid"])
            step10_valid = _truth(resolution["step10_valid"])
            physics_valid = _truth(resolution["physics5_valid"])
            record.update(
                {
                    "data_qc_status": resolution["new_data_qc_status"],
                    "original_review_reason": resolution["original_review_reason"],
                    "review_mapping_status": (
                        "GUIDE_VERIFIED"
                        if _truth(resolution["guide_mapping_verified"])
                        else "UNVERIFIED"
                    ),
                    "resolved_issue": resolution["step10_review_reason"],
                    "remaining_issue": resolution["remaining_issue"],
                    "review_resolution_evidence": resolution["evidence"],
                    "review_source_sha256": resolution["source_sha256"],
                    "p5_label_verified": _truth(resolution["p5_label_verified"]),
                    "step4_eligible": step4_valid,
                    "step10_eligible": step10_valid,
                    "p5_eligible": physics_valid,
                    "step4_reason": resolution["step4_review_reason"],
                    "step10_reason": resolution["step10_review_reason"],
                    "step4_segment_reference": resolution[
                        "step4_segment_reference"
                    ],
                    "p5_reason": (
                        ""
                        if physics_valid
                        else "Step 4 source did not pass the validated P5 feature gate."
                    ),
                    "baseline_eligible": _truth(resolution["baseline_eligible"]),
                    "two_modality_eligible": _truth(
                        resolution["two_modality_eligible"]
                    ),
                    "three_modality_eligible": _truth(
                        resolution["three_modality_eligible"]
                    ),
                    "reference_soh_percent": reference,
                    "reference_status": (
                        "accepted_data_qc_p5_ratio"
                        if _truth(resolution["p5_label_verified"])
                        else "candidate_ratio_not_verified"
                    ),
                    "reference_source": resolution["step4_segment_reference"],
                    "evaluation_eligible": False,
                    "evaluation_membership": resolution["evaluation_membership"],
                }
            )

    def _load_inventory(
        self,
    ) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], set[str]]:
        canonical_rows = _read_csv(CANONICAL_MAPPING_PATH)
        canonical_by_id: dict[str, dict[str, str]] = {}
        for row in canonical_rows:
            if (
                row.get("dataset_part") == "3"
                and row.get("record_type") == "cycling_guide_record"
                and row.get("rpt_number", "").strip()
            ):
                record_id = row.get("record_id", "").strip()
                if not record_id or record_id in canonical_by_id:
                    raise DashboardDataError(
                        f"Canonical Part 3 RPT identity is missing or duplicated: {record_id!r}"
                    )
                canonical_by_id[record_id] = row

        base_rows = _read_csv(AUDIT_PATH)
        records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        canonical_ids: set[str] = set()
        for row in base_rows:
            if row.get("dataset_part") != "3":
                continue
            key = _identity_key(row.get("group"), row.get("physical_cell_id"), row.get("rpt_index"))
            canonical_id = row.get("canonical_record_id", "").strip()
            canonical = canonical_by_id.get(canonical_id)
            if canonical is None:
                raise DashboardDataError(
                    f"Part 3 audit row has no exact canonical record: {key}"
                )
            if key != _identity_key(
                canonical.get("group_id"),
                canonical.get("physical_cell_id"),
                canonical.get("rpt_number"),
            ):
                raise DashboardDataError(
                    f"Canonical identity conflicts with modality audit for {key}"
                )
            source_file = row.get("source_file", "").strip()
            canonical_sources = {
                _normalized_path(source)
                for source in canonical.get("source_paths", "").split(";")
                if source.strip()
            }
            if _normalized_path(source_file) not in canonical_sources:
                raise DashboardDataError(
                    f"Canonical source path conflicts with modality audit for {key}"
                )
            if key in records:
                raise DashboardDataError(f"Duplicate Part 3 RPT identity: {key}")
            records[key] = self._base_record(row, canonical)
            canonical_ids.add(canonical_id)

        eol_mapping_rows = _read_csv(EOL_MAPPING_PATH)
        eol_audit_rows = _read_csv(EOL_AUDIT_PATH)
        eol_mapping_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in eol_mapping_rows:
            key = _identity_key(row.get("group"), row.get("cell"), row.get("rpt"))
            if key in eol_mapping_by_key:
                raise DashboardDataError(f"Duplicate demo-only EoL mapping: {key}")
            if (
                row.get("mapping_status") != "GUIDE_VERIFIED"
                or row.get("inventory_scope") != "VERIFIED_DEMO_ONLY"
                or not _truth(row.get("is_eol_procedure"))
                or not _truth(row.get("selectable"))
            ):
                raise DashboardDataError(
                    f"EoL mapping is not an explicitly selectable Guide-verified record: {key}"
                )
            eol_mapping_by_key[key] = row

        eol_audit_by_key: dict[tuple[str, str, str, str], dict[str, str]] = {}
        for row in eol_audit_rows:
            key = _identity_key(row.get("group"), row.get("cell"), row.get("rpt"))
            mapping = eol_mapping_by_key.get(key)
            if mapping is None:
                raise DashboardDataError(f"EoL audit row has no exact mapping: {key}")
            if (
                _normalized_path(row.get("source_file"))
                != _normalized_path(mapping.get("source_file"))
                or row.get("source_sha256") != mapping.get("source_sha256")
                or row.get("mapping_status") != "GUIDE_VERIFIED"
                or row.get("inventory_scope") != "VERIFIED_DEMO_ONLY"
            ):
                raise DashboardDataError(f"EoL audit provenance conflicts for {key}")
            if key in eol_audit_by_key:
                raise DashboardDataError(f"Duplicate demo-only EoL audit row: {key}")
            eol_audit_by_key[key] = row

        if set(eol_mapping_by_key) != set(eol_audit_by_key):
            raise DashboardDataError(
                "The EoL mapping and eligibility audit contain different identities."
            )
        for key, row in eol_audit_by_key.items():
            if key in records:
                raise DashboardDataError(
                    f"Demo-only EoL duplicates a canonical Part 3 RPT identity: {key}"
                )
            records[key] = self._eol_record(row)

        if not records:
            raise DashboardDataError("No Part 3 RPT observations were loaded.")
        return records, canonical_ids

    @staticmethod
    def _base_record(
        row: dict[str, str], canonical: dict[str, str]
    ) -> dict[str, Any]:
        key = _identity_key(row["group"], row["physical_cell_id"], row["rpt_index"])
        return {
            "key": key,
            "group": key[1],
            "cell": key[2],
            "rpt_index": int(key[3]),
            "is_eol": False,
            "inventory_scope": "GUIDE_MAPPED",
            "mapping_status": row.get("mapping_status", ""),
            "canonical_mapping_status": canonical.get("mapping_status", ""),
            "canonical_guide_mapping_status": canonical.get(
                "guide_mapping_status", ""
            ),
            "canonical_cell_identity_status": canonical.get(
                "cell_identity_status", ""
            ),
            "mapping_evidence": row.get("mapping_evidence", ""),
            "guide_mapping_status": row.get("guide_mapping_status", ""),
            "identity_provenance": row.get("identity_provenance", ""),
            "procedure": row.get("source_tpg", ""),
            "guide_stage": row.get("guide_stage", ""),
            "source_file": row.get("source_file", ""),
            "source_sha256": row.get("raw_source_sha256_snapshot", ""),
            "source_exists": _truth(row.get("source_exists_now")),
            "status": row.get("final2_observation_status", ""),
            "status_reason": row.get("final2_status_reason", "")
            or row.get("failure_reason", ""),
            "failure_reason": row.get("failure_reason", ""),
            "recommended_action": row.get("recommended_action", ""),
            "step4_eligible": _truth(row.get("step4_inference_eligible")),
            "step10_eligible": _truth(row.get("step10_inference_eligible")),
            "p5_eligible": _truth(row.get("physics_inference_eligible")),
            "step4_reason": row.get("step4_evidence", "")
            or row.get("step4_calculation_failure", ""),
            "step10_reason": row.get("step10_evidence", ""),
            "p5_reason": row.get("physics5_evidence", ""),
            "step4_segment_reference": row.get("step4_segment_reference", ""),
            "step10_segment_reference": row.get("step10_segment_reference", ""),
            "reference_soh_percent": _number(row.get("soh_label_percent")),
            "reference_status": row.get("soh_label_status", ""),
            "reference_source": row.get("soh_label_source", ""),
            "evaluation_eligible": _truth(row.get("soh_evaluation_eligible"))
            and _truth(row.get("label_evaluation_eligible")),
            "partition": row.get("final2_partition", ""),
            "pydma_available": _truth(row.get("pydma_charge_available")),
            "pydma_summary_status": row.get("pydma_charge_saved_qc_status", ""),
        }

    @staticmethod
    def _eol_record(row: dict[str, str]) -> dict[str, Any]:
        key = _identity_key(row["group"], row["cell"], row["rpt"])
        return {
            "key": key,
            "group": key[1],
            "cell": key[2],
            "rpt_index": int(key[3]),
            "is_eol": True,
            "inventory_scope": "VERIFIED_DEMO_ONLY",
            "mapping_status": row.get("mapping_status", ""),
            "canonical_mapping_status": "",
            "guide_mapping_status": "GUIDE_VERIFIED",
            "identity_provenance": row.get("identity_provenance", ""),
            "procedure": row.get("procedure", ""),
            "guide_stage": row.get("guide_entry", ""),
            "source_file": row.get("source_file", ""),
            "source_sha256": row.get("source_sha256", ""),
            "source_exists": Path(ROOT / Path(row["source_file"])).is_file(),
            "status": "DEMO_ONLY",
            "status_reason": "Verified cycling EoL; excluded from frozen evaluation.",
            "failure_reason": row.get("step10_review_reason", ""),
            "recommended_action": "Use the discharge-only baseline only.",
            "step4_eligible": _truth(row.get("discharge_only_inference_eligible")),
            "step10_eligible": _truth(row.get("step10_contiguous_extractor_eligible")),
            "p5_eligible": _truth(row.get("p5_physics_eligible")),
            "step4_reason": "",
            "step10_reason": row.get("step10_review_reason", ""),
            "p5_reason": "",
            "step4_segment_reference": row.get("step4_segment_reference", ""),
            "step10_segment_reference": "",
            "reference_soh_percent": _number(row.get("soh_percent")),
            "reference_status": row.get("soh_reference_status", ""),
            "reference_source": row.get("source_file", ""),
            "evaluation_eligible": False,
            "partition": "demo_only",
            "pydma_available": False,
            "pydma_summary_status": "not_fitted_for_demo_only_eol",
        }

    def _load_frozen_external(
        self,
    ) -> tuple[
        dict[tuple[str, str, str, str], dict[str, Any]],
        dict[tuple[str, str, str, str], float],
    ]:
        try:
            manifest = json.loads(DATASET_HASH_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise DashboardDataError(
                f"Could not read frozen dataset hash manifest: {error}"
            ) from error
        if manifest.get("dataset_identifier") != DATASET_IDENTIFIER:
            raise DashboardDataError("Frozen dataset identifier is not the expected version.")
        expected_hash = (
            manifest.get("artifacts", {})
            .get("external_test.npz", {})
            .get("sha256")
        )
        if not expected_hash or _sha256(EXTERNAL_NPZ_PATH) != expected_hash:
            raise DashboardDataError(
                "Frozen external tensor is missing or does not match its saved SHA-256."
            )
        try:
            with np.load(EXTERNAL_NPZ_PATH, allow_pickle=False) as archive:
                required = {
                    "X_discharge",
                    "X_charge",
                    "X_physics_raw",
                    "y_soh",
                    "dataset_part",
                    "physical_cell_id",
                    "rpt_index",
                }
                if set(archive.files) != required:
                    raise DashboardDataError(
                        "Frozen external tensor keys do not match the saved input contract."
                    )
                discharge = np.asarray(archive["X_discharge"], dtype=np.float32)
                charge = np.asarray(archive["X_charge"], dtype=np.float32)
                physics = np.asarray(archive["X_physics_raw"], dtype=np.float32)
                targets = np.asarray(archive["y_soh"], dtype=np.float32).reshape(-1)
                parts = np.asarray(archive["dataset_part"])
                cells = np.asarray(archive["physical_cell_id"])
                rpts = np.asarray(archive["rpt_index"])
        except (OSError, ValueError) as error:
            raise DashboardDataError(
                f"Could not load frozen external tensor: {error}"
            ) from error

        sample_count = int(targets.shape[0])
        if (
            discharge.shape != (sample_count, 128)
            or charge.shape != (sample_count, 128)
            or physics.shape != (sample_count, len(PHYSICS_FEATURE_NAMES))
            or parts.shape != (sample_count,)
            or cells.shape != (sample_count,)
            or rpts.shape != (sample_count,)
            or not all(
                np.isfinite(values).all()
                for values in (discharge, charge, physics, targets)
            )
        ):
            raise DashboardDataError("Frozen external tensor shapes or values are invalid.")

        inputs: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        references: dict[tuple[str, str, str, str], float] = {}
        for index in range(sample_count):
            if int(parts[index]) != 3:
                raise DashboardDataError(
                    "The Part 3 dashboard external tensor contains a non-Part-3 sample."
                )
            group, cell = _group_cell_from_npz(str(cells[index]))
            key = _identity_key(group, cell, rpts[index])
            record = self._records.get(key)
            if (
                record is None
                or record["is_eol"]
                or record["status"] != "ACCEPTED"
                or not record["evaluation_eligible"]
            ):
                raise DashboardDataError(
                    f"Frozen external sample has no matching eligible canonical record: {key}"
                )
            if key in inputs:
                raise DashboardDataError(f"Duplicate frozen external sample: {key}")
            reference = float(targets[index])
            audit_reference = record["reference_soh_percent"]
            if audit_reference is None or not math.isclose(
                reference, audit_reference, rel_tol=0, abs_tol=1e-4
            ):
                raise DashboardDataError(
                    f"Frozen target conflicts with the exact Part 3 audit label for {key}"
                )
            feature_row: dict[str, Any] = {
                "part": "3",
                "group": group,
                "physical_cell_id": cell,
                "rpt_index": key[3],
                "source_file": record["source_file"],
            }
            for name, value in zip(
                (f"step4_voltage_progress_{i:03d}" for i in range(128)),
                discharge[index],
            ):
                feature_row[name] = float(value)
            for name, value in zip(
                (f"step10_voltage_progress_{i:03d}" for i in range(128)),
                charge[index],
            ):
                feature_row[name] = float(value)
            feature_row["p5_physics_features"] = [
                float(value) for value in physics[index]
            ]
            inputs[key] = feature_row
            references[key] = reference

        expected_eligible = {
            key
            for key, record in self._records.items()
            if not record["is_eol"] and record["evaluation_eligible"]
        }
        if set(inputs) != expected_eligible:
            raise DashboardDataError(
                "Frozen external samples do not exactly match the eligible Part 3 cohort."
            )
        return inputs, references

    def metadata(self) -> dict[str, Any]:
        catalog = self.runtime.model_catalog()
        models = [
            {
                "name": item["name"],
                "architecture": item["architecture"],
                "modalities": item["modalities"],
                "checkpoint_id": item["checkpoint_id"],
                "checkpoint_sha256": self.runtime.manifest.get(item["name"], {}).get(
                    "best_sha256"
                ),
                "validation_metrics": item["validation_metrics"],
                "checkpoint_status": item["checkpoint_status"],
                "default": item["default"],
            }
            for item in catalog
        ]
        return {
            "dataset_identifier": self.dataset_identifier,
            "preprocessing_contract": PREPROCESSING_CONTRACT,
            "groups": sorted(self._cells_by_group, key=int),
            "cells_by_group": dict(self._cells_by_group),
            "model_catalog": models,
            "default_model": self.runtime.default_model(),
            "observation_count": len(self._records),
            "canonical_rpt_count": sum(
                not record["is_eol"] for record in self._records.values()
            ),
            "frozen_external_count": len(self._frozen_inputs),
            "demo_only_eol_count": sum(
                record["is_eol"] for record in self._records.values()
            ),
            "cell_count": sum(len(cells) for cells in self._cells_by_group.values()),
        }

    def health(self) -> dict[str, Any]:
        available = [
            item
            for item in self.runtime.model_catalog()
            if item["checkpoint_status"]["available"]
        ]
        return {
            "status": "ok" if available else "degraded",
            "dataset_identifier": self.dataset_identifier,
            "available_model_count": len(available),
            "default_model": self.runtime.default_model(),
        }

    def cell_data(
        self,
        group: str,
        cell: str,
        *,
        model_names: list[str] | None = None,
        include_eol: bool = True,
    ) -> dict[str, Any]:
        group_text, cell_text = str(int(group)), str(cell).strip()
        if cell_text not in self._cells_by_group.get(group_text, []):
            raise DashboardDataError(
                f"Cell {cell_text} is not present in Part 3 Group {group_text}."
            )
        catalog = {item["name"]: item for item in self.runtime.model_catalog()}
        selected_models = list(dict.fromkeys(model_names or [self.runtime.default_model()]))
        if not selected_models:
            selected_models = [self.runtime.default_model()]
        unknown = set(selected_models) - set(MODEL_NAMES)
        if unknown:
            raise DashboardDataError(
                "Unknown SoH model selection: " + ", ".join(sorted(unknown))
            )
        unavailable = [
            name
            for name in selected_models
            if not catalog[name]["checkpoint_status"]["available"]
        ]
        if unavailable:
            raise DashboardDataError(
                "Selected frozen model checkpoint is unavailable: "
                + ", ".join(unavailable)
            )

        rows = [
            dict(record)
            for key, record in self._records.items()
            if key[1] == group_text
            and key[2] == cell_text
            and (include_eol or not record["is_eol"])
        ]
        rows.sort(key=lambda row: (row["rpt_index"], row["is_eol"]))
        by_key = {row["key"]: row for row in rows}

        input_rows: dict[tuple[str, str, str, str], dict[str, Any]] = {
            key: self._frozen_inputs[key]
            for key in by_key
            if key in self._frozen_inputs
        }
        source_errors: dict[tuple[str, str, str, str], str] = {}
        source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["key"] in input_rows or not row["step4_eligible"]:
                continue
            if row["source_file"]:
                source_groups[row["source_file"]].append(row)
        for source_rows in source_groups.values():
            extracted, error = self._extract_source(source_rows[0])
            if error:
                for row in source_rows:
                    source_errors[row["key"]] = error
                continue
            for row in source_rows:
                extracted_record = extracted.get(row["key"])
                if extracted_record is None:
                    source_errors[row["key"]] = (
                        "verified_extractor_did_not_return_exact_rpt_identity"
                    )
                    continue
                feature_row = self._feature_row(extracted_record)
                if feature_row is not None:
                    input_rows[row["key"]] = feature_row

        for model_name in selected_models:
            self._predict_rows(model_name, by_key, input_rows)

        for row in rows:
            row["chart_rpt"] = (
                f"RPT{row['rpt_index']} EoL"
                if row["is_eol"]
                else f"RPT{row['rpt_index']}"
            )
            row["predictions"] = {}
            for model_name in selected_models:
                cache_key = self._prediction_key(model_name, row["key"])
                prediction = self._prediction_cache.get(cache_key)
                if prediction is None:
                    prediction = {
                        "predicted_soh_percent": None,
                        "status": "prediction_unavailable",
                        "reason": self._missing_input_reason(
                            row, model_name, source_errors.get(row["key"])
                        ),
                        "checkpoint_sha256": catalog[model_name].get(
                            "checkpoint_sha256"
                        ),
                    }
                row["predictions"][model_name] = prediction
            row["degradation"] = self._degradation(row)

        selected = selected_models[0]
        metrics = self._exploratory_metrics(rows, selected)
        summary = self._summary(
            rows,
            selected,
            metrics,
            input_rows,
            catalog[selected]["modalities"],
        )
        safe_rows = [self._public_record(row) for row in rows]
        return {
            "group": group_text,
            "cell": cell_text,
            "models": selected_models,
            "selected_model": selected,
            "observation_count": len(rows),
            "summary": summary,
            "metrics": metrics,
            "observations": safe_rows,
        }

    def rpt_detail(
        self,
        group: str,
        cell: str,
        rpt: str,
        *,
        include_eol: bool = True,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        response = self.cell_data(
            group,
            cell,
            model_names=[model_name or self.runtime.default_model()],
            include_eol=include_eol,
        )
        key = _identity_key(group, cell, rpt)
        record = self._records.get(key)
        if record is None or (record["is_eol"] and not include_eol):
            raise DashboardDataError(f"Part 3 RPT not found: {key}")
        source_record, error = self._curve_record(record)
        return {
            "observation": next(
                row for row in response["observations"] if row["key"] == list(key)
            ),
            "curves": self._curve_payload(source_record) if source_record else None,
            "curve_error": error,
        }

    def _extract_source(
        self, record: dict[str, Any]
    ) -> tuple[dict[tuple[str, str, str, str], dict[str, Any]], str]:
        source_file = str(record["source_file"])
        cache_key = (
            self.dataset_identifier,
            _normalized_path(source_file),
            str(record.get("source_sha256") or ""),
            PREPROCESSING_CONTRACT,
        )
        with self._source_lock:
            cached = self._source_cache.get(cache_key)
            if cached is not None:
                return cached
            relative = Path(source_file.replace("/", "\\"))
            source_path = (ROOT / relative).resolve()
            if not source_path.is_relative_to(ROOT.resolve()):
                return {}, "source_path_is_outside_project_root"
            if not source_path.is_file():
                return {}, "mapped_source_file_missing"
            try:
                adapter = _ExtractionRuntime(self.runtime)
                analysis = analyze_tpg_file(
                    source_path,
                    source_path.name,
                    adapter,
                    STEP4_BASELINE,
                )
            except (TpgIngestionError, OSError, UnicodeDecodeError, ValueError, KeyError) as error:
                result = ({}, f"verified_curve_extraction_failed:{error}")
                self._source_cache[cache_key] = result
                return result
            records: dict[tuple[str, str, str, str], dict[str, Any]] = {}
            for output in analysis.get("rpts", []):
                try:
                    key = _identity_key(
                        output.get("group", analysis.get("group_id")),
                        output.get("physical_cell_id", analysis.get("cell_id")),
                        output.get("rpt_index"),
                    )
                except DashboardDataError:
                    continue
                if key[0] != "3":
                    continue
                if key in records:
                    result = ({}, f"verified_extractor_returned_duplicate_identity:{key}")
                    self._source_cache[cache_key] = result
                    return result
                records[key] = output
            if not records:
                result = ({}, "verified_extractor_returned_no_part3_rpt_rows")
                self._source_cache[cache_key] = result
                return result
            for key, output in records.items():
                resolution = self._review_resolutions.get(key)
                if resolution is None:
                    continue
                try:
                    if key in STEP10_REPAIR_KEYS:
                        step10 = extract_rpt12_step10(source_path, resolution)
                        output.setdefault("inputs", {})["step10"] = {
                            "available": True,
                            "status": step10["status"],
                            "raw_voltage": step10["raw_voltage"],
                            "processed_128_voltage": step10[
                                "processed_128_voltage"
                            ],
                            "reason": "",
                            "row_start": step10["row_start"],
                            "row_end": step10["row_end"],
                            "observed_rows": step10["observed_rows"],
                        }
                        source_sha256 = resolution["source_sha256"]
                    else:
                        source_sha256 = verify_review_source(source_path, resolution)
                except ReviewPreprocessingError as error:
                    result = ({}, f"review_resolution_source_invalid:{error}")
                    self._source_cache[cache_key] = result
                    return result
                output["source_sha256"] = source_sha256
                output["source_checksum_verified"] = True
            result = (records, "")
            self._source_cache[cache_key] = result
            return result

    def _feature_row(self, output: dict[str, Any]) -> dict[str, Any] | None:
        inputs = output.get("inputs", {})
        step4 = inputs.get("step4", {})
        if not step4.get("available"):
            return None
        discharge = step4.get("processed_128_voltage", [])
        if len(discharge) != 128 or not all(_number(value) is not None for value in discharge):
            return None
        row: dict[str, Any] = {
            "part": "3",
            "group": str(output.get("group", "")),
            "physical_cell_id": str(output.get("physical_cell_id", "")),
            "rpt_index": _rpt_number(output.get("rpt_index")),
            "inventory_scope": output.get("inventory_scope", "GUIDE_MAPPED"),
            "procedure": output.get("procedure", ""),
            "guide_stage": output.get("guide_stage", ""),
            "mapping_status": output.get("mapping_status", ""),
            "mapping_provenance": output.get("mapping_provenance", ""),
            "mapping_evidence": output.get("mapping_evidence", ""),
            "source_file": output.get("source_file", ""),
            "source_sha256": output.get("source_sha256", ""),
            "source_checksum_verified": output.get("source_checksum_verified", False),
            "is_eol_procedure": output.get("is_eol_procedure", False),
            "demo_step4_valid": True,
            "demo_step10_valid": bool(inputs.get("step10", {}).get("available")),
            "demo_p5_valid": False,
        }
        for index, value in enumerate(discharge):
            row[f"step4_voltage_progress_{index:03d}"] = float(value)
        step10 = inputs.get("step10", {})
        charge = step10.get("processed_128_voltage", [])
        if step10.get("available") and len(charge) == 128 and all(
            _number(value) is not None for value in charge
        ):
            row["demo_step10_valid"] = True
            for index, value in enumerate(charge):
                row[f"step10_voltage_progress_{index:03d}"] = float(value)

        p5 = inputs.get("p5_physics_features", {})
        physics = [_number(p5.get(name, {}).get("value")) for name in PHYSICS_FEATURE_NAMES]
        if all(value is not None for value in physics):
            row["p5_physics_features"] = [float(value) for value in physics]
            row["demo_p5_valid"] = True
        return row

    def _predict_rows(
        self,
        model_name: str,
        records: dict[tuple[str, str, str, str], dict[str, Any]],
        input_rows: dict[tuple[str, str, str, str], dict[str, Any]],
    ) -> None:
        catalog = {item["name"]: item for item in self.runtime.model_catalog()}
        model_hash = str(catalog[model_name].get("checkpoint_sha256") or "")
        pending = [
            key
            for key in records
            if key in input_rows
            and self._prediction_key(model_name, key) not in self._prediction_cache
        ]
        if not pending:
            return
        with self._inference_lock:
            result = self.runtime.preprocess_rows(
                [input_rows[key] for key in pending], model_name
            )
        result_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}
        for item in result.get("results", []):
            try:
                key = _identity_key(
                    item.get("group"),
                    item.get("physical_cell_id"),
                    item.get("rpt_index"),
                )
            except DashboardDataError:
                continue
            if key in result_by_key:
                raise DashboardDataError(
                    f"Model returned duplicate Part 3 identity for {model_name}: {key}"
                )
            result_by_key[key] = item
        for key in pending:
            item = result_by_key.get(key)
            if item is None:
                prediction = {
                    "predicted_soh_percent": None,
                    "status": "prediction_unavailable",
                    "reason": "model_runtime_did_not_return_exact_rpt_identity",
                    "checkpoint_sha256": model_hash,
                }
            else:
                prediction = {
                    "predicted_soh_percent": _number(
                        item.get("predicted_soh_percent")
                    ),
                    "status": item.get("soh_status", "prediction_unavailable"),
                    "reason": item.get("reason", ""),
                    "checkpoint_sha256": item.get("checkpoint_sha256", model_hash),
                    "modalities": item.get("modalities", []),
                    "modality_eligibility": item.get("modality_eligibility", {}),
                    "modality_eligibility_reasons": item.get(
                        "modality_eligibility_reasons", {}
                    ),
                }
                if prediction["predicted_soh_percent"] is None and not prediction["reason"]:
                    prediction["reason"] = "prediction_unavailable"
            self._prediction_cache[self._prediction_key(model_name, key)] = prediction

    def _prediction_key(
        self, model_name: str, key: tuple[str, str, str, str]
    ) -> tuple[str, ...]:
        entry = self.runtime.manifest.get(model_name, {})
        return (
            self.dataset_identifier,
            model_name,
            str(entry.get("best_sha256") or ""),
            self.runtime.normalization_sha256,
            key[1],
            key[2],
            key[3],
            PREPROCESSING_CONTRACT,
        )

    def _missing_input_reason(
        self, record: dict[str, Any], model_name: str, source_error: str | None
    ) -> str:
        if source_error:
            return source_error
        if not record["step4_eligible"]:
            return "step4_ineligible:" + (
                record["step4_reason"] or record["status_reason"] or "not verified"
            )
        required = self.runtime.model_catalog()
        model = next((item for item in required if item["name"] == model_name), None)
        modalities = model["modalities"] if model else []
        if "step10_voltage" in modalities and not record["step10_eligible"]:
            return "step10_ineligible:" + (
                record["step10_reason"] or "no approved contiguous Step 10 segment"
            )
        if "p5_physics" in modalities and not record["p5_eligible"]:
            return "p5_physics_ineligible:" + (
                record["p5_reason"] or "verified five-feature vector unavailable"
            )
        return "verified_model_input_not_available"

    def _degradation(self, record: dict[str, Any]) -> dict[str, Any]:
        key = record["key"]
        if record["is_eol"]:
            return {
                "status": "no_saved_pydma_for_demo_only_eol",
                "reason": "No PyDMA fit was run for this demo-only EoL source.",
                "chart_eligible": False,
            }
        cached = self._degradation_cache.get(key)
        if cached is not None:
            return dict(cached)
        try:
            saved = load_shared_charge_degradation(*key)
        except (OSError, csv.Error, UnicodeDecodeError, ValueError) as error:
            saved = {
                "status": "charge_artifact_read_failed",
                "reason": f"{type(error).__name__}:{error}",
            }
        same_bol_ok = (
            saved.get("same_cell_bol_fit_accepted") is True
            and saved.get("same_cell_bol_status") == "same_cell_bol_fit_accepted"
            and bool(saved.get("same_cell_bol_source"))
        )
        fit_ok = (
            saved.get("status") == "accepted_charge_pydma_diagnostic_not_ground_truth"
            and _number(saved.get("fit_rmse_mV")) is not None
            and float(saved["fit_rmse_mV"]) < 15.0
            and _number(saved.get("accepted_runs")) is not None
            and int(saved["accepted_runs"]) >= 3
            and same_bol_ok
        )
        values = {
            "lli_percent": _number(saved.get("LLI_percent")),
            "lam_anode_percent": _number(saved.get("Anode_LAM_percent")),
            "lam_cathode_percent": _number(saved.get("Cathode_LAM_percent")),
        }
        if not fit_ok:
            values = {name: None for name in values}
        result = {
            "status": saved.get("status", "charge_fit_unavailable"),
            "reason": saved.get("reason", ""),
            "chart_eligible": fit_ok,
            **values,
            "diagnostic_values": (
                {
                    "lli_percent": _number(saved.get("LLI_percent")),
                    "lam_anode_percent": _number(saved.get("Anode_LAM_percent")),
                    "lam_cathode_percent": _number(saved.get("Cathode_LAM_percent")),
                }
                if saved.get("status")
                == "accepted_charge_fit_final2_observation_rejected_diagnostic_only"
                else None
            ),
            "fit_rmse_mV": _number(saved.get("fit_rmse_mV")),
            "accepted_runs": _number(saved.get("accepted_runs")),
            "total_runs": _number(saved.get("total_runs")),
            "source_file": saved.get("charge_source_file") or saved.get("source_file"),
            "source_artifact": saved.get("source_artifact"),
            "same_cell_bol_source": saved.get("same_cell_bol_source"),
            "same_cell_bol_fit_accepted": saved.get("same_cell_bol_fit_accepted"),
            "same_cell_bol_status": saved.get("same_cell_bol_status"),
            "same_cell_bol_note": saved.get("same_cell_bol_note"),
        }
        self._degradation_cache[key] = result
        return dict(result)

    @staticmethod
    def _exploratory_metrics(
        rows: list[dict[str, Any]], model_name: str
    ) -> dict[str, Any]:
        pairs = [
            (row["reference_soh_percent"], row["predictions"][model_name]["predicted_soh_percent"])
            for row in rows
            if row["evaluation_eligible"]
            and not row["is_eol"]
            and row["status"] == "ACCEPTED"
            and row["reference_soh_percent"] is not None
            and row["predictions"][model_name]["predicted_soh_percent"] is not None
        ]
        n = len(pairs)
        if not n:
            return {
                "scope": "exploratory per-cell diagnostics; frozen eligible observations only",
                "n": 0,
                "mae_pp": None,
                "rmse_pp": None,
                "r2": None,
                "r2_reason": "fewer_than_two_valid_pairs",
            }
        actual = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        predicted = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        error = predicted - actual
        variance = float(np.sum((actual - np.mean(actual)) ** 2))
        r2 = (
            1.0 - float(np.sum(error**2)) / variance
            if n >= 2 and variance > 0
            else None
        )
        return {
            "scope": "exploratory per-cell diagnostics; frozen eligible observations only",
            "n": n,
            "mae_pp": float(np.mean(np.abs(error))),
            "rmse_pp": float(np.sqrt(np.mean(error**2))),
            "r2": r2,
            "r2_reason": "" if r2 is not None else "fewer_than_two_pairs_or_zero_reference_variance",
        }

    @staticmethod
    def _summary(
        rows: list[dict[str, Any]],
        model_name: str,
        metrics: dict[str, Any],
        input_rows: dict[tuple[str, str, str, str], dict[str, Any]],
        modalities: list[str],
    ) -> dict[str, Any]:
        predictions = [
            row["predictions"][model_name]["predicted_soh_percent"]
            for row in rows
            if row["predictions"][model_name]["predicted_soh_percent"] is not None
        ]
        evaluation_rows = [
            row
            for row in rows
            if row["evaluation_eligible"] and not row["is_eol"]
        ]
        latest_prediction = next(
            (
                row["predictions"][model_name]["predicted_soh_percent"]
                for row in reversed(evaluation_rows)
                if row["predictions"][model_name]["predicted_soh_percent"] is not None
            ),
            None,
        )
        latest_reference = next(
            (
                row["reference_soh_percent"]
                for row in reversed(evaluation_rows)
                if row["reference_soh_percent"] is not None
            ),
            None,
        )
        return {
            "mapped_rpt_count": len(rows),
            "inference_eligible_count": sum(
                row["key"] in input_rows
                and all(
                    row[eligibility_field]
                    for modality, eligibility_field in (
                        ("step4_voltage", "step4_eligible"),
                        ("step10_voltage", "step10_eligible"),
                        ("p5_physics", "p5_eligible"),
                    )
                    if modality in modalities
                )
                for row in rows
            ),
            "successful_prediction_count": len(predictions),
            "frozen_evaluation_pair_count": metrics["n"],
            "latest_predicted_soh_percent": latest_prediction,
            "latest_reference_soh_percent": latest_reference,
            "frozen_eligible_observation_count": len(evaluation_rows),
        }

    def _public_record(self, record: dict[str, Any]) -> dict[str, Any]:
        safe = {
            key: value
            for key, value in record.items()
            if key
            not in {
                "key",
                "p5_reason",
                "step4_reason",
                "step10_reason",
                "source_sha256",
                "source_exists",
            }
        }
        safe["key"] = list(record["key"])
        safe["source_exists"] = bool(record["source_exists"])
        safe["source_sha256"] = record["source_sha256"]
        safe["modality_eligibility"] = {
            "step4_voltage": record["step4_eligible"],
            "step10_voltage": record["step10_eligible"],
            "p5_physics": record["p5_eligible"],
        }
        safe["modality_reasons"] = {
            "step4_voltage": record["step4_reason"],
            "step10_voltage": record["step10_reason"],
            "p5_physics": record["p5_reason"],
        }
        safe["display_reference_soh_percent"] = record["reference_soh_percent"]
        safe["reference_soh_percent"] = (
            self._frozen_references.get(record["key"])
            if record["key"] in self._frozen_references
            else record["reference_soh_percent"]
        )
        return safe

    def _curve_record(
        self, record: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str]:
        extracted, error = self._extract_source(record)
        if error:
            return None, error
        output = extracted.get(record["key"])
        if output is None:
            return None, "verified_extractor_did_not_return_exact_rpt_identity"
        return output, ""

    @staticmethod
    def _curve_payload(output: dict[str, Any]) -> dict[str, Any]:
        inputs = output.get("inputs", {})
        step4 = inputs.get("step4", {})
        step10 = inputs.get("step10", {})

        def voltage_values(values: Any) -> list[float]:
            result = [_number(value) for value in values]
            if any(value is None for value in result):
                raise DashboardDataError(
                    "Verified source curve contains a non-finite voltage value."
                )
            return [float(value) for value in result]

        return {
            "step4": {
                "available": bool(step4.get("available")),
                "raw_voltage": voltage_values(step4.get("raw_voltage", [])),
                "processed_128_voltage": voltage_values(
                    step4.get("processed_128_voltage", [])
                ),
                "reason": step4.get("reason", ""),
            },
            "step10": {
                "available": bool(step10.get("available")),
                "raw_voltage": voltage_values(step10.get("raw_voltage", [])),
                "processed_128_voltage": voltage_values(
                    step10.get("processed_128_voltage", [])
                ),
                "reason": step10.get("reason", ""),
            },
        }
