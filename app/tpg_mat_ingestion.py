from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from array import array
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_MAPPING_PATH = ROOT / "data" / "manifests" / "unified_raw_mapping.csv"
RPT_MANIFEST_PATH = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
PART3_ELIGIBILITY_PATH = ROOT / "app" / "part3_modality_eligibility.csv"
PART3_DEMO_EOL_MAPPING_PATH = (
    ROOT / "data" / "manifests" / "part3_demo_eol_mapping_v1.csv"
)
RPT10_ZERO_CURRENT_IDENTITY = ("3", "10", "G10-Cell2", "10")
FROZEN_FINAL2_DIR = (
    ROOT / "experiments" / "final2" / "training_6_models" / "frozen_dataset"
)
FROZEN_OBSERVATION_MANIFEST = FROZEN_FINAL2_DIR / "observation_manifest.csv"
G10_P5_EXTENSION_PATH = FROZEN_FINAL2_DIR / "g10_cell2_p5_label_extension.csv"
FROZEN_DATASET_SHA_MANIFEST = FROZEN_FINAL2_DIR / "dataset_sha256_manifest.json"
FROZEN_BUILD_MANIFEST = (
    ROOT / "experiments" / "final2" / "training_6_models" / "frozen_dataset_manifest_v2.json"
)
FROZEN_TRAINING_CONFIG = (
    ROOT / "experiments" / "final2" / "training_6_models" / "training_config_v2.json"
)
FROZEN_VALIDATION_SELECTION = (
    ROOT / "experiments" / "final2" / "training_6_models" / "run_001" / "validation_selection.json"
)
FROZEN_EXTERNAL_MANIFEST = (
    ROOT
    / "experiments"
    / "final2"
    / "training_6_models"
    / "external_evaluation"
    / "external_evaluation_manifest.json"
)
G10_OWNER_PROVENANCE = (
    "Group 10 Cell 2 identity confirmed by project owner; explicit Cell 2 label "
    "omitted from printed Guide."
)
G10_RPT10_SOURCE_FILE = (
    "data/raw/oxford/part3/Group_10/Group 10/TPG10.19-Cell2.csv"
)
G10_RPT10_SOURCE_SHA256 = (
    "7af1228384504c2e54b32088528cae7b5a4a043e899a365a1d4bcf6ed28e8fcb"
)
G10_OWNER_MAPPING_STATUS = "owner_confirmed_frozen_final2_exception"
FROZEN_DATASET_IDENTIFIER = "bafuse-final2-p5-g10-guide-verified-v2"
H2_V3_PATH = ROOT / "experiments" / "pydma_hybrid_direction_v3" / "h2_hybrid_v3.csv"
SPLIT_PATH = ROOT / "reports" / "bafuse_final_data_validation_v2" / "cell_split_12train_4val_12external.csv"
SOH_REFERENCE_PATHS = {
    "1": ROOT
    / "data"
    / "processed"
    / "label_soh_part1_2"
    / "run_007_part1_capacity_qc"
    / "labels_soh.csv",
    "2": ROOT
    / "data"
    / "processed"
    / "label_soh_part1_2"
    / "run_010_part2_soh_candidates"
    / "labels_soh_part2.csv",
    "3": ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part3_soh_candidates"
    / "labels_soh_candidates.csv",
}
VALID_REFERENCE_STATUS = "calculated_from_measured_capacity"
MATLAB_CONVERTER_DIR = ROOT / "app" / "matlab"
MATLAB_CONVERTER_SCRIPT = MATLAB_CONVERTER_DIR / "bafuse_export_tpg_mat.m"
RPT_NUMBER = re.compile(r"RPT\s*(\d+)", re.IGNORECASE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
REQUIRED_MEASUREMENT_COLUMNS = ("Step", "Amphr", "Amps", "Volts")
MEASUREMENT_COMPARE_TOLERANCE = 1e-9


class TpgIngestionError(ValueError):
    pass


class MatlabRuntimeUnavailable(TpgIngestionError):
    pass


def _norm(value: Any) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _basename(value: Any) -> str:
    return PurePosixPath(_norm(value)).name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_rows(path: Path, label: str) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise TpgIngestionError(f"{label} has no CSV header: {path}")
            return list(reader)
    except (OSError, csv.Error, UnicodeDecodeError) as error:
        raise TpgIngestionError(f"Could not read {label}: {error}") from error


def _read_json_manifest(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as error:
        raise TpgIngestionError(f"Could not read {label}: {error}") from error
    if not isinstance(value, dict):
        raise TpgIngestionError(f"{label} must contain a JSON object.")
    return value


def _verified_frozen_observation_rows() -> list[dict[str, str]]:
    hash_manifest = _read_json_manifest(
        FROZEN_DATASET_SHA_MANIFEST, "Final2 frozen dataset SHA-256 manifest"
    )
    build_manifest = _read_json_manifest(
        FROZEN_BUILD_MANIFEST, "Final2 frozen dataset build manifest"
    )
    training_config = _read_json_manifest(
        FROZEN_TRAINING_CONFIG, "Final2 frozen training configuration"
    )
    validation_selection = _read_json_manifest(
        FROZEN_VALIDATION_SELECTION, "Final2 validation selection manifest"
    )
    external_manifest = _read_json_manifest(
        FROZEN_EXTERNAL_MANIFEST, "Final2 external evaluation manifest"
    )

    hash_manifest_digest = _sha256(FROZEN_DATASET_SHA_MANIFEST)
    declared_hash_manifest_digests = {
        str(training_config.get("dataset_sha256_manifest_sha256", "")).casefold(),
        str(validation_selection.get("dataset_sha256_manifest_sha256", "")).casefold(),
        str(external_manifest.get("dataset_sha256_manifest_sha256", "")).casefold(),
        str(
            build_manifest.get("dataset_artifact_sha256", {}).get(
                "dataset_sha256_manifest.json", ""
            )
        ).casefold(),
    }
    if (
        len(declared_hash_manifest_digests) != 1
        or not _SHA256_RE.fullmatch(hash_manifest_digest)
        or declared_hash_manifest_digests != {hash_manifest_digest.casefold()}
    ):
        raise TpgIngestionError(
            "Final2 frozen checksum anchors disagree; refusing canonical source-hash fallback."
        )

    dataset_identifiers = {
        str(manifest.get("dataset_identifier", "")).strip()
        for manifest in (
            hash_manifest,
            build_manifest,
            training_config,
            validation_selection,
            external_manifest,
        )
    }
    if dataset_identifiers != {FROZEN_DATASET_IDENTIFIER}:
        raise TpgIngestionError(
            "Final2 frozen manifests refer to different datasets; refusing source-hash fallback."
        )

    relative_manifest_path = FROZEN_OBSERVATION_MANIFEST.relative_to(ROOT).as_posix()
    declared_artifact = hash_manifest.get("artifacts", {}).get(
        "observation_manifest.csv", {}
    )
    declared_observation_digest = str(declared_artifact.get("sha256", "")).casefold()
    build_observation_digest = str(
        build_manifest.get("dataset_artifact_sha256", {}).get(
            "observation_manifest.csv", ""
        )
    ).casefold()
    actual_observation_digest = _sha256(FROZEN_OBSERVATION_MANIFEST).casefold()
    if (
        _norm(declared_artifact.get("path", "")) != _norm(relative_manifest_path)
        or not _SHA256_RE.fullmatch(actual_observation_digest)
        or declared_observation_digest != actual_observation_digest
        or build_observation_digest != actual_observation_digest
    ):
        raise TpgIngestionError(
            "The Final2 frozen observation manifest failed checksum verification."
        )
    return _manifest_rows(
        FROZEN_OBSERVATION_MANIFEST, "Final2 frozen observation manifest"
    )


def _frozen_source_sha256(
    mapping: dict[str, str],
    observations: list[dict[str, str]],
) -> str:
    frozen_rows = _verified_frozen_observation_rows()
    source_path = _norm(mapping.get("repository_path", ""))
    hashes: set[str] = set()
    for observation in observations:
        rpt_index = str(observation.get("rpt_index", "")).strip()
        matching = [
            row
            for row in frozen_rows
            if row.get("dataset_part", "").strip()
            == mapping.get("dataset_part", "").strip()
            and row.get("group", "").strip() == mapping.get("group_id", "").strip()
            and row.get("physical_cell_id", "").strip()
            == mapping.get("physical_cell_id", "").strip()
            and row.get("rpt_index", "").strip() == rpt_index
            and _norm(row.get("source_file", "")) == source_path
        ]
        if len(matching) != 1:
            raise TpgIngestionError(
                "The frozen Final2 manifest does not contain one exact source/RPT identity "
                f"for {mapping.get('physical_cell_id', '')} RPT{rpt_index}."
            )
        row = matching[0]
        if row.get("mapping_status", "").strip().casefold() not in {
            "verified",
            "guide_verified",
        }:
            raise TpgIngestionError(
                f"The frozen source mapping is not verified for RPT{rpt_index}."
            )
        source_hashes = [
            str(row.get(name, "")).strip().casefold()
            for name in (
                "step4_source_sha256_snapshot",
                "step10_source_sha256_snapshot",
            )
            if str(row.get(name, "")).strip()
        ]
        if (
            not source_hashes
            or any(not _SHA256_RE.fullmatch(value) for value in source_hashes)
            or len(set(source_hashes)) != 1
        ):
            raise TpgIngestionError(
                f"The frozen source hash is missing or inconsistent for RPT{rpt_index}."
            )
        hashes.update(source_hashes)
    if len(hashes) != 1:
        raise TpgIngestionError(
            "The mapped RPTs do not share one verified source-file SHA-256."
        )
    return next(iter(hashes))


def frozen_soh_label_for_identity(
    identity: dict[str, Any],
) -> dict[str, Any] | None:
    source_path = _norm(identity.get("source_file", ""))
    matching = [
        row
        for row in _verified_frozen_observation_rows()
        if row.get("dataset_part", "").strip()
        == str(identity.get("part", "")).strip()
        and row.get("group", "").strip() == str(identity.get("group", "")).strip()
        and row.get("physical_cell_id", "").strip()
        == str(identity.get("physical_cell_id", "")).strip()
        and row.get("rpt_index", "").strip()
        == str(identity.get("rpt_index", "")).strip()
        and _norm(row.get("source_file", "")) == source_path
    ]
    if not matching:
        return None
    if len(matching) != 1:
        raise TpgIngestionError(
            "The frozen Final2 manifest contains duplicate label identities for this RPT."
        )
    row = matching[0]
    reference = {
        "value_percent": None,
        "status": "frozen_label_unavailable",
        "source_file": row.get("source_file", ""),
        "label_source": "verified_frozen_final2_observation_manifest",
        "source_sha256_provenance": (
            "verified_frozen_final2_observation_manifest"
        ),
    }
    if (
        row.get("status", "").strip().upper() != "ACCEPTED"
        or row.get("mapping_status", "").strip().casefold()
        not in {"verified", "guide_verified"}
    ):
        reference["status"] = "frozen_observation_not_accepted"
        return reference
    source_hashes = {
        str(row.get(name, "")).strip().casefold()
        for name in (
            "step4_source_sha256_snapshot",
            "step10_source_sha256_snapshot",
        )
        if str(row.get(name, "")).strip()
    }
    expected_hash = str(identity.get("source_sha256", "")).strip().casefold()
    if (
        len(source_hashes) != 1
        or not _SHA256_RE.fullmatch(expected_hash)
        or next(iter(source_hashes), "") != expected_hash
    ):
        reference["status"] = "frozen_label_source_sha256_mismatch"
        return reference
    try:
        value = float(row.get("soh_percent", ""))
    except (TypeError, ValueError):
        reference["status"] = "frozen_label_value_invalid"
        return reference
    if not np.isfinite(value):
        reference["status"] = "frozen_label_value_invalid"
        return reference
    reference.update(
        {
            "value_percent": value,
            "status": "calculated_from_measured_capacity",
            "source_sha256": expected_hash,
            "label_method": row.get("capacity_method", ""),
            "dataset_identifier": FROZEN_DATASET_IDENTIFIER,
        }
    )
    return reference


def _segment_signature(reference: str, expected_step: int) -> tuple[str, int, int]:
    text = str(reference).strip().replace("\\", "/")
    if "#step" in text.casefold():
        source, suffix = re.split(r"#step", text, maxsplit=1, flags=re.IGNORECASE)
        match = re.match(r"(\d+):.*?rows(\d+)-(\d+)", suffix, re.IGNORECASE)
        if not match or int(match.group(1)) != expected_step:
            raise TpgIngestionError(
                f"Frozen segment reference is not Step {expected_step}: {reference}"
            )
        return _norm(source), int(match.group(2)), int(match.group(3))
    match = re.match(r"(.+):rows(\d+)-(\d+)(?::.*)?$", text, re.IGNORECASE)
    if not match:
        raise TpgIngestionError(f"Malformed frozen segment reference: {reference}")
    return _norm(match.group(1)), int(match.group(2)), int(match.group(3))


def _integrate_step4_capacity(
    source_csv: Path,
    reference: str,
    *,
    expected_source_name: str,
) -> dict[str, Any]:
    source, start, end = _segment_signature(reference, 4)
    if Path(source).name.casefold() != expected_source_name.casefold():
        raise TpgIngestionError(
            f"Step 4 segment source does not match {expected_source_name}: {reference}"
        )
    try:
        with source_csv.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                raise TpgIngestionError(f"Step 4 source has no header: {source_csv}")
            indices: dict[str, int] = {}
            for index, name in enumerate(header):
                key = name.strip().casefold()
                if key in indices:
                    raise TpgIngestionError(
                        f"Step 4 source has duplicate column: {name}"
                    )
                indices[key] = index
            missing = {"step", "testtime", "amps", "volts"} - set(indices)
            if missing:
                raise TpgIngestionError(
                    "Step 4 source is missing columns: " + ", ".join(sorted(missing))
                )

            row_count = 0
            previous_time: float | None = None
            previous_current: float | None = None
            integral = 0.0
            testtime_values: list[float] = []
            current_values: list[float] = []
            voltage_values: list[float] = []
            delta_times: list[float] = []
            digest = hashlib.sha256()
            for values in reader:
                line_number = reader.line_num
                if line_number < start:
                    continue
                if line_number > end:
                    break
                row_count += 1
                digest.update(
                    ("\x1f".join(values) + "\n").encode(
                        "utf-8", errors="replace"
                    )
                )
                try:
                    step = float(values[indices["step"]])
                    testtime = float(values[indices["testtime"]])
                    current = float(values[indices["amps"]])
                    voltage = float(values[indices["volts"]])
                except (IndexError, TypeError, ValueError) as error:
                    raise TpgIngestionError(
                        f"Step 4 source has an invalid value at line {line_number}."
                    ) from error
                if not all(
                    np.isfinite(value)
                    for value in (step, testtime, current, voltage)
                ):
                    raise TpgIngestionError(
                        f"Step 4 source has a non-finite value at line {line_number}."
                    )
                if step != 4 or current >= 0:
                    raise TpgIngestionError(
                        f"Step 4 identity/current direction is invalid at line {line_number}."
                    )
                if previous_time is not None:
                    delta_time = testtime - previous_time
                    if delta_time <= 0:
                        raise TpgIngestionError(
                            f"Step 4 TestTime is not strictly increasing at line {line_number}."
                        )
                    delta_times.append(delta_time)
                    integral += (
                        (abs(previous_current) + abs(current))
                        * 0.5
                        * delta_time
                    )
                previous_time = testtime
                previous_current = current
                testtime_values.append(testtime)
                current_values.append(current)
                voltage_values.append(voltage)
    except TpgIngestionError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise TpgIngestionError(
            f"Could not read Step 4 source {source_csv}: {error}"
        ) from error

    expected_rows = end - start + 1
    if row_count != expected_rows or row_count < 2:
        raise TpgIngestionError(
            f"Step 4 segment row count mismatch: expected {expected_rows}, read {row_count}."
        )
    if not np.isfinite(integral) or integral <= 0:
        raise TpgIngestionError("Step 4 trapezoidal capacity integral is invalid.")
    return {
        "capacity_native_current_time": float(integral),
        "row_count": row_count,
        "source_block_sha256": digest.hexdigest(),
        "testtime_strictly_increasing": True,
        "testtime_delta_min": min(delta_times),
        "testtime_delta_median": float(np.median(delta_times)),
        "testtime_delta_max": max(delta_times),
        "testtime_values": testtime_values,
        "current_values": current_values,
        "voltage_values": voltage_values,
    }


def _demo_eol_mapping_for_file(
    filename: str,
) -> tuple[dict[str, str], list[dict[str, str]]] | None:
    if not PART3_DEMO_EOL_MAPPING_PATH.is_file():
        raise FileNotFoundError(
            f"Demo-only EoL mapping extension is missing: {PART3_DEMO_EOL_MAPPING_PATH}"
        )
    mappings = [
        row
        for row in _manifest_rows(
            PART3_DEMO_EOL_MAPPING_PATH, "Part 3 demo-only EoL mapping extension"
        )
        if row.get("record_type") == "cycling_eol"
        and _basename(row.get("source_file", "")) == Path(filename).name.casefold()
    ]
    if not mappings:
        return None
    if len(mappings) != 1:
        raise TpgIngestionError(
            "The EoL filename does not resolve to exactly one demo-only mapping."
        )
    record = mappings[0]
    required = (
        record.get("mapping_status") == "GUIDE_VERIFIED"
        and record.get("inventory_scope") == "VERIFIED_DEMO_ONLY"
        and record.get("is_eol_procedure", "").casefold() == "true"
        and record.get("selectable", "").casefold() == "true"
        and record.get("step10_eligible", "").casefold() == "false"
    )
    if not required:
        raise TpgIngestionError(
            "The EoL mapping lacks the required Guide, demo-scope, or Step 10 review status."
        )

    raw_rows = pd.read_csv(RAW_MAPPING_PATH, dtype=str, keep_default_na=False)
    exact = raw_rows[
        raw_rows["record_type"].eq("raw_file")
        & raw_rows["source_extension"].str.casefold().eq(".csv")
        & raw_rows["repository_path"].map(_norm).eq(_norm(record["source_file"]))
        & raw_rows["source_present"].str.casefold().eq("true")
    ]
    if len(exact) != 1:
        raise TpgIngestionError(
            "The EoL extension must match exactly one present raw-source inventory row."
        )
    inventory = exact.iloc[0].to_dict()
    source_path = ROOT / Path(record["source_file"])
    expected_hash = record["source_sha256"].strip().casefold()
    try:
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(record["source_size_bytes"])
            or source_path.stat().st_size
            != int(inventory.get("actual_size_bytes", "-1"))
            or not re.fullmatch(r"[0-9a-f]{64}", expected_hash)
            or _sha256(source_path).casefold() != expected_hash
        ):
            raise TpgIngestionError(
                "The EoL source path, size, or SHA-256 does not match the verified extension."
            )
    except OSError as error:
        raise TpgIngestionError(
            f"Could not verify the demo-only EoL source: {error}"
        ) from error

    reference = record["step4_segment_reference"]
    source, start, end = _segment_signature(reference, 4)
    if (
        source != _norm(record["source_file"])
        or end - start + 1 != int(record["step4_expected_rows"])
    ):
        raise TpgIngestionError(
            "The EoL Step 4 segment reference does not match its verified source and row count."
        )
    bol_path = ROOT / Path(record["bol_source_file"])
    bol_reference = record["bol_step4_segment_reference"]
    bol_source, bol_start, bol_end = _segment_signature(bol_reference, 4)
    if (
        not bol_path.is_file()
        or bol_source != _norm(record["bol_source_file"]).replace(
            "data/raw/oxford/part3/", ""
        )
        or bol_end < bol_start
        or not re.fullmatch(r"[0-9a-f]{64}", record["bol_source_sha256"].casefold())
    ):
        raise TpgIngestionError(
            "The same-cell BoL source or Step 4 reference is not valid."
        )
    try:
        if _sha256(bol_path).casefold() != record["bol_source_sha256"].casefold():
            raise TpgIngestionError(
                "The same-cell BoL source SHA-256 differs from its saved P5 snapshot."
            )
    except OSError as error:
        raise TpgIngestionError(
            f"Could not verify the same-cell BoL source: {error}"
        ) from error

    mapping = {
        **inventory,
        "dataset_part": "3",
        "group_id": record["group"],
        "physical_cell_id": record["cell"],
        "procedure": record["procedure"],
        "repository_path": record["source_file"],
        "source_sha256": expected_hash,
        "source_present": "true",
        "guide_stage": f"RPT{record['rpt']}",
        "guide_rpt_refs": f"RPT{record['rpt']}",
        "mapping_resolution_status": "GUIDE_VERIFIED",
        "mapping_resolution_provenance": record["identity_provenance"],
        "mapping_evidence": (
            f"Guide {record['guide_sheet']} row {record['guide_row']}; "
            "exact raw filename, source size, and SHA-256 verified"
        ),
        "inventory_scope": record["inventory_scope"],
        "is_eol_procedure": "true",
        "source_tpg": record["procedure"],
        "step10_eligible": record["step10_eligible"],
        "step10_review_reason": record["step10_review_reason"],
    }
    observation = {
        "part": record["part"],
        "group": record["group"],
        "physical_cell_id": record["cell"],
        "rpt_index": record["rpt"],
        "source_file": record["source_file"],
        "source_tpg": record["procedure"],
        "procedure": record["procedure"],
        "guide_stage": f"RPT{record['rpt']}",
        "mapping_status": record["mapping_status"],
        "mapping_resolution_status": record["mapping_status"],
        "mapping_resolution_provenance": record["identity_provenance"],
        "mapping_evidence": mapping["mapping_evidence"],
        "inventory_scope": record["inventory_scope"],
        "is_eol_procedure": "true",
        "source_sha256": expected_hash,
        "step4_segment_ref": reference,
        "step4_status": "verified",
        "step8_status": "not_mapped",
        "step10_segment_ref": "",
        "step10_status": "not_mapped",
        "bol_source_file": record["bol_source_file"],
        "bol_step4_segment_ref": bol_reference,
        "bol_step4_block_sha256": record["bol_step4_block_sha256"],
        "bol_source_sha256": record["bol_source_sha256"],
        "demo_step10_review_reason": record["step10_review_reason"],
    }
    return mapping, [observation]


def _owner_confirmed_g10_rpt10_mapping(
    filename: str, raw_mapping: dict[str, str]
) -> tuple[dict[str, str], list[dict[str, str]]]:
    expected_relpath = "group_10/group 10/tpg10.19-cell2.csv"
    if (
        Path(filename).suffix.casefold() != ".csv"
        or Path(filename).name.casefold() != "tpg10.19-cell2.csv"
        or raw_mapping.get("dataset_part") != "3"
        or raw_mapping.get("group_id") != "10"
        or raw_mapping.get("source_present", "").casefold() != "true"
        or _norm(raw_mapping.get("source_relative_path", "")) != expected_relpath
        or _norm(raw_mapping.get("repository_path", ""))
        != _norm(G10_RPT10_SOURCE_FILE)
    ):
        raise TpgIngestionError(
            "The owner-confirmed mapping exception is restricted to the exact "
            "Part 3 Group 10 TPG10.19-Cell2.csv source."
        )

    identity = RPT10_ZERO_CURRENT_IDENTITY
    observations = [
        row
        for row in _manifest_rows(
            FROZEN_OBSERVATION_MANIFEST, "frozen Final2 observation manifest"
        )
        if (
            row.get("dataset_part", ""),
            row.get("group", ""),
            row.get("physical_cell_id", ""),
            row.get("rpt_index", ""),
        )
        == identity
    ]
    extensions = [
        row
        for row in _manifest_rows(
            G10_P5_EXTENSION_PATH, "frozen G10 P5 label extension"
        )
        if (
            row.get("dataset_part", ""),
            row.get("group", ""),
            row.get("physical_cell_id", ""),
            row.get("rpt_index", ""),
        )
        == identity
    ]
    audits = [
        row
        for row in _manifest_rows(
            PART3_ELIGIBILITY_PATH, "Part 3 modality eligibility audit"
        )
        if (
            row.get("dataset_part", ""),
            row.get("group", ""),
            row.get("physical_cell_id", ""),
            row.get("rpt_index", ""),
        )
        == identity
    ]
    if len(observations) != 1 or len(extensions) != 1 or len(audits) != 1:
        raise TpgIngestionError(
            "The exact owner-confirmed RPT10 observation, extension, and modality "
            "audit rows must each be unique."
        )
    observation, extension, audit = observations[0], extensions[0], audits[0]

    segment_matches = True
    try:
        source4, start4, end4 = _segment_signature(
            observation["step4_segment_reference"], 4
        )
        ext_source4, ext_start4, ext_end4 = _segment_signature(
            extension["step4_segment_reference"], 4
        )
        audit_source4, audit_start4, audit_end4 = _segment_signature(
            audit["step4_segment_reference"], 4
        )
        source10, start10, end10 = _segment_signature(
            observation["step10_segment_reference"], 10
        )
        ext_source10, ext_start10, ext_end10 = _segment_signature(
            extension["step10_segment_reference"], 10
        )
        audit_source10, audit_start10, audit_end10 = _segment_signature(
            audit["step10_segment_reference"], 10
        )
        segment_matches = (
            (source4, start4, end4)
            == (ext_source4, ext_start4, ext_end4)
            == (audit_source4, audit_start4, audit_end4)
            and (source10, start10, end10)
            == (ext_source10, ext_start10, ext_end10)
            == (audit_source10, audit_start10, audit_end10)
        )
    except (KeyError, TpgIngestionError):
        segment_matches = False

    try:
        zero_current_rows = json.loads(extension["step10_zero_current_rows"])
    except (KeyError, TypeError, json.JSONDecodeError):
        zero_current_rows = None
    try:
        expected_zero_row = (
            isinstance(zero_current_rows, list)
            and len(zero_current_rows) == 1
            and zero_current_rows[0].get("line") == 2880746
            and float(zero_current_rows[0].get("amps", float("nan"))) == 0.0
        )
    except (AttributeError, TypeError, ValueError):
        expected_zero_row = False
    expected_fields = (
        observation.get("source_file") == G10_RPT10_SOURCE_FILE
        and extension.get("source_file") == G10_RPT10_SOURCE_FILE
        and observation.get("source_tpg") == "TPG10.19"
        and extension.get("source_tpg") == "TPG10.19"
        and observation.get("guide_stage") == "RPT10"
        and observation.get("mapping_status") == "GUIDE_VERIFIED"
        and extension.get("mapping_status") == "GUIDE_VERIFIED"
        and observation.get("identity_provenance") == G10_OWNER_PROVENANCE
        and extension.get("identity_provenance") == G10_OWNER_PROVENANCE
        and observation.get("status") == "ACCEPTED"
        and observation.get("final2_partition") == "external_test"
        and observation.get("step4_valid", "").casefold() == "true"
        and observation.get("step10_valid", "").casefold() == "true"
        and extension.get("step4_eligible", "").casefold() == "true"
        and extension.get("step10_eligible", "").casefold() == "true"
        and extension.get("external_inference_authorized", "").casefold() == "true"
        and observation.get("step4_block_sha256")
        == extension.get("step4_block_sha256")
        == audit.get("step4_block_sha256")
        and observation.get("step10_block_sha256")
        == extension.get("step10_block_sha256")
        == audit.get("step10_block_sha256")
        and observation.get("step4_source_sha256_snapshot")
        == extension.get("source_file_sha256")
        == G10_RPT10_SOURCE_SHA256
        and observation.get("step10_source_sha256_snapshot")
        == extension.get("source_file_sha256")
        == G10_RPT10_SOURCE_SHA256
        and audit.get("source_file") == G10_RPT10_SOURCE_FILE
        and audit.get("step4_inference_eligible", "").casefold() == "true"
        and audit.get("step10_inference_eligible", "").casefold() == "true"
        and audit.get("step10_zero_current_policy") == "exclude_zero_current"
        and audit.get("step10_zero_current_excluded_count") == "1"
        and audit.get("step10_zero_current_source_line") == "2880746"
        and extension.get("step10_zero_current_policy") == "exclude_zero_current"
        and extension.get("step10_zero_current_excluded_count") == "1"
        and expected_zero_row
        and segment_matches
    )
    if not expected_fields:
        raise TpgIngestionError(
            "The owner-confirmed RPT10 mapping failed frozen manifest, segment, "
            "source-hash, or zero-current audit cross-checks."
        )

    source_path = ROOT / Path(G10_RPT10_SOURCE_FILE)
    try:
        if (
            not source_path.is_file()
            or source_path.stat().st_size
            != int(raw_mapping.get("actual_size_bytes", "-1"))
            or _sha256(source_path).casefold() != G10_RPT10_SOURCE_SHA256
        ):
            raise TpgIngestionError(
                "The owner-confirmed TPG10.19 source path or SHA-256 does not match "
                "the frozen Final2 evidence."
            )
    except OSError as error:
        raise TpgIngestionError(
            f"Could not verify the owner-confirmed TPG10.19 source: {error}"
        ) from error

    resolved_mapping = {
        **raw_mapping,
        "dataset_part": "3",
        "group_id": "10",
        "physical_cell_id": "G10-Cell2",
        "procedure": "TPG10.19",
        "repository_path": G10_RPT10_SOURCE_FILE,
        "source_sha256": G10_RPT10_SOURCE_SHA256,
        "source_present": "true",
        "guide_stage": "RPT10",
        "guide_rpt_refs": "RPT10",
        "mapping_resolution_status": G10_OWNER_MAPPING_STATUS,
        "mapping_resolution_provenance": G10_OWNER_PROVENANCE,
        "inventory_scope": "FROZEN_FINAL2",
        "is_eol_procedure": "false",
        "mapping_evidence": (
            "frozen Final2 observation manifest, G10 P5 label extension, "
            "and exact Part 3 modality-eligibility row"
        ),
    }
    mapped_observation = {
        **observation,
        "part": observation["dataset_part"],
        "step4_status": "verified",
        "step10_status": "verified",
        "step4_segment_ref": extension["step4_segment_reference"],
        "step10_segment_ref": extension["step10_segment_reference"],
        "mapping_resolution_status": G10_OWNER_MAPPING_STATUS,
        "mapping_resolution_provenance": G10_OWNER_PROVENANCE,
    }
    return resolved_mapping, [mapped_observation]


def guide_mapping_for_file(
    filename: str,
    *,
    require_source_sha256: bool = False,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    extension = Path(filename).suffix.casefold()
    if extension not in {".mat", ".csv"}:
        raise TpgIngestionError("Upload a raw TPG .mat or .csv file.")
    if not RAW_MAPPING_PATH.is_file() or not RPT_MANIFEST_PATH.is_file():
        raise FileNotFoundError("Guide-verified source manifests are missing.")
    if extension == ".csv":
        eol_mapping = _demo_eol_mapping_for_file(filename)
        if eol_mapping is not None:
            return eol_mapping
    raw = pd.read_csv(RAW_MAPPING_PATH, dtype=str, keep_default_na=False)
    candidates = raw[
        raw["record_type"].eq("raw_file")
        & raw["source_extension"].str.casefold().eq(extension)
        & raw["source_present"].str.casefold().eq("true")
        & raw["source_relative_path"].map(_basename).eq(Path(filename).name.casefold())
    ]
    if len(candidates) != 1:
        raise TpgIngestionError(
            "This filename does not resolve to exactly one available raw TPG source in the Guide-verified inventory."
        )
    mapping = candidates.iloc[0].to_dict()
    guide_status = mapping["guide_source_mapping_status"].casefold()
    if "procedure_and_cell_match" not in guide_status:
        if (
            Path(filename).suffix.casefold() == ".csv"
            and Path(filename).name.casefold() == "tpg10.19-cell2.csv"
        ):
            return _owner_confirmed_g10_rpt10_mapping(filename, mapping)
        raise TpgIngestionError(
            "This file has an ambiguous Guide-to-cell mapping; no RPTs were inferred."
        )
    rpt_refs = RPT_NUMBER.findall(mapping["guide_rpt_refs"])
    guide_stage = mapping["guide_stage"].strip()
    if not rpt_refs and not RPT_NUMBER.search(guide_stage):
        raise TpgIngestionError(
            f"The Guide maps {Path(filename).name} as {guide_stage or 'a non-RPT procedure'}, not an RPT checkpoint."
        )

    observations = pd.read_csv(RPT_MANIFEST_PATH, dtype=str, keep_default_na=False)
    expected_csv_name = (
        Path(filename).name
        if extension == ".csv"
        else Path(filename).with_suffix(".csv").name
    ).casefold()
    matched = observations[
        observations["part"].eq(mapping["dataset_part"])
        & observations["group"].eq(mapping["group_id"])
        & observations["physical_cell_id"].eq(mapping["physical_cell_id"])
        & observations["source_file"].map(_basename).eq(expected_csv_name)
    ].copy()
    if rpt_refs:
        ref_set = {int(value) for value in rpt_refs}
        matched = matched[pd.to_numeric(matched["rpt_index"], errors="coerce").isin(ref_set)]
    elif guide_stage:
        stage_numbers = {int(value) for value in RPT_NUMBER.findall(guide_stage)}
        if stage_numbers:
            matched = matched[
                pd.to_numeric(matched["rpt_index"], errors="coerce").isin(stage_numbers)
            ]
    if matched.empty:
        raise TpgIngestionError(
            "No verified RPT boundaries match this Guide-mapped source file."
        )
    if matched.duplicated(["part", "physical_cell_id", "rpt_index"]).any():
        raise TpgIngestionError(
            "The Guide-linked observation manifest contains duplicate cell/RPT keys for this procedure."
        )
    matched = matched.sort_values(
        "chronological_position" if "chronological_position" in matched else "rpt_index",
        key=lambda values: pd.to_numeric(values, errors="coerce"),
        kind="stable",
    )
    mapped_observations = matched.to_dict(orient="records")
    expected_hash = str(mapping.get("source_sha256", "")).strip()
    if expected_hash:
        if not _SHA256_RE.fullmatch(expected_hash):
            raise TpgIngestionError(
                "The Guide source mapping contains a malformed SHA-256 checksum."
            )
        mapping["source_sha256_provenance"] = "unified_raw_mapping"
    elif extension == ".csv" and require_source_sha256:
        mapping["source_sha256"] = _frozen_source_sha256(
            mapping, mapped_observations
        )
        mapping["source_sha256_provenance"] = (
            "verified_frozen_final2_observation_manifest"
        )
    else:
        mapping["source_sha256_provenance"] = (
            "mat_source_verified_by_converted_csv_equivalence"
        )
    mapping["source_mapped_rpt_count"] = str(len(mapped_observations))
    return mapping, mapped_observations


def guide_mapping_for_mat(
    filename: str,
) -> tuple[dict[str, str], list[dict[str, str]]]:
    if Path(filename).suffix.casefold() != ".mat":
        raise TpgIngestionError("Upload a TPG .mat file.")
    return guide_mapping_for_file(filename)


def _matlab_executable() -> str:
    executable = shutil.which("matlab")
    if executable is None:
        raise MatlabRuntimeUnavailable(
            "MATLAB is required to decode TPG MAT files. Install MATLAB and add matlab.exe to PATH, then retry."
        )
    return executable


def convert_mat_with_matlab(input_path: Path, output_csv: Path) -> None:
    executable = _matlab_executable()
    if not MATLAB_CONVERTER_SCRIPT.is_file():
        raise FileNotFoundError(f"MATLAB converter script is missing: {MATLAB_CONVERTER_SCRIPT}")
    environment = os.environ.copy()
    environment["BAFUSE_MAT_INPUT"] = str(input_path.resolve())
    environment["BAFUSE_CSV_OUTPUT"] = str(output_csv.resolve())
    try:
        process = subprocess.run(
            [executable, "-batch", "bafuse_export_tpg_mat"],
            cwd=MATLAB_CONVERTER_DIR,
            env=environment,
            capture_output=True,
            text=True,
            timeout=1200,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise TpgIngestionError(
            "MATLAB conversion timed out after 20 minutes. Try a smaller supported TPG MAT file."
        ) from error
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip().splitlines()
        if detail:
            print("[MATLAB converter]", "\n".join(detail[-20:]), file=sys.stderr)
        raise TpgIngestionError(
            "MATLAB could not decode this TPG file as one supported measurement table."
        )
    if not output_csv.is_file() or output_csv.stat().st_size == 0:
        raise TpgIngestionError("MATLAB conversion produced no measurement-table output.")


def validate_mat_csv_equivalence(
    converted_csv: Path,
    reference_csv: Path,
    *,
    chunksize: int = 100_000,
) -> int:
    if not reference_csv.is_file():
        raise TpgIngestionError(
            "The Guide-linked reference CSV is unavailable; refusing to trust MAT row boundaries."
        )
    try:
        converted_header = pd.read_csv(converted_csv, nrows=0).columns.tolist()
        reference_header = pd.read_csv(reference_csv, nrows=0).columns.tolist()
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise TpgIngestionError(f"Could not inspect converted/reference CSV headers: {error}") from error
    def canonical_columns(columns: list[str], label: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for column in columns:
            canonical = str(column).lstrip("\ufeff").strip().casefold()
            if canonical in result:
                raise TpgIngestionError(f"{label} has duplicate column name: {column}.")
            result[canonical] = column
        return result

    converted_columns = canonical_columns(converted_header, "Converted/uploaded CSV")
    reference_columns = canonical_columns(reference_header, "Guide-linked CSV")
    required = tuple(name.casefold() for name in REQUIRED_MEASUREMENT_COLUMNS)
    if not set(required).issubset(converted_columns):
        raise TpgIngestionError(
            "Uploaded file is missing required raw measurement columns: Step, Amphr, Amps, Volts."
        )
    if not set(required).issubset(reference_columns):
        raise TpgIngestionError(
            "Guide-linked reference CSV is missing required raw measurement columns."
        )
    compare_pairs = [
        (converted_columns[name], reference_columns[name])
        for name in required
    ]
    for optional in ("temp1", "temp", "steptime"):
        if optional in converted_columns and optional in reference_columns:
            compare_pairs.append(
                (converted_columns[optional], reference_columns[optional])
            )

    row_count = 0
    with (
        pd.read_csv(
            converted_csv,
            usecols=[left for left, _ in compare_pairs],
            chunksize=chunksize,
        ) as csv_reader,
        pd.read_csv(
            reference_csv,
            usecols=[right for _, right in compare_pairs],
            chunksize=chunksize,
        ) as ref_reader,
    ):
        while True:
            try:
                converted_chunk = next(csv_reader)
            except StopIteration:
                converted_chunk = None
            try:
                reference_chunk = next(ref_reader)
            except StopIteration:
                reference_chunk = None
            if converted_chunk is None or reference_chunk is None:
                if converted_chunk is not reference_chunk:
                    raise TpgIngestionError(
                        "Converted MAT row count differs from the Guide-linked CSV; refusing RPT mapping."
                    )
                break
            if len(converted_chunk) != len(reference_chunk):
                raise TpgIngestionError(
                    "Converted MAT row count differs from the Guide-linked CSV; refusing RPT mapping."
                )
            for uploaded_column, reference_column in compare_pairs:
                converted_values = pd.to_numeric(
                    converted_chunk[uploaded_column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                reference_values = pd.to_numeric(
                    reference_chunk[reference_column], errors="coerce"
                ).to_numpy(dtype=np.float64)
                if not np.allclose(
                    converted_values,
                    reference_values,
                    rtol=0.0,
                    atol=MEASUREMENT_COMPARE_TOLERANCE,
                    equal_nan=True,
                ):
                    raise TpgIngestionError(
                        "Uploaded values differ from the Guide-linked CSV in "
                        f"{uploaded_column}; refusing RPT mapping."
                    )
            row_count += len(converted_chunk)
    if row_count == 0:
        raise TpgIngestionError("Converted TPG measurement table is empty.")
    return row_count


def _validate_source_equivalence(uploaded_csv: Path, reference_csv: Path) -> int:
    try:
        return validate_mat_csv_equivalence(uploaded_csv, reference_csv)
    except TpgIngestionError:
        raise
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
    ) as error:
        raise TpgIngestionError(
            f"Could not parse uploaded/reference TPG measurement CSV: {error}"
        ) from error


def _count_verified_csv_rows(source_csv: Path, *, chunksize: int = 100_000) -> int:
    row_count = 0
    try:
        with pd.read_csv(
            source_csv,
            usecols=["Step"],
            chunksize=chunksize,
        ) as csv_reader:
            for chunk in csv_reader:
                row_count += len(chunk)
    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        ValueError,
    ) as error:
        raise TpgIngestionError(
            f"Could not count rows in the SHA-256-verified source CSV: {error}"
        ) from error
    if row_count == 0:
        raise TpgIngestionError("The SHA-256-verified source CSV contains no data rows.")
    return row_count


def _approved_rpt10_zero_current_evidence(
    request: dict[str, Any],
    source_csv: Path,
    eligibility_path: Path = PART3_ELIGIBILITY_PATH,
) -> dict[str, Any] | None:
    row = request.get("row", {})
    identity = tuple(
        str(row.get(name, "")).strip()
        for name in ("part", "group", "physical_cell_id", "rpt_index")
    )
    if (
        request.get("step") != 10
        or identity != RPT10_ZERO_CURRENT_IDENTITY
        or not source_csv.is_file()
    ):
        return None
    try:
        with eligibility_path.open(encoding="utf-8-sig", newline="") as stream:
            matches = [
                record
                for record in csv.DictReader(stream)
                if (
                    record.get("dataset_part", "").strip(),
                    record.get("group", "").strip(),
                    record.get("physical_cell_id", "").strip(),
                    record.get("rpt_index", "").strip(),
                )
                == RPT10_ZERO_CURRENT_IDENTITY
            ]
    except (OSError, csv.Error, UnicodeDecodeError):
        return None
    if len(matches) != 1:
        return None
    record = matches[0]
    reference = record.get("step10_segment_reference", "")
    bounds = re.search(r":rows(\d+)-(\d+)$", reference)
    if not (
        record.get("source_tpg") == "TPG10.19"
        and str(record.get("source_file", "")).replace("\\", "/").casefold()
        == "data/raw/oxford/part3/group_10/group 10/tpg10.19-cell2.csv"
        and Path(record.get("source_file", "")).name.casefold()
        == "tpg10.19-cell2.csv"
        and record.get("step10_inference_eligible", "").casefold() == "true"
        and record.get("step10_zero_current_policy") == "exclude_zero_current"
        and record.get("step10_zero_current_excluded_count") == "1"
        and record.get("step10_zero_current_source_line") == "2880746"
        and str(row.get("mapping_status", "")).casefold() == "guide_verified"
        and str(row.get("step10_status", "")).casefold() == "verified"
        and bounds is not None
        and int(bounds.group(1)) == int(request["start"])
        and int(bounds.group(2)) == int(request["end"])
        and str(row.get("source_file", "")).replace("\\", "/").casefold()
        == str(record.get("source_file", "")).replace("\\", "/").casefold()
    ):
        return None
    return {
        "source_line": int(record["step10_zero_current_source_line"]),
        "start": int(bounds.group(1)),
        "end": int(bounds.group(2)),
    }


def _scan_approved_rpt10_step10(
    root: Path,
    request: dict[str, Any],
    evidence: dict[str, int],
) -> None:
    from bafuse.hybrid.soh_fusion import resample_curve

    phase2_dir = root / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
    if str(phase2_dir) not in sys.path:
        sys.path.insert(0, str(phase2_dir))
    from run_phase2 import _mark_segment_invalid, _parse_value

    source_path = Path(request["source"])
    if not source_path.is_absolute():
        source_path = root / source_path
    observed = 0
    zero_lines: list[int] = []
    invalid_current_direction = False
    voltages = array("f")
    currents = array("f")
    step_ids: set[int | None] = set()
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                _mark_segment_invalid(request, "source_has_no_header")
                return
            indices: dict[str, int] = {}
            for index, name in enumerate(header):
                key = name.strip().casefold()
                if key in indices:
                    _mark_segment_invalid(request, f"source_duplicate_column:{key}")
                    return
                indices[key] = index
            missing = {"step", "volts", "amps"} - set(indices)
            if missing:
                _mark_segment_invalid(
                    request, f"source_missing_columns:{','.join(sorted(missing))}"
                )
                return
            for values in reader:
                line_number = reader.line_num
                if line_number < evidence["start"]:
                    continue
                if line_number > evidence["end"]:
                    break
                observed += 1
                step = _parse_value(values, indices["step"])
                step_id = (
                    int(step)
                    if step is not None and step.is_integer()
                    else None
                )
                step_ids.add(step_id)
                voltage = _parse_value(values, indices["volts"])
                current = _parse_value(values, indices["amps"])
                if voltage is None:
                    request["missing_voltage"] += 1
                if current is None:
                    request["missing_current"] += 1
                if current == 0:
                    zero_lines.append(line_number)
                    if (
                        line_number == evidence["source_line"]
                        and step_id == 10
                        and voltage is not None
                    ):
                        continue
                elif current is not None and current < 0:
                    invalid_current_direction = True
                if step_id != 10 or voltage is None or current is None or current <= 0:
                    continue
                voltages.append(voltage)
                currents.append(current)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        _mark_segment_invalid(request, f"approved_rpt10_source_read_failed:{error}")
        return

    request["observed_rows"] = observed
    request["step_ids"] = step_ids
    expected_zero_policy = zero_lines == [evidence["source_line"]]
    if observed != request["expected_rows"]:
        _mark_segment_invalid(request, "segment_row_count_mismatch")
    elif step_ids != {10}:
        _mark_segment_invalid(request, "segment_step_id_mismatch")
    elif request["missing_voltage"] or request["missing_current"]:
        _mark_segment_invalid(request, "missing_or_nonfinite_voltage_or_current")
    elif not expected_zero_policy:
        _mark_segment_invalid(request, "approved_zero_current_row_mismatch")
    elif invalid_current_direction:
        _mark_segment_invalid(request, "unexpected_current_direction")
    elif len(voltages) < 2:
        _mark_segment_invalid(request, "segment_too_short_to_resample")
    else:
        request["row"]["step10_result"] = {
            "valid": True,
            "status": "verified_raw_segment_zero_current_excluded_by_approved_policy",
            "curve": resample_curve(voltages),
            "voltage_raw": voltages,
            "current_raw": currents,
            "temperature_raw": array("f"),
            "temperature_complete": False,
            "duration_native": None,
            "row_start": request["start"],
            "row_end": request["end"],
            "observed_rows": observed,
            "zero_current_excluded_count": 1,
            "zero_current_source_line": evidence["source_line"],
        }


def _scan_verified_source_requests(
    root: Path,
    requests: list[dict[str, Any]],
    source_csv: Path,
    eligibility_path: Path = PART3_ELIGIBILITY_PATH,
) -> None:
    phase2_dir = root / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
    if str(phase2_dir) not in sys.path:
        sys.path.insert(0, str(phase2_dir))
    from run_phase2 import _scan_source_requests

    approved = [
        (request, _approved_rpt10_zero_current_evidence(
            request, source_csv, eligibility_path
        ))
        for request in requests
    ]
    approved = [(request, evidence) for request, evidence in approved if evidence]
    if len(approved) != 1:
        _scan_source_requests(root, requests)
        return
    special, evidence = approved[0]
    if any(
        request is not special
        and request["start"] <= special["end"]
        and special["start"] <= request["end"]
        for request in requests
    ):
        _scan_source_requests(root, requests)
        return
    regular = [request for request in requests if request is not special]
    if regular:
        _scan_source_requests(root, regular)
    _scan_approved_rpt10_step10(root, special, evidence)


def _verified_feature_rows(
    source_csv: Path,
    observations: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, build_physics_features
    from bafuse.hybrid.p5_electrical_features import (
        P5_FEATURE_NAMES,
        build_p5_electrical_features,
    )

    phase2_dir = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
    if str(phase2_dir) not in sys.path:
        sys.path.insert(0, str(phase2_dir))
    from run_phase2 import _scan_source_requests, _segment_requests

    prepared = []
    for source in observations:
        row = dict(source)
        row["soh_percent"] = None
        row["step4_result"] = None
        row["step8_result"] = None
        row["step10_result"] = None
        row["physics_features"] = None
        prepared.append(row)
    requests_by_source = _segment_requests(
        prepared, include_unlabeled=True, steps=(4, 8, 10)
    )
    for requests in requests_by_source.values():
        for request in requests:
            request["source"] = str(source_csv.resolve())
        _scan_verified_source_requests(ROOT, requests, source_csv)

    model_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    for row in prepared:
        step4 = row["step4_result"]
        step8 = row["step8_result"]
        step10 = row["step10_result"]
        step4_status = _segment_status(step4, row.get("step4_status"), 4)
        step8_status = _segment_status(step8, row.get("step8_status"), 8)
        step10_status = _segment_status(step10, row.get("step10_status"), 10)
        identity = {
            "part": str(row["part"]),
            "group": str(row["group"]),
            "physical_cell_id": row["physical_cell_id"],
            "rpt_index": str(row["rpt_index"]),
            "mapping_resolution_status": row.get(
                "mapping_resolution_status", "guide_inventory_mapping"
            ),
            "identity_provenance": row.get("mapping_resolution_provenance", ""),
            "inventory_scope": row.get("inventory_scope", "GUIDE_MAPPED"),
            "procedure": row.get("procedure", row.get("source_tpg", "")),
            "guide_stage": row.get("guide_stage", ""),
            "mapping_status": row.get(
                "mapping_status",
                row.get("mapping_resolution_status", "not_recorded"),
            ),
            "source_file": row.get("source_file", ""),
            "source_sha256": row.get("source_sha256", ""),
            "mapping_provenance": row.get(
                "mapping_resolution_provenance", ""
            ),
            "mapping_evidence": row.get("mapping_evidence", ""),
            "source_checksum_verified": row.get(
                "source_checksum_verified", "false"
            ),
            "is_eol_procedure": row.get("is_eol_procedure", "false"),
            "rpt": f"RPT{row['rpt_index']}",
            "chronological_position": row.get("chronological_position", ""),
            "step4_status": step4_status,
            "step8_status": step8_status,
            "step10_status": step10_status,
            "step8_mapping_status": row.get("step8_status", "not_mapped"),
            "step8_extraction_status": (
                step8.get("status", "unavailable") if step8 else "unavailable"
            ),
        }
        reason = ""
        if not step4 or not step4["valid"]:
            reason = step4_status["reason"]

        feature_values: dict[str, float] = {}
        physics_ready = bool(
            step4
            and step4["valid"]
            and step4["temperature_complete"]
        )
        if physics_ready:
            feature_values = build_physics_features(
                step4["voltage_raw"],
                step4["current_raw"],
                step4["temperature_raw"],
            )
        p5_feature_values: dict[str, float] = {}
        if step4 and step4["valid"]:
            p5_feature_values = build_p5_electrical_features(
                step4["voltage_raw"],
                step4["current_raw"],
            )
        identity.update(
            {
                "demo_step4_valid": bool(step4 and step4.get("valid")),
                "demo_step10_valid": bool(step10 and step10.get("valid")),
                "demo_p5_valid": bool(p5_feature_values),
                "modality_eligibility": {
                    "step4_voltage": bool(step4 and step4.get("valid")),
                    "step10_voltage": bool(step10 and step10.get("valid")),
                    "p5_physics": bool(p5_feature_values),
                },
                "demo_step10_review_reason": row.get(
                    "demo_step10_review_reason", ""
                ),
            }
        )
        p5_feature_status = {
            name: {
                "value": p5_feature_values.get(name),
                "status": (
                    "computed_from_verified_step4_electrical_measurements"
                    if name in p5_feature_values
                    else f"unavailable:{step4_status['reason']}"
                ),
            }
            for name in P5_FEATURE_NAMES
        }
        if step4 and step4["valid"]:
            model_row: dict[str, Any] = {
                **identity,
                "partition": "external_test" if row["part"] == "3" else "unassigned",
            }
            for index, value in enumerate(step4["curve"]):
                model_row[f"step4_voltage_progress_{index:03d}"] = value
            if step10 and step10["valid"]:
                for index, value in enumerate(step10["curve"]):
                    model_row[f"step10_voltage_progress_{index:03d}"] = value
            if feature_values:
                for name in PHYSICS_FEATURE_NAMES:
                    model_row[name] = feature_values[name]
            if p5_feature_values:
                model_row["p5_physics_features"] = [
                    p5_feature_values[name] for name in P5_FEATURE_NAMES
                ]
            model_rows.append(model_row)

        feature_status: dict[str, dict[str, Any]] = {}
        for name in PHYSICS_FEATURE_NAMES:
            if name in feature_values:
                feature_status[name] = {
                    "value": feature_values[name],
                    "status": "computed_from_verified_step4_measurements",
                }
            elif not step4 or not step4["valid"]:
                feature_status[name] = {
                    "value": None,
                    "status": f"not_computed:{step4_status['reason']}",
                }
            elif not step4["temperature_complete"]:
                feature_status[name] = {
                    "value": None,
                    "status": (
                        "missing_step4_temperature"
                        if name
                        in (
                            "mean_discharge_temperature_native",
                            "discharge_temperature_rise_native",
                        )
                        else "not_computed:approved_physics_feature_block_requires_complete_temperature"
                    ),
                }
            else:
                feature_status[name] = {
                    "value": None,
                    "status": "not_computed:approved_physics_feature_inputs_unavailable",
                }

        checkpoint_rows.append(
            {
                **identity,
                "processing_status": (
                    reason
                    or (
                        "features_ready"
                        if step10 and step10["valid"]
                        else step10_status["reason"]
                    )
                ),
                "inputs": {
                    "step4": _segment_visualization(step4, step4_status, 4),
                    "step8": _segment_visualization(step8, step8_status, 8),
                    "step10": _segment_visualization(step10, step10_status, 10),
                    "physics_features": feature_status,
                    "p5_physics_features": p5_feature_status,
                },
            }
        )
    return model_rows, checkpoint_rows


def _segment_status(
    result: dict[str, Any] | None,
    mapping_status: str | None,
    step: int,
) -> dict[str, Any]:
    if result and result.get("valid"):
        return {
            "available": True,
            "mapping_status": mapping_status or "verified",
            "status": result.get("status", "verified_raw_segment"),
            "reason": "",
        }
    if result:
        reason = str(result.get("status", "invalid_segment"))
    elif mapping_status and mapping_status.casefold() not in {
        "verified",
        "verified_raw_segment",
    }:
        reason = f"missing_step{step}_mapping:{mapping_status}"
    else:
        reason = f"missing_step{step}_measurements"
    return {
        "available": False,
        "mapping_status": mapping_status or "not_mapped",
        "status": result.get("status", "unavailable") if result else "unavailable",
        "reason": reason,
    }


def _segment_visualization(
    result: dict[str, Any] | None,
    status: dict[str, Any],
    step: int,
) -> dict[str, Any]:
    visualization: dict[str, Any] = {
        **status,
        "step": step,
        "raw_voltage": [],
        "processed_128_voltage": [],
    }
    if not result or not result.get("valid"):
        if step == 4:
            missing_temperature = result.get("missing_temperature") if result else None
            temperature_available = bool(
                result
                and missing_temperature == 0
                and result.get("observed_rows") == result.get("expected_rows")
            )
            visualization["temperature_available"] = temperature_available
            visualization["temperature_status"] = (
                "temperature_present_but_step4_segment_invalid"
                if temperature_available
                else "missing_or_nonfinite_step4_temperature"
            )
            visualization["current_available"] = (
                result.get("missing_current") == 0
                if result and result.get("missing_current") is not None
                else False
            )
        return visualization
    visualization["raw_voltage"] = [
        float(value) for value in result["voltage_raw"]
    ]
    visualization["processed_128_voltage"] = [
        float(value) for value in result["curve"]
    ]
    visualization["row_start"] = result.get("row_start")
    visualization["row_end"] = result.get("row_end")
    visualization["observed_rows"] = result.get("observed_rows")
    if step == 4:
        visualization["temperature_available"] = bool(
            result.get("temperature_complete")
        )
        visualization["temperature_status"] = (
            "verified_step4_temperature"
            if result.get("temperature_complete")
            else "missing_or_nonfinite_step4_temperature"
        )
        visualization["current_available"] = bool(result.get("current_raw"))
    return visualization


def _saved_degradation(
    mapping: dict[str, str], observations: list[dict[str, str]]
) -> dict[tuple[str, str], dict[str, Any]]:
    if not H2_V3_PATH.is_file():
        return {}
    saved = pd.read_csv(H2_V3_PATH, low_memory=False)
    part = mapping["dataset_part"]
    group = mapping["group_id"]
    cell = mapping["physical_cell_id"]
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for observation in observations:
        rpt = str(observation["rpt_index"])
        rows = saved[
            saved["part"].astype(str).eq(part)
            & saved["group"].astype(str).eq(group)
            & saved["physical_cell_id"].eq(cell)
            & pd.to_numeric(saved["rpt_index"], errors="coerce").eq(int(rpt))
        ]
        for row in rows.to_dict(orient="records"):
            metric = row["metric"]
            accepted = str(row["h2_v3_selected_accepted"]).casefold() == "true"
            value = row.get("h2_v3_selected_percent")
            try:
                value = float(value)
                if not np.isfinite(value):
                    value = None
            except (TypeError, ValueError):
                value = None
            if not accepted:
                value = None
            key = (rpt, metric)
            if key in result:
                raise TpgIngestionError(f"Duplicate H2 v3 key for {cell} RPT{rpt} {metric}.")
            result[key] = {
                "value_percent": value,
                "status": (
                    "saved_h2_v3_accepted"
                    if accepted and value is not None
                    else row.get("h2_v3_selection_reason", "saved_fit_unavailable")
                ),
                "source_direction": row.get("h2_v3_source_direction", ""),
                "fit_rmse_mV": row.get("h2_v3_selected_fit_rmse_mV"),
                "fit_accepted_runs": row.get("h2_v3_selected_fit_accepted_runs"),
                "selection_policy": "H2 Hybrid direction v3",
            }
    return result


def _verified_upload_context(
    input_path: Path | None,
    original_filename: str,
    *,
    require_source_sha256: bool = False,
) -> tuple[dict[str, str], list[dict[str, str]], Path]:
    mapping, observations = guide_mapping_for_file(
        original_filename,
        require_source_sha256=require_source_sha256,
    )
    if mapping["source_present"].casefold() != "true":
        raise TpgIngestionError(
            "The Guide-mapped source is not present in the verified project inventory."
        )
    reference_path = Path(mapping["repository_path"])
    reference_csv = ROOT / (
        reference_path.with_suffix(".csv")
        if Path(original_filename).suffix.casefold() == ".mat"
        else reference_path
    )
    if not reference_csv.is_file():
        raise TpgIngestionError(
            "The Guide-linked reference CSV is unavailable; refusing to infer RPT row boundaries."
        )
    if input_path is None:
        return mapping, observations, reference_csv
    if not input_path.is_file() or input_path.stat().st_size == 0:
        raise TpgIngestionError("The uploaded TPG file is empty or unavailable.")
    extension = Path(original_filename).suffix.casefold()
    with input_path.open("rb") as stream:
        header_bytes = stream.read(128)
    if extension == ".mat":
        if b"MATLAB" not in header_bytes:
            raise TpgIngestionError(
                "The uploaded .mat file does not have a recognized MATLAB MAT-file header."
            )
    else:
        if b"\x00" in header_bytes:
            raise TpgIngestionError("The uploaded .csv file contains binary data, not a raw TPG table.")
        try:
            header_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise TpgIngestionError("Raw TPG CSV files must use UTF-8 encoding.") from error
    expected_hash = mapping["source_sha256"].strip()
    if (
        extension == ".csv"
        and require_source_sha256
        and not _SHA256_RE.fullmatch(expected_hash)
    ):
        raise TpgIngestionError(
            "The raw CSV has no valid SHA-256 from the verified Guide or frozen source manifest."
        )
    if expected_hash and _sha256(input_path).casefold() != expected_hash.casefold():
        raise TpgIngestionError(
            "Uploaded bytes do not match the saved checksum for the resolved source mapping."
        )
    mapping["source_checksum_verified"] = (
        "true" if expected_hash else "false"
    )
    return mapping, observations, reference_csv


def _inspection_result(
    mapping: dict[str, str], observations: list[dict[str, str]]
) -> dict[str, Any]:
    owner_confirmed = (
        mapping.get("mapping_resolution_status") == G10_OWNER_MAPPING_STATUS
    )
    mapping_status = (
        "owner_confirmed_frozen_final2_mapping; source_checksum_verified; "
        "not_independently_guide_mapped"
        if owner_confirmed
        else (
            "guide_mapped; source_checksum_verified"
            if mapping["source_sha256"].strip()
            else "guide_mapped; source_checksum_not_in_inventory"
        )
    )
    return {
        "cell_id": mapping["physical_cell_id"],
        "group_id": mapping["group_id"],
        "part": mapping["dataset_part"],
        "detected_rpt_count": len(observations),
        "rpt_checkpoints": [f"RPT{row['rpt_index']}" for row in observations],
        "mapping_status": mapping_status,
        "mapping_resolution_status": mapping.get(
            "mapping_resolution_status", "guide_inventory_mapping"
        ),
        "mapping_provenance": mapping.get(
            "mapping_resolution_provenance", ""
        ),
        "mapping_evidence": mapping.get("mapping_evidence", ""),
        "source_sha256": mapping.get("source_sha256", ""),
        "procedure": mapping.get("procedure", ""),
        "guide_stage": mapping.get("guide_stage", ""),
        "inventory_scope": mapping.get("inventory_scope", "GUIDE_MAPPED"),
        "is_eol_procedure": mapping.get("is_eol_procedure", "false").casefold()
        == "true",
        "step10_eligible": mapping.get("step10_eligible", ""),
        "step10_review_reason": mapping.get("step10_review_reason", ""),
    }


def inspect_tpg_file(original_filename: str) -> dict[str, Any]:
    mapping, observations = guide_mapping_for_file(original_filename)
    return _inspection_result(mapping, observations)


def inspect_tpg_mat(input_path: Path, original_filename: str) -> dict[str, Any]:
    mapping, observations, _ = _verified_upload_context(input_path, original_filename)
    return _inspection_result(mapping, observations)


def _select_rpt_observations(
    observations: list[dict[str, str]],
    selected_rpt_index: str | None,
) -> list[dict[str, str]]:
    if selected_rpt_index is None:
        if len(observations) != 1:
            raise TpgIngestionError(
                f"This source maps to {len(observations)} RPTs; an explicit RPT selection is required."
            )
        return observations

    selected = str(selected_rpt_index).strip()
    selected_match = re.fullmatch(r"(?:RPT\s*)?(\d+)", selected, re.IGNORECASE)
    if not selected_match:
        raise TpgIngestionError(f"Invalid selected RPT identifier: {selected_rpt_index!r}.")
    selected_number = str(int(selected_match.group(1)))
    matches = [
        observation
        for observation in observations
        if str(observation.get("rpt_index", "")).strip() == selected_number
    ]
    if len(matches) != 1:
        raise TpgIngestionError(
            f"Selected RPT{selected_number} is not a unique checkpoint in the verified Guide mapping."
        )
    return matches


def analyze_tpg_file(
    input_path: Path,
    original_filename: str,
    engine: Any,
    model_name: str | None = None,
    selected_rpt_index: str | None = None,
    require_source_sha256: bool = False,
) -> dict[str, Any]:
    mapping, observations, reference_csv = _verified_upload_context(
        input_path,
        original_filename,
        require_source_sha256=require_source_sha256,
    )
    mapping.setdefault("source_mapped_rpt_count", str(len(observations)))
    if selected_rpt_index is not None or require_source_sha256:
        observations = _select_rpt_observations(
            observations, selected_rpt_index
        )
    if Path(original_filename).suffix.casefold() == ".mat":
        with tempfile.TemporaryDirectory(prefix="bafuse-tpg-conversion-") as directory:
            source_csv = Path(directory) / "measurements.csv"
            convert_mat_with_matlab(input_path, source_csv)
            validated_rows = _validate_source_equivalence(source_csv, reference_csv)
            return _analyze_verified_csv(
                source_csv,
                original_filename,
                mapping,
                observations,
                validated_rows,
                engine,
                model_name,
            )
    if Path(original_filename).suffix.casefold() != ".csv":
        raise TpgIngestionError("Upload a raw TPG .mat or .csv file.")
    if require_source_sha256:
        if mapping.get("source_checksum_verified") != "true":
            raise TpgIngestionError(
                "The raw CSV source checksum was not verified against a trusted manifest."
            )
        validated_rows = _count_verified_csv_rows(input_path)
        mapping["source_csv_equivalence"] = (
            "uploaded_sha256_matches_trusted_source_manifest; "
            "row_count_read_from_Step_column"
        )
    else:
        validated_rows = _validate_source_equivalence(input_path, reference_csv)
        mapping["source_csv_equivalence"] = (
            "validated_raw_signals_and_row_count"
        )
    return _analyze_verified_csv(
        input_path,
        original_filename,
        mapping,
        observations,
        validated_rows,
        engine,
        model_name,
    )


def _analyze_verified_csv(
    source_csv: Path,
    original_filename: str,
    mapping: dict[str, str],
    observations: list[dict[str, str]],
    validated_rows: int,
    engine: Any,
    model_name: str | None = None,
) -> dict[str, Any]:
    for observation in observations:
        observation.setdefault(
            "inventory_scope", mapping.get("inventory_scope", "GUIDE_MAPPED")
        )
        observation.setdefault("procedure", mapping.get("procedure", ""))
        observation.setdefault("source_tpg", mapping.get("source_tpg", ""))
        observation.setdefault("guide_stage", mapping.get("guide_stage", ""))
        observation.setdefault(
            "mapping_status",
            observation.get(
                "mapping_resolution_status",
                mapping.get("mapping_resolution_status", "not_recorded"),
            ),
        )
        observation.setdefault(
            "source_checksum_verified",
            mapping.get("source_checksum_verified", "false"),
        )
        observation.setdefault("source_sha256", mapping.get("source_sha256", ""))
        observation.setdefault(
            "is_eol_procedure", mapping.get("is_eol_procedure", "false")
        )
    try:
        model_rows, processing_rows = _verified_feature_rows(source_csv, observations)
    except TpgIngestionError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error, ValueError) as error:
        raise TpgIngestionError(
            f"Could not parse/extract verified RPT measurements from the uploaded TPG file: {error}"
        ) from error
    selected_model = model_name or engine.final2_runtime.default_model()
    model_inference = (
        engine.predict_for_model(model_rows, selected_model)
        if model_rows
        else {"model_name": selected_model, "results": []}
    )
    prediction_by_key = {
        (
            row["part"],
            row["physical_cell_id"],
            row["rpt_index"],
        ): row
        for row in model_inference["results"]
    }
    normalized_by_key = {
        (
            row["part"],
            row["physical_cell_id"],
            row["rpt_index"],
        ): {
            **row.get("normalized_inputs", {}),
            "preprocessing": "frozen Final2 Train-only saved scalers",
        }
        for row in model_inference["results"]
        if row.get("normalized_inputs")
    }
    saved = _saved_degradation(mapping, observations)
    from app.model_runtime import load_shared_charge_degradation

    references = _reference_labels(
        str(mapping["dataset_part"]),
        mapping["physical_cell_id"],
        observations,
    )
    if mapping.get("inventory_scope") == "VERIFIED_DEMO_ONLY":
        references.update(_demo_eol_reference_labels(source_csv, observations))
    outputs = []
    for checkpoint in processing_rows:
        rpt_index = str(checkpoint["rpt_index"])
        identity_key = (
            str(checkpoint["part"]),
            checkpoint["physical_cell_id"],
            rpt_index,
        )
        prediction = prediction_by_key.get(identity_key)
        predicted_value = (
            prediction["predicted_soh_percent"] if prediction else None
        )
        reference = references.get(
            rpt_index,
            {
                "value_percent": None,
                "status": "missing_reference_soh_label_for_exact_part_cell_rpt",
                "label_scope": "",
                "source_file": "",
                "label_method": "",
                "capacity_unit": "",
            },
        )
        reference_value = reference["value_percent"]
        if predicted_value is not None and reference_value is not None:
            signed_error = float(predicted_value) - float(reference_value)
            absolute_error = abs(signed_error)
            error_status = "matched_finite_prediction_and_reference"
        elif predicted_value is None:
            signed_error = None
            absolute_error = None
            error_status = f"prediction_unavailable:{checkpoint['processing_status']}"
        else:
            signed_error = None
            absolute_error = None
            error_status = f"reference_unavailable:{reference['status']}"
        if identity_key in normalized_by_key:
            checkpoint["inputs"]["model_normalized"] = normalized_by_key[identity_key]
            p5_normalized = normalized_by_key[identity_key].get(
                "physics_features_normalized"
            )
            if p5_normalized is not None:
                from bafuse.hybrid.p5_electrical_features import P5_FEATURE_NAMES

                for index, name in enumerate(P5_FEATURE_NAMES):
                    checkpoint["inputs"]["p5_physics_features"][name][
                        "normalized_value"
                    ] = p5_normalized[index]
        degradation = {
            metric: saved.get((rpt_index, metric), {
                "value_percent": None,
                "status": "no_accepted_saved_h2_v3_fit",
                "source_direction": "",
                "fit_rmse_mV": None,
                "fit_accepted_runs": None,
                "selection_policy": "H2 Hybrid direction v3",
            })
            for metric in ("LLI", "Anode_LAM", "Cathode_LAM")
        }
        outputs.append(
            {
                **checkpoint,
                "predicted_soh_percent": predicted_value,
                "soh_status": (
                    prediction["soh_status"]
                    if prediction
                    else "prediction_unavailable:no_valid_step4_input_row"
                ),
                "model_prediction": prediction
                or {
                    "model_name": selected_model,
                    "predicted_soh_percent": None,
                    "soh_status": "prediction_unavailable",
                    "reason": checkpoint["processing_status"],
                },
                "reference_soh": reference,
                "signed_error_percentage_points": signed_error,
                "absolute_error_percentage_points": absolute_error,
                "error_status": error_status,
                "degradation": degradation,
                "charge_degradation": load_shared_charge_degradation(
                    mapping["dataset_part"],
                    mapping["group_id"],
                    mapping["physical_cell_id"],
                    rpt_index,
                ),
            }
        )
    processed_count = sum(row["predicted_soh_percent"] is not None for row in outputs)
    source_format = Path(original_filename).suffix.casefold().lstrip(".")
    metrics = calculate_soh_metrics(outputs)
    return {
        "cell_id": mapping["physical_cell_id"],
        "group_id": mapping["group_id"],
        "part": mapping["dataset_part"],
        "procedure": mapping["procedure"],
        "detected_rpt_count": int(
            mapping.get("source_mapped_rpt_count", len(observations))
        ),
        "selected_rpt_count": len(observations),
        "processed_rpt_count": processed_count,
        "successful_prediction_count": processed_count,
        "selected_model": selected_model,
        "model_validation_metrics": engine.final2_runtime.validation_metrics.get(
            selected_model
        ),
        "matched_evaluation_count": metrics["n"],
        "evaluation_metrics": metrics,
        "source_row_count_verified": validated_rows,
        "rpts": outputs,
        "provenance": {
            "source_format": source_format,
            "guide_stage": mapping["guide_stage"],
            "guide_rpt_refs": mapping["guide_rpt_refs"],
            "mapping_resolution_status": mapping.get(
                "mapping_resolution_status", "guide_inventory_mapping"
            ),
            "mapping_provenance": mapping.get(
                "mapping_resolution_provenance", ""
            ),
            "mapping_evidence": mapping.get("mapping_evidence", ""),
            "source_file_sha256": mapping.get("source_sha256", ""),
            "source_checksum_verified": (
                mapping.get("source_checksum_verified", "").casefold() == "true"
            ),
            "source_csv_equivalence": mapping.get(
                "source_csv_equivalence",
                "validated_raw_signals_and_row_count",
            ),
            "soh_model_version": selected_model,
            "reference_label_policy": (
                "provisional same-cell capacity-ratio labels; only "
                "calculated_from_measured_capacity rows are eligible"
            ),
            "part3_evaluation_context": (
                "external/domain-shift evaluation; Part 3 remains separate"
                if mapping["dataset_part"] == "3"
                else None
            ),
            "degradation_policy": (
                "saved H2 Hybrid direction v3 output is model-independent; "
                "accepted Charge PyDMA is separately joined by exact identity"
            ),
        },
    }


def analyze_tpg_mat(
    input_path: Path,
    original_filename: str,
    engine: Any,
) -> dict[str, Any]:
    if Path(original_filename).suffix.casefold() != ".mat":
        raise TpgIngestionError("Upload a TPG .mat file.")
    return analyze_tpg_file(input_path, original_filename, engine)


def _demo_eol_reference_labels(
    source_csv: Path,
    observations: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    references: dict[str, dict[str, Any]] = {}
    for row in observations:
        if row.get("inventory_scope") != "VERIFIED_DEMO_ONLY":
            continue
        if row.get("mapping_status") != "GUIDE_VERIFIED":
            raise TpgIngestionError(
                "A demo-only EoL reference requires a Guide-verified mapping."
            )
        if row.get("source_checksum_verified", "").casefold() != "true":
            raise TpgIngestionError(
                "A demo-only EoL reference requires a verified source checksum."
            )
        target = _integrate_step4_capacity(
            source_csv,
            row["step4_segment_ref"],
            expected_source_name=Path(row["source_file"]).name,
        )
        if row.get("bol_source_file") is None:
            raise TpgIngestionError("The demo-only EoL observation has no BoL source.")
        bol_path = ROOT / Path(row["bol_source_file"])
        bol = _integrate_step4_capacity(
            bol_path,
            row["bol_step4_segment_ref"],
            expected_source_name=Path(row["bol_source_file"]).name,
        )
        if bol["source_block_sha256"] != row["bol_step4_block_sha256"]:
            raise TpgIngestionError(
                "The same-cell BoL Step 4 block differs from the frozen P5 reference."
            )
        if _sha256(bol_path).casefold() != row["bol_source_sha256"].casefold():
            raise TpgIngestionError(
                "The same-cell BoL source differs from the frozen P5 source snapshot."
            )
        q_target = target["capacity_native_current_time"]
        q_bol = bol["capacity_native_current_time"]
        if not np.isfinite(q_target) or not np.isfinite(q_bol) or q_target <= 0 or q_bol <= 0:
            raise TpgIngestionError(
                "The demo-only target or same-cell BoL capacity integral is invalid."
            )
        rpt = str(row["rpt_index"])
        if rpt in references:
            raise TpgIngestionError(
                f"Duplicate demo-only reference for {row['physical_cell_id']} RPT{rpt}."
            )
        references[rpt] = {
            "value_percent": 100.0 * q_target / q_bol,
            "status": "demo_only_provisional_capacity_ratio_not_evaluation_eligible",
            "label_scope": "verified_demo_only",
            "source_file": row["source_file"],
            "label_method": (
                "100 * Step 4 trapezoidal integral of abs(Amps) against native "
                "TestTime / same-cell .0 BoL Step 4 integral"
            ),
            "capacity_unit": "native current*time; same-cell ratio",
            "source_block_sha256": target["source_block_sha256"],
            "bol_source_file": row["bol_source_file"],
            "bol_segment_reference": row["bol_step4_segment_ref"],
            "bol_source_sha256": row["bol_source_sha256"],
            "bol_block_sha256": bol["source_block_sha256"],
            "evaluation_eligible": False,
        }
    return references


def _reference_labels(
    part: str,
    physical_cell_id: str,
    observations: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    source_path = SOH_REFERENCE_PATHS.get(part)
    if source_path is None:
        return {}
    if not source_path.is_file():
        raise FileNotFoundError(f"Validated Part {part} SOH label table is missing: {source_path}")
    labels = pd.read_csv(source_path, dtype=str, keep_default_na=False)
    labels["dataset_part"] = labels["dataset_part"].astype(str)
    labels["rpt_order_numeric"] = pd.to_numeric(labels["rpt_order"], errors="coerce")
    selected = labels[
        labels["dataset_part"].eq(part)
        & labels["physical_cell_id"].eq(physical_cell_id)
    ]
    result: dict[str, dict[str, Any]] = {}
    for observation in observations:
        rpt = str(observation["rpt_index"])
        key_rows = selected[
            selected["rpt_order_numeric"].eq(pd.to_numeric(rpt, errors="coerce"))
        ]
        valid_rows = key_rows[
            key_rows["validation_status"].eq(VALID_REFERENCE_STATUS)
        ]
        if len(valid_rows) > 1:
            raise TpgIngestionError(
                f"Duplicate validated SOH labels for Part {part}, {physical_cell_id}, RPT{rpt}."
            )
        if len(valid_rows) == 1:
            label = valid_rows.iloc[0]
            try:
                value = float(label["soh_percent"])
            except (TypeError, ValueError):
                value = None
            if value is not None and np.isfinite(value):
                result[rpt] = {
                    "value_percent": value,
                    "status": VALID_REFERENCE_STATUS,
                    "label_scope": label.get("label_scope", label.get("label_confidence", "")),
                    "source_file": label.get("source_file", ""),
                    "label_method": label.get("label_method", ""),
                    "capacity_unit": label.get("capacity_unit", ""),
                }
                continue
            result[rpt] = {
                "value_percent": None,
                "status": "eligible_reference_label_has_nonfinite_soh_value",
                "label_scope": "",
                "source_file": label.get("source_file", ""),
                "label_method": label.get("label_method", ""),
                "capacity_unit": label.get("capacity_unit", ""),
            }
            continue
        if key_rows.empty:
            reason = "missing_reference_soh_label_for_exact_part_cell_rpt"
        else:
            statuses = sorted(
                {
                    str(value)
                    for value in key_rows["validation_status"].tolist()
                    if str(value).strip()
                }
            )
            reason = "reference_label_not_eligible:" + ",".join(statuses or ["unknown_status"])
        result[rpt] = {
            "value_percent": None,
            "status": reason,
            "label_scope": "",
            "source_file": "",
            "label_method": "",
            "capacity_unit": "",
        }
    return result


def calculate_soh_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = []
    for row in records:
        if (
            row.get("inventory_scope") == "VERIFIED_DEMO_ONLY"
            or row.get("reference_soh", {}).get("evaluation_eligible") is False
        ):
            continue
        prediction = row.get("predicted_soh_percent")
        reference = row.get("reference_soh", {}).get("value_percent")
        try:
            prediction_value = float(prediction)
            reference_value = float(reference)
        except (TypeError, ValueError):
            continue
        if np.isfinite(prediction_value) and np.isfinite(reference_value):
            pairs.append((prediction_value, reference_value))

    count = len(pairs)
    metrics: dict[str, Any] = {"n": count}
    if not count:
        for name in ("mae_percentage_points", "rmse_percentage_points", "r2"):
            metrics[name] = {
                "value": None,
                "reason": "No finite matched prediction/reference RPT pairs.",
            }
        return metrics

    errors = np.asarray(
        [prediction - reference for prediction, reference in pairs],
        dtype=np.float64,
    )
    metrics["mae_percentage_points"] = {
        "value": float(np.mean(np.abs(errors))),
        "reason": "",
    }
    metrics["rmse_percentage_points"] = {
        "value": float(np.sqrt(np.mean(np.square(errors)))),
        "reason": "",
    }
    references = np.asarray([reference for _, reference in pairs], dtype=np.float64)
    if count < 2:
        metrics["r2"] = {
            "value": None,
            "reason": "R2 requires at least two matched samples.",
        }
    elif float(np.sum(np.square(references - np.mean(references)))) == 0:
        metrics["r2"] = {
            "value": None,
            "reason": "R2 is undefined because reference SoH has zero variance.",
        }
    else:
        metrics["r2"] = {
            "value": float(
                1.0
                - np.sum(np.square(errors))
                / np.sum(np.square(references - np.mean(references)))
            ),
            "reason": "",
        }
    return metrics
