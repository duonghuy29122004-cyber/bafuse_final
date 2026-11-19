from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


CANONICAL_MANIFEST = Path("data/manifests/canonical_observation_manifest_v3.csv")
DATASET_REVIEW = Path("experiments/dataset_review/README.md")
MAPPING_AUDIT = Path("reports/final_mapping_label_audit/mapping_observation_audit.csv")
SOURCE_INTEGRITY_AUDIT = Path(
    "reports/final_mapping_label_audit/source_integrity_audit.csv"
)
MAPPING_AUDIT_SUMMARY = Path("reports/final_mapping_label_audit/summary.md")
FINAL2_PREPROCESSING = Path("src/bafuse/hybrid/final2_dataset.py")
FINAL2_ROOT = Path("experiments/final2/training_6_models")
FINAL2_OBSERVATIONS = FINAL2_ROOT / "frozen_dataset/observation_manifest.csv"
FINAL2_SUMMARY = FINAL2_ROOT / "frozen_dataset/dataset_summary.json"
FINAL2_SHA_MANIFEST = FINAL2_ROOT / "frozen_dataset/dataset_sha256_manifest.json"
FINAL2_BUILD_MANIFEST = FINAL2_ROOT / "frozen_dataset_manifest_v2.json"
FINAL2_G10_EXTENSION = FINAL2_ROOT / "frozen_dataset/g10_cell2_p5_label_extension.csv"
FINAL2_G10_ELIGIBILITY = FINAL2_ROOT / "frozen_dataset/g10_cell2_curve_eligibility.csv"
FINAL2_G10_SOURCE_HASHES = FINAL2_ROOT / "frozen_dataset/g10_cell2_source_sha256.json"
PART3_LABEL_CANDIDATES = Path(
    "experiments/results/data_audits/part3_soh_candidates/labels_soh_candidates.csv"
)
PART3_STEP4_SCAN = Path(
    "experiments/results/data_audits/part3_soh_candidates/combined_source_file_step4_scan.csv"
)
FINAL2_EXTERNAL_NPZ = FINAL2_ROOT / "frozen_dataset/external_test.npz"
FINAL2_MODEL_REGISTRY = Path("experiments/final2/v1/model_registry.json")
FINAL2_RUN_ROOT = FINAL2_ROOT / "run_001"
FINAL2_CHECKPOINT_MANIFEST = FINAL2_RUN_ROOT / "checkpoint_manifest.json"
FINAL2_VALIDATION_SELECTION = FINAL2_RUN_ROOT / "validation_selection.json"
FINAL2_TRAINING_REPORT = FINAL2_RUN_ROOT / "training_report.md"
FINAL2_EXTERNAL_ROOT = FINAL2_ROOT / "external_evaluation"
FINAL2_EXTERNAL_MANIFEST = FINAL2_EXTERNAL_ROOT / "external_evaluation_manifest.json"
FINAL2_EXTERNAL_METRICS = FINAL2_EXTERNAL_ROOT / "external_metrics.json"
P5_ROOT = Path(
    "experiments/soh_forecast/bafuse_architecture_benchmark_v1/"
    "bafuse_p5_electrical_v1_20261114"
)
P5_FEATURE_AUDIT = P5_ROOT / "bafuse_p5_feature_audit.csv"
P5_FEATURE_SCHEMA = P5_ROOT / "bafuse_p5_feature_schema.json"
P5_SHA_MANIFEST = P5_ROOT / "bafuse_p5_sha256_manifest.json"
P5_FEATURE_MODULE = Path("src/bafuse/hybrid/p5_electrical_features.py")
PYDMA_PART3 = Path(
    "data/processed/label_pydma_fit_part3/run_009_part3_pydma/pydma_results.csv"
)
PYDMA_G10_CELL2 = Path(
    "experiments/final2/phase_c2_g10_cell2_charge/charge_degradation_results.csv"
)
PYDMA_G10_CELL2_SHA_MANIFEST = Path(
    "experiments/final2/phase_c2_g10_cell2_charge/artifact_sha256_manifest.json"
)
OUTPUT_CSV = Path("app/part3_modality_eligibility.csv")
OUTPUT_REPORT = Path("app/report_not_demo.md")
_RAW_P5_FEATURE_CACHE: dict[tuple[str, int, int, int, int], dict[str, str]] = {}

REQUIRED_CSV_FIELDS = (
    "part",
    "group",
    "cell",
    "rpt",
    "source_file",
    "mapping_status",
    "final2_observation_status",
    "step4_valid",
    "step10_valid",
    "physics5_valid",
    "bol_reference_valid",
    "soh_label_available",
    "discharge_only_inference_eligible",
    "two_modality_inference_eligible",
    "three_modality_inference_eligible",
    "soh_evaluation_eligible",
    "pydma_charge_available",
    "pydma_charge_qc_status",
    "failure_reason",
    "recommended_action",
)

RICH_CSV_FIELDS = (
    "dataset_part",
    "physical_cell_id",
    "rpt_index",
    "source_tpg",
    "guide_stage",
    "canonical_record_id",
    "canonical_mapping_status",
    "canonical_identity_status",
    "canonical_source_path_status",
    "guide_mapping_status",
    "final2_mapping_status",
    "identity_provenance",
    "final2_partition",
    "final2_status",
    "final2_status_reason",
    "source_exists_now",
    "raw_source_scan_status",
    "raw_source_rows_scanned",
    "raw_mapping_status",
    "raw_mapping_audit_notes",
    "source_size_bytes_now",
    "source_size_bytes_reference",
    "source_size_matches_reference",
    "source_mtime_matches_reference",
    "raw_source_sha256_snapshot",
    "raw_source_sha256_currently_reverified",
    "step4_source_scan_status",
    "step4_source_scan_rows",
    "step4_capacity_finite_rows",
    "step4_capacity_span_native",
    "step4_current_signs",
    "step4_segment_reference",
    "step4_block_sha256",
    "step4_inference_eligible",
    "step4_evidence",
    "step10_segment_reference",
    "step10_block_sha256",
    "step10_inference_eligible",
    "step10_zero_current_policy",
    "step10_zero_current_excluded_count",
    "step10_zero_current_source_line",
    "step10_retained_voltage_decrease_count",
    "step10_evidence",
    "physics_inference_eligible",
    "physics_feature_basis",
    "physics5_evidence",
    "physics5_feature_names",
    "physics5_tensor_source",
    "physics5_value_source",
    "physics5_values_json",
    "physics5_segment_sha256",
    "physics5_segment_row_count",
    "physics5_calculation_failure",
    "bol_reference_evidence",
    "bol_source_file",
    "bol_segment_reference",
    "bol_block_sha256",
    "bol_source_sha256_snapshot",
    "temperature_inference_eligible",
    "temperature_evidence",
    "eis_inference_eligible",
    "eis_evidence",
    "inference_eligible",
    "inference_reason",
    "soh_label_status",
    "soh_label_percent",
    "soh_label_source",
    "label_evaluation_eligible",
    "label_evaluation_reason",
    "pydma_charge_fit_gate_passed",
    "pydma_charge_saved_qc_status",
    "pydma_charge_fit_rmse_mV",
    "pydma_charge_accepted_runs",
    "pydma_charge_total_runs",
    "pydma_charge_source_file",
    "pydma_charge_artifact",
    "pydma_charge_status_reason",
    "audit_evidence_paths",
    "uncertainties",
)
CSV_FIELDS = REQUIRED_CSV_FIELDS + RICH_CSV_FIELDS

PHYSICS_FEATURE_BASIS = (
    "same-RPT Step 4 only: mean absolute discharge current; discharge-current "
    "variability; voltage slope 20-80%; voltage derivative MAD 10-90%; "
    "voltage segment contrast 10-30% vs 70-90%"
)
STATIC_UNCERTAINTY = (
    "Capacity-ratio labels are reconstructed candidates, not official Oxford "
    "ground truth; native Amphr absolute units are unverified; same-cell BoL "
    "denominator policy is provisional. Part 3 temperature and EIS are not "
    "P5 inputs; EIS is not aligned to each RPT. Current raw CSV contents were "
    "not rehashed in this audit."
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _int(value: Any) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _float_text(value: Any) -> str:
    try:
        return str(float(value))
    except (TypeError, ValueError):
        return ""


def _key(cell: str, rpt: Any) -> tuple[str, int]:
    parsed = _int(rpt)
    if parsed is None:
        raise ValueError(f"Invalid RPT index for {cell}: {rpt!r}")
    return cell, parsed


def _index_unique(
    rows: list[dict[str, str]],
    key_fn: Any,
    description: str,
) -> dict[Any, dict[str, str]]:
    indexed: dict[Any, dict[str, str]] = {}
    for row in rows:
        key = key_fn(row)
        if key in indexed:
            raise ValueError(f"Duplicate {description} key: {key!r}")
        indexed[key] = row
    return indexed


def _normalize_source_path(value: str) -> str:
    normalized = str(value).replace("\\", "/").strip()
    prefix = "data/raw/oxford/part3/"
    if normalized.casefold().startswith(prefix):
        normalized = normalized[len(prefix) :]
    return normalized.casefold()


def _step4_segment_matches(left: str, right: str) -> bool:
    pattern = re.compile(
        r"^(?P<source>.+?)(?:#Step4:segment\d+)?[:#]rows"
        r"(?P<start>\d+)-(?P<end>\d+)$",
        flags=re.IGNORECASE,
    )
    left_match = pattern.fullmatch(str(left).strip())
    right_match = pattern.fullmatch(str(right).strip())
    return bool(
        left_match
        and right_match
        and _normalize_source_path(left_match.group("source"))
        == _normalize_source_path(right_match.group("source"))
        and left_match.group("start") == right_match.group("start")
        and left_match.group("end") == right_match.group("end")
    )


def _safe_repo_path(root: Path, relative_path: str) -> Path:
    normalized = relative_path.replace("\\", "/")
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Source path must be repository-relative: {relative_path!r}")
    return root.joinpath(*posix.parts)


def _verify_final2_artifacts(root: Path) -> None:
    manifest_path = root / FINAL2_SHA_MANIFEST
    manifest = _read_json(manifest_path)
    for name, metadata in manifest.get("artifacts", {}).items():
        path = _safe_repo_path(root, str(metadata["path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Final2 frozen artifact is missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata.get("sha256"):
            raise ValueError(f"Final2 frozen artifact checksum mismatch: {path}")


def _verify_p5_sources(root: Path) -> None:
    manifest = _read_json(root / P5_SHA_MANIFEST)
    artifacts = [
        manifest.get("outputs", {}).get("bafuse_p5_feature_audit.csv"),
        manifest.get("outputs", {}).get("bafuse_p5_feature_schema.json"),
        manifest.get("code", {}).get("p5_feature_module"),
    ]
    if any(not item for item in artifacts):
        raise ValueError("P5 checksum manifest lacks feature audit/schema/builder evidence.")
    for metadata in artifacts:
        path = _safe_repo_path(root, str(metadata["path"]))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != metadata.get("sha256"):
            raise ValueError(f"P5 source artifact checksum mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_p5_builder(root: Path) -> tuple[Any, tuple[str, ...]]:
    source_directory = str((root / "src").resolve())
    if source_directory not in sys.path:
        sys.path.insert(0, source_directory)
    try:
        from bafuse.hybrid.p5_electrical_features import (
            P5_FEATURE_NAMES,
            build_p5_electrical_features,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "The project PyTorch environment is required to import the existing "
            "P5 feature builder; run this audit with .venv-torch-cuda."
        ) from error
    return build_p5_electrical_features, tuple(P5_FEATURE_NAMES)


def _compute_p5_from_raw_step4(
    root: Path,
    observation: dict[str, str],
    raw_audit: dict[str, str],
    source_scan: dict[str, str] | None,
) -> dict[str, str]:
    reference = observation.get("step4_segment_reference", "")
    match = re.fullmatch(r"(?P<source>.+):rows(?P<start>\d+)-(?P<end>\d+)", reference)
    if not match:
        raise ValueError(f"Unrecognized exact Step 4 segment reference: {reference!r}")
    source = match.group("source")
    start = int(match.group("start"))
    end = int(match.group("end"))
    if end < start or _normalize_source_path(source) != _normalize_source_path(
        observation.get("source_file", "")
    ):
        raise ValueError("Step 4 segment source does not match the mapped RPT source.")
    if raw_audit.get("step4_valid", "").casefold() != "true":
        raise ValueError("Raw mapping audit does not mark the exact Step 4 segment valid.")
    if _normalize_source_path(raw_audit.get("step4_segment_reference", "")) != (
        _normalize_source_path(reference)
    ):
        raise ValueError("Final2 and raw mapping audits disagree on the Step 4 segment.")
    if not source_scan or source_scan.get("scan_status") != "one_contiguous_step4_candidate":
        raise ValueError("Raw source scan does not verify one contiguous Step 4 segment.")
    if source_scan.get("physical_cells") != observation.get("physical_cell_id"):
        raise ValueError("Raw Step 4 scan identity does not match the mapped physical cell.")
    if f"RPT{observation.get('rpt_index')}" not in source_scan.get(
        "measurement_ids", ""
    ).split(";"):
        raise ValueError("Raw Step 4 scan does not list the mapped RPT identity.")
    if int(source_scan.get("step4_rows", "0")) != end - start + 1:
        raise ValueError("Mapped Step 4 line range differs from the saved source-scan row count.")

    source_path = _safe_repo_path(root, source)
    if not source_path.is_file():
        raise FileNotFoundError(f"Mapped Step 4 raw source is missing: {source_path}")
    before = source_path.stat()
    expected_size = _int(source_scan.get("source_size_bytes"))
    expected_mtime = _int(source_scan.get("source_modified_ns"))
    if before.st_size != expected_size or before.st_mtime_ns != expected_mtime:
        raise ValueError("Mapped Step 4 source size/mtime differs from its saved raw scan.")

    cache_key = (str(source_path.resolve()), before.st_size, before.st_mtime_ns, start, end)
    cached = _RAW_P5_FEATURE_CACHE.get(cache_key)
    if cached is not None:
        return dict(cached)

    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError("Mapped Step 4 source has no CSV header.")
        folded = [column.strip().casefold() for column in header]
        columns: dict[str, int] = {}
        for name in ("step", "amps", "volts"):
            if folded.count(name) != 1:
                raise ValueError(
                    f"Expected exactly one {name!r} column, found {folded.count(name)}."
                )
            columns[name] = folded.index(name)

        previous_step: int | None = None
        next_step: int | None = None
        voltage_values: list[float] = []
        current_values: list[float] = []
        block_digest = hashlib.sha256()
        for row in reader:
            line_number = reader.line_num
            if line_number < start - 1:
                continue
            if line_number > end + 1:
                break
            step_value: int | None = None
            if columns["step"] < len(row):
                try:
                    parsed_step = float(row[columns["step"]])
                    if np.isfinite(parsed_step) and parsed_step.is_integer():
                        step_value = int(parsed_step)
                except (TypeError, ValueError):
                    pass
            if line_number == start - 1:
                previous_step = step_value
            elif start <= line_number <= end:
                if len(row) != len(header) or step_value != 4:
                    raise ValueError(
                        f"Step 4 row {line_number} is malformed or has a different Step ID."
                    )
                try:
                    current = float(row[columns["amps"]])
                    voltage = float(row[columns["volts"]])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Non-numeric Step 4 signal at row {line_number}.") from error
                if not np.isfinite(current) or not np.isfinite(voltage) or current > 0:
                    raise ValueError(
                        f"Step 4 row {line_number} has non-finite signals or a charge-current sample."
                    )
                current_values.append(current)
                voltage_values.append(voltage)
                block_digest.update(
                    ("\x1f".join(row) + "\n").encode("utf-8", errors="replace")
                )
            elif line_number == end + 1:
                next_step = step_value
                break

    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError("Mapped raw source changed while extracting the Step 4 segment.")
    if (
        len(voltage_values) != end - start + 1
        or not any(value < 0 for value in current_values)
        or previous_step is None
        or previous_step == 4
        or next_step is None
        or next_step == 4
        or voltage_values[0] <= voltage_values[-1]
    ):
        raise ValueError(
            "Exact Step 4 segment row count, adjacent Step boundary, or discharge "
            "voltage direction could not be validated."
        )

    build_features, feature_names = _load_p5_builder(root)
    if feature_names != tuple(
        (
            "mean_abs_discharge_current",
            "discharge_current_variability",
            "discharge_voltage_slope_20_80",
            "discharge_voltage_derivative_mad_10_90",
            "discharge_voltage_segment_contrast_10_30_vs_70_90",
        )
    ):
        raise ValueError("Existing P5 builder feature order differs from the saved schema.")
    values = build_features(voltage_values, current_values)
    ordered_values = [float(values[name]) for name in feature_names]
    if len(ordered_values) != 5 or not np.isfinite(ordered_values).all():
        raise ValueError("Existing P5 builder did not produce five finite values.")

    record = {
        "evidence": (
            "Computed by the existing build_p5_electrical_features from the exact "
            f"mapped raw Step 4 segment {reference}; verified {len(voltage_values)} "
            "finite Step 4 discharge/nonpositive-current rows (neutral zero-current "
            "samples retained), adjacent Step boundaries, source size/mtime, "
            "and all five finite named outputs. No label or evaluation artifact was used."
        ),
        "names": ";".join(feature_names),
        "tensor_source": "",
        "value_source": "computed_raw_step4",
        "values_json": json.dumps(
            dict(zip(feature_names, ordered_values)),
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        "segment_sha256": block_digest.hexdigest(),
        "segment_row_count": str(len(voltage_values)),
    }
    _RAW_P5_FEATURE_CACHE[cache_key] = record
    return dict(record)


def _load_physics_evidence(
    root: Path,
    observations: dict[tuple[str, int], dict[str, str]],
) -> dict[tuple[str, int], dict[str, str]]:
    _verify_p5_sources(root)
    schema = _read_json(root / P5_FEATURE_SCHEMA)
    feature_names = schema.get("modalities", {}).get("X_physics", {}).get(
        "feature_names_in_order", []
    )
    expected_names = [
        "mean_abs_discharge_current",
        "discharge_current_variability",
        "discharge_voltage_slope_20_80",
        "discharge_voltage_derivative_mad_10_90",
        "discharge_voltage_segment_contrast_10_30_vs_70_90",
    ]
    if feature_names != expected_names:
        raise ValueError("Saved P5 physics feature order does not match the reviewed contract.")

    feature_rows = [
        row
        for row in _read_csv(root / P5_FEATURE_AUDIT)
        if row.get("dataset_part") == "3"
    ]
    raw_columns = [f"p5_raw_{name}" for name in feature_names]
    feature_vectors: dict[tuple[str, int], np.ndarray] = {}
    feature_rows_by_key: dict[tuple[str, int], dict[str, str]] = {}
    for row in feature_rows:
        key = _key(row["physical_cell_id"], row["rpt_index"])
        if key in feature_vectors:
            raise ValueError(f"Duplicate Part 3 P5 feature audit key: {key}")
        observation = observations.get(key)
        if (
            observation is None
            or _normalize_source_path(row.get("step4_source_file", ""))
            != _normalize_source_path(observation.get("source_file", ""))
        ):
            raise ValueError(f"P5 feature audit source does not match Final2 mapping for {key}")
        vector = np.asarray([float(row[column]) for column in raw_columns], dtype=np.float32)
        if vector.shape != (5,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid saved P5 raw feature vector for {key}")
        feature_vectors[key] = vector
        feature_rows_by_key[key] = row

    extension_rows = _index_unique(
        _read_csv(root / FINAL2_G10_EXTENSION),
        lambda row: _key(row["physical_cell_id"], row["rpt_index"]),
        "G10-Cell2 saved P5 feature extension",
    )

    with np.load(root / FINAL2_EXTERNAL_NPZ, allow_pickle=False) as archive:
        required = {"X_physics_raw", "dataset_part", "physical_cell_id", "rpt_index"}
        if not required.issubset(archive.files):
            raise ValueError("Final2 external tensor lacks keyed raw P5 physics inputs.")
        values = np.asarray(archive["X_physics_raw"], dtype=np.float32)
        parts = np.asarray(archive["dataset_part"])
        cells = np.asarray(archive["physical_cell_id"])
        rpts = np.asarray(archive["rpt_index"])
    if values.ndim != 2 or values.shape[1] != 5 or not np.isfinite(values).all():
        raise ValueError("Final2 external raw physics tensor must be finite with shape (N, 5).")
    if not (len(parts) == len(cells) == len(rpts) == len(values)):
        raise ValueError("Final2 external physics keys do not align with the tensor.")

    result: dict[tuple[str, int], dict[str, str]] = {}
    for index, (part, cell, rpt) in enumerate(zip(parts, cells, rpts)):
        if int(part) != 3:
            continue
        key = _key(str(cell), rpt)
        if key in result:
            raise ValueError(f"Duplicate Final2 external physics tensor key: {key}")
        observation = observations.get(key)
        if (
            observation is None
            or observation.get("status") != "ACCEPTED"
            or observation.get("final2_partition") != "external_test"
        ):
            raise ValueError(
                f"Final2 raw physics tensor key is not an accepted external Part 3 "
                f"observation: {key}"
            )
        if key in feature_vectors:
            if not np.array_equal(values[index], feature_vectors[key]):
                raise ValueError(f"Final2 physics tensor differs from P5 raw audit for {key}")
            feature_row = feature_rows_by_key[key]
            segment_sha256 = feature_row.get("step4_block_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", segment_sha256):
                raise ValueError(f"P5 feature audit lacks a Step 4 block hash for {key}")
            evidence = (
                "Finite 5-value X_physics_raw row in saved Final2 external_test.npz; "
                "exact float32 match to the checksum-verified P5 feature audit, "
                "whose Step 4 source matches the mapped raw procedure file."
            )
            value_source = "saved_tensor_exact_p5_audit_match"
        else:
            extension = extension_rows.get(key)
            if (
                extension is None
                or extension.get("step4_eligible", "").casefold() != "true"
                or _normalize_source_path(extension.get("source_file", ""))
                != _normalize_source_path(observation.get("source_file", ""))
                or not _step4_segment_matches(
                    extension.get("step4_segment_reference", ""),
                    observation.get("step4_segment_reference", ""),
                )
            ):
                raise ValueError(f"Final2 tensor row lacks aligned G10-Cell2 feature provenance for {key}")
            segment_sha256 = extension.get("step4_block_sha256", "")
            if not re.fullmatch(r"[0-9a-f]{64}", segment_sha256):
                raise ValueError(f"G10-Cell2 extension lacks a Step 4 block hash for {key}")
            evidence = (
                "Finite 5-value X_physics_raw row in saved Final2 external_test.npz; "
                "key/source/Step 4 segment match the G10-Cell2 extension with "
                "verified Step 4 block hash."
            )
            value_source = "saved_tensor_g10_cell2_extension"
        result[key] = {
            "evidence": evidence,
            "names": ";".join(feature_names),
            "tensor_source": FINAL2_EXTERNAL_NPZ.as_posix(),
            "value_source": value_source,
            "values_json": json.dumps(
                dict(zip(feature_names, [float(value) for value in values[index]])),
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "segment_sha256": segment_sha256,
            "segment_row_count": "",
        }
    return result


def _pydma_status(
    saved_status: str,
    fit_gate: bool,
) -> tuple[str, str]:
    normalized = saved_status.strip().casefold()
    if normalized == "diagnostic_estimate_accepted" and fit_gate:
        return "accepted", "Saved diagnostic status accepted and saved PyDMA RMSE/run gate passed."
    if normalized == "diagnostic_only_final2_rejected":
        return (
            "rejected",
            "Saved downstream QC marks the diagnostic-only estimate rejected for the "
            "Final2 status available when that artifact was written.",
        )
    if normalized == "diagnostic_estimate_accepted" and not fit_gate:
        return "rejected", "Saved accepted status does not meet the recorded PyDMA RMSE/run gate."
    return "rejected", f"Saved PyDMA QC status is {saved_status or 'blank'}; not accepted."


def _load_pydma_charge_records(root: Path) -> dict[tuple[str, int], dict[str, str]]:
    part3_rows = _read_csv(root / PYDMA_PART3)
    c2_manifest = _read_json(root / PYDMA_G10_CELL2_SHA_MANIFEST)
    c2_metadata = c2_manifest.get("artifacts", {}).get("charge_degradation_results.csv")
    if not c2_metadata:
        raise ValueError("Phase C2 checksum manifest lacks charge_degradation_results.csv")
    c2_path = _safe_repo_path(root, str(c2_metadata["path"]))
    if (
        c2_path.stat().st_size != int(c2_metadata["size_bytes"])
        or hashlib.sha256(c2_path.read_bytes()).hexdigest() != c2_metadata["sha256"]
    ):
        raise ValueError("Phase C2 saved PyDMA Charge artifact checksum mismatch.")

    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in part3_rows:
        if row.get("direction", "").casefold() != "charge" or row.get("test_step") != "10":
            continue
        match = re.fullmatch(r"RPT(\d+)", row.get("checkup", "").strip(), re.IGNORECASE)
        group_match = re.search(r"(\d+)", row.get("group_id", ""))
        cell_match = re.search(r"(\d+)", row.get("cell_id", ""))
        if not (match and group_match and cell_match):
            continue
        key = (f"G{group_match.group(1)}-Cell{cell_match.group(1)}", int(match.group(1)))
        rmse = _float_text(row.get("fit_rmse_mV", ""))
        accepted_runs = _int(row.get("accepted_runs")) or 0
        total_runs = _int(row.get("total_runs")) or 0
        threshold = _float_text(row.get("fit_threshold_mV", "15"))
        fit_gate = (
            _truth(row.get("fit_accepted_by_pydma_15mV"))
            and bool(rmse)
            and float(rmse) < float(threshold or "15")
            and accepted_runs >= 3
            and total_runs >= 3
        )
        status, reason = _pydma_status(row.get("diagnostic_label_status", ""), fit_gate)
        record = {
            "available": "true",
            "qc_status": status,
            "saved_qc_status": row.get("diagnostic_label_status", ""),
            "fit_gate_passed": str(fit_gate).lower(),
            "rmse_mV": rmse,
            "accepted_runs": str(accepted_runs),
            "total_runs": str(total_runs),
            "source_file": row.get("source_file", ""),
            "artifact": PYDMA_PART3.as_posix(),
            "reason": reason,
        }
        if key in result:
            raise ValueError(f"Duplicate Part 3 PyDMA Charge key: {key}")
        result[key] = record

    for row in _read_csv(c2_path):
        key = _key(row["physical_cell_id"], row["rpt_index"])
        if row.get("fit_direction", "").casefold() != "charge" or row.get("step") != "10":
            continue
        rmse = _float_text(row.get("fit_rmse_mV", ""))
        accepted_runs = _int(row.get("accepted_runs")) or 0
        total_runs = _int(row.get("total_runs")) or 0
        fit_gate = (
            bool(rmse)
            and float(rmse) < 15.0
            and accepted_runs >= 3
            and total_runs >= 3
        )
        saved_status = row.get("quality_control_status", "")
        status, reason = _pydma_status(saved_status, fit_gate)
        if saved_status == "diagnostic_only_final2_rejected" and fit_gate:
            reason += (
                " The fit itself passed (<15 mV and at least 3 accepted runs), but "
                "the saved downstream label remains rejected; no refit or status "
                "promotion was performed."
            )
        record = {
            "available": "true",
            "qc_status": status,
            "saved_qc_status": saved_status,
            "fit_gate_passed": str(fit_gate).lower(),
            "rmse_mV": rmse,
            "accepted_runs": str(accepted_runs),
            "total_runs": str(total_runs),
            "source_file": row.get("source_file", ""),
            "artifact": PYDMA_G10_CELL2.as_posix(),
            "reason": reason,
        }
        if key in result:
            raise ValueError(f"Duplicate Part 3 PyDMA Charge key across artifacts: {key}")
        result[key] = record
    return result


def _verify_trained_run_artifacts(root: Path) -> dict[str, Any]:
    checkpoint_manifest = _read_json(root / FINAL2_CHECKPOINT_MANIFEST)
    selection = _read_json(root / FINAL2_VALIDATION_SELECTION)
    external_manifest = _read_json(root / FINAL2_EXTERNAL_MANIFEST)
    external_metrics = _read_json(root / FINAL2_EXTERNAL_METRICS)
    training_report = (root / FINAL2_TRAINING_REPORT).read_text(encoding="utf-8")
    checkpoints = external_manifest.get("checkpoint_sha256", {})
    selected = selection.get("selected_best_seed_by_model", {})
    if not checkpoints or set(checkpoints) != set(checkpoint_manifest) or set(checkpoints) != set(selected):
        raise ValueError("Saved training/evaluation checkpoint model sets do not match.")

    verified_models: list[str] = []
    for name, metadata in checkpoint_manifest.items():
        expected_sha = str(metadata.get("best_sha256", "")).lower()
        if (
            expected_sha != str(checkpoints.get(name, "")).lower()
            or expected_sha != str(selected[name].get("checkpoint_sha256", "")).lower()
            or metadata.get("best_path") != selected[name].get("checkpoint_path")
        ):
            raise ValueError(f"Selected checkpoint manifests disagree for {name}.")
        checkpoint_path = _safe_repo_path(
            root,
            f"{FINAL2_RUN_ROOT.as_posix()}/{metadata['best_path']}",
        )
        if not checkpoint_path.is_file() or _sha256_file(checkpoint_path) != expected_sha:
            raise ValueError(f"Selected trained checkpoint is missing or hash-mismatched: {name}")
        if name not in training_report:
            raise ValueError(f"Training report does not include selected model {name}.")
        verified_models.append(name)

    dataset_manifest_path = root / FINAL2_SHA_MANIFEST
    external_npz_path = root / FINAL2_EXTERNAL_NPZ
    external_manifest_path = root / FINAL2_EXTERNAL_MANIFEST
    if (
        _sha256_file(dataset_manifest_path)
        != external_manifest.get("dataset_sha256_manifest_sha256")
        or _sha256_file(external_npz_path) != external_manifest.get("external_npz_sha256")
        or _sha256_file(external_manifest_path)
        != selection.get("external_evaluation_manifest_sha256")
    ):
        raise ValueError("External evaluation manifest does not match saved data/selection artifacts.")
    if selection.get("external_evaluated") is not True:
        raise ValueError("Validation selection does not record external evaluation.")

    metrics_by_model = external_metrics.get("metrics_by_model", {})
    if (
        external_manifest.get("external_test_evaluation_count") != 1
        or external_metrics.get("external_test_evaluation_count") != 1
        or external_metrics.get("dataset_identifier")
        != external_manifest.get("dataset_identifier")
        or selection.get("dataset_identifier")
        != external_manifest.get("dataset_identifier")
        or external_metrics.get("sample_count") != external_manifest.get("sample_count")
        or external_metrics.get("cell_count") != external_manifest.get("cell_count")
        or set(metrics_by_model) != set(verified_models) | {"Train-Mean"}
        or any(
            metric.get("N") != external_manifest.get("sample_count")
            or metric.get("N_cells") != external_manifest.get("cell_count")
            or any(
                not np.isfinite(float(metric[name]))
                for name in ("MAE", "RMSE", "R2")
            )
            for metric in metrics_by_model.values()
        )
    ):
        raise ValueError("Saved external metrics do not align with the completed evaluation manifest.")

    selection_hash_matches_current = (
        _sha256_file(root / FINAL2_VALIDATION_SELECTION)
        == external_manifest.get("validation_selection_sha256")
    )
    selection_hash_note = ""
    if not selection_hash_matches_current:
        selection_hash_note = (
            "The external manifest's validation-selection SHA does not match the "
            "current selection JSON, which now contains the external-manifest back-reference; "
            "selected checkpoint hashes cross-match all current manifests."
        )
    return {
        "status": "trained/frozen",
        "verified_checkpoint_count": len(verified_models),
        "verified_models": sorted(verified_models),
        "external_evaluation_count": external_manifest["external_test_evaluation_count"],
        "external_sample_count": external_manifest["sample_count"],
        "external_cell_count": external_manifest["cell_count"],
        "external_metrics_model_count": len(metrics_by_model),
        "selection_hash_note": selection_hash_note,
        "training_report": FINAL2_TRAINING_REPORT.as_posix(),
        "checkpoint_manifest": FINAL2_CHECKPOINT_MANIFEST.as_posix(),
        "external_manifest": FINAL2_EXTERNAL_MANIFEST.as_posix(),
        "external_metrics": FINAL2_EXTERNAL_METRICS.as_posix(),
    }


def _validate_bol_reference(
    root: Path,
    cell: str,
    observation: dict[str, str],
    candidate_label: dict[str, str] | None,
    g10_extension: dict[str, str] | None,
    g10_source_hashes: dict[str, Any],
    bol_scan: dict[str, str] | None,
) -> tuple[bool, str, str, str, str, str]:
    if cell == "G10-Cell2":
        extension = g10_extension or {}
        bol_source = extension.get("bol_source_file", "")
        bol_segment = extension.get("bol_segment_reference", "")
        bol_block = extension.get("bol_block_sha256", "")
        bol_sha = extension.get("bol_source_sha256", "")
        hash_record = g10_source_hashes.get(bol_source, {})
        path = _safe_repo_path(root, bol_source) if bol_source else None
        valid = bool(
            path
            and path.is_file()
            and bol_segment
            and re.fullmatch(r"[0-9a-f]{64}", bol_block)
            and bol_sha
            and hash_record.get("sha256") == bol_sha
            and path.stat().st_size == _int(hash_record.get("size_bytes"))
            and _truth(observation.get("bol_source_exists_now"))
        )
        evidence = (
            "G10-Cell2 Final2 P5 extension supplies the same-cell BoL Step 4 "
            "segment/block hash and source SHA; source file exists and its size "
            "matches the saved source metadata (the current raw file was not rehashed)."
            if valid
            else "G10-Cell2 BoL source/segment/hash evidence is incomplete or inconsistent."
        )
        return valid, evidence, bol_source, bol_segment, bol_block, bol_sha

    label = candidate_label or {}
    bol_source_ref = label.get("baseline_source_file", "")
    segment_index = _int(label.get("baseline_segment_index"))
    normalized = _normalize_source_path(bol_source_ref)
    scan = bol_scan or {}
    path = (
        _safe_repo_path(root, f"data/raw/oxford/part3/{bol_source_ref}")
        if bol_source_ref
        else None
    )
    try:
        segments = json.loads(scan.get("step4_segments_json", "[]"))
    except json.JSONDecodeError:
        segments = []
    segment = next(
        (
            item
            for item in segments
            if int(item.get("segment_index", -1)) == segment_index
        ),
        None,
    )
    expected_size = _int(scan.get("source_size_bytes"))
    current_size = path.stat().st_size if path and path.is_file() else None
    identity_matches = (
        scan.get("physical_cells", "") == cell
        and "BoL" in scan.get("measurement_ids", "").split(";")
    )
    valid = bool(
        path
        and path.is_file()
        and segment_index is not None
        and scan.get("scan_status") == "one_contiguous_step4_candidate"
        and scan.get("step4_segment_count") == "1"
        and scan.get("step4_capacity_finite_rows", "0").isdigit()
        and int(scan.get("step4_capacity_finite_rows", "0")) > 0
        and _float_text(scan.get("step4_capacity_span", "0"))
        and float(scan.get("step4_capacity_span", "0")) > 0
        and scan.get("step4_current_signs") == "negative"
        and segment is not None
        and int(segment.get("capacity_finite_rows", 0)) > 0
        and identity_matches
        and current_size == expected_size
        and normalized == _normalize_source_path(scan.get("relative_path", ""))
    )
    evidence = (
        f"Part 3 SOH candidate BoL source {bol_source_ref}, segment index "
        f"{segment_index}; matched to the same-cell BoL source scan with one "
        "contiguous Step 4 segment, finite capacity, positive span, and matching "
        "current file size."
        if valid
        else (
            f"BoL candidate source {bol_source_ref or '(missing)'} could not be "
            "verified against a same-cell complete Step 4 source scan."
        )
    )
    return valid, evidence, bol_source_ref, f"Step4 segment index={segment_index or ''}", "", ""


def _build_row(
    root: Path,
    observation: dict[str, str],
    canonical: dict[str, str],
    raw_audit: dict[str, str],
    candidate_label: dict[str, str] | None,
    g10_extension: dict[str, str] | None,
    g10_curve: dict[str, str] | None,
    g10_source_hashes: dict[str, Any],
    source_scan: dict[str, str] | None,
    bol_scan: dict[str, str] | None,
    physics_features: dict[tuple[str, int], dict[str, str]],
    physics_failures: dict[tuple[str, int], str],
    pydma_records: dict[tuple[str, int], dict[str, str]],
) -> dict[str, str]:
    cell = observation["physical_cell_id"]
    rpt = int(observation["rpt_index"])
    source = observation["source_file"]
    source_path = _safe_repo_path(root, source)
    source_exists = source_path.is_file()
    source_size = source_path.stat().st_size if source_exists else None
    scan_reference = source_scan or {}
    source_hash_record = g10_source_hashes.get(source, {})
    size_reference = _int(scan_reference.get("source_size_bytes"))
    if size_reference is None:
        size_reference = _int(source_hash_record.get("size_bytes"))
    if size_reference is None:
        raise ValueError(f"No recorded raw source size for {source}")
    size_matches = source_size == size_reference
    mtime_reference = _int(scan_reference.get("source_modified_ns"))
    mtime_matches = (
        source_path.stat().st_mtime_ns == mtime_reference
        if source_exists and mtime_reference is not None
        else None
    )
    source_snapshot_hash = (
        str(source_hash_record.get("sha256", ""))
        or observation.get("step4_source_sha256_snapshot", "")
        or observation.get("step10_source_sha256_snapshot", "")
    )

    raw_scanned = raw_audit.get("source_scan_status") == "scanned"
    step4_valid = _truth(observation.get("step4_valid")) and _truth(
        raw_audit.get("step4_valid")
    )
    step10_valid = _truth(observation.get("step10_valid")) and _truth(
        raw_audit.get("step10_valid")
    )
    source_integrity_ok = (
        source_exists
        and size_matches
        and mtime_matches is not False
        and raw_scanned
    )
    step4_eligible = source_integrity_ok and step4_valid
    step10_eligible = source_integrity_ok and step10_valid

    zero_audit: dict[str, Any] = {}
    if g10_extension:
        try:
            zero_audit = json.loads(g10_extension.get("step10_zero_current_rows", "[]"))
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid Step 10 zero-current JSON for {cell} RPT{rpt}") from error
    zero_count = (
        _int(g10_extension.get("step10_zero_current_excluded_count"))
        if g10_extension
        else 0
    )
    zero_count = zero_count or 0
    zero_lines = [str(item.get("line", "")) for item in zero_audit if item.get("line")]
    zero_policy = (
        observation.get("step10_zero_current_policy", "")
        or (g10_extension or {}).get("step10_zero_current_policy", "")
        or ("exclude_zero_current" if cell == "G10-Cell2" else "strict")
    )
    if cell == "G10-Cell2":
        if not g10_curve or not _truth(g10_curve.get("step10_eligible")):
            step10_eligible = False
        if zero_policy != "exclude_zero_current":
            step10_eligible = False
        if zero_count > 1:
            step10_eligible = False
    elif zero_count:
        step10_eligible = False

    observation_key = (cell, rpt)
    physics_record = physics_features.get(observation_key)
    if physics_record is None and step4_eligible and step10_eligible:
        try:
            physics_record = _compute_p5_from_raw_step4(
                root,
                observation,
                raw_audit,
                source_scan,
            )
            physics_features[observation_key] = physics_record
        except Exception as error:
            physics_failures[observation_key] = (
                "P5 computation from the mapped Step 4 segment was not validated: "
                f"{type(error).__name__}: {error}"
            )
    physics5_valid = physics_record is not None
    physics_eligible = step4_eligible and physics5_valid
    bol_valid, bol_evidence, bol_source, bol_segment, bol_block, bol_sha = (
        _validate_bol_reference(
            root,
            cell,
            observation,
            candidate_label,
            g10_extension,
            g10_source_hashes,
            bol_scan,
        )
    )
    discharge_eligible = step4_eligible
    two_modality_eligible = step4_eligible and step10_eligible
    three_modality_eligible = two_modality_eligible and physics5_valid

    final2_status = observation["status"]
    final2_target = observation.get("soh_percent", "")
    candidate_target = (candidate_label or {}).get("soh_percent", "")
    label_target = final2_target or candidate_target
    label_available = bool(label_target)
    if final2_status == "ACCEPTED" and final2_target:
        label_status = "accepted_final2"
        label_source = (
            "Final2-only G10-Cell2 extension"
            if cell == "G10-Cell2"
            else "frozen Final2 P5 cohort"
        )
    elif candidate_target:
        label_status = "candidate_only_review"
        label_source = "Part 3 SOH candidate table; not approved in frozen Final2"
    else:
        label_status = "unavailable"
        label_source = "none"

    evaluation_eligible = bool(
        final2_status == "ACCEPTED"
        and final2_target
        and observation.get("final2_partition") == "external_test"
        and three_modality_eligible
        and bol_valid
    )
    if three_modality_eligible:
        inference_reason = (
            "Same-RPT Step 4 discharge, Step 10 charge, and all five "
            "saved-or-raw-computed P5 physics values are verified; this is input availability, not "
            "evidence that an inference was run."
        )
    else:
        blockers = []
        if not discharge_eligible:
            blockers.append("Step 4 discharge input is not eligible")
        if not step10_eligible:
            blockers.append("Step 10 charge input is not eligible")
        if not physics5_valid:
            blockers.append(
                "verified five-feature P5 vector is unavailable: "
                + physics_failures.get(
                    observation_key,
                    "no saved vector and raw Step 4 computation was not eligible",
                )
            )
        if not source_integrity_ok:
            blockers.append("raw source presence/scan/size check failed")
        inference_reason = "; ".join(blockers)
    if evaluation_eligible:
        evaluation_reason = (
            "ACCEPTED Final2 same-time label and complete P5 inputs; external "
            "test only with verified same-cell BoL. The target is a reconstructed capacity ratio, not "
            "Oxford ground truth."
        )
    elif final2_status == "REVIEW":
        evaluation_reason = (
            "Final2 status is REVIEW; do not score. Any numeric Part 3 "
            "capacity-ratio candidate remains review-only."
        )
    elif final2_status == "REJECTED":
        evaluation_reason = (
            "Final2 status is REJECTED; no evaluation target is available."
        )
    else:
        evaluation_reason = (
            "Current Final2 acceptance, external-test partition, and complete "
            "P5 input plus same-cell BoL eligibility are all required."
        )

    pydma = pydma_records.get(
        observation_key,
        {
            "available": "false",
            "qc_status": "missing",
            "saved_qc_status": "",
            "fit_gate_passed": "false",
            "rmse_mV": "",
            "accepted_runs": "",
            "total_runs": "",
            "source_file": "",
            "artifact": "",
            "reason": "No matching saved Part 3 Step 10 Charge PyDMA record.",
        },
    )
    failure_reasons = []
    if not step4_eligible:
        failure_reasons.append(
            f"Step 4 unavailable/invalid ({scan_reference.get('scan_status', 'raw QC failure')})."
        )
    if not step10_eligible:
        failure_reasons.append(
            f"Step 10 unavailable/invalid ({observation.get('mapping_status', '')}; "
            f"{raw_audit.get('audit_notes', '')})."
        )
    if not physics5_valid:
        failure_reasons.append(
            "The same five P5 feature values are not verifiable from saved or "
            "mapped raw evidence; Step 4 validity alone is insufficient. "
            + physics_failures.get(observation_key, "")
        )
    if not bol_valid:
        failure_reasons.append("Same-cell BoL Step 4 reference was not verified.")
    if not label_available:
        failure_reasons.append("No numeric SOH label/candidate is available.")
    elif not evaluation_eligible:
        failure_reasons.append(
            f"SOH value is not evaluation-approved (Final2 status={final2_status})."
        )
    failure_reason = "; ".join(failure_reasons) or "No eligibility failure identified."
    if evaluation_eligible:
        recommended_action = (
            "Eligible only for the existing external/domain-shift evaluation "
            "contract; this audit did not run inference or score the row."
        )
    elif final2_status == "REJECTED":
        recommended_action = (
            "Keep excluded; preserve the recorded mapping and do not infer a "
            "replacement Step 4 curve or SOH label."
        )
    elif not step10_eligible:
        recommended_action = (
            "Step 4-only baseline may use this input; keep out of 2M/3M until "
            "the existing Step 10 mapping/QC ambiguity is resolved."
        )
    elif not physics_eligible:
        recommended_action = (
            "Do not claim 3M eligibility; retain only the modalities otherwise "
            "verified and resolve the recorded P5 evidence/computation failure."
        )
    elif final2_status == "REVIEW":
        recommended_action = (
            "Keep inference-only/review status separate from scoring; resolve "
            "the existing mapping/label review before evaluation."
        )
    else:
        recommended_action = "Revisit only if new source evidence changes this row."

    if step4_eligible:
        step4_evidence = (
            "Existing raw mapping audit marked Step 4 valid; current raw source "
            "exists and matches the recorded source size."
        )
    else:
        step4_evidence = (
            "Existing raw mapping audit and/or Final2 QC does not mark Step 4 "
            "eligible; retain the recorded segment reference without assigning "
            f"a discharge modality. Step 4 source scan: "
            f"{scan_reference.get('scan_status', 'not in combined scan')}."
        )
    if step10_eligible:
        if cell == "G10-Cell2" and zero_count:
            step10_evidence = (
                f"Approved isolate policy {zero_policy}: excluded {zero_count} "
                f"zero-current row at raw source line {','.join(zero_lines)}; "
                "all positive-current rows retained and voltage decreases kept."
            )
        else:
            step10_evidence = (
                "Existing raw mapping audit and Final2 curve QC mark Step 10 "
                "valid; no unapproved zero-current exception was applied."
            )
    else:
        step10_evidence = (
            "Existing raw mapping audit and/or Final2 QC does not mark Step 10 "
            f"eligible; no replacement curve or inferred RPT mapping was used. "
            f"Raw audit note: {raw_audit.get('audit_notes', '')}"
        )
    step4_scan_status = (
        scan_reference.get("scan_status", "")
        or (
            "not_in_combined_scan; G10-Cell2-specific curve eligibility audit"
            if cell == "G10-Cell2"
            else ""
        )
    )

    audit_paths = (
        f"{DATASET_REVIEW.as_posix()};{CANONICAL_MANIFEST.as_posix()};"
        f"{MAPPING_AUDIT.as_posix()};{SOURCE_INTEGRITY_AUDIT.as_posix()};"
        f"{MAPPING_AUDIT_SUMMARY.as_posix()};{FINAL2_PREPROCESSING.as_posix()};"
        f"{FINAL2_OBSERVATIONS.as_posix()};{PART3_LABEL_CANDIDATES.as_posix()};"
        f"{PART3_STEP4_SCAN.as_posix()};{FINAL2_SUMMARY.as_posix()};"
        f"{FINAL2_SHA_MANIFEST.as_posix()};{FINAL2_BUILD_MANIFEST.as_posix()};"
        f"{FINAL2_G10_ELIGIBILITY.as_posix()};"
        f"{FINAL2_G10_EXTENSION.as_posix()};{FINAL2_G10_SOURCE_HASHES.as_posix()};"
        f"{FINAL2_EXTERNAL_NPZ.as_posix()};{P5_FEATURE_AUDIT.as_posix()};"
        f"{P5_FEATURE_SCHEMA.as_posix()};{P5_FEATURE_MODULE.as_posix()};"
        f"{P5_SHA_MANIFEST.as_posix()};"
        f"{PYDMA_PART3.as_posix()};{PYDMA_G10_CELL2.as_posix()};"
        f"{PYDMA_G10_CELL2_SHA_MANIFEST.as_posix()};"
        f"{FINAL2_MODEL_REGISTRY.as_posix()}"
    )
    uncertainties = STATIC_UNCERTAINTY
    if cell == "G10-Cell2":
        uncertainties += (
            " G10-Cell2 identity is user/project-owner confirmed because the "
            "printed Guide omits Cell 2; it is not an independent Guide label."
        )
    if final2_status != "ACCEPTED":
        uncertainties += (
            f" Final2 status={final2_status}: {observation.get('status_reason', '')}"
        )

    return {
        "part": "3",
        "dataset_part": "3",
        "group": observation["group"],
        "cell": cell.split("-Cell", 1)[1],
        "rpt": str(rpt),
        "physical_cell_id": cell,
        "rpt_index": str(rpt),
        "source_tpg": observation["source_tpg"],
        "guide_stage": observation["guide_stage"],
        "canonical_record_id": canonical.get("record_id", ""),
        "canonical_mapping_status": canonical.get("mapping_status", ""),
        "canonical_identity_status": canonical.get("cell_identity_status", ""),
        "canonical_source_path_status": canonical.get("source_path_status", ""),
        "guide_mapping_status": canonical.get("guide_mapping_status", ""),
        "final2_mapping_status": observation["mapping_status"],
        "identity_provenance": observation.get("identity_provenance", ""),
        "final2_partition": observation["final2_partition"],
        "final2_status": final2_status,
        "final2_observation_status": final2_status,
        "final2_status_reason": observation.get("status_reason", ""),
        "mapping_status": observation["mapping_status"],
        "source_file": source.replace("\\", "/"),
        "source_exists_now": str(source_exists).lower(),
        "raw_source_scan_status": raw_audit.get("source_scan_status", ""),
        "raw_source_rows_scanned": raw_audit.get("source_rows_scanned", ""),
        "raw_mapping_status": raw_audit.get("final_mapping_status", ""),
        "raw_mapping_audit_notes": raw_audit.get("audit_notes", ""),
        "source_size_bytes_now": str(source_size or ""),
        "source_size_bytes_reference": str(size_reference),
        "source_size_matches_reference": str(size_matches).lower(),
        "source_mtime_matches_reference": (
            "" if mtime_matches is None else str(mtime_matches).lower()
        ),
        "raw_source_sha256_snapshot": source_snapshot_hash,
        "raw_source_sha256_currently_reverified": "false",
        "step4_source_scan_status": step4_scan_status,
        "step4_source_scan_rows": scan_reference.get("step4_rows", ""),
        "step4_capacity_finite_rows": scan_reference.get(
            "step4_capacity_finite_rows", ""
        ),
        "step4_capacity_span_native": scan_reference.get("step4_capacity_span", ""),
        "step4_current_signs": scan_reference.get("step4_current_signs", ""),
        "step4_segment_reference": observation.get("step4_segment_reference", ""),
        "step4_block_sha256": observation.get("step4_block_sha256", ""),
        "step4_inference_eligible": str(step4_eligible).lower(),
        "step4_valid": str(step4_eligible).lower(),
        "discharge_only_inference_eligible": str(discharge_eligible).lower(),
        "step4_evidence": step4_evidence,
        "step10_segment_reference": observation.get("step10_segment_reference", ""),
        "step10_block_sha256": observation.get("step10_block_sha256", ""),
        "step10_inference_eligible": str(step10_eligible).lower(),
        "step10_valid": str(step10_eligible).lower(),
        "step10_zero_current_policy": zero_policy,
        "step10_zero_current_excluded_count": str(zero_count),
        "step10_zero_current_source_line": ";".join(zero_lines),
        "step10_retained_voltage_decrease_count": (
            (g10_curve or {}).get("step10_retained_voltage_decrease_count", "")
            or (
                json.loads((g10_curve or {}).get("step10_zero_current_audit", "{}"))
                .get("retained_voltage_decrease_count", "")
                if g10_curve
                else ""
            )
        ),
        "step10_evidence": step10_evidence,
        "physics_inference_eligible": str(physics_eligible).lower(),
        "physics5_valid": str(physics5_valid).lower(),
        "physics_feature_basis": PHYSICS_FEATURE_BASIS,
        "physics5_evidence": (
            physics_record["evidence"]
            if physics_record
            else physics_failures.get(
                observation_key,
                (
                    "No keyed finite five-value P5 row exists in the saved Final2 "
                    "external tensor; raw Step 4 calculation was not attempted because "
                    "Step 4 and Step 10 are not both eligible."
                ),
            )
        ),
        "physics5_feature_names": (
            physics_record["names"]
            if physics_record
            else (
                "mean_abs_discharge_current;discharge_current_variability;"
                "discharge_voltage_slope_20_80;"
                "discharge_voltage_derivative_mad_10_90;"
                "discharge_voltage_segment_contrast_10_30_vs_70_90"
            )
        ),
        "physics5_tensor_source": (
            physics_record["tensor_source"] if physics_record else ""
        ),
        "physics5_value_source": (
            physics_record["value_source"] if physics_record else ""
        ),
        "physics5_values_json": (
            physics_record["values_json"] if physics_record else ""
        ),
        "physics5_segment_sha256": (
            physics_record["segment_sha256"] if physics_record else ""
        ),
        "physics5_segment_row_count": (
            physics_record["segment_row_count"] if physics_record else ""
        ),
        "physics5_calculation_failure": physics_failures.get(observation_key, ""),
        "bol_reference_valid": str(bol_valid).lower(),
        "bol_reference_evidence": bol_evidence,
        "bol_source_file": bol_source,
        "bol_segment_reference": bol_segment,
        "bol_block_sha256": bol_block,
        "bol_source_sha256_snapshot": bol_sha,
        "temperature_inference_eligible": "false",
        "temperature_evidence": (
            "Temperature is not in the frozen P5 feature contract; Part 3 has "
            "known early-file temperature gaps and no imputation is performed."
        ),
        "eis_inference_eligible": "false",
        "eis_evidence": (
            "Part 3 EIS is checkpoint-level, not aligned to each RPT; no EIS "
            "spectrum is assigned or forward-filled as a same-RPT feature."
        ),
        "inference_eligible": str(three_modality_eligible).lower(),
        "inference_reason": inference_reason,
        "soh_label_available": str(label_available).lower(),
        "soh_label_status": label_status,
        "soh_label_percent": _float_text(label_target),
        "soh_label_source": label_source,
        "label_evaluation_eligible": str(evaluation_eligible).lower(),
        "soh_evaluation_eligible": str(evaluation_eligible).lower(),
        "two_modality_inference_eligible": str(two_modality_eligible).lower(),
        "three_modality_inference_eligible": str(three_modality_eligible).lower(),
        "label_evaluation_reason": evaluation_reason,
        "failure_reason": failure_reason,
        "recommended_action": recommended_action,
        "pydma_charge_available": pydma["available"],
        "pydma_charge_qc_status": pydma["qc_status"],
        "pydma_charge_fit_gate_passed": pydma["fit_gate_passed"],
        "pydma_charge_saved_qc_status": pydma["saved_qc_status"],
        "pydma_charge_fit_rmse_mV": pydma["rmse_mV"],
        "pydma_charge_accepted_runs": pydma["accepted_runs"],
        "pydma_charge_total_runs": pydma["total_runs"],
        "pydma_charge_source_file": pydma["source_file"],
        "pydma_charge_artifact": pydma["artifact"],
        "pydma_charge_status_reason": pydma["reason"],
        "audit_evidence_paths": audit_paths,
        "uncertainties": uncertainties,
    }


def build_audit_rows(root: Path) -> list[dict[str, str]]:
    root = root.resolve()
    _verify_final2_artifacts(root)
    canonical_rows = _read_csv(root / CANONICAL_MANIFEST)
    canonical_rows = [
        row
        for row in canonical_rows
        if row.get("dataset_part") == "3"
        and row.get("record_type") == "cycling_guide_record"
        and str(row.get("measurement_id", "")).startswith("RPT")
        and row.get("group_id") in {"7", "8", "9", "10"}
    ]
    canonical = _index_unique(
        canonical_rows,
        lambda row: _key(row["physical_cell_id"], row["rpt_number"]),
        "canonical Part 3 RPT",
    )

    observations = [
        row
        for row in _read_csv(root / FINAL2_OBSERVATIONS)
        if row.get("dataset_part") == "3"
        and row.get("group") in {"7", "8", "9", "10"}
    ]
    observations_by_key = _index_unique(
        observations,
        lambda row: _key(row["physical_cell_id"], row["rpt_index"]),
        "Final2 Part 3 RPT",
    )
    if set(canonical) != set(observations_by_key):
        missing_canonical = sorted(set(observations_by_key) - set(canonical))
        missing_final2 = sorted(set(canonical) - set(observations_by_key))
        raise ValueError(
            "Canonical and Final2 Part 3 RPT mappings differ; "
            f"missing canonical={missing_canonical}, missing Final2={missing_final2}"
        )

    raw_audits = _index_unique(
        [
            row
            for row in _read_csv(root / MAPPING_AUDIT)
            if row.get("part") == "3" and row.get("group") in {"7", "8", "9", "10"}
        ],
        lambda row: _key(row["physical_cell_id"], row["rpt_index"]),
        "raw-source Part 3 RPT audit",
    )
    candidate_labels = _index_unique(
        [
            row
            for row in _read_csv(root / PART3_LABEL_CANDIDATES)
            if row.get("dataset_part") == "3"
            and str(row.get("measurement_id", "")).startswith("RPT")
        ],
        lambda row: _key(
            row["physical_cell_id"], re.sub(r"^RPT", "", row["measurement_id"])
        ),
        "Part 3 candidate SOH label",
    )
    extensions = _index_unique(
        _read_csv(root / FINAL2_G10_EXTENSION),
        lambda row: _key(row["physical_cell_id"], row["rpt_index"]),
        "G10-Cell2 label extension",
    )
    curve_eligibility = _index_unique(
        _read_csv(root / FINAL2_G10_ELIGIBILITY),
        lambda row: _key(row["physical_cell_id"], row["rpt_index"]),
        "G10-Cell2 curve eligibility",
    )
    g10_source_hashes = _read_json(root / FINAL2_G10_SOURCE_HASHES).get(
        "source_files", {}
    )
    step4_scans = [
        row
        for row in _read_csv(root / PART3_STEP4_SCAN)
        if row.get("dataset_parts") == "3"
    ]
    scan_by_source = _index_unique(
        step4_scans,
        lambda row: _normalize_source_path(row["relative_path"]),
        "Part 3 source Step 4 scan",
    )
    physics_features = _load_physics_evidence(root, observations_by_key)
    physics_failures: dict[tuple[str, int], str] = {}
    pydma_records = _load_pydma_charge_records(root)

    result: list[dict[str, str]] = []
    for key, observation in observations_by_key.items():
        if key not in raw_audits:
            raise ValueError(f"Missing raw-source audit for {key}")
        canonical_row = canonical[key]
        if canonical_row.get("source_paths", "").replace("\\", "/").casefold() != (
            observation["source_file"].replace("\\", "/").casefold()
        ):
            raise ValueError(f"Canonical and Final2 source path mismatch for {key}")
        source_key = _normalize_source_path(observation["source_file"])
        source_scan = scan_by_source.get(source_key)
        extension = extensions.get(key)
        curve = curve_eligibility.get(key)
        candidate_label = candidate_labels.get(key)
        if observation["physical_cell_id"] == "G10-Cell2" and not (
            extension and curve
        ):
            raise ValueError(f"Missing approved G10-Cell2 extension/QC for {key}")
        bol_scan = (
            scan_by_source.get(
                _normalize_source_path(
                    (candidate_label or {}).get("baseline_source_file", "")
                )
            )
            if candidate_label
            else None
        )
        result.append(
            _build_row(
                root,
                observation,
                canonical_row,
                raw_audits[key],
                candidate_label,
                extension,
                curve,
                g10_source_hashes,
                source_scan,
                bol_scan,
                physics_features,
                physics_failures,
                pydma_records,
            )
        )

    result.sort(key=lambda row: (int(row["group"]), row["physical_cell_id"], int(row["rpt_index"])))
    if len(result) != 87:
        raise ValueError(f"Expected 87 mapped Part 3 RPTs; found {len(result)}")
    return result


def write_audit(root: Path, output_path: Path | None = None) -> list[dict[str, str]]:
    rows = build_audit_rows(root)
    destination = output_path or root / OUTPUT_CSV
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    report_destination = root / OUTPUT_REPORT
    report_destination.parent.mkdir(parents=True, exist_ok=True)
    write_report_from_csv(destination, report_destination, root)
    return rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    return {
        "rpt_rows": len(rows),
        "cells": len({row["physical_cell_id"] for row in rows}),
        "groups": dict(sorted(Counter(row["group"] for row in rows).items())),
        "final2_status": dict(sorted(Counter(row["final2_status"] for row in rows).items())),
        "mapping_status": dict(
            sorted(Counter(row["final2_mapping_status"] for row in rows).items())
        ),
        "raw_scanned": sum(row["raw_source_scan_status"] == "scanned" for row in rows),
        "raw_present": sum(_truth(row["source_exists_now"]) for row in rows),
        "step4_eligible": sum(_truth(row["step4_inference_eligible"]) for row in rows),
        "step10_eligible": sum(_truth(row["step10_inference_eligible"]) for row in rows),
        "physics5_valid": sum(_truth(row["physics5_valid"]) for row in rows),
        "bol_reference_valid": sum(_truth(row["bol_reference_valid"]) for row in rows),
        "discharge_only_inference_eligible": sum(
            _truth(row["discharge_only_inference_eligible"]) for row in rows
        ),
        "two_modality_inference_eligible": sum(
            _truth(row["two_modality_inference_eligible"]) for row in rows
        ),
        "three_modality_inference_eligible": sum(
            _truth(row["three_modality_inference_eligible"]) for row in rows
        ),
        "full_p5_inference_eligible": sum(
            _truth(row["three_modality_inference_eligible"]) for row in rows
        ),
        "soh_label_available": sum(_truth(row["soh_label_available"]) for row in rows),
        "label_evaluation_eligible": sum(
            _truth(row["soh_evaluation_eligible"]) for row in rows
        ),
        "pydma_charge_available": sum(
            _truth(row["pydma_charge_available"]) for row in rows
        ),
        "pydma_charge_qc_status": dict(
            sorted(Counter(row["pydma_charge_qc_status"] for row in rows).items())
        ),
    }


def _row_order(row: dict[str, str]) -> tuple[int, str, int]:
    return int(row["group"]), row["physical_cell_id"], int(row["rpt_index"])


def _rpt_keys(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "—"
    return ", ".join(
        f"G{row['group']}-Cell{row['physical_cell_id'].split('-Cell', 1)[1]}/RPT{row['rpt_index']}"
        for row in sorted(rows, key=_row_order)
    )


def _report_model_categories(root: Path) -> tuple[list[str], list[str], dict[str, Any]]:
    registry = _read_json(root / FINAL2_MODEL_REGISTRY)
    training_evidence = _verify_trained_run_artifacts(root)
    models = registry.get("models", [])
    two = [
        model["name"]
        for model in models
        if set(model.get("modalities", []))
        == {"discharge_voltage", "charge_voltage"}
    ]
    three = [
        model["name"]
        for model in models
        if set(model.get("modalities", []))
        == {"discharge_voltage", "charge_voltage", "p5_physics"}
    ]
    if not two or not three:
        raise ValueError("Final2 model registry lacks the expected 2M/3M modality contracts.")
    registered_model_names = set(two) | set(three)
    verified_model_names = set(training_evidence["verified_models"]) - {
        "CNN1D-Step4-Baseline",
        "Train-Mean",
    }
    if registered_model_names != verified_model_names:
        raise ValueError("Saved 2M/3M model registry does not match verified trained checkpoints.")
    training_evidence["registry_status_metadata"] = str(registry.get("status", ""))
    return two, three, training_evidence


def _render_report(rows: list[dict[str, str]], root: Path) -> str:
    counts = summarize(rows)
    by_group: dict[str, list[dict[str, str]]] = {}
    by_cell: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_group.setdefault(row["group"], []).append(row)
        by_cell.setdefault(row["physical_cell_id"], []).append(row)

    baseline_ineligible = [
        row for row in rows if not _truth(row["discharge_only_inference_eligible"])
    ]
    two_ineligible = [
        row for row in rows if not _truth(row["two_modality_inference_eligible"])
    ]
    three_ineligible = [
        row for row in rows if not _truth(row["three_modality_inference_eligible"])
    ]
    evaluation_ineligible = [
        row for row in rows if not _truth(row["soh_evaluation_eligible"])
    ]
    pydma_status_counts = Counter(row["pydma_charge_qc_status"] for row in rows)
    pydma_rejected = [
        row for row in rows if row["pydma_charge_qc_status"] == "rejected"
    ]
    pydma_missing = [
        row for row in rows if row["pydma_charge_qc_status"] == "missing"
    ]
    p5_exact_matches = sum(
        "exact float32 match" in row["physics5_evidence"] for row in rows
    )
    p5_extension_rows = sum(
        "G10-Cell2 extension" in row["physics5_evidence"] for row in rows
    )
    p5_raw_computed = sum(
        row["physics5_value_source"] == "computed_raw_step4"
        for row in rows
    )
    p5_saved_vectors = sum(bool(row["physics5_tensor_source"]) for row in rows)
    raw_scanned = sum(row["raw_source_scan_status"] == "scanned" for row in rows)
    raw_present = sum(_truth(row["source_exists_now"]) for row in rows)
    raw_size_matched = sum(_truth(row["source_size_matches_reference"]) for row in rows)
    mtime_referenced = sum(bool(row["source_mtime_matches_reference"]) for row in rows)
    mtime_matched = sum(
        _truth(row["source_mtime_matches_reference"]) for row in rows
    )
    mapping_counts = Counter(row["final2_mapping_status"] for row in rows)

    group_lines = [
        "| Group | Cells | RPT | ACCEPTED | REVIEW | REJECTED | Step 4 | Step 10 | P5-5 | BoL | Labels | SOH evaluation |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group in sorted(by_group, key=int):
        subset = by_group[group]
        statuses = Counter(row["final2_status"] for row in subset)
        group_lines.append(
            "| {group} | {cells} | {rpts} | {accepted} | {review} | {rejected} | "
            "{step4} | {step10} | {physics} | {bol} | {labels} | {evaluation} |".format(
                group=group,
                cells=len({row["physical_cell_id"] for row in subset}),
                rpts=len(subset),
                accepted=statuses.get("ACCEPTED", 0),
                review=statuses.get("REVIEW", 0),
                rejected=statuses.get("REJECTED", 0),
                step4=sum(_truth(row["step4_valid"]) for row in subset),
                step10=sum(_truth(row["step10_valid"]) for row in subset),
                physics=sum(_truth(row["physics5_valid"]) for row in subset),
                bol=sum(_truth(row["bol_reference_valid"]) for row in subset),
                labels=sum(_truth(row["soh_label_available"]) for row in subset),
                evaluation=sum(_truth(row["soh_evaluation_eligible"]) for row in subset),
            )
        )

    cell_lines = [
        "| Part | Group | Cell | RPTs | Step4 | Baseline-ineligible RPTs | 2M | 2M-ineligible RPTs | P5-5 | 3M | 3M-ineligible RPTs | BoL | SOH labels | SOH evaluation | PyDMA accepted RPTs | PyDMA rejected RPTs | PyDMA missing RPTs |",
        "|---:|---:|---|---:|---:|---|---:|---|---:|---:|---|---:|---:|---:|---|---|---|",
    ]
    for cell in sorted(by_cell, key=lambda value: _row_order(by_cell[value][0])):
        subset = sorted(by_cell[cell], key=_row_order)
        first = subset[0]
        cell_lines.append(
            "| 3 | {group} | {cell} | {rpts} | {step4} | {baseline_bad} | {two} | "
            "{two_bad} | {physics} | {three} | {three_bad} | {bol} | {labels} | "
            "{evaluation} | {pydma_ok} | {pydma_bad} | {pydma_missing} |".format(
                group=first["group"],
                cell=cell,
                rpts=len(subset),
                step4=sum(_truth(row["step4_valid"]) for row in subset),
                baseline_bad=_rpt_keys(
                    [row for row in subset if not _truth(row["discharge_only_inference_eligible"])]
                ),
                two=sum(_truth(row["two_modality_inference_eligible"]) for row in subset),
                two_bad=_rpt_keys(
                    [row for row in subset if not _truth(row["two_modality_inference_eligible"])]
                ),
                physics=sum(_truth(row["physics5_valid"]) for row in subset),
                three=sum(_truth(row["three_modality_inference_eligible"]) for row in subset),
                three_bad=_rpt_keys(
                    [row for row in subset if not _truth(row["three_modality_inference_eligible"])]
                ),
                bol=sum(_truth(row["bol_reference_valid"]) for row in subset),
                labels=sum(_truth(row["soh_label_available"]) for row in subset),
                evaluation=sum(_truth(row["soh_evaluation_eligible"]) for row in subset),
                pydma_ok=_rpt_keys(
                    [row for row in subset if row["pydma_charge_qc_status"] == "accepted"]
                ),
                pydma_bad=_rpt_keys(
                    [row for row in subset if row["pydma_charge_qc_status"] == "rejected"]
                ),
                pydma_missing=_rpt_keys(
                    [row for row in subset if row["pydma_charge_qc_status"] == "missing"]
                ),
            )
        )

    two_models, three_models, model_evidence = _report_model_categories(root)
    zero_rows = [
        row
        for row in rows
        if row["step10_zero_current_policy"] == "exclude_zero_current"
        and _int(row["step10_zero_current_excluded_count"])
    ]
    zero_lines = []
    for row in sorted(zero_rows, key=_row_order):
        zero_lines.append(
            f"- {row['physical_cell_id']} RPT{row['rpt_index']}: policy "
            f"`{row['step10_zero_current_policy']}`, excluded "
            f"{row['step10_zero_current_excluded_count']} row(s), source line(s) "
            f"{row['step10_zero_current_source_line'] or 'not recorded'}, block "
            f"SHA-256 `{row['step10_block_sha256']}`. {row['step10_evidence']}"
        )
    if not zero_lines:
        zero_lines = ["- Không có Step 10 zero-current exception được ghi trong CSV."]

    rejected_details = [
        f"{row['physical_cell_id']} RPT{row['rpt_index']} "
        f"({row['pydma_charge_saved_qc_status']}; fit gate passed="
        f"{row['pydma_charge_fit_gate_passed']}; RMSE="
        f"{row['pydma_charge_fit_rmse_mV']} mV)"
        for row in sorted(pydma_rejected, key=_row_order)
    ]
    rejected_text = ", ".join(rejected_details) if rejected_details else "Không có."

    return f"""# Kiểm toán eligibility modality Part 3 (Groups 7–10)

## 1. Mục tiêu và phạm vi

Đây là báo cáo kiểm toán dữ liệu, **không phải demo và không xác nhận inference hoặc evaluation mới đã chạy**. Kiểm tra mọi RPT Part 3 có trong canonical mapping và cohort Final2, kể cả ACCEPTED, REVIEW, REJECTED và ứng viên nhãn chưa được chấp nhận. Không huấn luyện, không fit scaler, không tối ưu PyDMA, không sửa raw hay dataset frozen.

## 2. Nguồn, cohort và provenance

Đã đọc [`experiments/dataset_review/README.md`](../experiments/dataset_review/README.md) trước khi thực hiện. CSV chi tiết là nguồn sự thật cho mọi con số trong báo cáo này.

- Canonical mapping: `{CANONICAL_MANIFEST.as_posix()}`; raw mapping/QC: `{MAPPING_AUDIT.as_posix()}` và `{SOURCE_INTEGRITY_AUDIT.as_posix()}`.
- Final2 observation/QC/tensors: `{FINAL2_OBSERVATIONS.as_posix()}`, `{FINAL2_SHA_MANIFEST.as_posix()}`, `{FINAL2_EXTERNAL_NPZ.as_posix()}`; preprocessing contract: `{FINAL2_PREPROCESSING.as_posix()}`.
- P5 feature audit/schema: `{P5_FEATURE_AUDIT.as_posix()}`, `{P5_FEATURE_SCHEMA.as_posix()}`.
- Part 3 PyDMA Charge: `{PYDMA_PART3.as_posix()}`; G10-Cell2 saved Charge QC: `{PYDMA_G10_CELL2.as_posix()}`.

Tổng cộng **{counts['rpt_rows']} RPT / {counts['cells']} cell** thuộc Groups {", ".join(sorted(counts["groups"], key=int))}; canonical mapping và Final2 đã được đối chiếu theo khóa vật lý cell/RPT trong audit. Trạng thái Final2: **{counts["final2_status"].get("ACCEPTED", 0)} ACCEPTED, {counts["final2_status"].get("REVIEW", 0)} REVIEW, {counts["final2_status"].get("REJECTED", 0)} REJECTED**. Group 10 Cell 2 có provenance xác nhận project-owner; không suy ra identity khác từ tên file.

Raw source evidence trong CSV: **{raw_scanned}/{counts['rpt_rows']}** RPT có source scan status `scanned`, **{raw_present}/{counts['rpt_rows']}** file hiện tồn tại, **{raw_size_matched}/{counts['rpt_rows']}** kích thước khớp snapshot; `mtime` khớp **{mtime_matched}/{mtime_referenced}** hàng có reference (`{counts['rpt_rows'] - mtime_referenced}` hàng không có mtime reference). Mapping status: {", ".join(f"`{key}`={value}" for key, value in sorted(mapping_counts.items()))}.

Final2 artifact checksum được kiểm tra bởi bộ audit. P5 `feature_audit`, `feature_schema` và feature-builder source được đối chiếu với checksum manifest. Tám raw Step 4 segment thiếu vector đã được đọc đúng line range để xác minh Step ID, current/voltage, biên Step và tính năm feature; các raw CSV khác không được parse đầy đủ hoặc rehash trong bước này. Với các file đó, báo cáo dùng scan/QC đã lưu và metadata kích thước/mtime.

## 3. Quy tắc eligibility

- **Baseline/discharge-only** cần `step4_valid`; **2M** cần Step 4 + Step 10; **3M** cần cả hai cùng 5 giá trị physics hữu hạn theo đúng thứ tự schema, dù được nạp từ artifact đã lưu hay tính hợp lệ từ raw.
- `physics5_valid` không được suy ra từ Step 4. Có **{p5_saved_vectors}** vector trong tensor Final2; **{p5_exact_matches}** khớp chính xác float32 với P5 feature audit và **{p5_extension_rows}** thuộc G10-Cell2 extension. Với hàng có Step 4 + Step 10 hợp lệ nhưng thiếu vector lưu, audit gọi `build_p5_electrical_features` trên đúng raw Step 4 segment đã map, rồi xác nhận đủ 5 tên/giá trị hữu hạn; có **{p5_raw_computed}** vector được tính và kiểm chứng như vậy. CSV lưu nguồn, giá trị, hash segment, số dòng và lỗi nếu phép tính không xác minh được.
- BoL chỉ hợp lệ khi có same-cell source/segment Step 4 evidence: candidate baseline và source scan một segment liên tục cho các cell có Guide identity; G10-Cell2 dùng extension với BoL source, segment và hash đã lưu.
- `soh_label_available` phân biệt nhãn số/candidate với `soh_evaluation_eligible`. Chỉ ACCEPTED Final2 external-test row có đủ 3M input và BoL mới đủ điều kiện chấm; nhãn capacity ratio là reconstructed, không phải ground truth chính thức.
- Đây là eligibility same-time `X(t) -> SOH(t)`, không phải future forecast. Không dùng RPT/cell/group/file/part ID làm feature; Part 3 giữ vai trò external/domain-shift cohort.

## 4. RPT không đủ điều kiện theo modality

### Baseline — `CNN1D-Step4-Baseline` (Step 4 discharge only)

Đủ điều kiện: **{counts['discharge_only_inference_eligible']}**; không đủ điều kiện: **{len(baseline_ineligible)}**.

{_rpt_keys(baseline_ineligible)}

### 2M — Step 4 discharge + Step 10 charge

Đủ điều kiện: **{counts['two_modality_inference_eligible']}**; không đủ điều kiện: **{len(two_ineligible)}**.

{_rpt_keys(two_ineligible)}

### 3M — Step 4 + Step 10 + năm P5 physics feature

Đủ điều kiện: **{counts['three_modality_inference_eligible']}**; không đủ điều kiện: **{len(three_ineligible)}**.

{_rpt_keys(three_ineligible)}

Với 3M, eligibility inference độc lập với nhãn và membership trong external-test tensor; P5 được tính từ raw mapped Step 4 khi thiếu saved vector, không tạo nhãn. Tổng quan các cờ trong CSV: Step 4 **{counts['step4_eligible']}**, Step 10 **{counts['step10_eligible']}**, P5-5 **{counts['physics5_valid']}**, BoL **{counts['bol_reference_valid']}**, có giá trị SOH **{counts['soh_label_available']}**, đủ SOH evaluation **{counts['label_evaluation_eligible']}**. Số hàng không đủ SOH evaluation: **{len(evaluation_ineligible)}** — {_rpt_keys(evaluation_ineligible)}.

| Group | Cells | RPT | ACCEPTED | REVIEW | REJECTED | Step 4 | Step 10 | P5-5 | BoL | Labels | SOH evaluation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(group_lines[2:])}

## 5. Tóm tắt theo cell

Các cột RPT-ineligible liệt kê key chính xác; PyDMA columns liệt kê các RPT theo từng trạng thái.

{chr(10).join(cell_lines)}

## 6. PyDMA Charge coverage (artifact đã lưu; không refit)

Coverage theo exact key `(Part 3, Group, Cell, RPT)`: **{counts['pydma_charge_available']}/{counts['rpt_rows']} có bản ghi**, gồm **{pydma_status_counts.get('accepted', 0)} accepted, {pydma_status_counts.get('rejected', 0)} rejected, {pydma_status_counts.get('missing', 0)} missing**. CSV ghi trạng thái và artifact nguồn cho từng RPT. Đây là model-derived diagnostic, không phải SOH label/ground truth.

RPT bị đánh dấu rejected trong saved downstream QC: {rejected_text}. G10-Cell2 RPT10 có 3/3 runs và RMSE dưới 15 mV (`pydma_charge_fit_gate_passed=true`) nhưng artifact Phase C2 cũ vẫn ghi `diagnostic_only_final2_rejected`; Final2 hiện hành ghi ACCEPTED sau zero-current policy. Báo cáo giữ nguyên status lưu trong PyDMA artifact, không promote/re-run fit. Missing keys: {_rpt_keys(pydma_missing)}.

## 7. Danh mục demo theo contract (không phải kết quả inference mới)

- **Baseline, Step 4 only:** `CNN1D-Step4-Baseline`. `Train-Mean` chỉ là reference theo label train, không phải modality model.
- **2M, discharge + charge:** {", ".join(f"`{name}`" for name in two_models)}.
- **3M, discharge + charge + P5 physics:** {", ".join(f"`{name}`" for name in three_models)}.

Trạng thái artifact hiện có được xác minh là **{model_evidence['status']}**: {model_evidence['verified_checkpoint_count']} checkpoint đã chọn tồn tại và SHA-256 khớp giữa checkpoint manifest, validation selection và external-evaluation manifest. Có **{model_evidence['external_evaluation_count']}** lượt external evaluation đã lưu, với **{model_evidence['external_sample_count']}** sample / **{model_evidence['external_cell_count']}** cell và {model_evidence['external_metrics_model_count']} bộ metrics (gồm Train-Mean reference). Bằng chứng: `{model_evidence['training_report']}`, `{model_evidence['checkpoint_manifest']}`, `{model_evidence['external_manifest']}`, `{model_evidence['external_metrics']}`. Registry metadata còn ghi `{model_evidence['registry_status_metadata']}` nên không được dùng làm trạng thái thực thi thay cho artifacts. External metrics có N={model_evidence['external_sample_count']}; tám vector P5 tính từ raw cho các REVIEW row không thuộc nhãn/evaluation được xác nhận và không được cộng vào các metrics đã lưu. Không training/evaluation lại.

{model_evidence['selection_hash_note'] or "Validation-selection SHA trong manifest khớp với selection artifact hiện tại."}

## 8. Step 10 zero-current handling đã duyệt

{chr(10).join(zero_lines)}

## 9. Giới hạn và bất định

- Cohort là **{counts['rpt_rows']} mapped RPT**, không xác nhận toàn bộ Part 3 CSV hay MAT. Chỉ tám exact Step 4 segments được parse lại phục vụ feature calculation; raw CSV không được rehash toàn file và các giá trị ngoài các segment đó không được kiểm chứng lại. `source_size_matches_reference`/mtime và QC là metadata/evidence đã lưu, không phải chứng minh byte-for-byte raw identity hiện tại.
- P5 có {p5_saved_vectors} vector từ tensor Final2 và {p5_raw_computed} vector tính trực tiếp từ raw Step 4 đã map khi Step 4 + Step 10 hợp lệ; không tạo nhãn. Candidate còn thiếu P5 sau cả hai nguồn: **{counts['rpt_rows'] - counts['physics5_valid']}** — {_rpt_keys([row for row in rows if not _truth(row['physics5_valid'])])}. REVIEW/REJECTED chỉ được tính feature cho inference khi cùng các kiểm tra Step 4/10 thành công.
- Part 3 SOH values là candidate/reconstructed capacity ratio; `Amphr` unit chưa được xác minh độc lập. BoL denominator policy vẫn theo project review provisional.
- PyDMA coverage/status phản ánh các file CSV đã lưu; PyDMA Part 3 artifact không có checksum manifest cùng phạm vi trong bộ này. Phase C2 CSV được kiểm tra theo SHA/size manifest.
- External-evaluation manifest, NPZ, dataset hash manifest và các checkpoint chọn được kiểm tra SHA-256; `external_metrics.json` được kiểm tra cấu trúc/N/cell nhưng manifest không ghi checksum riêng cho file metrics. Validation-selection SHA lưu trong external manifest không trùng file selection hiện tại; back-reference và SHA các checkpoint vẫn khớp như nêu ở Mục 7.
- Temperature thiếu ở một số file Part 3; EIS checkpoint không được căn chỉnh RPT-level. Không điền missingness hoặc gán EIS/temperature.
- Step 10 zero-current exception áp dụng chỉ cho hàng được nêu trong CSV; không mở rộng chính sách sang cell/RPT khác.

## 10. Mở rộng EoL chỉ cho demo

Guide-mapped cycling EoL Part 3 được kiểm toán riêng trong
`app/part3_eol_audit_report.md`; chi tiết từng mapping/modality nằm trong
`app/part3_demo_eol_eligibility.csv` và `app/part3_mapping_completeness.csv`.
Chín EoL observation có Step 4/P5 hợp lệ và provisional same-cell BoL ratio,
nhưng Step 10 bị giữ lại vì mỗi file có ba block phân tách bởi hàng thiếu dữ
liệu. Nhãn EoL không đủ điều kiện evaluation. 36 EIS EoL CSV là checkpoint,
không phải RPT; 36 MAT tương ứng không có. Các record này không được thêm vào
cohort, split, NPZ, scaler, checkpoint hoặc metrics frozen Final2.

## 11. Kết luận

CSV phân biệt inference eligibility theo ba contract với label/evaluation
eligibility, xác minh độc lập năm P5 features, same-cell BoL, và saved PyDMA
Charge status. Các con số và danh sách trong báo cáo được dựng bằng cách đọc
lại `{OUTPUT_CSV.as_posix()}`. EoL được bổ sung qua mapping extension demo-only;
raw và frozen Final2 artifacts không bị sửa.
"""


def write_report_from_csv(
    csv_path: Path,
    report_path: Path,
    root: Path,
) -> str:
    rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"Cannot generate report from an empty audit CSV: {csv_path}")
    missing_columns = set(REQUIRED_CSV_FIELDS) - set(rows[0])
    if missing_columns:
        raise ValueError(f"Audit CSV is missing required columns: {sorted(missing_columns)}")
    report = _render_report(rows, root)
    report_path.write_text(report, encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the read-only Part 3 modality and label eligibility audit."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    rows = write_audit(args.root, args.output)
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
