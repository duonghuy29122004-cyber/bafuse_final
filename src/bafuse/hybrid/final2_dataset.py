from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bafuse.hybrid.p5_normalization import PopulationZScore, fit_population_zscore
from bafuse.hybrid.p5_electrical_features import (
    P5_FEATURE_NAMES,
    build_p5_electrical_features,
)
from bafuse.hybrid.soh_fusion import resample_curve
from bafuse.labeling import raw_soh_reconstruction as raw


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIRECTORY = PROJECT_ROOT / "experiments" / "final2" / "v1" / "dataset"

PHYSICS_FEATURE_NAMES = (
    "mean_abs_discharge_current",
    "discharge_current_variability",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)
P5_RAW_FEATURE_COLUMNS = (
    "p5_raw_mean_abs_discharge_current",
    "p5_raw_discharge_current_variability",
    "p5_raw_discharge_voltage_slope_20_80",
    "p5_raw_discharge_voltage_derivative_mad_10_90",
    "p5_raw_discharge_voltage_segment_contrast_10_30_vs_70_90",
)
EXPECTED_PARTITION_CELL_COUNTS = {
    "train": 12,
    "validation": 4,
    "external_test": 12,
}
P5_PARTITION_FILES = {
    "train": "bafuse_train_p5.npz",
    "validation": "bafuse_validation_p5.npz",
    "external_test": "bafuse_external_p5.npz",
}
P5_REQUIRED_OUTPUTS = (
    *P5_PARTITION_FILES.values(),
    "bafuse_p5_split_manifest.csv",
    "bafuse_soh_labels_frozen.csv",
    "bafuse_p5_feature_audit.csv",
    "bafuse_p5_feature_schema.json",
    "bafuse_p5_preprocessing_config.json",
)
P5_LABEL_DEFINITION = (
    "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)"
)
G10_CELL2_PROVENANCE = (
    "Group 10 Cell 2 identity confirmed by project owner; "
    "explicit Cell 2 label omitted from printed Guide."
)
G10_CELL2_TPG_BY_RPT = {
    1: "TPG10.3",
    2: "TPG10.6",
    3: "TPG10.7",
    4: "TPG10.8",
    5: "TPG10.10",
    6: "TPG10.13",
    7: "TPG10.14",
    8: "TPG10.17",
    9: "TPG10.18",
    10: "TPG10.19",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


@dataclass(frozen=True)
class Final2Arrays:
    X_discharge: np.ndarray
    X_charge: np.ndarray
    X_physics_raw: np.ndarray
    y_soh: np.ndarray

    def validate(self) -> None:
        discharge = np.asarray(self.X_discharge)
        charge = np.asarray(self.X_charge)
        physics = np.asarray(self.X_physics_raw)
        target = np.asarray(self.y_soh)
        if discharge.ndim != 2 or discharge.shape[1] != 128:
            raise ValueError("X_discharge must have shape (N, 128).")
        if charge.shape != discharge.shape:
            raise ValueError("X_charge must match X_discharge shape.")
        if physics.shape != (discharge.shape[0], len(PHYSICS_FEATURE_NAMES)):
            raise ValueError("X_physics_raw must have shape (N, 5).")
        if target.shape not in {(discharge.shape[0],), (discharge.shape[0], 1)}:
            raise ValueError("y_soh must have shape (N,) or (N, 1).")
        if discharge.shape[0] == 0:
            raise ValueError("Final2 arrays must contain at least one observation.")
        if not all(
            np.isfinite(values).all()
            for values in (discharge, charge, physics, target)
        ):
            raise ValueError("Final2 arrays contain non-finite values.")

    @property
    def sample_count(self) -> int:
        return int(np.asarray(self.y_soh).shape[0])


@dataclass(frozen=True)
class Final2Scalers:
    discharge: PopulationZScore
    charge: PopulationZScore
    physics: PopulationZScore
    target: PopulationZScore

    def to_dict(
        self, dataset_identifier: str = "bafuse-final2-p5-g10-guide-verified-v1"
    ) -> dict[str, object]:
        return {
            "schema_version": 1,
            "dataset_identifier": dataset_identifier,
            "fit_partition": "train",
            "fit_scope": "accepted Final2 Train observations only",
            "variant": "P5 Variant D semantics with newly fitted Final2 parameters",
            "voltage_normalization": True,
            "physics_normalization": "P5 population z-score, Train only",
            "target_normalization": True,
            "discharge": self.discharge.to_dict(),
            "charge": self.charge.to_dict(),
            "physics": self.physics.to_dict(),
            "target": self.target.to_dict(),
        }


def compute_soh_percent(q_rpt: float, q_bol: float) -> float:
    values = (float(q_rpt), float(q_bol))
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Step 4 capacity integrals must be finite.")
    if values[0] <= 0 or values[1] <= 0:
        raise ValueError("Step 4 capacity integrals must be positive.")
    return 100.0 * values[0] / values[1]


def expected_bol_source_part(dataset_part: int, group: int) -> int | None:
    if dataset_part == 1:
        return 1
    if dataset_part == 2:
        return 1 if group in {1, 2, 3, 4} else None
    if dataset_part == 3:
        return 3
    raise ValueError(f"Unsupported dataset part: {dataset_part}.")


def fit_final2_scalers(
    train: Final2Arrays,
    *,
    accepted_cohort_complete: bool,
    training_authorized: bool,
) -> Final2Scalers:
    if not accepted_cohort_complete:
        raise RuntimeError(
            "Final2 scaler fitting is blocked until the accepted cohort is complete."
        )
    if not training_authorized:
        raise RuntimeError("Final2 normalization fitting is not authorized.")
    return _fit_final2_scalers(train)


def fit_final2_scalers_for_frozen_cohort(
    train: Final2Arrays,
    *,
    accepted_cohort_frozen: bool,
    training_authorized: bool,
) -> Final2Scalers:
    if not accepted_cohort_frozen:
        raise RuntimeError(
            "Final2 scaler fitting requires a verified frozen Accepted cohort."
        )
    if not training_authorized:
        raise RuntimeError("Final2 normalization fitting is not authorized.")
    return _fit_final2_scalers(train)


def _fit_final2_scalers(train: Final2Arrays) -> Final2Scalers:
    train.validate()
    physics_minimum_std = float.fromhex("0x0.0000000000001p-1022")
    target = np.asarray(train.y_soh, dtype=np.float64).reshape(-1, 1)
    return Final2Scalers(
        discharge=fit_population_zscore(train.X_discharge, minimum_std=1e-6),
        charge=fit_population_zscore(train.X_charge, minimum_std=1e-6),
        physics=fit_population_zscore(
            train.X_physics_raw, minimum_std=physics_minimum_std
        ),
        target=fit_population_zscore(target, minimum_std=1e-8),
    )


def transform_final2_arrays(
    arrays: Final2Arrays, scalers: Final2Scalers
) -> dict[str, np.ndarray]:
    arrays.validate()
    target = np.asarray(arrays.y_soh, dtype=np.float64).reshape(-1, 1)
    return {
        "X_discharge": scalers.discharge.transform(arrays.X_discharge),
        "X_charge": scalers.charge.transform(arrays.X_charge),
        "X_physics": scalers.physics.transform(arrays.X_physics_raw),
        "y_soh": scalers.target.transform(target),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_frozen_p5_artifacts(project_root: Path) -> dict[str, str]:
    p5_directory = (
        project_root
        / "experiments"
        / "soh_forecast"
        / "bafuse_architecture_benchmark_v1"
        / "bafuse_p5_electrical_v1_20261114"
    )
    manifest_path = p5_directory / "bafuse_p5_sha256_manifest.json"
    manifest = _read_json(manifest_path)
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Frozen P5 checksum manifest has no output inventory.")
    verified: dict[str, str] = {}
    for name in P5_REQUIRED_OUTPUTS:
        item = outputs.get(name)
        if not isinstance(item, dict):
            raise ValueError(f"P5 checksum manifest is missing {name}.")
        relative_path = str(item.get("path", "")).strip()
        expected_hash = str(item.get("sha256", "")).strip().lower()
        if not _SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"P5 checksum for {name} is invalid.")
        artifact_path = (project_root / relative_path).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Frozen P5 artifact is missing: {relative_path}")
        actual_hash = _sha256(artifact_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Frozen P5 artifact checksum mismatch: {relative_path}")
        verified[name] = actual_hash
    return verified


def _field(row: dict[str, str], *names: str, default: str = "") -> str:
    lookup = {str(key).casefold(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.casefold())
        if value is not None:
            return str(value).strip()
    return default


def _parse_part(value: str) -> int:
    text = str(value).strip()
    match = re.search(r"(?:part\s*)?([123])", text, re.IGNORECASE)
    if not match:
        raise ValueError(f"Unrecognized dataset part: {value!r}.")
    return int(match.group(1))


def _parse_group(cell_id: str, group_value: str = "") -> int:
    if group_value:
        return int(group_value)
    match = re.match(r"G(\d+)-", cell_id, re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot determine group for physical cell {cell_id!r}.")
    return int(match.group(1))


def _validate_source_cell(source_file: str, cell_id: str) -> None:
    source_cell = re.search(r"Cell[_ -]?(\d+)", source_file, re.IGNORECASE)
    expected_cell = re.search(r"Cell[_ -]?(\d+)", cell_id, re.IGNORECASE)
    source_group = re.search(r"TPG(\d+)", source_file, re.IGNORECASE)
    expected_group = re.match(r"G(\d+)-", cell_id, re.IGNORECASE)
    if not (source_cell and expected_cell and source_group and expected_group):
        raise ValueError(f"Cannot verify source identity for {source_file!r} and {cell_id!r}.")
    if source_cell.group(1) != expected_cell.group(1):
        raise ValueError(f"Source file cell identity differs from {cell_id!r}.")
    if source_group.group(1) != expected_group.group(1):
        raise ValueError(f"Source file group identity differs from {cell_id!r}.")


def _key(row: dict[str, str], *, part_field: str = "dataset_part") -> tuple[int, str, int]:
    part = _parse_part(_field(row, part_field, "part"))
    cell_id = _field(row, "physical_cell_id", "cell_id")
    if not cell_id:
        raise ValueError("Observation key is missing physical_cell_id.")
    try:
        rpt = int(_field(row, "rpt_index"))
    except ValueError as error:
        raise ValueError(f"Invalid RPT index for {cell_id!r}.") from error
    if rpt <= 0:
        raise ValueError(f"RPT index must be positive for {cell_id!r}.")
    return part, cell_id, rpt


def _bool(value: str) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _normalized_path(value: str) -> str:
    return str(value).strip().replace("\\", "/").casefold()


def _safe_project_path(project_root: Path, relative_path: str) -> Path:
    normalized = str(relative_path).strip().replace("\\", "/")
    if not normalized:
        raise ValueError("Source path is empty.")
    root = project_root.resolve()
    candidate = (root / normalized).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Source path escapes project root: {relative_path!r}.") from error
    return candidate


def _source_is_present(project_root: Path, source_file: str) -> bool:
    if not source_file or ";" in source_file or "," in source_file:
        return False
    return _safe_project_path(project_root, source_file).is_file()


def _source_from_segment(reference: str) -> str:
    text = str(reference).strip()
    if "#Step" in text:
        return text.split("#Step", 1)[0]
    if ":rows" in text:
        return text.split(":rows", 1)[0]
    raise ValueError(f"Unrecognized segment reference: {reference!r}.")


def _canonical_source_path(source: str, part: int) -> str:
    normalized = str(source).strip().replace("\\", "/")
    if normalized.casefold().startswith("data/raw/oxford/part"):
        return normalized
    return f"data/raw/oxford/part{part}/{normalized.lstrip('/')}"


def _segment_reference_key(reference: str, expected_step: int) -> tuple[str, int, int, int]:
    text = str(reference).strip()
    if "#Step" in text:
        source, step, start, end = raw._parse_segment(text)
        if step != expected_step:
            raise ValueError(f"Unexpected mapped step in segment reference: {reference!r}.")
        return _normalized_path(source), step, start, end
    match = re.fullmatch(r"(?P<source>.+):rows(?P<start>\d+)-(?P<end>\d+)", text)
    if match is None:
        raise ValueError(f"Unrecognized audit segment reference: {reference!r}.")
    return (
        _normalized_path(match.group("source")),
        expected_step,
        int(match.group("start")),
        int(match.group("end")),
    )


def _assert_sha256(value: str, description: str) -> None:
    if not _SHA256_RE.fullmatch(str(value).strip()):
        raise ValueError(f"{description} is missing a valid SHA-256.")


def _load_split_assignments(project_root: Path) -> tuple[dict[str, str], dict[str, int]]:
    split_path = (
        project_root
        / "reports"
        / "bafuse_final_data_validation_v2"
        / "cell_split_12train_4val_12external.csv"
    )
    rows = _read_csv(split_path)
    assignments: dict[str, str] = {}
    cell_counts: Counter[str] = Counter()
    for row in rows:
        cell_id = _field(row, "physical_cell_id")
        partition = _field(row, "partition").casefold()
        if not cell_id or partition not in EXPECTED_PARTITION_CELL_COUNTS:
            raise ValueError(f"Invalid Final2 split row: {row}.")
        if cell_id in assignments:
            raise ValueError(f"Physical cell appears more than once in split: {cell_id}.")
        assignments[cell_id] = partition
        cell_counts[partition] += 1
    if dict(cell_counts) != EXPECTED_PARTITION_CELL_COUNTS:
        raise ValueError(
            "Final2 split must contain 12 Train, 4 Validation, and 12 External cells."
        )
    return assignments, dict(cell_counts)


def ensure_cell_disjoint(cell_ids_by_partition: dict[str, list[str]]) -> None:
    partitions = sorted(cell_ids_by_partition)
    for index, first in enumerate(partitions):
        first_cells = set(cell_ids_by_partition[first])
        for second in partitions[index + 1 :]:
            overlap = first_cells & set(cell_ids_by_partition[second])
            if overlap:
                raise ValueError(
                    f"Cell leakage detected between {first} and {second}: "
                    f"{sorted(overlap)}."
                )


def _load_p5_records(
    project_root: Path,
) -> dict[tuple[int, str, int], dict[str, Any]]:
    p5_directory = (
        project_root
        / "experiments"
        / "soh_forecast"
        / "bafuse_architecture_benchmark_v1"
        / "bafuse_p5_electrical_v1_20261114"
    )
    split_rows = _read_csv(p5_directory / "bafuse_p5_split_manifest.csv")
    label_rows = _read_csv(p5_directory / "bafuse_soh_labels_frozen.csv")
    feature_rows = _read_csv(p5_directory / "bafuse_p5_feature_audit.csv")
    schema = _read_json(p5_directory / "bafuse_p5_feature_schema.json")
    preprocessing = _read_json(p5_directory / "bafuse_p5_preprocessing_config.json")

    if schema.get("label", {}).get("definition") != P5_LABEL_DEFINITION:
        raise ValueError("Frozen P5 label definition differs from the approved formula.")
    if tuple(schema.get("modalities", {}).get("X_physics", {}).get("feature_names_in_order", [])) != PHYSICS_FEATURE_NAMES:
        raise ValueError("Frozen P5 physics feature order does not match the P5 schema.")
    if preprocessing.get("feature_names_in_order") != list(PHYSICS_FEATURE_NAMES):
        raise ValueError("Frozen P5 preprocessing feature order has changed.")
    if preprocessing.get("fit_partition") != "train":
        raise ValueError("Frozen P5 physics preprocessing was not Train-fitted.")
    if preprocessing.get("external_statistics_used_for_selection") is not False:
        raise ValueError("Frozen P5 preprocessing used external statistics.")

    labels = {_key(row): row for row in label_rows}
    features = {_key(row): row for row in feature_rows}
    if len(labels) != len(label_rows) or len(features) != len(feature_rows):
        raise ValueError("Frozen P5 label or feature audit contains duplicate keys.")
    if len(split_rows) != 203 or len(labels) != 203 or len(features) != 203:
        raise ValueError("Expected exactly 203 frozen P5 labeled observations.")

    partition_records: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[int, str, int]] = set()
    for row in split_rows:
        key = _key(row)
        if key in seen:
            raise ValueError(f"Duplicate frozen P5 observation: {key}.")
        seen.add(key)
        partition = _field(row, "partition").casefold()
        if partition not in P5_PARTITION_FILES:
            raise ValueError(f"Unknown frozen P5 partition: {partition!r}.")
        partition_records[partition].append(row)

    curve_arrays: dict[str, dict[str, np.ndarray]] = {}
    for partition, file_name in P5_PARTITION_FILES.items():
        archive_path = _safe_project_path(
            project_root,
            str(
                _read_json(p5_directory / "bafuse_p5_sha256_manifest.json")[
                    "outputs"
                ][file_name]["path"]
            ),
        )
        with np.load(archive_path, allow_pickle=False) as archive:
            arrays = {
                name: np.asarray(archive[name], dtype=np.float32)
                for name in ("X_discharge", "X_charge", "X_physics", "y_soh")
            }
        expected_count = len(partition_records[partition])
        if arrays["X_discharge"].shape != (expected_count, 128):
            raise ValueError(f"Frozen P5 {partition} discharge shape is invalid.")
        if arrays["X_charge"].shape != (expected_count, 128):
            raise ValueError(f"Frozen P5 {partition} charge shape is invalid.")
        if arrays["X_physics"].shape != (expected_count, 5):
            raise ValueError(f"Frozen P5 {partition} physics shape is invalid.")
        if arrays["y_soh"].shape != (expected_count, 1):
            raise ValueError(f"Frozen P5 {partition} target shape is invalid.")
        if not all(np.isfinite(values).all() for values in arrays.values()):
            raise ValueError(f"Frozen P5 {partition} contains non-finite values.")
        curve_arrays[partition] = arrays

    records: dict[tuple[int, str, int], dict[str, Any]] = {}
    for partition, rows in partition_records.items():
        ordered_rows = sorted(
            rows, key=lambda row: int(_field(row, "split_row_index", default="0"))
        )
        arrays = curve_arrays[partition]
        for index, split_row in enumerate(ordered_rows):
            key = _key(split_row)
            label = labels.get(key)
            feature = features.get(key)
            if label is None or feature is None:
                raise ValueError(f"Frozen P5 inputs are not aligned for {key}.")
            if _field(label, "final_status").upper() != "ACCEPTED":
                raise ValueError(f"Frozen P5 label is not accepted for {key}.")
            if _bool(_field(label, "training_authorized")):
                raise ValueError("Historical P5 artifacts unexpectedly authorize training.")
            if _field(split_row, "label_status").upper() != "ACCEPTED":
                raise ValueError(f"P5 split manifest label status is not accepted for {key}.")
            if _bool(_field(split_row, "training_authorized")):
                raise ValueError("P5 split manifest unexpectedly authorizes training.")
            if _field(label, "target_definition") != P5_LABEL_DEFINITION:
                raise ValueError(f"P5 target definition differs for {key}.")
            capacity_method = _field(label, "capacity_method")
            if "trapezoidal integral" not in capacity_method.casefold():
                raise ValueError(f"P5 capacity method is not the verified integral for {key}.")

            group = _parse_group(key[1])
            expected_bol_part = expected_bol_source_part(key[0], group)
            bol_part_value = _field(label, "bol_source_part")
            if expected_bol_part is None or int(bol_part_value) != expected_bol_part:
                raise ValueError(f"P5 BoL source part is inconsistent for {key}.")
            if _field(label, "bol_resolution_status").casefold() != "resolved":
                raise ValueError(f"P5 BoL is not resolved for {key}.")
            bol_file = _field(label, "bol_step4_source_file")
            bol_reference = _field(label, "bol_step4_segment_ref")
            if _normalized_path(bol_file) != _normalized_path(
                _canonical_source_path(
                    _source_from_segment(bol_reference), expected_bol_part
                )
            ):
                raise ValueError(f"P5 BoL segment/source mismatch for {key}.")
            _validate_source_cell(bol_file, key[1])
            _assert_sha256(
                _field(label, "bol_step4_block_sha256"), f"{key} BoL Step 4 block"
            )
            bol_source_hash = _field(label, "bol_source_file_sha256")
            if bol_source_hash:
                _assert_sha256(bol_source_hash, f"{key} BoL source snapshot")

            q_rpt = float(_field(label, "q_rpt_step4_testtime_native_current_time"))
            q_bol = float(_field(label, "q_bol_step4_testtime_native_current_time"))
            target = float(_field(label, "soh_percent"))
            expected_target = compute_soh_percent(q_rpt, q_bol)
            if not math.isclose(target, expected_target, rel_tol=0.0, abs_tol=1e-10):
                raise ValueError(f"Frozen P5 target fails its capacity ratio for {key}.")
            if not math.isclose(
                float(_field(feature, "soh_percent_for_alignment_audit_only")),
                target,
                rel_tol=0.0,
                abs_tol=1e-7,
            ):
                raise ValueError(f"P5 target and feature audit are misaligned for {key}.")

            step4_file = _field(feature, "step4_source_file")
            step10_file = _field(feature, "step10_source_file")
            if not (
                _normalized_path(step4_file)
                == _normalized_path(_field(label, "rpt_step4_source_file"))
                == _normalized_path(_field(split_row, "rpt_source_file"))
            ):
                raise ValueError(f"P5 Step 4 source references disagree for {key}.")
            if _normalized_path(step10_file) != _normalized_path(
                _source_from_segment(_field(feature, "step10_segment_ref"))
            ):
                raise ValueError(f"P5 Step 10 segment/source mismatch for {key}.")
            if _normalized_path(step4_file) != _normalized_path(
                _source_from_segment(_field(feature, "step4_segment_ref"))
            ):
                raise ValueError(f"P5 Step 4 segment/source mismatch for {key}.")
            if _normalized_path(step10_file) != _normalized_path(step4_file):
                raise ValueError(f"P5 Step 4 and Step 10 are not from one RPT file for {key}.")

            if _bool(_field(feature, "predictive_temperature_feature")):
                raise ValueError(f"P5 feature audit marks temperature predictive for {key}.")
            if _bool(_field(feature, "predictive_label_or_bol_feature")):
                raise ValueError(f"P5 feature audit marks a target-derived feature for {key}.")
            if _bool(_field(feature, "future_rpt_feature")):
                raise ValueError(f"P5 feature audit marks a future-RPT feature for {key}.")
            if not _bool(_field(feature, "same_rpt_step_order_valid")):
                raise ValueError(f"P5 same-RPT step order is not verified for {key}.")

            _assert_sha256(_field(feature, "step4_block_sha256"), f"{key} Step 4 block")
            _assert_sha256(_field(feature, "step10_block_sha256"), f"{key} Step 10 block")
            step4_snapshot_hash = _field(feature, "step4_source_file_sha256_snapshot")
            step10_snapshot_hash = _field(feature, "step10_source_file_sha256_snapshot")
            _assert_sha256(step4_snapshot_hash, f"{key} Step 4 source snapshot")
            _assert_sha256(step10_snapshot_hash, f"{key} Step 10 source snapshot")
            if not _bool(_field(feature, "source_file_available", default="true")):
                raise ValueError(f"P5 source file is marked unavailable for {key}.")
            source_available = all(
                _safe_project_path(project_root, source).is_file()
                for source in (step4_file, step10_file, bol_file)
            )
            rpt_source_available = all(
                _safe_project_path(project_root, source).is_file()
                for source in (step4_file, step10_file)
            )
            bol_source_available = _safe_project_path(project_root, bol_file).is_file()

            raw_physics = np.asarray(
                [float(_field(feature, column)) for column in P5_RAW_FEATURE_COLUMNS],
                dtype=np.float32,
            )
            if not np.isfinite(raw_physics).all():
                raise ValueError(f"P5 raw electrical features are non-finite for {key}.")
            if not np.isclose(arrays["y_soh"][index, 0], target, atol=1e-5, rtol=0):
                raise ValueError(f"Frozen P5 tensor target differs from label for {key}.")

            records[key] = {
                "key": key,
                "label_source": "frozen_p5",
                "original_partition": partition,
                "source_file": step4_file,
                "step4_segment_ref": _field(feature, "step4_segment_ref"),
                "step10_segment_ref": _field(feature, "step10_segment_ref"),
                "step4_block_sha256": _field(feature, "step4_block_sha256"),
                "step10_block_sha256": _field(feature, "step10_block_sha256"),
                "step4_source_sha256": step4_snapshot_hash,
                "step10_source_sha256": step10_snapshot_hash,
                "bol_source_file": bol_file,
                "bol_segment_ref": bol_reference,
                "bol_block_sha256": _field(label, "bol_step4_block_sha256"),
                "bol_source_sha256": bol_source_hash,
                "rpt_source_available": rpt_source_available,
                "bol_source_available": bol_source_available,
                "bol_source_part": expected_bol_part,
                "target": target,
                "target_definition": P5_LABEL_DEFINITION,
                "capacity_method": capacity_method,
                "physics_raw": raw_physics,
                "X_discharge": arrays["X_discharge"][index],
                "X_charge": arrays["X_charge"][index],
                "source_available": source_available,
            }

    if len(records) != 203:
        raise ValueError(f"Expected 203 aligned frozen P5 observations, got {len(records)}.")
    return records


def _hash_raw_segments(
    path: Path, ranges: list[tuple[int, int]]
) -> dict[tuple[int, int], str]:
    unique_ranges = sorted(set(ranges))
    if not unique_ranges or any(start < 2 or end < start for start, end in unique_ranges):
        raise ValueError(f"Invalid mapped row ranges for {path}.")
    digests = {row_range: hashlib.sha256() for row_range in unique_ranges}
    row_counts = {row_range: 0 for row_range in unique_ranges}
    max_end = max(end for _, end in unique_ranges)
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        if not next(reader, None):
            raise ValueError(f"Mapped raw source has no header: {path}.")
        for row in reader:
            line_number = reader.line_num
            if line_number > max_end:
                break
            for row_range in unique_ranges:
                start, end = row_range
                if start <= line_number <= end:
                    digests[row_range].update(
                        ("\x1f".join(row) + "\n").encode("utf-8", errors="replace")
                    )
                    row_counts[row_range] += 1
    for row_range in unique_ranges:
        start, end = row_range
        if row_counts[row_range] != end - start + 1:
            raise ValueError(
                f"Mapped segment {start}-{end} has {row_counts[row_range]} rows "
                f"instead of {end - start + 1}: {path}."
            )
    return {row_range: digest.hexdigest() for row_range, digest in digests.items()}


def _filter_zero_current_step10(
    block: dict[str, Any],
    *,
    reference: str,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    current = np.asarray(block["current_values"], dtype=np.float64)
    voltage = np.asarray(block["voltage_values"], dtype=np.float64)
    capacity = np.asarray(block["capacity_values"], dtype=np.float64)
    testtime = np.asarray(block["testtime_values"], dtype=np.float64)
    steptime = np.asarray(block["steptime_values"], dtype=np.float64)
    arrays = (current, voltage, capacity, testtime, steptime)
    reasons: list[str] = []
    if any(values.size != block["row_count"] for values in arrays):
        reasons.append("step10_audit_arrays_do_not_match_raw_row_count")
    elif not all(np.isfinite(values).all() for values in arrays):
        reasons.append("step10_audit_values_contain_non_finite_samples")

    zero_mask = current == 0
    positive_mask = current > 0
    if np.any(current < 0):
        reasons.append("negative_current_rows_present")
    if not np.any(positive_mask):
        reasons.append("no_positive_current_rows")

    joined_intervals: list[dict[str, Any]] = []
    if not reasons:
        raw_testtime_delta = np.diff(testtime)
        raw_steptime_delta = np.diff(steptime)
        filtered_testtime = testtime[positive_mask]
        filtered_steptime = steptime[positive_mask]
        filtered_capacity = capacity[positive_mask]
        filtered_testtime_delta = np.diff(filtered_testtime)
        filtered_steptime_delta = np.diff(filtered_steptime)
        if np.any(raw_testtime_delta <= 0) or np.any(raw_steptime_delta <= 0):
            reasons.append("raw_step10_timestamps_not_strictly_increasing")
        if (
            np.any(filtered_testtime_delta <= 0)
            or np.any(filtered_steptime_delta <= 0)
        ):
            reasons.append("filtered_step10_timestamps_not_strictly_increasing")
        if np.any(np.diff(filtered_capacity) <= 0):
            reasons.append("filtered_step10_capacity_not_strictly_increasing")
        for timestamp_name, raw_delta, filtered_delta in (
            ("TestTime", raw_testtime_delta, filtered_testtime_delta),
            ("StepTime", raw_steptime_delta, filtered_steptime_delta),
        ):
            if raw_delta.size and filtered_delta.size:
                tolerance = np.finfo(np.float64).eps * max(
                    1.0, float(np.max(raw_delta))
                ) * 8
                if float(np.max(filtered_delta)) > float(np.max(raw_delta)) + tolerance:
                    reasons.append(f"filtered_{timestamp_name}_gap_exceeds_raw_max")

        zero_indices = np.flatnonzero(zero_mask)
        group_start = 0
        while group_start < len(zero_indices):
            group_end = group_start
            while (
                group_end + 1 < len(zero_indices)
                and zero_indices[group_end + 1] == zero_indices[group_end] + 1
            ):
                group_end += 1
            left = int(zero_indices[group_start]) - 1
            right = int(zero_indices[group_end]) + 1
            if left >= 0 and right < len(testtime):
                joined_intervals.append(
                    {
                        "left_line": block["start_line"] + left,
                        "right_line": block["start_line"] + right,
                        "testtime_delta": float(testtime[right] - testtime[left]),
                        "steptime_delta": float(steptime[right] - steptime[left]),
                    }
                )
            group_start = group_end + 1

    zero_rows = block["zero_current_rows"]
    audit = {
        "reference": reference,
        "raw_block_sha256": block["raw_block_sha256"],
        "raw_row_count": int(block["row_count"]),
        "positive_current_row_count": int(np.count_nonzero(positive_mask)),
        "zero_current_excluded_count": int(np.count_nonzero(zero_mask)),
        "negative_current_row_count": int(np.count_nonzero(current < 0)),
        "zero_current_rows": zero_rows,
        "joined_intervals_across_excluded_rows": joined_intervals,
        "raw_testtime_delta_max": (
            float(np.max(np.diff(testtime))) if testtime.size > 1 else None
        ),
        "filtered_testtime_delta_max": (
            float(np.max(np.diff(testtime[positive_mask])))
            if np.count_nonzero(positive_mask) > 1
            else None
        ),
        "raw_steptime_delta_max": (
            float(np.max(np.diff(steptime))) if steptime.size > 1 else None
        ),
        "filtered_steptime_delta_max": (
            float(np.max(np.diff(steptime[positive_mask])))
            if np.count_nonzero(positive_mask) > 1
            else None
        ),
        "filtered_capacity_min": (
            float(np.min(capacity[positive_mask]))
            if positive_mask.size == capacity.size and np.any(positive_mask)
            else None
        ),
        "filtered_capacity_max": (
            float(np.max(capacity[positive_mask]))
            if positive_mask.size == capacity.size and np.any(positive_mask)
            else None
        ),
        "positive_current_rows_sha256": block["positive_current_rows_sha256"],
        "retained_voltage_decrease_count": (
            int(np.count_nonzero(np.diff(voltage[positive_mask]) < 0))
            if voltage.size == current.size and np.any(positive_mask)
            else None
        ),
        "reasons": reasons,
    }
    if reasons:
        return None, audit
    if block["positive_current_row_count"] != audit["positive_current_row_count"]:
        audit["reasons"].append("positive_current_row_count_mismatch")
        return None, audit
    return voltage[positive_mask], audit


def _build_g10_cell2_p5_records(
    project_root: Path,
    mapping_rows: list[dict[str, str]],
    *,
    step10_zero_current_policy: str = "strict",
    training_authorized: bool = False,
    external_inference_authorized: bool = False,
) -> tuple[
    dict[tuple[int, str, int], dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[int, str, int], dict[str, Any]],
    list[dict[str, Any]],
]:
    if step10_zero_current_policy not in {"strict", "exclude_zero_current"}:
        raise ValueError(
            "Unsupported Step 10 zero-current policy: "
            f"{step10_zero_current_policy!r}."
        )
    if project_root.resolve() != raw.PROJECT_ROOT.resolve():
        raise ValueError("G10-Cell2 extraction requires the active repository root.")
    canonical_path = project_root / "data" / "manifests" / "bafuse_rpt_observations.csv"
    canonical_rows = _read_csv(canonical_path)
    cell_rows = [
        row
        for row in canonical_rows
        if row.get("part") == "3" and row.get("physical_cell_id") == "G10-Cell2"
    ]
    cell_rows.sort(key=lambda row: int(row["rpt_index"]))
    if len(cell_rows) != len(G10_CELL2_TPG_BY_RPT):
        raise ValueError("Canonical G10-Cell2 manifest must contain exactly ten RPTs.")

    audit_by_key = {
        (int(row["part"]), row["physical_cell_id"], int(row["rpt_index"])): row
        for row in mapping_rows
    }
    if len(audit_by_key) != len(mapping_rows):
        raise ValueError("Final2 mapping audit contains duplicate observation keys.")

    g10_audit_rows: dict[int, dict[str, str]] = {}
    for row in cell_rows:
        rpt = int(row["rpt_index"])
        if G10_CELL2_TPG_BY_RPT.get(rpt) != row["source_tpg"]:
            raise ValueError(f"Unexpected TPG mapping for G10-Cell2 RPT{rpt}.")
        if row.get("mapping_status") != "GUIDE_VERIFIED":
            raise ValueError(f"G10-Cell2 RPT{rpt} is not marked GUIDE_VERIFIED.")
        if row.get("split") != "external_test":
            raise ValueError(f"G10-Cell2 RPT{rpt} is not assigned to External Test.")
        if row.get("audit_notes") != G10_CELL2_PROVENANCE:
            raise ValueError(f"G10-Cell2 RPT{rpt} is missing the required provenance note.")
        if row.get("step4_status") != "verified" or row.get("step10_status") != "verified":
            raise ValueError(f"G10-Cell2 RPT{rpt} lacks verified Step 4/Step 10 boundaries.")
        audit = audit_by_key.get((3, "G10-Cell2", rpt))
        if audit is None:
            raise ValueError(f"Final2 mapping audit is missing G10-Cell2 RPT{rpt}.")
        if (
            audit.get("final_mapping_status") != "GUIDE_VERIFIED"
            or not _bool(audit.get("source_exists", ""))
            or not _bool(audit.get("step4_valid", ""))
            or not _bool(audit.get("step10_valid", ""))
        ):
            raise ValueError(f"G10-Cell2 RPT{rpt} failed mapping or curve eligibility.")
        if _normalized_path(row["source_file"]) != _normalized_path(audit["source_file"]):
            raise ValueError(f"G10-Cell2 RPT{rpt} source file differs between audits.")
        for manifest_field, audit_field, step in (
            ("step4_segment_ref", "step4_segment_reference", 4),
            ("step10_segment_ref", "step10_segment_reference", 10),
        ):
            if _segment_reference_key(row[manifest_field], step) != _segment_reference_key(
                audit[audit_field], step
            ):
                raise ValueError(
                    f"G10-Cell2 RPT{rpt} {manifest_field} differs between audits."
                )
        g10_audit_rows[rpt] = audit

    if [int(row["rpt_index"]) for row in cell_rows] != list(G10_CELL2_TPG_BY_RPT):
        raise ValueError("G10-Cell2 RPT chronology is not RPT1 through RPT10.")
    if len({row["source_file"] for row in cell_rows}) != len(cell_rows):
        raise ValueError("G10-Cell2 RPT source files are not unique.")

    bol_reference = cell_rows[0]["bol_step4_segment_ref"]
    bol_source_file = _canonical_source_path(_source_from_segment(bol_reference), 3)
    if any(row["bol_step4_segment_ref"] != bol_reference for row in cell_rows):
        raise ValueError("G10-Cell2 RPT rows do not share one Step 4 BoL reference.")
    if any(row["bol_reference_file"] != cell_rows[0]["bol_reference_file"] for row in cell_rows):
        raise ValueError("G10-Cell2 RPT rows do not share one BoL source file.")
    if _normalized_path(bol_source_file) != _normalized_path(
        _canonical_source_path(cell_rows[0]["bol_reference_file"], 3)
    ):
        raise ValueError("G10-Cell2 BoL source path differs from its segment reference.")
    _validate_source_cell(bol_source_file, "G10-Cell2")

    source_files = {bol_source_file, *(row["source_file"] for row in cell_rows)}
    source_manifest_path = (
        project_root / "reports" / "bafuse_final_data_validation_v2" / "run_manifest.json"
    )
    expected_source_hashes = _read_json(source_manifest_path).get("input_sha256", {})
    source_hashes: dict[str, dict[str, Any]] = {}
    for source_file in sorted(source_files):
        expected_hash = str(expected_source_hashes.get(source_file, "")).lower()
        if not _SHA256_RE.fullmatch(expected_hash):
            raise ValueError(f"Saved source hash is missing for {source_file}.")
        source_path = _safe_project_path(project_root, source_file)
        actual_hash = _sha256(source_path)
        if actual_hash != expected_hash:
            raise ValueError(f"Current source hash differs from the saved audit: {source_file}.")
        source_hashes[source_file] = {
            "sha256": actual_hash,
            "size_bytes": source_path.stat().st_size,
        }

    block_specs: dict[
        tuple[str, int, int, int, int], dict[str, Any]
    ] = {}
    for row in cell_rows:
        for field_name, expected_step, keep_physics in (
            ("step4_segment_ref", 4, True),
            ("step10_segment_ref", 10, False),
        ):
            source, step, start, end = raw._parse_segment(row[field_name])
            source = raw._resolve_source_name(source, 3)
            if step != expected_step or _normalized_path(source) != _normalized_path(
                row["source_file"]
            ):
                raise ValueError(f"G10-Cell2 RPT{row['rpt_index']} has an invalid {field_name}.")
            key = raw._block_key(source, 3, step, start, end)
            block_specs[key] = {
                "source_file": source,
                "step": step,
                "start": start,
                "end": end,
                "keep_physics": keep_physics,
                "keep_current": (
                    expected_step == 10
                    and step10_zero_current_policy == "exclude_zero_current"
                ),
                "reference": row[field_name],
            }

    parsed_bol_source, bol_step, bol_start, bol_end = raw._parse_segment(bol_reference)
    bol_source = raw._resolve_source_name(parsed_bol_source, 3)
    if bol_step != 4 or _normalized_path(bol_source) != _normalized_path(bol_source_file):
        raise ValueError("G10-Cell2 BoL reference is not a Part 3 Step 4 segment.")
    bol_key = raw._block_key(bol_source, 3, bol_step, bol_start, bol_end)
    block_specs[bol_key] = {
        "source_file": bol_source,
        "step": bol_step,
        "start": bol_start,
        "end": bol_end,
        "keep_physics": False,
        "keep_current": False,
        "reference": bol_reference,
    }

    ranges_by_source: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for spec in block_specs.values():
        ranges_by_source[spec["source_file"]].append((spec["start"], spec["end"]))
    block_requests: dict[tuple[str, int, int, int, int], raw.BlockRequest] = {}
    block_audit: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for source_file, ranges in ranges_by_source.items():
        source_path = _safe_project_path(project_root, source_file)
        before = source_path.stat()
        block_hashes = _hash_raw_segments(source_path, ranges)
        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Raw source changed while hashing mapped blocks: {source_file}.")
        for key, spec in block_specs.items():
            if spec["source_file"] != source_file:
                continue
            block_hash = block_hashes[(spec["start"], spec["end"])]
            block_requests[key] = raw.BlockRequest(
                source_file=source_file,
                source_part=3,
                step=spec["step"],
                start_line=spec["start"],
                end_line=spec["end"],
                expected_sha256=block_hash,
                keep_voltage=True,
                keep_physics=spec["keep_physics"],
                keep_current=spec["keep_current"],
            )
            block_audit[key] = {
                "source_file": source_file,
                "source_part": "3",
                "expected_step": str(spec["step"]),
                "start_line": str(spec["start"]),
                "end_line": str(spec["end"]),
                "raw_block_sha256": block_hash,
                "source_size_bytes": str(before.st_size),
                "source_mtime_ns": str(before.st_mtime_ns),
            }

    raw_blocks = raw._scan_raw_blocks(block_requests, block_audit)
    curve_eligibility: dict[tuple[int, str, int], dict[str, Any]] = {}
    eligibility_rows: list[dict[str, Any]] = []
    block_validity: dict[tuple[str, int, int, int, int], tuple[bool, list[str]]] = {}
    filtered_step10_voltage: dict[tuple[str, int, int, int, int], np.ndarray] = {}
    step10_zero_audits: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    for key, spec in block_specs.items():
        block = raw_blocks[key]
        current_sign = "negative" if spec["step"] == 4 else "positive"
        allow_zero_current = (
            spec["step"] == 10
            and step10_zero_current_policy == "exclude_zero_current"
        )
        valid, reasons = raw._block_validity(
            block,
            expected_step=spec["step"],
            expected_current_sign=current_sign,
            allow_zero_current=allow_zero_current,
        )
        if allow_zero_current:
            filtered_voltage, zero_audit = _filter_zero_current_step10(
                block,
                reference=spec["reference"],
            )
            step10_zero_audits[key] = zero_audit
            if filtered_voltage is None:
                valid = False
                reasons.extend(zero_audit["reasons"])
            else:
                filtered_step10_voltage[key] = filtered_voltage
        block_validity[key] = (valid, reasons)
        if valid and spec["step"] == 4 and block["voltage_first_v"] <= block["voltage_last_v"]:
            valid = False
            reasons = ["step4_voltage_not_decreasing"]
            block_validity[key] = (valid, reasons)
        step10_voltage = filtered_step10_voltage.get(
            key, np.asarray(block["voltage_values"], dtype=np.float64)
        )
        if valid and spec["step"] == 10 and step10_voltage[0] >= step10_voltage[-1]:
            valid = False
            reasons = ["step10_voltage_not_increasing"]
            block_validity[key] = (valid, reasons)
        if not valid and key == bol_key:
            raise ValueError(
                f"Mapped block failed Step {spec['step']} checks at "
                f"{spec['reference']}: {', '.join(reasons)}."
            )

    for row in cell_rows:
        rpt = int(row["rpt_index"])
        key = (3, "G10-Cell2", rpt)
        step4_source, step4_step, step4_start, step4_end = raw._parse_segment(
            row["step4_segment_ref"]
        )
        step10_source, step10_step, step10_start, step10_end = raw._parse_segment(
            row["step10_segment_ref"]
        )
        step4_key = raw._block_key(
            raw._resolve_source_name(step4_source, 3),
            3,
            step4_step,
            step4_start,
            step4_end,
        )
        step10_key = raw._block_key(
            raw._resolve_source_name(step10_source, 3),
            3,
            step10_step,
            step10_start,
            step10_end,
        )
        step4_valid, step4_reasons = block_validity[step4_key]
        step10_valid, step10_reasons = block_validity[step10_key]
        curve_eligibility[key] = {
            "step4_valid": step4_valid,
            "step10_valid": step10_valid,
            "step4_reasons": step4_reasons,
            "step10_reasons": step10_reasons,
            "step10_zero_current_audit": step10_zero_audits.get(step10_key),
        }
        eligibility_rows.append(
            {
                "physical_cell_id": "G10-Cell2",
                "rpt_index": rpt,
                "source_tpg": row["source_tpg"],
                "source_file": row["source_file"],
                "mapping_status": "GUIDE_VERIFIED",
                "step4_segment_reference": row["step4_segment_ref"],
                "step4_block_sha256": raw_blocks[step4_key]["raw_block_sha256"],
                "step4_eligible": step4_valid,
                "step4_reasons": ";".join(step4_reasons),
                "step10_segment_reference": row["step10_segment_ref"],
                "step10_block_sha256": raw_blocks[step10_key]["raw_block_sha256"],
                "step10_eligible": step10_valid,
                "step10_reasons": ";".join(step10_reasons),
                "current_sign_counts": json.dumps(
                    raw_blocks[step10_key]["current_signs"], sort_keys=True
                ),
                "step10_zero_current_policy": step10_zero_current_policy,
                "step10_zero_current_audit": json.dumps(
                    step10_zero_audits.get(step10_key, {}),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "included_as_labeled_observation": step4_valid and step10_valid,
                "identity_provenance": G10_CELL2_PROVENANCE,
            }
        )

    if tuple(P5_FEATURE_NAMES) != PHYSICS_FEATURE_NAMES:
        raise ValueError("P5 electrical feature order differs from the Final2 input contract.")
    bol_block = raw_blocks[bol_key]
    q_bol = abs(float(bol_block["testtime_integral_native_current_time"]))
    if not math.isfinite(q_bol) or q_bol <= 0:
        raise ValueError("G10-Cell2 BoL Step 4 capacity integral is invalid.")

    records: dict[tuple[int, str, int], dict[str, Any]] = {}
    extension_rows: list[dict[str, Any]] = []
    capacity_method = (
        "P5 trapezoidal integral of abs(Amps) against native TestTime "
        "within the verified Step 4 segment"
    )
    for row in cell_rows:
        rpt = int(row["rpt_index"])
        key = (3, "G10-Cell2", rpt)
        mapping = g10_audit_rows[rpt]
        step4_source, step4_step, step4_start, step4_end = raw._parse_segment(
            row["step4_segment_ref"]
        )
        step10_source, step10_step, step10_start, step10_end = raw._parse_segment(
            row["step10_segment_ref"]
        )
        step4_key = raw._block_key(
            raw._resolve_source_name(step4_source, 3), 3, step4_step, step4_start, step4_end
        )
        step10_key = raw._block_key(
            raw._resolve_source_name(step10_source, 3),
            3,
            step10_step,
            step10_start,
            step10_end,
        )
        step4 = raw_blocks[step4_key]
        step10 = raw_blocks[step10_key]
        if not (
            curve_eligibility[key]["step4_valid"]
            and curve_eligibility[key]["step10_valid"]
        ):
            continue
        q_rpt = abs(float(step4["testtime_integral_native_current_time"]))
        target = compute_soh_percent(q_rpt, q_bol)
        x_discharge = np.asarray(
            resample_curve(step4["voltage_values"], n_points=128), dtype=np.float32
        )
        step10_voltage = filtered_step10_voltage.get(
            step10_key, np.asarray(step10["voltage_values"], dtype=np.float64)
        )
        x_charge = np.asarray(
            resample_curve(step10_voltage, n_points=128), dtype=np.float32
        )
        physics = build_p5_electrical_features(
            step4["voltage_values"], step4["current_values"]
        )
        physics_raw = np.asarray(
            [physics[name] for name in PHYSICS_FEATURE_NAMES], dtype=np.float32
        )
        if not np.isfinite(np.concatenate((x_discharge, x_charge, physics_raw))).all():
            raise ValueError(f"G10-Cell2 RPT{rpt} produced non-finite P5 features.")

        rpt_source = row["source_file"]
        rpt_hash = source_hashes[rpt_source]["sha256"]
        bol_hash = source_hashes[bol_source]["sha256"]
        record = {
            "key": key,
            "label_source": "final2_g10_p5_extension",
            "original_partition": "external_test",
            "source_file": rpt_source,
            "step4_segment_ref": row["step4_segment_ref"],
            "step10_segment_ref": row["step10_segment_ref"],
            "step4_block_sha256": step4["raw_block_sha256"],
            "step10_block_sha256": step10["raw_block_sha256"],
            "step10_zero_current_policy": step10_zero_current_policy,
            "step4_source_sha256": rpt_hash,
            "step10_source_sha256": rpt_hash,
            "bol_source_file": bol_source,
            "bol_segment_ref": bol_reference,
            "bol_block_sha256": bol_block["raw_block_sha256"],
            "bol_source_sha256": bol_hash,
            "rpt_source_available": True,
            "bol_source_available": True,
            "bol_source_part": 3,
            "target": target,
            "target_definition": P5_LABEL_DEFINITION,
            "capacity_method": capacity_method,
            "physics_raw": physics_raw,
            "X_discharge": x_discharge,
            "X_charge": x_charge,
            "source_available": True,
        }
        records[key] = record
        extension_rows.append(
            {
                "dataset_part": 3,
                "group": 10,
                "physical_cell_id": "G10-Cell2",
                "rpt_index": rpt,
                "source_tpg": row["source_tpg"],
                "mapping_status": "GUIDE_VERIFIED",
                "final2_partition": "external_test",
                "source_file": rpt_source,
                "source_file_sha256": rpt_hash,
                "step4_segment_reference": row["step4_segment_ref"],
                "step4_block_sha256": step4["raw_block_sha256"],
                "step4_eligible": True,
                "step10_segment_reference": row["step10_segment_ref"],
                "step10_block_sha256": step10["raw_block_sha256"],
                "step10_zero_current_policy": step10_zero_current_policy,
                "step10_zero_current_excluded_count": step10_zero_audits.get(
                    step10_key, {}
                ).get("zero_current_excluded_count", 0),
                "step10_zero_current_rows": json.dumps(
                    step10_zero_audits.get(step10_key, {}).get("zero_current_rows", []),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "step10_filtered_positive_current_sha256": step10_zero_audits.get(
                    step10_key, {}
                ).get("positive_current_rows_sha256", ""),
                "step10_eligible": True,
                "bol_source_part": 3,
                "bol_source_file": bol_source,
                "bol_source_sha256": bol_hash,
                "bol_segment_reference": bol_reference,
                "bol_block_sha256": bol_block["raw_block_sha256"],
                "q_rpt_step4_testtime_native_current_time": q_rpt,
                "q_bol_step4_testtime_native_current_time": q_bol,
                "soh_percent": target,
                "target_definition": P5_LABEL_DEFINITION,
                "capacity_method": capacity_method,
                "identity_provenance": G10_CELL2_PROVENANCE,
                "label_source": "Final2-only P5 extension; frozen P5 artifacts unchanged",
                "training_authorized": training_authorized,
                "external_inference_authorized": external_inference_authorized,
            }
        )

    source_manifest = {
        "schema_version": 1,
        "source_hashes_checked_against": (
            "reports/bafuse_final_data_validation_v2/run_manifest.json"
        ),
        "source_files": source_hashes,
        "mapping_status": "GUIDE_VERIFIED",
        "identity_provenance": G10_CELL2_PROVENANCE,
        "step4_step10_validation_complete": True,
        "all_mapped_step4_step10_curves_eligible": all(
            item["step4_valid"] and item["step10_valid"]
            for item in curve_eligibility.values()
        ),
        "eligible_observation_count": len(records),
        "ineligible_observation_count": len(curve_eligibility) - len(records),
        "label_formula": P5_LABEL_DEFINITION,
        "step10_zero_current_policy": step10_zero_current_policy,
        "step10_zero_current_audit_by_rpt": {
            str(rpt): step10_zero_audits.get(
                raw._block_key(
                    raw._resolve_source_name(
                        raw._parse_segment(row["step10_segment_ref"])[0], 3
                    ),
                    3,
                    *raw._parse_segment(row["step10_segment_ref"])[1:],
                ),
                {},
            )
            for row in cell_rows
            for rpt in (int(row["rpt_index"]),)
        },
        "training_authorized": training_authorized,
        "external_inference_authorized": external_inference_authorized,
    }
    return records, extension_rows, source_manifest, curve_eligibility, eligibility_rows


def _candidate_status(
    row: dict[str, str],
    *,
    key: tuple[int, str, int],
    p5_records: dict[tuple[int, str, int], dict[str, Any]],
    project_root: Path,
    curve_eligibility: dict[tuple[int, str, int], dict[str, Any]] | None = None,
) -> tuple[str, str, str]:
    part, cell_id, _ = key
    group = _parse_group(cell_id, _field(row, "group"))
    source_file = _field(row, "source_file")
    split = _field(row, "partition", "split").casefold()
    if split == "external":
        split = "external_test"
    if split not in EXPECTED_PARTITION_CELL_COUNTS:
        split = ""

    source_exists_audit = _bool(_field(row, "source_exists"))
    if part == 2 and group in {5, 6}:
        return "REVIEW", "Part 2 Groups 5-6 are quarantined by policy", split
    mapping_status = _field(row, "final_mapping_status").casefold()
    if mapping_status in {"ambiguous_rpt", "needs_curve_audit"}:
        return "REVIEW", f"mapping status requires resolution: {mapping_status}", split
    if not source_exists_audit or not _source_is_present(project_root, source_file):
        return "REJECTED", "source file is absent from the current workspace", split
    actual_eligibility = (curve_eligibility or {}).get(key)
    if actual_eligibility is not None and not (
        actual_eligibility["step4_valid"] and actual_eligibility["step10_valid"]
    ):
        failed_steps = [
            f"Step {step}"
            for step in (4, 10)
            if not actual_eligibility[f"step{step}_valid"]
        ]
        failed_reasons = [
            reason
            for step in (4, 10)
            for reason in actual_eligibility[f"step{step}_reasons"]
        ]
        return (
            "REJECTED",
            f"raw mapped {' and '.join(failed_steps)} curve failed eligibility: "
            f"{', '.join(failed_reasons)}",
            split,
        )
    if not _bool(_field(row, "step4_valid")) or not _bool(_field(row, "step10_valid")):
        return "REJECTED", "required Step 4 or Step 10 curve is not valid", split

    p5_record = p5_records.get(key)
    if p5_record is not None:
        if _normalized_path(source_file) != _normalized_path(p5_record["source_file"]):
            return "REVIEW", "P5 key has a different mapping source file", split
        if not p5_record["source_available"]:
            return "REVIEW", "frozen P5 source file is missing from this workspace", split
        if p5_record.get("label_source") == "frozen_p5":
            return "ACCEPTED", "accepted frozen P5 label and verified feature provenance", split
        return "ACCEPTED", "accepted Final2-only P5 label extension with verified feature provenance", split

    return "REVIEW", "not in the frozen P5 labeled cohort; P5 label review required", split


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def prepare_final2_dataset(
    *,
    project_root: Path = PROJECT_ROOT,
    output_directory: Path = OUTPUT_DIRECTORY,
    dataset_identifier: str = "bafuse-final2-p5-g10-guide-verified-v1",
    step10_zero_current_policy: str = "strict",
    training_authorized: bool = False,
    external_inference_authorized: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    verified_artifacts = _verify_frozen_p5_artifacts(project_root)
    assignments, split_cells = _load_split_assignments(project_root)
    if assignments.get("G10-Cell2") != "external_test":
        raise ValueError("G10-Cell2 must remain in the External Test cell partition.")
    frozen_p5_records = _load_p5_records(project_root)
    mapping_audit_path = (
        project_root
        / "reports"
        / "final_mapping_label_audit"
        / "mapping_observation_audit.csv"
    )
    mapping_rows = _read_csv(mapping_audit_path)
    if len(mapping_rows) != 477:
        raise ValueError(f"Expected 477 Final2 mapping candidates, got {len(mapping_rows)}.")
    (
        extension_records,
        extension_rows,
        g10_source_manifest,
        g10_curve_eligibility,
        g10_eligibility_rows,
    ) = _build_g10_cell2_p5_records(
        project_root,
        mapping_rows,
        step10_zero_current_policy=step10_zero_current_policy,
        training_authorized=training_authorized,
        external_inference_authorized=external_inference_authorized,
    )
    if set(frozen_p5_records).intersection(extension_records):
        raise ValueError("G10-Cell2 extension duplicates a frozen P5 observation.")
    p5_records = {**frozen_p5_records, **extension_records}

    observations: list[dict[str, Any]] = []
    seen: set[tuple[int, str, int]] = set()
    accepted_by_partition: dict[str, list[dict[str, Any]]] = {
        partition: [] for partition in EXPECTED_PARTITION_CELL_COUNTS
    }
    status_counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()

    for row in mapping_rows:
        key = _key(row, part_field="part")
        if key in seen:
            raise ValueError(f"Final2 mapping audit contains duplicate keys: {key}.")
        seen.add(key)
        status, reason, partition = _candidate_status(
            row,
            key=key,
            p5_records=p5_records,
            project_root=project_root,
            curve_eligibility=g10_curve_eligibility,
        )
        if key[1] not in assignments:
            partition = ""
        else:
            partition = assignments[key[1]]
        status_counts[status] += 1
        reasons[reason] += 1
        p5_record = p5_records.get(key)
        observation = {
            "dataset_part": key[0],
            "group": _parse_group(key[1], _field(row, "group")),
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "source_tpg": _field(row, "source_tpg"),
            "source_file": _field(row, "source_file"),
            "guide_stage": _field(row, "guide_stage"),
            "mapping_status": _field(row, "final_mapping_status"),
            "identity_provenance": _field(row, "identity_provenance"),
            "final2_partition": partition,
            "step4_segment_reference": _field(row, "step4_segment_reference"),
            "step10_segment_reference": _field(row, "step10_segment_reference"),
            "step4_valid": str(
                g10_curve_eligibility[key]["step4_valid"]
                if key in g10_curve_eligibility
                else _bool(_field(row, "step4_valid"))
            ),
            "step10_valid": str(
                g10_curve_eligibility[key]["step10_valid"]
                if key in g10_curve_eligibility
                else _bool(_field(row, "step10_valid"))
            ),
            "source_exists_in_audit": _field(row, "source_exists"),
            "source_exists_now": str(
                _source_is_present(project_root, _field(row, "source_file"))
            ),
            "bol_source_exists_now": (
                str(p5_record["bol_source_available"])
                if p5_record is not None
                else ""
            ),
            "status": status,
            "status_reason": reason,
            "soh_percent": p5_record["target"] if status == "ACCEPTED" else "",
            "target_definition": (
                p5_record["target_definition"] if status == "ACCEPTED" else ""
            ),
            "capacity_method": p5_record["capacity_method"] if status == "ACCEPTED" else "",
            "step4_block_sha256": (
                p5_record["step4_block_sha256"] if status == "ACCEPTED" else ""
            ),
            "step10_block_sha256": (
                p5_record["step10_block_sha256"] if status == "ACCEPTED" else ""
            ),
            "step4_source_sha256_snapshot": (
                p5_record["step4_source_sha256"] if status == "ACCEPTED" else ""
            ),
            "step10_source_sha256_snapshot": (
                p5_record["step10_source_sha256"] if status == "ACCEPTED" else ""
            ),
            "bol_source_part": p5_record["bol_source_part"] if status == "ACCEPTED" else "",
            "bol_source_file": p5_record["bol_source_file"] if status == "ACCEPTED" else "",
            "bol_segment_reference": p5_record["bol_segment_ref"] if status == "ACCEPTED" else "",
            "bol_block_sha256": p5_record["bol_block_sha256"] if status == "ACCEPTED" else "",
            "bol_source_sha256": p5_record["bol_source_sha256"] if status == "ACCEPTED" else "",
        }
        observations.append(observation)
        if status == "ACCEPTED":
            accepted_by_partition[partition].append(p5_record)

    if (
        len(seen) != 477
        or len(frozen_p5_records) != 203
        or len(extension_records)
        != sum(
            item["step4_valid"] and item["step10_valid"]
            for item in g10_curve_eligibility.values()
        )
        or len(p5_records) != len(frozen_p5_records) + len(extension_records)
    ):
        raise ValueError("Final2 mapping, frozen P5, or G10-Cell2 extension inventory is incomplete.")

    extension_fields = [
        "dataset_part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "source_tpg",
        "mapping_status",
        "final2_partition",
        "source_file",
        "source_file_sha256",
        "step4_segment_reference",
        "step4_block_sha256",
        "step4_eligible",
        "step10_segment_reference",
        "step10_block_sha256",
        "step10_zero_current_policy",
        "step10_zero_current_excluded_count",
        "step10_zero_current_rows",
        "step10_filtered_positive_current_sha256",
        "step10_eligible",
        "bol_source_part",
        "bol_source_file",
        "bol_source_sha256",
        "bol_segment_reference",
        "bol_block_sha256",
        "q_rpt_step4_testtime_native_current_time",
        "q_bol_step4_testtime_native_current_time",
        "soh_percent",
        "target_definition",
        "capacity_method",
        "identity_provenance",
        "label_source",
        "training_authorized",
        "external_inference_authorized",
    ]
    _atomic_csv(
        output_directory / "g10_cell2_p5_label_extension.csv",
        extension_rows,
        extension_fields,
    )
    eligibility_fields = [
        "physical_cell_id",
        "rpt_index",
        "source_tpg",
        "source_file",
        "mapping_status",
        "step4_segment_reference",
        "step4_block_sha256",
        "step4_eligible",
        "step4_reasons",
        "step10_segment_reference",
        "step10_block_sha256",
        "step10_eligible",
        "step10_reasons",
        "current_sign_counts",
        "step10_zero_current_policy",
        "step10_zero_current_audit",
        "included_as_labeled_observation",
        "identity_provenance",
    ]
    _atomic_csv(
        output_directory / "g10_cell2_curve_eligibility.csv",
        g10_eligibility_rows,
        eligibility_fields,
    )
    _atomic_json(
        output_directory / "g10_cell2_source_sha256.json", g10_source_manifest
    )

    bol_reference_by_cell: dict[str, dict[str, Any]] = {}
    for records in accepted_by_partition.values():
        for record in records:
            cell_id = record["key"][1]
            reference = {
                "source_part": record["bol_source_part"],
                "source_file": record["bol_source_file"],
                "segment_reference": record["bol_segment_ref"],
                "block_sha256": record["bol_block_sha256"],
                "source_sha256": record["bol_source_sha256"],
                "source_exists_now": record["bol_source_available"],
            }
            prior = bol_reference_by_cell.get(cell_id)
            if prior is None:
                bol_reference_by_cell[cell_id] = reference
                continue
            for name in (
                "source_part",
                "source_file",
                "segment_reference",
                "block_sha256",
                "source_exists_now",
            ):
                if prior[name] != reference[name]:
                    raise ValueError(f"Accepted observations disagree on BoL for {cell_id}.")
            prior_source_hash = str(prior["source_sha256"])
            current_source_hash = str(reference["source_sha256"])
            if (
                prior_source_hash
                and current_source_hash
                and prior_source_hash != current_source_hash
            ):
                raise ValueError(f"Accepted observations disagree on BoL source hash for {cell_id}.")
            if not prior_source_hash and current_source_hash:
                prior["source_sha256"] = current_source_hash

    accepted_counts: dict[str, int] = {}
    accepted_cells: dict[str, int] = {}
    accepted_cell_ids: dict[str, list[str]] = {}
    assigned_cell_ids: dict[str, list[str]] = {
        partition: sorted(
            cell_id for cell_id, cell_partition in assignments.items()
            if cell_partition == partition
        )
        for partition in EXPECTED_PARTITION_CELL_COUNTS
    }
    for partition, records in accepted_by_partition.items():
        accepted_counts[partition] = len(records)
        accepted_cell_ids[partition] = sorted(
            {record["key"][1] for record in records}
        )
        accepted_cells[partition] = len(accepted_cell_ids[partition])
        arrays = Final2Arrays(
            X_discharge=np.asarray(
                [record["X_discharge"] for record in records], dtype=np.float32
            ),
            X_charge=np.asarray(
                [record["X_charge"] for record in records], dtype=np.float32
            ),
            X_physics_raw=np.asarray(
                [record["physics_raw"] for record in records], dtype=np.float32
            ),
            y_soh=np.asarray(
                [record["target"] for record in records], dtype=np.float32
            ).reshape(-1, 1),
        )
        arrays.validate()
        _atomic_npz(
            output_directory / f"{partition}.npz",
            {
                "X_discharge": arrays.X_discharge,
                "X_charge": arrays.X_charge,
                "X_physics_raw": arrays.X_physics_raw,
                "y_soh": arrays.y_soh,
                "dataset_part": np.asarray(
                    [record["key"][0] for record in records], dtype=np.int16
                ),
                "physical_cell_id": np.asarray(
                    [record["key"][1] for record in records], dtype="U32"
                ),
                "rpt_index": np.asarray(
                    [record["key"][2] for record in records], dtype=np.int16
                ),
            },
        )

    manifest_fields = [
        "dataset_part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "source_tpg",
        "source_file",
        "guide_stage",
        "mapping_status",
        "identity_provenance",
        "final2_partition",
        "step4_segment_reference",
        "step10_segment_reference",
        "step4_valid",
        "step10_valid",
        "source_exists_in_audit",
        "source_exists_now",
        "bol_source_exists_now",
        "status",
        "status_reason",
        "soh_percent",
        "target_definition",
        "capacity_method",
        "step4_block_sha256",
        "step10_block_sha256",
        "step4_source_sha256_snapshot",
        "step10_source_sha256_snapshot",
        "bol_source_part",
        "bol_source_file",
        "bol_segment_reference",
        "bol_block_sha256",
        "bol_source_sha256",
    ]
    _atomic_csv(output_directory / "observation_manifest.csv", observations, manifest_fields)

    cohort_complete = (
        status_counts["REVIEW"] == 0
        and status_counts["REJECTED"] == 0
        and status_counts["ACCEPTED"] == len(mapping_rows)
    )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_identifier": dataset_identifier,
        "status": "candidate_cohort_only",
        "p5_source_dataset": "bafuse-p5-electrical-v1-20261114",
        "p5_source_artifact_sha256": verified_artifacts,
        "final2_split_source": (
            (
                project_root
                / "reports"
                / "bafuse_final_data_validation_v2"
                / "cell_split_12train_4val_12external.csv"
            ).relative_to(project_root)
        ).as_posix(),
        "mapping_audit_source": mapping_audit_path.relative_to(project_root).as_posix(),
        "candidate_mapping_rows": len(mapping_rows),
        "frozen_p5_label_rows": len(frozen_p5_records),
        "final2_p5_label_extension_rows": len(extension_records),
        "accepted_label_rows_after_extension": len(p5_records),
        "mapping_status_counts": dict(
            Counter(_field(row, "final_mapping_status") for row in mapping_rows)
        ),
        "status_counts": dict(status_counts),
        "status_reason_counts": dict(reasons),
        "split_cell_counts": split_cells,
        "accepted_observations_by_partition": accepted_counts,
        "accepted_cells_by_partition": accepted_cells,
        "split_cells_without_accepted_p5_observations": {
            partition: sorted(
                set(assigned_cell_ids[partition]) - set(accepted_cell_ids[partition])
            )
            for partition in split_cells
        },
        "accepted_cohort_complete": cohort_complete,
        "accepted_cohort_frozen": False,
        "training_cohort_scope": (
            "Accepted observations only; REVIEW and REJECTED candidates are excluded."
        ),
        "training_authorized": training_authorized,
        "external_inference_authorized": external_inference_authorized,
        "normalization_parameters_fitted": False,
        "normalization_blocker": (
            None
            if training_authorized
            else (
                f"{status_counts['REVIEW']} mapping candidates remain under review and "
                f"{status_counts['REJECTED']} are rejected; training authorization is "
                "disabled."
            )
        ),
        "unresolved_candidate_note": (
            f"{status_counts['REVIEW']} REVIEW and {status_counts['REJECTED']} REJECTED "
            "candidates remain outside the approved Accepted training cohort."
        ),
        "g10_cell2_p5_extension": {
            "observations": len(extension_records),
            "partition": "external_test",
            "mapping_status": "GUIDE_VERIFIED",
            "provenance": G10_CELL2_PROVENANCE,
            "label_source": "Final2-only P5 extension; frozen P5 artifacts unchanged",
            "source_hash_manifest": "g10_cell2_source_sha256.json",
            "step10_zero_current_policy": step10_zero_current_policy,
        },
        "g10_cell2_curve_eligibility": {
            "mapped_candidates": len(g10_eligibility_rows),
            "eligible_observations": len(extension_records),
            "ineligible_observations": len(g10_eligibility_rows) - len(extension_records),
            "eligibility_manifest": "g10_cell2_curve_eligibility.csv",
            "ineligible_rpts": [
                {
                    "rpt_index": row["rpt_index"],
                    "failed_step4": not row["step4_eligible"],
                    "failed_step10": not row["step10_eligible"],
                    "step4_reasons": row["step4_reasons"],
                    "step10_reasons": row["step10_reasons"],
                }
                for row in g10_eligibility_rows
                if not row["included_as_labeled_observation"]
            ],
        },
        "feature_policy": {
            "discharge_curve": "Step 4 voltage, 128 points, P5 normalized-row-progress interpolation",
            "charge_curve": (
                "Step 10 voltage, 128 points, P5 normalized-row-progress interpolation; "
                f"zero-current policy={step10_zero_current_policy}"
            ),
            "physics_features": list(PHYSICS_FEATURE_NAMES),
            "physics_values_written": "raw P5 values; new Final2 Train-only scaler not fitted",
            "target": P5_LABEL_DEFINITION,
            "target_unit": "percent",
            "target_semantics": "reconstructed capacity-retention reference, not official Oxford ground truth",
        },
        "normalization_policy": (
            "P5 Variant D semantics; separate Step 4/Step 10 population z-scores, "
            "P5 physics population z-score, and target z-score, all fit on accepted "
            "Final2 Train rows only after the Accepted cohort is frozen and training is authorized."
        ),
        "artifacts_written": [
            "observation_manifest.csv",
            "g10_cell2_p5_label_extension.csv",
            "g10_cell2_curve_eligibility.csv",
            "g10_cell2_source_sha256.json",
            "train.npz",
            "validation.npz",
            "external_test.npz",
            "dataset_summary.json",
            "split_integrity.json",
            "dataset_sha256_manifest.json",
        ],
    }
    _atomic_json(output_directory / "dataset_summary.json", summary)

    split_integrity = {
        "cell_level_split": True,
        "split_cell_counts": split_cells,
        "accepted_observation_counts": accepted_counts,
        "accepted_cell_counts": accepted_cells,
        "accepted_cell_ids_by_partition": accepted_cell_ids,
        "bol_reference_by_accepted_cell": bol_reference_by_cell,
        "assigned_cell_ids_by_partition": assigned_cell_ids,
        "assigned_cells_without_accepted_p5_observations": {
            partition: sorted(
                set(assigned_cell_ids[partition]) - set(accepted_cell_ids[partition])
            )
            for partition in assigned_cell_ids
        },
        "p5_cells_missing_from_final2_split": sorted(
            {
                record["key"][1] for record in p5_records.values()
            }
            - set(assignments)
        ),
        "training_authorized": training_authorized,
    }
    partition_cells = split_integrity["accepted_cell_ids_by_partition"]
    ensure_cell_disjoint(partition_cells)
    _atomic_json(output_directory / "split_integrity.json", split_integrity)
    checksum_names = (
        "observation_manifest.csv",
        "g10_cell2_p5_label_extension.csv",
        "g10_cell2_curve_eligibility.csv",
        "g10_cell2_source_sha256.json",
        "train.npz",
        "validation.npz",
        "external_test.npz",
        "dataset_summary.json",
        "split_integrity.json",
    )
    output_hashes = {
        name: {
            "path": (output_directory / name).resolve().relative_to(project_root).as_posix(),
            "sha256": _sha256(output_directory / name),
        }
        for name in checksum_names
    }
    _atomic_json(
        output_directory / "dataset_sha256_manifest.json",
        {
            "schema_version": 1,
            "dataset_identifier": summary["dataset_identifier"],
            "artifacts": output_hashes,
            "manifest_includes_itself": False,
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the read-only-input Final2 candidate dataset from frozen P5 artifacts."
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-directory", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument(
        "--dataset-identifier",
        default="bafuse-final2-p5-g10-guide-verified-v1",
    )
    parser.add_argument(
        "--step10-zero-current-policy",
        choices=("strict", "exclude_zero_current"),
        default="strict",
    )
    parser.add_argument("--training-authorized", action="store_true")
    parser.add_argument("--external-inference-authorized", action="store_true")
    arguments = parser.parse_args()
    summary = prepare_final2_dataset(
        project_root=arguments.project_root,
        output_directory=arguments.output_directory,
        dataset_identifier=arguments.dataset_identifier,
        step10_zero_current_policy=arguments.step10_zero_current_policy,
        training_authorized=arguments.training_authorized,
        external_inference_authorized=arguments.external_inference_authorized,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
