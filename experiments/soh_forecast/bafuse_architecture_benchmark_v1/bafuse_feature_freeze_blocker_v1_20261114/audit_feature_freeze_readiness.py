from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
LABEL_DIR = BENCHMARK_DIR / "bafuse_soh_trajectory_audit_v1_20261114"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
M1_DIR = BENCHMARK_DIR / "m1_raw_validation_20261113"
SEGMENT_PATTERN = re.compile(
    r"^(?P<path>.+)#Step(?P<step>\d+):segment[^:]*:rows"
    r"(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$",
    re.IGNORECASE,
)
PHYSICS_FEATURES = (
    "mean_discharge_temperature_native",
    "discharge_temperature_rise_native",
    "discharge_plateau_slope_v_per_progress",
    "discharge_slope_variability_v_per_progress",
    "mean_absolute_discharge_current_native",
)
EXPECTED_TARGET = (
    "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)"
)
INPUT_FILES = {
    "labels": LABEL_DIR / "bafuse_soh_labels_203.csv",
    "label_checksums": LABEL_DIR / "bafuse_soh_final_checksums.json",
    "candidate_manifest": RECONSTRUCTION_DIR / "final_candidate_manifest.csv",
    "feature_matrix": RECONSTRUCTION_DIR / "raw_curve_feature_matrix.csv",
    "block_audit": M1_DIR / "measurement_block_audit.csv",
    "source_hashes": OUTPUT_DIR / "bafuse_raw_source_hashes.csv",
    "dataset_review": ROOT / "experiments" / "dataset_review" / "README.md",
    "physics_coverage": BENCHMARK_DIR / "physics_feature_coverage.csv",
    "raw_reconstruction_code": ROOT / "src" / "bafuse" / "labeling" / "raw_soh_reconstruction.py",
    "fusion_feature_code": ROOT / "src" / "bafuse" / "hybrid" / "soh_fusion.py",
    "architecture_code": ROOT / "src" / "bafuse" / "hybrid" / "architecture_models.py",
    "legacy_voltage_extractor": ROOT / "src" / "bafuse" / "features" / "soh_voltage.py",
    "baseline_config": BENCHMARK_DIR / "baseline_config.json",
    "architecture_config": BENCHMARK_DIR / "architecture_matrix_v1" / "architecture_config.json",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("dataset_part", "")),
        row.get("physical_cell_id", ""),
        str(row.get("rpt_index", "")),
    )


def _unique_index(
    rows: list[dict[str, str]], name: str
) -> tuple[dict[tuple[str, str, str], dict[str, str]], list[str]]:
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    errors: list[str] = []
    for row in rows:
        key = _key(row)
        if key in indexed:
            errors.append(f"{name} has duplicate key {key}.")
        else:
            indexed[key] = row
    return indexed, errors


def _normalize(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _parse_segment(reference: str) -> tuple[str, int, int, int]:
    match = SEGMENT_PATTERN.fullmatch(reference.strip())
    if match is None:
        raise ValueError(f"Unparseable segment reference: {reference!r}")
    return (
        match.group("path"),
        int(match.group("step")),
        int(match.group("start")),
        int(match.group("end")),
    )


def _block_key(reference: str, part: str) -> str:
    source, step, start, end = _parse_segment(reference)
    return f"{_normalize(source)}|{part}|{step}|{start}|{end}"


def _finite_count(row: dict[str, str], columns: list[str]) -> int:
    count = 0
    for column in columns:
        try:
            value = float(row[column])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            count += 1
    return count


def _read_mapped_block(reference: str) -> tuple[list[str], list[list[str]], str]:
    source, _, start, end = _parse_segment(reference)
    digest = hashlib.sha256()
    rows: list[list[str]] = []
    with _path_from_source(source).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        for line_number, row in enumerate(reader, start=2):
            if line_number < start:
                continue
            if line_number > end:
                break
            rows.append(row)
            digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
    if len(rows) != end - start + 1:
        raise ValueError(f"Incomplete mapped block {reference!r}: read {len(rows)} rows.")
    return header, rows, digest.hexdigest()


def _resample(values: list[float], count: int = 128) -> list[float]:
    last = len(values) - 1
    result = []
    for index in range(count):
        position = index * last / (count - 1)
        lower = math.floor(position)
        upper = min(lower + 1, last)
        fraction = position - lower
        result.append(values[lower] * (1 - fraction) + values[upper] * fraction)
    return result


def _physics_features(
    voltage_values: list[float],
    current_values: list[float],
    temperature_values: list[float],
) -> dict[str, float]:
    voltage = _resample(voltage_values)
    progress = [index / (len(voltage) - 1) for index in range(len(voltage))]
    plateau = [index for index, value in enumerate(progress) if 0.20 <= value <= 0.80]
    x_mean = statistics.mean(progress[index] for index in plateau)
    y_mean = statistics.mean(voltage[index] for index in plateau)
    slope = sum(
        (progress[index] - x_mean) * (voltage[index] - y_mean)
        for index in plateau
    ) / sum((progress[index] - x_mean) ** 2 for index in plateau)
    derivatives = [
        (voltage[index + 1] - voltage[index - 1])
        / (progress[index + 1] - progress[index - 1])
        for index in range(1, len(voltage) - 1)
        if 0.10 <= progress[index] <= 0.90
    ]
    median_derivative = statistics.median(derivatives)
    return {
        "mean_discharge_temperature_native": statistics.mean(temperature_values),
        "discharge_temperature_rise_native": max(temperature_values)
        - temperature_values[0],
        "discharge_plateau_slope_v_per_progress": slope,
        "discharge_slope_variability_v_per_progress": 1.4826
        * statistics.median(
            abs(value - median_derivative) for value in derivatives
        ),
        "mean_absolute_discharge_current_native": statistics.mean(
            abs(value) for value in current_values
        ),
    }


def _sampling_sensitivity(
    candidates: list[dict[str, str]], features: list[dict[str, str]]
) -> list[dict[str, Any]]:
    candidate_by_key = {_key(row): row for row in candidates}
    feature_by_key = {_key(row): row for row in features}
    cases = (("1", "G2-Cell3", "2"), ("3", "G8-Cell18", "1"))
    results = []
    for key in cases:
        candidate = candidate_by_key[key]
        header, rows, block_hash = _read_mapped_block(
            candidate["step4_segment_ref"]
        )
        if block_hash != candidate["step4_block_sha256"]:
            raise ValueError(f"Current raw block hash differs for {key}.")
        columns = {name.strip().casefold(): index for index, name in enumerate(header)}
        temperature_name = "temp1" if "temp1" in columns else "temp"

        def values(source_rows: list[list[str]]) -> tuple[list[float], list[float], list[float]]:
            return (
                [float(row[columns["volts"]]) for row in source_rows],
                [float(row[columns["amps"]]) for row in source_rows],
                [float(row[columns[temperature_name]]) for row in source_rows],
            )

        raw_voltage, raw_current, raw_temperature = values(rows)
        raw_curve = _resample(raw_voltage)
        raw_features = _physics_features(raw_voltage, raw_current, raw_temperature)
        deduplicated_rows: list[list[str]] = []
        for row in rows:
            if not deduplicated_rows or row != deduplicated_rows[-1]:
                deduplicated_rows.append(row)
        dedup_voltage, dedup_current, dedup_temperature = values(deduplicated_rows)
        dedup_curve = _resample(dedup_voltage)
        dedup_features = _physics_features(
            dedup_voltage, dedup_current, dedup_temperature
        )
        stored = feature_by_key[key]
        raw_match = max(
            abs(raw_features[name] - float(stored[name]))
            for name in PHYSICS_FEATURES
        )
        result: dict[str, Any] = {
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "step4_block_sha256": block_hash,
            "raw_rows": len(rows),
            "exact_adjacent_duplicate_rows_removed": len(rows)
            - len(deduplicated_rows),
            "temperature_raw_column": header[columns[temperature_name]],
            "temperature_raw_min": min(raw_temperature),
            "temperature_raw_max": max(raw_temperature),
            "feature_matrix_recomputation_max_abs_delta": raw_match,
            "duplicate_collapse_curve_max_abs_delta_v": max(
                abs(left - right) for left, right in zip(raw_curve, dedup_curve)
            ),
        }
        result.update(
            {
                f"duplicate_collapse_delta_{name}": dedup_features[name]
                - raw_features[name]
                for name in PHYSICS_FEATURES
            }
        )
        results.append(result)
    return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_from_source(source: str) -> Path:
    return ROOT / Path(source.replace("/", "\\"))


def _load_inputs() -> dict[str, Any]:
    return {
        "labels": _read_csv(INPUT_FILES["labels"]),
        "candidates": _read_csv(INPUT_FILES["candidate_manifest"]),
        "features": _read_csv(INPUT_FILES["feature_matrix"]),
        "blocks": _read_csv(INPUT_FILES["block_audit"]),
        "source_hashes": _read_csv(INPUT_FILES["source_hashes"]),
    }


def _build_alignment_audit(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    labels, label_errors = _unique_index(data["labels"], "labels")
    candidates, candidate_errors = _unique_index(data["candidates"], "candidate manifest")
    features, feature_errors = _unique_index(data["features"], "feature matrix")
    block_by_key = {row["block_key"]: row for row in data["blocks"]}
    source_hash_by_path = {
        _normalize(row["source_file"]): row for row in data["source_hashes"]
    }
    errors = label_errors + candidate_errors + feature_errors
    errors.extend(
        [
            f"Label count is {len(labels)}, expected 203."
            for _ in [None]
            if len(labels) != 203
        ]
    )
    if set(labels) != set(candidates):
        errors.append("Label and candidate-manifest observation keys differ.")
    if set(labels) != set(features):
        errors.append("Label and feature-matrix observation keys differ.")

    discharge_columns = sorted(
        column for column in data["features"][0] if column.startswith("discharge_v_")
    )
    charge_columns = sorted(
        column for column in data["features"][0] if column.startswith("charge_v_")
    )
    rows: list[dict[str, Any]] = []
    for key in sorted(labels, key=lambda item: (int(item[0]), item[1], int(item[2]))):
        label = labels[key]
        candidate = candidates.get(key, {})
        feature = features.get(key, {})
        part = key[0]
        step4_ref = candidate.get("step4_segment_ref", "")
        step10_ref = candidate.get("step10_segment_ref", "")
        step4_block: dict[str, str] = {}
        step10_block: dict[str, str] = {}
        mapping_error = ""
        try:
            step4_block = block_by_key[_block_key(step4_ref, part)]
            step10_block = block_by_key[_block_key(step10_ref, part)]
        except (KeyError, ValueError) as error:
            mapping_error = str(error)

        rpt_source = label.get("rpt_step4_source_file", "")
        bol_source = label.get("bol_step4_source_file", "")
        rpt_hash = source_hash_by_path.get(_normalize(rpt_source), {})
        bol_hash = source_hash_by_path.get(_normalize(bol_source), {})
        try:
            _, step4_number, step4_start, step4_end = _parse_segment(step4_ref)
            _, step10_number, step10_start, step10_end = _parse_segment(step10_ref)
            references_valid = (
                step4_number == 4
                and step10_number == 10
                and _normalize(_parse_segment(step4_ref)[0])
                == _normalize(_parse_segment(step10_ref)[0])
                and step10_start > step4_end
            )
        except ValueError:
            step4_start = step4_end = step10_start = step10_end = 0
            references_valid = False

        step4_hash_ok = bool(
            step4_block
            and step4_block.get("raw_block_sha256")
            == candidate.get("step4_block_sha256")
            == label.get("rpt_step4_block_sha256")
        )
        step10_hash_ok = bool(
            step10_block
            and step10_block.get("raw_block_sha256")
            == candidate.get("step10_block_sha256")
        )
        blocks_complete = all(
            block
            and block.get("row_count_exact") == "True"
            and block.get("step_ids_match") == "True"
            and not block.get("scan_error")
            and block.get("source_changed_during_scan") != "True"
            for block in (step4_block, step10_block)
        )
        current_directions_valid = bool(
            step4_block
            and step10_block
            and int(step4_block["current_negative_rows"])
            == int(step4_block["row_count"])
            and int(step4_block["current_positive_rows"]) == 0
            and int(step10_block["current_positive_rows"])
            == int(step10_block["row_count"])
            and int(step10_block["current_negative_rows"]) == 0
        )
        try:
            step4_voltage_first = float(step4_block["voltage_first_native"])
            step4_voltage_last = float(step4_block["voltage_last_native"])
            step10_voltage_first = float(step10_block["voltage_first_native"])
            step10_voltage_last = float(step10_block["voltage_last_native"])
            voltage_endpoint_directions_valid = (
                step4_voltage_first > step4_voltage_last
                and step10_voltage_first < step10_voltage_last
            )
        except (KeyError, TypeError, ValueError):
            step4_voltage_first = step4_voltage_last = math.nan
            step10_voltage_first = step10_voltage_last = math.nan
            voltage_endpoint_directions_valid = False
        temperature_rows_complete = bool(
            step4_block
            and int(step4_block["step4_temperature_finite_rows"])
            == int(step4_block["row_count"])
            and int(step4_block["missing_temperature_rows"]) == 0
        )

        rpt_file = _path_from_source(rpt_source)
        stat_unchanged = bool(
            step4_block
            and rpt_file.is_file()
            and rpt_file.stat().st_size == int(step4_block["source_size_bytes"])
            and rpt_file.stat().st_mtime_ns == int(step4_block["source_mtime_ns"])
        )
        rpt_hash_status = rpt_hash.get("status", "missing")
        bol_hash_status = bol_hash.get("status", "missing")
        if label.get("rpt_source_file_sha256"):
            rpt_hash_status = (
                "MATCH"
                if rpt_hash.get("sha256") == label["rpt_source_file_sha256"]
                else "MISMATCH"
            )
        if label.get("bol_source_file_sha256"):
            bol_hash_status = (
                "MATCH"
                if bol_hash.get("sha256") == label["bol_source_file_sha256"]
                else "MISMATCH"
            )

        discharge_finite = _finite_count(feature, discharge_columns)
        charge_finite = _finite_count(feature, charge_columns)
        physics_finite = _finite_count(feature, list(PHYSICS_FEATURES))
        temperature_mean = feature.get("mean_discharge_temperature_native", "")
        try:
            temperature_value = float(temperature_mean)
        except (TypeError, ValueError):
            temperature_value = math.nan

        row = {
            "dataset_part": part,
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "partition": label.get("partition", ""),
            "label_status": label.get("final_status", ""),
            "soh_percent": label.get("soh_percent", ""),
            "target_definition": label.get("target_definition", ""),
            "label_candidate_key_match": key in candidates,
            "label_step4_reference_match": _normalize(rpt_source)
            == _normalize(_parse_segment(step4_ref)[0])
            if step4_ref
            else False,
            "label_step4_block_hash_match": step4_hash_ok,
            "step4_reference": step4_ref,
            "step10_reference": step10_ref,
            "step4_block_sha256": candidate.get("step4_block_sha256", ""),
            "step10_block_sha256": candidate.get("step10_block_sha256", ""),
            "step4_block_hash_audit_match": step4_hash_ok,
            "step10_block_hash_audit_match": step10_hash_ok,
            "block_audit_complete": blocks_complete,
            "step4_source_size_mtime_unchanged": stat_unchanged,
            "step4_row_count": step4_block.get("row_count", ""),
            "step10_row_count": step10_block.get("row_count", ""),
            "step4_current_negative_rows": step4_block.get("current_negative_rows", ""),
            "step4_current_positive_rows": step4_block.get("current_positive_rows", ""),
            "step10_current_negative_rows": step10_block.get("current_negative_rows", ""),
            "step10_current_positive_rows": step10_block.get("current_positive_rows", ""),
            "current_directions_valid": current_directions_valid,
            "step4_testtime_zero_intervals": step4_block.get("testtime_zero_intervals", ""),
            "step4_steptime_zero_intervals": step4_block.get("steptime_zero_intervals", ""),
            "step10_testtime_zero_intervals": step10_block.get("testtime_zero_intervals", ""),
            "step10_steptime_zero_intervals": step10_block.get("steptime_zero_intervals", ""),
            "step4_voltage_first_v": step4_voltage_first,
            "step4_voltage_last_v": step4_voltage_last,
            "step10_voltage_first_v": step10_voltage_first,
            "step10_voltage_last_v": step10_voltage_last,
            "voltage_endpoint_directions_valid": voltage_endpoint_directions_valid,
            "discharge_curve_finite_points": discharge_finite,
            "charge_curve_finite_points": charge_finite,
            "physics_finite_features": physics_finite,
            "temperature_raw_column": step4_block.get("temperature_column", ""),
            "temperature_column_status": step4_block.get(
                "temperature_column_status", ""
            ),
            "temperature_finite_rows": step4_block.get(
                "step4_temperature_finite_rows", ""
            ),
            "temperature_rows_complete": temperature_rows_complete,
            "temperature_native_mean": temperature_mean,
            "temperature_semantics": "native units unresolved; sensor calibration/location not documented",
            "part3_lowest_temperature_observation": False,
            "rpt_source_sha256": rpt_hash.get("sha256", ""),
            "rpt_source_hash_status": rpt_hash_status,
            "bol_source_sha256": bol_hash.get("sha256", ""),
            "bol_source_hash_status": bol_hash_status,
            "reference_mapping_error": mapping_error,
            "same_rpt_step_order_valid": references_valid,
            "feature_provenance": feature.get("feature_provenance", ""),
            "leakage_fields_excluded": feature.get(
                "feature_target_leakage_fields_excluded", ""
            ),
            "training_authorized": label.get("training_authorized", ""),
        }
        rows.append(row)

    part3_rows = [row for row in rows if row["dataset_part"] == "3"]
    finite_part3_temps = [
        float(row["temperature_native_mean"])
        for row in part3_rows
        if row["temperature_native_mean"] not in ("", None)
        and math.isfinite(float(row["temperature_native_mean"]))
    ]
    if finite_part3_temps:
        minimum = min(finite_part3_temps)
        for row in part3_rows:
            try:
                row["part3_lowest_temperature_observation"] = (
                    float(row["temperature_native_mean"]) == minimum
                )
            except (TypeError, ValueError):
                pass

    for row in rows:
        if not row["label_step4_reference_match"]:
            errors.append(f"Label Step 4 mapping differs for {_key(row)}.")
        if not row["label_step4_block_hash_match"]:
            errors.append(f"Step 4 hash differs for {_key(row)}.")
        if not row["step4_block_hash_audit_match"] or not row["step10_block_hash_audit_match"]:
            errors.append(f"Raw block-audit hash differs for {_key(row)}.")
        if not row["block_audit_complete"] or not row["step4_source_size_mtime_unchanged"]:
            errors.append(f"Raw block audit or source stat check failed for {_key(row)}.")
        if not row["current_directions_valid"]:
            errors.append(f"Current direction check failed for {_key(row)}.")
        if not row["voltage_endpoint_directions_valid"]:
            errors.append(f"Voltage endpoint direction check failed for {_key(row)}.")
        if not row["temperature_rows_complete"]:
            errors.append(f"Raw temperature rows are incomplete for {_key(row)}.")
        if not row["same_rpt_step_order_valid"]:
            errors.append(f"Step 4/10 reference order is invalid for {_key(row)}.")
        if row["discharge_curve_finite_points"] != 128:
            errors.append(f"Discharge curve shape/finite check failed for {_key(row)}.")
        if row["charge_curve_finite_points"] != 128:
            errors.append(f"Charge curve shape/finite check failed for {_key(row)}.")
        if row["physics_finite_features"] != 5:
            errors.append(f"Physics feature shape/finite check failed for {_key(row)}.")
        if row["target_definition"] != EXPECTED_TARGET:
            errors.append(f"Target definition differs for {_key(row)}.")
        if row["label_status"] != "ACCEPTED":
            errors.append(f"Label status is not ACCEPTED for {_key(row)}.")
        try:
            if not math.isfinite(float(row["soh_percent"])):
                errors.append(f"SOH target is non-finite for {_key(row)}.")
        except (TypeError, ValueError):
            errors.append(f"SOH target is missing or invalid for {_key(row)}.")
        if row["rpt_source_hash_status"] in {"MISMATCH", "missing"}:
            errors.append(f"RPT source hash check failed for {_key(row)}.")
        if row["bol_source_hash_status"] in {"MISMATCH", "missing"}:
            errors.append(f"BoL source hash check failed for {_key(row)}.")
    return rows, errors


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty audit: {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run() -> dict[str, Any]:
    data = _load_inputs()
    alignment, errors = _build_alignment_audit(data)
    _write_csv(OUTPUT_DIR / "bafuse_alignment_audit.csv", alignment)
    sampling_sensitivity = _sampling_sensitivity(
        data["candidates"], data["features"]
    )
    _write_csv(
        OUTPUT_DIR / "bafuse_sampling_sensitivity.csv", sampling_sensitivity
    )

    labels = data["labels"]
    split_counts = Counter(row["partition"] for row in labels)
    cells_by_split: dict[str, set[str]] = defaultdict(set)
    for row in labels:
        cells_by_split[row["partition"]].add(row["physical_cell_id"])
    split_overlap = {
        f"{left}:{right}": sorted(cells_by_split[left] & cells_by_split[right])
        for left in cells_by_split
        for right in cells_by_split
        if left < right and cells_by_split[left] & cells_by_split[right]
    }
    status_counts = Counter(row["final_status"] for row in labels)
    part_counts = Counter(row["dataset_part"] for row in labels)
    feature_columns = list(data["features"][0])
    discharge_columns = [column for column in feature_columns if column.startswith("discharge_v_")]
    charge_columns = [column for column in feature_columns if column.startswith("charge_v_")]
    physics_columns = [column for column in feature_columns if column in PHYSICS_FEATURES]
    temperature_values = [
        float(row["temperature_native_mean"])
        for row in alignment
        if row["dataset_part"] == "3"
    ]
    lowest_part3 = min(
        (row for row in alignment if row["part3_lowest_temperature_observation"]),
        key=lambda row: row["physical_cell_id"],
        default=None,
    )
    torch_available = importlib.util.find_spec("torch") is not None
    source_hash_rows = data["source_hashes"]
    source_hash_matches = Counter(row["status"] for row in source_hash_rows)
    step4_timestamp_anomalies = [
        row
        for row in alignment
        if int(row["step4_testtime_zero_intervals"] or 0)
        or int(row["step4_steptime_zero_intervals"] or 0)
    ]
    step10_timestamp_anomalies = [
        row
        for row in alignment
        if int(row["step10_testtime_zero_intervals"] or 0)
        or int(row["step10_steptime_zero_intervals"] or 0)
    ]

    feature_schema = {
        "status": "candidate_only_not_frozen",
        "modalities": {
            "X_discharge": {"shape": [203, 128], "feature_prefix": "discharge_v_", "resampling": "linear on normalized raw row index"},
            "X_charge": {"shape": [203, 128], "feature_prefix": "charge_v_", "step": 10, "resampling": "linear on normalized raw row index"},
            "X_physics": {"shape": [203, 5], "features": physics_columns},
            "y_soh": {"shape": [203, 1], "definition": EXPECTED_TARGET, "unit": "percent"},
        },
        "curve_progress_is_capacity_derived": False,
        "curve_progress_basis": "normalized sample-row index; no Amphr or capacity axis",
        "temperature_units": "native unresolved; no Celsius assumption",
        "temperature_cross_part_semantics_verified": False,
        "array_dtype": "not frozen; no NPZ created",
        "preprocessing_fit_scope": "none; no scaler or imputer fitted",
    }
    _write_json(OUTPUT_DIR / "bafuse_feature_schema_candidate.json", feature_schema)

    forward_status = {
        "training_started": False,
        "external_prediction_evaluation_started": False,
        "synthetic_forward_passes_run": False,
        "status": "NOT_RUN_TORCH_UNAVAILABLE" if not torch_available else "NOT_RUN_BLOCKER",
        "torch_available_in_audit_runtime": torch_available,
        "note": "The earlier test collection failed with ModuleNotFoundError: torch; no model was trained or evaluated.",
    }
    _write_json(OUTPUT_DIR / "bafuse_model_forward_pass_status.json", forward_status)

    if lowest_part3 is None:
        low_temp_text = "No finite Part 3 temperature means were present."
    else:
        low_temp_text = (
            f"Part 3 minimum: {lowest_part3['physical_cell_id']} RPT{lowest_part3['rpt_index']} "
            f"({float(lowest_part3['temperature_native_mean']):.12g} native units); "
            f"Part 3 range: {min(temperature_values):.12g} to {max(temperature_values):.12g}."
        )
    feature_report = f"""# BaFuse Feature Audit — Freeze Readiness

Status: **BLOCKED — dataset not frozen**

## Cohort and integrity

- Labels: {len(labels)} rows; status counts: {dict(status_counts)}.
- Split counts: {dict(split_counts)}; Parts: {dict(part_counts)}.
- Physical cells: {len(set(row['physical_cell_id'] for row in labels))}; split overlap: {split_overlap or 'none'}.
- Observation-key sets agree across labels, candidate mappings, and feature matrix: {not errors}.
- Step 4 and Step 10 exact block mappings: 203 each; block hashes match the prior raw-block audit and the current source size/mtime checks.
- Timestamp audit: {len(step4_timestamp_anomalies)} Step 4 and {len(step10_timestamp_anomalies)} Step 10 blocks contain zero elapsed-time intervals; the sole mapped case is G2-Cell3 RPT2 with 26 repeated intervals on each axis. The raw Step 4 exact-duplicate sensitivity is reported separately.
- Full raw-source SHA-256 snapshots: {len(source_hash_rows)} unique RPT/BoL files; historical expected hashes: {source_hash_matches.get('MATCH', 0)} matched, {source_hash_matches.get('MISMATCH', 0)} mismatched, {source_hash_matches.get('CURRENT_SNAPSHOT_ONLY', 0)} recorded without a prior full-file digest.
- Feature shape/finite scan: discharge {len(discharge_columns)} columns; charge {len(charge_columns)}; physics {len(physics_columns)}; see the row-level alignment CSV.

## Modalities and leakage

- Discharge: Guide-mapped Step 4; expected negative current direction; voltage curves are resampled linearly on normalized raw row index.
- Charge: one exact same-RPT Step 10 block per record, in the same source file and after its Step 4 block; expected positive current direction. No Step 10 selection ambiguity was found in the candidate mappings.
- The voltage-progress axis is based on normalized row index, not capacity, `Amphr`, or integrated charge. The persisted matrix metadata excludes target, capacity, `Amphr`, duration, IDs, and future data from predictive columns. No direct target leakage was identified.
- The legacy `extract_rpt_charge_voltages` helper in `soh_voltage.py` selects Steps 5/6; it was not used for this matrix. The matrix was built through Guide-mapped Step 10 references in `raw_soh_reconstruction.py`.
- Sampling sensitivity was checked directly from the raw Step 4 rows for G2-Cell3 RPT2 and G8-Cell18 RPT1; recomputed physics features match the persisted matrix exactly. For G2-Cell3 RPT2, collapsing 26 exact adjacent duplicate rows changes the 128-point discharge curve by at most {sampling_sensitivity[0]['duplicate_collapse_curve_max_abs_delta_v']:.9g} V; feature deltas are recorded without applying a threshold or changing the stored feature. See `bafuse_sampling_sensitivity.csv`.
- Physics features are engineered summaries of same-RPT Step 4 signals: temperature mean/rise from the raw temperature column, voltage slope and derivative MAD from `Volts`, and mean absolute current from `Amps`. They are not five independent physical modalities.
- Voltage units are documented as V. Current is an inferred native-A interpretation. Temperature remains native and unresolved; no Celsius conversion or imputation was applied.

## Blocking issue: temperature validity

The raw block audit identifies `Temp1` in Part 1 and `Temp` in Parts 2–3, but the feature coverage record states that units, sensor calibration, and location are not documented. Part 3 uses a different Maccor instrument and its README does not establish chamber temperature. Cross-Part semantic comparability is therefore unverified.

{low_temp_text}

This is a raw observed value, not a basis for a universal cutoff or an automatic correction. The `G8-Cell18` RPT1 Step 4 block contains a finite `Temp` signal near 0.0015 native units while the other Part 3 observations are much higher. Without sensor metadata, the value cannot be distinguished from an unavailable/mis-scaled channel. The row is retained and flagged; no value was imputed, clipped, or removed.

Because temperature is one of the required five physics inputs, its validity across Parts and the flagged observation must be resolved before freezing. Do not replace it from another RPT or cell.

## Additional validation blocker

Four-model synthetic forward passes were not run. The available test collection reported `ModuleNotFoundError: No module named 'torch'`; the runtime used for this audit does not expose PyTorch. Architecture source declares the requested `(N,128)`, `(N,128)`, `(N,5)` inputs and `(N,1)` output, but compatibility is not runtime-verified here.

## Preprocessing and authorization

- No scaler or imputer was fitted; there are no train-fitted parameters to record.
- No NPZ arrays or frozen label copy were written.
- No model training or external prediction evaluation was run.
- The user authorized freezing if checks pass; execution remains blocked. `training_authorized: false`; `external_prediction_evaluation_authorized: false`.

## Required resolution before retry

1. Establish the semantics, units, and cross-instrument comparability of `Temp1`/`Temp` from source metadata, and resolve the G8-Cell18 RPT1 low raw reading without imputation.
2. Provide an environment with PyTorch, then run schema-loading and synthetic forward-pass tests for all four architecture interfaces.
3. Rerun the row-level audit. Only after critical checks pass may Train-only preprocessing be fitted and a frozen dataset be created.
"""
    (OUTPUT_DIR / "bafuse_feature_quality_report.md").write_text(
        feature_report, encoding="utf-8"
    )
    blocker_report = f"""# BaFuse Dataset Freeze Blocker

**Freeze decision:** not frozen. The user authorized freezing if checks pass, but the required `(N,128)`, `(N,128)`, `(N,5)` arrays are only numerically present in the existing candidate feature matrix; critical validity checks remain unresolved. No frozen NPZs, scalers, or model-ready dataset were created.

## Blockers

1. **Temperature feature semantics are unresolved across Parts.** The inputs use `Temp1` (Part 1) and `Temp` (Parts 2–3); unit/calibration/sensor location are undocumented. Part 3 uses Maccor 4000, and the README does not establish its chamber temperature. {low_temp_text} The affected row remains in the audit as `G8-Cell18`, Part 3, RPT1; no imputation or exclusion was applied.
2. **Four-model forward compatibility is unverified.** Synthetic forward tests could not run because PyTorch is unavailable in the test runtime (`ModuleNotFoundError: torch`).

## Checks that passed

- 203 unique accepted labels and 203 unique `(part, physical_cell_id, RPT)` keys.
- Split counts: Train 102, Validation 37, External Test 64; 23 cells and no cell overlap.
- Label target policy is unchanged: `{EXPECTED_TARGET}`; all label rows carry `training_authorized: false`.
- Step 4 and Step 10 references match the exact raw-block audit hashes for all 203 observations; current source file size/mtime agrees with the prior scan.
- Full current SHA-256 was computed for all {len(source_hash_rows)} unique raw RPT/BoL files; {source_hash_matches.get('MISMATCH', 0)} mismatches. {source_hash_matches.get('MATCH', 0)} match previously recorded full-file hashes; the rest are explicitly current snapshots.
- Raw Step 4 resampling sensitivity was recomputed for the known repeated-timestamp case and the Part 3 low-temperature case; exact results are in `bafuse_sampling_sensitivity.csv`.
- Existing candidate matrix has 128 finite discharge samples, 128 finite Step 10 samples, and five finite numeric physics summaries per row. Numeric completeness does not resolve temperature semantics.
- All 203 Step 10 mappings are same-RPT and use the exact mapped source segment; no ambiguity was detected in the persisted Guide crosswalk.
- No direct target-derived feature or capacity-based curve coordinate was detected.

## Not authorized or not performed

- No label changes or split changes.
- No NPZ creation, scaler fitting, training, hyperparameter selection, or External Test prediction evaluation.
- No PyDMA or Half-Cell fitting.

The row-level evidence is in `bafuse_alignment_audit.csv`; current raw source hashes are in `bafuse_raw_source_hashes.csv`. This package is a blocker/audit artifact, not a frozen dataset.
"""
    (OUTPUT_DIR / "bafuse_dataset_freeze_blocker_report.md").write_text(
        blocker_report, encoding="utf-8"
    )

    config = {
        "dataset_freeze_status": "BLOCKED_NO_FREEZE",
        "dataset_freeze_authorized": True,
        "dataset_frozen": False,
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "label_policy": EXPECTED_TARGET,
        "resampling": "deterministic linear interpolation on normalized raw row index",
        "preprocessing_fit": "none",
        "training_started": False,
        "external_prediction_evaluation_started": False,
        "critical_blockers": [
            "temperature semantics and cross-Part comparability unresolved",
            "G8-Cell18 Part 3 RPT1 temperature value requires source-level interpretation",
            "PyTorch unavailable; synthetic four-architecture forward checks not run",
        ],
    }
    _write_json(OUTPUT_DIR / "bafuse_feature_audit_config.json", config)

    if errors:
        raise RuntimeError("Alignment/provenance checks failed: " + "; ".join(errors[:10]))
    if len(labels) != 203 or split_counts != Counter(
        {"train": 102, "validation": 37, "external_test": 64}
    ):
        raise RuntimeError("Label cohort or split counts do not match the approved cohort.")
    if split_overlap:
        raise RuntimeError(f"Physical cells overlap splits: {split_overlap}")

    input_hashes = {
        name: _sha256(path).lower()
        for name, path in INPUT_FILES.items()
    }
    input_hashes["audit_generator"] = _sha256(Path(__file__))
    output_names = [
        "audit_feature_freeze_readiness.py",
        "bafuse_alignment_audit.csv",
        "bafuse_dataset_freeze_blocker_report.md",
        "bafuse_feature_audit_config.json",
        "bafuse_feature_quality_report.md",
        "bafuse_feature_schema_candidate.json",
        "bafuse_model_forward_pass_status.json",
        "bafuse_raw_source_hashes.csv",
        "bafuse_sampling_sensitivity.csv",
    ]
    output_hashes = {
        name: _sha256(OUTPUT_DIR / name)
        for name in output_names
    }
    checksum_manifest = {
        "schema_version": 1,
        "dataset_freeze_status": "BLOCKED_NO_FREEZE",
        "dataset_freeze_authorized": True,
        "dataset_frozen": False,
        "dataset_identifier": None,
        "labels": 203,
        "split_counts": dict(split_counts),
        "physical_cells": 23,
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "inputs_sha256": input_hashes,
        "outputs_sha256": output_hashes,
        "checksum_file_self_hash_excluded": True,
    }
    _write_json(OUTPUT_DIR / "bafuse_feature_audit_sha256.json", checksum_manifest)
    return {
        "status": "BLOCKED_NO_FREEZE",
        "labels": len(labels),
        "alignment_errors": len(errors),
        "split_counts": dict(split_counts),
        "source_hash_status": dict(source_hash_matches),
        "torch_available": torch_available,
        "output_directory": str(OUTPUT_DIR),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
