from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
import sys
from array import array
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from bafuse.hybrid.hybrid_battery_health import SavedCNN1D
from bafuse.hybrid.soh_fusion import (
    EXCLUDED_FEATURES,
    INPUT_SCHEMA_VERSION,
    MODEL_VERSION,
    PHYSICS_FEATURE_NAMES,
    ThreeEncoderSoH,
    build_physics_features,
    resample_curve,
)


MANIFEST_PATH = Path("data/manifests/bafuse_rpt_observations.csv")
SPLIT_PATH = Path(
    "reports/bafuse_final_data_validation_v2/"
    "cell_split_12train_4val_12external.csv"
)
LABEL_PATHS = {
    "1": Path(
        "data/processed/label_soh_part1_2/run_007_part1_capacity_qc/"
        "labels_soh.csv"
    ),
    "2": Path(
        "data/processed/label_soh_part1_2/run_010_part2_soh_candidates/"
        "labels_soh_part2.csv"
    ),
    "3": Path(
        "experiments/results/data_audits/part3_soh_candidates/"
        "labels_soh_candidates.csv"
    ),
}
BASELINE_CHECKPOINT = Path(
    "models/checkpoints/part1_cnn1d/cnn1d_checkpoint.pt"
)
SEGMENT_PATTERN = re.compile(r"^(.*?)#Step(\d+):.*?rows(\d+)-(\d+)")
CURVE_COLUMNS = tuple(f"voltage_progress_{index:03d}" for index in range(128))
PARTITIONS = ("train", "validation", "external_test")
def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _normalize_path(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def _numeric(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _load_split() -> tuple[dict[str, str], str]:
    rows = _read_csv(ROOT / SPLIT_PATH)
    assignments: dict[str, str] = {}
    versions = {row["split_version"] for row in rows}
    if len(versions) != 1:
        raise ValueError(f"Expected one split version, got {sorted(versions)}")
    for row in rows:
        cell = row["physical_cell_id"]
        if cell in assignments:
            raise ValueError(f"Duplicate physical cell in split manifest: {cell}")
        assignments[cell] = row["partition"]
    counts = {
        partition: sum(value == partition for value in assignments.values())
        for partition in PARTITIONS
    }
    if counts != {"train": 12, "validation": 4, "external_test": 12}:
        raise ValueError(f"Unexpected physical-cell split counts: {counts}")
    if any(value not in PARTITIONS for value in assignments.values()):
        raise ValueError("Split manifest contains an unsupported partition.")
    return assignments, next(iter(versions))


def _load_labels() -> dict[tuple[str, str, str], dict[str, str]]:
    labels: dict[tuple[str, str, str], dict[str, str]] = {}
    for part, relative_path in LABEL_PATHS.items():
        for row in _read_csv(ROOT / relative_path):
            measurement = row.get("measurement_id", "")
            if not measurement.startswith("RPT"):
                continue
            key = (part, row.get("physical_cell_id", ""), measurement[3:])
            if key in labels:
                raise ValueError(f"Duplicate SOH label key: {key}")
            labels[key] = row
    return labels


def _parse_segment(row: dict[str, str], step: int) -> tuple[str, int, int] | None:
    if row.get(f"step{step}_status") != "verified":
        return None
    reference = row.get(f"step{step}_segment_ref", "")
    match = SEGMENT_PATTERN.match(reference)
    if not match:
        return None
    source, ref_step, start, end = match.groups()
    if int(ref_step) != step or _normalize_path(source) != _normalize_path(
        row["source_file"]
    ):
        return None
    return source, int(start), int(end)


def _load_observations() -> tuple[list[dict[str, Any]], str]:
    split_by_cell, split_version = _load_split()
    labels = _load_labels()
    rows = _read_csv(ROOT / MANIFEST_PATH)
    observations: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for source_row in rows:
        part = source_row["part"]
        cell = source_row["physical_cell_id"]
        rpt = str(source_row["rpt_index"])
        key = (part, cell, rpt)
        if key in seen:
            raise ValueError(f"Duplicate mapped observation key: {key}")
        seen.add(key)
        if cell not in split_by_cell:
            raise ValueError(f"Cell absent from 12/4/12 split: {cell}")
        label = labels.get(key)
        target = _numeric(label.get("soh_percent", "")) if label else None
        valid_label = bool(
            label
            and label.get("validation_status")
            == "calculated_from_measured_capacity"
            and target is not None
        )
        row = dict(source_row)
        row["partition"] = split_by_cell[cell]
        row["split_version"] = split_version
        row["label_status"] = (
            label.get("validation_status", "label_status_unspecified")
            if label
            else "no_label_row"
        )
        row["soh_percent"] = target if valid_label else None
        row["label_scope"] = (
            "provisional_external_candidate"
            if valid_label and part == "3"
            else "calculated_internal_candidate"
            if valid_label
            else "not_labeled"
        )
        row["step4_mapping_verified"] = _parse_segment(row, 4) is not None
        row["step10_mapping_verified"] = _parse_segment(row, 10) is not None
        row["step4_result"] = None
        row["step10_result"] = None
        row["physics_features"] = None
        row["fusion_ready"] = False
        row["exclusion_reason"] = (
            "no_validated_soh_label"
            if not valid_label
            else "missing_verified_step4_mapping"
            if not row["step4_mapping_verified"]
            else "missing_verified_step10_mapping"
            if not row["step10_mapping_verified"]
            else "raw_curve_validation_pending"
        )
        observations.append(row)
    for row in observations:
        if row["part"] == "3" and row["partition"] != "external_test":
            raise ValueError(
                f"Part 3 cell is not external-test-only: {row['physical_cell_id']}"
            )
    return observations, split_version


def _segment_requests(
    observations: list[dict[str, Any]],
    *,
    include_unlabeled: bool = False,
    steps: tuple[int, ...] = (4, 10),
) -> dict[str, list[dict[str, Any]]]:
    requests_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in observations:
        if row["soh_percent"] is None and not include_unlabeled:
            continue
        for step in steps:
            parsed = _parse_segment(row, step)
            if parsed is None:
                continue
            source, start, end = parsed
            request_id = (row["part"], row["physical_cell_id"], row["rpt_index"])
            request = {
                "row": row,
                "step": step,
                "source": source,
                "start": start,
                "end": end,
                "request_id": request_id,
                "expected_rows": end - start + 1,
                "observed_rows": 0,
                "step_ids": set(),
                "voltage": array("f"),
                "current": array("f"),
                "temperature": array("f"),
                "step_time": array("d"),
                "missing_voltage": 0,
                "missing_current": 0,
                "missing_temperature": 0,
                "missing_step_time": 0,
            }
            requests_by_source[_normalize_path(source)].append(request)
    return requests_by_source


def _parse_value(values: list[str], index: int | None) -> float | None:
    if index is None or index >= len(values):
        return None
    return _numeric(values[index])


def _mark_segment_invalid(
    request: dict[str, Any], reason: str
) -> None:
    result = {
        "valid": False,
        "status": reason,
        "curve": None,
        "voltage_raw": None,
        "current_raw": None,
        "temperature_raw": None,
        "duration_native": None,
        "row_start": request.get("start"),
        "row_end": request.get("end"),
        "expected_rows": request.get("expected_rows"),
        "observed_rows": request.get("observed_rows"),
        "missing_voltage": request.get("missing_voltage"),
        "missing_current": request.get("missing_current"),
        "missing_temperature": request.get("missing_temperature"),
    }
    request["row"][f"step{request['step']}_result"] = result


def _finish_segment(request: dict[str, Any]) -> None:
    step = request["step"]
    if request["observed_rows"] != request["expected_rows"]:
        _mark_segment_invalid(request, "segment_row_count_mismatch")
        return
    if request["step_ids"] != {step}:
        _mark_segment_invalid(request, "segment_step_id_mismatch")
        return
    if request["missing_voltage"] or request["missing_current"]:
        _mark_segment_invalid(request, "missing_or_nonfinite_voltage_or_current")
        return
    if len(request["voltage"]) < 2:
        _mark_segment_invalid(request, "segment_too_short_to_resample")
        return
    direction_ok = (
        all(value < 0 for value in request["current"])
        if step in (4, 8)
        else all(value > 0 for value in request["current"])
    )
    if not direction_ok:
        _mark_segment_invalid(request, "unexpected_current_direction")
        return
    curve = resample_curve(request["voltage"])
    duration = None
    if step == 4 and len(request["step_time"]) == request["expected_rows"]:
        time_values = request["step_time"]
        if all(second > first for first, second in zip(time_values, time_values[1:])):
            duration = float(time_values[-1] - time_values[0])
    result = {
        "valid": True,
        "status": "verified_raw_segment",
        "curve": curve,
        "voltage_raw": request["voltage"],
        "current_raw": request["current"],
        "temperature_raw": request["temperature"],
        "temperature_complete": (
            step == 4
            and request["missing_temperature"] == 0
            and len(request["temperature"]) == request["expected_rows"]
        ),
        "duration_native": duration,
        "row_start": request["start"],
        "row_end": request["end"],
        "observed_rows": request["observed_rows"],
    }
    request["row"][f"step{step}_result"] = result


def _scan_source_requests(
    root: Path, requests: list[dict[str, Any]]
) -> None:
    requests.sort(key=lambda item: (item["start"], item["end"], item["step"]))
    conflicted: set[int] = set()
    active: list[int] = []
    for index, request in enumerate(requests):
        active = [
            prior
            for prior in active
            if requests[prior]["end"] >= request["start"]
        ]
        if active:
            conflicted.add(index)
            conflicted.update(active)
        active.append(index)
    for index in conflicted:
        _mark_segment_invalid(requests[index], "overlapping_mapped_segment_ranges")
    requests = [
        request for index, request in enumerate(requests) if index not in conflicted
    ]
    if not requests:
        return

    source_path = Path(requests[0]["source"])
    if not source_path.is_absolute():
        source_path = root / Path(str(source_path).replace("\\", "/"))
    if not source_path.is_file():
        raise FileNotFoundError(
            f"Verified mapped source file is missing: {source_path}"
        )
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Mapped source has no CSV header: {source_path}")
        indices: dict[str, int] = {}
        for index, name in enumerate(header):
            canonical_name = name.strip().casefold()
            if canonical_name in indices:
                raise ValueError(f"Mapped source has duplicate column name: {name}")
            indices[canonical_name] = index
        required = {"step", "volts", "amps"}
        missing = required - set(indices)
        if missing:
            for request in requests:
                _mark_segment_invalid(
                    request, f"source_missing_columns:{','.join(sorted(missing))}"
                )
            return
        temperature_column = (
            "temp1"
            if "temp1" in indices
            else "temp"
            if "temp" in indices
            else None
        )
        pending_index = 0
        active: list[dict[str, Any]] = []
        max_end = max(request["end"] for request in requests)
        for values in reader:
            line_number = reader.line_num
            if line_number > max_end:
                break
            while (
                pending_index < len(requests)
                and requests[pending_index]["start"] <= line_number
            ):
                active.append(requests[pending_index])
                pending_index += 1
            if not active:
                continue
            for request in active:
                if line_number > request["end"]:
                    continue
                request["observed_rows"] += 1
                step_value = _parse_value(values, indices["step"])
                request["step_ids"].add(
                    int(step_value)
                    if step_value is not None and step_value.is_integer()
                    else None
                )
                voltage = _parse_value(values, indices["volts"])
                current = _parse_value(values, indices["amps"])
                if voltage is None:
                    request["missing_voltage"] += 1
                else:
                    request["voltage"].append(voltage)
                if current is None:
                    request["missing_current"] += 1
                else:
                    request["current"].append(current)
                if request["step"] == 4:
                    temperature = _parse_value(
                        values,
                        indices.get(temperature_column)
                        if temperature_column
                        else None,
                    )
                    if temperature is None:
                        request["missing_temperature"] += 1
                    else:
                        request["temperature"].append(temperature)
                    step_time = _parse_value(values, indices.get("steptime"))
                    if step_time is None:
                        request["missing_step_time"] += 1
                    else:
                        request["step_time"].append(step_time)
            active = [
                request for request in active if line_number < request["end"]
            ]
    for request in requests:
        _finish_segment(request)


def _extract_observations(
    observations: list[dict[str, Any]]
) -> dict[tuple[str, str, str], dict[str, Any]]:
    requests_by_source = _segment_requests(observations)
    for requests in requests_by_source.values():
        _scan_source_requests(ROOT, requests)

    usable: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in observations:
        step4 = row["step4_result"]
        step10 = row["step10_result"]
        if row["soh_percent"] is None:
            row["exclusion_reason"] = "no_validated_soh_label"
        elif not row["step4_mapping_verified"]:
            row["exclusion_reason"] = "missing_verified_step4_mapping"
        elif not row["step10_mapping_verified"]:
            row["exclusion_reason"] = "missing_verified_step10_mapping"
        elif not step4 or not step4["valid"]:
            row["exclusion_reason"] = (
                step4["status"] if step4 else "step4_extraction_not_attempted"
            )
        elif not step10 or not step10["valid"]:
            row["exclusion_reason"] = (
                step10["status"] if step10 else "step10_extraction_not_attempted"
            )
        elif not step4["temperature_complete"]:
            row["exclusion_reason"] = "missing_or_nonfinite_step4_temperature"
        else:
            features = build_physics_features(
                step4["voltage_raw"],
                step4["current_raw"],
                step4["temperature_raw"],
            )
            row["physics_features"] = features
            row["fusion_ready"] = True
            row["exclusion_reason"] = ""
            row["step4_duration_native_audit_only"] = step4["duration_native"]
            usable[
                (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
            ] = row
    return usable


def _record_identity(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "part": row["part"],
        "dataset_part": row["part"],
        "group": row["group"],
        "group_id": row["group"],
        "physical_cell_id": row["physical_cell_id"],
        "rpt_index": row["rpt_index"],
        "measurement_id": f"RPT{row['rpt_index']}",
        "partition": row["partition"],
        "source_file": row["source_file"],
    }


def _write_data_outputs(
    output_dir: Path,
    observations: list[dict[str, Any]],
    usable: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    availability_rows: list[dict[str, Any]] = []
    step4_curve_rows: list[dict[str, Any]] = []
    charge_curve_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    for row in observations:
        identity = _record_identity(row)
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        step4 = row["step4_result"]
        step10 = row["step10_result"]
        is_usable = key in usable
        step4_checked = step4 is not None
        step10_checked = step10 is not None
        feature_checkable = bool(
            row["soh_percent"] is not None
            and row["step4_mapping_verified"]
            and row["step10_mapping_verified"]
        )
        manifest_rows.append(
            {
                **identity,
                "step4_segment_ref": row["step4_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "label_status": row["label_status"],
                "soh_percent": row["soh_percent"],
                "label_scope": row["label_scope"],
                "step4_raw_status": (
                    step4["status"]
                    if step4_checked
                    else "not_checked_no_valid_label_or_mapping"
                ),
                "step10_raw_status": (
                    step10["status"]
                    if step10_checked
                    else "not_checked_no_valid_label_or_mapping"
                ),
                "fusion_eligible": is_usable,
                "fusion_exclusion_reason": row["exclusion_reason"],
                "step4_duration_native_audit_only": row.get(
                    "step4_duration_native_audit_only"
                ),
            }
        )
        availability_rows.append(
            {
                **identity,
                "label_available": row["soh_percent"] is not None,
                "step4_mapping_verified": row["step4_mapping_verified"],
                "step10_mapping_verified": row["step10_mapping_verified"],
                "step4_raw_curve_valid": (
                    bool(step4["valid"]) if step4_checked else ""
                ),
                "step10_raw_curve_valid": (
                    bool(step10["valid"]) if step10_checked else ""
                ),
                "temperature_complete": (
                    bool(
                        step4
                        and step4["valid"]
                        and step4.get("temperature_complete", False)
                    )
                    if step4_checked
                    else ""
                ),
                "physics_features_complete": (
                    row["physics_features"] is not None
                    if feature_checkable
                    else ""
                ),
                "all_modalities_complete": is_usable,
                "provisional_external_label": row["label_scope"]
                == "provisional_external_candidate",
                "exclusion_reason": row["exclusion_reason"],
            }
        )
        if step4 and step4["valid"]:
            step4_curve_rows.append(
                {
                    **identity,
                    **{
                        column: value
                        for column, value in zip(
                            CURVE_COLUMNS, step4["curve"]
                        )
                    },
                }
            )
        if step10 and step10["valid"]:
            charge_curve_rows.append(
                {
                    **identity,
                    **{
                        column: value
                        for column, value in zip(
                            CURVE_COLUMNS, step10["curve"]
                        )
                    },
                }
            )
        if row["physics_features"] is not None:
            physics_rows.append(
                {
                    **identity,
                    **row["physics_features"],
                    "step4_discharge_duration_native_audit_only": row.get(
                        "step4_duration_native_audit_only"
                    ),
                    "duration_in_model_input": False,
                }
            )
    _write_csv(output_dir / "multimodal_feature_manifest.csv", manifest_rows)
    _write_csv(output_dir / "modality_availability.csv", availability_rows)
    if not step4_curve_rows or not charge_curve_rows or not physics_rows:
        raise ValueError("No complete raw features were extracted.")
    _write_csv(output_dir / "charge_features.csv", charge_curve_rows)
    _write_csv(output_dir / "physics_features.csv", physics_rows)
    _write_csv(output_dir / "discharge_features.csv", step4_curve_rows)


def _fit_scaler(values: np.ndarray, name: str) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError(f"Non-finite train-only {name} scaler values.")
    if np.any(scale < 1e-8):
        raise ValueError(f"Constant or near-constant train-only {name} inputs.")
    return mean.astype(np.float32), scale.astype(np.float32)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _r2(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if denominator <= 0:
        return None
    numerator = float(np.sum((actual - predicted) ** 2))
    return 1 - numerator / denominator


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    errors = predicted - actual
    return {
        "mae_soh_percentage_points": float(np.mean(np.abs(errors))),
        "rmse_soh_percentage_points": float(np.sqrt(np.mean(errors**2))),
        "r2": _r2(actual, predicted),
    }


def _evaluate(
    records: list[dict[str, Any]],
    predictions: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for partition in PARTITIONS:
        selected = [record for record in records if record["partition"] == partition]
        if not selected:
            raise ValueError(f"No eligible observations in {partition}.")
        for model_name, prediction_key in (
            ("train_fold_mean", "train_fold_mean"),
            ("historical_cnn1d", "historical_cnn1d"),
            ("three_encoder_fusion", "fusion"),
        ):
            actual = np.asarray(
                [record["soh_percent"] for record in selected], dtype=np.float64
            )
            predicted = np.asarray(
                [
                    predictions[
                        (
                            record["part"],
                            record["physical_cell_id"],
                            str(record["rpt_index"]),
                        )
                    ][prediction_key]
                    for record in selected
                ],
                dtype=np.float64,
            )
            overall = _metrics(actual, predicted)
            rows.append(
                {
                    "model": model_name,
                    "partition": partition,
                    "metric_scope": "overall",
                    "physical_cell_id": "",
                    "observation_count": len(selected),
                    "cell_count": len({record["physical_cell_id"] for record in selected}),
                    **overall,
                }
            )
            per_cell_metrics: list[dict[str, float | None]] = []
            for cell in sorted({record["physical_cell_id"] for record in selected}):
                cell_records = [
                    record
                    for record in selected
                    if record["physical_cell_id"] == cell
                ]
                cell_actual = np.asarray(
                    [record["soh_percent"] for record in cell_records],
                    dtype=np.float64,
                )
                cell_predicted = np.asarray(
                    [
                        predictions[
                            (
                                record["part"],
                                record["physical_cell_id"],
                                str(record["rpt_index"]),
                            )
                        ][prediction_key]
                        for record in cell_records
                    ],
                    dtype=np.float64,
                )
                cell_metrics = _metrics(cell_actual, cell_predicted)
                per_cell_metrics.append(cell_metrics)
                rows.append(
                    {
                        "model": model_name,
                        "partition": partition,
                        "metric_scope": "per_cell",
                        "physical_cell_id": cell,
                        "observation_count": len(cell_records),
                        "cell_count": 1,
                        **cell_metrics,
                    }
                )
            rows.append(
                {
                    "model": model_name,
                    "partition": partition,
                    "metric_scope": "macro_cell",
                    "physical_cell_id": "",
                    "observation_count": len(selected),
                    "cell_count": len(per_cell_metrics),
                    "mae_soh_percentage_points": statistics.mean(
                        item["mae_soh_percentage_points"]
                        for item in per_cell_metrics
                    ),
                    "rmse_soh_percentage_points": statistics.mean(
                        item["rmse_soh_percentage_points"]
                        for item in per_cell_metrics
                    ),
                    "r2": (
                        statistics.mean(
                            item["r2"]
                            for item in per_cell_metrics
                            if item["r2"] is not None
                        )
                        if any(item["r2"] is not None for item in per_cell_metrics)
                        else None
                    ),
                }
            )
    return rows


def _train(
    records: list[dict[str, Any]],
    baseline_checkpoint_path: Path,
    output_dir: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    batch_size: int,
    freeze_discharge: bool,
) -> tuple[dict[tuple[str, str, str], dict[str, float]], dict[str, Any]]:
    train_records = [row for row in records if row["partition"] == "train"]
    validation_records = [
        row for row in records if row["partition"] == "validation"
    ]
    if not train_records or not validation_records:
        raise ValueError("Training and validation require eligible labelled rows.")
    train_cells = {row["physical_cell_id"] for row in train_records}
    validation_cells = {row["physical_cell_id"] for row in validation_records}
    if train_cells & validation_cells:
        raise ValueError("Physical cells cross the training/validation boundary.")
    if any(row["part"] == "3" for row in train_records + validation_records):
        raise ValueError("Part 3 must remain external-test-only.")

    train_discharge = np.asarray(
        [row["step4_result"]["curve"] for row in train_records],
        dtype=np.float32,
    )
    train_charge = np.asarray(
        [row["step10_result"]["curve"] for row in train_records],
        dtype=np.float32,
    )
    train_physics = np.asarray(
        [
            [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES]
            for row in train_records
        ],
        dtype=np.float32,
    )
    train_target = np.asarray(
        [row["soh_percent"] for row in train_records], dtype=np.float32
    )
    discharge_mean, discharge_scale = _fit_scaler(
        train_discharge, "Step 4 voltage"
    )
    charge_mean, charge_scale = _fit_scaler(train_charge, "Step 10 voltage")
    physics_mean, physics_scale = _fit_scaler(train_physics, "physics")
    target_mean = float(train_target.mean())
    target_scale = float(train_target.std())
    if not math.isfinite(target_mean) or not math.isfinite(target_scale):
        raise ValueError("Non-finite training target scaler.")
    if target_scale < 1e-8:
        raise ValueError("Training SOH labels have no usable variation.")

    def normalized(records_subset: list[dict[str, Any]]) -> tuple[np.ndarray, ...]:
        discharge = np.asarray(
            [row["step4_result"]["curve"] for row in records_subset],
            dtype=np.float32,
        )
        charge = np.asarray(
            [row["step10_result"]["curve"] for row in records_subset],
            dtype=np.float32,
        )
        physics = np.asarray(
            [
                [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES]
                for row in records_subset
            ],
            dtype=np.float32,
        )
        target = np.asarray(
            [row["soh_percent"] for row in records_subset], dtype=np.float32
        )
        return (
            (discharge - discharge_mean) / discharge_scale,
            (charge - charge_mean) / charge_scale,
            (physics - physics_mean) / physics_scale,
            (target - target_mean) / target_scale,
        )

    train_arrays = normalized(train_records)
    validation_arrays = normalized(validation_records)
    tensors = TensorDataset(
        *(torch.from_numpy(value.astype(np.float32)) for value in train_arrays)
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        tensors,
        batch_size=min(batch_size, len(train_records)),
        shuffle=True,
        generator=generator,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ThreeEncoderSoH(freeze_discharge=freeze_discharge)
    historical = torch.load(
        baseline_checkpoint_path, map_location="cpu", weights_only=False
    )
    required_checkpoint = {
        "model_state_dict",
        "x_mean",
        "x_scale",
        "y_mean",
        "y_scale",
        "feature_names",
    }
    if not required_checkpoint.issubset(historical):
        raise ValueError("Historical CNN checkpoint is missing required fields.")
    initialized_keys = model.initialize_discharge_encoder(
        historical["model_state_dict"]
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
        weight_decay=1e-4,
    )
    criterion = torch.nn.MSELoss()
    validation_tensors = tuple(
        torch.from_numpy(value.astype(np.float32)).to(device)
        for value in validation_arrays
    )

    def evaluate_loss(values: tuple[torch.Tensor, ...]) -> float:
        model.eval()
        with torch.no_grad():
            output = model(values[0], values[1], values[2])
            return float(criterion(output, values[3].reshape(-1, 1)).item())

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for discharge, charge, physics, target in loader:
            discharge = discharge.to(device)
            charge = charge.to(device)
            physics = physics.to(device)
            target = target.to(device).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(discharge, charge, physics)
            loss = criterion(prediction, target)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        validation_loss = evaluate_loss(validation_tensors)
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": statistics.mean(batch_losses),
                "validation_mse_scaled": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - 1e-8:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break
    if best_state is None:
        raise RuntimeError("Training completed without a best validation checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)

    baseline = SavedCNN1D()
    baseline.load_state_dict(historical["model_state_dict"])
    baseline.to(device)
    baseline.eval()
    feature_names = tuple(historical["feature_names"])
    if len(feature_names) != 128 or len(set(feature_names)) != 128:
        raise ValueError("Historical CNN checkpoint must contain 128 unique features.")
    baseline_x_mean = np.asarray(historical["x_mean"], dtype=np.float32).reshape(128)
    baseline_x_scale = np.asarray(historical["x_scale"], dtype=np.float32).reshape(128)
    baseline_y_mean = float(historical["y_mean"])
    baseline_y_scale = float(historical["y_scale"])
    if not np.all(np.isfinite(baseline_x_mean)) or not np.all(
        np.isfinite(baseline_x_scale)
    ):
        raise ValueError("Historical CNN input scaler is non-finite.")
    if (
        np.any(baseline_x_scale <= 0)
        or not math.isfinite(baseline_y_mean)
        or not math.isfinite(baseline_y_scale)
        or baseline_y_scale <= 0
    ):
        raise ValueError("Historical CNN scaler has a non-positive scale.")

    predictions: dict[tuple[str, str, str], dict[str, float]] = {}
    model.eval()
    with torch.no_grad():
        for record in records:
            key = (
                record["part"],
                record["physical_cell_id"],
                str(record["rpt_index"]),
            )
            discharge = np.asarray(
                record["step4_result"]["curve"], dtype=np.float32
            )
            baseline_input = (discharge - baseline_x_mean) / baseline_x_scale
            baseline_tensor = torch.from_numpy(
                baseline_input[None, None, :]
            ).to(device)
            baseline_prediction = float(
                (baseline(baseline_tensor).reshape(-1)[0].item())
                * baseline_y_scale
                + baseline_y_mean
            )
            normalized_inputs = (
                (discharge - discharge_mean) / discharge_scale,
                (
                    np.asarray(record["step10_result"]["curve"], dtype=np.float32)
                    - charge_mean
                )
                / charge_scale,
                (
                    np.asarray(
                        [
                            record["physics_features"][name]
                            for name in PHYSICS_FEATURE_NAMES
                        ],
                        dtype=np.float32,
                    )
                    - physics_mean
                )
                / physics_scale,
            )
            prediction = model(
                *(
                    torch.from_numpy(value[None, :].astype(np.float32)).to(device)
                    for value in normalized_inputs
                )
            )
            fusion_prediction = float(
                prediction.reshape(-1)[0].item() * target_scale + target_mean
            )
            if not math.isfinite(baseline_prediction) or not math.isfinite(
                fusion_prediction
            ):
                raise ValueError(f"Non-finite prediction for {key}")
            predictions[key] = {
                "train_fold_mean": target_mean,
                "historical_cnn1d": baseline_prediction,
                "fusion": fusion_prediction,
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "fusion_checkpoint.pt"
    checkpoint_payload = {
        "model_version": MODEL_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "model_state_dict": best_state,
        "physics_feature_names": PHYSICS_FEATURE_NAMES,
        "discharge_mean": discharge_mean,
        "discharge_scale": discharge_scale,
        "charge_mean": charge_mean,
        "charge_scale": charge_scale,
        "physics_mean": physics_mean,
        "physics_scale": physics_scale,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "training_seed": seed,
        "best_epoch": best_epoch,
        "training_cells": sorted(train_cells),
        "split_version": records[0]["split_version"],
        "historical_checkpoint": str(baseline_checkpoint_path.relative_to(ROOT)),
        "historical_cnn_encoder_keys_initialized": list(initialized_keys),
        "discharge_encoder_frozen": freeze_discharge,
        "excluded_features": EXCLUDED_FEATURES,
        "missingness_policy": (
            "Complete same-RPT Step 4, Step 10, and physics observations only; "
            "no imputation or synthetic values."
        ),
    }
    torch.save(checkpoint_payload, checkpoint_path)
    _write_csv(output_dir / "fusion_training_history.csv", history)
    config = {
        "model_version": MODEL_VERSION,
        "input_schema_version": INPUT_SCHEMA_VERSION,
        "architecture": {
            "discharge_encoder": "Conv1D(1,16,9)->GELU->MaxPool(2)->Conv1D(16,32,5)->GELU->AdaptiveAvgPool(1)->Flatten",
            "charge_encoder": "independent Conv1D encoder with the same architecture",
            "physics_encoder": "Linear(5,32)->GELU->Linear(32,32)->GELU",
            "fusion": "concat 32+32+32; Linear(96,64)->GELU->Dropout(0.2)->Linear(64,32)->GELU->Linear(32,1)",
        },
        "physics_features": list(PHYSICS_FEATURE_NAMES),
        "excluded_features": EXCLUDED_FEATURES,
        "input_points": 128,
        "split_version": records[0]["split_version"],
        "training_seed": seed,
        "best_epoch": best_epoch,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "batch_size": batch_size,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "loss": "MSE on train-standardized SOH percent",
        "preprocessing_fit_scope": "12/4/12 training partition only",
        "training_observations": len(train_records),
        "training_cells_with_labels": len(train_cells),
        "validation_observations": len(validation_records),
        "validation_cells_with_labels": len(validation_cells),
        "initialized_from": str(baseline_checkpoint_path.relative_to(ROOT)),
        "discharge_encoder_frozen": freeze_discharge,
        "device": str(device),
    }
    with (output_dir / "fusion_model_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(config, stream, indent=2)
    return predictions, {
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "training_cells": train_cells,
        "validation_cells": validation_cells,
        "checkpoint_path": checkpoint_path,
        "history": history,
        "config": config,
    }


def _write_evaluation_outputs(
    output_dir: Path,
    records: list[dict[str, Any]],
    predictions: dict[tuple[str, str, str], dict[str, float]],
) -> list[dict[str, Any]]:
    prediction_rows: list[dict[str, Any]] = []
    for row in records:
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        values = predictions[key]
        prediction_rows.append(
            {
                **_record_identity(row),
                "reference_soh_percent": row["soh_percent"],
                "label_scope": row["label_scope"],
                "train_fold_mean_soh_percent": values["train_fold_mean"],
                "historical_cnn1d_predicted_soh_percent": values[
                    "historical_cnn1d"
                ],
                "fusion_v2_predicted_soh_percent": values["fusion"],
                "fusion_v2_model_version": MODEL_VERSION,
                "fusion_v2_input_schema_version": INPUT_SCHEMA_VERSION,
                "fusion_v2_missingness_policy": "complete_modalities_only",
            }
        )
    _write_csv(output_dir / "fusion_predictions.csv", prediction_rows)
    metrics = _evaluate(records, predictions)
    _write_csv(output_dir / "fusion_soh_evaluation.csv", metrics)
    return metrics


def _write_reports(
    output_dir: Path,
    observations: list[dict[str, Any]],
    usable: dict[tuple[str, str, str], dict[str, Any]],
    training: dict[str, Any],
    metrics: list[dict[str, Any]],
    split_version: str,
) -> None:
    counts = {
        partition: sum(
            row["partition"] == partition for row in usable.values()
        )
        for partition in PARTITIONS
    }
    cells = {
        partition: len(
            {
                row["physical_cell_id"]
                for row in usable.values()
                if row["partition"] == partition
            }
        )
        for partition in PARTITIONS
    }
    metrics_by_model_partition = {
        (row["model"], row["partition"], row["metric_scope"]): row
        for row in metrics
    }
    lines = [
        "# Three-encoder SoH architecture",
        "",
        f"- Model version: `{MODEL_VERSION}`.",
        f"- Split: `{split_version}`; physical-cell counts 12 train / 4 validation / 12 external.",
        "- Same-RPT estimation: Step 4 discharge + Step 10 charge + Step 4 physics map to one labeled cell/RPT.",
        "- Part 3 was used only for external evaluation; its SOH labels remain provisional capacity-ratio candidates.",
        "- No cell/group/file/part identifier or future observation is a model input.",
        "- Input curves were linearly resampled to 128 points on normalized sample progress.",
        "- Curve, physics, and target scalers were fit on training observations only.",
        "- The historical CNN1D checkpoint supplied only compatible discharge-convolution weights. All seven historical training cells are in the new training partition; its old scalers were not reused for fusion preprocessing.",
        "- Complete modalities only: no imputation or synthetic measurement values.",
        f"- Trained with seed 211, best validation epoch {training['best_epoch']}, device `{training['config']['device']}`.",
        "",
        "## Physics input and exclusions",
        "",
        "Physics encoder input: `"
        + "`, `".join(PHYSICS_FEATURE_NAMES)
        + "`.",
        "",
        "`step4_discharge_duration` is excluded from model input because constant-current Step 4 duration can nearly reconstruct delivered capacity and leak same-RPT SoH. It is retained only as an audit-only column in `physics_features.csv`.",
        "Temperature channels retain native values; their absolute units, sensor calibration/placement, and cross-part equivalence are unverified. Current is reported in native CSV units. Voltage slopes are deterministic summaries of the same Step 4 representation supplied to the discharge encoder.",
        "",
        "## Complete-modality label coverage",
        "",
        "| Partition | Eligible observations | Cells with labels |",
        "|---|---:|---:|",
    ]
    for partition in PARTITIONS:
        lines.append(f"| {partition} | {counts[partition]} | {cells[partition]} |")
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            "| Model | Partition | MAE (SOH pp) | RMSE (SOH pp) | R² |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for model_name in (
        "train_fold_mean",
        "historical_cnn1d",
        "three_encoder_fusion",
    ):
        for partition in PARTITIONS:
            row = metrics_by_model_partition[
                (model_name, partition, "overall")
            ]
            lines.append(
                f"| {model_name} | {partition} | "
                f"{row['mae_soh_percentage_points']:.4f} | "
                f"{row['rmse_soh_percentage_points']:.4f} | "
                f"{row['r2'] if row['r2'] is not None else 'n/a'} |"
            )
    lines.extend(
        [
            "",
            "Macro-cell MAE/RMSE and per-cell results are in `fusion_soh_evaluation.csv`. The train-fold mean is the simple same-time reference. The historical CNN1D model is the learned baseline, not a newly retrained discharge-only ablation. External results are descriptive and were not used for model selection.",
            "",
            "## PyDMA and scope",
            "",
            "PyDMA source, optimization settings, references, labels, and quality flags were not modified or used as SoH inputs. This phase implements and evaluates the SoH fusion branch only; Phase 3 ablations were not run.",
        ]
    )
    (output_dir / "fusion_architecture.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    integration = [
        "# Three-encoder integration readiness",
        "",
        "The versioned `fusion_predictions.csv` includes historical CNN1D and fusion predictions keyed by dataset part, physical cell, and RPT. The dashboard integration joins only on those exact keys; the existing PyDMA calculation and aggregation path remains unchanged.",
        "",
        f"Complete-modality prediction rows: {len(usable)}; total manifest rows: {len(observations)}.",
        f"Fusion checkpoint: `{training['checkpoint_path'].relative_to(ROOT)}`.",
        "",
        "Part 3 predictions are external-only and based on provisional candidate labels. The dashboard must preserve that provenance. The original CNN1D checkpoint and its historical artifacts remain unchanged.",
        "",
        "## Reproduction",
        "",
        "Train and evaluate: `.venv-torch-cuda/Scripts/python.exe experiments/soh_forecast/bafuse_three_encoder_v1/run_phase2.py`.",
        "Focused tests: `.venv-torch-cuda/Scripts/python.exe -m pytest -q tests/test_bafuse_three_encoder_soh.py tests/test_hybrid_battery_health.py tests/test_dashboard_paths.py`.",
        "Regenerate the hybrid dashboard in a separate output directory: `.venv-torch-cuda/Scripts/python.exe src/bafuse/hybrid/hybrid_battery_health.py --root . --output reports/bafuse_three_encoder_v1_hybrid`.",
    ]
    (output_dir / "fusion_integration_report.md").write_text(
        "\n".join(integration) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the approved BaFuse three-encoder SoH model."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=211)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--freeze-discharge-encoder", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        raise ValueError("Epoch, patience, and batch-size values must be positive.")
    output_dir = args.output_dir.resolve()
    _seed_everything(args.seed)
    observations, split_version = _load_observations()
    usable = _extract_observations(observations)
    _write_data_outputs(output_dir, observations, usable)
    records = sorted(
        usable.values(),
        key=lambda row: (
            PARTITIONS.index(row["partition"]),
            row["part"],
            row["physical_cell_id"],
            int(row["rpt_index"]),
        ),
    )
    if any(row["partition"] not in PARTITIONS for row in records):
        raise ValueError("Unsupported partition in extracted observations.")
    checkpoint = ROOT / BASELINE_CHECKPOINT
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Historical CNN1D checkpoint not found: {checkpoint}")
    predictions, training = _train(
        records,
        checkpoint,
        output_dir,
        args.seed,
        args.epochs,
        args.patience,
        args.batch_size,
        args.freeze_discharge_encoder,
    )
    metrics = _write_evaluation_outputs(output_dir, records, predictions)
    _write_reports(
        output_dir, observations, usable, training, metrics, split_version
    )
    print(
        f"Wrote {MODEL_VERSION} outputs to {output_dir}; "
        f"{len(records)} complete observations, best epoch {training['best_epoch']}."
    )


if __name__ == "__main__":
    main()
