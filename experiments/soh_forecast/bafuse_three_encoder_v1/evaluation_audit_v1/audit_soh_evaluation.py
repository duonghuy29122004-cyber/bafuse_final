from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[4]
PHASE2_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE2_DIR))

from bafuse.data.soh_step4_scan import scan_step4_file
from bafuse.hybrid.hybrid_battery_health import SavedCNN1D
from bafuse.hybrid.soh_fusion import (
    EXCLUDED_FEATURES,
    INPUT_SCHEMA_VERSION,
    MODEL_VERSION,
    PHYSICS_FEATURE_NAMES,
    ThreeEncoderSoH,
    build_physics_features,
)

import run_phase2 as phase2


CELL2 = "G10-Cell2"
PHASE2_OUTPUT = PHASE2_DIR
SPLIT_PATH = ROOT / phase2.SPLIT_PATH
MANIFEST_PATH = ROOT / phase2.MANIFEST_PATH
PART3_LABELS_PATH = ROOT / phase2.LABEL_PATHS["3"]
HYBRID_OUTPUT = ROOT / "reports" / "bafuse_three_encoder_v1_hybrid"
FUSION_CHECKPOINT = PHASE2_OUTPUT / "fusion_checkpoint.pt"
BASELINE_CHECKPOINT = ROOT / phase2.BASELINE_CHECKPOINT
CURVE_COLUMNS = phase2.CURVE_COLUMNS
PARTITIONS = phase2.PARTITIONS
EXPECTED_CURRENT_METRICS = {
    ("train", "historical_cnn1d"): (2.829405, 3.302004, 0.647125),
    ("train", "three_encoder_fusion"): (0.600448, 0.754045, 0.981598),
    ("validation", "historical_cnn1d"): (3.372109, 3.714346, 0.554423),
    ("validation", "three_encoder_fusion"): (0.457352, 0.586625, 0.988886),
    ("external_test", "historical_cnn1d"): (1.398226, 1.654440, 0.826163),
    ("external_test", "three_encoder_fusion"): (1.486690, 1.877927, 0.776026),
}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return pd.read_csv(path, keep_default_na=False)


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
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
    rpt_text = str(int(float(rpt_text)))
    return part_text, cell_text, rpt_text


def _key_from_row(row: dict[str, Any] | pd.Series) -> tuple[str, str, str]:
    part = row.get("dataset_part", row.get("part", ""))
    rpt = row.get("rpt_index", row.get("measurement_id", ""))
    return _key(part, row.get("physical_cell_id", ""), rpt)


def _records_by_key(
    frame: pd.DataFrame,
) -> dict[tuple[str, str, str], dict[str, Any]]:
    records: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in frame.to_dict(orient="records"):
        key = _key_from_row(row)
        if key in records:
            raise ValueError(f"Duplicate observation key: {key}")
        records[key] = row
    return records


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float | None]:
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual):
        raise ValueError("Metrics require non-empty, aligned one-dimensional arrays.")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("Metrics cannot be computed from non-finite values.")
    errors = predicted - actual
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    return {
        "mae_soh_percentage_points": float(np.mean(np.abs(errors))),
        "rmse_soh_percentage_points": float(np.sqrt(np.mean(errors**2))),
        "r2": (
            1.0 - float(np.sum(errors**2)) / denominator
            if denominator > 0
            else None
        ),
    }


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _segment_capacity(
    scan: dict[str, Any], source: str, start: int, end: int
) -> dict[str, Any]:
    segments = json.loads(scan["step4_segments_json"])
    matched = [
        item
        for item in segments
        if item["source_row_start"] == start and item["source_row_end"] == end
    ]
    if len(matched) != 1:
        raise ValueError(
            f"Expected one Step 4 capacity segment for {source} rows {start}-{end}; "
            f"found {len(matched)}."
        )
    return matched[0]


def _mapped_step4_ref(reference: str) -> tuple[str, int, int]:
    match = phase2.SEGMENT_PATTERN.match(reference)
    if not match:
        raise ValueError(f"Malformed mapped Step 4 segment reference: {reference}")
    source, step, start, end = match.groups()
    if int(step) != 4:
        raise ValueError(f"Expected a Step 4 reference, got: {reference}")
    return source, int(start), int(end)


def _resolve_source_path(source: str) -> Path:
    candidates = (
        ROOT / Path(source),
        ROOT / "data" / "raw" / "oxford" / "part3" / Path(source),
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Mapped source file does not exist: {source}")


def _capacity_is_eligible(segment: dict[str, Any]) -> tuple[bool, str]:
    rows = int(segment["step4_rows"])
    finite_rows = int(segment["capacity_finite_rows"])
    span = segment["capacity_span"]
    if rows < 10 or finite_rows != rows:
        return False, "insufficient_or_nonfinite_step4_capacity"
    if span is None or not math.isfinite(float(span)) or float(span) <= 0:
        return False, "invalid_step4_capacity_span"
    if int(segment["negative_capacity_deltas"]) != 0:
        return False, "step4_capacity_reversal"
    if int(segment["positive_capacity_deltas"]) == 0:
        return False, "step4_capacity_has_no_positive_progress"
    if segment["current_signs"] != ["negative"]:
        return False, "step4_current_direction_unverified"
    return True, "verified_step4_capacity_segment"


def _load_supplemental_cell2() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations, _ = phase2._load_observations()
    rows = [
        row
        for row in observations
        if row["part"] == "3" and row["physical_cell_id"] == CELL2
    ]
    if len(rows) != 10:
        raise ValueError(f"Expected 10 mapped {CELL2} RPTs, found {len(rows)}.")
    if any(row["partition"] != "external_test" for row in rows):
        raise ValueError(f"{CELL2} is not exclusively in the external-test split.")
    if any(row["soh_percent"] is not None for row in rows):
        raise ValueError(
            f"{CELL2} unexpectedly has historical labels; do not overwrite them."
        )

    requests_by_source = phase2._segment_requests(rows, include_unlabeled=True)
    for requests in requests_by_source.values():
        phase2._scan_source_requests(ROOT, requests)

    baseline_source, baseline_start, baseline_end = _mapped_step4_ref(
        rows[0]["bol_step4_segment_ref"]
    )
    if phase2._normalize_path(rows[0]["bol_reference_file"]) != phase2._normalize_path(
        baseline_source
    ):
        raise ValueError(f"{CELL2} BoL file and Step 4 segment reference disagree.")
    capacity_scans: dict[str, dict[str, Any]] = {}
    for row in rows:
        source, start, end = _mapped_step4_ref(row["step4_segment_ref"])
        if source not in capacity_scans:
            capacity_scans[source] = scan_step4_file(_resolve_source_path(source))
    capacity_scans[baseline_source] = scan_step4_file(
        _resolve_source_path(baseline_source)
    )
    baseline_scan = capacity_scans[baseline_source]
    baseline_segment = _segment_capacity(
        baseline_scan, baseline_source, baseline_start, baseline_end
    )
    baseline_ok, baseline_status = _capacity_is_eligible(baseline_segment)
    baseline_capacity = (
        float(baseline_segment["capacity_span"]) if baseline_ok else None
    )
    if any(row["bol_step4_segment_ref"] != rows[0]["bol_step4_segment_ref"] for row in rows):
        raise ValueError(f"{CELL2} mapped RPTs do not share one BoL reference.")

    capacity_rows: list[dict[str, Any]] = []
    capacity_rows.append(
        {
            "role": "earliest_mapped_bol_reference",
            "physical_cell_id": CELL2,
            "measurement_id": "BoL",
            "source_file": baseline_source,
            "source_row_start": baseline_start,
            "source_row_end": baseline_end,
            "step4_rows": baseline_segment["step4_rows"],
            "capacity_finite_rows": baseline_segment["capacity_finite_rows"],
            "capacity_span_native": baseline_segment["capacity_span"],
            "positive_capacity_deltas": baseline_segment["positive_capacity_deltas"],
            "negative_capacity_deltas": baseline_segment["negative_capacity_deltas"],
            "current_signs": ";".join(baseline_segment["current_signs"]),
            "capacity_status": baseline_status,
            "raw_scan_status": baseline_scan["scan_status"],
        }
    )

    supplemental: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda item: int(item["rpt_index"])):
        source, start, end = _mapped_step4_ref(row["step4_segment_ref"])
        scan = capacity_scans[source]
        segment = _segment_capacity(scan, source, start, end)
        capacity_ok, capacity_status = _capacity_is_eligible(segment)
        step4 = row["step4_result"]
        step10 = row["step10_result"]
        step4_valid = bool(step4 and step4.get("valid"))
        step10_valid = bool(step10 and step10.get("valid"))
        temperature_complete = bool(
            step4_valid and step4.get("temperature_complete")
        )
        if step4_valid and (
            step4["row_start"] != start or step4["row_end"] != end
        ):
            raise ValueError(
                f"Raw Step 4 boundaries disagree for {CELL2} RPT{row['rpt_index']}."
            )
        step10_mapping = phase2._parse_segment(row, 10)
        if step10_valid and step10_mapping is not None and (
            step10["row_start"] != step10_mapping[1]
            or step10["row_end"] != step10_mapping[2]
        ):
            raise ValueError(
                f"Raw Step 10 boundaries disagree for {CELL2} RPT{row['rpt_index']}."
            )
        reference_valid = bool(
            baseline_ok
            and capacity_ok
            and baseline_capacity
            and float(baseline_capacity) > 0
        )
        soh = (
            100.0 * float(segment["capacity_span"]) / float(baseline_capacity)
            if reference_valid
            else None
        )
        label_status = (
            "calculated_from_measured_capacity"
            if soh is not None and 50.0 <= soh <= 120.0
            else "candidate_outside_review_range"
            if soh is not None
            else "not_calculated"
        )
        if label_status != "calculated_from_measured_capacity":
            soh = None

        row["soh_percent"] = soh
        row["label_scope"] = "supplemental_provisional_part3_capacity_ratio"
        row["physics_features"] = None
        if step4_valid and temperature_complete:
            row["physics_features"] = build_physics_features(
                step4["voltage_raw"],
                step4["current_raw"],
                step4["temperature_raw"],
            )
        fusion_eligible = bool(
            soh is not None
            and step4_valid
            and step10_valid
            and temperature_complete
            and row["physics_features"] is not None
        )
        source_row = {
            "role": "rpt_candidate",
            "physical_cell_id": CELL2,
            "measurement_id": f"RPT{row['rpt_index']}",
            "source_file": source,
            "source_row_start": start,
            "source_row_end": end,
            "step4_rows": segment["step4_rows"],
            "capacity_finite_rows": segment["capacity_finite_rows"],
            "capacity_span_native": segment["capacity_span"],
            "positive_capacity_deltas": segment["positive_capacity_deltas"],
            "negative_capacity_deltas": segment["negative_capacity_deltas"],
            "current_signs": ";".join(segment["current_signs"]),
            "capacity_status": capacity_status,
            "raw_scan_status": scan["scan_status"],
        }
        capacity_rows.append(source_row)
        supplemental.append(
            {
                "row": row,
                "label": {
                    "dataset_part": "3",
                    "group_id": row["group"],
                    "physical_cell_id": CELL2,
                    "measurement_id": f"RPT{row['rpt_index']}",
                    "soh_percent": soh,
                    "validation_status": label_status,
                    "label_scope": "supplemental_provisional_part3_capacity_ratio",
                    "label_method": (
                        "100 * mapped Step 4 Amphr span / earliest mapped BoL "
                        "Step 4 Amphr span"
                    ),
                    "source_file": source,
                    "source_row_start": start,
                    "source_row_end": end,
                    "step4_segment_ref": row["step4_segment_ref"],
                    "step10_segment_ref": row["step10_segment_ref"],
                    "step4_observed_rows": (
                        step4.get("observed_rows") if step4 else None
                    ),
                    "step10_source_row_start": (
                        step10.get("row_start")
                        if step10 and step10.get("row_start") is not None
                        else step10_mapping[1]
                        if step10_mapping
                        else None
                    ),
                    "step10_source_row_end": (
                        step10.get("row_end")
                        if step10 and step10.get("row_end") is not None
                        else step10_mapping[2]
                        if step10_mapping
                        else None
                    ),
                    "step10_expected_rows": (
                        step10_mapping[2] - step10_mapping[1] + 1
                        if step10_mapping
                        else None
                    ),
                    "step10_observed_rows": (
                        step10.get("observed_rows") if step10 else None
                    ),
                    "bol_reference_file": baseline_source,
                    "bol_source_row_start": baseline_start,
                    "bol_source_row_end": baseline_end,
                    "reference_capacity_span_native": baseline_capacity,
                    "capacity_span_native": segment["capacity_span"],
                    "step4_raw_status": step4["status"] if step4 else "not_scanned",
                    "step10_raw_status": (
                        step10["status"] if step10 else "not_scanned"
                    ),
                    "temperature_complete": temperature_complete,
                    "fusion_eligible": fusion_eligible,
                    "identity_provenance": (
                        "user-confirmed Group 10 identity; Guide omits explicit cell "
                        "number; manifest mapping retained; candidate remains provisional"
                    ),
                    "historical_label_file_modified": False,
                },
            }
        )
    return supplemental, capacity_rows


def _infer_cell2(
    supplemental: list[dict[str, Any]], split_version: str
) -> list[dict[str, Any]]:
    fusion_checkpoint = torch.load(
        FUSION_CHECKPOINT, map_location="cpu", weights_only=False
    )
    if fusion_checkpoint.get("split_version") != split_version:
        raise ValueError("Saved fusion checkpoint split does not match the locked split.")
    if fusion_checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError("Saved fusion checkpoint model version is unexpected.")
    if fusion_checkpoint.get("input_schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError("Saved fusion checkpoint input schema is unexpected.")
    if tuple(fusion_checkpoint.get("physics_feature_names", ())) != PHYSICS_FEATURE_NAMES:
        raise ValueError("Saved fusion checkpoint physics schema is unexpected.")

    fusion = ThreeEncoderSoH()
    fusion.load_state_dict(fusion_checkpoint["model_state_dict"], strict=True)
    fusion.eval()
    historical = torch.load(
        BASELINE_CHECKPOINT, map_location="cpu", weights_only=False
    )
    baseline = SavedCNN1D()
    baseline.load_state_dict(historical["model_state_dict"], strict=True)
    baseline.eval()

    rows = []
    with torch.inference_mode():
        for item in supplemental:
            row = item["row"]
            label = item["label"]
            if (
                label["validation_status"] != "calculated_from_measured_capacity"
                or not row["step4_result"]
                or not row["step4_result"].get("valid")
            ):
                continue
            discharge = np.asarray(row["step4_result"]["curve"], dtype=np.float32)
            baseline_input = (
                discharge - _to_numpy(historical["x_mean"])
            ) / _to_numpy(historical["x_scale"])
            baseline_value = baseline(
                torch.from_numpy(baseline_input[None, None, :])
            )
            baseline_prediction = (
                float(baseline_value.reshape(-1)[0])
                * float(historical["y_scale"])
                + float(historical["y_mean"])
            )
            fusion_prediction = None
            if label["fusion_eligible"]:
                charge = np.asarray(row["step10_result"]["curve"], dtype=np.float32)
                physics = np.asarray(
                    [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES],
                    dtype=np.float32,
                )
                discharge_input = (
                    discharge - _to_numpy(fusion_checkpoint["discharge_mean"])
                ) / _to_numpy(fusion_checkpoint["discharge_scale"])
                charge_input = (
                    charge - _to_numpy(fusion_checkpoint["charge_mean"])
                ) / _to_numpy(fusion_checkpoint["charge_scale"])
                physics_input = (
                    physics - _to_numpy(fusion_checkpoint["physics_mean"])
                ) / _to_numpy(fusion_checkpoint["physics_scale"])
                fusion_value = fusion(
                    torch.from_numpy(discharge_input[None, :]),
                    torch.from_numpy(charge_input[None, :]),
                    torch.from_numpy(physics_input[None, :]),
                )
                fusion_prediction = (
                    float(fusion_value.reshape(-1)[0])
                    * float(fusion_checkpoint["target_scale"])
                    + float(fusion_checkpoint["target_mean"])
                )
            if not math.isfinite(baseline_prediction) or (
                fusion_prediction is not None
                and not math.isfinite(fusion_prediction)
            ):
                raise ValueError(f"Non-finite saved-checkpoint output for {label['measurement_id']}.")
            rows.append(
                {
                    **label,
                    "reference_soh_percent": label["soh_percent"],
                    "partition": "external_test",
                    "train_fold_mean_soh_percent": float(
                        fusion_checkpoint["target_mean"]
                    ),
                    "historical_cnn1d_predicted_soh_percent": baseline_prediction,
                    "fusion_v2_predicted_soh_percent": fusion_prediction,
                    "fusion_eligible": bool(label["fusion_eligible"]),
                    "fusion_exclusion_reason": (
                        ""
                        if label["fusion_eligible"]
                        else label["step4_raw_status"]
                        if label["step4_raw_status"] != "verified_raw_segment"
                        else label["step10_raw_status"]
                        if label["step10_raw_status"] != "verified_raw_segment"
                        else "missing_or_nonfinite_step4_temperature"
                        if not label["temperature_complete"]
                        else "missing_valid_provisional_soh_reference"
                    ),
                    "prediction_source": "saved_checkpoints_inference_only",
                    "model_retrained": False,
                }
            )
    return rows


def _metric_row(
    rows: list[dict[str, Any]], model: str, prediction_column: str
) -> dict[str, Any]:
    actual = np.asarray(
        [float(row["reference_soh_percent"]) for row in rows], dtype=np.float64
    )
    predicted = np.asarray(
        [float(row[prediction_column]) for row in rows], dtype=np.float64
    )
    return {
        "model": model,
        "partition": rows[0]["partition"],
        "metric_scope": "overall",
        "evaluation_population": "same_keys_as_fusion",
        "observation_count": len(rows),
        "physical_cell_count": len({row["physical_cell_id"] for row in rows}),
        **_metrics(actual, predicted),
    }


def _verify_historical_outputs(
    original_rows: list[dict[str, Any]],
    baseline_rows: dict[tuple[str, str, str], dict[str, Any]],
) -> None:
    for row in original_rows:
        key = _key_from_row(row)
        if key not in baseline_rows:
            raise ValueError(f"Historical CNN1D output is missing key {key}.")
        baseline = baseline_rows[key]
        if not math.isclose(
            float(row["historical_cnn1d_predicted_soh_percent"]),
            float(baseline["predicted_soh_percent"]),
            rel_tol=0.0,
            abs_tol=1e-4,
        ):
            raise ValueError(f"CNN1D prediction exports disagree for {key}.")
        if not math.isclose(
            float(row["reference_soh_percent"]),
            float(baseline["reference_soh_percent"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"CNN1D reference label exports disagree for {key}.")


def _overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for partition in PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        if not selected:
            raise ValueError(f"No shared-subset predictions for {partition}.")
        for model, column in (
            ("train_fold_mean", "train_fold_mean_soh_percent"),
            ("historical_cnn1d", "historical_cnn1d_predicted_soh_percent"),
            ("three_encoder_fusion", "fusion_v2_predicted_soh_percent"),
        ):
            output.append(_metric_row(selected, model, column))
    return output


def _full_cnn_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        raise ValueError("No eligible full-coverage CNN1D predictions were found.")
    output = []
    partitions = sorted({row["partition"] for row in rows})
    for partition in partitions:
        selected = [row for row in rows if row["partition"] == partition]
        overall = _metric_row(
            selected, "historical_cnn1d", "predicted_soh_percent"
        )
        overall["evaluation_population"] = "all_step4_eligible_external_labels"
        overall["physical_cell_id"] = ""
        output.append(overall)
        for cell in sorted({row["physical_cell_id"] for row in selected}):
            cell_rows = [
                row for row in selected if row["physical_cell_id"] == cell
            ]
            per_cell = _metric_row(
                cell_rows, "historical_cnn1d", "predicted_soh_percent"
            )
            per_cell["metric_scope"] = "per_cell"
            per_cell["evaluation_population"] = "all_step4_eligible_external_labels"
            per_cell["physical_cell_id"] = cell
            output.append(per_cell)
    return output


def _external_coverage(
    supplemental_predictions: list[dict[str, Any]],
    full_cnn_rows: list[dict[str, Any]],
    supplemental_labels: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    split = _read_csv(SPLIT_PATH)
    cells = sorted(
        split.loc[split["partition"] == "external_test", "physical_cell_id"].unique()
    )
    if len(cells) != 12:
        raise ValueError(f"Locked split must contain 12 external cells, found {len(cells)}.")
    manifest = _read_csv(MANIFEST_PATH)
    manifest = manifest[
        (manifest["part"].astype(str) == "3")
        & manifest["physical_cell_id"].isin(cells)
    ].copy()
    availability = _read_csv(PHASE2_OUTPUT / "modality_availability.csv")
    feature_manifest = _read_csv(
        PHASE2_OUTPUT / "multimodal_feature_manifest.csv"
    )
    for frame in (manifest, availability, feature_manifest):
        frame["_key"] = [
            _key_from_row(row)
            for row in frame.to_dict(orient="records")
        ]
    availability_rows = _records_by_key(availability)
    feature_rows = _records_by_key(feature_manifest)
    predicted_keys = {_key_from_row(row) for row in supplemental_predictions}
    full_cnn_keys = {_key_from_row(row) for row in full_cnn_rows}
    supplemental_label_map = {
        _key_from_row(row): row
        for row in supplemental_labels
        if row["validation_status"] == "calculated_from_measured_capacity"
    }
    fusion_keys = {
        _key_from_row(row)
        for row in _read_csv(PHASE2_OUTPUT / "fusion_predictions.csv").to_dict(
            orient="records"
        )
    } | predicted_keys

    coverage_rows = []
    exclusion_rows = []
    for cell in cells:
        cell_manifest = manifest[manifest["physical_cell_id"] == cell]
        cell_keys = set(cell_manifest["_key"])
        supplemental_keys = cell_keys.intersection(supplemental_label_map)
        mapped_step4 = int((cell_manifest["step4_status"] == "verified").sum())
        mapped_step10 = int((cell_manifest["step10_status"] == "verified").sum())
        valid_soh = sum(
            1
            for key in cell_keys
            if key in feature_rows
            and str(feature_rows[key]["label_status"])
            == "calculated_from_measured_capacity"
        )
        valid_step4 = 0
        valid_step10_raw = 0
        valid_physics = 0
        complete_modalities = 0
        reasons = []
        for key in sorted(cell_keys):
            feature = feature_rows.get(key, {})
            available = availability_rows.get(key, {})
            supplemental_label = supplemental_label_map.get(key)
            supplemental = supplemental_label is not None
            step4_status = (
                str(feature.get("step4_raw_status", "not_scanned"))
                if feature
                else "not_scanned"
            )
            step10_status = (
                str(feature.get("step10_raw_status", "not_scanned"))
                if feature
                else "not_scanned"
            )
            if supplemental_label is not None:
                step4_status = str(
                    supplemental_label.get("step4_raw_status", step4_status)
                )
                step10_status = str(
                    supplemental_label.get("step10_raw_status", step10_status)
                )
            step4_ok = step4_status == "verified_raw_segment"
            step10_ok = step10_status == "verified_raw_segment"
            temperature_ok = (
                str(available.get("temperature_complete", "")).casefold()
                == "true"
                or (
                    supplemental_label is not None
                    and str(supplemental_label["temperature_complete"]).casefold()
                    == "true"
                )
            )
            fusion_ok = key in fusion_keys
            valid_step4 += int(step4_ok)
            valid_step10_raw += int(step10_ok)
            valid_physics += int(step4_ok and temperature_ok)
            complete_modalities += int(fusion_ok)

            old_reason = str(
                feature.get(
                    "fusion_exclusion_reason",
                    available.get("exclusion_reason", ""),
                )
            )
            if supplemental and fusion_ok:
                status = "resolved_by_supplemental_provisional_label"
                exact_reason = ""
            elif supplemental:
                status = "supplemental_reference_but_incomplete_modalities"
                exact_reason = (
                    step4_status
                    if not step4_ok
                    else step10_status
                    if not step10_ok
                    else "missing_or_nonfinite_step4_temperature"
                    if not temperature_ok
                    else "incomplete_fusion_modalities"
                )
                reasons.append(exact_reason)
            elif fusion_ok:
                status = "eligible_fusion_prediction"
                exact_reason = ""
            else:
                status = "excluded"
                exact_reason = old_reason or "no_validated_soh_label"
                reasons.append(exact_reason)
            exclusion_rows.append(
                {
                    "dataset_part": "3",
                    "physical_cell_id": cell,
                    "measurement_id": f"RPT{key[2]}",
                    "original_fusion_exclusion_reason": old_reason,
                    "current_status": status,
                    "exact_exclusion_reason": exact_reason,
                    "supplemental_label_used": supplemental,
                    "eligible_fusion_prediction": key in fusion_keys,
                    "eligible_full_coverage_cnn1d_prediction": key in full_cnn_keys,
                    "step4_mapping_status": (
                        "verified"
                        if feature.get("step4_segment_ref")
                        else "verified"
                        if cell_manifest.loc[
                            cell_manifest["_key"] == key, "step4_status"
                        ].iloc[0]
                        == "verified"
                        else "unverified"
                    ),
                    "step4_raw_status": step4_status,
                    "step10_mapping_status": (
                        "verified"
                        if feature.get("step10_segment_ref")
                        else "verified"
                        if cell_manifest.loc[
                            cell_manifest["_key"] == key, "step10_status"
                        ].iloc[0]
                        == "verified"
                        else "unverified"
                    ),
                    "step10_raw_status": step10_status,
                    "temperature_complete": temperature_ok,
                }
            )
        label_count = valid_soh + len(supplemental_keys)
        fusion_count = sum(key in fusion_keys for key in cell_keys)
        cnn_count = sum(key in full_cnn_keys for key in cell_keys)
        coverage_rows.append(
            {
                "physical_cell_id": cell,
                "external_split_cell_count": 12,
                "mapped_step4_observations": mapped_step4,
                "available_step4_observations": valid_step4,
                "valid_soh_reference_observations": label_count,
                "verified_step10_charge_observations": mapped_step10,
                "valid_step10_charge_measurements": valid_step10_raw,
                "valid_physics_feature_observations": valid_physics,
                "complete_modality_observations": complete_modalities,
                "eligible_fusion_predictions": fusion_count,
                "eligible_full_coverage_cnn1d_predictions": cnn_count,
                "exact_exclusion_reasons": "; ".join(sorted(set(reasons))),
            }
        )
    return coverage_rows, exclusion_rows


def _feature_map(
    path: Path, columns: tuple[str, ...]
) -> dict[tuple[str, str, str], np.ndarray]:
    frame = _read_csv(path)
    records: dict[tuple[str, str, str], np.ndarray] = {}
    for row in frame.to_dict(orient="records"):
        key = _key_from_row(row)
        if key in records:
            raise ValueError(f"Duplicate feature key in {path.name}: {key}")
        values = np.asarray([float(row[column]) for column in columns], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite feature values in {path.name} for {key}.")
        records[key] = values
    return records


def _correlation(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.std() == 0 or right_values.std() == 0:
        return None
    return float(np.corrcoef(left_values, right_values)[0, 1])


def _generalization_statistics(
    output_dir: Path,
    shared_rows: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    physics_map = _feature_map(
        PHASE2_OUTPUT / "physics_features.csv", tuple(PHYSICS_FEATURE_NAMES)
    )
    discharge_map = _feature_map(
        PHASE2_OUTPUT / "discharge_features.csv", CURVE_COLUMNS
    )
    charge_map = _feature_map(
        PHASE2_OUTPUT / "charge_features.csv", CURVE_COLUMNS
    )
    shared_keys = {_key_from_row(row) for row in shared_rows}
    for item in supplemental:
        key = _key_from_row(item["label"])
        if key not in shared_keys:
            continue
        row = item["row"]
        physics_map[key] = np.asarray(
            [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES],
            dtype=np.float64,
        )
        discharge_map[key] = np.asarray(
            row["step4_result"]["curve"], dtype=np.float64
        )
        charge_map[key] = np.asarray(
            row["step10_result"]["curve"], dtype=np.float64
        )

    partition_by_key = {
        _key_from_row(row): row["partition"] for row in shared_rows
    }
    train_keys = [
        key for key, partition in partition_by_key.items() if partition == "train"
    ]
    external_keys = [
        key
        for key, partition in partition_by_key.items()
        if partition == "external_test"
    ]

    def matrix_for(
        mapping: dict[tuple[str, str, str], np.ndarray],
        keys: list[tuple[str, str, str]],
        name: str,
    ) -> np.ndarray:
        missing = [key for key in keys if key not in mapping]
        if missing:
            raise ValueError(f"{name} features are missing shared keys: {missing[:5]}")
        return np.vstack([mapping[key] for key in keys])

    training = {
        "physics": matrix_for(physics_map, train_keys, "physics"),
        "discharge": matrix_for(discharge_map, train_keys, "discharge"),
        "charge": matrix_for(charge_map, train_keys, "charge"),
    }
    external = {
        "physics": matrix_for(physics_map, external_keys, "physics"),
        "discharge": matrix_for(discharge_map, external_keys, "discharge"),
        "charge": matrix_for(charge_map, external_keys, "charge"),
    }
    checkpoint = torch.load(
        FUSION_CHECKPOINT, map_location="cpu", weights_only=False
    )
    scaler_deltas: dict[str, float] = {}
    for name in ("physics", "discharge", "charge"):
        saved_mean = _to_numpy(checkpoint[f"{name}_mean"])
        saved_scale = _to_numpy(checkpoint[f"{name}_scale"])
        mean_delta = float(np.max(np.abs(training[name].mean(axis=0) - saved_mean)))
        scale_delta = float(np.max(np.abs(training[name].std(axis=0) - saved_scale)))
        scaler_deltas[f"{name}_mean_abs_delta"] = mean_delta
        scaler_deltas[f"{name}_scale_abs_delta"] = scale_delta
        if mean_delta > 1e-4 or scale_delta > 1e-4:
            raise ValueError(
                f"Saved {name} scaler differs from reconstructed training features."
            )
    train_targets = np.asarray(
        [
            float(row["reference_soh_percent"])
            for row in shared_rows
            if row["partition"] == "train"
        ],
        dtype=np.float64,
    )
    target_mean_delta = abs(float(train_targets.mean()) - checkpoint["target_mean"])
    target_scale_delta = abs(float(train_targets.std()) - checkpoint["target_scale"])
    scaler_deltas["target_mean_abs_delta"] = target_mean_delta
    scaler_deltas["target_scale_abs_delta"] = target_scale_delta
    if target_mean_delta > 1e-4 or target_scale_delta > 1e-4:
        raise ValueError("Saved target scaler differs from reconstructed train labels.")

    feature_rows: list[dict[str, Any]] = []
    for index, name in enumerate(PHYSICS_FEATURE_NAMES):
        train_values = training["physics"][:, index]
        external_values = external["physics"][:, index]
        train_scale = float(train_values.std())
        feature_rows.append(
            {
                "feature_scope": "physics_feature",
                "feature_name": name,
                "train_observation_count": len(train_values),
                "external_observation_count": len(external_values),
                "train_mean": float(train_values.mean()),
                "train_std": train_scale,
                "external_mean": float(external_values.mean()),
                "external_std": float(external_values.std()),
                "standardized_mean_shift_vs_train": (
                    float((external_values.mean() - train_values.mean()) / train_scale)
                    if train_scale > 0
                    else None
                ),
                "mean_curve_profile_mae": "",
                "max_curve_profile_abs_difference": "",
            }
        )
    train_charge = training["charge"]
    external_charge = external["charge"]
    train_charge_mean = train_charge.mean(axis=0)
    external_charge_mean = external_charge.mean(axis=0)
    train_charge_std = float(train_charge.std())
    feature_rows.append(
        {
            "feature_scope": "charge_curve_profile",
            "feature_name": "128_point_voltage_progress",
            "train_observation_count": len(train_charge),
            "external_observation_count": len(external_charge),
            "train_mean": float(train_charge.mean()),
            "train_std": train_charge_std,
            "external_mean": float(external_charge.mean()),
            "external_std": float(external_charge.std()),
            "standardized_mean_shift_vs_train": (
                float((external_charge.mean() - train_charge.mean()) / train_charge_std)
                if train_charge_std > 0
                else None
            ),
            "mean_curve_profile_mae": float(
                np.mean(np.abs(external_charge_mean - train_charge_mean))
            ),
            "max_curve_profile_abs_difference": float(
                np.max(np.abs(external_charge_mean - train_charge_mean))
            ),
        }
    )

    feature_manifest = _read_csv(
        PHASE2_OUTPUT / "multimodal_feature_manifest.csv"
    )
    duration_by_key = {}
    for row in feature_manifest.to_dict(orient="records"):
        value = row.get("step4_duration_native_audit_only", "")
        if value not in ("", None):
            duration_by_key[_key_from_row(row)] = float(value)
    target_by_key = {
        _key_from_row(row): float(row["reference_soh_percent"])
        for row in shared_rows
    }
    for item in supplemental:
        key = _key_from_row(item["label"])
        duration = item["row"]["step4_result"]["duration_native"]
        if duration is not None:
            duration_by_key[key] = float(duration)
    duration_rows = []
    for partition in ("train", "external_test"):
        keys = [
            key
            for key, value in partition_by_key.items()
            if value == partition and key in duration_by_key
        ]
        correlation = _correlation(
            [duration_by_key[key] for key in keys],
            [target_by_key[key] for key in keys],
        )
        duration_rows.append(
            {
                "feature_scope": "leakage_audit",
                "feature_name": "step4_duration_native_audit_only",
                "partition": partition,
                "observation_count": len(keys),
                "pearson_correlation_with_same_rpt_soh": correlation,
                "included_in_model_input": False,
                "exclusion_rule": ",".join(sorted(EXCLUDED_FEATURES)),
            }
        )

    per_cell_rows = []
    external_shared = [
        row for row in shared_rows if row["partition"] == "external_test"
    ]
    for cell in sorted({row["physical_cell_id"] for row in external_shared}):
        selected = [
            row for row in external_shared if row["physical_cell_id"] == cell
        ]
        actual = np.asarray(
            [float(row["reference_soh_percent"]) for row in selected]
        )
        cnn = np.asarray(
            [float(row["historical_cnn1d_predicted_soh_percent"]) for row in selected]
        )
        fusion = np.asarray(
            [float(row["fusion_v2_predicted_soh_percent"]) for row in selected]
        )
        per_cell_rows.append(
            {
                "physical_cell_id": cell,
                "observation_count": len(selected),
                "cnn1d_mae_soh_percentage_points": float(
                    np.mean(np.abs(cnn - actual))
                ),
                "cnn1d_mean_signed_error_soh_percentage_points": float(
                    np.mean(cnn - actual)
                ),
                "fusion_mae_soh_percentage_points": float(
                    np.mean(np.abs(fusion - actual))
                ),
                "fusion_mean_signed_error_soh_percentage_points": float(
                    np.mean(fusion - actual)
                ),
                "fusion_max_absolute_error_soh_percentage_points": float(
                    np.max(np.abs(fusion - actual))
                ),
            }
        )
    _write_rows(output_dir / "generalization_feature_shift.csv", feature_rows)
    _write_rows(
        output_dir / "generalization_duration_leakage_audit.csv", duration_rows
    )
    _write_rows(output_dir / "generalization_per_cell_errors.csv", per_cell_rows)
    return feature_rows, duration_rows, scaler_deltas


def _write_generalization_report(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    exclusion_rows: list[dict[str, Any]],
    full_cnn_rows: list[dict[str, Any]],
    shared_rows: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    duration_rows: list[dict[str, Any]],
    scaler_deltas: dict[str, float],
) -> None:
    by_reason: dict[str, int] = {}
    for row in exclusion_rows:
        reason = row["exact_exclusion_reason"]
        if reason:
            by_reason[reason] = by_reason.get(reason, 0) + 1
    external_fusion = [
        row for row in shared_rows if row["partition"] == "external_test"
    ]
    fusion_errors = np.asarray(
        [
            float(row["fusion_v2_predicted_soh_percent"])
            - float(row["reference_soh_percent"])
            for row in external_fusion
        ],
        dtype=np.float64,
    )
    cnn_errors = np.asarray(
        [
            float(row["historical_cnn1d_predicted_soh_percent"])
            - float(row["reference_soh_percent"])
            for row in external_fusion
        ],
        dtype=np.float64,
    )
    cell_errors = []
    for cell in sorted({row["physical_cell_id"] for row in external_fusion}):
        records = [row for row in external_fusion if row["physical_cell_id"] == cell]
        errors = np.asarray(
            [
                float(row["fusion_v2_predicted_soh_percent"])
                - float(row["reference_soh_percent"])
                for row in records
            ],
            dtype=np.float64,
        )
        cell_errors.append(
            (
                cell,
                len(records),
                float(np.mean(np.abs(errors))),
                float(np.mean(errors)),
            )
        )
    worst_cell = max(cell_errors, key=lambda row: row[2]) if cell_errors else None
    max_error_row = max(
        external_fusion,
        key=lambda row: abs(
            float(row["fusion_v2_predicted_soh_percent"])
            - float(row["reference_soh_percent"])
        ),
    )
    metrics_by_key = {
        (row["partition"], row["model"]): row
        for row in metric_rows
        if row["metric_scope"] == "overall"
    }
    missing_lines = [
        f"- `{reason}`: {count} observations."
        for reason, count in sorted(by_reason.items())
    ] or ["- No external fusion exclusions remain."]
    coverage_lines = [
        f"| {row['physical_cell_id']} | {row['valid_soh_reference_observations']} | "
        f"{row['valid_physics_feature_observations']} | "
        f"{row['complete_modality_observations']} | "
        f"{row['eligible_fusion_predictions']} | "
        f"{row['eligible_full_coverage_cnn1d_predictions']} |"
        for row in coverage_rows
    ]
    cell2_coverage = next(
        row for row in coverage_rows if row["physical_cell_id"] == CELL2
    )
    cell2_predictions = cell2_coverage["eligible_fusion_predictions"]
    full_cnn_count = len(full_cnn_rows)
    cell_error_line = (
        f"`{worst_cell[0]}` had the largest fusion per-cell MAE "
        f"({worst_cell[2]:.4f} pp across {worst_cell[1]} observations; "
        f"mean signed error {worst_cell[3]:+.4f} pp)."
        if worst_cell
        else "No per-cell fusion errors were available."
    )
    absolute_max = abs(
        float(max_error_row["fusion_v2_predicted_soh_percent"])
        - float(max_error_row["reference_soh_percent"])
    )
    original_baseline = metrics_by_key[("external_test", "historical_cnn1d")]
    original_fusion = metrics_by_key[("external_test", "three_encoder_fusion")]
    full_cnn_metric = _full_cnn_metrics(full_cnn_rows)[0]
    charge_shift = next(
        row for row in feature_rows if row["feature_scope"] == "charge_curve_profile"
    )
    physics_shifts = [
        row for row in feature_rows if row["feature_scope"] == "physics_feature"
    ]
    temperature_shift = next(
        row
        for row in physics_shifts
        if row["feature_name"] == "discharge_temperature_rise_native"
    )
    current_shift = next(
        row
        for row in physics_shifts
        if row["feature_name"] == "mean_absolute_discharge_current_native"
    )
    duration_text = "; ".join(
        (
            f"{row['partition']} n={row['observation_count']}, "
            f"r={row['pearson_correlation_with_same_rpt_soh']:.4f}"
        )
        if row["pearson_correlation_with_same_rpt_soh"] is not None
        else f"{row['partition']} n={row['observation_count']}, r=n/a"
        for row in duration_rows
    )
    lines = [
        "# SoH overall evaluation and external coverage audit",
        "",
        "## Scope and safeguards",
        "",
        "- This is same-RPT SoH estimation, not future-RPT forecasting. The locked 12/4/12 physical-cell split was read-only.",
        "- No model was trained, fine-tuned, or selected using external results. The supplemental G10-Cell2 predictions use the saved CNN1D and fusion checkpoints with their saved scalers.",
        "- Historical Part 3 candidate labels were not edited. G10-Cell2's 10 Step 4 capacity ratios are recorded only in this versioned audit and remain provisional.",
        "- Parts 1-2 used Maccor 4200; Part 3 used Maccor 4000. Part 3 remains an external/domain-shift cohort; its temperature conditions are not established by its README.",
        "- PyDMA fitting code, references, labels, quality flags, and existing artifacts were not modified.",
        "",
        "## Primary overall metrics",
        "",
        "| Partition | Model | Observations | Cells | MAE (SOH pp) | RMSE (SOH pp) | R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        if row["metric_scope"] != "overall":
            continue
        lines.append(
            f"| {row['partition']} | {row['model']} | {row['observation_count']} | "
            f"{row['physical_cell_count']} | {row['mae_soh_percentage_points']:.6f} | "
            f"{row['rmse_soh_percentage_points']:.6f} | {row['r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"External complete-modality shared population: {len(external_fusion)} observations across "
            f"{len({row['physical_cell_id'] for row in external_fusion})} cells. "
            f"On that same population, CNN1D MAE/RMSE/R² were "
            f"{original_baseline['mae_soh_percentage_points']:.4f}/"
            f"{original_baseline['rmse_soh_percentage_points']:.4f}/"
            f"{original_baseline['r2']:.6f}; fusion was "
            f"{original_fusion['mae_soh_percentage_points']:.4f}/"
            f"{original_fusion['rmse_soh_percentage_points']:.4f}/"
            f"{original_fusion['r2']:.6f}. Fusion does not outperform CNN1D on this external shared population.",
            f"Full Step 4-eligible CNN1D coverage: {full_cnn_count} observations across "
            f"{len({row['physical_cell_id'] for row in full_cnn_rows})} cells; overall MAE/RMSE/R² "
            f"{full_cnn_metric['mae_soh_percentage_points']:.4f}/"
            f"{full_cnn_metric['rmse_soh_percentage_points']:.4f}/"
            f"{full_cnn_metric['r2']:.6f}. This is a different population from the fusion shared subset.",
            "",
            "## External 12-cell coverage",
            "",
            "| Cell | Valid SOH references | Valid physics availability | Complete modalities | Fusion predictions | CNN1D predictions |",
            "|---|---:|---:|---:|---:|---:|",
            *coverage_lines,
            "",
            f"G10-Cell2 was the missing 12th fusion cell in the earlier report. Its canonical manifest maps 10 Step 4/Step 10 observations; the previous candidate-label table omitted them. Raw Step 4 passed for {cell2_coverage['available_step4_observations']}/10, Step 10 passed for {cell2_coverage['valid_step10_charge_measurements']}/10, and {cell2_predictions} have complete modalities and saved-checkpoint fusion predictions. RPT10 is excluded from fusion because its Step 10 current direction is unexpected; its Step 4-only CNN1D prediction remains eligible. The Group 10 physical-cell identity is user-confirmed; the Guide does not specify the cell number, so this is not independent identity verification. These references use the earliest mapped BoL Step 4 capacity span and are supplemental provisional labels, not official Part 3 labels.",
            "",
            "Valid physics availability is counted from a verified Step 4 segment with complete Step 4 temperature; it does not require Step 10. The current feature writer only emits physics vectors after Step 10 validation, so that stored `physics_features_complete` flag can undercount physics availability when charge is missing. Fusion still correctly requires complete modalities.",
            "",
            "### Exact external exclusions",
            "",
            *missing_lines,
            "",
            "Step 4-only CNN1D was evaluated on every external candidate with a validated provisional reference and a verified raw Step 4 segment. No missing inputs were imputed or fabricated.",
            "",
            "## Generalization-gap evidence",
            "",
            f"On the external shared subset, fusion mean signed error was {float(np.mean(fusion_errors)):+.4f} pp and maximum absolute error was {absolute_max:.4f} pp at `{max_error_row['physical_cell_id']}` / `{max_error_row['measurement_id']}`. CNN1D mean signed error was {float(np.mean(cnn_errors)):+.4f} pp.",
            cell_error_line,
            f"Mean Step 4 temperature rise was {temperature_shift['train_mean']:.3f} native units in training versus {temperature_shift['external_mean']:.3f} externally ({temperature_shift['standardized_mean_shift_vs_train']:+.2f} training standard deviations). Mean absolute current has a {current_shift['standardized_mean_shift_vs_train']:+.2f} SD shift, but the absolute means are {current_shift['train_mean']:.6f} and {current_shift['external_mean']:.6f}; its training standard deviation is nearly zero, so that standardized shift is sensitive to small native-unit differences. External mean charge-curve profile MAE was {charge_shift['mean_curve_profile_mae']:.4f} V over 128 normalized progress points; this is descriptive, not a significance test.",
            "- The external cohort uses a different cycling instrument (Maccor 4000) from Parts 1-2 (Maccor 4200), and its temperature protocol is not established by the README. This supports a protocol/domain-shift interpretation but does not establish a causal explanation.",
            "- Recomputed training-only feature and target scaler means/scales match the saved checkpoint within 1e-4. Maximum absolute differences: "
            + ", ".join(
                f"{name}={value:.2e}"
                for name, value in sorted(scaler_deltas.items())
            )
            + ". No external feature fitting or scaler updates were performed.",
            "- Complete-modality filtering selects observations with both validated Step 4 and Step 10 data plus complete Step 4 temperature. Exclusions are reported by observation in `external_exclusion_reasons.csv`; this selection can bias which RPTs/cells enter fusion.",
            f"Step 4 duration is not a model input. Audit-only Pearson correlations with same-RPT SoH were {duration_text}. It is excluded because of its strong relationship to same-RPT capacity; same-RPT capacity spans and target-derived quantities are not used as predictors.",
            "- Part 3 target ratios remain provisional, so external metrics are descriptive and must not be interpreted as validation on official labels.",
            "- Numeric physics/charge shifts are in `generalization_feature_shift.csv`; duration correlations are in `generalization_duration_leakage_audit.csv`; per-cell errors are in `generalization_per_cell_errors.csv`.",
            "",
            "## Reproducibility",
            "",
            "From the repository root, run:",
            "",
            "```powershell",
            r".venv-torch-cuda\Scripts\python.exe experiments\soh_forecast\bafuse_three_encoder_v1\evaluation_audit_v1\audit_soh_evaluation.py",
            r".venv-torch-cuda\Scripts\python.exe -m pytest -q tests\test_bafuse_evaluation_audit.py tests\test_dashboard_paths.py",
            "```",
            "",
            "The audit writes only to `evaluation_audit_v1`; it does not write model checkpoints, historical candidate labels, locked split files, or PyDMA artifacts.",
        ]
    )
    (output_dir / "fusion_generalization_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def run(output_dir: Path) -> None:
    supplemental, capacity_rows = _load_supplemental_cell2()
    _, split_version = phase2._load_observations()
    supplemental_predictions = _infer_cell2(supplemental, split_version)
    supplemental_fusion_predictions = [
        row for row in supplemental_predictions if row["fusion_eligible"]
    ]

    current = _read_csv(PHASE2_OUTPUT / "fusion_predictions.csv")
    current_rows = current.to_dict(orient="records")
    baseline_output = _read_csv(HYBRID_OUTPUT / "hybrid_battery_health_results.csv")
    baseline_part3 = baseline_output[
        baseline_output["dataset_part"].astype(str) == "3"
    ].copy()
    baseline_part3["_key"] = [
        _key_from_row(row) for row in baseline_part3.to_dict(orient="records")
    ]
    if baseline_part3["_key"].duplicated().any():
        raise ValueError("Historical CNN1D export contains duplicate Part 3 keys.")
    baseline_by_key = _records_by_key(baseline_part3)
    _verify_historical_outputs(
        [row for row in current_rows if str(row["dataset_part"]) == "3"],
        baseline_by_key,
    )

    original_metrics = _overall_rows(current_rows)
    for row in original_metrics:
        expected = EXPECTED_CURRENT_METRICS.get((row["partition"], row["model"]))
        if expected is None:
            continue
        observed = (
            row["mae_soh_percentage_points"],
            row["rmse_soh_percentage_points"],
            row["r2"],
        )
        if not np.allclose(observed, expected, rtol=0.0, atol=1e-5):
            raise ValueError(
                f"Saved prediction metric mismatch for "
                f"{row['partition']}/{row['model']}: {observed} != {expected}."
            )

    shared_rows = current_rows + supplemental_fusion_predictions
    if len({_key_from_row(row) for row in shared_rows}) != len(shared_rows):
        raise ValueError("Supplemental predictions duplicate a historical key.")
    metric_rows = _overall_rows(shared_rows)
    comparison_rows = []
    for row in shared_rows:
        actual = float(row["reference_soh_percent"])
        cnn = float(row["historical_cnn1d_predicted_soh_percent"])
        fusion = float(row["fusion_v2_predicted_soh_percent"])
        comparison_rows.append(
            {
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "measurement_id": row["measurement_id"],
                "partition": row["partition"],
                "reference_soh_percent": actual,
                "label_scope": row["label_scope"],
                "historical_cnn1d_predicted_soh_percent": cnn,
                "historical_cnn1d_absolute_error_pp": abs(cnn - actual),
                "fusion_v2_predicted_soh_percent": fusion,
                "fusion_v2_absolute_error_pp": abs(fusion - actual),
                "prediction_source": row.get(
                    "prediction_source", "historical_saved_evaluation"
                ),
            }
        )

    labels = _read_csv(PART3_LABELS_PATH)
    valid_labels = labels[
        (labels["dataset_part"].astype(str) == "3")
        & (labels["validation_status"] == "calculated_from_measured_capacity")
        & labels["measurement_id"].astype(str).str.startswith("RPT")
    ].copy()
    valid_labels["_key"] = [
        _key_from_row(row) for row in valid_labels.to_dict(orient="records")
    ]
    if valid_labels["_key"].duplicated().any():
        raise ValueError("Historical Part 3 candidate labels contain duplicate keys.")
    feature_manifest = _read_csv(
        PHASE2_OUTPUT / "multimodal_feature_manifest.csv"
    )
    feature_manifest = feature_manifest[
        feature_manifest["dataset_part"].astype(str) == "3"
    ].copy()
    feature_manifest["_key"] = [
        _key_from_row(row) for row in feature_manifest.to_dict(orient="records")
    ]
    manifest_by_key = _records_by_key(feature_manifest)
    cnn_by_key = baseline_by_key
    full_cnn_rows = []
    for label in valid_labels.to_dict(orient="records"):
        key = _key_from_row(label)
        if key not in manifest_by_key:
            raise ValueError(f"Missing raw Step 4 audit row for labeled key {key}.")
        if key not in cnn_by_key:
            raise ValueError(f"Missing historical CNN1D prediction for {key}.")
        manifest_row = manifest_by_key[key]
        if manifest_row["step4_raw_status"] != "verified_raw_segment":
            continue
        baseline_row = cnn_by_key[key]
        if not math.isclose(
            float(label["soh_percent"]),
            float(baseline_row["reference_soh_percent"]),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(f"Candidate label differs from CNN1D target for {key}.")
        full_cnn_rows.append(
            {
                "dataset_part": "3",
                "physical_cell_id": label["physical_cell_id"],
                "measurement_id": label["measurement_id"],
                "partition": "external_test",
                "reference_soh_percent": float(label["soh_percent"]),
                "predicted_soh_percent": float(
                    baseline_row["predicted_soh_percent"]
                ),
                "label_scope": "provisional_external_candidate",
                "step4_raw_status": manifest_row["step4_raw_status"],
            }
        )
    for row in supplemental_predictions:
        full_cnn_rows.append(
            {
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "measurement_id": row["measurement_id"],
                "partition": row["partition"],
                "reference_soh_percent": row["soh_percent"],
                "predicted_soh_percent": row[
                    "historical_cnn1d_predicted_soh_percent"
                ],
                "label_scope": row["label_scope"],
                "step4_raw_status": "verified_raw_segment",
            }
        )
    full_cnn_metrics = _full_cnn_metrics(full_cnn_rows)
    coverage_rows, exclusion_rows = _external_coverage(
        supplemental_fusion_predictions,
        full_cnn_rows,
        [item["label"] for item in supplemental],
    )
    if sum(row["eligible_fusion_predictions"] for row in coverage_rows) != sum(
        row["partition"] == "external_test" for row in shared_rows
    ):
        raise ValueError("External fusion coverage does not match keyed predictions.")
    if sum(row["eligible_full_coverage_cnn1d_predictions"] for row in coverage_rows) != len(
        full_cnn_rows
    ):
        raise ValueError("CNN1D coverage does not match keyed predictions.")

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_rows, duration_rows, scaler_deltas = _generalization_statistics(
        output_dir, shared_rows, supplemental
    )
    _write_rows(output_dir / "g10_cell2_step4_capacity_scan.csv", capacity_rows)
    _write_rows(
        output_dir / "supplemental_g10_cell2_labels.csv",
        [item["label"] for item in supplemental],
    )
    _write_rows(
        output_dir / "supplemental_g10_cell2_predictions.csv",
        supplemental_predictions,
    )
    _write_rows(output_dir / "soh_overall_metrics.csv", metric_rows)
    _write_rows(output_dir / "external_12_cell_coverage.csv", coverage_rows)
    _write_rows(
        output_dir / "external_exclusion_reasons.csv", exclusion_rows
    )
    _write_rows(
        output_dir / "shared_subset_model_comparison.csv", comparison_rows
    )
    _write_rows(
        output_dir / "full_coverage_cnn1d_evaluation.csv", full_cnn_metrics
    )
    _write_rows(
        output_dir / "full_coverage_cnn1d_predictions.csv", full_cnn_rows
    )
    _write_generalization_report(
        output_dir,
        metric_rows,
        coverage_rows,
        exclusion_rows,
        full_cnn_rows,
        comparison_rows,
        feature_rows,
        duration_rows,
        scaler_deltas,
    )
    _write_final_report(
        output_dir,
        metric_rows,
        coverage_rows,
        full_cnn_metrics,
        supplemental,
        split_version,
    )
    print(
        f"Wrote overall evaluation audit to {output_dir}; "
        f"{len(shared_rows)} shared-key predictions, "
        f"{len(full_cnn_rows)} full-coverage CNN1D external predictions."
    )


def _write_final_report(
    output_dir: Path,
    metric_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    full_cnn_metrics: list[dict[str, Any]],
    supplemental: list[dict[str, Any]],
    split_version: str,
) -> None:
    external_rows = [
        row for row in metric_rows if row["partition"] == "external_test"
    ]
    full_overall = next(
        row for row in full_cnn_metrics if row["metric_scope"] == "overall"
    )
    fusion_external = next(
        row for row in external_rows if row["model"] == "three_encoder_fusion"
    )
    cnn_shared = next(
        row for row in external_rows if row["model"] == "historical_cnn1d"
    )
    eligible_cells = sum(
        row["eligible_fusion_predictions"] > 0 for row in coverage_rows
    )
    cell2_step4_valid = sum(
        item["label"]["step4_raw_status"] == "verified_raw_segment"
        for item in supplemental
    )
    cell2_step10_valid = sum(
        item["label"]["step10_raw_status"] == "verified_raw_segment"
        for item in supplemental
    )
    cell2_fusion_eligible = sum(
        bool(item["label"]["fusion_eligible"]) for item in supplemental
    )
    lines = [
        "# Corrected SoH evaluation report",
        "",
        "## Summary",
        "",
        f"- Locked split `{split_version}` was not changed: 12 train / 4 validation / 12 external physical cells.",
        "- Primary metrics are overall MAE, RMSE, and R² computed over all observations in the named partition. Per-cell rows are diagnostics only.",
        f"- The complete-modality comparison has {fusion_external['observation_count']} external observations across {fusion_external['physical_cell_count']} cells; full-coverage CNN1D has {full_overall['observation_count']} Step 4-eligible external observations across 12 cells.",
        f"- External fusion coverage is {eligible_cells}/12 cells after a supplemental audit-only evaluation for G10-Cell2.",
        "- Part 3 targets remain provisional capacity-ratio candidates. External metrics are descriptive, not ground-truth validation.",
        "- No retraining, split change, synthetic measurement, PyDMA modification, or external-test tuning was performed.",
        "",
        "## Overall metrics",
        "",
        "| Partition | Model | Observations | Cells | MAE (SOH pp) | RMSE (SOH pp) | R² |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['partition']} | {row['model']} | {row['observation_count']} | "
            f"{row['physical_cell_count']} | {row['mae_soh_percentage_points']:.6f} | "
            f"{row['rmse_soh_percentage_points']:.6f} | {row['r2']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## External population definitions",
            "",
            f"- Shared-subset CNN1D vs fusion: identical keyed observations; CNN1D MAE/RMSE/R² "
            f"{cnn_shared['mae_soh_percentage_points']:.6f}/"
            f"{cnn_shared['rmse_soh_percentage_points']:.6f}/"
            f"{cnn_shared['r2']:.6f}, fusion "
            f"{fusion_external['mae_soh_percentage_points']:.6f}/"
            f"{fusion_external['rmse_soh_percentage_points']:.6f}/"
            f"{fusion_external['r2']:.6f}. Fusion does not outperform CNN1D on these identical external keys.",
            f"- Full-coverage CNN1D: {full_overall['observation_count']} observations, "
            f"{full_overall['physical_cell_count']} cells, MAE/RMSE/R² "
            f"{full_overall['mae_soh_percentage_points']:.6f}/"
            f"{full_overall['rmse_soh_percentage_points']:.6f}/"
            f"{full_overall['r2']:.6f}. This population includes eligible Step 4 observations that lack Step 10 and is not directly comparable to fusion.",
            "",
            "## G10-Cell2 provenance",
            "",
            f"The canonical manifest maps {len(supplemental)} G10-Cell2 RPTs. Step 4 raw validation passed for {cell2_step4_valid}; Step 10 raw validation passed for {cell2_step10_valid}; {cell2_fusion_eligible} have all modalities and are eligible for fusion. Mapped Step 4 capacity segments are checked against source row boundaries. The Group 10 cell identity is user-confirmed because the Guide leaves the cell field blank; it is not independently established by the Guide. The denominator is the earliest mapped BoL Step 4 capacity span. These supplemental labels and predictions are separate from the historical Part 3 label file.",
            "The label is `100 * Step 4 Amphr span / BoL Step 4 Amphr span`; `Amphr` is retained as native units, not asserted to be Ah. Step 4 current direction, curve boundary, capacity progress, finite values, and temperature completeness are recorded in the accompanying scan and label CSVs.",
            "",
            "## Coverage by cell",
            "",
            "| Cell | Step 4 available | Valid references | Verified Step 10 | Valid physics | Complete modalities | Fusion | CNN1D |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage_rows:
        lines.append(
            f"| {row['physical_cell_id']} | {row['available_step4_observations']} | "
            f"{row['valid_soh_reference_observations']} | "
            f"{row['verified_step10_charge_observations']} | "
            f"{row['valid_physics_feature_observations']} | "
            f"{row['complete_modality_observations']} | "
            f"{row['eligible_fusion_predictions']} | "
            f"{row['eligible_full_coverage_cnn1d_predictions']} |"
        )
    lines.extend(
        [
            "",
            "Cell-level and observation-level exclusion details are in `external_12_cell_coverage.csv` and `external_exclusion_reasons.csv`.",
            "",
            "## Generalization and leakage notes",
            "",
            "- Part 3 is an external/domain-shift evaluation (Maccor 4000 vs Maccor 4200 in Parts 1-2). Its README does not state temperature conditions.",
            "- The saved fusion checkpoint and training-only scalers were used unchanged. No external data were used to fit preprocessing.",
            "- Fusion requires complete discharge, charge, and physics modalities. CNN1D uses only Step 4 voltage and a valid reference label.",
            "- Step 4 duration is excluded from model inputs; it is retained only as an audit variable because it is strongly associated with same-RPT capacity. No same-RPT capacity span or label-derived transform is included as a predictor.",
            "- Part 3 capacity-ratio labels are provisional; do not interpret these external scores as official-label performance.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            r".venv-torch-cuda\Scripts\python.exe experiments\soh_forecast\bafuse_three_encoder_v1\evaluation_audit_v1\audit_soh_evaluation.py",
            r".venv-torch-cuda\Scripts\python.exe -m pytest -q tests\test_bafuse_evaluation_audit.py tests\test_dashboard_paths.py",
            "```",
            "",
            "This evaluation audit does not invoke training or PyDMA fitting. The historical CNN1D evaluation, three-encoder checkpoint, locked split, and existing PyDMA branch remain unchanged.",
        ]
    )
    (output_dir / "corrected_soh_evaluation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved BaFuse SoH predictions and external coverage without training."
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run(args.output_dir.resolve())


if __name__ == "__main__":
    main()
