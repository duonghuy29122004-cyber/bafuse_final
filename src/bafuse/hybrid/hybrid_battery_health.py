from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

from bafuse.config import load_project_paths
from bafuse.labeling.pydma_policy import pydma_fit_passes_gate

KEY_COLUMNS = ["dataset_part", "physical_cell_id", "measurement_id"]
LABEL_KEY_COLUMNS = [
    "dataset_part",
    "group_id",
    "physical_cell_id",
    "measurement_id",
]
SOH_TARGET_COLUMNS = [
    "LLI_percent_vs_BoL",
    "LAM_anode_percent_vs_BoL",
    "LAM_cathode_percent_vs_BoL",
]
DEGRADATION_COLUMNS = {
    "LLI_percent_vs_BoL": "lli_estimated_pct",
    "LAM_anode_percent_vs_BoL": "lam_anode_estimated_pct",
    "LAM_cathode_percent_vs_BoL": "lam_cathode_estimated_pct",
}


class SavedCNN1D(nn.Module):
    """Single-output architecture serialized in the existing baseline checkpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1:] != (1, 128):
            raise ValueError("CNN1D input must have shape (batch, 1, 128).")
        return self.layers(values)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_checkpoint(root: Path) -> tuple[SavedCNN1D, dict[str, Any], list[str]]:
    checkpoint_path = load_project_paths(root).checkpoint
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("CNN1D checkpoint must contain a state dictionary bundle.")
    required = {
        "model_state_dict",
        "x_mean",
        "x_scale",
        "y_mean",
        "y_scale",
        "feature_names",
    }
    if not required.issubset(payload):
        raise ValueError(f"CNN1D checkpoint is missing keys: {sorted(required - set(payload))}")

    feature_names = [str(name) for name in payload["feature_names"]]
    if len(feature_names) != 128 or len(set(feature_names)) != 128:
        raise ValueError("CNN1D checkpoint must define exactly 128 unique input features.")
    x_mean = payload["x_mean"].detach().cpu().numpy()
    x_scale = payload["x_scale"].detach().cpu().numpy()
    if x_mean.shape != (128,) or x_scale.shape != (128,):
        raise ValueError("Saved CNN1D input scalers must each have shape (128,).")
    if not np.isfinite(x_mean).all() or not np.isfinite(x_scale).all():
        raise ValueError("Saved CNN1D input scalers must be finite.")
    if np.any(x_scale <= 0):
        raise ValueError("Saved CNN1D input scales must be positive.")
    if not math.isfinite(float(payload["y_mean"])) or not math.isfinite(
        float(payload["y_scale"])
    ):
        raise ValueError("Saved CNN1D target scalers must be finite.")
    if float(payload["y_scale"]) <= 0:
        raise ValueError("Saved CNN1D target scale must be positive.")

    model = SavedCNN1D()
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    return model, payload, feature_names


def _label_tables(root: Path) -> pd.DataFrame:
    paths = load_project_paths(root)
    tables = []
    for part, label_path in paths.soh_labels.items():
        table = pd.read_csv(label_path, dtype=str, keep_default_na=False)
        table["dataset_part"] = table["dataset_part"].astype(str)
        table["group_id"] = table["group_id"].map(_group_number)
        for column in ("physical_cell_id", "measurement_id"):
            table[column] = table[column].astype(str)
        if not (table["dataset_part"] == part).all():
            raise ValueError(f"SOH label table contains an unexpected dataset part: {label_path}")
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _group_number(value: object) -> str:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise ValueError(f"Cannot parse a numeric group identifier from {value!r}.")
    return str(int(match.group()))


def _part_number(value: object) -> int:
    match = re.search(r"\d+", str(value))
    if match is None:
        raise ValueError(f"Cannot parse a dataset part from {value!r}.")
    return int(match.group())


def _cell_id(group_id: object, cell_id: object) -> str:
    group = _group_number(group_id)
    cell = re.sub(r"\s+", "", str(cell_id))
    if not cell.startswith("Cell"):
        raise ValueError(f"Cannot parse a physical cell identifier from {cell_id!r}.")
    return f"G{group}-{cell}"


def _rpt_id(value: object) -> str:
    text = str(value).strip()
    order = int(text) if re.fullmatch(r"\d+", text) else _rpt_order(value)
    if order is None:
        raise ValueError(f"Cannot parse an RPT identifier from {value!r}.")
    return f"RPT{order}"


def _basename(value: object) -> str:
    return str(value).replace("\\", "/").rsplit("/", 1)[-1].casefold()


def _source_path_identity(value: object, part: int) -> str:
    normalized = str(value).replace("\\", "/").strip("/").casefold()
    prefix = f"data/raw/oxford/part{part}/"
    if prefix in normalized:
        normalized = normalized.split(prefix, 1)[1]
    return normalized


def _same_source_path(left: object, right: object, part: int) -> bool:
    left_path = _source_path_identity(left, part)
    right_path = _source_path_identity(right, part)
    return (
        left_path == right_path
        or left_path.endswith(f"/{right_path}")
        or right_path.endswith(f"/{left_path}")
    )


def _load_rpt_observation_manifest(root: Path) -> pd.DataFrame:
    path = load_project_paths(root).rpt_observation_manifest
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    table["dataset_part"] = table["part"].map(_part_number).astype(str)
    table["group_id"] = table["group"].map(_group_number)
    table["measurement_id"] = table["rpt_index"].map(_rpt_id)
    keys = ["dataset_part", "group_id", "physical_cell_id", "measurement_id"]
    if table.duplicated(keys).any():
        raise ValueError(f"RPT observation manifest contains duplicate keys: {path}")
    return table


def _load_bol_reference_policy(root: Path) -> pd.DataFrame:
    path = load_project_paths(root).part2_bol_policy_decisions
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if table.duplicated(["physical_cell_id"]).any():
        raise ValueError(f"Part 2 BoL policy contains duplicate physical cells: {path}")
    return table


def _attach_bol_reference_policy(frame: pd.DataFrame, policy: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    by_cell = policy.set_index("physical_cell_id").to_dict(orient="index")
    policy_fields = {
        "degradation_reference_policy": "reference_policy",
        "degradation_reference_file": "selected_reference_file",
        "degradation_reference_status": "decision_status",
        "degradation_reference_limitations": "limitations",
        "degradation_reference_evidence": "evidence",
        "degradation_reference_sources": "evidence_sources",
    }
    for output_column in policy_fields:
        result[output_column] = ""
    for index, row in result.iterrows():
        part = int(row["dataset_part"])
        physical_cell = str(row["physical_cell_id"])
        decision = by_cell.get(physical_cell)
        if part == 3:
            result.at[index, "degradation_reference_status"] = (
                "not_applicable_external_part3"
            )
            continue
        if decision is None:
            if part == 2:
                raise ValueError(
                    "Part 2 observation has no BoL policy decision for "
                    f"{physical_cell}."
                )
            result.at[index, "degradation_reference_status"] = (
                "no_part2_policy_for_physical_cell"
            )
            continue
        for output_column, source_column in policy_fields.items():
            result.at[index, output_column] = str(decision[source_column])
    return result


def _load_feature_rows(
    root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], SavedCNN1D, dict[str, Any], list[str]]:
    paths = load_project_paths(root)
    feature_dir = paths.feature_run
    baseline_dir = paths.baseline_run
    config = json.loads((baseline_dir / "run_config.json").read_text(encoding="utf-8"))
    expected_hashes = config.get("input_sha256", {})
    hash_sources = {
        "features.csv": feature_dir / "features.csv",
        "sample_manifest.csv": feature_dir / "sample_manifest.csv",
        "feature_run_metrics.json": feature_dir / "metrics.json",
        "part1_labels": paths.soh_labels["1"],
        "part2_labels": paths.soh_labels["2"],
        "part3_labels": paths.soh_labels["3"],
    }
    split_path = paths.split_manifest
    hash_sources["split"] = split_path
    for name, path in hash_sources.items():
        expected = expected_hashes.get(name)
        if expected is None or _sha256(path) != expected:
            raise ValueError(f"Saved CNN1D input hash does not match {name}: {path}")

    manifest = pd.read_csv(
        feature_dir / "sample_manifest.csv",
        dtype={"dataset_part": str, "group_id": str, "physical_cell_id": str},
    )
    features = pd.read_csv(
        feature_dir / "features.csv",
        dtype={"dataset_part": str, "physical_cell_id": str, "measurement_id": str},
    )
    manifest["dataset_part"] = manifest["dataset_part"].astype(str)
    manifest["group_id"] = manifest["group_id"].map(_group_number)
    for column in ("physical_cell_id", "measurement_id"):
        manifest[column] = manifest[column].astype(str)
    if manifest.duplicated(KEY_COLUMNS).any() or features.duplicated(KEY_COLUMNS).any():
        raise ValueError("Step 4 manifest/features contain duplicate checkpoint keys.")
    frame = manifest.merge(
        features,
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not frame["_merge"].eq("both").all():
        raise ValueError("Step 4 feature rows and observation manifest do not align one-to-one.")
    frame = frame.drop(columns="_merge")

    observations = _load_rpt_observation_manifest(root)
    observation_keys = ["dataset_part", "group_id", "physical_cell_id", "measurement_id"]
    observation_fields = {
        "source_tpg": "mapped_source_tpg",
        "source_file": "mapped_source_file",
        "bol_reference_file": "mapped_bol_reference_file",
        "step4_segment_ref": "step4_segment_ref",
        "step8_segment_ref": "step8_segment_ref",
        "step10_segment_ref": "step10_segment_ref",
        "mapping_status": "observation_mapping_status",
        "step4_status": "observation_step4_status",
        "step8_status": "observation_step8_status",
        "step10_status": "observation_step10_status",
        "ready_soh": "observation_ready_soh",
        "ready_degradation": "observation_ready_degradation",
        "needs_audit": "observation_needs_audit",
        "audit_notes": "observation_audit_notes",
    }
    observation_table = observations[
        observation_keys + list(observation_fields)
    ].rename(columns=observation_fields)
    frame = frame.merge(
        observation_table,
        on=observation_keys,
        how="left",
        validate="one_to_one",
        indicator="_observation_merge",
    )
    if not frame["_observation_merge"].eq("both").all():
        missing = frame.loc[
            frame["_observation_merge"] != "both", observation_keys
        ].to_dict(orient="records")
        raise ValueError(f"Feature checkpoints are missing RPT mappings: {missing[:5]}")
    frame = frame.drop(columns="_observation_merge")
    source_matches = [
        _source_path_identity(feature_path, int(part))
        == _source_path_identity(mapped_path, int(part))
        for part, feature_path, mapped_path in zip(
            frame["dataset_part"],
            frame["source_file"],
            frame["mapped_source_file"],
        )
    ]
    if not all(source_matches):
        raise ValueError("A Step 4 checkpoint source path disagrees with the RPT manifest.")
    frame["observation_source_match_basis"] = "exact_normalized_source_path"
    if not frame["observation_step4_status"].eq("verified").all():
        raise ValueError("A Step 4 checkpoint is not verified by the RPT manifest.")
    if not frame["observation_ready_soh"].map(_parse_bool).all():
        raise ValueError("A Step 4 checkpoint is not marked ready for SoH.")
    frame = _attach_bol_reference_policy(frame, _load_bol_reference_policy(root))

    label_table = _label_tables(root)
    valid_labels = label_table[
        label_table["validation_status"] == "calculated_from_measured_capacity"
    ].copy()
    if valid_labels.duplicated(LABEL_KEY_COLUMNS).any():
        raise ValueError("SOH labels contain duplicate part/group/cell/RPT keys.")
    valid_labels = valid_labels.rename(
        columns={
            "soh_percent": "reference_soh_percent",
            "nominal_cycle_age": "nominal_cycle_age",
            "cycle_age_status": "cycle_age_status",
            "validation_status": "reference_label_status",
        }
    )
    label_columns = LABEL_KEY_COLUMNS + [
        "reference_soh_percent",
        "nominal_cycle_age",
        "cycle_age_status",
        "reference_label_status",
    ]
    frame = frame.merge(
        valid_labels[label_columns],
        on=LABEL_KEY_COLUMNS,
        how="left",
        validate="one_to_one",
    )
    if frame["reference_soh_percent"].isna().any():
        raise ValueError("A Step 4 checkpoint has no valid capacity-based SOH reference.")
    frame["soh_percent"] = pd.to_numeric(frame["soh_percent"], errors="raise")
    frame["reference_soh_percent"] = pd.to_numeric(
        frame["reference_soh_percent"], errors="raise"
    )
    if not np.allclose(
        frame["soh_percent"].to_numpy(dtype=float),
        frame["reference_soh_percent"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-8,
    ):
        raise ValueError("Feature manifest SOH values differ from their source label tables.")

    frame["rpt_order"] = pd.to_numeric(frame["rpt_order"], errors="raise").astype(int)
    frame["nominal_cycle_age"] = pd.to_numeric(
        frame["nominal_cycle_age"], errors="raise"
    )
    expected_age = 48 * frame["rpt_order"].to_numpy(dtype=int)
    if not np.allclose(
        frame["nominal_cycle_age"].to_numpy(dtype=float), expected_age, rtol=0.0, atol=1e-8
    ):
        raise ValueError("A Step 4 label age is not the documented nominal 48 x RPT proxy.")
    if not frame["cycle_age_status"].eq("nominal_48_cycles_per_RPT").all():
        raise ValueError("Step 4 rows contain unsupported cycle-age provenance.")
    frame["cycle_count"] = np.nan
    frame["reference_label_scope"] = np.where(
        frame["dataset_part"].eq("3"),
        "provisional_part3_capacity_ratio",
        "capacity_ratio_label",
    )

    if frame.groupby("physical_cell_id")["partition"].nunique().gt(1).any():
        raise ValueError("A physical cell crosses locked train/validation/test partitions.")
    if not frame.loc[frame["dataset_part"] == "3", "partition"].eq("external_test").all():
        raise ValueError("Part 3 must remain isolated in the external-test partition.")
    if len(frame) != 231:
        raise ValueError(f"Expected 231 audited Step 4 checkpoints, found {len(frame)}.")

    model, checkpoint, feature_names = _read_checkpoint(root)
    if list(feature_names) != [
        f"voltage_progress_{index:03d}" for index in range(128)
    ]:
        raise ValueError("Checkpoint feature order is not the validated 128-point voltage shape.")
    if not set(feature_names).issubset(frame.columns):
        raise ValueError("The feature table is missing checkpoint input columns.")
    curves = frame[feature_names].to_numpy(dtype=np.float32)
    if curves.shape != (len(frame), 128) or not np.isfinite(curves).all():
        raise ValueError("Step 4 model input must be a finite (n, 128) array.")
    return frame, curves, config, model, checkpoint, feature_names


def _predict_soh(
    curves: np.ndarray,
    model: SavedCNN1D,
    checkpoint: dict[str, Any],
    batch_size: int = 64,
) -> np.ndarray:
    if curves.ndim != 2 or curves.shape[1] != 128:
        raise ValueError("CNN1D inference expects a (n, 128) array.")
    x_mean = checkpoint["x_mean"].detach().cpu().numpy()
    x_scale = checkpoint["x_scale"].detach().cpu().numpy()
    y_mean = float(checkpoint["y_mean"])
    y_scale = float(checkpoint["y_scale"])
    normalized = (curves.astype(np.float32) - x_mean) / x_scale
    if not np.isfinite(normalized).all():
        raise ValueError("Normalized CNN1D inputs contain non-finite values.")
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(normalized), batch_size):
            batch = torch.from_numpy(normalized[start : start + batch_size, None, :])
            prediction = model(batch)
            if prediction.shape != (len(batch), 1):
                raise ValueError("Saved CNN1D model returned an unexpected output shape.")
            outputs.append(prediction[:, 0].cpu().numpy())
    scaled = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    predictions = scaled.astype(np.float64) * y_scale + y_mean
    if not np.isfinite(predictions).all():
        raise ValueError("CNN1D inference produced a non-finite SoH prediction.")
    return predictions


def _validate_prediction_parity(
    root: Path, frame: pd.DataFrame, predictions: np.ndarray, tolerance: float = 1e-4
) -> float:
    saved = pd.read_csv(
        load_project_paths(root).baseline_run / "predictions.csv",
        dtype={"dataset_part": str, "physical_cell_id": str, "measurement_id": str},
    )
    compare = frame[KEY_COLUMNS].copy()
    compare["new_prediction"] = predictions
    compare = compare.merge(
        saved[KEY_COLUMNS + ["cnn1d_prediction_percent"]],
        on=KEY_COLUMNS,
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not compare["_merge"].eq("both").all():
        raise ValueError("Recomputed CNN1D predictions do not match the saved checkpoint rows.")
    difference = np.abs(
        compare["new_prediction"].to_numpy(dtype=float)
        - compare["cnn1d_prediction_percent"].to_numpy(dtype=float)
    )
    max_difference = float(difference.max(initial=0.0))
    if max_difference > tolerance:
        raise ValueError(
            f"Checkpoint inference parity failed: max difference {max_difference:.8g} "
            f"> {tolerance:.8g}."
        )
    return max_difference


def _r2_score(actual: np.ndarray, predicted: np.ndarray) -> float | None:
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    if len(actual) < 2 or denominator <= 0:
        return None
    return float(1.0 - np.sum((actual - predicted) ** 2) / denominator)


def _metric_row(
    subset: pd.DataFrame,
    group_name: str,
    group_value: str,
    label_scope: str,
    baseline_prediction: float,
) -> dict[str, Any]:
    actual = subset["reference_soh_percent"].to_numpy(dtype=float)
    predicted = subset["predicted_soh_percent"].to_numpy(dtype=float)
    baseline = np.full(len(subset), baseline_prediction, dtype=float)
    error = predicted - actual
    return {
        "group_name": group_name,
        "group_value": group_value,
        "label_scope": label_scope,
        "n_checkpoints": len(subset),
        "n_physical_cells": subset["physical_cell_id"].nunique(),
        "mae_percentage_points": float(np.mean(np.abs(error))) if len(error) else None,
        "rmse_percentage_points": (
            float(np.sqrt(np.mean(error**2))) if len(error) else None
        ),
        "r2": _r2_score(actual, predicted),
        "train_cell_mean_mae_percentage_points": (
            float(np.mean(np.abs(baseline - actual))) if len(error) else None
        ),
        "train_cell_mean_rmse_percentage_points": (
            float(np.sqrt(np.mean((baseline - actual) ** 2))) if len(error) else None
        ),
    }


def _evaluate_soh(frame: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    evaluated = frame.copy()
    evaluated["predicted_soh_percent"] = predictions
    train_rows = evaluated[evaluated["partition"] == "train"]
    train_cell_mean = float(
        train_rows.groupby("physical_cell_id")["reference_soh_percent"].mean().mean()
    )
    result = []
    for partition, subset in evaluated.groupby("partition", sort=True):
        scope = (
            "provisional_part3_capacity_ratio"
            if partition == "external_test"
            else "capacity_ratio_label"
        )
        result.append(
            _metric_row(subset, "partition", str(partition), scope, train_cell_mean)
        )
    for part, subset in evaluated.groupby("dataset_part", sort=True):
        scope = "provisional_part3_capacity_ratio" if part == "3" else "capacity_ratio_label"
        result.append(_metric_row(subset, "dataset_part", str(part), scope, train_cell_mean))
    return pd.DataFrame(result)


def _parse_optional_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _parse_optional_int(value: object) -> int | None:
    parsed = _parse_optional_float(value)
    return None if parsed is None else int(parsed)


def _parse_bool(value: object) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _rpt_order(value: object) -> int | None:
    match = re.fullmatch(r"\s*RPT\s*(\d+)\s*", str(value), flags=re.IGNORECASE)
    return None if match is None else int(match.group(1))


def _protocol_group(part: int, group: str) -> str:
    if part == 3:
        return "maccor4000_part3"
    if part == 2 and group in {"5", "6"}:
        return "part2_group5_6_special"
    return "maccor4200_parts1_2"


def _nominal_age(part: int, group: str, order: int | None) -> tuple[float | None, str]:
    if order is None:
        return None, "rpt_order_unavailable"
    nominal = (part == 1 and group in {"1", "2", "3", "4"}) or (
        part == 2 and group in {"1", "2", "3", "4"}
    ) or (part == 3 and group in {"7", "8", "9", "10"})
    if nominal:
        return float(48 * order), "nominal_48_cycles_per_RPT"
    if part == 2 and group == "6":
        return None, "calendar_aging_only"
    return None, "nominal_cycle_age_not_documented"


def _candidate_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        record["dataset_part"],
        record["group_id"],
        record["physical_cell_id"],
        record["rpt_id"],
        _basename(record["source_file"]),
        str(record["test_step"]),
        str(record["direction"]).casefold(),
    )


def _load_pydma_records(root: Path) -> pd.DataFrame:
    paths = load_project_paths(root)
    observations = _load_rpt_observation_manifest(root)
    bol_policy = _load_bol_reference_policy(root)
    part12 = pd.read_csv(paths.pydma_part12, dtype=str, keep_default_na=False)
    part3 = pd.read_csv(paths.pydma_part3, dtype=str, keep_default_na=False)
    part3["source_part"] = "Part 3"
    part3["source_run"] = "run_009_part3_pydma"
    candidate_audit = pd.read_csv(
        paths.pydma_g5_audit, dtype=str, keep_default_na=False
    )

    records: list[dict[str, Any]] = []
    source_rows: dict[tuple[Any, ...], list[int]] = {}
    for table_source, table in (("part1_2_results", part12), ("part3_results", part3)):
        for source_index, row in table.iterrows():
            part = _part_number(row["source_part"])
            group = _group_number(row["group_id"])
            physical_cell = _cell_id(group, row["cell_id"])
            rpt = str(row["checkup"])
            order = _rpt_order(rpt)
            audit_text = row.get("curve_audit_json", "")
            audit = json.loads(audit_text) if audit_text else {}
            if not isinstance(audit, dict):
                raise ValueError(f"PyDMA curve audit is not an object: {table_source}:{source_index}")
            segments = audit.get("segments", [])
            if not isinstance(segments, list):
                raise ValueError(f"PyDMA segments are malformed: {table_source}:{source_index}")
            selected = audit.get("selected_segment")
            selected_segment = next(
                (
                    segment
                    for segment in segments
                    if str(segment.get("segment_index")) == str(selected)
                ),
                {},
            )
            status = str(row.get("diagnostic_label_status", ""))
            accepted = (
                status == "diagnostic_estimate_accepted"
                and _parse_bool(row.get("fit_accepted_by_pydma_15mV"))
            )
            rmse = _parse_optional_float(row.get("fit_rmse_mV"))
            accepted_runs = _parse_optional_int(row.get("accepted_runs"))
            strict_gate = pydma_fit_passes_gate(
                rmse_mV=rmse,
                accepted_runs=accepted_runs,
                is_accepted=accepted,
            )
            curve_source = audit.get("source_file", "")
            if not curve_source:
                curve_source = row.get("source_file", "")
            record = {
                "result_record_id": f"{table_source}:{source_index}",
                "table_source": table_source,
                "source_part": f"Part {part}",
                "dataset_part": part,
                "group_id": group,
                "physical_cell_id": physical_cell,
                "rpt_id": f"RPT{order}" if order is not None else "",
                "rpt_order": order,
                "direction": str(row.get("direction", "")).casefold(),
                "test_step": str(row.get("test_step", "")),
                "source_file": str(row.get("source_file", "")),
                "source_file_basename": _basename(row.get("source_file", "")),
                "source_run": str(row.get("source_run", "")),
                "fit_rmse_mV": rmse,
                "fit_threshold_mV": _parse_optional_float(
                    row.get("fit_threshold_mV")
                ),
                "accepted_runs": accepted_runs,
                "total_runs": _parse_optional_int(row.get("total_runs")),
                "diagnostic_label_status": status,
                "fit_accepted_by_pydma_15mV": _parse_bool(
                    row.get("fit_accepted_by_pydma_15mV")
                ),
                "strict_gate_pass": strict_gate,
                "lli_estimated_pct": _parse_optional_float(
                    row.get("LLI_percent_vs_BoL")
                ),
                "lam_anode_estimated_pct": _parse_optional_float(
                    row.get("LAM_anode_percent_vs_BoL")
                ),
                "lam_cathode_estimated_pct": _parse_optional_float(
                    row.get("LAM_cathode_percent_vs_BoL")
                ),
                "curve_audit_json": audit_text,
                "selected_segment_index": _parse_optional_int(selected),
                "candidate_segment_count": len(segments),
                "selected_sample_count": _parse_optional_int(
                    selected_segment.get("sample_count")
                ),
                "selected_capacity_min_native": _parse_optional_float(
                    selected_segment.get("capacity_min")
                ),
                "selected_capacity_max_native": _parse_optional_float(
                    selected_segment.get("capacity_max")
                ),
                "raw_curve_source_path": str(curve_source),
                "parameter_vectors_saved": False,
                "fit_curve_arrays_saved": False,
                "candidate_mapping_status": "",
                "candidate_mapping_basis": "",
                "candidate_audit_status": "",
                "candidate_is_training_label": "",
                "candidate_values_consistent": "",
            }
            if order is not None:
                segment_id = (
                    str(record["selected_segment_index"])
                    if record["selected_segment_index"] is not None
                    else "unknown"
                )
                record["curve_id"] = (
                    f"P{part}|{physical_cell}|{_basename(record['source_file'])}|"
                    f"Step{record['test_step']}|{record['direction']}|segment{segment_id}"
                )
            else:
                record["curve_id"] = (
                    f"P{part}|{physical_cell}|{_basename(record['source_file'])}|"
                    f"Step{record['test_step']}|{record['direction']}|baseline"
                )
            records.append(record)
            source_rows.setdefault(_candidate_key(record), []).append(len(records) - 1)

    for candidate_index, row in candidate_audit.iterrows():
        group = _group_number(row["group_id"])
        part = 2
        physical_cell = str(row["physical_cell_id"])
        candidate_order = _rpt_order(row["candidate_rpt"])
        if candidate_order is None:
            candidate_order = int(row["candidate_rpt"])
        rpt = f"RPT{candidate_order}"
        record = {
            "dataset_part": part,
            "group_id": group,
            "physical_cell_id": physical_cell,
            "rpt_id": rpt,
            "direction": str(row["direction"]).casefold(),
            "test_step": str(row["test_step"]),
            "source_file": str(row["source_file"]),
        }
        key = _candidate_key(record)
        matches = source_rows.get(key, [])
        if len(matches) > 1:
            raise ValueError(f"Duplicate canonical PyDMA records match Group 5 audit row {key}.")
        audit_values = {
            "candidate_mapping_status": str(row.get("identity_status", "")),
            "candidate_mapping_basis": str(row.get("mapping_basis", "")),
            "candidate_audit_status": str(row.get("audit_status", "")),
            "candidate_is_training_label": str(row.get("is_training_label", "")),
            "candidate_segment_index": _parse_optional_int(
                row.get("candidate_segment_index")
            ),
            "candidate_source_row_start": _parse_optional_int(row.get("source_row_start")),
            "candidate_source_row_end": _parse_optional_int(row.get("source_row_end")),
            "candidate_values_consistent": None,
        }
        candidate_estimates = {
            "fit_rmse_mV": _parse_optional_float(row.get("fit_rmse_mV")),
            "lli_estimated_pct": _parse_optional_float(row.get("LLI_percent_vs_BoL")),
            "lam_anode_estimated_pct": _parse_optional_float(
                row.get("LAM_anode_percent_vs_BoL")
            ),
            "lam_cathode_estimated_pct": _parse_optional_float(
                row.get("LAM_cathode_percent_vs_BoL")
            ),
        }
        if matches:
            target = records[matches[0]]
            consistent = True
            for name, candidate_value in candidate_estimates.items():
                canonical_value = target.get(name)
                if (
                    candidate_value is not None
                    and canonical_value is not None
                    and not math.isclose(
                        candidate_value, canonical_value, rel_tol=0.0, abs_tol=1e-6
                    )
                ):
                    consistent = False
            target.update(audit_values)
            target["candidate_values_consistent"] = consistent
            target["candidate_segment_index"] = audit_values["candidate_segment_index"]
            continue

        rmse = candidate_estimates["fit_rmse_mV"]
        accepted_runs = _parse_optional_int(row.get("accepted_runs"))
        passed = _parse_bool(row.get("passes_strict_rmse15_gate"))
        strict_gate = pydma_fit_passes_gate(
            rmse_mV=rmse,
            accepted_runs=accepted_runs,
            is_accepted=_parse_bool(row.get("fit_is_accepted")) and passed,
        )
        record.update(
            {
                "result_record_id": f"candidate_g5:{candidate_index}",
                "table_source": "part2_g5_candidate_audit_only",
                "source_part": "Part 2",
                "rpt_order": _rpt_order(rpt),
                "source_file_basename": _basename(row["source_file"]),
                "source_run": "part2_g5_candidate_audit",
                "fit_rmse_mV": rmse,
                "fit_threshold_mV": 15.0,
                "accepted_runs": accepted_runs,
                "total_runs": _parse_optional_int(row.get("total_runs")),
                "diagnostic_label_status": (
                    "diagnostic_estimate_accepted" if strict_gate else "raw_diagnostic_estimate"
                ),
                "fit_accepted_by_pydma_15mV": _parse_bool(
                    row.get("fit_is_accepted")
                ),
                "strict_gate_pass": strict_gate,
                "lli_estimated_pct": candidate_estimates["lli_estimated_pct"],
                "lam_anode_estimated_pct": candidate_estimates["lam_anode_estimated_pct"],
                "lam_cathode_estimated_pct": candidate_estimates["lam_cathode_estimated_pct"],
                "curve_audit_json": "",
                "selected_segment_index": audit_values["candidate_segment_index"],
                "candidate_segment_count": None,
                "selected_sample_count": _parse_optional_int(row.get("curve_sample_count")),
                "selected_capacity_min_native": None,
                "selected_capacity_max_native": _parse_optional_float(
                    row.get("curve_capacity_span_native")
                ),
                "raw_curve_source_path": "",
                "parameter_vectors_saved": False,
                "fit_curve_arrays_saved": False,
                **audit_values,
                "curve_id": (
                    f"P2|{physical_cell}|{_basename(row['source_file'])}|"
                    f"Step{record['test_step']}|{record['direction']}|"
                    f"segment{audit_values['candidate_segment_index']}"
                ),
            }
        )
        records.append(record)
    for index, record in enumerate(records):
        record["result_record_id"] = f"{record['result_record_id']}:{index}"
    result = pd.DataFrame(records)
    observation_index = {
        (
            int(row.dataset_part),
            str(row.group_id),
            str(row.physical_cell_id),
            str(row.measurement_id),
        ): row._asdict()
        for row in observations.itertuples(index=False)
    }
    observation_columns = {
        "source_tpg": "mapped_source_tpg",
        "source_file": "mapped_source_file",
        "bol_reference_file": "mapped_bol_reference_file",
        "mapping_status": "observation_mapping_status",
        "step4_status": "observation_step4_status",
        "step8_status": "observation_step8_status",
        "step10_status": "observation_step10_status",
        "step4_segment_ref": "observation_step4_segment_ref",
        "step8_segment_ref": "observation_step8_segment_ref",
        "step10_segment_ref": "observation_step10_segment_ref",
        "ready_soh": "observation_ready_soh",
        "ready_degradation": "observation_ready_degradation",
        "audit_notes": "observation_audit_notes",
    }
    for output_column in observation_columns.values():
        result[output_column] = ""
    result["observation_mapping_status"] = "not_in_rpt_observation_manifest"
    result["observation_source_match_basis"] = "unmatched"
    result["observation_curve_step"] = ""
    result["observation_curve_step_status"] = "not_in_manifest"
    result["observation_curve_step_segment_ref"] = ""
    for index, row in result.iterrows():
        order = _rpt_order(row["rpt_id"])
        if order is None:
            continue
        key = (
            int(row["dataset_part"]),
            str(row["group_id"]),
            str(row["physical_cell_id"]),
            f"RPT{order}",
        )
        observation = observation_index.get(key)
        if observation is None:
            continue
        raw_source = str(row["raw_curve_source_path"])
        if raw_source:
            source_matches = _same_source_path(
                raw_source, observation["source_file"], int(row["dataset_part"])
            )
            match_basis = "full_source_path_suffix"
        else:
            source_matches = _basename(row["source_file"]) == _basename(
                observation["source_file"]
            )
            match_basis = "part_group_cell_rpt_and_basename"
        result.at[index, "observation_mapping_status"] = (
            str(observation["mapping_status"])
            if source_matches
            else "source_file_mismatch"
        )
        result.at[index, "observation_source_match_basis"] = (
            match_basis if source_matches else "source_path_mismatch"
        )
        for source_column, output_column in observation_columns.items():
            result.at[index, output_column] = str(observation[source_column])
        if not source_matches:
            result.at[index, "observation_audit_notes"] = (
                str(observation["audit_notes"])
                + ";PyDMA source basename does not match mapped observation source"
            ).strip(";")
            result.at[index, "observation_curve_step_status"] = "source_file_mismatch"
            continue
        step_match = re.search(r"\d+", str(row["test_step"]))
        step = step_match.group() if step_match is not None else ""
        result.at[index, "observation_curve_step"] = step
        if step in {"4", "8", "10"}:
            status_column = f"step{step}_status"
            ref_column = f"step{step}_segment_ref"
            result.at[index, "observation_curve_step_status"] = str(
                observation[status_column]
            )
            result.at[index, "observation_curve_step_segment_ref"] = str(
                observation[ref_column]
            )
            expected_direction = {"8": "discharge", "10": "charge"}.get(step)
            if (
                expected_direction is not None
                and str(row["direction"]).casefold() != expected_direction
            ):
                result.at[index, "observation_curve_step_status"] = (
                    "step_direction_mismatch"
                )
        else:
            result.at[index, "observation_curve_step_status"] = (
                "unsupported_curve_step"
            )
    result = _attach_bol_reference_policy(result, bol_policy)
    return result


def _map_pydma_to_features(
    curve_rows: pd.DataFrame, frame: pd.DataFrame
) -> pd.DataFrame:
    feature_index: dict[tuple[Any, ...], list[int]] = {}
    for index, row in frame.iterrows():
        key = (
            int(row["dataset_part"]),
            str(row["physical_cell_id"]),
            str(row["measurement_id"]),
            _basename(row["source_file"]),
        )
        feature_index.setdefault(key, []).append(int(index))
    mapped = curve_rows.copy()
    statuses = []
    feature_ids = []
    for _, row in mapped.iterrows():
        part = int(row["dataset_part"])
        group = str(row["group_id"])
        rpt = str(row["rpt_id"])
        if not rpt:
            statuses.append("baseline_not_checkpoint")
            feature_ids.append("")
            continue
        key = (
            part,
            str(row["physical_cell_id"]),
            rpt,
            str(row["source_file_basename"]),
        )
        candidates = feature_index.get(key, [])
        provisional = (
            row.get("candidate_mapping_status")
            == "provisional_guide_and_step_order_not_log_verified"
        )
        if provisional:
            status = "provisional_group5_file_order_mapping"
        elif len(candidates) == 1:
            status = "verified_exact_step4_source_match"
        elif len(candidates) > 1:
            status = "ambiguous_multiple_step4_matches"
        elif part == 2 and group == "6":
            status = "pydma_only_calendar_aging_no_step4_feature"
        else:
            status = "no_exact_step4_source_match"
        statuses.append(status)
        if len(candidates) == 1:
            feature = frame.iloc[candidates[0]]
            feature_ids.append(
                f"{part}|{feature['physical_cell_id']}|{feature['measurement_id']}"
            )
        else:
            feature_ids.append("")
    mapped["feature_mapping_status"] = statuses
    mapped["feature_checkpoint_id"] = feature_ids
    mapped["is_exact_feature_mapped"] = mapped[
        "feature_mapping_status"
    ].eq("verified_exact_step4_source_match")
    mapped["protocol_group"] = [
        _protocol_group(int(row.dataset_part), str(row.group_id))
        for row in mapped.itertuples()
    ]
    nominal = [
        _nominal_age(int(row.dataset_part), str(row.group_id), row.rpt_order)
        for row in mapped.itertuples()
    ]
    mapped["nominal_cycle_age"] = [value for value, _ in nominal]
    mapped["cycle_age_status"] = [status for _, status in nominal]

    mapped["quality_flags"] = ""
    for index, row in mapped.iterrows():
        flags = []
        if row["feature_mapping_status"] == "provisional_group5_file_order_mapping":
            flags.append("ambiguous_group5_rpt_identity")
        if (
            row["candidate_segment_count"] is not None
            and pd.notna(row["candidate_segment_count"])
            and int(row["candidate_segment_count"]) > 1
        ):
            flags.append("multiple_monotone_segments_only_longest_saved")
        if row["strict_gate_pass"] and any(
            row[column] is not None and float(row[column]) < 0
            for column in (
                "lli_estimated_pct",
                "lam_anode_estimated_pct",
                "lam_cathode_estimated_pct",
            )
        ):
            flags.append("negative_degradation_estimate_review")
        if row.get("candidate_values_consistent") is False:
            flags.append("candidate_audit_values_disagree")
        mapped.at[index, "quality_flags"] = ";".join(flags)

    mapped["statistical_outlier_targets"] = ""
    for _, group in mapped[
        mapped["strict_gate_pass"] & mapped["rpt_id"].ne("")
    ].groupby(["dataset_part", "protocol_group", "direction"]):
        for source_column, target_column in DEGRADATION_COLUMNS.items():
            values = pd.to_numeric(group[target_column], errors="coerce")
            finite = values.dropna()
            if len(finite) < 4:
                continue
            q1, q3 = finite.quantile([0.25, 0.75]).to_numpy(dtype=float)
            spread = q3 - q1
            low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
            outlier_indices = values.index[(values < low) | (values > high)]
            for index in outlier_indices:
                current = mapped.at[index, "statistical_outlier_targets"]
                mapped.at[index, "statistical_outlier_targets"] = (
                    f"{current};{source_column}".strip(";")
                )
                current_flags = mapped.at[index, "quality_flags"]
                mapped.at[index, "quality_flags"] = (
                    f"{current_flags};statistical_outlier_review".strip(";")
                )
    return mapped


def _quartile_dispersion(values: pd.Series) -> tuple[float | None, float | None]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if not len(finite):
        return None, None
    q1, median, q3 = np.quantile(finite, [0.25, 0.5, 0.75])
    return float(q3 - q1), float(median)


def _within_group_outlier(values: pd.Series) -> bool:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(finite) < 3:
        return False
    q1, q3 = np.quantile(finite, [0.25, 0.75])
    spread = float(q3 - q1)
    low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
    return bool(np.any((finite < low) | (finite > high)))


def _group_metadata(
    group: pd.DataFrame, columns: list[str], *, allow_multiple: bool = False
) -> dict[str, str]:
    metadata = {}
    for column in columns:
        if column not in group:
            continue
        values = group[column].fillna("").astype(str).unique()
        if len(values) > 1 and not allow_multiple:
            raise ValueError(f"PyDMA group has conflicting {column} provenance.")
        metadata[column] = ";".join(sorted(values)) if allow_multiple else (
            values[0] if len(values) else ""
        )
    return metadata


def _select_curve_records(group: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for _, curve_group in group.groupby("curve_id", dropna=False, sort=False):
        accepted = curve_group[curve_group["strict_gate_pass"]]
        pool = accepted if not accepted.empty else curve_group
        pool = pool.assign(
            _sort_rmse=pd.to_numeric(pool["fit_rmse_mV"], errors="coerce").fillna(np.inf)
        )
        selected.append(pool.sort_values("_sort_rmse", kind="stable").iloc[0].drop(labels="_sort_rmse"))
    return pd.DataFrame(selected)


def _aggregate_pydma(curve_rows: pd.DataFrame) -> pd.DataFrame:
    rpt_rows = curve_rows[curve_rows["rpt_id"].ne("")].copy()
    keys = [
        "dataset_part",
        "group_id",
        "physical_cell_id",
        "rpt_id",
        "rpt_order",
        "protocol_group",
        "direction",
    ]
    result = []
    for key, group in rpt_rows.groupby(keys, dropna=False, sort=True):
        part, group_id, cell, rpt_id, order, protocol, direction = key
        curves = _select_curve_records(group)
        provisional = group["feature_mapping_status"].eq(
            "provisional_group5_file_order_mapping"
        ).any()
        gate_accepted = curves[curves["strict_gate_pass"]]
        if "observation_curve_step_status" in curves:
            accepted = gate_accepted[
                gate_accepted["observation_curve_step_status"].eq("verified")
            ]
        else:
            accepted = gate_accepted
        values: dict[str, float | None] = {}
        dispersions: dict[str, float | None] = {}
        for source, target in DEGRADATION_COLUMNS.items():
            dispersion, median = _quartile_dispersion(accepted[target])
            values[target] = median if median is not None else None
            dispersions[f"{target}_iqr"] = dispersion
        rmse_iqr, median_rmse = _quartile_dispersion(accepted["fit_rmse_mV"])
        flags = []
        if provisional:
            flags.append("AMBIGUOUS_GROUP5_RPT_MAPPING")
            values = {target: None for target in DEGRADATION_COLUMNS.values()}
            dispersions = {
                f"{target}_iqr": None for target in DEGRADATION_COLUMNS.values()
            }
            median_rmse = None
        elif not len(accepted) and len(gate_accepted):
            flags.append("UNAVAILABLE_UNVERIFIED_OBSERVATION_STEP")
            values = {target: None for target in DEGRADATION_COLUMNS.values()}
            dispersions = {
                f"{target}_iqr": None for target in DEGRADATION_COLUMNS.values()
            }
            median_rmse = None
        elif not len(accepted):
            flags.append("UNAVAILABLE_NO_STRICTLY_ACCEPTED_CURVE")
            values = {target: None for target in DEGRADATION_COLUMNS.values()}
            dispersions = {
                f"{target}_iqr": None for target in DEGRADATION_COLUMNS.values()
            }
            median_rmse = None
        else:
            if len(accepted) < len(gate_accepted):
                flags.append("REVIEW_UNVERIFIED_OBSERVATION_STEP_EXCLUDED")
            if len(accepted) == 1:
                flags.append("LOW_CONFIDENCE_SINGLE_CURVE")
            elif len(accepted) == 2:
                flags.append("REVIEW_MULTIPLE_CURVES_LOW_N")
            elif any(
                _within_group_outlier(accepted[target])
                for target in DEGRADATION_COLUMNS.values()
            ):
                flags.append("AMBIGUOUS_CURVE_DISAGREEMENT")
                values = {target: None for target in DEGRADATION_COLUMNS.values()}
            if curves["candidate_segment_count"].fillna(0).gt(1).any():
                flags.append("REVIEW_MULTIPLE_SOURCE_SEGMENTS")
            if any(
                accepted[target].map(
                    lambda value: value is not None
                    and pd.notna(value)
                    and float(value) < 0
                ).any()
                for target in DEGRADATION_COLUMNS.values()
            ):
                flags.append("REVIEW_NEGATIVE_ESTIMATE")
            if accepted["statistical_outlier_targets"].fillna("").ne("").any():
                flags.append("REVIEW_STATISTICAL_OUTLIER")

        exact_mapping = bool(group["is_exact_feature_mapped"].any())
        if provisional:
            quality = "AMBIGUOUS"
        elif not len(accepted):
            quality = "UNAVAILABLE"
        elif any(flag.startswith("AMBIGUOUS") for flag in flags):
            quality = "AMBIGUOUS"
        elif flags:
            quality = "REVIEW"
        else:
            quality = "PASS"
        nominal_age, age_status = _nominal_age(int(part), str(group_id), order)
        result.append(
            {
                "dataset_part": int(part),
                "group_id": str(group_id),
                "physical_cell_id": str(cell),
                "rpt_id": str(rpt_id),
                "rpt_order": order,
                "cycle_count": None,
                "nominal_cycle_age": nominal_age,
                "cycle_age_status": age_status,
                "protocol_group": str(protocol),
                "direction": str(direction),
                "selected_protocol": str(protocol),
                "aggregation_method": (
                    "discharge_curve_median"
                    if direction == "discharge"
                    else "charge_curve_median_separate"
                ),
                "n_available_curves": int(curves["curve_id"].nunique()),
                "n_accepted_curves": int(gate_accepted["curve_id"].nunique()),
                "n_mapping_eligible_accepted_curves": int(
                    accepted["curve_id"].nunique()
                ),
                "n_aggregated_curves": (
                    int(accepted["curve_id"].nunique())
                    if accepted.empty or not any(f.startswith("AMBIGUOUS") for f in flags)
                    else 0
                ),
                "median_voltage_fit_rmse_mV": median_rmse,
                "voltage_fit_rmse_iqr_mV": rmse_iqr,
                **values,
                **dispersions,
                "quality_flag": quality,
                "quality_flags": ";".join(flags),
                "exact_step4_feature_mapping": exact_mapping,
                "mapping_status": (
                    "provisional_group5_rpt_identity"
                    if provisional
                    else (
                        "exact_step4_source_match"
                        if exact_mapping
                        else "pydma_rpt_only_not_joined_to_step4"
                    )
                ),
                "curve_ids": ";".join(sorted(set(curves["curve_id"].astype(str)))),
                **_group_metadata(
                    group,
                    [
                        "observation_mapping_status",
                        "observation_source_match_basis",
                        "observation_curve_step_status",
                        "observation_curve_step_segment_ref",
                        "mapped_bol_reference_file",
                        "observation_step4_status",
                        "observation_step8_status",
                        "observation_step10_status",
                        "observation_step4_segment_ref",
                        "observation_step8_segment_ref",
                        "observation_step10_segment_ref",
                        "observation_ready_soh",
                        "observation_ready_degradation",
                        "observation_needs_audit",
                        "observation_audit_notes",
                        "degradation_reference_policy",
                        "degradation_reference_file",
                        "degradation_reference_status",
                        "degradation_reference_limitations",
                        "degradation_reference_evidence",
                        "degradation_reference_sources",
                    ],
                ),
            }
        )
    return pd.DataFrame(result)


def _build_hybrid_results(
    predictions: pd.DataFrame,
    aggregates: pd.DataFrame,
    fusion_predictions: pd.DataFrame | None = None,
) -> pd.DataFrame:
    merged = predictions.copy()
    for direction, prefix in (("discharge", "pydma_discharge"), ("charge", "pydma_charge")):
        selected = aggregates[
            (aggregates["direction"] == direction)
            & aggregates["exact_step4_feature_mapping"]
        ].copy()
        selected["dataset_part"] = selected["dataset_part"].astype(str)
        selected = selected.rename(columns={"rpt_id": "measurement_id"})
        rename = {
            "lli_estimated_pct": f"{prefix}_lli_pct",
            "lam_anode_estimated_pct": f"{prefix}_lam_anode_pct",
            "lam_cathode_estimated_pct": f"{prefix}_lam_cathode_pct",
            "median_voltage_fit_rmse_mV": f"{prefix}_median_rmse_mV",
            "n_available_curves": f"{prefix}_n_available_curves",
            "n_accepted_curves": f"{prefix}_n_accepted_curves",
            "n_mapping_eligible_accepted_curves": (
                f"{prefix}_n_mapping_eligible_accepted_curves"
            ),
            "n_aggregated_curves": f"{prefix}_n_aggregated_curves",
            "quality_flag": f"{prefix}_quality_flag",
            "quality_flags": f"{prefix}_quality_flags",
            "mapping_status": f"{prefix}_mapping_status",
            "curve_ids": f"{prefix}_curve_ids",
            "observation_mapping_status": f"{prefix}_observation_mapping_status",
            "observation_curve_step_status": (
                f"{prefix}_observation_curve_step_status"
            ),
            "observation_step8_status": f"{prefix}_observation_step8_status",
            "observation_step10_status": f"{prefix}_observation_step10_status",
            "observation_step8_segment_ref": f"{prefix}_observation_step8_segment_ref",
            "observation_step10_segment_ref": f"{prefix}_observation_step10_segment_ref",
            "degradation_reference_policy": f"{prefix}_reference_policy",
            "degradation_reference_file": f"{prefix}_reference_file",
            "degradation_reference_status": f"{prefix}_reference_status",
            "degradation_reference_limitations": f"{prefix}_reference_limitations",
        }
        selected = selected.rename(columns=rename)
        keep = KEY_COLUMNS + list(rename.values())
        merged = merged.merge(
            selected[keep],
            on=KEY_COLUMNS,
            how="left",
            validate="one_to_one",
        )
        for name in rename.values():
            if name.endswith(
                (
                    "_n_available_curves",
                    "_n_accepted_curves",
                    "_n_mapping_eligible_accepted_curves",
                    "_n_aggregated_curves",
                )
            ):
                merged[name] = merged[name].fillna(0).astype(int)
            else:
                merged[name] = merged[name].fillna("")
    for target in ("lli", "lam_anode", "lam_cathode"):
        charge = pd.to_numeric(merged[f"pydma_charge_{target}_pct"], errors="coerce")
        discharge = pd.to_numeric(merged[f"pydma_discharge_{target}_pct"], errors="coerce")
        merged[f"charge_minus_discharge_{target}_pp"] = charge - discharge
    merged["degradation_quality_flag"] = np.where(
        merged["pydma_discharge_quality_flag"].eq(""),
        "UNAVAILABLE_NO_EXACT_DISCHARGE_FIT",
        merged["pydma_discharge_quality_flag"],
    )
    merged["target_relationship"] = "same-RPT estimation; no future observations used"
    merged["soh_model_input"] = "128-point Step 4 discharge voltage shape"
    merged["pydma_primary_direction"] = "discharge; charge remains separate"
    if fusion_predictions is not None:
        fusion_keys = [
            "dataset_part",
            "physical_cell_id",
            "measurement_id",
        ]
        fusion_columns = [
            *fusion_keys,
            "fusion_v2_predicted_soh_percent",
            "fusion_v2_model_version",
            "fusion_v2_input_schema_version",
            "fusion_v2_missingness_policy",
        ]
        missing_columns = set(fusion_columns) - set(fusion_predictions.columns)
        if missing_columns:
            raise ValueError(
                "Fusion predictions are missing columns: "
                f"{sorted(missing_columns)}"
            )
        fusion_key_values = fusion_predictions[fusion_keys].astype(str)
        if fusion_key_values.duplicated().any():
            raise ValueError("Fusion predictions contain duplicate observation keys.")
        fusion_values = pd.to_numeric(
            fusion_predictions["fusion_v2_predicted_soh_percent"],
            errors="raise",
        )
        if not np.isfinite(fusion_values.dropna()).all():
            raise ValueError("Fusion predictions contain non-finite SoH values.")
        fusion_join_keys = [f"_fusion_join_{key}" for key in fusion_keys]
        fusion_join = fusion_predictions[fusion_columns].copy()
        merged_with_keys = merged.copy()
        for key, join_key in zip(fusion_keys, fusion_join_keys):
            merged_with_keys[join_key] = merged_with_keys[key].astype(str)
            fusion_join[join_key] = fusion_join[key].astype(str)
        fusion_join = fusion_join.drop(columns=fusion_keys)
        merged = merged_with_keys.merge(
            fusion_join,
            on=fusion_join_keys,
            how="left",
            validate="one_to_one",
        )
        merged = merged.drop(columns=fusion_join_keys)
        if not merged["fusion_v2_predicted_soh_percent"].notna().any():
            raise ValueError(
                "No fusion prediction keys matched the hybrid result rows."
            )
        merged["fusion_v2_status"] = np.where(
            merged["fusion_v2_predicted_soh_percent"].notna(),
            "available_complete_modalities",
            "no_complete_same_rpt_fusion_prediction",
        )
    return merged


def _build_checkpoint_manifest(
    frame: pd.DataFrame, curve_rows: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    feature_keys = set()
    for _, row in frame.iterrows():
        part = int(row["dataset_part"])
        cell = str(row["physical_cell_id"])
        rpt = str(row["measurement_id"])
        checkpoint_id = f"{part}|{cell}|{rpt}"
        feature_keys.add((part, cell, rpt))
        related = curve_rows[
            (curve_rows["dataset_part"] == part)
            & (curve_rows["physical_cell_id"] == cell)
            & (curve_rows["rpt_id"] == rpt)
        ]
        exact = related[related["is_exact_feature_mapped"]]
        rows.append(
            {
                "checkpoint_id": checkpoint_id,
                "physical_cell_id": cell,
                "dataset_part": part,
                "group_id": str(row["group_id"]),
                "rpt_id": rpt,
                "measurement_id": rpt,
                "rpt_order": int(row["rpt_order"]),
                "cycle_count": None,
                "nominal_cycle_age": float(row["nominal_cycle_age"]),
                "cycle_age_status": str(row["cycle_age_status"]),
                "step4_curve_id": (
                    f"P{part}|{cell}|{rpt}|{_basename(row['source_file'])}|"
                    f"Step4Segment{row['step4_segment_index']}"
                ),
                "step4_source_file": str(row["source_file"]),
                "step4_segment_index": row["step4_segment_index"],
                "step4_row_count": row["step4_row_count"],
                "cnn1d_input_available": True,
                "predicted_soh_percent": row["predicted_soh_percent"],
                "reference_soh_percent": row["reference_soh_percent"],
                "reference_label_scope": row["reference_label_scope"],
                "mapped_source_tpg": row["mapped_source_tpg"],
                "mapped_source_file": row["mapped_source_file"],
                "mapped_bol_reference_file": row["mapped_bol_reference_file"],
                "observation_mapping_status": row["observation_mapping_status"],
                "observation_source_match_basis": row[
                    "observation_source_match_basis"
                ],
                "observation_step4_status": row["observation_step4_status"],
                "observation_step8_status": row["observation_step8_status"],
                "observation_step10_status": row["observation_step10_status"],
                "step4_segment_ref": row["step4_segment_ref"],
                "step8_segment_ref": row["step8_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "observation_ready_soh": row["observation_ready_soh"],
                "observation_ready_degradation": row[
                    "observation_ready_degradation"
                ],
                "observation_needs_audit": row["observation_needs_audit"],
                "observation_audit_notes": row["observation_audit_notes"],
                "degradation_reference_policy": row[
                    "degradation_reference_policy"
                ],
                "degradation_reference_file": row["degradation_reference_file"],
                "degradation_reference_status": row[
                    "degradation_reference_status"
                ],
                "degradation_reference_limitations": row[
                    "degradation_reference_limitations"
                ],
                "degradation_reference_evidence": row[
                    "degradation_reference_evidence"
                ],
                "pocv_curve_ids": ";".join(sorted(set(related["curve_id"].astype(str)))),
                "mapped_pocv_curve_ids": ";".join(sorted(set(exact["curve_id"].astype(str)))),
                "pocv_directions": ";".join(sorted(set(related["direction"].astype(str)))),
                "pocv_curve_count": int(related["curve_id"].nunique()),
                "mapped_pocv_curve_count": int(exact["curve_id"].nunique()),
                "mapping_status": (
                    "verified_cell_rpt_and_source_file"
                    if len(exact)
                    else "step4_checkpoint_without_exact_pydma_curve"
                ),
            }
        )

    pydma_rpt = curve_rows[curve_rows["rpt_id"].ne("")]
    for key, group in pydma_rpt.groupby(
        ["dataset_part", "group_id", "physical_cell_id", "rpt_id"], sort=True
    ):
        part, group_id, cell, rpt = key
        if (int(part), str(cell), str(rpt)) in feature_keys:
            continue
        order = _rpt_order(rpt)
        age, age_status = _nominal_age(int(part), str(group_id), order)
        provisional = group["feature_mapping_status"].eq(
            "provisional_group5_file_order_mapping"
        ).any()
        mapped_ids = group.loc[group["is_exact_feature_mapped"], "curve_id"].astype(str)
        rows.append(
            {
                "checkpoint_id": f"{part}|{cell}|{rpt}",
                "physical_cell_id": str(cell),
                "dataset_part": int(part),
                "group_id": str(group_id),
                "rpt_id": str(rpt),
                "measurement_id": str(rpt),
                "rpt_order": order,
                "cycle_count": None,
                "nominal_cycle_age": age,
                "cycle_age_status": age_status,
                "step4_curve_id": "",
                "step4_source_file": "",
                "step4_segment_index": "",
                "step4_row_count": "",
                "cnn1d_input_available": False,
                "predicted_soh_percent": None,
                "reference_soh_percent": None,
                "reference_label_scope": "unavailable_no_step4_checkpoint",
                "pocv_curve_ids": ";".join(sorted(set(group["curve_id"].astype(str)))),
                "mapped_pocv_curve_ids": ";".join(sorted(set(mapped_ids))),
                "pocv_directions": ";".join(sorted(set(group["direction"].astype(str)))),
                "pocv_curve_count": int(group["curve_id"].nunique()),
                "mapped_pocv_curve_count": int(mapped_ids.nunique()),
                **_group_metadata(
                    group,
                    [
                        "mapped_source_tpg",
                        "mapped_source_file",
                        "observation_mapping_status",
                        "observation_source_match_basis",
                        "mapped_bol_reference_file",
                        "observation_step4_status",
                        "observation_step8_status",
                        "observation_step10_status",
                        "observation_step4_segment_ref",
                        "observation_step8_segment_ref",
                        "observation_step10_segment_ref",
                        "observation_ready_soh",
                        "observation_ready_degradation",
                        "observation_needs_audit",
                        "observation_audit_notes",
                        "degradation_reference_policy",
                        "degradation_reference_file",
                        "degradation_reference_status",
                        "degradation_reference_limitations",
                        "degradation_reference_evidence",
                    ],
                    allow_multiple=True,
                ),
                "mapping_status": (
                    "provisional_group5_rpt_mapping_quarantined"
                    if provisional
                    else "pydma_rpt_without_step4_checkpoint"
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["dataset_part", "physical_cell_id", "rpt_order", "checkpoint_id"],
        na_position="last",
    )


def _quality_report(
    curve_rows: pd.DataFrame,
    aggregates: pd.DataFrame,
    manifest: pd.DataFrame,
) -> pd.DataFrame:
    summaries = []
    for part, direction in sorted(
        set(zip(curve_rows["dataset_part"].astype(int), curve_rows["direction"].astype(str)))
    ):
        rows = curve_rows[
            (curve_rows["dataset_part"] == part)
            & (curve_rows["direction"] == direction)
            & curve_rows["rpt_id"].ne("")
        ]
        summaries.append(
            {
                "dataset_part": part,
                "direction": direction,
                "result_records": len(rows),
                "unique_curves": int(rows["curve_id"].nunique()),
                "exact_step4_mapped_curves": int(
                    rows.loc[rows["is_exact_feature_mapped"], "curve_id"].nunique()
                ),
                "strictly_accepted_curves": int(
                    rows.loc[rows["strict_gate_pass"], "curve_id"].nunique()
                ),
                "step_mapping_eligible_accepted_curves": int(
                    rows.loc[
                        rows["strict_gate_pass"]
                        & rows["observation_curve_step_status"].eq("verified"),
                        "curve_id",
                    ].nunique()
                ),
                "step_mapping_excluded_curves": int(
                    rows.loc[
                        rows["strict_gate_pass"]
                        & ~rows["observation_curve_step_status"].eq("verified"),
                        "curve_id",
                    ].nunique()
                ),
                "provisional_mapping_curves": int(
                    rows.loc[
                        rows["feature_mapping_status"]
                        == "provisional_group5_file_order_mapping",
                        "curve_id",
                    ].nunique()
                ),
                "unmatched_curves": int(
                    rows.loc[
                        ~rows["is_exact_feature_mapped"]
                        & ~rows["feature_mapping_status"].str.contains("provisional"),
                        "curve_id",
                    ].nunique()
                ),
                "multi_segment_records": int(
                    rows["candidate_segment_count"].fillna(0).gt(1).sum()
                ),
                "negative_estimate_records": int(
                    rows["quality_flags"].str.contains("negative_degradation_estimate_review").sum()
                ),
                "statistical_outlier_records": int(
                    rows["statistical_outlier_targets"].fillna("").ne("").sum()
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    if not summary.empty:
        summary.insert(0, "section", "part_direction")
    unavailable = pd.DataFrame(
        [
            {
                "section": "checkpoint_coverage",
                "dataset_part": int(part),
                "direction": "",
                "result_records": len(rows),
                "unique_curves": int(rows["pocv_curve_count"].sum()),
                "exact_step4_mapped_curves": int(rows["mapped_pocv_curve_count"].sum()),
                "strictly_accepted_curves": "",
                "provisional_mapping_curves": int(
                    rows["mapping_status"].str.contains("provisional").sum()
                ),
                "unmatched_curves": int((~rows["cnn1d_input_available"]).sum()),
                "multi_segment_records": "",
                "negative_estimate_records": "",
                "statistical_outlier_records": "",
            }
            for part, rows in manifest.groupby("dataset_part")
        ]
    )
    return pd.concat([summary, unavailable], ignore_index=True, sort=False)


def _trajectory_review_flags(aggregates: pd.DataFrame) -> pd.DataFrame:
    result = aggregates.copy()
    result["trajectory_review_flags"] = ""
    values = list(DEGRADATION_COLUMNS.values())
    for _, group in result[
        (result["direction"] == "discharge")
        & result["quality_flag"].isin(["PASS", "REVIEW"])
    ].groupby(["dataset_part", "physical_cell_id"], sort=False):
        ordered = group.sort_values("rpt_order")
        for column in values:
            numeric = pd.to_numeric(ordered[column], errors="coerce")
            changes = numeric.diff().dropna()
            if len(changes) < 3:
                continue
            median_change = float(changes.abs().median())
            mad = float((changes.abs() - median_change).abs().median())
            threshold = median_change + 3.5 * mad
            if threshold <= 0:
                continue
            flagged_indices = changes.index[changes.abs() > threshold]
            for index in flagged_indices:
                current = result.at[index, "trajectory_review_flags"]
                result.at[index, "trajectory_review_flags"] = (
                    f"{current};sudden_{column}_change".strip(";")
                )
    return result


def _make_trajectories(
    root: Path,
    output_dir: Path,
    checkpoint_manifest: pd.DataFrame,
    hybrid: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> list[str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    discharge = aggregates[aggregates["direction"] == "discharge"].copy()
    checkpoint_manifest = checkpoint_manifest.copy()
    checkpoint_manifest["dataset_part"] = checkpoint_manifest["dataset_part"].astype(str)
    discharge["dataset_part"] = discharge["dataset_part"].astype(str)
    trajectory = checkpoint_manifest.merge(
        hybrid[
            KEY_COLUMNS
            + [
                "predicted_soh_percent",
                "reference_soh_percent",
                "reference_label_scope",
            ]
        ],
        on=KEY_COLUMNS,
        how="left",
        validate="one_to_one",
        suffixes=("", "_hybrid"),
    )
    trajectory = trajectory.merge(
        discharge[
            [
                "dataset_part",
                "physical_cell_id",
                "rpt_id",
                "lli_estimated_pct",
                "lam_anode_estimated_pct",
                "lam_cathode_estimated_pct",
                "lli_estimated_pct_iqr",
                "lam_anode_estimated_pct_iqr",
                "lam_cathode_estimated_pct_iqr",
                "quality_flag",
            ]
        ],
        left_on=["dataset_part", "physical_cell_id", "rpt_id"],
        right_on=["dataset_part", "physical_cell_id", "rpt_id"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_pydma"),
    )
    figures = []
    for cell, rows in trajectory.groupby("physical_cell_id", sort=True):
        rows = rows.sort_values(["rpt_order", "dataset_part"])
        has_nominal = rows["nominal_cycle_age"].notna().any()
        x = (
            pd.to_numeric(rows["nominal_cycle_age"], errors="coerce").to_numpy(dtype=float)
            if has_nominal
            else pd.to_numeric(rows["rpt_order"], errors="coerce").to_numpy(dtype=float)
        )
        x_label = (
            "Nominal cycle-age proxy (48 x RPT; not exact cycles)"
            if has_nominal
            else "RPT order (cycle age unavailable)"
        )
        fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
        soh = pd.to_numeric(rows["predicted_soh_percent"], errors="coerce")
        axes[0].plot(x, soh, "o-", label="CNN1D predicted SoH")
        reference = pd.to_numeric(rows["reference_soh_percent"], errors="coerce")
        if reference.notna().any():
            axes[0].plot(x, reference, "x--", label="Capacity-ratio reference")
        axes[0].set_ylabel("SoH (%)")
        axes[0].legend(loc="best", fontsize=8)
        metric_columns = [
            ("lli_estimated_pct", "LLI (%)"),
            ("lam_anode_estimated_pct", "Anode LAM (%)"),
            ("lam_cathode_estimated_pct", "Cathode LAM (%)"),
        ]
        for axis, (column, label) in zip(axes[1:], metric_columns):
            estimate = pd.to_numeric(rows[column], errors="coerce")
            axis.plot(x, estimate, "o-", label=label)
            dispersion_column = f"{column}_iqr"
            dispersion = pd.to_numeric(rows[dispersion_column], errors="coerce")
            valid = estimate.notna() & dispersion.notna()
            if valid.any():
                axis.errorbar(
                    x[valid.to_numpy()],
                    estimate[valid].to_numpy(dtype=float),
                    yerr=dispersion[valid].to_numpy(dtype=float) / 2.0,
                    fmt="none",
                    capsize=2,
                    alpha=0.6,
                )
            missing = estimate.isna().to_numpy()
            if missing.any():
                axis.scatter(
                    x[missing],
                    np.full(int(missing.sum()), 0.05),
                    transform=axis.get_xaxis_transform(),
                    marker="x",
                    color="#b22222",
                    label="Unavailable/quarantined",
                )
            review = rows["quality_flag"].fillna("").isin(["REVIEW", "AMBIGUOUS"])
            if review.any():
                axis.scatter(
                    x[review.to_numpy()],
                    estimate[review].to_numpy(dtype=float),
                    facecolors="none",
                    edgecolors="#e87500",
                    s=90,
                    label="Review",
                )
            axis.set_ylabel(label)
            axis.legend(loc="best", fontsize=8)
        axes[-1].set_xlabel(x_label)
        for axis in axes:
            axis.grid(True, alpha=0.25)
        fig.suptitle(f"{cell} | measured RPT checkpoints; no smoothing")
        fig.tight_layout()
        filename = f"{re.sub(r'[^A-Za-z0-9._-]+', '_', str(cell))}_trajectory.png"
        path = figure_dir / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figures.append(
            str(path.relative_to(root)) if path.is_relative_to(root) else str(path)
        )
    return figures


def _write_readme(output_dir: Path, counts: dict[str, Any]) -> None:
    partition_metrics = "\n".join(
        f"  | `{row['partition']}` | {row['n_checkpoints']} | "
        f"{row['mae_percentage_points']:.3f} | "
        f"{row['rmse_percentage_points']:.3f} | `{row['label_scope']}` |"
        for row in counts["partition_soh_metrics"]
    )
    observation_mapping_summary = ", ".join(
        f"{status}: {count}"
        for status, count in sorted(counts["observation_mapping_status_counts"].items())
    )
    reference_policy_summary = ", ".join(
        f"{status}: {count}"
        for status, count in sorted(counts["degradation_reference_status_counts"].items())
    )
    text = f"""# Hybrid CNN1D + PyDMA Battery Health

## Run

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = 'src'
& .\\.venv-torch-cuda\\Scripts\\python.exe -m bafuse.hybrid.hybrid_battery_health
```

The pipeline uses the saved single-output CNN1D checkpoint and saved scalers.
It does not train a model or run the PyDMA optimizer. It regenerates the
checkpoint/curve manifests, fit QC, direction-specific aggregates, evaluation
tables, trajectory figures, and the offline dashboard in this directory.

## Generated files

- `checkpoint_manifest.csv`: Step 4/PyDMA mapping, including unavailable and
  quarantined PyDMA-only RPTs.
- `pydma_curve_level_results.csv`: preserved fit summaries and source audit
  metadata, including failed, unmatched, baseline, and provisional records.
- `pydma_rpt_aggregated.csv`: per-RPT, protocol- and direction-specific
  estimates and quality flags. Discharge is primary; charge is not averaged
  with discharge.
- `hybrid_battery_health_results.csv`: one row per validated Step 4 feature
  checkpoint, with CNN1D SoH and only exact-mapped PyDMA estimates.
- `pydma_quality_report.csv`, `soh_evaluation.csv`, `run_metadata.json`,
  `dashboard.html`, and `figures/`.

## Scope and limitations

- The input feature run contains {counts["step4_checkpoints"]} checkpoints:
  {counts["part1_checkpoints"]} Part 1, {counts["part2_checkpoints"]} Part 2
  Groups 1–4, and {counts["part3_checkpoints"]} Part 3. The locked split is
  preserved; Part 3 is external/domain-shift evaluation only.
- The fit export contains {counts["pydma_result_records"]} curve-level
  records; {counts["pydma_strictly_accepted_curves"]} pass the strict gate
  (RMSE < 15 mV and at least 3 accepted runs). Of these,
  {counts["pydma_strictly_accepted_exact_mapped_records"]} also have an exact
  Step 4 source match. The {counts["pydma_exact_step4_mapped_records"]} exact
  mapped-record count includes fits that did not pass the strict gate.
- RPT and curve-step provenance is joined from the canonical RPT observation
  manifest. Locked Step 4 checkpoints require an exact source-path match,
  verified Step 4, and `ready_soh`; the other Step 8/10 audit flags are retained.
  Feature mapping statuses: {observation_mapping_summary}.
- PyDMA fit-gate acceptance is distinct from Step 8/10 mapping eligibility.
  Curves without a verified matching step remain visible as diagnostics but
  are excluded from degradation aggregates. Part 2 BoL policy decisions are
  reported separately from the SoH capacity-ratio reference:
  {reference_policy_summary}.
- {counts["hybrid_discharge_estimates_available"]} of the
  {counts["step4_checkpoints"]} Step 4 checkpoints have an accepted,
  exact-mapped discharge estimate;
  {counts["hybrid_discharge_estimates_unavailable"]} do not. A single
  accepted curve remains flagged `REVIEW` for reduced confidence.
- Same-time SoH evaluation (percentage points; Part 3 is provisional):

  | Split | Checkpoints | MAE | RMSE | Label scope |
  |---|---:|---:|---:|---|
{partition_metrics}

- Part 3 capacity-ratio labels remain provisional because its BoL denominator
  policy is unresolved. Its MAE/RMSE/R² are labeled provisional, not validated
  accuracy.
- Cycle count is not available. The output carries documented `48 x RPT`
  nominal proxies only where applicable; Part 2 Groups 5–6 are not assigned a
  cycle-age proxy.
- Group 5 `TPG5.3` file-order mappings are quarantined. Part 2 Group 6 is
  calendar-aging-only and has no Step 4 input in this baseline feature run.
- PyDMA fit rows contain degradation estimates, fit RMSE, accepted/total run
  counts, and curve audits, but no optimized parameter vectors or fitted
  arrays. No batch refit was run. Prior diagnostic overlays are linked only
  where available; other curves are marked measured-only/unavailable.
- The PyDMA Step 8/10 inputs are the documented C/24 pseudo-OCV curves, not
  equilibrium OCV. Multiple monotone segments are retained as a review flag
  when the source table saved only the selected longest segment.
- PyDMA-derived LLI/LAM are model diagnostics, not independently measured
  ground truth. Negative values are preserved and flagged, never clipped.
- Charge and discharge estimates are reported separately. Their difference is
  shown for review; there is no cross-direction averaging or fit-selection
  based on degradation values.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def run_pipeline(root: Path, output_dir: Path) -> dict[str, Any]:
    root = root.resolve()
    output_dir = output_dir.resolve()
    paths = load_project_paths(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, curves, config, model, checkpoint, feature_names = _load_feature_rows(root)
    predictions = _predict_soh(curves, model, checkpoint)
    parity_difference = _validate_prediction_parity(root, frame, predictions)

    prediction_frame = frame.copy()
    prediction_frame["predicted_soh_percent"] = predictions
    train_rows = prediction_frame[prediction_frame["partition"] == "train"]
    train_cell_mean = float(
        train_rows.groupby("physical_cell_id")["reference_soh_percent"].mean().mean()
    )
    prediction_frame["cycle_count"] = np.nan
    prediction_frame["soh_model_input"] = (
        "128-point Step 4 discharge voltage shape; saved train-only scaling"
    )
    evaluation = _evaluate_soh(prediction_frame, predictions)

    pydma = _load_pydma_records(root)
    pydma = _map_pydma_to_features(pydma, prediction_frame)
    aggregates = _trajectory_review_flags(_aggregate_pydma(pydma))
    fusion_predictions_path = (
        root
        / "experiments"
        / "soh_forecast"
        / "bafuse_three_encoder_v1"
        / "fusion_predictions.csv"
    )
    fusion_predictions = (
        pd.read_csv(fusion_predictions_path)
        if fusion_predictions_path.is_file()
        else None
    )
    if fusion_predictions is None:
        print(
            "Three-encoder SoH predictions not found; "
            "retaining CNN1D and PyDMA outputs only."
        )
    hybrid = _build_hybrid_results(
        prediction_frame, aggregates, fusion_predictions=fusion_predictions
    )
    manifest = _build_checkpoint_manifest(prediction_frame, pydma)
    quality = _quality_report(pydma, aggregates, manifest)

    output_files = {
        "checkpoint_manifest.csv": manifest,
        "pydma_curve_level_results.csv": pydma,
        "pydma_rpt_aggregated.csv": aggregates,
        "hybrid_battery_health_results.csv": hybrid,
        "pydma_quality_report.csv": quality,
        "soh_evaluation.csv": evaluation,
    }
    for filename, table in output_files.items():
        table.to_csv(output_dir / filename, index=False)

    figure_paths = _make_trajectories(root, output_dir, manifest, hybrid, aggregates)
    from app.hybrid_dashboard import write_dashboard

    write_dashboard(
        root=root,
        output_dir=output_dir,
        feature_names=feature_names,
        feature_frame=prediction_frame,
        checkpoint_manifest=manifest,
        curve_rows=pydma,
        hybrid=hybrid,
        aggregates=aggregates,
    )
    counts = {
        "step4_checkpoints": int(len(prediction_frame)),
        "part1_checkpoints": int((prediction_frame["dataset_part"] == "1").sum()),
        "part2_checkpoints": int((prediction_frame["dataset_part"] == "2").sum()),
        "part3_checkpoints": int((prediction_frame["dataset_part"] == "3").sum()),
        "pydma_result_records": int(len(pydma)),
        "pydma_strictly_accepted_curves": int(pydma["strict_gate_pass"].sum()),
        "pydma_step_mapping_eligible_curves": int(
            (
                pydma["strict_gate_pass"]
                & pydma["observation_curve_step_status"].eq("verified")
            ).sum()
        ),
        "observation_mapping_status_counts": {
            str(status): int(count)
            for status, count in prediction_frame[
                "observation_mapping_status"
            ].value_counts().items()
        },
        "degradation_reference_status_counts": {
            str(status): int(count)
            for status, count in prediction_frame[
                "degradation_reference_status"
            ].value_counts().items()
        },
        "pydma_exact_step4_mapped_records": int(pydma["is_exact_feature_mapped"].sum()),
        "pydma_strictly_accepted_exact_mapped_records": int(
            (
                pydma["strict_gate_pass"]
                & pydma["is_exact_feature_mapped"]
            ).sum()
        ),
        "pydma_multisegment_records": int(
            pydma["candidate_segment_count"].fillna(0).gt(1).sum()
        ),
        "hybrid_discharge_estimates_available": int(
            pd.to_numeric(
                hybrid["pydma_discharge_lli_pct"], errors="coerce"
            ).notna().sum()
        ),
        "hybrid_discharge_estimates_unavailable": int(
            pd.to_numeric(
                hybrid["pydma_discharge_lli_pct"], errors="coerce"
            ).isna().sum()
        ),
        "partition_soh_metrics": [
            {
                "partition": str(row.group_value),
                "label_scope": str(row.label_scope),
                "n_checkpoints": int(row.n_checkpoints),
                "mae_percentage_points": float(row.mae_percentage_points),
                "rmse_percentage_points": float(row.rmse_percentage_points),
            }
            for row in evaluation[evaluation["group_name"] == "partition"].itertuples(
                index=False
            )
        ],
        "hybrid_rows": int(len(hybrid)),
        "checkpoint_manifest_rows": int(len(manifest)),
        "prediction_parity_max_abs_difference": parity_difference,
        "train_cell_mean_prediction_percent": train_cell_mean,
        "split_version": json.loads(
            (paths.baseline_run / "metrics.json").read_text(encoding="utf-8")
        )["split_version"],
        "checkpoint_path": (
            str(paths.checkpoint.relative_to(root))
            if paths.checkpoint.is_relative_to(root)
            else str(paths.checkpoint)
        ),
        "input_feature_names": feature_names,
        "training_or_refit_performed": False,
        "trajectory_figures": figure_paths,
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(counts, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_readme(output_dir, counts)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run existing-checkpoint CNN1D inference and PyDMA result QC."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    paths = load_project_paths(args.root)
    output = args.output or paths.hybrid_output
    if not output.is_absolute():
        output = args.root / output
    summary = run_pipeline(args.root, output)
    print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
