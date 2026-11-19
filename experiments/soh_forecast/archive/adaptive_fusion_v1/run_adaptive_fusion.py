from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[2]
SOH_DIR = ROOT / "experiments" / "soh_forecast"
PHASE2_DIR = SOH_DIR / "bafuse_three_encoder_v1"
PHASE2_AUDIT_DIR = PHASE2_DIR / "evaluation_audit_v1"
PHASE3_DIR = SOH_DIR / "bafuse_three_encoder_ablation_v1"
DEFAULT_OUTPUT_DIR = (
    SOH_DIR / "bafuse_adaptive_v1" / "adaptive_modality_v1"
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(SOH_DIR))
sys.path.insert(0, str(PHASE2_DIR))
sys.path.insert(0, str(PHASE2_AUDIT_DIR))

import run_ablation_phase3 as phase3
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, build_physics_features


phase2 = phase3.phase2
PARTITIONS = phase3.PARTITIONS
SEEDS = phase3.SEEDS
CONFIGURATIONS = phase3.CONFIGURATIONS
MODALITY_CODES = {
    "discharge": "D",
    "charge": "C",
    "physics": "P",
}
EXPECTED_ELIGIBILITY = {
    "M1": {"train": 112, "validation": 43, "external_test": 82},
    "M2": {"train": 106, "validation": 41, "external_test": 77},
    "M3": {"train": 112, "validation": 43, "external_test": 78},
    "M4": {"train": 106, "validation": 41, "external_test": 73},
    "M5": {"train": 106, "validation": 41, "external_test": 73},
    "M6": {"train": 107, "validation": 42, "external_test": 79},
    "M7": {"train": 112, "validation": 43, "external_test": 78},
}
MODEL_INPUTS = {
    name: tuple(phase3.MODALITY_FEATURES[modality] for modality in modalities)
    for name, modalities in CONFIGURATIONS.items()
}
CURVE_COLUMNS = phase3.CURVE_COLUMNS
FEATURE_PATHS = {
    "discharge": PHASE2_DIR / "discharge_features.csv",
    "charge": PHASE2_DIR / "charge_features.csv",
    "physics": PHASE2_DIR / "physics_features.csv",
}
MODALITY_AVAILABILITY_PATH = PHASE2_DIR / "modality_availability.csv"
EXTERNAL_COVERAGE_PATH = PHASE2_AUDIT_DIR / "external_12_cell_coverage.csv"
PHASE3_PREDICTIONS_PATH = PHASE3_DIR / "ablation_predictions.csv"
MIN_DELTA = 1e-8
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 16
ROUTING_POLICIES = (
    "adaptive_validation_selected",
    "fixed_M4_for_DCP",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required adaptive fusion input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _finite(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {name}: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"Non-finite {name}: {value!r}")
    return parsed


def _key(part: Any, cell: Any, rpt: Any) -> tuple[str, str, str]:
    return phase3._key(part, cell, rpt)


def _feature_map(
    path: Path,
    columns: tuple[str, ...],
    partitions: set[str],
) -> dict[tuple[str, str, str], np.ndarray]:
    result: dict[tuple[str, str, str], np.ndarray] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"Feature file has no header: {path}")
        missing = set(columns) - set(reader.fieldnames)
        if missing:
            raise ValueError(f"{path.name} lacks feature columns: {sorted(missing)}")
        for row in reader:
            if row["partition"] not in partitions:
                continue
            key = _key(
                row.get("dataset_part", row.get("part", "")),
                row["physical_cell_id"],
                row.get("rpt_index", row.get("measurement_id", "")),
            )
            if key in result:
                raise ValueError(f"Duplicate feature key in {path.name}: {key}")
            result[key] = np.asarray(
                [_finite(row[column], f"{path.name}:{column}") for column in columns],
                dtype=np.float32,
            )
    return result


def _availability_index() -> dict[tuple[str, str, str], dict[str, str]]:
    result: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv(MODALITY_AVAILABILITY_PATH):
        key = _key(row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in result:
            raise ValueError(f"Duplicate modality availability key: {key}")
        result[key] = row
    return result


def _modality_bits(features: dict[str, np.ndarray]) -> tuple[str, ...]:
    return tuple(
        code
        for modality, code in MODALITY_CODES.items()
        if modality in features
    )


def _modality_pattern(features: dict[str, np.ndarray]) -> str:
    bits = _modality_bits(features)
    return "+".join(bits) if bits else "none"


def _is_complete_dcp(record: dict[str, Any]) -> bool:
    return all(name in record["features"] for name in ("discharge", "charge", "physics"))


def _expected_g10_counts() -> dict[str, int]:
    rows = _read_csv(EXTERNAL_COVERAGE_PATH)
    matches = [row for row in rows if row["physical_cell_id"] == "G10-Cell2"]
    if len(matches) != 1 or len(rows) != 12:
        raise ValueError("External coverage audit must list exactly 12 cells including G10-Cell2.")
    row = matches[0]
    required = {
        "valid_soh_reference_observations": 10,
        "available_step4_observations": 10,
        "valid_step10_charge_measurements": 9,
        "valid_physics_feature_observations": 10,
        "complete_modality_observations": 9,
    }
    for column, expected in required.items():
        observed = int(row[column])
        if observed != expected:
            raise ValueError(
                f"Unexpected G10-Cell2 audit count for {column}: {observed}"
            )
    return {
        "D": int(row["available_step4_observations"]),
        "C": int(row["valid_step10_charge_measurements"]),
        "P": int(row["valid_physics_feature_observations"]),
        "DC": int(row["complete_modality_observations"]),
        "DP": int(row["valid_physics_feature_observations"]),
        "CP": int(row["complete_modality_observations"]),
        "DCP": int(row["complete_modality_observations"]),
        "any": int(row["valid_soh_reference_observations"]),
    }


def _metadata_eligibility() -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    availability = _read_csv(MODALITY_AVAILABILITY_PATH)
    labeled = [row for row in availability if _truth(row["label_available"])]
    counts = {
        partition: Counter()
        for partition in PARTITIONS
    }
    for row in labeled:
        d = _truth(row["step4_raw_curve_valid"])
        c = _truth(row["step10_raw_curve_valid"])
        p = d and _truth(row["temperature_complete"])
        bits = {
            "discharge": d,
            "charge": c,
            "physics": p,
        }
        counts[row["partition"]]["labels"] += 1
        for config, modalities in CONFIGURATIONS.items():
            if all(bits[name] for name in modalities):
                counts[row["partition"]][config] += 1

    supplement = _expected_g10_counts()
    counts["external_test"]["labels"] += supplement["any"]
    for config, modalities in CONFIGURATIONS.items():
        codes = tuple(MODALITY_CODES[name] for name in modalities)
        if codes == ("D",):
            add = supplement["D"]
        elif codes == ("D", "C"):
            add = supplement["DC"]
        elif codes == ("D", "P"):
            add = supplement["DP"]
        elif codes == ("D", "C", "P"):
            add = supplement["DCP"]
        elif codes == ("C", "P"):
            add = supplement["CP"]
        elif codes == ("C",):
            add = supplement["C"]
        elif codes == ("P",):
            add = supplement["P"]
        else:
            raise ValueError(f"Unsupported adaptive configuration: {config}")
        counts["external_test"][config] += add

    expected_labels = {"train": 113, "validation": 44, "external_test": 84}
    for partition, expected in expected_labels.items():
        if counts[partition]["labels"] != expected:
            raise ValueError(
                f"Expected {expected} valid labels in {partition}, "
                f"found {counts[partition]['labels']}"
            )
    observed = {
        config: {
            partition: counts[partition][config]
            for partition in PARTITIONS
        }
        for config in CONFIGURATIONS
    }
    if observed != EXPECTED_ELIGIBILITY:
        raise ValueError(
            "Pre-training modality eligibility does not match the requested "
            f"counts. Expected {EXPECTED_ELIGIBILITY}, observed {observed}"
        )
    labels = {partition: counts[partition]["labels"] for partition in PARTITIONS}
    return observed, labels


def preflight() -> dict[str, Any]:
    split_by_cell, split_version = phase2._load_split()
    external_cells = {
        cell for cell, partition in split_by_cell.items() if partition == "external_test"
    }
    coverage = _read_csv(EXTERNAL_COVERAGE_PATH)
    if {row["physical_cell_id"] for row in coverage} != external_cells:
        raise ValueError("External coverage audit does not match the locked split.")
    eligibility, labels = _metadata_eligibility()
    return {
        "split_version": split_version,
        "physical_cell_counts": {
            partition: sum(value == partition for value in split_by_cell.values())
            for partition in PARTITIONS
        },
        "valid_label_counts": labels,
        "eligibility": eligibility,
        "g10_cell2_external_label_count": 10,
        "recovered_g3_rpts_included": False,
        "group5_group6_unlabeled_rows_included": False,
    }


def _feature_columns() -> dict[str, tuple[str, ...]]:
    return {
        "discharge": CURVE_COLUMNS,
        "charge": CURVE_COLUMNS,
        "physics": tuple(PHYSICS_FEATURE_NAMES),
    }


def _raw_physics_for_missing_keys(
    observations: list[dict[str, Any]],
    missing_keys: set[tuple[str, str, str]],
    discharge_map: dict[tuple[str, str, str], np.ndarray],
) -> dict[tuple[str, str, str], np.ndarray]:
    if not missing_keys:
        return {}
    observation_by_key = {
        (row["part"], row["physical_cell_id"], str(row["rpt_index"])): row
        for row in observations
        if row["soh_percent"] is not None
    }
    selected = [
        observation_by_key[key]
        for key in missing_keys
        if key in observation_by_key
    ]
    if len(selected) != len(missing_keys):
        raise ValueError(
            f"Missing canonical observation rows for physics keys: "
            f"{sorted(missing_keys - set(observation_by_key))}"
        )
    requests_by_source = phase2._segment_requests(selected)
    found_keys: set[tuple[str, str, str]] = set()
    for requests in requests_by_source.values():
        step4_requests = [
            request
            for request in requests
            if request["step"] == 4 and request["request_id"] in missing_keys
        ]
        if not step4_requests:
            continue
        found_keys.update(request["request_id"] for request in step4_requests)
        phase2._scan_source_requests(ROOT, step4_requests)
    if found_keys != missing_keys:
        raise ValueError(
            f"Raw Step 4 mapping missing for physics keys: "
            f"{sorted(missing_keys - found_keys)}"
        )

    result: dict[tuple[str, str, str], np.ndarray] = {}
    for row in selected:
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        step4 = row["step4_result"]
        if not step4 or not step4["valid"]:
            raise ValueError(f"Step 4 raw validation failed for physics key {key}")
        if not step4.get("temperature_complete", False):
            raise ValueError(f"Incomplete Step 4 temperature for physics key {key}")
        cached_discharge = discharge_map.get(key)
        raw_discharge = np.asarray(step4["curve"], dtype=np.float32)
        if cached_discharge is None or not np.allclose(
            raw_discharge, cached_discharge, rtol=0.0, atol=1e-5
        ):
            raise ValueError(
                f"Raw/cached discharge feature mismatch for physics key {key}"
            )
        values = build_physics_features(
            step4["voltage_raw"],
            step4["current_raw"],
            step4["temperature_raw"],
        )
        result[key] = np.asarray(
            [values[name] for name in PHYSICS_FEATURE_NAMES], dtype=np.float32
        )
    if set(result) != missing_keys:
        raise ValueError("Did not derive all requested Step 4 physics vectors.")
    return result


def _base_records(
    partitions: set[str],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str], dict[str, str]]]:
    availability = _availability_index()
    observations, split_version = phase2._load_observations()
    selected_observations = [
        row
        for row in observations
        if row["partition"] in partitions and row["soh_percent"] is not None
    ]
    partition_feature_names = _feature_columns()
    feature_maps = {
        name: _feature_map(
            FEATURE_PATHS[name],
            columns,
            partitions,
        )
        for name, columns in partition_feature_names.items()
    }

    expected_physics_keys: set[tuple[str, str, str]] = set()
    for row in selected_observations:
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        availability_row = availability.get(key)
        if availability_row is None or not _truth(availability_row["label_available"]):
            raise ValueError(f"Valid label absent from modality audit for {key}")
        d_valid = _truth(availability_row["step4_raw_curve_valid"])
        temp_valid = _truth(availability_row["temperature_complete"])
        if d_valid and temp_valid:
            expected_physics_keys.add(key)

    existing_physics_keys = set(feature_maps["physics"])
    unexpected_physics = existing_physics_keys - expected_physics_keys
    if unexpected_physics:
        raise ValueError(
            f"Cached physics feature has no valid Step 4/temp evidence: "
            f"{sorted(unexpected_physics)[:5]}"
        )

    missing_physics_keys = expected_physics_keys - existing_physics_keys
    feature_maps["physics"].update(
        _raw_physics_for_missing_keys(
            observations,
            missing_physics_keys,
            feature_maps["discharge"],
        )
    )
    records: list[dict[str, Any]] = []
    feature_sources: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in selected_observations:
        key = (row["part"], row["physical_cell_id"], str(row["rpt_index"]))
        availability_row = availability[key]
        flags = {
            "discharge": _truth(availability_row["step4_raw_curve_valid"]),
            "charge": _truth(availability_row["step10_raw_curve_valid"]),
            "physics": (
                _truth(availability_row["step4_raw_curve_valid"])
                and _truth(availability_row["temperature_complete"])
            ),
        }
        features: dict[str, np.ndarray] = {}
        sources: dict[str, str] = {}
        for modality, expected in flags.items():
            values = feature_maps[modality].get(key)
            if expected != (values is not None):
                raise ValueError(
                    f"{modality} availability/cache mismatch for {key}: "
                    f"metadata={expected}, cached={values is not None}"
                )
            if values is not None:
                features[modality] = values
                sources[modality] = (
                    "recomputed_from_verified_step4"
                    if modality == "physics" and key in missing_physics_keys
                    else f"saved_phase2_{modality}_features"
                )
        records.append(
            {
                "key": key,
                "part": key[0],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": int(row["rpt_index"]),
                "measurement_id": f"RPT{row['rpt_index']}",
                "partition": row["partition"],
                "soh_percent": _finite(row["soh_percent"], f"SOH label for {key}"),
                "label_scope": row["label_scope"],
                "label_status": row["label_status"],
                "source_file": row["source_file"],
                "step4_segment_ref": row["step4_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "identity_provenance": "canonical_observation_manifest",
                "feature_source": "approved_phase2_cache_and_step4_recovery",
                "features": features,
                "feature_sources": sources,
                "split_version": split_version,
            }
        )
        feature_sources[key] = sources

    if any(record["part"] == "3" for record in records if record["partition"] != "external_test"):
        raise ValueError("Part 3 records cannot be in Train or Validation.")
    return records, availability


def _supplemental_external_records() -> list[dict[str, Any]]:
    from audit_soh_evaluation import _load_supplemental_cell2

    supplemental, _ = _load_supplemental_cell2()
    records: list[dict[str, Any]] = []
    for item in supplemental:
        row = item["row"]
        label = item["label"]
        if label["validation_status"] != "calculated_from_measured_capacity":
            continue
        if label["label_scope"] != "supplemental_provisional_part3_capacity_ratio":
            raise ValueError("G10-Cell2 supplemental label provenance was not preserved.")
        if row["partition"] != "external_test":
            raise ValueError("G10-Cell2 supplemental data are not external-only.")
        step4 = row["step4_result"]
        step10 = row["step10_result"]
        features: dict[str, np.ndarray] = {}
        sources: dict[str, str] = {}
        if step4 and step4["valid"]:
            features["discharge"] = np.asarray(step4["curve"], dtype=np.float32)
            sources["discharge"] = "verified_g10_cell2_step4_raw_segment"
        if step10 and step10["valid"]:
            features["charge"] = np.asarray(step10["curve"], dtype=np.float32)
            sources["charge"] = "verified_g10_cell2_step10_raw_segment"
        physics = row.get("physics_features")
        if physics is not None:
            features["physics"] = np.asarray(
                [physics[name] for name in PHYSICS_FEATURE_NAMES], dtype=np.float32
            )
            sources["physics"] = "verified_g10_cell2_step4_physics"
        if any(not np.isfinite(values).all() for values in features.values()):
            raise ValueError(f"Non-finite G10-Cell2 feature for RPT{row['rpt_index']}")
        key = _key(
            label["dataset_part"],
            label["physical_cell_id"],
            label["measurement_id"],
        )
        records.append(
            {
                "key": key,
                "part": key[0],
                "physical_cell_id": key[1],
                "rpt_index": int(key[2]),
                "measurement_id": label["measurement_id"],
                "partition": "external_test",
                "soh_percent": _finite(label["soh_percent"], f"G10 label {key}"),
                "label_scope": label["label_scope"],
                "label_status": label["validation_status"],
                "source_file": label["source_file"],
                "step4_segment_ref": label["step4_segment_ref"],
                "step10_segment_ref": label["step10_segment_ref"],
                "identity_provenance": label["identity_provenance"],
                "feature_source": "verified_g10_cell2_supplemental_raw_segments",
                "features": features,
                "feature_sources": sources,
                "split_version": phase2._load_split()[1],
            }
        )
    if len(records) != 10 or len({row["key"] for row in records}) != 10:
        raise ValueError(f"Expected 10 distinct accepted G10-Cell2 labels, got {len(records)}.")
    return records


def _eligible_records(
    records: list[dict[str, Any]], config_id: str
) -> list[dict[str, Any]]:
    required = set(CONFIGURATIONS[config_id])
    return [row for row in records if required.issubset(row["features"])]


def _validate_records(
    records: list[dict[str, Any]],
    split_by_cell: dict[str, str],
    expected: dict[str, dict[str, int]],
) -> None:
    keys = [row["key"] for row in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate adaptive observation keys.")
    if any(row["physical_cell_id"] not in split_by_cell for row in records):
        raise ValueError("Adaptive data contain a cell outside the locked split.")
    if any(split_by_cell[row["physical_cell_id"]] != row["partition"] for row in records):
        raise ValueError("Adaptive observation partition disagrees with locked split.")
    if any(row["physical_cell_id"].startswith(("G5-", "G6-")) for row in records):
        raise ValueError("Unlabeled Group 5/6 observations entered supervised records.")
    forbidden_recovered = {
        ("1", "G3-Cell14", "11"),
        ("1", "G3-Cell14", "12"),
    }
    if forbidden_recovered & set(keys):
        raise ValueError("Provisional recovered G3-Cell14 labels entered the experiment.")
    train_cells = {
        row["physical_cell_id"] for row in records if row["partition"] == "train"
    }
    validation_cells = {
        row["physical_cell_id"] for row in records if row["partition"] == "validation"
    }
    if train_cells & validation_cells:
        raise ValueError("Physical cells overlap between Train and Validation.")
    observed = {
        config: {
            partition: len(
                [
                    row
                    for row in _eligible_records(records, config)
                    if row["partition"] == partition
                ]
            )
            for partition in PARTITIONS
        }
        for config in CONFIGURATIONS
    }
    if observed != expected:
        raise ValueError(
            f"Feature-backed eligibility differs from verified preflight counts: {observed}"
        )
    for config in CONFIGURATIONS:
        for partition in ("train", "validation"):
            selected = [
                row
                for row in _eligible_records(records, config)
                if row["partition"] == partition
            ]
            if not selected:
                raise ValueError(f"{config} has no {partition} observations.")


def _train_one(
    records: list[dict[str, Any]],
    config_id: str,
    seed: int,
    checkpoint_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    modalities = CONFIGURATIONS[config_id]
    eligible = _eligible_records(records, config_id)
    train_records = [row for row in eligible if row["partition"] == "train"]
    validation_records = [
        row for row in eligible if row["partition"] == "validation"
    ]
    if not train_records or not validation_records:
        raise ValueError(f"{config_id} requires non-empty Train and Validation.")
    if {row["physical_cell_id"] for row in train_records} & {
        row["physical_cell_id"] for row in validation_records
    }:
        raise ValueError(f"{config_id} contains physical-cell split leakage.")

    phase3._seed_everything(seed)
    train_inputs, train_target = phase3._arrays_for(train_records, modalities)
    validation_inputs, validation_target = phase3._arrays_for(
        validation_records, modalities
    )
    scalers: dict[str, dict[str, list[float]]] = {}
    normalized_train: dict[str, np.ndarray] = {}
    normalized_validation: dict[str, np.ndarray] = {}
    for name in modalities:
        mean, scale = phase3._fit_scaler(train_inputs[name], name)
        scalers[name] = {"mean": mean.tolist(), "scale": scale.tolist()}
        normalized_train[name] = (train_inputs[name] - mean) / scale
        normalized_validation[name] = (validation_inputs[name] - mean) / scale
    target_mean = float(train_target.mean())
    target_scale = float(train_target.std())
    if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale < 1e-8:
        raise ValueError(f"Invalid training-only target scaler for {config_id}/{seed}.")
    normalized_train_target = (train_target - target_mean) / target_scale
    normalized_validation_target = (validation_target - target_mean) / target_scale
    tensors = [
        torch.from_numpy(normalized_train[name].astype(np.float32))
        for name in modalities
    ]
    tensors.append(torch.from_numpy(normalized_train_target.astype(np.float32)))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(*tensors),
        batch_size=min(BATCH_SIZE, len(train_records)),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    model = phase3.AblationSoH(modalities).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    validation_tensors = {
        name: torch.from_numpy(values.astype(np.float32)).to(device)
        for name, values in normalized_validation.items()
    }
    validation_target_tensor = torch.from_numpy(
        normalized_validation_target.astype(np.float32)
    ).to(device)

    history: list[dict[str, Any]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        batch_losses: list[float] = []
        for batch in loader:
            inputs = {
                name: batch[index].to(device)
                for index, name in enumerate(modalities)
            }
            target = batch[-1].to(device).reshape(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss for {config_id}/{seed}.")
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))
        model.eval()
        with torch.no_grad():
            validation_prediction = model(validation_tensors)
            validation_loss = float(
                criterion(
                    validation_prediction,
                    validation_target_tensor.reshape(-1, 1),
                ).item()
            )
        if not math.isfinite(validation_loss):
            raise RuntimeError(f"Non-finite validation loss for {config_id}/{seed}.")
        history.append(
            {
                "epoch": epoch,
                "train_mse_scaled": statistics.mean(batch_losses),
                "validation_mse_scaled": validation_loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        if validation_loss < best_loss - MIN_DELTA:
            best_loss = validation_loss
            best_epoch = epoch
            stale_epochs = 0
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError(f"No Validation-selected checkpoint for {config_id}/{seed}.")
    model.load_state_dict(best_state)
    model.to(device)

    checkpoint_dir.mkdir(parents=True, exist_ok=False)
    checkpoint_path = checkpoint_dir / "adaptive_checkpoint.pt"
    checkpoint_payload = {
        "experiment_version": "bafuse-adaptive-modality-v1",
        "model_configuration": config_id,
        "modalities": modalities,
        "model_state_dict": best_state,
        "input_scalers": scalers,
        "target_mean_train_only": target_mean,
        "target_scale_train_only": target_scale,
        "training_seed": seed,
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "training_observation_keys": [row["key"] for row in train_records],
        "validation_observation_keys": [row["key"] for row in validation_records],
        "split_version": train_records[0]["split_version"],
        "pretraining": "none; independent adaptive configuration",
        "preprocessing_fit_scope": "training observations only",
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "early_stopping_minimum_improvement": MIN_DELTA,
        "batch_size": BATCH_SIZE,
        "optimizer": "AdamW(lr=0.001, weight_decay=0.0001)",
        "loss": "MSE on training-standardized SOH percent",
        "device": str(device),
        "external_used_for_fit_or_selection": False,
    }
    torch.save(checkpoint_payload, checkpoint_path)
    with (checkpoint_dir / "training_history.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    validation_scaled_prediction = phase3._predict_scaled(
        model, normalized_validation, device
    )
    validation_prediction = validation_scaled_prediction * target_scale + target_mean
    return {
        "checkpoint_path": checkpoint_path,
        "best_epoch": best_epoch,
        "best_validation_mse_scaled": best_loss,
        "validation_predictions": {
            row["key"]: float(value)
            for row, value in zip(validation_records, validation_prediction)
        },
        "validation_records": validation_records,
    }


def _select_dcp_route(
    run_results: dict[tuple[str, int], dict[str, Any]],
    train_validation_records: list[dict[str, Any]],
) -> dict[str, Any]:
    dcp_validation = [
        row
        for row in train_validation_records
        if row["partition"] == "validation" and _is_complete_dcp(row)
    ]
    expected_keys = {row["key"] for row in dcp_validation}
    if len(expected_keys) != 41:
        raise ValueError(f"Expected 41 complete-modality Validation keys, got {len(expected_keys)}.")
    per_seed: dict[str, dict[int, float]] = {"M2": {}, "M4": {}}
    for config in ("M2", "M4"):
        for seed in SEEDS:
            result = run_results[(config, seed)]
            prediction_map = result["validation_predictions"]
            if not expected_keys.issubset(prediction_map):
                raise ValueError(f"{config}/{seed} lacks complete-modality Validation keys.")
            actual = np.asarray(
                [row["soh_percent"] for row in dcp_validation], dtype=np.float64
            )
            predicted = np.asarray(
                [prediction_map[row["key"]] for row in dcp_validation],
                dtype=np.float64,
            )
            per_seed[config][seed] = phase3._metrics(actual, predicted)[
                "mae_soh_percentage_points"
            ]
    means = {
        config: float(np.mean(list(values.values())))
        for config, values in per_seed.items()
    }
    selected = (
        "M2"
        if means["M2"] <= means["M4"]
        else "M4"
    )
    return {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_partition": "validation",
        "selection_metric": "overall MAE in SOH percentage points on identical 41 D+C+P keys; mean over seeds",
        "selected_DCP_model": selected,
        "tie_break": "M2 (fewer modalities) if mean Validation MAE is exactly tied",
        "validation_mean_mae_by_seed": {
            config: {str(seed): value for seed, value in values.items()}
            for config, values in per_seed.items()
        },
        "validation_mean_mae_across_seeds": means,
        "validation_DCP_observations": len(expected_keys),
        "external_used_for_selection": False,
    }


def _predict_checkpoint(
    checkpoint_path: Path,
    records: list[dict[str, Any]],
    device: torch.device,
) -> dict[tuple[str, str, str], float]:
    if not records:
        return {}
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modalities = tuple(payload["modalities"])
    arrays, _ = phase3._arrays_for(records, modalities)
    scalers = payload["input_scalers"]
    normalized = {
        name: (
            arrays[name]
            - np.asarray(scalers[name]["mean"], dtype=np.float32)
        )
        / np.asarray(scalers[name]["scale"], dtype=np.float32)
        for name in modalities
    }
    model = phase3.AblationSoH(modalities)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    scaled = phase3._predict_scaled(model, normalized, device)
    predictions = (
        scaled * float(payload["target_scale_train_only"])
        + float(payload["target_mean_train_only"])
    )
    if not np.isfinite(predictions).all():
        raise ValueError(f"Non-finite predictions from {checkpoint_path}")
    return {
        row["key"]: float(value)
        for row, value in zip(records, predictions)
    }


def _route_model(
    features: dict[str, np.ndarray],
    policy: str,
    selected_dcp_model: str,
) -> str | None:
    bits = set(_modality_bits(features))
    if bits == {"D", "C", "P"}:
        return selected_dcp_model if policy == "adaptive_validation_selected" else "M4"
    if bits == {"D", "C"}:
        return "M2"
    if bits == {"D", "P"}:
        return "M3"
    if bits == {"C", "P"}:
        return "M5"
    if bits == {"D"}:
        return "M1"
    if bits == {"C"}:
        return "M6"
    if bits == {"P"}:
        return "M7"
    if not bits:
        return None
    raise ValueError(f"Unsupported modality pattern: {sorted(bits)}")


def _metric(actual: list[float], predicted: list[float]) -> dict[str, float | None]:
    return phase3._metrics(
        np.asarray(actual, dtype=np.float64),
        np.asarray(predicted, dtype=np.float64),
    )


def _population_scope(records: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    complete = [row for row in records if _is_complete_dcp(row)]
    incomplete = [row for row in records if not _is_complete_dcp(row)]
    scopes = [("overall_eligible", records)]
    if complete:
        scopes.append(("complete_DCP", complete))
    if incomplete:
        scopes.append(("incomplete_DCP", incomplete))
    return scopes


def _metric_row(
    *,
    result_type: str,
    model_configuration: str,
    routing_policy: str,
    seed: str,
    partition: str,
    population_scope: str,
    records: list[dict[str, Any]],
    predictions: dict[tuple[str, str, str], float],
    summary_statistic: str = "per_seed",
) -> dict[str, Any]:
    actual = [row["soh_percent"] for row in records]
    predicted = [predictions[row["key"]] for row in records]
    return {
        "result_type": result_type,
        "model_configuration": model_configuration,
        "routing_policy": routing_policy,
        "training_seed": seed,
        "partition": partition,
        "population_scope": population_scope,
        "summary_statistic": summary_statistic,
        "observation_count": len(records),
        "physical_cell_count": len({row["physical_cell_id"] for row in records}),
        **_metric(actual, predicted),
    }


def _summarize_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_fields = (
        "result_type",
        "model_configuration",
        "routing_policy",
        "partition",
        "population_scope",
    )
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["summary_statistic"] != "per_seed":
            continue
        grouped[tuple(row[field] for field in group_fields)].append(row)
    summaries: list[dict[str, Any]] = []
    metrics = (
        "mae_soh_percentage_points",
        "rmse_soh_percentage_points",
        "r2",
    )
    for key, selected in sorted(grouped.items()):
        summary = {field: value for field, value in zip(group_fields, key)}
        summary.update(
            {
                "training_seed": "",
                "summary_statistic": "mean_and_sample_std_across_seeds",
                "observation_count": selected[0]["observation_count"],
                "physical_cell_count": selected[0]["physical_cell_count"],
                "successful_training_seeds": len(selected),
            }
        )
        for metric in metrics:
            values = [
                float(row[metric]) for row in selected if row[metric] is not None
            ]
            summary[metric] = float(np.mean(values)) if values else ""
            summary[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else ""
            )
        summaries.append(summary)
    return summaries


def _prediction_row(
    record: dict[str, Any],
    prediction: float | None,
    *,
    prediction_type: str,
    model_configuration: str,
    routing_policy: str,
    selected_model: str,
    seed: int,
    checkpoint: str,
    status: str,
) -> dict[str, Any]:
    return {
        "prediction_type": prediction_type,
        "model_configuration": model_configuration,
        "routing_policy": routing_policy,
        "training_seed": seed,
        "partition": record["partition"],
        "dataset_part": record["part"],
        "physical_cell_id": record["physical_cell_id"],
        "measurement_id": record["measurement_id"],
        "rpt_index": record["rpt_index"],
        "source_file": record["source_file"],
        "modalities_available": _modality_pattern(record["features"]),
        "complete_modality_observation": _is_complete_dcp(record),
        "selected_model": selected_model,
        "checkpoint": checkpoint,
        "reference_soh_percent": record["soh_percent"],
        "label_scope": record["label_scope"],
        "label_status": record["label_status"],
        "predicted_soh_percent": "" if prediction is None else prediction,
        "absolute_error_soh_percentage_points": (
            "" if prediction is None else abs(prediction - record["soh_percent"])
        ),
        "prediction_status": status,
        "feature_source": record["feature_source"],
        "identity_provenance": record["identity_provenance"],
    }


def _routing_rows(
    records: list[dict[str, Any]],
    prediction_maps: dict[tuple[str, int], dict[tuple[str, str, str], float]],
    run_results: dict[tuple[str, int], dict[str, Any]],
    selected_dcp_model: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int], list[dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    route_records: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for policy in ROUTING_POLICIES:
        for seed in SEEDS:
            routed: list[dict[str, Any]] = []
            for record in records:
                model_name = _route_model(record["features"], policy, selected_dcp_model)
                prediction = (
                    prediction_maps[(model_name, seed)].get(record["key"])
                    if model_name is not None
                    else None
                )
                if model_name is not None and prediction is None:
                    raise ValueError(
                        f"Route {policy}/{seed} lacks {model_name} prediction for "
                        f"{record['key']}"
                    )
                routed_record = {
                    **record,
                    "selected_model": model_name or "",
                    "prediction": prediction,
                    "routing_policy": policy,
                }
                routed.append(routed_record)
                checkpoint = (
                    str(run_results[(model_name, seed)]["checkpoint_path"].relative_to(ROOT))
                    if model_name
                    else ""
                )
                rows.append(
                    _prediction_row(
                        record,
                        prediction,
                        prediction_type="routed_system",
                        model_configuration=policy,
                        routing_policy=policy,
                        selected_model=model_name or "",
                        seed=seed,
                        checkpoint=checkpoint,
                        status=(
                            "predicted"
                            if prediction is not None
                            else "unavailable_no_valid_modality"
                        ),
                    )
                )
            route_records[(policy, seed)] = routed
    return rows, route_records


def _route_coverage_rows(
    records: list[dict[str, Any]],
    selected_dcp_model: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    patterns = ("D+C+P", "D+C", "D+P", "C+P", "D", "C", "P", "none")
    for policy in ROUTING_POLICIES:
        for partition in PARTITIONS:
            partition_records = [row for row in records if row["partition"] == partition]
            for pattern in patterns:
                selected = [
                    row
                    for row in partition_records
                    if _modality_pattern(row["features"]) == pattern
                ]
                bits = {part for part in pattern.split("+") if part in {"D", "C", "P"}}
                features = {
                    name: np.asarray([0.0])
                    for name, code in MODALITY_CODES.items()
                    if code in bits
                }
                route = (
                    _route_model(features, policy, selected_dcp_model)
                    if bits
                    else None
                )
                if selected:
                    available = sum(bool(route) for _ in selected)
                    cells = len({row["physical_cell_id"] for row in selected})
                    provisional = sum(
                        str(row["label_scope"]).startswith("supplemental_provisional")
                        or row["label_scope"] == "provisional_external_candidate"
                        for row in selected
                    )
                else:
                    available = cells = provisional = 0
                output.append(
                    {
                        "routing_policy": policy,
                        "partition": partition,
                        "modality_pattern": pattern,
                        "population_scope": (
                            "unavailable" if not bits else "complete_DCP" if bits == {"D", "C", "P"} else "incomplete_DCP"
                        ),
                        "selected_model": route or "",
                        "observation_count": len(selected),
                        "physical_cell_count": cells,
                        "prediction_available_observations": available,
                        "unavailable_observations": len(selected) - available,
                        "provisional_part3_label_observations": provisional,
                    }
                )
    return output


def _metrics_and_prediction_rows(
    all_records: list[dict[str, Any]],
    prediction_maps: dict[tuple[str, int], dict[tuple[str, str, str], float]],
    run_results: dict[tuple[str, int], dict[str, Any]],
    selected_dcp_model: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, int], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    metric_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        eligible = _eligible_records(all_records, config)
        for seed in SEEDS:
            pred_map = prediction_maps[(config, seed)]
            if set(pred_map) != {row["key"] for row in eligible}:
                raise ValueError(f"{config}/{seed} prediction coverage mismatch.")
            for partition in PARTITIONS:
                partition_records = [
                    row for row in eligible if row["partition"] == partition
                ]
                for scope, scoped in _population_scope(partition_records):
                    metric_rows.append(
                        _metric_row(
                            result_type="individual_model",
                            model_configuration=config,
                            routing_policy="",
                            seed=str(seed),
                            partition=partition,
                            population_scope=scope,
                            records=scoped,
                            predictions=pred_map,
                        )
                    )
            for record in eligible:
                prediction_rows.append(
                    _prediction_row(
                        record,
                        pred_map[record["key"]],
                        prediction_type="model_specific",
                        model_configuration=config,
                        routing_policy="",
                        selected_model=config,
                        seed=seed,
                        checkpoint=str(
                            run_results[(config, seed)]["checkpoint_path"].relative_to(ROOT)
                        ),
                        status="predicted",
                    )
                )

    route_prediction_rows, route_records = _routing_rows(
        all_records,
        prediction_maps,
        run_results,
        selected_dcp_model,
    )
    prediction_rows.extend(route_prediction_rows)
    for policy in ROUTING_POLICIES:
        for seed in SEEDS:
            pred_map = {
                row["key"]: row["prediction"]
                for row in route_records[(policy, seed)]
                if row["prediction"] is not None
            }
            predicted_records = [
                row for row in route_records[(policy, seed)] if row["prediction"] is not None
            ]
            for partition in PARTITIONS:
                partition_records = [
                    row for row in predicted_records if row["partition"] == partition
                ]
                for scope, scoped in _population_scope(partition_records):
                    metric_rows.append(
                        _metric_row(
                            result_type="routed_system",
                            model_configuration=policy,
                            routing_policy=policy,
                            seed=str(seed),
                            partition=partition,
                            population_scope=scope,
                            records=scoped,
                            predictions=pred_map,
                        )
                    )
    metric_rows.extend(_summarize_metric_rows(metric_rows))
    return metric_rows, prediction_rows, route_records, _route_coverage_rows(
        all_records, selected_dcp_model
    )


def _per_cell_rows(
    prediction_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row["prediction_status"] != "predicted":
            continue
        population = (
            "complete_DCP"
            if _truth(row["complete_modality_observation"])
            else "incomplete_DCP"
        )
        grouped[
            (
                row["prediction_type"],
                row["model_configuration"],
                int(row["training_seed"]),
                row["partition"],
                row["physical_cell_id"],
                population,
            )
        ].append(row)
    per_seed: list[dict[str, Any]] = []
    for (prediction_type, model, seed, partition, cell, population), selected in sorted(
        grouped.items()
    ):
        scores = _metric(
            [float(row["reference_soh_percent"]) for row in selected],
            [float(row["predicted_soh_percent"]) for row in selected],
        )
        per_seed.append(
            {
                "prediction_type": prediction_type,
                "model_configuration": model,
                "training_seed": seed,
                "partition": partition,
                "physical_cell_id": cell,
                "population_scope": population,
                "observation_count": len(selected),
                "summary_statistic": "per_seed",
                **scores,
            }
        )
    summary_groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        key = (
            row["prediction_type"],
            row["model_configuration"],
            row["partition"],
            row["physical_cell_id"],
            row["population_scope"],
        )
        summary_groups[key].append(row)
    summaries: list[dict[str, Any]] = []
    for key, selected in sorted(summary_groups.items()):
        summary: dict[str, Any] = {
            "prediction_type": key[0],
            "model_configuration": key[1],
            "training_seed": "",
            "partition": key[2],
            "physical_cell_id": key[3],
            "population_scope": key[4],
            "observation_count": selected[0]["observation_count"],
            "summary_statistic": "mean_and_sample_std_across_seeds",
            "successful_training_seeds": len(selected),
        }
        for name in ("mae_soh_percentage_points", "rmse_soh_percentage_points", "r2"):
            values = [float(row[name]) for row in selected if row[name] is not None]
            summary[name] = float(np.mean(values)) if values else ""
            summary[f"{name}_std"] = (
                float(np.std(values, ddof=1)) if len(values) > 1 else ""
            )
        summaries.append(summary)
    return per_seed + summaries


def _phase3_comparison_rows(
    model_prediction_rows: list[dict[str, Any]],
    routed_prediction_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical = _read_csv(PHASE3_PREDICTIONS_PATH)
    old: dict[tuple[str, int, str, tuple[str, str, str]], dict[str, Any]] = {}
    for row in historical:
        key = _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        index = (
            row["model_configuration"],
            int(row["training_seed"]),
            row["partition"],
            key,
        )
        if index in old:
            raise ValueError(f"Duplicate Phase 3 prediction key: {index}")
        old[index] = row

    adaptive: dict[tuple[str, int, str, tuple[str, str, str]], dict[str, Any]] = {}
    for row in model_prediction_rows:
        key = _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        adaptive[
            (row["model_configuration"], int(row["training_seed"]), row["partition"], key)
        ] = row

    output: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        for seed in SEEDS:
            for partition in PARTITIONS:
                adaptive_keys = {
                    index[3]
                    for index in adaptive
                    if index[:3] == (config, seed, partition)
                }
                phase3_keys = {
                    index[3]
                    for index in old
                    if index[:3] == (config, seed, partition)
                }
                common = sorted(adaptive_keys & phase3_keys)
                if not common:
                    raise ValueError(
                        f"No matched Phase 3 keys for {config}/{seed}/{partition}."
                    )
                current = [adaptive[(config, seed, partition, key)] for key in common]
                previous = [old[(config, seed, partition, key)] for key in common]
                for left, right in zip(current, previous):
                    if not math.isclose(
                        float(left["reference_soh_percent"]),
                        _finite(right["reference_soh_percent"], "Phase 3 label"),
                        rel_tol=0.0,
                        abs_tol=1e-8,
                    ):
                        raise ValueError(f"Phase 3 label mismatch for {left['physical_cell_id']}.")
                current_scores = _metric(
                    [float(row["reference_soh_percent"]) for row in current],
                    [float(row["predicted_soh_percent"]) for row in current],
                )
                previous_scores = _metric(
                    [float(row["reference_soh_percent"]) for row in previous],
                    [float(row["predicted_soh_percent"]) for row in previous],
                )
                output.append(
                    {
                        "comparison_type": "adaptive_model_vs_phase3_same_configuration",
                        "model_configuration": config,
                        "routing_policy": "",
                        "training_seed": seed,
                        "partition": partition,
                        "population_scope": "matched_keys",
                        "adaptive_eligible_observations": len(adaptive_keys),
                        "phase3_observations": len(phase3_keys),
                        "matched_observations": len(common),
                        "matched_physical_cells": len({key[1] for key in common}),
                        "expanded_adaptive_observations_outside_match": len(
                            adaptive_keys - set(common)
                        ),
                        "adaptive_mae_soh_percentage_points": current_scores[
                            "mae_soh_percentage_points"
                        ],
                        "phase3_mae_soh_percentage_points": previous_scores[
                            "mae_soh_percentage_points"
                        ],
                        "delta_mae_adaptive_minus_phase3": (
                            current_scores["mae_soh_percentage_points"]
                            - previous_scores["mae_soh_percentage_points"]
                        ),
                        "adaptive_rmse_soh_percentage_points": current_scores[
                            "rmse_soh_percentage_points"
                        ],
                        "phase3_rmse_soh_percentage_points": previous_scores[
                            "rmse_soh_percentage_points"
                        ],
                        "delta_rmse_adaptive_minus_phase3": (
                            current_scores["rmse_soh_percentage_points"]
                            - previous_scores["rmse_soh_percentage_points"]
                        ),
                        "adaptive_r2": current_scores["r2"],
                        "phase3_r2": previous_scores["r2"],
                        "delta_r2_adaptive_minus_phase3": (
                            current_scores["r2"] - previous_scores["r2"]
                            if current_scores["r2"] is not None
                            and previous_scores["r2"] is not None
                            else ""
                        ),
                    }
                )

    dcp_keys_by_partition = {
        partition: {
            row["key"]
            for row in records
            if row["partition"] == partition and _is_complete_dcp(row)
        }
        for partition in PARTITIONS
    }
    for policy_row in routed_prediction_rows:
        if policy_row["prediction_type"] != "routed_system":
            continue
        policy = policy_row["routing_policy"]
        if policy not in ROUTING_POLICIES:
            raise ValueError(f"Unexpected routed comparison policy: {policy}")
    route_index = {
        (
            row["routing_policy"],
            int(row["training_seed"]),
            row["partition"],
            _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"]),
        ): row
        for row in routed_prediction_rows
        if row["prediction_type"] == "routed_system"
        and row["prediction_status"] == "predicted"
    }
    for policy in ROUTING_POLICIES:
        for seed in SEEDS:
            for partition in PARTITIONS:
                common = sorted(dcp_keys_by_partition[partition])
                adaptive_rows = [
                    route_index[(policy, seed, partition, key)] for key in common
                ]
                baseline_rows = [
                    old[("M4", seed, partition, key)] for key in common
                ]
                adaptive_scores = _metric(
                    [float(row["reference_soh_percent"]) for row in adaptive_rows],
                    [float(row["predicted_soh_percent"]) for row in adaptive_rows],
                )
                baseline_scores = _metric(
                    [float(row["reference_soh_percent"]) for row in baseline_rows],
                    [float(row["predicted_soh_percent"]) for row in baseline_rows],
                )
                output.append(
                    {
                        "comparison_type": "routed_policy_vs_phase3_M4_on_DCP_keys",
                        "model_configuration": "M4",
                        "routing_policy": policy,
                        "training_seed": seed,
                        "partition": partition,
                        "population_scope": "matched_complete_DCP_keys",
                        "adaptive_eligible_observations": "",
                        "phase3_observations": "",
                        "matched_observations": len(common),
                        "matched_physical_cells": len({key[1] for key in common}),
                        "expanded_adaptive_observations_outside_match": "",
                        "adaptive_mae_soh_percentage_points": adaptive_scores[
                            "mae_soh_percentage_points"
                        ],
                        "phase3_mae_soh_percentage_points": baseline_scores[
                            "mae_soh_percentage_points"
                        ],
                        "delta_mae_adaptive_minus_phase3": (
                            adaptive_scores["mae_soh_percentage_points"]
                            - baseline_scores["mae_soh_percentage_points"]
                        ),
                        "adaptive_rmse_soh_percentage_points": adaptive_scores[
                            "rmse_soh_percentage_points"
                        ],
                        "phase3_rmse_soh_percentage_points": baseline_scores[
                            "rmse_soh_percentage_points"
                        ],
                        "delta_rmse_adaptive_minus_phase3": (
                            adaptive_scores["rmse_soh_percentage_points"]
                            - baseline_scores["rmse_soh_percentage_points"]
                        ),
                        "adaptive_r2": adaptive_scores["r2"],
                        "phase3_r2": baseline_scores["r2"],
                        "delta_r2_adaptive_minus_phase3": (
                            adaptive_scores["r2"] - baseline_scores["r2"]
                            if adaptive_scores["r2"] is not None
                            and baseline_scores["r2"] is not None
                            else ""
                        ),
                    }
                )
    return output


def _training_manifest_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        modalities = CONFIGURATIONS[config]
        for record in _eligible_records(records, config):
            rows.append(
                {
                    "model_configuration": config,
                    "training_seeds": ";".join(str(seed) for seed in SEEDS),
                    "dataset_part": record["part"],
                    "physical_cell_id": record["physical_cell_id"],
                    "measurement_id": record["measurement_id"],
                    "rpt_index": record["rpt_index"],
                    "partition": record["partition"],
                    "required_modalities": "+".join(
                        MODALITY_CODES[name] for name in modalities
                    ),
                    "available_modalities": _modality_pattern(record["features"]),
                    "source_file": record["source_file"],
                    "step4_segment_ref": record["step4_segment_ref"],
                    "step10_segment_ref": record["step10_segment_ref"],
                    "reference_soh_percent": record["soh_percent"],
                    "label_scope": record["label_scope"],
                    "label_status": record["label_status"],
                    "identity_provenance": record["identity_provenance"],
                    "feature_source": record["feature_source"],
                    "feature_sources": json.dumps(record["feature_sources"], sort_keys=True),
                    "used_for_training": record["partition"] == "train",
                    "used_for_validation_selection": record["partition"] == "validation",
                    "external_evaluation_only": record["partition"] == "external_test",
                    "split_version": record["split_version"],
                }
            )
    return rows


def _protected_paths() -> list[Path]:
    paths = set(phase3.PROTECTED_FILES)
    paths.add(ROOT / phase2.SPLIT_PATH)
    paths.add(ROOT / phase2.MANIFEST_PATH)
    paths.update(ROOT / value for value in phase2.LABEL_PATHS.values())
    paths.update(FEATURE_PATHS.values())
    paths.add(PHASE2_DIR / "modality_availability.csv")
    paths.add(PHASE2_DIR / "multimodal_feature_manifest.csv")
    if not PHASE3_DIR.is_dir():
        raise FileNotFoundError(f"Historical Phase 3 output is missing: {PHASE3_DIR}")
    paths.update(path for path in PHASE3_DIR.rglob("*") if path.is_file())
    missing = sorted(str(path) for path in paths if not path.is_file())
    if missing:
        raise FileNotFoundError(f"Cannot protect missing historical artifacts: {missing}")
    return sorted(paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): _sha256(path) for path in paths}


def _integrity_rows(
    before: dict[str, str], after: dict[str, str]
) -> list[dict[str, Any]]:
    if set(before) != set(after):
        raise ValueError("Protected artifact set changed during adaptive experiment.")
    rows = [
        {
            "protected_path": path,
            "sha256_before": before[path],
            "sha256_after": after[path],
            "unchanged": before[path] == after[path],
        }
        for path in sorted(before)
    ]
    if any(not row["unchanged"] for row in rows):
        raise RuntimeError("A protected Phase 2, Phase 3, label, split, or PyDMA artifact changed.")
    return rows


def _summary_lookup(
    metrics: list[dict[str, Any]],
    result_type: str,
    model: str,
    partition: str,
    population_scope: str = "overall_eligible",
) -> dict[str, Any]:
    for row in metrics:
        if (
            row["result_type"] == result_type
            and row["model_configuration"] == model
            and row["partition"] == partition
            and row["population_scope"] == population_scope
            and row["summary_statistic"] == "mean_and_sample_std_across_seeds"
        ):
            return row
    raise ValueError(
        f"Missing summary for {result_type}/{model}/{partition}/{population_scope}"
    )


def _fmt(mean: Any, std: Any, digits: int = 3) -> str:
    if mean in ("", None):
        return "n/a"
    if std in ("", None):
        return f"{float(mean):.{digits}f}"
    return f"{float(mean):.{digits}f} ± {float(std):.{digits}f}"


def _report(
    preflight_result: dict[str, Any],
    metrics: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    routing_decision: dict[str, Any],
    all_records: list[dict[str, Any]],
    integrity_rows: list[dict[str, Any]],
) -> str:
    expected = preflight_result["eligibility"]
    chosen = routing_decision["selected_DCP_model"]
    lines = [
        "# BaFuse Adaptive Fusion — Existing Validated Labels",
        "",
        "## Scope and safeguards",
        "",
        "- New, versioned experiment; the locked physical-cell split is "
        f"`{preflight_result['split_version']}` with 12/4/12 cells.",
        "- Supervision uses only already accepted Part 1/2 labels plus the "
        "existing external Part 3 candidate labels, including the 10 separate "
        "G10-Cell2 provisional labels.",
        "- The two provisional G3-Cell14 recovered candidates from Phase A/B are "
        "excluded. Group 5/6 unlabeled RPTs are excluded.",
        "- No imputation, label rewriting, use of PyDMA outputs, or changes to "
        "Phase 2/Phase 3 artifacts.",
        "- All input scalers and the target scaler are fit on each configuration's "
        "Train observations only. Validation alone selects checkpoints and the "
        "D+C+P route.",
        "- External Part 3 features and predictions are evaluated only after the "
        "route is frozen. Part 3 labels remain provisional.",
        "",
        "## Pre-training eligibility verification",
        "",
        "| Model | Modalities | Train | Validation | External Part 3 |",
        "|---|---|---:|---:|---:|",
    ]
    for config, modalities in CONFIGURATIONS.items():
        lines.append(
            f"| {config} | {'+'.join(MODALITY_CODES[name] for name in modalities)} | "
            f"{expected[config]['train']} | {expected[config]['validation']} | "
            f"{expected[config]['external_test']} |"
        )
    external_routed_count = sum(
        row["partition"] == "external_test"
        and _route_model(
            row["features"],
            ROUTING_POLICIES[0],
            routing_decision["selected_DCP_model"],
        )
        is not None
        for row in all_records
    )
    lines += [
        "",
        "The counts above were checked before training against verified Step 4, "
        "Step 10, and temperature availability. Physics availability is Step 4 "
        "plus complete temperature, using the existing five approved physics "
        "features. The historical physics cache omitted discharge-only cases "
        "because its previous feature writer required Step 10 as well; those "
        "missing vectors were recomputed from verified Step 4 raw segments with "
        "the same `build_physics_features` function. The cache itself was not changed.",
        "",
        "| Partition | Valid labels in experiment |",
        "|---|---:|",
    ]
    for partition in PARTITIONS:
        lines.append(
            f"| {partition} | {preflight_result['valid_label_counts'][partition]} |"
        )
    lines += [
        "",
        "## Training protocol",
        "",
        f"- Successful independent runs: {len(CONFIGURATIONS) * len(SEEDS)}/"
        f"{len(CONFIGURATIONS) * len(SEEDS)}; seeds {', '.join(map(str, SEEDS))}.",
        f"- Training device: `{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}`.",
        f"- Maximum {MAX_EPOCHS} epochs; Validation-loss early stopping with "
        f"patience {PATIENCE}, minimum improvement {MIN_DELTA}; batch size {BATCH_SIZE}.",
        "- AdamW (lr 0.001, weight decay 0.0001); MSE on Train-standardized SOH.",
        "- Each configuration is randomly initialized and saved independently "
        "for each seed. The Phase 2 and Phase 3 checkpoints are not warm starts.",
        "",
        "## Validation-only D+C+P route selection",
        "",
        "| Candidate | Validation MAE on identical 41 D+C+P keys (mean ± sample SD across seeds) |",
        "|---|---:|",
    ]
    for config in ("M2", "M4"):
        values = list(
            routing_decision["validation_mean_mae_by_seed"][config].values()
        )
        lines.append(
            f"| {config} | {_fmt(float(np.mean(values)), float(np.std(values, ddof=1)))} |"
        )
    lines += [
        "",
        f"Selected route for D+C+P: **{chosen}**, because its mean Validation MAE "
        f"across the three seeds is lower. The fixed comparison policy routes "
        "D+C+P to M4. No external result influenced this choice.",
        "",
        "## Overall model metrics",
        "",
        "MAE/RMSE are SOH percentage points. Values are overall metrics over each "
        "partition and eligible population, shown as seed mean ± sample standard "
        "deviation; R² is computed per seed over all eligible observations.",
        "",
    ]
    for partition in PARTITIONS:
        lines.extend(
            [
                f"### {partition}",
                "",
                "| Model / policy | Observations | Cells | MAE | RMSE | R² |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for model in (*CONFIGURATIONS.keys(), *ROUTING_POLICIES):
            kind = "individual_model" if model in CONFIGURATIONS else "routed_system"
            row = _summary_lookup(metrics, kind, model, partition)
            lines.append(
                f"| {model} | {row['observation_count']} | "
                f"{row['physical_cell_count']} | "
                f"{_fmt(row['mae_soh_percentage_points'], row['mae_soh_percentage_points_std'])} | "
                f"{_fmt(row['rmse_soh_percentage_points'], row['rmse_soh_percentage_points_std'])} | "
                f"{_fmt(row['r2'], row['r2_std'])} |"
            )
        lines.append("")

    lines += [
        "## Complete versus incomplete modality observations",
        "",
        "The complete cohort is D+C+P. Incomplete includes any non-empty strict "
        "subset of D/C/P. Each model's scores below use only keys eligible for "
        "that model; routed scores use all observations routed to a valid model.",
        "",
    ]
    for partition in PARTITIONS:
        lines.extend(
            [
                f"### {partition}",
                "",
                "| Policy | Population | Observations | Cells | MAE | RMSE | R² |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy in ROUTING_POLICIES:
            for scope in ("complete_DCP", "incomplete_DCP"):
                try:
                    row = _summary_lookup(
                        metrics, "routed_system", policy, partition, scope
                    )
                except ValueError:
                    continue
                lines.append(
                    f"| {policy} | {scope} | {row['observation_count']} | "
                    f"{row['physical_cell_count']} | "
                    f"{_fmt(row['mae_soh_percentage_points'], row['mae_soh_percentage_points_std'])} | "
                    f"{_fmt(row['rmse_soh_percentage_points'], row['rmse_soh_percentage_points_std'])} | "
                    f"{_fmt(row['r2'], row['r2_std'])} |"
                )
        lines.append("")

    lines += [
        "## Phase 3 matched-key comparison",
        "",
        "The CSV compares each adaptive model with the same Phase 3 configuration "
        "on identical keys, and compares each routed policy with historical Phase "
        "3 M4 on the common D+C+P keys. Expanded-coverage metrics above are not "
        "presented as matched-key comparisons. Differences can reflect changed "
        "training populations for M1/M3/M6/M7 as well as modality availability; "
        "they are not causal modality contributions.",
        "",
        f"Matched comparison rows generated: {len(comparison_rows)}.",
        "",
        "## External Part 3 coverage",
        "",
        f"The external split has 12 physical cells. The adaptive route produced "
        f"predictions for {external_routed_count}"
        " observations in the selected policy. Per-cell and per-modality coverage, "
        "selected route, provisional label status, and exact RPTs are in "
        "`adaptive_routing_coverage.csv` and `adaptive_predictions.csv`.",
        "",
        "All Part 3 references are provisional capacity-ratio labels, including "
        "the separate supplemental G10-Cell2 candidates. External results are "
        "descriptive only and were not used to select checkpoints, features, "
        "hyperparameters, or routing.",
        "",
        "## Integrity and artifacts",
        "",
        f"Protected historical artifacts verified unchanged: "
        f"{sum(row['unchanged'] for row in integrity_rows)}/{len(integrity_rows)}.",
        "No claim is made that adding modalities necessarily improves prediction. "
        "Per-cell diagnostics and seed-level uncertainty are in "
        "`adaptive_per_cell_metrics.csv` and `adaptive_model_metrics.csv`.",
        "",
        "## Reproducibility",
        "",
        "From the repository root:",
        "",
        "```powershell",
        ".venv-torch-cuda\\Scripts\\python.exe experiments\\soh_forecast\\run_adaptive_fusion.py --preflight-only",
        ".venv-torch-cuda\\Scripts\\python.exe experiments\\soh_forecast\\run_adaptive_fusion.py",
        ".venv-torch-cuda\\Scripts\\python.exe -m pytest -q tests\\test_bafuse_adaptive_fusion.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def run(output_dir: Path = DEFAULT_OUTPUT_DIR) -> None:
    preflight_result = preflight()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a non-empty adaptive experiment: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    protected_paths = _protected_paths()
    protected_before = _hashes(protected_paths)
    split_by_cell, _ = phase2._load_split()

    train_validation_records, _ = _base_records({"train", "validation"})
    train_validation_expected = {
        config: {
            "train": EXPECTED_ELIGIBILITY[config]["train"],
            "validation": EXPECTED_ELIGIBILITY[config]["validation"],
            "external_test": 0,
        }
        for config in CONFIGURATIONS
    }
    _validate_records(
        train_validation_records,
        split_by_cell,
        train_validation_expected,
    )
    run_results: dict[tuple[str, int], dict[str, Any]] = {}
    for config in CONFIGURATIONS:
        for seed in SEEDS:
            checkpoint_dir = output_dir / "checkpoints" / config / f"seed_{seed}"
            result = _train_one(
                train_validation_records,
                config,
                seed,
                checkpoint_dir,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            run_results[(config, seed)] = result
            print(
                f"{config} seed={seed}: best epoch {result['best_epoch']}, "
                f"validation MSE={result['best_validation_mse_scaled']:.6f}"
            )

    routing_decision = _select_dcp_route(run_results, train_validation_records)
    _write_json(output_dir / "routing_policy_frozen.json", routing_decision)
    print(
        f"Validation-frozen D+C+P route: {routing_decision['selected_DCP_model']}; "
        "starting the one-time external evaluation."
    )

    external_records, _ = _base_records({"external_test"})
    external_records.extend(_supplemental_external_records())
    all_records = train_validation_records + external_records
    full_expected = EXPECTED_ELIGIBILITY
    _validate_records(all_records, split_by_cell, full_expected)
    label_counts = Counter(row["partition"] for row in all_records)
    if dict(label_counts) != {"train": 113, "validation": 44, "external_test": 84}:
        raise ValueError(f"Unexpected final label populations: {dict(label_counts)}")
    external_supplemental = [
        row
        for row in all_records
        if row["physical_cell_id"] == "G10-Cell2"
    ]
    if len(external_supplemental) != 10 or any(
        row["partition"] != "external_test"
        or row["label_scope"] != "supplemental_provisional_part3_capacity_ratio"
        for row in external_supplemental
    ):
        raise ValueError("G10-Cell2 supplemental provenance/coverage mismatch.")

    prediction_maps: dict[
        tuple[str, int], dict[tuple[str, str, str], float]
    ] = {}
    for config in CONFIGURATIONS:
        eligible_all = _eligible_records(all_records, config)
        train_validation = [
            row for row in eligible_all if row["partition"] != "external_test"
        ]
        external = [
            row for row in eligible_all if row["partition"] == "external_test"
        ]
        for seed in SEEDS:
            result = run_results[(config, seed)]
            predictions = _predict_checkpoint(
                result["checkpoint_path"],
                train_validation,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
            predictions.update(
                _predict_checkpoint(
                    result["checkpoint_path"],
                    external,
                    torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                )
            )
            if set(predictions) != {row["key"] for row in eligible_all}:
                raise ValueError(f"Final prediction coverage mismatch: {config}/{seed}")
            prediction_maps[(config, seed)] = predictions

    manifest_rows = _training_manifest_rows(all_records)
    _write_csv(output_dir / "adaptive_training_manifest.csv", manifest_rows)
    metric_rows, prediction_rows, routed_records, coverage_rows = (
        _metrics_and_prediction_rows(
            all_records,
            prediction_maps,
            run_results,
            routing_decision["selected_DCP_model"],
        )
    )
    comparison_rows = _phase3_comparison_rows(
        [
            row
            for row in prediction_rows
            if row["prediction_type"] == "model_specific"
        ],
        [
            row
            for row in prediction_rows
            if row["prediction_type"] == "routed_system"
        ],
        all_records,
    )
    per_cell = _per_cell_rows(prediction_rows)
    integrity_rows = _integrity_rows(
        protected_before,
        _hashes(_protected_paths()),
    )
    _write_csv(output_dir / "adaptive_model_metrics.csv", metric_rows)
    _write_csv(output_dir / "adaptive_predictions.csv", prediction_rows)
    _write_csv(output_dir / "adaptive_routing_coverage.csv", coverage_rows)
    _write_csv(output_dir / "adaptive_per_cell_metrics.csv", per_cell)
    _write_csv(output_dir / "adaptive_vs_phase3_comparison.csv", comparison_rows)
    _write_csv(output_dir / "protected_artifact_integrity.csv", integrity_rows)

    report = _report(
        preflight_result,
        metric_rows,
        comparison_rows,
        routing_decision,
        all_records,
        integrity_rows,
    )
    (output_dir / "adaptive_fusion_final_report.md").write_text(
        report, encoding="utf-8"
    )
    print(
        f"Adaptive experiment complete: {len(run_results)}/{len(CONFIGURATIONS) * len(SEEDS)} "
        f"runs; outputs at {output_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate versioned BaFuse adaptive modality models."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Verify expected label and per-modality counts without training.",
    )
    args = parser.parse_args()
    if args.preflight_only:
        print(json.dumps(preflight(), indent=2))
        return
    run(args.output_dir)


if __name__ == "__main__":
    main()
