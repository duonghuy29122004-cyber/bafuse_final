from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
P5 = BENCHMARK / "bafuse_p5_electrical_v1_20261114"
P5_RUN = BENCHMARK / "bafuse_p5_trainval_benchmark_v2_20261114"
LEGACY = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
LEGACY_SPLIT = ROOT / "reports" / "bafuse_final_data_validation_v2" / "cell_split_12train_4val_12external.csv"
P5_CELL_SPLIT = BENCHMARK / "cell_split.csv"
OBSERVATIONS = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
RAW_EXCLUSIONS = BENCHMARK / "raw_soh_reconstruction_v1_20261113" / "review_exclusions.csv"
FORENSIC = ROOT / "experiments" / "soh_forecast" / "bafuse_legacy_p5_forensic_audit_v1_20261114"
FORENSIC_MANIFEST = FORENSIC / "forensic_audit_sha256_manifest.json"
FORENSIC_ALIGNMENT = FORENSIC / "legacy_vs_p5_observation_alignment.csv"
FORENSIC_MANIFEST_SHA256 = "8cebb0cc17772fba0bce63fa022ffb766f89397166de57ed6bfcd86bfef52dc5"
P5_MANIFEST = P5 / "bafuse_p5_sha256_manifest.json"
EXPECTED_P5_MANIFEST_SHA256 = "a6f8e5bf199cf4d059f0db784885e249d27a1cf50ed410c9ebcbd0d1815c8457"
SEEDS = (42, 123, 2026)
P5_FEATURE_NAMES = (
    "mean_abs_discharge_current",
    "discharge_current_variability",
    "discharge_voltage_slope_20_80",
    "discharge_voltage_derivative_mad_10_90",
    "discharge_voltage_segment_contrast_10_30_vs_70_90",
)
P5_RAW_COLUMNS = tuple(f"p5_raw_{name}" for name in P5_FEATURE_NAMES)
P5_SCALED_COLUMNS = tuple(f"p5_scaled_{name}" for name in P5_FEATURE_NAMES)
LEGACY_FEATURE_NAMES = (
    "mean_discharge_temperature_native",
    "discharge_temperature_rise_native",
    "discharge_plateau_slope_v_per_progress",
    "discharge_slope_variability_v_per_progress",
    "mean_absolute_discharge_current_native",
)
LEGACY_COLUMN_MAP = {
    name: name for name in LEGACY_FEATURE_NAMES
}
TRAINING = {
    "seeds": list(SEEDS),
    "max_epochs": 300,
    "batch_size": 16,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "loss": "MSELoss",
    "early_stopping_patience": 35,
    "early_stopping_min_delta": 0.0,
    "checkpoint_metric": "validation_RMSE",
    "target_transform": None,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def key(row: dict[str, str]) -> tuple[str, str, str]:
    return row["dataset_part"], row["physical_cell_id"], row["rpt_index"]


def forensic_key(row: dict[str, str]) -> tuple[str, str, str]:
    rpt_id = row["rpt_id"]
    if not rpt_id.startswith("RPT"):
        raise ValueError(f"Unexpected forensic RPT identifier: {rpt_id}")
    return row["dataset_part"], row["physical_cell_id"], rpt_id[3:]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _verify_p5_manifest() -> dict[str, Any]:
    observed = sha256(P5_MANIFEST)
    if observed != EXPECTED_P5_MANIFEST_SHA256:
        raise ValueError(f"P5 frozen manifest hash mismatch: {observed}")
    manifest = _read_json(P5_MANIFEST)
    for section in ("outputs", "inputs", "code"):
        entries = manifest.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f"P5 manifest is missing {section}.")
        for name, entry in entries.items():
            path = P5 / name if section == "outputs" else ROOT / entry["path"]
            if not path.is_file():
                raise FileNotFoundError(f"P5 manifest input is missing: {path}")
            observed_file_hash = sha256(path)
            if observed_file_hash != entry["sha256"]:
                raise ValueError(
                    f"P5 manifest hash mismatch for {path}: {observed_file_hash}"
                )
    return manifest


def _verify_prior_forensic_alignment() -> list[dict[str, str]]:
    if sha256(FORENSIC_MANIFEST) != FORENSIC_MANIFEST_SHA256:
        raise ValueError("Prior forensic manifest hash changed.")
    manifest = _read_json(FORENSIC_MANIFEST)
    declared = manifest["outputs"]["legacy_vs_p5_observation_alignment.csv"]
    if sha256(FORENSIC_ALIGNMENT) != declared:
        raise ValueError("Prior forensic observation-alignment hash mismatch.")
    rows = read_csv(FORENSIC_ALIGNMENT)
    if len(rows) != 220:
        raise ValueError("Prior forensic alignment no longer has 220 rows.")
    return rows


def _read_npz(path: Path, count: int) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        expected = {"X_discharge", "X_charge", "X_physics", "y_soh"}
        if set(archive.files) != expected:
            raise ValueError(f"Unexpected NPZ members in {path.name}.")
        arrays = {name: archive[name].copy() for name in archive.files}
    expected_shapes = {
        "X_discharge": (count, 128),
        "X_charge": (count, 128),
        "X_physics": (count, 5),
        "y_soh": (count, 1),
    }
    for name, values in arrays.items():
        if values.dtype != np.float32 or values.shape != expected_shapes[name]:
            raise ValueError(f"Unexpected dtype/shape in {path.name}:{name}.")
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite values in {path.name}:{name}.")
    return arrays


def _summary(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std_population": float(np.std(values)),
        "min": float(np.min(values)),
        "q25": float(np.quantile(values, 0.25)),
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "max": float(np.max(values)),
    }


def _verify_and_stage_internal() -> tuple[list[dict[str, str]], dict[str, Any]]:
    labels = read_csv(P5 / "bafuse_soh_labels_frozen.csv")
    split_rows = read_csv(P5 / "bafuse_p5_split_manifest.csv")
    feature_rows = read_csv(P5 / "bafuse_p5_feature_audit.csv")
    label_by_key = {key(row): row for row in labels}
    split_by_key = {key(row): row for row in split_rows}
    audit_by_key = {key(row): row for row in feature_rows}
    if len(labels) != 203 or len(label_by_key) != 203:
        raise ValueError("Frozen P5 labels do not contain 203 unique keys.")
    if len(split_rows) != 203 or len(split_by_key) != 203:
        raise ValueError("Frozen P5 split does not contain 203 unique keys.")
    if len(feature_rows) != 203 or len(audit_by_key) != 203:
        raise ValueError("Frozen P5 feature audit does not contain 203 unique keys.")
    if set(label_by_key) != set(split_by_key) or set(split_by_key) != set(audit_by_key):
        raise ValueError("Frozen P5 labels, split, and features have different keys.")
    counts = Counter(row["partition"] for row in split_rows)
    if counts != Counter({"train": 102, "validation": 37, "external_test": 64}):
        raise ValueError(f"Frozen P5 partition counts changed: {dict(counts)}")
    cell_partition: dict[str, str] = {}
    for row in split_rows:
        cell = row["physical_cell_id"]
        previous = cell_partition.setdefault(cell, row["partition"])
        if previous != row["partition"]:
            raise ValueError(f"Physical cell crosses partitions: {cell}")
        if label_by_key[key(row)]["final_status"] != "ACCEPTED":
            raise ValueError(f"Non-accepted label found for {key(row)}")
        audit = audit_by_key[key(row)]
        if audit["partition"] != row["partition"]:
            raise ValueError(f"Feature partition mismatch for {key(row)}")
        if row["partition"] != "external_test":
            if audit["same_rpt_step_order_valid"] != "True":
                raise ValueError(f"Invalid Step mapping for {key(row)}")
            for flag in (
                "predictive_label_or_bol_feature",
                "future_rpt_feature",
                "predictive_temperature_feature",
            ):
                if audit[flag] != "False":
                    raise ValueError(f"Leakage guard {flag} failed for {key(row)}")
            if audit["step4_source_file"] != audit["step10_source_file"]:
                raise ValueError(f"Step 4/10 source mismatch for {key(row)}")

    preprocessing = _read_json(P5 / "bafuse_p5_preprocessing_config.json")
    mean = np.asarray(preprocessing["mean"], dtype=np.float64)
    scale = np.asarray(preprocessing["scale"], dtype=np.float64)
    if preprocessing["fit_partition"] != "train" or preprocessing["fit_row_count"] != 102:
        raise ValueError("Frozen P5 scaler is not fitted on the 102 Train rows.")
    if preprocessing["validation_and_external_fit_or_update"] is not False:
        raise ValueError("Frozen P5 scaler was updated beyond Train.")
    legacy_partition_by_cell = {
        row["physical_cell_id"]: row["partition"] for row in read_csv(LEGACY_SPLIT)
    }

    internal_rows: list[dict[str, str]] = []
    internal_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    all_feature_values: dict[tuple[str, str, str], list[float]] = {}
    for row in feature_rows:
        row_key = key(row)
        all_feature_values[row_key] = [
            float(row[column]) for column in P5_RAW_COLUMNS
        ]
        if row["partition"] == "external_test":
            continue
        if any(not math.isfinite(value) for value in all_feature_values[row_key]):
            raise ValueError(f"Nonfinite P5 physics feature for {row_key}")
        internal_rows.append(split_by_key[row_key])

    p5_inputs: dict[str, dict[str, np.ndarray]] = {}
    for partition, filename, count in (
        ("train", "bafuse_train_p5.npz", 102),
        ("validation", "bafuse_validation_p5.npz", 37),
    ):
        p5_inputs[partition] = _read_npz(P5 / filename, count)
        ordered = sorted(
            (row for row in split_rows if row["partition"] == partition),
            key=lambda row: int(row["split_row_index"]),
        )
        for index, row in enumerate(ordered):
            row_key = key(row)
            label_value = np.float32(label_by_key[row_key]["soh_percent"])
            if p5_inputs[partition]["y_soh"][index, 0] != label_value:
                raise ValueError(f"NPZ target mismatch for {row_key}")
            raw_physics = np.asarray(all_feature_values[row_key], dtype=np.float64)
            expected_physics = ((raw_physics - mean) / scale).astype(np.float32)
            if not np.allclose(
                p5_inputs[partition]["X_physics"][index],
                expected_physics,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError(f"P5 physics scaler alignment mismatch for {row_key}")
            internal_arrays["dataset_part"].append(np.asarray(row["dataset_part"]))
            internal_arrays["physical_cell_id"].append(np.asarray(row["physical_cell_id"]))
            internal_arrays["rpt_index"].append(np.asarray(row["rpt_index"]))
            internal_arrays["partition"].append(np.asarray(partition))
            internal_arrays["legacy_partition"].append(
                np.asarray(legacy_partition_by_cell[row["physical_cell_id"]])
            )
            for name in ("X_discharge", "X_charge", "y_soh"):
                internal_arrays[name].append(p5_inputs[partition][name][index])
            internal_arrays["X_physics_p5_raw"].append(raw_physics.astype(np.float32))
            internal_arrays["X_physics_p5_scaled"].append(
                p5_inputs[partition]["X_physics"][index]
            )
            legacy_feature = _legacy_feature_by_key[row_key]
            internal_arrays["X_physics_legacy_raw"].append(
                np.asarray(
                    [float(legacy_feature[LEGACY_COLUMN_MAP[name]]) for name in LEGACY_FEATURE_NAMES],
                    dtype=np.float32,
                )
            )
    if len(internal_rows) != 139 or len(internal_arrays["y_soh"]) != 139:
        raise ValueError("Expected 139 internal Train/Validation observations.")
    data = {
        name: np.asarray(values)
        for name, values in internal_arrays.items()
    }
    if set(data["partition"].tolist()) != {"train", "validation"}:
        raise ValueError("External rows entered the staged train/validation data.")
    if any(str(part) == "3" for part in data["dataset_part"].tolist()):
        raise ValueError("Part 3 entered the staged train/validation data.")
    np.savez_compressed(OUT / "data" / "internal_trainval_raw.npz", **data)

    summary = {
        "dataset_identifier": "bafuse-p5-electrical-v1-20261114",
        "source_manifest_sha256": sha256(P5_MANIFEST),
        "internal_observations": 139,
        "train_observations": 102,
        "validation_observations": 37,
        "external_rows_loaded_by_training_process": False,
        "external_npz_opened": False,
        "input_keys": ["dataset_part", "physical_cell_id", "rpt_index"],
        "input_tensors": {
            "X_discharge": {"shape": [139, 128], "dtype": "float32", "unit": "V"},
            "X_charge": {"shape": [139, 128], "dtype": "float32", "unit": "V"},
            "X_physics_p5_raw": {"shape": [139, 5], "dtype": "float32"},
            "X_physics_legacy_raw": {"shape": [139, 5], "dtype": "float32"},
            "y_soh": {"shape": [139, 1], "dtype": "float32", "unit": "percent"},
        },
        "source_files": {
            "labels": "experiments/soh_forecast/bafuse_architecture_benchmark_v1/bafuse_p5_electrical_v1_20261114/bafuse_soh_labels_frozen.csv",
            "split": "experiments/soh_forecast/bafuse_architecture_benchmark_v1/bafuse_p5_electrical_v1_20261114/bafuse_p5_split_manifest.csv",
            "feature_audit": "experiments/soh_forecast/bafuse_architecture_benchmark_v1/bafuse_p5_electrical_v1_20261114/bafuse_p5_feature_audit.csv",
            "legacy_physics": "experiments/soh_forecast/bafuse_three_encoder_v1/physics_features.csv",
        },
    }
    (OUT / "prepared_data_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return internal_rows, summary


_legacy_feature_by_key: dict[tuple[str, str, str], dict[str, str]] = {}


def _formula_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "feature_family": "curve",
            "name": "X_discharge[0:128]",
            "raw_measurement": "same-RPT mapped Step 4 Volts",
            "definition": "Linear interpolation at 128 equally spaced normalized source-row positions; native row order",
            "unit": "V",
            "uses_capacity_or_time_axis": False,
            "uses_target_or_BoL": False,
            "notes": "Voltage direction is preserved; Step 4 discharge current is negative and voltage decreases overall.",
        },
        {
            "feature_family": "curve",
            "name": "X_charge[0:128]",
            "raw_measurement": "same-RPT mapped Step 10 Volts",
            "definition": "Linear interpolation at 128 equally spaced normalized source-row positions; native row order",
            "unit": "V",
            "uses_capacity_or_time_axis": False,
            "uses_target_or_BoL": False,
            "notes": "Step 10 same-RPT source; charge current is positive and voltage increases overall.",
        },
    ]
    p5_definitions = (
        ("sample-weighted mean(abs(Step 4 Amps))", "inferred A"),
        ("1.4826 * median(abs(I - median(I))) over raw Step 4 Amps", "inferred A"),
        ("OLS slope of resampled Step 4 Volts versus normalized progress, 0.20<=p<=0.80", "V/progress"),
        ("1.4826 * MAD of centered finite differences of resampled Step 4 Volts, 0.10<=p<=0.90", "V/progress"),
        ("mean(V[0.10<=p<=0.30]) - mean(V[0.70<=p<=0.90])", "V"),
    )
    for name, (definition, unit) in zip(P5_FEATURE_NAMES, p5_definitions):
        rows.append(
            {
                "feature_family": "P5_physics",
                "name": name,
                "raw_measurement": "same-RPT Step 4 Amps and/or Volts",
                "definition": definition,
                "unit": unit,
                "uses_capacity_or_time_axis": False,
                "uses_target_or_BoL": False,
                "notes": "Engineered summary of the same-RPT electrical input; not an independent sensor.",
            }
        )
    legacy_definitions = (
        ("mean_discharge_temperature_native", "mean(Step 4 temp1)", "native units unresolved"),
        ("discharge_temperature_rise_native", "max(Step 4 temp1) - initial Step 4 temp1", "native units unresolved"),
        ("discharge_plateau_slope_v_per_progress", "OLS slope of resampled Step 4 Volts, 0.20<=p<=0.80", "V/progress"),
        ("discharge_slope_variability_v_per_progress", "1.4826 * MAD of centered voltage differences, 0.10<=p<=0.90", "V/progress"),
        ("mean_absolute_discharge_current_native", "mean(abs(Step 4 Amps))", "inferred A"),
    )
    for name, definition, unit in legacy_definitions:
        rows.append(
            {
                "feature_family": "legacy_physics",
                "name": name,
                "raw_measurement": "same-RPT Step 4 temperature, Volts, and/or Amps",
                "definition": definition,
                "unit": unit,
                "uses_capacity_or_time_axis": False,
                "uses_target_or_BoL": False,
                "notes": "Temperature calibration/location is undocumented; legacy temperature is not carried into P5.",
            }
        )
    rows.append(
        {
            "feature_family": "excluded",
            "name": "Step 4 duration, Amphr, capacity, cell/group/file/part/RPT identifiers",
            "raw_measurement": "raw timestamps/counters/metadata",
            "definition": "Not supplied to any model branch.",
            "unit": "n/a",
            "uses_capacity_or_time_axis": False,
            "uses_target_or_BoL": False,
            "notes": "Same-RPT capacity/counter-derived values could directly encode the label; identifiers are metadata only.",
        }
    )
    return rows


def _write_protocol() -> None:
    lines = [
        "# BaFuse formula and population audit; preregistered ablation protocol",
        "",
        "This protocol is written before launching any new training runs. The approved objective is Train-only controlled ablation on the unchanged P5 frozen labels and splits. No model is trained on Part 3; no External prediction is evaluated.",
        "",
        "## Frozen cohort and split policies",
        "",
        "- P5 cohort: 203 accepted labels, partitioned 102 Train / 37 Validation / 64 Part 3 External Test.",
        "- Legacy comparison cohort: 220 historical rows, 106 Train / 41 Validation / 73 External. Only the 203 common keys are considered; no omitted label is restored.",
        "- P5 Train/Validation experiments use 139 Part 1-2 observations. The historical split has 99 Train / 40 Validation among those same keys; P5 uses 102 / 37.",
        "- Every split is by physical cell. Part 1-2 continuation keys remain together; Part 3 remains External-only.",
        "- All P5 labels remain unchanged. The label is the existing same-RPT Step 4 discharge capacity-retention reference, not official Oxford ground truth.",
        "",
        "## Variants",
        "",
        "| ID | Curve inputs | Physics inputs | Network branch family | Purpose |",
        "|---|---|---|---|---|",
        "| A1 | Step 4 discharge | none | P5 CNN1D | discharge-only reference |",
        "| A2 | Step 4 discharge + Step 10 charge | none | P5 CNN1D | charge-branch ablation |",
        "| A3 | Step 4 discharge | P5 electrical physics | P5 CNN1D | P5 physics contribution without charge |",
        "| A4 | Step 4 discharge + Step 10 charge | P5 electrical physics | P5 CNN1D | full P5 reference; reuse exact saved outputs |",
        "| C1 | both curves | legacy physics | legacy GELU/dropout CNN1D | architecture-by-physics factorial |",
        "| C2 | both curves | P5 electrical physics | legacy GELU/dropout CNN1D | architecture-by-physics factorial |",
        "| C3 | both curves | legacy physics | P5 ReLU CNN1D | architecture-by-physics factorial |",
        "| C4 | both curves | P5 electrical physics | P5 ReLU CNN1D | same fitted model as A4; reuse exact outputs |",
        "| D-legacy | both curves | P5 electrical physics | P5 ReLU CNN1D | historical cell split sensitivity |",
        "| D-P5 | both curves | P5 electrical physics | P5 ReLU CNN1D | P5 split reference; same fitted model as A4 |",
        "",
        "A1-A3, C1-C3, and D-legacy are the only new model runs (21 runs). A4/C4/D-P5 are explicit references to the three saved P5 full-model runs, not duplicate training.",
        "",
        "## Training and preprocessing",
        "",
        "- Seeds: 42, 123, 2026. Maximum 300 epochs; batch size 16; AdamW, learning rate 0.001, weight decay 0.0001; MSELoss on raw SoH percent; patience 35; min_delta 0.",
        "- Select the best checkpoint independently per run by minimum Validation RMSE. No cross-seed or External selection.",
        "- All curve inputs stay in volts and use the frozen 128-point normalized-source-row interpolation. No capacity/time coordinate is used.",
        "- Physics features use population standardization fitted only on the Train rows for that split. For D-P5 reuse, use the exact frozen P5 Train-only scaler. For the historical split, refit on its 99 Train rows only.",
        "- No target scaler, imputer, augmentation, smoothing, feature selection, or label-derived transform is used.",
        "- The legacy network uses the legacy GELU activations and 0.2 fusion dropout; the P5 network uses ReLU and no fusion dropout. Curves and target scaling are held common at P5 raw-voltage/raw-target settings so these runs isolate the architecture block, not all historical pipeline differences.",
        "- A1-A3 remove branches, so their parameter counts differ. Report counts and do not interpret their differences as information-only effects.",
        "- Each run records overall and per-cell Train/Validation metrics; summary across seeds is mean and sample SD. No External model predictions or metrics are produced.",
        "",
        "## Split-sensitivity limitation",
        "",
        "The 99/40 historical and 102/37 P5 policies have disjoint Validation cell sets. Their own-validation metrics therefore combine training-composition, validation-difficulty, and checkpoint-selection differences. No common held-out cell set exists within these two locked policies; the comparison is descriptive and cannot identify those effects separately.",
        "",
        "## Guardrails",
        "",
        "- Training authorization applies only to these Train/Validation experiments. `external_prediction_evaluation_authorized=false`.",
        "- The training process reads only a sanitized 139-row Part 1-2 archive. It cannot access External arrays or labels.",
        "- External feature distribution summaries may be computed separately for descriptive domain-shift audit; they cannot influence feature definitions, preprocessing, run selection, or architecture decisions.",
        "- No PyDMA/Half-Cell fitting, RAG, dataset/label modification, frozen dataset rewrite, or historical checkpoint overwrite.",
        "",
        "## Reported comparisons",
        "",
        "- A1-A4: branch ablations; paired seed deltas and parameter counts.",
        "- C1-C4: physics effect within each architecture and architecture effect under each physics set; report paired seed deltas and difference-in-differences.",
        "- D-legacy versus D-P5: Train/Validation composition sensitivity with the limitations above.",
        "",
    ]
    (OUT / "ablation_experiment_protocol.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    preregistration = {
        "experiment_identifier": "bafuse-formula-ablation-v1-20261114",
        "training_authorized": True,
        "train_evaluation_authorized": True,
        "validation_evaluation_authorized": True,
        "checkpoint_selection_authorized": True,
        "external_prediction_evaluation_authorized": False,
        "dataset_modification_authorized": False,
        "label_modification_authorized": False,
        "pydma_fitting_authorized": False,
        "half_cell_fitting_authorized": False,
        "training": TRAINING,
        "new_run_count": 21,
        "seeds": list(SEEDS),
        "new_variants": list(
            (
                "A1_discharge_only",
                "A2_discharge_charge",
                "A3_discharge_p5_physics",
                "C1_legacy_arch_legacy_physics",
                "C2_legacy_arch_p5_physics",
                "C3_p5_arch_legacy_physics",
                "D_legacy_split_p5_full",
            )
        ),
        "reused_p5_full_runs": 3,
        "external_arrays_loaded_by_training_process": False,
    }
    (OUT / "ablation_preregistration_config.json").write_text(
        json.dumps(preregistration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _population_audit() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_predictions = read_csv(LEGACY / "fusion_predictions.csv")
    forensic_rows = _verify_prior_forensic_alignment()
    p5_labels = read_csv(P5 / "bafuse_soh_labels_frozen.csv")
    p5_split = read_csv(P5 / "bafuse_p5_split_manifest.csv")
    old_model_keys = {key(row) for row in old_predictions}
    legacy_keys = {
        forensic_key(row)
        for row in forensic_rows
        if row["present_in_legacy"].casefold() == "true"
    }
    p5_keys = {key(row) for row in p5_labels}
    if len(old_model_keys) != 211 or len(legacy_keys) != 220 or len(p5_keys) != 203:
        raise ValueError(
            f"Unexpected population sizes: legacy_saved_model={len(old_model_keys)}, "
            f"legacy_forensic_union={len(legacy_keys)}, P5={len(p5_keys)}"
        )
    p5_only = p5_keys - legacy_keys
    old_only = legacy_keys - p5_keys
    if len(p5_only) != 0 or len(old_only) != 17:
        raise ValueError(f"Unexpected cohort difference: P5-only={p5_only}, legacy-only={len(old_only)}")
    observation_rows = read_csv(OBSERVATIONS)
    observation_by_key = {
        (row["part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in observation_rows
    }
    exclusion_rows = read_csv(RAW_EXCLUSIONS)
    exclusion_by_cell = {
        row["physical_cell_id"]: row
        for row in exclusion_rows
        if row["record_type"] == "assigned_cell_without_candidate"
    }
    legacy_prediction_by_key = {key(row): row for row in old_predictions}
    forensic_by_key = {
        forensic_key(row): row for row in forensic_rows
    }
    rows: list[dict[str, Any]] = []
    for record_key in sorted(old_only, key=lambda value: (int(value[0]), value[1], int(value[2]))):
        part, cell, rpt = record_key
        forensic = forensic_by_key[record_key]
        if part == "3":
            excluded = exclusion_by_cell.get(cell)
            reason = excluded["reason"] if excluded else "No Guide-mapped P5 candidate row."
            source_status = "assigned_cell_without_candidate"
            evidence = "review_exclusions.csv: no candidate RPT row in the M1 Guide-mapped manifest"
            step4_notes = forensic["legacy_source_file"]
        else:
            observation = observation_by_key.get(record_key)
            if observation is None:
                raise ValueError(f"Legacy-only internal key is absent from observation manifest: {record_key}")
            reason = observation["audit_notes"]
            source_status = observation["mapping_status"]
            evidence = (
                f"Step4={observation['step4_status']}; Step10={observation['step10_status']}; "
                f"ready_soh={observation['ready_soh']}; needs_audit={observation['needs_audit']}"
            )
            step4_notes = observation["step4_segment_ref"]
            if (
                observation["mapping_status"] != "needs_curve_audit"
                or observation["needs_audit"].casefold() != "true"
                or "expected 1; no ordinal assignment made" not in observation["audit_notes"]
            ):
                raise ValueError(f"Internal legacy-only reason is not the expected curve ambiguity: {record_key}")
        old_row = legacy_prediction_by_key.get(record_key)
        rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "legacy_partition": forensic["legacy_partition"],
                "legacy_prediction_exists": forensic["present_in_legacy"],
                "legacy_prediction_source": (
                    "fusion_predictions.csv"
                    if old_row is not None
                    else "supplemental G10-Cell2 saved-checkpoint output in prior forensic audit"
                ),
                "p5_accepted_label_exists": False,
                "p5_candidate_or_feature_qc_applied": False,
                "p5_exclusion_basis": source_status,
                "raw_mapping_evidence": evidence,
                "source_file_or_step4_ref": (
                    step4_notes
                    or (old_row["source_file"] if old_row is not None else forensic["legacy_source_file"])
                ),
                "recommended_interpretation": (
                    "Quarantine is scientifically justified pending unique Step 4 segment assignment."
                    if part in {"1", "2"}
                    else "External exclusion is justified until Guide-backed physical-cell identity and RPT crosswalk are resolved."
                ),
                "reason": reason,
            }
        )
    if sum(row["dataset_part"] in {"1", "2"} for row in rows) != 8:
        raise ValueError("Expected eight internal legacy-only keys.")
    if sum(row["dataset_part"] == "3" for row in rows) != 9:
        raise ValueError("Expected nine External legacy-only keys.")
    write_csv(OUT / "cohort_population_audit.csv", rows)

    old_split_rows = read_csv(LEGACY_SPLIT)
    p5_cell_rows = read_csv(P5_CELL_SPLIT)
    old_by_cell = {row["physical_cell_id"]: row for row in old_split_rows}
    p5_by_cell = {row["physical_cell_id"]: row for row in p5_cell_rows}
    p5_internal_cells = sorted(
        {
            row["physical_cell_id"]
            for row in p5_split
            if row["partition"] in {"train", "validation"}
        }
    )
    assignments = []
    for cell in p5_internal_cells:
        old = old_by_cell[cell]
        new = p5_by_cell[cell]
        assignments.append(
            {
                "physical_cell_id": cell,
                "legacy_partition": old["partition"],
                "legacy_assignment_basis": old["assignment_basis"],
                "p5_partition": new["partition"],
                "p5_assignment_basis": new["assignment_basis"],
                "assignment_changed": old["partition"] != new["partition"],
            }
        )
    if len(assignments) != 12 or sum(row["assignment_changed"] for row in assignments) != 6:
        raise ValueError("Unexpected internal physical-cell assignment comparison.")
    write_csv(OUT / "cell_split_assignment_audit.csv", assignments)
    counts = Counter(row["partition"] for row in old_predictions)
    forensic_counts = Counter(row["legacy_partition"] for row in forensic_rows)
    p5_counts = Counter(row["partition"] for row in p5_labels)
    common_internal = [
        row for row in p5_split if row["partition"] in {"train", "validation"}
    ]
    old_partition_by_cell = {cell: row["partition"] for cell, row in old_by_cell.items()}
    shared_old_counts = Counter(old_partition_by_cell[row["physical_cell_id"]] for row in common_internal)
    summary = {
        "legacy_observations": len(legacy_keys),
        "legacy_primary_saved_model_rows": len(old_model_keys),
        "legacy_supplemental_g10_saved_checkpoint_rows": len(legacy_keys - old_model_keys),
        "legacy_partition_counts": dict(forensic_counts),
        "legacy_primary_saved_model_partition_counts": dict(counts),
        "p5_observations": len(p5_keys),
        "p5_partition_counts": dict(p5_counts),
        "common_keys": len(legacy_keys & p5_keys),
        "legacy_only": len(old_only),
        "p5_only": len(p5_only),
        "legacy_only_internal": 8,
        "legacy_only_external": 9,
        "legacy_partition_counts_on_139_common_internal": dict(shared_old_counts),
        "p5_internal_cells_changed_assignment": 6,
        "split_change_source": "Different explicit split manifests; no split seed was used to regenerate these assignments.",
    }
    (OUT / "population_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return rows, summary


def _feature_distribution() -> dict[str, Any]:
    audit_rows = read_csv(P5 / "bafuse_p5_feature_audit.csv")
    feature_rows = read_csv(LEGACY / "physics_features.csv")
    global _legacy_feature_by_key
    _legacy_feature_by_key = {key(row): row for row in feature_rows}
    p5_by_key = {key(row): row for row in audit_rows}
    if len(_legacy_feature_by_key) != len(feature_rows):
        raise ValueError("Legacy physics feature keys are not unique.")
    if not set(p5_by_key).issubset(_legacy_feature_by_key):
        missing = set(p5_by_key) - set(_legacy_feature_by_key)
        raise ValueError(f"Legacy physics features do not cover P5 keys: {sorted(missing)[:3]}")

    split_rows = read_csv(P5 / "bafuse_p5_split_manifest.csv")
    p5_train_keys = [key(row) for row in split_rows if row["partition"] == "train"]
    feature_info: list[tuple[str, str, dict[tuple[str, str, str], float]]] = []
    for index, name in enumerate(P5_FEATURE_NAMES):
        feature_info.append(
            (
                "P5",
                name,
                {row_key: float(row[P5_RAW_COLUMNS[index]]) for row_key, row in p5_by_key.items()},
            )
        )
    for name in LEGACY_FEATURE_NAMES:
        feature_info.append(
            (
                "legacy",
                name,
                {row_key: float(row[LEGACY_COLUMN_MAP[name]]) for row_key, row in _legacy_feature_by_key.items()},
            )
        )
    result_rows: list[dict[str, Any]] = []
    scaler_info: dict[str, Any] = {}
    for family, name, values_by_key in feature_info:
        train_values = np.asarray([values_by_key[row_key] for row_key in p5_train_keys], dtype=np.float64)
        if not np.isfinite(train_values).all():
            raise ValueError(f"Nonfinite Train values in {family}:{name}")
        if family == "P5":
            pre = _read_json(P5 / "bafuse_p5_preprocessing_config.json")
            column_index = P5_FEATURE_NAMES.index(name)
            mean = float(pre["mean"][column_index])
            scale = float(pre["scale"][column_index])
        else:
            mean = float(np.mean(train_values))
            scale = float(np.std(train_values))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"Invalid Train-only scaler for {family}:{name}")
        train_min = float(np.min(train_values))
        train_max = float(np.max(train_values))
        scaler_info[f"{family}:{name}"] = {
            "fit_partition": "P5_train",
            "fit_rows": 102,
            "mean": mean,
            "scale_population_std": scale,
            "train_min": train_min,
            "train_max": train_max,
        }
        groups: list[tuple[str, str, list[tuple[str, str, str]]]] = []
        for partition in ("train", "validation", "external_test"):
            selected = [
                key(row)
                for row in split_rows
                if row["partition"] == partition and key(row) in values_by_key
            ]
            groups.append(("partition", partition, selected))
        for part in ("1", "2", "3"):
            selected = [row_key for row_key in p5_by_key if row_key[0] == part]
            groups.append(("dataset_part", part, selected))
        for group_kind, group_name, selected_keys in groups:
            values = np.asarray([values_by_key[row_key] for row_key in selected_keys], dtype=np.float64)
            z_values = (values - mean) / scale
            stats = _summary(values)
            result_rows.append(
                {
                    "feature_family": family,
                    "feature": name,
                    "group_kind": group_kind,
                    "group": group_name,
                    "n": len(values),
                    **stats,
                    "train_fit_mean": mean,
                    "train_fit_scale_population_std": scale,
                    "train_raw_min": train_min,
                    "train_raw_max": train_max,
                    "count_outside_train_raw_range": int(np.sum((values < train_min) | (values > train_max))),
                    "z_mean_under_train_scaler": float(np.mean(z_values)),
                    "z_std_under_train_scaler": float(np.std(z_values)),
                    "z_min_under_train_scaler": float(np.min(z_values)),
                    "z_max_under_train_scaler": float(np.max(z_values)),
                }
            )
    write_csv(OUT / "physics_feature_distribution_audit.csv", result_rows)
    (OUT / "physics_scaler_audit.json").write_text(
        json.dumps(scaler_info, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    figure_dir = OUT / "plots"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for name in P5_FEATURE_NAMES:
        index = P5_FEATURE_NAMES.index(name)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4))
        for axis, family, source_map, columns, title in (
            (
                axes[0],
                "P5",
                p5_by_key,
                P5_RAW_COLUMNS,
                "P5 electrical features",
            ),
            (
                axes[1],
                "legacy",
                _legacy_feature_by_key,
                LEGACY_FEATURE_NAMES,
                "Legacy physics features",
            ),
        ):
            vals = []
            labels = []
            for part in ("1", "2", "3"):
                selected = [
                    float(row[columns[index]])
                    for row_key, row in source_map.items()
                    if row_key[0] == part
                ]
                if selected:
                    vals.append(selected)
                    labels.append(f"Part {part}")
            axis.boxplot(vals, tick_labels=labels, showfliers=True)
            axis.set_title(title)
            axis.set_ylabel(name if family == "P5" else LEGACY_FEATURE_NAMES[index])
        figure.suptitle(f"{name}: descriptive raw feature distributions by Part")
        figure.tight_layout()
        figure.savefig(figure_dir / f"part_feature_{name}.png", dpi=150)
        plt.close(figure)
    return scaler_info


def _write_formula_audit(population: dict[str, Any], manifest: dict[str, Any]) -> None:
    formulas = _formula_rows()
    write_csv(OUT / "feature_formula_audit.csv", formulas)
    p5_features = read_csv(P5 / "bafuse_p5_feature_audit.csv")
    step4_delta = max(float(row["old_step4_curve_max_abs_delta_v"]) for row in p5_features)
    step10_delta = max(float(row["old_step10_curve_max_abs_delta_v"]) for row in p5_features)
    status_counts = Counter(row["step4_source_hash_status"] for row in p5_features)
    lines = [
        "# Formula and population audit",
        "",
        f"- Frozen P5 manifest SHA-256 verified: `{sha256(P5_MANIFEST)}`.",
        f"- Dataset: 203 accepted observations; 102 Train / 37 Validation / 64 External Test.",
        f"- Historical legacy comparison population: {population['legacy_observations']} rows, including {population['legacy_primary_saved_model_rows']} primary saved-model rows and {population['legacy_supplemental_g10_saved_checkpoint_rows']} supplemental G10-Cell2 saved-checkpoint rows; partition counts `{population['legacy_partition_counts']}`.",
        f"- Common keys: {population['common_keys']}; legacy-only: {population['legacy_only']}; P5-only: {population['p5_only']}.",
        f"- Legacy-only keys comprise {population['legacy_only_internal']} internal and {population['legacy_only_external']} Part 3 records.",
        "",
        "## Formula comparison",
        "",
        "- Both pipelines resample Step 4 discharge voltage and same-RPT Step 10 charge voltage by linear interpolation on normalized source-row index, retaining native direction. The coordinate is row progress, not capacity or time.",
        f"- On the 203 common observations, recorded maximum legacy-versus-P5 resampled curve deltas are {step4_delta:.9g} V (Step 4) and {step10_delta:.9g} V (Step 10); P5 used the existing source-defined 1e-6 V audit tolerance.",
        "- Legacy model physics: native Step 4 temperature mean and max-minus-first; OLS voltage slope over 20-80%; 1.4826-scaled MAD of centered voltage derivatives over 10-90%; mean absolute Step 4 current. Step 4 duration is explicitly excluded.",
        "- P5 physics: mean absolute Step 4 current; 1.4826-scaled raw-current MAD; 20-80% resampled-voltage slope; 10-90% derivative MAD; and 10-30% minus 70-90% voltage segment contrast. No plateau fraction is used.",
        "- These physics inputs are engineered summaries of same-RPT electrical measurements, not independent sensors. Legacy temperature channel `temp1` has undocumented calibration/location and native units; it is not assumed to be Celsius.",
        "- Historical legacy preprocessing standardized each voltage point and target using Train-only fits; P5 leaves curves in volts and target in percent, while standardizing physics on Train. New controlled runs use common raw-voltage and raw-percent settings, with Train-only physics scaling, to avoid confounding architecture/features with historical input/target scaling.",
        "- Step 4 `Amphr`, integrated current-time capacity, duration, capacity/BoL-derived quantities, SoH-derived inputs, and identifiers are not predictive features. The 128-point curve progression itself is independent of capacity and time.",
        "",
        "## Formula-level safety and protocol checks",
        "",
        "- P5 row audit marks same-RPT step order true and target/BoL, future-RPT, and predictive-temperature flags false for all internal rows.",
        "- Step 4 is negative-current discharge with voltage decreasing overall; Step 10 is positive-current charge with voltage increasing overall. The raw blocks are same-RPT and same source file, per the frozen row audit.",
        f"- Step 4 source-hash status counts in the P5 row audit: `{dict(status_counts)}`. `CURRENT_SNAPSHOT_ONLY` is not a permanent source archive; block and dataset hashes preserve audit-time traceability.",
        "- Repeated timestamp rows are preserved by P5 curve construction; duplicate removal is sensitivity-only. Neither formula uses timestamps, so row density/repetition can still affect sample weighting and interpolation.",
        "",
        "## Population discrepancies",
        "",
        "- Six internal cells changed assignment between explicit split manifests: the old manifest says `preserved_existing_train_validation_or_part3_external_assignment`; P5 says `user_explicit_assignment_2026-11-13`. This was not a different random seed.",
        "- Among the 139 shared Part 1-2 keys, legacy assignment counts are 99 Train / 40 Validation and P5 is 102 / 37. The validation physical-cell sets are disjoint, so validation score differences conflate training composition, validation difficulty, and checkpoint selection.",
        "- Eight legacy-only internal observations have ambiguous Step 4 segment cardinality in the Guide observation audit (multiple candidate segments where one was expected, with no ordinal assignment). This is an upstream label/source-boundary eligibility issue, not a P5 feature-modality failure. Quarantine is scientifically justified pending unique Step 4 resolution; the existing legacy extraction does not establish that its selected segment is the correct RPT discharge.",
        "- Nine historical G10-Cell2 External rows lack candidate RPT rows in the M1 Guide-mapped manifest. The prior split marks the cell External by a user-confirmed identity; the later frozen P5 candidate audit still has no Guide-backed rows. Their exclusion is justified until the source identity and RPT crosswalk are resolved; no feature/label is fabricated.",
        "- No omitted row is restored and no label, split, frozen NPZ, or historical checkpoint is changed.",
        "",
        "## Domain-shift and feature-population cautions",
        "",
        "- Part 1-2 used Maccor 4200 and Step 4 CC-CV discharge; Part 3 used Maccor 4000 and Step 4 CC discharge. Part 3 feature statistics are descriptive only and never inform fitting or selection.",
        "- P5 mean absolute current is highly concentrated: the Train-only scale is recorded in `physics_scaler_audit.json`; no arbitrary near-constant cutoff is imposed. Its standardized validation/external values and out-of-Train-range counts are reported for sensitivity.",
        "- The temperature features have unresolved native-unit/calibration semantics and missingness in the wider dataset; they cannot be considered universally comparable across Parts. P5 avoids temperature and does not impute it.",
        "",
        f"- The P5 frozen manifest covers and hash-verifies inputs/outputs/code; manifest SHA-256: `{sha256(P5_MANIFEST)}`.",
        "",
    ]
    (OUT / "formula_population_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_source_hashes() -> None:
    paths = [
        P5_MANIFEST,
        P5 / "bafuse_soh_labels_frozen.csv",
        P5 / "bafuse_p5_split_manifest.csv",
        P5 / "bafuse_p5_feature_audit.csv",
        P5 / "bafuse_p5_feature_schema.json",
        P5 / "bafuse_p5_preprocessing_config.json",
        P5 / "bafuse_train_p5.npz",
        P5 / "bafuse_validation_p5.npz",
        P5 / "bafuse_external_p5.npz",
        P5 / "p5_run_config.json",
        P5_RUN / "benchmark_config.json",
        P5_RUN / "benchmark_run_metrics.csv",
        P5_RUN / "train_validation_predictions.csv",
        LEGACY / "fusion_predictions.csv",
        LEGACY / "physics_features.csv",
        LEGACY / "fusion_model_config.json",
        LEGACY / "fusion_training_history.csv",
        FORENSIC_MANIFEST,
        FORENSIC_ALIGNMENT,
        LEGACY / "evaluation_audit_v1" / "supplemental_g10_cell2_predictions.csv",
        LEGACY_SPLIT,
        P5_CELL_SPLIT,
        OBSERVATIONS,
        RAW_EXCLUSIONS,
        ROOT / "src" / "bafuse" / "hybrid" / "soh_fusion.py",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark_models.py",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_benchmark.py",
        ROOT / "src" / "bafuse" / "hybrid" / "p5_electrical_features.py",
    ]
    rows = [
        {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}
        for path in paths
    ]
    write_csv(OUT / "source_sha256_manifest.csv", rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "data").mkdir(exist_ok=True)
    p5_manifest = _verify_p5_manifest()
    _verify_prior_forensic_alignment()

    global _legacy_feature_by_key
    _legacy_feature_by_key = {
        key(row): row for row in read_csv(LEGACY / "physics_features.csv")
    }
    population_rows, population_summary = _population_audit()
    internal_rows, staged_summary = _verify_and_stage_internal()
    scaler_info = _feature_distribution()
    _write_protocol()
    _write_formula_audit(population_summary, p5_manifest)
    _write_source_hashes()
    print(
        json.dumps(
            {
                "population_rows": len(population_rows),
                "internal_rows_staged": len(internal_rows),
                "feature_scalers": len(scaler_info),
                "staged_data": staged_summary["input_tensors"],
                "p5_manifest_sha256": sha256(P5_MANIFEST),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
