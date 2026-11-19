"""Train one staged Final2 model from the frozen cohort."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark import MODELS

import numpy as np
import torch

from bafuse.features.physics_coverage import FEATURES, UNITS, key, read_csv
from bafuse.hybrid.final2_dataset import fit_final2_scalers_for_frozen_cohort
from bafuse.hybrid.final2_experiment import (
    DATASET_IDENTIFIER, _load_partition, _metric_values, _predict_percent, _read_json, _sha256, _tensor_inputs,
    _train_one_run, _write_json,
)
from bafuse.hybrid.final2_models import build_final2_model
from bafuse.hybrid.final2_runtime import resolve_device


def train_candidate(root: Path, coverage: Path, *, model_name: str = "CNN-3M", output: Path | None = None) -> dict:
    study = root / "experiments/final2/training_6_models"
    frozen = study / "frozen_dataset"
    freeze = _read_json(study / "frozen_dataset_manifest_v2.json")
    if (freeze["dataset_identifier"] != DATASET_IDENTIFIER
            or not freeze["accepted_cohort_frozen"] or not freeze["training_authorized"]):
        raise ValueError("Unexpected frozen dataset authority")
    for filename, digest in freeze["dataset_artifact_sha256"].items():
        if _sha256(frozen / filename) != digest:
            raise ValueError(f"Frozen hash mismatch for {filename}")
    if model_name not in MODELS:
        raise ValueError(f"Unsupported official model: {model_name}")
    use_physics = model_name == "CNN-3M"
    selection_metric = "MAE" if use_physics else "RMSE"
    if use_physics:
        rows = read_csv(coverage)
        by_key = {key(row): row for row in rows}
        accepted = {
            key(row): row for row in read_csv(frozen / "observation_manifest.csv")
            if row["status"] == "ACCEPTED"
        }
        if len(rows) != 213 or len(by_key) != 213 or set(by_key) != set(accepted):
            raise ValueError("Coverage keys do not match the frozen cohort")
        for identifier, row in by_key.items():
            source = accepted[identifier]
            if (row["status"] != "verified" or row["missing_reason"]
                    or any(row[f] != source[f] for f in (
                        "final2_partition", "source_file", "step10_segment_reference",
                        "step10_source_sha256_snapshot"))
                    or any(row[name + "_unit"] != unit for name, unit in zip(FEATURES, UNITS))):
                raise ValueError(f"Unverified feature or provenance for {identifier}")
            if float(row["nominal_cycle_age"]) != 48 * identifier[2]:
                raise ValueError(f"Nominal age disagrees with the frozen ordinal for {identifier}")
        for cell in {identifier[1] for identifier in by_key}:
            ordinals = [rpt for _, cell_id, rpt in sorted(by_key) if cell_id == cell]
            if any(right <= left for left, right in zip(ordinals, ordinals[1:])):
                raise ValueError(f"Frozen RPT ordinal does not continue across parts for {cell}")

    def development_partition(name):
        partition = _load_partition(frozen / f"{name}.npz")
        if not use_physics:
            return partition
        keys = zip(map(int, partition.dataset_part), partition.physical_cell_id, map(int, partition.rpt_index))
        values = np.asarray([[float(by_key[k][f]) for f in FEATURES] for k in keys], dtype=np.float32)
        if not np.array_equal(values[:, 2:], partition.arrays.X_physics_raw[:, 2:]):
            raise ValueError(f"Frozen Step 4 descriptors differ for {name}")
        arrays = replace(partition.arrays, X_physics_raw=values)
        arrays.validate()
        return replace(partition, arrays=arrays)

    train = development_partition("train")
    validation = development_partition("validation")
    if (train.sample_count, train.cell_count, validation.sample_count, validation.cell_count) != (99, 9, 40, 3):
        raise ValueError("Unexpected development partition counts")
    if set(train.physical_cell_id) & set(validation.physical_cell_id):
        raise ValueError("Development cells overlap")
    scalers = fit_final2_scalers_for_frozen_cohort(
        train.arrays, accepted_cohort_frozen=freeze["accepted_cohort_frozen"],
        training_authorized=freeze["training_authorized"],
    )
    source_config = study / "training_config_v2.json"
    if _sha256(source_config) != freeze["training_config_sha256"]:
        raise ValueError("Training configuration does not match the freeze manifest")
    training = _read_json(source_config)["training"]
    device = resolve_device()
    if output is None:
        output = study / ("updated_physics_candidate" if use_physics else f"{model_name.lower()}_candidate")
    output.mkdir(exist_ok=False)
    feature_metadata = {}
    if use_physics:
        features_path = output / "development_physics.npz"
        temporary = features_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary,
                            train=train.arrays.X_physics_raw, validation=validation.arrays.X_physics_raw,
                            train_keys=train.keys, validation_keys=validation.keys, feature_names=FEATURES)
        temporary.replace(features_path)
        feature_metadata = {
            "feature_names": list(FEATURES), "feature_units": list(UNITS),
            "coverage_sha256": _sha256(coverage), "development_physics_sha256": _sha256(features_path),
        }
    config_path = output / "config.json"
    _write_json(config_path, {
        "architecture": model_name, "dataset_identifier": DATASET_IDENTIFIER,
        "modalities": ["Discharge", "Charge"] + (["Physics"] if use_physics else []),
        **feature_metadata,
        "training": training, "device": str(device),
        "selection_metric": f"Validation overall {selection_metric}",
    })
    scaler_data = scalers.to_dict(dataset_identifier=DATASET_IDENTIFIER)
    if use_physics:
        scaler_data.update(feature_names=list(FEATURES), variant="Updated Physics, Train-only population z-score",
                       physics_normalization="Updated Physics population z-score, Train only")
    scaler_path = output / "normalization_parameters.json"
    _write_json(scaler_path, scaler_data)
    inputs = _tensor_inputs(train.arrays, scalers)
    val_inputs = _tensor_inputs(validation.arrays, scalers)
    target = torch.as_tensor(scalers.target.transform(train.arrays.y_soh), dtype=torch.float32).reshape(-1)
    metrics = []
    checkpoint = output / "best.pt"
    with TemporaryDirectory(prefix="seed-staging-", dir=output) as temporary_dir:
        staging = Path(temporary_dir)
        for seed in training["seeds"]:
            run, _, _, _ = _train_one_run(
                name=model_name, seed=seed, train=train, validation=validation,
                train_inputs=inputs, validation_inputs=val_inputs, train_targets_normalized=target,
                target_scaler=scalers.target, output_dir=staging,
                dataset_manifest_sha256=freeze["dataset_artifact_sha256"]["dataset_sha256_manifest.json"],
                config_sha256=_sha256(config_path), device=device, amp_enabled=device.type == "cuda",
                training=training, selection_metric=selection_metric, print_progress=True,
            )
            metrics.append(run)
        selected_run = min(metrics, key=lambda run: (run["validation"][selection_metric], run["seed"]))
        (staging / selected_run["checkpoint_path"]).rename(checkpoint)
    selected = {key: value for key, value in selected_run.items() if not key.startswith("last_checkpoint_")}
    selected["checkpoint_path"] = checkpoint.name
    result = {
        f"seed_validation_{selection_metric.lower()}_pp": {
            run["seed"]: run["validation"][selection_metric] for run in metrics
        },
        "selected": selected, "external_evaluated": False,
        "selection_frozen": True, "selection_metric": f"Validation overall {selection_metric}",
        "device": f"CUDA / {torch.cuda.get_device_name(device)}" if device.type == "cuda" else "CPU",
        "normalization_sha256": _sha256(scaler_path),
        "promoted": False, "promotion_status": "retained as candidate; approved checkpoint unchanged",
    }
    metrics_path = output / "metrics.json"
    _write_json(metrics_path, result)

    if _sha256(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint hash changed before External evaluation")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (payload["model"] != model_name or payload["seed"] != selected["seed"]
            or payload["best_epoch"] != selected["best_epoch"]
            or payload["training_config_sha256"] != _sha256(config_path)
            or _read_json(scaler_path) != scaler_data):
        raise ValueError("Selected checkpoint, scaler, or configuration mismatch")
    model = build_final2_model(model_name).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    external = development_partition("external_test")
    if external.sample_count != 74 or external.cell_count != 12:
        raise ValueError("Unexpected External partition counts")
    if set(external.physical_cell_id) & (set(train.physical_cell_id) | set(validation.physical_cell_id)):
        raise ValueError("External cells overlap development cells")
    predicted = _predict_percent(
        model, model_name, _tensor_inputs(external.arrays, scalers), scalers.target,
        device, device.type == "cuda",
    )
    result["external"] = _metric_values(external.arrays.y_soh.reshape(-1), predicted, external.cell_count)
    result["external_evaluated"] = True
    result["checkpoint"] = checkpoint.relative_to(root).as_posix()
    _write_json(metrics_path, result)
    return result
