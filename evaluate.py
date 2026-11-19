"""Compute read-only Final2 results on selected frozen splits."""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

from benchmark import MODELS, STUDY, historical_catalog, read_json, verify

ROOT = Path(__file__).resolve().parent


def research_results(splits):
    sys.path.insert(0, str(ROOT / "src"))
    import numpy as np
    import torch
    from app.model_runtime import Final2ModelRuntime
    from bafuse.features.physics_coverage import FEATURES, key, read_csv
    from bafuse.hybrid.final2_dataset import Final2Scalers
    from bafuse.hybrid.final2_experiment import _load_partition, _metric_values, _predict_percent, _tensor_inputs
    from bafuse.hybrid.final2_models import build_final2_model
    from bafuse.hybrid.p5_normalization import PopulationZScore

    freeze = read_json(STUDY / "frozen_dataset_manifest_v2.json")
    frozen = STUDY / "frozen_dataset"
    for name, digest in freeze["dataset_artifact_sha256"].items():
        verify(frozen / name, digest)
    historical = {record["model"]: row for record, row in historical_catalog()}
    candidate = STUDY / "updated_physics_candidate"
    summary = read_json(candidate / "metrics.json")
    config = read_json(candidate / "config.json")
    selected = summary["selected"]
    coverage = ROOT / "data/processed/physics/coverage.csv"
    verify(coverage, config["coverage_sha256"])
    verify(candidate / "best.pt", selected["checkpoint_sha256"])
    verify(candidate / "normalization_parameters.json", summary["normalization_sha256"])
    verify(candidate / "development_physics.npz", config["development_physics_sha256"])
    metadata = read_json(candidate / "normalization_parameters.json")
    if (not summary["selection_frozen"] or not summary["external_evaluated"]
            or config["dataset_identifier"] != freeze["dataset_identifier"]
            or metadata["fit_partition"] != "train"
            or tuple(config["feature_names"]) != FEATURES
            or tuple(metadata["feature_names"]) != FEATURES):
        raise ValueError("Updated CNN-3M schema or frozen selection mismatch")
    rows = read_csv(coverage)
    by_key = {key(row): row for row in rows}
    if len(rows) != 213 or len(by_key) != 213 or any(row["status"] != "verified" for row in rows):
        raise ValueError("Updated Physics coverage is incomplete")
    runtime = Final2ModelRuntime()
    device = runtime.device
    amp = device.type == "cuda"
    names = ("discharge", "charge", "physics", "target")
    updated_scalers = Final2Scalers(*(PopulationZScore.from_dict(metadata[name]) for name in names))
    prior_scalers = Final2Scalers(*(runtime.scalers[name] for name in names))
    if tuple(len(getattr(updated_scalers, name).mean) for name in names) != (128, 128, 5, 1):
        raise ValueError("Updated scaler widths do not match the model")
    payload = torch.load(candidate / "best.pt", map_location="cpu", weights_only=True)
    verify(candidate / "config.json", payload["training_config_sha256"])
    if (payload["model"] != "CNN-3M" or payload["seed"] != selected["seed"]
            or payload["best_epoch"] != selected["best_epoch"]):
        raise ValueError("Updated checkpoint metadata disagrees with selection")
    updated = build_final2_model("CNN-3M").to(device)
    updated.load_state_dict(payload["model_state_dict"], strict=True)
    models = {"CNN-3M": updated}
    for name in MODELS[1:]:
        loaded = runtime._load_model(name)
        if loaded is None:
            raise ValueError(f"Selected {name} failed to load: {runtime.load_errors.get(name)}")
        models[name] = loaded[0]

    counts = {"train": (99, 9), "validation": (40, 3), "external": (74, 12)}
    stored_external = {}
    if "external" in splits:
        external_dir = STUDY / "external_evaluation"
        verify(external_dir / "external_evaluation_manifest.json",
               runtime.validation_selection["external_evaluation_manifest_sha256"])
        stored_external = read_json(external_dir / "external_metrics.json")["metrics_by_model"]
    result = {}
    for split in splits:
        name = "external_test" if split == "external" else split
        part = _load_partition(frozen / (name + ".npz"))
        if (part.sample_count, part.cell_count) != counts[split]:
            raise ValueError(f"Frozen counts differ for {split}")
        keys = list(zip(part.dataset_part, part.physical_cell_id, part.rpt_index))
        if set(keys) != {k for k, row in by_key.items() if row["final2_partition"] == name}:
            raise ValueError(f"Updated feature identity mismatch for {split}")
        values = np.asarray([[float(by_key[k][f]) for f in FEATURES] for k in keys], dtype=np.float32)
        if not np.array_equal(values[:, 2:], part.arrays.X_physics_raw[:, 2:]):
            raise ValueError(f"Updated descriptors differ from frozen Step 4 for {split}")
        if split != "external":
            with np.load(candidate / "development_physics.npz", allow_pickle=False) as trained:
                if not np.array_equal(values, trained[split]) or tuple(trained[split + "_keys"]) != part.keys:
                    raise ValueError(f"Candidate development inputs differ for {split}")
        new_arrays = replace(part.arrays, X_physics_raw=values)
        new_arrays.validate()
        updated_inputs = _tensor_inputs(new_arrays, updated_scalers)
        historical_inputs = _tensor_inputs(part.arrays, prior_scalers)
        result[split] = {}
        for model_name in MODELS:
            inputs, scaler = ((updated_inputs, updated_scalers.target) if model_name == "CNN-3M"
                              else (historical_inputs, prior_scalers.target))
            pred = _predict_percent(models[model_name], model_name, inputs, scaler, device, amp)
            result[split][model_name] = _metric_values(part.arrays.y_soh, pred, part.cell_count)
        expected = summary["external"] if split == "external" else selected[split]
        if amp and any(abs(result[split]["CNN-3M"][m] - expected[m]) > 1e-4 for m in ("MAE", "RMSE", "R2")):
            raise ValueError(f"Updated CNN-3M does not reproduce saved {split} metrics")
        if amp:
            for model_name in MODELS[1:]:
                archived = (stored_external[model_name] if split == "external" else
                            {m: float(historical[model_name][f"{split}_{m}"]) for m in ("MAE", "RMSE", "R2")})
                result[split][model_name]["archived_delta"] = {
                    m: result[split][model_name][m] - archived[m] for m in ("MAE", "RMSE", "R2")
                }
    return result, str(device)


def print_results(results, device):
    print(f"Device: {device}. CNN-3M: updated-Physics candidate (not promoted).")
    for split, rows in results.items():
        print(f"\n{split.upper()} ? overall by observation")
        print(f"{'Model':<16}{'N':>5}{'MAE':>11}{'RMSE':>11}{'R?':>11}")
        for name in MODELS:
            row = rows[name]
            print(f"{name:<16}{row['N']:>5}{row['MAE']:>11.4f}{row['RMSE']:>11.4f}{row['R2']:>11.4f}")
    print("\nMAE / RMSE: SoH percentage points. External labels: provisional reconstructed references.")
    deltas = [row["archived_delta"] for rows in results.values() for row in rows.values()
              if "archived_delta" in row]
    if deltas:
        print("Maximum absolute difference from saved 2M metrics: "
              + ", ".join(f"{m} {max(abs(row[m]) for row in deltas):.6f}" for m in ("MAE", "RMSE", "R2")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "external"))
    args = parser.parse_args()
    print_results(*research_results((args.split,) if args.split else ("train", "validation")))


if __name__ == "__main__":
    main()
