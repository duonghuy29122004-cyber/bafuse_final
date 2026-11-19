"""Report the five official architectures and their frozen study evidence."""
import csv
import hashlib
import json
from pathlib import Path

STUDY = Path(__file__).resolve().parent / "experiments/final2/training_6_models"
RUN = STUDY / "run_001"
MODELS = ("CNN-3M", "TCN-2M", "ResNet-2M", "DenseNet-2M", "ConvNeXt-2M")


def read_json(path):
    return json.loads(path.read_bytes())


def verify(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch for {path}")


def historical_catalog():
    freeze = read_json(STUDY / "frozen_dataset_manifest_v2.json")
    selection = read_json(RUN / "validation_selection.json")
    if not freeze["accepted_cohort_frozen"] or not selection["selection_frozen"]:
        raise ValueError("The cohort and validation selection must be frozen.")
    if freeze["dataset_identifier"] != selection["dataset_identifier"]:
        raise ValueError("Selection and dataset identifiers differ.")
    verify(RUN / "normalization_parameters.json", selection["normalization_parameters_sha256"])
    verify(STUDY / "training_config_v2.json", selection["training_config_sha256"])
    manifest = read_json(RUN / "checkpoint_manifest.json")
    with (RUN / "selected_validation_metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        metrics = {row["model"]: row for row in csv.DictReader(stream)}
    for name in MODELS:
        record = manifest[name]
        row = metrics[name]
        verify(RUN / record["best_path"], record["best_sha256"])
        if row["checkpoint_sha256"] != record["best_sha256"] or int(row["seed"]) != record["seed"]:
            raise ValueError(f"Stored metrics do not match the selected checkpoint for {name}.")
        yield {
            "model": name,
            "modalities": ["Discharge", "Charge"] + (["Physics"] if name == "CNN-3M" else []),
            "physics_revision": "historical current-based P5" if name == "CNN-3M" else None,
            "dataset": freeze["dataset_identifier"],
            "checkpoint": record["best_path"],
            "sha256": record["best_sha256"],
            "seed": record["seed"],
            "epoch": record["best_epoch"],
            "validation_RMSE_pp": record["validation_rmse_pp"],
        }, row


def catalog():
    """List the five entries in the research comparison with their approval status."""
    candidate = STUDY / "updated_physics_candidate"
    metrics = read_json(candidate / "metrics.json")
    selected = metrics["selected"]
    verify(candidate / "best.pt", selected["checkpoint_sha256"])
    for record, row in historical_catalog():
        if record["model"] == "CNN-3M":
            yield {
                **record,
                "physics_revision": "updated Physics",
                "checkpoint": metrics["checkpoint"],
                "sha256": selected["checkpoint_sha256"],
                "seed": selected["seed"],
                "epoch": selected["best_epoch"],
                "validation_RMSE_pp": selected["validation"]["RMSE"],
                "promoted": metrics["promoted"],
            }, selected
        else:
            yield {**record, "promoted": True}, row


if __name__ == "__main__":
    from evaluate import print_results, research_results

    results, device = research_results(("train", "validation", "external"))
    print("Five-model research comparison")
    print(f"{'Model':<16}{'Train MAE':>11}{'Val MAE':>11}{'Ext MAE':>11}{'Ext RMSE':>11}{'Ext R?':>11}")
    for name in MODELS:
        external = results["external"][name]
        print(f"{name:<16}{results['train'][name]['MAE']:>11.4f}{results['validation'][name]['MAE']:>11.4f}"
              f"{external['MAE']:>11.4f}{external['RMSE']:>11.4f}{external['R2']:>11.4f}")
    print_results(results, device)
