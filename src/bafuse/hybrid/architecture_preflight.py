from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from bafuse.hybrid.architecture_models import ARCHITECTURE_NAMES, architecture_parameter_counts
from bafuse.hybrid.training_authorization import TrainingAuthorizationError, build_training_scope


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PREPARATION_DIR = (
    PROJECT_ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "architecture_matrix_v1"
)
CONFIG_PATH = PREPARATION_DIR / "architecture_config.json"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _resolve(path: str) -> Path:
    return PROJECT_ROOT / Path(path.replace("/", "\\"))


def _has_unset(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, dict):
        return any(_has_unset(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_has_unset(item) for item in value)
    return False


def run_preflight(
    config_path: Path = CONFIG_PATH,
    output_dir: Path = PREPARATION_DIR,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("training_authorized") is not False:
        raise ValueError("Preflight requires training_authorized to remain false.")
    if config.get("training_started") is not False:
        raise ValueError("Preflight requires training_started to remain false.")
    if config.get("architectures") != list(ARCHITECTURE_NAMES):
        raise ValueError("Architecture config does not match the implemented four-model matrix.")

    split_path = _resolve(config["split_manifest"])
    split_rows = _read_csv(split_path)
    split_cells: dict[str, str] = {}
    for row in split_rows:
        cell = row["physical_cell_id"]
        if cell in split_cells:
            raise ValueError(f"Duplicate physical cell in split manifest: {cell}")
        split_cells[cell] = row["partition"]
    split_counts = Counter(split_cells.values())
    expected_cells = {"train": 9, "validation": 3, "external_test": 12}
    if dict(split_counts) != expected_cells:
        raise ValueError(f"Expected the exact 9/3/0/12 split assignments, got {dict(split_counts)}.")

    label_path = _resolve(config["label_manifest"])
    feature_path = _resolve(config["feature_matrix"])
    references: list[dict[str, str]] = []
    feature_rows: list[dict[str, str]] = []
    data_ready = label_path.is_file() and feature_path.is_file()
    if data_ready:
        references = _read_csv(label_path)
        feature_rows = _read_csv(feature_path)
        feature_keys = {
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            for row in feature_rows
        }
        accepted = [
            row for row in references
            if row["label_disposition"] == "ACCEPTED"
            and row["final_eligibility_status"] == "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
        ]
        for row in accepted:
            if split_cells.get(row["physical_cell_id"]) != row["partition"]:
                raise ValueError(f"Split mismatch for accepted label {row['physical_cell_id']}.")
            key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            if key not in feature_keys:
                raise ValueError(f"Accepted label has no raw feature row: {key}")
        internal = [row for row in accepted if row["partition"] in {"train", "validation"}]
        train = [row for row in internal if row["partition"] == "train"]
        validation = [row for row in internal if row["partition"] == "validation"]
        distribution: dict[str, dict[str, float | int]] = {}
        for name, group in (("train", train), ("validation", validation)):
            targets = [float(row["reconstructed_soh_percent"]) for row in group]
            distribution[name] = (
                {
                    "n": len(targets),
                    "min_percent": min(targets),
                    "max_percent": max(targets),
                    "mean_percent": sum(targets) / len(targets),
                }
                if targets
                else {"n": 0}
            )
        eligible_by_split = dict(Counter(row["partition"] for row in accepted))
        cell_counts = {
            name: len({row["physical_cell_id"] for row in internal if row["partition"] == name})
            for name in ("train", "validation")
        }
        feasibility = {
            "accepted_complete_records_by_split": eligible_by_split,
            "internal_train_records": len(train),
            "internal_validation_records": len(validation),
            "internal_train_physical_cells": cell_counts["train"],
            "internal_validation_physical_cells": cell_counts["validation"],
            "accepted_target_distribution_percent": distribution,
            "raw_feature_rows": len(feature_rows),
            "assessment": (
                "review_required_user_approval"
                if train and validation
                else "blocked_no_eligible_internal_cohort"
            ),
            "complexity_note": (
                "Compare measured parameter counts with the actual record and cell counts; "
                "the small number of independent training cells warrants review before fitting."
            ),
        }
    else:
        feasibility = {
            "assessment": "blocked_reconstruction_outputs_missing",
            "missing_inputs": [
                str(path) for path in (label_path, feature_path) if not path.is_file()
            ],
        }

    counts = architecture_parameter_counts()
    output = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "training_authorized": False,
        "training_started": False,
        "external_evaluation_authorized": False,
        "architectures": list(ARCHITECTURE_NAMES),
        "trainable_parameter_counts": counts,
        "split_cell_counts": dict(split_counts),
        "post_m1_feasibility_gate": feasibility,
        "authorization_scope_ready": False,
        "authorization_scope_blockers": [],
    }

    budget = config["training_proposal_not_approved"]
    if _has_unset(budget):
        output["authorization_scope_blockers"].append(
            "Device and maximum total device-hours are not specified and the training proposal is not approved."
        )
    if not data_ready:
        output["authorization_scope_blockers"].append(
            "Versioned reconstructed labels and raw curve features must exist before scoping authorization."
        )
    if not output["authorization_scope_blockers"]:
        scope = build_training_scope(
            label_manifest=label_path,
            feature_matrix=feature_path,
            split_manifest=split_path,
            experiment_config=config_path,
            compute_budget=budget,
            output_directory=_resolve(config["output_directory"]),
            architectures=config["architectures"],
        )
        output["authorization_scope"] = scope
        output["authorization_scope_ready"] = True
        (output_dir / "training_scope.json").write_text(
            json.dumps(scope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        (output_dir / "training_scope.json").write_text(
            json.dumps(
                {
                    "status": "blocked",
                    "blockers": output["authorization_scope_blockers"],
                    "training_authorized": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "architecture_preflight.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "architecture_parameter_counts.json").write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output


def main() -> int:
    result = run_preflight()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
