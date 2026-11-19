from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


CONFIRMED_CELLS = (
    "G1-Cell15",
    "G1-Cell20",
    "G1-Cell9",
    "G2-Cell3",
    "G2-Cell4",
    "G2-Cell8",
    "G3-Cell10",
    "G3-Cell11",
    "G3-Cell14",
    "G4-Cell12",
    "G4-Cell18",
    "G4-Cell19",
)
SOH_STATUSES = {"baseline_reference", "calculated_from_measured_capacity"}
PYDMA_ACCEPTED = "diagnostic_only_not_independently_validated"
TARGET_COLUMNS = (
    "lli_percent",
    "lam_anode_percent",
    "lam_cathode_percent",
)
RPT_STAGE_RE = re.compile(r"rpt", re.IGNORECASE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _norm(value: str) -> str:
    return value.replace("\\", "/").strip().casefold()


def _norm_rpt(value: str) -> str:
    compact = re.sub(r"\s+", "", value).casefold()
    match = re.fullmatch(r"rpt(\d+)", compact)
    return f"rpt{int(match.group(1))}" if match else compact


def _identity(row: dict[str, str], direction: bool = True) -> tuple[str, ...]:
    result = (
        row["physical_cell_id"].strip(),
        _norm(row["source_file"]),
        _norm_rpt(row["measurement_id"]),
    )
    return (*result, row["direction"].strip().casefold()) if direction else result


def _fit_identity(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["physical_cell_id"].strip(),
        _norm_rpt(row["measurement_id"]),
        row["direction"].strip().casefold(),
    )


def _is_accepted(row: dict[str, str]) -> bool:
    return bool(
        row.get("fit_success", "").strip().casefold() == "true"
        and row.get("identifiability_status", "").strip() == PYDMA_ACCEPTED
        and all(row.get(column, "").strip() for column in TARGET_COLUMNS)
    )


def _row_rank(row: dict[str, str]) -> tuple[int, float]:
    if _is_accepted(row):
        return (4, 0.0)
    status = row.get("identifiability_status", "").strip()
    if status == "withheld_fit_rejected":
        return (3, 0.0)
    if status == "withheld_baseline_fit_failed":
        return (2, 0.0)
    if status == "not_run":
        return (1, 0.0)
    if status == "reference_baseline":
        return (0, 0.0)
    return (0, 0.0)


def _load_best_degradation_rows(
    labels_dir: Path, supplemental_paths: list[Path]
) -> list[dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for path in (labels_dir / "labels_degradation.csv", *supplemental_paths):
        for row in _read_csv(path):
            if row.get("physical_cell_id", "") not in CONFIRMED_CELLS:
                continue
            direction = row.get("direction", "").strip().casefold()
            if direction not in {"charge", "discharge"}:
                continue
            key = _fit_identity(row)
            current = selected.get(key)
            if current is None or _row_rank(row) > _row_rank(current):
                selected[key] = row
    return sorted(
        selected.values(),
        key=lambda row: (
            row["physical_cell_id"],
            _norm_rpt(row["measurement_id"]),
            row["direction"],
        ),
    )


def _load_usable_soh(labels_dir: Path) -> dict[tuple[str, ...], dict[str, str]]:
    selected: dict[tuple[str, ...], dict[str, str]] = {}
    for row in _read_csv(labels_dir / "labels_soh.csv"):
        if (
            row.get("physical_cell_id", "") in CONFIRMED_CELLS
            and row.get("validation_status", "") in SOH_STATUSES
            and row.get("soh_percent", "").strip()
        ):
            selected[_identity(row, direction=False)] = row
    return selected


def _load_source_records(labels_dir: Path) -> dict[tuple[str, str], dict[str, str]]:
    records: dict[tuple[str, str], dict[str, str]] = {}
    for row in _read_csv(labels_dir / "dataset_manifest.csv"):
        key = (row.get("physical_cell_id", ""), _norm(row.get("relative_path", "")))
        records[key] = row
    return records


def _load_eis_counts(
    labels_dir: Path,
) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in _read_csv(labels_dir / "dataset_manifest.csv"):
        if (
            row.get("modality") == "EIS"
            and row.get("file_type") == "csv"
            and row.get("physical_cell_id") in CONFIRMED_CELLS
        ):
            counts[row["physical_cell_id"]] += 1
    return dict(counts)


def _load_step4_cache_keys(feature_cache_dir: Path) -> set[tuple[str, ...]]:
    provenance = _read_csv(feature_cache_dir / "sample_provenance.csv")
    return {
        (
            row["physical_cell_id"],
            _norm(row["source_file"]),
            _norm_rpt(row["measurement_id"]),
        )
        for row in provenance
    }


def _make_sample_manifest(
    degradation: list[dict[str, str]],
    soh: dict[tuple[str, ...], dict[str, str]],
    source_records: dict[tuple[str, str], dict[str, str]],
    eis_counts: dict[str, int],
    step4_cache_keys: set[tuple[str, ...]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in degradation:
        source = source_records.get(
            (row["physical_cell_id"], _norm(row["source_file"]))
        )
        measurement_summary: dict[str, Any] = {}
        if source and source.get("measurement_summary_json", "").strip():
            try:
                measurement_summary = json.loads(source["measurement_summary_json"])
            except json.JSONDecodeError:
                measurement_summary = {}
        step_counts = measurement_summary.get("step_counts", {})
        soh_row = soh.get(_identity(row, direction=False))
        accepted = _is_accepted(row)
        rows.append(
            {
                "sample_kind": "rpt_direction",
                "physical_cell_id": row["physical_cell_id"],
                "group_id": row["group_id"],
                "measurement_id": row["measurement_id"],
                "direction": row["direction"],
                "source_file": row["source_file"],
                "source_record_status": "mapped" if source else "not_mapped",
                "discharge_step4_available": (
                    "true" if int(step_counts.get("4", 0) or 0) > 0 else "unknown"
                ),
                "charge_step10_available": (
                    "true" if int(step_counts.get("10", 0) or 0) > 0 else "unknown"
                ),
                "eis_status": (
                    "present_cell_level_not_rpt_aligned"
                    if eis_counts.get(row["physical_cell_id"], 0)
                    else "not_observed"
                ),
                "physics_status": "not_ready",
                "step4_feature_cache_status": (
                    "cached_for_same_source_rpt"
                    if _identity(row, direction=False) in step4_cache_keys
                    else "not_cached_not_evaluated"
                ),
                "soh_label_status": (
                    soh_row["validation_status"] if soh_row else "not_usable"
                ),
                "soh_percent": soh_row["soh_percent"] if soh_row else "",
                "fit_status": (
                    "accepted_diagnostic" if accepted else row["identifiability_status"]
                ),
                "training_eligible": "true" if accepted else "false",
                "lli_percent": row["lli_percent"] if accepted else "",
                "lam_anode_percent": row["lam_anode_percent"] if accepted else "",
                "lam_cathode_percent": row["lam_cathode_percent"] if accepted else "",
                "fit_residual_mV": row["fit_residual_mV"],
                "accepted_runs": row["accepted_runs"],
                "total_runs": row["total_runs"],
                "label_source": row["label_source"],
                "source_result_file": row["source_result_file"],
            }
        )
    return rows


def _cell_summary(
    samples: list[dict[str, Any]], soh: dict[tuple[str, ...], dict[str, str]], eis_counts: dict[str, int]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for cell in CONFIRMED_CELLS:
        cell_samples = [row for row in samples if row["physical_cell_id"] == cell]
        accepted = [row for row in cell_samples if row["training_eligible"] == "true"]
        result.append(
            {
                "physical_cell_id": cell,
                "group_id": cell.split("-", 1)[0].removeprefix("G"),
                "usable_soh_rows": sum(
                    row["physical_cell_id"] == cell for row in soh.values()
                ),
                "canonical_rpt_direction_rows": len(cell_samples),
                "accepted_degradation_fit_points": len(accepted),
                "accepted_charge_fit_points": sum(
                    row["direction"] == "charge" for row in accepted
                ),
                "accepted_discharge_fit_points": sum(
                    row["direction"] == "discharge" for row in accepted
                ),
                "eis_csv_count": eis_counts.get(cell, 0),
                "eis_alignment": "not_verified",
                "physics_status": "not_ready",
            }
        )
    return result


def _candidate_splits() -> dict[str, dict[str, list[str]]]:
    return {
        "coverage_balanced": {
            "train": [
                "G1-Cell20",
                "G2-Cell3",
                "G2-Cell4",
                "G2-Cell8",
                "G3-Cell11",
                "G4-Cell12",
                "G4-Cell18",
                "G4-Cell19",
            ],
            "validation": ["G1-Cell15", "G3-Cell10"],
            "test": ["G1-Cell9", "G3-Cell14"],
        },
        "group_spread": {
            "train": [
                "G1-Cell15",
                "G1-Cell20",
                "G2-Cell3",
                "G2-Cell8",
                "G3-Cell11",
                "G3-Cell14",
                "G4-Cell18",
                "G4-Cell19",
            ],
            "validation": ["G1-Cell9", "G2-Cell4"],
            "test": ["G3-Cell10", "G4-Cell12"],
        },
    }


def _validate_candidate(name: str, split: dict[str, list[str]]) -> None:
    all_cells = set(CONFIRMED_CELLS)
    assigned = [cell for cells in split.values() for cell in cells]
    if set(assigned) != all_cells or len(assigned) != len(set(assigned)):
        raise ValueError(f"Invalid {name} split: cells are not a disjoint partition.")
    if {key: len(value) for key, value in split.items()} != {
        "train": 8,
        "validation": 2,
        "test": 2,
    }:
        raise ValueError(f"Invalid {name} split size.")


def build_phase1_manifest(
    labels_dir: Path,
    supplemental_paths: list[Path],
    feature_cache_dir: Path,
    output_dir: Path,
    locked_candidate: str | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    degradation = _load_best_degradation_rows(labels_dir, supplemental_paths)
    soh = _load_usable_soh(labels_dir)
    source_records = _load_source_records(labels_dir)
    eis_counts = _load_eis_counts(labels_dir)
    step4_cache_keys = _load_step4_cache_keys(feature_cache_dir)
    samples = _make_sample_manifest(
        degradation, soh, source_records, eis_counts, step4_cache_keys
    )
    cells = _cell_summary(samples, soh, eis_counts)

    candidates = _candidate_splits()
    if locked_candidate is not None and locked_candidate not in candidates:
        raise ValueError(f"Unknown split candidate: {locked_candidate}")
    candidate_rows: list[dict[str, Any]] = []
    candidate_stats: dict[str, Any] = {}
    for name, split in candidates.items():
        _validate_candidate(name, split)
        for split_name, split_cells in split.items():
            for cell in split_cells:
                candidate_rows.append(
                    {
                        "candidate_id": name,
                        "split": split_name,
                        "physical_cell_id": cell,
                        "status": (
                            "locked"
                            if locked_candidate == name
                            else "proposed_not_locked"
                        ),
                    }
                )
        by_cell = {row["physical_cell_id"]: row for row in cells}
        candidate_stats[name] = {
            "train_cells": split["train"],
            "validation_cells": split["validation"],
            "test_cells": split["test"],
            "train_accepted_fit_points": sum(
                by_cell[cell]["accepted_degradation_fit_points"]
                for cell in split["train"]
            ),
            "validation_accepted_fit_points": sum(
                by_cell[cell]["accepted_degradation_fit_points"]
                for cell in split["validation"]
            ),
            "test_accepted_fit_points": sum(
                by_cell[cell]["accepted_degradation_fit_points"]
                for cell in split["test"]
            ),
            "validation_usable_soh_rows": sum(
                by_cell[cell]["usable_soh_rows"] for cell in split["validation"]
            ),
            "test_usable_soh_rows": sum(
                by_cell[cell]["usable_soh_rows"] for cell in split["test"]
            ),
        }

    if locked_candidate is not None:
        locked_rows = [
            {
                "split_version": "phase1-8-2-2-v1",
                "candidate_id": locked_candidate,
                "split": split_name,
                "physical_cell_id": cell,
                "status": "locked",
                "all_rpts_same_split": "true",
                "selection_basis": "coverage_and_cell_group_audit_not_model_metrics",
            }
            for split_name, cells_for_split in candidates[locked_candidate].items()
            for cell in cells_for_split
        ]
        _write_csv(output_dir / "split_manifest.csv", locked_rows)

    _write_csv(output_dir / "sample_manifest.csv", samples)
    _write_csv(output_dir / "cell_summary.csv", cells)
    _write_csv(output_dir / "split_candidates.csv", candidate_rows)
    summary = {
        "phase": "1",
        "status": (
            "manifest_ready_split_locked"
            if locked_candidate is not None
            else "manifest_ready_split_not_locked"
        ),
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
        "source_labels": [
            str(labels_dir / "labels_degradation.csv"),
            *(str(path) for path in supplemental_paths),
        ],
        "physical_cells": list(CONFIRMED_CELLS),
        "sample_manifest": {
            "rows": len(samples),
            "accepted_training_rows": sum(
                row["training_eligible"] == "true" for row in samples
            ),
            "withheld_or_unrun_rows": sum(
                row["training_eligible"] != "true" for row in samples
            ),
        },
        "cell_summary_file": "cell_summary.csv",
        "split_candidates_file": "split_candidates.csv",
        "locked_split_file": "split_manifest.csv" if locked_candidate else None,
        "locked_candidate": locked_candidate,
        "split_version": "phase1-8-2-2-v1" if locked_candidate else None,
        "split_policy": (
            "All RPT/direction rows of a physical cell remain in one split. "
            "Candidates are proposed from coverage and group representation, "
            "never from model metrics."
        ),
        "recommended_candidate": "coverage_balanced",
        "candidate_rationale": (
            (
                "Selected by the user because it provides the larger proposed "
                "test label count while retaining usable SOH rows in both "
                "validation and test cells; the two-cell test is concentrated "
                "in G1/G3."
            )
            if locked_candidate
            else (
                "It provides the larger proposed test label count while retaining "
                "usable SOH rows in both validation and test cells. It is not "
                "locked because the two-cell test is concentrated in G1/G3."
            )
        ),
        "candidate_stats": candidate_stats,
        "modality_status": {
            "discharge": "Step 4 source/cache availability is recorded per sample; "
            "full feature QC remains a training-pipeline concern.",
            "charge": "Step 10 source availability is recorded where present; "
            "direction-specific PyDMA rows remain separate.",
            "eis": "Cell-level CSV presence exists, but RPT alignment and frequency axis are not verified.",
            "physics": "not_ready",
        },
        "gpu_policy": {
            "phase1_manifest": "CPU/read-only data processing; GPU not required.",
            "future_training": (
                "GPU required: must run in .venv-torch-cuda with an explicit "
                "CUDA availability preflight and fail if CUDA is unavailable."
            ),
        },
    }
    _write_json(output_dir / "phase1_manifest_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a read-only-derived Part 1 sample manifest and proposed cell splits."
    )
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--supplemental", type=Path, action="append", required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lock-candidate",
        choices=tuple(_candidate_splits()),
        help="Lock one deterministic 8/2/2 candidate in split_manifest.csv.",
    )
    args = parser.parse_args()
    summary = build_phase1_manifest(
        args.labels.resolve(),
        [path.resolve() for path in args.supplemental],
        args.feature_cache.resolve(),
        args.output.resolve(),
        args.lock_candidate,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
