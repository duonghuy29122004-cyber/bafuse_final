from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from . import __version__
from .audit import (
    FileRecord,
    GuideEntry,
    audit_dataset,
    feature_schema,
    read_guide_entries,
    validate_cached_manifest,
)
from .labels import build_degradation_labels, build_soh_labels
from .pydma_runner import run_pydma_for_available_cells, run_pydma_for_cell
from .pydma_policy import PYDMA_MIN_ACCEPTED_RUNS, PYDMA_RMSE_THRESHOLD_MV

_UNFLAGGED_SOH_STATUSES = {"baseline_reference", "calculated_from_measured_capacity"}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def _quality_rows(records: list[FileRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        if record.file_type == "csv":
            try:
                missing = json.loads(record.missing_counts_json or "{}")
            except json.JSONDecodeError:
                missing = {}
            try:
                types = json.loads(record.column_types_json or "{}")
            except json.JSONDecodeError:
                types = {}
            for column in json.loads(record.columns_json or "[]"):
                rows.append(
                    {
                        "relative_path": record.relative_path,
                        "group_id": record.group_id,
                        "physical_cell_id": record.physical_cell_id,
                        "modality": record.modality,
                        "column": column,
                        "dtype": types.get(column, "unknown"),
                        "missing_count": missing.get(column, ""),
                        "row_count": record.row_count,
                        "status": record.decode_status,
                        "warning": record.warnings,
                        "error": record.error,
                    }
                )
        else:
            rows.append(
                {
                    "relative_path": record.relative_path,
                    "group_id": record.group_id,
                    "physical_cell_id": record.physical_cell_id,
                    "modality": record.modality,
                    "column": "",
                    "dtype": "",
                    "missing_count": "",
                    "row_count": "",
                    "status": record.decode_status,
                    "warning": record.warnings,
                    "error": record.error,
                }
            )
    return rows


def _leakage_audit(records: list[FileRecord]) -> list[dict[str, Any]]:
    observed = sorted(
        {
            name
            for record in records
            if record.file_type == "csv"
            for name in json.loads(record.columns_json or "[]")
        }
    )
    rows = [
        {
            "target": "SOH",
            "feature_or_source": "Amphr / Capacity / Step 4 discharge capacity",
            "role": "target_definition_source",
            "allowed_as_model_input": False,
            "reason": "The same measured discharge capacity defines the SOH ratio; including it leaks the target.",
        },
        {
            "target": "SOH",
            "feature_or_source": "Future RPT, future EIS, or later cycling data",
            "role": "future_measurement",
            "allowed_as_model_input": False,
            "reason": "Temporal prediction must only use measurements available at or before prediction time.",
        },
        {
            "target": "SOH",
            "feature_or_source": "EIS features at the same/earlier prediction checkpoint",
            "role": "candidate_input",
            "allowed_as_model_input": True,
            "reason": "Potentially valid only with time-aligned observations and group/cell-level holdout; do not use future MoL/BoL records.",
        },
        {
            "target": "SOH",
            "feature_or_source": "Group/cell identifiers, file id, RPT order, raw cycle counter",
            "role": "metadata",
            "allowed_as_model_input": False,
            "reason": "Identifiers/order are provenance metadata, not measurements; prevent memorization and temporal leakage.",
        },
        {
            "target": "LLI/LAM",
            "feature_or_source": "Pseudo-OCV Step 8/10 voltage and capacity axes used by PyDMA",
            "role": "label_generation_signal",
            "allowed_as_model_input": "conditional",
            "reason": "A model using the same pOCV signals may reproduce the PyDMA label-generation procedure rather than predict an independent target.",
        },
        {
            "target": "all",
            "feature_or_source": "Observed CSV columns",
            "role": "inventory",
            "allowed_as_model_input": "requires_review",
            "reason": "Field names do not prove predictive availability or unit; inspect feature timestamps and remove target-defining fields.",
        },
    ]
    for name in observed:
        if name.strip().lower() in {"amphr", "capacity"}:
            rows.append(
                {
                    "target": "SOH",
                    "feature_or_source": name,
                    "role": "observed_column",
                    "allowed_as_model_input": False,
                    "reason": "Capacity-like field is used in target calculation or may encode the target directly.",
                }
            )
    return rows


def _cell_identity_inventory(
    records: list[FileRecord], guide_entries: list[GuideEntry]
) -> dict[str, Any]:
    documented_pairs = {(entry.group_id, entry.cell_id) for entry in guide_entries}
    filename_pairs = {
        (record.group_id, record.cell_id)
        for record in records
        if record.group_id and record.cell_id
    }
    unmatched_pairs = filename_pairs - documented_pairs
    eis_pairs = {
        (record.group_id, record.cell_id)
        for record in records
        if record.modality == "EIS" and record.group_id and record.cell_id
    }
    tpg_pairs = {
        (record.group_id, record.cell_id)
        for record in records
        if record.modality == "cycling_and_RPT" and record.group_id and record.cell_id
    }
    return {
        "documented_physical_cells": len(documented_pairs),
        "filename_group_cell_pairs": len(filename_pairs),
        "filename_pairs_not_in_guide": len(unmatched_pairs),
        "unlinked_eis_group_cell_pairs": len(eis_pairs - documented_pairs),
        "unmapped_tpg_group_cell_pairs": len(tpg_pairs - documented_pairs),
        "documented_pairs": documented_pairs,
    }


def _coverage_rows(
    records: list[FileRecord],
    soh_rows: list[dict[str, Any]],
    degradation_rows: list[dict[str, Any]],
    guide_entries: list[GuideEntry],
) -> list[dict[str, Any]]:
    cells = sorted({record.physical_cell_id for record in records if record.physical_cell_id})
    identity_inventory = _cell_identity_inventory(records, guide_entries)
    soh_by_cell: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    degradation_by_cell: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in soh_rows:
        soh_by_cell[row["physical_cell_id"]].append(row)
    for row in degradation_rows:
        degradation_by_cell[row["physical_cell_id"]].append(row)
    rows = []
    for physical in cells:
        cell_records = [record for record in records if record.physical_cell_id == physical]
        group_id, cell_id = physical[1:].split("-Cell", 1)
        documented = (group_id, cell_id) in identity_inventory["documented_pairs"]
        labels = soh_by_cell.get(physical, [])
        dlabels = degradation_by_cell.get(physical, [])
        base = {
            "group_id": physical.split("-", 1)[0].removeprefix("G"),
            "physical_cell_id": physical,
            "identity_status": "confirmed_in_guide" if documented else "filename_identity_unlinked",
            "csv_files": sum(record.file_type == "csv" for record in cell_records),
            "mat_files": sum(record.file_type == "mat" for record in cell_records),
            "eis_files": sum(record.modality == "EIS" for record in cell_records),
            "rpt_observations": len(labels),
            "soh_unflagged_observations": sum(row["validation_status"] in _UNFLAGGED_SOH_STATUSES for row in labels),
            "soh_missing_or_flagged_observations": sum(row["soh_percent"] is None or row["validation_status"].startswith(("unavailable", "review_", "physically_")) for row in labels),
            "degradation_fit_observations": len(dlabels),
            "degradation_accepted_directions": sum(bool(row["fit_success"]) for row in dlabels),
            "degradation_unavailable_or_withheld": sum(not bool(row["fit_success"]) for row in dlabels),
            "notes": (
                "Guide-listed TPG Group+Cell."
                if documented
                else "Filename-derived Group+Cell is not linked to a physical TPG cell in the local Guide; do not count as a confirmed distinct physical cell."
            )
            + " EIS SOC/BoL/MoL are extracted from filenames. MAT contents may be opaque and are not counted as decoded observations.",
        }
        rows.append({"condition": "ALL", "label_family": "coverage_summary", **base})
        condition_counts = Counter(
            f"{record.stage} SOC {record.soc}%"
            for record in cell_records
            if record.modality == "EIS" and record.stage and record.soc
        )
        for condition, count in sorted(condition_counts.items()):
            rows.append(
                {
                    "condition": condition,
                    "label_family": "EIS_measurement_coverage",
                    **{**base, "rpt_observations": 0, "soh_unflagged_observations": 0, "soh_missing_or_flagged_observations": 0, "degradation_fit_observations": 0, "degradation_accepted_directions": 0, "degradation_unavailable_or_withheld": 0},
                    "eis_files": count,
                    "notes": "Filename-derived EIS stage/SOC; this is measurement coverage, not a degradation label.",
                }
            )
        if labels:
            rows.append(
                {
                    "condition": "RPT",
                    "label_family": "SOH",
                    **{**base, "csv_files": 0, "mat_files": 0, "eis_files": 0, "degradation_fit_observations": 0, "degradation_accepted_directions": 0, "degradation_unavailable_or_withheld": 0},
                    "notes": "SOH coverage by same-cell RPT; units and baseline confidence are recorded in labels_soh.csv.",
                }
            )
        for direction in ("charge", "discharge"):
            directional = [row for row in dlabels if row["direction"] == direction]
            if directional:
                rows.append(
                    {
                        "condition": f"RPT {direction}",
                        "label_family": "LLI_LAM",
                        **{**base, "csv_files": 0, "mat_files": 0, "eis_files": 0, "rpt_observations": 0, "soh_unflagged_observations": 0, "soh_missing_or_flagged_observations": 0},
                        "degradation_fit_observations": len(directional),
                        "degradation_accepted_directions": sum(bool(row["fit_success"]) for row in directional),
                        "degradation_unavailable_or_withheld": sum(not bool(row["fit_success"]) for row in directional),
                        "notes": (
                            f"PyDMA diagnostic coverage by direction; accepted counts "
                            f"require RMSE <{PYDMA_RMSE_THRESHOLD_MV:g} mV and at least "
                            f"{PYDMA_MIN_ACCEPTED_RUNS} accepted runs."
                        ),
                    }
                )
    return rows


def _plot_soh(path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row["soh_percent"] is not None]
    fig, axis = plt.subplots(figsize=(10, 6))
    for physical in sorted({row["physical_cell_id"] for row in valid}):
        observations = [row for row in valid if row["physical_cell_id"] == physical]
        observations.sort(key=lambda item: item["measurement_order"])
        axis.plot(
            range(len(observations)),
            [row["soh_percent"] for row in observations],
            marker="o",
            linewidth=1,
            label=physical,
        )
    axis.set_xlabel("Available RPT order (not lifetime cycle count)")
    axis.set_ylabel("SOH estimate (%)")
    axis.set_title("Part 1 SOH from measured Step 4 discharge capacity")
    axis.grid(True, alpha=0.3)
    if valid:
        axis.legend(fontsize=7, ncol=2)
    else:
        axis.text(0.5, 0.5, "No justified SOH observations", ha="center", va="center", transform=axis.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_degradation(path: Path, rows: list[dict[str, Any]]) -> None:
    valid = [row for row in rows if row["fit_success"]]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharex=True)
    for axis, key, title in zip(
        axes,
        ("lli_percent", "lam_anode_percent", "lam_cathode_percent"),
        ("LLI (%)", "LAM anode (%)", "LAM cathode (%)"),
    ):
        series = sorted({(row["physical_cell_id"], row["direction"]) for row in valid})
        for physical, direction in series:
            observations = [row for row in valid if row["physical_cell_id"] == physical and row["direction"] == direction]
            observations.sort(key=lambda item: (0 if item["measurement_id"].casefold().startswith("bol") else int("".join(char for char in item["measurement_id"] if char.isdigit()) or 0)))
            axis.plot([row["measurement_id"] for row in observations], [row[key] for row in observations], marker="o", label=f"{physical} {direction}")
        axis.set_title(title)
        axis.grid(True, alpha=0.3)
        axis.tick_params(axis="x", labelrotation=60)
    axes[0].set_ylabel("Model-based diagnostic estimate")
    if valid:
        axes[-1].legend(fontsize=7)
    fig.suptitle("PyDMA estimates at accepted RPT checkpoints only (charge/discharge fits)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _summary_rows(
    records: list[FileRecord],
    guide_entries: list[GuideEntry],
    soh_rows: list[dict[str, Any]],
    degradation_rows: list[dict[str, Any]],
    dataset_hash: str,
) -> list[dict[str, Any]]:
    statuses = Counter(row["identifiability_status"] for row in degradation_rows)
    soh_counts = Counter(row["validation_status"] for row in soh_rows)
    identities = _cell_identity_inventory(records, guide_entries)
    numeric_soh_count = sum(row["soh_percent"] is not None for row in soh_rows)
    unflagged_soh_count = sum(row["validation_status"] in _UNFLAGGED_SOH_STATUSES for row in soh_rows)
    return [
        {"section": "dataset", "metric": "scope", "value": "Part 1 only"},
        {"section": "dataset", "metric": "csv_files", "value": sum(record.file_type == "csv" for record in records)},
        {"section": "dataset", "metric": "mat_files", "value": sum(record.file_type == "mat" for record in records)},
        {"section": "dataset", "metric": "documented_physical_tpg_cells", "value": identities["documented_physical_cells"]},
        {"section": "dataset", "metric": "distinct_filename_group_cell_pairs", "value": identities["filename_group_cell_pairs"]},
        {"section": "dataset", "metric": "filename_pairs_not_linked_in_guide", "value": identities["filename_pairs_not_in_guide"]},
        {"section": "dataset", "metric": "unlinked_eis_group_cell_pairs", "value": identities["unlinked_eis_group_cell_pairs"]},
        {"section": "dataset", "metric": "unmapped_tpg_group_cell_pairs", "value": identities["unmapped_tpg_group_cell_pairs"]},
        {"section": "dataset", "metric": "manifest_fingerprint", "value": dataset_hash},
        {"section": "SOH", "metric": "observations_with_numeric_labels", "value": numeric_soh_count},
        {"section": "SOH", "metric": "unflagged_numeric_observations", "value": unflagged_soh_count},
        {"section": "SOH", "metric": "numeric_review_observations", "value": numeric_soh_count - unflagged_soh_count},
        {"section": "SOH", "metric": "unavailable_or_flagged", "value": sum(row["soh_percent"] is None or row["validation_status"].startswith(("review_", "physically_", "unavailable")) for row in soh_rows)},
        {"section": "SOH", "metric": "status_counts", "value": json.dumps(soh_counts, sort_keys=True)},
        {"section": "degradation", "metric": "accepted_estimates", "value": sum(bool(row["fit_success"]) for row in degradation_rows)},
        {"section": "degradation", "metric": "withheld_or_unrun", "value": sum(not bool(row["fit_success"]) for row in degradation_rows)},
        {"section": "degradation", "metric": "status_counts", "value": json.dumps(statuses, sort_keys=True)},
        {"section": "issues", "metric": "mat_files_not_decoded", "value": sum(record.file_type == "mat" for record in records)},
        {"section": "issues", "metric": "capacity_absolute_unit", "value": "Unverified; SOH ratio uses like-for-like native Amphr spans, absolute Ah omitted."},
        {"section": "issues", "metric": "cell_identity", "value": "The local Readme/Guide documents 12 TPG cells. Additional Group+Cell pairs parsed from filenames are retained as unlinked identities, not counted as confirmed physical cells."},
    ]


def _load_cached_audit(
    cache_dir: Path, dataset_root: Path
) -> tuple[list[FileRecord], list[dict[str, Any]], list[GuideEntry], str]:
    metadata_path = cache_dir / "run_metadata.json"
    manifest_path = cache_dir / "dataset_manifest.csv"
    quality_path = cache_dir / "data_quality_report.csv"
    for path in (metadata_path, manifest_path, quality_path):
        if not path.is_file():
            raise FileNotFoundError(f"Cached audit is missing required file: {path}")

    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if Path(metadata.get("dataset_root", "")).resolve() != dataset_root:
        raise ValueError("Cached audit was generated from a different dataset root.")
    dataset_hash = metadata.get("dataset_manifest_hash")
    if not isinstance(dataset_hash, str) or not dataset_hash:
        raise ValueError("Cached audit does not contain a valid dataset fingerprint.")

    integer_fields = {"size_bytes", "modified_ns"}
    optional_integer_fields = {"file_id", "row_count"}
    records: list[FileRecord] = []
    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            for field_name in integer_fields:
                row[field_name] = int(row[field_name])
            for field_name in optional_integer_fields:
                row[field_name] = int(row[field_name]) if row[field_name] else None
            records.append(FileRecord(**row))
    validate_cached_manifest(dataset_root, records, dataset_hash)

    with quality_path.open(encoding="utf-8-sig", newline="") as stream:
        quality_rows = list(csv.DictReader(stream))
    guide_entries = read_guide_entries(dataset_root / "Guide_to_Datafiles.xlsx")
    return records, quality_rows, guide_entries, dataset_hash


def run_pipeline(
    dataset_root: Path,
    output_dir: Path,
    imported_pydma_results: Path | None = None,
    pydma_cell: tuple[str, str] | None = None,
    reuse_audit_dir: Path | None = None,
    pydma_all: bool = False,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    selected_pydma_modes = sum(
        (
            imported_pydma_results is not None,
            pydma_cell is not None,
            pydma_all,
        )
    )
    if selected_pydma_modes > 1:
        raise ValueError("Choose only one of --pydma-results, --pydma-cell, or --pydma-all.")
    try:
        output_dir.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Output directory must not be inside the source Part 1 dataset.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty; refusing to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if reuse_audit_dir is None:
        records, quality_rows, guide_entries, dataset_hash = audit_dataset(dataset_root)
    else:
        records, quality_rows, guide_entries, dataset_hash = _load_cached_audit(
            reuse_audit_dir.resolve(), dataset_root
        )
    for record in records:
        if record.file_type == "mat":
            continue
        record.format = "CSV"
    manifest = [record.__dict__ for record in records]
    soh_rows, soh_validation = build_soh_labels(records, guide_entries)
    if pydma_cell is not None:
        generated_pydma_results = run_pydma_for_cell(
            dataset_root, records, guide_entries, *pydma_cell
        )
    elif pydma_all:
        generated_pydma_results = run_pydma_for_available_cells(
            dataset_root, records, guide_entries
        )
    else:
        generated_pydma_results = None
    degradation_rows, degradation_validation = build_degradation_labels(
        records,
        guide_entries,
        imported_pydma_results,
        generated_pydma_results,
    )
    if reuse_audit_dir is None:
        quality_rows = _quality_rows(records)
    validation_rows = soh_validation + degradation_validation
    leakage_rows = _leakage_audit(records)
    coverage_rows = _coverage_rows(records, soh_rows, degradation_rows, guide_entries)
    identity_inventory = _cell_identity_inventory(records, guide_entries)
    summary = {
        "pipeline_version": __version__,
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
        "dataset_root": str(dataset_root),
        "dataset_manifest_hash": dataset_hash,
        "manifest_hash_method": "SHA-256 over relative path, size, and last modified timestamp; not a full content hash.",
        "csv_files_discovered": sum(record.file_type == "csv" for record in records),
        "mat_files_discovered": sum(record.file_type == "mat" for record in records),
        "documented_physical_tpg_cells": identity_inventory["documented_physical_cells"],
        "distinct_filename_group_cell_pairs": identity_inventory["filename_group_cell_pairs"],
        "filename_pairs_not_linked_in_guide": identity_inventory["filename_pairs_not_in_guide"],
        "unlinked_eis_group_cell_pairs": identity_inventory["unlinked_eis_group_cell_pairs"],
        "unmapped_tpg_group_cell_pairs": identity_inventory["unmapped_tpg_group_cell_pairs"],
        "soh_observations": len(soh_rows),
        "soh_numeric_observations": sum(row["soh_percent"] is not None for row in soh_rows),
        "soh_unflagged_numeric_observations": sum(
            row["validation_status"] in _UNFLAGGED_SOH_STATUSES for row in soh_rows
        ),
        "soh_numeric_review_observations": sum(
            row["soh_percent"] is not None and row["validation_status"] not in _UNFLAGGED_SOH_STATUSES
            for row in soh_rows
        ),
        "degradation_directions": len(degradation_rows),
        "accepted_pydma_estimates": sum(bool(row["fit_success"]) for row in degradation_rows),
        "pydma_import": str(imported_pydma_results) if imported_pydma_results else None,
        "pydma_cell_run": f"Group {pydma_cell[0]} Cell {pydma_cell[1]}" if pydma_cell else None,
        "pydma_all_available_cells_run": pydma_all,
        "soh_capacity_unit": "Amphr absolute unit is not explicitly documented; dimensionless ratios only.",
        "official_soh_labels_found": False,
        "mat_decoding": "Headers inspected; MATLAB variables and MCOS structures were not decoded.",
        "deep_learning_started": False,
        "source_files_modified": False,
    }
    _write_csv(output_dir / "dataset_manifest.csv", manifest)
    _write_csv(output_dir / "data_quality_report.csv", quality_rows)
    _write_json(output_dir / "feature_schema.json", feature_schema(records, dataset_hash))
    _write_csv(output_dir / "labels_soh.csv", soh_rows)
    _write_csv(output_dir / "labels_degradation.csv", degradation_rows)
    _write_csv(output_dir / "label_validation_report.csv", validation_rows)
    _write_csv(output_dir / "label_coverage_report.csv", coverage_rows)
    _write_csv(output_dir / "label_leakage_audit.csv", leakage_rows)
    _write_csv(
        output_dir / "pipeline_summary.csv",
        _summary_rows(records, guide_entries, soh_rows, degradation_rows, dataset_hash),
    )
    _write_json(output_dir / "run_metadata.json", summary)
    _plot_soh(output_dir / "soh_by_rpt_order.png", soh_rows)
    _plot_degradation(output_dir / "lli_lam_by_rpt.png", degradation_rows)
    failures = [record for record in records if record.error]
    if failures:
        _write_csv(
            output_dir / "audit_errors.csv",
            [{"relative_path": record.relative_path, "error": record.error} for record in failures],
        )
        raise RuntimeError(
            f"Audit completed with {len(failures)} CSV parsing failure(s); see {output_dir / 'audit_errors.csv'}."
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Oxford Part 1 and build validated diagnostic labels.")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to Oxford Part 1 source files.",
    )
    parser.add_argument("--output", type=Path, required=True, help="New or empty directory outside the source dataset")
    parser.add_argument("--pydma-results", type=Path, help="Optional prior PyDMA output CSV; imported estimates are revalidated, not recomputed.")
    parser.add_argument("--pydma-cell", help="Run PyDMA for one cell, written GROUP:CELL (e.g. 1:15); may take several minutes per RPT/direction.")
    parser.add_argument("--pydma-all", action="store_true", help="Run PyDMA for every Guide-listed cell with matched cycling/RPT CSVs; may take a long time.")
    parser.add_argument("--reuse-audit", type=Path, help="Reuse a prior audit output after validating its dataset fingerprint; skips rereading CSV contents.")
    args = parser.parse_args()
    selected_cell = None
    if args.pydma_cell:
        parts = args.pydma_cell.split(":")
        if len(parts) != 2 or not all(part.isdigit() for part in parts):
            parser.error("--pydma-cell must be GROUP:CELL using numeric IDs, e.g. 1:15")
        selected_cell = (parts[0], parts[1])
    result = run_pipeline(
        args.dataset,
        args.output,
        args.pydma_results,
        selected_cell,
        args.reuse_audit,
        args.pydma_all,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
