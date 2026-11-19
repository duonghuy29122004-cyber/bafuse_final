from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any
from importlib.metadata import version

from bafuse.labeling.audit import FileRecord, read_guide_entries
from bafuse.labeling.labels import build_degradation_labels
from bafuse.labeling.pydma_runner import run_pydma_for_cell
from bafuse.labeling.pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
)


DEFAULT_TARGET_CELLS = (("1", "20"), ("2", "4"), ("3", "11"), ("4", "19"))


def _priority_cell_targets(labels_dir: Path) -> list[tuple[str, str]]:
    labels_path = labels_dir / "labels_degradation.csv"
    if not labels_path.is_file():
        return list(DEFAULT_TARGET_CELLS)
    rows = _read_csv(labels_path)
    by_cell: dict[tuple[str, str], dict[str, str | None]] = {}
    for row in rows:
        key = (str(row.get("group_id", "")).strip(), str(row.get("cell_id", "")).strip())
        if not key[0] or not key[1]:
            continue
        cell = by_cell.setdefault(key, {"charge": None, "discharge": None})
        direction = str(row.get("direction", "")).strip().casefold()
        status = str(row.get("identifiability_status", "")).strip()
        if direction == "charge" and cell["charge"] is None:
            cell["charge"] = status
        if direction == "discharge" and cell["discharge"] is None:
            cell["discharge"] = status

    missing_cell = [
        (group, cell)
        for (group, cell), state in by_cell.items()
        if state.get("charge") == "not_run" and state.get("discharge") == "not_run"
    ]
    partial_cell = [
        (group, cell)
        for (group, cell), state in by_cell.items()
        if (state.get("charge") == "not_run" and state.get("discharge") != "not_run")
        or (state.get("discharge") == "not_run" and state.get("charge") != "not_run")
    ]
    ordered = missing_cell + partial_cell
    if not ordered:
        return list(DEFAULT_TARGET_CELLS)
    unique: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in ordered:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unaccepted_fit_targets(
    label_paths: list[Path],
    dataset_root: Path,
    guide_entries: list[GuideEntry] | None = None,
) -> tuple[dict[tuple[str, str], set[tuple[int, str]]], dict[str, int]]:
    slots: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    source_pattern = re.compile(r"TPG(?P<group>\d+)(?:[._](?P<file_id>\d+))?", re.IGNORECASE)
    guide_entries = guide_entries or read_guide_entries(
        dataset_root / "Guide_to_Datafiles.xlsx"
    )
    guide_file_ids: dict[tuple[str, str, int], set[int]] = {}
    for entry in guide_entries:
        stage = re.search(r"\brpt\s*(\d+)\b", entry.stage, re.IGNORECASE)
        if stage is not None:
            guide_file_ids.setdefault(
                (entry.group_id, entry.cell_id, int(stage.group(1))), set()
            ).add(entry.file_id)
    for path in label_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Label input does not exist: {path}")
        for row in _read_csv(path):
            if str(row.get("measurement_id", "")).strip().casefold().startswith("bol"):
                continue
            group = str(row.get("group_id", "")).strip()
            cell = str(row.get("cell_id", "")).strip()
            direction = str(row.get("direction", "")).strip().casefold()
            measurement = re.search(r"\brpt\s*(\d+)\b", str(row.get("measurement_id", "")), re.IGNORECASE)
            if not group or not cell or direction not in {"charge", "discharge"} or measurement is None:
                raise ValueError(f"Cannot identify RPT fit slot from {path}: {row}")
            match = source_pattern.search(str(row.get("source_file", "")))
            if match is not None and match.group("group") != group:
                raise ValueError(f"Source group does not match label row in {path}: {row}")
            file_id = int(match.group("file_id") or 1) if match is not None else None
            key = (group, cell, int(measurement.group(1)), direction)
            state = slots.setdefault(key, {"statuses": [], "accepted": False, "file_ids": set()})
            status = str(row.get("identifiability_status", "")).strip()
            state["statuses"].append(status)
            if file_id is not None:
                state["file_ids"].add(file_id)
            state["accepted"] |= (
                str(row.get("fit_success", "")).strip().casefold() == "true"
                and status == "diagnostic_only_not_independently_validated"
            )

    fit_targets: dict[tuple[str, str], set[tuple[int, str]]] = {}
    accepted_count = 0
    unavailable_count = 0
    for (group, cell, _rpt_number, direction), state in slots.items():
        if state["accepted"]:
            accepted_count += 1
        else:
            is_unavailable = all(status.startswith("not_run") for status in state["statuses"])
            if is_unavailable:
                unavailable_count += 1
            file_ids = state["file_ids"] or guide_file_ids.get(
                (group, cell, _rpt_number), set()
            )
            if len(file_ids) != 1:
                raise ValueError(
                    "An unaccepted RPT slot must resolve to exactly one source file ID "
                    "from its CSV path or the Part 1 guide: "
                    f"{group}/{cell} RPT {_rpt_number} {direction} -> {file_ids}"
                )
            file_id = next(iter(file_ids))
            fit_targets.setdefault((group, cell), set()).add((file_id, direction))

    target_count = sum(len(targets) for targets in fit_targets.values())
    return fit_targets, {
        "unique_fit_slots": len(slots),
        "already_accepted": accepted_count,
        "not_run_for_curve_recheck": unavailable_count,
        "selected_for_optimizer_retry": target_count - unavailable_count,
        "selected_total_for_fit_or_curve_recheck": target_count,
    }


def _record_from_row(row: dict[str, str]) -> FileRecord:
    converted = dict(row)
    for field in ("size_bytes", "modified_ns"):
        converted[field] = int(converted[field])
    for field in ("file_id", "row_count"):
        converted[field] = int(converted[field]) if converted[field] else None
    return FileRecord(**converted)


def run_missing_cell_pydma(
    dataset_root: Path,
    labels_dir: Path,
    output_dir: Path,
    target_cells_override: list[tuple[str, str]] | None = None,
    fit_targets_override: dict[tuple[str, str], set[tuple[int, str]]] | None = None,
    optimizer_options: dict[str, int] | None = None,
    fit_target_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    dataset_root = dataset_root.resolve()
    labels_dir = labels_dir.resolve()
    output_dir = output_dir.resolve()
    if dataset_root.name not in {
        "Path_Dependent_Battery_Degradation_Dataset_Part 1",
        "part1",
    }:
        raise ValueError("Only Oxford Part 1 is permitted for this supplemental PyDMA run.")
    if labels_dir.name != "labeling_output_part1_pydma_all":
        raise ValueError("The source labels must be labeling_output_part1_pydma_all.")
    target_cells = (
        target_cells_override
        if target_cells_override is not None
        else (
            sorted(fit_targets_override)
            if fit_targets_override is not None
            else _priority_cell_targets(labels_dir)
        )
    )
    fit_options = optimizer_options or {}
    run_metadata = json.loads((labels_dir / "run_metadata.json").read_text(encoding="utf-8"))
    if Path(run_metadata["dataset_root"]).resolve() != dataset_root:
        raise ValueError("The selected source labels refer to a different dataset root.")
    try:
        output_dir.relative_to(dataset_root)
    except ValueError:
        pass
    else:
        raise ValueError("Supplemental output must be outside the original dataset.")
    progress_path = output_dir / "run_progress.json"
    partial_results_path = output_dir / "pydma_raw_results.partial.csv"
    resume_state: dict[str, Any] | None = None
    if output_dir.exists() and any(output_dir.iterdir()):
        if not progress_path.is_file():
            raise FileExistsError(f"Refusing to use output without this runner's checkpoint: {output_dir}")
        resume_state = json.loads(progress_path.read_text(encoding="utf-8"))
        if resume_state.get("status") != "running":
            raise FileExistsError(f"Refusing to overwrite completed supplemental output: {output_dir}")
        if (
            Path(resume_state.get("dataset_root", "")).resolve() != dataset_root
            or Path(resume_state.get("labels_dir", "")).resolve() != labels_dir
            or tuple(tuple(item) for item in resume_state.get("target_cells", [])) != tuple(target_cells)
            or resume_state.get("fit_targets", {})
            != {
                f"G{group}-Cell{cell}": sorted(
                    [file_id, direction] for file_id, direction in targets
                )
                for (group, cell), targets in (fit_targets_override or {}).items()
            }
            or resume_state.get("optimizer_options", {}) != fit_options
        ):
            raise ValueError("Checkpoint inputs differ from the requested supplemental PyDMA run.")

    old_degradation = _read_csv(labels_dir / "labels_degradation.csv")
    for group, cell in target_cells:
        existing_rows = [
            row
            for row in old_degradation
            if row["group_id"] == group and row["cell_id"] == cell
        ]
        if not existing_rows:
            continue
        accepted_directions = {
            row["direction"].strip().casefold()
            for row in existing_rows
            if row["fit_success"].strip().casefold() == "true"
            and row["identifiability_status"] == "diagnostic_only_not_independently_validated"
        }
        if accepted_directions == {"charge", "discharge"}:
            continue
        if accepted_directions == set() and not any(
            row["identifiability_status"].strip() != "not_run" for row in existing_rows
        ):
            continue
        # Allow partial cells to be rerun: a cell with one accepted direction but a missing or withheld second direction
        # is still worth optimizing because the goal is to maximize valid labels without fabricating estimates.

    records = [_record_from_row(row) for row in _read_csv(labels_dir / "dataset_manifest.csv")]
    dataset_relative_paths = {record.relative_path for record in records}
    guide_entries = read_guide_entries(dataset_root / "Guide_to_Datafiles.xlsx")
    guide_lookup = {(entry.group_id, entry.cell_id, entry.file_id): entry for entry in guide_entries}

    added_csv_files: list[Path] = []
    for group, cell in target_cells:
        source_dir = dataset_root / f"Group_{group}" / f"Group {group}"
        filename_pattern = re.compile(
            rf"^TPG{re.escape(group)}(?:[._]\d+)?\s*[-_ ]*Cell\s*{re.escape(cell)}(?:\D|$)",
            re.IGNORECASE,
        )
        for path in sorted(source_dir.glob("*.csv")):
            if not filename_pattern.search(path.stem):
                continue
            relative = path.relative_to(dataset_root).as_posix()
            if relative in dataset_relative_paths:
                continue
            file_match = re.match(
                rf"^TPG{re.escape(group)}(?:[._](\d+))?",
                path.stem,
                re.IGNORECASE,
            )
            if file_match is None:
                continue
            file_id = int(file_match.group(1) or 1)
            entry = guide_lookup.get((group, cell, file_id))
            if entry is None:
                continue
            stat = path.stat()
            record = FileRecord(
                relative_path=relative,
                file_type="csv",
                size_bytes=stat.st_size,
                modified_ns=stat.st_mtime_ns,
                format="CSV",
                decode_status="not_full_audit_supplemental_fit_only",
                group_id=group,
                cell_id=cell,
                physical_cell_id=f"G{group}-Cell{cell}",
                file_id=file_id,
                procedure=entry.procedure,
                stage=entry.stage,
                modality="cycling_and_RPT",
                condition=entry.stage,
            )
            records.append(record)
            dataset_relative_paths.add(relative)
            added_csv_files.append(path)

    cell20_source = next(
        (
            path
            for path in (
                dataset_root / "Group_1" / "Group 1" / "TPG1 - Cell 20.csv",
                dataset_root / "Group_1" / "Group 1" / "TPG1 - Cell20.csv",
            )
            if path.is_file()
        ),
        None,
    )
    if cell20_source is None:
        raise FileNotFoundError("The added G1-Cell20 BoL CSV could not be located.")
    relative = cell20_source.relative_to(dataset_root).as_posix()
    if relative not in dataset_relative_paths:
        record, _, _ = scan_csv(cell20_source, dataset_root, guide_lookup)
        records.append(record)

    missing_rpt_csvs = []
    for entry in guide_entries:
        if (entry.group_id, entry.cell_id) not in target_cells:
            continue
        if not (entry.stage.casefold().startswith("bol") or "rpt" in entry.stage.casefold()):
            continue
        has_csv = any(
            record.file_type == "csv"
            and record.group_id == entry.group_id
            and record.cell_id == entry.cell_id
            and record.file_id == entry.file_id
            for record in records
        )
        if not has_csv:
            missing_rpt_csvs.append(
                {
                    "group_id": entry.group_id,
                    "cell_id": entry.cell_id,
                    "file_id": entry.file_id,
                    "stage": entry.stage,
                    "status": "not_fit_no_readable_csv; source MAT remains undecoded",
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    if resume_state is None:
        raw_results: list[dict[str, Any]] = []
        runtime_by_cell: list[dict[str, Any]] = []
        cell_status = {}
        resume_state = {
            "status": "running",
            "dataset_root": str(dataset_root),
            "labels_dir": str(labels_dir),
            "target_cells": [list(item) for item in target_cells],
            "fit_targets": {
                f"G{group}-Cell{cell}": sorted(
                    [file_id, direction] for file_id, direction in targets
                )
                for (group, cell), targets in (fit_targets_override or {}).items()
            },
            "optimizer_options": fit_options,
            "current_cell": None,
            "cell_status": cell_status,
            "runtime_by_cell": runtime_by_cell,
            "raw_rows_checkpointed": 0,
            "started_at_unix": time.time(),
        }
        _write_json(progress_path, resume_state)
    else:
        raw_results = _read_csv(partial_results_path) if partial_results_path.is_file() else []
        runtime_by_cell = list(resume_state.get("runtime_by_cell", []))
        cell_status = dict(resume_state.get("cell_status", {}))
        resume_state["cell_status"] = cell_status
        resume_state["runtime_by_cell"] = runtime_by_cell

    for group, cell in target_cells:
        key = f"G{group}-Cell{cell}"
        if cell_status.get(key, {}).get("status") == "completed":
            print(f"Skipping already checkpointed cell {key}", flush=True)
            continue
        start = time.perf_counter()
        print(f"Running PyDMA only for missing-label Group {group}, Cell {cell}", flush=True)
        resume_state["current_cell"] = key
        resume_state["cell_status"][key] = {
            "status": "running",
            "started_at_unix": time.time(),
            "rows_checkpointed_before_cell": len(raw_results),
        }
        _write_json(progress_path, resume_state)
        try:
            rows = run_pydma_for_cell(
                dataset_root,
                records,
                guide_entries,
                group,
                cell,
                use_cuda=True,
                fit_targets=(
                    fit_targets_override.get((group, cell))
                    if fit_targets_override is not None
                    else None
                ),
                optimizer_options=fit_options,
            )
            cell_state = {
                "status": "completed",
                "rows_returned": len(rows),
                "elapsed_seconds": time.perf_counter() - start,
            }
        except Exception as error:
            rows = []
            cell_state = {
                "status": "failed",
                "rows_returned": 0,
                "elapsed_seconds": time.perf_counter() - start,
                "error": f"{type(error).__name__}: {error}",
            }
            print(f"Cell-level PyDMA error {key}: {cell_state['error']}", flush=True)
        runtime_by_cell.append({"group_id": group, "cell_id": cell, **cell_state})
        raw_results.extend(rows)
        _write_csv(partial_results_path, raw_results)
        resume_state["cell_status"][key] = cell_state
        resume_state["runtime_by_cell"] = runtime_by_cell
        resume_state["raw_rows_checkpointed"] = len(raw_results)
        resume_state["current_cell"] = None
        _write_json(progress_path, resume_state)
        print(
            f"Checkpointed {key}: status={cell_state['status']} rows={len(raw_results)} "
            f"elapsed_s={cell_state['elapsed_seconds']:.1f}",
            flush=True,
        )

    derived_all, validation_all = build_degradation_labels(
        records,
        guide_entries,
        imported_results=None,
        generated_results=raw_results,
    )
    target_physical_cells = {f"G{group}-Cell{cell}" for group, cell in target_cells}
    supplemental = [row for row in derived_all if row["physical_cell_id"] in target_physical_cells]
    validation = [row for row in validation_all if row["physical_cell_id"] in target_physical_cells]
    _write_csv(output_dir / "pydma_raw_results.csv", raw_results)
    _write_csv(output_dir / "labels_degradation_supplemental.csv", supplemental)
    _write_csv(output_dir / "label_validation_supplemental.csv", validation)
    _write_csv(output_dir / "cell20_unavailable_rpt_mat.csv", missing_rpt_csvs)

    accepted = [
        row
        for row in supplemental
        if row["fit_success"]
        and row["identifiability_status"] == "diagnostic_only_not_independently_validated"
    ]
    coverage = []
    for group, cell in target_cells:
        physical = f"G{group}-Cell{cell}"
        cell_rows = [row for row in supplemental if row["physical_cell_id"] == physical]
        cell_accepted = [row for row in accepted if row["physical_cell_id"] == physical]
        coverage.append(
            {
                "group_id": group,
                "physical_cell_id": physical,
                "generated_rows": len(cell_rows),
                "accepted_rows": len(cell_accepted),
                "accepted_charge": sum(row["direction"] == "charge" for row in cell_accepted),
                "accepted_discharge": sum(row["direction"] == "discharge" for row in cell_accepted),
                "not_run_or_withheld": len(cell_rows) - len(cell_accepted),
            }
        )
    _write_csv(output_dir / "cell_coverage.csv", coverage)

    config = {
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
        "selected_existing_labels": str(labels_dir),
        "dataset_manifest_hash_before_cell20_addition": run_metadata["dataset_manifest_hash"],
        "added_base_file": {
            "relative_path": cell20_source.relative_to(dataset_root).as_posix(),
            "size_bytes": cell20_source.stat().st_size,
            "sha256": _sha256(cell20_source),
        },
        "added_csv_files": [
            path.relative_to(dataset_root).as_posix() for path in added_csv_files
        ],
        "target_cells_only": [f"G{group}-Cell{cell}" for group, cell in target_cells],
        "target_fit_slots": {
            f"G{group}-Cell{cell}": [
                {"file_id": file_id, "direction": direction}
                for file_id, direction in sorted(targets)
            ]
            for (group, cell), targets in (fit_targets_override or {}).items()
        },
        "fit_target_selection": fit_target_summary,
        "optimizer_options": fit_options,
        "pydma_version_this_run": version("pydma"),
        "pydma_config_this_run": {
            "charge_step": 10,
            "discharge_step": 8,
            "current_filter_a": "directional sign and abs(Amps) <= 0.2",
            "data_length": 500,
            "weight_ocv": 100.0,
            "weight_dva": 0.0,
            "weight_ica": 0.0,
            "speed_preset": "fast",
            "req_accepted": 3,
            "max_tries_overall": fit_options.get("max_tries", 10),
            "random_seed": 211,
            "analyze_maxiter": fit_options.get("maxiter", 50),
            "analyze_popsize": fit_options.get("popsize", 30),
            "polish": True,
            "optimizer_backend": "PyTorch CUDA batched best1bin with GPU L-BFGS polish",
            "acceptance": (
                f"is_accepted, RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV, at least "
                f"{PYDMA_MIN_ACCEPTED_RUNS} accepted runs; compare with "
                "same-direction BoL fit"
            ),
        },
        "elapsed_by_cell": runtime_by_cell,
        "cell_status": cell_status,
        "missing_cell20_rpt_sources": missing_rpt_csvs,
        "accepted_supplemental_rows": len(accepted),
        "accepted_by_cell": coverage,
        "original_label_directory_modified": False,
        "older_run_directories_overwritten": False,
        "training_started": False,
        "parts_2_3_read": False,
        "label_warning": "Supplemental LLI/LAM values are PyDMA model-derived labels, not independently verified physical ground truth.",
    }
    _write_json(output_dir / "run_metadata.json", config)
    resume_state["status"] = "completed"
    resume_state["completed_at_unix"] = time.time()
    resume_state["current_cell"] = None
    _write_json(progress_path, resume_state)
    if partial_results_path.exists():
        partial_results_path.unlink()
    return config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate supplemental PyDMA labels for selected Oxford Part 1 cells."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--target-cell",
        nargs=2,
        action="append",
        metavar=("GROUP", "CELL"),
        help="Run only this group/cell pair; may be supplied more than once.",
    )
    parser.add_argument(
        "--optimize-unaccepted-from",
        type=Path,
        action="append",
        help="Label CSV to include when selecting RPT/direction slots without accepted labels; repeat for each source.",
    )
    parser.add_argument("--max-tries", type=int, default=10)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--popsize", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if min(args.max_tries, args.maxiter, args.popsize) < 1:
        parser.error("--max-tries, --maxiter, and --popsize must be positive.")
    target_cells = (
        [(str(group), str(cell)) for group, cell in args.target_cell]
        if args.target_cell
        else None
    )
    fit_targets = None
    fit_target_summary = None
    if args.optimize_unaccepted_from:
        if args.target_cell:
            parser.error("--target-cell cannot be combined with --optimize-unaccepted-from.")
        fit_targets, fit_target_summary = _unaccepted_fit_targets(
            args.optimize_unaccepted_from,
            args.dataset,
        )
        target_cells = sorted(fit_targets)
        if args.dry_run:
            print(
                json.dumps(
                    {
                        "target_cells": [f"G{group}-Cell{cell}" for group, cell in target_cells],
                        "fit_target_selection": fit_target_summary,
                        "optimizer_options": {
                            "max_tries": args.max_tries,
                            "maxiter": args.maxiter,
                            "popsize": args.popsize,
                        },
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
    summary = run_missing_cell_pydma(
        args.dataset,
        args.labels,
        args.output,
        target_cells_override=target_cells,
        fit_targets_override=fit_targets,
        optimizer_options={
            "max_tries": args.max_tries,
            "maxiter": args.maxiter,
            "popsize": args.popsize,
        },
        fit_target_summary=fit_target_summary,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()