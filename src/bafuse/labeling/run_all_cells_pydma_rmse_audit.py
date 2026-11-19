from __future__ import annotations

import argparse
import csv
import json
import re
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from bafuse.labeling.audit import FileRecord, read_guide_entries, scan_csv
from bafuse.labeling.pydma_policy import PYDMA_RMSE_THRESHOLD_MV
from bafuse.labeling.pydma_runner import run_pydma_for_cell


CONFIRMED_CELLS = (
    ("1", "9"),
    ("1", "15"),
    ("1", "20"),
    ("2", "3"),
    ("2", "4"),
    ("2", "8"),
    ("3", "10"),
    ("3", "11"),
    ("3", "14"),
    ("4", "12"),
    ("4", "18"),
    ("4", "19"),
)
RMSE_THRESHOLDS_MV = (5.0, 10.0, 11.0, 12.0, 15.0, 20.0)
RMSE_BINS = (
    ("lt_5mV", lambda value: value < 5.0),
    ("5_to_lt_10mV", lambda value: 5.0 <= value < 10.0),
    ("10_to_lt_11mV", lambda value: 10.0 <= value < 11.0),
    ("11_to_lt_12mV", lambda value: 11.0 <= value < 12.0),
    ("12_to_lt_15mV", lambda value: 12.0 <= value < 15.0),
    ("15_to_lt_20mV", lambda value: 15.0 <= value < 20.0),
    ("gte_20mV", lambda value: value >= 20.0),
)


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


def _record_from_row(row: dict[str, str]) -> FileRecord:
    converted = dict(row)
    for field in ("size_bytes", "modified_ns"):
        converted[field] = int(converted[field])
    for field in ("file_id", "row_count"):
        converted[field] = int(converted[field]) if converted[field] else None
    return FileRecord(**converted)


def _collect_records(
    dataset_root: Path,
    labels_dir: Path,
    guide_entries: list[Any],
) -> tuple[list[FileRecord], list[str]]:
    records = [
        _record_from_row(row)
        for row in _read_csv(labels_dir / "dataset_manifest.csv")
    ]
    existing = {record.relative_path for record in records}
    guide_lookup = {(entry.group_id, entry.cell_id, entry.file_id): entry for entry in guide_entries}
    added: list[str] = []
    filename_pattern = re.compile(
        r"^TPG(?P<group>\d+)(?:[._]\d+)?\s*[-_ ]*Cell\s*(?P<cell>\d+)(?:\D|$)",
        re.IGNORECASE,
    )
    for group, cell in CONFIRMED_CELLS:
        source_dir = dataset_root / f"Group_{group}" / f"Group {group}"
        for path in sorted(source_dir.glob("*.csv")):
            match = filename_pattern.match(path.stem)
            if match is None or (match.group("group"), match.group("cell")) != (group, cell):
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
            relative = path.relative_to(dataset_root).as_posix()
            if relative in existing:
                continue
            record, _, _ = scan_csv(path, dataset_root, guide_lookup)
            records.append(record)
            existing.add(relative)
            added.append(relative)
    return records, added


def _numeric_rmse(row: dict[str, Any]) -> float | None:
    value = str(row.get("fit_rmse_mV", "")).strip()
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    return parsed if np.isfinite(parsed) else None


def _fit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if str(row.get("checkup", "")).strip().casefold() != "bol"
        and _numeric_rmse(row) is not None
    ]


def _summarize(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    fit_rows = _fit_rows(rows)
    by_cell: list[dict[str, Any]] = []
    for group, cell in CONFIRMED_CELLS:
        physical = f"G{group}-Cell{cell}"
        cell_rows = [
            row
            for row in rows
            if row.get("group_id") == f"Group {group}"
            and row.get("cell_id") == f"Cell {cell}"
            and str(row.get("checkup", "")).strip().casefold() != "bol"
        ]
        cell_fits = _fit_rows(cell_rows)
        values = np.asarray([_numeric_rmse(row) for row in cell_fits], dtype=np.float64)
        values = values[np.isfinite(values)]
        counts = {
            f"count_rmse_le_{str(threshold).replace('.', '_')}mV": int(
                np.sum(values <= threshold)
            )
            for threshold in RMSE_THRESHOLDS_MV
        }
        bins = {
            f"count_{name}": sum(
                predicate(float(value)) for value in values
            )
            for name, predicate in RMSE_BINS
        }
        by_cell.append(
            {
                "physical_cell_id": physical,
                "rpt_rows_returned": len(cell_rows),
                "numeric_rpt_fits": int(values.size),
                "rpt_rows_without_numeric_fit": len(cell_rows) - int(values.size),
                "rmse_mean_mV": float(np.mean(values)) if values.size else None,
                "rmse_median_mV": float(np.median(values)) if values.size else None,
                "rmse_min_mV": float(np.min(values)) if values.size else None,
                "rmse_max_mV": float(np.max(values)) if values.size else None,
                **counts,
                **bins,
            }
        )

    all_values = np.asarray([_numeric_rmse(row) for row in fit_rows], dtype=np.float64)
    all_values = all_values[np.isfinite(all_values)]
    threshold_summary = {
        f"count_rmse_le_{str(threshold).replace('.', '_')}mV": int(
            np.sum(all_values <= threshold)
        )
        for threshold in RMSE_THRESHOLDS_MV
    }
    bin_summary = {
        f"count_{name}": sum(predicate(float(value)) for value in all_values)
        for name, predicate in RMSE_BINS
    }
    overall = {
        "rpt_rows_returned_excluding_bol": len(
            [
                row
                for row in rows
                if str(row.get("checkup", "")).strip().casefold() != "bol"
            ]
        ),
        "numeric_rpt_fits": int(all_values.size),
        "rpt_rows_without_numeric_fit": (
            len(
                [
                    row
                    for row in rows
                    if str(row.get("checkup", "")).strip().casefold() != "bol"
                ]
            )
            - int(all_values.size)
        ),
        "rmse_mean_mV": float(np.mean(all_values)) if all_values.size else None,
        "rmse_median_mV": float(np.median(all_values)) if all_values.size else None,
        "rmse_min_mV": float(np.min(all_values)) if all_values.size else None,
        "rmse_max_mV": float(np.max(all_values)) if all_values.size else None,
        **threshold_summary,
        **bin_summary,
    }
    return by_cell, overall


def run_audit(
    dataset_root: Path,
    labels_dir: Path,
    output_dir: Path,
    max_tries: int,
    maxiter: int,
    popsize: int,
) -> dict[str, Any]:
    os_environment = __import__("os")
    os_environment.environ["PYTHONPATH"] = str(dataset_root.parent)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if not (labels_dir / "dataset_manifest.csv").is_file():
        raise FileNotFoundError("The source dataset manifest is missing.")
    output_dir.mkdir(parents=True, exist_ok=True)
    guide_entries = read_guide_entries(dataset_root / "Guide_to_Datafiles.xlsx")
    records, added_files = _collect_records(dataset_root, labels_dir, guide_entries)
    fit_options = {
        "max_tries": max_tries,
        "maxiter": maxiter,
        "popsize": popsize,
    }
    raw_rows: list[dict[str, Any]] = []
    cell_runtime: list[dict[str, Any]] = []
    started = time.perf_counter()
    for group, cell in CONFIRMED_CELLS:
        cell_started = time.perf_counter()
        print(f"Running raw PyDMA RMSE audit for G{group}-Cell{cell}", flush=True)
        rows = run_pydma_for_cell(
            dataset_root,
            records,
            guide_entries,
            group,
            cell,
            use_cuda=True,
            optimizer_options=fit_options,
            allow_raw_comparison=True,
        )
        raw_rows.extend(rows)
        cell_runtime.append(
            {
                "physical_cell_id": f"G{group}-Cell{cell}",
                "rows_returned": len(rows),
                "elapsed_s": time.perf_counter() - cell_started,
            }
        )
        _write_csv(output_dir / "pydma_raw_results.partial.csv", raw_rows)

    _write_csv(output_dir / "pydma_raw_results.csv", raw_rows)
    partial_path = output_dir / "pydma_raw_results.partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    by_cell, overall = _summarize(raw_rows)
    _write_csv(output_dir / "rmse_summary_by_cell.csv", by_cell)
    summary = {
        "experiment": "12-cell PyDMA raw RMSE audit without public-label RMSE gate",
        "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
        "cells": [f"G{group}-Cell{cell}" for group, cell in CONFIRMED_CELLS],
        "runtime": {
            "pydma_version": version("pydma"),
            "optimizer_backend": "PyTorch CUDA batched best1bin with GPU L-BFGS polish",
            "cuda_requested": True,
            "elapsed_s": time.perf_counter() - started,
        },
        "optimizer_options": fit_options,
        "optimizer_rmse_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
        "rmse_policy": (
            "No post-fit RMSE filter is applied to the audit counts; every numeric "
            "RPT fit RMSE is retained and counted. The optimizer's accepted-run "
            f"threshold is {PYDMA_RMSE_THRESHOLD_MV:g} mV. BoL reference fits are "
            "excluded from RPT fit totals but remain in pydma_raw_results.csv."
        ),
        "source_policy": {
            "labels_directory_read_only": True,
            "parts_2_3_read": False,
            "added_manifest_records": len(added_files),
            "added_files": added_files,
        },
        "overall": overall,
        "cell_runtime": cell_runtime,
    }
    _write_json(output_dir / "rmse_audit_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit all Guide-confirmed Part 1 cells and summarize raw PyDMA RMSE with the configured optimizer gate."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tries", type=int, default=30)
    parser.add_argument("--maxiter", type=int, default=50)
    parser.add_argument("--popsize", type=int, default=30)
    args = parser.parse_args()
    if min(args.max_tries, args.maxiter, args.popsize) < 1:
        parser.error("Optimizer options must be positive.")
    summary = run_audit(
        args.dataset.resolve(),
        args.labels.resolve(),
        args.output.resolve(),
        args.max_tries,
        args.maxiter,
        args.popsize,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
