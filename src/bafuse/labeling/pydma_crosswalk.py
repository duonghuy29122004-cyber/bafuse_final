from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from bafuse.labeling.audit import FileRecord, GuideEntry


DIRECTIONS = ("charge", "discharge")


def is_unique_checkpoint(row: dict[str, str]) -> bool:
    return (
        row.get("measurement_mapping_status") == "single_checkpoint"
        and row.get("source_mapping_status") == "unique_source_file"
        and row.get("source_file_count") == "1"
        and bool(row.get("source_file"))
    )


def source_order(row: dict[str, str]) -> tuple[int, str]:
    try:
        file_id = int(row["procedure_file_id"])
    except (KeyError, ValueError):
        file_id = 2**31 - 1
    return file_id, row.get("source_file", "").casefold()


def manifest_path(row: dict[str, str], roots: dict[str, Path]) -> Path:
    root = roots[row["dataset_part"]].resolve()
    relative_path = Path(row["source_file"])
    if relative_path.is_absolute():
        raise ValueError(f"Expected a relative manifest source path: {relative_path}")
    path = (root / relative_path).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Manifest source escapes its dataset root: {relative_path}") from error
    if not path.is_file():
        raise FileNotFoundError(f"Manifest source file is missing: {path}")
    expected_size = int(row["source_file_size_bytes"])
    if path.stat().st_size != expected_size:
        raise ValueError(
            f"Manifest source size changed for {path}: "
            f"expected {expected_size}, found {path.stat().st_size}."
        )
    return path


def cell_inputs(
    plan: dict[str, Any],
    roots: dict[str, Path],
    directions: tuple[str, ...] = DIRECTIONS,
) -> tuple[list[FileRecord], list[GuideEntry], set[tuple[int, str]]]:
    rows = [plan["baseline"], *plan["targets"]]
    unique_rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in rows:
        key = (
            row["dataset_part"],
            row["procedure_file_id"],
            row["source_file"],
        )
        unique_rows.setdefault(key, row)

    records: list[FileRecord] = []
    entries: list[GuideEntry] = []
    synthetic_ids: dict[tuple[str, str, str], int] = {}
    ordered_rows = sorted(
        unique_rows.items(),
        key=lambda item: (item[1]["dataset_part"], source_order(item[1])),
    )
    for synthetic_id, (key, row) in enumerate(ordered_rows, start=1):
        synthetic_ids[key] = synthetic_id
        source_path = manifest_path(row, roots)
        records.append(
            FileRecord(
                relative_path=str(source_path),
                file_type="csv",
                size_bytes=source_path.stat().st_size,
                modified_ns=source_path.stat().st_mtime_ns,
                format="CSV",
                decode_status="manifest_crosswalk",
                group_id=row["group_id"],
                cell_id=row["cell_id"],
                physical_cell_id=row["physical_cell_id"],
                file_id=synthetic_id,
                procedure=row["procedure"],
                stage=row["guide_stage"],
                modality="cycling_and_RPT",
            )
        )
        entries.append(
            GuideEntry(
                group_id=row["group_id"],
                cell_id=row["cell_id"],
                procedure=row["procedure"],
                stage=row["guide_stage"],
                file_id=synthetic_id,
            )
        )

    fit_targets = {
        (
            synthetic_ids[
                (
                    row["dataset_part"],
                    row["procedure_file_id"],
                    row["source_file"],
                )
            ],
            direction,
        )
        for row in plan["targets"]
        for direction in directions
    }
    return records, entries, fit_targets


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
