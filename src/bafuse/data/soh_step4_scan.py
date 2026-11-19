from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_FILE_FIELDS = [
    "relative_path",
    "source_size_bytes",
    "source_modified_ns",
    "expected_source_size_bytes",
    "source_mapping_rows",
    "dataset_parts",
    "physical_cells",
    "measurement_ids",
    "csv_status",
    "rows_scanned",
    "step4_rows",
    "step4_segment_count",
    "step4_capacity_finite_rows",
    "step4_capacity_min",
    "step4_capacity_max",
    "step4_capacity_span",
    "step4_current_signs",
    "step4_positive_capacity_deltas",
    "step4_negative_capacity_deltas",
    "step4_zero_capacity_deltas",
    "step4_segments_json",
    "scan_status",
    "scan_error",
]


def _update_segment_capacity(
    segment: dict[str, Any],
    capacity: np.ndarray,
    current: np.ndarray | None,
    source_row_start: int,
    source_row_end: int,
) -> None:
    if segment["source_row_start"] is None:
        segment["source_row_start"] = source_row_start
    segment["source_row_end"] = source_row_end
    segment["step4_rows"] += int(len(capacity))
    finite = np.isfinite(capacity)
    finite_values = capacity[finite]
    if finite_values.size:
        segment["finite_rows"] += int(finite_values.size)
        minimum = float(finite_values.min())
        maximum = float(finite_values.max())
        segment["capacity_min"] = (
            minimum
            if segment["capacity_min"] is None
            else min(segment["capacity_min"], minimum)
        )
        segment["capacity_max"] = (
            maximum
            if segment["capacity_max"] is None
            else max(segment["capacity_max"], maximum)
        )

        if segment["last_capacity"] is not None and finite[0]:
            delta = float(finite_values[0]) - float(segment["last_capacity"])
            if delta > 0:
                segment["positive_deltas"] += 1
            elif delta < 0:
                segment["negative_deltas"] += 1
            else:
                segment["zero_deltas"] += 1

        adjacent = finite[1:] & finite[:-1]
        deltas = np.diff(capacity)[adjacent]
        segment["positive_deltas"] += int(np.count_nonzero(deltas > 0))
        segment["negative_deltas"] += int(np.count_nonzero(deltas < 0))
        segment["zero_deltas"] += int(np.count_nonzero(deltas == 0))

    segment["last_capacity"] = (
        float(capacity[-1]) if len(capacity) and finite[-1] else None
    )

    if current is not None:
        finite_current = current[np.isfinite(current)]
        if finite_current.size:
            if np.any(finite_current < 0):
                segment["current_signs"].add("negative")
            if np.any(finite_current > 0):
                segment["current_signs"].add("positive")
            if np.any(finite_current == 0):
                segment["current_signs"].add("zero")


def _new_segment(index: int) -> dict[str, Any]:
    return {
        "segment_index": index,
        "source_row_start": None,
        "source_row_end": None,
        "step4_rows": 0,
        "finite_rows": 0,
        "capacity_min": None,
        "capacity_max": None,
        "positive_deltas": 0,
        "negative_deltas": 0,
        "zero_deltas": 0,
        "current_signs": set(),
        "last_capacity": None,
    }


def _finish_segment(segment: dict[str, Any]) -> dict[str, Any]:
    capacity_min = segment["capacity_min"]
    capacity_max = segment["capacity_max"]
    return {
        "segment_index": segment["segment_index"],
        "source_row_start": segment["source_row_start"],
        "source_row_end": segment["source_row_end"],
        "step4_rows": segment["step4_rows"],
        "capacity_finite_rows": segment["finite_rows"],
        "capacity_min": capacity_min,
        "capacity_max": capacity_max,
        "capacity_span": (
            capacity_max - capacity_min
            if capacity_min is not None and capacity_max is not None
            else None
        ),
        "current_signs": sorted(segment["current_signs"]),
        "positive_capacity_deltas": segment["positive_deltas"],
        "negative_capacity_deltas": segment["negative_deltas"],
        "zero_capacity_deltas": segment["zero_deltas"],
    }


def scan_step4_file(path: Path, chunk_rows: int = 100_000) -> dict[str, Any]:
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("CSV chunk size must be a positive integer.")
    path = path.resolve()
    stat = path.stat()
    result: dict[str, Any] = {
        "relative_path": "",
        "source_size_bytes": stat.st_size,
        "source_modified_ns": stat.st_mtime_ns,
        "source_mapping_rows": 0,
        "expected_source_size_bytes": "",
        "dataset_parts": "",
        "physical_cells": "",
        "measurement_ids": "",
        "csv_status": "pending",
        "rows_scanned": 0,
        "step4_rows": 0,
        "step4_segment_count": 0,
        "step4_capacity_finite_rows": 0,
        "step4_capacity_min": None,
        "step4_capacity_max": None,
        "step4_capacity_span": None,
        "step4_current_signs": "",
        "step4_positive_capacity_deltas": 0,
        "step4_negative_capacity_deltas": 0,
        "step4_zero_capacity_deltas": 0,
        "step4_segments_json": "[]",
        "scan_status": "pending",
        "scan_error": "",
    }

    try:
        columns = list(pd.read_csv(path, nrows=0).columns)
        missing_columns = {"Step", "Amphr"}.difference(columns)
        if missing_columns:
            result["csv_status"] = "missing_required_columns"
            result["scan_error"] = (
                "Missing required column(s): " + ", ".join(sorted(missing_columns))
            )
            result["scan_status"] = "not_scanned"
            return result

        selected_columns = ["Step", "Amphr"]
        if "Amps" in columns:
            selected_columns.append("Amps")
        result["csv_status"] = "columns_available"
        result["scan_status"] = "scanning"
        segments: list[dict[str, Any]] = []
        open_segment: dict[str, Any] | None = None

        for chunk in pd.read_csv(
            path,
            usecols=selected_columns,
            chunksize=chunk_rows,
            low_memory=False,
        ):
            chunk_row_offset = result["rows_scanned"]
            result["rows_scanned"] += len(chunk)
            step_values = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy()
            step4_mask = step_values == 4
            capacity_values = (
                pd.to_numeric(chunk["Amphr"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=float)
            )
            current_values = (
                pd.to_numeric(chunk["Amps"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=float)
                if "Amps" in chunk
                else None
            )

            padded = np.concatenate(([False], step4_mask, [False]))
            changes = np.diff(padded.astype(np.int8))
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1)
            if not len(starts):
                if open_segment is not None:
                    segments.append(_finish_segment(open_segment))
                    open_segment = None
                continue

            if open_segment is not None and starts[0] != 0:
                segments.append(_finish_segment(open_segment))
                open_segment = None

            for start, end in zip(starts, ends):
                if start == 0 and open_segment is not None:
                    segment = open_segment
                else:
                    if open_segment is not None:
                        segments.append(_finish_segment(open_segment))
                    segment = _new_segment(len(segments) + 1)

                _update_segment_capacity(
                    segment,
                    capacity_values[start:end],
                    current_values[start:end] if current_values is not None else None,
                    chunk_row_offset + int(start) + 2,
                    chunk_row_offset + int(end) + 1,
                )
                reaches_chunk_end = end == len(step4_mask)
                if reaches_chunk_end:
                    open_segment = segment
                else:
                    segments.append(_finish_segment(segment))
                    open_segment = None

        if open_segment is not None:
            segments.append(_finish_segment(open_segment))

        result["step4_segment_count"] = len(segments)
        result["step4_rows"] = sum(segment["step4_rows"] for segment in segments)
        result["step4_capacity_finite_rows"] = sum(
            segment["capacity_finite_rows"] for segment in segments
        )
        finite_segments = [
            segment
            for segment in segments
            if segment["capacity_min"] is not None
            and segment["capacity_max"] is not None
        ]
        if finite_segments:
            result["step4_capacity_min"] = min(
                segment["capacity_min"] for segment in finite_segments
            )
            result["step4_capacity_max"] = max(
                segment["capacity_max"] for segment in finite_segments
            )
            result["step4_capacity_span"] = (
                result["step4_capacity_max"] - result["step4_capacity_min"]
            )
        signs = sorted(
            {
                sign
                for segment in segments
                for sign in segment["current_signs"]
            }
        )
        result["step4_current_signs"] = ";".join(signs)
        result["step4_positive_capacity_deltas"] = sum(
            segment["positive_capacity_deltas"] for segment in segments
        )
        result["step4_negative_capacity_deltas"] = sum(
            segment["negative_capacity_deltas"] for segment in segments
        )
        result["step4_zero_capacity_deltas"] = sum(
            segment["zero_capacity_deltas"] for segment in segments
        )
        result["step4_segments_json"] = json.dumps(
            segments, ensure_ascii=False, allow_nan=False, sort_keys=True
        )
        result["scan_status"] = _classify_file_scan(result)
        return result
    except (
        OSError,
        UnicodeError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        ValueError,
    ) as error:
        result["scan_status"] = "scan_failed"
        result["scan_error"] = f"{type(error).__name__}: {error}"
        return result


def _classify_file_scan(result: dict[str, Any]) -> str:
    if result["step4_rows"] == 0:
        return "no_step4_rows"
    if result["step4_segment_count"] != 1:
        return "multiple_step4_segments"
    if result["step4_capacity_finite_rows"] < 10:
        return "too_few_step4_capacity_rows"
    span = result["step4_capacity_span"]
    if span is None or not math.isfinite(span) or span <= 0:
        return "invalid_step4_capacity_span"
    signs = set(filter(None, result["step4_current_signs"].split(";")))
    if {"negative", "positive"}.issubset(signs):
        return "mixed_step4_current_signs"
    return "one_contiguous_step4_candidate"


def _source_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def build_step4_scan(
    manifest_path: Path,
    dataset_roots: dict[int, Path],
    chunk_rows: int = 100_000,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Guide observation manifest not found: {manifest_path}")
    source_manifest_path = manifest_path.with_name("summary.json")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Guide manifest summary not found: {source_manifest_path}")
    source_summary = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    for part, root_value in dataset_roots.items():
        root = root_value.resolve()
        if str(root) != source_summary.get("dataset_roots", {}).get(str(part)):
            raise ValueError(
                f"Part {part} root differs from the root recorded in the Guide manifest."
            )
        guide_path = next(root.glob("Guide*.xlsx"), None)
        if guide_path is None:
            raise FileNotFoundError(f"Guide workbook is missing from {root}")
        actual_hash = hashlib.sha256(guide_path.read_bytes()).hexdigest()
        if actual_hash != source_summary.get("guide_sha256", {}).get(str(part)):
            raise ValueError(f"Guide workbook changed since manifest creation: {guide_path}")

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        observations = list(csv.DictReader(stream))
    if not observations:
        raise ValueError(f"Guide observation manifest contains no rows: {manifest_path}")

    path_to_observations: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        if row.get("measurement_mapping_status") != "single_checkpoint":
            continue
        for relative_path in _source_paths(row.get("source_file", "")):
            path_to_observations[
                f"{row['dataset_part']}:{relative_path}"
            ].append(row)

    file_results: list[dict[str, Any]] = []
    path_results: dict[str, dict[str, Any]] = {}
    total_files = len(path_to_observations)
    for index, (key, mapped_observations) in enumerate(
        sorted(path_to_observations.items()), 1
    ):
        part_text, relative_path = key.split(":", 1)
        part = int(part_text)
        root = dataset_roots.get(part)
        if root is None:
            raise ValueError(f"No dataset root was supplied for Part {part}.")
        resolved_root = root.resolve()
        source_path = (resolved_root / Path(relative_path)).resolve()
        try:
            source_path.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(f"Source path escapes the Part {part} dataset root: {relative_path}") from error
        if not source_path.is_file():
            scan_result = {
                "relative_path": relative_path,
                "source_size_bytes": "",
                "source_modified_ns": "",
                "source_mapping_rows": len(mapped_observations),
                "expected_source_size_bytes": "",
                "dataset_parts": str(part),
                "physical_cells": ";".join(
                    sorted({row["physical_cell_id"] for row in mapped_observations})
                ),
                "measurement_ids": ";".join(
                    sorted({row["measurement_id"] for row in mapped_observations})
                ),
                "csv_status": "missing_source_file",
                "rows_scanned": 0,
                "step4_rows": 0,
                "step4_segment_count": 0,
                "step4_capacity_finite_rows": 0,
                "step4_capacity_min": None,
                "step4_capacity_max": None,
                "step4_capacity_span": None,
                "step4_current_signs": "",
                "step4_positive_capacity_deltas": 0,
                "step4_negative_capacity_deltas": 0,
                "step4_zero_capacity_deltas": 0,
                "step4_segments_json": "[]",
                "scan_status": "source_file_missing",
                "scan_error": "Mapped source file does not exist.",
            }
        else:
            expected_sizes = {
                int(row["source_file_size_bytes"])
                for row in mapped_observations
                if row.get("source_mapping_status") == "unique_source_file"
                and row.get("source_file_count") == "1"
                and row.get("source_file") == relative_path
                and row.get("source_file_size_bytes", "").strip()
            }
            if len(expected_sizes) > 1:
                raise ValueError(
                    f"Conflicting source sizes in manifest for {relative_path}"
                )
            source_size = source_path.stat().st_size
            expected_size = next(iter(expected_sizes), None)
            if expected_size is not None and source_size != expected_size:
                scan_result = {
                    "relative_path": relative_path,
                    "source_size_bytes": source_size,
                    "source_modified_ns": source_path.stat().st_mtime_ns,
                    "source_mapping_rows": len(mapped_observations),
                    "expected_source_size_bytes": expected_size,
                    "dataset_parts": str(part),
                    "physical_cells": ";".join(
                        sorted({row["physical_cell_id"] for row in mapped_observations})
                    ),
                    "measurement_ids": ";".join(
                        sorted({row["measurement_id"] for row in mapped_observations})
                    ),
                    "csv_status": "source_size_mismatch",
                    "rows_scanned": 0,
                    "step4_rows": 0,
                    "step4_segment_count": 0,
                    "step4_capacity_finite_rows": 0,
                    "step4_capacity_min": None,
                    "step4_capacity_max": None,
                    "step4_capacity_span": None,
                    "step4_current_signs": "",
                    "step4_positive_capacity_deltas": 0,
                    "step4_negative_capacity_deltas": 0,
                    "step4_zero_capacity_deltas": 0,
                    "step4_segments_json": "[]",
                    "scan_status": "source_size_mismatch",
                    "scan_error": (
                        f"Expected {expected_size} bytes from Guide manifest; "
                        f"found {source_size}."
                    ),
                }
            else:
                scan_result = scan_step4_file(source_path, chunk_rows)
                scan_result["relative_path"] = relative_path
                scan_result["source_mapping_rows"] = len(mapped_observations)
                scan_result["expected_source_size_bytes"] = expected_size or ""
                scan_result["dataset_parts"] = str(part)
                scan_result["physical_cells"] = ";".join(
                    sorted({row["physical_cell_id"] for row in mapped_observations})
                )
                scan_result["measurement_ids"] = ";".join(
                    sorted({row["measurement_id"] for row in mapped_observations})
                )
        file_results.append(scan_result)
        path_results[key] = scan_result
        print(
            f"Scanned {index}/{total_files}: Part {part} {relative_path} "
            f"-> {scan_result['scan_status']}",
            flush=True,
        )

    updated_observations: list[dict[str, Any]] = []
    for row in observations:
        updated = dict(row)
        candidates = [
            path_results[f"{row['dataset_part']}:{relative_path}"]
            for relative_path in _source_paths(row.get("source_file", ""))
            if f"{row['dataset_part']}:{relative_path}" in path_results
        ]
        updated["step4_candidate_source_file"] = ""
        if row.get("measurement_mapping_status") != "single_checkpoint":
            updated["step4_boundary_status"] = "blocked_compound_guide_stage"
        elif not candidates:
            updated["step4_boundary_status"] = "no_mapped_source_file"
        else:
            eligible = [
                candidate
                for candidate in candidates
                if candidate["scan_status"] == "one_contiguous_step4_candidate"
            ]
            if len(eligible) == 1:
                updated["step4_candidate_source_file"] = eligible[0]["relative_path"]
                updated["step4_boundary_status"] = (
                    "one_contiguous_step4_candidate"
                    if len(candidates) == 1
                    else "unique_source_among_multiple_candidates"
                )
            elif len(eligible) > 1:
                updated["step4_boundary_status"] = "multiple_step4_source_candidates"
            else:
                statuses = sorted({candidate["scan_status"] for candidate in candidates})
                updated["step4_boundary_status"] = ";".join(statuses)
        updated_observations.append(updated)

    summary = {
        "manifest": "Guide-level SOH observation candidates with targeted Step 4 scan",
        "source_guide_manifest": str(manifest_path),
        "source_guide_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "dataset_roots": {
            str(part): str(root.resolve())
            for part, root in sorted(dataset_roots.items())
        },
        "guide_sha256": source_summary.get("guide_sha256", {}),
        "candidate_rows": len(updated_observations),
        "distinct_source_files_scanned": len(file_results),
        "source_file_scan_status_counts": dict(
            Counter(row["scan_status"] for row in file_results)
        ),
        "step4_segment_count_distribution": dict(
            Counter(str(row["step4_segment_count"]) for row in file_results)
        ),
        "observation_boundary_status_counts": dict(
            Counter(row["step4_boundary_status"] for row in updated_observations)
        ),
        "scan_chunk_rows": chunk_rows,
        "source_file_results": [
            {
                "dataset_part": row["dataset_parts"],
                "relative_path": row["relative_path"],
                "source_size_bytes": row["source_size_bytes"],
                "source_modified_ns": row["source_modified_ns"],
                "scan_status": row["scan_status"],
            }
            for row in file_results
        ],
        "limitations": [
            "Step 4 contiguity is based on observed Step values; it does not prove that the Guide-to-procedure mapping or chemistry interpretation is correct.",
            "A one-segment candidate is a QC screening result, not an accepted SOH label.",
            "Compound Guide stages remain unassigned to a single RPT and were not scanned for label assignment.",
            "Capacity unit remains the native Amphr column unit, not verified Ah.",
            "Nominal cycle age remains an ordinal proxy and is not reconstructed exposure.",
        ],
    }
    return file_results, updated_observations, summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_step4_scan(
    manifest_path: Path,
    dataset_roots: dict[int, Path],
    output_dir: Path,
    chunk_rows: int = 100_000,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment output directory is not empty: {output_dir}")
    file_results, observations, summary = build_step4_scan(
        manifest_path, dataset_roots, chunk_rows
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "source_file_step4_scan.csv",
        file_results,
        _FILE_FIELDS,
    )
    observation_fields = list(
        dict.fromkeys(key for row in observations for key in row)
    )
    _write_csv(
        output_dir / "guide_observation_manifest_step4.csv",
        observations,
        observation_fields,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream only Step/Amphr/Amps to screen mapped RPT files for Step 4 boundaries."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--part-1-root", type=Path, required=True)
    parser.add_argument("--part-2-root", type=Path, required=True)
    parser.add_argument("--part-3-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    summary = write_step4_scan(
        args.manifest,
        {
            1: args.part_1_root,
            2: args.part_2_root,
            3: args.part_3_root,
        },
        args.output,
        args.chunk_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = [
        status
        for status in summary["source_file_scan_status_counts"]
        if status
        in {
            "scan_failed",
            "source_file_missing",
            "source_size_mismatch",
            "not_scanned",
        }
    ]
    if failures:
        raise RuntimeError(
            "Step 4 screening finished with file errors; see source_file_step4_scan.csv."
        )


if __name__ == "__main__":
    main()
