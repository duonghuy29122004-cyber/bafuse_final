from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_COMPARISON_FIELDS = [
    "physical_cell_id",
    "first_measurement_id",
    "first_procedure",
    "first_procedure_file_id",
    "first_source_file",
    "first_source_candidate_count",
    "first_segment_index",
    "first_segment_rows",
    "first_capacity_span",
    "second_measurement_id",
    "second_procedure",
    "second_procedure_file_id",
    "second_source_file",
    "second_source_candidate_count",
    "second_segment_index",
    "second_segment_rows",
    "second_capacity_span",
    "capacity_row_counts_match",
    "step_exact_match",
    "amphr_exact_match",
    "capacity_match_atol_1e-12",
    "amps_exact_match",
    "voltage_column",
    "voltage_exact_match",
    "cyc_exact_match",
    "core_signals_match_except_time",
    "testtime_constant_offset",
    "testtime_offset_native_units",
    "testtime_offset_range_native_units",
    "capacity_max_abs_difference",
    "comparison_status",
    "interpretation",
]


def _read_step4_segments(
    path: Path, chunk_rows: int
) -> list[dict[str, np.ndarray]]:
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError("CSV chunk size must be a positive integer.")

    completed: list[dict[str, list[np.ndarray]]] = []
    open_segment: dict[str, list[np.ndarray]] | None = None
    columns = list(pd.read_csv(path, nrows=0).columns)
    if not {"Step", "Amphr"}.issubset(columns):
        raise ValueError(f"CSV is missing Step or Amphr columns: {path}")
    selected_columns = [
        column
        for column in ("Step", "Amphr", "Amps", "Volts", "Voltage", "Cyc", "TestTime")
        if column in columns
    ]

    for chunk in pd.read_csv(
        path,
        usecols=selected_columns,
        chunksize=chunk_rows,
        low_memory=False,
    ):
        values = {
            column: (
                pd.to_numeric(chunk[column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .to_numpy(dtype=float)
            )
            for column in selected_columns
        }
        step = values["Step"]
        mask = step == 4
        padded = np.concatenate(([False], mask, [False]))
        changes = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)

        if not len(starts):
            if open_segment is not None:
                completed.append(open_segment)
                open_segment = None
            continue

        if open_segment is not None and starts[0] != 0:
            completed.append(open_segment)
            open_segment = None

        for start, end in zip(starts, ends):
            if start == 0 and open_segment is not None:
                segment = open_segment
            else:
                if open_segment is not None:
                    completed.append(open_segment)
                segment = {}
            for column in selected_columns:
                segment.setdefault(column, []).append(values[column][start:end])
            if end == len(mask):
                open_segment = segment
            else:
                completed.append(segment)
                open_segment = None

    if open_segment is not None:
        completed.append(open_segment)
    return [
        {
            column: (
                np.concatenate(chunks)
                if chunks
                else np.array([], dtype=float)
            )
            for column, chunks in segment.items()
        }
        for segment in completed
    ]


def _source_paths(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def _capacity_span(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.max() - finite.min()) if finite.size else None


def _candidate_sources(
    observation: dict[str, str],
    scan_by_path: dict[tuple[str, str], dict[str, str]],
) -> list[tuple[str, dict[str, str], list[dict[str, Any]]]]:
    candidates = []
    for relative_path in _source_paths(observation.get("source_file", "")):
        file_scan = scan_by_path.get((observation["dataset_part"], relative_path))
        if file_scan is None:
            continue
        segments = json.loads(file_scan.get("step4_segments_json", "[]"))
        if segments:
            candidates.append((relative_path, file_scan, segments))
    return candidates


def _empty_comparison(
    first: dict[str, str],
    second: dict[str, str],
    first_candidates: list[tuple[str, dict[str, str], list[dict[str, Any]]]],
    second_candidates: list[tuple[str, dict[str, str], list[dict[str, Any]]]],
    status: str,
) -> dict[str, Any]:
    return {
        "physical_cell_id": first["physical_cell_id"],
        "first_measurement_id": first["measurement_id"],
        "first_procedure": first["procedure"],
        "first_procedure_file_id": first["procedure_file_id"],
        "first_source_file": ";".join(item[0] for item in first_candidates),
        "first_source_candidate_count": len(first_candidates),
        "first_segment_index": "",
        "first_segment_rows": "",
        "first_capacity_span": "",
        "second_measurement_id": second["measurement_id"],
        "second_procedure": second["procedure"],
        "second_procedure_file_id": second["procedure_file_id"],
        "second_source_file": ";".join(item[0] for item in second_candidates),
        "second_source_candidate_count": len(second_candidates),
        "second_segment_index": "",
        "second_segment_rows": "",
        "second_capacity_span": "",
        "capacity_row_counts_match": "",
        "step_exact_match": "",
        "amphr_exact_match": "",
        "capacity_match_atol_1e-12": "",
        "amps_exact_match": "",
        "voltage_column": "",
        "voltage_exact_match": "",
        "cyc_exact_match": "",
        "core_signals_match_except_time": "",
        "testtime_constant_offset": "",
        "testtime_offset_native_units": "",
        "testtime_offset_range_native_units": "",
        "capacity_max_abs_difference": "",
        "comparison_status": status,
        "interpretation": (
            "No curve was selected; this row records a blocked comparison only."
        ),
    }


def build_part1_bol_comparison(
    observation_manifest: Path,
    source_scan: Path,
    dataset_root: Path,
    chunk_rows: int = 100_000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observations: list[dict[str, str]]
    with observation_manifest.open(encoding="utf-8-sig", newline="") as stream:
        observations = list(csv.DictReader(stream))
    with source_scan.open(encoding="utf-8-sig", newline="") as stream:
        scan_rows = list(csv.DictReader(stream))

    scan_by_path = {
        (row["dataset_parts"], row["relative_path"]): row for row in scan_rows
    }
    part1_bol_by_cell: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in observations:
        if row.get("dataset_part") == "1" and row.get("measurement_id", "").casefold() == "bol":
            part1_bol_by_cell[row["physical_cell_id"]].append(row)

    arrays_by_path: dict[str, list[dict[str, np.ndarray]]] = {}
    comparisons: list[dict[str, Any]] = []
    paired_observations = 0
    unique_source_pairs = 0

    for cell_id, bol_rows in sorted(part1_bol_by_cell.items()):
        bol_rows.sort(key=lambda row: int(row["procedure_file_id"]))
        for first, second in zip(bol_rows, bol_rows[1:]):
            paired_observations += 1
            first_candidates = _candidate_sources(first, scan_by_path)
            second_candidates = _candidate_sources(second, scan_by_path)
            if len(first_candidates) != 1 or len(second_candidates) != 1:
                comparisons.append(
                    _empty_comparison(
                        first,
                        second,
                        first_candidates,
                        second_candidates,
                        "blocked_non_unique_step4_source",
                    )
                )
                continue

            unique_source_pairs += 1
            first_path, first_scan, first_segment_summaries = first_candidates[0]
            second_path, second_scan, second_segment_summaries = second_candidates[0]
            for relative_path, file_scan in (
                (first_path, first_scan),
                (second_path, second_scan),
            ):
                root = dataset_root.resolve()
                source_path = (root / Path(relative_path)).resolve()
                try:
                    source_path.relative_to(root)
                except ValueError as error:
                    raise ValueError(
                        f"Source path escapes the Part 1 dataset root: {relative_path}"
                    ) from error
                if not source_path.is_file():
                    raise FileNotFoundError(f"Mapped source file is missing: {source_path}")
                stat = source_path.stat()
                if (
                    int(file_scan["source_size_bytes"]) != stat.st_size
                    or int(file_scan["source_modified_ns"]) != stat.st_mtime_ns
                ):
                    raise ValueError(
                        f"Source file changed since Step 4 scan: {source_path}"
                    )
                if relative_path not in arrays_by_path:
                    arrays_by_path[relative_path] = _read_step4_segments(
                        source_path, chunk_rows
                    )

            first_segments = arrays_by_path[first_path]
            second_segments = arrays_by_path[second_path]
            if len(first_segments) != len(first_segment_summaries):
                raise ValueError(
                    f"Step 4 segment count changed since scan: {first_path}"
                )
            if len(second_segments) != len(second_segment_summaries):
                raise ValueError(
                    f"Step 4 segment count changed since scan: {second_path}"
                )

            for first_index, first_curve in enumerate(first_segments, 1):
                for second_index, second_curve in enumerate(second_segments, 1):
                    first_capacity = first_curve["Amphr"]
                    second_capacity = second_curve["Amphr"]
                    same_length = len(first_capacity) == len(second_capacity)
                    step_match = (
                        same_length
                        and np.array_equal(
                            first_curve["Step"],
                            second_curve["Step"],
                            equal_nan=True,
                        )
                    )
                    exact_capacity_match = bool(
                        same_length
                        and np.array_equal(first_capacity, second_capacity, equal_nan=True)
                    )
                    close_match = bool(
                        same_length
                        and np.allclose(
                            first_capacity,
                            second_capacity,
                            rtol=0,
                            atol=1e-12,
                            equal_nan=True,
                        )
                    )
                    current_match = (
                        bool(
                            same_length
                            and np.array_equal(
                                first_curve["Amps"],
                                second_curve["Amps"],
                                equal_nan=True,
                            )
                        )
                        if "Amps" in first_curve and "Amps" in second_curve
                        else None
                    )
                    voltage_column = next(
                        (
                            name
                            for name in ("Volts", "Voltage")
                            if name in first_curve and name in second_curve
                        ),
                        "",
                    )
                    voltage_match = (
                        bool(
                            same_length
                            and np.array_equal(
                                first_curve[voltage_column],
                                second_curve[voltage_column],
                                equal_nan=True,
                            )
                        )
                        if voltage_column
                        else None
                    )
                    cyc_match = (
                        bool(
                            same_length
                            and np.array_equal(
                                first_curve["Cyc"],
                                second_curve["Cyc"],
                                equal_nan=True,
                            )
                        )
                        if "Cyc" in first_curve and "Cyc" in second_curve
                        else None
                    )
                    core_signal_matches = [
                        step_match,
                        exact_capacity_match,
                        *(
                            [current_match]
                            if current_match is not None
                            else []
                        ),
                        *(
                            [voltage_match]
                            if voltage_match is not None
                            else []
                        ),
                        *([cyc_match] if cyc_match is not None else []),
                    ]
                    testtime_offsets = None
                    testtime_constant_offset = None
                    if (
                        same_length
                        and "TestTime" in first_curve
                        and "TestTime" in second_curve
                    ):
                        first_time = first_curve["TestTime"]
                        second_time = second_curve["TestTime"]
                        finite_times = np.isfinite(first_time) & np.isfinite(second_time)
                        if finite_times.any():
                            testtime_offsets = (
                                first_time[finite_times] - second_time[finite_times]
                            )
                            testtime_constant_offset = bool(
                                np.allclose(
                                    testtime_offsets,
                                    testtime_offsets[0],
                                    rtol=0,
                                    atol=1e-9,
                                )
                            )
                    finite_pair = (
                        np.isfinite(first_capacity) & np.isfinite(second_capacity)
                        if same_length
                        else np.array([], dtype=bool)
                    )
                    max_difference = (
                        float(
                            np.max(
                                np.abs(
                                    first_capacity[finite_pair]
                                    - second_capacity[finite_pair]
                                )
                            )
                        )
                        if finite_pair.any()
                        else None
                    )
                    comparisons.append(
                        {
                            "physical_cell_id": cell_id,
                            "first_measurement_id": first["measurement_id"],
                            "first_procedure": first["procedure"],
                            "first_procedure_file_id": first["procedure_file_id"],
                            "first_source_file": first_path,
                            "first_source_candidate_count": len(first_candidates),
                            "first_segment_index": first_index,
                            "first_segment_rows": len(first_capacity),
                            "first_capacity_span": _capacity_span(first_capacity),
                            "second_measurement_id": second["measurement_id"],
                            "second_procedure": second["procedure"],
                            "second_procedure_file_id": second["procedure_file_id"],
                            "second_source_file": second_path,
                            "second_source_candidate_count": len(second_candidates),
                            "second_segment_index": second_index,
                            "second_segment_rows": len(second_capacity),
                            "second_capacity_span": _capacity_span(second_capacity),
                            "capacity_row_counts_match": same_length,
                            "step_exact_match": step_match,
                            "amphr_exact_match": exact_capacity_match,
                            "capacity_match_atol_1e-12": close_match,
                            "amps_exact_match": (
                                current_match if current_match is not None else ""
                            ),
                            "voltage_column": voltage_column,
                            "voltage_exact_match": (
                                voltage_match if voltage_match is not None else ""
                            ),
                            "cyc_exact_match": cyc_match if cyc_match is not None else "",
                            "core_signals_match_except_time": (
                                all(core_signal_matches) if core_signal_matches else False
                            ),
                            "testtime_constant_offset": (
                                testtime_constant_offset
                                if testtime_constant_offset is not None
                                else ""
                            ),
                            "testtime_offset_native_units": (
                                float(np.median(testtime_offsets))
                                if testtime_offsets is not None
                                else ""
                            ),
                            "testtime_offset_range_native_units": (
                                float(np.ptp(testtime_offsets))
                                if testtime_offsets is not None
                                else ""
                            ),
                            "capacity_max_abs_difference": max_difference,
                            "comparison_status": (
                                "capacity_sequences_match_within_tolerance"
                                if close_match
                                else "capacity_sequences_differ"
                            ),
                            "interpretation": (
                                "Core Step 4 signals are compared where available; "
                                "similarity does not establish measurement identity "
                                "or select a baseline."
                            ),
                        }
                    )

    summary = {
        "review": "Part 1 repeated BoL Step 4 capacity-curve comparison",
        "observation_manifest": str(observation_manifest.resolve()),
        "observation_manifest_sha256": hashlib.sha256(
            observation_manifest.read_bytes()
        ).hexdigest(),
        "source_scan": str(source_scan.resolve()),
        "source_scan_sha256": hashlib.sha256(source_scan.read_bytes()).hexdigest(),
        "part1_dataset_root": str(dataset_root.resolve()),
        "paired_consecutive_bol_entries": paired_observations,
        "pairs_with_one_step4_source_each": unique_source_pairs,
        "segment_pair_comparisons": len(comparisons),
        "capacity_tolerance": {"rtol": 0, "atol": 1e-12},
        "matching_capacity_sequences": sum(
            row["capacity_match_atol_1e-12"] is True for row in comparisons
        ),
        "matching_core_signal_sequences": sum(
            row["core_signals_match_except_time"] is True
            for row in comparisons
            if row["capacity_row_counts_match"] is True
        ),
        "blocked_pairs": sum(
            row["comparison_status"] == "blocked_non_unique_step4_source"
            for row in comparisons
        ),
        "decision": "review_only_no_baseline_selected",
        "limitations": [
            "Core signal comparison covers available Step, Amphr, Amps, Volts/Voltage, and Cyc fields; it does not establish Guide semantics or measurement identity.",
            "TestTime is compared only as a relative offset because it may reset between files.",
            "A matching sequence does not establish which BoL measurement should define the SOH denominator.",
            "This report does not create labels or alter historical outputs.",
        ],
    }
    return comparisons, summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_COMPARISON_FIELDS, extrasaction="raise"
        )
        writer.writeheader()
        writer.writerows(rows)


def write_part1_bol_comparison(scan_dir: Path, output_dir: Path, chunk_rows: int) -> dict[str, Any]:
    scan_dir = scan_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment output directory is not empty: {output_dir}")
    scan_summary_path = scan_dir / "summary.json"
    if not scan_summary_path.is_file():
        raise FileNotFoundError(f"Step 4 scan summary not found: {scan_summary_path}")
    scan_summary = json.loads(scan_summary_path.read_text(encoding="utf-8"))
    dataset_root_value = scan_summary.get("dataset_roots", {}).get("1")
    if not dataset_root_value:
        raise ValueError("Step 4 scan summary does not contain a Part 1 dataset root.")
    comparisons, summary = build_part1_bol_comparison(
        scan_dir / "guide_observation_manifest_step4.csv",
        scan_dir / "source_file_step4_scan.csv",
        Path(dataset_root_value),
        chunk_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "part1_bol_curve_comparison.csv", comparisons)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare successive Guide-mapped Part 1 BoL Step 4 capacity curves without selecting a denominator."
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    args = parser.parse_args()
    summary = write_part1_bol_comparison(
        args.scan_dir, args.output, args.chunk_rows
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
