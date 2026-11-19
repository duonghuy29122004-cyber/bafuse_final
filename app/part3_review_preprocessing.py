from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

from bafuse.hybrid.soh_fusion import resample_curve


ROOT = Path(__file__).resolve().parents[1]
RESOLUTION_PATH = ROOT / "app" / "part3_review_resolution.csv"
REVIEW_KEYS = {
    ("3", "7", "G7-Cell4", "1"),
    ("3", "7", "G7-Cell19", "1"),
    ("3", "7", "G7-Cell27", "1"),
    ("3", "8", "G8-Cell20", "1"),
    ("3", "10", "G10-Cell22", "2"),
    ("3", "10", "G10-Cell22", "4"),
    ("3", "10", "G10-Cell22", "11"),
    ("3", "10", "G10-Cell22", "12"),
    ("3", "10", "G10-Cell25", "11"),
    ("3", "10", "G10-Cell25", "12"),
}
STEP10_REPAIR_KEYS = {
    ("3", "10", "G10-Cell22", "12"),
    ("3", "10", "G10-Cell25", "12"),
}
REQUIRED_AUDIT_FIELDS = {
    "group",
    "cell",
    "rpt",
    "raw_source_path",
    "source_sha256",
    "new_data_qc_status",
    "evaluation_membership",
    "baseline_eligible",
    "two_modality_eligible",
    "three_modality_eligible",
    "step10_block_ranges",
    "step10_separator_rows",
}


class ReviewPreprocessingError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise ReviewPreprocessingError(f"Could not hash review source {path}: {error}") from error
    return digest.hexdigest()


def _identity(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        str(row.get("part", "3")).strip() or "3",
        str(int(row.get("group", ""))),
        str(row.get("cell", "")).strip(),
        str(int(row.get("rpt", ""))),
    )


def load_review_resolution_index(
    path: Path = RESOLUTION_PATH,
) -> dict[tuple[str, str, str, str], dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, strict=True)
            if not reader.fieldnames or not REQUIRED_AUDIT_FIELDS.issubset(reader.fieldnames):
                raise ReviewPreprocessingError(
                    f"Review-resolution audit is missing required fields: {path}"
                )
            rows = list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ReviewPreprocessingError(
            f"Could not read review-resolution audit {path}: {error}"
        ) from error

    index: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for row in rows:
        key = _identity(row)
        if key in index:
            raise ReviewPreprocessingError(
                f"Duplicate review-resolution identity: {key}"
            )
        index[key] = row
    if set(index) != REVIEW_KEYS:
        missing = sorted(REVIEW_KEYS - set(index))
        extra = sorted(set(index) - REVIEW_KEYS)
        raise ReviewPreprocessingError(
            f"Review-resolution identities differ from the approved ten: "
            f"missing={missing}; extra={extra}"
        )
    return index


def verify_review_source(source_path: Path, row: dict[str, str]) -> str:
    expected_path = (
        ROOT / Path(row["raw_source_path"].replace("/", "\\"))
    ).resolve()
    actual_path = source_path.resolve()
    if actual_path != expected_path:
        raise ReviewPreprocessingError(
            f"Review source path mismatch: expected {expected_path}, got {actual_path}"
        )
    actual_hash = _sha256(actual_path)
    if actual_hash != row["source_sha256"]:
        raise ReviewPreprocessingError(
            f"Review source SHA-256 mismatch for {actual_path}: {actual_hash}"
        )
    return actual_hash


def extract_rpt12_step10(
    source_path: Path,
    row: dict[str, str],
) -> dict[str, Any]:
    key = _identity(row)
    if key not in STEP10_REPAIR_KEYS:
        raise ReviewPreprocessingError(
            f"Step 10 review repair is not authorized for {key}"
        )
    verify_review_source(source_path, row)
    try:
        ranges = json.loads(row["step10_block_ranges"])
        separators = {int(value) for value in json.loads(row["step10_separator_rows"])}
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReviewPreprocessingError(
            f"Invalid Step 10 repair ranges for {key}: {error}"
        ) from error
    if (
        len(ranges) != 2
        or any(set(item) != {"start", "end"} for item in ranges)
        or any(int(item["start"]) > int(item["end"]) for item in ranges)
        or int(ranges[1]["start"]) <= int(ranges[0]["end"])
    ):
        raise ReviewPreprocessingError(f"Invalid Step 10 repair ranges for {key}")
    expected_separators = set(
        range(int(ranges[0]["end"]) + 1, int(ranges[1]["start"]))
    )
    if separators != expected_separators:
        raise ReviewPreprocessingError(
            f"Unmapped rows occur between Step 10 blocks for {key}"
        )

    block_ranges = [(int(item["start"]), int(item["end"])) for item in ranges]
    signal_rows: list[tuple[int, float, float, float, float, float]] = []
    separator_values: list[dict[str, Any]] = []
    segment_digest = hashlib.sha256()
    separator_digest = hashlib.sha256()
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if not header:
                raise ReviewPreprocessingError(
                    f"Step 10 repair source has no header: {source_path}"
                )
            columns: dict[str, int] = {}
            for column, name in enumerate(header):
                normalized = name.strip().casefold()
                if normalized in columns:
                    raise ReviewPreprocessingError(
                        f"Duplicate Step 10 source column: {name}"
                    )
                columns[normalized] = column
            required = {
                "step",
                "testtime",
                "steptime",
                "amphr",
                "amps",
                "volts",
            }
            if not required.issubset(columns):
                raise ReviewPreprocessingError(
                    "Step 10 source is missing required columns: "
                    + ", ".join(sorted(required - set(columns)))
                )
            bounds = [value for pair in block_ranges for value in pair]
            max_line = max((*bounds, *separators))
            seen_separators: set[int] = set()
            for values in reader:
                line = reader.line_num
                if line > max_line:
                    break
                if line in separators:
                    if len(values) != len(header):
                        raise ReviewPreprocessingError(
                            f"Separator row width mismatch at source line {line}"
                        )
                    nonmissing = []
                    for column, value in zip(header, values):
                        try:
                            number = float(value)
                            finite = math.isfinite(number)
                        except (TypeError, ValueError):
                            finite = False
                        if finite:
                            if not column.strip().casefold().startswith("varx"):
                                raise ReviewPreprocessingError(
                                    "Finite non-VAR measurement in Step 10 separator "
                                    f"at source line {line}: {column}={value}"
                                )
                            nonmissing.append((column, value))
                    separator_digest.update(
                        ("\x1f".join(values) + "\n").encode("utf-8", errors="replace")
                    )
                    separator_values.append(
                        {"line": line, "finite_var_fields": dict(nonmissing)}
                    )
                    seen_separators.add(line)
                    continue
                if not any(start <= line <= end for start, end in block_ranges):
                    continue
                if len(values) != len(header):
                    raise ReviewPreprocessingError(
                        f"Step 10 row width mismatch at source line {line}"
                    )
                parsed: dict[str, float] = {}
                for name in ("step", "testtime", "steptime", "amphr", "amps", "volts"):
                    try:
                        number = float(values[columns[name]])
                    except (ValueError, TypeError, IndexError) as error:
                        raise ReviewPreprocessingError(
                            f"Missing Step 10 {name} at source line {line}"
                        ) from error
                    if not math.isfinite(number):
                        raise ReviewPreprocessingError(
                            f"Non-finite Step 10 {name} at source line {line}"
                        )
                    parsed[name] = number
                if parsed["step"] != 10 or parsed["amps"] <= 0:
                    raise ReviewPreprocessingError(
                        f"Step 10 identity/current direction failed at source line {line}"
                    )
                signal_rows.append(
                    (
                        line,
                        parsed["testtime"],
                        parsed["steptime"],
                        parsed["amphr"],
                        parsed["amps"],
                        parsed["volts"],
                    )
                )
                segment_digest.update(
                    ("\x1f".join(values) + "\n").encode(
                        "utf-8", errors="replace"
                    )
                )
    except ReviewPreprocessingError:
        raise
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ReviewPreprocessingError(
            f"Could not parse Step 10 source {source_path}: {error}"
        ) from error

    if seen_separators != separators:
        raise ReviewPreprocessingError(
            f"Step 10 separator rows are missing for {key}: "
            f"expected={sorted(separators)}, seen={sorted(seen_separators)}"
        )
    expected_count = sum(end - start + 1 for start, end in block_ranges)
    if len(signal_rows) != expected_count:
        raise ReviewPreprocessingError(
            f"Step 10 row count mismatch for {key}: "
            f"expected {expected_count}, got {len(signal_rows)}"
        )
    times = [item[1] for item in signal_rows]
    step_times = [item[2] for item in signal_rows]
    capacity = [item[3] for item in signal_rows]
    voltage = [item[5] for item in signal_rows]
    time_deltas = [right - left for left, right in zip(times, times[1:])]
    step_time_deltas = [
        right - left for left, right in zip(step_times, step_times[1:])
    ]
    capacity_deltas = [
        right - left for left, right in zip(capacity, capacity[1:])
    ]
    if (
        any(delta <= 0 for delta in time_deltas)
        or any(delta <= 0 for delta in step_time_deltas)
        or any(delta <= 0 for delta in capacity_deltas)
    ):
        raise ReviewPreprocessingError(
            f"Step 10 time/capacity is not strictly increasing for {key}"
        )
    range_ends = {end for _, end in block_ranges}
    if any(
        right[0] != left[0] + 1
        for left, right in zip(signal_rows, signal_rows[1:])
        if left[0] not in range_ends
    ):
        raise ReviewPreprocessingError(
            f"Unexplained Step 10 row discontinuity for {key}"
        )
    within_deltas: list[float] = []
    for start, end in block_ranges:
        block = [item for item in signal_rows if start <= item[0] <= end]
        within_deltas.extend(
            right[1] - left[1] for left, right in zip(block, block[1:])
        )
    if len(block_ranges) == 2:
        first_end = next(item for item in signal_rows if item[0] == block_ranges[0][1])
        second_start = next(item for item in signal_rows if item[0] == block_ranges[1][0])
        boundary_delta_value = second_start[1] - first_end[1]
        cadence = statistics.median(within_deltas)
        if not math.isclose(boundary_delta_value, cadence, rel_tol=1e-6, abs_tol=1e-9):
            raise ReviewPreprocessingError(
                f"Step 10 time gap across separators is not the observed sample cadence "
                f"for {key}: {boundary_delta_value} vs {cadence}"
            )
    else:
        boundary_delta_value = None

    return {
        "available": True,
        "status": "verified_review_step10_contiguous_blocks_v1",
        "raw_voltage": voltage,
        "processed_128_voltage": resample_curve(voltage, n_points=128),
        "current_raw": [item[4] for item in signal_rows],
        "capacity_raw": capacity,
        "testtime_raw": times,
        "steptime_raw": step_times,
        "row_start": block_ranges[0][0],
        "row_end": block_ranges[-1][1],
        "observed_rows": len(signal_rows),
        "row_ranges": block_ranges,
        "separator_rows": separator_values,
        "time_delta_min": min(time_deltas),
        "time_delta_median": statistics.median(time_deltas),
        "time_delta_max": max(time_deltas),
        "separator_boundary_time_delta": boundary_delta_value,
        "capacity_span_native_amphr": capacity[-1] - capacity[0],
        "voltage_first": voltage[0],
        "voltage_last": voltage[-1],
        "segment_sha256": segment_digest.hexdigest(),
        "separator_sha256": separator_digest.hexdigest(),
    }
