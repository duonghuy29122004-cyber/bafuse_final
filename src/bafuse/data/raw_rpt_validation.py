from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, build_physics_features


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = (
    ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
)
OUTPUT_DIR = BENCHMARK_DIR / "m1_raw_validation_20261113"
OBSERVATIONS_PATH = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
SPLIT_PATH = BENCHMARK_DIR / "cell_split.csv"
TRAIN_CANDIDATES_PATH = BENCHMARK_DIR / "pretraining_dataset_candidates.csv"
EXTERNAL_CANDIDATES_PATH = (
    BENCHMARK_DIR / "external_part3_evaluation_candidates.csv"
)
STRICT_PART1_LABELS_PATH = (
    ROOT / "reports" / "bafuse_final_data_validation_v2"
    / "soh_labels_verified_candidates.csv"
)
PART2_LABELS_PATH = (
    ROOT / "data" / "processed" / "label_soh_part1_2"
    / "run_010_part2_soh_candidates" / "labels_soh_part2.csv"
)
PART3_LABELS_PATH = (
    ROOT / "experiments" / "results" / "data_audits"
    / "part3_soh_candidates" / "labels_soh_candidates.csv"
)
PART3_BOL_CROSSWALK_README = (
    ROOT / "experiments" / "results" / "data_audits"
    / "part3_bol_crosswalk" / "README.md"
)
DATASET_REVIEW_PATH = ROOT / "experiments" / "dataset_review" / "README.md"
BASELINE_CONFIG_PATH = BENCHMARK_DIR / "baseline_config.json"
RPT8_PREVIOUS_AUDIT_PATH = BENCHMARK_DIR / "rpt8_reextraction.json"

EXPECTED_CANDIDATES_BY_SPLIT = {
    "train": 102,
    "validation": 37,
    "external_test": 64,
}
EXPECTED_SPLIT_CELLS = {
    "train": 9,
    "validation": 3,
    "external_test": 12,
}
EXPECTED_PART1_STRICT_KEYS = 94
CORE_COLUMNS = ("step", "testtime", "steptime", "amphr", "amps", "volts")
SEGMENT_PATTERN = re.compile(
    r"#Step(?P<step>\d+):.*?rows(?P<start>\d+)-(?P<end>\d+)",
    re.IGNORECASE,
)
SEGMENT_INDEX_PATTERN = re.compile(r"segment(?P<index>\d+)", re.IGNORECASE)
PART_PREFIX_PATTERN = re.compile(r"^data/raw/oxford/part(?P<part>[123])/", re.I)
UNITS_STATUS = "unverified_from_local_source_documentation"
VALIDATOR_VERSION = "m1-validator-1"
SOH_ARITHMETIC_REL_TOL = 1e-12
SOH_ARITHMETIC_ABS_TOL = 1e-12
MAX_TIME_DELTAS_FOR_MEDIAN = 2048
RAW_PARTS = ("part1", "part2", "part3")


@dataclass(frozen=True)
class SegmentReference:
    source_text: str
    source_relative: str
    step: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SegmentRequest:
    block_key: str
    source_path: Path
    source_part: str
    reference: SegmentReference


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required M1 input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_segment_reference(value: str) -> SegmentReference:
    source, separator, locator = value.partition("#")
    match = SEGMENT_PATTERN.search(value)
    if not separator or not source or not match:
        raise ValueError(f"Malformed source segment reference: {value!r}")
    step = int(match.group("step"))
    start = int(match.group("start"))
    end = int(match.group("end"))
    if start < 2 or end < start:
        raise ValueError(f"Invalid source line range: {value!r}")
    if not locator.startswith(f"Step{step}:"):
        raise ValueError(f"Step locator and segment reference disagree: {value!r}")
    return SegmentReference(
        source_text=value,
        source_relative=source.replace("\\", "/"),
        step=step,
        start_line=start,
        end_line=end,
    )


def _normal_relative_source(value: str) -> str:
    normalized = str(value).replace("\\", "/").strip().lstrip("./").casefold()
    match = PART_PREFIX_PATTERN.match(normalized)
    if match:
        normalized = normalized[match.end() :]
    return normalized


def _safe_resolve(raw_root: Path, source: str) -> Path:
    normalized = source.replace("\\", "/").strip()
    posix = PurePosixPath(normalized)
    if not normalized or posix.is_absolute() or ".." in posix.parts:
        raise ValueError(f"Unsafe or empty raw source path: {source!r}")
    if normalized.casefold().startswith("data/raw/oxford/"):
        candidate = ROOT.joinpath(*posix.parts).resolve()
        allowed_root = (ROOT / "data" / "raw" / "oxford").resolve()
    else:
        candidate = raw_root.joinpath(*posix.parts).resolve()
        allowed_root = raw_root.resolve()
    if not candidate.is_relative_to(allowed_root):
        raise ValueError(f"Raw source path escapes its dataset root: {source!r}")
    return candidate


def _resolve_bol_source(
    part: str, reference: str, label_baseline_file: str
) -> tuple[Path | None, str]:
    ref_source = reference.partition("#")[0]
    if not ref_source:
        return None, "missing_bol_source_reference"
    if (
        _normal_relative_source(ref_source)
        != _normal_relative_source(label_baseline_file)
    ):
        return None, "bol_crosswalk_and_label_baseline_source_disagree"

    normalized = ref_source.replace("\\", "/").strip()
    posix = PurePosixPath(normalized)
    if posix.is_absolute() or ".." in posix.parts:
        return None, "unsafe_bol_source_reference"
    candidates: list[Path] = []
    if normalized.casefold().startswith("data/raw/oxford/"):
        candidate = ROOT.joinpath(*posix.parts).resolve()
        if candidate.is_file():
            candidates.append(candidate)
    else:
        for part_name in RAW_PARTS:
            candidate = (ROOT / "data" / "raw" / "oxford" / part_name).joinpath(
                *posix.parts
            )
            if candidate.is_file():
                candidates.append(candidate.resolve())
    if len(candidates) != 1:
        return None, (
            "bol_source_missing"
            if not candidates
            else "bol_source_path_is_ambiguous_across_parts"
        )
    source_part_match = PART_PREFIX_PATTERN.match(
        candidates[0].relative_to(ROOT).as_posix()
    )
    actual_source_part = source_part_match.group("part") if source_part_match else ""
    expected_bol_part = "1" if part == "2" else part
    if actual_source_part != expected_bol_part:
        return candidates[0], (
            f"bol_part_mismatch_expected_{expected_bol_part}_found_{actual_source_part}"
        )
    return candidates[0], "resolved"


def _block_key(path: Path, reference: SegmentReference, source_part: str) -> str:
    return "|".join(
        (
            _display_source_path(path).casefold(),
            source_part,
            str(reference.step),
            str(reference.start_line),
            str(reference.end_line),
        )
    )


def _display_source_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _request(
    source_path: Path, source_part: str, reference: SegmentReference
) -> SegmentRequest:
    return SegmentRequest(
        block_key=_block_key(source_path, reference, source_part),
        source_path=source_path,
        source_part=source_part,
        reference=reference,
    )


class _SegmentAccumulator:
    def __init__(
        self,
        request: SegmentRequest,
        columns: dict[str, int],
        temperature_column: str | None,
        temperature_column_status: str,
    ) -> None:
        self.request = request
        self.columns = columns
        self.temperature_column = temperature_column
        self.temperature_column_status = temperature_column_status
        self.row_count = 0
        self.step_ids: set[int | None] = set()
        self.missing_counts = Counter()
        self.current_signs = Counter()
        self.capacity_deltas = Counter()
        self.voltage_direction_deltas = Counter()
        self.time_deltas = {
            "testtime": {"negative": 0, "zero": 0, "positive": 0},
            "steptime": {"negative": 0, "zero": 0, "positive": 0},
        }
        self.integration_raw = {"testtime": 0.0, "steptime": 0.0}
        self.integration_intervals = {"testtime": 0, "steptime": 0}
        self.time_delta_stats = {
            "testtime": {"min": None, "max": None, "sample": []},
            "steptime": {"min": None, "max": None, "sample": []},
        }
        self.previous: dict[str, float | None] = {
            "testtime": None,
            "steptime": None,
            "amphr": None,
            "amps": None,
            "volts": None,
        }
        self.first: dict[str, float | None] = {}
        self.minimum: dict[str, float | None] = {}
        self.maximum: dict[str, float | None] = {}
        self.temperature_sum = 0.0
        self.temperature_count = 0
        self.temperature_first: float | None = None
        self.temperature_max: float | None = None
        self.voltage_values: list[float] = []
        self.current_values: list[float] = []
        self.temperature_values: list[float] = []
        self.block_digest = hashlib.sha256()
        self.rows_with_wrong_width = 0

    def _value(self, row: list[str], key: str) -> float | None:
        index = self.columns[key]
        return _finite_float(row[index] if index < len(row) else None)

    def update(self, row: list[str]) -> None:
        self.row_count += 1
        self.block_digest.update(
            ("\x1f".join(row) + "\n").encode("utf-8", errors="replace")
        )
        if len(row) != self.columns["_header_length"]:
            self.rows_with_wrong_width += 1
        values = {key: self._value(row, key) for key in CORE_COLUMNS}
        step_value = values["step"]
        if step_value is not None and step_value.is_integer():
            self.step_ids.add(int(step_value))
        else:
            self.step_ids.add(None)
            self.missing_counts["step"] += 1

        for key in ("testtime", "steptime", "amphr", "amps", "volts"):
            number = values[key]
            if number is None:
                self.missing_counts[key] += 1
                self.previous[key] = None
                continue
            if key not in self.first:
                self.first[key] = number
            self.minimum[key] = min(self.minimum.get(key, number), number)
            self.maximum[key] = max(self.maximum.get(key, number), number)

        current = values["amps"]
        voltage = values["volts"]
        capacity = values["amphr"]
        if current is not None:
            self.current_signs[
                "negative" if current < 0 else "positive" if current > 0 else "zero"
            ] += 1
        if voltage is not None and self.request.reference.step == 4:
            self.voltage_values.append(voltage)
        elif voltage is not None:
            self.voltage_values.append(voltage)
        if current is not None:
            self.current_values.append(current)

        if capacity is not None and self.previous["amphr"] is not None:
            delta = capacity - float(self.previous["amphr"])
            self.capacity_deltas[
                "negative" if delta < 0 else "positive" if delta > 0 else "zero"
            ] += 1
        if voltage is not None and self.previous["volts"] is not None:
            delta = voltage - float(self.previous["volts"])
            self.voltage_direction_deltas[
                "negative" if delta < 0 else "positive" if delta > 0 else "zero"
            ] += 1

        for time_key in ("testtime", "steptime"):
            timestamp = values[time_key]
            previous_time = self.previous[time_key]
            if (
                timestamp is not None
                and previous_time is not None
                and current is not None
                and self.previous["amps"] is not None
            ):
                delta_time = timestamp - float(previous_time)
                direction = (
                    "negative"
                    if delta_time < 0
                    else "positive"
                    if delta_time > 0
                    else "zero"
                )
                self.time_deltas[time_key][direction] += 1
                if delta_time > 0:
                    prior_current = float(self.previous["amps"])
                    self.integration_raw[time_key] += (
                        (prior_current + current) * 0.5 * delta_time
                    )
                    self.integration_intervals[time_key] += 1
                    stats = self.time_delta_stats[time_key]
                    stats["min"] = (
                        delta_time
                        if stats["min"] is None
                        else min(float(stats["min"]), delta_time)
                    )
                    stats["max"] = (
                        delta_time
                        if stats["max"] is None
                        else max(float(stats["max"]), delta_time)
                    )
                    if len(stats["sample"]) < MAX_TIME_DELTAS_FOR_MEDIAN:
                        stats["sample"].append(delta_time)
            if timestamp is not None:
                self.previous[time_key] = timestamp
            else:
                self.previous[time_key] = None

        if current is not None:
            self.previous["amps"] = current
        else:
            self.previous["amps"] = None
        if capacity is not None:
            self.previous["amphr"] = capacity
        else:
            self.previous["amphr"] = None
        if voltage is not None:
            self.previous["volts"] = voltage
        else:
            self.previous["volts"] = None

        if self.temperature_column is None:
            self.missing_counts["temperature"] += 1
        else:
            temperature = _finite_float(
                row[self.columns[self.temperature_column]]
                if self.columns[self.temperature_column] < len(row)
                else None
            )
            if temperature is None:
                self.missing_counts["temperature"] += 1
            else:
                self.temperature_sum += temperature
                self.temperature_count += 1
                self.temperature_values.append(temperature)
                if self.temperature_first is None:
                    self.temperature_first = temperature
                self.temperature_max = (
                    temperature
                    if self.temperature_max is None
                    else max(self.temperature_max, temperature)
                )

    def finish(
        self,
        *,
        source_size_bytes: int,
        source_mtime_ns: int,
        source_ctime_ns: int,
        line_range_exact: bool,
        scan_error: str = "",
    ) -> dict[str, Any]:
        reference = self.request.reference
        expected_count = reference.end_line - reference.start_line + 1
        row_count_exact = self.row_count == expected_count
        step_ids = sorted(-1 if item is None else item for item in self.step_ids)
        capacity_span = None
        if self.minimum.get("amphr") is not None and self.maximum.get("amphr") is not None:
            capacity_span = (
                float(self.maximum["amphr"]) - float(self.minimum["amphr"])
            )
        result: dict[str, Any] = {
            "block_key": self.request.block_key,
            "source_file": _display_source_path(self.request.source_path),
            "source_part": self.request.source_part,
            "expected_step": reference.step,
            "start_line": reference.start_line,
            "end_line": reference.end_line,
            "expected_row_count": expected_count,
            "row_count": self.row_count,
            "row_count_exact": row_count_exact,
            "line_range_exact": line_range_exact,
            "source_size_bytes": source_size_bytes,
            "source_mtime_ns": source_mtime_ns,
            "source_ctime_ns": source_ctime_ns,
            "observed_step_ids": ";".join(str(value) for value in step_ids),
            "step_ids_match": self.step_ids == {reference.step},
            "rows_with_wrong_width": self.rows_with_wrong_width,
            "missing_step_rows": self.missing_counts["step"],
            "missing_testtime_rows": self.missing_counts["testtime"],
            "missing_steptime_rows": self.missing_counts["steptime"],
            "missing_amphr_rows": self.missing_counts["amphr"],
            "missing_amps_rows": self.missing_counts["amps"],
            "missing_volts_rows": self.missing_counts["volts"],
            "missing_temperature_rows": self.missing_counts["temperature"],
            "temperature_column": self.temperature_column or "",
            "temperature_column_status": self.temperature_column_status,
            "voltage_first_native": self.first.get("volts"),
            "voltage_last_native": self.previous.get("volts"),
            "voltage_min_v": self.minimum.get("volts"),
            "voltage_max_v": self.maximum.get("volts"),
            "current_first_native": self.first.get("amps"),
            "current_last_native": self.previous.get("amps"),
            "current_min_native": self.minimum.get("amps"),
            "current_max_native": self.maximum.get("amps"),
            "current_negative_rows": self.current_signs["negative"],
            "current_zero_rows": self.current_signs["zero"],
            "current_positive_rows": self.current_signs["positive"],
            "capacity_first_native_amphr": self.first.get("amphr"),
            "capacity_last_native_amphr": self.previous.get("amphr"),
            "capacity_min_native_amphr": self.minimum.get("amphr"),
            "capacity_max_native_amphr": self.maximum.get("amphr"),
            "capacity_span_native_amphr": capacity_span,
            "capacity_positive_deltas": self.capacity_deltas["positive"],
            "capacity_negative_deltas": self.capacity_deltas["negative"],
            "capacity_zero_deltas": self.capacity_deltas["zero"],
            "voltage_positive_deltas": self.voltage_direction_deltas["positive"],
            "voltage_negative_deltas": self.voltage_direction_deltas["negative"],
            "testtime_first_native": self.first.get("testtime"),
            "testtime_last_native": self.previous.get("testtime"),
            "steptime_first_native": self.first.get("steptime"),
            "steptime_last_native": self.previous.get("steptime"),
            "testtime_negative_intervals": self.time_deltas["testtime"]["negative"],
            "testtime_zero_intervals": self.time_deltas["testtime"]["zero"],
            "testtime_positive_intervals": self.time_deltas["testtime"]["positive"],
            "steptime_negative_intervals": self.time_deltas["steptime"]["negative"],
            "steptime_zero_intervals": self.time_deltas["steptime"]["zero"],
            "steptime_positive_intervals": self.time_deltas["steptime"]["positive"],
            "testtime_integral_native_current_time": self.integration_raw["testtime"],
            "steptime_integral_native_current_time": self.integration_raw["steptime"],
            "testtime_integral_if_minutes_native_capacity": abs(
                self.integration_raw["testtime"]
            )
            / 60.0,
            "steptime_integral_if_minutes_native_capacity": abs(
                self.integration_raw["steptime"]
            )
            / 60.0,
            "time_unit_status": UNITS_STATUS,
            "current_unit_status": UNITS_STATUS,
            "capacity_unit_status": "native_Amphr_absolute_unit_unverified",
            "testtime_dt_min_native": self.time_delta_stats["testtime"]["min"],
            "testtime_dt_max_native": self.time_delta_stats["testtime"]["max"],
            "testtime_dt_median_first_sample_native": (
                statistics.median(self.time_delta_stats["testtime"]["sample"])
                if self.time_delta_stats["testtime"]["sample"]
                else None
            ),
            "testtime_dt_sample_count": len(
                self.time_delta_stats["testtime"]["sample"]
            ),
            "steptime_dt_min_native": self.time_delta_stats["steptime"]["min"],
            "steptime_dt_max_native": self.time_delta_stats["steptime"]["max"],
            "steptime_dt_median_first_sample_native": (
                statistics.median(self.time_delta_stats["steptime"]["sample"])
                if self.time_delta_stats["steptime"]["sample"]
                else None
            ),
            "steptime_dt_sample_count": len(
                self.time_delta_stats["steptime"]["sample"]
            ),
            "step4_temperature_mean_native": (
                self.temperature_sum / self.temperature_count
                if self.temperature_count
                else None
            ),
            "step4_temperature_rise_native": (
                self.temperature_max - self.temperature_first
                if self.temperature_count and self.temperature_first is not None
                and self.temperature_max is not None
                else None
            ),
            "step4_temperature_finite_rows": self.temperature_count,
            "step4_mean_absolute_current_native": (
                statistics.mean(abs(value) for value in self.current_values)
                if self.current_values
                else None
            ),
            "raw_block_sha256": self.block_digest.hexdigest(),
            "scan_error": scan_error,
        }
        if (
            reference.step == 4
            and self.row_count >= 2
            and not self.missing_counts["volts"]
            and not self.missing_counts["amps"]
            and self.temperature_count == self.row_count
        ):
            try:
                result["physics_features"] = build_physics_features(
                    self.voltage_values,
                    self.current_values,
                    self.temperature_values,
                )
                result["physics_status"] = "computable_native_units"
            except (ValueError, ArithmeticError) as error:
                result["physics_features"] = {}
                result["physics_status"] = f"computation_error:{error}"
        else:
            result["physics_features"] = {}
            result["physics_status"] = "incomplete_required_step4_physics_inputs"
        return result


def _choose_temperature_column(
    part: str, columns: dict[str, int]
) -> tuple[str | None, str]:
    preferred = "temp1" if part == "1" else "temp"
    alternate = "temp" if preferred == "temp1" else "temp1"
    if preferred in columns:
        return preferred, "part_preferred_header"
    if alternate in columns:
        return alternate, "alternate_temperature_header_used"
    return None, "temperature_column_absent"


def _configured_step4_cutoff_tolerance_v(path: Path) -> float | None:
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    if config.get("training_authorized") is not False:
        raise ValueError("M1 requires training_authorized to remain false.")
    value = _finite_float(
        config.get("preflight_gates", {}).get("step4_cutoff_tolerance_v")
    )
    if value is not None and value < 0:
        raise ValueError("Configured Step 4 cutoff tolerance cannot be negative.")
    return value


def _scan_source_file(
    source_path: Path,
    requests: list[SegmentRequest],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    stat_before = source_path.stat()
    unique: dict[str, SegmentRequest] = {}
    for request in requests:
        unique.setdefault(request.block_key, request)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item.reference.start_line,
            item.reference.end_line,
            item.reference.step,
        ),
    )
    overlap_keys: set[str] = set()
    for index, current in enumerate(ordered):
        for other in ordered[index + 1 :]:
            if other.reference.start_line > current.reference.end_line:
                break
            if (
                current.reference.start_line != other.reference.start_line
                or current.reference.end_line != other.reference.end_line
                or current.reference.step != other.reference.step
            ):
                overlap_keys.add(current.block_key)
                overlap_keys.add(other.block_key)

    results: dict[str, dict[str, Any]] = {}
    accumulators: dict[str, _SegmentAccumulator] = {}
    file_rows_consumed = 0
    file_error = ""
    try:
        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                raise ValueError("CSV has no header.")
            folded = [name.strip().casefold() for name in header]
            columns: dict[str, int] = {
                name: index for index, name in enumerate(folded)
            }
            for name in CORE_COLUMNS:
                if folded.count(name) != 1:
                    raise ValueError(
                        f"Expected exactly one {name!r} column, found {folded.count(name)}."
                    )
            columns["_header_length"] = len(header)
            source_part = ordered[0].source_part
            temperature_column, temperature_status = _choose_temperature_column(
                source_part, columns
            )
            start_map: dict[int, list[SegmentRequest]] = defaultdict(list)
            for request in ordered:
                start_map[request.reference.start_line].append(request)
                accumulators[request.block_key] = _SegmentAccumulator(
                    request,
                    columns,
                    temperature_column,
                    temperature_status,
                )
            active: list[SegmentRequest] = []
            previous_line = 1
            max_end = max(item.reference.end_line for item in ordered)
            for row in reader:
                line_number = reader.line_num
                file_rows_consumed += 1
                if line_number - previous_line > 1:
                    for request in ordered:
                        if (
                            previous_line < request.reference.start_line <= line_number
                            or previous_line < request.reference.end_line < line_number
                        ):
                            overlap_keys.add(request.block_key)
                previous_line = line_number
                active.extend(start_map.get(line_number, ()))
                if line_number > max_end:
                    break
                for request in active:
                    if line_number <= request.reference.end_line:
                        accumulators[request.block_key].update(row)
                active = [
                    request
                    for request in active
                    if line_number < request.reference.end_line
                ]
                if line_number >= max_end:
                    break
    except (OSError, UnicodeDecodeError, csv.Error, ValueError, StopIteration) as error:
        file_error = f"{type(error).__name__}: {error}"

    stat_after = source_path.stat()
    changed = (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
        or stat_before.st_ctime_ns != stat_after.st_ctime_ns
    )
    file_metadata = {
        "source_file": _display_source_path(source_path),
        "source_size_bytes": stat_before.st_size,
        "source_mtime_ns": stat_before.st_mtime_ns,
        "source_ctime_ns": stat_before.st_ctime_ns,
        "source_changed_during_scan": changed,
        "csv_data_rows_consumed_to_last_required_line": file_rows_consumed,
        "scan_error": file_error,
    }

    for request in ordered:
        accumulator = accumulators.get(request.block_key)
        if accumulator is None:
            result = {
                **file_metadata,
                "block_key": request.block_key,
                "expected_step": request.reference.step,
                "start_line": request.reference.start_line,
                "end_line": request.reference.end_line,
                "expected_row_count": (
                    request.reference.end_line - request.reference.start_line + 1
                ),
                "row_count": 0,
                "row_count_exact": False,
                "line_range_exact": False,
                "observed_step_ids": "",
                "step_ids_match": False,
                "raw_block_sha256": "",
                "physics_features": {},
                "physics_status": "not_scanned",
                "scan_error": file_error or "CSV header could not be processed.",
            }
        else:
            result = accumulator.finish(
                source_size_bytes=stat_before.st_size,
                source_mtime_ns=stat_before.st_mtime_ns,
                source_ctime_ns=stat_before.st_ctime_ns,
                line_range_exact=request.block_key not in overlap_keys,
                scan_error=file_error,
            )
            result.update(file_metadata)
            if request.block_key in overlap_keys:
                result["line_range_exact"] = False
        results[request.block_key] = result
    return results, file_metadata


class _BlockCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS blocks (
                block_key TEXT PRIMARY KEY,
                source_size_bytes INTEGER NOT NULL,
                source_mtime_ns INTEGER NOT NULL,
                source_ctime_ns INTEGER NOT NULL,
                result_json TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def get(self, request: SegmentRequest) -> dict[str, Any] | None:
        stat = request.source_path.stat()
        row = self.connection.execute(
            """
            SELECT result_json FROM blocks
            WHERE block_key = ? AND source_size_bytes = ?
              AND source_mtime_ns = ? AND source_ctime_ns = ?
            """,
            (
                request.block_key,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        ).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, result: dict[str, Any]) -> None:
        self.connection.execute(
            """
            INSERT OR REPLACE INTO blocks
                (block_key, source_size_bytes, source_mtime_ns, source_ctime_ns, result_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                result["block_key"],
                result["source_size_bytes"],
                result["source_mtime_ns"],
                result["source_ctime_ns"],
                json.dumps(result, sort_keys=True, allow_nan=False),
            ),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def _load_split() -> tuple[dict[str, str], str]:
    rows = _read_csv(SPLIT_PATH)
    cell_to_split: dict[str, str] = {}
    versions = {row["split_version"] for row in rows}
    if len(versions) != 1 or not rows:
        raise ValueError("Cell split manifest is empty or has multiple versions.")
    for row in rows:
        cell = row["physical_cell_id"]
        partition = row["partition"]
        if not cell or partition not in EXPECTED_SPLIT_CELLS or cell in cell_to_split:
            raise ValueError(f"Invalid or duplicate cell split row: {row}")
        cell_to_split[cell] = partition
    counts = Counter(cell_to_split.values())
    if dict(counts) != EXPECTED_SPLIT_CELLS:
        raise ValueError(
            f"Expected cell split {EXPECTED_SPLIT_CELLS}, found {dict(counts)}."
        )
    return cell_to_split, next(iter(versions))


def _load_labels() -> tuple[
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[str, str, str], dict[str, str]],
    dict[tuple[str, str, str], dict[str, str]],
]:
    strict = _read_csv(STRICT_PART1_LABELS_PATH)
    if len(strict) != EXPECTED_PART1_STRICT_KEYS:
        raise ValueError(
            f"Expected exactly {EXPECTED_PART1_STRICT_KEYS} strict Part 1 keys, "
            f"found {len(strict)}."
        )
    p1: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in strict:
        key = ("1", row["physical_cell_id"], row["measurement_id"].removeprefix("RPT"))
        if (
            row.get("audit_status") != "verified_source_and_arithmetic"
            or row.get("validation_status_v2") != "VERIFIED_DERIVED_LABEL"
            or key in p1
            or _finite_float(row.get("stored_soh_percent")) is None
        ):
            raise ValueError(f"Invalid or duplicate strict Part 1 label: {key}")
        p1[key] = row

    selected_p2_rows = [
        row
        for row in _read_csv(PART2_LABELS_PATH)
        if row.get("dataset_part") == "2"
        and row.get("condition") == "RPT"
        and row.get("group_id") in {"1", "2", "3", "4"}
        and row.get("validation_status") == "calculated_from_measured_capacity"
        and row.get("label_confidence") == "medium_ratio_only"
    ]
    if len(selected_p2_rows) != 54:
        raise ValueError(
            f"Expected 54 selected Part 2 source candidates, found {len(selected_p2_rows)}."
        )
    p2: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in selected_p2_rows:
        key = ("2", row["physical_cell_id"], row["measurement_id"].removeprefix("RPT"))
        if key in p2 or _finite_float(row.get("soh_percent")) is None:
            raise ValueError(f"Invalid or duplicate Part 2 label candidate: {key}")
        p2[key] = row

    p3: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv(PART3_LABELS_PATH):
        if (
            row.get("dataset_part") != "3"
            or row.get("condition") != "RPT"
            or row.get("validation_status") != "calculated_from_measured_capacity"
            or not row.get("measurement_id", "").startswith("RPT")
        ):
            continue
        key = ("3", row["physical_cell_id"], row["measurement_id"].removeprefix("RPT"))
        if key in p3 or _finite_float(row.get("soh_percent")) is None:
            raise ValueError(f"Invalid or duplicate Part 3 label candidate: {key}")
        p3[key] = row
    return p1, p2, p3


def _load_observations() -> dict[tuple[str, str, str], dict[str, str]]:
    observations: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in _read_csv(OBSERVATIONS_PATH):
        key = (row["part"], row["physical_cell_id"], row["rpt_index"])
        if key in observations:
            raise ValueError(f"Duplicate Guide/source observation key: {key}")
        observations[key] = row
    return observations


def _source_part(path: Path) -> str:
    match = PART_PREFIX_PATTERN.match(path.relative_to(ROOT).as_posix())
    return match.group("part") if match else ""


def _load_candidate_rows(
    cell_to_split: dict[str, str],
    observations: dict[tuple[str, str, str], dict[str, str]],
    labels: tuple[
        dict[tuple[str, str, str], dict[str, str]],
        dict[tuple[str, str, str], dict[str, str]],
        dict[tuple[str, str, str], dict[str, str]],
    ],
) -> list[dict[str, Any]]:
    p1_labels, p2_labels, p3_labels = labels
    candidate_rows: list[dict[str, Any]] = []
    for path in (TRAIN_CANDIDATES_PATH, EXTERNAL_CANDIDATES_PATH):
        candidate_rows.extend(_read_csv(path))
    count_by_split = Counter(row["partition"] for row in candidate_rows)
    if dict(count_by_split) != EXPECTED_CANDIDATES_BY_SPLIT:
        raise ValueError(
            "Candidate input set changed; expected current candidate counts "
            f"{EXPECTED_CANDIDATES_BY_SPLIT}, found {dict(count_by_split)}."
        )

    seen: set[tuple[str, str, str]] = set()
    enriched: list[dict[str, Any]] = []
    for candidate in candidate_rows:
        key = (
            candidate["part"],
            candidate["physical_cell_id"],
            candidate["rpt_index"],
        )
        if key in seen:
            raise ValueError(f"Duplicate proposed candidate key: {key}")
        seen.add(key)
        part, cell, rpt = key
        if cell not in cell_to_split or candidate["partition"] != cell_to_split[cell]:
            raise ValueError(f"Candidate split disagrees with cell split: {key}")
        if part == "3" and candidate["partition"] != "external_test":
            raise ValueError(f"Part 3 candidate is not external-only: {key}")
        if part in {"1", "2"} and candidate["partition"] not in {"train", "validation"}:
            raise ValueError(f"Part 1/2 candidate has an invalid partition: {key}")
        observation = observations.get(key)
        if observation is None:
            enriched.append(
                {
                    **candidate,
                    "_key": key,
                    "_observation": None,
                    "_label": None,
                    "_mapping_status": "missing_guide_observation",
                    "_step4_request": None,
                    "_step10_request": None,
                    "_bol_request": None,
                    "_bol_resolution_status": "not_resolved",
                }
            )
            continue

        label = (p1_labels if part == "1" else p2_labels if part == "2" else p3_labels).get(key)
        status = "verified"
        if observation.get("mapping_status") != "verified":
            status = "guide_mapping_not_verified"
        if observation.get("step4_status") != "verified":
            status = "step4_guide_mapping_not_verified"
        if observation.get("step10_status") != "verified":
            status = "step10_guide_mapping_not_verified"
        if (
            candidate.get("source_file") != observation.get("source_file")
            or candidate.get("step4_segment_ref") != observation.get("step4_segment_ref")
            or candidate.get("step10_segment_ref") != observation.get("step10_segment_ref")
        ):
            status = "candidate_and_guide_source_references_disagree"
        if label is None:
            status = "candidate_label_source_missing"

        step4_request = None
        step10_request = None
        bol_request = None
        bol_resolution_status = "not_resolved"
        try:
            step4_ref = parse_segment_reference(observation["step4_segment_ref"])
            step10_ref = parse_segment_reference(observation["step10_segment_ref"])
            if step4_ref.step != 4 or step10_ref.step != 10:
                status = "unexpected_step_number_in_guide_reference"
            step4_path = _safe_resolve(
                ROOT,
                step4_ref.source_relative,
            )
            step10_path = _safe_resolve(
                ROOT,
                step10_ref.source_relative,
            )
            expected_source = _safe_resolve(ROOT, observation["source_file"])
            if (
                step4_path != expected_source
                or step10_path != expected_source
                or step4_path != step10_path
            ):
                status = "measurement_blocks_do_not_resolve_to_exact_guide_source"
            source_part = _source_part(expected_source)
            if source_part != part:
                status = "guide_source_part_mismatch"
            if step4_path.is_file():
                step4_request = _request(step4_path, source_part, step4_ref)
            if step10_path.is_file():
                step10_request = _request(step10_path, source_part, step10_ref)

            if label is not None:
                bol_path, bol_resolution_status = _resolve_bol_source(
                    part,
                    observation.get("bol_step4_segment_ref", ""),
                    label.get("baseline_source_file", ""),
                )
                bol_text = observation.get("bol_step4_segment_ref", "")
                bol_ref = parse_segment_reference(bol_text)
                if bol_ref.step != 4:
                    bol_resolution_status = "bol_reference_is_not_step4"
                if (
                    label.get("baseline_segment_index")
                    and SEGMENT_INDEX_PATTERN.search(bol_text)
                    and int(label["baseline_segment_index"])
                    != int(SEGMENT_INDEX_PATTERN.search(bol_text).group("index"))
                ):
                    bol_resolution_status = "label_and_guide_bol_segment_indexes_disagree"
                if bol_path is not None and bol_path.is_file():
                    bol_part = _source_part(bol_path)
                    bol_request = _request(bol_path, bol_part, bol_ref)
        except (KeyError, ValueError, OSError) as error:
            status = f"source_reference_error:{type(error).__name__}:{error}"

        enriched.append(
            {
                **candidate,
                "_key": key,
                "_observation": observation,
                "_label": label,
                "_mapping_status": status,
                "_step4_request": step4_request,
                "_step10_request": step10_request,
                "_bol_request": bol_request,
                "_bol_resolution_status": bol_resolution_status,
            }
        )

    strict_keys = set(p1_labels)
    admitted_part1 = [
        row for row in enriched if row["part"] == "1" and row.get("strict_part1_key") == "True"
    ]
    unapproved_part1 = [
        row for row in enriched if row["part"] == "1" and row.get("strict_part1_key") != "True"
    ]
    if any(row["_key"] not in strict_keys for row in admitted_part1):
        raise ValueError("An unapproved Part 1 key is marked as a strict candidate.")
    if unapproved_part1:
        raise ValueError(
            "Proposed Part 1 candidate input contains keys outside the exact "
            f"94-key whitelist: {[row['_key'] for row in unapproved_part1[:5]]}"
        )
    return enriched


def _build_requests(
    candidates: list[dict[str, Any]],
    observations: dict[tuple[str, str, str], dict[str, str]],
) -> tuple[dict[str, list[SegmentRequest]], dict[str, str]]:
    requests_by_file: dict[str, dict[str, SegmentRequest]] = defaultdict(dict)
    request_aliases: dict[str, str] = {}
    for candidate in candidates:
        for field_name in ("_step4_request", "_step10_request", "_bol_request"):
            request = candidate.get(field_name)
            if request is None:
                continue
            requests_by_file[str(request.source_path)][request.block_key] = request
            request_aliases[
                f"{candidate['_key'][0]}|{candidate['_key'][1]}|{candidate['_key'][2]}|{field_name}"
            ] = request.block_key

    rpt8_key = ("1", "G1-Cell9", "8")
    rpt8 = observations.get(rpt8_key)
    if rpt8:
        for field_name in ("step4_segment_ref", "step10_segment_ref"):
            try:
                reference = parse_segment_reference(rpt8[field_name])
                path = _safe_resolve(ROOT, reference.source_relative)
                request = _request(path, "1", reference)
                requests_by_file[str(path)][request.block_key] = request
                request_aliases[f"RPT8_CONTROL|{field_name}"] = request.block_key
            except (ValueError, OSError):
                continue
    return (
        {path: list(items.values()) for path, items in requests_by_file.items()},
        request_aliases,
    )


def _scan_requested_blocks(
    requests_by_file: dict[str, list[SegmentRequest]],
    cache: _BlockCache,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    block_results: dict[str, dict[str, Any]] = {}
    file_results: list[dict[str, Any]] = []
    for file_path in sorted(requests_by_file):
        requests = requests_by_file[file_path]
        uncached = [request for request in requests if cache.get(request) is None]
        cached_results = {
            request.block_key: cache.get(request)
            for request in requests
            if request not in uncached
        }
        block_results.update(
            {
                key: value
                for key, value in cached_results.items()
                if value is not None
            }
        )
        if not uncached:
            file_results.append(
                {
                    "source_file": _display_source_path(Path(file_path)),
                    "cache_status": "all_blocks_reused",
                    "blocks": len(requests),
                }
            )
            continue
        try:
            results, metadata = _scan_source_file(
                Path(file_path),
                uncached,
            )
            file_results.append(
                {
                    **metadata,
                    "cache_status": "scanned",
                    "blocks": len(uncached),
                }
            )
        except OSError as error:
            results = {}
            file_results.append(
                {
                    "source_file": _display_source_path(Path(file_path)),
                    "cache_status": "scan_failed",
                    "blocks": len(uncached),
                    "scan_error": f"{type(error).__name__}: {error}",
                }
            )
        for request in uncached:
            result = results.get(request.block_key)
            if result is None:
                result = {
                    "block_key": request.block_key,
                    "source_file": _display_source_path(request.source_path),
                    "source_part": request.source_part,
                    "expected_step": request.reference.step,
                    "start_line": request.reference.start_line,
                    "end_line": request.reference.end_line,
                    "expected_row_count": (
                        request.reference.end_line
                        - request.reference.start_line
                        + 1
                    ),
                    "row_count": 0,
                    "row_count_exact": False,
                    "line_range_exact": False,
                    "step_ids_match": False,
                    "raw_block_sha256": "",
                    "physics_features": {},
                    "physics_status": "not_scanned",
                    "scan_error": "No block result returned from source scan.",
                }
            cache.put(result)
            block_results[request.block_key] = result
    return block_results, file_results


def _block_validity(
    metrics: dict[str, Any] | None,
    expected_step: int,
    step4_cutoff_tolerance_v: float | None = None,
) -> tuple[str, list[str]]:
    if not metrics:
        return "REVIEW", ["raw_block_not_scanned_or_unmapped"]
    reasons: list[str] = []
    if metrics.get("scan_error"):
        reasons.append(f"source_scan_error:{metrics['scan_error']}")
    if not metrics.get("line_range_exact"):
        reasons.append("source_line_range_ambiguous_or_not_exact")
    if not metrics.get("row_count_exact"):
        reasons.append("mapped_block_row_count_mismatch")
    if metrics.get("observed_step_ids") != str(expected_step):
        reasons.append("observed_step_ids_do_not_match_expected")
    if any(
        int(metrics.get(f"missing_{column}_rows", 0) or 0) > 0
        for column in ("step", "amphr", "amps", "volts")
    ):
        reasons.append("missing_or_nonfinite_core_measurement_values")
    if int(metrics.get("rows_with_wrong_width", 0) or 0) > 0:
        reasons.append("raw_rows_have_inconsistent_column_counts")
    if int(metrics.get("row_count", 0) or 0) < 2:
        reasons.append("fewer_than_two_mapped_measurement_rows")
    if reasons:
        return "REJECTED", reasons

    wrong_direction = (
        metrics.get("current_positive_rows", 0)
        if expected_step == 4
        else metrics.get("current_negative_rows", 0)
    )
    if wrong_direction:
        return "REJECTED", ["current_direction_conflicts_with_documented_step"]
    expected_direction_rows = (
        metrics.get("current_negative_rows", 0)
        if expected_step == 4
        else metrics.get("current_positive_rows", 0)
    )
    if not expected_direction_rows:
        return "REJECTED", ["no_current_rows_in_documented_step_direction"]
    if metrics.get("capacity_negative_deltas", 0):
        return "REVIEW", ["native_capacity_counter_has_negative_deltas"]
    if (
        metrics.get("testtime_negative_intervals", 0)
        or metrics.get("steptime_negative_intervals", 0)
        or metrics.get("testtime_zero_intervals", 0)
        or metrics.get("steptime_zero_intervals", 0)
    ):
        return "REVIEW", ["time_axis_has_reset_or_nonincreasing_interval"]

    min_voltage = _finite_float(metrics.get("voltage_min_v"))
    max_voltage = _finite_float(metrics.get("voltage_max_v"))
    if expected_step == 4 and (min_voltage is None or min_voltage > 2.5):
        if step4_cutoff_tolerance_v is None:
            return "REVIEW", ["step4_cutoff_tolerance_unverified"]
        if min_voltage > 2.5 + step4_cutoff_tolerance_v:
            return "REJECTED", ["step4_does_not_reach_configured_cutoff_tolerance"]
    if expected_step == 10 and (max_voltage is None or max_voltage < 4.2):
        return "REVIEW", ["step10_does_not_reach_documented_4_2v_charge_endpoint"]
    return "ACCEPTED", []


def _candidate_label_value(label: dict[str, str] | None, part: str) -> float | None:
    if label is None:
        return None
    return _finite_float(
        label.get("stored_soh_percent") if part == "1" else label.get("soh_percent")
    )


def _label_capacity_values(
    label: dict[str, str] | None, part: str
) -> tuple[float | None, float | None]:
    if label is None:
        return None, None
    if part == "1":
        return (
            _finite_float(label.get("raw_capacity_span_native")),
            _finite_float(label.get("raw_baseline_capacity_native")),
        )
    return (
        _finite_float(label.get("capacity_raw")),
        _finite_float(label.get("reference_capacity_raw")),
    )


def _reference_disposition(
    *,
    part: str,
    mapping_verified: bool,
    step4_status: str,
    bol_status: str,
    label_sources_match: bool,
    candidate_label_matches_source: bool,
    arithmetic_matches: bool,
    integration_units_verified: bool,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if step4_status == "REJECTED" or bol_status == "REJECTED":
        return "REJECTED", ["invalid_or_incomplete_step4_or_bol_measurement"]
    if not mapping_verified:
        reasons.append("guide_or_bol_mapping_unresolved")
    if step4_status != "ACCEPTED":
        reasons.append("step4_measurement_not_accepted")
    if bol_status != "ACCEPTED":
        reasons.append("bol_measurement_not_accepted")
    if not label_sources_match:
        reasons.append("label_source_provenance_mismatch")
    if not candidate_label_matches_source:
        reasons.append("candidate_manifest_and_source_label_disagree")
    if not arithmetic_matches:
        reasons.append("candidate_label_arithmetic_or_source_capacity_mismatch")
    if reasons:
        return "REVIEW", reasons
    if part in {"2", "3"}:
        return "REVIEW", ["part_label_policy_remains_provisional_pending_user_approval"]
    if not integration_units_verified:
        return "REVIEW", ["current_integration_time_and_current_units_unverified"]
    return "ACCEPTED", []


def _source_label_agrees(
    candidate: dict[str, Any], observation: dict[str, str], label: dict[str, str] | None
) -> tuple[bool, list[str]]:
    if not label:
        return False, ["source_label_record_missing"]
    issues: list[str] = []
    if (
        _normal_relative_source(label.get("source_file", ""))
        != _normal_relative_source(observation.get("source_file", ""))
    ):
        issues.append("label_numerator_source_does_not_match_guide_source")
    if (
        _normal_relative_source(label.get("baseline_source_file", ""))
        != _normal_relative_source(observation.get("bol_reference_file", ""))
    ):
        issues.append("label_denominator_source_does_not_match_guide_bol")
    step4_ref = observation.get("step4_segment_ref", "")
    step4_match = SEGMENT_INDEX_PATTERN.search(step4_ref)
    label_numerator_index = label.get(
        "step4_segment_index", label.get("step4_segment_index", "")
    )
    parsed_numerator_index = _finite_float(label_numerator_index)
    if (
        step4_match
        and label_numerator_index
        and (
            parsed_numerator_index is None
            or int(parsed_numerator_index) != int(step4_match.group("index"))
        )
    ):
        issues.append("label_numerator_segment_index_disagrees_with_guide")
    bol_match = SEGMENT_INDEX_PATTERN.search(
        observation.get("bol_step4_segment_ref", "")
    )
    label_bol_index = label.get(
        "baseline_step4_segment_index", label.get("baseline_segment_index", "")
    )
    parsed_bol_index = _finite_float(label_bol_index)
    if (
        bol_match
        and label_bol_index
        and (
            parsed_bol_index is None
            or int(parsed_bol_index) != int(bol_match.group("index"))
        )
    ):
        issues.append("label_bol_segment_index_disagrees_with_guide")
    return not issues, issues


def _audit_one_candidate(
    candidate: dict[str, Any],
    request_aliases: dict[str, str],
    block_results: dict[str, dict[str, Any]],
    step4_cutoff_tolerance_v: float | None,
) -> dict[str, Any]:
    part, cell, rpt = candidate["_key"]
    observation = candidate.get("_observation") or {}
    label = candidate.get("_label")

    def result_for(field_name: str) -> dict[str, Any] | None:
        alias = f"{part}|{cell}|{rpt}|{field_name}"
        block_key = request_aliases.get(alias)
        return block_results.get(block_key) if block_key else None

    step4 = result_for("_step4_request")
    step10 = result_for("_step10_request")
    bol = result_for("_bol_request")
    step4_tolerance = (
        step4_cutoff_tolerance_v
        if part in {"1", "2"}
        else None
    )
    step4_status, step4_reasons = _block_validity(
        step4, 4, step4_cutoff_tolerance_v=step4_tolerance
    )
    step10_status, step10_reasons = _block_validity(step10, 10)
    bol_status, bol_reasons = _block_validity(
        bol, 4, step4_cutoff_tolerance_v=step4_tolerance
    )
    mapping_status = candidate.get("_mapping_status", "not_verified")
    bol_mapping_status = candidate.get("_bol_resolution_status", "not_resolved")
    if mapping_status != "verified":
        step4_status = "REVIEW"
        step10_status = "REVIEW"
        step4_reasons.append(mapping_status)
        step10_reasons.append(mapping_status)
    if bol_mapping_status != "resolved":
        bol_status = "REVIEW"
        bol_reasons.append(bol_mapping_status)

    expected_bol_part = "1" if part == "2" else part
    protocol = (
        "Maccor 4200; Step 4 CC-CV discharge C/2 to 2.5 V"
        if part in {"1", "2"}
        else "Maccor 4000; Step 4 CC discharge C/2 to 2.5 V"
    )
    label_value = _candidate_label_value(label, part)
    candidate_manifest_label = _finite_float(candidate.get("soh_candidate_percent"))
    label_capacity, label_bol_capacity = _label_capacity_values(label, part)
    numerator_capacity = (
        _finite_float(step4.get("capacity_span_native_amphr")) if step4 else None
    )
    denominator_capacity = (
        _finite_float(bol.get("capacity_span_native_amphr")) if bol else None
    )
    recomputed_soh = None
    if (
        numerator_capacity is not None
        and denominator_capacity is not None
        and denominator_capacity > 0
    ):
        recomputed_soh = 100.0 * numerator_capacity / denominator_capacity

    label_source_matches, label_source_issues = _source_label_agrees(
        candidate,
        observation,
        label,
    )
    arithmetic_matches = (
        label_capacity is not None
        and label_bol_capacity is not None
        and numerator_capacity is not None
        and denominator_capacity is not None
        and math.isclose(
            label_capacity,
            numerator_capacity,
            rel_tol=SOH_ARITHMETIC_REL_TOL,
            abs_tol=SOH_ARITHMETIC_ABS_TOL,
        )
        and math.isclose(
            label_bol_capacity,
            denominator_capacity,
            rel_tol=SOH_ARITHMETIC_REL_TOL,
            abs_tol=SOH_ARITHMETIC_ABS_TOL,
        )
        and recomputed_soh is not None
        and label_value is not None
        and math.isclose(
            label_value,
            recomputed_soh,
            rel_tol=SOH_ARITHMETIC_REL_TOL,
            abs_tol=SOH_ARITHMETIC_ABS_TOL,
        )
    )

    candidate_label_matches_source = (
        candidate_manifest_label is not None
        and label_value is not None
        and math.isclose(
            candidate_manifest_label,
            label_value,
            rel_tol=SOH_ARITHMETIC_REL_TOL,
            abs_tol=SOH_ARITHMETIC_ABS_TOL,
        )
    )
    label_disposition, label_reasons = _reference_disposition(
        part=part,
        mapping_verified=mapping_status == "verified",
        step4_status=step4_status,
        bol_status=bol_status,
        label_sources_match=label_source_matches and not label_source_issues,
        candidate_label_matches_source=candidate_label_matches_source,
        arithmetic_matches=arithmetic_matches,
        integration_units_verified=False,
    )
    if step4_status != "ACCEPTED":
        label_reasons.extend(step4_reasons)
    if bol_status != "ACCEPTED":
        label_reasons.extend(f"bol_{reason}" for reason in bol_reasons)
    if not label_source_matches:
        label_reasons.extend(label_source_issues)

    physics = (step4 or {}).get("physics_features", {})
    physics_required = set(PHYSICS_FEATURE_NAMES)
    physics_structurally_complete = (
        step4_status == "ACCEPTED"
        and physics_required.issubset(physics)
        and all(_finite_float(physics.get(name)) is not None for name in physics_required)
    )
    temperature_available = (
        step4 is not None
        and step4.get("missing_temperature_rows") == 0
        and bool(step4.get("temperature_column"))
    )
    physics_status = (
        "STRUCTURALLY_COMPLETE_UNITS_UNVERIFIED"
        if physics_structurally_complete
        else "INCOMPLETE"
    )
    structurally_complete_modalities = bool(
        step4_status == "ACCEPTED"
        and step10_status == "ACCEPTED"
        and physics_structurally_complete
    )
    final_eligible = bool(
        structurally_complete_modalities
        and label_disposition == "ACCEPTED"
        and part in {"1", "2"}
        and candidate["partition"] in {"train", "validation"}
    )
    external_eligible = bool(
        structurally_complete_modalities
        and label_disposition == "ACCEPTED"
        and part == "3"
        and candidate["partition"] == "external_test"
    )
    final_status = (
        "M1_COMPLETE_PENDING_A1_A1B"
        if final_eligible
        else "NOT_ELIGIBLE_FOR_PRIMARY_BENCHMARK"
    )
    if part == "3" and not external_eligible:
        final_status = "NOT_ELIGIBLE_FOR_EXTERNAL_EVALUATION"
    if structurally_complete_modalities and label_disposition == "REVIEW":
        final_status = "STRUCTURAL_MODALITIES_COMPLETE_LABEL_REVIEW"

    reasons = list(dict.fromkeys(
        step4_reasons + step10_reasons + bol_reasons + label_reasons
    ))
    return {
        "dataset_part": part,
        "physical_cell_id": cell,
        "rpt_index": int(rpt),
        "measurement_id": f"RPT{rpt}",
        "partition": candidate["partition"],
        "candidate_record": True,
        "source_tpg": candidate.get("source_tpg", ""),
        "source_record_id": observation.get("source_record_id", ""),
        "guide_stage": observation.get("guide_stage", candidate.get("guide_stage", "")),
        "guide_mapping_status": mapping_status,
        "source_file": candidate.get("source_file", ""),
        "source_file_size_bytes": (
            step4.get("source_size_bytes") if step4 else None
        ),
        "step4_segment_ref": observation.get("step4_segment_ref", ""),
        "step4_validity": step4_status,
        "step4_reasons": "|".join(dict.fromkeys(step4_reasons)),
        "step4_row_count": step4.get("row_count") if step4 else None,
        "step4_voltage_first_v": step4.get("voltage_first_native") if step4 else None,
        "step4_voltage_last_v": step4.get("voltage_last_native") if step4 else None,
        "step4_voltage_min_v": step4.get("voltage_min_v") if step4 else None,
        "step4_voltage_max_v": step4.get("voltage_max_v") if step4 else None,
        "step4_current_signs": (
            f"negative={step4.get('current_negative_rows',0)};"
            f"zero={step4.get('current_zero_rows',0)};"
            f"positive={step4.get('current_positive_rows',0)}"
            if step4
            else ""
        ),
        "step4_capacity_span_native_amphr": numerator_capacity,
        "step4_testtime_integral_if_minutes_native_capacity": (
            step4.get("testtime_integral_if_minutes_native_capacity")
            if step4
            else None
        ),
        "step4_testtime_integral_minus_amphr_span_if_minutes": (
            step4.get("testtime_integral_if_minutes_native_capacity")
            - numerator_capacity
            if step4
            and step4.get("testtime_integral_if_minutes_native_capacity") is not None
            and numerator_capacity is not None
            else None
        ),
        "step4_steptime_integral_if_minutes_native_capacity": (
            step4.get("steptime_integral_if_minutes_native_capacity")
            if step4
            else None
        ),
        "step4_steptime_integral_minus_amphr_span_if_minutes": (
            step4.get("steptime_integral_if_minutes_native_capacity")
            - numerator_capacity
            if step4
            and step4.get("steptime_integral_if_minutes_native_capacity") is not None
            and numerator_capacity is not None
            else None
        ),
        "integration_capacity_comparison_status": (
            "conditional_minutes_assumption_only_units_unverified"
            if step4
            else "not_computed"
        ),
        "step4_capacity_negative_deltas": (
            step4.get("capacity_negative_deltas") if step4 else None
        ),
        "step4_testtime_first_native": (
            step4.get("testtime_first_native") if step4 else None
        ),
        "step4_testtime_last_native": (
            step4.get("testtime_last_native") if step4 else None
        ),
        "step4_steptime_first_native": (
            step4.get("steptime_first_native") if step4 else None
        ),
        "step4_steptime_last_native": (
            step4.get("steptime_last_native") if step4 else None
        ),
        "step4_testtime_integral_native_current_time": (
            step4.get("testtime_integral_native_current_time") if step4 else None
        ),
        "step4_steptime_integral_native_current_time": (
            step4.get("steptime_integral_native_current_time") if step4 else None
        ),
        "step4_integral_if_minutes_native_capacity": (
            step4.get("testtime_integral_if_minutes_native_capacity")
            if step4
            else None
        ),
        "step4_time_unit_status": step4.get("time_unit_status", "") if step4 else "",
        "step4_temperature_column": step4.get("temperature_column", "") if step4 else "",
        "step4_temperature_column_status": (
            step4.get("temperature_column_status", "") if step4 else ""
        ),
        "mean_discharge_temperature_native": physics.get(
            "mean_discharge_temperature_native"
        ),
        "discharge_temperature_rise_native": physics.get(
            "discharge_temperature_rise_native"
        ),
        "discharge_plateau_slope_v_per_progress": physics.get(
            "discharge_plateau_slope_v_per_progress"
        ),
        "discharge_slope_variability_v_per_progress": physics.get(
            "discharge_slope_variability_v_per_progress"
        ),
        "mean_absolute_discharge_current_native": physics.get(
            "mean_absolute_discharge_current_native"
        ),
        "physics_status": physics_status,
        "physics_feature_units_status": (
            "temperature_and_current_units_unverified_voltage_native"
            if physics_structurally_complete
            else "not_complete"
        ),
        "temperature_available": temperature_available,
        "step10_segment_ref": observation.get("step10_segment_ref", ""),
        "modalities_aligned_by_physical_cell_and_rpt": (
            mapping_status == "verified"
            and bool(step4)
            and bool(step10)
            and step4.get("source_file") == step10.get("source_file")
            and candidate.get("_key") == (part, cell, rpt)
        ),
        "step10_validity": step10_status,
        "step10_reasons": "|".join(dict.fromkeys(step10_reasons)),
        "step10_row_count": step10.get("row_count") if step10 else None,
        "step10_voltage_first_v": step10.get("voltage_first_native") if step10 else None,
        "step10_voltage_last_v": step10.get("voltage_last_native") if step10 else None,
        "step10_voltage_min_v": step10.get("voltage_min_v") if step10 else None,
        "step10_voltage_max_v": step10.get("voltage_max_v") if step10 else None,
        "step10_current_signs": (
            f"negative={step10.get('current_negative_rows',0)};"
            f"zero={step10.get('current_zero_rows',0)};"
            f"positive={step10.get('current_positive_rows',0)}"
            if step10
            else ""
        ),
        "step10_capacity_span_native_amphr": (
            step10.get("capacity_span_native_amphr") if step10 else None
        ),
        "step10_testtime_integral_native_current_time": (
            step10.get("testtime_integral_native_current_time")
            if step10
            else None
        ),
        "step10_testtime_integral_if_minutes_native_capacity": (
            step10.get("testtime_integral_if_minutes_native_capacity")
            if step10
            else None
        ),
        "step10_testtime_integral_minus_amphr_span_if_minutes": (
            step10.get("testtime_integral_if_minutes_native_capacity")
            - step10.get("capacity_span_native_amphr")
            if step10
            and step10.get("testtime_integral_if_minutes_native_capacity") is not None
            and step10.get("capacity_span_native_amphr") is not None
            else None
        ),
        "step10_integration_capacity_comparison_status": (
            "conditional_minutes_assumption_only_units_unverified"
            if step10
            else "not_computed"
        ),
        "step10_testtime_negative_intervals": (
            step10.get("testtime_negative_intervals") if step10 else None
        ),
        "bol_reference_file": observation.get("bol_reference_file", ""),
        "bol_step4_segment_ref": observation.get("bol_step4_segment_ref", ""),
        "bol_resolution_status": bol_mapping_status,
        "bol_validity": bol_status,
        "bol_reasons": "|".join(dict.fromkeys(bol_reasons)),
        "bol_source_file": bol.get("source_file", "") if bol else "",
        "bol_capacity_span_native_amphr": denominator_capacity,
        "bol_protocol": protocol,
        "bol_source_part": expected_bol_part,
        "reference_protocol_compatibility": (
            "same_cell_step4_and_part_protocol_crosswalk"
            if bol_status == "ACCEPTED"
            else "unresolved_or_invalid"
        ),
        "candidate_label_source": candidate.get("label_source", ""),
        "candidate_label_status": (
            label.get("validation_status_v2", label.get("validation_status", ""))
            if label
            else ""
        ),
        "candidate_label_is_official_ground_truth": False if label else "",
        "candidate_soh_percent": label_value,
        "recomputed_soh_percent_from_raw_spans": recomputed_soh,
        "stored_capacity_span_native_amphr": label_capacity,
        "stored_bol_capacity_native_amphr": label_bol_capacity,
        "label_source_matches_guide": label_source_matches,
        "label_arithmetic_matches_raw_spans": arithmetic_matches,
        "label_disposition": label_disposition,
        "label_reasons": "|".join(dict.fromkeys(label_reasons)),
        "structurally_complete_modalities": structurally_complete_modalities,
        "accepted_soh_reference": label_disposition == "ACCEPTED",
        "primary_benchmark_eligible": final_eligible,
        "external_evaluation_eligible_pending_A3": external_eligible,
        "final_eligibility_status": final_status,
        "training_authorized": False,
        "instrument_by_part": "Maccor 4200" if part in {"1", "2"} else "Maccor 4000",
        "time_unit_policy": "native elapsed time retained; no conversion asserted",
        "reason_summary": "|".join(reasons),
        "step4_block_sha256": step4.get("raw_block_sha256", "") if step4 else "",
        "step10_block_sha256": step10.get("raw_block_sha256", "") if step10 else "",
        "bol_block_sha256": bol.get("raw_block_sha256", "") if bol else "",
    }


def _audit_rpt8_control(
    observations: dict[tuple[str, str, str], dict[str, str]],
    request_aliases: dict[str, str],
    block_results: dict[str, dict[str, Any]],
    step4_cutoff_tolerance_v: float | None,
) -> dict[str, Any]:
    key = ("1", "G1-Cell9", "8")
    observation = observations.get(key)
    if not observation:
        return {
            "present_in_guide_observations": False,
            "status": "REVIEW",
            "reason": "mapped_rpt8_observation_missing",
        }
    results: dict[str, Any] = {}
    for field_name, expected_step in (
        ("step4_segment_ref", 4),
        ("step10_segment_ref", 10),
    ):
        alias = f"RPT8_CONTROL|{field_name}"
        metrics = block_results.get(request_aliases.get(alias, ""))
        status, reasons = _block_validity(
            metrics,
            expected_step,
            step4_cutoff_tolerance_v=(
                step4_cutoff_tolerance_v if expected_step == 4 else None
            ),
        )
        results[f"step{expected_step}"] = {
            "validity": status,
            "reasons": reasons,
            "source_ref": observation[field_name],
            "row_count": metrics.get("row_count") if metrics else None,
            "voltage_min_v": metrics.get("voltage_min_v") if metrics else None,
            "voltage_max_v": metrics.get("voltage_max_v") if metrics else None,
            "capacity_span_native_amphr": (
                metrics.get("capacity_span_native_amphr") if metrics else None
            ),
            "block_sha256": metrics.get("raw_block_sha256", "") if metrics else "",
        }
    step4_rejected = results["step4"]["validity"] == "REJECTED"
    return {
        "present_in_guide_observations": True,
        "candidate_record": False,
        "physical_cell_id": key[1],
        "rpt_index": 8,
        "source_file": observation["source_file"],
        "previous_exclusion_preserved": True,
        "step4": results["step4"],
        "step10": results["step10"],
        "status": (
            "REJECTED_PRESERVED"
            if step4_rejected
            else "EXCLUDED_REVIEW_PRESERVED"
        ),
        "reason": (
            "Step4 does not meet the inherited 2.5 V cutoff tolerance; "
            "Step10 does not substitute for a discharge-capacity reference."
            if step4_rejected
            else "Prior exclusion retained; no new evidence re-admits this record."
        ),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_json(path: Path, value: Any) -> None:
    payload = json.dumps(
        _jsonable(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    _atomic_write_text(path, payload)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing M1 artifact: {path}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty M1 artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing M1 artifact: {path}")
    fieldnames = list(rows[0])
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _aggregate(
    manifest: list[dict[str, Any]],
    file_results: list[dict[str, Any]],
    rpt8: dict[str, Any],
    cell_to_split: dict[str, str],
) -> dict[str, Any]:
    split_summary: dict[str, Any] = {}
    for split in ("train", "validation", "external_test"):
        rows = [row for row in manifest if row["partition"] == split]
        labels = [
            row["candidate_soh_percent"]
            for row in rows
            if _finite_float(row.get("candidate_soh_percent")) is not None
        ]
        split_summary[split] = {
            "candidate_records": len(rows),
            "distinct_physical_cells": len({row["physical_cell_id"] for row in rows}),
            "step4_missing": sum(not row.get("step4_row_count") for row in rows),
            "step10_missing": sum(not row.get("step10_row_count") for row in rows),
            "reference_label_missing": sum(
                _finite_float(row.get("candidate_soh_percent")) is None
                for row in rows
            ),
            "bol_reference_missing": sum(
                not row.get("bol_source_file")
                or _finite_float(row.get("bol_capacity_span_native_amphr")) is None
                for row in rows
            ),
            "guide_mapping_unresolved": sum(
                row["guide_mapping_status"] != "verified" for row in rows
            ),
            "structurally_complete_modality_records": sum(
                bool(row["structurally_complete_modalities"]) for row in rows
            ),
            "records_with_accepted_soh_references": sum(
                bool(row["accepted_soh_reference"]) for row in rows
            ),
            "primary_benchmark_eligible": sum(
                bool(row["primary_benchmark_eligible"]) for row in rows
            ),
            "external_evaluation_eligible_pending_A3": sum(
                bool(row["external_evaluation_eligible_pending_A3"]) for row in rows
            ),
            "step4_invalid_or_incomplete": sum(
                row["step4_validity"] == "REJECTED" for row in rows
            ),
            "step4_review": sum(row["step4_validity"] == "REVIEW" for row in rows),
            "step10_invalid_or_incomplete": sum(
                row["step10_validity"] == "REJECTED" for row in rows
            ),
            "step10_review": sum(row["step10_validity"] == "REVIEW" for row in rows),
            "temperature_missing": sum(not bool(row["temperature_available"]) for row in rows),
            "physics_incomplete": sum(
                row["physics_status"] == "INCOMPLETE" for row in rows
            ),
            "label_disposition_counts": dict(
                Counter(row["label_disposition"] for row in rows)
            ),
            "candidate_label_count": len(labels),
            "candidate_label_distribution_percent": (
                {
                    "min": min(labels),
                    "median": statistics.median(labels),
                    "max": max(labels),
                }
                if labels
                else None
            ),
        }

    dispositions = Counter(row["label_disposition"] for row in manifest)
    per_cell = []
    for cell in sorted(cell_to_split):
        rows = [row for row in manifest if row["physical_cell_id"] == cell]
        per_cell.append(
            {
                "physical_cell_id": cell,
                "partition": cell_to_split[cell],
                "candidate_records": len(rows),
                "part1_candidates": sum(row["dataset_part"] == "1" for row in rows),
                "part2_candidates": sum(row["dataset_part"] == "2" for row in rows),
                "part3_candidates": sum(row["dataset_part"] == "3" for row in rows),
                "step4_missing": sum(not row.get("step4_row_count") for row in rows),
                "step10_missing": sum(not row.get("step10_row_count") for row in rows),
                "temperature_missing": sum(
                    not bool(row["temperature_available"]) for row in rows
                ),
                "reference_label_missing": sum(
                    _finite_float(row.get("candidate_soh_percent")) is None
                    for row in rows
                ),
                "step4_complete": sum(row["step4_validity"] == "ACCEPTED" for row in rows),
                "step10_complete": sum(row["step10_validity"] == "ACCEPTED" for row in rows),
                "physics_complete": sum(
                    row["physics_status"] == "STRUCTURALLY_COMPLETE_UNITS_UNVERIFIED"
                    for row in rows
                ),
                "accepted_soh": sum(row["label_disposition"] == "ACCEPTED" for row in rows),
                "review_soh": sum(row["label_disposition"] == "REVIEW" for row in rows),
                "rejected_soh": sum(row["label_disposition"] == "REJECTED" for row in rows),
                "primary_benchmark_eligible": sum(
                    bool(row["primary_benchmark_eligible"]) for row in rows
                ),
            }
        )

    reason_counts = Counter()
    for row in manifest:
        for reason in filter(None, row["reason_summary"].split("|")):
            reason_counts[reason] += 1
    feature_specs = {
        "mean_discharge_temperature_native": (
            "mean(Step4 temperature)",
            "Step 4",
            "native temperature unit; unverified",
        ),
        "discharge_temperature_rise_native": (
            "max(Step4 temperature) - first(Step4 temperature)",
            "Step 4",
            "native temperature unit; unverified",
        ),
        "discharge_plateau_slope_v_per_progress": (
            "OLS slope of resampled voltage on normalized row progress, 20%-80%",
            "Step 4",
            "native voltage per normalized progress; voltage unit unverified",
        ),
        "discharge_slope_variability_v_per_progress": (
            "1.4826 * MAD of resampled voltage derivatives, 10%-90%",
            "Step 4",
            "native voltage per normalized progress; voltage unit unverified",
        ),
        "mean_absolute_discharge_current_native": (
            "mean(abs(Step4 current))",
            "Step 4",
            "native current unit; unverified",
        ),
    }
    physics_feature_coverage = []
    for part in ("1", "2", "3"):
        part_rows = [row for row in manifest if row["dataset_part"] == part]
        for feature, (formula, source_step, unit_status) in feature_specs.items():
            available = sum(
                _finite_float(row.get(feature)) is not None for row in part_rows
            )
            physics_feature_coverage.append(
                {
                    "dataset_part": part,
                    "feature": feature,
                    "formula": formula,
                    "source_step": source_step,
                    "unit_status": unit_status,
                    "candidate_records": len(part_rows),
                    "available_records": available,
                    "missing_records": len(part_rows) - available,
                    "feature_status": (
                        "native_value_computable_units_unverified"
                        if available
                        else "unavailable_in_candidate_part"
                    ),
                }
            )
    return {
        "candidate_counts": {
            split: EXPECTED_CANDIDATES_BY_SPLIT[split]
            for split in ("train", "validation", "external_test")
        },
        "candidate_total": len(manifest),
        "split_summary": split_summary,
        "label_disposition_counts": {
            label: dispositions[label] for label in ("ACCEPTED", "REVIEW", "REJECTED")
        },
        "structurally_complete_modality_records": sum(
            bool(row["structurally_complete_modalities"]) for row in manifest
        ),
        "records_with_accepted_soh_references": sum(
            bool(row["accepted_soh_reference"]) for row in manifest
        ),
        "primary_benchmark_eligible_records": sum(
            bool(row["primary_benchmark_eligible"]) for row in manifest
        ),
        "external_evaluation_eligible_records_pending_A3": sum(
            bool(row["external_evaluation_eligible_pending_A3"]) for row in manifest
        ),
        "reason_counts": dict(reason_counts.most_common()),
        "per_cell_coverage": per_cell,
        "physics_feature_coverage": physics_feature_coverage,
        "source_files_scanned_or_reused": len(file_results),
        "rpt8_exclusion_control": rpt8,
        "training_authorized": False,
        "training_started": False,
        "degradation_fitting_started": False,
    }


def _markdown_report(
    summary: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    lines = [
        "# BaFuse M1 Raw Data Validation",
        "",
        f"- Generated UTC: `{metadata['generated_at_utc']}`",
        f"- Split version: `{metadata['split_version']}`",
        f"- Candidates: **{summary['candidate_total']}** (input candidate counts, not expected eligible counts)",
        "- Training authorization: `false`",
        "- Training started: `false`",
        "- Degradation fitting started: `false`",
        "",
        "## Split eligibility summary",
        "",
        "| Split | Candidate RPTs | Structural D/C/physics complete | Accepted references | Primary eligible | External eligible pending A3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "external_test"):
        row = summary["split_summary"][split]
        lines.append(
            "| {split} | {candidate_records} | {structurally_complete_modality_records} | "
            "{records_with_accepted_soh_references} | {primary_benchmark_eligible} | "
            "{external_evaluation_eligible_pending_A3} |".format(
                split=split,
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Modality and reference availability",
            "",
            "| Split | Step 4 missing | Step 10 missing | Temperature missing | Physics incomplete | Reference label missing | BoL reference missing | Guide mapping unresolved |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for split in ("train", "validation", "external_test"):
        row = summary["split_summary"][split]
        lines.append(
            "| {split} | {step4_missing} | {step10_missing} | {temperature_missing} | "
            "{physics_incomplete} | {reference_label_missing} | {bol_reference_missing} | "
            "{guide_mapping_unresolved} |".format(split=split, **row)
        )
    lines.extend(
        [
            "",
            "## Reference-label disposition",
            "",
            "| ACCEPTED | REVIEW | REJECTED |",
            "|---:|---:|---:|",
            "| {ACCEPTED} | {REVIEW} | {REJECTED} |".format(
                **summary["label_disposition_counts"]
            ),
            "",
            "## Validation policy and unresolved evidence",
            "",
            "- Guide identity and exact segment ranges are required; filename numbering is not used to assign an RPT.",
            "- Part 1/2 use the documented Maccor 4200 Step 4 CC-CV discharge protocol; Part 3 uses Maccor 4000 Step 4 CC discharge. Step 10 is audited as charge, not as the SOH numerator.",
            "- Candidate current integration is reported in native current × native elapsed-time units. A minutes-based conversion is emitted only as a conditional diagnostic; local source documentation does not establish elapsed-time or current units.",
            f"- The Step 4 endpoint tolerance is inherited from the frozen baseline preflight config (`{metadata.get('step4_cutoff_tolerance_v')}` V) for Parts 1-2 only; it is not asserted as an instrument accuracy. Part 3 remains REVIEW when it does not reach exactly 2.5 V because a Part 3 tolerance was not independently verified.",
            "- The integration comparison against native `Amphr` spans is conditional on elapsed time being minutes and current being in compatible units; it is diagnostic only and does not establish capacity units or label validity.",
            "- `Amphr` stays native and absolute units are unverified. Capacity ratios are recomputed from same-cell Step 4 spans only when mapped BoL blocks are available and protocol-compatible.",
            "- No instrument accuracy, calibration tolerance, sampling-gap threshold, or uncertainty bound is invented. Time interval summaries are descriptive.",
            "- Part 2 and Part 3 capacity-ratio references remain provisional and therefore REVIEW pending policy approval. Part 1 labels remain derived candidates; unresolved integration units keep them REVIEW.",
            "- Physics features use only the mapped same-RPT Step 4 block. Temperature/current are retained in native units; no missing values are filled from other cells, RPTs, or future measurements.",
            "- G1-Cell9 RPT8 is audited as an excluded control, not one of the 203 candidates.",
            "",
            "## Most frequent record-level reasons",
            "",
        ]
    )
    if summary["reason_counts"]:
        for reason, count in list(summary["reason_counts"].items())[:20]:
            lines.append(f"- `{reason}`: {count}")
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- Canonical manifest SHA-256: `{metadata.get('canonical_manifest_sha256', '')}`",
            f"- Candidate input manifest SHA-256: `{metadata['input_sha256']['pretraining_dataset_candidates.csv']}`",
            f"- External candidate manifest SHA-256: `{metadata['input_sha256']['external_part3_evaluation_candidates.csv']}`",
            f"- Block content hashes are included per Step 4, Step 10, and BoL reference.",
            "- Raw CSVs are opened read-only and streamed to the last mapped line needed for requested blocks; no full source collection is loaded into memory.",
            "- The resume cache is keyed by exact block reference and raw file size/mtime/ctime. Cache reuse is recorded in the machine-readable run metadata.",
            "",
            "This report is an M1 validation result only. It does not approve the manifest (A1), feasibility (A1b), training (A2), or external evaluation (A3).",
            "",
        ]
    )
    return "\n".join(lines)


def run_m1_audit(
    *,
    root: Path = ROOT,
    output_dir: Path = OUTPUT_DIR,
    resume: bool = False,
) -> dict[str, Any]:
    global ROOT
    ROOT = root.resolve()
    if output_dir.exists():
        existing = list(output_dir.iterdir())
        if existing and not resume:
            raise FileExistsError(
                f"Output directory is non-empty; use --resume only for an incomplete run: {output_dir}"
            )
        if (output_dir / "validation_report.json").exists():
            raise FileExistsError(
                f"Completed M1 outputs already exist and will not be overwritten: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    cell_to_split, split_version = _load_split()
    observations = _load_observations()
    labels = _load_labels()
    candidates = _load_candidate_rows(cell_to_split, observations, labels)
    step4_cutoff_tolerance_v = _configured_step4_cutoff_tolerance_v(
        BASELINE_CONFIG_PATH
    )
    requests_by_file, request_aliases = _build_requests(candidates, observations)
    cache = _BlockCache(output_dir / "block_validation_cache.sqlite")
    try:
        block_results, file_results = _scan_requested_blocks(
            requests_by_file,
            cache,
        )
    finally:
        cache.close()

    manifest = [
        _audit_one_candidate(
            row,
            request_aliases,
            block_results,
            step4_cutoff_tolerance_v,
        )
        for row in candidates
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    for row in manifest:
        row["validation_run_id"] = generated_at
        row["validator_version"] = VALIDATOR_VERSION
    rpt8 = _audit_rpt8_control(
        observations,
        request_aliases,
        block_results,
        step4_cutoff_tolerance_v,
    )
    summary = _aggregate(manifest, file_results, rpt8, cell_to_split)

    input_files = {
        "bafuse_rpt_observations.csv": OBSERVATIONS_PATH,
        "cell_split.csv": SPLIT_PATH,
        "pretraining_dataset_candidates.csv": TRAIN_CANDIDATES_PATH,
        "external_part3_evaluation_candidates.csv": EXTERNAL_CANDIDATES_PATH,
        "strict_part1_labels.csv": STRICT_PART1_LABELS_PATH,
        "part2_candidate_labels.csv": PART2_LABELS_PATH,
        "part3_candidate_labels.csv": PART3_LABELS_PATH,
        "dataset_review_README.md": DATASET_REVIEW_PATH,
        "part3_bol_crosswalk_README.md": PART3_BOL_CROSSWALK_README,
        "baseline_config.json": BASELINE_CONFIG_PATH,
        "rpt8_previous_audit.json": RPT8_PREVIOUS_AUDIT_PATH,
    }
    input_hashes = {
        name: _sha256_file(path)
        for name, path in input_files.items()
        if path.is_file()
    }
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "validator_version": VALIDATOR_VERSION,
        "validator_sha256": _sha256_file(Path(__file__)),
        "audit_name": "bafuse_m1_raw_data_validation",
        "generated_at_utc": generated_at,
        "python_version": sys.version,
        "split_version": split_version,
        "input_sha256": input_hashes,
        "candidate_total": len(manifest),
        "candidate_counts_by_split": EXPECTED_CANDIDATES_BY_SPLIT,
        "unique_source_files_requested": len(requests_by_file),
        "raw_block_count": len(block_results),
        "source_file_scan_results": file_results,
        "timebase_unit_verification_status": UNITS_STATUS,
        "current_unit_verification_status": UNITS_STATUS,
        "capacity_unit_verification_status": "native_Amphr_absolute_unit_unverified",
        "step4_cutoff_tolerance_v": step4_cutoff_tolerance_v,
        "step4_tolerance_provenance": (
            "inherited from baseline_config.json preflight_gates; not an instrument accuracy"
        ),
        "training_authorized": False,
        "training_started": False,
        "degradation_fitting_started": False,
        "external_data_used_for_model_selection": False,
        "cache_policy": (
            "Exact source block and resolved path keyed by file size, mtime_ns, and ctime_ns; "
            "block SHA-256 is stored with scanned results."
        ),
    }
    _write_csv(output_dir / "canonical_m1_manifest.csv", manifest)
    block_rows = list(block_results.values())
    _write_csv(output_dir / "measurement_block_audit.csv", block_rows)
    _write_csv(
        output_dir / "per_cell_coverage.csv",
        summary["per_cell_coverage"],
    )
    _write_csv(
        output_dir / "physics_feature_coverage.csv",
        summary["physics_feature_coverage"],
    )
    _atomic_write_json(output_dir / "eligibility_summary.json", summary)
    _atomic_write_json(output_dir / "run_metadata.json", metadata)
    metadata["canonical_manifest_sha256"] = _sha256_file(
        output_dir / "canonical_m1_manifest.csv"
    )
    _atomic_write_json(output_dir / "reproducibility.json", metadata)
    _atomic_write_text(
        output_dir / "validation_report.md",
        _markdown_report(summary, metadata),
    )
    _atomic_write_json(
        output_dir / "validation_report.json",
        {"summary": summary, "metadata": metadata},
    )
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only streaming validation for the BaFuse 203-record M1 candidate set."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="New M1 output directory; non-empty completed output is never overwritten.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse valid per-block cache entries from an incomplete output directory.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    summary = run_m1_audit(output_dir=args.output_dir, resume=args.resume)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
