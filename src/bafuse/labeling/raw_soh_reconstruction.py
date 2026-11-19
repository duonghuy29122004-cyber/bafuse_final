from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bafuse.data.raw_rpt_validation import OUTPUT_DIR as M1_OUTPUT_DIR
from bafuse.hybrid.soh_fusion import PHYSICS_FEATURE_NAMES, build_physics_features, resample_curve


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
OUTPUT_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
M1_MANIFEST = M1_OUTPUT_DIR / "canonical_m1_manifest.csv"
M1_BLOCK_AUDIT = M1_OUTPUT_DIR / "measurement_block_audit.csv"
CELL_SPLIT = BENCHMARK_DIR / "cell_split.csv"
RPT_OBSERVATIONS = PROJECT_ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
GUIDE_MANIFEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "experiment_manifests"
    / "guide_observation_manifest"
)
PART1_BOL_COMPARISON = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part1_bol_signal_review"
    / "part1_bol_curve_comparison.csv"
)

_SEGMENT_RE = re.compile(
    r"^(?P<path>.+)#Step(?P<step>\d+):segment[^:]*:rows"
    r"(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$",
    re.IGNORECASE,
)
_TIME_UNIT_EVIDENCE = (
    "TestTime and StepTime are treated as minutes because the Maccor export "
    "documentation identifies elapsed-time fields but omits their numeric "
    "unit, while independent trapezoidal integration of mapped Step 4 current "
    "under minutes agrees closely with the native Amphr span across Parts 1-3. "
    "The original export profile is unavailable; this is a strong, documented "
    "inference rather than a recovered export setting."
)
_CURRENT_UNIT_EVIDENCE = (
    "Amps is interpreted as A from Maccor export documentation and the "
    "documented C/2 protocol magnitude for nominal 3 Ah cells; observed Step 4 "
    "current is approximately 1.5 A."
)
_CAPACITY_UNIT_EVIDENCE = (
    "Maccor export documentation labels its capacity field Ah, but the Oxford "
    "Amphr-to-export-field mapping is not archived. The primary target uses "
    "integrated current times native elapsed time, so the common unit scale "
    "cancels within the compatible same-protocol numerator/BoL ratio. "
    "A-minute/60 Ah values are retained as an inferred consistency diagnostic, "
    "not an absolute-capacity claim."
)
_ENDPOINT_EVIDENCE = (
    "No voltage tolerance is applied. A Step 4 endpoint at or below the "
    "documented 2.5 V target is below the cutoff; an endpoint above it is "
    "accepted only if it exactly equals the closest above-target voltage "
    "observed in the mapped raw Step 4 samples and the following row changes "
    "Step ID. This is a data-derived endpoint rule, not an instrument-accuracy "
    "claim."
)


@dataclass(frozen=True)
class BlockRequest:
    source_file: str
    source_part: int
    step: int
    start_line: int
    end_line: int
    expected_sha256: str
    keep_voltage: bool
    keep_physics: bool
    keep_current: bool = False

    @property
    def key(self) -> tuple[str, int, int, int, int]:
        return (
            _normalize_source(self.source_file),
            self.source_part,
            self.step,
            self.start_line,
            self.end_line,
        )


@dataclass
class _BlockAccumulator:
    request: BlockRequest
    columns: dict[str, int]
    width: int
    digest: Any = field(default_factory=hashlib.sha256)
    row_count: int = 0
    wrong_width: int = 0
    missing: Counter[str] = field(default_factory=Counter)
    steps: set[int | None] = field(default_factory=set)
    current_signs: Counter[str] = field(default_factory=Counter)
    capacity_deltas: Counter[str] = field(default_factory=Counter)
    testtime_deltas: Counter[str] = field(default_factory=Counter)
    steptime_deltas: Counter[str] = field(default_factory=Counter)
    testtime_integral_raw: float = 0.0
    steptime_integral_raw: float = 0.0
    testtime_gaps: list[float] = field(default_factory=list)
    steptime_gaps: list[float] = field(default_factory=list)
    previous: dict[str, float | None] = field(
        default_factory=lambda: {
            "amps": None,
            "amphr": None,
            "testtime": None,
            "steptime": None,
        }
    )
    capacity_min: float | None = None
    capacity_max: float | None = None
    voltage_min: float | None = None
    voltage_max: float | None = None
    voltage_first: float | None = None
    voltage_last: float | None = None
    voltage_closest_below_2_5: float | None = None
    voltage_closest_above_2_5: float | None = None
    current_min: float | None = None
    current_max: float | None = None
    temperature_values: list[float] = field(default_factory=list)
    voltage_values: list[float] = field(default_factory=list)
    current_values: list[float] = field(default_factory=list)
    capacity_values: list[float] = field(default_factory=list)
    testtime_values: list[float] = field(default_factory=list)
    steptime_values: list[float] = field(default_factory=list)
    zero_current_rows: list[dict[str, Any]] = field(default_factory=list)
    positive_current_digest: Any = field(default_factory=hashlib.sha256)
    positive_current_row_count: int = 0

    def _number(self, row: list[str], name: str) -> float | None:
        index = self.columns.get(name)
        if index is None or index >= len(row):
            return None
        try:
            value = float(row[index])
        except (ValueError, TypeError):
            return None
        return value if math.isfinite(value) else None

    def update(self, row: list[str]) -> None:
        self.row_count += 1
        self.digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
        self.wrong_width += int(len(row) != self.width)

        previous_values = dict(self.previous)
        values = {
            name: self._number(row, name)
            for name in ("step", "testtime", "steptime", "amphr", "amps", "volts")
        }
        step_value = values["step"]
        if step_value is not None and step_value.is_integer():
            self.steps.add(int(step_value))
        else:
            self.steps.add(None)
            self.missing["step"] += 1

        for name in ("testtime", "steptime", "amphr", "amps", "volts"):
            if values[name] is None:
                self.missing[name] += 1

        current = values["amps"]
        capacity = values["amphr"]
        voltage = values["volts"]
        for name, value in (("amps", current), ("amphr", capacity)):
            previous = self.previous[name]
            if value is not None and previous is not None and name == "amphr":
                delta = value - previous
                self.capacity_deltas[
                    "negative" if delta < 0 else "positive" if delta > 0 else "zero"
                ] += 1
            if value is not None:
                self.previous[name] = value
            else:
                self.previous[name] = None

        if current is not None:
            self.current_signs[
                "negative" if current < 0 else "positive" if current > 0 else "zero"
            ] += 1
            self.current_min = current if self.current_min is None else min(self.current_min, current)
            self.current_max = current if self.current_max is None else max(self.current_max, current)
            if self.request.keep_physics or self.request.keep_current:
                self.current_values.append(current)
            if self.request.keep_current:
                if current > 0:
                    self.positive_current_digest.update(
                        ("\x1f".join(row) + "\n").encode("utf-8", errors="replace")
                    )
                    self.positive_current_row_count += 1
                elif current == 0:
                    self.zero_current_rows.append(
                        {
                            "line": self.request.start_line + self.row_count - 1,
                            "testtime": values["testtime"],
                            "steptime": values["steptime"],
                            "amphr": capacity,
                            "amps": current,
                            "volts": voltage,
                        }
                    )
        elif self.request.keep_current:
            self.current_values.append(math.nan)

        if self.request.keep_current:
            self.capacity_values.append(
                math.nan if capacity is None else capacity
            )
            self.testtime_values.append(
                math.nan if values["testtime"] is None else values["testtime"]
            )
            self.steptime_values.append(
                math.nan if values["steptime"] is None else values["steptime"]
            )

        if capacity is not None:
            self.capacity_min = capacity if self.capacity_min is None else min(self.capacity_min, capacity)
            self.capacity_max = capacity if self.capacity_max is None else max(self.capacity_max, capacity)

        if voltage is not None:
            self.voltage_min = voltage if self.voltage_min is None else min(self.voltage_min, voltage)
            self.voltage_max = voltage if self.voltage_max is None else max(self.voltage_max, voltage)
            if voltage <= 2.5:
                self.voltage_closest_below_2_5 = (
                    voltage
                    if self.voltage_closest_below_2_5 is None
                    else max(self.voltage_closest_below_2_5, voltage)
                )
            else:
                self.voltage_closest_above_2_5 = (
                    voltage
                    if self.voltage_closest_above_2_5 is None
                    else min(self.voltage_closest_above_2_5, voltage)
                )
            if self.voltage_first is None:
                self.voltage_first = voltage
            self.voltage_last = voltage
            if self.request.keep_voltage:
                self.voltage_values.append(voltage)

        if self.request.keep_physics and self.columns.get("temperature") is not None:
            temperature = self._number(row, "temperature")
            if temperature is not None:
                self.temperature_values.append(temperature)
            else:
                self.missing["temperature"] += 1
        elif self.request.keep_physics:
            self.missing["temperature"] += 1

        for name, integral_name, gap_name in (
            ("testtime", "testtime_integral_raw", "testtime_gaps"),
            ("steptime", "steptime_integral_raw", "steptime_gaps"),
        ):
            timestamp = values[name]
            previous_time = previous_values[name]
            if timestamp is not None and previous_time is not None:
                delta = timestamp - previous_time
                direction = "negative" if delta < 0 else "positive" if delta > 0 else "zero"
                getattr(self, f"{name}_deltas")[direction] += 1
                previous_current = previous_values["amps"]
                if delta > 0 and current is not None and previous_current is not None:
                    setattr(
                        self,
                        integral_name,
                        getattr(self, integral_name)
                        + (previous_current + current) * 0.5 * delta,
                    )
                    getattr(self, gap_name).append(delta)
            self.previous[name] = timestamp

    def finish(self, next_step: float | None) -> dict[str, Any]:
        capacity_span = (
            None
            if self.capacity_min is None or self.capacity_max is None
            else self.capacity_max - self.capacity_min
        )
        return {
            "source_file": self.request.source_file,
            "source_part": self.request.source_part,
            "step": self.request.step,
            "start_line": self.request.start_line,
            "end_line": self.request.end_line,
            "row_count": self.row_count,
            "expected_row_count": self.request.end_line - self.request.start_line + 1,
            "raw_block_sha256": self.digest.hexdigest(),
            "expected_raw_block_sha256": self.request.expected_sha256,
            "step_ids": sorted(self.steps, key=lambda value: -1 if value is None else value),
            "step_ids_match": self.steps == {self.request.step},
            "next_step_after_block": next_step,
            "next_row_transitions_step": next_step is not None and next_step != self.request.step,
            "wrong_width_rows": self.wrong_width,
            "missing_core_values": dict(self.missing),
            "current_signs": dict(self.current_signs),
            "current_min": self.current_min,
            "current_max": self.current_max,
            "capacity_span_native_amphr": capacity_span,
            "capacity_negative_deltas": self.capacity_deltas["negative"],
            "capacity_positive_deltas": self.capacity_deltas["positive"],
            "capacity_zero_deltas": self.capacity_deltas["zero"],
            "voltage_min_v": self.voltage_min,
            "voltage_max_v": self.voltage_max,
            "voltage_first_v": self.voltage_first,
            "voltage_last_v": self.voltage_last,
            "voltage_closest_below_2_5_v": self.voltage_closest_below_2_5,
            "voltage_closest_above_2_5_v": self.voltage_closest_above_2_5,
            "testtime_integral_native_current_time": self.testtime_integral_raw,
            "steptime_integral_native_current_time": self.steptime_integral_raw,
            "testtime_negative_intervals": self.testtime_deltas["negative"],
            "testtime_zero_intervals": self.testtime_deltas["zero"],
            "testtime_positive_intervals": self.testtime_deltas["positive"],
            "steptime_negative_intervals": self.steptime_deltas["negative"],
            "steptime_zero_intervals": self.steptime_deltas["zero"],
            "steptime_positive_intervals": self.steptime_deltas["positive"],
            "testtime_dt_min_native": min(self.testtime_gaps) if self.testtime_gaps else None,
            "testtime_dt_median_native": statistics.median(self.testtime_gaps) if self.testtime_gaps else None,
            "testtime_dt_max_native": max(self.testtime_gaps) if self.testtime_gaps else None,
            "steptime_dt_min_native": min(self.steptime_gaps) if self.steptime_gaps else None,
            "steptime_dt_median_native": statistics.median(self.steptime_gaps) if self.steptime_gaps else None,
            "steptime_dt_max_native": max(self.steptime_gaps) if self.steptime_gaps else None,
            "temperature_values": self.temperature_values,
            "voltage_values": self.voltage_values,
            "current_values": self.current_values,
            "capacity_values": self.capacity_values,
            "testtime_values": self.testtime_values,
            "steptime_values": self.steptime_values,
            "zero_current_rows": self.zero_current_rows,
            "positive_current_row_count": self.positive_current_row_count,
            "positive_current_rows_sha256": (
                self.positive_current_digest.hexdigest()
                if self.request.keep_current
                else None
            ),
        }


def _normalize_source(source: str) -> str:
    return source.replace("\\", "/").casefold()


def _resolve_source_name(source: str, part: int) -> str:
    normalized = source.replace("\\", "/")
    if normalized.casefold().startswith("data/raw/oxford/part"):
        return normalized
    return f"data/raw/oxford/part{part}/{normalized.lstrip('/')}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_segment(reference: str) -> tuple[str, int, int, int]:
    match = _SEGMENT_RE.fullmatch(reference.strip())
    if not match:
        raise ValueError(f"Unparseable Guide-mapped block reference: {reference!r}")
    return (
        match.group("path"),
        int(match.group("step")),
        int(match.group("start")),
        int(match.group("end")),
    )


def _block_key(
    source_file: str, source_part: int, step: int, start: int, end: int
) -> tuple[str, int, int, int, int]:
    return (_normalize_source(source_file), source_part, step, start, end)


def _build_block_requests(
    candidates: list[dict[str, str]],
    block_audit: list[dict[str, str]],
    rpt8_observations: list[dict[str, str]],
) -> tuple[dict[tuple[str, int, int, int, int], BlockRequest], dict[tuple[str, int, int, int, int], dict[str, str]]]:
    audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for row in block_audit:
        key = _block_key(
            row["source_file"],
            int(row["source_part"]),
            int(row["expected_step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        audit_by_key[key] = row

    requested: dict[tuple[str, int, int, int, int], BlockRequest] = {}

    def add(reference: str, part: int, expected_hash: str, keep_voltage: bool, keep_physics: bool) -> None:
        source, step, start, end = _parse_segment(reference)
        source = _resolve_source_name(source, part)
        key = _block_key(source, part, step, start, end)
        audit = audit_by_key.get(key)
        if audit is None:
            raise ValueError(f"M1 block audit has no exact match for {reference!r}.")
        if audit["raw_block_sha256"] != expected_hash:
            raise ValueError(f"M1 block-hash mismatch for {reference!r}.")
        prior = requested.get(key)
        request = BlockRequest(
            source_file=source,
            source_part=part,
            step=step,
            start_line=start,
            end_line=end,
            expected_sha256=expected_hash,
            keep_voltage=keep_voltage,
            keep_physics=keep_physics,
        )
        if prior is not None:
            request = BlockRequest(
                source_file=source,
                source_part=part,
                step=step,
                start_line=start,
                end_line=end,
                expected_sha256=expected_hash,
                keep_voltage=prior.keep_voltage or keep_voltage,
                keep_physics=prior.keep_physics or keep_physics,
            )
        requested[key] = request

    for row in candidates:
        part = int(row["dataset_part"])
        add(row["step4_segment_ref"], part, row["step4_block_sha256"], True, True)
        add(row["step10_segment_ref"], part, row["step10_block_sha256"], True, False)
        add(row["bol_step4_segment_ref"], int(row["bol_source_part"]), row["bol_block_sha256"], False, False)

    for row in rpt8_observations:
        if row.get("part") != "1" or row.get("physical_cell_id") != "G1-Cell9" or row.get("rpt_index") != "8":
            continue
        for field_name, step in (("step4_segment_ref", 4), ("step10_segment_ref", 10)):
            reference = row.get(field_name, "")
            if not reference:
                raise ValueError(f"G1-Cell9 RPT8 lacks its Guide-mapped {field_name}.")
            source, parsed_step, start, end = _parse_segment(reference)
            if parsed_step != step:
                raise ValueError(f"G1-Cell9 RPT8 has an unexpected {field_name} step.")
            source = _resolve_source_name(source, 1)
            key = _block_key(source, 1, step, start, end)
            audit = audit_by_key.get(key)
            if audit is None:
                raise ValueError(f"M1 block audit lacks G1-Cell9 RPT8 {field_name}.")
            requested[key] = BlockRequest(
                source,
                1,
                step,
                start,
                end,
                audit["raw_block_sha256"],
                keep_voltage=False,
                keep_physics=False,
            )
    return requested, audit_by_key


def _scan_raw_blocks(
    requests: dict[tuple[str, int, int, int, int], BlockRequest],
    audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> dict[tuple[str, int, int, int, int], dict[str, Any]]:
    by_source: defaultdict[str, list[BlockRequest]] = defaultdict(list)
    for request in requests.values():
        by_source[request.source_file].append(request)
    output: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}

    ordered_sources = sorted(by_source.items())
    for source_index, (source_name, source_requests) in enumerate(ordered_sources, start=1):
        print(
            f"Scanning raw mapped sources {source_index}/{len(ordered_sources)}: {source_name}",
            flush=True,
        )
        source_path = PROJECT_ROOT / Path(source_name.replace("/", "\\"))
        if not source_path.is_file():
            raise FileNotFoundError(f"Mapped raw source file is missing: {source_path}")
        source_requests.sort(key=lambda item: (item.start_line, item.end_line))
        before = source_path.stat()
        audit_rows = [
            audit_by_key[request.key]
            for request in source_requests
        ]
        if any(int(row["source_size_bytes"]) != before.st_size for row in audit_rows):
            raise ValueError(f"Raw source size changed since M1: {source_name}")
        if any(int(row["source_mtime_ns"]) != before.st_mtime_ns for row in audit_rows):
            raise ValueError(f"Raw source mtime changed since M1: {source_name}")

        accumulators: dict[tuple[str, int, int, int, int], _BlockAccumulator] = {}
        start_map: defaultdict[int, list[BlockRequest]] = defaultdict(list)
        max_end = max(request.end_line for request in source_requests)
        active: list[BlockRequest] = []
        next_steps: dict[tuple[str, int, int, int, int], float | None] = {}

        with source_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, None)
            if not header:
                raise ValueError(f"Mapped raw source has no header: {source_name}")
            folded = [name.strip().casefold() for name in header]
            columns: dict[str, int] = {}
            for name in ("step", "testtime", "steptime", "amphr", "amps", "volts"):
                if folded.count(name) != 1:
                    raise ValueError(
                        f"Expected exactly one {name!r} in {source_name}, "
                        f"found {folded.count(name)}."
                    )
                columns[name] = folded.index(name)
            temperature_candidates = (
                ("temp1", "temp", "temperature")
                if source_requests[0].source_part == 1
                else ("temp", "temp1", "temperature")
            )
            temperature_name = next(
                (name for name in temperature_candidates if folded.count(name) == 1),
                None,
            )
            if temperature_name is not None:
                columns["temperature"] = folded.index(temperature_name)
            for request in source_requests:
                start_map[request.start_line].append(request)

            for row in reader:
                line_number = reader.line_num
                active.extend(start_map.get(line_number, ()))
                if line_number > max_end + 1:
                    break
                step_number: float | None = None
                step_index = columns["step"]
                if step_index < len(row):
                    try:
                        parsed = float(row[step_index])
                        step_number = parsed if math.isfinite(parsed) else None
                    except (TypeError, ValueError):
                        pass
                for request in active:
                    if line_number <= request.end_line:
                        accumulator = accumulators.get(request.key)
                        if accumulator is None:
                            accumulator = _BlockAccumulator(
                                request=request,
                                columns=columns,
                                width=len(header),
                            )
                            accumulators[request.key] = accumulator
                        accumulator.update(row)
                    elif line_number == request.end_line + 1:
                        next_steps[request.key] = step_number
                active = [
                    request
                    for request in active
                    if line_number <= request.end_line
                ]
                if line_number >= max_end + 1:
                    break

        after = source_path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError(f"Raw source changed while scanning: {source_name}")

        for request in source_requests:
            accumulator = accumulators.get(request.key)
            if accumulator is None:
                raise ValueError(f"Mapped source block was not reached: {request}")
            result = accumulator.finish(next_steps.get(request.key))
            audit = audit_by_key[request.key]
            if (
                result["row_count"] != result["expected_row_count"]
                or result["raw_block_sha256"] != request.expected_sha256
                or result["raw_block_sha256"] != audit["raw_block_sha256"]
            ):
                raise ValueError(
                    f"Raw block hash/row-count mismatch: {source_name} "
                    f"Step {request.step} lines {request.start_line}-{request.end_line}"
                )
            output[request.key] = result
    return output


def _block_for_reference(
    reference: str,
    part: int,
    raw_blocks: dict[tuple[str, int, int, int, int], dict[str, Any]],
) -> dict[str, Any]:
    source, step, start, end = _parse_segment(reference)
    source = _resolve_source_name(source, part)
    key = _block_key(source, part, step, start, end)
    try:
        return raw_blocks[key]
    except KeyError as error:
        raise ValueError(f"Raw re-read did not produce the mapped block {reference!r}.") from error


def _observed_cutoff_bracket(
    raw_blocks: dict[tuple[str, int, int, int, int], dict[str, Any]],
) -> tuple[float, float]:
    step4_blocks = [block for block in raw_blocks.values() if block["step"] == 4]
    below = [
        block["voltage_closest_below_2_5_v"]
        for block in step4_blocks
        if block["voltage_closest_below_2_5_v"] is not None
    ]
    above = [
        block["voltage_closest_above_2_5_v"]
        for block in step4_blocks
        if block["voltage_closest_above_2_5_v"] is not None
    ]
    if not below or not above:
        raise ValueError("Mapped raw Step 4 samples do not bracket the 2.5 V cutoff.")
    return max(below), min(above)


def _capacity_audit(block: dict[str, Any]) -> dict[str, Any]:
    q_testtime_native = abs(float(block["testtime_integral_native_current_time"]))
    q_steptime_native = abs(float(block["steptime_integral_native_current_time"]))
    q_testtime = q_testtime_native / 60.0
    q_steptime = q_steptime_native / 60.0
    span = _number(block["capacity_span_native_amphr"])
    testtime_pct = (
        None
        if span in (None, 0)
        else 100.0 * abs(q_testtime - span) / span
    )
    steptime_pct = (
        None
        if span in (None, 0)
        else 100.0 * abs(q_steptime - span) / span
    )
    return {
        "integrated_capacity_testtime_native_current_time": q_testtime_native,
        "integrated_capacity_steptime_native_current_time": q_steptime_native,
        "integrated_capacity_testtime_ah_inferred": q_testtime,
        "integrated_capacity_steptime_ah_inferred": q_steptime,
        "native_amphr_capacity_span": span,
        "testtime_vs_native_relative_discrepancy_percent": testtime_pct,
        "steptime_vs_native_relative_discrepancy_percent": steptime_pct,
        "testtime_vs_steptime_abs_difference_ah": abs(q_testtime - q_steptime),
    }


def _block_validity(
    block: dict[str, Any],
    *,
    expected_step: int,
    expected_current_sign: str,
    allow_zero_current: bool = False,
    require_endpoint: bool = False,
    closest_above_cutoff_v: float | None = None,
) -> tuple[bool, list[str]]:
    if allow_zero_current and (
        expected_step != 10 or expected_current_sign != "positive"
    ):
        raise ValueError(
            "Zero-current exclusion is supported only for charge Step 10."
        )
    reasons: list[str] = []
    if block["row_count"] != block["expected_row_count"]:
        reasons.append("raw_block_row_count_mismatch")
    if block["raw_block_sha256"] != block["expected_raw_block_sha256"]:
        reasons.append("raw_block_hash_mismatch")
    if not block["step_ids_match"] or block["step"] != expected_step:
        reasons.append("unexpected_or_mixed_step_id")
    if not block["next_row_transitions_step"]:
        reasons.append("step_block_boundary_not_confirmed")
    if block["wrong_width_rows"]:
        reasons.append("row_width_mismatch")
    if any(block["missing_core_values"].get(name, 0) for name in ("step", "testtime", "steptime", "amphr", "amps", "volts")):
        reasons.append("missing_or_nonfinite_core_measurement")
    directed_rows = block["current_signs"].get(expected_current_sign, 0)
    if allow_zero_current:
        directed_rows += block["current_signs"].get("zero", 0)
    if directed_rows != block["row_count"]:
        reasons.append("current_direction_not_consistent")
    if block["capacity_span_native_amphr"] is None or block["capacity_span_native_amphr"] <= 0:
        reasons.append("invalid_native_capacity_span")
    if block["capacity_negative_deltas"]:
        reasons.append("capacity_counter_reversal")
    if block["testtime_negative_intervals"] or block["steptime_negative_intervals"]:
        reasons.append("time_axis_reset_or_negative_interval")
    if block["testtime_zero_intervals"] or block["steptime_zero_intervals"]:
        reasons.append("repeated_timestamp_interval")
    if require_endpoint and (
        block["voltage_first_v"] is None
        or block["voltage_first_v"] <= 2.5
        or block["voltage_last_v"] is None
        or (
            block["voltage_last_v"] > 2.5
            and (
                closest_above_cutoff_v is None
                or block["voltage_last_v"] != closest_above_cutoff_v
            )
        )
    ):
        reasons.append("step4_endpoint_not_supported_by_observed_cutoff_sample")
    return not reasons, reasons


def _protocol_compatible(row: dict[str, str]) -> tuple[bool, str]:
    part = int(row["dataset_part"])
    bol_part = int(row["bol_source_part"])
    cell_match = re.fullmatch(r"G(?P<group>\d+)-Cell\d+", row["physical_cell_id"], re.IGNORECASE)
    if cell_match is None:
        return False, "Physical-cell group is not parseable from the canonical cell identity."
    if row["bol_resolution_status"] != "resolved":
        return False, "Guide-mapped BoL identity is not resolved."
    if row["reference_protocol_compatibility"] != "same_cell_step4_and_part_protocol_crosswalk":
        return False, "BoL reference protocol compatibility is unresolved or invalid."
    expected_bol_part = 3 if part == 3 else 1
    if bol_part != expected_bol_part:
        return False, f"Expected same-protocol BoL from Part {expected_bol_part}."
    if part == 2 and int(cell_match.group("group")) not in {1, 2, 3, 4}:
        return False, "Part 2 cell is outside the documented Part 1 continuation groups."
    if part == 3 and bol_part != part:
        return False, "Part 3 must use its separate Part 3 BoL."
    protocol = row["bol_protocol"]
    expected_instrument = "Maccor 4000" if part == 3 else "Maccor 4200"
    if expected_instrument not in protocol or "Step 4" not in protocol or "2.5 V" not in protocol:
        return False, "BoL protocol text does not match the dataset-part Step 4 protocol."
    return True, "Guide-mapped same-cell Step 4 BoL and dataset-part protocol match."


def _load_secondary_bol() -> dict[str, dict[str, str]]:
    if not PART1_BOL_COMPARISON.is_file():
        raise FileNotFoundError(f"Part 1 secondary BoL comparison is missing: {PART1_BOL_COMPARISON}")
    rows = _read_csv(PART1_BOL_COMPARISON)
    selected: dict[str, dict[str, str]] = {}
    for row in rows:
        if row.get("first_segment_index") != "1":
            continue
        cell = row["physical_cell_id"]
        if cell in selected:
            raise ValueError(f"Duplicate secondary BoL candidate for {cell}.")
        selected[cell] = row
    return selected


def _format_float(value: Any) -> str:
    number = _number(value)
    return "" if number is None else format(number, ".15g")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(dict.fromkeys(name for row in rows for name in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _target_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [
        float(row["reconstructed_soh_percent"])
        for row in rows
        if row["label_disposition"] == "ACCEPTED"
        and row["reconstructed_soh_percent"] != ""
        and row["partition"] in {"train", "validation", "external_test"}
    ]
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "min_percent": min(values),
        "max_percent": max(values),
        "mean_percent": statistics.mean(values),
        "median_percent": statistics.median(values),
        "stdev_percent": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def reconstruct(
    *,
    m1_manifest: Path = M1_MANIFEST,
    m1_block_audit: Path = M1_BLOCK_AUDIT,
    split_path: Path = CELL_SPLIT,
    output_dir: Path = OUTPUT_DIR,
) -> dict[str, Any]:
    candidates = _read_csv(m1_manifest)
    block_audit = _read_csv(m1_block_audit)
    split_rows = _read_csv(split_path)
    observations = _read_csv(RPT_OBSERVATIONS)
    if len(candidates) != 203:
        raise ValueError(f"Expected 203 M1 candidates; found {len(candidates)}.")

    assignments: dict[str, str] = {}
    for row in split_rows:
        cell = row["physical_cell_id"]
        if cell in assignments:
            raise ValueError(f"Duplicate physical-cell split assignment: {cell}")
        assignments[cell] = row["partition"]
    for row in candidates:
        if assignments.get(row["physical_cell_id"]) != row["partition"]:
            raise ValueError(
                f"Candidate split mismatch for {row['physical_cell_id']} RPT{row['rpt_index']}."
            )

    requests, audit_by_key = _build_block_requests(candidates, block_audit, observations)
    raw_blocks = _scan_raw_blocks(requests, audit_by_key)
    cutoff_below_v, cutoff_above_v = _observed_cutoff_bracket(raw_blocks)
    source_hashes = {
        str(path): _sha256(path)
        for path in (
            m1_manifest,
            m1_block_audit,
            split_path,
            RPT_OBSERVATIONS,
            GUIDE_MANIFEST_DIR / "guide_observation_manifest.csv",
            GUIDE_MANIFEST_DIR / "summary.json",
            PART1_BOL_COMPARISON,
        )
    }
    guide_summary = json.loads(
        (GUIDE_MANIFEST_DIR / "summary.json").read_text(encoding="utf-8")
    )
    secondary_bol = _load_secondary_bol()

    reconstructed: list[dict[str, Any]] = []
    final_manifest: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    integration_rows_by_key: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}
    bol_rows_by_key: dict[tuple[str, int, int, int, int], dict[str, Any]] = {}

    for row in candidates:
        part = int(row["dataset_part"])
        discharge = _block_for_reference(row["step4_segment_ref"], part, raw_blocks)
        charge = _block_for_reference(row["step10_segment_ref"], part, raw_blocks)
        bol = _block_for_reference(
            row["bol_step4_segment_ref"], int(row["bol_source_part"]), raw_blocks
        )
        discharge_ok, discharge_reasons = _block_validity(
            discharge,
            expected_step=4,
            expected_current_sign="negative",
            require_endpoint=True,
            closest_above_cutoff_v=cutoff_above_v,
        )
        charge_ok, charge_reasons = _block_validity(
            charge, expected_step=10, expected_current_sign="positive"
        )
        bol_ok, bol_reasons = _block_validity(
            bol,
            expected_step=4,
            expected_current_sign="negative",
            require_endpoint=True,
            closest_above_cutoff_v=cutoff_above_v,
        )

        protocol_ok, protocol_note = _protocol_compatible(row)
        if not protocol_ok:
            bol_reasons.append("incompatible_or_unresolved_bol_protocol")

        physics_values: dict[str, float] = {}
        physics_ok = False
        if (
            len(discharge["voltage_values"]) >= 2
            and len(discharge["current_values"]) == discharge["row_count"]
            and len(discharge["temperature_values"]) == discharge["row_count"]
        ):
            try:
                physics_values = build_physics_features(
                    discharge["voltage_values"],
                    discharge["current_values"],
                    discharge["temperature_values"],
                )
                physics_ok = all(math.isfinite(value) for value in physics_values.values())
            except ValueError:
                physics_ok = False
        if not physics_ok:
            discharge_reasons.append("physics_features_unavailable_or_nonfinite")

        discharge_q = _capacity_audit(discharge)
        bol_q = _capacity_audit(bol)
        label = ""
        label_disposition = "REVIEW"
        label_reasons: list[str] = []
        if not discharge_ok:
            label_reasons.extend(discharge_reasons)
        if not bol_ok:
            label_reasons.extend(f"bol_{reason}" for reason in bol_reasons)
        if not protocol_ok:
            label_reasons.append("bol_protocol_compatibility_unresolved")
        if not label_reasons:
            denominator = bol_q["integrated_capacity_testtime_native_current_time"]
            if denominator <= 0:
                label_reasons.append("nonpositive_integrated_bol_capacity")
            else:
                label = (
                    100.0
                    * discharge_q["integrated_capacity_testtime_native_current_time"]
                    / denominator
                )
                label_disposition = "ACCEPTED"
        if label_reasons:
            label_disposition = "REVIEW"

        modality_reasons = []
        if not discharge_ok:
            modality_reasons.append("step4_discharge_invalid")
        if not charge_ok:
            modality_reasons.append("step10_charge_invalid")
        if not physics_ok:
            modality_reasons.append("physics_features_invalid")
        eligible = label_disposition == "ACCEPTED" and not modality_reasons
        final_status = (
            "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
            if eligible
            else "REVIEW_OR_INCOMPLETE"
        )
        result = {
            "dataset_part": part,
            "physical_cell_id": row["physical_cell_id"],
            "rpt_index": row["rpt_index"],
            "measurement_id": row["measurement_id"],
            "partition": row["partition"],
            "source_file": row["source_file"],
            "step4_segment_ref": row["step4_segment_ref"],
            "step10_segment_ref": row["step10_segment_ref"],
            "bol_source_file": row["bol_source_file"],
            "bol_step4_segment_ref": row["bol_step4_segment_ref"],
            "bol_source_part": row["bol_source_part"],
            "bol_resolution_status": row["bol_resolution_status"],
            "reference_protocol_compatibility": row["reference_protocol_compatibility"],
            "protocol_compatibility_note": protocol_note,
            "protocol_identity": row["bol_protocol"],
            "q_rpt_integrated_testtime_native_current_time": discharge_q["integrated_capacity_testtime_native_current_time"],
            "q_rpt_integrated_steptime_native_current_time": discharge_q["integrated_capacity_steptime_native_current_time"],
            "q_rpt_integrated_testtime_ah_inferred": discharge_q["integrated_capacity_testtime_ah_inferred"],
            "q_rpt_integrated_steptime_ah_inferred": discharge_q["integrated_capacity_steptime_ah_inferred"],
            "q_rpt_native_amphr_span": discharge_q["native_amphr_capacity_span"],
            "q_bol_integrated_testtime_native_current_time": bol_q["integrated_capacity_testtime_native_current_time"],
            "q_bol_integrated_steptime_native_current_time": bol_q["integrated_capacity_steptime_native_current_time"],
            "q_bol_integrated_testtime_ah_inferred": bol_q["integrated_capacity_testtime_ah_inferred"],
            "q_bol_integrated_steptime_ah_inferred": bol_q["integrated_capacity_steptime_ah_inferred"],
            "q_bol_native_amphr_span": bol_q["native_amphr_capacity_span"],
            "reconstructed_soh_percent": label,
            "capacity_extraction_method": "trapezoidal_abs_current_times_native_elapsed_time; numerator_and_bol_share_compatible_time_scale",
            "time_unit_evidence": "minutes_strongly_inferred;not_an_archived_export_setting",
            "current_unit_evidence": "A_documented_and_C_over_2_consistent",
            "capacity_unit_evidence": "native_integrated_current_time_ratio;Ah_conversion_retained_as_inferred_diagnostic",
            "discharge_measurement_validity": "ACCEPTED" if discharge_ok else "REVIEW",
            "charge_measurement_validity": "ACCEPTED" if charge_ok else "REVIEW",
            "bol_measurement_validity": "ACCEPTED" if bol_ok and protocol_ok else "REVIEW",
            "physics_validity": "STRUCTURALLY_COMPLETE" if physics_ok else "REVIEW",
            "physics_unit_status": (
                "voltage_V_documented;current_A_inferred;temperature_native_unresolved"
            ),
            "label_disposition": label_disposition,
            "label_reasons": ";".join(dict.fromkeys(label_reasons)),
            "modality_reasons": ";".join(modality_reasons),
            "final_eligibility_status": final_status,
            "historical_soh_percent_for_comparison_only": row["candidate_soh_percent"],
            "historical_label_is_target": False,
            "step4_block_sha256": discharge["raw_block_sha256"],
            "step10_block_sha256": charge["raw_block_sha256"],
            "bol_block_sha256": bol["raw_block_sha256"],
        }
        reconstructed.append(result)
        final_manifest.append(
            {
                **result,
                "candidate_record": True,
                "source_guide_stage": row["guide_stage"],
                "source_tpg": row["source_tpg"],
                "training_authorized": False,
            }
        )
        provenance.append(
            {
                "dataset_part": part,
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "partition": row["partition"],
                "guide_stage": row["guide_stage"],
                "guide_mapping_status": row["guide_mapping_status"],
                "source_file": row["source_file"],
                "step4_segment_ref": row["step4_segment_ref"],
                "step4_block_sha256": discharge["raw_block_sha256"],
                "step10_segment_ref": row["step10_segment_ref"],
                "step10_block_sha256": charge["raw_block_sha256"],
                "bol_source_part": row["bol_source_part"],
                "bol_source_file": row["bol_source_file"],
                "bol_resolution_status": row["bol_resolution_status"],
                "reference_protocol_compatibility": row["reference_protocol_compatibility"],
                "protocol_compatibility_note": protocol_note,
                "bol_step4_segment_ref": row["bol_step4_segment_ref"],
                "bol_block_sha256": bol["raw_block_sha256"],
                "bol_protocol": row["bol_protocol"],
                "time_unit_evidence": _TIME_UNIT_EVIDENCE,
                "current_unit_evidence": _CURRENT_UNIT_EVIDENCE,
                "capacity_unit_evidence": _CAPACITY_UNIT_EVIDENCE,
                "endpoint_evidence": _ENDPOINT_EVIDENCE,
                "mapped_step4_closest_observed_voltage_below_cutoff_v": cutoff_below_v,
                "mapped_step4_closest_observed_voltage_above_cutoff_v": cutoff_above_v,
                "historical_candidate_soh_percent_comparison_only": row["candidate_soh_percent"],
                "historical_soh_used_as_target": False,
                "m1_label_source": row["candidate_label_source"],
            }
        )

        feature_row: dict[str, Any] = {
            "dataset_part": part,
            "physical_cell_id": row["physical_cell_id"],
            "rpt_index": row["rpt_index"],
            "partition": row["partition"],
            "step4_voltage_units": "V",
            "step10_voltage_units": "V",
            "temperature_units": "native_unresolved",
            "current_units": "A_inferred",
            "feature_provenance": "same_RPT_raw_Step4_and_Step10; normalized_row_progress",
            "feature_target_leakage_fields_excluded": "capacity;Amphr;duration;RPT_index;cell_ID;part_ID;future_data",
            "physics_validity": "STRUCTURALLY_COMPLETE" if physics_ok else "REVIEW",
            "physics_unit_status": (
                "voltage_V_documented;current_A_inferred;temperature_native_unresolved"
            ),
        }
        if len(discharge["voltage_values"]) >= 2:
            feature_row.update(
                {
                    f"discharge_v_{index:03d}": value
                    for index, value in enumerate(resample_curve(discharge["voltage_values"]))
                }
            )
        if len(charge["voltage_values"]) >= 2:
            feature_row.update(
                {
                    f"charge_v_{index:03d}": value
                    for index, value in enumerate(resample_curve(charge["voltage_values"]))
                }
            )
        for name in PHYSICS_FEATURE_NAMES:
            feature_row[name] = physics_values.get(name, "")
        feature_rows.append(feature_row)

        for block, role in ((discharge, "rpt_step4"), (bol, "bol_step4")):
            key = _block_key(
                block["source_file"], int(block["source_part"]), 4, int(block["start_line"]), int(block["end_line"])
            )
            if key not in integration_rows_by_key:
                audit = _capacity_audit(block)
                integration_rows_by_key[key] = {
                    "block_role": role,
                    "dataset_part": block["source_part"],
                    "physical_cell_id": row["physical_cell_id"] if role == "bol_step4" else "",
                    "source_file": block["source_file"],
                    "step": 4,
                    "start_line": block["start_line"],
                    "end_line": block["end_line"],
                    "row_count": block["row_count"],
                    "raw_block_sha256": block["raw_block_sha256"],
                    "expected_raw_block_sha256": block["expected_raw_block_sha256"],
                    "current_signs": json.dumps(block["current_signs"], sort_keys=True),
                    "testtime_negative_intervals": block["testtime_negative_intervals"],
                    "testtime_zero_intervals": block["testtime_zero_intervals"],
                    "steptime_negative_intervals": block["steptime_negative_intervals"],
                    "steptime_zero_intervals": block["steptime_zero_intervals"],
                    "testtime_dt_min_native": block["testtime_dt_min_native"],
                    "testtime_dt_median_native": block["testtime_dt_median_native"],
                    "testtime_dt_max_native": block["testtime_dt_max_native"],
                    "steptime_dt_min_native": block["steptime_dt_min_native"],
                    "steptime_dt_median_native": block["steptime_dt_median_native"],
                    "steptime_dt_max_native": block["steptime_dt_max_native"],
                    "native_capacity_span": block["capacity_span_native_amphr"],
                    **audit,
                }

        bol_key = _block_key(
            bol["source_file"], int(bol["source_part"]), 4, int(bol["start_line"]), int(bol["end_line"])
        )
        if bol_key not in bol_rows_by_key:
            bol_rows_by_key[bol_key] = {
                "physical_cell_id": row["physical_cell_id"],
                "source_part": row["bol_source_part"],
                "source_file": row["bol_source_file"],
                "source_block_ref": row["bol_step4_segment_ref"],
                "protocol": row["bol_protocol"],
                "q_bol_integrated_testtime_native_current_time": bol_q["integrated_capacity_testtime_native_current_time"],
                "q_bol_integrated_steptime_native_current_time": bol_q["integrated_capacity_steptime_native_current_time"],
                "q_bol_integrated_testtime_ah_inferred": bol_q["integrated_capacity_testtime_ah_inferred"],
                "q_bol_integrated_steptime_ah_inferred": bol_q["integrated_capacity_steptime_ah_inferred"],
                "q_bol_native_amphr_span": bol_q["native_amphr_capacity_span"],
                "bol_measurement_status": "ACCEPTED" if bol_ok and protocol_ok else "REVIEW",
                "bol_reasons": ";".join(bol_reasons),
                "block_sha256": bol["raw_block_sha256"],
                "reference_role": "primary_label_denominator",
            }

    # Preserve the second Part 1 BoL procedure as a QC-only sensitivity source.
    secondary_rows: list[dict[str, Any]] = []
    for candidate in reconstructed:
        if int(candidate["dataset_part"]) not in {1, 2}:
            continue
        comparison = secondary_bol.get(candidate["physical_cell_id"])
        if comparison is None:
            continue
        primary_span = _number(candidate["q_bol_native_amphr_span"])
        secondary_span = _number(comparison.get("second_capacity_span"))
        rpt_span = _number(candidate["q_rpt_native_amphr_span"])
        primary_ratio = (
            ""
            if primary_span in (None, 0) or rpt_span is None
            else 100.0 * rpt_span / primary_span
        )
        secondary_ratio = (
            ""
            if secondary_span in (None, 0) or rpt_span is None
            else 100.0 * rpt_span / secondary_span
        )
        secondary_rows.append(
            {
                "dataset_part": candidate["dataset_part"],
                "physical_cell_id": candidate["physical_cell_id"],
                "rpt_index": candidate["rpt_index"],
                "partition": candidate["partition"],
                "primary_bol_source_file": candidate["bol_source_file"],
                "primary_bol_capacity_span_native_amphr": primary_span,
                "secondary_bol_procedure": comparison["second_procedure"],
                "secondary_bol_source_file": comparison["second_source_file"],
                "secondary_segment_index": comparison["second_segment_index"],
                "secondary_bol_capacity_span_native_amphr": secondary_span,
                "rpt_capacity_span_native_amphr": rpt_span,
                "primary_bol_native_counter_soh_percent_qc": primary_ratio,
                "secondary_bol_native_counter_soh_percent_qc": secondary_ratio,
                "secondary_minus_primary_soh_pp_qc": (
                    ""
                    if primary_ratio == "" or secondary_ratio == ""
                    else secondary_ratio - primary_ratio
                ),
                "used_for_primary_label": False,
                "interpretation": "QC/sensitivity only; not an additional chronological RPT or target denominator",
            }
        )

    # Add the known excluded RPT8 and missing external-cell mapping without
    # manufacturing candidate labels.
    rpt8_observation = next(
        row
        for row in observations
        if row.get("part") == "1"
        and row.get("physical_cell_id") == "G1-Cell9"
        and row.get("rpt_index") == "8"
    )
    rpt8_step4 = _block_for_reference(
        rpt8_observation["step4_segment_ref"], 1, raw_blocks
    )
    rpt8_step10 = _block_for_reference(
        rpt8_observation["step10_segment_ref"], 1, raw_blocks
    )
    rpt8_key = _block_key(
        rpt8_step4["source_file"],
        1,
        4,
        int(rpt8_step4["start_line"]),
        int(rpt8_step4["end_line"]),
    )
    if rpt8_key not in integration_rows_by_key:
        integration_rows_by_key[rpt8_key] = {
            "block_role": "excluded_control",
            "dataset_part": 1,
            "physical_cell_id": "G1-Cell9",
            "source_file": rpt8_step4["source_file"],
            "step": 4,
            "start_line": rpt8_step4["start_line"],
            "end_line": rpt8_step4["end_line"],
            "row_count": rpt8_step4["row_count"],
            "raw_block_sha256": rpt8_step4["raw_block_sha256"],
            "expected_raw_block_sha256": rpt8_step4["expected_raw_block_sha256"],
            "current_signs": json.dumps(rpt8_step4["current_signs"], sort_keys=True),
            "testtime_negative_intervals": rpt8_step4["testtime_negative_intervals"],
            "testtime_zero_intervals": rpt8_step4["testtime_zero_intervals"],
            "steptime_negative_intervals": rpt8_step4["steptime_negative_intervals"],
            "steptime_zero_intervals": rpt8_step4["steptime_zero_intervals"],
            "testtime_dt_min_native": rpt8_step4["testtime_dt_min_native"],
            "testtime_dt_median_native": rpt8_step4["testtime_dt_median_native"],
            "testtime_dt_max_native": rpt8_step4["testtime_dt_max_native"],
            "steptime_dt_min_native": rpt8_step4["steptime_dt_min_native"],
            "steptime_dt_median_native": rpt8_step4["steptime_dt_median_native"],
            "steptime_dt_max_native": rpt8_step4["steptime_dt_max_native"],
            "native_capacity_span": rpt8_step4["capacity_span_native_amphr"],
            **_capacity_audit(rpt8_step4),
        }
    exceptions = [
        {
            "record_type": "excluded_control",
            "dataset_part": 1,
            "physical_cell_id": "G1-Cell9",
            "rpt_index": 8,
            "partition": "validation",
            "status": "REJECTED",
            "reason": "Mapped Step 4 contains 20 rows and only spans 3.738765545129-3.801480117335 V; it is a partial discharge and cannot supply a complete Step 4 reference.",
            "source_file": rpt8_observation["source_file"],
            "step4_segment_ref": rpt8_observation["step4_segment_ref"],
            "step4_block_sha256": rpt8_step4["raw_block_sha256"],
            "step10_segment_ref": rpt8_observation["step10_segment_ref"],
            "step10_block_sha256": rpt8_step10["raw_block_sha256"],
            "historical_label_used": False,
        },
        {
            "record_type": "assigned_cell_without_candidate",
            "dataset_part": 3,
            "physical_cell_id": "G10-Cell2",
            "rpt_index": "",
            "partition": "external_test",
            "status": "REVIEW",
            "reason": "The cell is assigned to External but has no candidate RPT row in the M1 Guide-mapped manifest; no sample or label is fabricated.",
            "source_file": "",
            "step4_segment_ref": "",
            "step4_block_sha256": "",
            "step10_segment_ref": "",
            "step10_block_sha256": "",
            "historical_label_used": False,
        },
    ]

    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty reconstruction output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    integration_rows = list(integration_rows_by_key.values())
    bol_rows = list(bol_rows_by_key.values())
    _write_csv(output_dir / "reconstructed_soh_references.csv", reconstructed)
    _write_csv(output_dir / "capacity_integration_audit.csv", integration_rows)
    _write_csv(output_dir / "bol_mapping.csv", bol_rows)
    _write_csv(output_dir / "secondary_bol_sensitivity.csv", secondary_rows)
    _write_csv(output_dir / "label_provenance.csv", provenance)
    _write_csv(output_dir / "final_candidate_manifest.csv", final_manifest)
    _write_csv(output_dir / "raw_curve_feature_matrix.csv", feature_rows)
    _write_csv(output_dir / "review_exclusions.csv", exceptions + [
        {
            "record_type": "candidate_record",
            "dataset_part": row["dataset_part"],
            "physical_cell_id": row["physical_cell_id"],
            "rpt_index": row["rpt_index"],
            "partition": row["partition"],
            "status": row["label_disposition"],
            "reason": row["label_reasons"] or row["modality_reasons"],
            "historical_label_used": False,
        }
        for row in reconstructed
        if row["label_disposition"] != "ACCEPTED" or row["final_eligibility_status"] != "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
    ])

    by_split: dict[str, dict[str, Any]] = {}
    for split in ("train", "validation", "external_test"):
        split_rows = [row for row in final_manifest if row["partition"] == split]
        accepted = [row for row in split_rows if row["label_disposition"] == "ACCEPTED"]
        by_split[split] = {
            "candidate_records": len(split_rows),
            "accepted_references": len(accepted),
            "review_references": sum(row["label_disposition"] == "REVIEW" for row in split_rows),
            "rejected_references": sum(row["label_disposition"] == "REJECTED" for row in split_rows),
            "complete_three_modality_eligible": sum(
                row["final_eligibility_status"] == "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
                for row in split_rows
            ),
            "physical_cells_with_candidates": len({row["physical_cell_id"] for row in split_rows}),
            "target_distribution_accepted_only": _target_distribution(accepted),
        }
    per_cell = []
    for cell in sorted(assignments):
        cell_rows = [row for row in final_manifest if row["physical_cell_id"] == cell]
        if not cell_rows:
            per_cell.append(
                {
                    "physical_cell_id": cell,
                    "partition": assignments[cell],
                    "candidate_records": 0,
                    "accepted_references": 0,
                    "review_references": 0,
                    "eligible_records": 0,
                    "accepted_target_distribution_percent": {"n": 0},
                }
            )
            continue
        accepted_cell = [
            row for row in cell_rows if row["label_disposition"] == "ACCEPTED"
        ]
        per_cell.append(
            {
                "physical_cell_id": cell,
                "partition": assignments[cell],
                "candidate_records": len(cell_rows),
                "accepted_references": sum(row["label_disposition"] == "ACCEPTED" for row in cell_rows),
                "review_references": sum(row["label_disposition"] == "REVIEW" for row in cell_rows),
                "eligible_records": sum(
                    row["final_eligibility_status"] == "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
                    for row in cell_rows
                ),
                "accepted_target_distribution_percent": _target_distribution(accepted_cell),
            }
        )

    dispositions = Counter(row["label_disposition"] for row in reconstructed)
    review_reason_counts = Counter(
        reason
        for row in reconstructed
        if row["label_disposition"] != "ACCEPTED"
        for reason in row["label_reasons"].split(";")
        if reason
    )
    historical_differences = [
        abs(
            float(row["reconstructed_soh_percent"])
            - float(row["historical_soh_percent_for_comparison_only"])
        )
        for row in reconstructed
        if row["label_disposition"] == "ACCEPTED"
        and row["reconstructed_soh_percent"] != ""
        and _number(row["historical_soh_percent_for_comparison_only"]) is not None
    ]
    modality_counts = {
        "step4_discharge_review": sum(row["discharge_measurement_validity"] != "ACCEPTED" for row in reconstructed),
        "step10_charge_review": sum(row["charge_measurement_validity"] != "ACCEPTED" for row in reconstructed),
        "physics_review": sum(
            row["physics_validity"] != "STRUCTURALLY_COMPLETE"
            for row in reconstructed
        ),
        "complete_three_modality_eligible": sum(
            row["final_eligibility_status"] == "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
            for row in reconstructed
        ),
    }
    summary = {
        "schema_version": 1,
        "reconstruction_version": "raw-soh-reconstruction-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(candidates),
        "excluded_control_count": 1,
        "assigned_cells_without_candidates": 1,
        "candidate_counts_by_part": dict(Counter(row["dataset_part"] for row in candidates)),
        "candidate_counts_by_split": dict(Counter(row["partition"] for row in candidates)),
        "label_disposition_counts": dict(dispositions),
        "review_reason_counts": dict(review_reason_counts),
        "historical_label_crosscheck_abs_difference_pp": (
            {
                "n": len(historical_differences),
                "mean": statistics.mean(historical_differences),
                "max": max(historical_differences),
            }
            if historical_differences
            else {"n": 0}
        ),
        "modality_counts": modality_counts,
        "by_split": by_split,
        "per_cell_coverage": per_cell,
        "training_authorized": False,
        "training_started": False,
        "degradation_fitting_started": False,
        "external_used_for_model_selection": False,
        "raw_blocks_reread_and_hash_verified": len(raw_blocks),
        "raw_source_files_reread": len({request.source_file for request in requests.values()}),
        "input_sha256": source_hashes,
        "guide_source_sha256": guide_summary.get("guide_sha256", {}),
        "time_unit_evidence": _TIME_UNIT_EVIDENCE,
        "current_unit_evidence": _CURRENT_UNIT_EVIDENCE,
        "capacity_unit_evidence": _CAPACITY_UNIT_EVIDENCE,
        "endpoint_evidence": _ENDPOINT_EVIDENCE,
        "mapped_step4_closest_observed_voltage_below_cutoff_v": cutoff_below_v,
        "mapped_step4_closest_observed_voltage_above_cutoff_v": cutoff_above_v,
        "limitations": [
            "These are reconstructed capacity-retention SoH references, not official Oxford ground truth.",
            "The original Maccor export profile is unavailable; minutes is a strong cross-validated inference, not an archived setting.",
            "Temperature remains in native units; sensor calibration/location is not established by the reviewed source documentation.",
            "Physics status means raw feature values are finite and sourced from the mapped Step 4 block; it does not mean temperature calibration or absolute units are independently verified.",
            "The Part 1 second BoL procedure is retained only for QC/sensitivity and is not the primary denominator.",
            "G2-Cell3 RPT2 remains REVIEW because both elapsed-time axes contain repeated timestamp intervals.",
            "G10-Cell2 has no candidate RPT mapping and is not fabricated.",
        ],
    }
    secondary_differences = [
        float(row["secondary_minus_primary_soh_pp_qc"])
        for row in secondary_rows
        if row["secondary_minus_primary_soh_pp_qc"] != ""
    ]
    summary["secondary_bol_sensitivity_qc"] = (
        {
            "n": len(secondary_differences),
            "mean_absolute_difference_pp": statistics.mean(
                abs(value) for value in secondary_differences
            ),
            "max_absolute_difference_pp": max(
                abs(value) for value in secondary_differences
            ),
        }
        if secondary_differences
        else {"n": 0}
    )
    (output_dir / "eligibility_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_m1_manifest": str(m1_manifest),
                "source_m1_block_audit": str(m1_block_audit),
                "split_manifest": str(split_path),
                "input_sha256": source_hashes,
                "raw_blocks_requested": len(requests),
                "raw_blocks_verified": len(raw_blocks),
                "raw_source_files_scanned": summary["raw_source_files_reread"],
                "training_authorized": False,
                "training_started": False,
                "degradation_fitting_started": False,
                "label_source": "Guide-mapped raw Step 4 current/elapsed-time integration",
                "historical_labels_used_as_targets": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_report(output_dir / "validation_report.md", summary, reconstructed, integration_rows, exceptions)
    return summary


def _write_report(
    path: Path,
    summary: dict[str, Any],
    references: list[dict[str, Any]],
    integration_rows: list[dict[str, Any]],
    exceptions: list[dict[str, Any]],
) -> None:
    statuses = summary["label_disposition_counts"]
    text = [
        "# Raw-derived capacity-retention SoH reconstruction",
        "",
        f"- Generated UTC: `{summary['generated_at_utc']}`",
        f"- M1 candidates: **{summary['candidate_count']}**",
        f"- Raw blocks re-read and SHA-256 verified: **{summary['raw_blocks_reread_and_hash_verified']}**",
        f"- Source files scanned once each, streaming: **{summary['raw_source_files_reread']}**",
        f"- Candidate label disposition: ACCEPTED {statuses.get('ACCEPTED', 0)}, REVIEW {statuses.get('REVIEW', 0)}, REJECTED {statuses.get('REJECTED', 0)}; G1-Cell9 RPT8 is a separate rejected control, not an M1 candidate.",
        "- `training_authorized: false`; no training or degradation fitting was run.",
        "",
        "## Measurement semantics",
        "",
        f"- **Time:** {_TIME_UNIT_EVIDENCE}",
        f"- **Current:** {_CURRENT_UNIT_EVIDENCE}",
        f"- **Capacity:** {_CAPACITY_UNIT_EVIDENCE}",
        f"- **Step 4 cutoff:** {_ENDPOINT_EVIDENCE}",
        f"- Closest mapped Step 4 observations bracketing 2.5 V: `{summary['mapped_step4_closest_observed_voltage_below_cutoff_v']:.12f} V` below and `{summary['mapped_step4_closest_observed_voltage_above_cutoff_v']:.12f} V` above.",
        "- TestTime and StepTime trapezoidal integrations are both retained in native current-time units. TestTime integration supplies the numerator/denominator; the common scale cancels in each compatible same-cell ratio. StepTime and native Amphr are independent checks. A-minute/60 Ah values are inferred diagnostics only. No numerical instrument tolerance is applied.",
        "- Unequal sampling intervals are integrated using trapezoids over actual positive timestamp differences. Negative or repeated timestamps remain review conditions, not silently repaired.",
        "",
        "## Candidate and eligibility counts",
        "",
        "| Split | Candidates | Accepted references | Review | Rejected | Complete three-modality eligible | Cells with candidates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, entry in summary["by_split"].items():
        text.append(
            f"| {split} | {entry['candidate_records']} | {entry['accepted_references']} | {entry['review_references']} | {entry['rejected_references']} | {entry['complete_three_modality_eligible']} | {entry['physical_cells_with_candidates']} |"
        )
    text.extend(
        [
            "",
            "## Modality coverage",
            "",
            f"- Step 4 discharge review/incomplete: {summary['modality_counts']['step4_discharge_review']}",
            f"- Step 10 charge review/incomplete: {summary['modality_counts']['step10_charge_review']}",
            f"- Physics-feature review/incomplete: {summary['modality_counts']['physics_review']}",
            f"- Complete three-modality eligible records: {summary['modality_counts']['complete_three_modality_eligible']}",
            "",
            "## Review reasons",
            "",
            "| Reason | Candidate records |",
            "|---|---:|",
        ]
    )
    for reason, count in sorted(summary["review_reason_counts"].items()):
        text.append(f"| {reason} | {count} |")
    text.extend(
        [
            "",
            "## Integration consistency",
            "",
            "The following discrepancy summaries are descriptive; they are not pass/fail thresholds.",
            "",
            "| Dataset part | Step 4 blocks | Median TestTime vs Amphr discrepancy (%) | Maximum (%) |",
            "|---|---:|---:|---:|",
        ]
    )
    for part in (1, 2, 3):
        rows = [
            row for row in integration_rows
            if int(row["dataset_part"]) == part
            and row["block_role"] == "rpt_step4"
            and row["testtime_vs_native_relative_discrepancy_percent"] not in ("", None)
        ]
        values = [float(row["testtime_vs_native_relative_discrepancy_percent"]) for row in rows]
        text.append(
            f"| {part} | {len(values)} | {statistics.median(values) if values else float('nan'):.8f} | {max(values) if values else float('nan'):.8f} |"
        )
    text.extend(
        [
            "",
            "## Per-split accepted target distribution",
            "",
            "| Split | N | Min (%) | Median (%) | Mean (%) | Max (%) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for split, entry in summary["by_split"].items():
        distribution = entry["target_distribution_accepted_only"]
        if distribution.get("n"):
            text.append(
                f"| {split} | {distribution['n']} | {distribution['min_percent']:.4f} | {distribution['median_percent']:.4f} | {distribution['mean_percent']:.4f} | {distribution['max_percent']:.4f} |"
            )
        else:
            text.append(f"| {split} | 0 | n/a | n/a | n/a | n/a |")
    text.extend(
        [
            "",
            "## Exceptions and unresolved evidence",
            "",
            "- G1-Cell9 RPT8 remains excluded: its mapped Step 4 block is a short partial discharge; the complete Step 10 charge is not a substitute.",
            "- G2-Cell3 RPT2 remains REVIEW because repeated timestamp intervals occur on both elapsed-time axes; no interpolation or dropped-row repair was applied.",
            "- G10-Cell22 and G10-Cell25 are retained only where exact Guide-mapped Step 4 endpoints are supported by the nearest observed above-cutoff sample; no voltage tolerance was invented.",
            "- G10-Cell2 remains an assigned External cell with no candidate RPT row; no label is created.",
            "- Temperature units and sensor calibration/location remain unresolved. Physics values preserve source-native temperature units and are not imputed.",
            "- Physics status indicates finite, same-RPT raw features; it does not assert verified calibration or absolute temperature units.",
            "- The second Part 1 BoL procedure is preserved in `secondary_bol_sensitivity.csv` for QC only and is not used as a chronological RPT or primary denominator.",
            f"- Historical-label cross-check (not target authority): n={summary['historical_label_crosscheck_abs_difference_pp']['n']}, mean absolute difference={summary['historical_label_crosscheck_abs_difference_pp'].get('mean', 0.0):.6f} pp, maximum={summary['historical_label_crosscheck_abs_difference_pp'].get('max', 0.0):.6f} pp.",
            f"- Secondary Part 1 BoL sensitivity (native Amphr ratio, QC only): n={summary['secondary_bol_sensitivity_qc']['n']}, mean absolute target difference={summary['secondary_bol_sensitivity_qc'].get('mean_absolute_difference_pp', 0.0):.6f} pp, maximum={summary['secondary_bol_sensitivity_qc'].get('max_absolute_difference_pp', 0.0):.6f} pp.",
            "",
            "## Per-cell coverage",
            "",
            "| Physical cell | Split | Candidates | Accepted labels | Review labels | Complete eligible | Accepted target min-max (%) |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in summary["per_cell_coverage"]:
        distribution = row["accepted_target_distribution_percent"]
        target_range = (
            f"{distribution['min_percent']:.3f}-{distribution['max_percent']:.3f}"
            if distribution.get("n")
            else "n/a"
        )
        text.append(
            f"| {row['physical_cell_id']} | {row['partition']} | {row['candidate_records']} | {row['accepted_references']} | {row['review_references']} | {row['eligible_records']} | {target_range} |"
        )
    text.extend(
        [
            "",
            "## Reproducibility and boundaries",
            "",
            "- The candidate identities, source blocks, and cell split come from the versioned M1 Guide-mapped manifest; historical SOH fields are provenance-only and are never targets.",
            "- Every requested raw block was re-read in one streaming pass per source file. Each block row count and SHA-256 was compared against the M1 raw-block audit; source file size and modification time were also checked.",
            "- The raw curve feature matrix contains only Step 4 voltage, Step 10 voltage, and the five configured same-RPT physics features. It does not contain capacity, Amphr, duration, cell ID, RPT index, or part ID as a predictive feature; IDs are retained only as join keys.",
            "- External records are held for external/domain-shift evaluation only and are not used in model selection or preprocessing.",
            "- No raw source, historical label, M1 output, checkpoint, or prior experiment configuration was modified.",
            "",
            "## Approval items",
            "",
            "1. Confirm use of the native integrated-current-time ratio as the reference; minutes/Ah remain strong inferences for absolute-capacity reporting because the export profile is unavailable.",
            "2. Confirm the data-derived nearest-observed-sample interpretation at the 2.5 V Step 4 boundary; it is not an instrument tolerance.",
            "3. Resolve G2-Cell3 RPT2 repeated timestamps before admitting it to the primary benchmark.",
            "4. Approve use of raw temperature-channel values in their native units as a predictive feature, or revise the feature contract.",
            "5. Approve the architecture matrix, compute budget, output location, and the exact label/split/config hashes in a scope-bound authorization before training.",
            "",
        ]
    )
    path.write_text("\n".join(text), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reconstruct SOH references from Guide-mapped raw RPT blocks.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    result = reconstruct(output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
