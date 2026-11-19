from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
INDEPENDENT_AUDIT_DIR = BENCHMARK_DIR / "independent_raw_soh_audit_v1_20261113"
OUTPUT_DIR = BENCHMARK_DIR / "soh_multimethod_recovery_v1_20261113"
CANDIDATE_MANIFEST = RECONSTRUCTION_DIR / "final_candidate_manifest.csv"
RECONSTRUCTED_REFERENCES = RECONSTRUCTION_DIR / "reconstructed_soh_references.csv"
INDEPENDENT_BLOCK_AUDIT = INDEPENDENT_AUDIT_DIR / "independent_block_audit.csv"
GUIDE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "experiment_manifests"
    / "guide_observation_manifest"
    / "guide_observation_manifest.csv"
)
PART3_BOL_CROSSWALK = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part3_bol_crosswalk"
    / "guide_observation_manifest.csv"
)
PART1_BOL_COMPARISON = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part1_bol_signal_review"
    / "part1_bol_curve_comparison.csv"
)
PART1_SECONDARY_BOL_SENSITIVITY = RECONSTRUCTION_DIR / "secondary_bol_sensitivity.csv"

_SEGMENT_RE = re.compile(
    r"^(?P<path>.+)#Step(?P<step>\d+):segment[^:]*:rows"
    r"(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$",
    re.IGNORECASE,
)
_CORE_COLUMNS = ("step", "testtime", "steptime", "amphr", "amps", "volts")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_segment(reference: str) -> tuple[str, int, int, int]:
    match = _SEGMENT_RE.fullmatch(reference.strip())
    if match is None:
        raise ValueError(f"Unparseable Guide-mapped block reference: {reference!r}")
    return (
        match.group("path").replace("\\", "/"),
        int(match.group("step")),
        int(match.group("start")),
        int(match.group("end")),
    )


def _key(source: str, part: int, step: int, start: int, end: int) -> tuple[str, int, int, int, int]:
    return source.replace("\\", "/").casefold(), part, step, start, end


def _float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _ratio_percent(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or numerator <= 0 or denominator <= 0:
        return None
    return 100.0 * numerator / denominator


def _metric_summary(rows: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    paired = [
        (_float(row.get(left)), _float(row.get(right)))
        for row in rows
        if _float(row.get(left)) is not None and _float(row.get(right)) is not None
    ]
    differences = [right_value - left_value for left_value, right_value in paired]
    absolute = [abs(value) for value in differences]
    if not differences:
        return {
            "n": 0,
            "mean_absolute_difference_pp": None,
            "median_absolute_difference_pp": None,
            "max_absolute_difference_pp": None,
            "bias_right_minus_left_pp": None,
        }
    return {
        "n": len(differences),
        "mean_absolute_difference_pp": statistics.fmean(absolute),
        "median_absolute_difference_pp": statistics.median(absolute),
        "max_absolute_difference_pp": max(absolute),
        "bias_right_minus_left_pp": statistics.fmean(differences),
    }


def _block_quality(
    row: dict[str, str],
    *,
    expected_step: int,
    current_direction: str,
    require_time: bool,
    require_charge_endpoint: bool = False,
    step4_voltage_above_cutoff: float | None = None,
) -> list[str]:
    reasons: list[str] = []
    for field in ("row_count_exact", "block_hash_exact", "step_ids_exact", "next_row_transitions_step"):
        if row.get(field, "").casefold() != "true":
            reasons.append(f"{field}_false")
    if int(row.get("wrong_width_rows", "0") or 0) != 0:
        reasons.append("row_width_mismatch")
    if int(row.get("missing_testtime", "0") or 0) or int(row.get("missing_amps", "0") or 0):
        reasons.append("missing_time_or_current")
    if int(row.get("missing_amphr", "0") or 0) or int(row.get("missing_volts", "0") or 0):
        reasons.append("missing_capacity_or_voltage")
    positive = int(row.get("current_positive_rows", "0") or 0)
    negative = int(row.get("current_negative_rows", "0") or 0)
    zero = int(row.get("current_zero_rows", "0") or 0)
    if current_direction == "positive" and (positive == 0 or negative or zero):
        reasons.append("current_direction_not_consistently_positive")
    if current_direction == "negative" and (negative == 0 or positive or zero):
        reasons.append("current_direction_not_consistently_negative")
    if float(row.get("amphr_span_native", "0") or 0) <= 0:
        reasons.append("nonpositive_native_capacity_span")
    if int(row.get("amphr_negative_deltas", "0") or 0) > 0:
        reasons.append("native_capacity_counter_reversal")
    if require_time:
        for name in ("testtime", "steptime"):
            if int(row.get(f"{name}_zero_intervals", "0") or 0):
                reasons.append(f"{name}_repeated_timestamp")
            if int(row.get(f"{name}_negative_intervals", "0") or 0):
                reasons.append(f"{name}_time_reset_or_negative_interval")
    if expected_step == 4:
        first = _float(row.get("voltage_first"))
        last = _float(row.get("voltage_last"))
        if first is None or first <= 2.5 or last is None:
            reasons.append("step4_start_or_endpoint_missing")
        elif last > 2.5 and (
            step4_voltage_above_cutoff is None or last != step4_voltage_above_cutoff
        ):
            reasons.append("step4_endpoint_not_supported_by_observed_cutoff_sample")
    if require_charge_endpoint:
        maximum = _float(row.get("voltage_max"))
        if maximum is None or maximum < 4.2:
            reasons.append("step10_did_not_reach_documented_4_2_v_target")
    if row.get("raw_block_status") != "ACCEPTED":
        reasons.append("independent_raw_block_audit_not_accepted")
    return list(dict.fromkeys(reasons))


class _Step10Segment:
    def __init__(self, start_line: int) -> None:
        self.start_line = start_line
        self.end_line = start_line
        self.row_count = 0
        self.digest = hashlib.sha256()
        self.current_signs: Counter[str] = Counter()
        self.missing: Counter[str] = Counter()
        self.testtime_integral = 0.0
        self.steptime_integral = 0.0
        self.time_intervals: Counter[str] = Counter()
        self.step_intervals: Counter[str] = Counter()
        self.capacity_deltas: Counter[str] = Counter()
        self.capacity_min: float | None = None
        self.capacity_max: float | None = None
        self.voltage_first: float | None = None
        self.voltage_last: float | None = None
        self.voltage_min: float | None = None
        self.voltage_max: float | None = None
        self.previous: dict[str, float | None] = {
            "amps": None,
            "amphr": None,
            "testtime": None,
            "steptime": None,
        }

    @staticmethod
    def _value(row: list[str], columns: dict[str, int], field: str) -> float | None:
        index = columns[field]
        if index >= len(row):
            return None
        return _float(row[index])

    def update(self, row: list[str], columns: dict[str, int], line_number: int) -> None:
        self.end_line = line_number
        self.row_count += 1
        self.digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
        values = {name: self._value(row, columns, name) for name in _CORE_COLUMNS}
        for name, value in values.items():
            if value is None:
                self.missing[name] += 1
        current = values["amps"]
        capacity = values["amphr"]
        voltage = values["volts"]
        if current is not None:
            self.current_signs[
                "positive" if current > 0 else "negative" if current < 0 else "zero"
            ] += 1
        if capacity is not None:
            self.capacity_min = capacity if self.capacity_min is None else min(self.capacity_min, capacity)
            self.capacity_max = capacity if self.capacity_max is None else max(self.capacity_max, capacity)
            previous_capacity = self.previous["amphr"]
            if previous_capacity is not None:
                delta = capacity - previous_capacity
                self.capacity_deltas[
                    "negative" if delta < 0 else "positive" if delta > 0 else "zero"
                ] += 1
        if voltage is not None:
            if self.voltage_first is None:
                self.voltage_first = voltage
            self.voltage_last = voltage
            self.voltage_min = voltage if self.voltage_min is None else min(self.voltage_min, voltage)
            self.voltage_max = voltage if self.voltage_max is None else max(self.voltage_max, voltage)

        for time_name, integral_name, counter in (
            ("testtime", "testtime_integral", self.time_intervals),
            ("steptime", "steptime_integral", self.step_intervals),
        ):
            timestamp = values[time_name]
            prior_time = self.previous[time_name]
            prior_current = self.previous["amps"]
            if timestamp is not None and prior_time is not None:
                delta_time = timestamp - prior_time
                counter[
                    "negative" if delta_time < 0 else "positive" if delta_time > 0 else "zero"
                ] += 1
                if delta_time > 0 and current is not None and prior_current is not None:
                    setattr(
                        self,
                        integral_name,
                        getattr(self, integral_name)
                        + 0.5 * (abs(prior_current) + abs(current)) * delta_time,
                    )
            self.previous[time_name] = timestamp
        self.previous["amps"] = current
        self.previous["amphr"] = capacity

    def finish(self, source_file: str, source_part: int, next_step: int | None) -> dict[str, Any]:
        span = (
            None
            if self.capacity_min is None or self.capacity_max is None
            else self.capacity_max - self.capacity_min
        )
        return {
            "source_file": source_file,
            "source_part": source_part,
            "step": 10,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "row_count": self.row_count,
            "raw_block_sha256": self.digest.hexdigest(),
            "next_step": next_step,
            "current_signs": dict(self.current_signs),
            "missing": dict(self.missing),
            "testtime_integral_abs_current_native": self.testtime_integral,
            "steptime_integral_abs_current_native": self.steptime_integral,
            "testtime_zero_intervals": self.time_intervals["zero"],
            "testtime_negative_intervals": self.time_intervals["negative"],
            "steptime_zero_intervals": self.step_intervals["zero"],
            "steptime_negative_intervals": self.step_intervals["negative"],
            "amphr_span_native": span,
            "amphr_negative_deltas": self.capacity_deltas["negative"],
            "voltage_first": self.voltage_first,
            "voltage_last": self.voltage_last,
            "voltage_min": self.voltage_min,
            "voltage_max": self.voltage_max,
        }


def _scan_step10_segments(source_file: str, source_part: int) -> list[dict[str, Any]]:
    source_path = PROJECT_ROOT / Path(source_file.replace("/", "\\"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Guide-mapped BoL source file is missing: {source_path}")
    before = source_path.stat()
    segments: list[dict[str, Any]] = []
    active: _Step10Segment | None = None
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Guide-mapped BoL source has no header: {source_path}")
        folded = [item.strip().casefold() for item in header]
        columns: dict[str, int] = {}
        for name in _CORE_COLUMNS:
            if folded.count(name) != 1:
                raise ValueError(
                    f"Expected exactly one {name!r} in {source_file}, found {folded.count(name)}."
                )
            columns[name] = folded.index(name)
        for row in reader:
            line_number = reader.line_num
            step = _float(row[columns["step"]]) if columns["step"] < len(row) else None
            step_value = int(step) if step is not None and step.is_integer() else None
            if step_value == 10:
                if active is None:
                    active = _Step10Segment(line_number)
                active.update(row, columns, line_number)
            elif active is not None:
                segments.append(active.finish(source_file, source_part, step_value))
                active = None
        if active is not None:
            segments.append(active.finish(source_file, source_part, None))
    after = source_path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Mapped BoL source changed while reading: {source_file}")
    for segment in segments:
        segment["source_size_bytes"] = before.st_size
        segment["source_mtime_ns"] = before.st_mtime_ns
        segment["source_block_ref"] = (
            f"{source_file}#Step10:segment_rows"
            f"{segment['start_line']}-{segment['end_line']}"
        )
        segment["charge_validity_reasons"] = _step10_segment_reasons(segment)
        segment["charge_valid"] = not segment["charge_validity_reasons"]
    return segments


def _step10_segment_reasons(segment: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if segment["row_count"] < 2:
        reasons.append("too_few_rows_to_integrate")
    if segment["current_signs"].get("positive", 0) != segment["row_count"]:
        reasons.append("current_direction_not_consistently_positive")
    if segment["missing"]:
        reasons.append("missing_or_nonfinite_core_measurement")
    if segment["testtime_zero_intervals"] or segment["testtime_negative_intervals"]:
        reasons.append("testtime_not_strictly_increasing")
    if segment["steptime_zero_intervals"] or segment["steptime_negative_intervals"]:
        reasons.append("steptime_not_strictly_increasing")
    if segment["amphr_negative_deltas"]:
        reasons.append("native_capacity_counter_reversal")
    if segment["amphr_span_native"] is None or segment["amphr_span_native"] <= 0:
        reasons.append("nonpositive_native_capacity_span")
    if _float(segment["testtime_integral_abs_current_native"]) is None or segment[
        "testtime_integral_abs_current_native"
    ] <= 0:
        reasons.append("nonpositive_integrated_charge")
    if segment["voltage_max"] is None or segment["voltage_max"] < 4.2:
        reasons.append("did_not_reach_documented_4_2_v_target")
    return reasons


def _select_primary_step10_bol(
    source_file: str, source_part: int
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str]:
    segments = _scan_step10_segments(source_file, source_part)
    valid = [segment for segment in segments if segment["charge_valid"]]
    if len(valid) == 1:
        return valid[0], segments, "unique_complete_positive_step10_segment"
    if not valid:
        return None, segments, "no_complete_positive_step10_segment"
    return None, segments, "multiple_complete_positive_step10_segments"


def _guide_bol_match(candidate: dict[str, str], guide_rows: list[dict[str, str]]) -> tuple[bool, str]:
    part = int(candidate["dataset_part"])
    expected_bol_part = 3 if part == 3 else 1
    bol_part = int(candidate["bol_source_part"])
    if bol_part != expected_bol_part:
        return False, f"expected_part{expected_bol_part}_bol_but_manifest_uses_part{bol_part}"
    if part == 2:
        group_match = re.fullmatch(r"G(?P<group>\d+)-Cell\d+", candidate["physical_cell_id"])
        if group_match is None or int(group_match.group("group")) not in {1, 2, 3, 4}:
            return False, "part2_cell_not_in_documented_part1_continuation_groups"
    prefix = f"data/raw/oxford/part{expected_bol_part}/"
    source_relative = candidate["bol_source_file"].replace("\\", "/")
    if not source_relative.startswith(prefix):
        return False, "bol_source_path_does_not_match_expected_dataset_part"
    source_relative = source_relative[len(prefix) :]
    if part == 3:
        matching = [
            row
            for row in _read_csv(PART3_BOL_CROSSWALK)
            if row.get("physical_cell_id") == candidate["physical_cell_id"]
            and "bol" in row.get("guide_stage", "").casefold()
            and row.get("source_file", "").replace("\\", "/") == source_relative
            and row.get("source_mapping_rule") == "part3_unsuffixed_bol_to_zero_suffix"
            and row.get("source_mapping_status") == "unique_source_file"
        ]
    else:
        matching = [
            row
            for row in guide_rows
            if row.get("dataset_part") == str(expected_bol_part)
            and row.get("physical_cell_id") == candidate["physical_cell_id"]
            and "bol" in row.get("guide_stage", "").casefold()
            and row.get("source_file", "").replace("\\", "/") == source_relative
            and row.get("measurement_mapping_status") == "single_checkpoint"
            and row.get("source_mapping_status") == "unique_source_file"
        ]
    if len(matching) != 1:
        return False, f"expected_one_exact_guide_bol_mapping_found_{len(matching)}"
    return True, "exact_same_cell_guide_bol_source_and_documented_part_protocol"


def _lookup_block(
    audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]],
    reference: str,
    part: int,
) -> dict[str, str]:
    source, step, start, end = _parse_segment(reference)
    if not source.startswith("data/raw/"):
        source = f"data/raw/oxford/part{part}/{source}"
    key = _key(source, part, step, start, end)
    row = audit_by_key.get(key)
    if row is None:
        raise ValueError(f"Independent raw block audit has no exact match for {reference!r}.")
    return row


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return format(value, ".15g")
    return str(value)


def _make_comparison(
    candidate: dict[str, str],
    discharge: dict[str, str],
    charge: dict[str, str],
    bol_discharge: dict[str, str],
    bol_charge: dict[str, Any] | None,
    *,
    guide_compatible: bool,
    guide_note: str,
    cutoff_above_v: float | None,
) -> dict[str, Any]:
    part = int(candidate["dataset_part"])
    discharge_reasons = _block_quality(
        discharge,
        expected_step=4,
        current_direction="negative",
        require_time=True,
        step4_voltage_above_cutoff=cutoff_above_v,
    )
    bol_discharge_reasons = _block_quality(
        bol_discharge,
        expected_step=4,
        current_direction="negative",
        require_time=True,
        step4_voltage_above_cutoff=cutoff_above_v,
    )
    charge_reasons = _block_quality(
        charge,
        expected_step=10,
        current_direction="positive",
        require_time=True,
        require_charge_endpoint=True,
    )
    if bol_charge is None:
        bol_charge_reasons = ["no_unique_complete_step10_bol_segment"]
    else:
        bol_charge_reasons = list(bol_charge["charge_validity_reasons"])
    if not guide_compatible:
        bol_discharge_reasons.append("guide_bol_compatibility_unresolved")
        bol_charge_reasons.append("guide_bol_compatibility_unresolved")
    q4 = _float(discharge.get("testtime_integrated_abs_current_native"))
    q10 = _float(charge.get("testtime_integrated_abs_current_native"))
    q4_bol = _float(bol_discharge.get("testtime_integrated_abs_current_native"))
    q10_bol = (
        None
        if bol_charge is None
        else _float(bol_charge.get("testtime_integral_abs_current_native"))
    )
    span4 = _float(discharge.get("amphr_span_native"))
    span4_bol = _float(bol_discharge.get("amphr_span_native"))
    span10 = _float(charge.get("amphr_span_native"))
    span10_bol = None if bol_charge is None else _float(bol_charge.get("amphr_span_native"))
    m1 = _ratio_percent(q4, q4_bol)
    m2 = _ratio_percent(q10, q10_bol)
    m3_discharge = _ratio_percent(span4, span4_bol)
    m3_charge = _ratio_percent(span10, span10_bol)
    m1_valid = not discharge_reasons and not bol_discharge_reasons and guide_compatible and m1 is not None
    m2_valid = not charge_reasons and not bol_charge_reasons and guide_compatible and m2 is not None
    m3_discharge_valid = (
        span4 is not None
        and span4_bol is not None
        and span4 > 0
        and span4_bol > 0
        and int(discharge.get("amphr_negative_deltas", "0") or 0) == 0
        and int(bol_discharge.get("amphr_negative_deltas", "0") or 0) == 0
        and int(discharge.get("current_negative_rows", "0") or 0)
        == int(discharge.get("row_count", "0") or 0)
        and int(discharge.get("current_positive_rows", "0") or 0) == 0
        and int(discharge.get("current_zero_rows", "0") or 0) == 0
        and int(bol_discharge.get("current_negative_rows", "0") or 0)
        == int(bol_discharge.get("row_count", "0") or 0)
        and int(bol_discharge.get("current_positive_rows", "0") or 0) == 0
        and int(bol_discharge.get("current_zero_rows", "0") or 0) == 0
        and discharge.get("raw_block_status") == "ACCEPTED"
        and bol_discharge.get("raw_block_status") == "ACCEPTED"
        and guide_compatible
    )
    m3_charge_valid = (
        span10 is not None
        and span10_bol is not None
        and span10 > 0
        and span10_bol > 0
        and int(charge.get("amphr_negative_deltas", "0") or 0) == 0
        and bol_charge is not None
        and not bol_charge_reasons
        and guide_compatible
    )
    bol_voltage_min = None if bol_charge is None else _float(bol_charge.get("voltage_min"))
    rpt_voltage_min = _float(charge.get("voltage_min"))
    bol_voltage_max = None if bol_charge is None else _float(bol_charge.get("voltage_max"))
    rpt_voltage_max = _float(charge.get("voltage_max"))
    overlap = None
    if None not in (bol_voltage_min, rpt_voltage_min, bol_voltage_max, rpt_voltage_max):
        overlap = max(
            0.0,
            min(float(bol_voltage_max), float(rpt_voltage_max))
            - max(float(bol_voltage_min), float(rpt_voltage_min)),
        )
    reasons = list(dict.fromkeys(discharge_reasons + bol_discharge_reasons))
    return {
        "dataset_part": part,
        "physical_cell_id": candidate["physical_cell_id"],
        "rpt_index": candidate["rpt_index"],
        "measurement_id": candidate["measurement_id"],
        "partition": candidate["partition"],
        "original_disposition": candidate["label_disposition"],
        "original_review_reason": candidate.get("label_reasons", ""),
        "protocol_domain": "Maccor 4000; Part 3" if part == 3 else "Maccor 4200; Parts 1-2",
        "guide_bol_compatible": guide_compatible,
        "guide_bol_evidence": guide_note,
        "primary_bol_source_file": candidate["bol_source_file"],
        "primary_bol_step4_segment_ref": candidate["bol_step4_segment_ref"],
        "primary_bol_step4_block_sha256": bol_discharge["block_sha256_independent"],
        "primary_bol_step10_segment_ref": "" if bol_charge is None else bol_charge["source_block_ref"],
        "primary_bol_step10_block_sha256": "" if bol_charge is None else bol_charge["raw_block_sha256"],
        "rpt_step4_segment_ref": candidate["step4_segment_ref"],
        "rpt_step4_block_sha256": discharge["block_sha256_independent"],
        "rpt_step10_segment_ref": candidate["step10_segment_ref"],
        "rpt_step10_block_sha256": charge["block_sha256_independent"],
        "q_discharge_rpt_native_current_time": q4,
        "q_discharge_bol_native_current_time": q4_bol,
        "soh_discharge_integrated_percent": m1,
        "m1_status": "VALID" if m1_valid else "REVIEW",
        "m1_invalid_reasons": ";".join(reasons),
        "q_charge_rpt_native_current_time": q10,
        "q_charge_bol_native_current_time": q10_bol,
        "soh_charge_integrated_percent": m2,
        "m2_status": "VALID" if m2_valid else "REVIEW",
        "m2_invalid_reasons": ";".join(dict.fromkeys(charge_reasons + bol_charge_reasons)),
        "native_discharge_rpt_span": span4,
        "native_discharge_bol_span": span4_bol,
        "soh_native_discharge_percent": m3_discharge,
        "m3_discharge_status": "VALID" if m3_discharge_valid else "REVIEW",
        "native_charge_rpt_span": span10,
        "native_charge_bol_span": span10_bol,
        "soh_native_charge_percent": m3_charge,
        "m3_charge_status": "VALID" if m3_charge_valid else "REVIEW",
        "step10_rpt_voltage_min_v": _float(charge.get("voltage_min")),
        "step10_rpt_voltage_max_v": _float(charge.get("voltage_max")),
        "step10_bol_voltage_min_v": bol_voltage_min,
        "step10_bol_voltage_max_v": bol_voltage_max,
        "step10_voltage_window_overlap_v": overlap,
        "step10_bol_segment_status": (
            "MISSING_OR_AMBIGUOUS"
            if bol_charge is None
            else "VALID"
            if not bol_charge_reasons
            else "REVIEW"
        ),
        "step4_time_zero_intervals_rpt": discharge.get("testtime_zero_intervals", ""),
        "step4_time_negative_intervals_rpt": discharge.get("testtime_negative_intervals", ""),
        "step10_time_zero_intervals_rpt": charge.get("testtime_zero_intervals", ""),
        "step10_time_negative_intervals_rpt": charge.get("testtime_negative_intervals", ""),
        "historical_m1_reference_percent_comparison_only": _float(
            candidate.get("reconstructed_soh_percent")
        ),
        "m1_vs_existing_reference_abs_difference_pp": (
            None
            if m1 is None or _float(candidate.get("reconstructed_soh_percent")) is None
            else abs(m1 - float(candidate["reconstructed_soh_percent"]))
        ),
        "training_authorized": False,
    }


def _review_outcome(row: dict[str, Any]) -> tuple[str, str]:
    if row["m1_status"] == "VALID":
        return (
            "RECOVERABLE_DISCHARGE",
            "Independent Guide/protocol crosswalk and Step 4 measurement checks support the discharge ratio; retain original REVIEW until separately approved.",
        )
    if row["m2_status"] == "VALID":
        return (
            "CHARGE_PROXY_ONLY",
            "Step 10 charge ratio is computable as a separate target; it does not resolve or replace the Step 4 discharge reference.",
        )
    if row["m3_discharge_status"] == "VALID":
        return (
            "NATIVE_COUNTER_SUPPORT_ONLY",
            "Native Step 4 counter ratio is available but cannot resolve the discharge timestamp or integration issue.",
        )
    if "step4_endpoint_not_supported_by_observed_cutoff_sample" in row["m1_invalid_reasons"]:
        return "EXCLUDED", "Mapped Step 4 discharge does not meet the observed endpoint evidence."
    return "UNRESOLVED", "Insufficient compatible evidence; keep the original REVIEW disposition."


def _summaries(comparison: list[dict[str, Any]], agreement: list[dict[str, Any]]) -> dict[str, Any]:
    accepted = agreement
    original_accepted = [
        row for row in comparison if row["original_disposition"] == "ACCEPTED"
    ]
    accepted_valid = [
        row
        for row in original_accepted
        if row["m1_status"] == "VALID" and row["m2_status"] == "VALID"
    ]
    by_domain: dict[str, Any] = {}
    for domain in sorted({row["protocol_domain"] for row in accepted}):
        subset = [row for row in accepted if row["protocol_domain"] == domain]
        by_domain[domain] = _metric_summary(
            subset, "soh_discharge_integrated_percent", "soh_charge_integrated_percent"
        )
    by_cell: dict[str, Any] = {}
    for cell in sorted({row["physical_cell_id"] for row in accepted}):
        subset = [row for row in accepted if row["physical_cell_id"] == cell]
        by_cell[cell] = _metric_summary(
            subset, "soh_discharge_integrated_percent", "soh_charge_integrated_percent"
        )
    disposition_counts = Counter(row["original_disposition"] for row in comparison)
    return {
        "candidate_count": len(comparison),
        "original_disposition_counts": dict(sorted(disposition_counts.items())),
        "valid_method_counts": {
            "M1_step4_discharge_integrated": sum(row["m1_status"] == "VALID" for row in comparison),
            "M2_step10_charge_integrated": sum(row["m2_status"] == "VALID" for row in comparison),
            "M3_native_step4_counter": sum(
                row["m3_discharge_status"] == "VALID" for row in comparison
            ),
            "M3_native_step10_counter": sum(row["m3_charge_status"] == "VALID" for row in comparison),
        },
        "accepted_record_method_counts": {
            "M1": sum(row["m1_status"] == "VALID" for row in accepted_valid),
            "M2": sum(row["m2_status"] == "VALID" for row in accepted_valid),
            "M3_native_step4": sum(
                row["m3_discharge_status"] == "VALID" for row in accepted_valid
            ),
            "M3_native_step10": sum(
                row["m3_charge_status"] == "VALID" for row in accepted_valid
            ),
        },
        "accepted_records_not_in_valid_M1_M2_pairs": len(original_accepted) - len(agreement),
        "accepted_pair_exclusion_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in original_accepted
                    if row["m1_status"] != "VALID" or row["m2_status"] != "VALID"
                    for reason in {
                        item
                        for item in (
                            str(row["m1_invalid_reasons"])
                            + ";"
                            + str(row["m2_invalid_reasons"])
                        ).split(";")
                        if item
                    }
                ).items()
            )
        ),
        "accepted_M1_vs_M2": _metric_summary(
            accepted, "soh_discharge_integrated_percent", "soh_charge_integrated_percent"
        ),
        "accepted_M1_vs_M3_native_step4": _metric_summary(
            accepted, "soh_discharge_integrated_percent", "soh_native_discharge_percent"
        ),
        "accepted_M2_vs_M3_native_step10": _metric_summary(
            accepted, "soh_charge_integrated_percent", "soh_native_charge_percent"
        ),
        "accepted_M1_vs_existing_reference": _metric_summary(
            [
                row
                for row in comparison
                if row["original_disposition"] == "ACCEPTED"
                and row["m1_status"] == "VALID"
            ],
            "soh_discharge_integrated_percent",
            "historical_m1_reference_percent_comparison_only",
        ),
        "m1_vs_m2_by_protocol_domain": by_domain,
        "m1_vs_m2_by_cell": by_cell,
        "review_recovery_disposition_counts": dict(
            sorted(Counter(row["recovery_outcome"] for row in comparison if row["original_disposition"] == "REVIEW").items())
        ),
        "recovery_records_not_promoted": sum(row["original_disposition"] == "REVIEW" for row in comparison),
        "numerical_acceptance_threshold_applied": False,
        "training_authorized": False,
        "training_started": False,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _text(row.get(key)) for key in fields})


def _agreement_plot(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    colors = {"Maccor 4200; Parts 1-2": "#2166ac", "Maccor 4000; Part 3": "#b2182b"}
    figure, axis = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    for domain, color in colors.items():
        subset = [row for row in rows if row["protocol_domain"] == domain]
        x = [float(row["soh_discharge_integrated_percent"]) for row in subset]
        y = [float(row["soh_charge_integrated_percent"]) for row in subset]
        if x:
            axis.scatter(x, y, s=24, alpha=0.78, label=f"{domain} (n={len(x)})", color=color)
    points = [
        float(value)
        for row in rows
        for value in (row["soh_discharge_integrated_percent"], row["soh_charge_integrated_percent"])
        if _float(value) is not None
    ]
    lower = min(points) - 1.0
    upper = max(points) + 1.0
    axis.plot([lower, upper], [lower, upper], linestyle="--", color="#333333", linewidth=1)
    axis.set(xlim=(lower, upper), ylim=(lower, upper))
    axis.set_xlabel("Step 4 discharge retention, M1 (%)")
    axis.set_ylabel("Step 10 charge retention, M2 (%)")
    axis.set_title("Cross-method agreement; charge remains a separate proxy")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _write_report(
    path: Path,
    summary: dict[str, Any],
    review_rows: list[dict[str, Any]],
    bol_segments: list[dict[str, Any]],
    secondary_segments: list[dict[str, Any]],
) -> None:
    m1_m2 = summary["accepted_M1_vs_M2"]
    lines = [
        "# Multi-method SoH reconstruction and REVIEW recovery audit",
        "",
        "Status: data-quality investigation only. Original labels and dispositions were not modified. "
        "`training_authorized: false`; no model training, external prediction evaluation, PyDMA, or Half-Cell fitting was run.",
        "",
        "## Methods and provenance",
        "",
        "- M1: Step 4 discharge retention, `100 × integral(abs(Amps), TestTime)_RPT / integral(abs(Amps), TestTime)_BoL`.",
        "- M2: Step 10 pseudo-OCV charge retention, computed separately with the same-cell, same-protocol Step 10 BoL; it is a distinct charge-proxy target, never substituted for M1.",
        "- M3: native `Amphr` span ratio, reported separately on Step 4 discharge and Step 10 charge windows.",
        "- RPT block integrals and counters use the prior independently re-read, row-count- and SHA-256-verified block audit. The new implementation independently forms ratios and directly integrates Step 10 in the Guide-selected BoL procedure files. It does not repeat the 227-source scan.",
        "- The Part 3 unsuffixed Guide BoL procedure maps uniquely to `.0` files in the corrected crosswalk. Part 1/2 continuation records use the same-cell Part 1 Guide BoL. Part 2 is treated as continuing Parts 1–2 only for Guide groups 1–4.",
        "- Time units and absolute `Amphr` units remain unverified. Values are retained in native current-time/counter units; only within-cell ratios are reported.",
        "- Step 10 completion uses documented 4.2 V as a reached-maximum check, with no added instrument tolerance. Actual voltage-window minima/maxima and overlap are reported per record.",
        "",
        "## Method counts",
        "",
        f"- Candidate rows: {summary['candidate_count']}; original dispositions: `{summary['original_disposition_counts']}`.",
        f"- Valid M1 Step 4 integrated ratios: {summary['valid_method_counts']['M1_step4_discharge_integrated']}.",
        f"- Valid M2 Step 10 integrated ratios: {summary['valid_method_counts']['M2_step10_charge_integrated']}.",
        f"- Valid M3 native Step 4 ratios: {summary['valid_method_counts']['M3_native_step4_counter']}.",
        f"- Valid M3 native Step 10 ratios: {summary['valid_method_counts']['M3_native_step10_counter']}.",
        f"- Original ACCEPTED records with valid M1/M2 pairs: {m1_m2['n']}/178; not included in paired statistics: {summary['accepted_records_not_in_valid_M1_M2_pairs']}.",
        f"- Accepted-pair exclusion reasons: `{summary['accepted_pair_exclusion_reasons']}`.",
        "",
        "## M1 versus M2 on original ACCEPTED records",
        "",
        f"- Valid paired N: {m1_m2['n']}.",
        f"- Mean absolute difference: {m1_m2['mean_absolute_difference_pp']:.6f} percentage points.",
        f"- Median absolute difference: {m1_m2['median_absolute_difference_pp']:.6f} percentage points.",
        f"- Maximum absolute difference: {m1_m2['max_absolute_difference_pp']:.6f} percentage points.",
        f"- Bias (M2 − M1): {m1_m2['bias_right_minus_left_pp']:.6f} percentage points.",
        "",
        "M3 native-counter consistency checks on paired ACCEPTED records:",
        "",
        f"- M1 versus native Step 4 counter: N={summary['accepted_M1_vs_M3_native_step4']['n']}, "
        f"mean absolute difference={summary['accepted_M1_vs_M3_native_step4']['mean_absolute_difference_pp']:.6f} pp, "
        f"maximum={summary['accepted_M1_vs_M3_native_step4']['max_absolute_difference_pp']:.6f} pp.",
        f"- M2 versus native Step 10 counter: N={summary['accepted_M2_vs_M3_native_step10']['n']}, "
        f"mean absolute difference={summary['accepted_M2_vs_M3_native_step10']['mean_absolute_difference_pp']:.6f} pp, "
        f"maximum={summary['accepted_M2_vs_M3_native_step10']['max_absolute_difference_pp']:.6f} pp.",
        "",
        "| Protocol domain | N | Mean absolute difference (pp) | Median absolute difference (pp) | Maximum absolute difference (pp) | Bias M2−M1 (pp) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, metric in summary["m1_vs_m2_by_protocol_domain"].items():
        lines.append(
            f"| {domain} | {metric['n']} | {metric['mean_absolute_difference_pp']:.6f} | "
            f"{metric['median_absolute_difference_pp']:.6f} | {metric['max_absolute_difference_pp']:.6f} | "
            f"{metric['bias_right_minus_left_pp']:.6f} |"
        )
    lines.extend(
        [
            "",
            "### Per-cell agreement",
            "",
            "| Physical cell | N | Mean absolute difference (pp) | Median absolute difference (pp) | Maximum absolute difference (pp) | Bias M2−M1 (pp) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell, metric in summary["m1_vs_m2_by_cell"].items():
        lines.append(
            f"| {cell} | {metric['n']} | {metric['mean_absolute_difference_pp']:.6f} | "
            f"{metric['median_absolute_difference_pp']:.6f} | {metric['max_absolute_difference_pp']:.6f} | "
            f"{metric['bias_right_minus_left_pp']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Per-cell paired differences are recorded in `soh_charge_discharge_agreement.csv`; the companion PNG plots M1 against M2. No numerical acceptance threshold was invented or applied. Similarity is a cross-method diagnostic, not proof that charge and discharge retention are interchangeable.",
            "",
            "## Step 10 compatibility interpretation",
            "",
            f"- Directly scanned Guide-selected primary BoL Step 10 source segments: {len(bol_segments)}.",
            f"- Unique complete positive Step 10 BoL segments: {sum(bool(row.get('charge_valid')) for row in bol_segments)}.",
            f"- Alternative Part 1 BoL source files scanned: {len({row['source_file'] for row in secondary_segments})}; files with a unique complete Step 10 charge segment: {len({row['source_file'] for row in secondary_segments if row.get('charge_valid')})}.",
            "- The Oxford READMEs specify Step 10 as pseudo-OCV charge at C/24 to 4.2 V; that is not the Step 4 CC-CV discharge measurement. M2 is therefore suitable only as a separate charge-retention proxy when the Guide mapping, direction, time axis, counter, and 4.2 V reach checks pass.",
            "- Voltage-window overlap is empirical and varies by record. It is not converted into a pass/fail cutoff because the sources provide no validated SOC conversion or tolerance for the lower voltage endpoint.",
            "",
            "## REVIEW recovery decisions",
            "",
            "| Outcome | Count |",
            "|---|---:|",
        ]
    )
    for outcome, count in summary["review_recovery_disposition_counts"].items():
        lines.append(f"| {outcome} | {count} |")
    lines.extend(
        [
            "",
            f"All {summary['recovery_records_not_promoted']} original REVIEW rows remain REVIEW in the source manifest and are not promoted by this audit.",
            "",
            "| Cell | RPT | Outcome | M1 status | M2 status | M3 Step 4 | Original reason |",
            "|---|---:|---|---|---|---|---|",
        ]
    )
    for row in review_rows:
        lines.append(
            f"| {row['physical_cell_id']} | {row['rpt_index']} | {row['recovery_outcome']} | "
            f"{row['m1_status']} | {row['m2_status']} | {row['m3_discharge_status']} | "
            f"{row['original_review_reason']} |"
        )
    lines.extend(
        [
            "",
            "Alternative Step 10 BoL mappings for REVIEW records:",
            "",
            "| Cell | Alternative Step 10 BoL status | Secondary Step 4 native-counter QC SoH (%) |",
            "|---|---|---:|",
        ]
    )
    for row in review_rows:
        lines.append(
            f"| {row['physical_cell_id']} | {row.get('secondary_bol_step10_status', 'no_alternative_part1_bol')} | "
            f"{_text(row.get('secondary_bol_step4_native_counter_soh_percent_qc'))} |"
        )
    lines.extend(
        [
            "",
            "The 24 Part 3 cases can be independently re-evaluated against the corrected same-cell Guide `.0` BoL crosswalk and documented Maccor 4000 Step 4 protocol. Their recovery outcome is a review candidate only; it does not edit the previous `unresolved_or_invalid` flag or original label status. G2-Cell3 RPT2 retains its repeated Step 4 timestamp issue; a valid M2 charge proxy, if present, remains a separate target and does not resolve M1.",
            "",
            "Fourteen original ACCEPTED records for G4-Cell18 and G4-Cell19 are not included in M1/M2 agreement statistics: the Guide lists two source CSVs for each selected primary BoL procedure, while the M1 candidate manifest selected one exact file. The audit preserves that existing selection and its raw block hash but does not choose between Guide source files or treat this mapping ambiguity as resolved.",
            "",
            "## Alternative BoL references and label policy",
            "",
            "- Part 1’s second Guide BoL procedure is retained as QC only. Its Step 4 native-counter sensitivity remains available from the prior audit. A secondary Step 10 denominator is considered usable only if the exact second-procedure file contains one complete, positive-direction Step 10 block reaching the documented upper voltage; no filename-order substitution is made.",
            "- Preserve the existing 178 ACCEPTED reconstructed discharge references unchanged. Keep all 25 REVIEW records out of the primary M1 cohort until a separate reviewer approves any corrected mapping/protocol evidence.",
            "- Keep any CHARGE_PROXY_ONLY estimate in a separate target table/target type; do not merge it with M1 labels or describe it as recovered discharge SoH.",
            "- No acceptance threshold was applied. If one is desired for M1/M2 agreement, it requires explicit scientific justification and approval.",
            "- G10-Cell2 still has no candidate mapping and is not fabricated.",
            "",
            "## Remaining limitations",
            "",
            "- Part 3 README does not state chamber temperature; actual cell temperature and its effect on charge/discharge capacity comparability remain unresolved.",
            "- The Step 10 pseudo-OCV charge window is not the Step 4 discharge window. Cross-method agreement cannot establish independent ground truth.",
            "- Exact Maccor time-unit configuration and Oxford `Amphr` export semantics remain unarchived; no absolute Ah claim or instrument tolerance is made.",
            "- Review outcomes are proposed audit dispositions only. User approval is required before any label-status change.",
            "",
            "`training_authorized: false`; existing manifests and historical outputs were preserved.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    candidates = _read_csv(CANDIDATE_MANIFEST)
    blocks = _read_csv(INDEPENDENT_BLOCK_AUDIT)
    guide_rows = _read_csv(GUIDE_MANIFEST)
    if len(candidates) != 203:
        raise ValueError(f"Expected 203 canonical M1 candidates; found {len(candidates)}.")
    if sum(row.get("label_disposition") == "ACCEPTED" for row in candidates) != 178:
        raise ValueError("The expected 178 original ACCEPTED candidates changed.")
    if sum(row.get("label_disposition") == "REVIEW" for row in candidates) != 25:
        raise ValueError("The expected 25 original REVIEW candidates changed.")
    if any(row.get("training_authorized", "").casefold() != "false" for row in candidates):
        raise ValueError("Candidate manifest unexpectedly authorizes training.")

    audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for block in blocks:
        key = _key(
            block["source_file"],
            int(block["source_part"]),
            int(block["step"]),
            int(block["start_line"]),
            int(block["end_line"]),
        )
        if key in audit_by_key:
            raise ValueError(f"Duplicate independent block-audit key: {key}")
        audit_by_key[key] = block
    step4_above = [
        _float(block.get("voltage_closest_above_2_5"))
        for block in blocks
        if block.get("step") == "4" and _float(block.get("voltage_closest_above_2_5")) is not None
    ]
    cutoff_above_v = min(step4_above) if step4_above else None

    primary_bol_sources: dict[tuple[str, int], None] = {}
    for candidate in candidates:
        primary_bol_sources[
            (candidate["bol_source_file"].replace("\\", "/"), int(candidate["bol_source_part"]))
        ] = None
    primary_bol_by_source: dict[tuple[str, int], dict[str, Any] | None] = {}
    primary_segments_by_source: dict[tuple[str, int], list[dict[str, Any]]] = {}
    primary_segment_status: dict[tuple[str, int], str] = {}
    for source_file, source_part in sorted(primary_bol_sources):
        selected, segments, status = _select_primary_step10_bol(source_file, source_part)
        primary_bol_by_source[(source_file, source_part)] = selected
        primary_segments_by_source[(source_file, source_part)] = segments
        primary_segment_status[(source_file, source_part)] = status

    part1_comparison = _read_csv(PART1_BOL_COMPARISON)
    secondary_sources: dict[str, str] = {}
    for row in part1_comparison:
        raw = row["second_source_file"].replace("\\", "/")
        source = raw if raw.startswith("data/raw/") else f"data/raw/oxford/part1/{raw}"
        secondary_sources[source] = row["physical_cell_id"]
    secondary_bol_by_source: dict[str, dict[str, Any] | None] = {}
    secondary_segment_rows: list[dict[str, Any]] = []
    secondary_status: dict[str, str] = {}
    for source_file in sorted(secondary_sources):
        selected, segments, status = _select_primary_step10_bol(source_file, 1)
        secondary_bol_by_source[source_file] = selected
        secondary_status[source_file] = status
        for segment in segments:
            secondary_segment_rows.append(
                {
                    **segment,
                    "physical_cell_id": secondary_sources[source_file],
                    "bol_role": "secondary_part1_guide_bol_qc_only",
                    "selection_status": status,
                }
            )

    primary_segment_rows: list[dict[str, Any]] = []
    for key, segments in primary_segments_by_source.items():
        for segment in segments:
            primary_segment_rows.append(
                {
                    **segment,
                    "physical_cell_id": next(
                        row["physical_cell_id"]
                        for row in candidates
                        if row["bol_source_file"].replace("\\", "/") == key[0]
                        and int(row["bol_source_part"]) == key[1]
                    ),
                    "bol_role": "primary_same_cell_guide_bol",
                    "selection_status": primary_segment_status[key],
                }
            )

    comparisons: list[dict[str, Any]] = []
    for candidate in candidates:
        part = int(candidate["dataset_part"])
        discharge = _lookup_block(audit_by_key, candidate["step4_segment_ref"], part)
        charge = _lookup_block(audit_by_key, candidate["step10_segment_ref"], part)
        bol_part = int(candidate["bol_source_part"])
        bol_discharge = _lookup_block(audit_by_key, candidate["bol_step4_segment_ref"], bol_part)
        source_key = (candidate["bol_source_file"].replace("\\", "/"), bol_part)
        bol_charge = primary_bol_by_source[source_key]
        guide_compatible, guide_note = _guide_bol_match(candidate, guide_rows)
        result = _make_comparison(
            candidate,
            discharge,
            charge,
            bol_discharge,
            bol_charge,
            guide_compatible=guide_compatible,
            guide_note=guide_note,
            cutoff_above_v=cutoff_above_v,
        )
        comparisons.append(result)

    reference_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in _read_csv(RECONSTRUCTED_REFERENCES)
    }
    if len(reference_by_key) != 203:
        raise ValueError("Reconstructed reference keys are missing or duplicated.")
    for result in comparisons:
        reference = reference_by_key[
            (str(result["dataset_part"]), result["physical_cell_id"], result["rpt_index"])
        ]
        if reference.get("label_disposition") != result["original_disposition"]:
            raise ValueError(
                f"Original label disposition changed for {result['physical_cell_id']} RPT{result['rpt_index']}."
            )

    review_rows: list[dict[str, Any]] = []
    prior_sensitivity = {
        (row["physical_cell_id"], row["rpt_index"]): row
        for row in _read_csv(PART1_SECONDARY_BOL_SENSITIVITY)
    }
    for result in comparisons:
        if result["original_disposition"] != "REVIEW":
            continue
        outcome, explanation = _review_outcome(result)
        result["recovery_outcome"] = outcome
        result["recovery_explanation"] = explanation
        result["status_promoted"] = False
        secondary_source = next(
            (
                source
                for source, cell in secondary_sources.items()
                if cell == result["physical_cell_id"]
            ),
            "",
        )
        secondary = secondary_bol_by_source.get(secondary_source)
        secondary_span = None if secondary is None else secondary.get("amphr_span_native")
        secondary_step10_q = (
            None
            if secondary is None
            else secondary.get("testtime_integral_abs_current_native")
        )
        result["secondary_bol_step10_status"] = (
            secondary_status.get(secondary_source, "no_alternative_part1_bol")
            if secondary_source
            else "no_alternative_part1_bol"
        )
        result["secondary_bol_step10_segment_ref"] = (
            "" if secondary is None else secondary["source_block_ref"]
        )
        result["secondary_bol_step10_block_sha256"] = (
            "" if secondary is None else secondary["raw_block_sha256"]
        )
        result["secondary_bol_step10_integrated_native_current_time"] = secondary_step10_q
        result["secondary_bol_step10_native_amphr_span"] = secondary_span
        sensitivity = prior_sensitivity.get(
            (result["physical_cell_id"], result["rpt_index"]), {}
        )
        result["secondary_bol_step4_native_counter_soh_percent_qc"] = _float(
            sensitivity.get("secondary_bol_native_counter_soh_percent_qc")
        )
        result["secondary_minus_primary_step4_native_counter_pp_qc"] = _float(
            sensitivity.get("secondary_minus_primary_soh_pp_qc")
        )
        review_rows.append(dict(result))

    agreement: list[dict[str, Any]] = []
    for result in comparisons:
        if (
            result["original_disposition"] == "ACCEPTED"
            and result["m1_status"] == "VALID"
            and result["m2_status"] == "VALID"
        ):
            agreement.append(
                {
                    "dataset_part": result["dataset_part"],
                    "physical_cell_id": result["physical_cell_id"],
                    "rpt_index": result["rpt_index"],
                    "partition": result["partition"],
                    "protocol_domain": result["protocol_domain"],
                    "soh_discharge_integrated_percent": result["soh_discharge_integrated_percent"],
                    "soh_charge_integrated_percent": result["soh_charge_integrated_percent"],
                    "soh_native_discharge_percent": result["soh_native_discharge_percent"],
                    "soh_native_charge_percent": result["soh_native_charge_percent"],
                    "charge_minus_discharge_pp": (
                        float(result["soh_charge_integrated_percent"])
                        - float(result["soh_discharge_integrated_percent"])
                    ),
                    "native_discharge_minus_integrated_pp": (
                        float(result["soh_native_discharge_percent"])
                        - float(result["soh_discharge_integrated_percent"])
                    ),
                    "native_charge_minus_integrated_pp": (
                        float(result["soh_native_charge_percent"])
                        - float(result["soh_charge_integrated_percent"])
                    ),
                    "absolute_difference_pp": abs(
                        float(result["soh_charge_integrated_percent"])
                        - float(result["soh_discharge_integrated_percent"])
                    ),
                    "step10_voltage_window_overlap_v": result["step10_voltage_window_overlap_v"],
                    "rpt_step10_voltage_min_v": result["step10_rpt_voltage_min_v"],
                    "rpt_step10_voltage_max_v": result["step10_rpt_voltage_max_v"],
                    "bol_step10_voltage_min_v": result["step10_bol_voltage_min_v"],
                    "bol_step10_voltage_max_v": result["step10_bol_voltage_max_v"],
                }
            )
    if len(agreement) > 178:
        raise ValueError(f"Valid paired ACCEPTED records exceed the original 178: {len(agreement)}.")

    summary = _summaries(comparisons, agreement)
    for result in comparisons:
        if result["original_disposition"] == "REVIEW" and "recovery_outcome" not in result:
            raise AssertionError("A REVIEW row lacks a recovery outcome.")
    review_rows = [row for row in comparisons if row["original_disposition"] == "REVIEW"]

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing audit outputs: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "soh_multimethod_comparison.csv"
    review_path = output_dir / "soh_review_recovery_audit.csv"
    agreement_path = output_dir / "soh_charge_discharge_agreement.csv"
    plot_path = output_dir / "soh_charge_discharge_agreement.png"
    report_path = output_dir / "soh_recovery_report.md"
    checksum_path = output_dir / "soh_recovery_checksums.json"
    _write_csv(comparison_path, comparisons)
    _write_csv(review_path, review_rows)
    _write_csv(agreement_path, agreement)
    _agreement_plot(plot_path, agreement)
    _write_report(
        report_path, summary, review_rows, primary_segment_rows, secondary_segment_rows
    )

    source_paths = [
        CANDIDATE_MANIFEST,
        RECONSTRUCTED_REFERENCES,
        INDEPENDENT_BLOCK_AUDIT,
        GUIDE_MANIFEST,
        PART3_BOL_CROSSWALK,
        PART1_BOL_COMPARISON,
        PART1_SECONDARY_BOL_SENSITIVITY,
        PROJECT_ROOT / "experiments" / "dataset_review" / "README.md",
        PROJECT_ROOT / "data" / "raw" / "oxford" / "part1" / "Readme.txt",
        PROJECT_ROOT / "data" / "raw" / "oxford" / "part2" / "Readme.txt",
        PROJECT_ROOT / "data" / "raw" / "oxford" / "part3" / "Readme.txt",
    ]
    input_hashes = {str(path): _sha256_file(path) for path in source_paths}
    output_hashes = {
        path.name: _sha256_file(path)
        for path in (comparison_path, review_path, agreement_path, plot_path, report_path)
    }
    checksum_manifest = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorized": False,
        "training_started": False,
        "candidate_count": summary["candidate_count"],
        "input_sha256": input_hashes,
        "output_sha256": output_hashes,
        "primary_step10_bol_block_sha256": {
            segment["source_block_ref"]: segment["raw_block_sha256"]
            for segment in primary_segment_rows
        },
        "secondary_step10_bol_block_sha256": {
            segment["source_block_ref"]: segment["raw_block_sha256"]
            for segment in secondary_segment_rows
        },
        "valid_method_counts": summary["valid_method_counts"],
    }
    checksum_path.write_text(
        json.dumps(checksum_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary["output_directory"] = str(output_dir)
    summary["output_checksums_manifest"] = str(checksum_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare independent Step 4/Step 10/native-counter SoH references."
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    run_audit(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
