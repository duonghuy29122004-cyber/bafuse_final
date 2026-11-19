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


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
M1_DIR = BENCHMARK_DIR / "m1_raw_validation_20261113"
GUIDE_MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "manifests"
    / "experiment_manifests"
    / "guide_observation_manifest"
    / "guide_observation_manifest.csv"
)
PART3_GUIDE_MANIFEST = (
    PROJECT_ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part3_bol_crosswalk"
    / "guide_observation_manifest.csv"
)
SPLIT_MANIFEST = BENCHMARK_DIR / "cell_split.csv"
OUTPUT_DIR = BENCHMARK_DIR / "independent_raw_soh_audit_v1_20261113"
BLOCK_REFERENCE_RE = re.compile(
    r"^(?P<source>.+)#Step(?P<step>\d+):segment[^:]*:rows"
    r"(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$",
    re.IGNORECASE,
)
CURVE_FIELDS = tuple(f"{name}_v_{index:03d}" for name in ("discharge", "charge") for index in range(128))
PHYSICS_FIELDS = (
    "mean_discharge_temperature_native",
    "discharge_temperature_rise_native",
    "discharge_plateau_slope_v_per_progress",
    "discharge_slope_variability_v_per_progress",
    "mean_absolute_discharge_current_native",
)


@dataclass(frozen=True)
class BlockReference:
    source: str
    step: int
    start_line: int
    end_line: int


@dataclass
class BlockRequest:
    source_file: str
    source_part: int
    step: int
    start_line: int
    end_line: int
    expected_hash: str
    expected_source_size: int
    expected_source_mtime_ns: int
    roles: list[tuple[str, str]] = field(default_factory=list)

    @property
    def key(self) -> tuple[str, int, int, int, int]:
        return (
            _normalize_path(self.source_file),
            self.source_part,
            self.step,
            self.start_line,
            self.end_line,
        )

    @property
    def expected_rows(self) -> int:
        return self.end_line - self.start_line + 1


class BlockSamples:
    def __init__(self, request: BlockRequest, columns: dict[str, int], width: int):
        self.request = request
        self.columns = columns
        self.width = width
        self.digest = hashlib.sha256()
        self.row_count = 0
        self.wrong_width_rows = 0
        self.values: dict[str, list[float | None]] = {
            name: [] for name in ("step", "testtime", "steptime", "amphr", "amps", "volts")
        }
        self.values["temperature"] = []

    def _value(self, row: list[str], name: str) -> float | None:
        column = self.columns.get(name)
        if column is None or column >= len(row):
            return None
        try:
            value = float(row[column])
        except (ValueError, TypeError):
            return None
        return value if math.isfinite(value) else None

    def add(self, row: list[str]) -> None:
        self.row_count += 1
        self.digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
        self.wrong_width_rows += int(len(row) != self.width)
        for name in self.values:
            self.values[name].append(self._value(row, name))

    def result(self, next_step: int | None) -> dict[str, Any]:
        steps = self.values["step"]
        integral_testtime, testtime_stats = _integrate(
            self.values["testtime"], self.values["amps"]
        )
        integral_steptime, steptime_stats = _integrate(
            self.values["steptime"], self.values["amps"]
        )
        capacities = [value for value in self.values["amphr"] if value is not None]
        voltages = [value for value in self.values["volts"] if value is not None]
        currents = [value for value in self.values["amps"] if value is not None]
        temperature = [value for value in self.values["temperature"] if value is not None]
        signs = Counter(
            "negative" if value < 0 else "positive" if value > 0 else "zero"
            for value in currents
        )
        capacity_changes = [
            right - left
            for left, right in zip(self.values["amphr"], self.values["amphr"][1:])
            if left is not None and right is not None
        ]
        step_ids = sorted({int(value) for value in steps if value is not None and value.is_integer()})
        missing = {
            name: sum(value is None for value in values)
            for name, values in self.values.items()
        }
        cap_span = max(capacities) - min(capacities) if capacities else None
        cap_integral_discrepancy = (
            100.0 * abs(integral_testtime / 60.0 - cap_span) / cap_span
            if cap_span not in (None, 0)
            else None
        )
        result = {
            "source_file": self.request.source_file,
            "source_part": self.request.source_part,
            "step": self.request.step,
            "start_line": self.request.start_line,
            "end_line": self.request.end_line,
            "expected_row_count": self.request.expected_rows,
            "row_count": self.row_count,
            "row_count_exact": self.row_count == self.request.expected_rows,
            "block_sha256_expected": self.request.expected_hash,
            "block_sha256_independent": self.digest.hexdigest(),
            "block_hash_exact": self.digest.hexdigest() == self.request.expected_hash,
            "step_ids": ";".join(str(value) for value in step_ids),
            "step_ids_exact": step_ids == [self.request.step] and missing["step"] == 0,
            "next_step": next_step,
            "next_row_transitions_step": next_step is not None and next_step != self.request.step,
            "wrong_width_rows": self.wrong_width_rows,
            "missing_testtime": missing["testtime"],
            "missing_steptime": missing["steptime"],
            "missing_amphr": missing["amphr"],
            "missing_amps": missing["amps"],
            "missing_volts": missing["volts"],
            "missing_temperature": missing["temperature"],
            "current_negative_rows": signs["negative"],
            "current_positive_rows": signs["positive"],
            "current_zero_rows": signs["zero"],
            "current_min": min(currents) if currents else None,
            "current_max": max(currents) if currents else None,
            "amphr_min": min(capacities) if capacities else None,
            "amphr_max": max(capacities) if capacities else None,
            "amphr_span_native": cap_span,
            "amphr_negative_deltas": sum(value < 0 for value in capacity_changes),
            "amphr_zero_deltas": sum(value == 0 for value in capacity_changes),
            "amphr_positive_deltas": sum(value > 0 for value in capacity_changes),
            "voltage_first": next((value for value in self.values["volts"] if value is not None), None),
            "voltage_last": next((value for value in reversed(self.values["volts"]) if value is not None), None),
            "voltage_min": min(voltages) if voltages else None,
            "voltage_max": max(voltages) if voltages else None,
            "voltage_closest_below_2_5": max(
                (value for value in voltages if value <= 2.5), default=None
            ),
            "voltage_closest_above_2_5": min(
                (value for value in voltages if value > 2.5), default=None
            ),
            "testtime_integrated_abs_current_native": integral_testtime,
            "steptime_integrated_abs_current_native": integral_steptime,
            "testtime_zero_intervals": testtime_stats["zero"],
            "testtime_negative_intervals": testtime_stats["negative"],
            "testtime_invalid_intervals": testtime_stats["invalid"],
            "testtime_positive_dt_min": testtime_stats["positive_min"],
            "testtime_positive_dt_median": testtime_stats["positive_median"],
            "testtime_positive_dt_max": testtime_stats["positive_max"],
            "steptime_zero_intervals": steptime_stats["zero"],
            "steptime_negative_intervals": steptime_stats["negative"],
            "steptime_invalid_intervals": steptime_stats["invalid"],
            "steptime_positive_dt_min": steptime_stats["positive_min"],
            "steptime_positive_dt_median": steptime_stats["positive_median"],
            "steptime_positive_dt_max": steptime_stats["positive_max"],
            "temperature_min_native": min(temperature) if temperature else None,
            "temperature_max_native": max(temperature) if temperature else None,
            "temperature_mean_native": statistics.mean(temperature) if temperature else None,
            "testtime_vs_amphr_discrepancy_if_minutes_percent": cap_integral_discrepancy,
            "raw_block_status": "PENDING",
        }
        finite_voltage = [value for value in self.values["volts"] if value is not None]
        if len(finite_voltage) == len(self.values["volts"]) and len(finite_voltage) >= 2:
            result["_resampled_voltage_values"] = _resample(finite_voltage)
        if (
            self.request.step == 4
            and len(temperature) == len(self.values["temperature"])
            and len(currents) == len(self.values["amps"])
            and len(finite_voltage) == len(self.values["volts"])
        ):
            try:
                physical = recompute_physics_features(finite_voltage, currents, temperature)
            except ValueError:
                physical = {}
            for name in PHYSICS_FIELDS:
                result[f"independent_physics_{name}"] = physical.get(name)
        return result


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").strip("/").casefold()


def parse_block_reference(reference: str) -> BlockReference:
    match = BLOCK_REFERENCE_RE.fullmatch(reference.strip())
    if match is None:
        raise ValueError(f"Unparseable source block reference: {reference!r}")
    start, end = int(match.group("start")), int(match.group("end"))
    if start < 1 or end < start:
        raise ValueError(f"Invalid source line interval: {reference!r}")
    return BlockReference(
        match.group("source"),
        int(match.group("step")),
        start,
        end,
    )


def _integrate(
    time_values: list[float | None], current_values: list[float | None]
) -> tuple[float, dict[str, float | int | None]]:
    if len(time_values) != len(current_values):
        raise ValueError("Time and current sample counts differ.")
    areas: list[float] = []
    gaps: list[float] = []
    zero = negative = invalid = 0
    for index in range(1, len(time_values)):
        t0, t1 = time_values[index - 1], time_values[index]
        i0, i1 = current_values[index - 1], current_values[index]
        if t0 is None or t1 is None or i0 is None or i1 is None:
            invalid += 1
            continue
        delta = t1 - t0
        if delta == 0:
            zero += 1
        elif delta < 0:
            negative += 1
        else:
            gaps.append(delta)
            areas.append((abs(i0) + abs(i1)) * delta * 0.5)
    stats: dict[str, float | int | None] = {
        "zero": zero,
        "negative": negative,
        "invalid": invalid,
        "positive_min": min(gaps) if gaps else None,
        "positive_median": statistics.median(gaps) if gaps else None,
        "positive_max": max(gaps) if gaps else None,
    }
    return math.fsum(areas), stats


def _resample(values: list[float], count: int = 128) -> list[float]:
    if len(values) < 2:
        raise ValueError("At least two raw samples are required.")
    last = len(values) - 1
    result = []
    for index in range(count):
        position = index * last / (count - 1)
        low = int(position)
        high = min(low + 1, last)
        fraction = position - low
        result.append(values[low] * (1.0 - fraction) + values[high] * fraction)
    return result


def _slope(x: list[float], y: list[float]) -> float:
    x_mean = statistics.mean(x)
    y_mean = statistics.mean(y)
    denominator = math.fsum((value - x_mean) ** 2 for value in x)
    if denominator <= 0:
        raise ValueError("No variation in normalized curve progress.")
    return math.fsum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in zip(x, y)
    ) / denominator


def recompute_physics_features(
    voltage: list[float], current: list[float], temperature: list[float]
) -> dict[str, float]:
    curve = _resample(voltage)
    progress = [index / (len(curve) - 1) for index in range(len(curve))]
    plateau = [index for index, value in enumerate(progress) if 0.2 <= value <= 0.8]
    plateau_slope = _slope(
        [progress[index] for index in plateau],
        [curve[index] for index in plateau],
    )
    derivatives = [
        (curve[index + 1] - curve[index - 1])
        / (progress[index + 1] - progress[index - 1])
        for index in range(1, len(curve) - 1)
        if 0.1 <= progress[index] <= 0.9
    ]
    median_derivative = statistics.median(derivatives)
    slope_variability = 1.4826 * statistics.median(
        abs(value - median_derivative) for value in derivatives
    )
    return {
        "mean_discharge_temperature_native": statistics.mean(temperature),
        "discharge_temperature_rise_native": max(temperature) - temperature[0],
        "discharge_plateau_slope_v_per_progress": plateau_slope,
        "discharge_slope_variability_v_per_progress": slope_variability,
        "mean_absolute_discharge_current_native": statistics.mean(abs(value) for value in current),
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _resolve_raw_file(relative: str) -> Path:
    path = PROJECT_ROOT.joinpath(*relative.replace("\\", "/").split("/"))
    if not path.is_file():
        raise FileNotFoundError(f"Mapped raw source does not exist: {path}")
    return path


def _tail_for_part(path: str, part: int) -> str:
    normalized = path.replace("\\", "/").strip("/")
    marker = f"/part{part}/"
    folded = normalized.casefold()
    position = folded.find(marker)
    if position >= 0:
        return normalized[position + len(marker):].casefold()
    return normalized.casefold()


def _same_mapped_file(reference_source: str, source_file: str, part: int) -> bool:
    return _tail_for_part(reference_source, part) == _tail_for_part(source_file, part)


def _split_map(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    split = {row["physical_cell_id"]: row["partition"] for row in rows}
    if len(split) != len(rows):
        raise ValueError("Duplicate physical cell in split manifest.")
    counts = Counter(split.values())
    if dict(counts) != {"train": 9, "validation": 3, "external_test": 12}:
        raise ValueError(f"Unexpected physical-cell partition counts: {dict(counts)}")
    return split


def _add_request(
    requests: dict[tuple[str, int, int, int, int], BlockRequest],
    block_audit: dict[tuple[str, int, int, int, int], dict[str, str]],
    *,
    reference: str,
    source_file: str,
    source_part: int,
    expected_step: int,
    expected_hash: str,
    role: str,
    record_key: str,
) -> None:
    parsed = parse_block_reference(reference)
    if parsed.step != expected_step:
        raise ValueError(f"{role} expected Step {expected_step}, found Step {parsed.step}: {reference}")
    if not _same_mapped_file(parsed.source, source_file, source_part):
        raise ValueError(
            f"{role} reference path does not match mapped source {source_file!r}: {reference!r}"
        )
    normalized_source = _normalize_path(source_file)
    key = (normalized_source, source_part, parsed.step, parsed.start_line, parsed.end_line)
    audit = block_audit.get(key)
    if audit is None:
        raise ValueError(f"M1 block audit lacks exact mapped block for {role}: {reference}")
    if audit["raw_block_sha256"] != expected_hash:
        raise ValueError(f"Manifest/M1 block hash mismatch for {role}: {reference}")
    request = requests.get(key)
    if request is None:
        request = BlockRequest(
            source_file=source_file.replace("\\", "/"),
            source_part=source_part,
            step=parsed.step,
            start_line=parsed.start_line,
            end_line=parsed.end_line,
            expected_hash=expected_hash,
            expected_source_size=int(audit["source_size_bytes"]),
            expected_source_mtime_ns=int(audit["source_mtime_ns"]),
        )
        requests[key] = request
    elif request.expected_hash != expected_hash:
        raise ValueError(f"Conflicting expected hashes for shared block: {reference}")
    request.roles.append((role, record_key))


def _scan_one_source(
    source_file: str,
    source_requests: list[BlockRequest],
) -> list[dict[str, Any]]:
    path = _resolve_raw_file(source_file)
    before = path.stat()
    for request in source_requests:
        if before.st_size != request.expected_source_size:
            raise ValueError(f"Raw file size differs from M1 audit: {source_file}")
        if before.st_mtime_ns != request.expected_source_mtime_ns:
            raise ValueError(f"Raw file modification time differs from M1 audit: {source_file}")
    source_requests.sort(key=lambda item: (item.start_line, item.end_line))
    start_map: defaultdict[int, list[BlockRequest]] = defaultdict(list)
    end_map: defaultdict[int, list[BlockRequest]] = defaultdict(list)
    for request in source_requests:
        start_map[request.start_line].append(request)
        end_map[request.end_line + 1].append(request)
    max_end = max(request.end_line for request in source_requests)
    active: list[BlockRequest] = []
    states: dict[tuple[str, int, int, int, int], BlockSamples] = {}
    next_steps: dict[tuple[str, int, int, int, int], int | None] = {}
    results: list[dict[str, Any]] = []

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Missing header in mapped file: {source_file}")
        folded = [value.strip().casefold() for value in header]
        columns: dict[str, int] = {}
        for name in ("step", "testtime", "steptime", "amphr", "amps", "volts"):
            matches = [index for index, value in enumerate(folded) if value == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} field in {source_file}; found {len(matches)}")
            columns[name] = matches[0]
        temperature_names = ("temp1", "temp", "temperature")
        temperature = next(
            (name for name in temperature_names if folded.count(name) == 1),
            None,
        )
        if temperature is not None:
            columns["temperature"] = folded.index(temperature)
        for row in reader:
            line_number = reader.line_num
            if line_number > max_end + 1:
                break
            try:
                raw_step = float(row[columns["step"]])
                step = int(raw_step) if math.isfinite(raw_step) and raw_step.is_integer() else None
            except (IndexError, ValueError):
                step = None
            for request in end_map.get(line_number, ()):
                next_steps[request.key] = step
            active.extend(start_map.get(line_number, ()))
            for request in active:
                if line_number <= request.end_line:
                    state = states.get(request.key)
                    if state is None:
                        state = BlockSamples(request, columns, len(header))
                        states[request.key] = state
                    state.add(row)
            active = [request for request in active if line_number < request.end_line]
            if line_number >= max_end + 1:
                break

    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Raw file changed during independent audit: {source_file}")
    for request in source_requests:
        state = states.get(request.key)
        if state is None:
            raise ValueError(f"Mapped block was not reached: {request}")
        result = state.result(next_steps.get(request.key))
        result["roles"] = ";".join(f"{role}:{record}" for role, record in request.roles)
        required_missing = sum(
            result[f"missing_{field}"]
            for field in ("testtime", "steptime", "amphr", "amps", "volts")
        )
        integrity_ok = (
            result["row_count_exact"]
            and result["block_hash_exact"]
            and result["step_ids_exact"]
            and result["wrong_width_rows"] == 0
            and required_missing == 0
            and result["next_row_transitions_step"]
        )
        result["raw_block_status"] = "ACCEPTED" if integrity_ok else "REVIEW"
        result["temperature_column"] = temperature or ""
        results.append(result)
    return results


def _guide_mapping_status(
    row: dict[str, str],
    guide_by_key: dict[tuple[str, str, str], list[dict[str, str]]],
    guide_by_cell: dict[tuple[str, str], list[dict[str, str]]],
    part3_guide_by_cell: dict[tuple[str, str], list[dict[str, str]]] | None = None,
) -> tuple[str, str]:
    part = row["dataset_part"]
    key = (part, row["physical_cell_id"], row["measurement_id"])
    matches = guide_by_key.get(key, [])
    if len(matches) != 1:
        return "REVIEW", f"Guide observation key matched {len(matches)} rows"
    guide = matches[0]
    if not _same_mapped_file(guide["source_file"], row["source_file"], int(part)):
        return "REVIEW", "Candidate source file differs from the Guide observation mapping"
    if int(guide["rpt_order"]) != int(row["rpt_index"]):
        return "REVIEW", "Candidate RPT index differs from Guide RPT order"

    bol_key = (row["bol_source_part"], row["physical_cell_id"])
    bol_manifest = (
        part3_guide_by_cell
        if int(row["bol_source_part"]) == 3 and part3_guide_by_cell is not None
        else guide_by_cell
    )
    bol_matches = [
        item
        for item in bol_manifest.get(bol_key, [])
        if item["measurement_id"].casefold() == "bol"
        and any(
            _same_mapped_file(source.strip(), row["bol_source_file"], int(row["bol_source_part"]))
            for source in item["source_file"].split(";")
        )
    ]
    if len(bol_matches) != 1:
        return "REVIEW", f"Guide BoL source mapping matched {len(bol_matches)} rows"
    if guide["measurement_mapping_status"] != "single_checkpoint":
        return "REVIEW", f"Guide measurement mapping status is {guide['measurement_mapping_status']}"
    if guide["source_mapping_status"] != "unique_source_file":
        return "REVIEW", f"Guide source mapping status is {guide['source_mapping_status']}"
    if bol_matches[0]["measurement_mapping_status"] != "single_checkpoint":
        return "REVIEW", (
            "Guide BoL measurement mapping status is "
            f"{bol_matches[0]['measurement_mapping_status']}"
        )
    if bol_matches[0]["source_mapping_status"] not in {
        "unique_source_file",
        "multiple_source_files",
    }:
        return "REVIEW", f"Guide BoL source mapping status is {bol_matches[0]['source_mapping_status']}"
    return (
        "ACCEPTED",
        "Part/cell/RPT/source and exact same-cell BoL source member match the Guide crosswalk; Step blocks are checked against M1 and raw hashes",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty audit table: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _jsonable(path: Path) -> str:
    return str(path.resolve())


def run_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty audit output: {output_dir}")
    candidates = _read_csv(RECONSTRUCTION_DIR / "final_candidate_manifest.csv")
    provenance = _read_csv(RECONSTRUCTION_DIR / "label_provenance.csv")
    old_references = _read_csv(RECONSTRUCTION_DIR / "reconstructed_soh_references.csv")
    feature_rows = _read_csv(RECONSTRUCTION_DIR / "raw_curve_feature_matrix.csv")
    m1_blocks = _read_csv(M1_DIR / "measurement_block_audit.csv")
    guide_rows = _read_csv(GUIDE_MANIFEST)
    part3_guide_rows = _read_csv(PART3_GUIDE_MANIFEST)
    split = _split_map(SPLIT_MANIFEST)
    if len(candidates) != 203:
        raise ValueError(f"Expected 203 canonical candidates, found {len(candidates)}")
    candidate_keys = [
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in candidates
    ]
    if len(set(candidate_keys)) != len(candidate_keys):
        raise ValueError("Duplicate (part, physical cell, RPT) candidate keys.")

    block_audit: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for row in m1_blocks:
        key = (
            _normalize_path(row["source_file"]),
            int(row["source_part"]),
            int(row["expected_step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        if key in block_audit:
            raise ValueError(f"Duplicate M1 block audit key: {key}")
        block_audit[key] = row

    requests: dict[tuple[str, int, int, int, int], BlockRequest] = {}
    guide_by_key: defaultdict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    guide_by_cell: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    part3_guide_by_cell: defaultdict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in guide_rows:
        guide_by_key[
            (row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        ].append(row)
        guide_by_cell[(row["dataset_part"], row["physical_cell_id"])].append(row)
    for row in part3_guide_rows:
        part3_guide_by_cell[(row["dataset_part"], row["physical_cell_id"])].append(row)

    mapping_by_key: dict[tuple[str, str, str], tuple[str, str]] = {}
    for row in candidates:
        part = int(row["dataset_part"])
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if split.get(row["physical_cell_id"]) != row["partition"]:
            raise ValueError(f"Split mismatch for {key}")
        if split.get(row["physical_cell_id"]) not in {"train", "validation", "external_test"}:
            raise ValueError(f"Unassigned physical cell for {key}")
        mapping_by_key[key] = _guide_mapping_status(
            row, guide_by_key, guide_by_cell, part3_guide_by_cell
        )
        for role, ref_column, source_column, source_part, step, hash_column in (
            ("rpt_step4", "step4_segment_ref", "source_file", part, 4, "step4_block_sha256"),
            ("rpt_step10", "step10_segment_ref", "source_file", part, 10, "step10_block_sha256"),
            ("bol_step4", "bol_step4_segment_ref", "bol_source_file", int(row["bol_source_part"]), 4, "bol_block_sha256"),
        ):
            _add_request(
                requests,
                block_audit,
                reference=row[ref_column],
                source_file=row[source_column],
                source_part=source_part,
                expected_step=step,
                expected_hash=row[hash_column],
                role=role,
                record_key="|".join(key),
            )

    control_observations = [
        row
        for row in _read_csv(PROJECT_ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv")
        if row.get("part") == "1"
        and row.get("physical_cell_id") == "G1-Cell9"
        and row.get("rpt_index") == "8"
    ]
    if len(control_observations) != 1:
        raise ValueError(f"Expected one G1-Cell9 RPT8 control mapping, found {len(control_observations)}")
    control = control_observations[0]
    for role, ref_column, step in (
        ("excluded_control_step4", "step4_segment_ref", 4),
        ("excluded_control_step10", "step10_segment_ref", 10),
    ):
        reference = parse_block_reference(control[ref_column])
        source = control["source_file"]
        key = (_normalize_path(source), 1, step, reference.start_line, reference.end_line)
        audit = block_audit.get(key)
        if audit is None or int(audit["expected_step"]) != step:
            raise ValueError(f"M1 raw block audit lacks the G1-Cell9 RPT8 {role} block")
        _add_request(
            requests,
            block_audit,
            reference=control[ref_column],
            source_file=source,
            source_part=1,
            expected_step=step,
            expected_hash=audit["raw_block_sha256"],
            role=role,
            record_key="1|G1-Cell9|8",
        )

    by_source: defaultdict[str, list[BlockRequest]] = defaultdict(list)
    for request in requests.values():
        by_source[request.source_file].append(request)
    block_results: list[dict[str, Any]] = []
    for source_file, source_requests in sorted(by_source.items()):
        block_results.extend(_scan_one_source(source_file, source_requests))
    block_by_key = {
        (
            _normalize_path(row["source_file"]),
            int(row["source_part"]),
            int(row["step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        ): row
        for row in block_results
    }
    cutoff_below = max(
        (float(row["voltage_closest_below_2_5"]) for row in block_results
         if int(row["step"]) == 4 and row["voltage_closest_below_2_5"] is not None),
        default=None,
    )
    cutoff_above = min(
        (float(row["voltage_closest_above_2_5"]) for row in block_results
         if int(row["step"]) == 4 and row["voltage_closest_above_2_5"] is not None),
        default=None,
    )
    provenance_cutoffs = {
        (
            float(row["mapped_step4_closest_observed_voltage_below_cutoff_v"]),
            float(row["mapped_step4_closest_observed_voltage_above_cutoff_v"]),
        )
        for row in provenance
    }
    if len(provenance_cutoffs) != 1 or (cutoff_below, cutoff_above) not in provenance_cutoffs:
        raise ValueError(
            f"Independent raw cutoff bracket differs from provenance: "
            f"{(cutoff_below, cutoff_above)} versus {provenance_cutoffs}"
        )

    old_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in old_references:
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in old_by_key:
            raise ValueError(f"Duplicate reconstructed reference key: {key}")
        old_by_key[key] = row
    if set(old_by_key) != set(candidate_keys):
        raise ValueError("Reconstructed reference keys differ from canonical candidates.")
    feature_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in feature_rows:
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in feature_by_key:
            raise ValueError(f"Duplicate raw feature key: {key}")
        feature_by_key[key] = row
    provenance_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in provenance:
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in provenance_by_key:
            raise ValueError(f"Duplicate label provenance key: {key}")
        provenance_by_key[key] = row
    if set(provenance_by_key) != set(candidate_keys):
        raise ValueError("Label provenance keys differ from canonical candidates.")

    record_rows: list[dict[str, Any]] = []
    mapping_mismatches = 0
    label_mismatches = 0
    raw_block_review_records = 0
    for row in candidates:
        part = int(row["dataset_part"])
        identity = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        old = old_by_key[identity]
        provenance_row = provenance_by_key.get(identity)
        if provenance_row is None:
            raise ValueError(f"Missing unique label provenance for {identity}")
        guide_status, guide_note = mapping_by_key[identity]
        rpt_ref = parse_block_reference(row["step4_segment_ref"])
        charge_ref = parse_block_reference(row["step10_segment_ref"])
        bol_ref = parse_block_reference(row["bol_step4_segment_ref"])
        rpt_key = (_normalize_path(row["source_file"]), part, 4, rpt_ref.start_line, rpt_ref.end_line)
        charge_key = (_normalize_path(row["source_file"]), part, 10, charge_ref.start_line, charge_ref.end_line)
        bol_key = (
            _normalize_path(row["bol_source_file"]),
            int(row["bol_source_part"]),
            4,
            bol_ref.start_line,
            bol_ref.end_line,
        )
        discharge = block_by_key[rpt_key]
        charge = block_by_key[charge_key]
        bol = block_by_key[bol_key]
        ratio = None
        steptime_ratio = None
        if float(bol["testtime_integrated_abs_current_native"]) > 0:
            ratio = (
                100.0
                * float(discharge["testtime_integrated_abs_current_native"])
                / float(bol["testtime_integrated_abs_current_native"])
            )
        if float(bol["steptime_integrated_abs_current_native"]) > 0:
            steptime_ratio = (
                100.0
                * float(discharge["steptime_integrated_abs_current_native"])
                / float(bol["steptime_integrated_abs_current_native"])
            )
        old_soh = float(old["reconstructed_soh_percent"]) if old["reconstructed_soh_percent"] else None
        delta_pp = ratio - old_soh if ratio is not None and old_soh is not None else None
        label_computation_matches = delta_pp is None or abs(delta_pp) <= 1e-8
        if not label_computation_matches:
            label_mismatches += 1
        raw_blocks = (discharge, charge, bol)
        raw_ok = all(item["raw_block_status"] == "ACCEPTED" for item in raw_blocks)
        if not raw_ok:
            raw_block_review_records += 1
        guide_ok = guide_status == "ACCEPTED"
        if not guide_ok:
            mapping_mismatches += 1
        sign_ok = (
            discharge["current_negative_rows"] == discharge["row_count"]
            and charge["current_positive_rows"] == charge["row_count"]
            and bol["current_negative_rows"] == bol["row_count"]
        )
        time_ok = (
            discharge["testtime_negative_intervals"] == 0
            and discharge["testtime_zero_intervals"] == 0
            and bol["testtime_negative_intervals"] == 0
            and bol["testtime_zero_intervals"] == 0
            and discharge["steptime_negative_intervals"] == 0
            and discharge["steptime_zero_intervals"] == 0
            and bol["steptime_negative_intervals"] == 0
            and bol["steptime_zero_intervals"] == 0
        )
        protocol = row["reference_protocol_compatibility"]
        if (
            row["bol_resolution_status"] != "resolved"
            or protocol != "same_cell_step4_and_part_protocol_crosswalk"
            or not guide_ok
        ):
            protocol_status = "REVIEW"
        else:
            protocol_status = "ACCEPTED"
        endpoint_ok = (
            discharge["voltage_last"] is not None
            and (
                float(discharge["voltage_last"]) <= 2.5
                or float(discharge["voltage_last"])
                == float(discharge["voltage_closest_above_2_5"])
            )
            and discharge["next_row_transitions_step"]
        )
        feature = feature_by_key.get(identity)
        feature_status = "REVIEW"
        physics_max_abs_delta = None
        curve_max_abs_delta = None
        if feature is not None:
            try:
                discharge_curve_deltas = [
                    abs(
                        float(discharge["_resampled_voltage_values"][index])
                        - float(feature[f"discharge_v_{index:03d}"])
                    )
                    for index in range(128)
                ]
                charge_curve_deltas = [
                    abs(
                        float(charge["_resampled_voltage_values"][index])
                        - float(feature[f"charge_v_{index:03d}"])
                    )
                    for index in range(128)
                ]
                curve_max_abs_delta = max(discharge_curve_deltas + charge_curve_deltas)
                physical_deltas = [
                    abs(
                        float(discharge[f"independent_physics_{name}"])
                        - float(feature[name])
                    )
                    for name in PHYSICS_FIELDS
                ]
                physics_max_abs_delta = max(physical_deltas)
                feature_status = (
                    "ACCEPTED"
                    if curve_max_abs_delta <= 1e-10 and physics_max_abs_delta <= 1e-8
                    else "REVIEW"
                )
            except (KeyError, TypeError, ValueError):
                feature_status = "REVIEW"
        provenance_matches = (
            provenance_row["step4_block_sha256"] == discharge["block_sha256_independent"]
            and provenance_row["step10_block_sha256"] == charge["block_sha256_independent"]
            and provenance_row["bol_block_sha256"] == bol["block_sha256_independent"]
        )
        if not provenance_matches:
            guide_status = "REVIEW"
            guide_note += "; source-provenance block hashes do not match independently read blocks"
        per_record_disposition = (
            "ACCEPTED"
            if raw_ok and guide_ok and sign_ok and time_ok and endpoint_ok and protocol_status == "ACCEPTED" and feature_status == "ACCEPTED" and provenance_matches and ratio is not None and label_computation_matches
            else "REVIEW"
        )
        record_rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "measurement_id": row["measurement_id"],
                "partition": row["partition"],
                "raw_source_file": row["source_file"],
                "step4_segment_ref": row["step4_segment_ref"],
                "step10_segment_ref": row["step10_segment_ref"],
                "bol_source_part": row["bol_source_part"],
                "bol_source_file": row["bol_source_file"],
                "bol_step4_segment_ref": row["bol_step4_segment_ref"],
                "guide_mapping_status": guide_status,
                "guide_mapping_note": guide_note,
                "provenance_hashes_match_independent_raw": provenance_matches,
                "raw_step4_status": discharge["raw_block_status"],
                "raw_step10_status": charge["raw_block_status"],
                "raw_bol_step4_status": bol["raw_block_status"],
                "protocol_compatibility_status": protocol_status,
                "bol_same_cell_identity_check": guide_ok,
                "step4_current_direction_valid": discharge["current_negative_rows"] == discharge["row_count"],
                "step10_current_direction_valid": charge["current_positive_rows"] == charge["row_count"],
                "bol_current_direction_valid": bol["current_negative_rows"] == bol["row_count"],
                "step4_testtime_zero_intervals": discharge["testtime_zero_intervals"],
                "step4_testtime_negative_intervals": discharge["testtime_negative_intervals"],
                "step4_steptime_zero_intervals": discharge["steptime_zero_intervals"],
                "step4_steptime_negative_intervals": discharge["steptime_negative_intervals"],
                "step4_voltage_first_v": discharge["voltage_first"],
                "step4_voltage_last_v": discharge["voltage_last"],
                "step4_voltage_min_v": discharge["voltage_min"],
                "step4_voltage_max_v": discharge["voltage_max"],
                "step4_endpoint_valid_under_documented_2_5v_rule": endpoint_ok,
                "step4_integrated_capacity_testtime_native": discharge["testtime_integrated_abs_current_native"],
                "step4_integrated_capacity_steptime_native": discharge["steptime_integrated_abs_current_native"],
                "step4_amphr_span_native": discharge["amphr_span_native"],
                "bol_integrated_capacity_testtime_native": bol["testtime_integrated_abs_current_native"],
                "bol_integrated_capacity_steptime_native": bol["steptime_integrated_abs_current_native"],
                "bol_amphr_span_native": bol["amphr_span_native"],
                "independent_soh_testtime_percent": ratio,
                "independent_soh_steptime_percent": steptime_ratio,
                "stored_reconstructed_soh_percent": old_soh,
                "independent_minus_stored_soh_pp": delta_pp,
                "stored_label_matches_independent_within_float_check": label_computation_matches,
                "native_amphr_time_conversion_diagnostic_only_percent": discharge["testtime_vs_amphr_discrepancy_if_minutes_percent"],
                "temperature_unit_status": row["physics_unit_status"],
                "physics_feature_alignment_status": feature_status,
                "physics_feature_max_abs_difference": physics_max_abs_delta,
                "curve_feature_max_abs_difference_v": curve_max_abs_delta,
                "raw_recomputation_disposition": per_record_disposition,
                "current_manifest_disposition": row["label_disposition"],
                "training_disposition": "REVIEW_PENDING_GLOBAL_UNIT_AND_UNCERTAINTY_EVIDENCE",
                "training_authorized": False,
            }
        )

    control_step4 = parse_block_reference(control["step4_segment_ref"])
    control_step10 = parse_block_reference(control["step10_segment_ref"])
    control_step4_row = block_by_key[
        (_normalize_path(control["source_file"]), 1, 4, control_step4.start_line, control_step4.end_line)
    ]
    control_step10_row = block_by_key[
        (_normalize_path(control["source_file"]), 1, 10, control_step10.start_line, control_step10.end_line)
    ]
    record_rows.append(
        {
            "dataset_part": 1,
            "physical_cell_id": "G1-Cell9",
            "rpt_index": 8,
            "measurement_id": "RPT8",
            "partition": "validation",
            "raw_source_file": control["source_file"],
            "step4_segment_ref": control["step4_segment_ref"],
            "step10_segment_ref": control["step10_segment_ref"],
            "step4_segment_rows": control_step4_row["row_count"],
            "step4_voltage_first_v": control_step4_row["voltage_first"],
            "step4_voltage_last_v": control_step4_row["voltage_last"],
            "step4_voltage_min_v": control_step4_row["voltage_min"],
            "step4_voltage_max_v": control_step4_row["voltage_max"],
            "raw_step4_status": control_step4_row["raw_block_status"],
            "raw_step10_status": control_step10_row["raw_block_status"],
            "raw_recomputation_disposition": "REJECTED",
            "current_manifest_disposition": "EXCLUDED_CONTROL",
            "training_disposition": "EXCLUDED_INCOMPLETE_STEP4_DISCHARGE",
            "training_authorized": False,
        }
    )

    # Report the global endpoint bracket from observed block endpoint values only as a diagnostic;
    # the exact cutoff comparison above remains tied to each Guide-mapped Step 4 endpoint.
    feature_rows_internal = [
        row for row in record_rows
        if row["partition"] in {"train", "validation"}
        and row["current_manifest_disposition"] == "ACCEPTED"
    ]
    cells_by_partition: defaultdict[str, set[str]] = defaultdict(set)
    for row in record_rows:
        cells_by_partition[row["partition"]].add(row["physical_cell_id"])
    candidate_audits = [
        row for row in record_rows if row.get("current_manifest_disposition") != "EXCLUDED_CONTROL"
    ]
    originally_accepted = [
        row for row in candidate_audits if row["current_manifest_disposition"] == "ACCEPTED"
    ]
    recomputed_dispositions = Counter(
        row["raw_recomputation_disposition"] for row in candidate_audits
    )
    original_dispositions = Counter(
        row["current_manifest_disposition"] for row in candidate_audits
    )
    max_abs_delta = max(
        (abs(float(row["independent_minus_stored_soh_pp"])) for row in originally_accepted
         if row["independent_minus_stored_soh_pp"] is not None),
        default=None,
    )
    summary = {
        "schema_version": 1,
        "audit_version": "independent-raw-soh-audit-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "independent_implementation": "stdlib csv streaming; per-interval trapezoid areas collected and math.fsum-aggregated; no reconstruction module imported",
        "candidate_records": len(candidates),
        "original_candidate_disposition_counts": dict(original_dispositions),
        "raw_recomputation_disposition_counts": dict(recomputed_dispositions),
        "originally_accepted_candidate_records_recomputed": len(originally_accepted),
        "distinct_mapped_raw_blocks_recomputed": len(block_results),
        "source_files_scanned": len(by_source),
        "guide_mapping_mismatches": mapping_mismatches,
        "records_with_raw_block_review": raw_block_review_records,
        "accepted_label_discrepancy_count_over_1e-8_pp_computational_check": label_mismatches,
        "maximum_accepted_label_absolute_difference_pp": max_abs_delta,
        "internal_train_validation_candidate_records_with_accepted_original_label": len(feature_rows_internal),
        "physical_cells_by_partition": {name: len(value) for name, value in cells_by_partition.items()},
        "step4_cutoff_observed_bracket_diagnostic_v": [cutoff_below, cutoff_above],
        "time_unit_status": "Unresolved archived numeric unit; minute-based agreement with Amphr is a consistency inference, not proof.",
        "capacity_unit_status": "Absolute Amphr-to-export mapping remains unverified; same-protocol native current-time ratios are conditional on common scale.",
        "uncertainty_status": "Instrument accuracy/repeatability and propagated capacity uncertainty are not available in the inspected artifacts.",
        "training_eligible_records_cleared_by_this_audit": 0,
        "training_authorized": False,
        "degradation_fitting_started": False,
        "external_used_for_model_selection": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_dir / "independent_block_audit.csv",
        [
            {name: value for name, value in row.items() if not name.startswith("_")}
            for row in block_results
        ],
    )
    _write_csv(output_dir / "independent_record_audit.csv", record_rows)
    (output_dir / "independent_audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "audit_version": "independent-raw-soh-audit-v1",
                "training_authorized": False,
                "training_started": False,
                "degradation_fitting_started": False,
                "external_used_for_model_selection": False,
                "inputs": {
                    "candidate_manifest": _jsonable(RECONSTRUCTION_DIR / "final_candidate_manifest.csv"),
                    "label_provenance": _jsonable(RECONSTRUCTION_DIR / "label_provenance.csv"),
                    "references": _jsonable(RECONSTRUCTION_DIR / "reconstructed_soh_references.csv"),
                    "feature_matrix": _jsonable(RECONSTRUCTION_DIR / "raw_curve_feature_matrix.csv"),
                    "m1_block_audit": _jsonable(M1_DIR / "measurement_block_audit.csv"),
                    "guide_manifest": _jsonable(GUIDE_MANIFEST),
                    "part3_bol_crosswalk": _jsonable(PART3_GUIDE_MANIFEST),
                    "split_manifest": _jsonable(SPLIT_MANIFEST),
                },
                "raw_files_source_size_and_mtime_checked": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Independently audit raw-derived BaFuse SOH references.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    print(json.dumps(run_audit(args.output_dir), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
