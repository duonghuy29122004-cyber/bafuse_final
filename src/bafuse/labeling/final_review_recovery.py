from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
INDEPENDENT_DIR = BENCHMARK_DIR / "independent_raw_soh_audit_v1_20261113"
MULTIMETHOD_DIR = BENCHMARK_DIR / "soh_multimethod_recovery_v1_20261113"
PART3_CROSSWALK_DIR = PROJECT_ROOT / "experiments" / "results" / "data_audits" / "part3_bol_crosswalk"
PART1_BOL_DIR = PROJECT_ROOT / "experiments" / "results" / "data_audits" / "part1_bol_signal_review"
OUTPUT_DIR = BENCHMARK_DIR / "soh_final_review_recovery_v1_20261114"
OFFICIAL_SOURCE_DOCUMENTS = (
    {
        "name": "Part 1 Guide_to_Datafiles.xlsx",
        "url": "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/dz029p477c",
        "sha256": "54f8fddb5d71e9c7179ac25d45b751b12f2b413573a7f190a8ffef95135a6aa7",
    },
    {
        "name": "Part 1 Readme.txt",
        "url": "https://ora.ox.ac.uk/objects/uuid:de62b5d2-6154-426d-bcbb-30253ddb7d1e/files/d7s75dc40b",
        "sha256": "59489534eaa5cddd2cef74b057855d2074bbef802d3aa12db6e082f9886dc59c",
    },
    {
        "name": "Part 2 Guide_to_Datafiles_2.xlsx",
        "url": "https://ora.ox.ac.uk/objects/uuid:be3d304e-51fd-4b37-a818-b6fa1ac2ba9d/files/dh989r335p",
        "sha256": "8031e1ddc78ff13935542882eccbc11a3708bda01cf76cda3987ada2971d04e7",
    },
    {
        "name": "Part 2 Readme.txt",
        "url": "https://ora.ox.ac.uk/objects/uuid:be3d304e-51fd-4b37-a818-b6fa1ac2ba9d/files/dsq87bt675",
        "sha256": "2d834e1afb16bfcc75dddae4910b3c8a8b250324e58511ac8eb641adb5411d4c",
    },
    {
        "name": "Part 3 Guide_to_Datafiles_3.xlsx",
        "url": "https://ora.ox.ac.uk/objects/uuid:78f66fa8-deb9-468a-86f3-63983a7391a9/files/ddv13zt35f",
        "sha256": "4b1dde65eda4f43138037748704ec87a2c9c456e64ddd83518fb0e7e63adf5b5",
    },
    {
        "name": "Part 3 Readme.txt",
        "url": "https://ora.ox.ac.uk/objects/uuid:78f66fa8-deb9-468a-86f3-63983a7391a9/files/dcj82k741w",
        "sha256": "c42db6dc02f854dd2c19eb272f5e61204e42c47e17dd1a034f7f775df852b43c",
    },
)

_BLOCK_RE = re.compile(
    r"#Step(?P<step>\d+):segment[^:]*:rows(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$",
    re.IGNORECASE,
)
_CELL_RE = re.compile(r"^G(?P<group>\d+)-Cell(?P<cell>\d+)$")
_FIELDS = ("step", "testtime", "steptime", "amphr", "amps", "volts")


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


def _number(value: str | float | int | None) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _parse_ref(reference: str) -> tuple[int, int, int]:
    match = _BLOCK_RE.search(reference.strip())
    if match is None:
        raise ValueError(f"Unparseable raw block reference: {reference!r}")
    return int(match.group("step")), int(match.group("start")), int(match.group("end"))


def _parse_row(raw: bytes) -> list[str]:
    return next(csv.reader([raw.decode("utf-8", errors="replace")]))


def _integral(times: list[float | None], currents: list[float | None]) -> tuple[float, int, int]:
    area = 0.0
    zero_intervals = 0
    negative_intervals = 0
    for left_t, right_t, left_i, right_i in zip(times, times[1:], currents, currents[1:]):
        if None in (left_t, right_t, left_i, right_i):
            raise ValueError("A required time/current value is missing inside a mapped block.")
        dt = float(right_t) - float(left_t)
        if dt == 0:
            zero_intervals += 1
        elif dt < 0:
            negative_intervals += 1
        else:
            area += 0.5 * (abs(float(left_i)) + abs(float(right_i))) * dt
    return area, zero_intervals, negative_intervals


def _scan_block(
    source_file: str,
    reference: str,
    expected_hash: str,
) -> dict[str, Any]:
    step, start_line, end_line = _parse_ref(reference)
    source_path = PROJECT_ROOT / Path(source_file.replace("/", "\\"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Mapped raw source is missing: {source_path}")

    source_digest = hashlib.sha256()
    block_digest = hashlib.sha256()
    header: list[str] | None = None
    columns: dict[str, int] = {}
    rows: list[dict[str, float | None]] = []
    raw_rows: list[bytes] = []
    next_step: int | None = None
    line_count = 0
    with source_path.open("rb") as stream:
        for line_count, raw in enumerate(stream, 1):
            source_digest.update(raw)
            if line_count == 1:
                header = next(csv.reader([raw.decode("utf-8-sig", errors="replace")]))
                columns = {name.strip().casefold(): i for i, name in enumerate(header)}
            if start_line <= line_count <= end_line + 1:
                row = _parse_row(raw)
                if line_count <= end_line:
                    block_digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
                    raw_rows.append(raw.rstrip(b"\r\n"))
                    if header is None:
                        raise ValueError(f"Missing header in {source_path}")
                    if len(row) != len(header):
                        raise ValueError(
                            f"Row width changed at {source_path}:{line_count}: "
                            f"{len(row)} != {len(header)}"
                        )
                    values: dict[str, float | None] = {}
                    for field in _FIELDS:
                        index = columns.get(field)
                        values[field] = (
                            None
                            if index is None or index >= len(row)
                            else _number(row[index])
                        )
                    rows.append(values)
                else:
                    step_index = columns.get("step")
                    next_step = (
                        None
                        if step_index is None or step_index >= len(row)
                        else int(float(row[step_index]))
                    )

    if header is None:
        raise ValueError(f"Empty mapped raw source: {source_path}")
    if len(rows) != end_line - start_line + 1:
        raise ValueError(
            f"Raw block row-count mismatch at {source_path}:{start_line}-{end_line}: "
            f"{len(rows)}"
        )
    observed_hash = block_digest.hexdigest()
    if observed_hash != expected_hash:
        raise ValueError(
            f"Raw block hash mismatch at {source_path}:{start_line}-{end_line}: "
            f"{observed_hash} != {expected_hash}"
        )
    observed_steps = {int(row["step"]) for row in rows if row["step"] is not None}
    if observed_steps != {step} or any(row["step"] is None for row in rows):
        raise ValueError(f"Unexpected Step values in {source_path}:{start_line}-{end_line}")

    testtime = [row["testtime"] for row in rows]
    steptime = [row["steptime"] for row in rows]
    current = [row["amps"] for row in rows]
    test_integral, test_zero, test_negative = _integral(testtime, current)
    step_integral, step_zero, step_negative = _integral(steptime, current)
    amphr = [row["amphr"] for row in rows]
    voltage = [row["volts"] for row in rows]
    if any(value is None for value in (*testtime, *steptime, *current, *amphr, *voltage)):
        raise ValueError(f"Incomplete core measurement columns in {source_path}:{reference}")
    currents = [float(value) for value in current if value is not None]
    capacities = [float(value) for value in amphr if value is not None]
    voltages = [float(value) for value in voltage if value is not None]
    return {
        "source_file": source_file.replace("\\", "/"),
        "source_file_sha256": source_digest.hexdigest(),
        "source_file_size_bytes": source_path.stat().st_size,
        "step": step,
        "start_line": start_line,
        "end_line": end_line,
        "row_count": len(rows),
        "block_sha256": observed_hash,
        "step_ids_exact": observed_steps == {step},
        "next_row_step": next_step,
        "next_row_transitions_step": next_step is not None and next_step != step,
        "testtime_integrated_abs_current_native": test_integral,
        "steptime_integrated_abs_current_native": step_integral,
        "testtime_zero_intervals": test_zero,
        "testtime_negative_intervals": test_negative,
        "steptime_zero_intervals": step_zero,
        "steptime_negative_intervals": step_negative,
        "amphr_span_native": max(capacities) - min(capacities),
        "current_min": min(currents),
        "current_max": max(currents),
        "current_mean": statistics.fmean(currents),
        "voltage_first": voltages[0],
        "voltage_last": voltages[-1],
        "voltage_min": min(voltages),
        "voltage_max": max(voltages),
        "rows": rows,
        "raw_rows": raw_rows,
    }


def _scan_source_segments(source_file: str, selected_steps: set[int]) -> dict[str, Any]:
    source_path = PROJECT_ROOT / Path(source_file.replace("/", "\\"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Mapped raw source is missing: {source_path}")
    digest = hashlib.sha256()
    segments: list[dict[str, Any]] = []
    header: list[str] | None = None
    columns: dict[str, int] = {}
    active: dict[str, Any] | None = None

    def finish(segment: dict[str, Any] | None) -> None:
        if segment is None or segment["step"] not in selected_steps:
            return
        fields = segment["values"]
        amphr = [value for value in fields["amphr"] if value is not None]
        amps = [value for value in fields["amps"] if value is not None]
        volts = [value for value in fields["volts"] if value is not None]
        if fields["testtime"] and amps and len(fields["testtime"]) == len(fields["amps"]):
            integral, zero, negative = _integral(fields["testtime"], fields["amps"])
        else:
            integral, zero, negative = None, None, None
        signs = Counter("positive" if v > 0 else "negative" if v < 0 else "zero" for v in amps)
        time_values = [value for value in fields["testtime"] if value is not None]
        step_time_values = [value for value in fields["steptime"] if value is not None]
        segments.append(
            {
                "source_file": source_file.replace("\\", "/"),
                "step": segment["step"],
                "start_line": segment["start_line"],
                "end_line": segment["end_line"],
                "row_count": segment["row_count"],
                "block_sha256": segment["digest"].hexdigest(),
                "testtime_integrated_abs_current_native": integral,
                "testtime_zero_intervals": zero,
                "testtime_negative_intervals": negative,
                "testtime_min": min(time_values) if time_values else None,
                "testtime_max": max(time_values) if time_values else None,
                "steptime_zero_intervals": sum(
                    b == a for a, b in zip(step_time_values, step_time_values[1:])
                ),
                "steptime_negative_intervals": sum(
                    b < a for a, b in zip(step_time_values, step_time_values[1:])
                ),
                "amphr_span_native": max(amphr) - min(amphr) if amphr else None,
                "amps_min": min(amps) if amps else None,
                "amps_max": max(amps) if amps else None,
                "current_sign_counts": dict(signs),
                "voltage_first": volts[0] if volts else None,
                "voltage_last": volts[-1] if volts else None,
                "voltage_min": min(volts) if volts else None,
                "voltage_max": max(volts) if volts else None,
            }
        )

    with source_path.open("rb") as stream:
        for line_count, raw in enumerate(stream, 1):
            digest.update(raw)
            row = _parse_row(raw)
            if line_count == 1:
                header = row
                columns = {name.strip().casefold(): i for i, name in enumerate(header)}
                continue
            if header is None:
                raise ValueError(f"Missing header in {source_path}")
            step_index = columns.get("step")
            if step_index is None or step_index >= len(row):
                raise ValueError(f"Missing Step field at {source_path}:{line_count}")
            step = int(float(row[step_index]))
            if active is None or active["step"] != step:
                finish(active)
                active = {
                    "step": step,
                    "start_line": line_count,
                    "end_line": line_count,
                    "row_count": 0,
                    "digest": hashlib.sha256(),
                    "values": {
                        field: [] for field in ("testtime", "steptime", "amphr", "amps", "volts")
                    },
                }
            active["end_line"] = line_count
            active["row_count"] += 1
            active["digest"].update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
            for field in active["values"]:
                index = columns.get(field)
                active["values"][field].append(
                    None if index is None or index >= len(row) else _number(row[index])
                )
    finish(active)
    return {
        "source_file": source_file.replace("\\", "/"),
        "source_file_sha256": digest.hexdigest(),
        "source_file_size_bytes": source_path.stat().st_size,
        "segments": segments,
    }


def _ratio_percent(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Capacity numerator and denominator must be positive.")
    return 100.0 * numerator / denominator


def _validate_step4_pair(
    numerator: dict[str, Any],
    denominator: dict[str, Any],
) -> bool:
    for role, block in (("RPT", numerator), ("BoL", denominator)):
        if block["testtime_zero_intervals"] or block["testtime_negative_intervals"]:
            raise ValueError(f"{role} TestTime is not strictly increasing.")
        if block["steptime_zero_intervals"] or block["steptime_negative_intervals"]:
            raise ValueError(f"{role} StepTime is not strictly increasing.")
        if block["current_max"] >= 0:
            raise ValueError(f"{role} Step 4 block does not have discharge-only current.")
        if block["voltage_last"] != block["voltage_min"]:
            raise ValueError(f"{role} Step 4 boundary does not end at its lowest sampled voltage.")
    return True


def _proposal_status(*verified_criteria: bool) -> str:
    return "PROMOTION_SUPPORTED" if verified_criteria and all(verified_criteria) else "REVIEW_RETAINED"


def _guide_entry(
    rows: list[dict[str, str]],
    *,
    part: int,
    cell: str,
    measurement_id: str,
    source_file: str | None = None,
    procedure: str | None = None,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if row.get("dataset_part") == str(part)
        and row.get("physical_cell_id") == cell
        and row.get("measurement_id") == measurement_id
        and (
            source_file is None
            or row.get("source_file", "").replace("\\", "/").casefold()
            == source_file.replace("\\", "/").casefold()
        )
        and (procedure is None or row.get("procedure") == procedure)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Guide mapping must be unique for Part {part} {cell} {measurement_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _pair_summary(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    differences = [
        float(row["soh_charge_percent"]) - float(row["soh_discharge_percent"])
        for row in rows
    ]
    absolute = [abs(value) for value in differences]
    return {
        "n": len(differences),
        "mean_absolute_difference_pp": statistics.fmean(absolute),
        "median_absolute_difference_pp": statistics.median(absolute),
        "max_absolute_difference_pp": max(absolute),
        "bias_charge_minus_discharge_pp": statistics.fmean(differences),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to create an empty evidence table: {path.name}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _verify_split(candidates: list[dict[str, str]], splits: list[dict[str, str]]) -> None:
    if any(row["dataset_part"] == "3" and row["partition"] != "external_test" for row in candidates):
        raise ValueError("A Part 3 observation is not isolated to External Test.")
    split_by_cell = {row["physical_cell_id"]: row["partition"] for row in splits}
    seen: dict[str, set[str]] = {}
    for row in candidates:
        cell = row["physical_cell_id"]
        expected = split_by_cell.get(cell)
        if expected != row["partition"]:
            raise ValueError(f"Partition mismatch for {cell}: {row['partition']} != {expected}")
        seen.setdefault(cell, set()).add(row["partition"])
    if any(len(partitions) != 1 for partitions in seen.values()):
        raise ValueError("A physical cell appears in more than one partition.")


def _load_raw_audit() -> tuple[
    list[dict[str, str]],
    dict[tuple[str, int, int, int, int], dict[str, str]],
]:
    block_rows = _read_csv(INDEPENDENT_DIR / "independent_block_audit.csv")
    record_rows = _read_csv(INDEPENDENT_DIR / "independent_record_audit.csv")
    by_key: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for row in block_rows:
        key = (
            row["source_file"].replace("\\", "/").casefold(),
            int(row["source_part"]),
            int(row["step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        if key in by_key:
            raise ValueError(f"Duplicate independent block key: {key}")
        by_key[key] = row
    return record_rows, by_key


def _block_audit_row(
    block_rows: dict[tuple[str, int, int, int, int], dict[str, str]],
    source_file: str,
    part: int,
    reference: str,
) -> dict[str, str]:
    step, start, end = _parse_ref(reference)
    key = (source_file.replace("\\", "/").casefold(), part, step, start, end)
    if key not in block_rows:
        raise ValueError(f"Independent raw block audit lacks {source_file} {reference}")
    return block_rows[key]


def _review_evidence(
    *,
    reviews: list[dict[str, str]],
    candidates: list[dict[str, str]],
    guide_rows: list[dict[str, str]],
    part3_step4_rows: list[dict[str, str]],
    block_rows: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], list[dict[str, Any]]]:
    candidates_by_key = {
        (row["physical_cell_id"], row["rpt_index"]): row for row in candidates
    }
    part3_step4_by_cell = {row["physical_cell_id"]: row for row in part3_step4_rows}
    evidence: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    raw_hashes: dict[str, str] = {}
    g2_blocks: list[dict[str, Any]] = []
    p3_verified = 0

    for review in reviews:
        cell, rpt_text = review["physical_cell_id"], review["rpt_index"]
        candidate = candidates_by_key[(cell, rpt_text)]
        part = int(candidate["dataset_part"])
        step4_source = candidate["source_file"]
        step4_ref = candidate["step4_segment_ref"]
        step4_audit = _block_audit_row(block_rows, step4_source, part, step4_ref)
        rpt_block = _scan_block(step4_source, step4_ref, step4_audit["block_sha256_independent"])
        raw_hashes[rpt_block["source_file"]] = rpt_block["source_file_sha256"]

        bol_source = candidate["bol_source_file"]
        bol_ref = candidate["bol_step4_segment_ref"]
        bol_audit = _block_audit_row(block_rows, bol_source, int(candidate["bol_source_part"]), bol_ref)
        bol_block = _scan_block(bol_source, bol_ref, bol_audit["block_sha256_independent"])
        raw_hashes[bol_block["source_file"]] = bol_block["source_file_sha256"]

        q_rpt = float(rpt_block["testtime_integrated_abs_current_native"])
        q_bol = float(bol_block["testtime_integrated_abs_current_native"])
        soh = _ratio_percent(q_rpt, q_bol)
        counter_soh = _ratio_percent(
            float(rpt_block["amphr_span_native"]),
            float(bol_block["amphr_span_native"]),
        )
        stored_soh = _number(review.get("soh_discharge_integrated_percent"))
        stored_q_rpt = _number(review.get("q_discharge_rpt_native_current_time"))
        stored_q_bol = _number(review.get("q_discharge_bol_native_current_time"))
        for observed, stored, name in (
            (soh, stored_soh, "SOH"),
            (q_rpt, stored_q_rpt, "RPT integrated capacity"),
            (q_bol, stored_q_bol, "BoL integrated capacity"),
        ):
            if stored is None or not math.isclose(observed, stored, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(
                    f"Independent {name} recomputation differs from prior audit for "
                    f"{cell} RPT{rpt_text}."
                )
        check_source = rpt_block["source_file"].replace("\\", "/")
        check_file_size = int(rpt_block["source_file_size_bytes"])
        mapping = _guide_entry(
            guide_rows,
            part=part,
            cell=cell,
            measurement_id=f"RPT{rpt_text}",
        )
        bol_source_relative = bol_source.replace("\\", "/").split(
            f"data/raw/oxford/part{part}/", 1
        )[-1]
        bol_guide = _guide_entry(
            guide_rows,
            part=part,
            cell=cell,
            measurement_id="BoL",
            source_file=bol_source_relative,
        )
        guide_rpt_path = f"data/raw/oxford/part{part}/{mapping['source_file']}"
        exact_rpt_membership = (
            mapping.get("source_mapping_status") == "unique_source_file"
            and int(mapping.get("source_file_count", "0")) == 1
            and guide_rpt_path.casefold() == check_source.casefold()
            and int(mapping.get("source_file_size_bytes", "0")) == check_file_size
        )
        if not exact_rpt_membership:
            raise ValueError(f"Guide RPT membership does not match raw file for {cell} RPT{rpt_text}")
        guide_bol_path = f"data/raw/oxford/part{part}/{bol_guide['source_file']}"
        exact_bol_membership = (
            bol_guide.get("source_mapping_status") == "unique_source_file"
            and int(bol_guide.get("source_file_count", "0")) == 1
            and guide_bol_path.casefold() == bol_block["source_file"].casefold()
            and int(bol_guide.get("source_file_size_bytes", "0"))
            == int(bol_block["source_file_size_bytes"])
        )
        if not exact_bol_membership:
            raise ValueError(f"Guide BoL membership does not match raw file for {cell}")
        same_cell_bol_entries = [
            row
            for row in guide_rows
            if row.get("dataset_part") == str(part)
            and row.get("physical_cell_id") == cell
            and row.get("measurement_id") == "BoL"
        ]
        if not same_cell_bol_entries or bol_guide["procedure_file_id"] != min(
            same_cell_bol_entries,
            key=lambda row: int(row["procedure_file_id"]),
        )["procedure_file_id"]:
            raise ValueError(f"Selected BoL is not the earliest Guide-mapped procedure for {cell}")

        if part == 3:
            group_match = _CELL_RE.fullmatch(cell)
            if group_match is None:
                raise ValueError(f"Unparseable physical cell identity: {cell}")
            expected_bol_source = candidate["bol_source_file"].replace("\\", "/")
            p3_bol_audit = part3_step4_by_cell.get(cell)
            if p3_bol_audit is None:
                raise ValueError(f"Part 3 Step 4 baseline audit missing for {cell}")
            bol_member = (
                bol_guide.get("source_mapping_rule") == "part3_unsuffixed_bol_to_zero_suffix"
                and bol_block["source_file"].casefold() == expected_bol_source.casefold()
                and Path(expected_bol_source).name.casefold().endswith(
                    f"-cell{group_match.group('cell')}.csv"
                )
                and p3_bol_audit["source_file"].replace("\\", "/").casefold()
                    == expected_bol_source.casefold().removeprefix("data/raw/oxford/part3/")
                and p3_bol_audit["scan_status"] == "one_contiguous_step4_candidate"
            )
            if not bol_member:
                raise ValueError(f"Part 3 corrected BoL crosswalk failed validation for {cell}")
            step4_protocol_compatible = _validate_step4_pair(rpt_block, bol_block)
            if rpt_block["next_row_step"] == 4 or bol_block["next_row_step"] == 4:
                raise ValueError(f"Step 4 boundary not closed for {cell} RPT{rpt_text}")
            p3_verified += 1
            recommendation = _proposal_status(
                exact_rpt_membership,
                exact_bol_membership,
                bol_member,
                rpt_block["block_sha256"] == step4_audit["block_sha256_independent"],
                bol_block["block_sha256"] == bol_audit["block_sha256_independent"],
                step4_protocol_compatible,
            )
            if recommendation != "PROMOTION_SUPPORTED":
                raise ValueError(f"Part 3 evidence gates did not pass for {cell} RPT{rpt_text}")
            rationale = (
                "Corrected Guide maps the same-cell unsuffixed BoL procedure uniquely to its .0 "
                "source; the Guide-mapped RPT and BoL each contain hash-matched, complete Step 4 "
                "discharge blocks under the documented Part 3 Maccor 4000 CC C/2-to-2.5-V protocol."
            )
        else:
            if cell != "G2-Cell3" or rpt_text != "2":
                raise ValueError(f"Unexpected non-Part-3 REVIEW record: {cell} RPT{rpt_text}")
            if rpt_block["testtime_zero_intervals"] != 26 or rpt_block["steptime_zero_intervals"] != 26:
                raise ValueError("G2-Cell3 RPT2 repeated-time count changed from the audited 26.")
            if rpt_block["testtime_negative_intervals"] or rpt_block["steptime_negative_intervals"]:
                raise ValueError("G2-Cell3 RPT2 contains negative time intervals.")
            g2_blocks = [rpt_block, bol_block]
            exact_duplicate_groups: dict[str, int] = {}
            for axis in ("testtime", "steptime"):
                exact_duplicate_groups[axis] = sum(
                    rpt_block["rows"][i - 1][axis] == rpt_block["rows"][i][axis]
                    for i in range(1, len(rpt_block["rows"]))
                )
            if exact_duplicate_groups != {"testtime": 26, "steptime": 26}:
                raise ValueError(f"G2 exact duplicate timestamp count changed: {exact_duplicate_groups}")
            duplicate_positions = {
                i
                for axis in ("testtime", "steptime")
                for i in range(1, len(rpt_block["rows"]))
                if rpt_block["rows"][i - 1][axis] == rpt_block["rows"][i][axis]
            }
            if not all(
                rpt_block["raw_rows"][i - 1] == rpt_block["raw_rows"][i]
                for i in duplicate_positions
            ):
                raise ValueError("G2 repeated-time rows are not exact duplicates across exported fields.")
            recommendation = _proposal_status(
                exact_rpt_membership,
                exact_bol_membership,
                rpt_block["block_sha256"] == step4_audit["block_sha256_independent"],
                bol_block["block_sha256"] == bol_audit["block_sha256_independent"],
                rpt_block["testtime_zero_intervals"] == 26,
                rpt_block["steptime_zero_intervals"] == 26,
            )
            rationale = (
                "The 26 repeated intervals on each elapsed-time axis are exact duplicate rows; "
                "retaining or collapsing the identical rows leaves the Step 4 integral unchanged. "
                "The origin of duplicate emission is still unknown, but it does not alter this "
                "capacity calculation."
            )

        evidence_row = {
            "dataset_part": part,
            "physical_cell_id": cell,
            "rpt_index": int(rpt_text),
            "partition": candidate["partition"],
            "original_status": candidate["label_disposition"],
            "original_review_reason": review.get("original_review_reason", ""),
            "rpt_source_file": rpt_block["source_file"],
            "rpt_step4_reference": step4_ref,
            "rpt_step4_block_sha256_expected": step4_audit["block_sha256_independent"],
            "rpt_step4_block_sha256_recomputed": rpt_block["block_sha256"],
            "rpt_source_file_sha256": rpt_block["source_file_sha256"],
            "rpt_step4_row_count": rpt_block["row_count"],
            "rpt_step4_integrated_native_current_time": q_rpt,
            "rpt_step4_native_amphr_span": rpt_block["amphr_span_native"],
            "rpt_step4_current_min": rpt_block["current_min"],
            "rpt_step4_current_max": rpt_block["current_max"],
            "rpt_step4_current_mean": rpt_block["current_mean"],
            "rpt_step4_voltage_first": rpt_block["voltage_first"],
            "rpt_step4_voltage_last": rpt_block["voltage_last"],
            "rpt_step4_voltage_min": rpt_block["voltage_min"],
            "rpt_step4_voltage_max": rpt_block["voltage_max"],
            "rpt_step4_terminal_offset_from_2_5_v": rpt_block["voltage_last"] - 2.5,
            "rpt_step4_testtime_zero_intervals": rpt_block["testtime_zero_intervals"],
            "rpt_step4_steptime_zero_intervals": rpt_block["steptime_zero_intervals"],
            "bol_source_file": bol_block["source_file"],
            "bol_step4_reference": bol_ref,
            "bol_step4_block_sha256_expected": bol_audit["block_sha256_independent"],
            "bol_step4_block_sha256_recomputed": bol_block["block_sha256"],
            "bol_source_file_sha256": bol_block["source_file_sha256"],
            "bol_step4_row_count": bol_block["row_count"],
            "bol_step4_integrated_native_current_time": q_bol,
            "bol_step4_native_amphr_span": bol_block["amphr_span_native"],
            "bol_step4_current_min": bol_block["current_min"],
            "bol_step4_current_max": bol_block["current_max"],
            "bol_step4_current_mean": bol_block["current_mean"],
            "bol_step4_voltage_first": bol_block["voltage_first"],
            "bol_step4_voltage_last": bol_block["voltage_last"],
            "bol_step4_voltage_min": bol_block["voltage_min"],
            "bol_step4_voltage_max": bol_block["voltage_max"],
            "bol_step4_terminal_offset_from_2_5_v": bol_block["voltage_last"] - 2.5,
            "step4_mean_absolute_current_difference": abs(
                abs(rpt_block["current_mean"]) - abs(bol_block["current_mean"])
            ),
            "step4_voltage_window_overlap": max(
                0.0,
                min(rpt_block["voltage_max"], bol_block["voltage_max"])
                - max(rpt_block["voltage_min"], bol_block["voltage_min"]),
            ),
            "recomputed_soh_discharge_percent": soh,
            "native_counter_soh_percent": counter_soh,
            "native_counter_minus_integrated_pp": counter_soh - soh,
            "stored_m1_soh_percent": review.get("soh_discharge_integrated_percent", ""),
            "stored_m1_difference_pp": (
                soh - float(review["soh_discharge_integrated_percent"])
                if review.get("soh_discharge_integrated_percent")
                else None
            ),
            "guide_rpt_procedure": mapping["procedure"],
            "guide_rpt_stage": mapping["guide_stage"],
            "guide_rpt_source_count": mapping["source_file_count"],
            "guide_bol_procedure": bol_guide["procedure"],
            "guide_bol_source_count": bol_guide["source_file_count"],
            "guide_bol_mapping_rule": (
                bol_guide.get("source_mapping_rule", "") if part == 3 else "same_cell_Guide_TPG2_BoL"
            ),
            "protocol_evidence": (
                "Oxford Part 3 README: Maccor 4000; Step 4 CC discharge C/2 to 2.5 V."
                if part == 3
                else "Oxford Parts 1-2 README: Maccor 4200; Step 4 CC-CV discharge C/2 to 2.5 V."
            ),
            "protocol_metadata_limit": (
                "Archived tester settings, exact time units, and Part 3 chamber temperature unavailable."
            ),
            "recommended_status": recommendation,
            "recommendation_rationale": rationale,
            "status_promoted": False,
            "training_authorized": False,
        }
        evidence.append(evidence_row)
        proposals.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": int(rpt_text),
                "partition": candidate["partition"],
                "original_status": candidate["label_disposition"],
                "original_review_reason": candidate.get("label_reasons", ""),
                "proposed_status": recommendation,
                "status_change": f"{candidate['label_disposition']} -> ACCEPTED" if recommendation == "PROMOTION_SUPPORTED" else candidate["label_disposition"],
                "decision": "PROPOSED_ONLY_USER_APPROVAL_REQUIRED",
                "decision_rationale": rationale,
                "charge_proxy_used_as_discharge_label": False,
                "training_authorized": False,
            }
        )
    if p3_verified != 24 or len(g2_blocks) != 2 or len(evidence) != 25:
        raise ValueError(f"Expected 24 verified Part 3 rows and one G2 row; got {p3_verified}/{len(evidence)}")
    return evidence, proposals, raw_hashes, g2_blocks


def _unpaired_audit(
    *,
    comparison: list[dict[str, str]],
    guide_rows: list[dict[str, str]],
    bol_curve_rows: list[dict[str, str]],
    block_rows: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    accepted_unpaired = [
        row
        for row in comparison
        if row["original_disposition"] == "ACCEPTED"
        and row["physical_cell_id"] in {"G4-Cell18", "G4-Cell19"}
        and (row["m1_status"] != "VALID" or row["m2_status"] != "VALID")
    ]
    if len(accepted_unpaired) != 14:
        raise ValueError(f"Expected 14 G4 unpaired ACCEPTED rows; found {len(accepted_unpaired)}")

    sources: dict[str, dict[str, Any]] = {}
    for cell in ("G4-Cell18", "G4-Cell19"):
        guide_bol = _guide_entry(
            guide_rows, part=1, cell=cell, measurement_id="BoL", procedure="TPG4"
        )
        source_candidates = guide_bol["source_file"].split(";")
        if len(source_candidates) != 2:
            raise ValueError(f"Expected two Guide source candidates for {cell} TPG4 BoL.")
        for relative in source_candidates:
            source = f"data/raw/oxford/part1/{relative}"
            sources[source] = _scan_source_segments(source, {4, 10})

    raw_hashes = {source: row["source_file_sha256"] for source, row in sources.items()}
    result_rows: list[dict[str, Any]] = []
    newly_pairable: list[dict[str, Any]] = []
    cell_step10: dict[str, list[dict[str, Any]]] = {}
    for cell in ("G4-Cell18", "G4-Cell19"):
        guide_bol = _guide_entry(
            guide_rows, part=1, cell=cell, measurement_id="BoL", procedure="TPG4"
        )
        candidate_files = [
            f"data/raw/oxford/part1/{item}" for item in guide_bol["source_file"].split(";")
        ]
        step10_options = []
        for source in candidate_files:
            for segment in sources[source]["segments"]:
                if segment["step"] != 10:
                    continue
                signs = segment["current_sign_counts"]
                valid_charge = (
                    signs.get("positive", 0) > 0
                    and signs.get("negative", 0) == 0
                    and signs.get("zero", 0) == 0
                    and segment["testtime_zero_intervals"] == 0
                    and segment["testtime_negative_intervals"] == 0
                    and segment["voltage_max"] is not None
                    and segment["voltage_max"] >= 4.2
                    and segment["amphr_span_native"] is not None
                    and segment["amphr_span_native"] > 0
                )
                if valid_charge:
                    step10_options.append(
                        {
                            **segment,
                            "source_file_sha256": sources[source]["source_file_sha256"],
                        }
                    )
        cell_step10[cell] = step10_options

    for row in accepted_unpaired:
        cell, rpt_text = row["physical_cell_id"], row["rpt_index"]
        source_a, source_b = [
            f"data/raw/oxford/part1/{item}"
            for item in _guide_entry(
                guide_rows, part=1, cell=cell, measurement_id="BoL", procedure="TPG4"
            )[
                "source_file"
            ].split(";")
        ]
        step4_options = []
        for source in (source_a, source_b):
            for segment in sources[source]["segments"]:
                if (
                    segment["step"] == 4
                    and segment["current_sign_counts"].get("negative", 0) > 0
                    and segment["current_sign_counts"].get("positive", 0) == 0
                    and segment["current_sign_counts"].get("zero", 0) == 0
                    and segment["testtime_zero_intervals"] == 0
                    and segment["testtime_negative_intervals"] == 0
                    and segment["amphr_span_native"] is not None
                    and segment["amphr_span_native"] > 0
                ):
                    step4_options.append(segment)
        expected_start = _parse_ref(row["primary_bol_step4_segment_ref"])[1]
        matching_step4 = [
            segment
            for segment in step4_options
            if segment["source_file"] == row["primary_bol_source_file"]
            and segment["start_line"] == expected_start
        ]
        if len(matching_step4) != 1:
            raise ValueError(f"Could not uniquely verify the selected M1 BoL for {cell}.")
        if matching_step4[0]["block_sha256"] != row["primary_bol_step4_block_sha256"]:
            raise ValueError(f"G4 selected Step 4 BoL raw hash mismatch for {cell}.")
        selected_source = row["primary_bol_source_file"]
        selected_source_segments = [
            segment for segment in step4_options if segment["source_file"] == selected_source
        ]
        if not selected_source_segments or min(
            segment["start_line"] for segment in selected_source_segments
        ) != matching_step4[0]["start_line"]:
            raise ValueError(f"Selected G4 Step 4 BoL is not the earliest complete segment for {cell}.")
        matched_later_bol = [
            item
            for item in bol_curve_rows
            if item["physical_cell_id"] == cell
            and item["first_procedure"] == "TPG4"
            and item["second_procedure"] == "TPG4.2"
            and item["first_segment_index"] == "2"
        ]
        if len(matched_later_bol) != 1 or matched_later_bol[0]["core_signals_match_except_time"] != "True":
            raise ValueError(f"G4 later-segment cross-check is missing for {cell}.")
        other_source_step4 = [
            segment for segment in step4_options if segment["source_file"] != selected_source
        ]
        charge_options = cell_step10[cell]
        rpt_m1 = float(row["soh_discharge_integrated_percent"])
        rpt_q10 = float(row["q_charge_rpt_native_current_time"])
        candidate_m2 = [
            {
                "source_file": segment["source_file"],
                "start_line": segment["start_line"],
                "end_line": segment["end_line"],
                "block_sha256": segment["block_sha256"],
                "bol_q_native_current_time": segment["testtime_integrated_abs_current_native"],
                "bol_amphr_span_native": segment["amphr_span_native"],
                "soh_charge_percent": _ratio_percent(
                    rpt_q10, segment["testtime_integrated_abs_current_native"]
                ),
            }
            for segment in charge_options
        ]
        if cell == "G4-Cell19" and len(candidate_m2) != 1:
            raise ValueError("Expected one valid G4-Cell19 Step 10 BoL segment.")
        if cell == "G4-Cell18" and len(candidate_m2) != 2:
            raise ValueError("Expected two valid G4-Cell18 Step 10 BoL candidates.")
        if cell == "G4-Cell19":
            selected = candidate_m2[0]
            newly_pairable.append(
                {
                    "physical_cell_id": cell,
                    "rpt_index": int(rpt_text),
                    "soh_discharge_percent": rpt_m1,
                    "soh_charge_percent": selected["soh_charge_percent"],
                }
            )
            pair_recommendation = "INCLUDE_IN_M1_M2_AGREEMENT"
            charge_status = "UNIQUE_VALID_STEP10_BLOCK"
            m2_candidate_range = f"{selected['soh_charge_percent']:.12f}"
        else:
            pair_recommendation = "EXCLUDE_M1_M2_AGREEMENT"
            charge_status = "TWO_VALID_GUID_STEP10_CANDIDATES"
            values = [item["soh_charge_percent"] for item in candidate_m2]
            m2_candidate_range = f"{min(values):.12f}..{max(values):.12f}"
        result_rows.append(
            {
                "physical_cell_id": cell,
                "rpt_index": int(rpt_text),
                "partition": row["partition"],
                "original_status": "ACCEPTED",
                "original_review_reason": "",
                "prior_m1_invalid_reasons": row["m1_invalid_reasons"],
                "prior_m2_invalid_reasons": row["m2_invalid_reasons"],
                "prior_m1_status": row["m1_status"],
                "prior_m2_status": row["m2_status"],
                "rpt_source_file": row["rpt_step4_segment_ref"].split("#", 1)[0],
                "rpt_step4_reference": row["rpt_step4_segment_ref"],
                "rpt_step4_block_sha256": row["rpt_step4_block_sha256"],
                "rpt_step4_capacity_native_current_time": row[
                    "q_discharge_rpt_native_current_time"
                ],
                "rpt_step4_native_amphr_span": row["native_discharge_rpt_span"],
                "primary_bol_source_file": row["primary_bol_source_file"],
                "primary_bol_step4_reference": row["primary_bol_step4_segment_ref"],
                "primary_bol_step4_block_sha256": row["primary_bol_step4_block_sha256"],
                "primary_bol_step4_capacity_native_current_time": row[
                    "q_discharge_bol_native_current_time"
                ],
                "primary_bol_step4_native_amphr_span": row["native_discharge_bol_span"],
                "rpt_step10_reference": row["rpt_step10_segment_ref"],
                "rpt_step10_block_sha256": row["rpt_step10_block_sha256"],
                "rpt_step10_charge_capacity_native_current_time": row[
                    "q_charge_rpt_native_current_time"
                ],
                "guide_bol_source_candidates": f"{source_a};{source_b}",
                "step4_eligible_guide_source_count": len(
                    {segment["source_file"] for segment in step4_options}
                ),
                "step4_selected_candidate_count": len(matching_step4),
                "step4_earliest_complete_reference_verified": len(matching_step4) == 1,
                "step4_alternative_candidate_files_have_no_valid_step4": not other_source_step4,
                "step10_valid_candidate_count": len(candidate_m2),
                "step10_candidate_source_and_segment_hashes": json.dumps(
                    [
                        {
                            "source_file": item["source_file"],
                            "start_line": item["start_line"],
                            "end_line": item["end_line"],
                            "block_sha256": item["block_sha256"],
                        }
                        for item in candidate_m2
                    ],
                    sort_keys=True,
                ),
                "step10_m2_candidate_soh_percent_or_range": m2_candidate_range,
                "m1_soh_percent": rpt_m1,
                "m1_status_recommendation": "ACCEPTED_RETAINED",
                "m1_reason": (
                    "Guide-level source count is two, but only the selected .1 source has a "
                    "complete Step 4 discharge; the other listed source has no Step 4 block. "
                    "The earliest complete Step 4 segment is independently identified and "
                    "the later TPG4.2 segment is retained as a separate BoL sensitivity."
                ),
                "m2_resolution": charge_status,
                "agreement_pair_recommendation": pair_recommendation,
                "m2_charge_native_counter_soh_percent": row["soh_native_charge_percent"],
                "m1_discharge_native_counter_soh_percent": row["soh_native_discharge_percent"],
                "m1_native_counter_minus_integrated_pp": (
                    float(row["soh_native_discharge_percent"])
                    - float(row["soh_discharge_integrated_percent"])
                ),
                "protocol_compatibility": (
                    "Maccor 4200 Part 1; Step 4 CC-CV discharge C/2 to 2.5 V; "
                    "same-cell, earliest complete Step 4 BoL selected."
                ),
                "remaining_uncertainty": (
                    "Cell18 has two valid Step 10 BoL sources; Cell19 Step 10 is uniquely mapped. "
                    "Archived tester settings and absolute counter units remain unverified."
                ),
                "confidence_rationale": (
                    "M1 source identity is confirmed by the Guide and direct Step 4 scan; selected "
                    "raw block hash matches the existing independent audit. M2 ambiguity is handled "
                    "separately and does not invalidate M1."
                ),
                "status_promoted": False,
                "training_authorized": False,
            }
        )

    original_pairs = [
        row
        for row in comparison
        if row["original_disposition"] == "ACCEPTED"
        and row["m1_status"] == "VALID"
        and row["m2_status"] == "VALID"
    ]
    paired_values = [
        {
            "soh_discharge_percent": float(row["soh_discharge_integrated_percent"]),
            "soh_charge_percent": float(row["soh_charge_integrated_percent"]),
        }
        for row in original_pairs
    ] + newly_pairable
    return result_rows, newly_pairable, raw_hashes, {
        "previous_valid_pair_count": len(original_pairs),
        "step_specific_valid_pair_additions": len(newly_pairable),
        "revised_valid_pair_count": len(paired_values),
        "previous_pair_summary": _pair_summary(paired_values[: len(original_pairs)]),
        "revised_pair_summary": _pair_summary(paired_values),
    }


def _g2_investigation(
    g2_block: dict[str, Any], g2_bol_block: dict[str, Any]
) -> dict[str, Any]:
    rows = g2_block["rows"]
    duplicate_by_axis: dict[str, int] = {}
    exact_repeats: dict[str, int] = {}
    for axis in ("testtime", "steptime"):
        index = axis
        duplicate_positions = [
            i
            for i in range(1, len(rows))
            if rows[i - 1][index] == rows[i][index]
        ]
        duplicate_by_axis[axis] = len(duplicate_positions)
        exact_repeats[axis] = sum(rows[i - 1] == rows[i] for i in duplicate_positions)
    if duplicate_by_axis != {"testtime": 26, "steptime": 26}:
        raise ValueError(f"Unexpected G2 duplicate interval counts: {duplicate_by_axis}")
    if exact_repeats != duplicate_by_axis:
        raise ValueError("G2 repeated timestamps are not all full-row duplicates.")
    reduced = [row for i, row in enumerate(rows) if i == 0 or row != rows[i - 1]]
    for axis in ("testtime", "steptime"):
        original_q, zero, negative = _integral(
            [row[axis] for row in rows], [row["amps"] for row in rows]
        )
        collapsed_q, collapsed_zero, collapsed_negative = _integral(
            [row[axis] for row in reduced], [row["amps"] for row in reduced]
        )
        if original_q != collapsed_q or negative or collapsed_zero or collapsed_negative:
            raise ValueError("Collapsing exact G2 duplicate rows changed the capacity integral.")
        if zero != 26:
            raise ValueError(f"Unexpected G2 repeated timestamp intervals on {axis}: {zero}")
    q_step4 = float(g2_block["testtime_integrated_abs_current_native"])
    testtime_soh = _ratio_percent(
        q_step4, float(g2_bol_block["testtime_integrated_abs_current_native"])
    )
    steptime_soh = _ratio_percent(
        float(g2_block["steptime_integrated_abs_current_native"]),
        float(g2_bol_block["steptime_integrated_abs_current_native"]),
    )
    return {
        "exact_duplicate_pairs_testtime": exact_repeats["testtime"],
        "exact_duplicate_pairs_steptime": exact_repeats["steptime"],
        "exported_rows": len(rows),
        "unique_rows_after_exact_adjacent_duplicate_collapse": len(reduced),
        "testtime_integral_original_native_current_time": q_step4,
        "steptime_integral_original_native_current_time": float(
            g2_block["steptime_integrated_abs_current_native"]
        ),
        "bol_testtime_integral_native_current_time": float(
            g2_bol_block["testtime_integrated_abs_current_native"]
        ),
        "bol_steptime_integral_native_current_time": float(
            g2_bol_block["steptime_integrated_abs_current_native"]
        ),
        "soh_testtime_percent": testtime_soh,
        "soh_steptime_percent": steptime_soh,
        "steptime_minus_testtime_soh_pp": steptime_soh - testtime_soh,
        "integral_delta_after_collapse_testtime": 0.0,
        "integral_delta_after_collapse_steptime": 0.0,
        "duplicate_origin": (
            "Unresolved between tester duplicate emission and export duplication; exact repeated "
            "rows rule out differing simultaneous values in the exported record but do not identify "
            "the upstream cause."
        ),
    }


def _write_g2_report(path: Path, summary: dict[str, Any], evidence_row: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# G2-Cell3 RPT2 repeated-timestamp investigation",
                "",
                "Disposition proposal only; the authoritative candidate manifest remains unchanged.",
                "",
                "## Raw observations",
                "",
                f"- Source: `{evidence_row['rpt_source_file']}`",
                f"- Step 4 block: `{evidence_row['rpt_step4_reference']}`",
                f"- Raw block SHA-256: `{evidence_row['rpt_step4_block_sha256_recomputed']}`",
                f"- Rows: {summary['exported_rows']}; exact adjacent duplicate rows: "
                f"{summary['exact_duplicate_pairs_testtime']} on TestTime and "
                f"{summary['exact_duplicate_pairs_steptime']} on StepTime.",
                "- Each repeated pair is identical across the complete exported row, not just its timestamp.",
                "",
                "## Capacity sensitivity",
                "",
                f"- Step 4 integrated capacity: {summary['testtime_integral_original_native_current_time']:.12f} "
                "native current-time units on TestTime and "
                f"{summary['steptime_integral_original_native_current_time']:.12f} on StepTime.",
                f"- TestTime and StepTime retention: {summary['soh_testtime_percent']:.12f}% and "
                f"{summary['soh_steptime_percent']:.12f}% "
                f"(difference {summary['steptime_minus_testtime_soh_pp']:.12f} pp).",
                "- Removing one row from every exact adjacent duplicate pair changes both integrals by exactly 0.",
                f"- Reconstructed discharge retention: {evidence_row['recomputed_soh_discharge_percent']:.12f}%.",
                f"- Native Amphr span ratio: {evidence_row['native_counter_soh_percent']:.12f}% "
                f"(difference {evidence_row['native_counter_minus_integrated_pp']:.12f} pp).",
                "",
                "## Interpretation",
                "",
                "The raw export cannot distinguish tester-side repeated emission from duplication during export. "
                "There is no evidence in this file alone to attribute the cause to timestamp quantization or a "
                "data-acquisition fault. Because the duplicate rows are exact copies and occur at zero elapsed "
                "time, neither retaining nor collapsing them changes trapezoidal capacity. The repeat-origin "
                "uncertainty therefore does not materially affect this capacity ratio.",
                "",
                "Recommendation: `PROMOTION_SUPPORTED` for the Step 4 discharge reference, subject to explicit "
                "user approval. Step 10 charge retention was not used as a replacement. No tolerance was invented.",
                "`training_authorized: false`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    *,
    evidence: list[dict[str, Any]],
    proposals: list[dict[str, Any]],
    unpaired: list[dict[str, Any]],
    pair_summary: dict[str, Any],
    g2_summary: dict[str, Any],
) -> None:
    proposed = Counter(row["proposed_status"] for row in proposals)
    by_split = Counter(row["partition"] for row in proposals)
    lines = [
        "# Final REVIEW recovery and SoH label verification",
        "",
        "Status: forensic label-quality review and proposed changes only. No authoritative label or split was modified.",
        "`training_authorized: false`; no training, external prediction evaluation, PyDMA, or Half-Cell fitting was run.",
        "",
        "## Decision summary",
        "",
        f"- Original candidate counts: 203 total; 178 ACCEPTED; 25 REVIEW.",
        f"- REVIEW evidence: {proposed.get('PROMOTION_SUPPORTED', 0)} promotion-supported candidates "
        "and 0 excluded candidates; all remain in REVIEW pending explicit approval.",
        "- The 24 Part 3 records are proposed for external-only label recovery after the corrected Guide `.0` "
        "BoL mapping and Step 4 evidence were checked against raw blocks.",
        "- G2-Cell3 RPT2 is proposed for recovery from Step 4 only: repeated entries are 26 identical full-row "
        "duplicates, and exact duplicate collapse changes neither integration.",
        "- The 14 unpaired ACCEPTED G4 labels remain accepted; a non-step-specific Guide-count gate caused "
        "over-exclusion from cross-method comparison.",
        "",
        "## Proposed cohort counts (only if approved)",
        "",
        "| Split | Existing ACCEPTED | Newly promotion-supported | Proposed discharge labels | Remaining REVIEW | EXCLUDED |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Train | 101 | 1 | 102 | 0 | 0 |",
        f"| Validation | 37 | 0 | 37 | 0 | 0 |",
        f"| Part 3 External Test | 40 | 24 | 64 | 0 | 0 |",
        "| Total | 178 | 25 | 203 | 0 | 0 |",
        "",
        "These are proposal counts, not current dispositions. Before approval, the authoritative table remains "
        "178 ACCEPTED and 25 REVIEW. Part 3 remains External Test only; it is not Train/Validation data.",
        "",
        "## Part 3: 24 original REVIEW records",
        "",
        "All 24 have unique Guide RPT source membership, the corrected same-cell unsuffixed BoL-to-`.0` mapping, "
        "a unique complete Step 4 baseline candidate, Step 4-only numerator/denominator, exact raw block hashes, "
        "strictly increasing TestTime and StepTime, discharge-direction current, and a terminal sample at the "
        "lowest observed voltage. Raw terminal samples may sit slightly above 2.5 V; those offsets are recorded "
        "per record, and no instrument tolerance is inferred. Current profiles and voltage windows were compared from the raw blocks; "
        "the Part 3 README documents Maccor 4000 Step 4 CC discharge at C/2 to 2.5 V.",
        "",
        "Independent raw recomputation checks both trapezoidal current-time capacity and the native Amphr span. "
        "The block hash matches the retained independent audit. This resolves the original specific blocker "
        "`bol_protocol_compatibility_unresolved`: the previous mapping selected the `.1` Profile rather than "
        "the unsuffixed BoL procedure's `.0` file.",
        "",
        "Limitations remain: the archived Maccor setting export, exact cutoff tolerance, numeric elapsed-time unit, and Part 3 chamber "
        "temperature were not available. These records are reconstructed same-cell capacity-retention references, "
        "not official Oxford ground truth. These global limitations do not change the resolved Guide/Step 4 mapping.",
        "",
        "| Cell | RPTs | Step 4 BoL source | Recomputed SOH range (%) |",
        "|---|---|---|---:|",
    ]
    by_cell: dict[str, list[dict[str, Any]]] = {}
    for row in evidence:
        if int(row["dataset_part"]) == 3:
            by_cell.setdefault(row["physical_cell_id"], []).append(row)
    for cell, rows in sorted(by_cell.items()):
        lines.append(
            f"| {cell} | {', '.join(str(row['rpt_index']) for row in rows)} | "
            f"`{rows[0]['bol_source_file']}` | "
            f"{min(float(row['recomputed_soh_discharge_percent']) for row in rows):.6f}–"
            f"{max(float(row['recomputed_soh_discharge_percent']) for row in rows):.6f} |"
        )
    lines.extend(
        [
            "",
            "## G2-Cell3 RPT2",
            "",
            f"The raw Step 4 block has {g2_summary['exact_duplicate_pairs_testtime']} exact repeated intervals on "
            "each time field. First-copy collapse changes the Step 4 integral by exactly 0 native current-time "
            f"units. TestTime and StepTime retention are {g2_summary['soh_testtime_percent']:.12f}% and "
            f"{g2_summary['soh_steptime_percent']:.12f}% "
            f"(difference {g2_summary['steptime_minus_testtime_soh_pp']:.12e} pp). "
            "The duplicate origin remains unresolved, but the "
            "integration ambiguity is immaterial to the ratio. Proposed status: `PROMOTION_SUPPORTED`, "
            "subject to user approval. The Step 10 charge proxy was not used.",
            "",
            "## Fourteen previously unpaired ACCEPTED records",
            "",
            "The old comparison implementation gated M1 and M2 together on a single-file Guide source count. "
            "That is too coarse. The original Guide lists two files for TPG4 BoL, but raw Step 4 scanning found "
            "the complete Step 4 discharge only in the `_1` file for both cells; the other Guide-listed file "
            "has no Step 4 block. The selected earliest Step 4 block remains the correct M1 denominator. The "
            "later TPG4.2 Step 4 curve matches the later segment in the primary source and remains sensitivity "
            "evidence, not a replacement denominator.",
            "",
            "- G4-Cell18: M1 remains accepted for 8 records. Both Guide-listed source files contain complete "
            "positive Step 10 charge blocks, so M2 denominator identity is ambiguous; keep all 8 out of M1/M2 "
            "agreement statistics.",
            "- G4-Cell19: M1 remains accepted for 6 records. The `_1` file contains one complete positive "
            "Step 10 segment; the `_2` file contains only zero-current Step 10 plateaus. This supports a unique "
            "Step 10 denominator for these six cross-method pairs.",
            "- No discharge label is demoted solely because an M2 comparison was absent.",
            "",
            f"Valid M1/M2 pairs increase from {pair_summary['previous_valid_pair_count']} to "
            f"{pair_summary['revised_valid_pair_count']} when those six pairs are added. Revised M1/M2 "
            f"mean absolute difference is {pair_summary['revised_pair_summary']['mean_absolute_difference_pp']:.6f} pp; "
            f"median {pair_summary['revised_pair_summary']['median_absolute_difference_pp']:.6f} pp; "
            f"maximum {pair_summary['revised_pair_summary']['max_absolute_difference_pp']:.6f} pp; "
            f"bias M2−M1 {pair_summary['revised_pair_summary']['bias_charge_minus_discharge_pp']:.6f} pp. "
            "Cell18 remains excluded from the paired statistics pending resolution of the two valid Step 10 BoL sources.",
            "",
            "## Status proposal and approval",
            "",
            "- Proposed changes: 24 Part 3 External REVIEW -> ACCEPTED (External Test only); G2-Cell3 RPT2 Train REVIEW -> ACCEPTED.",
            "- No status has been changed. The 14 existing G4 ACCEPTED labels are retained.",
            "- If approved, proposed discharge-label total is 203: Train 102, Validation 37, External Test 64.",
            "- If not approved, authoritative current remains 178 ACCEPTED / 25 REVIEW.",
            "- No charge-proxy labels are mixed into discharge SoH.",
            "- Explicit user approval is required before editing the authoritative label table or any split/eligibility manifest.",
            "",
            "## Reproducibility",
            "",
            "Exact SHA-256 values for input manifests, official source documents, directly scanned raw files, "
            "audit code/configuration, and the five requested deliverables are in "
            "`soh_recovery_sha256_manifest.csv`. The manifest does not hash itself to avoid a circular checksum. "
            "Historical candidate/reference files and all source labels remain unchanged.",
            "",
            "`training_authorized: false`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite versioned audit output: {output_dir}")
    output_dir.mkdir(parents=True)

    candidates = _read_csv(RECONSTRUCTION_DIR / "final_candidate_manifest.csv")
    reviews = _read_csv(MULTIMETHOD_DIR / "soh_review_recovery_audit.csv")
    comparison = _read_csv(MULTIMETHOD_DIR / "soh_multimethod_comparison.csv")
    guide_rows = _read_csv(PART3_CROSSWALK_DIR / "guide_observation_manifest.csv")
    part3_step4_rows = _read_csv(PART3_CROSSWALK_DIR / "part3_bol_step4_capacity_audit.csv")
    part1_bol_rows = _read_csv(PART1_BOL_DIR / "part1_bol_curve_comparison.csv")
    split_rows = _read_csv(BENCHMARK_DIR / "cell_split.csv")
    record_rows, block_rows = _load_raw_audit()
    del record_rows

    if len(candidates) != 203 or len(reviews) != 25:
        raise ValueError("Candidate/review counts changed; stop for explicit review.")
    if sum(row["label_disposition"] == "ACCEPTED" for row in candidates) != 178:
        raise ValueError("Expected exactly 178 original ACCEPTED candidates.")
    if sum(row["label_disposition"] == "REVIEW" for row in candidates) != 25:
        raise ValueError("Expected exactly 25 original REVIEW candidates.")
    _verify_split(candidates, split_rows)

    evidence, proposals, review_raw_hashes, g2_blocks = _review_evidence(
        reviews=reviews,
        candidates=candidates,
        guide_rows=guide_rows,
        part3_step4_rows=part3_step4_rows,
        block_rows=block_rows,
    )
    unpaired_rows, new_pairs, unpaired_raw_hashes, pair_summary = _unpaired_audit(
        comparison=comparison,
        guide_rows=guide_rows,
        bol_curve_rows=part1_bol_rows,
        block_rows=block_rows,
    )
    del new_pairs
    g2_summary = _g2_investigation(g2_blocks[0], g2_blocks[1])

    evidence_by_key = {(row["physical_cell_id"], str(row["rpt_index"])): row for row in evidence}
    for row in proposals:
        row["approval_required"] = True
        row["authoritative_label_modified"] = False
    for row in evidence:
        if row["physical_cell_id"] == "G2-Cell3":
            row.update(
                {
                    "exact_duplicate_pairs_testtime": g2_summary["exact_duplicate_pairs_testtime"],
                    "exact_duplicate_pairs_steptime": g2_summary["exact_duplicate_pairs_steptime"],
                    "duplicate_collapse_integral_delta_testtime": 0.0,
                    "duplicate_collapse_integral_delta_steptime": 0.0,
                    "soh_testtime_percent": g2_summary["soh_testtime_percent"],
                    "soh_steptime_percent": g2_summary["soh_steptime_percent"],
                    "steptime_minus_testtime_soh_pp": g2_summary[
                        "steptime_minus_testtime_soh_pp"
                    ],
                    "duplicate_origin": g2_summary["duplicate_origin"],
                }
            )

    _write_csv(output_dir / "soh_review_recovery_evidence.csv", evidence)
    _write_csv(output_dir / "soh_review_promotion_proposal.csv", proposals)
    _write_csv(output_dir / "soh_unpaired_accepted_audit.csv", unpaired_rows)
    _write_g2_report(output_dir / "soh_g2_timestamp_investigation.md", g2_summary, evidence_by_key[("G2-Cell3", "2")])

    config = {
        "schema_version": 1,
        "training_authorized": False,
        "training_started": False,
        "external_records_used_for_model_selection": False,
        "authoritative_labels_modified": False,
        "scope": {
            "review_records": 25,
            "part3_review_records": 24,
            "g2_cell3_rpt2": 1,
            "unpaired_original_accepted": 14,
        },
        "methods": {
            "discharge_soh": "100 * trapezoidal_integral(abs(Amps), TestTime)_RPT / same-cell Step4 BoL",
            "counter_qc": "100 * native Amphr span_RPT / native Amphr span_BoL",
            "timestamp_duplicate_policy": "preserve raw rows; separately collapse exact adjacent duplicate rows for sensitivity",
        },
        "decision_policy": {
            "promotion_requires": [
                "Guide mapping and physical-cell identity verified",
                "earliest complete compatible Step4 BoL verified",
                "raw block count, Step IDs, transition, and SHA-256 verified",
                "monotone time and discharge direction confirmed",
                "independent integrated ratio and native counter cross-check recorded",
            ],
            "promotion_is_proposal_only": True,
            "part3_partition": "external_test_only",
        },
        "official_source_evidence": list(OFFICIAL_SOURCE_DOCUMENTS),
        "revised_pair_summary": pair_summary,
        "proposal_counts": {
            "train": 102,
            "validation": 37,
            "part3_external_test": 64,
            "newly_promotion_supported": 25,
            "excluded": 0,
            "authoritative_label_changes": 0,
        },
    }
    config_path = output_dir / "soh_review_audit_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_report(
        output_dir / "soh_final_recovery_report.md",
        evidence=evidence,
        proposals=proposals,
        unpaired=unpaired_rows,
        pair_summary=pair_summary,
        g2_summary=g2_summary,
    )

    code_path = Path(__file__).resolve()
    input_paths = [
        RECONSTRUCTION_DIR / "final_candidate_manifest.csv",
        RECONSTRUCTION_DIR / "reconstructed_soh_references.csv",
        INDEPENDENT_DIR / "independent_block_audit.csv",
        INDEPENDENT_DIR / "independent_record_audit.csv",
        MULTIMETHOD_DIR / "soh_review_recovery_audit.csv",
        MULTIMETHOD_DIR / "soh_multimethod_comparison.csv",
        PART3_CROSSWALK_DIR / "guide_observation_manifest.csv",
        PART3_CROSSWALK_DIR / "part3_bol_step4_capacity_audit.csv",
        PART3_CROSSWALK_DIR / "README.md",
        PART1_BOL_DIR / "part1_bol_curve_comparison.csv",
        PROJECT_ROOT / "experiments" / "dataset_review" / "README.md",
        BENCHMARK_DIR / "cell_split.csv",
        PROJECT_ROOT / "tests" / "test_final_review_recovery.py",
    ]
    input_entries: list[dict[str, Any]] = []
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required reproducibility input is missing: {path}")
        digest = _sha256_file(path)
        input_entries.append(
            {
                "role": "input",
                "path": str(path.resolve()),
                "sha256": digest,
            }
        )
    input_entries.extend(
        {
            "role": "official_source_document",
            "path": item["url"],
            "sha256": item["sha256"],
        }
        for item in OFFICIAL_SOURCE_DOCUMENTS
    )
    for path, digest in sorted({**review_raw_hashes, **unpaired_raw_hashes}.items()):
        input_entries.append(
            {
                "role": "raw_source_file",
                "path": path,
                "sha256": digest,
            }
        )
    input_entries.extend(
        [
            {"role": "code", "path": str(code_path), "sha256": _sha256_file(code_path)},
            {
                "role": "configuration",
                "path": str(config_path),
                "sha256": _sha256_file(config_path),
            },
        ]
    )
    output_names = [
        "soh_review_recovery_evidence.csv",
        "soh_review_promotion_proposal.csv",
        "soh_unpaired_accepted_audit.csv",
        "soh_g2_timestamp_investigation.md",
        "soh_final_recovery_report.md",
        "soh_review_audit_config.json",
    ]
    manifest_rows = input_entries + [
        {
            "role": "output",
            "path": str((output_dir / name).resolve()),
            "sha256": _sha256_file(output_dir / name),
        }
        for name in output_names
    ]
    _write_csv(output_dir / "soh_recovery_sha256_manifest.csv", manifest_rows)
    return {
        "output_dir": str(output_dir),
        "review_rows": len(evidence),
        "review_proposal_counts": dict(Counter(row["proposed_status"] for row in proposals)),
        "unpaired_accepted_rows": len(unpaired_rows),
        "revised_valid_m1_m2_pair_count": pair_summary["revised_valid_pair_count"],
        "revised_m1_m2_summary": pair_summary["revised_pair_summary"],
        "part3_review_count": sum(row["dataset_part"] == 3 for row in evidence),
        "training_authorized": False,
        "training_started": False,
        "authoritative_labels_modified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit final REVIEW recovery evidence without changing labels.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    print(json.dumps(run_audit(args.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
