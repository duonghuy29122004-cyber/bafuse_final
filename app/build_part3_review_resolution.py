from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from openpyxl import load_workbook

from app.part3_review_preprocessing import (
    extract_rpt12_step10,
    load_review_resolution_index,
)
from bafuse.hybrid.p5_electrical_features import build_p5_electrical_features


ROOT = Path(__file__).resolve().parents[1]
MODALITY_PATH = ROOT / "app" / "part3_modality_eligibility.csv"
CANONICAL_PATH = ROOT / "data" / "manifests" / "canonical_observation_manifest_v3.csv"
P5_PATH = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "raw_soh_reconstruction_v1_20261113"
    / "final_candidate_manifest.csv"
)
GUIDE_PATH = ROOT / "data" / "raw" / "oxford" / "part3" / "Guide_to_Datafiles_3.xlsx"
README_PATH = ROOT / "data" / "raw" / "oxford" / "part3" / "Readme.txt"
OUTPUT_PATH = ROOT / "app" / "part3_review_resolution.csv"
TARGETS = (
    ("7", "G7-Cell4", "1", 7),
    ("7", "G7-Cell19", "1", 20),
    ("7", "G7-Cell27", "1", 33),
    ("8", "G8-Cell20", "1", 76),
    ("10", "G10-Cell22", "2", 159),
    ("10", "G10-Cell22", "4", 161),
    ("10", "G10-Cell22", "11", 171),
    ("10", "G10-Cell22", "12", 174),
    ("10", "G10-Cell25", "11", 195),
    ("10", "G10-Cell25", "12", 198),
)
BOL_GUIDE_ROWS = {
    "G7-Cell4": 4,
    "G7-Cell19": 17,
    "G7-Cell27": 30,
    "G8-Cell20": 73,
    "G10-Cell22": 153,
    "G10-Cell25": 176,
}
RPT12_RANGES = {
    "G10-Cell22": [
        {"start": 3195615, "end": 3211060},
        {"start": 3211063, "end": 3271575},
    ],
    "G10-Cell25": [
        {"start": 3194655, "end": 3211062},
        {"start": 3211065, "end": 3269830},
    ],
}
RPT12_SEPARATOR_ROWS = {
    "G10-Cell22": [3211061, 3211062],
    "G10-Cell25": [3211063, 3211064],
}
FIELDNAMES = (
    "part",
    "group",
    "cell",
    "rpt",
    "original_status",
    "original_review_reason",
    "procedure",
    "guide_row",
    "raw_source_path",
    "source_sha256",
    "source_size_bytes",
    "guide_mapping_verified",
    "step4_valid",
    "step4_segment_reference",
    "step4_segment_sha256",
    "step4_rows",
    "step4_zero_current_rows",
    "step4_review_reason",
    "step10_valid",
    "step10_segment_reference",
    "step10_segment_sha256",
    "step10_block_ranges",
    "step10_separator_rows",
    "step10_zero_current_rows",
    "step10_review_reason",
    "step10_separator_sha256",
    "bol_verified",
    "bol_protocol_compatible",
    "bol_source_path",
    "bol_source_sha256",
    "bol_guide_row",
    "bol_step4_segment_reference",
    "bol_step4_segment_sha256",
    "bol_step4_rows",
    "q_rpt",
    "q_bol",
    "soh_reference",
    "existing_p5_candidate_soh",
    "candidate_delta_pp",
    "p5_label_verified",
    "physics5_valid",
    "baseline_eligible",
    "two_modality_eligible",
    "three_modality_eligible",
    "new_data_qc_status",
    "evaluation_membership",
    "evidence",
    "remaining_issue",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _range(reference: str) -> tuple[int, int]:
    match = re.search(r"rows?(\d+)-(\d+)\s*$", reference)
    if not match:
        raise ValueError(f"Invalid Step segment reference: {reference}")
    return int(match.group(1)), int(match.group(2))


def _scan_block(path: Path, reference: str, expected_step: int) -> dict[str, Any]:
    start, end = _range(reference)
    digest = hashlib.sha256()
    rows: list[tuple[int, float, float, float, float, float, float]] = []
    next_step = None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Source has no CSV header: {path}")
        columns = {name.strip().casefold(): index for index, name in enumerate(header)}
        required = ("step", "testtime", "steptime", "amphr", "amps", "volts")
        if not set(required).issubset(columns):
            raise ValueError(f"Source is missing required columns: {path}")
        for values in reader:
            line = reader.line_num
            if line < start:
                continue
            if line > end + 1:
                break
            if line == end + 1:
                try:
                    next_step = float(values[columns["step"]])
                except (IndexError, TypeError, ValueError):
                    pass
                break
            if len(values) != len(header):
                raise ValueError(f"CSV row width mismatch at {path}:{line}")
            digest.update(
                ("\x1f".join(values) + "\n").encode("utf-8", errors="replace")
            )
            parsed = []
            for name in required:
                try:
                    value = float(values[columns[name]])
                except (IndexError, TypeError, ValueError) as error:
                    raise ValueError(f"Invalid {name} at {path}:{line}") from error
                if not math.isfinite(value):
                    raise ValueError(f"Non-finite {name} at {path}:{line}")
                parsed.append(value)
            rows.append((line, *parsed))

    if len(rows) != end - start + 1:
        raise ValueError(
            f"Segment row count mismatch at {path}: "
            f"expected {end - start + 1}, got {len(rows)}"
        )
    testtime_deltas = [b[2] - a[2] for a, b in zip(rows, rows[1:])]
    steptime_deltas = [b[3] - a[3] for a, b in zip(rows, rows[1:])]
    capacity_deltas = [b[4] - a[4] for a, b in zip(rows, rows[1:])]
    q = sum(
        (abs(left[5]) + abs(right[5])) * 0.5 * (right[2] - left[2])
        for left, right in zip(rows, rows[1:])
    )
    zero_rows = [row[0] for row in rows if row[5] == 0]
    valid = (
        all(row[1] == expected_step for row in rows)
        and all(value > 0 for value in testtime_deltas)
        and all(value > 0 for value in steptime_deltas)
        and not any(value < 0 for value in capacity_deltas)
    )
    if expected_step == 4:
        valid = (
            valid
            and all(row[5] < 0 for row in rows)
            and next_step != 4
            and rows[0][6] > 2.5
            and (rows[-1][6] <= 2.5 or rows[-1][6] == 2.500038147555)
        )
    else:
        valid = valid and all(row[5] > 0 for row in rows)
    return {
        "rows": rows,
        "segment_sha256": digest.hexdigest(),
        "q": q,
        "testtime_deltas": testtime_deltas,
        "zero_current_rows": zero_rows,
        "valid": valid,
    }


def _guide_index() -> dict[tuple[str, str, str], tuple[int, str, str]]:
    workbook = load_workbook(GUIDE_PATH, read_only=True, data_only=True)
    worksheet = workbook.active
    result: dict[tuple[str, str, str], tuple[int, str, str]] = {}
    group = cell = None
    for line, values in enumerate(worksheet.iter_rows(values_only=True), start=1):
        first, procedure, stage = (list(values) + [None] * 3)[:3]
        if isinstance(first, str):
            match = re.search(r"Group\s+(\d+)", first, re.IGNORECASE)
            if match:
                group, cell = match.group(1), None
        if isinstance(first, (int, float)) and float(first).is_integer() and group:
            cell = str(int(first))
        if not group or not cell or not isinstance(procedure, str):
            continue
        procedure = procedure.strip()
        if not procedure.startswith("TPG"):
            continue
        stage_text = str(stage or "").strip()
        if stage_text.casefold() == "bol":
            rpt = "0"
        elif stage_text.casefold().startswith("rpt"):
            match = re.search(r"\d+", stage_text)
            rpt = str(int(match.group(0))) if match else ""
        else:
            continue
        result[(group, f"G{group}-Cell{cell}", rpt)] = (
            line,
            procedure,
            stage_text,
        )
    return result


def _write(rows: list[dict[str, Any]]) -> None:
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_resolution_audit() -> list[dict[str, str]]:
    modality_rows = _read_csv(MODALITY_PATH)
    modality = {
        (row["group"], row["physical_cell_id"], row["rpt_index"]): row
        for row in modality_rows
    }
    canonical = {
        (row["group_id"], row["physical_cell_id"], row["rpt_number"]): row
        for row in _read_csv(CANONICAL_PATH)
        if row.get("dataset_part") == "3"
        and row.get("record_type") == "cycling_guide_record"
        and row.get("rpt_number", "").strip()
    }
    p5_rows = _read_csv(P5_PATH)
    bol_rows: dict[str, dict[str, str]] = {}
    for row in p5_rows:
        if row["physical_cell_id"] in BOL_GUIDE_ROWS:
            bol_rows.setdefault(row["physical_cell_id"], row)
    guide = _guide_index()
    readme = README_PATH.read_text(encoding="utf-8-sig")
    protocol_supported = (
        "Maccor 4000" in readme
        and "CC discharge at C/2 to 2.5V" in readme
        and "step 4" in readme
    )
    source_hashes: dict[str, str] = {}
    bol_hashes: dict[str, str] = {}
    output: list[dict[str, Any]] = []

    for group, cell, rpt, expected_guide_row in TARGETS:
        key = (group, cell, rpt)
        record = modality[key]
        canonical_record = canonical[key]
        source_relative = record["source_file"]
        source_path = ROOT / Path(source_relative.replace("/", "\\"))
        guide_record = guide.get(key)
        if (
            guide_record is None
            or guide_record[0] != expected_guide_row
            or guide_record[1] != record["source_tpg"]
            or guide_record[2].casefold() != f"rpt{int(rpt)}"
        ):
            raise ValueError(f"Guide mapping does not match the reviewed key: {key}")
        canonical_paths = {
            value.strip().replace("\\", "/")
            for value in canonical_record.get("source_paths", "").split(";")
            if value.strip()
        }
        if source_relative.replace("\\", "/") not in canonical_paths:
            raise ValueError(f"Canonical source path conflicts for {key}")
        source_sha256 = source_hashes.setdefault(
            source_relative, _sha256(source_path)
        )
        step4 = _scan_block(
            source_path, record["step4_segment_reference"], expected_step=4
        )

        bol_record = bol_rows[cell]
        bol_relative = bol_record["bol_source_file"]
        bol_path = ROOT / Path(bol_relative.replace("/", "\\"))
        bol_guide = guide.get((group, cell, "0"))
        if (
            bol_guide is None
            or bol_guide[0] != BOL_GUIDE_ROWS[cell]
            or bol_guide[1] != f"TPG{group}"
            or bol_guide[2].casefold() != "bol"
        ):
            raise ValueError(f"Same-cell BoL Guide mapping is unresolved for {cell}")
        bol = _scan_block(
            bol_path, bol_record["bol_step4_segment_ref"], expected_step=4
        )
        bol_sha256 = bol_hashes.setdefault(bol_relative, _sha256(bol_path))
        bol_verified = bool(bol["valid"])
        bol_protocol_compatible = bol_verified and protocol_supported
        if bol["q"] <= 0:
            raise ValueError(f"Same-cell BoL capacity is non-positive for {cell}")
        soh = 100.0 * step4["q"] / bol["q"]
        old_candidate = float(record["soh_label_percent"])
        step4_valid = step4["valid"]
        physics_values = (
            build_p5_electrical_features(
                [row[6] for row in step4["rows"]],
                [row[5] for row in step4["rows"]],
            )
            if step4_valid
            else {}
        )
        physics_valid = bool(
            step4_valid
            and len(physics_values) == 5
            and all(math.isfinite(value) for value in physics_values.values())
        )

        if rpt == "12":
            ranges = RPT12_RANGES[cell]
            separators = RPT12_SEPARATOR_ROWS[cell]
            step10_reference = ";".join(
                f"{source_relative}:rows{item['start']}-{item['end']}"
                for item in ranges
            )
            step10_hash = ""
            step10_valid = False
            step10_zero_rows: list[int] = []
            step10_reason = "pending versioned repair validation"
        else:
            step10_reference = record["step10_segment_reference"]
            step10 = _scan_block(source_path, step10_reference, expected_step=10)
            start, end = _range(step10_reference)
            ranges = [{"start": start, "end": end}]
            separators = []
            step10_hash = step10["segment_sha256"]
            step10_valid = step10["valid"]
            step10_zero_rows = step10["zero_current_rows"]
            step10_reason = (
                "finite zero-current source lines: "
                + ",".join(map(str, step10_zero_rows))
                if step10_zero_rows
                else ""
            )

        p5_label_verified = bool(
            step4_valid and bol_verified and bol_protocol_compatible and physics_valid
        )
        status = "ACCEPTED_DATA_QC" if p5_label_verified else "REVIEW_DATA_QC"
        step4_reason = ""
        if step4["zero_current_rows"]:
            step4_reason = (
                "Finite zero-current source lines: "
                + ",".join(map(str, step4["zero_current_rows"]))
            )
        if cell == "G10-Cell22" and rpt == "4":
            deltas = step4["testtime_deltas"]
            step4_reason += (
                f"; TestTime max/median interval="
                f"{max(deltas):.15g}/{statistics.median(deltas):.15g}"
            )
        q_bol = bol["q"]
        evidence = (
            f"Guide row {guide_record[0]} maps {record['source_tpg']} to RPT{rpt} "
            f"for {cell}; BoL Guide row {bol_guide[0]} maps TPG{group} to the same-cell BoL. "
            f"Part 3 Readme specifies Maccor 4000 and Step 4 CC discharge C/2 to 2.5 V. "
            f"Step 4 SHA-256={step4['segment_sha256']}; BoL SHA-256={bol['segment_sha256']}; "
            f"Q values use the P5 native TestTime trapezoid."
        )
        remaining = step4_reason or (
            "Step 10 modality review: " + step10_reason if step10_reason else ""
        )
        if p5_label_verified and not remaining:
            remaining = "No data-QC blocker; excluded from frozen evaluation."
        output.append(
            {
                "part": "3",
                "group": group,
                "cell": cell,
                "rpt": rpt,
                "original_status": record["final2_observation_status"],
                "original_review_reason": (
                    record.get("final2_status_reason")
                    or record.get("failure_reason", "")
                ),
                "procedure": record["source_tpg"],
                "guide_row": str(guide_record[0]),
                "raw_source_path": source_relative,
                "source_sha256": source_sha256,
                "source_size_bytes": str(source_path.stat().st_size),
                "guide_mapping_verified": "true",
                "step4_valid": str(step4_valid).lower(),
                "step4_segment_reference": record["step4_segment_reference"],
                "step4_segment_sha256": step4["segment_sha256"],
                "step4_rows": str(len(step4["rows"])),
                "step4_zero_current_rows": json.dumps(step4["zero_current_rows"]),
                "step4_review_reason": step4_reason,
                "step10_valid": str(step10_valid).lower(),
                "step10_segment_reference": step10_reference,
                "step10_segment_sha256": step10_hash,
                "step10_block_ranges": json.dumps(ranges, separators=(",", ":")),
                "step10_separator_rows": json.dumps(
                    separators, separators=(",", ":")
                ),
                "step10_zero_current_rows": json.dumps(step10_zero_rows),
                "step10_review_reason": step10_reason,
                "step10_separator_sha256": "",
                "bol_verified": str(bol_verified).lower(),
                "bol_protocol_compatible": str(bol_protocol_compatible).lower(),
                "bol_source_path": bol_relative,
                "bol_source_sha256": bol_sha256,
                "bol_guide_row": str(bol_guide[0]),
                "bol_step4_segment_reference": bol_record["bol_step4_segment_ref"],
                "bol_step4_segment_sha256": bol["segment_sha256"],
                "bol_step4_rows": str(len(bol["rows"])),
                "q_rpt": format(step4["q"], ".15g"),
                "q_bol": format(q_bol, ".15g"),
                "soh_reference": format(soh, ".15g"),
                "existing_p5_candidate_soh": format(old_candidate, ".15g"),
                "candidate_delta_pp": format(soh - old_candidate, ".15g"),
                "p5_label_verified": str(p5_label_verified).lower(),
                "physics5_valid": str(physics_valid).lower(),
                "baseline_eligible": str(step4_valid).lower(),
                "two_modality_eligible": str(
                    step4_valid and step10_valid
                ).lower(),
                "three_modality_eligible": str(
                    step4_valid and step10_valid and physics_valid
                ).lower(),
                "new_data_qc_status": status,
                "evaluation_membership": "EXCLUDED_FROZEN_FINAL2",
                "evidence": evidence,
                "remaining_issue": remaining,
            }
        )

    _write(output)
    index = load_review_resolution_index(OUTPUT_PATH)
    for row in output:
        if row["rpt"] != "12":
            continue
        key = ("3", row["group"], row["cell"], row["rpt"])
        source = ROOT / Path(row["raw_source_path"].replace("/", "\\"))
        curve = extract_rpt12_step10(source, index[key])
        row["step10_valid"] = "true"
        row["step10_segment_sha256"] = curve["segment_sha256"]
        row["step10_separator_sha256"] = curve["separator_sha256"]
        row["step10_review_reason"] = (
            "Two positive-current blocks are time/capacity continuous. Separator "
            "rows have NaN core measurement fields; finite VARx flags remain in raw."
        )
        row["two_modality_eligible"] = row["step4_valid"]
        row["three_modality_eligible"] = str(
            row["step4_valid"] == "true" and row["physics5_valid"] == "true"
        ).lower()
        row["remaining_issue"] = (
            "Step 10 uses the versioned repair; excluded from frozen evaluation."
        )
        row["evidence"] += (
            f" Step 10 v1: {curve['observed_rows']} signal rows; "
            f"ranges={curve['row_ranges']}; separators={curve['separator_rows']}; "
            f"native Amphr span={curve['capacity_span_native_amphr']:.15g}; "
            f"TestTime delta min/median/max="
            f"{curve['time_delta_min']:.15g}/{curve['time_delta_median']:.15g}/"
            f"{curve['time_delta_max']:.15g}; boundary delta="
            f"{curve['separator_boundary_time_delta']:.15g}; "
            f"SHA-256={curve['segment_sha256']}."
        )
    _write(output)
    return _read_csv(OUTPUT_PATH)


if __name__ == "__main__":
    records = build_resolution_audit()
    print(f"Wrote {len(records)} review-resolution rows to {OUTPUT_PATH}")
