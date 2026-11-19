from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
INDEPENDENT_DIR = BENCHMARK_DIR / "independent_raw_soh_audit_v1_20261113"
RECOVERY_DIR = BENCHMARK_DIR / "soh_final_review_recovery_v1_20261114"
OUTPUT_DIR = BENCHMARK_DIR / "bafuse_soh_trajectory_audit_v1_20261114"
APPROVED_RECOVERY_SHA256 = "b0146c9da6f42459bc10982cc3a641ccc130e3ec10dd64b35eb12880e34d8791"
ROBUST_Z_THRESHOLD = 3.5
ROBUST_MIN_SAMPLE_SIZE = 5
NUMERIC_LABEL_TOLERANCE_PP = 1e-8

_BLOCK_RE = re.compile(r"#Step(?P<step>\d+):[^:]*:rows(?P<start>\d+)-(?P<end>\d+)(?::sub\d+)?$", re.I)
_CELL_RE = re.compile(r"^G(?P<group>\d+)-Cell(?P<cell>\d+)$")
_RAW_FIELDS = ("step", "testtime", "steptime", "amphr", "amps", "volts")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
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
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _finite(value: float | None, name: str) -> float:
    if value is None or not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _parse_block_ref(reference: str) -> tuple[int, int, int]:
    match = _BLOCK_RE.search(reference.strip())
    if match is None:
        raise ValueError(f"Unparseable Step block reference: {reference!r}")
    return int(match.group("step")), int(match.group("start")), int(match.group("end"))


def _ratio_percent(numerator: float, denominator: float) -> float:
    if numerator <= 0 or denominator <= 0:
        raise ValueError("Step 4 capacity numerator and BoL denominator must be positive.")
    return 100.0 * numerator / denominator


def _block_key(source: str, part: int, reference: str) -> tuple[str, int, int, int, int]:
    step, start, end = _parse_block_ref(reference)
    return (_normalize_path(source), part, step, start, end)


def _verify_split(
    candidates: list[dict[str, str]], splits: list[dict[str, str]]
) -> dict[str, str]:
    split_by_cell = {row["physical_cell_id"]: row["partition"] for row in splits}
    seen: dict[str, set[str]] = defaultdict(set)
    for row in candidates:
        cell = row["physical_cell_id"]
        partition = row["partition"]
        if split_by_cell.get(cell) != partition:
            raise ValueError(f"Physical-cell split mismatch for {cell}: {partition}")
        if row["dataset_part"] == "3" and partition != "external_test":
            raise ValueError(f"Part 3 record is not External Test only: {cell}")
        seen[cell].add(partition)
    overlap = {cell: values for cell, values in seen.items() if len(values) != 1}
    if overlap:
        raise ValueError(f"Physical cells overlap partitions: {overlap}")
    return split_by_cell


def _verify_approved_recovery() -> tuple[
    list[dict[str, str]], list[dict[str, str]], list[dict[str, str]], dict[str, str]
]:
    manifest_path = RECOVERY_DIR / "soh_recovery_sha256_manifest.csv"
    actual_manifest_hash = _sha256(manifest_path)
    if actual_manifest_hash != APPROVED_RECOVERY_SHA256:
        raise ValueError(
            "Approved recovery manifest hash changed; refusing label promotion: "
            f"{actual_manifest_hash}"
        )
    entries = _read_csv(manifest_path)
    local_entries = [
        row
        for row in entries
        if row["role"] in {"input", "raw_source_file", "code", "configuration", "output"}
    ]

    def verify_entry(row: dict[str, str]) -> tuple[str, str]:
        path = Path(row["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            return row["path"], "MISSING"
        actual = _sha256(path)
        return row["path"], "OK" if actual == row["sha256"] else f"MISMATCH:{actual}"

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(verify_entry, local_entries))
    failures = [(path, status) for path, status in results if status != "OK"]
    if failures:
        raise ValueError(f"Recovery source/output hash changed; refusing promotion: {failures[:10]}")

    evidence = _read_csv(RECOVERY_DIR / "soh_review_recovery_evidence.csv")
    proposals = _read_csv(RECOVERY_DIR / "soh_review_promotion_proposal.csv")
    if len(evidence) != 25 or len(proposals) != 25:
        raise ValueError("Approved promotion evidence must contain exactly 25 records.")
    evidence_by_key = {
        (row["physical_cell_id"], row["rpt_index"], row["dataset_part"]): row
        for row in evidence
    }
    if len(evidence_by_key) != 25:
        raise ValueError("Promotion evidence contains duplicate observation keys.")
    raw_hash_by_path = {
        _normalize_path(row["path"]): row["sha256"]
        for row in entries
        if row["role"] == "raw_source_file"
    }
    for proposal in proposals:
        key = (
            proposal["physical_cell_id"],
            proposal["rpt_index"],
            proposal["dataset_part"],
        )
        item = evidence_by_key.get(key)
        if (
            item is None
            or proposal["original_status"] != "REVIEW"
            or proposal["proposed_status"] != "PROMOTION_SUPPORTED"
            or proposal["decision"] != "PROPOSED_ONLY_USER_APPROVAL_REQUIRED"
            or item["original_status"] != "REVIEW"
            or item["recommended_status"] != "PROMOTION_SUPPORTED"
            or item["status_promoted"] != "False"
        ):
            raise ValueError(f"Approved status/evidence mismatch for {key}")
        if (
            item["rpt_step4_block_sha256_expected"]
            != item["rpt_step4_block_sha256_recomputed"]
            or item["bol_step4_block_sha256_expected"]
            != item["bol_step4_block_sha256_recomputed"]
        ):
            raise ValueError(f"Promotion block hash mismatch for {key}")
        for field in ("rpt_source_file", "bol_source_file"):
            if raw_hash_by_path.get(_normalize_path(item[field])) != item[
                "rpt_source_file_sha256" if field == "rpt_source_file" else "bol_source_file_sha256"
            ]:
                raise ValueError(f"Promotion raw-file hash mismatch for {key}: {field}")
        if item["dataset_part"] == "3":
            cell_number = item["physical_cell_id"].split("Cell", 1)[1]
            if not Path(item["bol_source_file"]).name.casefold().endswith(
                f".0-cell{cell_number}.csv"
            ):
                raise ValueError(f"Part 3 promotion does not use corrected .0 BoL for {key}")
        elif (
            item["physical_cell_id"] != "G2-Cell3"
            or item["rpt_index"] != "2"
            or item["exact_duplicate_pairs_testtime"] != "26"
            or item["exact_duplicate_pairs_steptime"] != "26"
            or float(item["duplicate_collapse_integral_delta_testtime"]) != 0.0
            or float(item["duplicate_collapse_integral_delta_steptime"]) != 0.0
        ):
            raise ValueError(f"G2-Cell3 duplicate-row evidence changed for {key}")
    return entries, evidence, proposals, raw_hash_by_path


def _load_block_audit() -> dict[tuple[str, int, int, int, int], dict[str, str]]:
    rows = _read_csv(INDEPENDENT_DIR / "independent_block_audit.csv")
    blocks: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for row in rows:
        key = (
            _normalize_path(row["source_file"]),
            int(row["source_part"]),
            int(row["step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        if key in blocks:
            raise ValueError(f"Duplicate independent block audit entry: {key}")
        blocks[key] = row
    return blocks


def _promote_labels(
    candidates: list[dict[str, str]],
    references: list[dict[str, str]],
    evidence: list[dict[str, str]],
    splits: list[dict[str, str]],
) -> list[dict[str, Any]]:
    if len(candidates) != 203:
        raise ValueError(f"Expected 203 candidate records, found {len(candidates)}")
    disposition = Counter(row["label_disposition"] for row in candidates)
    if disposition != Counter({"ACCEPTED": 178, "REVIEW": 25}):
        raise ValueError(f"Unexpected pre-promotion dispositions: {disposition}")
    _verify_split(candidates, splits)
    candidate_by_key: dict[tuple[str, str, str], dict[str, str]] = {}
    for row in candidates:
        key = (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        if key in candidate_by_key:
            raise ValueError(f"Duplicate candidate observation key: {key}")
        candidate_by_key[key] = row

    reference_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in references
        if row["label_disposition"] == "ACCEPTED"
    }
    if len(reference_by_key) != 178:
        raise ValueError(f"Expected 178 historical accepted references, found {len(reference_by_key)}")
    evidence_by_key = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"]): row
        for row in evidence
    }
    if len(evidence_by_key) != 25:
        raise ValueError("Promotion evidence key count changed.")
    final_rows: list[dict[str, Any]] = []
    for key, candidate in candidate_by_key.items():
        original_status = candidate["label_disposition"]
        promoted = original_status == "REVIEW"
        if promoted:
            item = evidence_by_key.get(key)
            if item is None:
                raise ValueError(f"Approved promotion is missing evidence: {key}")
            q_rpt = _finite(_number(item["rpt_step4_integrated_native_current_time"]), "RPT Step 4 capacity")
            q_bol = _finite(_number(item["bol_step4_integrated_native_current_time"]), "BoL Step 4 capacity")
            soh = _ratio_percent(q_rpt, q_bol)
            if not math.isclose(
                soh,
                float(item["recomputed_soh_discharge_percent"]),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(f"Approved promotion ratio changed for {key}")
            if key[0] == "3" and not item["bol_source_file"].casefold().endswith(
                f".0-cell{key[1].split('Cell', 1)[1]}.csv"
            ):
                raise ValueError(f"Approved Part 3 denominator changed for {key}")
            evidence_path = RECOVERY_DIR / "soh_review_recovery_evidence.csv"
            full_rpt_sha = item["rpt_source_file_sha256"]
            full_bol_sha = item["bol_source_file_sha256"]
            promotion_status = "APPROVED_AND_PROMOTED"
            provenance = (
                "User-approved promotion on 2026-11-14; independently recomputed from verified "
                "Step 4 discharge blocks; see the linked recovery evidence row."
            )
            step4_ref = item["rpt_step4_reference"]
            bol_ref = item["bol_step4_reference"]
            rpt_source = item["rpt_source_file"]
            bol_source = item["bol_source_file"]
            rpt_hash = item["rpt_step4_block_sha256_recomputed"]
            bol_hash = item["bol_step4_block_sha256_recomputed"]
            q_rpt_step = _number(candidate["q_rpt_integrated_steptime_native_current_time"])
            q_bol_step = _number(candidate["q_bol_integrated_steptime_native_current_time"])
            native_rpt = _number(item["rpt_step4_native_amphr_span"])
            native_bol = _number(item["bol_step4_native_amphr_span"])
        else:
            reference = reference_by_key.get(key)
            if reference is None:
                raise ValueError(f"Existing ACCEPTED record is missing from reconstructed references: {key}")
            q_rpt = _finite(
                _number(candidate["q_rpt_integrated_testtime_native_current_time"]),
                "RPT Step 4 capacity",
            )
            q_bol = _finite(
                _number(candidate["q_bol_integrated_testtime_native_current_time"]),
                "BoL Step 4 capacity",
            )
            soh = _ratio_percent(q_rpt, q_bol)
            reference_soh = _finite(_number(reference["reconstructed_soh_percent"]), "Historical SoH")
            if not math.isclose(soh, reference_soh, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"Existing ACCEPTED target formula mismatch for {key}")
            if not math.isclose(
                soh,
                _finite(_number(candidate["reconstructed_soh_percent"]), "Candidate SoH"),
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(f"Candidate/reconstruction target mismatch for {key}")
            item = None
            evidence_path = ""
            full_rpt_sha = ""
            full_bol_sha = ""
            promotion_status = "EXISTING_ACCEPTED_PRESERVED"
            provenance = (
                "Original ACCEPTED Step 4 discharge-capacity ratio retained unchanged; "
                "recomputed from the recorded integrated-current capacities."
            )
            step4_ref = candidate["step4_segment_ref"]
            bol_ref = candidate["bol_step4_segment_ref"]
            rpt_source = candidate["source_file"]
            bol_source = candidate["bol_source_file"]
            rpt_hash = candidate["step4_block_sha256"]
            bol_hash = candidate["bol_block_sha256"]
            q_rpt_step = _number(candidate["q_rpt_integrated_steptime_native_current_time"])
            q_bol_step = _number(candidate["q_bol_integrated_steptime_native_current_time"])
            native_rpt = _number(candidate["q_rpt_native_amphr_span"])
            native_bol = _number(candidate["q_bol_native_amphr_span"])

        if any(value is None for value in (q_rpt_step, q_bol_step, native_rpt, native_bol)):
            raise ValueError(f"Missing comparison capacity for {key}")
        soh_steptime = _ratio_percent(float(q_rpt_step), float(q_bol_step))
        soh_native = _ratio_percent(float(native_rpt), float(native_bol))
        final_rows.append(
            {
                "dataset_part": int(candidate["dataset_part"]),
                "physical_cell_id": candidate["physical_cell_id"],
                "rpt_index": int(candidate["rpt_index"]),
                "measurement_id": candidate["measurement_id"],
                "partition": candidate["partition"],
                "original_status": original_status,
                "final_status": "ACCEPTED",
                "promotion_status": promotion_status,
                "promotion_provenance": provenance,
                "label_promotion_authorized": True,
                "soh_percent": soh,
                "target_definition": "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)",
                "target_semantics": "Reconstructed capacity-retention reference; not official Oxford ground truth.",
                "capacity_method": "Trapezoidal integral of abs(Amps) against native elapsed time within the verified Step 4 block.",
                "q_rpt_step4_testtime_native_current_time": q_rpt,
                "q_bol_step4_testtime_native_current_time": q_bol,
                "q_rpt_step4_steptime_native_current_time": float(q_rpt_step),
                "q_bol_step4_steptime_native_current_time": float(q_bol_step),
                "soh_step4_steptime_qc_percent": soh_steptime,
                "steptime_minus_testtime_soh_pp": soh_steptime - soh,
                "rpt_step4_native_amphr_span": float(native_rpt),
                "bol_step4_native_amphr_span": float(native_bol),
                "soh_native_amphr_qc_percent": soh_native,
                "native_amphr_minus_integrated_soh_pp": soh_native - soh,
                "rpt_step4_source_file": rpt_source,
                "rpt_step4_segment_ref": step4_ref,
                "rpt_step4_block_sha256": rpt_hash,
                "rpt_source_file_sha256": full_rpt_sha,
                "bol_step4_source_file": bol_source,
                "bol_step4_segment_ref": bol_ref,
                "bol_step4_block_sha256": bol_hash,
                "bol_source_file_sha256": full_bol_sha,
                "boL_source_part": int(candidate["bol_source_part"]),
                "bol_resolution_status": candidate["bol_resolution_status"],
                "protocol_identity": candidate["protocol_identity"],
                "protocol_compatibility": (
                    "Approved same-cell .0 BoL / Step 4 crosswalk"
                    if item and candidate["dataset_part"] == "3"
                    else "Approved same-cell Step 4 evidence; duplicate-row policy applied"
                    if item
                    else candidate["reference_protocol_compatibility"]
                ),
                "original_review_reason": item["original_review_reason"] if item else "",
                "promotion_evidence_file": str(evidence_path) if promoted else "",
                "promotion_evidence_rpt_file_sha256": full_rpt_sha,
                "promotion_evidence_bol_file_sha256": full_bol_sha,
                "chronology_basis": "Guide-mapped RPT ordinal; not exact elapsed lifetime or cycle age.",
                "nominal_cycle_age_used": False,
                "external_test_only": candidate["partition"] == "external_test",
                "training_authorized": False,
            }
        )
    if len(final_rows) != 203 or len(
        {(row["dataset_part"], row["physical_cell_id"], row["rpt_index"]) for row in final_rows}
    ) != 203:
        raise ValueError("Final accepted label table does not have 203 unique observation keys.")
    if Counter(row["partition"] for row in final_rows) != Counter(
        {"train": 102, "validation": 37, "external_test": 64}
    ):
        raise ValueError("Final split counts do not match the approved cohort.")
    if Counter(row["final_status"] for row in final_rows) != Counter({"ACCEPTED": 203}):
        raise ValueError("Final label table contains a non-ACCEPTED or missing label.")
    return sorted(
        final_rows,
        key=lambda row: (
            row["partition"],
            row["physical_cell_id"],
            row["rpt_index"],
            row["dataset_part"],
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required table: {path}")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def _write_promotion_report(
    path: Path,
    labels: list[dict[str, Any]],
    recovery_manifest_sha: str,
) -> None:
    promoted = [row for row in labels if row["promotion_status"] == "APPROVED_AND_PROMOTED"]
    lines = [
        "# BaFuse SoH label promotion report",
        "",
        "User approval: all 25 evidence-supported REVIEW promotions approved on 2026-11-14.",
        "This creates a new authoritative label table without changing historical labels, source data, or prior audit artifacts.",
        "",
        "## Final disposition",
        "",
        "| Split | ACCEPTED | REVIEW |",
        "|---|---:|---:|",
        "| Train | 102 | 0 |",
        "| Validation | 37 | 0 |",
        "| External Test (Part 3) | 64 | 0 |",
        "| Total | 203 | 0 |",
        "",
        f"Approved recovery manifest SHA-256: `{recovery_manifest_sha}`.",
        "All local sources, raw source hashes, output hashes, and promotion block hashes were reverified before promotion.",
        "",
        "## Target construction",
        "",
        "SoH (%) = 100 × Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL).",
        "Q is the trapezoidal integral of absolute current against the native time field within the verified Step 4 segment. "
        "No Step 10 charge retention or nominal 3 Ah value is used as the label.",
        "The ratio is retained as a reconstructed capacity-retention reference, not official Oxford ground truth. "
        "Absolute time and Amphr unit semantics remain unverified.",
        "",
        "## Approved promotions",
        "",
        "| Part | Cell | RPT | Split | Original status | Final SoH (%) | BoL source | BoL Step 4 block SHA-256 | Promotion evidence |",
        "|---:|---|---:|---|---|---:|---|---|---|",
    ]
    for row in promoted:
        lines.append(
            f"| {row['dataset_part']} | {row['physical_cell_id']} | {row['rpt_index']} | "
            f"{row['partition']} | {row['original_status']} | {row['soh_percent']:.9f} | "
            f"`{row['bol_step4_source_file']}` | `{row['bol_step4_block_sha256']}` | "
            f"`{Path(row['promotion_evidence_file']).name}` |"
        )
    lines.extend(
        [
            "",
            "All 24 Part 3 promotions use the corrected same-cell `.0` BoL source and remain External Test only. "
            "G2-Cell3 RPT2 uses its Step 4 discharge reconstruction; 26 exact repeated rows per elapsed-time axis "
            "were retained, and collapsing exact duplicates changes neither integral. No charge-proxy labels were mixed in.",
            "",
            "## Preserved records and safeguards",
            "",
            "- All 178 historical ACCEPTED Step 4 labels were retained without alteration.",
            "- Every promoted record carries original status, final status, RPT/BoL segment references, block hashes, full source-file hashes, and promotion evidence provenance in `bafuse_soh_labels_203.csv`.",
            "- Physical-cell partitions are unchanged; all Part 3 rows remain External Test.",
            "- `dataset_freeze_authorized: false`; `training_authorized: false`; no training or external prediction evaluation was run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _independent_scan_block(
    source_file: str,
    part: int,
    reference: str,
    expected_block_hash: str,
    file_hash_cache: dict[str, str],
) -> dict[str, Any]:
    step, start_line, end_line = _parse_block_ref(reference)
    source_key = _normalize_path(source_file)
    source_path = PROJECT_ROOT / Path(source_file.replace("/", "\\"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Raw Step 4 source is missing: {source_path}")
    if source_key not in file_hash_cache:
        file_hash_cache[source_key] = _sha256(source_path)
    source_sha = file_hash_cache[source_key]
    block_hash = hashlib.sha256()
    header: list[str] | None = None
    columns: dict[str, int] = {}
    raw_rows: list[bytes] = []
    parsed_rows: list[dict[str, float | None]] = []
    next_step: int | None = None
    with source_path.open("rb") as stream:
        for line_num, raw in enumerate(stream, 1):
            if line_num == 1:
                header = next(csv.reader([raw.decode("utf-8-sig", errors="replace")]))
                columns = {name.strip().casefold(): i for i, name in enumerate(header)}
            if start_line <= line_num <= end_line + 1:
                row = next(csv.reader([raw.decode("utf-8", errors="replace")]))
                if line_num <= end_line:
                    if header is None or len(row) != len(header):
                        raise ValueError(f"Raw row width mismatch at {source_path}:{line_num}")
                    block_hash.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
                    raw_rows.append(raw.rstrip(b"\r\n"))
                    values: dict[str, float | None] = {}
                    for field in _RAW_FIELDS:
                        index = columns.get(field)
                        if index is None or index >= len(row):
                            values[field] = None
                        else:
                            values[field] = _number(row[index])
                    parsed_rows.append(values)
                else:
                    step_index = columns.get("step")
                    next_step = (
                        int(float(row[step_index]))
                        if step_index is not None and step_index < len(row)
                        else None
                    )
    actual_hash = block_hash.hexdigest()
    if actual_hash != expected_block_hash:
        raise ValueError(
            f"Raw Step 4 block changed for {source_file} {reference}: "
            f"{actual_hash} != {expected_block_hash}"
        )
    if len(parsed_rows) != end_line - start_line + 1:
        raise ValueError(f"Raw row count mismatch at {source_file} {reference}")
    if any(row["step"] != step for row in parsed_rows):
        raise ValueError(f"Unexpected Step values at {source_file} {reference}")
    for field in ("testtime", "steptime", "amphr", "amps", "volts"):
        if any(row[field] is None for row in parsed_rows):
            raise ValueError(f"Missing {field} values at {source_file} {reference}")

    def integrate(axis: str) -> tuple[float, int, int]:
        area = 0.0
        zero = 0
        negative = 0
        for before, after in zip(parsed_rows, parsed_rows[1:]):
            dt = float(after[axis]) - float(before[axis])
            if dt == 0:
                zero += 1
            elif dt < 0:
                negative += 1
            else:
                area += 0.5 * (abs(float(before["amps"])) + abs(float(after["amps"]))) * dt
        return area, zero, negative

    q_test, test_zero, test_negative = integrate("testtime")
    q_step, step_zero, step_negative = integrate("steptime")
    capacities = [float(row["amphr"]) for row in parsed_rows]
    currents = [float(row["amps"]) for row in parsed_rows]
    voltages = [float(row["volts"]) for row in parsed_rows]
    dt_test = [
        float(after["testtime"]) - float(before["testtime"])
        for before, after in zip(parsed_rows, parsed_rows[1:])
    ]
    dt_step = [
        float(after["steptime"]) - float(before["steptime"])
        for before, after in zip(parsed_rows, parsed_rows[1:])
    ]
    test_interval_summary = _positive_interval_summary(dt_test)
    step_interval_summary = _positive_interval_summary(dt_step)
    return {
        "source_file": source_file,
        "source_file_sha256": source_sha,
        "part": part,
        "step": step,
        "row_count": len(parsed_rows),
        "block_sha256": actual_hash,
        "next_row_step": next_step,
        "next_row_transitions_step": next_step is not None and next_step != step,
        "q_testtime": q_test,
        "q_steptime": q_step,
        "amphr_span_native": max(capacities) - min(capacities),
        "testtime_zero_intervals": test_zero,
        "testtime_negative_intervals": test_negative,
        "steptime_zero_intervals": step_zero,
        "steptime_negative_intervals": step_negative,
        "testtime_nonpositive_intervals": sum(value <= 0 for value in dt_test),
        "steptime_nonpositive_intervals": sum(value <= 0 for value in dt_step),
        "testtime_positive_interval_min": test_interval_summary["minimum"],
        "testtime_positive_interval_median": test_interval_summary["median"],
        "testtime_positive_interval_max": test_interval_summary["maximum"],
        "testtime_max_interval_to_median": test_interval_summary["maximum_to_median"],
        "steptime_positive_interval_min": step_interval_summary["minimum"],
        "steptime_positive_interval_median": step_interval_summary["median"],
        "steptime_positive_interval_max": step_interval_summary["maximum"],
        "steptime_max_interval_to_median": step_interval_summary["maximum_to_median"],
        "current_min": min(currents),
        "current_max": max(currents),
        "current_mean": statistics.fmean(currents),
        "voltage_first": voltages[0],
        "voltage_last": voltages[-1],
        "voltage_min": min(voltages),
        "voltage_max": max(voltages),
        "raw_rows": raw_rows,
        "parsed_rows": parsed_rows,
        "exact_duplicate_pairs": sum(
            raw_rows[i - 1] == raw_rows[i] for i in range(1, len(raw_rows))
        ),
        "testtime_duplicate_pairs": test_zero,
        "steptime_duplicate_pairs": step_zero,
    }


def _mad_scores(values: list[float]) -> tuple[list[float | None], dict[str, Any]]:
    if len(values) < ROBUST_MIN_SAMPLE_SIZE:
        return [None] * len(values), {
            "n": len(values),
            "center": statistics.median(values) if values else None,
            "mad": None,
            "status": "insufficient_sample_size",
        }
    center = statistics.median(values)
    mad = statistics.median(abs(value - center) for value in values)
    if mad == 0:
        return [None] * len(values), {
            "n": len(values),
            "center": center,
            "mad": 0.0,
            "status": "zero_mad_unscorable",
        }
    scale = 1.4826 * mad
    return [(value - center) / scale for value in values], {
        "n": len(values),
        "center": center,
        "mad": mad,
        "status": "scored",
    }


def _time_axis_scores(values: list[float]) -> tuple[list[float | None], dict[str, Any]]:
    if max((abs(value) for value in values), default=0.0) <= NUMERIC_LABEL_TOLERANCE_PP:
        return [None] * len(values), {
            "n": len(values),
            "center": statistics.median(values) if values else None,
            "mad": statistics.median(abs(value - statistics.median(values)) for value in values)
            if values
            else None,
            "status": "below_numerical_comparison_resolution",
        }
    return _mad_scores(values)


def _positive_interval_summary(intervals: list[float]) -> dict[str, float | None]:
    positive = [value for value in intervals if value > 0]
    if not positive:
        return {"minimum": None, "median": None, "maximum": None, "maximum_to_median": None}
    minimum = min(positive)
    median = statistics.median(positive)
    maximum = max(positive)
    return {
        "minimum": minimum,
        "median": median,
        "maximum": maximum,
        "maximum_to_median": maximum / median if median else None,
    }


def _theil_sen(xs: list[float], ys: list[float]) -> tuple[float, float]:
    slopes = [
        (ys[j] - ys[i]) / (xs[j] - xs[i])
        for i in range(len(xs))
        for j in range(i + 1, len(xs))
        if xs[j] != xs[i]
    ]
    if not slopes:
        raise ValueError("Theil-Sen trend requires at least two distinct RPT ordinals.")
    slope = statistics.median(slopes)
    intercept = statistics.median(y - slope * x for x, y in zip(xs, ys))
    return slope, intercept


def _extreme_text(values: list[float], maximum: bool) -> str:
    if not values:
        return "none"
    return f"{(max(values) if maximum else min(values)):.6f}"


def _trajectory_diagnostics(
    labels: list[dict[str, Any]],
    block_audit: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    block_by_key: dict[tuple[str, int, int, int, int], dict[str, str]] = block_audit
    for row in labels:
        cell_match = _CELL_RE.fullmatch(row["physical_cell_id"])
        if cell_match is None:
            raise ValueError(f"Malformed physical-cell identifier: {row['physical_cell_id']}")
        part = int(row["dataset_part"])
        rpt_ref = row["rpt_step4_segment_ref"]
        bol_ref = row["bol_step4_segment_ref"]
        rpt_source = row["rpt_step4_source_file"]
        bol_source = row["bol_step4_source_file"]
        rpt_key = _block_key(rpt_source, part, rpt_ref)
        bol_part = int(row["boL_source_part"])
        bol_key = _block_key(bol_source, bol_part, bol_ref)
        rpt_block = block_by_key.get(rpt_key)
        bol_block = block_by_key.get(bol_key)
        if rpt_block is None or bol_block is None:
            raise ValueError(
                f"Independent Step 4 audit does not contain both blocks for "
                f"{row['physical_cell_id']} RPT{row['rpt_index']}"
            )
        for role, block, expected_hash in (
            ("RPT", rpt_block, row["rpt_step4_block_sha256"]),
            ("BoL", bol_block, row["bol_step4_block_sha256"]),
        ):
            if (
                block["block_sha256_independent"] != expected_hash
                or block["row_count_exact"].casefold() != "true"
                or block["block_hash_exact"].casefold() != "true"
                or block["step_ids_exact"].casefold() != "true"
            ):
                raise ValueError(f"{role} Step 4 block audit mismatch for {row['physical_cell_id']} RPT{row['rpt_index']}")
        row = dict(row)
        row["rpt_block_q_testtime"] = _finite(
            _number(rpt_block["testtime_integrated_abs_current_native"]), "RPT block capacity"
        )
        row["rpt_block_q_steptime"] = _finite(
            _number(rpt_block["steptime_integrated_abs_current_native"]), "RPT StepTime capacity"
        )
        row["rpt_block_amphr_span"] = _finite(_number(rpt_block["amphr_span_native"]), "RPT Amphr span")
        row["bol_block_q_testtime"] = _finite(
            _number(bol_block["testtime_integrated_abs_current_native"]), "BoL block capacity"
        )
        row["bol_block_q_steptime"] = _finite(
            _number(bol_block["steptime_integrated_abs_current_native"]), "BoL StepTime capacity"
        )
        row["bol_block_amphr_span"] = _finite(_number(bol_block["amphr_span_native"]), "BoL Amphr span")
        row["soh_testtime_independent_audit"] = _ratio_percent(
            row["rpt_block_q_testtime"], row["bol_block_q_testtime"]
        )
        row["soh_steptime_independent_audit"] = _ratio_percent(
            row["rpt_block_q_steptime"], row["bol_block_q_steptime"]
        )
        row["soh_native_independent_audit"] = _ratio_percent(
            row["rpt_block_amphr_span"], row["bol_block_amphr_span"]
        )
        row["audit_vs_label_diff_pp"] = row["soh_testtime_independent_audit"] - row["soh_percent"]
        row["time_axis_diff_pp"] = (
            row["soh_steptime_independent_audit"] - row["soh_testtime_independent_audit"]
        )
        row["native_counter_diff_pp"] = (
            row["soh_native_independent_audit"] - row["soh_testtime_independent_audit"]
        )
        row["rpt_raw_block_status"] = rpt_block["raw_block_status"]
        row["bol_raw_block_status"] = bol_block["raw_block_status"]
        row["rpt_testtime_zero_intervals"] = int(rpt_block["testtime_zero_intervals"])
        row["rpt_steptime_zero_intervals"] = int(rpt_block["steptime_zero_intervals"])
        row["rpt_testtime_negative_intervals"] = int(rpt_block["testtime_negative_intervals"])
        row["rpt_steptime_negative_intervals"] = int(rpt_block["steptime_negative_intervals"])
        row["rpt_voltage_last"] = _number(rpt_block["voltage_last"])
        row["rpt_voltage_endpoint_offset_2_5v"] = (
            row["rpt_voltage_last"] - 2.5 if row["rpt_voltage_last"] is not None else None
        )
        row["rpt_current_magnitude_mean"] = _number(
            rpt_block["independent_physics_mean_absolute_discharge_current_native"]
        )
        row["rpt_block_row_count"] = int(rpt_block["row_count"])
        row["rpt_block_missing_signal_count"] = sum(
            int(rpt_block[field])
            for field in (
                "missing_testtime",
                "missing_steptime",
                "missing_amphr",
                "missing_amps",
                "missing_volts",
                "wrong_width_rows",
            )
        )
        row["direct_raw_reopened"] = False
        row["direct_raw_status"] = "not_required_unflagged"
        row["direct_raw_rpt_source_sha256"] = ""
        row["direct_raw_bol_source_sha256"] = ""
        row["direct_raw_rpt_block_sha256"] = ""
        row["direct_raw_bol_block_sha256"] = ""
        row["direct_raw_soh_testtime_percent"] = None
        row["direct_raw_soh_steptime_percent"] = None
        row["direct_raw_soh_native_percent"] = None
        row["direct_raw_duplicate_pairs"] = 0
        row["direct_raw_duplicate_collapse_delta_testtime"] = None
        row["direct_raw_duplicate_collapse_delta_steptime"] = None
        row["proposed_corrected_soh_percent"] = None
        row["classification"] = "VERIFIED"
        row["statistical_flags"] = []
        groups[row["physical_cell_id"]].append(row)

    summaries: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    per_part_endpoint: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observations in groups.values():
        observations.sort(key=lambda row: (row["rpt_index"], row["dataset_part"]))
        xs = [float(row["rpt_index"]) for row in observations]
        ys = [float(row["soh_percent"]) for row in observations]
        slope, intercept = _theil_sen(xs, ys)
        trend_residuals = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
        local_residuals: list[float] = []
        local_indexes: list[int] = []
        for i, row in enumerate(observations):
            row["delta_previous_observed_rpt_pp"] = None
            row["previous_rpt_index"] = None
            row["rpt_gap_from_previous_observation"] = None
            row["local_neighbor_expected_soh_percent"] = None
            row["local_neighbor_deviation_pp"] = None
            row["theil_sen_slope_pp_per_rpt_ordinal"] = slope
            row["theil_sen_intercept_percent"] = intercept
            row["theil_sen_trend_residual_pp"] = trend_residuals[i]
            row["robust_z_delta"] = None
            row["robust_z_local"] = None
            row["robust_z_trend"] = None
            row["robust_z_native_counter"] = None
            row["robust_z_endpoint"] = None
            row["robust_z_current_profile"] = None
            row["robust_z_time_axis"] = None
            row["rpt_gap_skipped_ordinals"] = 0
            if i:
                previous = observations[i - 1]
                gap = int(row["rpt_index"]) - int(previous["rpt_index"])
                if gap <= 0:
                    raise ValueError(f"Non-increasing RPT chronology within {row['physical_cell_id']}")
                row["previous_rpt_index"] = previous["rpt_index"]
                row["rpt_gap_from_previous_observation"] = gap
                row["rpt_gap_skipped_ordinals"] = max(0, gap - 1)
                row["delta_previous_observed_rpt_pp"] = float(row["soh_percent"]) - float(
                    previous["soh_percent"]
                )
            if 0 < i < len(observations) - 1:
                before, after = observations[i - 1], observations[i + 1]
                x0, x1 = float(before["rpt_index"]), float(after["rpt_index"])
                expected = float(before["soh_percent"]) + (
                    (float(after["soh_percent"]) - float(before["soh_percent"]))
                    * (xs[i] - x0)
                    / (x1 - x0)
                )
                row["local_neighbor_expected_soh_percent"] = expected
                row["local_neighbor_deviation_pp"] = float(row["soh_percent"]) - expected
                local_residuals.append(row["local_neighbor_deviation_pp"])
                local_indexes.append(i)

        delta_indexes = [i for i, row in enumerate(observations) if row["delta_previous_observed_rpt_pp"] is not None]
        delta_values = [float(observations[i]["delta_previous_observed_rpt_pp"]) for i in delta_indexes]
        delta_scores, delta_meta = _mad_scores(delta_values)
        for i, score in zip(delta_indexes, delta_scores):
            observations[i]["robust_z_delta"] = score
        local_scores, local_meta = _mad_scores(local_residuals)
        for i, score in zip(local_indexes, local_scores):
            observations[i]["robust_z_local"] = score
        trend_scores, trend_meta = _mad_scores(trend_residuals)
        for row, score in zip(observations, trend_scores):
            row["robust_z_trend"] = score
        counter_values = [float(row["native_counter_diff_pp"]) for row in observations]
        counter_scores, counter_meta = _mad_scores(counter_values)
        for row, score in zip(observations, counter_scores):
            row["robust_z_native_counter"] = score
        axis_values = [float(row["time_axis_diff_pp"]) for row in observations]
        axis_scores, axis_meta = _time_axis_scores(axis_values)
        for row, score in zip(observations, axis_scores):
            row["robust_z_time_axis"] = score
        current_values = [
            float(row["rpt_current_magnitude_mean"])
            for row in observations
            if row["rpt_current_magnitude_mean"] is not None
        ]
        current_scores, current_meta = _mad_scores(current_values)
        current_score_index = 0
        for row in observations:
            if row["rpt_current_magnitude_mean"] is not None:
                row["robust_z_current_profile"] = current_scores[current_score_index]
                current_score_index += 1
        for row in observations:
            if row["rpt_voltage_endpoint_offset_2_5v"] is not None:
                per_part_endpoint[str(row["dataset_part"])].append(row)

        deltas = [float(row["delta_previous_observed_rpt_pp"]) for row in observations if row["delta_previous_observed_rpt_pp"] is not None]
        increases = [value for value in deltas if value > 0]
        decreases = [value for value in deltas if value < 0]
        skipped = sum(int(row["rpt_gap_skipped_ordinals"]) for row in observations)
        summaries.append(
            {
                "physical_cell_id": observations[0]["physical_cell_id"],
                "partition": observations[0]["partition"],
                "dataset_parts": ";".join(sorted({str(row["dataset_part"]) for row in observations}, key=int)),
                "observation_count": len(observations),
                "first_rpt_index": observations[0]["rpt_index"],
                "last_rpt_index": observations[-1]["rpt_index"],
                "rpt_ordinal_gaps_between_observations": skipped,
                "first_soh_percent": ys[0],
                "last_soh_percent": ys[-1],
                "total_soh_change_pp": ys[-1] - ys[0],
                "increase_transition_count": len(increases),
                "decrease_transition_count": len(decreases),
                "unchanged_transition_count": len(deltas) - len(increases) - len(decreases),
                "largest_positive_jump_pp": max(increases) if increases else None,
                "largest_negative_jump_pp": min(decreases) if decreases else None,
                "max_abs_local_neighbor_deviation_pp": max(
                    (abs(float(row["local_neighbor_deviation_pp"])) for row in observations if row["local_neighbor_deviation_pp"] is not None),
                    default=None,
                ),
                "theil_sen_trend_pp_per_rpt_ordinal": slope,
                "trend_direction": "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat",
                "theil_sen_residual_robust_score_status": trend_meta["status"],
                "difference_robust_score_status": delta_meta["status"],
                "local_robust_score_status": local_meta["status"],
                "native_counter_robust_score_status": counter_meta["status"],
                "time_axis_robust_score_status": axis_meta["status"],
                "current_profile_robust_score_status": current_meta["status"],
                "promoted_record_count": sum(row["promotion_status"] == "APPROVED_AND_PROMOTED" for row in observations),
                "rpt_order_is_ordinal_not_exact_age": True,
            }
        )
        trajectory.extend(observations)

    endpoint_meta: dict[str, dict[str, Any]] = {}
    for part, observations in per_part_endpoint.items():
        values = [float(row["rpt_voltage_endpoint_offset_2_5v"]) for row in observations]
        scores, meta = _mad_scores(values)
        endpoint_meta[part] = meta
        for row, score in zip(observations, scores):
            row["robust_z_endpoint"] = score

    for row in trajectory:
        flags: list[str] = []
        for field, name in (
            ("robust_z_delta", "consecutive_delta_modified_z_gt_3_5"),
            ("robust_z_local", "local_neighbor_modified_z_gt_3_5"),
            ("robust_z_trend", "theil_sen_residual_modified_z_gt_3_5"),
            ("robust_z_native_counter", "native_counter_discrepancy_modified_z_gt_3_5"),
            ("robust_z_time_axis", "testtime_steptime_difference_modified_z_gt_3_5"),
            ("robust_z_current_profile", "current_profile_modified_z_gt_3_5"),
            ("robust_z_endpoint", "voltage_endpoint_offset_modified_z_gt_3_5"),
        ):
            score = row[field]
            if score is not None and abs(float(score)) > ROBUST_Z_THRESHOLD:
                flags.append(name)
        if abs(float(row["audit_vs_label_diff_pp"])) > NUMERIC_LABEL_TOLERANCE_PP:
            flags.append("step4_capacity_ratio_mismatch")
        if abs(float(row["time_axis_diff_pp"])) > NUMERIC_LABEL_TOLERANCE_PP:
            flags.append("testtime_steptime_ratio_difference")
        if row["rpt_raw_block_status"] != "ACCEPTED" or row["bol_raw_block_status"] != "ACCEPTED":
            flags.append("independent_raw_block_audit_not_accepted")
        if row["rpt_block_missing_signal_count"] > 0:
            flags.append("step4_missing_or_malformed_raw_rows")
        if (
            row["rpt_testtime_zero_intervals"]
            or row["rpt_steptime_zero_intervals"]
            or row["rpt_testtime_negative_intervals"]
            or row["rpt_steptime_negative_intervals"]
        ):
            flags.append("step4_elapsed_time_anomaly")
        row["statistical_flags"] = flags
        row["statistically_flagged"] = bool(flags)
        row["flagged_by"] = ";".join(flags)
        if flags:
            row["direct_raw_reopened"] = True
            row["direct_raw_status"] = "pending"
            row["classification"] = "UNRESOLVED_ANOMALY"
    return trajectory, summaries, {"endpoint_score_status_by_part": endpoint_meta}


def _investigate_flagged_observations(
    trajectory: list[dict[str, Any]],
    file_hash_cache: dict[str, str],
) -> tuple[int, int]:
    flagged = [row for row in trajectory if row["statistically_flagged"]]
    corrected_count = 0
    unresolved_count = 0
    for row in flagged:
        try:
            rpt = _independent_scan_block(
                row["rpt_step4_source_file"],
                int(row["dataset_part"]),
                row["rpt_step4_segment_ref"],
                row["rpt_step4_block_sha256"],
                file_hash_cache,
            )
            bol = _independent_scan_block(
                row["bol_step4_source_file"],
                int(row["boL_source_part"]),
                row["bol_step4_segment_ref"],
                row["bol_step4_block_sha256"],
                file_hash_cache,
            )
        except (ValueError, FileNotFoundError) as error:
            row["direct_raw_status"] = f"unresolved:{error}"
            row["classification"] = "UNRESOLVED_ANOMALY"
            unresolved_count += 1
            continue

        cell_match = _CELL_RE.fullmatch(row["physical_cell_id"])
        cell_number = cell_match.group("cell") if cell_match else ""
        cell_token = re.compile(rf"cell\s*{re.escape(cell_number)}(?:\D|$)", re.I)
        same_cell_sources = bool(
            cell_token.search(rpt["source_file"]) and cell_token.search(bol["source_file"])
        )
        direct_soh = _ratio_percent(rpt["q_testtime"], bol["q_testtime"])
        direct_step_soh = _ratio_percent(rpt["q_steptime"], bol["q_steptime"])
        native_soh = _ratio_percent(rpt["amphr_span_native"], bol["amphr_span_native"])
        row["direct_raw_source_hashes"] = json.dumps(
            {
                "rpt": rpt["source_file_sha256"],
                "bol": bol["source_file_sha256"],
            },
            sort_keys=True,
        )
        row["direct_raw_rpt_source_sha256"] = rpt["source_file_sha256"]
        row["direct_raw_bol_source_sha256"] = bol["source_file_sha256"]
        row["direct_raw_rpt_block_sha256"] = rpt["block_sha256"]
        row["direct_raw_bol_block_sha256"] = bol["block_sha256"]
        row["direct_raw_rpt_row_count"] = rpt["row_count"]
        row["direct_raw_bol_row_count"] = bol["row_count"]
        row["direct_raw_soh_testtime_percent"] = direct_soh
        row["direct_raw_soh_steptime_percent"] = direct_step_soh
        row["direct_raw_soh_native_percent"] = native_soh
        row["direct_raw_testtime_steptime_difference_pp"] = direct_step_soh - direct_soh
        row["direct_raw_native_minus_integrated_pp"] = native_soh - direct_soh
        row["direct_raw_rpt_testtime_zero_intervals"] = rpt["testtime_zero_intervals"]
        row["direct_raw_rpt_steptime_zero_intervals"] = rpt["steptime_zero_intervals"]
        row["direct_raw_bol_testtime_zero_intervals"] = bol["testtime_zero_intervals"]
        row["direct_raw_bol_steptime_zero_intervals"] = bol["steptime_zero_intervals"]
        row["direct_raw_duplicate_pairs"] = rpt["exact_duplicate_pairs"]
        row["direct_raw_bol_duplicate_pairs"] = bol["exact_duplicate_pairs"]
        row["direct_raw_rpt_voltage_last"] = rpt["voltage_last"]
        row["direct_raw_bol_voltage_last"] = bol["voltage_last"]
        row["direct_raw_rpt_current_min"] = rpt["current_min"]
        row["direct_raw_rpt_current_max"] = rpt["current_max"]
        row["direct_raw_bol_current_min"] = bol["current_min"]
        row["direct_raw_bol_current_max"] = bol["current_max"]
        for role, block in (("rpt", rpt), ("bol", bol)):
            for axis in ("testtime", "steptime"):
                for statistic in ("min", "median", "max", "max_to_median"):
                    raw_name = (
                        f"{axis}_positive_interval_{statistic}"
                        if statistic != "max_to_median"
                        else f"{axis}_max_interval_to_median"
                    )
                    row[f"direct_raw_{role}_{raw_name}"] = block[raw_name]
        row["direct_raw_interval_interpretation"] = (
            "Observed adjacent positive-interval min/median/max and max-to-median ratios are descriptive only; "
            "an expected tester sampling cadence was not established."
        )
        row["direct_raw_source_identity_verified"] = same_cell_sources
        all_times_monotone = not (
            rpt["testtime_negative_intervals"]
            or rpt["steptime_negative_intervals"]
            or bol["testtime_negative_intervals"]
            or bol["steptime_negative_intervals"]
        )
        discharge_direction = rpt["current_max"] < 0 and bol["current_max"] < 0
        boundary_closed = rpt["next_row_transitions_step"] and bol["next_row_transitions_step"]
        formula_matches = abs(direct_soh - float(row["soh_percent"])) <= NUMERIC_LABEL_TOLERANCE_PP
        exact_repeat_stable = True
        for block, axis in ((rpt, "testtime"), (rpt, "steptime"), (bol, "testtime"), (bol, "steptime")):
            original = block["q_testtime"] if axis == "testtime" else block["q_steptime"]
            zero_intervals = (
                block["testtime_zero_intervals"]
                if axis == "testtime"
                else block["steptime_zero_intervals"]
            )
            if zero_intervals != block["exact_duplicate_pairs"]:
                exact_repeat_stable = False
            reduced_rows = [
                values
                for index, values in enumerate(block["parsed_rows"])
                if index == 0 or block["raw_rows"][index] != block["raw_rows"][index - 1]
            ]
            reduced_integral = 0.0
            for before, after in zip(reduced_rows, reduced_rows[1:]):
                dt = float(after[axis]) - float(before[axis])
                if dt > 0:
                    reduced_integral += 0.5 * (
                        abs(float(before["amps"])) + abs(float(after["amps"]))
                    ) * dt
            if block["exact_duplicate_pairs"] and not math.isclose(
                original, reduced_integral, rel_tol=1e-12, abs_tol=1e-9
            ):
                exact_repeat_stable = False
        row["direct_raw_duplicate_collapse_delta_testtime"] = (
            rpt["q_testtime"]
            - _integral_after_duplicate_collapse(rpt["parsed_rows"], rpt["raw_rows"], "testtime")
        )
        row["direct_raw_duplicate_collapse_delta_steptime"] = (
            rpt["q_steptime"]
            - _integral_after_duplicate_collapse(rpt["parsed_rows"], rpt["raw_rows"], "steptime")
        )
        row["direct_raw_step4_boundaries_verified"] = boundary_closed
        row["direct_raw_discharge_current_verified"] = discharge_direction
        row["direct_raw_elapsed_times_monotone"] = all_times_monotone
        row["direct_raw_ratio_matches_final_label"] = formula_matches
        row["proposed_corrected_soh_percent"] = direct_soh if not formula_matches else None
        unresolved = (
            not same_cell_sources
            or not all_times_monotone
            or not discharge_direction
            or not boundary_closed
            or not exact_repeat_stable
        )
        if not formula_matches and not unresolved:
            row["classification"] = "CONFIRMED_LABEL_ERROR"
            corrected_count += 1
        elif unresolved:
            row["classification"] = "UNRESOLVED_ANOMALY"
            unresolved_count += 1
        elif any(
            flag.startswith(
                (
                    "consecutive_delta_",
                    "local_neighbor_",
                    "theil_sen_residual_",
                )
            )
            for flag in row["statistical_flags"]
        ):
            row["classification"] = "PLAUSIBLE_VARIATION"
        else:
            row["classification"] = "VERIFIED"
        row["direct_raw_status"] = (
            "raw_reopened_hashes_boundaries_identity_and_independent_ratios_checked"
        )
    return corrected_count, unresolved_count


def _integral_after_duplicate_collapse(
    rows: list[dict[str, float | None]], raw_rows: list[bytes], axis: str
) -> float:
    reduced = [
        values
        for index, values in enumerate(rows)
        if index == 0 or raw_rows[index] != raw_rows[index - 1]
    ]
    total = 0.0
    for before, after in zip(reduced, reduced[1:]):
        dt = float(after[axis]) - float(before[axis])
        if dt > 0:
            total += 0.5 * (abs(float(before["amps"])) + abs(float(after["amps"]))) * dt
    return total


def _write_plots(output_dir: Path, trajectory: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "bafuse_soh_trajectory_plots"
    plot_dir.mkdir()
    saved: list[str] = []
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_part: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectory:
        by_cell[row["physical_cell_id"]].append(row)
        by_part[str(row["dataset_part"])].append(row)

    for cell, rows in sorted(by_cell.items()):
        rows.sort(key=lambda item: int(item["rpt_index"]))
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        ax, delta_ax = axes
        xs = [int(row["rpt_index"]) for row in rows]
        ys = [float(row["soh_percent"]) for row in rows]
        ax.plot(xs, ys, color="0.55", linewidth=1.2, zorder=1)
        for before, after in zip(rows, rows[1:]):
            change = float(after["soh_percent"]) - float(before["soh_percent"])
            if change > 0:
                ax.plot(
                    [before["rpt_index"], after["rpt_index"]],
                    [before["soh_percent"], after["soh_percent"]],
                    color="darkorange",
                    linewidth=2.2,
                    zorder=2,
                )
        for row in rows:
            flagged = bool(row["statistically_flagged"])
            promoted = row["promotion_status"] == "APPROVED_AND_PROMOTED"
            ax.scatter(
                row["rpt_index"],
                row["soh_percent"],
                s=66 if flagged or promoted else 34,
                c="crimson" if flagged else "tab:blue",
                edgecolors="darkorange" if promoted else "black",
                linewidths=1.6 if promoted else 0.4,
                zorder=3,
            )
            ax.annotate(f"R{row['rpt_index']}", (row["rpt_index"], row["soh_percent"]), xytext=(3, 4), textcoords="offset points", fontsize=7)
        deltas = [row["delta_previous_observed_rpt_pp"] for row in rows]
        delta_x = [row["rpt_index"] for row in rows]
        for x, delta, row in zip(delta_x, deltas, rows):
            if delta is None:
                continue
            delta_ax.bar(
                x,
                delta,
                color="crimson" if row["statistically_flagged"] else "darkorange" if delta > 0 else "steelblue",
                width=0.72,
            )
        ax.set_ylabel("Step 4 capacity-retention SoH (%)")
        ax.set_title(f"{cell} | split={rows[0]['partition']} | RPT ordinal, not exact lifetime")
        ax.grid(True, alpha=0.25)
        delta_ax.axhline(0, color="black", linewidth=0.8)
        delta_ax.set_ylabel("Change (pp)")
        delta_ax.set_xlabel("Guide-mapped RPT ordinal")
        delta_ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = plot_dir / f"{cell}_trajectory.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path.relative_to(output_dir)))

    colors = plt.get_cmap("tab20")
    for part, rows in sorted(by_part.items(), key=lambda item: int(item[0])):
        fig, ax = plt.subplots(figsize=(12, 7))
        cells = sorted({row["physical_cell_id"] for row in rows})
        for index, cell in enumerate(cells):
            series = sorted(
                (row for row in rows if row["physical_cell_id"] == cell),
                key=lambda item: int(item["rpt_index"]),
            )
            ax.plot(
                [row["rpt_index"] for row in series],
                [row["soh_percent"] for row in series],
                marker="o",
                markersize=3.5,
                linewidth=1.1,
                color=colors(index % 20),
                label=cell,
            )
            for row in series:
                if row["promotion_status"] == "APPROVED_AND_PROMOTED":
                    ax.scatter(row["rpt_index"], row["soh_percent"], s=72, facecolors="none", edgecolors="darkorange", linewidths=1.8, zorder=4)
                if row["statistically_flagged"]:
                    ax.scatter(row["rpt_index"], row["soh_percent"], s=80, marker="x", color="crimson", linewidths=1.8, zorder=5)
        ax.set_title(f"Part {part} Step 4 capacity-retention trajectories (RPT ordinal)")
        ax.set_xlabel("Guide-mapped RPT ordinal")
        ax.set_ylabel("SoH (%)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        path = plot_dir / f"part_{part}_combined_trajectories.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        saved.append(str(path.relative_to(output_dir)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    delta_ax, residual_ax = axes
    for row in trajectory:
        delta = row["delta_previous_observed_rpt_pp"]
        if delta is not None:
            delta_ax.scatter(
                row["rpt_index"],
                delta,
                marker="x" if row["statistically_flagged"] else "o",
                color="crimson" if row["statistically_flagged"] else f"C{int(row['dataset_part'])}",
                alpha=0.8,
                s=40,
            )
        local = row["local_neighbor_deviation_pp"]
        if local is not None:
            residual_ax.scatter(
                row["rpt_index"],
                local,
                marker="x" if row["statistically_flagged"] else "o",
                color="crimson" if row["statistically_flagged"] else f"C{int(row['dataset_part'])}",
                alpha=0.8,
                s=40,
            )
    delta_ax.axhline(0, color="black", linewidth=0.8)
    residual_ax.axhline(0, color="black", linewidth=0.8)
    delta_ax.set_title("Observed neighboring-RPT changes")
    delta_ax.set_xlabel("To-RPT ordinal")
    delta_ax.set_ylabel("SoH change (pp)")
    residual_ax.set_title("Deviation from adjacent-neighbor interpolation")
    residual_ax.set_xlabel("RPT ordinal")
    residual_ax.set_ylabel("Local deviation (pp)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Robust diagnostics are flags for review, not label rejection")
    fig.tight_layout()
    path = plot_dir / "outlier_diagnostics.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)
    saved.append(str(path.relative_to(output_dir)))
    return saved


def _write_trend_report(
    path: Path,
    labels: list[dict[str, Any]],
    trajectory: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    flagged: list[dict[str, Any]],
    confirmed_errors: int,
    unresolved: int,
    report_metadata: dict[str, Any],
) -> None:
    deltas = [
        float(row["delta_previous_observed_rpt_pp"])
        for row in trajectory
        if row["delta_previous_observed_rpt_pp"] is not None
    ]
    increases = [value for value in deltas if value > 0]
    decreases = [value for value in deltas if value < 0]
    increasing_cells = sum(row["trend_direction"] == "increasing" for row in summaries)
    decreasing_cells = sum(row["trend_direction"] == "decreasing" for row in summaries)
    flags = Counter(
        flag
        for row in flagged
        for flag in row["statistical_flags"]
    )
    classifications = Counter(row["classification"] for row in flagged)
    maximum_time_axis_difference = max(
        (abs(float(row["time_axis_diff_pp"])) for row in trajectory),
        default=0.0,
    )
    train_count = sum(row["partition"] == "train" for row in labels)
    validation_count = sum(row["partition"] == "validation" for row in labels)
    external_count = sum(row["partition"] == "external_test" for row in labels)
    lines = [
        "# Longitudinal SoH trajectory quality audit",
        "",
        "`label_promotion_authorized: true`; `trajectory_audit_authorized: true`.",
        "`dataset_freeze_authorized: false`; `training_authorized: false`; "
        "`external_prediction_evaluation_authorized: false`.",
        "No training, freeze, external prediction evaluation, PyDMA, or Half-Cell fitting was run.",
        "",
        "## Cohort and interpretation",
        "",
        f"- {len(labels)} accepted Step 4 capacity-retention labels across {len(summaries)} physical cells.",
        f"- Split counts: Train {train_count}; Validation {validation_count}; "
        f"External Test / Part 3 {external_count}.",
        "- Chronology uses the Guide-mapped RPT ordinal. For Parts 1–2, the RPT index preserves the continuation; "
        "for Part 3 it is the Guide-mapped sequence. Ordinal spacing is not exact cycle count or elapsed lifetime.",
        "- No missing RPT was interpolated, and no label was smoothed, clipped, removed, or overwritten.",
        "- Target: 100 × Step 4 discharge capacity at RPT / same-cell Step 4 BoL capacity. This is a reconstructed "
        "capacity-retention reference, not official Oxford ground truth.",
        "",
        "## Trajectory summary",
        "",
        f"- Observed adjacent-record transitions: {len(deltas)}.",
        f"- Increases: {len(increases)}; decreases: {len(decreases)}; unchanged: {len(deltas)-len(increases)-len(decreases)}.",
        f"- Largest increase: {_extreme_text(increases, True)} pp; "
        f"largest decrease: {_extreme_text(decreases, False)} pp.",
        f"- Per-cell Theil–Sen slope is increasing in {increasing_cells}, decreasing in {decreasing_cells}, "
        f"and flat in {len(summaries)-increasing_cells-decreasing_cells} cells.",
        f"- Unique statistically flagged observations: {len(flagged)}.",
        f"- Flagged classifications: {dict(classifications)}.",
        f"- Confirmed label errors: {confirmed_errors}; unresolved anomalies: {unresolved}.",
        "",
        "Observed increases are descriptive, not presumed errors. The maxima include transitions spanning an RPT ordinal gap; "
        "see `bafuse_per_cell_soh_trajectory.csv` for each `rpt_gap_from_previous_observation`.",
        "",
        "## Diagnostic methods and limitations",
        "",
        f"- Modified robust z-score: `(x - median) / (1.4826 × MAD)`; potential outlier if absolute score > "
        f"{ROBUST_Z_THRESHOLD:g}. This is the Iglewicz–Hoaglin modified-z convention, not a physical threshold.",
        f"- Scores are only applied with at least {ROBUST_MIN_SAMPLE_SIZE} values in the relevant within-cell (or same-part endpoint) series. "
        "Zero MAD and small samples are marked unscorable; no fallback scale is invented.",
        "- Screens include observed adjacent-RPT changes, leave-one-out interpolation residuals, Theil–Sen trend residuals, "
        "native Amphr-vs-integrated capacity differences, and voltage-endpoint offsets by Part.",
        "- Theil–Sen slope is a robust descriptive trend per RPT ordinal; no p-values or confidence intervals are claimed. "
        "The observations are sparse, serially dependent, and have unequal RPT gaps.",
        f"- TestTime and StepTime integrations were compared for all observations. Maximum absolute SoH-ratio difference: "
        f"{maximum_time_axis_difference:.12g} pp. Differences at or below {NUMERIC_LABEL_TOLERANCE_PP:g} pp are below "
        "the numerical comparison resolution used here and are not treated as physical anomalies.",
        "- Exact formula agreement uses a numerical comparison tolerance of 1e-8 percentage points only to detect calculation "
        "mismatch; it is not an instrument tolerance or acceptance criterion.",
        "- Raw Step 4 audit metadata and hashes were checked for all observations. Original CSV blocks were reopened and independently "
        "integrated for every flagged observation; only statistical/data-quality flags prompted this targeted raw reread.",
        "- For each reopened RPT and BoL block, the outlier table includes positive TestTime/StepTime adjacent-interval "
        "minimum, median, maximum, and max-to-median ratios, plus row counts and exact duplicate counts. These are descriptive "
        "screens only: without a verified sampling cadence, an interval ratio is not called a missing sample.",
        "- Native Amphr is a QC cross-check, not the target; its instrument/export semantics and absolute scale remain unverified.",
        "- Voltage endpoints are reported relative to 2.5 V without imposing an undocumented tester tolerance. Parts 1–2 use Maccor 4200 "
        "CC-CV Step 4; Part 3 uses Maccor 4000 CC Step 4 and its README does not state chamber temperature.",
        "",
        "The 3.5 modified-z screening convention is described by Iglewicz and Hoaglin, *How to Detect and Handle Outliers* (1993). "
        "Because the per-cell series are small, serially dependent, and not identically distributed, these are investigation flags only; "
        "they do not imply rejection.",
        "",
        "## Flag counts",
        "",
        "| Diagnostic | Flagged observations |",
        "|---|---:|",
    ]
    for name, count in sorted(flags.items()):
        lines.append(f"| `{name}` | {count} |")
    if not flags:
        lines.append("| No modified-z or structural flags | 0 |")
    lines.extend(
        [
            "",
            "## Raw investigation disposition",
            "",
            "The outlier table includes all 203 observations; flagged records include source hashes, Step 4 block hashes, "
            "recomputed TestTime/StepTime/native-counter ratios, boundary/current checks, and duplicate-row sensitivity where relevant. "
            "Classification never changes `bafuse_soh_labels_203.csv`.",
            "",
            f"- Promotion rows in final table: {sum(row['promotion_status'] == 'APPROVED_AND_PROMOTED' for row in labels)}.",
            f"- Explicitly promoted labels are identified in per-cell and combined figures with orange outlines; increasing transitions are orange; "
            "statistical flags are red.",
            f"- Full recovery manifest SHA-256 reverified: `{report_metadata['recovery_manifest_sha256']}`.",
            "",
            "## Freeze/training readiness",
            "",
            "The 203-label table and longitudinal audit are complete as versioned research artifacts. The dataset is **not frozen** and "
            "training is **not authorized**. Remaining limitations include reconstructed rather than official labels, unresolved absolute "
            "time/Amphr semantics, lack of archived tester-setting records, sparse and unequal RPT spacing, and Part 3 domain shift. "
            "Any `UNRESOLVED_ANOMALY` or `CONFIRMED_LABEL_ERROR` remains a review item; no correction is automatic.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing versioned audit: {output_dir}")
    _, recovery_evidence, _, recovery_raw_hashes = _verify_approved_recovery()
    recovery_manifest_hash = _sha256(RECOVERY_DIR / "soh_recovery_sha256_manifest.csv")

    candidate_path = RECONSTRUCTION_DIR / "final_candidate_manifest.csv"
    reference_path = RECONSTRUCTION_DIR / "reconstructed_soh_references.csv"
    split_path = BENCHMARK_DIR / "cell_split.csv"
    candidates = _read_csv(candidate_path)
    references = _read_csv(reference_path)
    splits = _read_csv(split_path)
    block_audit = _load_block_audit()
    labels = _promote_labels(candidates, references, recovery_evidence, splits)

    output_dir.mkdir(parents=True)
    label_path = output_dir / "bafuse_soh_labels_203.csv"
    _write_csv(label_path, labels)
    promotion_path = output_dir / "bafuse_label_promotion_report.md"
    _write_promotion_report(promotion_path, labels, recovery_manifest_hash)

    trajectory, summaries, score_meta = _trajectory_diagnostics(labels, block_audit)
    raw_hash_cache: dict[str, str] = dict(recovery_raw_hashes)
    confirmed_errors, unresolved = _investigate_flagged_observations(trajectory, raw_hash_cache)
    _write_csv(output_dir / "bafuse_per_cell_soh_trajectory.csv", trajectory)
    _write_csv(output_dir / "bafuse_soh_monotonicity_summary.csv", summaries)
    _write_csv(output_dir / "bafuse_soh_outlier_investigation.csv", trajectory)

    plots = _write_plots(output_dir, trajectory)
    label_freeze_config = {
        "label_promotion_authorized": True,
        "trajectory_audit_authorized": True,
        "dataset_freeze_authorized": False,
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "target": "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)",
        "chronology": "Guide-mapped RPT ordinal; no interpolated observations or exact lifetime claim.",
        "modified_z_threshold": ROBUST_Z_THRESHOLD,
        "minimum_modified_z_series_size": ROBUST_MIN_SAMPLE_SIZE,
        "numeric_label_comparison_tolerance_pp": NUMERIC_LABEL_TOLERANCE_PP,
        "robust_score_metadata": score_meta,
        "recovery_manifest_sha256": recovery_manifest_hash,
        "official_label_ground_truth": False,
    }
    config_path = output_dir / "bafuse_soh_audit_config.json"
    config_path.write_text(json.dumps(label_freeze_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    flagged_rows = [row for row in trajectory if row["statistically_flagged"]]
    trend_report_path = output_dir / "bafuse_soh_trend_quality_report.md"
    _write_trend_report(
        trend_report_path,
        labels,
        trajectory,
        summaries,
        flagged_rows,
        confirmed_errors,
        unresolved,
        {"recovery_manifest_sha256": recovery_manifest_hash},
    )

    input_files = [
        candidate_path,
        reference_path,
        split_path,
        INDEPENDENT_DIR / "independent_block_audit.csv",
        INDEPENDENT_DIR / "independent_record_audit.csv",
        RECOVERY_DIR / "soh_recovery_sha256_manifest.csv",
        RECOVERY_DIR / "soh_review_recovery_evidence.csv",
        RECOVERY_DIR / "soh_review_promotion_proposal.csv",
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests" / "test_longitudinal_soh_audit.py",
    ]
    input_hashes = {str(path.resolve()): _sha256(path) for path in input_files}
    output_files = [
        label_path,
        promotion_path,
        output_dir / "bafuse_per_cell_soh_trajectory.csv",
        output_dir / "bafuse_soh_monotonicity_summary.csv",
        output_dir / "bafuse_soh_outlier_investigation.csv",
        trend_report_path,
        config_path,
        *(output_dir / plot for plot in plots),
    ]
    output_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
        for path in output_files
    }
    checksums = {
        "schema_version": 1,
        "created_date": "2026-11-14",
        "label_promotion_authorized": True,
        "trajectory_audit_authorized": True,
        "dataset_freeze_authorized": False,
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "recovery_manifest_sha256": recovery_manifest_hash,
        "inputs_sha256": input_hashes,
        "raw_source_sha256": raw_hash_cache,
        "outputs_sha256": output_hashes,
        "checksum_file_self_hash_excluded": True,
    }
    checksum_path = output_dir / "bafuse_soh_final_checksums.json"
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    transitions = [
        float(row["delta_previous_observed_rpt_pp"])
        for row in trajectory
        if row["delta_previous_observed_rpt_pp"] is not None
    ]
    return {
        "output_dir": str(output_dir),
        "labels": len(labels),
        "physical_cells": len(summaries),
        "promoted": sum(row["promotion_status"] == "APPROVED_AND_PROMOTED" for row in labels),
        "increases": sum(value > 0 for value in transitions),
        "decreases": sum(value < 0 for value in transitions),
        "maximum_increase_pp": max((value for value in transitions if value > 0), default=None),
        "maximum_decrease_pp": min((value for value in transitions if value < 0), default=None),
        "statistically_flagged_observations": len(flagged_rows),
        "confirmed_label_errors": confirmed_errors,
        "unresolved_anomalies": unresolved,
        "training_authorized": False,
        "dataset_freeze_authorized": False,
    }


def finalize_incomplete_audit(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    required = [
        "bafuse_soh_labels_203.csv",
        "bafuse_label_promotion_report.md",
        "bafuse_per_cell_soh_trajectory.csv",
        "bafuse_soh_monotonicity_summary.csv",
        "bafuse_soh_outlier_investigation.csv",
        "bafuse_soh_audit_config.json",
    ]
    if not output_dir.is_dir() or any(not (output_dir / name).is_file() for name in required):
        raise FileNotFoundError("Cannot resume: required intermediate audit outputs are missing.")
    trend_report_path = output_dir / "bafuse_soh_trend_quality_report.md"
    checksum_path = output_dir / "bafuse_soh_final_checksums.json"
    if trend_report_path.exists() or checksum_path.exists():
        raise FileExistsError("Resume is only allowed before the final report/checksum exists.")

    _, _, _, raw_hashes = _verify_approved_recovery()
    labels = _read_csv(output_dir / "bafuse_soh_labels_203.csv")
    trajectory = _read_csv(output_dir / "bafuse_per_cell_soh_trajectory.csv")
    summaries = _read_csv(output_dir / "bafuse_soh_monotonicity_summary.csv")
    if len(labels) != 203 or len(trajectory) != 203 or len(summaries) != 23:
        raise ValueError("Incomplete outputs do not match the verified 203-label cohort.")
    if len(
        {
            (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
            for row in labels
        }
    ) != 203:
        raise ValueError("Incomplete label table does not have 203 unique observation keys.")
    if Counter(row["final_status"] for row in labels) != Counter({"ACCEPTED": 203}):
        raise ValueError("Incomplete label table contains non-ACCEPTED records.")

    def as_float(value: str) -> float | None:
        result = _number(value)
        return result

    parsed_trajectory: list[dict[str, Any]] = []
    for row in trajectory:
        item: dict[str, Any] = dict(row)
        item["dataset_part"] = int(row["dataset_part"])
        item["rpt_index"] = int(row["rpt_index"])
        item["soh_percent"] = float(row["soh_percent"])
        item["statistically_flagged"] = row["statistically_flagged"].casefold() == "true"
        flags = row.get("statistical_flags", "")
        try:
            item["statistical_flags"] = json.loads(flags) if flags else []
        except json.JSONDecodeError:
            item["statistical_flags"] = [flag for flag in flags.split(";") if flag]
        for field in ("delta_previous_observed_rpt_pp", "local_neighbor_deviation_pp"):
            item[field] = as_float(row.get(field, ""))
        parsed_trajectory.append(item)
    parsed_labels = [
        {
            "dataset_part": int(row["dataset_part"]),
            "partition": row["partition"],
            "promotion_status": row["promotion_status"],
        }
        for row in labels
    ]
    parsed_summaries = [
        {
            "trend_direction": row["trend_direction"],
        }
        for row in summaries
    ]
    flagged_rows = [row for row in parsed_trajectory if row["statistically_flagged"]]
    confirmed_errors = sum(row["classification"] == "CONFIRMED_LABEL_ERROR" for row in flagged_rows)
    unresolved = sum(row["classification"] == "UNRESOLVED_ANOMALY" for row in flagged_rows)
    for row in flagged_rows:
        for source_field, hash_field in (
            ("rpt_step4_source_file", "direct_raw_rpt_source_sha256"),
            ("bol_step4_source_file", "direct_raw_bol_source_sha256"),
        ):
            expected_hash = row.get(hash_field, "")
            if not expected_hash:
                raise ValueError(f"Missing reopened raw source hash for flagged observation {row['measurement_id']}")
            source_path = PROJECT_ROOT / Path(row[source_field].replace("/", "\\"))
            actual_hash = _sha256(source_path)
            if actual_hash != expected_hash:
                raise ValueError(f"Reopened raw source changed for flagged observation {row['measurement_id']}")
            raw_hashes[_normalize_path(row[source_field])] = actual_hash
    recovery_manifest_hash = _sha256(RECOVERY_DIR / "soh_recovery_sha256_manifest.csv")
    _write_trend_report(
        trend_report_path,
        parsed_labels,
        parsed_trajectory,
        parsed_summaries,
        flagged_rows,
        confirmed_errors,
        unresolved,
        {"recovery_manifest_sha256": recovery_manifest_hash},
    )

    input_files = [
        RECONSTRUCTION_DIR / "final_candidate_manifest.csv",
        RECONSTRUCTION_DIR / "reconstructed_soh_references.csv",
        BENCHMARK_DIR / "cell_split.csv",
        INDEPENDENT_DIR / "independent_block_audit.csv",
        INDEPENDENT_DIR / "independent_record_audit.csv",
        RECOVERY_DIR / "soh_recovery_sha256_manifest.csv",
        RECOVERY_DIR / "soh_review_recovery_evidence.csv",
        RECOVERY_DIR / "soh_review_promotion_proposal.csv",
        Path(__file__).resolve(),
        PROJECT_ROOT / "tests" / "test_longitudinal_soh_audit.py",
    ]
    input_hashes = {str(path.resolve()): _sha256(path) for path in input_files}
    plot_dir = output_dir / "bafuse_soh_trajectory_plots"
    plots = sorted(path for path in plot_dir.glob("*.png") if path.is_file())
    output_files = [
        output_dir / name
        for name in required
    ] + [trend_report_path, *plots]
    output_hashes = {
        str(path.relative_to(output_dir)).replace("\\", "/"): _sha256(path)
        for path in output_files
    }
    checksums = {
        "schema_version": 1,
        "created_date": "2026-11-14",
        "label_promotion_authorized": True,
        "trajectory_audit_authorized": True,
        "dataset_freeze_authorized": False,
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "recovery_manifest_sha256": recovery_manifest_hash,
        "inputs_sha256": input_hashes,
        "raw_source_sha256": raw_hashes,
        "outputs_sha256": output_hashes,
        "checksum_file_self_hash_excluded": True,
        "resumed_from_verified_intermediate_outputs": True,
    }
    checksum_path.write_text(json.dumps(checksums, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    transitions = [
        float(row["delta_previous_observed_rpt_pp"])
        for row in parsed_trajectory
        if row["delta_previous_observed_rpt_pp"] is not None
    ]
    return {
        "output_dir": str(output_dir),
        "labels": len(labels),
        "physical_cells": len(summaries),
        "promoted": sum(row["promotion_status"] == "APPROVED_AND_PROMOTED" for row in labels),
        "increases": sum(value > 0 for value in transitions),
        "decreases": sum(value < 0 for value in transitions),
        "maximum_increase_pp": max((value for value in transitions if value > 0), default=None),
        "maximum_decrease_pp": min((value for value in transitions if value < 0), default=None),
        "statistically_flagged_observations": len(flagged_rows),
        "confirmed_label_errors": confirmed_errors,
        "unresolved_anomalies": unresolved,
        "training_authorized": False,
        "dataset_freeze_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote approved SoH labels and audit all cell trajectories.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--resume-incomplete", action="store_true")
    arguments = parser.parse_args()
    operation = finalize_incomplete_audit if arguments.resume_incomplete else run_audit
    print(json.dumps(operation(arguments.output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
