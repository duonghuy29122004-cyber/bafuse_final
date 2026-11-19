from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "eol_coverage_v1"
AUDIT_INPUT = (
    ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "pydma_eol_audit_v1"
    / "eol_rpt_coverage_audit.csv"
)
CANONICAL_MANIFEST = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
FROZEN_CHECKPOINT = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_three_encoder_v1"
    / "fusion_checkpoint.pt"
)
SPLIT_FILE = (
    ROOT
    / "reports"
    / "bafuse_final_data_validation_v2"
    / "cell_split_12train_4val_12external.csv"
)
SEGMENT_PATTERN = re.compile(
    r"^(.*?)#Step(?P<step>\d+):segment(?P<segment>\d+)"
    r":rows(?P<start>\d+)-(?P<end>\d+)(?::sub(?P<subsegment>\d+))?$"
)
IDENTITY_PATTERN = re.compile(
    r"TPG(?P<group>\d+).*?Cell\s*(?P<cell>\d+)", re.IGNORECASE
)
_FIT_DIRECTIONS = (("discharge", 8), ("charge", 10))
_PERCENT_COLUMNS = ("lli_percent", "lam_anode_percent", "lam_cathode_percent")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    replace: bool = False,
) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_text(path: Path, text: str, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(f"Refusing to overwrite an existing output: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _normalized(value: str) -> str:
    return value.strip().replace("\\", "/").casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_capacity_ratio(
    rpt_capacity_span: float | None,
    bol_capacity_span: float | None,
) -> tuple[float | None, str]:
    if (
        rpt_capacity_span is None
        or bol_capacity_span is None
        or not math.isfinite(rpt_capacity_span)
        or not math.isfinite(bol_capacity_span)
        or rpt_capacity_span <= 0
        or bol_capacity_span <= 0
    ):
        return None, "requires_manual_review_missing_or_invalid_step4_capacity"
    return 100.0 * rpt_capacity_span / bol_capacity_span, (
        "provisional_capacity_ratio_candidate_not_official"
    )


def checkpoint_identity(part: str, cell: str, procedure: str, rpt: str) -> str:
    if part == "3":
        if not rpt:
            raise ValueError("A Part 3 EoL record must retain its Guide RPT assignment.")
        return f"EOL-P3-{cell}-RPT{rpt}-{procedure}"
    if rpt:
        raise ValueError("Part 2 EoL checkpoints must not be assigned an invented RPT.")
    return f"EOL-P2-{cell}-{procedure}-UNNUMBERED"


def validate_eol_keys(
    records: list[dict[str, Any]],
    canonical_rows: list[dict[str, str]],
) -> None:
    checkpoint_ids = [str(row["checkpoint_id"]) for row in records]
    if len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Duplicate EoL checkpoint identifiers were found.")
    source_keys = [
        (_normalized(str(row["source_file"])), str(row["physical_cell_id"]))
        for row in records
    ]
    if len(source_keys) != len(set(source_keys)):
        raise ValueError("An EoL source file is assigned to multiple physical cells.")
    for row in records:
        if row["part"] != "3" or not row["guide_rpt_assignment"]:
            continue
        key = (
            "3",
            str(row["physical_cell_id"]),
            str(row["guide_rpt_assignment"]),
        )
        if any(
            (item["part"], item["physical_cell_id"], item["rpt_index"]) == key
            for item in canonical_rows
        ):
            raise ValueError(
                f"Part 3 EoL assignment duplicates a canonical RPT key: {key}"
            )


def _parse_segment_ref(reference: str) -> dict[str, Any] | None:
    match = SEGMENT_PATTERN.fullmatch(reference.strip())
    if not match:
        return None
    parsed = match.groupdict()
    return {
        "source": parsed["0"] if "0" in parsed else reference.split("#", 1)[0],
        "step": int(parsed["step"]),
        "segment_index": int(parsed["segment"]),
        "source_row_start": int(parsed["start"]),
        "source_row_end": int(parsed["end"]),
        "subsegment_index": (
            int(parsed["subsegment"]) if parsed["subsegment"] else None
        ),
    }


def _segment_audit_row(
    checkpoint_id: str,
    record_type: str,
    source_file: str,
    step: int,
    segment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_id": checkpoint_id,
        "record_type": record_type,
        "source_file": source_file,
        "step": step,
        "segment_index": segment["segment_index"],
        "subsegment_index": segment.get("subsegment_index", ""),
        "source_row_start": segment["source_row_start"],
        "source_row_end": segment["source_row_end"],
        "source_row_indexing": "one-based CSV line numbers; header is line 1",
        "sample_count": segment["sample_count"],
        "capacity_min_native": segment["capacity_min"],
        "capacity_max_native": segment["capacity_max"],
        "capacity_span_native": segment["capacity_span"],
        "current_signs": segment.get("current_signs", ""),
        "valid": segment["valid"],
        "status": segment["status"],
    }


def _unique_valid_segment(
    scan: dict[str, Any], step: int
) -> tuple[dict[str, Any] | None, str]:
    if scan["scan_status"] != "scanned":
        return None, f"source_scan_{scan['scan_status']}:{scan['scan_error']}"
    candidates = scan["steps"][str(step)]
    valid = [segment for segment in candidates if segment["valid"]]
    if not valid:
        return None, "no_valid_segment"
    if step == 4:
        complete = [
            segment
            for segment in valid
            if segment["sample_count"] == segment["raw_step_rows"]
        ]
        if not complete:
            return None, "step4_has_missing_or_nonfinite_capacity_voltage_rows"
        valid = complete
    if len(valid) != 1:
        return None, f"ambiguous_{len(valid)}_valid_segments"
    return valid[0], "verified_unique_valid_segment"


def _gap_is_blank(
    path: Path,
    first_line: int,
    last_line: int,
    step: int,
) -> bool:
    if first_line > last_line:
        return True
    missing_tokens = {"", "na", "n/a", "nan", "null", "none"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            return False
        indices = {name: index for index, name in enumerate(header)}
        required = [
            name
            for name in (
                "Step",
                "Cyc",
                "TestTime",
                "StepTime",
                "Amphr",
                "Amps",
                "Volts",
                "Temp1",
                "Temp",
            )
            if name in indices
        ]
        if not {"Step", "TestTime", "StepTime", "Amphr"}.issubset(indices):
            return False
        left: list[str] | None = None
        right: list[str] | None = None
        for values in reader:
            line = reader.line_num
            if line < first_line:
                if line == first_line - 1:
                    left = values
                continue
            if line > last_line:
                right = values
                break
            if any(
                indices[name] < len(values)
                and values[indices[name]].strip().casefold() not in missing_tokens
                for name in required
            ):
                return False
    if left is None or right is None:
        return False

    def numeric(row: list[str], name: str) -> float | None:
        index = indices[name]
        if index >= len(row) or row[index].strip().casefold() in missing_tokens:
            return None
        try:
            value = float(row[index])
        except ValueError:
            return None
        return value if math.isfinite(value) else None

    left_values = {
        name: numeric(left, name) for name in ("Step", "Amphr", "TestTime", "StepTime")
    }
    right_values = {
        name: numeric(right, name) for name in ("Step", "Amphr", "TestTime", "StepTime")
    }
    return bool(
        all(value is not None for value in (*left_values.values(), *right_values.values()))
        and left_values["Step"] == step
        and right_values["Step"] == step
        and right_values["Amphr"] > left_values["Amphr"]
        and right_values["TestTime"] > left_values["TestTime"]
        and right_values["StepTime"] > left_values["StepTime"]
    )


def _resolve_pocv_curve(
    scan: dict[str, Any],
    path: Path,
    step: int,
) -> tuple[dict[str, Any] | None, str]:
    candidates = [
        segment
        for segment in scan["steps"][str(step)]
        if segment["valid"]
    ]
    if not candidates:
        return None, "no_valid_segment"
    candidates.sort(key=lambda item: item["source_row_start"])
    groups: list[list[dict[str, Any]]] = []
    for candidate in candidates:
        if not groups:
            groups.append([candidate])
            continue
        previous = groups[-1][-1]
        blank_gap = _gap_is_blank(
            path,
            previous["source_row_end"] + 1,
            candidate["source_row_start"] - 1,
            step,
        )
        capacity_continues = candidate["capacity_min"] > previous["capacity_max"]
        if blank_gap and capacity_continues:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    if len(groups) != 1:
        return None, f"ambiguous_{len(groups)}_valid_curve_groups"
    components = groups[0]
    first, last = components[0], components[-1]
    combined = {
        **first,
        "source_row_start": first["source_row_start"],
        "source_row_end": last["source_row_end"],
        "sample_count": sum(item["sample_count"] for item in components),
        "capacity_min": min(item["capacity_min"] for item in components),
        "capacity_max": max(item["capacity_max"] for item in components),
        "capacity_span": (
            max(item["capacity_max"] for item in components)
            - min(item["capacity_min"] for item in components)
        ),
        "valid": True,
        "status": (
            "valid_monotone_curve"
            if len(components) == 1
            else "valid_monotone_curve_stitched_across_missing_documented_channels"
        ),
        "components": components,
    }
    return combined, (
        "verified_unique_valid_segment"
        if len(components) == 1
        else "verified_unique_curve_stitched_across_missing_documented_channels"
    )


def _read_exact_frame(
    path: Path,
    start_line: int,
    end_line: int,
    columns: list[str],
) -> pd.DataFrame:
    if start_line < 2 or end_line < start_line:
        raise ValueError(f"Invalid source-row range {start_line}-{end_line} for {path}.")
    return pd.read_csv(
        path,
        usecols=columns,
        skiprows=range(1, start_line - 1),
        nrows=end_line - start_line + 1,
        low_memory=False,
    )


def _verify_step4_source_rows(
    path: Path,
    segment: dict[str, Any] | None,
) -> tuple[bool, str]:
    if segment is None:
        return False, "no_unique_valid_step4_segment"
    if segment["sample_count"] != segment["raw_step_rows"]:
        return False, "step4_has_missing_or_nonfinite_capacity_voltage_rows"
    frame = _read_exact_frame(
        path,
        segment["source_row_start"],
        segment["source_row_end"],
        ["Step", "Amphr", "Volts", "Amps"],
    )
    if len(frame) != segment["sample_count"]:
        return False, "step4_exact_row_count_mismatch"
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    values = numeric[["Step", "Amphr", "Volts", "Amps"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        return False, "step4_missing_or_nonfinite_signal_values"
    if not np.all(numeric["Step"].to_numpy(dtype=float) == 4):
        return False, "step4_source_range_contains_other_step_ids"
    if not np.all(numeric["Amps"].to_numpy(dtype=float) < 0):
        return False, "step4_source_range_contains_non_discharge_current"
    return True, "verified_complete_step4_discharge_source_rows"


def _source_path(root: Path, relative: str) -> Path:
    path = Path(relative.replace("/", "\\"))
    resolved = path if path.is_absolute() else root / path
    if not resolved.is_file():
        raise FileNotFoundError(f"Mapped raw source is missing: {resolved}")
    return resolved


def _verify_manifest_source_hashes(
    root: Path,
    manifest: list[dict[str, str]],
) -> None:
    seen: dict[str, str] = {}
    for row in manifest:
        for source_key, hash_key in (
            ("source_file", "source_sha256"),
            ("baseline_source_file", "baseline_source_sha256"),
        ):
            relative = row.get(source_key, "")
            expected = row.get(hash_key, "")
            if not relative:
                continue
            path = _source_path(root, relative)
            normalized = _normalized(relative)
            actual = seen.get(normalized)
            if actual is None:
                actual = _sha256(path)
                seen[normalized] = actual
            if not expected or actual != expected:
                raise ValueError(
                    f"Raw source changed or lacks its audit hash: {relative}."
                )


def _part_root(root: Path, part: str) -> Path:
    return root / "data" / "raw" / "oxford" / f"part{part}"


def _source_relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _baseline_manifest_row(
    manifest_rows: list[dict[str, str]],
    part: str,
    cell: str,
) -> tuple[dict[str, str] | None, str]:
    candidates = [
        row
        for row in manifest_rows
        if row.get("part") == part and row.get("physical_cell_id") == cell
    ]
    if not candidates:
        return None, "missing_same_cell_observation_in_canonical_manifest"
    refs = {
        (row.get("bol_reference_file", ""), row.get("bol_step4_segment_ref", ""))
        for row in candidates
    }
    if len(refs) != 1:
        return None, "ambiguous_same_cell_bol_reference_in_canonical_manifest"
    bol_file, bol_step4_ref = next(iter(refs))
    if not bol_file or not bol_step4_ref:
        return None, "missing_bol_source_or_step4_reference"
    return {
        "bol_reference_file": bol_file,
        "bol_step4_segment_ref": bol_step4_ref,
    }, "unique_same_cell_bol_reference_in_canonical_manifest"


def _bol_step4_matches(
    root: Path,
    baseline_path: Path,
    baseline_segment: dict[str, Any] | None,
    bol_file: str,
    bol_ref: str,
    baseline_part: str,
    cell: str,
) -> tuple[bool, str]:
    if baseline_segment is None:
        return False, "no_unique_valid_bol_step4_segment"
    parsed = _parse_segment_ref(bol_ref)
    if parsed is None or parsed["step"] != 4:
        return False, "canonical_bol_step4_reference_unparseable"
    if _normalized(Path(parsed["source"]).name) != _normalized(Path(bol_file).name):
        return False, "canonical_bol_reference_filename_mismatch"
    expected_baseline = _part_root(root, baseline_part) / Path(
        bol_file.replace("/", "\\")
    )
    if expected_baseline.resolve() != baseline_path.resolve():
        return False, "canonical_bol_source_path_mismatch"
    match = IDENTITY_PATTERN.search(baseline_path.name)
    cell_group = re.fullmatch(r"G(?P<group>\d+)-Cell(?P<cell>\d+)", cell)
    if (
        match is None
        or cell_group is None
        or match.group("group") != cell_group.group("group")
        or match.group("cell") != cell_group.group("cell")
    ):
        return False, "bol_filename_does_not_confirm_physical_cell"
    segment_key = (
        parsed["segment_index"],
        parsed["source_row_start"],
        parsed["source_row_end"],
    )
    observed_key = (
        baseline_segment["segment_index"],
        baseline_segment["source_row_start"],
        baseline_segment["source_row_end"],
    )
    if segment_key != observed_key:
        return False, "canonical_bol_step4_boundary_differs_from_raw_scan"
    return True, "same_cell_bol_step4_boundary_verified_against_raw_source"


def _protected_files(root: Path, output_dir: Path) -> list[Path]:
    roots = (
        root / "experiments" / "soh_forecast",
        root / "experiments" / "results",
        root / "data" / "processed" / "label_soh_part1_2",
        root / "data" / "processed" / "label_pydma_fit_part1_2",
        root / "data" / "processed" / "label_pydma_fit_part3",
        root / "data" / "manifests" / "bafuse_rpt_observations.csv",
        root / "reports" / "bafuse_final_data_validation_v2" / SPLIT_FILE.name,
        root / "models" / "checkpoints" / "part1_cnn1d" / "cnn1d_checkpoint.pt",
    )
    paths: set[Path] = set()
    for item in roots:
        if item.is_file():
            paths.add(item.resolve())
        elif item.is_dir():
            paths.update(
                path.resolve()
                for path in item.rglob("*")
                if path.is_file()
                and output_dir.resolve() not in path.resolve().parents
            )
    return sorted(paths, key=lambda path: path.as_posix().casefold())


def _capture_before_hashes(root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "protected_artifact_integrity_before.csv"
    if path.exists():
        return
    rows = [
        {
            "path": file.relative_to(root).as_posix(),
            "sha256_before": _sha256(file),
            "size_bytes_before": file.stat().st_size,
        }
        for file in _protected_files(root, output_dir)
    ]
    if not rows:
        raise ValueError("No protected artifacts were found for the integrity baseline.")
    _write_csv(path, rows)


def _write_integrity_comparison(root: Path, output_dir: Path) -> None:
    before_path = output_dir / "protected_artifact_integrity_before.csv"
    before_rows = _read_csv(before_path)
    before = {row["path"]: row for row in before_rows}
    after_files = _protected_files(root, output_dir)
    after = {file.relative_to(root).as_posix(): file for file in after_files}
    rows = []
    for relative in sorted(set(before) | set(after)):
        before_row = before.get(relative)
        after_path = after.get(relative)
        before_hash = before_row["sha256_before"] if before_row else ""
        after_hash = _sha256(after_path) if after_path else ""
        rows.append(
            {
                "path": relative,
                "sha256_before": before_hash,
                "sha256_after": after_hash,
                "existed_before": bool(before_row),
                "exists_after": bool(after_path),
                "unchanged": bool(before_row and after_path and before_hash == after_hash),
            }
        )
    _write_csv(
        output_dir / "protected_artifact_integrity.csv",
        rows,
        replace=True,
    )


def _build_eol_manifest(root: Path, output_dir: Path) -> None:
    audit_rows = _read_csv(AUDIT_INPUT)
    canonical_rows = _read_csv(CANONICAL_MANIFEST)
    if len(audit_rows) != 24:
        raise ValueError(f"Expected 24 audit cell rows, found {len(audit_rows)}.")
    selected = [row for row in audit_rows if row.get("guide_eol_tpg")]
    if len(selected) != 14:
        raise ValueError(f"Expected 14 Guide-mapped EoL procedures, found {len(selected)}.")

    scans: dict[Path, dict[str, Any]] = {}
    manifest: list[dict[str, Any]] = []
    curve_audit: list[dict[str, Any]] = []
    for audit in selected:
        part = "3" if audit["part_scope"] == "Part 3" else "2"
        cell = audit["physical_cell_id"]
        group = str(audit["group"])
        procedure = audit["guide_eol_tpg"]
        guide_rpt = str(audit.get("guide_eol_rpt_number", "")).strip()
        if part == "2" and guide_rpt:
            raise ValueError(f"Part 2 EoL must remain unnumbered: {cell} {procedure}.")
        if part == "3" and not guide_rpt:
            raise ValueError(f"Part 3 Guide RPT assignment is missing: {cell} {procedure}.")
        checkpoint_id = checkpoint_identity(part, cell, procedure, guide_rpt)
        source_file = audit["guide_eol_tpg_path"].replace("\\", "/")
        source = _source_path(root, source_file)
        identity = IDENTITY_PATTERN.search(source.name)
        if (
            identity is None
            or identity.group("group") != group
            or identity.group("cell") != cell.split("-Cell")[-1]
        ):
            raise ValueError(
                f"Raw filename identity does not match Guide cell/group: {source_file}."
            )
        if not _truth(audit.get("guide_eol_raw_mapping_source_present", "")):
            raise ValueError(f"Raw mapping audit did not confirm EoL source: {source_file}.")

        canonical_part = part
        bol_part = "1" if part == "2" else "3"
        bol_row, bol_manifest_status = _baseline_manifest_row(
            canonical_rows, canonical_part, cell
        )
        bol_file = bol_row["bol_reference_file"] if bol_row else ""
        bol_ref = bol_row["bol_step4_segment_ref"] if bol_row else ""
        bol_rel = (
            f"data/raw/oxford/part{bol_part}/{bol_file.replace(chr(92), '/')}"
            if bol_file
            else ""
        )
        bol_source = _source_path(root, bol_rel) if bol_rel else None
        target_scan = scans.setdefault(source.resolve(), None)
        if target_scan is None:
            target_scan = scans[source.resolve()] = _scan_tpg(source)
        baseline_scan = None
        if bol_source:
            baseline_scan = scans.setdefault(bol_source.resolve(), None)
            if baseline_scan is None:
                baseline_scan = scans[bol_source.resolve()] = _scan_tpg(bol_source)

        target_segments: dict[int, dict[str, Any] | None] = {}
        target_status: dict[int, str] = {}
        for step in (4, 8, 10):
            segment, status = (
                _unique_valid_segment(target_scan, step)
                if step == 4
                else _resolve_pocv_curve(target_scan, source, step)
            )
            if step == 4:
                complete, complete_status = _verify_step4_source_rows(source, segment)
                if not complete:
                    segment = None
                    status = complete_status
            target_segments[step] = segment
            target_status[step] = status
            for item in target_scan["steps"][str(step)]:
                curve_audit.append(
                    _segment_audit_row(
                        checkpoint_id,
                        "EOL_TARGET",
                        source_file,
                        step,
                        item,
                    )
                )

        baseline_segments: dict[int, dict[str, Any] | None] = {}
        baseline_status: dict[int, str] = {}
        if baseline_scan:
            for step in (4, 8, 10):
                if step == 4:
                    bol_ref_match = _parse_segment_ref(bol_ref)
                    exact = [
                        item
                        for item in baseline_scan["steps"]["4"]
                        if bol_ref_match
                        and item["segment_index"]
                        == bol_ref_match["segment_index"]
                        and item["source_row_start"]
                        == bol_ref_match["source_row_start"]
                        and item["source_row_end"]
                        == bol_ref_match["source_row_end"]
                    ]
                    if len(exact) == 1 and exact[0]["valid"]:
                        segment, status = exact[0], (
                            "verified_exact_canonical_bol_step4_segment"
                        )
                    elif len(exact) == 1:
                        segment, status = None, "canonical_bol_step4_segment_invalid"
                    else:
                        segment, status = None, (
                            "canonical_bol_step4_segment_not_unique_in_raw_scan"
                        )
                    complete, complete_status = _verify_step4_source_rows(
                        bol_source, segment
                    )
                    if not complete:
                        segment, status = None, complete_status
                else:
                    segment, status = _resolve_pocv_curve(
                        baseline_scan, bol_source, step
                    )
                baseline_segments[step] = segment
                baseline_status[step] = status
                for item in baseline_scan["steps"][str(step)]:
                    curve_audit.append(
                        _segment_audit_row(
                            checkpoint_id,
                            "SAME_CELL_BOL_REFERENCE",
                            bol_rel,
                            step,
                            item,
                        )
                    )
        else:
            baseline_segments = {4: None, 8: None, 10: None}
            baseline_status = {
                step: "unavailable_no_unique_canonical_bol_reference"
                for step in (4, 8, 10)
            }

        bol_step4_ok, bol_step4_status = _bol_step4_matches(
            root,
            bol_source,
            baseline_segments[4],
            bol_file,
            bol_ref,
            bol_part,
            cell,
        ) if bol_source else (False, bol_manifest_status)
        q_rpt = (
            float(target_segments[4]["capacity_span"])
            if target_segments[4] is not None
            else None
        )
        q_bol = (
            float(baseline_segments[4]["capacity_span"])
            if bol_step4_ok and baseline_segments[4] is not None
            else None
        )
        provisional_soh, capacity_ratio_status = candidate_capacity_ratio(q_rpt, q_bol)
        if not bol_step4_ok:
            capacity_ratio_status = f"requires_manual_review:{bol_step4_status}"

        step_refs = {
            step: (
                _segment_refs(source_file, target_segments[step])
                if target_segments[step] is not None
                else ""
            )
            for step in (4, 8, 10)
        }
        bol_step_refs = {
            step: (
                _segment_refs(bol_rel, baseline_segments[step])
                if baseline_segments[step] is not None
                else ""
            )
            for step in (4, 8, 10)
        }
        mapping_status = (
            "guide_procedure_and_raw_filename_identity_verified; "
            + (
                "Guide RPT assignment verified"
                if part == "3"
                else "unnumbered Part 2 EoL checkpoint"
            )
        )
        manifest.append(
            {
                "checkpoint_id": checkpoint_id,
                "part": part,
                "part_label": "Part 3" if part == "3" else "Part 2 continuation",
                "group": group,
                "physical_cell_id": cell,
                "guide_procedure": audit["guide_eol_guide_procedure"],
                "source_tpg": procedure,
                "guide_stage": audit["guide_eol_guide_stage"],
                "guide_rpt_assignment": guide_rpt,
                "rpt_index": guide_rpt if part == "3" else "",
                "source_file": source_file,
                "source_sha256": _sha256(source),
                "mapping_confidence": (
                    "high_guide_procedure_and_raw_cell_filename_identity"
                ),
                "mapping_status": mapping_status,
                "guide_index_source_present": audit.get(
                    "guide_eol_source_file_present_index", ""
                ),
                "raw_mapping_source_present": audit.get(
                    "guide_eol_raw_mapping_source_present", ""
                ),
                "prior_audit_last_mapped_rpt": audit.get("last_mapped_rpt", ""),
                "baseline_source_part": bol_part,
                "baseline_reference_file": bol_file,
                "baseline_source_file": bol_rel,
                "baseline_source_sha256": (
                    _sha256(bol_source) if bol_source else ""
                ),
                "baseline_manifest_status": bol_manifest_status,
                "baseline_step4_segment_ref": bol_ref,
                "baseline_step4_segment_ref_match_status": bol_step4_status,
                "baseline_step4_segment_ref_exact": bol_step_refs[4],
                "baseline_step8_segment_ref_exact": bol_step_refs[8],
                "baseline_step10_segment_ref_exact": bol_step_refs[10],
                "baseline_step4_status": (
                    "verified_same_cell_bol_step4" if bol_step4_ok else baseline_status[4]
                ),
                "baseline_step8_status": baseline_status[8],
                "baseline_step10_status": baseline_status[10],
                "step4_segment_ref": step_refs[4],
                "step8_segment_ref": step_refs[8],
                "step10_segment_ref": step_refs[10],
                "step4_components_json": json.dumps(
                    target_segments[4].get("components", [target_segments[4]])
                    if target_segments[4]
                    else []
                ),
                "step8_components_json": json.dumps(
                    target_segments[8].get("components", [target_segments[8]])
                    if target_segments[8]
                    else []
                ),
                "step10_components_json": json.dumps(
                    target_segments[10].get("components", [target_segments[10]])
                    if target_segments[10]
                    else []
                ),
                "baseline_step4_components_json": json.dumps(
                    baseline_segments[4].get("components", [baseline_segments[4]])
                    if baseline_segments[4]
                    else []
                ),
                "baseline_step8_components_json": json.dumps(
                    baseline_segments[8].get("components", [baseline_segments[8]])
                    if baseline_segments[8]
                    else []
                ),
                "baseline_step10_components_json": json.dumps(
                    baseline_segments[10].get("components", [baseline_segments[10]])
                    if baseline_segments[10]
                    else []
                ),
                "step4_status": target_status[4],
                "step8_status": target_status[8],
                "step10_status": target_status[10],
                "step4_sample_count": (
                    target_segments[4]["sample_count"] if target_segments[4] else ""
                ),
                "step8_sample_count": (
                    target_segments[8]["sample_count"] if target_segments[8] else ""
                ),
                "step10_sample_count": (
                    target_segments[10]["sample_count"] if target_segments[10] else ""
                ),
                "step4_capacity_span_native": q_rpt if q_rpt is not None else "",
                "step8_capacity_span_native": (
                    target_segments[8]["capacity_span"] if target_segments[8] else ""
                ),
                "step10_capacity_span_native": (
                    target_segments[10]["capacity_span"] if target_segments[10] else ""
                ),
                "baseline_step4_capacity_span_native": (
                    q_bol if q_bol is not None else ""
                ),
                "provisional_reference_soh_percent": (
                    provisional_soh if provisional_soh is not None else ""
                ),
                "capacity_ratio_status": capacity_ratio_status,
                "capacity_unit": "native Amphr; absolute unit unverified",
                "source_provenance": (
                    "Guide EoL mapping plus prior raw filename/cell crosswalk; "
                    "raw source re-scanned and segment row boundaries independently verified"
                ),
            }
        )
    validate_eol_keys(manifest, canonical_rows)
    if len({row["physical_cell_id"] for row in manifest}) != len(manifest):
        raise ValueError("Expected one EoL checkpoint per included physical cell.")
    _write_csv(output_dir / "eol_checkpoint_manifest.csv", manifest)
    if curve_audit:
        _write_csv(output_dir / "eol_curve_audit.csv", curve_audit)
    else:
        raise ValueError("No raw Step 4/8/10 segment audit records were produced.")


def _scan_tpg(path: Path) -> dict[str, Any]:
    from bafuse.data.rpt_mapping_audit import scan_tpg_file

    scan = scan_tpg_file(path)
    if scan["scan_status"] != "scanned":
        raise ValueError(
            f"Raw TPG scan failed for {path}: {scan['scan_status']} {scan['scan_error']}"
        )
    return scan


def _segment_ref(source_file: str, segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    suffix = (
        f"#Step{segment['step']}:segment{segment['segment_index']}"
        f":rows{segment['source_row_start']}-{segment['source_row_end']}"
    )
    if segment.get("subsegment_index"):
        suffix += f":sub{segment['subsegment_index']}"
    return source_file + suffix


def _segment_refs(source_file: str, segment: dict[str, Any] | None) -> str:
    if not segment:
        return ""
    components = segment.get("components", [segment])
    return ";".join(_segment_ref(source_file, component) for component in components)


def _read_pocv_components(
    root: Path,
    source_file: str,
    components_json: str,
    step: int,
    direction: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    components = json.loads(components_json)
    if not components:
        raise ValueError(f"No exact Step {step} curve components were recorded.")
    if direction != ("discharge" if step == 8 else "charge"):
        raise ValueError(f"Step {step} does not match direction {direction!r}.")
    path = _source_path(root, source_file)
    capacities: list[np.ndarray] = []
    voltages: list[np.ndarray] = []
    component_audits = []
    for previous, current_component in zip(components, components[1:]):
        if not _gap_is_blank(
            path,
            int(previous["source_row_end"]) + 1,
            int(current_component["source_row_start"]) - 1,
            step,
        ):
            raise ValueError(
                f"Step {step} component gap no longer passes documented-channel checks: "
                f"{source_file}."
            )
        if float(current_component["capacity_min"]) <= float(
            previous["capacity_max"]
        ):
            raise ValueError(
                f"Step {step} capacity is not increasing across components: {source_file}."
            )
    for component in components:
        start = int(component["source_row_start"])
        end = int(component["source_row_end"])
        frame = _read_exact_frame(
            path,
            start,
            end,
            ["Step", "Amphr", "Amps", "Volts"],
        )
        values = frame.apply(pd.to_numeric, errors="coerce")
        step_values = values["Step"].to_numpy(dtype=np.float64)
        capacity = values["Amphr"].to_numpy(dtype=np.float64)
        current = values["Amps"].to_numpy(dtype=np.float64)
        voltage = values["Volts"].to_numpy(dtype=np.float64)
        sign = current < 0 if direction == "discharge" else current > 0
        mask = (
            np.isfinite(step_values)
            & (step_values == step)
            & np.isfinite(capacity)
            & np.isfinite(voltage)
            & np.isfinite(current)
            & sign
            & (np.abs(current) <= 0.2)
        )
        selected_capacity = capacity[mask]
        selected_voltage = voltage[mask]
        selected_rows = np.arange(start, end + 1, dtype=np.int64)[mask]
        if (
            len(selected_capacity) != int(component["sample_count"])
            or not len(selected_rows)
            or selected_rows[0] != start
            or selected_rows[-1] != end
        ):
            raise ValueError(
                f"Step {step} exact component rows changed since the audit: "
                f"{source_file}:{start}-{end}."
            )
        if not np.all(np.diff(selected_capacity) > 0):
            raise ValueError(
                f"Step {step} component is not strictly monotone: {source_file}:{start}-{end}."
            )
        capacities.append(selected_capacity)
        voltages.append(selected_voltage)
        component_audits.append(
            {
                "segment_index": component["segment_index"],
                "subsegment_index": component.get("subsegment_index", ""),
                "source_row_start": start,
                "source_row_end": end,
                "sample_count": len(selected_capacity),
                "capacity_min_native": float(selected_capacity.min()),
                "capacity_max_native": float(selected_capacity.max()),
            }
        )
    combined_capacity = np.concatenate(capacities)
    combined_voltage = np.concatenate(voltages)
    if not np.all(np.diff(combined_capacity) > 0):
        raise ValueError(
            f"Step {step} components are not capacity-contiguous in source order: {source_file}."
        )
    audit = {
        "status": (
            "verified_exact_source_ranges"
            if len(components) == 1
            else "verified_exact_ranges_stitched_across_missing_documented_channels"
        ),
        "source_file": source_file,
        "step": step,
        "direction": direction,
        "source_row_start": int(component_audits[0]["source_row_start"]),
        "source_row_end": int(component_audits[-1]["source_row_end"]),
        "source_row_indexing": "one-based CSV line numbers; header is line 1",
        "component_audits": component_audits,
        "sample_count": len(combined_capacity),
        "capacity_min_native": float(combined_capacity.min()),
        "capacity_max_native": float(combined_capacity.max()),
        "capacity_span_native": float(np.ptp(combined_capacity)),
        "voltage_min_v": float(combined_voltage.min()),
        "voltage_max_v": float(combined_voltage.max()),
    }
    return combined_capacity, combined_voltage, audit


def _run_frozen_inference(root: Path, output_dir: Path) -> None:
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(
        0,
        str(root / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"),
    )
    import torch
    import run_phase2 as phase2
    from bafuse.hybrid.soh_fusion import (
        INPUT_SCHEMA_VERSION,
        MODEL_VERSION,
        PHYSICS_FEATURE_NAMES,
        ThreeEncoderSoH,
        build_physics_features,
        resample_curve,
    )

    manifest = _read_csv(output_dir / "eol_checkpoint_manifest.csv")
    _verify_manifest_source_hashes(root, manifest)
    model_rows: list[dict[str, Any]] = []
    extraction_rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for item in manifest:
        row = dict(item)
        row["rpt_index"] = (
            item["guide_rpt_assignment"]
            if item["part"] == "3"
            else f"EOL:{item['checkpoint_id']}"
        )
        for step in (4, 10):
            status_key = f"step{step}_status"
            row[status_key] = (
                "verified"
                if step == 4
                and item[status_key]
                in {
                    "verified_unique_valid_segment",
                }
                else ""
            )
        row["soh_percent"] = None
        row["step4_result"] = None
        row["step10_result"] = None
        row["physics_features"] = None
        row["step4_duration_native_audit_only"] = None
        observations.append(row)

    requests = phase2._segment_requests(observations, include_unlabeled=True)
    for source_requests in requests.values():
        phase2._scan_source_requests(root, source_requests)

    for row, item in zip(observations, manifest):
        step4 = row["step4_result"]
        step10 = None
        if item["step10_status"].startswith("verified_unique"):
            capacity, voltage, audit = _read_pocv_components(
                root,
                item["source_file"],
                item["step10_components_json"],
                10,
                "charge",
            )
            step10 = {
                "valid": True,
                "status": audit["status"],
                "curve": resample_curve(voltage.tolist()),
                "voltage_raw": voltage,
                "current_raw": None,
                "source_row_start": audit["source_row_start"],
                "source_row_end": audit["source_row_end"],
                "observed_rows": audit["sample_count"],
            }
            row["step10_result"] = step10
        discharge_valid = bool(step4 and step4["valid"])
        charge_valid = bool(step10 and step10["valid"])
        physics_status = "unavailable_step4_invalid_or_missing"
        if discharge_valid and step4.get("temperature_complete", False):
            row["physics_features"] = build_physics_features(
                step4["voltage_raw"],
                step4["current_raw"],
                step4["temperature_raw"],
            )
            physics_status = "verified_complete_step4_temperature_and_signals"
        elif discharge_valid:
            physics_status = "unavailable_step4_temperature_missing_or_incomplete"
        available = []
        if discharge_valid:
            available.append("D")
        if charge_valid:
            available.append("C")
        if row["physics_features"] is not None:
            available.append("P")
        complete = len(available) == 3
        extraction_rows.append(
            {
                "checkpoint_id": item["checkpoint_id"],
                "physical_cell_id": item["physical_cell_id"],
                "part": item["part"],
                "step4_raw_status": step4["status"] if step4 else "not_mapped",
                "step4_source_rows": (
                    f"{step4['row_start']}-{step4['row_end']}"
                    if step4 and step4.get("valid")
                    else ""
                ),
                "step10_raw_status": step10["status"] if step10 else "not_mapped",
                "step10_source_rows": (
                    f"{step10['source_row_start']}-{step10['source_row_end']}"
                    if step10 and step10.get("valid")
                    else ""
                ),
                "step4_temperature_complete": (
                    bool(step4.get("temperature_complete", False)) if step4 else False
                ),
                "physics_status": physics_status,
                "available_modalities": "+".join(available),
                "three_encoder_inference_eligible": complete,
                "eligibility_reason": (
                    "all validated D+C+P modalities present"
                    if complete
                    else "requires complete Step 4 discharge, Step 10 charge, and Step 4 physics"
                ),
            }
        )

    checkpoint_hash = _sha256(FROZEN_CHECKPOINT)
    payload = torch.load(FROZEN_CHECKPOINT, map_location="cpu", weights_only=False)
    if (
        payload.get("model_version") != MODEL_VERSION
        or payload.get("input_schema_version") != INPUT_SCHEMA_VERSION
    ):
        raise ValueError("Frozen checkpoint model or input schema does not match.")
    required = {
        "model_state_dict",
        "discharge_mean",
        "discharge_scale",
        "charge_mean",
        "charge_scale",
        "physics_mean",
        "physics_scale",
        "target_mean",
        "target_scale",
        "physics_feature_names",
    }
    if not required.issubset(payload):
        raise ValueError(
            "Frozen Phase 2 checkpoint is missing scaler/model fields: "
            + ", ".join(sorted(required.difference(payload)))
        )
    if tuple(payload["physics_feature_names"]) != tuple(PHYSICS_FEATURE_NAMES):
        raise ValueError("Frozen checkpoint physics feature order differs.")

    model = ThreeEncoderSoH()
    model.load_state_dict(payload["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    scalers = {}
    for key, width in (
        ("discharge_mean", 128),
        ("discharge_scale", 128),
        ("charge_mean", 128),
        ("charge_scale", 128),
        ("physics_mean", len(PHYSICS_FEATURE_NAMES)),
        ("physics_scale", len(PHYSICS_FEATURE_NAMES)),
    ):
        values = np.asarray(payload[key], dtype=np.float32).reshape(-1)
        if values.shape != (width,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid frozen checkpoint scaler: {key}")
        if key.endswith("_scale") and np.any(values <= 0):
            raise ValueError(f"Non-positive frozen checkpoint scaler: {key}")
        scalers[key] = values
    target_mean = float(payload["target_mean"])
    target_scale = float(payload["target_scale"])
    if not math.isfinite(target_mean) or not math.isfinite(target_scale) or target_scale <= 0:
        raise ValueError("Invalid frozen checkpoint target scaler.")

    extraction_by_id = {row["checkpoint_id"]: row for row in extraction_rows}
    for row, item in zip(observations, manifest):
        extraction = extraction_by_id[item["checkpoint_id"]]
        prediction: float | None = None
        displayed_prediction: float | None = None
        status = "unavailable_incomplete_DCP"
        plausibility_status = "not_evaluated_no_model_output"
        if extraction["three_encoder_inference_eligible"]:
            discharge = np.asarray(row["step4_result"]["curve"], dtype=np.float32)
            charge = np.asarray(row["step10_result"]["curve"], dtype=np.float32)
            physics = np.asarray(
                [row["physics_features"][name] for name in PHYSICS_FEATURE_NAMES],
                dtype=np.float32,
            )
            inputs = (
                (discharge - scalers["discharge_mean"]) / scalers["discharge_scale"],
                (charge - scalers["charge_mean"]) / scalers["charge_scale"],
                (physics - scalers["physics_mean"]) / scalers["physics_scale"],
            )
            with torch.no_grad():
                output = model(
                    *(
                        torch.from_numpy(value[None, :]).to(device)
                        for value in inputs
                    )
                )
            prediction = float(output.reshape(-1)[0].item() * target_scale + target_mean)
            if not math.isfinite(prediction):
                raise ValueError(
                    f"Frozen checkpoint returned a non-finite prediction for {item['checkpoint_id']}."
                )
            if prediction < 0:
                status = (
                    "withheld_negative_raw_output_incompatible_with_positive_capacity_ratio"
                )
                plausibility_status = (
                    "negative_output_incompatible_with_positive_capacity_ratio"
                )
            else:
                displayed_prediction = prediction
                status = "predicted_by_frozen_phase2_three_encoder_checkpoint"
                plausibility_status = "nonnegative_model_output"
        model_rows.append(
            {
                "checkpoint_id": item["checkpoint_id"],
                "part": item["part"],
                "physical_cell_id": item["physical_cell_id"],
                "guide_rpt_assignment": item["guide_rpt_assignment"],
                "available_modalities": extraction["available_modalities"],
                "inference_eligible": extraction["three_encoder_inference_eligible"],
                "raw_model_output_soh_percent": (
                    prediction if prediction is not None else ""
                ),
                "predicted_soh_percent": (
                    displayed_prediction
                    if displayed_prediction is not None
                    else ""
                ),
                "prediction_plausibility_status": plausibility_status,
                "provisional_reference_soh_percent": item[
                    "provisional_reference_soh_percent"
                ],
                "reference_status": item["capacity_ratio_status"],
                "soh_status": status,
                "checkpoint_path": FROZEN_CHECKPOINT.relative_to(root).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "inference_device": str(device),
                "training_or_refit_performed": False,
                "external_model_selection_performed": False,
            }
        )
    _write_csv(output_dir / "eol_modality_inference_audit.csv", extraction_rows)
    _write_csv(output_dir / "eol_model_predictions.csv", model_rows)


def _fit_record(
    *,
    part: str,
    cell: str,
    rpt: str,
    direction: str,
    observation_type: str,
    fit_id: str,
    curve: np.ndarray,
    voltage: np.ndarray,
    curve_audit: dict[str, Any],
    references: dict[str, Any],
    source_file: str,
    result: Any | None,
    runtime_seconds: float,
    error: str = "",
) -> dict[str, Any]:
    from bafuse.data.pydma_stage2 import _fit_record as stage2_fit_record

    return stage2_fit_record(
        fit_id=fit_id,
        observation_type=observation_type,
        part=part,
        cell=cell,
        rpt=rpt,
        direction=direction,
        seed=211,
        curve=curve,
        voltage=voltage,
        curve_audit=curve_audit,
        references=references,
        source_file=source_file,
        baseline_reference_id=f"same_cell_bol:{cell}:{direction}",
        result=result,
        runtime_seconds=runtime_seconds,
        error=error,
    )


def _run_one_fit(
    *,
    direction: str,
    curve: np.ndarray,
    voltage: np.ndarray,
    references: tuple[Any, Any, dict[str, Any]],
    context: str,
    use_cuda: bool,
) -> tuple[Any | None, float, str]:
    from bafuse.labeling.pydma_runner import _fit_one

    started = time.perf_counter()
    try:
        result = _fit_one(
            direction,
            curve,
            voltage,
            references[0],
            references[1],
            seed=211,
            context=context,
            use_cuda=use_cuda,
            maxiter=50,
            popsize=30,
            max_tries=10,
        )
    except Exception as error:
        message = f"{type(error).__name__}: {error}"
        print(f"PyDMA fit failed {context}: {message}", flush=True)
        return None, time.perf_counter() - started, message
    return result, time.perf_counter() - started, ""


def _run_eol_pydma(root: Path, output_dir: Path) -> None:
    import torch
    from pydma import DMAAnalyzer, DMAConfig
    from bafuse.data.pydma_stage2 import _load_references
    from bafuse.labeling.pydma_policy import pydma_fit_passes_gate

    manifest = _read_csv(output_dir / "eol_checkpoint_manifest.csv")
    _verify_manifest_source_hashes(root, manifest)
    fit_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    target_cache: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    baseline_cache: dict[
        tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    references = {
        direction: _load_references(root, direction)
        for direction, _ in _FIT_DIRECTIONS
    }

    for row in manifest:
        for direction, step in _FIT_DIRECTIONS:
            if not row[f"step{step}_status"].startswith("verified_unique"):
                continue
            target_curve = _read_pocv_components(
                root,
                row["source_file"],
                row[f"step{step}_components_json"],
                step,
                direction,
            )
            target_cache[(row["checkpoint_id"], direction)] = target_curve
            if (
                not row[f"baseline_step{step}_status"].startswith("verified")
                or not row[f"baseline_step{step}_components_json"]
                or row[f"baseline_step{step}_components_json"] == "[]"
            ):
                continue
            key = (row["physical_cell_id"], direction)
            if key not in baseline_cache:
                baseline_cache[key] = _read_pocv_components(
                    root,
                    row["baseline_source_file"],
                    row[f"baseline_step{step}_components_json"],
                    step,
                    direction,
                )

    use_cuda = bool(torch.cuda.is_available())
    print(
        f"Starting EoL PyDMA fits: target directions={len(target_cache)}, "
        f"same-cell BoL direction references={len(baseline_cache)}, CUDA={use_cuda}",
        flush=True,
    )
    baseline_results: dict[tuple[str, str], Any | None] = {}
    baseline_fit_rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for key, (capacity, voltage, audit) in sorted(baseline_cache.items()):
        cell, direction = key
        part = next(
            row["baseline_source_part"]
            for row in manifest
            if row["physical_cell_id"] == cell
            and row[f"baseline_step{8 if direction == 'discharge' else 10}_status"]
            == "verified_unique_valid_segment"
        )
        result, runtime, error = _run_one_fit(
            direction=direction,
            curve=capacity,
            voltage=voltage,
            references=references[direction],
            context=f"same-cell BoL {cell} {direction}",
            use_cuda=use_cuda,
        )
        baseline_results[key] = result
        source_file = audit["source_file"]
        record = _fit_record(
            part=part,
            cell=cell,
            rpt="0",
            direction=direction,
            observation_type="BOL_REFERENCE",
            fit_id=f"bol-{cell}-{direction}-seed211",
            curve=capacity,
            voltage=voltage,
            curve_audit=audit,
            references=references[direction][2],
            source_file=source_file,
            result=result,
            runtime_seconds=runtime,
            error=error,
        )
        record["fit_gate_passed"] = bool(
            result
            and pydma_fit_passes_gate(
                rmse_mV=float(result.rmse * 1000),
                accepted_runs=int(result.n_accepted_runs),
                is_accepted=bool(result.is_accepted),
            )
        )
        record["fit_scope"] = (
            "same_cell_same_direction_bol_reference_for_eol_comparison"
        )
        baseline_fit_rows_by_key[key] = record
        baseline_rows.append(record)

    target_results: dict[tuple[str, str], Any | None] = {}
    for row in manifest:
        for direction, step in _FIT_DIRECTIONS:
            key = (row["checkpoint_id"], direction)
            target_entry = target_cache.get(key)
            if target_entry is None:
                fit_rows.append(
                    {
                        "fit_id": f"{row['checkpoint_id']}-{direction}",
                        "checkpoint_id": row["checkpoint_id"],
                        "part": row["part"],
                        "physical_cell_id": row["physical_cell_id"],
                        "guide_rpt_assignment": row["guide_rpt_assignment"],
                        "direction": direction,
                        "step": step,
                        "source_file": row["source_file"],
                        "source_row_start": "",
                        "source_row_end": "",
                        "sample_count": "",
                        "capacity_span_native": "",
                        "curve_preflight_status": row[f"step{step}_status"],
                        "fit_status": "not_run_curve_unavailable_or_ambiguous",
                        "fit_rmse_mV": "",
                        "accepted_runs": 0,
                        "total_runs": 0,
                        "fit_gate_passed": False,
                        "degradation_estimate_status": "not_run_curve_unavailable_or_ambiguous",
                        "lli_percent": "",
                        "lam_anode_percent": "",
                        "lam_cathode_percent": "",
                        "negative_estimate_flags": "",
                        "parameter_bounds_json": "[]",
                        "parameter_std_json": "null",
                        "fitted_parameters_json": "{}",
                        "baseline_reference_status": row[
                            f"baseline_step{step}_status"
                        ],
                        "baseline_fit_status": "not_run",
                        "baseline_fit_rmse_mV": "",
                        "error": "",
                    }
                )
                continue
            capacity, voltage, audit = target_entry
            result, runtime, error = _run_one_fit(
                direction=direction,
                curve=capacity,
                voltage=voltage,
                references=references[direction],
                context=(
                    f"EoL {row['checkpoint_id']} {direction}"
                ),
                use_cuda=use_cuda,
            )
            target_results[key] = result
            target_fit = _fit_record(
                part=row["part"],
                cell=row["physical_cell_id"],
                rpt=row["guide_rpt_assignment"],
                direction=direction,
                observation_type="EOL_CHECKPOINT",
                fit_id=f"{row['checkpoint_id']}-{direction}-seed211",
                curve=capacity,
                voltage=voltage,
                curve_audit=audit,
                references=references[direction][2],
                source_file=row["source_file"],
                result=result,
                runtime_seconds=runtime,
                error=error,
            )
            fit_gate = bool(
                result
                and pydma_fit_passes_gate(
                    rmse_mV=float(result.rmse * 1000),
                    accepted_runs=int(result.n_accepted_runs),
                    is_accepted=bool(result.is_accepted),
                )
            )
            target_fit.update(
                checkpoint_id=row["checkpoint_id"],
                part=row["part"],
                guide_rpt_assignment=row["guide_rpt_assignment"],
                fit_gate_passed=fit_gate,
                lli_percent="",
                lam_anode_percent="",
                lam_cathode_percent="",
                negative_estimate_flags="",
                baseline_reference_status=row[
                    f"baseline_step{step}_status"
                ],
                baseline_fit_status="not_run",
                baseline_fit_rmse_mV="",
                degradation_estimate_status=(
                    "withheld_fit_rejected" if result else "fit_exception"
                ),
            )
            baseline_key = (row["physical_cell_id"], direction)
            baseline_result = baseline_results.get(baseline_key)
            baseline_record = baseline_fit_rows_by_key.get(baseline_key)
            if baseline_record:
                target_fit["baseline_fit_status"] = baseline_record["fit_status"]
                target_fit["baseline_fit_rmse_mV"] = baseline_record["rmse_mV"]
            if not fit_gate:
                target_fit["degradation_estimate_status"] = (
                    "withheld_fit_rejected" if result else "fit_exception"
                )
            elif not baseline_record or not baseline_record["fit_gate_passed"]:
                target_fit["degradation_estimate_status"] = (
                    "withheld_baseline_fit_failed_or_unavailable"
                )
            else:
                baseline_capacity = baseline_cache[baseline_key][0]
                try:
                    comparison = DMAAnalyzer(
                        config=DMAConfig(direction=direction),
                        anode=references[direction][0],
                        cathode=references[direction][1],
                    ).compare_with_reference(
                        reference_params=baseline_result.fitted_params,
                        current_params=result.fitted_params,
                        reference_capacity=float(np.ptp(baseline_capacity)),
                        current_capacity=float(np.ptp(capacity)),
                    )
                except Exception as error:
                    target_fit["degradation_estimate_status"] = (
                        f"withheld_comparison_exception:{type(error).__name__}"
                    )
                    target_fit["error"] = (
                        f"{target_fit.get('error', '')}; "
                        f"{type(error).__name__}: {error}"
                    ).strip("; ")
                else:
                    values = {
                        "lli_percent": float(comparison.lli * 100),
                        "lam_anode_percent": float(comparison.lam_anode * 100),
                        "lam_cathode_percent": float(comparison.lam_cathode * 100),
                    }
                    target_fit.update(values)
                    target_fit["negative_estimate_flags"] = ";".join(
                        name
                        for name, value in values.items()
                        if value < 0
                    )
                    target_fit["degradation_estimate_status"] = (
                        "diagnostic_estimate_accepted"
                    )
            target_fit["parameter_at_bound_flags"] = _parameter_bound_flags(
                target_fit.get("parameter_bounds_json", "[]")
            )
            fit_rows.append(target_fit)
            print(
                f"Completed EoL PyDMA {row['checkpoint_id']} {direction}: "
                f"gate={fit_gate} estimate={target_fit['degradation_estimate_status']}",
                flush=True,
            )

    _write_csv(output_dir / "eol_pydma_fits.csv", fit_rows)
    if baseline_rows:
        _write_csv(output_dir / "eol_pydma_baseline_fits.csv", baseline_rows)
    quality_rows = _build_quality_diagnostics(root, fit_rows, baseline_rows)
    _write_csv(output_dir / "pydma_quality_diagnostics.csv", quality_rows)
    consistency_rows = _build_direction_consistency(root, fit_rows)
    if consistency_rows:
        _write_csv(output_dir / "pydma_direction_consistency.csv", consistency_rows)
    dashboard_rows = _build_dashboard_records(output_dir, fit_rows)
    _write_csv(output_dir / "eol_dashboard_records.csv", dashboard_rows)
    _write_text(
        output_dir / "pydma_eol_quality_report.md",
        _pydma_quality_report(fit_rows, baseline_rows, use_cuda),
    )


def _parameter_bound_flags(bounds_json: str) -> str:
    try:
        items = json.loads(bounds_json or "[]")
    except json.JSONDecodeError:
        return "parameter_bounds_json_invalid"
    return ";".join(
        item["parameter"]
        for item in items
        if item.get("exact_or_near_bound")
    )


def _build_quality_diagnostics(
    root: Path,
    eol_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical_path = (
        root
        / "experiments"
        / "results"
        / "data_audits"
        / "pydma_eol_audit_v1"
        / "pydma_negative_estimate_audit.csv"
    )
    stability_path = (
        root
        / "experiments"
        / "results"
        / "data_audits"
        / "pydma_eol_audit_v1"
        / "pydma_fit_stability.csv"
    )
    historical = _read_csv(historical_path)
    stability = _read_csv(stability_path)
    stability_by_key: dict[tuple[str, str, str, str], list[dict[str, str]]] = {}
    for row in stability:
        key = (
            row["part"],
            row["physical_cell_id"],
            row["rpt"],
            row["direction"],
        )
        stability_by_key.setdefault(key, []).append(row)

    output: list[dict[str, Any]] = []
    for row in historical:
        part = row["part"]
        rpt = row["rpt"]
        key = (part, row["physical_cell_id"], rpt, row["direction"])
        repeats = stability_by_key.get(key, [])
        values = [
            row.get("LLI_percent_vs_BoL", ""),
            row.get("LAM_anode_percent_vs_BoL", ""),
            row.get("LAM_cathode_percent_vs_BoL", ""),
        ]
        negatives = [
            name
            for name, value in zip(_PERCENT_COLUMNS, values)
            if value != "" and float(value) < 0
        ]
        output.append(
            {
                "source": "existing_pydma_eol_audit_v1",
                "dataset_part": part,
                "physical_cell_id": row["physical_cell_id"],
                "checkpoint_id": rpt,
                "rpt_or_checkpoint": rpt,
                "direction": row["direction"],
                "test_step": row["test_step"],
                "source_file": row["source_file"],
                "fit_rmse_mV": row["fit_rmse_mV"],
                "accepted_runs": row["accepted_runs"],
                "total_runs": row["total_runs"],
                "fit_gate_passed": row["diagnostic_label_status"]
                == "diagnostic_estimate_accepted",
                "lli_percent": values[0],
                "lam_anode_percent": values[1],
                "lam_cathode_percent": values[2],
                "negative_estimate_flags": ";".join(negatives),
                "parameter_at_bound_flags": "unavailable_saved_parameter_vectors_absent",
                "parameter_std_json": "unavailable_saved_parameter_vectors_absent",
                "within_fit_parameter_stability_status": (
                    "not_available_saved_parameter_vectors_absent"
                ),
                "multi_seed_stability_status": (
                    f"prior_{len(repeats)}_seed_case_available"
                    if repeats
                    else "not_assessed_for_this_saved_fit"
                ),
                "baseline_reference_consistency": (
                    "same_direction_bol_comparison_recorded_in_existing_pipeline"
                    if row["diagnostic_label_status"] == "diagnostic_estimate_accepted"
                    else row["diagnostic_label_status"]
                ),
                "degradation_estimate_status": row["diagnostic_label_status"],
                "error": "",
            }
        )

    for row in baseline_rows:
        output.append(
            _quality_row(
                row,
                source="eol_same_cell_bol_fit",
                checkpoint_id="BoL",
                estimate_status="reference_fit_no_degradation_estimate",
            )
        )
    for row in eol_rows:
        output.append(
            _quality_row(
                row,
                source="eol_checkpoint_fit",
                checkpoint_id=row.get("checkpoint_id", ""),
                estimate_status=row.get("degradation_estimate_status", ""),
            )
        )
    return output


def _quality_row(
    row: dict[str, Any],
    *,
    source: str,
    checkpoint_id: str,
    estimate_status: str,
) -> dict[str, Any]:
    return {
        "source": source,
        "dataset_part": row.get("dataset_part", row.get("part", "")),
        "physical_cell_id": row.get("physical_cell_id", ""),
        "checkpoint_id": checkpoint_id,
        "rpt_or_checkpoint": row.get("rpt_index", checkpoint_id),
        "direction": row.get("direction", ""),
        "test_step": row.get("step", ""),
        "source_file": row.get("source_file", ""),
        "fit_rmse_mV": row.get("rmse_mV", row.get("fit_rmse_mV", "")),
        "accepted_runs": row.get("accepted_runs", 0),
        "total_runs": row.get("total_runs", 0),
        "fit_gate_passed": row.get(
            "fit_gate_passed", row.get("passes_primary_gate_15mV_3runs", False)
        ),
        "lli_percent": row.get("lli_percent", ""),
        "lam_anode_percent": row.get("lam_anode_percent", ""),
        "lam_cathode_percent": row.get("lam_cathode_percent", ""),
        "negative_estimate_flags": row.get("negative_estimate_flags", ""),
        "parameter_at_bound_flags": row.get(
            "parameter_at_bound_flags",
            _parameter_bound_flags(row.get("parameter_bounds_json", "[]")),
        ),
        "parameter_std_json": row.get("parameter_std_json", "null"),
        "within_fit_parameter_stability_status": (
            "within_optimizer_accepted_runs_parameter_std_recorded"
            if row.get("parameter_std_json") not in {"", "null", None}
            else "unavailable"
        ),
        "multi_seed_stability_status": "not_repeated_across_seed_single_seed_211",
        "baseline_reference_consistency": row.get(
            "baseline_reference_status", "not_applicable_bol_reference"
        ),
        "degradation_estimate_status": estimate_status,
        "error": row.get("error", ""),
    }


def _build_direction_consistency(
    root: Path,
    eol_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    historical_path = (
        root
        / "experiments"
        / "results"
        / "data_audits"
        / "pydma_eol_audit_v1"
        / "pydma_negative_estimate_audit.csv"
    )
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in _read_csv(historical_path):
        key = (row["part"], row["physical_cell_id"], row["rpt"])
        groups.setdefault(key, []).append(
            {
                "direction": row["direction"],
                **{
                    name: row.get(source, "")
                    for name, source in zip(
                        _PERCENT_COLUMNS,
                        (
                            "LLI_percent_vs_BoL",
                            "LAM_anode_percent_vs_BoL",
                            "LAM_cathode_percent_vs_BoL",
                        ),
                    )
                },
                "accepted": row["diagnostic_label_status"]
                == "diagnostic_estimate_accepted",
            }
        )
    output = []
    for (part, cell, rpt), records in sorted(groups.items()):
        output.append(_consistency_row(part, cell, rpt, records, "existing_pydma"))

    eol_groups: dict[str, list[dict[str, Any]]] = {}
    for row in eol_rows:
        eol_groups.setdefault(row.get("checkpoint_id", ""), []).append(
            {
                "direction": row.get("direction", ""),
                **{name: row.get(name, "") for name in _PERCENT_COLUMNS},
                "accepted": row.get("degradation_estimate_status")
                == "diagnostic_estimate_accepted",
            }
        )
    for checkpoint, records in sorted(eol_groups.items()):
        first = next(row for row in eol_rows if row.get("checkpoint_id") == checkpoint)
        output.append(
            _consistency_row(
                first["part"],
                first["physical_cell_id"],
                checkpoint,
                records,
                "eol_checkpoint",
            )
        )
    return output


def _consistency_row(
    part: str,
    cell: str,
    checkpoint: str,
    records: list[dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    accepted = {row["direction"]: row for row in records if row["accepted"]}
    charge = accepted.get("charge")
    discharge = accepted.get("discharge")
    values = {}
    for metric in _PERCENT_COLUMNS:
        charge_value = _float_or_none(charge.get(metric)) if charge else None
        discharge_value = _float_or_none(discharge.get(metric)) if discharge else None
        values[f"{metric}_charge"] = (
            charge_value if charge_value is not None else ""
        )
        values[f"{metric}_discharge"] = (
            discharge_value if discharge_value is not None else ""
        )
        values[f"{metric}_absolute_difference"] = (
            abs(charge_value - discharge_value)
            if charge_value is not None and discharge_value is not None
            else ""
        )
    return {
        "source": source,
        "part": part,
        "physical_cell_id": cell,
        "rpt_or_checkpoint": checkpoint,
        "charge_estimate_status": (
            "accepted" if charge else "no_accepted_charge_estimate"
        ),
        "discharge_estimate_status": (
            "accepted" if discharge else "no_accepted_discharge_estimate"
        ),
        "consistency_status": (
            "difference_reported_no_acceptance_threshold"
            if charge and discharge
            else "single_or_no_accepted_direction"
        ),
        **values,
    }


def _float_or_none(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _build_dashboard_records(
    output_dir: Path,
    fit_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    manifest = _read_csv(output_dir / "eol_checkpoint_manifest.csv")
    predictions = {
        row["checkpoint_id"]: row
        for row in _read_csv(output_dir / "eol_model_predictions.csv")
    }
    fit_by_key = {
        (row.get("checkpoint_id", ""), row.get("direction", "")): row
        for row in fit_rows
    }
    modality = {
        row["checkpoint_id"]: row
        for row in _read_csv(output_dir / "eol_modality_inference_audit.csv")
    }
    output = []
    for item in manifest:
        prediction = predictions[item["checkpoint_id"]]
        availability = modality[item["checkpoint_id"]]
        for direction, step in _FIT_DIRECTIONS:
            fit = fit_by_key.get((item["checkpoint_id"], direction), {})
            output.append(
                {
                    "checkpoint_id": item["checkpoint_id"],
                    "part_label": item["part_label"],
                    "physical_cell_id": item["physical_cell_id"],
                    "guide_rpt_assignment": item["guide_rpt_assignment"],
                    "direction": direction,
                    "modality_status": availability["available_modalities"] or "none",
                    "predicted_soh_percent": prediction["predicted_soh_percent"],
                    "raw_model_output_soh_percent": prediction[
                        "raw_model_output_soh_percent"
                    ],
                    "prediction_plausibility_status": prediction[
                        "prediction_plausibility_status"
                    ],
                    "provisional_reference_soh_percent": prediction[
                        "provisional_reference_soh_percent"
                    ],
                    "soh_status": prediction["soh_status"],
                    "fit_status": fit.get(
                        "degradation_estimate_status",
                        "not_run_curve_unavailable_or_ambiguous",
                    ),
                    "fit_rmse_mV": fit.get("rmse_mV", fit.get("fit_rmse_mV", "")),
                    "accepted_runs": fit.get("accepted_runs", ""),
                    "lli_percent": fit.get("lli_percent", ""),
                    "lam_anode_percent": fit.get("lam_anode_percent", ""),
                    "lam_cathode_percent": fit.get("lam_cathode_percent", ""),
                    "negative_estimate_flags": fit.get(
                        "negative_estimate_flags", ""
                    ),
                    "parameter_at_bound_flags": fit.get(
                        "parameter_at_bound_flags",
                        _parameter_bound_flags(
                            fit.get("parameter_bounds_json", "[]")
                        ),
                    ),
                    "baseline_reference_status": item[
                        f"baseline_step{step}_status"
                    ],
                    "mapping_status": item["mapping_status"],
                }
            )
    return output


def _pydma_quality_report(
    fit_rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    use_cuda: bool,
) -> str:
    attempted = [
        row
        for row in fit_rows
        if row.get("fit_status") != "not_run_curve_unavailable_or_ambiguous"
    ]
    accepted = [
        row
        for row in fit_rows
        if row.get("fit_gate_passed") is True
        or str(row.get("fit_gate_passed", "")).casefold() == "true"
    ]
    estimates = [
        row
        for row in fit_rows
        if row.get("degradation_estimate_status") == "diagnostic_estimate_accepted"
    ]
    negative_counts = {
        name: sum(
            _float_or_none(row.get(name)) is not None
            and float(row[name]) < 0
            for row in estimates
        )
        for name in _PERCENT_COLUMNS
    }
    return (
        "# EoL PyDMA fit quality diagnostics\n\n"
        f"- EoL direction records scheduled: {len(fit_rows)}; fits attempted: {len(attempted)}; "
        f"not run because curves were unavailable or ambiguous: {len(fit_rows) - len(attempted)}.\n"
        f"- EoL fits passing the strict gate (RMSE < 15 mV and at least 3 accepted runs): {len(accepted)}.\n"
        f"- EoL fit attempts rejected by the strict gate or fit failure: {len(attempted) - len(accepted)}.\n"
        f"- Same-cell, direction-matched BoL fits attempted: {len(baseline_rows)}.\n"
        f"- EoL degradation estimates available after both fit gates: {len(estimates)}.\n"
        f"- Negative accepted EoL estimates retained: LLI={negative_counts['lli_percent']}, "
        f"anode LAM={negative_counts['lam_anode_percent']}, "
        f"cathode LAM={negative_counts['lam_cathode_percent']}.\n"
        f"- CUDA optimizer used: {use_cuda}.\n\n"
        "Every estimate is a PyDMA model-derived diagnostic, not independent ground truth. "
        "No negative result was clipped or used to select a fit. Parameter-bound flags and "
        "within-fit parameter standard deviations are retained per fit. This EoL run uses "
        "one optimizer seed (211); it does not establish across-seed stability. The prior "
        "five-seed stability audit remains available in "
        "`../data_audits/pydma_eol_audit_v1/pydma_fit_stability.csv` for its three reviewed cases.\n\n"
        "Charge/discharge differences are descriptive only; no unapproved consistency "
        "threshold is applied. BoL comparisons use the same physical cell, same direction, "
        "and the verified direction-specific Step 8/10 curves. Capacity is stored in native "
        "Amphr units; absolute units remain unverified.\n"
    )


def _run_dashboard(root: Path, output_dir: Path) -> None:
    if not (output_dir / "eol_dashboard_records.csv").is_file():
        raise FileNotFoundError(
            "EoL dashboard records must be generated before rendering the dashboard."
        )
    dashboard_dir = output_dir / "dashboard"
    if dashboard_dir.exists():
        raise FileExistsError(f"Refusing to overwrite dashboard output: {dashboard_dir}")
    sys.path.insert(0, str(root / "src"))
    from bafuse.hybrid.hybrid_battery_health import run_pipeline

    run_pipeline(root, dashboard_dir)


def _write_final_report(root: Path, output_dir: Path) -> None:
    manifest = _read_csv(output_dir / "eol_checkpoint_manifest.csv")
    predictions = _read_csv(output_dir / "eol_model_predictions.csv")
    inference = _read_csv(output_dir / "eol_modality_inference_audit.csv")
    fits = _read_csv(output_dir / "eol_pydma_fits.csv")
    valid_predictions = [
        row for row in predictions if row["predicted_soh_percent"] != ""
    ]
    complete_modality_inputs = [
        row
        for row in inference
        if str(row["three_encoder_inference_eligible"]).casefold() == "true"
    ]
    provisional = [
        row
        for row in manifest
        if row["provisional_reference_soh_percent"] != ""
    ]
    accepted_fits = [
        row
        for row in fits
        if str(row.get("fit_gate_passed", "")).casefold() == "true"
    ]
    attempted_fits = [
        row
        for row in fits
        if row.get("fit_status") != "not_run_curve_unavailable_or_ambiguous"
    ]
    accepted_estimates = [
        row
        for row in fits
        if row.get("degradation_estimate_status")
        == "diagnostic_estimate_accepted"
    ]
    modality_by_id = {row["checkpoint_id"]: row for row in inference}
    prediction_by_id = {row["checkpoint_id"]: row for row in predictions}
    fit_by_id: dict[str, list[dict[str, str]]] = {}
    for row in fits:
        fit_by_id.setdefault(row.get("checkpoint_id", ""), []).append(row)
    lines = [
        "# EoL coverage and PyDMA diagnostics — v1",
        "",
        "## Scope and safeguards",
        "",
        "- New, versioned descriptive EoL analysis; canonical manifests, labels, splits, checkpoints, historical PyDMA outputs, and the existing dashboard were not overwritten.",
        "- Part 3 Guide assignments are retained as RPT6/RPT7; Part 2 endpoints remain unnumbered and use explicit EoL checkpoint identifiers.",
        "- SoH inference uses only complete, validated D+C+P at the same measured checkpoint and the frozen Phase 2 checkpoint. No model was retrained, tuned, or selected on EoL data.",
        "- Capacity-ratio references are candidate values only, not official SoH labels. They require a valid Step 4 target and the verified same-cell BoL Step 4 reference.",
        "- PyDMA uses direction-matched Step 8/10 curves and same-cell BoL fits. Estimates are diagnostic, not ground truth; negative estimates are preserved.",
        "- Group 5/6 are not introduced by this experiment. No ambiguous Part 2 EoL mapping was assigned an RPT number.",
        "",
        "## Coverage summary",
        "",
        f"- Guide-mapped EoL checkpoints audited: {len(manifest)} ({sum(row['part'] == '2' for row in manifest)} Part 2 unnumbered; {sum(row['part'] == '3' for row in manifest)} Part 3).",
        f"- Complete D+C+P inputs passed to the frozen checkpoint: {len(complete_modality_inputs)} / {len(manifest)}.",
        f"- Nonnegative frozen-checkpoint outputs retained as SoH predictions: {len(valid_predictions)}; negative raw outputs are preserved separately and withheld as physically incompatible with a positive capacity ratio.",
        f"- Provisional capacity-ratio reference candidates: {len(provisional)} / {len(manifest)}.",
        f"- EoL PyDMA direction records scheduled: {len(fits)}; fits attempted: {len(attempted_fits)}; curves not fit: {len(fits) - len(attempted_fits)}.",
        f"- EoL direction fits passing the strict RMSE < 15 mV / >= 3 accepted-run gate: {len(accepted_fits)}.",
        f"- EoL fit attempts rejected by the strict gate or fit failure: {len(attempted_fits) - len(accepted_fits)}.",
        f"- Degradation estimates with both target and same-direction BoL fits passing the gate: {len(accepted_estimates)}.",
        f"- Frozen Phase 2 checkpoint SHA-256: `{_sha256(FROZEN_CHECKPOINT)}`.",
        "",
        "## Per-checkpoint evidence",
        "",
        "| Checkpoint | Part | Cell | Guide RPT | Step 4 / 8 / 10 | D+C+P | Predicted SoH (%) | Raw model output (%) | Provisional reference (%) | Charge / discharge PyDMA estimates |",
        "|---|---|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in manifest:
        checkpoint = row["checkpoint_id"]
        prediction = prediction_by_id[checkpoint]
        modality = modality_by_id[checkpoint]
        directional = fit_by_id.get(checkpoint, [])
        fit_text = []
        for direction in ("charge", "discharge"):
            record = next(
                (item for item in directional if item.get("direction") == direction),
                None,
            )
            if record and record.get("degradation_estimate_status") == "diagnostic_estimate_accepted":
                fit_text.append(
                    f"{direction}: RMSE {record.get('rmse_mV', '')} mV, "
                    f"LLI {record.get('lli_percent', '')}%, "
                    f"Anode LAM {record.get('lam_anode_percent', '')}%, "
                    f"Cathode LAM {record.get('lam_cathode_percent', '')}%"
                )
            elif record:
                fit_text.append(
                    f"{direction}: {record.get('degradation_estimate_status', '')}"
                )
            else:
                fit_text.append(f"{direction}: not attempted")
        soh = prediction["predicted_soh_percent"] or "unavailable"
        raw_soh = prediction["raw_model_output_soh_percent"] or "unavailable"
        ref = row["provisional_reference_soh_percent"] or "unavailable"
        complete = "yes" if modality["three_encoder_inference_eligible"] == "True" else "no"
        lines.append(
            f"| {checkpoint} | Part {row['part']} | {row['physical_cell_id']} | "
            f"{row['guide_rpt_assignment'] or 'unnumbered'} | "
            f"{row['step4_status']} / {row['step8_status']} / {row['step10_status']} | "
            f"{complete} | {soh} | {raw_soh} | {ref} | {'; '.join(fit_text)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and limitations",
            "",
            "- A mapped EoL procedure is not automatically a valid model observation, SoH label, or accepted PyDMA degradation estimate. Eligibility was evaluated independently for each modality/direction.",
            "- Step 4, Step 8, and Step 10 boundaries are recorded as one-based CSV line ranges in `eol_curve_audit.csv`; Step 8/10 components are joined only across rows with missing documented Step/time/capacity/current/voltage/temperature fields and strictly increasing capacity, TestTime, and StepTime. `VARx*`/`FLGx*` fields are not used. Capacity spans are native `Amphr`, not asserted as Ah.",
            "- Part 3 capacity-ratio values remain provisional and descriptive. The dataset README does not define an official Part 3 SOH label.",
            "- A complete D+C+P prediction is an inference from operating state at that checkpoint, not a future-RPT forecast.",
            "- Parameter bounds and the optimizer's within-run parameter standard deviations are reported. This EoL run uses seed 211 once per fit; across-seed stability was not tested here.",
            "- Charge/discharge differences are reported without a pass/fail threshold. They must not be interpreted as physical consistency acceptance.",
            "",
            "## Reproducibility",
            "",
            "Use the repository CUDA environment used for the PyDMA optimizer:",
            "",
            "```powershell",
            "$env:PYTHONPATH='src'",
            r".\.venv-torch-cuda\Scripts\python.exe -m bafuse.data.eol_coverage_v1 --root . --phase all",
            r".\.venv-torch-cuda\Scripts\python.exe -m pytest tests\test_eol_coverage_v1.py -q",
            "```",
            "",
            "Run `--phase all` only with a fresh output directory. The phase commands refuse to overwrite existing result files; for staged execution use `--phase audit`, then `inference`, `pydma`, and `dashboard`.",
            "",
            "## Protected artifact integrity",
            "",
            "See `protected_artifact_integrity.csv`. Every file under the existing Phase 2/Phase 3 experiment tree and existing results tree, plus canonical labels/manifests/split and saved PyDMA label outputs, was SHA-256 compared before and after this experiment.",
            "",
            "## Output files",
            "",
            "- `eol_checkpoint_manifest.csv` — Guide identity, EoL status, and exact target/BoL segment references.",
            "- `eol_curve_audit.csv` — raw Step 4/8/10 segment boundaries and validity.",
            "- `eol_modality_inference_audit.csv` and `eol_model_predictions.csv` — strict D+C+P eligibility and frozen-checkpoint predictions.",
            "- `eol_pydma_fits.csv` and `eol_pydma_baseline_fits.csv` — EoL and same-cell BoL fit gates, parameters, bounds, and estimates.",
            "- `pydma_quality_diagnostics.csv` and `pydma_direction_consistency.csv` — historical and EoL quality flags, with unavailable parameter provenance marked explicitly.",
            "- `eol_dashboard_records.csv` and `dashboard/dashboard.html` — separate EoL dashboard section and its CSV source.",
            "- `pydma_eol_quality_report.md` — PyDMA gate and diagnostic summary.",
        ]
    )
    _write_text(output_dir / "eol_coverage_report.md", "\n".join(lines) + "\n")


def run_phase(
    phase: str,
    root: Path = ROOT,
    output_dir: Path = DEFAULT_OUTPUT,
) -> None:
    root = root.resolve()
    output_dir = output_dir.resolve()
    try:
        output_dir.relative_to(root)
    except ValueError as error:
        raise ValueError("EoL output must remain inside the repository.") from error
    _capture_before_hashes(root, output_dir)
    if phase in {"audit", "all"}:
        _build_eol_manifest(root, output_dir)
    if phase in {"inference", "all"}:
        _run_frozen_inference(root, output_dir)
    if phase in {"pydma", "all"}:
        _run_eol_pydma(root, output_dir)
    if phase in {"dashboard", "all"}:
        _run_dashboard(root, output_dir)
        _write_final_report(root, output_dir)
    _write_integrity_comparison(root, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit Guide-mapped EoL checkpoints with frozen BaFuse and PyDMA."
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase",
        choices=("audit", "inference", "pydma", "dashboard", "all"),
        default="all",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output_dir
    if not output.is_absolute():
        output = root / output
    run_phase(args.phase, root, output)


if __name__ == "__main__":
    main()
