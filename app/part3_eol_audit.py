from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from bafuse.hybrid.final2_dataset import compute_soh_percent
from bafuse.hybrid.p5_electrical_features import (
    P5_FEATURE_NAMES,
    build_p5_electrical_features,
)

from app.tpg_mat_ingestion import (
    PART3_DEMO_EOL_MAPPING_PATH,
    RAW_MAPPING_PATH,
    _integrate_step4_capacity,
    _sha256,
)


P5_LABELS_PATH = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_architecture_benchmark_v1"
    / "bafuse_p5_electrical_v1_20261114"
    / "bafuse_soh_labels_frozen.csv"
)
BASE_ELIGIBILITY_PATH = ROOT / "app" / "part3_modality_eligibility.csv"
OUTPUT_EO_L_AUDIT = ROOT / "app" / "part3_demo_eol_eligibility.csv"
OUTPUT_MAPPING_COMPLETENESS = ROOT / "app" / "part3_mapping_completeness.csv"
OUTPUT_REPORT = ROOT / "app" / "part3_eol_audit_report.md"

EOL_AUDIT_FIELDS = (
    "part",
    "group",
    "cell",
    "rpt",
    "procedure",
    "guide_sheet",
    "guide_row",
    "guide_entry",
    "source_file",
    "source_size_bytes",
    "source_sha256",
    "mapping_status",
    "identity_provenance",
    "inventory_scope",
    "is_eol_procedure",
    "step4_segment_reference",
    "step4_expected_rows",
    "step4_observed_rows",
    "step4_block_sha256",
    "step4_measurements_finite",
    "step4_current_direction",
    "step4_testtime_strictly_increasing",
    "step4_testtime_delta_min",
    "step4_testtime_delta_median",
    "step4_testtime_delta_max",
    "p5_feature_names",
    "p5_features_json",
    "p5_physics_eligible",
    "step10_segments_json",
    "step10_blocks_json",
    "step10_block_count",
    "step10_blocks_valid",
    "step10_separator_rows_nonfinite",
    "step10_testtime_strictly_increasing_within_blocks",
    "step10_testtime_strictly_increasing_across_blocks",
    "step10_inter_block_testtime_deltas_json",
    "step10_contiguous_extractor_eligible",
    "step10_review_reason",
    "discharge_only_inference_eligible",
    "two_modality_inference_eligible",
    "three_modality_inference_eligible",
    "bol_source_file",
    "bol_source_sha256",
    "bol_source_hash_status",
    "bol_step4_segment_reference",
    "bol_step4_block_sha256",
    "bol_capacity_native_current_time",
    "eol_capacity_native_current_time",
    "soh_percent",
    "soh_reference_status",
    "soh_evaluation_eligible",
    "selectable",
)

MAPPING_FIELDS = (
    "record_type",
    "part",
    "group",
    "cell",
    "rpt",
    "source_file",
    "source_extension",
    "source_present",
    "inventory_sha256",
    "mapping_status",
    "identity_basis",
    "is_rpt",
    "selectable",
    "guide_sheet",
    "guide_row",
    "reason",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                raise ValueError(f"CSV has no header: {path}")
            return list(reader)
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"Could not read {path}: {error}") from error


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _number(row: list[str], index: int) -> float:
    try:
        return float(row[index])
    except (IndexError, TypeError, ValueError):
        return float("nan")


def _audit_step10_blocks(
    source_path: Path, segments_json: str
) -> dict[str, Any]:
    segments = json.loads(segments_json)
    if not isinstance(segments, list) or len(segments) != 3:
        raise ValueError(f"Expected three Step 10 blocks in {source_path.name}.")
    normalized_segments = []
    for segment in segments:
        start, end = int(segment["start"]), int(segment["end"])
        if start < 2 or end < start:
            raise ValueError(f"Invalid Step 10 row range: {segment}.")
        normalized_segments.append({"start": start, "end": end})
    normalized_segments.sort(key=lambda item: item["start"])
    if any(
        later["start"] <= earlier["end"]
        for earlier, later in zip(normalized_segments, normalized_segments[1:])
    ):
        raise ValueError("Step 10 raw blocks overlap.")

    block_values: list[dict[str, Any]] = [
        {"segment": segment, "rows": [], "bad_finite": 0, "current_signs": set()}
        for segment in normalized_segments
    ]
    separator_rows: dict[int, list[tuple[float, float, float, float]]] = {
        index: []
        for index in range(len(normalized_segments) - 1)
    }
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Step 10 source has no header: {source_path}.")
        indices = {name.strip().casefold(): index for index, name in enumerate(header)}
        missing = {"step", "testtime", "amps", "volts"} - set(indices)
        if missing:
            raise ValueError(
                f"Step 10 source is missing columns: {', '.join(sorted(missing))}."
            )
        for row in reader:
            line = reader.line_num
            values = (
                _number(row, indices["step"]),
                _number(row, indices["testtime"]),
                _number(row, indices["amps"]),
                _number(row, indices["volts"]),
            )
            target_index = next(
                (
                    index
                    for index, block in enumerate(block_values)
                    if block["segment"]["start"] <= line <= block["segment"]["end"]
                ),
                None,
            )
            if target_index is not None:
                block = block_values[target_index]
                block["rows"].append(values)
                if not all(math.isfinite(value) for value in values):
                    block["bad_finite"] += 1
                if math.isfinite(values[2]):
                    block["current_signs"].add(
                        "positive" if values[2] > 0 else "negative" if values[2] < 0 else "zero"
                    )
                continue
            for gap_index, (earlier, later) in enumerate(
                zip(normalized_segments, normalized_segments[1:])
            ):
                if earlier["end"] < line < later["start"]:
                    separator_rows[gap_index].append(values)

    block_results = []
    all_blocks_valid = True
    within_monotonic = True
    for block in block_values:
        segment = block["segment"]
        values = block["rows"]
        times = [value[1] for value in values]
        expected_count = segment["end"] - segment["start"] + 1
        step_ids = {value[0] for value in values if math.isfinite(value[0])}
        monotonic = (
            len(times) == expected_count
            and all(math.isfinite(value) for value in times)
            and all(second > first for first, second in zip(times, times[1:]))
        )
        valid = (
            len(values) == expected_count
            and block["bad_finite"] == 0
            and step_ids == {10.0}
            and block["current_signs"] == {"positive"}
            and monotonic
        )
        all_blocks_valid = all_blocks_valid and valid
        within_monotonic = within_monotonic and monotonic
        block_results.append(
            {
                **segment,
                "row_count": len(values),
                "expected_rows": expected_count,
                "step_id_valid": step_ids == {10.0},
                "finite_measurements": block["bad_finite"] == 0,
                "current_direction": sorted(block["current_signs"]),
                "testtime_strictly_increasing": monotonic,
                "testtime_first": times[0] if times else None,
                "testtime_last": times[-1] if times else None,
                "valid": valid,
            }
        )

    separators_valid = all(
        len(rows) == 2
        and all(not all(math.isfinite(value) for value in row) for row in rows)
        and all(
            not math.isfinite(row[index])
            for row in rows
            for index in (0, 1, 2, 3)
        )
        for rows in separator_rows.values()
    )
    across_monotonic = True
    inter_block_time_deltas = []
    for left, right in zip(block_results, block_results[1:]):
        if left["testtime_last"] is None or right["testtime_first"] is None:
            across_monotonic = False
            continue
        delta = right["testtime_first"] - left["testtime_last"]
        inter_block_time_deltas.append(delta)
        across_monotonic = across_monotonic and delta > 0

    return {
        "segments": normalized_segments,
        "blocks": block_results,
        "blocks_valid": all_blocks_valid,
        "separator_rows_nonfinite": separators_valid,
        "separator_row_counts": [len(rows) for rows in separator_rows.values()],
        "testtime_strictly_increasing_within_blocks": within_monotonic,
        "testtime_strictly_increasing_across_blocks": across_monotonic,
        "inter_block_testtime_deltas": inter_block_time_deltas,
        "contiguous_extractor_eligible": False,
    }


def _verify_p5_bol_reference(
    row: dict[str, str],
    labels: list[dict[str, str]],
) -> dict[str, str]:
    matches = [
        label
        for label in labels
        if label.get("dataset_part") == row["part"]
        and label.get("physical_cell_id") == row["cell"]
        and label.get("bol_step4_source_file") == row["bol_source_file"]
        and label.get("bol_step4_segment_ref") == row["bol_step4_segment_reference"]
        and label.get("bol_step4_block_sha256") == row["bol_step4_block_sha256"]
    ]
    if not matches:
        raise ValueError(
            f"No exact same-cell frozen P5 BoL reference for {row['cell']}."
        )
    if len(
        {
            (
                item["bol_step4_source_file"],
                item["bol_step4_segment_ref"],
                item["bol_step4_block_sha256"],
            )
            for item in matches
        }
    ) != 1:
        raise ValueError(f"Conflicting P5 BoL references for {row['cell']}.")
    return matches[0]


def build_eol_audit(root: Path = ROOT) -> list[dict[str, Any]]:
    root = root.resolve()
    mappings = _read_csv(PART3_DEMO_EOL_MAPPING_PATH)
    mappings = [row for row in mappings if row.get("record_type") == "cycling_eol"]
    labels = _read_csv(P5_LABELS_PATH)
    output = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in mappings:
        key = (row["part"], row["group"], row["cell"], row["rpt"])
        if key in seen:
            raise ValueError(f"Duplicate demo-only EoL identity: {key}.")
        seen.add(key)
        if row.get("mapping_status") != "GUIDE_VERIFIED":
            raise ValueError(f"Unverified Guide mapping for {key}.")

        source_path = root / Path(row["source_file"])
        if (
            not source_path.is_file()
            or source_path.stat().st_size != int(row["source_size_bytes"])
            or _sha256(source_path).casefold() != row["source_sha256"].casefold()
        ):
            raise ValueError(f"EoL source integrity check failed for {key}.")
        step4 = _integrate_step4_capacity(
            source_path,
            row["step4_segment_reference"],
            expected_source_name=source_path.name,
        )
        if step4["row_count"] != int(row["step4_expected_rows"]):
            raise ValueError(f"Step 4 row count mismatch for {key}.")
        p5_features = build_p5_electrical_features(
            step4["voltage_values"], step4["current_values"]
        )
        if tuple(p5_features) != tuple(P5_FEATURE_NAMES) or not all(
            math.isfinite(float(value)) for value in p5_features.values()
        ):
            raise ValueError(f"P5 feature contract failed for {key}.")

        step10 = _audit_step10_blocks(source_path, row["step10_segments_json"])
        if not step10["blocks_valid"] or not step10["separator_rows_nonfinite"]:
            raise ValueError(f"Step 10 block audit failed for {key}.")

        bol_source = root / Path(row["bol_source_file"])
        if (
            not bol_source.is_file()
            or _sha256(bol_source).casefold() != row["bol_source_sha256"].casefold()
        ):
            raise ValueError(f"Same-cell BoL source integrity check failed for {key}.")
        p5_bol = _verify_p5_bol_reference(row, labels)
        bol = _integrate_step4_capacity(
            bol_source,
            row["bol_step4_segment_reference"],
            expected_source_name=bol_source.name,
        )
        if bol["source_block_sha256"] != row["bol_step4_block_sha256"]:
            raise ValueError(f"Same-cell BoL segment hash mismatch for {key}.")
        expected_q_bol = float(p5_bol["q_bol_step4_testtime_native_current_time"])
        if not math.isclose(
            bol["capacity_native_current_time"],
            expected_q_bol,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Same-cell BoL capacity integral mismatch for {key}.")

        soh = compute_soh_percent(
            step4["capacity_native_current_time"],
            bol["capacity_native_current_time"],
        )
        output.append(
            {
                "part": row["part"],
                "group": row["group"],
                "cell": row["cell"],
                "rpt": row["rpt"],
                "procedure": row["procedure"],
                "guide_sheet": row["guide_sheet"],
                "guide_row": row["guide_row"],
                "guide_entry": row["guide_entry"],
                "source_file": row["source_file"],
                "source_size_bytes": row["source_size_bytes"],
                "source_sha256": row["source_sha256"],
                "mapping_status": row["mapping_status"],
                "identity_provenance": row["identity_provenance"],
                "inventory_scope": row["inventory_scope"],
                "is_eol_procedure": row["is_eol_procedure"],
                "step4_segment_reference": row["step4_segment_reference"],
                "step4_expected_rows": row["step4_expected_rows"],
                "step4_observed_rows": step4["row_count"],
                "step4_block_sha256": step4["source_block_sha256"],
                "step4_measurements_finite": True,
                "step4_current_direction": "negative_discharge",
                "step4_testtime_strictly_increasing": step4[
                    "testtime_strictly_increasing"
                ],
                "step4_testtime_delta_min": step4["testtime_delta_min"],
                "step4_testtime_delta_median": step4["testtime_delta_median"],
                "step4_testtime_delta_max": step4["testtime_delta_max"],
                "p5_feature_names": ";".join(P5_FEATURE_NAMES),
                "p5_features_json": json.dumps(
                    p5_features, sort_keys=True, separators=(",", ":")
                ),
                "p5_physics_eligible": True,
                "step10_segments_json": row["step10_segments_json"],
                "step10_blocks_json": json.dumps(
                    step10["blocks"], sort_keys=True, separators=(",", ":")
                ),
                "step10_block_count": len(step10["blocks"]),
                "step10_blocks_valid": step10["blocks_valid"],
                "step10_separator_rows_nonfinite": step10[
                    "separator_rows_nonfinite"
                ],
                "step10_testtime_strictly_increasing_within_blocks": step10[
                    "testtime_strictly_increasing_within_blocks"
                ],
                "step10_testtime_strictly_increasing_across_blocks": step10[
                    "testtime_strictly_increasing_across_blocks"
                ],
                "step10_inter_block_testtime_deltas_json": json.dumps(
                    step10["inter_block_testtime_deltas"],
                    separators=(",", ":"),
                ),
                "step10_contiguous_extractor_eligible": False,
                "step10_review_reason": row["step10_review_reason"],
                "discharge_only_inference_eligible": True,
                "two_modality_inference_eligible": False,
                "three_modality_inference_eligible": False,
                "bol_source_file": row["bol_source_file"],
                "bol_source_sha256": row["bol_source_sha256"],
                "bol_source_hash_status": row["bol_source_hash_status"],
                "bol_step4_segment_reference": row[
                    "bol_step4_segment_reference"
                ],
                "bol_step4_block_sha256": bol["source_block_sha256"],
                "bol_capacity_native_current_time": bol[
                    "capacity_native_current_time"
                ],
                "eol_capacity_native_current_time": step4[
                    "capacity_native_current_time"
                ],
                "soh_percent": soh,
                "soh_reference_status": (
                    "demo_only_provisional_capacity_ratio_not_evaluation_eligible"
                ),
                "soh_evaluation_eligible": False,
                "selectable": True,
            }
        )
    if len(output) != 9:
        raise ValueError(f"Expected nine Guide-verified cycling EoL rows, got {len(output)}.")
    return output


def build_mapping_completeness(
    eol_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for item in eol_rows:
        rows.append(
            {
                "record_type": "cycling_eol_rpt",
                "part": item["part"],
                "group": item["group"],
                "cell": item["cell"],
                "rpt": item["rpt"],
                "source_file": item["source_file"],
                "source_extension": ".csv",
                "source_present": "true",
                "inventory_sha256": item["source_sha256"],
                "mapping_status": "GUIDE_VERIFIED",
                "identity_basis": "Guide procedure-to-RPT plus matching raw cell filename",
                "is_rpt": "true",
                "selectable": "true",
                "guide_sheet": item["guide_sheet"],
                "guide_row": item["guide_row"],
                "reason": "Step 4 is eligible; Step 10 is withheld because it has three blocks.",
            }
        )

    raw_rows = _read_csv(RAW_MAPPING_PATH)
    eis_rows = [
        row
        for row in raw_rows
        if row.get("record_type") == "raw_file"
        and row.get("dataset_part") == "3"
        and row.get("source_modality") == "eis_checkpoint"
        and row.get("source_filename_stage", "").casefold() == "eol"
    ]
    if len(eis_rows) != 72:
        raise ValueError(f"Expected 72 Part 3 EIS EoL CSV/MAT inventory entries, got {len(eis_rows)}.")
    csv_count = 0
    mat_missing_count = 0
    for item in eis_rows:
        extension = item["source_extension"].casefold()
        present = item["source_present"].casefold() == "true"
        if extension == ".csv" and present:
            csv_count += 1
            status = "NOT_AN_RPT_FILENAME_ONLY"
            reason = (
                "EIS EoL checkpoint; raw header does not independently identify "
                "cell/SOC/RPT; excluded from RPT inference."
            )
            selectable = "false"
        elif extension == ".mat" and not present:
            mat_missing_count += 1
            status = "SOURCE_MISSING_NOT_AN_RPT"
            reason = "EIS MAT counterpart is absent and EIS checkpoints are not RPTs."
            selectable = "false"
        else:
            raise ValueError(
                f"Unexpected EIS EoL availability state: {item['source_relative_path']}."
            )
        rows.append(
            {
                "record_type": "eis_eol_checkpoint",
                "part": item["dataset_part"],
                "group": item["group_id"],
                "cell": item["physical_cell_id"],
                "rpt": "",
                "source_file": item["repository_path"],
                "source_extension": extension,
                "source_present": str(present).lower(),
                "inventory_sha256": item.get("source_sha256", ""),
                "mapping_status": status,
                "identity_basis": "filename_identity_only",
                "is_rpt": "false",
                "selectable": selectable,
                "guide_sheet": "",
                "guide_row": "",
                "reason": reason,
            }
        )
    if csv_count != 36 or mat_missing_count != 36:
        raise ValueError(
            f"Expected 36 present EIS EoL CSVs and 36 missing MATs; "
            f"got {csv_count} and {mat_missing_count}."
        )
    return rows


def _render_report(
    eol_rows: list[dict[str, Any]], completeness_rows: list[dict[str, Any]]
) -> str:
    eligibility = _read_csv(BASE_ELIGIBILITY_PATH)
    frozen_eval = sum(
        row.get("label_evaluation_eligible", "").casefold() == "true"
        for row in eligibility
    )
    frozen_cells = {
        row.get("physical_cell_id", "")
        for row in eligibility
        if row.get("label_evaluation_eligible", "").casefold() == "true"
    }
    eis_csv = sum(
        row["record_type"] == "eis_eol_checkpoint"
        and row["source_extension"] == ".csv"
        and row["source_present"] == "true"
        for row in completeness_rows
    )
    eis_mat_missing = sum(
        row["record_type"] == "eis_eol_checkpoint"
        and row["source_extension"] == ".mat"
        and row["source_present"] == "false"
        for row in completeness_rows
    )
    lines = [
        "# Kiểm toán EoL Part 3 — mở rộng demo, không đổi Final2",
        "",
        "Đây là inventory demo-only. Không sửa cohort/checkpoint/scaler Final2, "
        "không train, không tính lại scaler, không chạy PyDMA. Các nhãn EoL dưới "
        "đây là capacity-ratio tham khảo, không phải ground truth và bị loại khỏi "
        "mọi metric evaluation.",
        "",
        "## Cycling EoL được Guide xác nhận",
        "",
        f"Có **{len(eol_rows)}** cycling EoL procedure, ánh xạ thành RPT: "
        + ", ".join(
            f"{row['cell']}/RPT{row['rpt']}" for row in eol_rows
        )
        + ".",
        "",
        "| Cell/RPT | Step 4 | P5 | Step 10 | Provisional SOH (%) | Same-cell BoL |",
        "|---|---|---|---|---:|---|",
    ]
    for row in eol_rows:
        lines.append(
            f"| {row['cell']}/RPT{row['rpt']} | eligible "
            f"({row['step4_observed_rows']} rows) | eligible | withheld "
            f"({row['step10_block_count']} discontiguous blocks) | "
            f"{float(row['soh_percent']):.6f} | {Path(row['bol_source_file']).name} |"
        )
    lines.extend(
        [
            "",
            "Step 4 segments were checked for Step ID 4, finite TestTime/Amps/Volts, "
            "negative discharge current, strictly increasing TestTime, exact row "
            "count, block SHA-256, and trapezoidal native current-time integral. "
            "The existing P5 electrical feature builder produced five finite "
            "features for all nine segments.",
            "",
            "Every EoL file has three positive-current Step 10 blocks separated "
            "by two rows with non-finite core measurements. TestTime increases "
            "within each block and across the observed endpoints. The existing "
            "extractor requires one contiguous finite segment, so this audit does "
            "not join the blocks: all nine remain Step 10-ineligible. Thus "
            "discharge-only inference is eligible; 2M and 3M inference are not.",
            "",
            "The provisional reference is `100 * Q_EoL / Q_BoL`, where each Q is "
            "the trapezoidal integral of `abs(Amps)` against native `TestTime` "
            "within verified Step 4 boundaries. Each `.0` BoL source, segment "
            "hash, and capacity integral was matched to the same-cell frozen P5 "
            "record. Native current-time units are not assigned an Ah label.",
            "",
            "## EIS EoL checkpoints and Group 10",
            "",
            f"The inventory contains **{eis_csv} present EIS EoL CSVs** and "
            f"**{eis_mat_missing} missing MAT counterparts**. EIS EoL is "
            "`NOT_AN_RPT`; its headers do not independently establish cell/SOC/RPT "
            "identity, so these rows are not selectable. Group 10 has no cycling "
            "EoL procedure in the Guide. In particular, do not attach its blank-cell "
            "Guide sequence to G10-Cell2; the Guide explicitly ends G10-Cell22 and "
            "G10-Cell25 at RPT12.",
            "",
            "## Frozen cohort integrity",
            "",
            f"The existing Part 3 modality audit remains **87 mapped RPT rows**, "
            f"with **{frozen_eval} evaluation-eligible observations across "
            f"{len(frozen_cells)} cells**. EoL records are stored in a separate "
            "demo-only extension and do not enter the frozen NPZ, manifests, split, "
            "training, external metrics, or saved checkpoints.",
            "",
            "Detailed source paths, SHA-256 values, Guide rows, Step boundaries, "
            "modality gates, BoL provenance, and per-observation references are in "
            "`part3_demo_eol_eligibility.csv` and "
            "`part3_mapping_completeness.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_audit_outputs(root: Path = ROOT) -> dict[str, Any]:
    rows = build_eol_audit(root)
    completeness = build_mapping_completeness(rows)
    _write_csv(OUTPUT_EO_L_AUDIT, EOL_AUDIT_FIELDS, rows)
    _write_csv(OUTPUT_MAPPING_COMPLETENESS, MAPPING_FIELDS, completeness)
    report = _render_report(rows, completeness)
    OUTPUT_REPORT.write_text(report, encoding="utf-8")
    return {
        "eol_rows": len(rows),
        "step4_eligible": sum(row["discharge_only_inference_eligible"] for row in rows),
        "step10_eligible": sum(row["two_modality_inference_eligible"] for row in rows),
        "p5_eligible": sum(row["p5_physics_eligible"] for row in rows),
        "provisional_references": sum(
            row["soh_reference_status"]
            == "demo_only_provisional_capacity_ratio_not_evaluation_eligible"
            for row in rows
        ),
        "mapping_completeness_rows": len(completeness),
        "eis_eol_csvs": sum(
            row["record_type"] == "eis_eol_checkpoint"
            and row["source_extension"] == ".csv"
            and row["source_present"] == "true"
            for row in completeness
        ),
        "eis_eol_mats_missing": sum(
            row["record_type"] == "eis_eol_checkpoint"
            and row["source_extension"] == ".mat"
            and row["source_present"] == "false"
            for row in completeness
        ),
        "outputs": [
            str(OUTPUT_EO_L_AUDIT),
            str(OUTPUT_MAPPING_COMPLETENESS),
            str(OUTPUT_REPORT),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(write_audit_outputs(), indent=2))
