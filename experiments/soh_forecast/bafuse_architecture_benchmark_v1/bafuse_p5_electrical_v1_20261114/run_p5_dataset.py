from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import shutil
import statistics
import sys
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from bafuse.hybrid.architecture_models import ARCHITECTURE_NAMES, build_architecture
from bafuse.hybrid.p5_electrical_features import (
    P5_FEATURE_NAMES,
    build_p5_electrical_features,
)
from bafuse.hybrid.soh_fusion import resample_curve
from bafuse.labeling import raw_soh_reconstruction as raw


P5_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = P5_DIR.parents[3]
BENCHMARK_DIR = PROJECT_ROOT / "experiments" / "soh_forecast" / "bafuse_architecture_benchmark_v1"
LABELS_PATH = (
    BENCHMARK_DIR
    / "bafuse_soh_trajectory_audit_v1_20261114"
    / "bafuse_soh_labels_203.csv"
)
CANDIDATES_PATH = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113" / "final_candidate_manifest.csv"
ALIGNMENT_PATH = (
    BENCHMARK_DIR
    / "bafuse_feature_freeze_blocker_v1_20261114"
    / "bafuse_alignment_audit.csv"
)
BLOCKER_MANIFEST_PATH = (
    BENCHMARK_DIR
    / "bafuse_feature_freeze_blocker_v1_20261114"
    / "bafuse_feature_audit_sha256.json"
)
SOURCE_HASHES_PATH = (
    BENCHMARK_DIR
    / "bafuse_feature_freeze_blocker_v1_20261114"
    / "bafuse_raw_source_hashes.csv"
)
FEATURE_BLOCKER_CONFIG_PATH = (
    BENCHMARK_DIR
    / "bafuse_feature_freeze_blocker_v1_20261114"
    / "bafuse_feature_audit_config.json"
)
OLD_MATRIX_PATH = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113" / "raw_curve_feature_matrix.csv"
CELL_SPLIT_PATH = BENCHMARK_DIR / "cell_split.csv"
LOCK_PATH = PROJECT_ROOT / "requirements-lock-cuda-cu126.txt"
RUN_CONFIG_PATH = P5_DIR / "p5_run_config.json"
OUTPUT_LABEL = P5_DIR / "bafuse_soh_labels_frozen.csv"

EXPECTED_LABEL_SHA256 = "c7dd659f4a73d35f777b33125349dd2695564ad33ed5b885f6261abfc9f5c915"
EXPECTED_BLOCKER_SHA256 = "2f71e806bb8910b3ebd73350dbf1188988d58acc40f414514d0d1bba0329f7c7"
EXPECTED_COUNTS = {"train": 102, "validation": 37, "external_test": 64}
TARGET_DEFINITION = (
    "100 * Q_Step4_discharge(RPT) / Q_Step4_discharge(same-cell BoL)"
)
KEY_FIELDS = ("dataset_part", "physical_cell_id", "rpt_index")
PARTITION_ORDER = {"train": 0, "validation": 1, "external_test": 2}
SPLIT_FILE_NAMES = {
    "train": "bafuse_train_p5.npz",
    "validation": "bafuse_validation_p5.npz",
    "external_test": "bafuse_external_p5.npz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_json_atomic(path: Path, content: dict[str, Any]) -> None:
    _write_text_atomic(
        path,
        json.dumps(content, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    )


def _write_csv_atomic(
    path: Path, fieldnames: list[str], rows: list[dict[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with zipfile.ZipFile(
            temporary,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=9,
            strict_timestamps=False,
        ) as archive:
            for name in sorted(arrays):
                array = np.asarray(arrays[name])
                if array.dtype.hasobject:
                    raise TypeError(f"Object arrays are not allowed in NPZ: {name}")
                payload = io.BytesIO()
                np.lib.format.write_array(
                    payload, array, version=(1, 0), allow_pickle=False
                )
                entry = zipfile.ZipInfo(
                    filename=f"{name}.npy",
                    date_time=(1980, 1, 1, 0, 0, 0),
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                entry.external_attr = 0o600 << 16
                entry.create_system = 3
                archive.writestr(
                    entry,
                    payload.getvalue(),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _key(row: dict[str, str]) -> tuple[int, str, int]:
    return (
        int(row["dataset_part"]),
        row["physical_cell_id"],
        int(row["rpt_index"]),
    )


def _normalized_source(source: str) -> str:
    return source.replace("\\", "/").casefold()


def _verify_approved_inputs() -> None:
    label_hash = _sha256(LABELS_PATH)
    blocker_hash = _sha256(BLOCKER_MANIFEST_PATH)
    if label_hash != EXPECTED_LABEL_SHA256:
        raise ValueError(
            f"Authoritative label SHA-256 changed: expected "
            f"{EXPECTED_LABEL_SHA256}, got {label_hash}."
        )
    if blocker_hash != EXPECTED_BLOCKER_SHA256:
        raise ValueError(
            f"Feature-blocker manifest SHA-256 changed: expected "
            f"{EXPECTED_BLOCKER_SHA256}, got {blocker_hash}."
        )
    print(f"Verified approved labels: {label_hash}", flush=True)
    print(f"Verified feature-blocker manifest: {blocker_hash}", flush=True)


def _verify_cohort(
    labels: list[dict[str, str]],
    candidates: list[dict[str, str]],
    alignments: list[dict[str, str]],
    cell_splits: list[dict[str, str]],
) -> tuple[list[dict[str, str]], dict[tuple[int, str, int], dict[str, str]]]:
    if len(labels) != 203 or len({_key(row) for row in labels}) != 203:
        raise ValueError("Authoritative label table must contain 203 unique keys.")
    if any(row["final_status"] != "ACCEPTED" for row in labels):
        raise ValueError("The authoritative cohort contains a non-ACCEPTED label.")
    if any(row["target_definition"] != TARGET_DEFINITION for row in labels):
        raise ValueError("The approved SoH target definition changed.")
    if any(row["training_authorized"].lower() != "false" for row in labels):
        raise ValueError("Training authorization must remain false.")

    counts: dict[str, int] = defaultdict(int)
    cells_by_partition: dict[str, set[str]] = defaultdict(set)
    for row in labels:
        partition = row["partition"]
        if partition not in EXPECTED_COUNTS:
            raise ValueError(f"Unexpected partition in label table: {partition}")
        soh = float(row["soh_percent"])
        if not math.isfinite(soh):
            raise ValueError(f"Non-finite SoH label at {_key(row)}.")
        counts[partition] += 1
        cells_by_partition[partition].add(row["physical_cell_id"])
    if dict(counts) != EXPECTED_COUNTS:
        raise ValueError(f"Unexpected label split counts: {dict(counts)}.")
    if (
        cells_by_partition["train"] & cells_by_partition["validation"]
        or cells_by_partition["train"] & cells_by_partition["external_test"]
        or cells_by_partition["validation"] & cells_by_partition["external_test"]
    ):
        raise ValueError("A physical cell overlaps partitions.")

    split_by_cell: dict[str, str] = {}
    for row in cell_splits:
        cell = row["physical_cell_id"]
        partition = row["partition"]
        if cell in split_by_cell:
            raise ValueError(f"Duplicate physical-cell assignment: {cell}.")
        split_by_cell[cell] = partition
    label_cells = {row["physical_cell_id"] for row in labels}
    if not label_cells.issubset(split_by_cell):
        raise ValueError("A labeled physical cell is absent from the locked cell split.")

    candidate_by_key = {_key(row): row for row in candidates}
    alignment_by_key = {_key(row): row for row in alignments}
    if len(candidate_by_key) != len(candidates) or len(alignment_by_key) != len(alignments):
        raise ValueError("Candidate or alignment manifest contains duplicate keys.")
    if set(candidate_by_key) != {_key(row) for row in labels}:
        raise ValueError("Candidate manifest keys differ from the approved labels.")
    if set(alignment_by_key) != {_key(row) for row in labels}:
        raise ValueError("Alignment audit keys differ from the approved labels.")

    ordered = sorted(
        labels,
        key=lambda row: (
            PARTITION_ORDER[row["partition"]],
            int(row["dataset_part"]),
            row["physical_cell_id"],
            int(row["rpt_index"]),
        ),
    )
    for row in ordered:
        key = _key(row)
        if split_by_cell[row["physical_cell_id"]] != row["partition"]:
            raise ValueError(f"Locked split mismatch for {key}.")
        candidate = candidate_by_key[key]
        alignment = alignment_by_key[key]
        promoted = (
            row["original_status"] == "REVIEW"
            and row["promotion_status"] == "APPROVED_AND_PROMOTED"
            and row["final_status"] == "ACCEPTED"
        )
        if promoted:
            if not row["promotion_provenance"]:
                raise ValueError(f"Approved promotion lacks provenance at {key}.")
            if (
                row["promotion_evidence_rpt_file_sha256"]
                != row["rpt_source_file_sha256"]
                or row["promotion_evidence_bol_file_sha256"]
                != row["bol_source_file_sha256"]
            ):
                raise ValueError(f"Promotion evidence source hashes differ at {key}.")
        elif (
            row["final_status"] != "ACCEPTED"
            or candidate["label_disposition"] != "ACCEPTED"
            or candidate["final_eligibility_status"]
            != "ELIGIBLE_COMPLETE_THREE_MODALITY_RECORD"
        ):
            raise ValueError(
                f"Non-promoted candidate is not historically accepted: {key}."
            )
        if candidate["step4_segment_ref"] != row["rpt_step4_segment_ref"]:
            raise ValueError(f"Step 4 mapping changed for {key}.")
        if candidate["step4_block_sha256"] != row["rpt_step4_block_sha256"]:
            raise ValueError(f"Step 4 block hash changed for {key}.")
        if candidate["reconstructed_soh_percent"] and not math.isclose(
            float(candidate["reconstructed_soh_percent"]),
            float(row["soh_percent"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(f"Label value differs from verified candidate at {key}.")
        for field in (
            "label_step4_reference_match",
            "label_step4_block_hash_match",
            "step4_block_hash_audit_match",
            "step10_block_hash_audit_match",
            "block_audit_complete",
            "step4_source_size_mtime_unchanged",
            "current_directions_valid",
            "voltage_endpoint_directions_valid",
            "same_rpt_step_order_valid",
        ):
            if alignment[field].lower() != "true":
                raise ValueError(f"Alignment check {field} failed for {key}.")
        if alignment["reference_mapping_error"]:
            raise ValueError(
                f"Prior raw-source alignment has an unresolved mapping issue at {key}."
            )
        if row["partition"] == "external_test" and row["external_test_only"].lower() != "true":
            raise ValueError(f"Part 3 record is not marked external-only: {key}.")
    return ordered, candidate_by_key


def _audit_index(
    block_audit_rows: list[dict[str, str]],
) -> dict[tuple[str, int, int, int, int], dict[str, str]]:
    result: dict[tuple[str, int, int, int, int], dict[str, str]] = {}
    for row in block_audit_rows:
        key = raw._block_key(
            row["source_file"],
            int(row["source_part"]),
            int(row["expected_step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        if key in result:
            raise ValueError(f"Duplicate M1 measurement-block key: {key}.")
        result[key] = row
    return result


def _mapped_request(
    reference: str,
    part: int,
    expected_hash: str,
    audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]],
    requests: dict[tuple[str, int, int, int, int], raw.BlockRequest],
    *,
    keep_physics: bool,
) -> tuple[tuple[str, int, int, int, int], dict[str, str]]:
    source, step, start, end = raw._parse_segment(reference)
    source = raw._resolve_source_name(source, part)
    key = raw._block_key(source, part, step, start, end)
    audit = audit_by_key.get(key)
    if audit is None:
        raise ValueError(f"No exact M1 block audit match for {reference!r}.")
    if audit["raw_block_sha256"] != expected_hash:
        raise ValueError(f"Block SHA-256 differs from its M1 audit: {reference!r}.")
    if audit["line_range_exact"].lower() != "true":
        raise ValueError(f"Mapped line range is not exact: {reference!r}.")
    request = raw.BlockRequest(
        source_file=source,
        source_part=part,
        step=step,
        start_line=start,
        end_line=end,
        expected_sha256=expected_hash,
        keep_voltage=True,
        keep_physics=keep_physics,
    )
    prior = requests.get(key)
    if prior is not None and prior.expected_sha256 != request.expected_sha256:
        raise ValueError(f"Conflicting hashes for a repeated mapped block: {reference!r}.")
    requests[key] = raw.BlockRequest(
        source_file=source,
        source_part=part,
        step=step,
        start_line=start,
        end_line=end,
        expected_sha256=expected_hash,
        keep_voltage=True,
        keep_physics=keep_physics or (prior.keep_physics if prior else False),
    )
    return key, audit


def _validate_raw_block(block: dict[str, Any], expected_step: int) -> None:
    if block["step"] != expected_step or not block["step_ids_match"]:
        raise ValueError(f"Raw Step ID mismatch for mapped Step {expected_step}.")
    if not block["next_row_transitions_step"]:
        raise ValueError(f"Step {expected_step} end boundary is not confirmed.")
    if block["wrong_width_rows"]:
        raise ValueError(f"Step {expected_step} has malformed-width rows.")
    if block["row_count"] != block["expected_row_count"]:
        raise ValueError(f"Step {expected_step} row count does not match mapping.")
    if block["raw_block_sha256"] != block["expected_raw_block_sha256"]:
        raise ValueError(f"Step {expected_step} block hash does not match mapping.")
    missing = block["missing_core_values"]
    if any(missing.get(name, 0) for name in ("step", "testtime", "steptime", "amphr", "amps", "volts")):
        raise ValueError(f"Step {expected_step} contains missing core measurements.")
    if (
        block["testtime_negative_intervals"]
        or block["steptime_negative_intervals"]
    ):
        raise ValueError(f"Step {expected_step} contains a negative time interval.")
    if len(block["voltage_values"]) != block["row_count"]:
        raise ValueError(f"Step {expected_step} voltage rows are incomplete.")
    if expected_step == 4:
        if len(block["current_values"]) != block["row_count"]:
            raise ValueError("Step 4 current rows are incomplete.")
        if block["current_signs"].get("negative", 0) != block["row_count"]:
            raise ValueError("Step 4 current direction is not consistently discharge.")
        if block["voltage_first_v"] <= block["voltage_last_v"]:
            raise ValueError("Step 4 voltage does not decrease overall.")
    elif block["current_signs"].get("positive", 0) != block["row_count"]:
        raise ValueError("Step 10 current direction is not consistently charge.")
    elif block["voltage_first_v"] >= block["voltage_last_v"]:
        raise ValueError("Step 10 voltage does not increase overall.")


def _read_exact_duplicate_sensitivity(
    source_path: Path,
    start_line: int,
    end_line: int,
) -> tuple[list[float], list[float], int]:
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Mapped raw source has no header: {source_path}.")
        folded = [name.strip().casefold() for name in header]
        if folded.count("amps") != 1 or folded.count("volts") != 1:
            raise ValueError(f"Expected unique Amps and Volts columns: {source_path}.")
        amps_index = folded.index("amps")
        volts_index = folded.index("volts")
        previous_row: list[str] | None = None
        duplicates = 0
        currents: list[float] = []
        voltages: list[float] = []
        for row in reader:
            line_number = reader.line_num
            if line_number < start_line:
                continue
            if line_number > end_line:
                break
            if previous_row is not None and row == previous_row:
                duplicates += 1
                continue
            try:
                current = float(row[amps_index])
                voltage = float(row[volts_index])
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"Malformed Step 4 row during duplicate sensitivity: "
                    f"{source_path}:{line_number}."
                ) from error
            if not math.isfinite(current) or not math.isfinite(voltage):
                raise ValueError(
                    f"Non-finite Step 4 row during duplicate sensitivity: "
                    f"{source_path}:{line_number}."
                )
            currents.append(current)
            voltages.append(voltage)
            previous_row = row
    return voltages, currents, duplicates


def _duplicate_sensitivity(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requested = {
        (1, "G2-Cell3", 2): 26,
        (3, "G8-Cell18", 1): 0,
    }
    results: list[dict[str, Any]] = []
    by_key = {record["key"]: record for record in records}
    for key, expected_duplicates in requested.items():
        record = by_key.get(key)
        if record is None:
            raise ValueError(f"Required duplicate-sensitivity sample is absent: {key}.")
        source_path = PROJECT_ROOT / Path(
            record["step4_source"].replace("/", "\\")
        )
        voltage_dedup, current_dedup, duplicate_count = _read_exact_duplicate_sensitivity(
            source_path,
            record["step4_start_line"],
            record["step4_end_line"],
        )
        if duplicate_count != expected_duplicates:
            raise ValueError(
                f"Exact adjacent duplicate count changed for {key}: "
                f"expected {expected_duplicates}, got {duplicate_count}."
            )
        dedup_features = build_p5_electrical_features(voltage_dedup, current_dedup)
        dedup_curve = np.asarray(resample_curve(voltage_dedup), dtype=np.float64)
        primary_curve = np.asarray(
            resample_curve(record["step4"]["voltage_values"], n_points=128),
            dtype=np.float64,
        )
        sensitivity: dict[str, Any] = {
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "step4_raw_row_count": record["step4_row_count"],
            "exact_adjacent_duplicate_rows_removed_for_sensitivity_only": duplicate_count,
            "primary_policy": "preserve all raw rows",
            "deduplicated_rows": len(voltage_dedup),
            "max_abs_voltage_curve_delta_v": float(
                np.max(np.abs(primary_curve - dedup_curve))
            ),
            "rms_voltage_curve_delta_v": float(
                np.sqrt(np.mean((primary_curve - dedup_curve) ** 2))
            ),
        }
        for name in P5_FEATURE_NAMES:
            sensitivity[f"primary_{name}"] = record["p5_raw"][name]
            sensitivity[f"deduplicated_{name}"] = dedup_features[name]
            sensitivity[f"delta_{name}"] = (
                dedup_features[name] - record["p5_raw"][name]
            )
        results.append(sensitivity)
    return results


def _fit_train_scaler(
    raw_features: np.ndarray, partitions: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_mask = np.asarray([partition == "train" for partition in partitions])
    if int(train_mask.sum()) != EXPECTED_COUNTS["train"]:
        raise ValueError("Physics scaler fit set must be exactly the 102 Train rows.")
    train_values = raw_features[train_mask]
    means = train_values.mean(axis=0, dtype=np.float64)
    scales = train_values.std(axis=0, ddof=0, dtype=np.float64)
    constant = scales == 0.0
    effective_scales = scales.copy()
    effective_scales[constant] = 1.0
    transformed = (raw_features - means) / effective_scales
    if not np.isfinite(transformed).all():
        raise ValueError("Train-fitted P5 scaling produced non-finite values.")
    return transformed, means, effective_scales, constant


def _run_synthetic_forward_passes() -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    statuses: list[dict[str, Any]] = []
    torch.manual_seed(20261114)
    discharge = torch.zeros((2, 128), dtype=torch.float32, device=device)
    charge = torch.zeros((2, 128), dtype=torch.float32, device=device)
    physics = torch.zeros((2, 5), dtype=torch.float32, device=device)
    with torch.inference_mode():
        for name in ARCHITECTURE_NAMES:
            model = build_architecture(name).to(device)
            model.eval()
            output = model(discharge, charge, physics)
            if tuple(output.shape) != (2, 1):
                raise ValueError(
                    f"{name} returned unexpected synthetic output shape "
                    f"{tuple(output.shape)}."
                )
            if not torch.isfinite(output).all().item():
                raise ValueError(f"{name} returned non-finite synthetic output.")
            statuses.append(
                {
                    "architecture": name,
                    "synthetic_batch_size": 2,
                    "input_shapes": {
                        "X_discharge": [2, 128],
                        "X_charge": [2, 128],
                        "X_physics": [2, 5],
                    },
                    "output_shape": [2, 1],
                    "output_finite": True,
                    "passed": True,
                }
            )
    return {
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "training_started": False,
        "prediction_evaluation_started": False,
        "forward_passes": statuses,
    }


def _feature_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": float(min(values)),
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }


def _cross_part_report(records: list[dict[str, Any]]) -> str:
    lines = [
        "# P5 Cross-Part Protocol and Sampling Audit",
        "",
        "Part-level summaries below are descriptive compatibility checks only. "
        "Part 3 (External Test) was not used to select features, define thresholds, "
        "or fit preprocessing.",
        "",
        "| Part | N | Protocol family | Step 4 rows (median, min–max) | "
        "Median native TestTime interval | Step 4 voltage first/last range (V) |",
        "|---:|---:|---|---:|---:|---:|",
    ]
    for part in (1, 2, 3):
        rows = [record for record in records if record["key"][0] == part]
        if not rows:
            continue
        row_counts = [record["step4_row_count"] for record in rows]
        intervals = [
            record["step4_testtime_dt_median"]
            for record in rows
            if record["step4_testtime_dt_median"] is not None
        ]
        starts = [record["step4_voltage_first_v"] for record in rows]
        ends = [record["step4_voltage_last_v"] for record in rows]
        protocol = (
            "Maccor 4200; Step 4 CC-CV discharge"
            if part in (1, 2)
            else "Maccor 4000; Step 4 CC discharge"
        )
        interval_text = (
            f"{statistics.median(intervals):.9g}"
            if intervals
            else "unavailable"
        )
        lines.append(
            f"| {part} | {len(rows)} | {protocol} | "
            f"{statistics.median(row_counts):.0f}, {min(row_counts)}–{max(row_counts)} | "
            f"{interval_text} | "
            f"{min(starts):.6g}–{max(starts):.6g} / "
            f"{min(ends):.6g}–{max(ends):.6g} |"
        )

    lines.extend(
        [
            "",
            "## Electrical feature distributions by Part",
            "",
            "These descriptive ranges expose domain shift; they are not used to "
            "choose features, transformations, or model settings.",
            "",
            "| Part | Feature | Min | Median | Max |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for part in (1, 2, 3):
        rows = [record for record in records if record["key"][0] == part]
        for name in P5_FEATURE_NAMES:
            stats = _feature_stats([record["p5_raw"][name] for record in rows])
            lines.append(
                f"| {part} | `{name}` | {stats['min']:.9g} | "
                f"{stats['median']:.9g} | {stats['max']:.9g} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Parts 1–2 and Part 3 use different Maccor instruments and "
            "documented Step 4 protocols (CC-CV versus CC); Part 3 remains an "
            "external/domain-shift partition.",
            "- Curves and voltage-shape summaries use normalized source-row "
            "progress, not time or capacity. Unequal sampling density can change "
            "the weighting of raw samples and the interpolation path; 128-point "
            "interpolation does not remove protocol or instrument differences.",
            "- Current mean and scaled MAD are sample-weighted. Their numerical "
            "comparability is descriptive, not guaranteed by the protocol; current "
            "units are recorded as inferred A in the source audit.",
            "- Step 4 and Step 10 directions, exact mapped blocks, raw block "
            "hashes, row counts, time monotonicity, and source stat snapshots were "
            "validated for every record. Zero timestamp intervals are permitted "
            "when the source contains repeated rows; P5 does not use timestamps.",
            "- The five P5 values are engineered summaries of the same-RPT "
            "Step 4 electrical inputs, not independent sensors or validated "
            "electrochemical states.",
            "- Temperature metadata remain in the row-level audit only; temperature "
            "is not included in P5 or imputed.",
            "",
        ]
    )
    return "\n".join(lines)


def _quality_report(
    records: list[dict[str, Any]],
    raw_physics: np.ndarray,
    scaled_physics: np.ndarray,
    means: np.ndarray,
    scales: np.ndarray,
    constant_features: np.ndarray,
    duplicate_rows: list[dict[str, Any]],
    forward_status: dict[str, Any],
) -> str:
    train_records = [record for record in records if record["partition"] == "train"]
    train_current_mads = [
        record["p5_raw"]["discharge_current_variability"]
        for record in train_records
    ]
    train_current_stds = [
        float(np.std(record["step4"]["current_values"], ddof=0))
        for record in train_records
    ]
    lines = [
        "# P5 Electrical Feature Quality Report",
        "",
        "## Cohort and alignment",
        "",
        f"- Observations: {len(records)} unique `(part, physical cell, RPT)` keys.",
        "- Split counts: Train 102, Validation 37, External Test 64; 23 physical "
        "cells; no cell overlap.",
        "- All 203 approved ACCEPTED labels were preserved byte-for-byte in the "
        "frozen label copy. The label definition was not recalculated.",
        "- Step 4 discharge and Step 10 charge blocks were independently re-read "
        "from the raw source files; exact mapping, row count, block SHA-256, "
        "current direction, step boundary, and same-RPT alignment passed.",
        "- No row was dropped, imputed, or assigned another cell/RPT's values.",
        "",
        "## P5 definitions",
        "",
        "1. `mean_abs_discharge_current`: sample-weighted mean of `abs(Amps)` "
        "within the verified Step 4 block.",
        "2. `discharge_current_variability`: `1.4826 × median(abs(I - median(I)))` "
        "over raw Step 4 current samples. The scaled MAD is robust to isolated "
        "current excursions; Train-only MAD and population-standard-deviation "
        "diagnostics are summarized below.",
        "3. `discharge_voltage_slope_20_80`: ordinary least-squares slope of "
        "resampled Step 4 voltage against normalized row progress for points "
        "with `0.20 ≤ p ≤ 0.80`; units V per progress.",
        "4. `discharge_voltage_derivative_mad_10_90`: `1.4826 × MAD` of centered "
        "finite differences `(V[i+1]-V[i-1])/(p[i+1]-p[i-1])` for interior "
        "resampled points with `0.10 ≤ p ≤ 0.90`; units V per progress.",
        "5. `discharge_voltage_segment_contrast_10_30_vs_70_90`: mean resampled "
        "voltage for `0.10 ≤ p ≤ 0.30` minus mean voltage for "
        "`0.70 ≤ p ≤ 0.90`; units V.",
        "",
        "The original plateau-fraction proposal was not implemented: no source-"
        "defined or independently justified derivative threshold was established. "
        "The predefined segment contrast is target-independent and avoids a "
        "cohort-relative threshold.",
        "",
        "## Train-only current variability diagnostics",
        "",
        f"- Train records: {len(train_records)}.",
        f"- Scaled current MAD: min={min(train_current_mads):.9g}, "
        f"median={statistics.median(train_current_mads):.9g}, "
        f"max={max(train_current_mads):.9g}.",
        f"- Current population standard deviation: min={min(train_current_stds):.9g}, "
        f"median={statistics.median(train_current_stds):.9g}, "
        f"max={max(train_current_stds):.9g}.",
        "- These are descriptive Train-only values. No universal threshold for "
        "negligible variation was invented; instrument noise-floor metadata are "
        "not available. External values were not used for feature selection.",
        "",
        "## Observed P5 numerical ranges",
        "",
        "| Feature | Train min | Train median | Train max | "
        "Train scaler mean | Train scaler scale |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(P5_FEATURE_NAMES):
        values = raw_physics[:]
        train_mask = np.asarray([record["partition"] == "train" for record in records])
        stats = _feature_stats(values[train_mask, index].tolist())
        lines.append(
            f"| `{name}` | {stats['min']:.9g} | {stats['median']:.9g} | "
            f"{stats['max']:.9g} | {means[index]:.12g} | {scales[index]:.12g} |"
        )
    lines.extend(
        [
            "",
            "Voltage features are correlated shape summaries of `X_discharge`; "
            "they are not independent measurements. Current variability may "
            "reflect tester regulation, noise, and sampling density as well as "
            "cell state.",
            "",
            "## Duplicate-row sensitivity",
            "",
            "The primary representation preserves every verified raw row. Exact "
            "adjacent duplicate removal was performed only as a sensitivity check:",
            "",
            "| Key | Exact duplicate rows | Max 128-point voltage delta (V) | "
            "Max absolute P5 feature delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in duplicate_rows:
        feature_deltas = [abs(float(row[f"delta_{name}"])) for name in P5_FEATURE_NAMES]
        lines.append(
            f"| Part {row['dataset_part']} / {row['physical_cell_id']} / "
            f"RPT{row['rpt_index']} | "
            f"{row['exact_adjacent_duplicate_rows_removed_for_sensitivity_only']} | "
            f"{row['max_abs_voltage_curve_delta_v']:.9g} | "
            f"{max(feature_deltas):.9g} |"
        )
    lines.extend(
        [
            "",
            "No raw file or primary curve was changed. Normalized-row-progress "
            "features are sensitive to row duplication by construction; any "
            "observed delta is documented rather than corrected silently.",
            "",
            "## Leakage and preprocessing",
            "",
            "- Feature extraction receives only same-RPT Step 4 `Volts` and `Amps`. "
            "It does not receive SoH, BoL, `Amphr`, integrated capacity, time, "
            "cell/RPT/Part identifiers, or future measurements.",
            "- Curve progress is normalized source-row position, not cumulative "
            "capacity or a capacity-derived coordinate.",
            "- Same-RPT Step 10 contributes only its voltage curve; charge "
            "capacity is not used as a target or feature.",
            "- Only the five P5 scalar features are standardized. Population means "
            "and standard deviations were fitted on Train rows only; Validation "
            "and External use those unchanged parameters.",
            "- Voltage curves remain in volts after deterministic within-curve "
            "linear interpolation to 128 normalized-row-progress points.",
            "- Temperature is preserved in audit metadata and excluded from all "
            "predictive tensors.",
            "- Four architecture smoke tests used synthetic zero-valued tensors "
            f"on `{forward_status['device_name']}`. All outputs had shape (2, 1) "
            "and were finite. No model was trained and no predictions were "
            "evaluated.",
            "",
            "## Known limitations",
            "",
            "- Parts 1–2 use Maccor 4200 Step 4 CC-CV; Part 3 uses Maccor 4000 "
            "Step 4 CC. Part 3 is an external/domain-shift test only.",
            "- The five engineered features are not validated as protocol-invariant "
            "health indicators. Part 3 distributions are reported for shift "
            "description only.",
            "- Current unit A is an evidence-based inference in the existing raw "
            "audit, not a recovered export setting. Temperature units remain "
            "unresolved and are not used.",
            "",
        ]
    )
    return "\n".join(lines)


def _freeze_report(
    records: list[dict[str, Any]],
    forward_status: dict[str, Any],
    means: np.ndarray,
    scales: np.ndarray,
    output_hashes: dict[str, str],
) -> str:
    lines = [
        "# BaFuse P5 Dataset Freeze Report",
        "",
        "## Decision",
        "",
        f"- Frozen dataset identifier: `bafuse-p5-electrical-v1-20261114`.",
        "- Status: FROZEN for schema/loading and future approved training; no "
        "training or external prediction evaluation was performed.",
        "- Training authorization: false.",
        "- External prediction evaluation authorization: false.",
        "- All 203 approved ACCEPTED labels were retained without reconstruction "
        "or modification; frozen CSV is a byte-for-byte copy.",
        "",
        "## Frozen tensors",
        "",
        "| Split | X_discharge | X_charge | X_physics | y_soh |",
        "|---|---:|---:|---:|---:|",
    ]
    for partition, count in EXPECTED_COUNTS.items():
        lines.append(
            f"| {partition} | ({count}, 128) float32 | ({count}, 128) float32 | "
            f"({count}, 5) float32 | ({count}, 1) float32 |"
        )
    lines.extend(
        [
            "",
            "The label remains a reconstructed same-cell Step 4 discharge-capacity "
            "retention reference, not official Oxford ground truth. All Part 3 "
            "records remain External Test only.",
            "",
            "## Preprocessing",
            "",
            "- Curves: deterministic linear interpolation on normalized source-row "
            "progress; no dataset-level curve scaler.",
            "- P5: population-standard-score scaling fitted to the 102 Train rows "
            "only.",
            f"- Train scaler means: `{[float(value) for value in means]}`.",
            f"- Train scaler scales: `{[float(value) for value in scales]}`.",
            "- Validation and External used the unchanged Train-fitted parameters.",
            "",
            "## Architecture compatibility",
            "",
            f"- PyTorch {forward_status['torch_version']}; CUDA "
            f"{forward_status['cuda_version']}; available="
            f"{forward_status['cuda_available']}; device="
            f"{forward_status['device_name']}.",
            f"- Synthetic forward passes: {len(forward_status['forward_passes'])}/4 "
            "passed with finite (2, 1) outputs.",
            "- No optimization, training, or prediction metric evaluation occurred.",
            "",
            "## Output SHA-256 values",
            "",
        ]
    )
    for name, digest in sorted(output_hashes.items()):
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "The checksum manifest covers every frozen data/report artifact except "
            "itself; its own SHA-256 is reported separately after creation.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify_frozen_npz(path: Path, expected_count: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"X_discharge", "X_charge", "X_physics", "y_soh"}:
            raise ValueError(f"Unexpected NPZ members in {path.name}.")
        expected_shapes = {
            "X_discharge": (expected_count, 128),
            "X_charge": (expected_count, 128),
            "X_physics": (expected_count, 5),
            "y_soh": (expected_count, 1),
        }
        for name, shape in expected_shapes.items():
            array = archive[name]
            if array.shape != shape or array.dtype != np.float32:
                raise ValueError(
                    f"{path.name}:{name} has shape/dtype "
                    f"{array.shape}/{array.dtype}, expected {shape}/float32."
                )
            if not np.isfinite(array).all():
                raise ValueError(f"{path.name}:{name} contains non-finite values.")


def run() -> dict[str, Any]:
    _verify_approved_inputs()
    run_config = json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))
    if run_config["p5_feature_names"] != list(P5_FEATURE_NAMES):
        raise ValueError("Run configuration and P5 feature schema differ.")
    if run_config["training_authorized"] or run_config["external_prediction_evaluation_authorized"]:
        raise ValueError("Training and External prediction evaluation must remain disabled.")

    labels = _read_csv(LABELS_PATH)
    candidates = _read_csv(CANDIDATES_PATH)
    alignments = _read_csv(ALIGNMENT_PATH)
    cell_splits = _read_csv(CELL_SPLIT_PATH)
    ordered_labels, candidate_by_key = _verify_cohort(
        labels, candidates, alignments, cell_splits
    )

    m1_rows = raw._read_csv(raw.M1_BLOCK_AUDIT)
    audit_by_key = _audit_index(m1_rows)
    source_hash_rows = _read_csv(SOURCE_HASHES_PATH)
    source_hash_by_path = {
        _normalized_source(row["source_file"]): row for row in source_hash_rows
    }
    if len(source_hash_by_path) != len(source_hash_rows):
        raise ValueError("Source SHA-256 manifest contains duplicate file entries.")
    old_matrix_rows = _read_csv(OLD_MATRIX_PATH)
    old_matrix_by_key = {_key(row): row for row in old_matrix_rows}
    if len(old_matrix_by_key) != len(old_matrix_rows):
        raise ValueError("Historical raw feature matrix contains duplicate keys.")

    requests: dict[tuple[str, int, int, int, int], raw.BlockRequest] = {}
    record_mappings: dict[tuple[int, str, int], dict[str, Any]] = {}
    for label in ordered_labels:
        key = _key(label)
        candidate = candidate_by_key[key]
        if key not in old_matrix_by_key:
            raise ValueError(f"Historical curve matrix is missing {key}.")
        step4_key, step4_audit = _mapped_request(
            candidate["step4_segment_ref"],
            int(label["dataset_part"]),
            candidate["step4_block_sha256"],
            audit_by_key,
            requests,
            keep_physics=True,
        )
        step10_key, step10_audit = _mapped_request(
            candidate["step10_segment_ref"],
            int(label["dataset_part"]),
            candidate["step10_block_sha256"],
            audit_by_key,
            requests,
            keep_physics=False,
        )
        for source_name, source_audit in (
            (step4_audit["source_file"], step4_audit),
            (step10_audit["source_file"], step10_audit),
        ):
            source_entry = source_hash_by_path.get(_normalized_source(source_name))
            if source_entry is None:
                raise ValueError(f"Source hash manifest lacks {source_name}.")
            if (
                source_entry["status"] not in {"MATCH", "CURRENT_SNAPSHOT_ONLY"}
                or not source_entry["sha256"]
                or int(source_entry["source_size_bytes"])
                != int(source_audit["source_size_bytes"])
            ):
                raise ValueError(f"Raw source hash provenance differs for {source_name}.")
            if (
                source_entry["expected_sha256"]
                and source_entry["sha256"] != source_entry["expected_sha256"]
            ):
                raise ValueError(f"Historical source hash mismatch for {source_name}.")
            label_source_hash = label["rpt_source_file_sha256"]
            if (
                label_source_hash
                and _normalized_source(label["rpt_step4_source_file"])
                == _normalized_source(source_name)
                and source_entry["sha256"] != label_source_hash
            ):
                raise ValueError(f"Label source SHA-256 differs for {source_name}.")
        record_mappings[key] = {
            "label": label,
            "candidate": candidate,
            "step4_key": step4_key,
            "step10_key": step10_key,
            "step4_audit": step4_audit,
            "step10_audit": step10_audit,
        }

    print(
        f"Re-reading {len(requests)} exact Step 4/Step 10 blocks "
        f"from {len({request.source_file for request in requests.values()})} raw sources.",
        flush=True,
    )
    raw_blocks = raw._scan_raw_blocks(requests, audit_by_key)
    records: list[dict[str, Any]] = []
    for label in ordered_labels:
        key = _key(label)
        mapping = record_mappings[key]
        step4 = raw_blocks[mapping["step4_key"]]
        step10 = raw_blocks[mapping["step10_key"]]
        _validate_raw_block(step4, 4)
        _validate_raw_block(step10, 10)
        x_discharge = np.asarray(
            resample_curve(step4["voltage_values"], n_points=128),
            dtype=np.float32,
        )
        x_charge = np.asarray(
            resample_curve(step10["voltage_values"], n_points=128),
            dtype=np.float32,
        )
        p5 = build_p5_electrical_features(
            step4["voltage_values"], step4["current_values"]
        )
        old_row = old_matrix_by_key[key]
        old_discharge = np.asarray(
            [float(old_row[f"discharge_v_{index:03d}"]) for index in range(128)],
            dtype=np.float64,
        )
        old_charge = np.asarray(
            [float(old_row[f"charge_v_{index:03d}"]) for index in range(128)],
            dtype=np.float64,
        )
        if not np.allclose(x_discharge, old_discharge, rtol=0.0, atol=1e-6):
            raise ValueError(f"Raw Step 4 resampling differs from historical matrix at {key}.")
        if not np.allclose(x_charge, old_charge, rtol=0.0, atol=1e-6):
            raise ValueError(f"Raw Step 10 resampling differs from historical matrix at {key}.")

        source, step, start, end = raw._parse_segment(
            mapping["candidate"]["step4_segment_ref"]
        )
        if step != 4:
            raise ValueError(f"Unexpected Step 4 reference at {key}.")
        step10_source, parsed_step10, step10_start, step10_end = raw._parse_segment(
            mapping["candidate"]["step10_segment_ref"]
        )
        if parsed_step10 != 10:
            raise ValueError(f"Unexpected Step 10 reference at {key}.")
        records.append(
            {
                "key": key,
                "partition": label["partition"],
                "label": label,
                "candidate": mapping["candidate"],
                "step4": step4,
                "step10": step10,
                "step4_source": raw._resolve_source_name(source, key[0]),
                "step10_source": raw._resolve_source_name(step10_source, key[0]),
                "step4_start_line": start,
                "step4_end_line": end,
                "step10_start_line": step10_start,
                "step10_end_line": step10_end,
                "step4_row_count": step4["row_count"],
                "step10_row_count": step10["row_count"],
                "step4_testtime_dt_median": step4["testtime_dt_median_native"],
                "step4_steptime_dt_median": step4["steptime_dt_median_native"],
                "step4_voltage_first_v": step4["voltage_first_v"],
                "step4_voltage_last_v": step4["voltage_last_v"],
                "step10_voltage_first_v": step10["voltage_first_v"],
                "step10_voltage_last_v": step10["voltage_last_v"],
                "p5_raw": p5,
                "x_discharge": x_discharge,
                "x_charge": x_charge,
                "old_discharge_max_abs_delta_v": float(
                    np.max(np.abs(x_discharge.astype(np.float64) - old_discharge))
                ),
                "old_charge_max_abs_delta_v": float(
                    np.max(np.abs(x_charge.astype(np.float64) - old_charge))
                ),
            }
        )
    if len(records) != 203:
        raise ValueError(f"Raw feature extraction yielded {len(records)} records, not 203.")

    duplicate_rows = _duplicate_sensitivity(records)
    partitions = [record["partition"] for record in records]
    raw_physics = np.asarray(
        [
            [record["p5_raw"][name] for name in P5_FEATURE_NAMES]
            for record in records
        ],
        dtype=np.float64,
    )
    scaled_physics, scaler_means, scaler_scales, constant_features = (
        _fit_train_scaler(raw_physics, partitions)
    )
    x_discharge_all = np.asarray(
        [record["x_discharge"] for record in records], dtype=np.float32
    )
    x_charge_all = np.asarray(
        [record["x_charge"] for record in records], dtype=np.float32
    )
    x_physics_all = np.asarray(scaled_physics, dtype=np.float32)
    y_all = np.asarray(
        [[float(record["label"]["soh_percent"])] for record in records],
        dtype=np.float32,
    )
    if not (
        np.isfinite(x_discharge_all).all()
        and np.isfinite(x_charge_all).all()
        and np.isfinite(x_physics_all).all()
        and np.isfinite(y_all).all()
    ):
        raise ValueError("One or more frozen modalities contain non-finite values.")

    forward_status = _run_synthetic_forward_passes()
    if len(forward_status["forward_passes"]) != 4:
        raise ValueError("Four architecture forward passes were not completed.")

    # Keep one deterministic key order for every output tensor and its manifest.
    output_paths: dict[str, Path] = {}
    _copy_file_atomic(LABELS_PATH, OUTPUT_LABEL)
    output_paths[OUTPUT_LABEL.name] = OUTPUT_LABEL
    for partition, count in EXPECTED_COUNTS.items():
        indices = [index for index, value in enumerate(partitions) if value == partition]
        if len(indices) != count:
            raise ValueError(f"Unexpected tensor row count for {partition}.")
        path = P5_DIR / SPLIT_FILE_NAMES[partition]
        _write_deterministic_npz(
            path,
            {
                "X_discharge": x_discharge_all[indices],
                "X_charge": x_charge_all[indices],
                "X_physics": x_physics_all[indices],
                "y_soh": y_all[indices],
            },
        )
        _verify_frozen_npz(path, count)
        output_paths[path.name] = path

    schema = {
        "schema_version": 1,
        "dataset_identifier": "bafuse-p5-electrical-v1-20261114",
        "feature_branch_description": "physics-inspired electrical feature branch",
        "label": {
            "definition": TARGET_DEFINITION,
            "semantics": "reconstructed capacity-retention reference; not official Oxford ground truth",
            "unit": "percent",
            "source": str(LABELS_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        },
        "observation_key": ["dataset_part", "physical_cell_id", "rpt_index"],
        "modalities": {
            "X_discharge": {
                "shape_per_split": ["N", 128],
                "dtype": "float32",
                "raw_columns": ["Step 4 Volts"],
                "resampling": "linear on normalized source-row progress; 128 points",
                "capacity_or_time_axis_used": False,
                "unit": "V",
            },
            "X_charge": {
                "shape_per_split": ["N", 128],
                "dtype": "float32",
                "raw_columns": ["same-RPT Step 10 Volts"],
                "resampling": "linear on normalized source-row progress; 128 points",
                "capacity_or_time_axis_used": False,
                "unit": "V",
            },
            "X_physics": {
                "shape_per_split": ["N", 5],
                "dtype": "float32",
                "feature_names_in_order": list(P5_FEATURE_NAMES),
                "definitions": {
                    "mean_abs_discharge_current": "sample-weighted mean(abs(Step 4 Amps)); current unit A is inferred",
                    "discharge_current_variability": "1.4826 * median(abs(I - median(I))) on raw Step 4 Amps; sample-weighted; current unit A is inferred",
                    "discharge_voltage_slope_20_80": "OLS slope of 128-point Step 4 Volts versus normalized row progress for 0.20 <= p <= 0.80; V/progress",
                    "discharge_voltage_derivative_mad_10_90": "1.4826 * MAD of centered finite differences on 128-point Step 4 voltage for 0.10 <= p <= 0.90; V/progress",
                    "discharge_voltage_segment_contrast_10_30_vs_70_90": "mean(V at 0.10 <= p <= 0.30) - mean(V at 0.70 <= p <= 0.90); V",
                },
                "temperature_included": False,
                "target_or_bol_derived": False,
            },
            "y_soh": {
                "shape_per_split": ["N", 1],
                "dtype": "float32",
                "unit": "percent",
            },
        },
        "duplicate_row_policy": "preserve all exact raw rows; duplicate removal is sensitivity-only",
        "plateau_fraction": {
            "included": False,
            "reason": "No defensible source- or instrument-grounded derivative threshold was established.",
            "documented_replacement": "predefined voltage segment contrast",
        },
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
    }
    schema_path = P5_DIR / "bafuse_p5_feature_schema.json"
    _write_json_atomic(schema_path, schema)
    output_paths[schema_path.name] = schema_path

    preprocess = {
        "preprocessing_version": "bafuse-p5-train-scaler-v1",
        "curve_transform": "within-curve linear interpolation only; curves remain in V",
        "physics_transform": "standard score using population standard deviation",
        "fit_partition": "train",
        "fit_row_count": EXPECTED_COUNTS["train"],
        "feature_names_in_order": list(P5_FEATURE_NAMES),
        "mean": [float(value) for value in scaler_means],
        "scale": [float(value) for value in scaler_scales],
        "constant_train_features": [
            name
            for name, is_constant in zip(P5_FEATURE_NAMES, constant_features)
            if bool(is_constant)
        ],
        "constant_feature_scale_policy": "scale set to 1.0 only for exact zero Train standard deviation; centered values remain zero",
        "validation_and_external_fit_or_update": False,
        "external_statistics_used_for_selection": False,
        "imputer": None,
    }
    preprocess_path = P5_DIR / "bafuse_p5_preprocessing_config.json"
    _write_json_atomic(preprocess_path, preprocess)
    output_paths[preprocess_path.name] = preprocess_path

    split_manifest_rows: list[dict[str, Any]] = []
    feature_audit_rows: list[dict[str, Any]] = []
    split_indices: dict[str, int] = defaultdict(int)
    alignment_by_key = {_key(row): row for row in alignments}
    for index, record in enumerate(records):
        key = record["key"]
        partition = record["partition"]
        label = record["label"]
        candidate = record["candidate"]
        step4_audit = record["step4"]
        step10_audit = record["step10"]
        split_indices[partition] += 1
        split_manifest_rows.append(
            {
                "split_row_index": split_indices[partition] - 1,
                "dataset_part": key[0],
                "physical_cell_id": key[1],
                "rpt_index": key[2],
                "partition": partition,
                "is_external_only": partition == "external_test",
                "rpt_source_file": record["step4_source"],
                "step4_segment_ref": candidate["step4_segment_ref"],
                "step4_block_sha256": candidate["step4_block_sha256"],
                "step4_source_file_sha256_snapshot": source_hash_by_path[
                    _normalized_source(record["step4_source"])
                ]["sha256"],
                "step4_source_hash_status": source_hash_by_path[
                    _normalized_source(record["step4_source"])
                ]["status"],
                "step10_segment_ref": candidate["step10_segment_ref"],
                "step10_block_sha256": candidate["step10_block_sha256"],
                "step10_source_file_sha256_snapshot": source_hash_by_path[
                    _normalized_source(record["step10_source"])
                ]["sha256"],
                "step10_source_hash_status": source_hash_by_path[
                    _normalized_source(record["step10_source"])
                ]["status"],
                "label_status": label["final_status"],
                "training_authorized": False,
            }
        )
        alignment = alignment_by_key[key]
        audit_row: dict[str, Any] = {
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "partition": partition,
            "soh_percent_for_alignment_audit_only": label["soh_percent"],
            "target_definition": label["target_definition"],
            "step4_source_file": record["step4_source"],
            "step4_segment_ref": candidate["step4_segment_ref"],
            "step4_block_sha256": step4_audit["raw_block_sha256"],
            "step4_source_file_sha256_snapshot": source_hash_by_path[
                _normalized_source(record["step4_source"])
            ]["sha256"],
            "step4_source_hash_status": source_hash_by_path[
                _normalized_source(record["step4_source"])
            ]["status"],
            "step10_source_file": record["step10_source"],
            "step10_segment_ref": candidate["step10_segment_ref"],
            "step10_block_sha256": step10_audit["raw_block_sha256"],
            "step10_source_file_sha256_snapshot": source_hash_by_path[
                _normalized_source(record["step10_source"])
            ]["sha256"],
            "step10_source_hash_status": source_hash_by_path[
                _normalized_source(record["step10_source"])
            ]["status"],
            "step4_row_count": step4_audit["row_count"],
            "step10_row_count": step10_audit["row_count"],
            "step4_testtime_zero_intervals": step4_audit["testtime_zero_intervals"],
            "step4_steptime_zero_intervals": step4_audit["steptime_zero_intervals"],
            "step10_testtime_zero_intervals": step10_audit["testtime_zero_intervals"],
            "step10_steptime_zero_intervals": step10_audit["steptime_zero_intervals"],
            "step4_testtime_median_interval_native": step4_audit["testtime_dt_median_native"],
            "step4_steptime_median_interval_native": step4_audit["steptime_dt_median_native"],
            "step4_voltage_first_v": step4_audit["voltage_first_v"],
            "step4_voltage_last_v": step4_audit["voltage_last_v"],
            "step10_voltage_first_v": step10_audit["voltage_first_v"],
            "step10_voltage_last_v": step10_audit["voltage_last_v"],
            "temperature_raw_column": alignment["temperature_raw_column"],
            "temperature_column_status": alignment["temperature_column_status"],
            "temperature_native_mean_audit_only": alignment["temperature_native_mean"],
            "temperature_semantics": alignment["temperature_semantics"],
            "same_rpt_step_order_valid": alignment["same_rpt_step_order_valid"],
            "predictive_temperature_feature": False,
            "predictive_label_or_bol_feature": False,
            "future_rpt_feature": False,
            "curve_progress_basis": "normalized source-row index",
            "old_step4_curve_max_abs_delta_v": record["old_discharge_max_abs_delta_v"],
            "old_step10_curve_max_abs_delta_v": record["old_charge_max_abs_delta_v"],
        }
        for feature_index, name in enumerate(P5_FEATURE_NAMES):
            audit_row[f"p5_raw_{name}"] = record["p5_raw"][name]
            audit_row[f"p5_scaled_{name}"] = float(
                scaled_physics[index, feature_index]
            )
        feature_audit_rows.append(audit_row)

    split_manifest_path = P5_DIR / "bafuse_p5_split_manifest.csv"
    _write_csv_atomic(
        split_manifest_path,
        list(split_manifest_rows[0]),
        split_manifest_rows,
    )
    output_paths[split_manifest_path.name] = split_manifest_path
    feature_audit_path = P5_DIR / "bafuse_p5_feature_audit.csv"
    _write_csv_atomic(
        feature_audit_path,
        list(feature_audit_rows[0]),
        feature_audit_rows,
    )
    output_paths[feature_audit_path.name] = feature_audit_path

    duplicate_path = P5_DIR / "bafuse_p5_duplicate_sensitivity.csv"
    _write_csv_atomic(duplicate_path, list(duplicate_rows[0]), duplicate_rows)
    output_paths[duplicate_path.name] = duplicate_path

    forward_path = P5_DIR / "bafuse_p5_model_forward_pass_status.json"
    _write_json_atomic(forward_path, forward_status)
    output_paths[forward_path.name] = forward_path

    crosspart_path = P5_DIR / "bafuse_p5_cross_part_audit.md"
    _write_text_atomic(crosspart_path, _cross_part_report(records))
    output_paths[crosspart_path.name] = crosspart_path
    quality_path = P5_DIR / "bafuse_p5_quality_report.md"
    _write_text_atomic(
        quality_path,
        _quality_report(
            records,
            raw_physics,
            scaled_physics,
            scaler_means,
            scaler_scales,
            constant_features,
            duplicate_rows,
            forward_status,
        ),
    )
    output_paths[quality_path.name] = quality_path

    for partition, count in EXPECTED_COUNTS.items():
        path = P5_DIR / SPLIT_FILE_NAMES[partition]
        _verify_frozen_npz(path, count)
    if _sha256(OUTPUT_LABEL) != EXPECTED_LABEL_SHA256:
        raise ValueError("Frozen label copy does not match the approved source.")
    if len(split_manifest_rows) != 203 or len(feature_audit_rows) != 203:
        raise ValueError("Row-level frozen manifests do not contain all 203 records.")

    output_hashes = {
        name: _sha256(path) for name, path in sorted(output_paths.items())
    }
    freeze_report_path = P5_DIR / "bafuse_p5_dataset_freeze_report.md"
    _write_text_atomic(
        freeze_report_path,
        _freeze_report(records, forward_status, scaler_means, scaler_scales, output_hashes),
    )
    output_paths[freeze_report_path.name] = freeze_report_path
    output_hashes[freeze_report_path.name] = _sha256(freeze_report_path)

    input_hash_paths = {
        "authoritative_labels": LABELS_PATH,
        "feature_blocker_manifest": BLOCKER_MANIFEST_PATH,
        "candidate_manifest": CANDIDATES_PATH,
        "alignment_audit": ALIGNMENT_PATH,
        "m1_measurement_block_audit": raw.M1_BLOCK_AUDIT,
        "raw_source_hash_manifest": SOURCE_HASHES_PATH,
        "feature_blocker_config": FEATURE_BLOCKER_CONFIG_PATH,
        "historical_curve_matrix_crosscheck_only": OLD_MATRIX_PATH,
        "locked_cell_split": CELL_SPLIT_PATH,
        "cuda_dependency_lock": LOCK_PATH,
        "run_config": RUN_CONFIG_PATH,
    }
    code_paths = {
        "p5_runner": Path(__file__).resolve(),
        "p5_feature_module": PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "p5_electrical_features.py",
        "architecture_models": PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "architecture_models.py",
        "raw_reconstruction_scanner": PROJECT_ROOT / "src" / "bafuse" / "labeling" / "raw_soh_reconstruction.py",
        "p5_regression_tests": PROJECT_ROOT / "tests" / "test_bafuse_p5_electrical_features.py",
    }
    checksum_manifest = {
        "manifest_version": 1,
        "dataset_identifier": "bafuse-p5-electrical-v1-20261114",
        "training_authorized": False,
        "external_prediction_evaluation_authorized": False,
        "inputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in sorted(input_hash_paths.items())
        },
        "code": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in sorted(code_paths.items())
        },
        "raw_source_hash_manifest_entries": {
            name: {
                "sha256": row["sha256"],
                "expected_sha256": row["expected_sha256"],
                "status": row["status"],
                "source_size_bytes": int(row["source_size_bytes"]),
            }
            for name, row in sorted(source_hash_by_path.items())
        },
        "outputs": {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in sorted(output_paths.items())
        },
        "manifest_self_hash": "excluded; calculate after writing this file",
    }
    checksum_path = P5_DIR / "bafuse_p5_sha256_manifest.json"
    _write_json_atomic(checksum_path, checksum_manifest)
    checksum_manifest_hash = _sha256(checksum_path)
    print(f"Dataset identifier: {checksum_manifest['dataset_identifier']}", flush=True)
    print(f"SHA-256 manifest: {checksum_manifest_hash}", flush=True)
    print(
        f"Split counts: {EXPECTED_COUNTS}; P5 shapes validated; "
        f"forward passes: {len(forward_status['forward_passes'])}/4.",
        flush=True,
    )
    return {
        "dataset_identifier": checksum_manifest["dataset_identifier"],
        "checksum_manifest_path": checksum_path,
        "checksum_manifest_sha256": checksum_manifest_hash,
        "counts": dict(EXPECTED_COUNTS),
        "records": len(records),
        "feature_audit_path": feature_audit_path,
        "duplicate_sensitivity": duplicate_rows,
        "forward_status": forward_status,
        "scaler_means": [float(value) for value in scaler_means],
        "scaler_scales": [float(value) for value in scaler_scales],
        "manifest_outputs": output_hashes,
    }


if __name__ == "__main__":
    try:
        result = run()
    except Exception as error:
        print(f"P5 freeze aborted: {type(error).__name__}: {error}", file=sys.stderr)
        raise
    print(json.dumps(result, indent=2, default=str), flush=True)
