from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_DIR = Path(__file__).resolve().parent
OBSERVATIONS_PATH = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
SPLIT_PATH = BENCHMARK_DIR / "cell_split.csv"
STRICT_PART1_PATH = (
    ROOT
    / "reports"
    / "bafuse_final_data_validation_v2"
    / "soh_labels_verified_candidates.csv"
)
PART2_LABELS_PATH = (
    ROOT
    / "data"
    / "processed"
    / "label_soh_part1_2"
    / "run_010_part2_soh_candidates"
    / "labels_soh_part2.csv"
)
PART3_LABELS_PATH = (
    ROOT
    / "experiments"
    / "results"
    / "data_audits"
    / "part3_soh_candidates"
    / "labels_soh_candidates.csv"
)
MODALITY_PATH = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_three_encoder_v1"
    / "modality_availability.csv"
)
RAW_OXFORD_DIR = ROOT / "data" / "raw" / "oxford"

EXPECTED_PART1_STRICT_KEYS = 94
EXPECTED_PART2_SELECTED_KEYS = 54
PARTITION_COUNTS = {"train": 9, "validation": 3, "external_test": 12}
CUTOFF_VOLTAGE_V = 2.5
CUTOFF_TOLERANCE_V = 0.05

Key = tuple[str, str, str]
SEGMENT_PATTERN = re.compile(r"#Step(\d+):.*?rows(\d+)-(\d+)", re.IGNORECASE)
EIS_FILENAME_PATTERN = re.compile(
    r"Group(?P<group>\d+)_(?P<checkpoint>BoL|MoL|EoL)_"
    r"Cell(?P<cell>\d+)_SOC(?P<soc>\d+)(?P<suffix>.*)\.csv$",
    re.IGNORECASE,
)
EIS_SUFFIX_ID_PATTERN = re.compile(r"G(?P<group>\d+)Cell(?P<cell>\d+)", re.IGNORECASE)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Required benchmark input is missing: {path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty preparation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _key(part: str, cell: str, measurement: str) -> Key:
    measurement = measurement.strip()
    if not measurement.upper().startswith("RPT"):
        raise ValueError(f"Expected an RPT measurement, got {measurement!r}")
    return (str(int(part)), cell.strip(), measurement[3:].strip())


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _flag(value: Any) -> bool | None:
    if value is True or str(value).strip().casefold() == "true":
        return True
    if value is False or str(value).strip().casefold() == "false":
        return False
    return None


def _normalize_path(value: str) -> str:
    return str(value).replace("\\", "/").lstrip("./").casefold()


def load_strict_part1_labels(path: Path = STRICT_PART1_PATH) -> dict[Key, dict[str, str]]:
    rows = _read_csv(path)
    if len(rows) != EXPECTED_PART1_STRICT_KEYS:
        raise ValueError(
            "The strict Part 1 audit must contain exactly "
            f"{EXPECTED_PART1_STRICT_KEYS} rows, found {len(rows)}."
        )

    labels: dict[Key, dict[str, str]] = {}
    for row in rows:
        if (
            row.get("dataset_part") != "1"
            or row.get("audit_status") != "verified_source_and_arithmetic"
            or row.get("validation_status_v2") != "VERIFIED_DERIVED_LABEL"
        ):
            raise ValueError(
                "Strict Part 1 audit contains a row outside the approved status."
            )
        key = _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        if key in labels:
            raise ValueError(f"Duplicate strict Part 1 audit key: {key}")
        if _finite_float(row.get("stored_soh_percent")) is None:
            raise ValueError(f"Strict Part 1 key has no finite SOH value: {key}")
        labels[key] = row
    if len(labels) != EXPECTED_PART1_STRICT_KEYS:
        raise ValueError("Strict Part 1 audit keys are not unique.")
    return labels


def load_selected_part2_labels(
    path: Path = PART2_LABELS_PATH,
) -> dict[Key, dict[str, str]]:
    rows = _read_csv(path)
    selected = [
        row
        for row in rows
        if row.get("dataset_part") == "2"
        and row.get("condition") == "RPT"
        and row.get("group_id") in {"1", "2", "3", "4"}
        and row.get("validation_status") == "calculated_from_measured_capacity"
        and row.get("label_confidence") == "medium_ratio_only"
    ]
    if len(selected) != EXPECTED_PART2_SELECTED_KEYS:
        raise ValueError(
            "The selected Part 2 candidate table must contribute exactly "
            f"{EXPECTED_PART2_SELECTED_KEYS} Group 1-4 RPT keys, found {len(selected)}."
        )
    labels: dict[Key, dict[str, str]] = {}
    for row in selected:
        key = _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        if key in labels:
            raise ValueError(f"Duplicate selected Part 2 candidate key: {key}")
        if _finite_float(row.get("soh_percent")) is None:
            raise ValueError(f"Selected Part 2 key has no finite SOH value: {key}")
        labels[key] = row
    return labels


def load_part3_external_labels(path: Path = PART3_LABELS_PATH) -> dict[Key, dict[str, str]]:
    rows = _read_csv(path)
    labels: dict[Key, dict[str, str]] = {}
    for row in rows:
        if (
            row.get("dataset_part") != "3"
            or row.get("condition") != "RPT"
            or row.get("validation_status") != "calculated_from_measured_capacity"
            or not row.get("measurement_id", "").startswith("RPT")
        ):
            continue
        key = _key(row["dataset_part"], row["physical_cell_id"], row["measurement_id"])
        if key in labels:
            raise ValueError(f"Duplicate Part 3 external candidate key: {key}")
        if _finite_float(row.get("soh_percent")) is None:
            raise ValueError(f"Part 3 external candidate has no finite SOH value: {key}")
        labels[key] = row
    return labels


def _parse_segment(reference: str) -> tuple[int, int, int] | None:
    match = SEGMENT_PATTERN.search(reference)
    if not match:
        return None
    step, start, end = (int(value) for value in match.groups())
    if start < 2 or end < start:
        raise ValueError(f"Invalid source line range in segment reference: {reference}")
    return step, start, end


def _segment_summary(
    source_path: Path, expected_step: int, start: int, end: int
) -> dict[str, Any]:
    with source_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Raw TPG file has no header: {source_path}")
        columns = {name.strip().casefold(): index for index, name in enumerate(header)}
        required = {"step", "volts", "amps", "amphr", "steptime"}
        missing = required - set(columns)
        if missing:
            raise ValueError(
                f"Raw TPG file is missing required columns {sorted(missing)}: {source_path}"
            )
        temperature_column = columns.get("temp1", columns.get("temp"))
        row_count = 0
        step_ids: set[int | None] = set()
        voltage_values: list[float] = []
        current_values: list[float] = []
        capacity_values: list[float] = []
        step_times: list[float] = []
        temperatures: list[float] = []
        missing_temperature = 0
        missing_core_values = 0
        for values in reader:
            line_number = reader.line_num
            if line_number > end:
                break
            if line_number < start:
                continue
            row_count += 1

            def value(column: str) -> float | None:
                try:
                    return _finite_float(values[columns[column]])
                except (IndexError, KeyError):
                    return None

            step_value = value("step")
            step_ids.add(
                int(step_value)
                if step_value is not None and step_value.is_integer()
                else None
            )
            for name, sink in (
                ("volts", voltage_values),
                ("amps", current_values),
                ("amphr", capacity_values),
                ("steptime", step_times),
            ):
                number = value(name)
                if number is None:
                    missing_core_values += 1
                else:
                    sink.append(number)
            if temperature_column is None:
                missing_temperature += 1
            else:
                temp = _finite_float(
                    values[temperature_column]
                    if temperature_column < len(values)
                    else None
                )
                if temp is None:
                    missing_temperature += 1
                else:
                    temperatures.append(temp)

    expected_count = end - start + 1
    if row_count != expected_count:
        raise ValueError(
            f"Mapped Step {expected_step} expected {expected_count} rows, "
            f"observed {row_count} in {source_path}."
        )
    return {
        "expected_step": expected_step,
        "start_line": start,
        "end_line": end,
        "row_count": row_count,
        "observed_step_ids": sorted(
            -1 if step is None else step for step in step_ids
        ),
        "voltage_min_v": min(voltage_values) if voltage_values else None,
        "voltage_max_v": max(voltage_values) if voltage_values else None,
        "current_min_native": min(current_values) if current_values else None,
        "current_max_native": max(current_values) if current_values else None,
        "current_negative_rows": sum(value < 0 for value in current_values),
        "current_zero_rows": sum(value == 0 for value in current_values),
        "current_positive_rows": sum(value > 0 for value in current_values),
        "capacity_min_native_amphr": min(capacity_values) if capacity_values else None,
        "capacity_max_native_amphr": max(capacity_values) if capacity_values else None,
        "capacity_span_native_amphr": (
            max(capacity_values) - min(capacity_values) if capacity_values else None
        ),
        "steptime_first_native": step_times[0] if step_times else None,
        "steptime_last_native": step_times[-1] if step_times else None,
        "temperature_finite_rows": len(temperatures),
        "temperature_missing_rows": missing_temperature,
        "missing_core_values": missing_core_values,
        "all_core_values_finite": missing_core_values == 0,
        "step_ids_match": step_ids == {expected_step},
    }


def reextract_g1_cell9_rpt8(
    observations: list[dict[str, str]] | None = None,
    strict_labels: dict[Key, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if observations is None:
        observations = _read_csv(OBSERVATIONS_PATH)
    if strict_labels is None:
        strict_labels = load_strict_part1_labels()
    rows = [
        row
        for row in observations
        if row.get("part") == "1"
        and row.get("physical_cell_id") == "G1-Cell9"
        and row.get("rpt_index") == "8"
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one mapped G1-Cell9 RPT8 record, found {len(rows)}.")
    row = rows[0]
    key = ("1", "G1-Cell9", "8")
    label = strict_labels.get(key)
    if label is None:
        raise ValueError("G1-Cell9 RPT8 is absent from the strict Part 1 audit.")
    step4_ref = _parse_segment(row.get("step4_segment_ref", ""))
    step10_ref = _parse_segment(row.get("step10_segment_ref", ""))
    if step4_ref is None or step10_ref is None:
        raise ValueError("G1-Cell9 RPT8 has no parseable Step 4/Step 10 mapping.")
    if step4_ref[0] != 4 or step10_ref[0] != 10:
        raise ValueError("G1-Cell9 RPT8 segment references have unexpected step IDs.")
    if row.get("step4_status") != "verified" or row.get("step10_status") != "verified":
        raise ValueError("G1-Cell9 RPT8 Guide-mapped Step 4/10 is not verified.")

    source_path = ROOT / Path(row["source_file"].replace("\\", "/"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Mapped G1-Cell9 RPT8 source is missing: {source_path}")
    step4 = _segment_summary(source_path, *step4_ref)
    step10 = _segment_summary(source_path, *step10_ref)
    label_value = _finite_float(label.get("stored_soh_percent"))
    raw_capacity = _finite_float(label.get("raw_capacity_span_native"))
    baseline_capacity = _finite_float(label.get("raw_baseline_capacity_native"))
    if label_value is None or raw_capacity is None or baseline_capacity in (None, 0):
        raise ValueError("G1-Cell9 RPT8 strict label provenance is incomplete.")
    recomputed_soh = 100.0 * raw_capacity / baseline_capacity
    step4_reaches_cutoff = (
        step4["voltage_min_v"] is not None
        and step4["voltage_min_v"] <= CUTOFF_VOLTAGE_V + CUTOFF_TOLERANCE_V
    )
    result = {
        "part": 1,
        "physical_cell_id": "G1-Cell9",
        "rpt_index": 8,
        "source_tpg": row["source_tpg"],
        "source_file": row["source_file"],
        "guide_stage": row["guide_stage"],
        "guide_mapping_status": row["mapping_status"],
        "strict_part1_key": True,
        "label_audit_status": label["audit_status"],
        "label_source_status": label["validation_status_v2"],
        "label_source_file": label["source_file"],
        "baseline_source_file": label["baseline_source_file"],
        "raw_capacity_span_native_amphr": raw_capacity,
        "raw_baseline_capacity_native_amphr": baseline_capacity,
        "recomputed_soh_percent": recomputed_soh,
        "stored_soh_percent": label_value,
        "label_arithmetic_matches": math.isclose(
            recomputed_soh, label_value, rel_tol=0.0, abs_tol=1e-10
        ),
        "step4": step4,
        "step4_cutoff_v": CUTOFF_VOLTAGE_V,
        "step4_cutoff_tolerance_v": CUTOFF_TOLERANCE_V,
        "step4_reaches_documented_cutoff": step4_reaches_cutoff,
        "step10": step10,
        "temperature_present_in_step4": step4["temperature_missing_rows"] == 0,
        "benchmark_label_eligible": bool(
            step4_reaches_cutoff
            and step4["all_core_values_finite"]
            and step4["step_ids_match"]
            and step10["all_core_values_finite"]
            and step10["step_ids_match"]
            and math.isclose(recomputed_soh, label_value, rel_tol=0.0, abs_tol=1e-10)
        ),
        "eligibility_reason": (
            "excluded: mapped Step 4 is a partial discharge that does not reach "
            "the documented 2.5 V cutoff; strict audit verifies source/arithmetic "
            "but not completeness of the discharge capacity curve"
            if not step4_reaches_cutoff
            else "eligible subject to remaining benchmark preflight checks"
        ),
    }
    return result


def _load_cell_splits() -> dict[str, str]:
    rows = _read_csv(SPLIT_PATH)
    assignments: dict[str, str] = {}
    for row in rows:
        cell = row["physical_cell_id"]
        if cell in assignments:
            raise ValueError(f"Duplicate physical cell in benchmark split: {cell}")
        assignments[cell] = row["partition"]
    counts = Counter(assignments.values())
    if dict(counts) != PARTITION_COUNTS or len(assignments) != 24:
        raise ValueError(f"Unexpected benchmark cell split: {dict(counts)}")
    return assignments


def _index_unique(rows: list[dict[str, str]], source: str) -> dict[Key, dict[str, str]]:
    indexed: dict[Key, dict[str, str]] = {}
    for row in rows:
        key = _key(row["part"], row["physical_cell_id"], f"RPT{row['rpt_index']}")
        if key in indexed:
            raise ValueError(f"Duplicate key in {source}: {key}")
        indexed[key] = row
    return indexed


def _load_eis_inventory(
    known_cells_by_part: dict[str, set[str]],
) -> list[dict[str, Any]]:
    paths = sorted(
        path
        for path in RAW_OXFORD_DIR.rglob("*.csv")
        if "eis" in {part.casefold() for part in path.parts}
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        relative = path.relative_to(RAW_OXFORD_DIR)
        part_dir = relative.parts[0]
        part = part_dir.removeprefix("part")
        match = EIS_FILENAME_PATTERN.match(path.name)
        group: str | None = None
        cell_number: str | None = None
        checkpoint = ""
        soc = ""
        suffix = ""
        parsed_cell = ""
        identity_status = "unparsed_filename"
        if match:
            group = match.group("group")
            cell_number = match.group("cell")
            checkpoint = match.group("checkpoint")
            soc = match.group("soc")
            suffix = match.group("suffix")
            parsed_cell = f"G{group}-Cell{cell_number}"
            suffix_match = EIS_SUFFIX_ID_PATTERN.search(suffix)
            if suffix_match and (
                suffix_match.group("group") != group
                or suffix_match.group("cell") != cell_number
            ):
                identity_status = "ambiguous_prefix_suffix_identity"
            elif parsed_cell in known_cells_by_part.get(part, set()):
                identity_status = "matched_exact_part_cell_manifest"
            else:
                identity_status = "filename_identity_not_in_part_manifest"

        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader, [])
            data_rows = sum(1 for _ in reader)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        spectrum_type = (
            "roh_only_not_full_spectrum"
            if "roh" in path.name.casefold()
            else "spectrum"
        )
        exact_identity = (
            parsed_cell
            if identity_status == "matched_exact_part_cell_manifest"
            else ""
        )
        rows.append(
            {
                "part": part,
                "physical_cell_id": exact_identity,
                "filename_group": group or "",
                "filename_cell": cell_number or "",
                "cell_identity_status": identity_status,
                "checkpoint": checkpoint,
                "soc_percent": soc,
                "source_file": relative.as_posix(),
                "spectrum_type": spectrum_type,
                "data_row_count": data_rows,
                "frequency_column_present": any(
                    "freq" in name.casefold() for name in header
                ),
                "sha256": digest,
                "rpt_index": "",
                "rpt_alignment_status": "checkpoint_only_no_exact_rpt_mapping",
                "same_rpt_predictor_eligible": False,
            }
        )
    digest_counts = Counter(row["sha256"] for row in rows)
    for row in rows:
        row["identical_file_count_across_parts"] = digest_counts[row["sha256"]]
    return rows


def _build_preflight_data() -> dict[str, Any]:
    splits = _load_cell_splits()
    strict_p1 = load_strict_part1_labels()
    selected_p2 = load_selected_part2_labels()
    external_p3 = load_part3_external_labels()
    observations = _read_csv(OBSERVATIONS_PATH)
    modalities = _index_unique(_read_csv(MODALITY_PATH), str(MODALITY_PATH))

    observation_keys: set[Key] = set()
    selected_observations: list[dict[str, str]] = []
    known_cells_by_part: dict[str, set[str]] = defaultdict(set)
    for row in observations:
        key = _key(row["part"], row["physical_cell_id"], f"RPT{row['rpt_index']}")
        if key in observation_keys:
            raise ValueError(f"Duplicate canonical observation key: {key}")
        observation_keys.add(key)
        known_cells_by_part[row["part"]].add(row["physical_cell_id"])
        if row["physical_cell_id"] in splits:
            selected_observations.append(row)
    for key in strict_p1:
        if key not in observation_keys:
            raise ValueError(f"Strict Part 1 key is absent from mapped observations: {key}")
    for key in selected_p2:
        if key not in observation_keys:
            raise ValueError(f"Selected Part 2 key is absent from mapped observations: {key}")
    for key in external_p3:
        if key not in observation_keys:
            raise ValueError(f"Part 3 external key is absent from mapped observations: {key}")

    rpt8 = reextract_g1_cell9_rpt8(observations, strict_p1)
    part1_keys = set(strict_p1)
    part2_keys = set(selected_p2)
    part3_keys = set(external_p3)
    records: list[dict[str, Any]] = []
    for source in selected_observations:
        part = source["part"]
        cell = source["physical_cell_id"]
        rpt = str(source["rpt_index"])
        key = (part, cell, rpt)
        partition = splits[cell]
        if part == "3" and partition != "external_test":
            raise ValueError(f"Part 3 cell is not external-only: {cell}")
        if part in {"1", "2"} and partition not in {"train", "validation"}:
            raise ValueError(f"Part 1/2 cell has an invalid benchmark split: {cell}")
        labels = strict_p1 if part == "1" else selected_p2 if part == "2" else external_p3
        label = labels.get(key)
        modality = modalities.get(key)
        rpt8_record = key == ("1", "G1-Cell9", "8")
        source_exists = (ROOT / Path(source["source_file"].replace("\\", "/"))).is_file()
        step4_mapped = source.get("step4_status") == "verified"
        step10_mapped = source.get("step10_status") == "verified"
        step4_previous = _flag(
            modality.get("step4_raw_curve_valid") if modality else None
        )
        step10_previous = _flag(
            modality.get("step10_raw_curve_valid") if modality else None
        )
        temperature_previous = _flag(
            modality.get("temperature_complete") if modality else None
        )
        physics_previous = _flag(
            modality.get("physics_features_complete") if modality else None
        )
        all_modalities_previous = _flag(
            modality.get("all_modalities_complete") if modality else None
        )
        step4_curve_available = bool(step4_mapped and step4_previous is True)
        step10_curve_available = bool(step10_mapped and step10_previous is True)
        physics_available = physics_previous is True
        temperature_available = temperature_previous
        step4_cutoff_status = "not_independently_rechecked_in_this_preparation"
        if rpt8_record:
            step4_curve_available = False
            physics_available = False
            temperature_available = bool(rpt8["temperature_present_in_step4"])
            step4_cutoff_status = "failed_2.5V_cutoff_check"
            all_modalities_previous = False

        selected_label = (
            _finite_float(
                label.get("stored_soh_percent")
                if part == "1"
                else label.get("soh_percent")
            )
            if label
            else None
        )
        label_source = (
            "strict_part1_94_key_audit"
            if part == "1" and label
            else "user_selected_part2_group1_4_candidates"
            if part == "2" and label
            else "provisional_part3_external_candidate"
            if part == "3" and label
            else "not_selected_by_benchmark_label_policy"
        )
        label_policy_selected = label is not None
        if not label_policy_selected:
            reason = (
                "part1_label_key_not_in_exact_94_key_audit"
                if part == "1"
                else "part2_label_key_not_in_selected_54_candidates"
                if part == "2"
                else "no_provisional_part3_rpt_label"
            )
        elif rpt8_record:
            reason = "incomplete_step4_discharge_does_not_reach_2.5V_cutoff"
        elif not source_exists:
            reason = "mapped_source_file_missing"
        elif not step4_mapped or not step10_mapped:
            reason = "step4_or_step10_mapping_not_verified"
        elif not step4_curve_available or not step10_curve_available:
            reason = "raw_step4_or_step10_not_complete_in_existing_export"
        elif not physics_available:
            reason = "step4_physics_features_not_complete_in_existing_export"
        elif all_modalities_previous is not True:
            reason = "existing_export_does_not_mark_all_modalities_complete"
        else:
            reason = "candidate_from_existing_complete_modality_export"

        input_complete_from_previous_export = bool(
            source_exists
            and step4_mapped
            and step10_mapped
            and step4_curve_available
            and step10_curve_available
            and physics_available
            and all_modalities_previous is True
        )
        dataset_admitted = bool(
            partition in {"train", "validation"}
            and part in {"1", "2"}
            and label_policy_selected
            and input_complete_from_previous_export
            and not rpt8_record
        )
        external_candidate = bool(
            part == "3"
            and partition == "external_test"
            and label_policy_selected
            and input_complete_from_previous_export
        )
        records.append(
            {
                "part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "measurement_id": f"RPT{rpt}",
                "partition": partition,
                "source_tpg": source["source_tpg"],
                "source_file": source["source_file"],
                "guide_stage": source["guide_stage"],
                "mapping_status": source["mapping_status"],
                "step4_segment_ref": source["step4_segment_ref"],
                "step10_segment_ref": source["step10_segment_ref"],
                "source_file_exists": source_exists,
                "strict_part1_key": key in part1_keys,
                "selected_part2_key": key in part2_keys,
                "provisional_part3_key": key in part3_keys,
                "label_policy_selected": label_policy_selected,
                "label_source": label_source,
                "soh_candidate_percent": (
                    selected_label if selected_label is not None else ""
                ),
                "label_is_official_ground_truth": False if label else "",
                "step4_mapping_verified": step4_mapped,
                "step10_mapping_verified": step10_mapped,
                "step4_raw_curve_valid_previous_export": (
                    step4_previous if step4_previous is not None else ""
                ),
                "step10_raw_curve_valid_previous_export": (
                    step10_previous if step10_previous is not None else ""
                ),
                "step4_protocol_cutoff_status": step4_cutoff_status,
                "temperature_complete_previous_export": (
                    temperature_previous if temperature_previous is not None else ""
                ),
                "temperature_measurements_present": (
                    temperature_available
                    if temperature_available is not None
                    else ""
                ),
                "temperature_measurement_provenance": (
                    "raw_RPT8_reextraction"
                    if rpt8_record
                    else "existing_modality_export"
                    if temperature_previous is not None
                    else "not_verified"
                ),
                "step4_curve_complete_for_model": step4_curve_available,
                "physics_features_complete_previous_export": (
                    physics_available if physics_previous is not None or rpt8_record else ""
                ),
                "all_modalities_complete_previous_export": (
                    all_modalities_previous
                    if all_modalities_previous is not None
                    else ""
                ),
                "input_complete_from_existing_export": input_complete_from_previous_export,
                "dataset_construction_admitted": dataset_admitted,
                "external_part3_evaluation_candidate": external_candidate,
                "training_authorized": False,
                "eligibility_reason": reason,
            }
        )

    for record in records:
        if record["part"] == "1" and record["dataset_construction_admitted"]:
            key = (
                record["part"],
                record["physical_cell_id"],
                record["rpt_index"],
            )
            if key not in part1_keys:
                raise ValueError(f"Unapproved Part 1 key entered dataset construction: {key}")
        if record["part"] == "3" and record["dataset_construction_admitted"]:
            raise ValueError("Part 3 record entered train/validation dataset construction.")
        if record["part"] == "3" and record["partition"] != "external_test":
            raise ValueError("Part 3 record is not isolated to external evaluation.")

    eis_rows = _load_eis_inventory(known_cells_by_part)
    eis_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for eis_row in eis_rows:
        if eis_row["physical_cell_id"]:
            eis_by_cell[(eis_row["part"], eis_row["physical_cell_id"])].append(eis_row)
    alignment_rows: list[dict[str, Any]] = []
    for source in selected_observations:
        part = source["part"]
        cell = source["physical_cell_id"]
        rpt = str(source["rpt_index"])
        record = next(
            item
            for item in records
            if item["part"] == part
            and item["physical_cell_id"] == cell
            and item["rpt_index"] == rpt
        )
        step4_ref = _parse_segment(source.get("step4_segment_ref", ""))
        step10_ref = _parse_segment(source.get("step10_segment_ref", ""))
        step4_source = (
            source["step4_segment_ref"].split("#", 1)[0]
            if "#" in source["step4_segment_ref"]
            else ""
        )
        step10_source = (
            source["step10_segment_ref"].split("#", 1)[0]
            if "#" in source["step10_segment_ref"]
            else ""
        )
        aligned = bool(
            step4_ref
            and step10_ref
            and step4_ref[0] == 4
            and step10_ref[0] == 10
            and _normalize_path(step4_source) == _normalize_path(source["source_file"])
            and _normalize_path(step10_source) == _normalize_path(source["source_file"])
            and source["step4_status"] == "verified"
            and source["step10_status"] == "verified"
        )
        checkpoints = sorted(
            {
                f"{item['checkpoint']}:SOC{item['soc_percent']}"
                for item in eis_by_cell.get((part, cell), [])
            }
        )
        alignment_rows.append(
            {
                "part": part,
                "physical_cell_id": cell,
                "rpt_index": source["rpt_index"],
                "source_file": source["source_file"],
                "step4_same_cell_rpt_mapping_verified": aligned,
                "step4_step": 4,
                "step10_same_cell_rpt_mapping_verified": aligned,
                "step10_step": 10,
                "physics_source": (
                    "same_cell_same_RPT_Step4"
                    if aligned
                    else "unverified_not_joined"
                ),
                "physics_temperature_availability": (
                    "present_in_raw_RPT8_reextraction"
                    if record["temperature_measurement_provenance"]
                    == "raw_RPT8_reextraction"
                    and record["temperature_measurements_present"]
                    else "missing_in_existing_export"
                    if record["temperature_measurements_present"] is False
                    else "complete_in_existing_export"
                    if record["temperature_measurements_present"] is True
                    else "not_verified"
                ),
                "eis_checkpoints_for_exact_part_cell": ";".join(checkpoints),
                "eis_rpt_index": "",
                "eis_alignment_status": "checkpoint_only_no_exact_RPT_or_timestamp_join",
                "eis_same_rpt_feature_allowed": False,
            }
        )

    physics_rows = _build_physics_coverage(records)
    cell_rows = _build_cell_eligibility(records, splits)
    admitted_rows = [
        row for row in records if row["dataset_construction_admitted"]
    ]
    external_rows = [
        row for row in records if row["external_part3_evaluation_candidate"]
    ]
    if any(row["part"] == "1" and not row["strict_part1_key"] for row in admitted_rows):
        raise ValueError("Pretraining dataset candidates include an unapproved Part 1 key.")
    return {
        "strict_p1_labels": strict_p1,
        "selected_p2_labels": selected_p2,
        "external_p3_labels": external_p3,
        "records": records,
        "admitted": admitted_rows,
        "external": external_rows,
        "cells": cell_rows,
        "physics": physics_rows,
        "eis": eis_rows,
        "alignment": alignment_rows,
        "rpt8": rpt8,
        "split_counts": dict(Counter(splits.values())),
    }


def _build_physics_coverage(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    features = [
        (
            "mean_discharge_temperature_native",
            "Temp1 (Part 1); Temp (Parts 2-3)",
            "mean(Step4 temperature)",
            "native temperature units; °C not verified from column definition",
            "temperature",
            "descriptive thermal summary; sensor calibration/location not documented",
        ),
        (
            "discharge_temperature_rise_native",
            "Temp1 (Part 1); Temp (Parts 2-3)",
            "max(Step4 temperature) - first(Step4 temperature)",
            "native temperature units; °C not verified from column definition",
            "temperature",
            "same-RPT thermal change; not an independently validated degradation mechanism",
        ),
        (
            "discharge_plateau_slope_v_per_progress",
            "Volts",
            "OLS slope of resampled Step4 voltage vs normalized sample progress, 20%-80%",
            "V per unit normalized progress",
            "voltage",
            "the 20%-80% window is operational, not a dataset-verified electrochemical plateau",
        ),
        (
            "discharge_slope_variability_v_per_progress",
            "Volts",
            "1.4826 * MAD of central dV/dprogress over 10%-90% of the 128-point curve",
            "V per unit normalized progress",
            "voltage",
            "derivative noise/stability has not been validated across all RPTs",
        ),
        (
            "mean_absolute_discharge_current_native",
            "Amps",
            "mean(abs(Step4 current))",
            "native current units; header/protocol magnitude is consistent with A but not declared",
            "current",
            "primarily reflects the fixed C/2 protocol setpoint; independent predictive value unproven",
        ),
    ]
    output: list[dict[str, Any]] = []
    for name, columns, formula, units, category, meaning in features:
        row: dict[str, Any] = {
            "feature_name": name,
            "category": category,
            "source_step": 4,
            "source_columns": columns,
            "formula": formula,
            "units": units,
            "temporal_scope": "same physical cell and same RPT; Step4 only; no future observations",
            "target_leakage_assessment": (
                "no capacity/Amphr axis or target-derived quantity used; "
                "same-RPT feature only"
            ),
            "physical_meaning_status": meaning,
            "temperature_missingness_policy": "preserve missing; never impute from another RPT/cell",
        }
        for part in ("1", "2", "3"):
            candidates = [
                record
                for record in records
                if record["part"] == part and record["label_policy_selected"]
            ]
            if category == "temperature":
                available = sum(
                    record["step4_curve_complete_for_model"]
                    and record["temperature_measurements_present"] is True
                    for record in candidates
                )
            else:
                available = sum(
                    record["step4_curve_complete_for_model"]
                    for record in candidates
                )
            row[f"part{part}_label_candidate_count"] = len(candidates)
            row[f"part{part}_available_count_proxy"] = available
            row[f"part{part}_coverage_source"] = (
                "existing modality_availability.csv; not a new all-row raw-value scan"
            )
        output.append(row)
    return output


def _build_cell_eligibility(
    records: list[dict[str, Any]], splits: dict[str, str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell, partition in sorted(splits.items()):
        cell_records = [row for row in records if row["physical_cell_id"] == cell]
        rows.append(
            {
                "physical_cell_id": cell,
                "partition": partition,
                "dataset_parts": "1;2" if partition in {"train", "validation"} else "3",
                "mapped_rpt_records": len(cell_records),
                "part1_strict_label_keys": sum(
                    row["part"] == "1" and row["strict_part1_key"]
                    for row in cell_records
                ),
                "part2_selected_label_keys": sum(
                    row["part"] == "2" and row["selected_part2_key"]
                    for row in cell_records
                ),
                "part3_provisional_rpt_keys": sum(
                    row["part"] == "3" and row["provisional_part3_key"]
                    for row in cell_records
                ),
                "pretraining_dataset_candidates": sum(
                    row["dataset_construction_admitted"] for row in cell_records
                ),
                "external_part3_evaluation_candidates": sum(
                    row["external_part3_evaluation_candidate"]
                    for row in cell_records
                ),
                "rpt8_quality_exclusion": cell == "G1-Cell9",
                "training_authorized": False,
            }
        )
    return rows


def write_preflight_artifacts(data: dict[str, Any]) -> dict[str, Any]:
    _write_csv(BENCHMARK_DIR / "record_eligibility.csv", data["records"])
    _write_csv(BENCHMARK_DIR / "cell_eligibility.csv", data["cells"])
    _write_csv(BENCHMARK_DIR / "physics_feature_coverage.csv", data["physics"])
    _write_csv(BENCHMARK_DIR / "eis_checkpoint_manifest.csv", data["eis"])
    _write_csv(BENCHMARK_DIR / "modality_alignment.csv", data["alignment"])
    _write_csv(
        BENCHMARK_DIR / "pretraining_dataset_candidates.csv", data["admitted"]
    )
    _write_csv(
        BENCHMARK_DIR / "external_part3_evaluation_candidates.csv", data["external"]
    )
    (BENCHMARK_DIR / "rpt8_reextraction.json").write_text(
        json.dumps(data["rpt8"], indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "training_authorized": False,
        "training_started": False,
        "split_counts": data["split_counts"],
        "strict_part1_audit_keys": len(data["strict_p1_labels"]),
        "part2_selected_candidate_keys": len(data["selected_p2_labels"]),
        "part3_provisional_rpt_candidate_keys": len(data["external_p3_labels"]),
        "mapped_records_in_benchmark_cells": len(data["records"]),
        "pretraining_dataset_candidates": len(data["admitted"]),
        "external_part3_evaluation_candidates": len(data["external"]),
        "part1_admitted_records": sum(
            row["part"] == "1" for row in data["admitted"]
        ),
        "part1_unapproved_admitted_records": sum(
            row["part"] == "1" and not row["strict_part1_key"]
            for row in data["admitted"]
        ),
        "g1_cell9_rpt8_benchmark_label_eligible": data["rpt8"][
            "benchmark_label_eligible"
        ],
        "eis_csv_files": len(data["eis"]),
        "eis_full_spectra": sum(row["spectrum_type"] == "spectrum" for row in data["eis"]),
        "eis_roh_only_files": sum(
            row["spectrum_type"] == "roh_only_not_full_spectrum" for row in data["eis"]
        ),
        "eis_records_with_exact_part_cell_identity": sum(
            bool(row["physical_cell_id"]) for row in data["eis"]
        ),
        "eis_records_with_exact_rpt_mapping": 0,
    }
    (BENCHMARK_DIR / "preparation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    data = _build_preflight_data()
    summary = write_preflight_artifacts(data)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
