from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from bafuse.features.soh_voltage import (
    EXPECTED_PARTITION_SIZES,
    FEATURE_NAMES,
    SPLIT_VERSION,
    extract_step4_voltage,
    load_split,
    resolve_source_file,
    sha256,
)
from .soh_label_generation import _LABEL_FIELDS, _empty_label, _read_csv

_GROUP5_CELLS = ("G5-Cell5", "G5-Cell6", "G5-Cell16")
_GROUP6_CELLS = ("G6-Cell1",)
_EXPECTED_BASELINES = {
    "G5-Cell5": 2.96727385409,
    "G5-Cell6": 2.958036978968,
    "G5-Cell16": 2.975439868434,
    "G6-Cell1": 2.933130265846,
}
_REQUIRED_GROUP5_STEPS = tuple(range(5, 37))
_MIN_COMPLETE_STEP4_ROWS = 1000
_MIN_COMPLETE_CAPACITY_SPAN = 1.0
_CHUNK_ROWS = 250_000
_LABEL_METHOD = (
    "Part 2 Step 4 discharge capacity span / earliest complete Step 4 segment "
    "in the same cell's first Guide-mapped local BoL procedure * 100"
)
_LABEL_NOTES = (
    "User-approved provisional local-BoL denominator; native Amphr unit is "
    "unverified. RPT order is ordinal; exact cycle age is unavailable."
)
_AUDIT_FIELDS = [
    "physical_cell_id",
    "procedure",
    "source_file",
    "segment_index",
    "step4_rows",
    "capacity_finite_rows",
    "capacity_span",
    "positive_capacity_deltas",
    "negative_capacity_deltas",
    "current_signs",
    "voltage_min_v",
    "voltage_max_v",
    "group5_rpt_signature",
    "selection",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_step4_file(path: Path, chunk_rows: int = _CHUNK_ROWS) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    step_runs: list[tuple[int, int]] = []
    segment_count = 0
    previous_step4 = False

    for chunk in pd.read_csv(
        path,
        usecols=["Step", "Amphr", "Amps", "Volts"],
        chunksize=chunk_rows,
    ):
        step = pd.to_numeric(chunk["Step"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        finite_step = np.isfinite(step)
        rounded_step = np.rint(step[finite_step])
        if not np.allclose(step[finite_step], rounded_step, rtol=0, atol=1e-6):
            raise ValueError(f"Non-integer Step values found in {path}.")
        step_ids = np.full(step.shape, -1, dtype=np.int32)
        step_ids[finite_step] = rounded_step.astype(np.int32)

        if step_ids.size:
            run_starts = np.r_[0, np.flatnonzero(step_ids[1:] != step_ids[:-1]) + 1]
            run_ends = np.r_[run_starts[1:], step_ids.size]
            for start, end in zip(run_starts, run_ends):
                run_step = int(step_ids[start])
                run_length = int(end - start)
                if step_runs and step_runs[-1][0] == run_step:
                    prior_step, prior_length = step_runs[-1]
                    step_runs[-1] = (prior_step, prior_length + run_length)
                else:
                    step_runs.append((run_step, run_length))

        is_step4 = step_ids == 4
        starts = is_step4 & ~np.r_[previous_step4, is_step4[:-1]]
        segment_ids = segment_count + np.cumsum(starts)
        selected_ids = np.unique(segment_ids[is_step4])
        capacity = pd.to_numeric(chunk["Amphr"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        current = pd.to_numeric(chunk["Amps"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        voltage = pd.to_numeric(chunk["Volts"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        for segment_id in selected_ids:
            while len(segments) < segment_id:
                segments.append(
                    {
                        "segment_index": len(segments) + 1,
                        "capacity_parts": [],
                        "current_parts": [],
                        "voltage_parts": [],
                        "step4_rows": 0,
                    }
                )
            mask = is_step4 & (segment_ids == segment_id)
            segment = segments[int(segment_id) - 1]
            segment["capacity_parts"].append(capacity[mask])
            segment["current_parts"].append(current[mask])
            segment["voltage_parts"].append(voltage[mask])
            segment["step4_rows"] += int(mask.sum())

        segment_count += int(starts.sum())
        if is_step4.size:
            previous_step4 = bool(is_step4[-1])

    step4_run_positions = [
        index for index, (step_id, _) in enumerate(step_runs) if step_id == 4
    ]
    if len(step4_run_positions) != len(segments):
        raise ValueError(f"Step 4 boundary scan did not reconcile for {path}.")

    for index, segment in enumerate(segments):
        capacity = np.concatenate(segment.pop("capacity_parts"))
        current = np.concatenate(segment.pop("current_parts"))
        voltage = np.concatenate(segment.pop("voltage_parts"))
        finite_capacity = np.isfinite(capacity)
        finite_voltage = voltage[np.isfinite(voltage)]
        finite_current = current[np.isfinite(current)]
        adjacent = finite_capacity[1:] & finite_capacity[:-1]
        deltas = np.diff(capacity)[adjacent]
        current_signs = sorted(
            sign
            for sign, present in (
                ("negative", bool(np.any(finite_current < 0))),
                ("positive", bool(np.any(finite_current > 0))),
                ("zero", bool(np.any(finite_current == 0))),
            )
            if present
        )
        run_position = step4_run_positions[index]
        next_position = (
            step4_run_positions[index + 1]
            if index + 1 < len(step4_run_positions)
            else len(step_runs)
        )
        post_steps = [
            step_id
            for step_id, _ in step_runs[run_position + 1 : next_position]
            if step_id >= 0
        ]
        segment.update(
            {
                "capacity_finite_rows": int(finite_capacity.sum()),
                "capacity_span": (
                    float(np.max(capacity[finite_capacity]) - np.min(capacity[finite_capacity]))
                    if finite_capacity.any()
                    else 0.0
                ),
                "positive_capacity_deltas": int(np.count_nonzero(deltas > 0)),
                "negative_capacity_deltas": int(np.count_nonzero(deltas < 0)),
                "current_signs": current_signs,
                "finite_voltage_samples": int(finite_voltage.size),
                "voltage_min_v": (
                    float(finite_voltage.min()) if finite_voltage.size else None
                ),
                "voltage_max_v": (
                    float(finite_voltage.max()) if finite_voltage.size else None
                ),
                "post_steps": post_steps,
            }
        )
    return segments


def _is_complete_discharge(segment: dict[str, Any]) -> bool:
    return (
        segment["step4_rows"] >= _MIN_COMPLETE_STEP4_ROWS
        and segment["capacity_finite_rows"] >= _MIN_COMPLETE_STEP4_ROWS
        and segment["capacity_span"] >= _MIN_COMPLETE_CAPACITY_SPAN
        and segment["positive_capacity_deltas"] > 0
        and segment["negative_capacity_deltas"] == 0
        and segment["current_signs"] == ["negative"]
        and segment["finite_voltage_samples"] >= len(FEATURE_NAMES)
        and segment["voltage_min_v"] is not None
        and segment["voltage_min_v"] <= 2.6
        and segment["voltage_max_v"] is not None
        and segment["voltage_max_v"] <= 4.3
    )


def _is_group5_rpt_signature(segment: dict[str, Any]) -> bool:
    return tuple(segment["post_steps"][: len(_REQUIRED_GROUP5_STEPS)]) == (
        _REQUIRED_GROUP5_STEPS
    )


def _select_unique_discharge_segment(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [segment for segment in segments if _is_complete_discharge(segment)]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one complete Step 4 discharge; found {len(candidates)}."
        )
    return candidates[0]


def _select_earliest_complete_discharge(
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    candidates = [segment for segment in segments if _is_complete_discharge(segment)]
    if not candidates:
        raise ValueError("No complete Step 4 discharge segment was found.")
    return min(candidates, key=lambda segment: segment["segment_index"])


def _capacity_ratio(capacity: float, reference_capacity: float) -> float:
    if not math.isfinite(capacity) or not math.isfinite(reference_capacity):
        raise ValueError("Capacity values must be finite.")
    if reference_capacity <= 0:
        raise ValueError("Reference capacity must be positive.")
    return 100.0 * capacity / reference_capacity


def _find_observation(
    observations: list[dict[str, str]],
    *,
    cell: str,
    predicate: Callable[[dict[str, str]], bool],
) -> dict[str, str]:
    matches = [
        row
        for row in observations
        if row.get("physical_cell_id") == cell and predicate(row)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one Guide observation for {cell}; found {len(matches)}.")
    return matches[0]


def _local_bol_rows(
    observations: list[dict[str, str]], cell: str
) -> list[dict[str, str]]:
    rows = [
        row
        for row in observations
        if row.get("physical_cell_id") == cell
        and row.get("measurement_id", "").strip().casefold() == "bol"
    ]
    return sorted(rows, key=lambda row: int(row.get("procedure_file_id", "0")))


def _scan_observation_source(
    observation: dict[str, str],
    dataset_root: Path,
) -> tuple[Path, list[dict[str, Any]], str, int]:
    relative_paths = [
        path.strip()
        for path in observation.get("source_file", "").split(";")
        if path.strip()
    ]
    if len(relative_paths) != 1:
        raise ValueError(
            f"Expected one Guide-mapped source for {observation.get('physical_cell_id')}; "
            f"found {len(relative_paths)}."
        )
    source_path = resolve_source_file(dataset_root, relative_paths[0])
    stat_before = source_path.stat()
    segments = _scan_step4_file(source_path)
    stat_after = source_path.stat()
    if (
        stat_before.st_size != stat_after.st_size
        or stat_before.st_mtime_ns != stat_after.st_mtime_ns
    ):
        raise ValueError(f"Source file changed while being scanned: {source_path}")
    return source_path, segments, relative_paths[0], stat_after.st_size


def _baseline_for_cell(
    observations: list[dict[str, str]],
    cell: str,
    dataset_root: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    bol_rows = _local_bol_rows(observations, cell)
    if not bol_rows:
        raise ValueError(f"No Guide-mapped local BoL observation for {cell}.")
    first = bol_rows[0]
    if first.get("measurement_mapping_status") != "single_checkpoint":
        raise ValueError(f"First BoL mapping is not unique for {cell}.")
    _, baseline_segments, baseline_relative, _ = _scan_observation_source(
        first, dataset_root
    )
    baseline_segment = _select_earliest_complete_discharge(baseline_segments)
    baseline_capacity = float(baseline_segment["capacity_span"])
    expected = _EXPECTED_BASELINES[cell]
    if not math.isclose(baseline_capacity, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(
            f"Provisional BoL capacity changed for {cell}: "
            f"{baseline_capacity:.12f} vs audited {expected:.12f}."
        )
    baseline = {
        "capacity": baseline_capacity,
        "source_file": baseline_relative,
        "procedure": first.get("procedure", ""),
        "segment": baseline_segment,
    }

    alternate: dict[str, Any] | None = None
    if cell in _GROUP5_CELLS and len(bol_rows) > 1:
        later = bol_rows[1]
        if later.get("measurement_mapping_status") != "single_checkpoint":
            raise ValueError(f"Later BoL mapping is not unique for {cell}.")
        _, alt_segments, alt_relative, _ = _scan_observation_source(
            later, dataset_root
        )
        alternate_segment = _select_earliest_complete_discharge(alt_segments)
        alternate = {
            "capacity": float(alternate_segment["capacity_span"]),
            "source_file": alt_relative,
            "procedure": later.get("procedure", ""),
            "segment": alternate_segment,
        }

    audit = []
    for segment in baseline_segments:
        audit.append(
            _audit_row(
                cell,
                first.get("procedure", ""),
                baseline_relative,
                segment,
                "local_bol_reference" if segment is baseline_segment else "not_selected",
            )
        )
    if alternate is not None:
        for segment in alt_segments:
            audit.append(
                _audit_row(
                    cell,
                    alternate["procedure"],
                    alternate["source_file"],
                    segment,
                    "secondary_bol_sensitivity"
                    if segment is alternate["segment"]
                    else "not_selected",
                )
            )
    return baseline, alternate, audit


def _audit_row(
    cell: str,
    procedure: str,
    source_file: str,
    segment: dict[str, Any],
    selection: str,
) -> dict[str, Any]:
    return {
        "physical_cell_id": cell,
        "procedure": procedure,
        "source_file": source_file,
        "segment_index": segment["segment_index"],
        "step4_rows": segment["step4_rows"],
        "capacity_finite_rows": segment["capacity_finite_rows"],
        "capacity_span": segment["capacity_span"],
        "positive_capacity_deltas": segment["positive_capacity_deltas"],
        "negative_capacity_deltas": segment["negative_capacity_deltas"],
        "current_signs": ";".join(segment["current_signs"]),
        "voltage_min_v": segment["voltage_min_v"],
        "voltage_max_v": segment["voltage_max_v"],
        "group5_rpt_signature": _is_group5_rpt_signature(segment),
        "selection": selection,
    }


def _build_labels_and_features(
    observations: list[dict[str, str]],
    dataset_root: Path,
    cell_to_partition: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    supplemental_labels: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    expected_counts = {cell: 0 for cell in (*_GROUP5_CELLS, *_GROUP6_CELLS)}

    for cell in (*_GROUP5_CELLS, *_GROUP6_CELLS):
        baseline, alternate, baseline_audit = _baseline_for_cell(
            observations, cell, dataset_root
        )
        audit_rows.extend(baseline_audit)

        if cell in _GROUP5_CELLS:
            compound = _find_observation(
                observations,
                cell=cell,
                predicate=lambda row: row.get("procedure") == "TPG5.3",
            )
            match = re.search(r"RPT(\d+)-RPT(\d+)", compound.get("guide_stage", ""))
            if (
                match is None
                or match.groups() != ("1", "4")
                or compound.get("measurement_mapping_status")
                != "ambiguous_compound_stage"
            ):
                raise ValueError(f"Unexpected Guide compound-stage mapping for {cell}.")
            source_path, segments, relative_source, _ = _scan_observation_source(
                compound, dataset_root
            )
            rpt_segments = [
                segment
                for segment in segments
                if _is_group5_rpt_signature(segment)
                and _is_complete_discharge(segment)
            ]
            if len(rpt_segments) != 4:
                raise ValueError(
                    f"{cell} must have four complete Guide-signature RPT curves; "
                    f"found {len(rpt_segments)}."
                )
            ordered_segments = sorted(
                rpt_segments, key=lambda segment: segment["segment_index"]
            )
            rpt_by_segment = {
                segment["segment_index"]: rpt_order
                for rpt_order, segment in enumerate(ordered_segments, start=1)
            }
            audit_rows.extend(
                _audit_row(
                    cell,
                    compound.get("procedure", ""),
                    relative_source,
                    segment,
                    (
                        f"RPT{rpt_by_segment[segment['segment_index']]}"
                        if segment["segment_index"] in rpt_by_segment
                        else "not_selected"
                    ),
                )
                for segment in segments
            )
            observations_for_cell = []
            for rpt_order, segment in enumerate(ordered_segments, start=1):
                observation = dict(compound)
                observation.update(
                    {
                        "measurement_id": f"RPT{rpt_order}",
                        "rpt_order": str(rpt_order),
                        "cycle_age_status": "cadence_not_established",
                        "nominal_cycle_age": "",
                    }
                )
                observations_for_cell.append((observation, segment))
        else:
            observations_for_cell = []
            guide_rows = [
                row
                for row in observations
                if row.get("physical_cell_id") == cell
                and row.get("dataset_part") == "2"
                and row.get("measurement_id", "").startswith("RPT")
            ]
            guide_rows.sort(key=lambda row: int(row.get("rpt_order", "0")))
            rpt_orders = [int(row["rpt_order"]) for row in guide_rows]
            if len(guide_rows) != 13 or len(set(rpt_orders)) != 13:
                raise ValueError(
                    f"Expected 13 uniquely mapped Group 6 RPTs for {cell}; "
                    f"found orders {rpt_orders}."
                )
            for observation in guide_rows:
                if (
                    observation.get("measurement_mapping_status")
                    != "single_checkpoint"
                    or observation.get("cycle_age_status") != "calendar_aging_only"
                    or observation.get("measurement_id")
                    != f"RPT{observation.get('rpt_order')}"
                ):
                    raise ValueError(f"Unsupported Group 6 Guide mapping for {cell}.")
                _, segments, relative_source, _ = _scan_observation_source(
                    observation, dataset_root
                )
                segment = _select_unique_discharge_segment(segments)
                audit_rows.extend(
                    _audit_row(
                        cell,
                        observation.get("procedure", ""),
                        relative_source,
                        candidate,
                        "selected_discharge"
                        if candidate is segment
                        else "not_selected",
                    )
                    for candidate in segments
                )
                observations_for_cell.append((dict(observation), segment))

        for observation, segment in observations_for_cell:
            if observation.get("cycle_age_status") not in {
                "cadence_not_established",
                "calendar_aging_only",
            }:
                raise ValueError("Supplemental labels require non-nominal age provenance.")
            if observation.get("nominal_cycle_age", "").strip():
                raise ValueError("Non-nominal Part 2 rows must not have a cycle-age value.")
            source_path = resolve_source_file(
                dataset_root,
                observation.get("source_file", ""),
            )
            voltage_features, voltage_metadata = extract_step4_voltage(
                source_path, int(segment["segment_index"])
            )
            if not _is_complete_discharge(segment):
                raise ValueError("Selected curve no longer passes Step 4 discharge QC.")

            capacity = float(segment["capacity_span"])
            soh = _capacity_ratio(capacity, float(baseline["capacity"]))
            status = "calculated_from_measured_capacity"
            notes = _LABEL_NOTES
            if soh < 0 or soh > 120:
                status = "physically_implausible_review"
                notes += " SOH is outside 0-120%; retained without clipping."
            elif soh < 50:
                status = "review_capacity_span_outlier"
                notes += " Below the 50% review floor; excluded from training."
            label_segment = dict(segment)
            label_segment["_source_file"] = observation.get("source_file", "")
            label = _empty_label(
                observation,
                status=status,
                notes=notes,
                baseline=baseline,
                capacity=capacity,
                segment=label_segment,
                alternate_capacity=(
                    float(alternate["capacity"]) if alternate is not None else None
                ),
                soh=soh,
                sensitivity_soh=(
                    _capacity_ratio(capacity, float(alternate["capacity"]))
                    if alternate is not None
                    else None
                ),
            )
            label["label_method"] = _LABEL_METHOD
            supplemental_labels.append(label)
            expected_counts[label["physical_cell_id"]] += 1

            partition = cell_to_partition[label["physical_cell_id"]]
            manifest_rows.append(
                {
                    "dataset_part": "2",
                    "group_id": str(observation.get("group_id", "")),
                    "physical_cell_id": label["physical_cell_id"],
                    "measurement_id": label["measurement_id"],
                    "rpt_order": str(label["rpt_order"]),
                    "soh_percent": format(soh, ".17g"),
                    "source_file": observation.get("source_file", ""),
                    "step4_segment_index": str(segment["segment_index"]),
                    "step4_row_count": str(segment["step4_rows"]),
                    "partition": partition,
                    "step4_rows": str(voltage_metadata["step4_rows"]),
                    "finite_voltage_samples": str(
                        voltage_metadata["finite_voltage_samples"]
                    ),
                    "step4_segment_count": str(len(segments)),
                    "voltage_min_v": format(
                        voltage_metadata["voltage_min_v"], ".17g"
                    ),
                    "voltage_max_v": format(
                        voltage_metadata["voltage_max_v"], ".17g"
                    ),
                }
            )
            feature_rows.append(
                {
                    "dataset_part": "2",
                    "physical_cell_id": label["physical_cell_id"],
                    "measurement_id": label["measurement_id"],
                    **{
                        name: format(float(value), ".17g")
                        for name, value in zip(FEATURE_NAMES, voltage_features)
                    },
                }
            )

    expected = {**{cell: 4 for cell in _GROUP5_CELLS}, "G6-Cell1": 13}
    if expected_counts != expected:
        raise ValueError(f"Supplemental label counts differ from the audited map: {expected_counts}")
    return supplemental_labels, manifest_rows, feature_rows, audit_rows


def _write_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        return list(reader.fieldnames or []), list(reader)


def run(
    scan_dir: Path,
    dataset_root: Path,
    base_feature_run: Path,
    split_path: Path,
    part1_labels: Path,
    part2_labels: Path,
    part3_labels: Path,
    output: Path,
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve()
    dataset_root = dataset_root.resolve()
    base_feature_run = base_feature_run.resolve()
    split_path = split_path.resolve()
    label_paths = {
        "1": part1_labels.resolve(),
        "2": part2_labels.resolve(),
        "3": part3_labels.resolve(),
    }
    output = output.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output directory is not empty: {output}")
    for protected in (dataset_root, label_paths["1"].parent):
        try:
            output.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(f"Output directory must not be inside {protected}.")

    scan_summary_path = scan_dir / "summary.json"
    manifest_path = scan_dir / "guide_observation_manifest_step4.csv"
    scan_summary = json.loads(scan_summary_path.read_text(encoding="utf-8"))
    if scan_summary.get("dataset_roots", {}).get("2") != str(dataset_root):
        raise ValueError("Part 2 dataset root differs from the Step 4 scan.")
    guide_paths = list(dataset_root.glob("Guide*.xlsx"))
    if len(guide_paths) != 1:
        raise ValueError(f"Expected one Guide workbook in {dataset_root}.")
    if _sha256(guide_paths[0]) != scan_summary.get("guide_sha256", {}).get("2"):
        raise ValueError("Part 2 Guide changed since the Step 4 scan.")
    integrity = scan_summary.get("post_scan_integrity_verification", {})
    if (
        integrity.get("source_paths_missing") != 0
        or integrity.get("source_size_mismatches") != 0
        or integrity.get("source_paths_modified_after_scan") != 0
    ):
        raise ValueError("The Step 4 scan did not pass source integrity checks.")

    cell_to_partition = load_split(split_path)
    if Counter(cell_to_partition.values()) != Counter(EXPECTED_PARTITION_SIZES):
        raise ValueError("Locked split partition sizes differ from the expected 10/3/3/11.")
    base_metrics_path = base_feature_run / "metrics.json"
    base_metrics = json.loads(base_metrics_path.read_text(encoding="utf-8"))
    expected_hashes = base_metrics.get("input_sha256", {})
    source_hashes = {
        "split": split_path,
        "part1_labels": label_paths["1"],
        "part2_labels": label_paths["2"],
        "part3_labels": label_paths["3"],
    }
    for name, path in source_hashes.items():
        if expected_hashes.get(name) != sha256(path):
            raise ValueError(f"{name} differs from the base feature-run inputs.")

    observations = _read_csv(manifest_path)
    labels, new_manifest, new_features, audit_rows = _build_labels_and_features(
        observations, dataset_root, cell_to_partition
    )
    if len(labels) != 25 or len(new_manifest) != 25 or len(new_features) != 25:
        raise ValueError("Expected 25 validated supplemental Part 2 observations.")

    manifest_fields, old_manifest = _read_rows(base_feature_run / "sample_manifest.csv")
    feature_fields, old_features = _read_rows(base_feature_run / "features.csv")
    if feature_fields != ["dataset_part", "physical_cell_id", "measurement_id", *FEATURE_NAMES]:
        raise ValueError("Base feature table has an unexpected schema.")
    keys = ("dataset_part", "physical_cell_id", "measurement_id")
    old_keys = [tuple(row[key] for key in keys) for row in old_manifest]
    new_keys = [tuple(row[key] for key in keys) for row in new_manifest]
    if len(set(old_keys)) != len(old_keys) or set(old_keys) & set(new_keys):
        raise ValueError("New feature sample identities duplicate existing observations.")

    combined_label_path = output / "labels_soh_part2_combined.csv"
    old_label_rows = _read_csv(label_paths["2"])
    combined_labels = old_label_rows + labels
    if any(list(row) != _LABEL_FIELDS for row in combined_labels):
        raise ValueError("Part 2 candidate-label schema does not match the label generator.")

    combined_manifest = old_manifest + new_manifest
    combined_features = old_features + new_features
    partition_counts = Counter(row["partition"] for row in combined_manifest)
    cells_by_partition = {
        partition: len(
            {
                row["physical_cell_id"]
                for row in combined_manifest
                if row["partition"] == partition
            }
        )
        for partition in EXPECTED_PARTITION_SIZES
    }
    if dict(partition_counts) != {
        "train": 101,
        "validation": 39,
        "internal_test": 42,
        "external_test": 74,
    } or cells_by_partition != EXPECTED_PARTITION_SIZES:
        raise ValueError(
            f"Extended feature counts differ from expected: "
            f"samples={dict(partition_counts)}, cells={cells_by_partition}"
        )

    output.mkdir(parents=True, exist_ok=True)
    feature_output = output / "feature_run"
    feature_output.mkdir()
    _write_rows(output / "labels_soh_part2_supplemental_g5_g6.csv", _LABEL_FIELDS, labels)
    _write_rows(combined_label_path, _LABEL_FIELDS, combined_labels)
    _write_rows(
        output / "step4_segment_audit.csv",
        _AUDIT_FIELDS,
        audit_rows,
    )

    _write_rows(feature_output / "sample_manifest.csv", manifest_fields, combined_manifest)
    _write_rows(feature_output / "features.csv", feature_fields, combined_features)
    excluded_path = base_feature_run / "excluded_samples.csv"
    if excluded_path.is_file():
        (feature_output / "excluded_samples.csv").write_bytes(excluded_path.read_bytes())

    metrics = dict(base_metrics)
    metrics.update(
        {
            "input_sha256": {
                **expected_hashes,
                "part2_labels": sha256(combined_label_path),
            },
            "samples_by_partition": dict(partition_counts),
            "labeled_cells_by_partition": cells_by_partition,
            "split_cells_without_labels": [],
            "n_feature_extraction_exclusions": 0,
            "supplemental_extension": {
                "labels": len(labels),
                "cells": sorted((*_GROUP5_CELLS, *_GROUP6_CELLS)),
                "age_policy": (
                    "Group 5: cadence_not_established; Group 6: calendar_aging_only; "
                    "no nominal cycle-age values assigned."
                ),
            },
        }
    )
    (feature_output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    secondary_bol_by_cell = {
        row["physical_cell_id"]: row
        for row in audit_rows
        if row["selection"] == "secondary_bol_sensitivity"
    }
    label_by_cell = {
        cell: next(row for row in labels if row["physical_cell_id"] == cell)
        for cell in (*_GROUP5_CELLS, *_GROUP6_CELLS)
    }
    bol_rows = []
    for cell in (*_GROUP5_CELLS, *_GROUP6_CELLS):
        row = label_by_cell[cell]
        secondary = secondary_bol_by_cell.get(cell, {})
        bol_rows.append(
            {
                "physical_cell_id": row["physical_cell_id"],
                "source_file": row["baseline_source_file"],
                "procedure": row["baseline_procedure"],
                "segment_index": row["baseline_segment_index"],
                "reference_capacity_raw": row["reference_capacity_raw"],
                "secondary_bol_source_file": secondary.get("source_file", ""),
                "secondary_bol_segment_index": secondary.get("segment_index", ""),
                "secondary_reference_capacity_raw": secondary.get(
                    "capacity_span", ""
                ),
            }
        )
    _write_rows(
        output / "bol_denominator_audit.csv",
        list(bol_rows[0]) if bol_rows else [],
        bol_rows,
    )

    summary = {
        "operation": "Supplemental Part 2 G5/G6 SOH labels and CNN1D feature extension",
        "split_version": SPLIT_VERSION,
        "provisional_policy": (
            "SOH = 100 * validated Step 4 discharge capacity span / earliest complete "
            "Step 4 discharge in the same cell's first Guide-mapped local BoL procedure."
        ),
        "policy_status": "user_selected_provisional_protocol_unverified",
        "capacity_unit": "native Amphr; absolute unit unverified",
        "age_provenance": {
            "Group 5": "cadence_not_established; verified RPT order only",
            "Group 6": "calendar_aging_only; verified RPT order only",
        },
        "labels_by_cell": dict(Counter(row["physical_cell_id"] for row in labels)),
        "validation_status_counts": dict(
            Counter(row["validation_status"] for row in labels)
        ),
        "samples_by_partition": dict(partition_counts),
        "labeled_cells_by_partition": cells_by_partition,
        "group5_rpt_mapping": (
            "TPG5.3 Step 4 segments assigned RPT1-RPT4 only when their following "
            "step sequence begins 5,6,...,36, and the segment passes discharge QC."
        ),
        "group6_rpt_mapping": (
            "Each Guide-mapped RPT file must have exactly one complete negative-current "
            "Step 4 discharge; partial segments are excluded."
        ),
        "source_files_modified": False,
        "source_provenance": {
            "step4_scan_directory": str(scan_dir),
            "guide_sha256": scan_summary["guide_sha256"]["2"],
            "observation_manifest_sha256": _sha256(manifest_path),
            "split_sha256": sha256(split_path),
            "original_part2_labels_sha256": sha256(label_paths["2"]),
            "combined_part2_labels_sha256": sha256(combined_label_path),
        },
        "limitations": [
            "Candidate labels only; the user-approved local-BoL policy has not been verified against the source protocol.",
            "Amphr units are not verified; same-cell ratios do not resolve step, sign, or integration-window semantics.",
            "Group 5 and Group 6 do not receive nominal cycle-age values.",
            "Part 3 remains an external/domain-shift evaluation.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate provisional Part 2 G5/G6 SOH candidates and extend CNN1D features."
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--base-feature-run", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--part1-labels", type=Path, required=True)
    parser.add_argument("--part2-labels", type=Path, required=True)
    parser.add_argument("--part3-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        args.scan_dir,
        args.dataset_root,
        args.base_feature_run,
        args.split,
        args.part1_labels,
        args.part2_labels,
        args.part3_labels,
        args.output,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
