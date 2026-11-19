from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from .soh_label_generation import (
    _LABEL_FIELDS,
    _MINIMUM_REVIEW_SOH_PERCENT,
    _empty_label,
    _is_bol,
    _read_csv,
    _resolve_curve,
    _sha256,
    _source_paths,
    build_part1_soh_labels,
)

_LABELED_GROUPS = {"1", "2", "3", "4"}
_LABEL_METHOD = (
    "Part 2 Step 4 discharge capacity span / earliest complete Step 4 segment "
    "of the same cell's first Guide-mapped Part 1 BoL procedure * 100"
)
_NOT_LABELED_NOTES = (
    "Part 2 Groups 5-6 are intentionally not labeled: Group 5 RPTs are bundled "
    "in compound Guide stages, Group 6 mostly has multiple Step 4 segments and is "
    "calendar-aging only, and no BoL denominator policy was approved for them."
)


def _cell_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    def number(value: Any) -> int:
        text = str(value)
        return int(text) if text.isdigit() else 10**9

    return number(row["group_id"]), number(row["cell_id"]), number(row["rpt_order"])


def build_part2_soh_labels(
    observations: list[dict[str, str]],
    source_scan_rows: list[dict[str, str]],
    part1_baselines: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    part2 = [
        row
        for row in observations
        if row.get("dataset_part") == "2" and row.get("physical_cell_id", "").strip()
    ]
    if not part2:
        raise ValueError("The Step 4 manifest has no identified Part 2 rows.")

    scan_by_path: dict[str, dict[str, str]] = {}
    for row in source_scan_rows:
        if "2" not in {p.strip() for p in row.get("dataset_parts", "").split(";")}:
            continue
        path = row.get("relative_path", "")
        if path in scan_by_path:
            raise ValueError(f"Duplicate Part 2 source scan row: {path}")
        scan_by_path[path] = row

    rpt_counts = Counter(
        (row["physical_cell_id"], row.get("measurement_id", ""))
        for row in part2
        if not _is_bol(row)
    )
    labels: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    numeric_by_cell: Counter[str] = Counter()
    for observation in part2:
        cell = observation["physical_cell_id"]
        if observation.get("group_id") not in _LABELED_GROUPS:
            label = _empty_label(
                observation,
                status="withheld_group5_6_not_labeled",
                notes=_NOT_LABELED_NOTES,
            )
        elif cell not in part1_baselines:
            label = _empty_label(
                observation,
                status="unavailable_no_part1_same_cell_bol_reference",
                notes="No Part 1 same-cell BoL denominator; no fallback is used.",
            )
        else:
            baseline = part1_baselines[cell]
            if _is_bol(observation):
                label = _empty_label(
                    observation,
                    status="unavailable_part2_bol_not_expected",
                    notes="Groups 1-4 have no Part 2 BoL row; unexpected BoL row not used.",
                    baseline=baseline,
                )
            elif rpt_counts[(cell, observation.get("measurement_id", ""))] != 1:
                label = _empty_label(
                    observation,
                    status="unavailable_duplicate_rpt_mapping",
                    notes="Duplicate RPT identity within the cell; withheld.",
                    baseline=baseline,
                )
            elif observation.get("measurement_mapping_status") != "single_checkpoint":
                label = _empty_label(
                    observation,
                    status="unavailable_ambiguous_rpt_mapping",
                    notes="Compound Guide stage; withheld.",
                    baseline=baseline,
                )
            else:
                segment, error = _resolve_curve(observation, scan_by_path, baseline=False)
                if segment is None:
                    label = _empty_label(
                        observation,
                        status=error,
                        notes="Step 4 curve did not pass the single-segment QC gates.",
                        baseline=baseline,
                    )
                else:
                    capacity = float(segment["capacity_span"])
                    soh = 100.0 * capacity / baseline["capacity"]
                    status = "calculated_from_measured_capacity"
                    notes = (
                        "Same-cell capacity ratio against the Part 1 BoL denominator "
                        "(provisional policy); native Amphr, absolute unit unverified."
                    )
                    if soh < 0 or soh > 120:
                        status = "physically_implausible_review"
                        notes += " SOH is outside 0-120%; retained without clipping."
                    elif soh < _MINIMUM_REVIEW_SOH_PERCENT:
                        status = "review_capacity_span_outlier"
                        notes += " Below the 50% review floor; excluded from training."
                    else:
                        numeric_by_cell[cell] += 1
                    label = _empty_label(
                        observation,
                        status=status,
                        notes=notes,
                        baseline=baseline,
                        capacity=capacity,
                        segment=segment,
                        soh=soh,
                    )
        label["label_method"] = _LABEL_METHOD
        labels.append(label)
        status_counts[label["validation_status"]] += 1

    labels.sort(key=_cell_sort_key)
    summary = {
        "operation": "Part 2 Groups 1-4 SOH candidate labels with Part 1 same-cell BoL",
        "baseline_policy": (
            "Part 1 provisional policy (run 007): earliest complete Step 4 "
            "segment of the first Guide-mapped Part 1 BoL procedure"
        ),
        "baseline_policy_status": "user_selected_provisional_protocol_unverified",
        "labeled_groups": sorted(_LABELED_GROUPS),
        "groups_5_6_labeled": False,
        "label_count": len(labels),
        "numeric_rpt_label_count": sum(numeric_by_cell.values()),
        "numeric_rpt_labels_by_cell": dict(sorted(numeric_by_cell.items())),
        "labeled_cell_count": len(numeric_by_cell),
        "validation_status_counts": dict(sorted(status_counts.items())),
        "official_soh_labels_found": False,
        "source_files_modified": False,
        "limitations": [
            "Candidate labels only: the denominator is a user-selected provisional policy and Amphr is native/unverified.",
            "Part 2 Groups 1-4 RPTs continue the same physical cells' Part 1 chain; keep each cell in one partition across both parts.",
            "Groups 5-6 receive no labels; their cells have no SOH observations in this table.",
            "Nominal cycle age is a documented-cadence proxy, not reconstructed exposure.",
        ],
    }
    return labels, summary


def run_part2_soh_label_generation(
    scan_dir: Path,
    part1_labels_path: Path,
    dataset_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    scan_dir = scan_dir.resolve()
    part1_labels_path = part1_labels_path.resolve()
    dataset_root = dataset_root.resolve()
    output_dir = output_dir.resolve()
    for protected in (dataset_root, scan_dir, part1_labels_path.parent):
        try:
            output_dir.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(f"Output directory must not be inside {protected}.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")

    scan_summary = json.loads((scan_dir / "summary.json").read_text(encoding="utf-8"))
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
        raise ValueError("The Step 4 scan did not pass its source integrity checks.")

    observation_path = scan_dir / "guide_observation_manifest_step4.csv"
    scan_path = scan_dir / "source_file_step4_scan.csv"
    observations = _read_csv(observation_path)
    scan_rows = _read_csv(scan_path)

    part1_rows, _ = build_part1_soh_labels(observations, scan_rows)
    baselines: dict[str, dict[str, Any]] = {}
    for row in part1_rows:
        if row["validation_status"] != "baseline_reference":
            continue
        cell = row["physical_cell_id"]
        if cell in baselines:
            raise ValueError(f"Duplicate Part 1 baseline for {cell}.")
        baselines[cell] = {
            "capacity": float(row["capacity_raw"]),
            "source_file": row["source_file"],
            "procedure": row["procedure"],
            "segment": {"segment_index": row["step4_segment_index"]},
        }
    saved = {
        row["physical_cell_id"]: float(row["capacity_raw"])
        for row in _read_csv(part1_labels_path)
        if row["validation_status"] == "baseline_reference"
    }
    if saved.keys() != baselines.keys() or any(
        not math.isclose(saved[cell], baselines[cell]["capacity"], rel_tol=1e-12)
        for cell in saved
    ):
        raise ValueError("Recomputed Part 1 baselines differ from the supplied Part 1 label table.")

    part2_scan = {
        row["relative_path"]: row
        for row in scan_rows
        if "2" in {p.strip() for p in row.get("dataset_parts", "").split(";")}
    }
    for observation in observations:
        if (
            observation.get("dataset_part") != "2"
            or observation.get("group_id") not in _LABELED_GROUPS
            or observation.get("measurement_mapping_status") != "single_checkpoint"
        ):
            continue
        for relative_path in _source_paths(observation):
            scan_row = part2_scan.get(relative_path)
            if scan_row is None:
                raise ValueError(f"Mapped Part 2 source was not scanned: {relative_path}")
            source_path = (dataset_root / relative_path).resolve()
            source_path.relative_to(dataset_root)
            stat = source_path.stat()
            if (
                str(stat.st_size) != scan_row.get("source_size_bytes")
                or str(stat.st_mtime_ns) != scan_row.get("source_modified_ns")
            ):
                raise ValueError(f"Source file changed since the Step 4 scan: {source_path}")

    labels, summary = build_part2_soh_labels(observations, scan_rows, baselines)
    summary.update(
        {
            "step4_scan_directory": str(scan_dir),
            "dataset_root": str(dataset_root),
            "guide_sha256": scan_summary["guide_sha256"]["2"],
            "observation_manifest_sha256": _sha256(observation_path),
            "source_scan_sha256": _sha256(scan_path),
            "part1_labels_path": str(part1_labels_path),
            "part1_labels_sha256": _sha256(part1_labels_path),
            "source_integrity_rechecked": True,
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "labels_soh_part2.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=_LABEL_FIELDS)
        writer.writeheader()
        writer.writerows(labels)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Part 2 Groups 1-4 SOH candidates using Part 1 same-cell BoL."
    )
    parser.add_argument("--scan-dir", type=Path, required=True)
    parser.add_argument("--part1-labels", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = run_part2_soh_label_generation(
        args.scan_dir, args.part1_labels, args.dataset_root, args.output
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
