from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bafuse.labeling.audit import _identify_file, read_guide_entries


_BOL_STAGE = re.compile(r"\s*BoL(?:\s+RPT)?\s*", re.IGNORECASE)
_RPT_STAGE = re.compile(r"\s*RPT\s*(\d+)\s*", re.IGNORECASE)
_MEASUREMENT_MARKER = re.compile(r"BoL|RPT", re.IGNORECASE)
_FIELDNAMES = [
    "dataset_part",
    "group_id",
    "cell_id",
    "physical_cell_id",
    "procedure",
    "procedure_file_id",
    "guide_stage",
    "measurement_id",
    "rpt_order",
    "measurement_mapping_status",
    "source_file",
    "source_file_id",
    "source_mapping_rule",
    "source_file_count",
    "source_mapping_status",
    "source_file_size_bytes",
    "nominal_cycle_age",
    "cycle_age_status",
    "chronology_basis",
    "step4_boundary_status",
]


def _measurement_identity(stage: str) -> tuple[str, int | None, str]:
    if _BOL_STAGE.fullmatch(stage):
        return "BoL", 0, "single_checkpoint"
    match = _RPT_STAGE.fullmatch(stage)
    if match:
        order = int(match.group(1))
        if order > 0:
            return f"RPT{order}", order, "single_checkpoint"
    return "", None, "ambiguous_compound_stage"


def _cycle_age(part: int, group_id: str, rpt_order: int | None) -> tuple[str, str]:
    if rpt_order is None:
        return "", "unresolved_checkpoint"
    group = int(group_id)
    has_documented_cadence = (
        (part == 1 and group in {1, 2, 3, 4})
        or (part == 2 and group in {1, 2, 3, 4})
        or (part == 3 and group in {7, 8, 9, 10})
    )
    if has_documented_cadence:
        return str(48 * rpt_order), "nominal_48_cycles_per_RPT"
    if part == 2 and group == 6:
        return "", "calendar_aging_only"
    if part == 2 and group == 5:
        return "", "cadence_not_established"
    return "", "cycle_age_not_documented"


def build_observation_manifest(
    dataset_roots: dict[int, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not dataset_roots:
        raise ValueError("At least one dataset part root is required.")
    if set(dataset_roots).difference({1, 2, 3}):
        raise ValueError("Dataset part keys must be 1, 2, or 3.")

    rows: list[dict[str, Any]] = []
    guide_hashes: dict[str, str] = {}
    source_roots: dict[str, str] = {}
    for part, root_value in sorted(dataset_roots.items()):
        root = root_value.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Part {part} dataset directory does not exist: {root}")
        guide_paths = sorted(root.glob("Guide*.xlsx"))
        if len(guide_paths) != 1:
            raise ValueError(
                f"Expected exactly one Guide XLSX in Part {part} root, found {len(guide_paths)}: {root}"
            )
        guide_path = guide_paths[0]
        guide_entries = read_guide_entries(guide_path)
        if not guide_entries:
            raise ValueError(f"No cell-mapped Guide entries were parsed from {guide_path}")
        guide_hashes[str(part)] = hashlib.sha256(guide_path.read_bytes()).hexdigest()
        source_roots[str(part)] = str(root)

        source_files: defaultdict[tuple[str, str, int], list[Path]] = defaultdict(list)
        for path in root.rglob("*.csv"):
            metadata = _identify_file(path, root)
            if metadata["modality"] != "cycling_and_RPT" or metadata["file_id"] is None:
                continue
            key = (
                metadata["group_id"],
                metadata["cell_id"],
                metadata["file_id"],
            )
            source_files[key].append(path)

        for entry in guide_entries:
            if not _MEASUREMENT_MARKER.search(entry.stage):
                continue
            measurement_id, rpt_order, mapping_status = _measurement_identity(entry.stage)
            source_file_id = entry.file_id
            source_mapping_rule = "guide_procedure_file_id"
            if (
                part == 3
                and _BOL_STAGE.fullmatch(entry.stage)
                and re.fullmatch(r"TPG\d+", entry.procedure)
            ):
                source_file_id = 0
                source_mapping_rule = "part3_unsuffixed_bol_to_zero_suffix"
            matches = source_files.get(
                (entry.group_id, entry.cell_id, source_file_id), []
            )
            if len(matches) == 1:
                source_file = matches[0]
                source_path = source_file.relative_to(root).as_posix()
                source_file_size = source_file.stat().st_size
                source_mapping_status = "unique_source_file"
            else:
                source_path = ";".join(
                    path.relative_to(root).as_posix() for path in sorted(matches)
                )
                source_file_size = ""
                source_mapping_status = (
                    "source_file_not_found" if not matches else "multiple_source_files"
                )

            nominal_age, age_status = _cycle_age(part, entry.group_id, rpt_order)
            rows.append(
                {
                    "dataset_part": part,
                    "group_id": entry.group_id,
                    "cell_id": entry.cell_id,
                    "physical_cell_id": f"G{entry.group_id}-Cell{entry.cell_id}",
                    "procedure": entry.procedure,
                    "procedure_file_id": entry.file_id,
                    "guide_stage": entry.stage,
                    "measurement_id": measurement_id,
                    "rpt_order": "" if rpt_order is None else rpt_order,
                    "measurement_mapping_status": mapping_status,
                    "source_file": source_path,
                    "source_file_id": source_file_id,
                    "source_mapping_rule": source_mapping_rule,
                    "source_file_count": len(matches),
                    "source_mapping_status": source_mapping_status,
                    "source_file_size_bytes": source_file_size,
                    "nominal_cycle_age": nominal_age,
                    "cycle_age_status": age_status,
                    "chronology_basis": (
                        "guide_rpt_ordinal"
                        if rpt_order is not None
                        else "unresolved_compound_guide_stage"
                    ),
                    "step4_boundary_status": (
                        "pending_curve_scan"
                        if mapping_status == "single_checkpoint"
                        else "blocked_compound_guide_stage"
                    ),
                }
            )

    rows.sort(
        key=lambda row: (
            row["dataset_part"],
            int(row["group_id"]),
            int(row["cell_id"]),
            row["rpt_order"] if row["rpt_order"] != "" else 10**9,
            row["procedure"],
        )
    )
    summary = {
        "manifest": "Guide-level SOH observation candidates",
        "purpose": (
            "Crosswalk Guide BoL/RPT entries to source CSVs and record ordinal/"
            "nominal age metadata; curve boundaries and labels are not validated here."
        ),
        "dataset_roots": source_roots,
        "guide_sha256": guide_hashes,
        "candidate_rows": len(rows),
        "unique_source_file_rows": sum(
            row["source_mapping_status"] == "unique_source_file" for row in rows
        ),
        "missing_source_file_rows": sum(
            row["source_mapping_status"] == "source_file_not_found" for row in rows
        ),
        "multiple_source_file_rows": sum(
            row["source_mapping_status"] == "multiple_source_files" for row in rows
        ),
        "measurement_mapping_status_counts": dict(
            Counter(row["measurement_mapping_status"] for row in rows)
        ),
        "cycle_age_status_counts": dict(Counter(row["cycle_age_status"] for row in rows)),
        "limitations": [
            "This is a Guide-level candidate manifest, not a validated SOH-label table.",
            "Step 4 row/curve boundaries have not been scanned in this run.",
            "Compound Guide stages are retained but are not assigned to a single RPT.",
            "Unmapped Guide sequences without a cell identifier are absent from read_guide_entries and remain quarantined in the dataset review.",
            "48 cycles per RPT is recorded only as a nominal proxy for documented groups.",
        ],
    }
    return rows, summary


def write_observation_manifest(
    dataset_roots: dict[int, Path], output_dir: Path
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Experiment output directory is not empty: {output_dir}")
    rows, summary = build_observation_manifest(dataset_roots)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "guide_observation_manifest.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=_FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Guide-level crosswalk for SOH RPT observation candidates."
    )
    for part in (1, 2, 3):
        parser.add_argument(
            f"--part-{part}-root",
            type=Path,
            required=True,
            help=f"Path to the Part {part} dataset directory.",
        )
    parser.add_argument(
        "--output", type=Path, required=True, help="New experiment output directory."
    )
    args = parser.parse_args()
    roots = {
        1: args.part_1_root,
        2: args.part_2_root,
        3: args.part_3_root,
    }
    summary = write_observation_manifest(roots, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
