from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from bafuse.labeling.independent_soh_audit import (
    BENCHMARK_DIR,
    PROJECT_ROOT,
    _tail_for_part,
    parse_block_reference,
)


AUDIT_DIR = BENCHMARK_DIR / "independent_raw_soh_audit_v1_20261113"
RECONSTRUCTION_DIR = BENCHMARK_DIR / "raw_soh_reconstruction_v1_20261113"
OUTPUT_DIR = BENCHMARK_DIR / "measurement_semantics_resolution_v1_20261113"
PART_READMES = (
    PROJECT_ROOT / "data" / "raw" / "oxford" / "part1" / "Readme.txt",
    PROJECT_ROOT / "data" / "raw" / "oxford" / "part2" / "Readme.txt",
    PROJECT_ROOT / "data" / "raw" / "oxford" / "part3" / "Readme.txt",
)
SOURCE_COLUMNS = ("testtime", "steptime", "amps", "amphr")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty report table: {path}")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_source(source: str, part: int) -> Path:
    normalized = source.replace("\\", "/").strip("/")
    if normalized.casefold().startswith("data/raw/oxford/"):
        relative = normalized
    else:
        relative = f"data/raw/oxford/part{part}/{normalized}"
    path = (PROJECT_ROOT / Path(relative)).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()) or not path.is_file():
        raise FileNotFoundError(f"Mapped raw source is unavailable or outside project: {path}")
    return path


def _read_mapped_block(
    reference: str,
    part: int,
    block_audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> dict[str, Any]:
    parsed = parse_block_reference(reference)
    path = _resolve_source(parsed.source, part)
    source_key = _tail_for_part(str(path), part)
    key = (source_key, part, parsed.step, parsed.start_line, parsed.end_line)
    audited = block_audit_by_key.get(key)
    if audited is None:
        raise ValueError(f"Mapped block is absent from the retained audit: {reference}")

    before = path.stat()
    values: dict[str, list[float]] = {name: [] for name in SOURCE_COLUMNS}
    row_digest = hashlib.sha256()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if not header:
            raise ValueError(f"Missing source header: {path}")
        folded = [value.strip().casefold() for value in header]
        columns: dict[str, int] = {}
        for name in (*SOURCE_COLUMNS, "step"):
            matches = [index for index, value in enumerate(folded) if value == name]
            if len(matches) != 1:
                raise ValueError(f"Expected one {name} field in {path}, got {len(matches)}")
            columns[name] = matches[0]
        for row in reader:
            line = reader.line_num
            if line < parsed.start_line:
                continue
            if line > parsed.end_line:
                break
            try:
                step = int(float(row[columns["step"]]))
                if step != parsed.step:
                    raise ValueError(f"Unexpected step at {path}:{line}")
                for name in SOURCE_COLUMNS:
                    value = float(row[columns[name]])
                    if not math.isfinite(value):
                        raise ValueError(f"Nonfinite {name} at {path}:{line}")
                    values[name].append(value)
            except (IndexError, ValueError) as error:
                if isinstance(error, ValueError) and str(error).startswith(
                    ("Unexpected step", "Nonfinite")
                ):
                    raise
                raise ValueError(f"Invalid source value at {path}:{line}") from error
            row_digest.update(("\x1f".join(row) + "\n").encode("utf-8", errors="replace"))
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        raise ValueError(f"Raw source changed while reading a targeted block: {path}")
    row_count = parsed.end_line - parsed.start_line + 1
    if len(values["amps"]) != row_count:
        raise ValueError(f"Mapped row count changed for {reference}")
    observed_hash = row_digest.hexdigest()
    if observed_hash != audited["block_sha256_independent"]:
        raise ValueError(f"Raw block hash differs from retained audit for {reference}")
    return {
        **values,
        "path": str(path),
        "row_count": row_count,
        "block_sha256": observed_hash,
    }


def _integrate(time: list[float], current: list[float]) -> tuple[float, int, int]:
    if len(time) != len(current):
        raise ValueError("Time and current arrays must have equal lengths.")
    areas = []
    zero_intervals = 0
    negative_intervals = 0
    for index in range(len(time) - 1):
        delta = time[index + 1] - time[index]
        if delta == 0:
            zero_intervals += 1
        elif delta < 0:
            negative_intervals += 1
        else:
            areas.append(
                (abs(current[index]) + abs(current[index + 1])) * 0.5 * delta
            )
    return math.fsum(areas), zero_intervals, negative_intervals


def _coalesce_duplicate_times(
    time: list[float], current: list[float], policy: str
) -> tuple[list[float], list[float]]:
    if policy not in {"first", "last", "mean"}:
        raise ValueError(f"Unknown duplicate-time policy: {policy}")
    if len(time) != len(current):
        raise ValueError("Time and current arrays must have equal lengths.")
    result_time: list[float] = []
    result_current: list[float] = []
    index = 0
    while index < len(time):
        end = index + 1
        while end < len(time) and time[end] == time[index]:
            end += 1
        current_group = current[index:end]
        if policy == "first":
            selected = current_group[0]
        elif policy == "last":
            selected = current_group[-1]
        else:
            selected = math.fsum(current_group) / len(current_group)
        result_time.append(time[index])
        result_current.append(selected)
        index = end
    return result_time, result_current


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "n": 0,
            "mean": None,
            "mean_abs": None,
            "median": None,
            "p95_abs": None,
            "max_abs": None,
        }
    absolute = sorted(abs(value) for value in values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "mean_abs": statistics.mean(absolute),
        "median": statistics.median(values),
        "p95_abs": absolute[int(0.95 * (len(absolute) - 1))],
        "max_abs": absolute[-1],
    }


def _index_blocks(rows: list[dict[str, str]]) -> dict[
    tuple[str, int, int, int, int], dict[str, str]
]:
    result = {}
    for row in rows:
        key = (
            _tail_for_part(row["source_file"], int(row["source_part"])),
            int(row["source_part"]),
            int(row["step"]),
            int(row["start_line"]),
            int(row["end_line"]),
        )
        if key in result:
            raise ValueError(f"Duplicate raw block audit key: {key}")
        result[key] = row
    return result


def _timestamp_sensitivity(
    candidate: dict[str, str],
    block_audit_by_key: dict[tuple[str, int, int, int, int], dict[str, str]],
) -> list[dict[str, Any]]:
    part = int(candidate["dataset_part"])
    rpt = _read_mapped_block(candidate["step4_segment_ref"], part, block_audit_by_key)
    bol = _read_mapped_block(
        candidate["bol_step4_segment_ref"],
        int(candidate["bol_source_part"]),
        block_audit_by_key,
    )
    rows = []
    for axis in ("testtime", "steptime"):
        baseline_capacity, _, _ = _integrate(bol[axis], bol["amps"])
        original_capacity, zero_intervals, negative_intervals = _integrate(
            rpt[axis], rpt["amps"]
        )
        original_ratio = 100.0 * original_capacity / baseline_capacity
        for policy in ("as_recorded", "first", "last", "mean"):
            if policy == "as_recorded":
                times, currents = rpt[axis], rpt["amps"]
                bol_times, bol_currents = bol[axis], bol["amps"]
            else:
                times, currents = _coalesce_duplicate_times(
                    rpt[axis], rpt["amps"], policy
                )
                bol_times, bol_currents = _coalesce_duplicate_times(
                    bol[axis], bol["amps"], policy
                )
            numerator, _, _ = _integrate(times, currents)
            denominator, _, _ = _integrate(bol_times, bol_currents)
            ratio = 100.0 * numerator / denominator
            rows.append(
                {
                    "dataset_part": candidate["dataset_part"],
                    "physical_cell_id": candidate["physical_cell_id"],
                    "rpt_index": candidate["rpt_index"],
                    "time_axis": axis,
                    "duplicate_policy": policy,
                    "rpt_row_count": rpt["row_count"],
                    "rpt_zero_intervals": zero_intervals,
                    "rpt_negative_intervals": negative_intervals,
                    "rpt_integrated_native": numerator,
                    "bol_integrated_native": denominator,
                    "capacity_retention_soh_percent": ratio,
                    "delta_from_as_recorded_pp": ratio - original_ratio,
                    "amphr_span_ratio_percent": 100.0
                    * (
                        (max(rpt["amphr"]) - min(rpt["amphr"]))
                        / (max(bol["amphr"]) - min(bol["amphr"]))
                    ),
                    "rpt_block_sha256": rpt["block_sha256"],
                    "bol_block_sha256": bol["block_sha256"],
                    "raw_recomputation_disposition": candidate[
                        "raw_recomputation_disposition"
                    ],
                }
            )
    return rows


def build_report(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    record_rows = _read_csv(AUDIT_DIR / "independent_record_audit.csv")
    candidate_rows = [
        row
        for row in record_rows
        if row["current_manifest_disposition"] != "EXCLUDED_CONTROL"
    ]
    block_rows = _read_csv(AUDIT_DIR / "independent_block_audit.csv")
    block_audit_by_key = _index_blocks(block_rows)
    integration_rows = _read_csv(RECONSTRUCTION_DIR / "capacity_integration_audit.csv")
    secondary_rows = _read_csv(RECONSTRUCTION_DIR / "secondary_bol_sensitivity.csv")

    accepted = [
        row for row in candidate_rows if row["raw_recomputation_disposition"] == "ACCEPTED"
    ]
    axis_delta = [
        float(row["independent_soh_testtime_percent"])
        - float(row["independent_soh_steptime_percent"])
        for row in accepted
    ]
    counter_delta = [
        100.0
        * float(row["step4_amphr_span_native"])
        / float(row["bol_amphr_span_native"])
        - float(row["independent_soh_testtime_percent"])
        for row in accepted
    ]
    accepted_keys = {
        (row["dataset_part"], row["physical_cell_id"], row["rpt_index"])
        for row in accepted
    }
    secondary_delta = [
        float(row["secondary_minus_primary_soh_pp_qc"])
        for row in secondary_rows
        if (
            row["dataset_part"],
            row["physical_cell_id"],
            row["rpt_index"],
        )
        in accepted_keys
    ]
    block_discrepancies = [
        abs(float(row["testtime_vs_native_relative_discrepancy_percent"]))
        for row in integration_rows
        if row["testtime_vs_native_relative_discrepancy_percent"]
    ]
    if not secondary_delta or not block_discrepancies:
        raise ValueError("Required capacity sensitivity evidence is incomplete.")

    by_split = Counter(row["partition"] for row in accepted)
    if by_split != Counter({"train": 101, "validation": 37, "external_test": 40}):
        raise ValueError(f"Unexpected accepted split counts: {dict(by_split)}")
    exception = next(
        row
        for row in candidate_rows
        if row["physical_cell_id"] == "G2-Cell3" and row["rpt_index"] == "2"
    )
    timestamp_rows = _timestamp_sensitivity(exception, block_audit_by_key)
    g2_testtime = next(
        row
        for row in timestamp_rows
        if row["time_axis"] == "testtime"
        and row["duplicate_policy"] == "as_recorded"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "g2_cell3_rpt2_duplicate_time_sensitivity.csv", timestamp_rows)
    summary = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_authorized": False,
        "raw_227_file_scan_repeated": False,
        "targeted_raw_blocks_read": [
            exception["step4_segment_ref"],
            exception["bol_step4_segment_ref"],
        ],
        "targeted_raw_block_hashes_verified_against_retained_audit": True,
        "raw_recomputation_disposition_counts": {
            disposition: sum(
                row["raw_recomputation_disposition"] == disposition
                for row in candidate_rows
            )
            for disposition in ("ACCEPTED", "REVIEW", "REJECTED")
        },
        "proposed_future_training_records": {
            "train": by_split["train"],
            "validation": by_split["validation"],
            "external_test_training": 0,
            "external_test_candidate_references": by_split["external_test"],
            "review_records_quarantined": sum(
                row["raw_recomputation_disposition"] == "REVIEW"
                for row in candidate_rows
            ),
            "currently_cleared_for_training": 0,
        },
        "accepted_record_sensitivity": {
            "testtime_minus_steptime_soh_percentage_points": _distribution(axis_delta),
            "native_amphr_ratio_minus_integrated_current_time_soh_percentage_points": _distribution(
                counter_delta
            ),
            "accepted_step4_zero_or_negative_elapsed_intervals": {
                "testtime_zero": sum(int(row["step4_testtime_zero_intervals"]) for row in accepted),
                "testtime_negative": sum(int(row["step4_testtime_negative_intervals"]) for row in accepted),
                "steptime_zero": sum(int(row["step4_steptime_zero_intervals"]) for row in accepted),
                "steptime_negative": sum(int(row["step4_steptime_negative_intervals"]) for row in accepted),
            },
            "alternate_part1_bol_procedure_native_counter_qc_difference_pp": {
                "n": len(secondary_delta),
                "mean_absolute": statistics.mean(abs(value) for value in secondary_delta),
                "max_absolute": max(abs(value) for value in secondary_delta),
            },
        },
        "capacity_block_sensitivity": {
            "blocks_with_native_counter_comparison": len(block_discrepancies),
            "integrated_capacity_vs_native_amphr_relative_discrepancy_percent": {
                "median_absolute": statistics.median(block_discrepancies),
                "p95_absolute": sorted(block_discrepancies)[
                    int(0.95 * (len(block_discrepancies) - 1))
                ],
                "max_absolute": max(block_discrepancies),
            },
            "quadrature_method_sensitivity": {
                "primary_method": "composite trapezoidal integration over ordered samples with positive elapsed-time intervals",
                "alternative_rectangle_or_simpson_methods_recomputed": False,
                "reason": "Per-interval sample arrays are not retained in the audit outputs; calculating another quadrature rule would require reopening the mapped source blocks. No evidence currently favors an alternative rule.",
            },
        },
        "g2_cell3_rpt2_duplicate_timestamp_sensitivity": {
            "disposition": exception["raw_recomputation_disposition"],
            "testtime_repeated_intervals": int(exception["step4_testtime_zero_intervals"]),
            "steptime_repeated_intervals": int(exception["step4_steptime_zero_intervals"]),
            "policies": ["as_recorded", "first", "last", "mean"],
            "testtime_integrated_soh_percent": g2_testtime[
                "capacity_retention_soh_percent"
            ],
            "native_amphr_soh_percent": g2_testtime["amphr_span_ratio_percent"],
            "native_amphr_minus_integrated_soh_pp": g2_testtime[
                "amphr_span_ratio_percent"
            ]
            - g2_testtime["capacity_retention_soh_percent"],
            "remains_quarantined": True,
        },
        "measurement_semantics": {
            "testtime_steptime_units": "elapsed-time semantics confirmed; numeric unit unresolved; minutes are a strong consistency inference",
            "amps": "Maccor export documentation defines current in A; Oxford Amps header mapping is strongly supported but archived export profile is absent",
            "amphr": "Maccor export documentation defines capacity in Ah; Oxford Amphr mapping is strongly supported by native-counter comparison but archived export profile is absent",
            "temperature": "not used in capacity integration; numeric temperature interpretation and calibration remain unverified",
            "soh_scale": "common positive current/time scale cancels in a same-cell numerator/BoL ratio when both blocks share the scale",
        },
    }
    summary_path = output_dir / "capacity_sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report(output_dir, summary, timestamp_rows)
    _write_checksums(output_dir)
    return summary


def _write_report(
    output_dir: Path,
    summary: dict[str, Any],
    timestamp_rows: list[dict[str, Any]],
) -> None:
    record_sensitivity = summary["accepted_record_sensitivity"]
    counter = record_sensitivity[
        "native_amphr_ratio_minus_integrated_current_time_soh_percentage_points"
    ]
    axis = record_sensitivity["testtime_minus_steptime_soh_percentage_points"]
    secondary = record_sensitivity[
        "alternate_part1_bol_procedure_native_counter_qc_difference_pp"
    ]
    block_summary = summary["capacity_block_sensitivity"]
    block = block_summary[
        "integrated_capacity_vs_native_amphr_relative_discrepancy_percent"
    ]
    g2_deltas = [abs(float(row["delta_from_as_recorded_pp"])) for row in timestamp_rows]
    lines = [
        "# SoH measurement semantics and architecture blocker resolution",
        "",
        "This is a non-training evidence report. It does not change any label disposition or authorize training.",
        "",
        "## Measurement semantics: confirmed versus inferred",
        "",
        "| Item | Confirmed evidence | Remaining limitation |",
        "|---|---|---|",
        "| Protocol/instrument | Oxford Part 1 and Part 2 READMEs identify Maccor 4200; Part 3 identifies Maccor 4000. The READMEs define Step 4 discharge to 2.5 V and Step 10 pseudo-OCV charge; Parts 1–2 say Step 4 is CC-CV, while Part 3 says CC. | Part 3 is a distinct protocol/instrument and must remain external/domain-shift data. |",
        "| Time fields | Oxford CSV headers contain `TestTime` and `StepTime`; Maccor documentation calls these test and within-step elapsed time. | Neither Oxford README nor the inspected Maccor export page specifies the numeric unit or proves the Oxford export profile. Minutes are an inference from approximately 1-second sample increments and current-integral/native-counter agreement. |",
        "| Current/capacity fields | Oxford CSV headers contain `Amps` and `Amphr`. Maccor's general export documentation labels current as A and capacity as Ah. Step 4 current is discharge-direction and Step 10 current is charge-direction in the audited blocks. | The original Oxford Maccor export profile is absent, so the exact Oxford column mapping/scaling is not forensic proof. Keep reconstructed absolute capacities in native/current-time units; do not claim absolute Ah. |",
        "| Step 4/BoL comparability | ACCEPTED numerator/denominator pairs map to the same physical cell and compatible Step 4 procedure within their instrument/protocol domain. Parts 1–2 continuation cells use Part 1 BoL; Part 3 uses its own BoL and is separate. | Do not pool Part 3 with ordinary Train/Validation. Do not treat unresolved protocol/BoL REVIEW rows as compatible. |",
        "| Temperature | Temperature is not in the capacity integration or SoH formula. Parts 1–2 READMEs specify a 24 °C chamber; raw columns are `Temp1`/`Temp`. | Channel units, calibration, placement, and actual cell temperature are unverified. This does not change the arithmetic target, but actual measurement temperature can alter capacity and therefore target comparability; it also affects interpretation of the model's thermal features. |",
        "",
        "The generic Maccor export reference is [Exporting Battery Test Data](https://www.maccor.com/battery-testing-knowledge-center/data-analysis/exporting-battery-test-data.html). It supports the A/Ah field conventions, but does not recover the Oxford-specific saved export profile or the elapsed-time unit. The original source documents are the local Part 1–3 `Readme.txt` files; the dataset review records the same limitations.",
        "",
        "## SoH scale and numerical sensitivity",
        "",
        "The target is a capacity-retention ratio from independently integrated `|Amps|` over elapsed time. If both same-cell numerator and BoL denominator share a positive multiplicative current/time scale `c`, then `100 × (cQ_RPT)/(cQ_BoL) = 100 × Q_RPT/Q_BoL`; therefore seconds-versus-minutes or A-versus-mA does not change the ratio when the scale is common. This does not establish absolute Ah or prove that export scaling is common; same tester, matching columns, and protocol crosswalk support that assumption for each within-domain pair.",
        "",
        f"- Across {axis['n']} ACCEPTED records, TestTime-vs-StepTime SoH difference: mean absolute `{axis['mean_abs']:.3g}` pp; maximum absolute `{axis['max_abs']:.3g}` pp.",
        f"- Across {counter['n']} ACCEPTED records, native Amphr-span ratio minus integrated-current-time ratio: mean absolute `{counter['mean_abs']:.6g}` pp; maximum absolute `{counter['max_abs']:.6g}` pp.",
        f"- Across {block_summary['blocks_with_native_counter_comparison']} capacity blocks, integrated capacity vs native Amphr relative discrepancy: median absolute `{block['median_absolute']:.6g}%`; 95th percentile `{block['p95_absolute']:.6g}%`; maximum `{block['max_absolute']:.6g}%`.",
        f"- Native-Amphr QC ratio using the alternate Part 1 BoL procedure, for `{secondary['n']}` accepted observations, differs by mean absolute `{secondary['mean_absolute']:.6f}` pp and maximum `{secondary['max_absolute']:.6f}` pp. This is a counter-based sensitivity, not a recalculation of the primary integrated-current target. Denominator choice is materially more consequential than the observed integration-axis differences; retain the selected earliest complete Step 4 baseline and do not silently substitute the alternate.",
        f"- G2-Cell3 RPT2 has {summary['g2_cell3_rpt2_duplicate_timestamp_sensitivity']['testtime_repeated_intervals']} repeated intervals on each elapsed-time axis. First/last/mean duplicate collapse changes the integrated ratio by at most `{max(g2_deltas):.6g}` pp in the targeted block. It remains REVIEW because repeated timestamp semantics are unresolved; this numerical stability is not a disposition change.",
        f"- For G2-Cell3 RPT2, the TestTime integrated ratio is `{summary['g2_cell3_rpt2_duplicate_timestamp_sensitivity']['testtime_integrated_soh_percent']:.8f}%`; the native-Amphr ratio is `{summary['g2_cell3_rpt2_duplicate_timestamp_sensitivity']['native_amphr_soh_percent']:.8f}%` (difference `{summary['g2_cell3_rpt2_duplicate_timestamp_sensitivity']['native_amphr_minus_integrated_soh_pp']:.6f}` pp). It remains quarantined.",
        "- The signed-current alternative would be negative capacity for the audited discharge direction; using absolute current is required for a positive capacity numerator. A consistent sign reversal in both numerator and denominator would mathematically cancel in the ratio, but is not an acceptable measurement convention.",
        "- The independent sensitivity pass does not recompute left/right rectangle or Simpson quadrature over all accepted raw blocks because per-interval arrays were not persisted. The primary method remains trapezoidal; do not claim those alternative quadrature sensitivities were measured. A new mapped-block extraction is needed if the user requires that comparison.",
        "",
        "## Proposed label eligibility (not a status promotion)",
        "",
        "| Split | Structurally accepted and label ACCEPTED | REVIEW quarantined | Proposed future cohort |",
        "|---|---:|---:|---:|",
        "| Train | 101 | 1 | 101, conditional |",
        "| Validation | 37 | 0 | 37, conditional |",
        "| Part 3 External | 40 | 24 | 0 for training; 40 remain external-only candidates |",
        "",
        "Current training-cleared count remains **0** and `training_authorized` remains false. The proposed 101/37 cohort is conditional on explicit approval of (1) same-cell common-scale ratio labels without absolute-Ah claims, (2) the verified same-protocol BoL mappings, (3) the native-temperature feature policy below, and (4) the current label/split hashes. Keep all 25 REVIEW records quarantined. Missing instrument accuracy is a limitation to report, not a basis for inventing tolerances; the target remains a reconstructed reference rather than official Oxford ground truth.",
        "",
        "Eligibility gates: exact Guide/procedure/cell/RPT identity; complete Step 4 discharge with correct sign, endpoint and finite monotonic-time measurements; matched same-cell same-protocol Step 4 BoL; numerator and denominator from a common scale domain; finite aligned non-leaking modalities; split isolation by physical cell; and no REVIEW label. For the five-feature architecture, either approve raw temperature in native scale without a °C claim (Train/Validation are Parts 1–2 on Maccor 4200) or revise the feature contract before training. Do not impute temperature.",
        "",
        "## Architecture specification resolution",
        "",
        "The approved candidate definitions in `specs/bafuse/spec.md` specify a two-layer unidirectional LSTM and an attention branch that pools the CNN stem to 64 positions before self-attention. The former implementation had one LSTM layer; its attention branch used a different 32-channel `k=5/k=3` stem and attended over 128 positions. The implementation is now explicitly aligned to the specification; no training was run.",
        "",
        "External inputs are discharge `(B,128)`, charge `(B,128)`, and physics `(B,5)`. Each curve encoder and the physics encoder output `(B,32)`. Fixed concatenation produces `(B,96)`, shared MLP head `96→64→32→1`, output `(B,1)`. Parameter counts from the instantiated updated models:",
        "",
        "| Architecture | Curve representation per branch | Parameters per curve encoder | Fusion/output | Full-model parameters |",
        "|---|---|---:|---|---:|",
        "| CNN1D + CNN1D + MLP | Conv1d `1→16,k9`, pool-2, Conv1d `16→32,k5`, adaptive pool `(B,32)` | 2,752 | `(B,96)→(B,1)` | 15,073 |",
        "| LSTM + LSTM + MLP | 2-layer unidirectional LSTM, hidden 32, dropout 0, final state `(B,32)` | 12,928 | `(B,96)→(B,1)` | 35,425 |",
        "| TCN + TCN + MLP | causal residual channel widths `1→16→16→32→32`, global pool `(B,32)` | 6,096 | `(B,96)→(B,1)` | 21,761 |",
        "| CNN1D-Attention + CNN1D-Attention + MLP | CNN stem `(B,32,64)`, batch-first 4-head Transformer input `(B,64,32)`, FFN 32→64→32, mean pool `(B,32)` | 11,296 | `(B,96)→(B,1)` | 32,161 |",
        "",
        "The parameter differences are intentional and must be reported; they do not imply equal model capacity. MACs/latency/peak memory remain analytical or unmeasured, not benchmark results.",
        "",
        "## Reproducibility and approval",
        "",
        "The latest audit SHA-256 manifest is verified in the resolution run. Its 31 listed files matched their recorded hashes before this continuation. This resolution creates a separate manifest for its inputs, code, tests, and outputs.",
        "",
        "The original raw scan did not capture a launch-time source-code hash. This weakens exact forensic reproducibility of the historical process: the raw rows are tied to retained block SHA-256 values, and the final scanner/report code is hashed, but it does not prove which exact source revision executed at scan launch. The integration implementation is recorded as unchanged; Guide post-processing was corrected from retained rows. Do not claim exact launch-code provenance. No full rescan is needed for this report; only the two targeted G2-Cell3 RPT2/BoL blocks were reread and their hashes matched the retained audit.",
        "",
        "Before training, obtain approval for the conditional target policy and temperature feature policy; confirm 101 Train/37 Validation keys and keep all 25 REVIEW records quarantined; approve the spec-aligned definitions, target/split/config/code hashes, compute budget and output path; and create a scope-matched authorization file. Keep Part 3 locked to a separately approved final external evaluation.",
        "",
        "Focused regression tests cover the specified architecture shapes/parameter counts and duplicate-timestamp integration policies. `training_authorized: false`; no training, external prediction evaluation, PyDMA, or Half-Cell fitting was run.",
        "",
    ]
    (output_dir / "measurement_semantics_architecture_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_checksums(output_dir: Path) -> None:
    targets = {
        "output:capacity_sensitivity_summary": output_dir / "capacity_sensitivity_summary.json",
        "output:g2_duplicate_timestamp_sensitivity": output_dir / "g2_cell3_rpt2_duplicate_time_sensitivity.csv",
        "output:measurement_report": output_dir / "measurement_semantics_architecture_report.md",
        "output:updated_approval_checklist": output_dir / "updated_approval_checklist.md",
        "output:test_results": output_dir / "test_results.json",
        "code:measurement_semantics_audit": Path(__file__).resolve(),
        "code:architecture_models": PROJECT_ROOT / "src" / "bafuse" / "hybrid" / "architecture_models.py",
        "code:measurement_sensitivity_tests": PROJECT_ROOT / "tests" / "test_soh_measurement_sensitivity.py",
        "code:architecture_tests": PROJECT_ROOT / "tests" / "test_bafuse_architecture_models.py",
        "config:architecture_config": BENCHMARK_DIR / "architecture_matrix_v1" / "architecture_config.json",
        "documentation:architecture_readme": BENCHMARK_DIR / "architecture_matrix_v1" / "README.md",
        "source:part1_readme": PART_READMES[0],
        "source:part2_readme": PART_READMES[1],
        "source:part3_readme": PART_READMES[2],
        "source:dataset_review": PROJECT_ROOT / "experiments" / "dataset_review" / "README.md",
        "source:research_spec": PROJECT_ROOT / "specs" / "bafuse" / "spec.md",
        "source:prior_audit_manifest": AUDIT_DIR / "sha256_manifest.json",
        "source:prior_audit_summary": AUDIT_DIR / "independent_audit_summary.json",
    }
    files = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in targets.items()
    }
    (output_dir / "sha256_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "algorithm": "SHA-256",
                "training_authorized": False,
                "external_references": [
                    {
                        "name": "Maccor Exporting Battery Test Data",
                        "url": "https://www.maccor.com/battery-testing-knowledge-center/data-analysis/exporting-battery-test-data.html",
                        "retrieved_for": "generic current/capacity field units and elapsed-time semantics",
                    }
                ],
                "files": files,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the targeted SoH semantics report.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    result = build_report(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
