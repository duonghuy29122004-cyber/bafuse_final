from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from bafuse.labeling.pydma_policy import pydma_fit_passes_gate
from bafuse.labeling.pydma_runner import _fit_one, _read_pocv_with_audit
from pydma import DMAAnalyzer, DMAConfig
from pydma.preprocessing.loader import load_ocp


P12_RESULTS = ROOT / "data/processed/label_pydma_fit_part1_2/pydma_results_rmse15.csv"
P3_RESULTS = ROOT / "data/processed/label_pydma_fit_part3/run_009_part3_pydma/pydma_results.csv"
P2_RMSE10_RESULTS = ROOT / "experiments/results/data_audits/part2_pydma_review/pydma_results.csv"
RAW_MAPPING = ROOT / "data/manifests/unified_raw_mapping.csv"
OBSERVATIONS = ROOT / "data/manifests/bafuse_rpt_observations.csv"
GUIDE_INDEX = ROOT / "experiments/dataset_review/guide_file_index.csv"
PREDICTIONS = ROOT / "experiments/soh_forecast/bafuse_three_encoder_v1/fusion_predictions.csv"
SUPPLEMENTAL_PREDICTIONS = (
    ROOT
    / "experiments/soh_forecast/bafuse_three_encoder_v1/evaluation_audit_v1/"
    / "supplemental_g10_cell2_predictions.csv"
)
HALF_CELL_ROOT = ROOT / "data/raw/oxford/part1/Half_Cell/Half Cell"

VALUE_COLUMNS = (
    ("LLI_percent_vs_BoL", "negative_lli"),
    ("LAM_anode_percent_vs_BoL", "negative_lam_anode"),
    ("LAM_cathode_percent_vs_BoL", "negative_lam_cathode"),
)
PARAMETER_NAMES = ("alpha_an", "beta_an", "alpha_ca", "beta_ca")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def group_number(value: str) -> str:
    match = re.search(r"\d+", value or "")
    return match.group(0) if match else ""


def cell_number(value: str) -> str:
    match = re.search(r"\d+", value or "")
    return match.group(0) if match else ""


def rpt_number(value: str) -> int | None:
    match = re.search(r"rpt\s*(\d+)", value or "", re.IGNORECASE)
    return int(match.group(1)) if match else None


def part_rank(value: str) -> int:
    match = re.search(r"\d+", value or "")
    return int(match.group(0)) if match else 0


def parse_json(value: str) -> dict[str, Any]:
    try:
        data = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value if value is not None else "").replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def source_fit_rows() -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, str]]]:
    raw: list[dict[str, Any]] = []
    p12_all = read_rows(P12_RESULTS)
    p3_all = read_rows(P3_RESULTS)
    sources = (
        ("Part 1/2", p12_all),
        ("Part 3", p3_all),
    )
    for source_name, rows in sources:
        for item in rows:
            group = group_number(item.get("group_id", ""))
            if item.get("diagnostic_label_status") != "diagnostic_estimate_accepted":
                continue
            if source_name == "Part 1/2" and group not in {"1", "2", "3", "4"}:
                continue
            if source_name == "Part 3" and group not in {"7", "8", "9", "10"}:
                continue
            part = item.get("source_part", "") if source_name == "Part 1/2" else "Part 3"
            curve = parse_json(item.get("curve_audit_json", ""))
            selected = next(
                (
                    segment
                    for segment in curve.get("segments", [])
                    if segment.get("segment_index") == curve.get("selected_segment")
                ),
                {},
            )
            metrics = {key: number(item.get(key)) for key, _ in VALUE_COLUMNS}
            row = {
                "part": part,
                "group": group,
                "physical_cell_id": f"G{group}-Cell{cell_number(item.get('cell_id', ''))}",
                "rpt": item.get("checkup", ""),
                "rpt_number": rpt_number(item.get("checkup", "")),
                "direction": item.get("direction", ""),
                "test_step": item.get("test_step", ""),
                "source_file": item.get("source_file", ""),
                "curve_source_path_recorded": curve.get("source_file", ""),
                "curve_status": curve.get("status", ""),
                "selected_segment": curve.get("selected_segment", ""),
                "curve_sample_count": selected.get("sample_count", ""),
                "capacity_min_native": selected.get("capacity_min", ""),
                "capacity_max_native": selected.get("capacity_max", ""),
                "capacity_span_native": (
                    number(selected.get("capacity_max")) - number(selected.get("capacity_min"))
                    if number(selected.get("capacity_max")) is not None
                    and number(selected.get("capacity_min")) is not None
                    else ""
                ),
                "fit_rmse_mV": number(item.get("fit_rmse_mV")),
                "rmse_threshold_mV": number(item.get("fit_threshold_mV")) or 15.0,
                "accepted_runs": int(item.get("accepted_runs") or 0),
                "total_runs": int(item.get("total_runs") or 0),
                "diagnostic_label_status": item.get("diagnostic_label_status", ""),
                "source_run": item.get("source_run", ""),
                **metrics,
            }
            threshold = row["rmse_threshold_mV"]
            if (
                row["fit_rmse_mV"] is None
                or row["fit_rmse_mV"] >= threshold
                or row["accepted_runs"] < 3
            ):
                raise ValueError(
                    "Accepted-status row violates the reported gate: "
                    f"{row['physical_cell_id']} {row['rpt']} {row['direction']}"
                )
            for value_key, flag_key in VALUE_COLUMNS:
                value = metrics[value_key]
                row[flag_key] = value is not None and value < 0
            raw.append(row)
    return raw, p12_all, p3_all


def distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}
    return {
        "min": ordered[0],
        "q1": float(np.percentile(ordered, 25)),
        "median": float(statistics.median(ordered)),
        "q3": float(np.percentile(ordered, 75)),
        "max": ordered[-1],
        "mean": float(statistics.mean(ordered)),
        "std": float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
    }


def negative_report_rows(fits: list[dict[str, Any]]) -> list[list[Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in fits:
        grouped[
            (row["part"], row["group"], row["physical_cell_id"], row["direction"])
        ].append(row)
    result = []
    for (part, group, cell, direction), rows in sorted(grouped.items()):
        negative_counts = [
            sum(bool(row[flag]) for row in rows)
            for _, flag in VALUE_COLUMNS
        ]
        if any(negative_counts):
            result.append(
                [part, f"Group {group}", cell, direction, len(rows), *negative_counts]
            )
    return result


def make_negative_report(
    fits: list[dict[str, Any]],
    strict10_rows: list[dict[str, str]],
    multi_rows: list[dict[str, Any]],
    use_cuda: bool,
    baseline_summary: str,
) -> str:
    parts = sorted({row["part"] for row in fits}, key=part_rank)
    lines = [
        "# PyDMA Negative-Estimate and Fit-Stability Audit",
        "",
        "Read-only analysis of saved RMSE-15 PyDMA outputs for Part 1, Part 2 Groups 1-4, "
        "and Part 3. Parts 2 Groups 5-6 were excluded. The separate Part 2 RMSE-10 run is "
        "reported only as a sensitivity check.",
        "",
        "## Accepted-fit summary",
        "",
        f"Primary cohort: **{len(fits)} accepted directional fits**. An accepted diagnostic "
        "estimate required `diagnostic_label_status=diagnostic_estimate_accepted`, RMSE below "
        "the source threshold, and at least three accepted optimizer runs. This is a fit-quality "
        "gate, not proof that the degradation parameters are identifiable or physically unique.",
        "",
    ]
    summary_rows = []
    for part in parts:
        group = [row for row in fits if row["part"] == part]
        rmse = [row["fit_rmse_mV"] for row in group if row["fit_rmse_mV"] is not None]
        summary_rows.append(
            [
                part,
                len(group),
                sum(row["direction"] == "charge" for row in group),
                sum(row["direction"] == "discharge" for row in group),
                f"{min(rmse):.3f}-{max(rmse):.3f}" if rmse else "",
                f"{statistics.mean(rmse):.3f}" if rmse else "",
                min(row["accepted_runs"] for row in group),
                max(row["accepted_runs"] for row in group),
            ]
        )
    lines.append(
        markdown_table(
            [
                "Part",
                "Accepted fits",
                "Charge",
                "Discharge",
                "RMSE min-max (mV)",
                "Mean RMSE (mV)",
                "Min accepted runs",
                "Max accepted runs",
            ],
            summary_rows,
        )
    )
    lines.append("")

    distribution_rows = []
    for part in ["All included", *parts]:
        sample = fits if part == "All included" else [row for row in fits if row["part"] == part]
        for value_key, _ in VALUE_COLUMNS:
            values = [row[value_key] for row in sample if row[value_key] is not None]
            dist = distribution(values)
            negatives = sum(value < 0 for value in values)
            distribution_rows.append(
                [
                    part,
                    value_key.replace("_percent_vs_BoL", "").replace("_", " "),
                    len(values),
                    f"{negatives} ({100 * negatives / len(values):.2f}%)" if values else "0",
                    f"{dist['min']:.3f}" if dist else "",
                    f"{dist['median']:.3f}" if dist else "",
                    f"{dist['max']:.3f}" if dist else "",
                ]
            )
    lines.extend(
        [
            "## Negative estimates and distributions",
            "",
            "The CSV lists every accepted fit with Part, cell, RPT, direction, RMSE, accepted "
            "run count, native selected-curve span, each estimate, and per-mechanism negative flags.",
            "",
            markdown_table(
                ["Part", "Mechanism", "N", "Negative n (%)", "Minimum (%)", "Median (%)", "Maximum (%)"],
                distribution_rows,
            ),
            "",
            "All negative values in the included RMSE-15 cohort are in discharge-direction fits; "
            "the charge-direction fits have no negative estimate for LLI, anode LAM, or cathode LAM. "
            "The row-level CSV gives the exact cell/RPT/direction for each negative estimate.",
            "",
            "### Nonzero negative counts by cell and direction",
            "",
            markdown_table(
                [
                    "Part",
                    "Group",
                    "Physical cell",
                    "Direction",
                    "Accepted fits",
                    "Negative LLI",
                    "Negative anode LAM",
                    "Negative cathode LAM",
                ],
                negative_report_rows(fits),
            ),
            "",
        ]
    )

    strict10_accepted = [
        row
        for row in strict10_rows
        if row.get("diagnostic_label_status") == "diagnostic_estimate_accepted"
        and group_number(row.get("group_id", "")) in {"1", "2", "3", "4"}
    ]
    strict_rows = []
    for value_key, _ in VALUE_COLUMNS:
        values = [number(row.get(value_key)) for row in strict10_accepted]
        values = [value for value in values if value is not None]
        strict_rows.append(
            [
                value_key.replace("_percent_vs_BoL", "").replace("_", " "),
                len(values),
                sum(value < 0 for value in values),
                f"{min(values):.3f}" if values else "",
            ]
        )
    lines.extend(
        [
            "## Part 2 RMSE-10 sensitivity",
            "",
            f"The separate stricter Part 2 run has **{len(strict10_accepted)} accepted fits** "
            "in Groups 1-4. It is not merged with the RMSE-15 primary cohort.",
            "",
            markdown_table(
                ["Mechanism", "Accepted fits", "Negative", "Minimum (%)"],
                strict_rows,
            ),
            "",
            "## Fitting and physical-plausibility checks",
            "",
            "- Charge fits use `CH_HC_anode.csv` and `CH_HC_cathode.csv`; discharge fits use "
            "`DCH_HC_anode.csv` and `DHC_HC_cathode.csv`. The half-cell reference files contain "
            "positive current for the charge pair and negative current for the discharge pair. "
            "PyDMA's `ElectrodeOCP` validates/mirrors the half-cell voltage orientation to its "
            "internal convention. The full-cell extraction uses Step 10 with positive current "
            "for charge and Step 8 with negative current for discharge.",
            "- The runner uses the earliest same-direction Guide-listed BoL curve as reference. "
            f"{baseline_summary} The Part 2 Groups 1-4 runner selects the earliest unique "
            "same-cell Part 1 BoL source (see `run_part2_pydma.py`). No negative estimate is "
            "calculated when the same-direction BoL fit is not accepted.",
            "- Pseudo-OCV extraction filters `|Amps| <= 0.2 A`, keeps the protocol direction, "
            "splits at non-increasing `Amphr`, and selects the longest valid monotone segment "
            "(ties within 1% go to the earliest). The optimizer normalizes the curve axis to 0-1, "
            "but separately retains the native `Amphr` span for capacity/degradation calculations. "
            "The dataset does not establish absolute `Amphr` units; relative estimates assume the "
            "same scale is consistent between each cell's BoL and RPT.",
            "- Fitting uses OCV-only weight 100 (DVA/ICA weights 0), 500 resampled points, "
            "`fast` differential evolution, `maxiter=50`, `popsize=30`, polish enabled, seed 211, "
            "up to 10 attempts to obtain at least 3 accepted runs, and a 15 mV RMSE gate. Blended "
            "electrodes, first-cycle inhomogeneity, and resistance offset are disabled. The four "
            "free shape parameters are bounded by alpha-anode [1, 2], beta-anode [-1, 0], "
            "alpha-cathode [1, 2.1], beta-cathode [-1, 0]; the configured initial vector is "
            "[1.2, 0, 1.1, -0.1].",
            "- Reported LAM and LLI are relative transformations, not constrained nonnegative "
            "outputs. For example, anode LAM is "
            "`1 - (alpha_an_current * Q_current) / (alpha_an_BoL * Q_BoL)`. LLI similarly "
            "compares `(alpha_ca + beta_ca - beta_an) * Q` against the BoL inventory. A negative "
            "value therefore means the fitted current component/inventory exceeds its fitted BoL "
            "reference. This can be physically real, a reference/curve mismatch, or an optimizer "
            "equifinality; the RMSE/run-count gate alone cannot distinguish those cases.",
            "- The saved results do not contain parameter vectors, so their parameter-level "
            "identifiability cannot be audited from the CSV alone. The following multi-start "
            "experiment reruns three representative negative discharge fits and their BoL references "
            "with seeds 211-215, retaining every parameter vector and outcome. "
            f"CUDA optimizer used: **{use_cuda}**.",
            "",
            "## Multi-start parameter stability",
            "",
            "No fit was selected by whether its estimate was positive. Each seed is reported; "
            "a result is marked accepted only under the original fit gate.",
            "",
        ]
    )
    cases = sorted({row["case_id"] for row in multi_rows})
    for case in cases:
        sample = [row for row in multi_rows if row["case_id"] == case]
        target_rows = [row for row in sample if row["fit_role"] == "target"]
        accepted = [row for row in target_rows if row["target_gate_passed"] and row["baseline_gate_passed"]]
        case_info = target_rows[0] if target_rows else {}
        lines.extend(
            [
                f"### {case}: {case_info.get('physical_cell_id', '')} "
                f"{case_info.get('rpt', '')} {case_info.get('direction', '')}",
                "",
                f"Stored estimate: LLI={case_info.get('stored_lli_percent', '')}%, "
                f"anode LAM={case_info.get('stored_lam_anode_percent', '')}%, "
                f"cathode LAM={case_info.get('stored_lam_cathode_percent', '')}%. "
                f"Accepted paired seed results: {len(accepted)}/{len(target_rows)}.",
            ]
        )
        if accepted:
            estimates = []
            for key, label in (
                ("lli_percent", "LLI"),
                ("lam_anode_percent", "anode LAM"),
                ("lam_cathode_percent", "cathode LAM"),
            ):
                values = [float(row[key]) for row in accepted if row.get(key) not in ("", None)]
                if values:
                    estimates.append(
                        f"{label} range {min(values):.3f} to {max(values):.3f}%"
                    )
            lines.append("Multi-start estimate ranges: " + "; ".join(estimates) + ".")
            for index, name in enumerate(PARAMETER_NAMES):
                values = []
                for row in accepted:
                    params = parse_json(row["target_parameters_json"])
                    vector = params.get("values", [])
                    if len(vector) > index and number(vector[index]) is not None:
                        values.append(float(vector[index]))
                if values:
                    lines.append(
                        f"- `{name}` target fit: mean {statistics.mean(values):.6g}, "
                        f"SD {statistics.stdev(values) if len(values) > 1 else 0.0:.6g}, "
                        f"range {min(values):.6g} to {max(values):.6g}."
                    )
        else:
            lines.append("No seed produced a paired accepted baseline and target fit.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation and follow-up",
            "",
            "The negative estimates are not a sign error established by the current evidence: "
            "the selected raw curves use the documented current directions, and the half-cell files "
            "are direction-specific. The transformation explicitly permits negative estimates. "
            "The evidence does establish that RMSE acceptance is not a physical-plausibility or "
            "parameter-identifiability gate. Use the multi-start ranges to distinguish stable "
            "parameter/estimate results from low-RMSE but decomposition-sensitive fits. Do not clip, "
            "take absolute values, or select fits based on sign. No canonical PyDMA output was changed.",
            "",
            "## Reproducibility",
            "",
            "From the repository root, run:",
            "",
            "```powershell",
            r".\.venv-torch-cuda\Scripts\python.exe experiments\results\data_audits\pydma_eol_audit_v1\run_audit.py",
            "```",
            "",
            "The command reads the source manifests/results and writes only to this versioned audit "
            "directory. It does not retrain SoH models, modify labels, checkpoints, PyDMA canonical "
            "outputs, or the dashboard.",
            "",
        ]
    )
    return "\n".join(lines)


def fit_stability(
    fits: list[dict[str, Any]],
    p12_all: list[dict[str, str]],
    p3_all: list[dict[str, str]],
    raw_mapping: list[dict[str, str]],
    observations: list[dict[str, str]],
    seeds: list[int],
    use_cuda: bool,
) -> list[dict[str, Any]]:
    p12_fit = [row for row in fits if row["part"] in {"Part 1", "Part 2"}]
    p3_fit = [row for row in fits if row["part"] == "Part 3"]
    cases: list[tuple[str, dict[str, Any], list[dict[str, str]], str]] = []

    lowest_part1_lli = min(
        (
            row
            for row in p12_fit
            if row["part"] == "Part 1"
            and row["direction"] == "discharge"
            and row["LLI_percent_vs_BoL"] is not None
            and row["LLI_percent_vs_BoL"] < 0
        ),
        key=lambda row: row["LLI_percent_vs_BoL"],
    )
    cases.append(("p1_negative_lli", lowest_part1_lli, p12_all, "Part 1"))

    lowest_part2_anode = min(
        (
            row
            for row in p12_fit
            if row["part"] == "Part 2"
            and row["direction"] == "discharge"
            and row["LAM_anode_percent_vs_BoL"] is not None
            and row["LAM_anode_percent_vs_BoL"] < 0
        ),
        key=lambda row: row["LAM_anode_percent_vs_BoL"],
    )
    cases.append(("p2_negative_anode_lam", lowest_part2_anode, p12_all, "Part 1"))

    lowest_p3_lli = min(
        (
            row
            for row in p3_fit
            if row["direction"] == "discharge"
            and row["LLI_percent_vs_BoL"] is not None
            and row["LLI_percent_vs_BoL"] < 0
        ),
        key=lambda row: row["LLI_percent_vs_BoL"],
    )
    cases.append(("p3_negative_lli_lam", lowest_p3_lli, p3_all, ""))

    references = {
        "charge": (
            load_ocp(
                HALF_CELL_ROOT / "CH_HC_anode.csv",
                electrode_type="anode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
            load_ocp(
                HALF_CELL_ROOT / "CH_HC_cathode.csv",
                electrode_type="cathode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
        ),
        "discharge": (
            load_ocp(
                HALF_CELL_ROOT / "DCH_HC_anode.csv",
                electrode_type="anode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
            load_ocp(
                HALF_CELL_ROOT / "DHC_HC_cathode.csv",
                electrode_type="cathode",
                soc_col="SoC",
                voltage_col="Volts",
            ),
        ),
    }
    results: list[dict[str, Any]] = []
    for case_id, target, source_rows, baseline_part in cases:
        source_item = next(
            row
            for row in source_rows
            if row.get("group_id") == f"Group {target['group']}"
            and row.get("cell_id") == f"Cell {target['physical_cell_id'].split('-Cell')[-1]}"
            and row.get("direction") == target["direction"]
            and row.get("diagnostic_label_status") == "reference_baseline"
            and (not baseline_part or row.get("source_part") == baseline_part)
        )
        target_file_id = None
        match = re.search(r"TPG\d+[._](\d+)", target["source_file"], re.IGNORECASE)
        if match:
            target_file_id = int(match.group(1))
        target_part = 3 if target["part"] == "Part 3" else part_rank(target["part"])
        target_source = next(
            (
                item
                for item in raw_mapping
                if item.get("dataset_part") == str(target_part)
                and item.get("physical_cell_id") == target["physical_cell_id"]
                and item.get("source_modality") == "cycler_procedure"
                and item.get("source_extension", "").casefold() == ".csv"
                and item.get("source_status") == "present"
                and item.get("procedure_file_id") == str(target_file_id)
            ),
            None,
        )
        if target_source is None:
            raise FileNotFoundError(
                f"Cannot map target curve for {case_id}: {target['source_file']}"
            )
        observation = next(
            (
                row
                for row in observations
                if row.get("physical_cell_id") == target["physical_cell_id"]
                and row.get("part") == str(target_part)
            ),
            None,
        )
        if observation is None:
            raise ValueError(f"No canonical observation row for {target['physical_cell_id']}")
        baseline_path = (
            ROOT
            / "data/raw/oxford"
            / f"part{1 if target_part == 2 else target_part}"
            / observation["bol_reference_file"]
        )
        target_path = ROOT / target_source["repository_path"]
        if not baseline_path.exists() or not target_path.exists():
            raise FileNotFoundError(
                f"Multi-start input missing for {case_id}: baseline={baseline_path.exists()}, "
                f"target={target_path.exists()}"
            )

        baseline_curve, baseline_audit = _read_pocv_with_audit(
            baseline_path, 10 if target["direction"] == "charge" else 8, target["direction"]
        )
        target_curve, target_audit = _read_pocv_with_audit(
            target_path, 10 if target["direction"] == "charge" else 8, target["direction"]
        )
        if baseline_curve is None or target_curve is None:
            raise ValueError(
                f"Multi-start input curve unavailable for {case_id}: "
                f"baseline={baseline_audit.get('status')}, target={target_audit.get('status')}"
            )
        anode, cathode = references[target["direction"]]
        stored_values = {key: target[key] for key, _ in VALUE_COLUMNS}
        print(
            f"Multi-start {case_id}: {target['physical_cell_id']} {target['rpt']} "
            f"seeds={seeds[0]}-{seeds[-1]} cuda={use_cuda}",
            flush=True,
        )
        for seed in seeds:
            record: dict[str, Any] = {
                "case_id": case_id,
                "physical_cell_id": target["physical_cell_id"],
                "group": target["group"],
                "part": target["part"],
                "rpt": target["rpt"],
                "direction": target["direction"],
                "seed": seed,
                "stored_lli_percent": stored_values["LLI_percent_vs_BoL"],
                "stored_lam_anode_percent": stored_values["LAM_anode_percent_vs_BoL"],
                "stored_lam_cathode_percent": stored_values["LAM_cathode_percent_vs_BoL"],
                "baseline_source": baseline_path.as_posix(),
                "target_source": target_path.as_posix(),
                "cuda_used": use_cuda,
            }
            baseline_result = _fit_one(
                target["direction"],
                baseline_curve[0],
                baseline_curve[1],
                anode,
                cathode,
                seed=seed,
                context=f"{case_id} BoL seed={seed}",
                use_cuda=use_cuda,
                maxiter=50,
                popsize=30,
                max_tries=10,
            )
            target_result = _fit_one(
                target["direction"],
                target_curve[0],
                target_curve[1],
                anode,
                cathode,
                seed=seed,
                context=f"{case_id} target seed={seed}",
                use_cuda=use_cuda,
                maxiter=50,
                popsize=30,
                max_tries=10,
            )
            baseline_gate = pydma_fit_passes_gate(
                rmse_mV=baseline_result.rmse * 1000,
                accepted_runs=baseline_result.n_accepted_runs,
                is_accepted=baseline_result.is_accepted,
            )
            target_gate = pydma_fit_passes_gate(
                rmse_mV=target_result.rmse * 1000,
                accepted_runs=target_result.n_accepted_runs,
                is_accepted=target_result.is_accepted,
            )
            record.update(
                {
                    "fit_role": "target",
                    "baseline_fit_rmse_mV": baseline_result.rmse * 1000,
                    "baseline_accepted_runs": baseline_result.n_accepted_runs,
                    "baseline_total_runs": baseline_result.n_total_runs,
                    "baseline_gate_passed": baseline_gate,
                    "baseline_parameters_json": json.dumps(
                        {
                            "names": PARAMETER_NAMES,
                            "values": [
                                float(value)
                                for value in baseline_result.fitted_params.to_array()[:4]
                            ],
                        },
                        allow_nan=False,
                    ),
                    "target_fit_rmse_mV": target_result.rmse * 1000,
                    "target_accepted_runs": target_result.n_accepted_runs,
                    "target_total_runs": target_result.n_total_runs,
                    "target_gate_passed": target_gate,
                    "target_parameters_json": json.dumps(
                        {
                            "names": PARAMETER_NAMES,
                            "values": [
                                float(value)
                                for value in target_result.fitted_params.to_array()[:4]
                            ],
                        },
                        allow_nan=False,
                    ),
                    "estimate_status": (
                        "accepted"
                        if baseline_gate and target_gate
                        else "raw_comparison_not_accepted"
                    ),
                    "lli_percent": "",
                    "lam_anode_percent": "",
                    "lam_cathode_percent": "",
                }
            )
            comparison = DMAAnalyzer(
                config=DMAConfig(direction=target["direction"]),
                anode=anode,
                cathode=cathode,
            ).compare_with_reference(
                reference_params=baseline_result.fitted_params,
                current_params=target_result.fitted_params,
                reference_capacity=float(np.ptp(baseline_curve[0])),
                current_capacity=float(np.ptp(target_curve[0])),
            )
            record["lli_percent"] = comparison.lli * 100
            record["lam_anode_percent"] = comparison.lam_anode * 100
            record["lam_cathode_percent"] = comparison.lam_cathode * 100
            results.append(record)
    return results


def baseline_consistency_summary(rows: list[dict[str, str]]) -> str:
    baselines = [
        row
        for row in rows
        if row.get("diagnostic_label_status") == "reference_baseline"
        and group_number(row.get("group_id", "")) in {"1", "2", "3", "4"}
    ]
    keys = sorted(
        {
            (row.get("group_id", ""), row.get("cell_id", ""), row.get("direction", ""))
            for row in baselines
        }
    )
    paired = 0
    mismatched = 0
    part1_only = 0
    part2_only = 0
    for key in keys:
        subset = [row for row in baselines if (row.get("group_id"), row.get("cell_id"), row.get("direction")) == key]
        by_part = {row.get("source_part"): row for row in subset}
        if "Part 1" in by_part and "Part 2" in by_part:
            paired += 1
            left = by_part["Part 1"]
            right = by_part["Part 2"]
            if (
                left.get("source_file") != right.get("source_file")
                or left.get("fit_rmse_mV") != right.get("fit_rmse_mV")
            ):
                mismatched += 1
        elif "Part 1" in by_part:
            part1_only += 1
        elif "Part 2" in by_part:
            part2_only += 1
    return (
        f"Among saved Part 1/2 baseline rows, {paired} cell-direction keys have both Part 1 "
        f"and Part 2 records; {mismatched} paired source/RMSE values differ. There are "
        f"{part1_only} Part 1-only and {part2_only} Part 2-only baseline rows in this result table."
    )


def segments_for_capacity(capacity: np.ndarray, voltage: np.ndarray) -> list[dict[str, Any]]:
    valid = np.isfinite(capacity) & np.isfinite(voltage)
    capacity = capacity[valid]
    voltage = voltage[valid]
    boundaries = np.flatnonzero(np.diff(capacity) <= 0) + 1
    starts = np.r_[0, boundaries]
    ends = np.r_[boundaries, capacity.size]
    output = []
    for index, (start, end) in enumerate(zip(starts, ends), 1):
        segment_capacity = capacity[start:end]
        segment_voltage = voltage[start:end]
        if segment_capacity.size < 1:
            continue
        output.append(
            {
                "segment_index": index,
                "sample_count": int(segment_capacity.size),
                "capacity_min": float(segment_capacity.min()),
                "capacity_max": float(segment_capacity.max()),
                "voltage_min": float(segment_voltage.min()),
                "voltage_max": float(segment_voltage.max()),
                "valid": bool(
                    segment_capacity.size >= 10
                    and np.all(np.diff(segment_capacity) > 0)
                    and np.ptp(segment_capacity) > 0
                ),
            }
        )
    return output


def audit_step4(path: Path) -> dict[str, Any]:
    capacity_parts: list[np.ndarray] = []
    voltage_parts: list[np.ndarray] = []
    current_min = float("inf")
    current_max = float("-inf")
    positive = negative = zero = row_count = 0
    for chunk in pd.read_csv(path, usecols=["Step", "Amphr", "Amps", "Volts"], chunksize=250_000):
        selected = chunk.loc[chunk["Step"].eq(4), ["Amphr", "Amps", "Volts"]]
        if selected.empty:
            continue
        amps = pd.to_numeric(selected["Amps"], errors="coerce").to_numpy(dtype=np.float64)
        finite_amp = np.isfinite(amps)
        positive += int(np.sum(amps[finite_amp] > 0))
        negative += int(np.sum(amps[finite_amp] < 0))
        zero += int(np.sum(amps[finite_amp] == 0))
        if finite_amp.any():
            current_min = min(current_min, float(np.min(amps[finite_amp])))
            current_max = max(current_max, float(np.max(amps[finite_amp])))
        step_discharge = selected.loc[selected["Amps"].lt(0), ["Amphr", "Volts"]]
        if not step_discharge.empty:
            capacity_parts.append(step_discharge["Amphr"].to_numpy(dtype=np.float64))
            voltage_parts.append(step_discharge["Volts"].to_numpy(dtype=np.float64))
        row_count += len(selected)
    if not row_count:
        return {
            "status": "no_step4_rows",
            "sample_count": 0,
            "negative_current_rows": 0,
            "positive_current_rows": 0,
            "current_min_a": "",
            "current_max_a": "",
            "selected_segment": "",
            "capacity_span_native": "",
            "voltage_min_v": "",
            "voltage_max_v": "",
        }
    if not capacity_parts:
        return {
            "status": "step4_rows_without_negative_current",
            "sample_count": row_count,
            "negative_current_rows": negative,
            "positive_current_rows": positive,
            "current_min_a": current_min if np.isfinite(current_min) else "",
            "current_max_a": current_max if np.isfinite(current_max) else "",
            "selected_segment": "",
            "capacity_span_native": "",
            "voltage_min_v": "",
            "voltage_max_v": "",
        }
    segments = segments_for_capacity(
        np.concatenate(capacity_parts),
        np.concatenate(voltage_parts),
    )
    valid_segments = [segment for segment in segments if segment["valid"]]
    selected = max(valid_segments, key=lambda segment: segment["sample_count"], default=None)
    return {
        "status": "selected_monotone_segment" if selected else "no_valid_monotone_segment",
        "segment_count": len(segments),
        "valid_segment_count": len(valid_segments),
        "sample_count": row_count,
        "negative_current_rows": negative,
        "positive_current_rows": positive,
        "current_min_a": current_min if np.isfinite(current_min) else "",
        "current_max_a": current_max if np.isfinite(current_max) else "",
        "selected_segment": selected["segment_index"] if selected else "",
        "capacity_span_native": (
            selected["capacity_max"] - selected["capacity_min"] if selected else ""
        ),
        "voltage_min_v": selected["voltage_min"] if selected else "",
        "voltage_max_v": selected["voltage_max"] if selected else "",
    }


def observation_label(row: dict[str, Any] | None) -> str:
    if row is None:
        return ""
    return f"P{row['part']}/RPT{row['rpt_number']} ({row.get('source_tpg', '')})"


def latest_by_part_rpt(rows: list[dict[str, Any]], part_key: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (part_rank(str(row.get(part_key, ""))), int(row.get("rpt_number") or -1)),
    )


def make_eol_audit(
    raw_mapping: list[dict[str, str]],
    observations: list[dict[str, str]],
    guide_index: list[dict[str, str]],
    fits: list[dict[str, Any]],
    p12_all: list[dict[str, str]],
    p3_all: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], str, str]:
    def procedure_key(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.casefold())

    eligible_observations = []
    for row in observations:
        part = part_rank(row.get("part", ""))
        group = group_number(row.get("group", ""))
        if (part in {1, 2} and group in {"1", "2", "3", "4"}) or (
            part == 3 and group in {"7", "8", "9", "10"}
        ):
            eligible_observations.append(row)
    cells: dict[str, dict[str, Any]] = {}
    for row in eligible_observations:
        cell = row["physical_cell_id"]
        part = part_rank(row["part"])
        scope = "Parts 1+2" if part in {1, 2} else "Part 3"
        cells.setdefault(
            cell,
            {"physical_cell_id": cell, "group": group_number(row["group"]), "part_scope": scope},
        )
    if len(cells) != 24:
        raise ValueError(f"Expected 24 included physical cells, found {len(cells)}.")

    raw_by_cell_part: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in raw_mapping:
        if (
            row.get("source_modality") == "cycler_procedure"
            and row.get("source_extension", "").casefold() == ".csv"
            and row.get("source_status") == "present"
            and row.get("source_present", "").casefold() == "true"
            and row.get("physical_cell_id") in cells
        ):
            raw_by_cell_part[(row["physical_cell_id"], row["dataset_part"])].append(row)

    guide_rpts: dict[str, list[dict[str, Any]]] = defaultdict(list)
    guide_eol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in guide_index:
        cell = f"G{group_number(row.get('group', ''))}-Cell{cell_number(row.get('cell', ''))}"
        if cell not in cells:
            continue
        part = part_rank(row.get("part", ""))
        scope_parts = {"1", "2"} if cells[cell]["part_scope"] == "Parts 1+2" else {"3"}
        if str(part) not in scope_parts:
            continue
        stage = row.get("guide_tag", "")
        index = rpt_number(stage)
        if index is not None:
            guide_rpts[cell].append(
                {
                    "part": str(part),
                    "rpt_number": index,
                    "source_tpg": row.get("procedure", ""),
                    "guide_stage": stage,
                }
            )
        procedure = row.get("procedure", "")
        if stage.casefold() == "eol" or re.fullmatch(r"tpg\d+eol", procedure_key(procedure)):
            guide_eol[cell].append(row)

    eol_eis: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in raw_mapping:
        if (
            row.get("source_modality") == "eis_checkpoint"
            and row.get("source_filename_stage", "").casefold() == "eol"
            and row.get("source_extension", "").casefold() == ".csv"
            and row.get("source_status") == "present"
            and row.get("source_present", "").casefold() == "true"
            and row.get("physical_cell_id") in cells
        ):
            expected_part = "2" if cells[row["physical_cell_id"]]["part_scope"] == "Parts 1+2" else "3"
            if row.get("dataset_part") == expected_part:
                eol_eis[row["physical_cell_id"]].append(row)

    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_rows(PREDICTIONS):
        if not row.get("fusion_v2_predicted_soh_percent"):
            continue
        cell = row.get("physical_cell_id", "")
        if cell not in cells:
            continue
        predictions[cell].append(
            {
                **row,
                "part": row.get("dataset_part", row.get("part", "")),
                "rpt_number": rpt_number(row.get("measurement_id", "")) or int(row.get("rpt_index") or 0),
            }
        )
    for row in read_rows(SUPPLEMENTAL_PREDICTIONS):
        if not row.get("fusion_v2_predicted_soh_percent"):
            continue
        cell = row.get("physical_cell_id", "")
        if cell not in cells:
            continue
        predictions[cell].append(
            {
                **row,
                "part": row.get("dataset_part", ""),
                "rpt_number": rpt_number(row.get("measurement_id", "")) or int(row.get("rpt_index") or 0),
                "prediction_source": row.get("prediction_source", "saved_checkpoints_inference_only"),
            }
        )

    all_pydma = p12_all + p3_all
    pydma_all_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pydma_accepted_by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_pydma:
        group = group_number(row.get("group_id", ""))
        cell = f"G{group}-Cell{cell_number(row.get('cell_id', ''))}"
        if cell not in cells:
            continue
        source_part = row.get("source_part", "Part 3")
        pydma_row = {
            **row,
            "physical_cell_id": cell,
            "part": source_part,
            "rpt_number": rpt_number(row.get("checkup", "")),
        }
        if pydma_row["rpt_number"] is not None:
            pydma_all_by_cell[cell].append(pydma_row)
            if row.get("diagnostic_label_status") == "diagnostic_estimate_accepted":
                pydma_accepted_by_cell[cell].append(pydma_row)

    eol_curve_results: dict[str, dict[str, Any]] = {}
    for cell, guide_rows in guide_eol.items():
        candidates = []
        for guide in guide_rows:
            part = str(guide.get("part"))
            group = group_number(guide.get("group", ""))
            procedure = guide.get("procedure", "")
            candidates.extend(
                item
                for item in raw_by_cell_part.get((cell, part), [])
                if procedure_key(item.get("procedure", "")) == procedure_key(procedure)
                and group_number(item.get("group_id", "")) == group
            )
        unique = {item["repository_path"]: item for item in candidates}
        csv_candidates = sorted(unique.values(), key=lambda item: item["repository_path"])
        guide_row = max(guide_rows, key=lambda row: part_rank(row.get("part", "")))
        guide_stage = guide_row.get("guide_tag", "")
        if not csv_candidates:
            eol_curve_results[cell] = {
                "procedure": guide_row.get("procedure", ""),
                "guide_procedure": guide_row.get("procedure", ""),
                "guide_stage": guide_stage,
                "guide_rpt_number": rpt_number(guide_stage),
                "guide_index_source_file_present": guide_row.get("source_file_present", ""),
                "status": "source_not_mapped",
            }
            continue
        source = csv_candidates[0]
        path = ROOT / source["repository_path"]
        try:
            step4 = audit_step4(path)
            step8_curve, step8_audit = _read_pocv_with_audit(path, 8, "discharge")
            step10_curve, step10_audit = _read_pocv_with_audit(path, 10, "charge")

            def curve_span(curve: tuple[np.ndarray, np.ndarray] | None) -> float | str:
                return float(np.ptp(curve[0])) if curve is not None else ""

            eol_curve_results[cell] = {
                "procedure": source.get("procedure", ""),
                "guide_procedure": guide_row.get("procedure", ""),
                "path": source.get("repository_path", ""),
                "guide_stage": guide_stage,
                "guide_rpt_number": rpt_number(guide_stage),
                "guide_index_source_file_present": guide_row.get("source_file_present", ""),
                "raw_mapping_source_present": source.get("source_present", ""),
                "status": "audited",
                "step4": step4,
                "step8_status": step8_audit.get("status", ""),
                "step8_segment_count": len(step8_audit.get("segments", [])),
                "step8_valid_segment_count": sum(
                    bool(segment.get("valid")) for segment in step8_audit.get("segments", [])
                ),
                "step8_selected_segment": step8_audit.get("selected_segment", ""),
                "step8_sample_count": (
                    next(
                        (
                            segment.get("sample_count", "")
                            for segment in step8_audit.get("segments", [])
                            if segment.get("segment_index") == step8_audit.get("selected_segment")
                        ),
                        "",
                    )
                ),
                "step8_capacity_span_native": curve_span(step8_curve),
                "step10_status": step10_audit.get("status", ""),
                "step10_segment_count": len(step10_audit.get("segments", [])),
                "step10_valid_segment_count": sum(
                    bool(segment.get("valid")) for segment in step10_audit.get("segments", [])
                ),
                "step10_selected_segment": step10_audit.get("selected_segment", ""),
                "step10_sample_count": (
                    next(
                        (
                            segment.get("sample_count", "")
                            for segment in step10_audit.get("segments", [])
                            if segment.get("segment_index") == step10_audit.get("selected_segment")
                        ),
                        "",
                    )
                ),
                "step10_capacity_span_native": curve_span(step10_curve),
            }
        except Exception as error:
            eol_curve_results[cell] = {
                "procedure": source.get("procedure", ""),
                "guide_procedure": guide_row.get("procedure", ""),
                "path": source.get("repository_path", ""),
                "guide_stage": guide_stage,
                "guide_rpt_number": rpt_number(guide_stage),
                "guide_index_source_file_present": guide_row.get("source_file_present", ""),
                "raw_mapping_source_present": source.get("source_present", ""),
                "status": f"ERROR {type(error).__name__}: {error}",
            }

    rows_out: list[dict[str, Any]] = []
    for cell, metadata in sorted(cells.items()):
        part_scope = metadata["part_scope"]
        raw_parts = ["1", "2"] if part_scope == "Parts 1+2" else ["3"]
        raw_by_part: dict[str, dict[str, str] | None] = {}
        for part in raw_parts:
            candidates = raw_by_cell_part.get((cell, part), [])
            latest = max(
                candidates,
                key=lambda row: (int(row.get("procedure_file_id") or -1), row.get("repository_path", "")),
                default=None,
            )
            raw_by_part[part] = latest
        last_raw_part = "2" if part_scope == "Parts 1+2" and raw_by_part.get("2") else (
            "1" if part_scope == "Parts 1+2" else "3"
        )
        last_raw = raw_by_part.get(last_raw_part)

        guide_latest = latest_by_part_rpt(guide_rpts.get(cell, []), "part")
        cell_observations = [row for row in eligible_observations if row["physical_cell_id"] == cell]
        mapped_rows = [
            {
                **row,
                "part": row.get("part", ""),
                "rpt_number": int(row.get("rpt_index") or 0),
            }
            for row in cell_observations
        ]
        mapped_latest = latest_by_part_rpt(mapped_rows, "part")
        valid_step4 = [
            row for row in mapped_rows if row.get("step4_status") == "verified"
        ]
        valid_pocv = [
            row
            for row in mapped_rows
            if row.get("step8_status") == "verified" or row.get("step10_status") == "verified"
        ]
        step4_latest = latest_by_part_rpt(valid_step4, "part")
        pocv_latest = latest_by_part_rpt(valid_pocv, "part")
        eol_curve = eol_curve_results.get(cell, {})
        eol_rpt_number = eol_curve.get("guide_rpt_number")
        if eol_rpt_number is not None and eol_curve.get("status") == "audited":
            eol_rpt_row = {
                "part": "3" if part_scope == "Part 3" else (
                    "2" if eol_curve.get("procedure", "").startswith("TPG") else "1"
                ),
                "rpt_number": eol_rpt_number,
                "source_tpg": eol_curve.get("procedure", ""),
                "source_file": eol_curve.get("path", ""),
                "mapping_status": "Guide-mapped EoL procedure; canonical RPT manifest omission under audit",
                "step4_status": eol_curve.get("step4", {}).get("status", ""),
                "step8_status": eol_curve.get("step8_status", ""),
                "step10_status": eol_curve.get("step10_status", ""),
            }
            if eol_rpt_row["step4_status"] == "selected_monotone_segment":
                step4_latest = latest_by_part_rpt([*valid_step4, eol_rpt_row], "part")
            if (
                eol_rpt_row["step8_status"] == "selected_monotone_segment"
                or eol_rpt_row["step10_status"] == "selected_monotone_segment"
            ):
                pocv_latest = latest_by_part_rpt([*valid_pocv, eol_rpt_row], "part")
        pred_latest = latest_by_part_rpt(predictions.get(cell, []), "part")
        pydma_latest = latest_by_part_rpt(pydma_accepted_by_cell.get(cell, []), "part")
        all_pydma_rows = pydma_all_by_cell.get(cell, [])
        latest_pydma_candidate = latest_by_part_rpt(all_pydma_rows, "part")
        latest_candidate_rows = (
            [
                row
                for row in all_pydma_rows
                if row.get("source_part", "Part 3") == latest_pydma_candidate.get("part")
                and row.get("checkup") == latest_pydma_candidate.get("checkup")
            ]
            if latest_pydma_candidate
            else []
        )
        eol_files = eol_eis.get(cell, [])
        soc_values = sorted(
            {
                match.group(1)
                for item in eol_files
                if (match := re.search(r"SOC(\d+)", item.get("source_relative_path", ""), re.I))
            },
            key=int,
        )
        latest_mapped_details = {
            "mapping_status": mapped_latest.get("mapping_status", "") if mapped_latest else "",
            "step4_status": mapped_latest.get("step4_status", "") if mapped_latest else "",
            "step8_status": mapped_latest.get("step8_status", "") if mapped_latest else "",
            "step10_status": mapped_latest.get("step10_status", "") if mapped_latest else "",
        }
        missing = []
        if guide_latest and mapped_latest and (
            part_rank(guide_latest["part"]), guide_latest["rpt_number"]
        ) > (part_rank(mapped_latest["part"]), mapped_latest["rpt_number"]):
            missing.append("last Guide RPT is later than the last canonical mapped RPT")
        if mapped_latest and not pred_latest:
            missing.append(
                "no original three-encoder prediction row for the last mapped RPT; "
                f"recorded statuses={latest_mapped_details}"
            )
        elif mapped_latest and pred_latest and (
            part_rank(pred_latest["part"]), pred_latest["rpt_number"]
        ) < (part_rank(mapped_latest["part"]), mapped_latest["rpt_number"]):
            missing.append(
                "no original three-encoder prediction at the last mapped RPT; "
                f"recorded statuses={latest_mapped_details}"
            )
        if guide_latest and pred_latest and (
            part_rank(guide_latest["part"]), guide_latest["rpt_number"]
        ) > (part_rank(pred_latest["part"]), pred_latest["rpt_number"]):
            missing.append(
                f"the Guide's later {observation_label(guide_latest)} has no saved original "
                "three-encoder prediction"
            )
        if guide_eol.get(cell) and eol_curve.get("guide_rpt_number") is not None:
            missing.append(
                f"Guide maps {eol_curve.get('procedure', '')} to RPT"
                f"{eol_curve['guide_rpt_number']}, but the canonical RPT manifest has no row "
                "for this present raw procedure"
            )
        elif guide_eol.get(cell):
            missing.append(
                f"Guide marks {eol_curve.get('procedure', '')} as EoL without a numbered RPT; "
                "it is not promoted into the RPT sequence"
            )
        else:
            missing.append(
                "no Guide-tagged cycling EoL TPG for this cell; EoL EIS is a separate checkpoint"
            )
        if pydma_latest and latest_pydma_candidate and (
            part_rank(pydma_latest["part"]), pydma_latest["rpt_number"]
        ) < (part_rank(latest_pydma_candidate["part"]), latest_pydma_candidate["rpt_number"]):
            missing.append("a later PyDMA candidate exists but did not produce an accepted diagnostic fit")
        elif mapped_latest and (
            not pydma_latest
            or (
                part_rank(pydma_latest["part"]), pydma_latest["rpt_number"]
            )
            < (part_rank(mapped_latest["part"]), mapped_latest["rpt_number"])
        ):
            missing.append("no accepted PyDMA fit is recorded for the last mapped RPT")
        if guide_latest and pydma_latest and (
            part_rank(guide_latest["part"]), guide_latest["rpt_number"]
        ) > (
            part_rank(pydma_latest["part"]),
            pydma_latest["rpt_number"],
        ):
            missing.append(
                f"no accepted PyDMA fit is recorded for the Guide's later "
                f"{observation_label(guide_latest)}"
            )

        if guide_eol.get(cell) and eol_curve.get("guide_rpt_number") is not None:
            action = (
                "Add the already Guide-assigned RPT number to a versioned manifest mapping using "
                "the present TPGx.EoL CSV; retain its EoL procedure provenance and verified curve "
                "segments. Do not infer the file number from the filename."
            )
        elif guide_eol.get(cell):
            action = (
                "Keep it separate from numbered RPTs. If product owners want EoL display, add a "
                "Guide-keyed EoL record after protocol and modality validation; do not backfill "
                "or infer an RPT number."
            )
        else:
            action = (
                "Keep EoL EIS availability explicit. Do not treat it as a Step 4 capacity RPT or "
                "a pseudo-OCV PyDMA curve; do not assume the last raw TPG is EoL."
            )

        def display_rpt(row: dict[str, Any] | None) -> str:
            return observation_label(row)

        rows_out.append(
            {
                "part_scope": part_scope,
                "group": metadata["group"],
                "physical_cell_id": cell,
                "last_raw_tpg_part1": (
                    raw_by_part["1"].get("procedure", "") if raw_by_part.get("1") else ""
                ),
                "last_raw_tpg_part1_path": (
                    raw_by_part["1"].get("repository_path", "") if raw_by_part.get("1") else ""
                ),
                "last_raw_tpg_part2": (
                    raw_by_part["2"].get("procedure", "") if raw_by_part.get("2") else ""
                ),
                "last_raw_tpg_part2_path": (
                    raw_by_part["2"].get("repository_path", "") if raw_by_part.get("2") else ""
                ),
                "last_raw_tpg_part3": (
                    raw_by_part["3"].get("procedure", "") if raw_by_part.get("3") else ""
                ),
                "last_raw_tpg_part3_path": (
                    raw_by_part["3"].get("repository_path", "") if raw_by_part.get("3") else ""
                ),
                "last_raw_tpg_overall": last_raw.get("procedure", "") if last_raw else "",
                "last_raw_tpg_overall_path": last_raw.get("repository_path", "") if last_raw else "",
                "last_guide_eol_or_eol_rpt_tpg": eol_curve.get("procedure", ""),
                "last_guide_eol_or_eol_rpt_guide_stage": eol_curve.get("guide_stage", ""),
                "guide_eol_source_file_present_index": eol_curve.get(
                    "guide_index_source_file_present", ""
                ),
                "guide_eol_raw_mapping_source_present": eol_curve.get(
                    "raw_mapping_source_present", ""
                ),
                "last_guide_rpt": display_rpt(guide_latest),
                "last_guide_rpt_guide_stage": guide_latest.get("guide_stage", "") if guide_latest else "",
                "last_mapped_rpt": display_rpt(mapped_latest),
                "last_mapped_rpt_mapping_status": (
                    mapped_latest.get("mapping_status", "") if mapped_latest else ""
                ),
                "last_valid_step4_discharge_rpt": display_rpt(step4_latest),
                "last_valid_step4_source": step4_latest.get("source_file", "") if step4_latest else "",
                "last_valid_step8_or_step10_rpt": display_rpt(pocv_latest),
                "last_valid_step8_status": pocv_latest.get("step8_status", "") if pocv_latest else "",
                "last_valid_step10_status": pocv_latest.get("step10_status", "") if pocv_latest else "",
                "last_valid_step8_or_step10_source": (
                    pocv_latest.get("source_file", "") if pocv_latest else ""
                ),
                "last_soh_prediction_rpt": (
                    f"P{pred_latest['part']}/RPT{pred_latest['rpt_number']}"
                    if pred_latest
                    else ""
                ),
                "last_soh_prediction_percent": (
                    pred_latest.get("fusion_v2_predicted_soh_percent", "") if pred_latest else ""
                ),
                "last_soh_prediction_source": (
                    pred_latest.get("prediction_source", "saved_phase2_checkpoint")
                    if pred_latest
                    else ""
                ),
                "last_soh_prediction_label_scope": (
                    pred_latest.get("label_scope", "") if pred_latest else ""
                ),
                "last_pydma_accepted_rpt": (
                    f"{pydma_latest['part']}/{pydma_latest['checkup']}" if pydma_latest else ""
                ),
                "last_pydma_accepted_directions": (
                    ";".join(
                        sorted(
                            {
                                row.get("direction", "")
                                for row in pydma_accepted_by_cell.get(cell, [])
                                if row.get("checkup") == pydma_latest.get("checkup")
                                and row.get("source_part", "Part 3") == pydma_latest.get("part")
                            }
                        )
                    )
                    if pydma_latest
                    else ""
                ),
                "last_pydma_candidate_status": (
                    ";".join(
                        f"{row.get('direction', '')}:{row.get('diagnostic_label_status', '')}"
                        for row in sorted(latest_candidate_rows, key=lambda item: item.get("direction", ""))
                    )
                ),
                "eol_eis_csv_count": len({row.get("source_relative_path", "") for row in eol_files}),
                "eol_eis_soc_percentages": ";".join(soc_values),
                "guide_eol_tpg": eol_curve.get("procedure", ""),
                "guide_eol_guide_procedure": eol_curve.get("guide_procedure", ""),
                "guide_eol_guide_stage": eol_curve.get("guide_stage", ""),
                "guide_eol_rpt_number": eol_curve.get("guide_rpt_number", ""),
                "guide_eol_tpg_path": eol_curve.get("path", ""),
                "guide_eol_step4_status": eol_curve.get("step4", {}).get("status", ""),
                "guide_eol_step4_samples": eol_curve.get("step4", {}).get("sample_count", ""),
                "guide_eol_step4_segment_count": eol_curve.get("step4", {}).get("segment_count", ""),
                "guide_eol_step4_valid_segment_count": eol_curve.get("step4", {}).get("valid_segment_count", ""),
                "guide_eol_step4_current_min_a": eol_curve.get("step4", {}).get("current_min_a", ""),
                "guide_eol_step4_current_max_a": eol_curve.get("step4", {}).get("current_max_a", ""),
                "guide_eol_step4_capacity_span_native": eol_curve.get("step4", {}).get("capacity_span_native", ""),
                "guide_eol_step8_status": eol_curve.get("step8_status", ""),
                "guide_eol_step8_samples": eol_curve.get("step8_sample_count", ""),
                "guide_eol_step8_segment_count": eol_curve.get("step8_segment_count", ""),
                "guide_eol_step8_valid_segment_count": eol_curve.get("step8_valid_segment_count", ""),
                "guide_eol_step8_capacity_span_native": eol_curve.get("step8_capacity_span_native", ""),
                "guide_eol_step10_status": eol_curve.get("step10_status", ""),
                "guide_eol_step10_samples": eol_curve.get("step10_sample_count", ""),
                "guide_eol_step10_segment_count": eol_curve.get("step10_segment_count", ""),
                "guide_eol_step10_valid_segment_count": eol_curve.get("step10_valid_segment_count", ""),
                "guide_eol_step10_capacity_span_native": eol_curve.get("step10_capacity_span_native", ""),
                "missing_reason": "; ".join(missing),
                "recommended_action": action,
            }
        )

    summary_rows = []
    for scope in ("Parts 1+2", "Part 3"):
        subset = [row for row in rows_out if row["part_scope"] == scope]
        summary_rows.append(
            [
                scope,
                len(subset),
                sum(bool(row["last_guide_rpt"]) for row in subset),
                sum(bool(row["last_mapped_rpt"]) for row in subset),
                sum(bool(row["last_valid_step4_discharge_rpt"]) for row in subset),
                sum(bool(row["last_valid_step8_or_step10_rpt"]) for row in subset),
                sum(bool(row["last_soh_prediction_rpt"]) for row in subset),
                sum(bool(row["last_pydma_accepted_rpt"]) for row in subset),
                sum(row["eol_eis_csv_count"] == 3 for row in subset),
                sum(bool(row["guide_eol_tpg"]) for row in subset),
            ]
        )
    numbered_guide_eol_count = sum(row["guide_eol_rpt_number"] != "" for row in rows_out)
    guide_presence_false_negatives = sum(
        str(row["guide_eol_source_file_present_index"]).casefold() == "false"
        and str(row["guide_eol_raw_mapping_source_present"]).casefold() == "true"
        for row in rows_out
    )
    guide_procedure_format_differences = sum(
        bool(row["guide_eol_guide_procedure"])
        and row["guide_eol_guide_procedure"] != row["guide_eol_tpg"]
        and procedure_key(row["guide_eol_guide_procedure"]) == procedure_key(row["guide_eol_tpg"])
        for row in rows_out
    )
    later_guide_mapping_gaps = sum(
        "last Guide RPT is later than the last canonical mapped RPT" in row["missing_reason"]
        for row in rows_out
    )
    report_lines = [
        "# EoL RPT Coverage Audit",
        "",
        "Scope: Part 1/2 Groups 1-4 as continued physical cells, plus Part 3 Groups 7-10. "
        "Part 2 Groups 5-6 are excluded. The audit uses the canonical RPT manifest, Guide file "
        "index, raw-file mapping, saved Phase 2 three-encoder predictions, supplemental G10-Cell2 "
        "predictions, and accepted PyDMA result statuses.",
        "",
        "## Coverage summary",
        "",
        markdown_table(
            [
                "Part scope",
                "Cells",
                "Guide RPT mapped",
                "Canonical RPT mapped",
                "Valid Step 4 RPT",
                "Valid Step 8/10 RPT",
                "Saved SoH predictions",
                "Accepted PyDMA",
                "Cells with 3 EoL EIS CSVs",
                "Guide EoL TPG",
            ],
            summary_rows,
        ),
        "",
        f"The per-cell table is `{OUTPUT.name}/eol_rpt_coverage_audit.csv`. Every included cell has "
        f"{sum(row['eol_eis_csv_count'] == 3 for row in rows_out)}/{len(rows_out)} complete EoL EIS "
        "checkpoint triplets in the raw mapping. EIS EoL is not evidence of an RPT capacity curve. "
        f"The audit found {numbered_guide_eol_count} Guide-numbered `TPGx.EoL` checkpoints; "
        f"{later_guide_mapping_gaps} are later than the canonical mapped RPT. The Guide index has "
        f"{guide_presence_false_negatives} false-negative source-presence flags for files the raw "
        "mapping confirms are present. "
        f"{guide_procedure_format_differences} Guide procedure tokens differ in punctuation or spacing "
        "from the raw-mapping filename but normalize to the same procedure identifier.",
        "",
        "## Cell-by-cell gaps",
        "",
        markdown_table(
            [
                "Scope",
                "Cell",
                "Highest-numbered raw TPG",
                "Last Guide RPT",
                "Last mapped RPT",
                "Last Step 4 RPT",
                "Last Step 8/10 RPT",
                "Last prediction",
                "Last accepted PyDMA",
                "EoL evidence",
                "Missing reason",
            ],
            [
                [
                    row["part_scope"],
                    row["physical_cell_id"],
                    row["last_raw_tpg_overall"],
                    row["last_guide_rpt"],
                    row["last_mapped_rpt"],
                    row["last_valid_step4_discharge_rpt"],
                    row["last_valid_step8_or_step10_rpt"],
                    row["last_soh_prediction_rpt"],
                    row["last_pydma_accepted_rpt"],
                    (
                        f"EIS {row['eol_eis_csv_count']} SOCs {row['eol_eis_soc_percentages']}; "
                        f"Guide TPG {row['guide_eol_tpg']} ({row['guide_eol_guide_stage']})"
                    ),
                    row["missing_reason"],
                ]
                for row in rows_out
            ],
        ),
        "",
        "## Verified Guide-tagged EoL TPGs",
        "",
        "Part 2's five Guide EoL procedures are unnumbered EoL stages. Part 3's nine `TPGx.EoL` "
        "procedures are explicitly assigned Guide RPT6/RPT7 stages. Step 4 was checked using all "
        "Step 4 samples and discharge-current direction; Steps 8/10 used the existing "
        "direction-filtered pseudo-OCV segment parser. `Amphr` spans are native units, not labeled Ah.",
        "",
    ]
    eol_rows = [
        [
            row["physical_cell_id"],
            row["part_scope"],
            row["guide_eol_tpg"],
            row["guide_eol_guide_procedure"],
            row["guide_eol_guide_stage"],
            row["guide_eol_source_file_present_index"],
            row["guide_eol_raw_mapping_source_present"],
            row["guide_eol_step4_status"],
            row["guide_eol_step4_samples"],
            row["guide_eol_step4_segment_count"],
            row["guide_eol_step4_capacity_span_native"],
            row["guide_eol_step8_status"],
            row["guide_eol_step8_segment_count"],
            row["guide_eol_step8_capacity_span_native"],
            row["guide_eol_step10_status"],
            row["guide_eol_step10_segment_count"],
            row["guide_eol_step10_capacity_span_native"],
        ]
        for row in rows_out
        if row["guide_eol_tpg"]
    ]
    report_lines.extend(
        [
            markdown_table(
                [
                    "Cell",
                    "Part scope",
                    "Guide EoL TPG",
                    "Guide procedure token",
                    "Guide stage",
                    "Index says source present",
                    "Raw mapping says present",
                    "Step 4 status",
                    "Step 4 rows",
                    "Step 4 segments",
                    "Step 4 span",
                    "Step 8 status",
                    "Step 8 segments",
                    "Step 8 span",
                    "Step 10 status",
                    "Step 10 segments",
                    "Step 10 span",
                ],
                eol_rows,
            ),
            "",
            "The original Part 3 Guide workbook (`Guide_to_Datafiles_3.xlsx`, Sheet1) maps "
            "`TPG7.EoL` to RPT6 for G7-Cell27 and to RPT7 for G7-Cell19/G7-Cell4, and maps "
            "`TPG8.EoL`/`TPG9.EoL` to RPT7 for the six Group 8/9 cells. All nine CSVs are present "
            "in `unified_raw_mapping.csv`, although `guide_file_index.csv` marks them absent. Their "
            "Step 4, Step 8, and Step 10 curves pass the existing monotone-segment selector. The "
            "canonical RPT manifest omits these Guide-assigned RPTs. The separate Part 2 EoL files "
            "remain unnumbered and must not be assigned RPT numbers.",
            "",
            "A structurally selected monotone segment is not automatically a complete or "
            "capacity-validated observation. Segment counts and native spans are retained; short "
            "Step 10 spans are reported rather than promoted to usable curves.",
            "",
            "## Source and reproducibility notes",
            "",
            "- The canonical manifest distinguishes `verified`, `needs_curve_audit`, and missing steps; "
            "only `verified` is counted as a valid Step 4 or pseudo-OCV RPT.",
            "- The saved Phase 2 prediction file is `fusion_predictions.csv`; its supplemental "
            "G10-Cell2 rows are read separately and retain their provisional label scope and saved-"
            "checkpoint inference provenance.",
            "- EoL EIS CSVs are reported independently from cycling TPG/RPT records.",
            "- Run from repository root with "
            "`\\.venv-torch-cuda\\Scripts\\python.exe experiments\\results\\data_audits\\"
            "pydma_eol_audit_v1\\run_audit.py`.",
            "",
        ]
    )

    plan_lines = [
        "# Dashboard EoL Fix Plan",
        "",
        "No dashboard code or behavior was changed during this read-only audit.",
        "",
        "## Evidence",
        "",
        "- EoL EIS exists for the included cells as separate SOC checkpoint files; EIS does not "
        "supply Step 4 discharge capacity or Step 8/10 pseudo-OCV.",
        "- Part 2 has five unnumbered Guide EoL TPG procedures with mixed Step 4/8/10 availability.",
        "- Part 3 has nine `TPGx.EoL` CSVs explicitly mapped by the original Guide to RPT6/RPT7. "
        "The raw files are present and have selected Step 4/8/10 curves, but the canonical RPT "
        "manifest and current three-encoder/PyDMA outputs stop earlier for those cells.",
        "- The `guide_file_index.csv` false-negative presence flags and numeric-only procedure "
        "crosswalk explain the exclusion: `TPGx.EoL` has no numeric file ID, even though its Guide "
        "stage has a numbered RPT identity. Do not “fix” this by assigning a guessed TPG file number.",
        "",
        "## Proposed correction if EoL display is approved",
        "",
        "1. For the nine Part 3 files, add the Guide-assigned RPT6/RPT7 mapping from the original "
        "Guide workbook to a versioned manifest. Preserve `procedure=TPGx.EoL`; keep filename file "
        "ID null rather than coercing it to a number.",
        "2. For the five Part 2 unnumbered EoL procedures, use a separate Guide-keyed "
        "`observation_kind=eol` record and do not invent an RPT number.",
        "3. Extract Step 4, Step 8, and Step 10 independently with their verified current directions "
        "and curve boundaries. Require explicit completeness/quality criteria before a curve is "
        "considered usable; short monotone segments must remain partial.",
        "4. Display the saved Phase 2 SoH prediction only when the existing D+C+P input schema is "
        "satisfied. Do not fabricate missing modalities, labels, or predictions. Clearly mark any "
        "prediction as an EoL inference, not a validated EoL metric.",
        "5. Display PyDMA degradation only for a valid pseudo-OCV curve with an accepted same-direction "
        "BoL fit, retaining RMSE, accepted-run count, and diagnostic-only status.",
        "6. Keep EoL EIS availability as its own display item and never use EIS-only records as "
        "capacity-RPT or PyDMA observations.",
        "7. Preserve current Phase 2/Phase 3 metrics and checkpoints; add regression tests for explicit "
        "Guide EoL mapping, incomplete Step 4/8/10 handling, and no inferred RPT identity.",
        "",
        "The audit supports a targeted correction for the nine explicitly Guide-numbered Part 3 "
        "EoL RPTs and a separate representation for Part 2's unnumbered EoL procedures. No dashboard "
        "or canonical mapping change is included here.",
        "",
    ]
    return rows_out, "\n".join(report_lines), "\n".join(plan_lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-multistart", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    raw_mapping = read_rows(RAW_MAPPING)
    observations = read_rows(OBSERVATIONS)
    guide_index = read_rows(GUIDE_INDEX)
    fits, p12_all, p3_all = source_fit_rows()
    strict10_rows = read_rows(P2_RMSE10_RESULTS)
    part3_rows = [row for row in fits if row["part"] == "Part 3"]
    use_cuda = False
    try:
        import torch

        use_cuda = bool(torch.cuda.is_available())
    except ImportError:
        use_cuda = False
    multi_rows = [] if args.skip_multistart else fit_stability(
        fits,
        p12_all,
        p3_all,
        raw_mapping,
        observations,
        seeds=list(range(211, 216)),
        use_cuda=use_cuda,
    )

    fit_fields = [
        "part",
        "group",
        "physical_cell_id",
        "rpt",
        "rpt_number",
        "direction",
        "test_step",
        "source_file",
        "curve_source_path_recorded",
        "curve_status",
        "selected_segment",
        "curve_sample_count",
        "capacity_min_native",
        "capacity_max_native",
        "capacity_span_native",
        "fit_rmse_mV",
        "rmse_threshold_mV",
        "accepted_runs",
        "total_runs",
        "diagnostic_label_status",
        "source_run",
        "LLI_percent_vs_BoL",
        "LAM_anode_percent_vs_BoL",
        "LAM_cathode_percent_vs_BoL",
        "negative_lli",
        "negative_lam_anode",
        "negative_lam_cathode",
    ]
    write_rows(OUTPUT / "pydma_negative_estimate_audit.csv", fits, fit_fields)
    multi_fields = [
        "case_id",
        "physical_cell_id",
        "group",
        "part",
        "rpt",
        "direction",
        "seed",
        "stored_lli_percent",
        "stored_lam_anode_percent",
        "stored_lam_cathode_percent",
        "baseline_source",
        "target_source",
        "cuda_used",
        "baseline_fit_rmse_mV",
        "baseline_accepted_runs",
        "baseline_total_runs",
        "baseline_gate_passed",
        "baseline_parameters_json",
        "target_fit_rmse_mV",
        "target_accepted_runs",
        "target_total_runs",
        "target_gate_passed",
        "target_parameters_json",
        "estimate_status",
        "lli_percent",
        "lam_anode_percent",
        "lam_cathode_percent",
    ]
    write_rows(OUTPUT / "pydma_fit_stability.csv", multi_rows, multi_fields)
    (OUTPUT / "pydma_fit_stability_report.md").write_text(
        make_negative_report(
            fits,
            strict10_rows,
            multi_rows,
            use_cuda,
            baseline_consistency_summary(p12_all),
        ),
        encoding="utf-8",
    )

    eol_rows, eol_report, dashboard_plan = make_eol_audit(
        raw_mapping,
        observations,
        guide_index,
        fits,
        p12_all,
        p3_all,
    )
    eol_fields = list(eol_rows[0]) if eol_rows else []
    write_rows(OUTPUT / "eol_rpt_coverage_audit.csv", eol_rows, eol_fields)
    (OUTPUT / "eol_missing_reasons.md").write_text(eol_report, encoding="utf-8")
    (OUTPUT / "dashboard_eol_fix_plan.md").write_text(dashboard_plan, encoding="utf-8")

    print(
        json.dumps(
            {
                "accepted_primary_fits": len(fits),
                "part1_part2_accepted": sum(row["part"] in {"Part 1", "Part 2"} for row in fits),
                "part3_accepted": len(part3_rows),
                "negative_lli": sum(bool(row["negative_lli"]) for row in fits),
                "negative_lam_anode": sum(bool(row["negative_lam_anode"]) for row in fits),
                "negative_lam_cathode": sum(bool(row["negative_lam_cathode"]) for row in fits),
                "multi_start_rows": len(multi_rows),
                "eol_cells": len(eol_rows),
                "eol_eis_triplets": sum(row["eol_eis_csv_count"] == 3 for row in eol_rows),
                "guide_eol_tpgs": sum(bool(row["guide_eol_tpg"]) for row in eol_rows),
                "output_directory": str(OUTPUT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
