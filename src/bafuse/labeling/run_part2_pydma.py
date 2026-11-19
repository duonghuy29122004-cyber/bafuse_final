from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from bafuse.labeling.pydma_runner import run_pydma_for_cell
from bafuse.labeling.pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
)
from bafuse.labeling.pydma_crosswalk import (
    cell_inputs as _cell_inputs,
    is_unique_checkpoint as _is_unique_checkpoint,
    manifest_path as _manifest_path,
    source_order as _source_order,
    write_csv as _write_csv,
)


_DIRECTIONS = ("charge", "discharge")
_SUPPORTED_GROUPS = {"1", "2", "3", "4", "6"}
_OPTIMIZER_OPTIONS = {"maxiter": 50, "popsize": 30, "max_tries": 10}


def _build_cell_plans(
    manifest_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    part1_baselines: dict[str, list[dict[str, str]]] = {}
    part2_baselines: dict[str, list[dict[str, str]]] = {}
    part2_targets: list[dict[str, str]] = []
    for row in manifest_rows:
        if not _is_unique_checkpoint(row):
            continue
        physical_cell_id = row["physical_cell_id"]
        if row["dataset_part"] == "1" and row["measurement_id"].casefold() == "bol":
            part1_baselines.setdefault(physical_cell_id, []).append(row)
        elif row["dataset_part"] == "2":
            if row["measurement_id"].casefold() == "bol":
                part2_baselines.setdefault(physical_cell_id, []).append(row)
            elif row["measurement_id"].casefold().startswith("rpt"):
                part2_targets.append(row)

    plans: dict[str, dict[str, Any]] = {}
    for target in part2_targets:
        group_id = target["group_id"]
        if group_id not in _SUPPORTED_GROUPS:
            continue
        physical_cell_id = target["physical_cell_id"]
        if group_id in {"1", "2", "3", "4"}:
            candidates = part1_baselines.get(physical_cell_id, [])
        else:
            candidates = part2_baselines.get(physical_cell_id, [])
        if not candidates:
            raise ValueError(
                f"No unique same-cell BoL source is available for "
                f"{physical_cell_id} Part 2 {target['measurement_id']}."
            )
        baseline = min(candidates, key=_source_order)
        plan = plans.setdefault(
            physical_cell_id,
            {
                "group_id": group_id,
                "cell_id": target["cell_id"],
                "baseline": baseline,
                "targets": [],
            },
        )
        if plan["group_id"] != group_id or plan["baseline"] is not baseline:
            raise ValueError(f"Conflicting cell or baseline mapping for {physical_cell_id}.")
        plan["targets"].append(target)

    for plan in plans.values():
        plan["targets"].sort(key=_source_order)
    return dict(sorted(plans.items()))


def run_part2(
    part1_root: Path,
    part2_root: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_gpu: str | None,
) -> dict[str, Any]:
    import torch

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("The Part 2 PyDMA experiment requires CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if expected_gpu and expected_gpu.casefold() not in device_name.casefold():
        raise RuntimeError(
            f"Expected GPU containing {expected_gpu!r}, found {device_name!r}."
        )

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    plans = _build_cell_plans(manifest_rows)
    roots = {"1": part1_root, "2": part2_root}
    fit_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for physical_cell_id, plan in plans.items():
        print(
            f"Running Part 2 PyDMA {physical_cell_id}: "
            f"{len(plan['targets'])} single-checkpoint RPTs",
            flush=True,
        )
        records, entries, fit_targets = _cell_inputs(plan, roots)
        fit_results.extend(
            run_pydma_for_cell(
                part2_root,
                records,
                entries,
                plan["group_id"],
                plan["cell_id"],
                use_cuda=True,
                fit_targets=fit_targets,
                optimizer_options=_OPTIMIZER_OPTIONS,
                allow_raw_comparison=True,
                half_cell_reference_root=part1_root,
            )
        )

    part2_rows = [
        row for row in manifest_rows
        if row.get("dataset_part") == "2"
        and row.get("measurement_id", "").casefold().startswith("rpt")
    ]
    selected_sources = {
        (
            target["group_id"],
            target["cell_id"],
            target["procedure_file_id"],
            target["source_file"],
        )
        for plan in plans.values()
        for target in plan["targets"]
    }
    crosswalk_audit = []
    for row in part2_rows:
        key = (
            row["group_id"],
            row["cell_id"],
            row["procedure_file_id"],
            row["source_file"],
        )
        if key in selected_sources:
            status = "selected_unique_single_checkpoint"
        elif row.get("measurement_mapping_status") != "single_checkpoint":
            status = "withheld_ambiguous_or_compound_checkpoint"
        elif row.get("source_mapping_status") != "unique_source_file":
            status = "withheld_non_unique_source_mapping"
        elif row.get("group_id") == "5":
            status = "withheld_group5_baseline_or_checkpoint_mapping_review"
        else:
            status = "withheld_unsupported_or_unmapped_checkpoint"
        crosswalk_audit.append({**row, "pydma_selection_status": status})

    status_counts = Counter(
        row["diagnostic_label_status"] for row in fit_results
    )
    numeric_label_count = sum(
        row["diagnostic_label_status"]
        in {"diagnostic_estimate_accepted", "raw_diagnostic_estimate"}
        for row in fit_results
    ) * 3
    summary = {
        "experiment": "Part 2 PyDMA LLI/LAM estimates with same-cell baselines",
        "dataset_scope": "Part 2 Groups 1-4 continuation and Group 6 single-checkpoint observations",
        "runtime": {
            "device": "cuda:0",
            "gpu_name": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_options": _OPTIMIZER_OPTIONS,
        },
        "baseline_policy": {
            "groups_1_to_4": "Earliest unique Guide-mapped Part 1 BoL source for the same physical cell.",
            "group_6": "Earliest unique Guide-mapped Part 2 BoL source for the same physical cell.",
            "group_5": "Withheld because the available Guide stages do not uniquely identify RPT checkpoints.",
            "half_cell_references": "Part 1 Half_Cell reference files.",
        },
        "fit_policy": {
            "target_curves": "Part 2 single-checkpoint Guide mappings with one unique CSV source; pOCV Step 8 discharge and Step 10 charge.",
            "rmse_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
            "minimum_accepted_runs": PYDMA_MIN_ACCEPTED_RUNS,
            "accepted_gate": (
                f"PyDMA accepted fit, RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV, "
                f"and at least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs."
            ),
            "raw_estimates": "Retained when allow_raw_comparison is enabled; not treated as accepted labels.",
        },
        "input_manifest": str(manifest_path),
        "part1_root": str(part1_root),
        "part2_root": str(part2_root),
        "cell_count": len(plans),
        "single_checkpoint_rpt_count": sum(
            len(plan["targets"]) for plan in plans.values()
        ),
        "directional_fit_result_count": len(fit_results),
        "degradation_value_count": numeric_label_count,
        "fit_status_counts": dict(sorted(status_counts.items())),
        "source_files_modified": False,
        "limitations": [
            "LLI/LAM values are PyDMA-derived diagnostic estimates, not independent physical ground truth.",
            "Part 2 Group 5 is excluded because its available Guide stages do not uniquely identify RPT checkpoints.",
            "Part 3 is excluded from this run; external-cohort labels and fits were not evaluated.",
            "Part 2 compound checkpoints remain withheld rather than assigned to individual RPTs.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "pydma_results.csv", fit_results)
    _write_csv(output_dir / "crosswalk_audit.csv", crosswalk_audit)
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    (output_dir / "README.md").write_text(
        "# Part 2 PyDMA degradation-label run\n\n"
        "This run uses same-cell Part 1 BoL references for Part 2 Groups 1-4 "
        "and a local Part 2 BoL for Group 6. Only unique single-checkpoint "
        "Guide mappings are fitted. Compound Group 5 checkpoints and Part 3 "
        "are withheld. Raw diagnostic estimates are retained separately from "
        "accepted estimates; neither is independent physical ground truth. "
        f"The accepted-fit gate is RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV with "
        f"at least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs.\n\n"
        "See `run_summary.json`, `pydma_results.csv`, and `crosswalk_audit.csv`.\n\n"
        "Reproduce from the repository root in PowerShell:\n\n"
        "```powershell\n"
        '$env:PYTHONPATH = "src"\n'
        "python -m bafuse.labeling.run_part2_pydma `\n"
        f'  --part1-root "{part1_root}" `\n'
        f'  --part2-root "{part2_root}" `\n'
        f'  --manifest "{manifest_path}" `\n'
        f'  --output "{output_dir}" `\n'
        f'  --expected-gpu "{expected_gpu or ""}"\n'
        "```\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate separate Part 2 PyDMA diagnostic labels."
    )
    parser.add_argument("--part1-root", type=Path, required=True)
    parser.add_argument("--part2-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", default=None)
    args = parser.parse_args()
    summary = run_part2(
        args.part1_root,
        args.part2_root,
        args.manifest,
        args.output,
        args.expected_gpu,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
