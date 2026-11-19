from __future__ import annotations

import argparse
import csv
import hashlib
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
    cell_inputs,
    is_unique_checkpoint,
    source_order,
    write_csv,
)


_DIRECTIONS = ("charge", "discharge")
_SUPPORTED_GROUPS = {"7", "8", "9", "10"}
_OPTIMIZER_OPTIONS = {"maxiter": 50, "popsize": 30, "max_tries": 10}
_PART3_BOL_SOURCE_RULE = "part3_unsuffixed_bol_to_zero_suffix"


def _row_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["group_id"],
        row["cell_id"],
        row["procedure_file_id"],
        row["source_file"],
    )


def _has_cell_identity(row: dict[str, str]) -> bool:
    return bool(row.get("group_id") and row.get("cell_id") and row.get("physical_cell_id"))


def _build_cell_plans(
    manifest_rows: list[dict[str, str]],
) -> dict[str, dict[str, Any]]:
    part3_rows = [row for row in manifest_rows if row.get("dataset_part") == "3"]
    baselines: dict[str, list[dict[str, str]]] = {}
    targets: list[dict[str, str]] = []
    for row in part3_rows:
        measurement_id = row.get("measurement_id", "").casefold()
        if measurement_id == "bol" and is_unique_checkpoint(row):
            if not _has_cell_identity(row):
                continue
            baselines.setdefault(row["physical_cell_id"], []).append(row)
        elif measurement_id.startswith("rpt"):
            targets.append(row)

    plans: dict[str, dict[str, Any]] = {}
    seen_targets: set[tuple[str, str]] = set()
    for target in targets:
        if (
            not is_unique_checkpoint(target)
            or not _has_cell_identity(target)
            or target["group_id"] not in _SUPPORTED_GROUPS
        ):
            continue
        expected_physical_cell_id = f"G{target['group_id']}-Cell{target['cell_id']}"
        if target["physical_cell_id"] != expected_physical_cell_id:
            raise ValueError(
                f"Conflicting physical-cell identity for {target['measurement_id']}: "
                f"{target['physical_cell_id']}."
            )
        physical_cell_id = target["physical_cell_id"]
        baseline_candidates = baselines.get(physical_cell_id, [])
        if len(baseline_candidates) != 1:
            raise ValueError(
                f"Expected one unique same-cell Part 3 BoL source for "
                f"{physical_cell_id} {target['measurement_id']}; "
                f"found {len(baseline_candidates)}."
            )
        baseline = baseline_candidates[0]
        if (
            baseline.get("group_id") != target["group_id"]
            or baseline.get("cell_id") != target["cell_id"]
            or baseline.get("physical_cell_id") != physical_cell_id
            or baseline.get("source_file_id") != "0"
            or baseline.get("source_mapping_rule") != _PART3_BOL_SOURCE_RULE
        ):
            raise ValueError(
                f"Part 3 BoL source for {physical_cell_id} does not match the "
                "corrected unsuffixed-to-.0 crosswalk."
            )

        target_key = (physical_cell_id, target["measurement_id"].casefold())
        if target_key in seen_targets:
            raise ValueError(
                f"Duplicate Guide RPT mapping for {physical_cell_id} "
                f"{target['measurement_id']}."
            )
        seen_targets.add(target_key)
        plan = plans.setdefault(
            physical_cell_id,
            {
                "group_id": target["group_id"],
                "cell_id": target["cell_id"],
                "baseline": baseline,
                "targets": [],
            },
        )
        if (
            plan["group_id"] != target["group_id"]
            or plan["cell_id"] != target["cell_id"]
            or plan["baseline"] is not baseline
        ):
            raise ValueError(f"Conflicting cell or baseline mapping for {physical_cell_id}.")
        plan["targets"].append(target)

    for plan in plans.values():
        plan["targets"].sort(key=source_order)
    return dict(sorted(plans.items()))


def _crosswalk_audit(
    manifest_rows: list[dict[str, str]],
    plans: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    selected_sources = {
        _row_key(target)
        for plan in plans.values()
        for target in plan["targets"]
    }
    audit_rows = []
    for row in manifest_rows:
        if (
            row.get("dataset_part") != "3"
            or not row.get("measurement_id", "").casefold().startswith("rpt")
        ):
            continue
        if not _has_cell_identity(row):
            status = "withheld_unresolved_cell_identity"
        elif row.get("group_id") not in _SUPPORTED_GROUPS:
            status = "withheld_unsupported_group"
        elif _row_key(row) in selected_sources:
            status = "selected_unique_single_checkpoint"
        elif row.get("measurement_mapping_status") != "single_checkpoint":
            status = "withheld_ambiguous_or_compound_checkpoint"
        elif (
            row.get("source_mapping_status") != "unique_source_file"
            or row.get("source_file_count") != "1"
        ):
            status = "withheld_non_unique_source_mapping"
        else:
            status = "withheld_unmapped_or_no_unique_same_cell_bol"
        audit_rows.append({**row, "pydma_selection_status": status})
    return audit_rows


def run_part3(
    part1_root: Path,
    part3_root: Path,
    manifest_path: Path,
    output_dir: Path,
    expected_gpu: str | None,
) -> dict[str, Any]:
    import torch

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("The Part 3 PyDMA experiment requires CUDA.")
    device_name = torch.cuda.get_device_name(0)
    if expected_gpu and expected_gpu.casefold() not in device_name.casefold():
        raise RuntimeError(
            f"Expected GPU containing {expected_gpu!r}, found {device_name!r}."
        )

    with manifest_path.open(encoding="utf-8-sig", newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    part3_rows = [row for row in manifest_rows if row.get("dataset_part") == "3"]
    plans = _build_cell_plans(manifest_rows)
    if not plans:
        raise ValueError("The manifest contains no eligible Part 3 cell/RPT plans.")

    roots = {"3": part3_root}
    fit_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for physical_cell_id, plan in plans.items():
        print(
            f"Running Part 3 PyDMA {physical_cell_id}: "
            f"{len(plan['targets'])} single-checkpoint RPTs",
            flush=True,
        )
        records, entries, fit_targets = cell_inputs(plan, roots, _DIRECTIONS)
        fit_results.extend(
            run_pydma_for_cell(
                part3_root,
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

    rpt_results = [row for row in fit_results if row["checkup"] != "BoL"]
    accepted_rpt_results = [
        row
        for row in rpt_results
        if row["diagnostic_label_status"] == "diagnostic_estimate_accepted"
    ]
    raw_rpt_results = [
        row
        for row in rpt_results
        if row["diagnostic_label_status"] == "raw_diagnostic_estimate"
    ]
    status_counts = Counter(row["diagnostic_label_status"] for row in rpt_results)
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    summary = {
        "experiment": "Part 3 PyDMA LLI/LAM estimates with same-cell BoL references",
        "dataset_scope": (
            "Part 3 Groups 7-10; Guide-identified cells and unique "
            "single-checkpoint RPT mappings only."
        ),
        "runtime": {
            "device": "cuda:0",
            "gpu_name": device_name,
            "torch_version": torch.__version__,
            "elapsed_seconds": time.perf_counter() - started,
            "optimizer_options": _OPTIMIZER_OPTIONS,
        },
        "input_manifest": str(manifest_path),
        "input_manifest_sha256": manifest_hash,
        "part1_root": str(part1_root),
        "part3_root": str(part3_root),
        "manifest_part3_row_count": len(part3_rows),
        "manifest_bol_count": sum(
            row.get("measurement_id", "").casefold() == "bol" for row in part3_rows
        ),
        "manifest_rpt_count": sum(
            row.get("measurement_id", "").casefold().startswith("rpt")
            for row in part3_rows
        ),
        "cell_count": len(plans),
        "selected_single_checkpoint_rpt_count": sum(
            len(plan["targets"]) for plan in plans.values()
        ),
        "baseline_direction_fit_count": sum(
            row["checkup"] == "BoL" for row in fit_results
        ),
        "rpt_direction_result_count": len(rpt_results),
        "accepted_rpt_direction_count": len(accepted_rpt_results),
        "raw_rpt_direction_count": len(raw_rpt_results),
        "numeric_rpt_degradation_value_count": (
            len(accepted_rpt_results) + len(raw_rpt_results)
        )
        * 3,
        "rpt_fit_status_counts": dict(sorted(status_counts.items())),
        "baseline_policy": {
            "part3": (
                "Use the unique same-cell Guide-listed BoL source; the corrected "
                "unsuffixed BoL procedure maps to source file id 0 (.0)."
            ),
            "half_cell_references": "Part 1 Half_Cell reference files.",
            "interpretation": (
                "The BoL mapping is a crosswalk choice for this diagnostic run, "
                "not a validated SOH denominator."
            ),
        },
        "fit_policy": {
            "target_curves": (
                "Unique single-checkpoint Part 3 RPT mappings; pseudo-OCV "
                "Step 8 discharge and Step 10 charge."
            ),
            "rmse_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
            "minimum_accepted_runs": PYDMA_MIN_ACCEPTED_RUNS,
            "accepted_gate": (
                f"PyDMA accepted fit, RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV, "
                f"and at least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs."
            ),
            "raw_estimates": (
                "Retained separately when the fit runs but does not pass the "
                "accepted-label gate."
            ),
        },
        "source_files_modified": False,
        "limitations": [
            "LLI/LAM values are PyDMA-derived diagnostic estimates, not independent physical ground truth or SOH labels.",
            "The unnamed Group 10 Guide sequence remains quarantined; no cell identity is inferred from filenames.",
            "Part 3 is a different instrument/protocol domain and these results are not used for model selection.",
            "Part 3 temperature is not stated in its README; temperature is not used by this PyDMA fit.",
            "Mapped RPT order is preserved; no exact lifetime cycle/time is reconstructed by this run.",
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "pydma_results.csv", fit_results)
    write_csv(output_dir / "accepted_rpt_estimates.csv", accepted_rpt_results)
    write_csv(output_dir / "raw_rpt_estimates.csv", raw_rpt_results)
    write_csv(output_dir / "crosswalk_audit.csv", _crosswalk_audit(manifest_rows, plans))
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    (output_dir / "README.md").write_text(
        "# Part 3 PyDMA degradation-diagnostic run\n\n"
        "This run uses the corrected, same-cell Part 3 BoL crosswalk and fits "
        "only unique single-checkpoint Guide-mapped RPTs. Unresolved Guide "
        "identities remain excluded. PyDMA LLI/LAM outputs are model-derived "
        "diagnostics, not SOH labels or independent physical ground truth. "
        f"The accepted-fit gate is RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV with "
        f"at least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs. Accepted and raw "
        "RPT estimates are written separately; BoL reference fits remain in "
        "`pydma_results.csv` for QC.\n\n"
        "See `run_summary.json`, `pydma_results.csv`, "
        "`accepted_rpt_estimates.csv`, `raw_rpt_estimates.csv`, and "
        "`crosswalk_audit.csv`.\n\n"
        "Reproduce from the repository root in PowerShell:\n\n"
        "```powershell\n"
        '$env:PYTHONPATH = "src"\n'
        "python -m bafuse.labeling.run_part3_pydma `\n"
        f'  --part1-root "{part1_root}" `\n'
        f'  --part3-root "{part3_root}" `\n'
        f'  --manifest "{manifest_path}" `\n'
        f'  --output "{output_dir}" `\n'
        f'  --expected-gpu "{expected_gpu or ""}"\n'
        "```\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate separate Part 3 PyDMA diagnostic estimates."
    )
    parser.add_argument("--part1-root", type=Path, required=True)
    parser.add_argument("--part3-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", default=None)
    args = parser.parse_args()
    summary = run_part3(
        args.part1_root,
        args.part3_root,
        args.manifest,
        args.output,
        args.expected_gpu,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
