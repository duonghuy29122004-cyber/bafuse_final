from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
V3_DIR = ROOT / "experiments" / "pydma_hybrid_direction_v3"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(V3_DIR))

from run_hybrid_direction_v3 import (  # noqa: E402
    _aggregate_trajectories,
    _number,
    _select_v3_row,
    _trajectory_stats,
    _truth,
)


INPUTS = {
    "h2_hybrid_v3.csv": V3_DIR / "h2_hybrid_v3.csv",
    "h2_direction_policy_v3.csv": V3_DIR / "h2_direction_policy_v3.csv",
    "trajectory_method_values.csv": V3_DIR / "trajectory_method_values.csv",
    "trajectory_trend_per_cell.csv": V3_DIR / "trajectory_trend_per_cell.csv",
}
CELL_FIELDS = [
    "evaluation_domain",
    "group",
    "physical_cell_id",
    "metric",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_direction(charge_count: int, discharge_count: int) -> tuple[str, str]:
    if charge_count > 0 and discharge_count == 0:
        return "charge", "single_direction_charge_only"
    if discharge_count > 0 and charge_count == 0:
        return "discharge", "single_direction_discharge_only"
    if charge_count > 0 and discharge_count > 0:
        return "", "ambiguous_both_directions_without_matched_checkpoints"
    return "", "no_accepted_single_direction_fits"


def _audit_undetermined(policies: pd.DataFrame) -> pd.DataFrame:
    undetermined = policies[policies["primary_direction"].eq("undetermined")].copy()
    rows = []
    for row in undetermined.to_dict("records"):
        charge_count = int(row["charge_accepted_metric_checkpoints_in_rate_scope"])
        discharge_count = int(row["discharge_accepted_metric_checkpoints_in_rate_scope"])
        direction, candidate_status = _candidate_direction(
            charge_count, discharge_count
        )
        rows.append(
            {
                "evaluation_domain": row["evaluation_domain"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "metric": row["metric"],
                "accepted_charge_fit_checkpoints": charge_count,
                "accepted_discharge_fit_checkpoints": discharge_count,
                "matched_accepted_checkpoints": int(
                    row["paired_accepted_checkpoints"]
                ),
                "h2_v3_reason": row["selection_reason"],
                "h2_v4_candidate_direction": direction or "undetermined",
                "h2_v4_candidate_status": candidate_status,
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 6:
        raise ValueError(f"Expected six undetermined H2 v3 policies, found {len(result)}.")
    if result.duplicated(CELL_FIELDS).any():
        raise ValueError("Undetermined-policy audit contains duplicate cell/metric keys.")
    return result


def _make_h2_v4(
    h2_v3: pd.DataFrame,
    audit: pd.DataFrame,
) -> pd.DataFrame:
    result = h2_v3.copy()
    result["h2_v4_candidate_direction"] = result["h2_v3_primary_direction"]
    result["h2_v4_candidate_selected_percent"] = result["h2_v3_selected_percent"]
    result["h2_v4_candidate_selected_accepted"] = result[
        "h2_v3_selected_accepted"
    ].map(_truth)
    result["h2_v4_candidate_source_direction"] = result["h2_v3_source_direction"]
    result["h2_v4_candidate_fallback_applied"] = result[
        "h2_v3_fallback_applied"
    ].map(_truth)
    result["h2_v4_candidate_fallback_reason"] = result["h2_v3_fallback_reason"]
    result["h2_v4_candidate_status"] = "unchanged_h2_v3_policy"

    audit_lookup = {
        (row["physical_cell_id"], row["metric"]): row
        for row in audit.to_dict("records")
    }
    for index, row in result.iterrows():
        policy = audit_lookup.get((row["physical_cell_id"], row["metric"]))
        if policy is None:
            continue
        direction = policy["h2_v4_candidate_direction"]
        result.at[index, "h2_v4_candidate_direction"] = direction
        result.at[index, "h2_v4_candidate_status"] = policy[
            "h2_v4_candidate_status"
        ]
        if direction not in {"charge", "discharge"}:
            result.at[index, "h2_v4_candidate_selected_percent"] = np.nan
            result.at[index, "h2_v4_candidate_selected_accepted"] = False
            result.at[index, "h2_v4_candidate_source_direction"] = ""
            result.at[index, "h2_v4_candidate_fallback_applied"] = False
            result.at[index, "h2_v4_candidate_fallback_reason"] = (
                "undetermined_policy_no_single_direction_evidence"
            )
            continue
        selection = _select_v3_row(row.to_dict(), direction)
        result.at[index, "h2_v4_candidate_selected_percent"] = selection[
            "h2_v3_selected_percent"
        ]
        result.at[index, "h2_v4_candidate_selected_accepted"] = selection[
            "h2_v3_selected_accepted"
        ]
        result.at[index, "h2_v4_candidate_source_direction"] = selection[
            "h2_v3_source_direction"
        ]
        result.at[index, "h2_v4_candidate_fallback_applied"] = selection[
            "h2_v3_fallback_applied"
        ]
        result.at[index, "h2_v4_candidate_fallback_reason"] = selection[
            "h2_v3_fallback_reason"
        ]

    key_fields = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "checkpoint_id",
        "metric",
    ]
    if result.duplicated(key_fields).any():
        raise ValueError("H2 v4 candidate contains duplicate checkpoint/metric keys.")
    return result


def _coverage_summary(v3: pd.DataFrame, v4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for domain in ("parts1_2_internal", "external_part3"):
        for metric in ("LLI", "Anode_LAM", "Cathode_LAM", "OVERALL"):
            for scope in ("regular", "eol"):
                base = v4.copy()
                base = base[base["evaluation_domain"].eq(domain)]
                if metric != "OVERALL":
                    base = base[base["metric"].eq(metric)]
                base = base[base["is_eol"].map(_truth).eq(scope == "eol")]
                old_selected = base["h2_v3_selected_accepted"].map(_truth).tolist()
                new_selected = base["h2_v4_candidate_selected_accepted"].map(_truth).tolist()
                new_values = [
                    _number(value) if accepted else None
                    for value, accepted in zip(
                        base["h2_v4_candidate_selected_percent"], new_selected
                    )
                ]
                old_count = sum(old_selected)
                new_count = sum(new_selected)
                rows.append(
                    {
                        "evaluation_domain": domain,
                        "metric": metric,
                        "checkpoint_scope": scope,
                        "mapped_checkpoint_metric_rows": len(base),
                        "h2_v3_selected_estimates": old_count,
                        "h2_v4_candidate_selected_estimates": new_count,
                        "newly_selected_estimates": sum(
                            not old and new
                            for old, new in zip(old_selected, new_selected)
                        ),
                        "negative_h2_v4_candidate_estimates": sum(
                            value is not None and value < 0 for value in new_values
                        ),
                        "cells_with_h2_v3_estimates": int(
                            base.loc[
                                base["h2_v3_selected_accepted"].map(_truth),
                                "physical_cell_id",
                            ].nunique()
                        ),
                        "cells_with_h2_v4_candidate_estimates": int(
                            base.loc[
                                base["h2_v4_candidate_selected_accepted"].map(_truth),
                                "physical_cell_id",
                            ].nunique()
                        ),
                        "unavailable_h2_v4_candidate_estimates": len(base) - new_count,
                    }
                )
    return pd.DataFrame(rows)


def _candidate_method_values(
    h2_v4: pd.DataFrame, method_values: pd.DataFrame
) -> pd.DataFrame:
    base = method_values[method_values["method"].eq("H2 Hybrid v3")].copy()
    keys = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "checkpoint_id",
        "metric",
    ]
    joined = base.merge(
        h2_v4[
            keys
            + [
                "h2_v4_candidate_selected_percent",
                "h2_v4_candidate_selected_accepted",
                "h2_v4_candidate_source_direction",
                "h2_v4_candidate_fallback_applied",
                "h2_v4_candidate_fallback_reason",
                "h2_v4_candidate_direction",
            ]
        ],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    accepted = joined["h2_v4_candidate_selected_accepted"].map(_truth)
    numeric = pd.to_numeric(
        joined["h2_v4_candidate_selected_percent"], errors="coerce"
    )
    joined["method"] = "H2 Hybrid v4 candidate"
    joined["value_percent"] = numeric.where(accepted & joined["mapping_verified"].map(_truth))
    joined["candidate_value_percent"] = numeric
    joined["valid_value"] = joined["value_percent"].notna()
    joined["fit_accepted"] = accepted
    joined["source_direction"] = joined["h2_v4_candidate_source_direction"].fillna("")
    joined["fallback_applied"] = joined[
        "h2_v4_candidate_fallback_applied"
    ].map(_truth)
    joined["primary_direction"] = joined["h2_v4_candidate_direction"].fillna("")
    return joined.drop(
        columns=[
            "h2_v4_candidate_selected_percent",
            "h2_v4_candidate_selected_accepted",
        ]
    )


def _trajectory_outputs(
    v4_values: pd.DataFrame,
    v3_per_cell: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identities = v4_values[CELL_FIELDS].drop_duplicates().to_dict("records")
    per_cell = []
    transition_rows = []
    for identity in identities:
        mask = pd.Series(True, index=v4_values.index)
        for field in CELL_FIELDS:
            mask &= v4_values[field].eq(identity[field])
        rows = v4_values[mask & v4_values["valid_value"]].copy()
        verified = set(
            pd.to_numeric(
                v4_values.loc[
                    mask
                    & v4_values["rpt_index"].notna()
                    & ~v4_values["is_eol"].map(_truth),
                    "rpt_index",
                ],
                errors="coerce",
            )
            .dropna()
            .astype(int)
        )
        for sensitivity, include_eol in (("regular_only", False), ("eol_inclusive", True)):
            selected = rows if include_eol else rows[~rows["is_eol"].map(_truth)]
            stats, details = _trajectory_stats(
                selected,
                sensitivity,
                "all_available",
                {**identity, "method": "H2 Hybrid v4 candidate"},
                verified,
            )
            per_cell.append(stats)
            transition_rows.extend(details)
    candidate_per_cell = pd.DataFrame(per_cell)
    candidate_transitions = pd.DataFrame(transition_rows)
    v3 = v3_per_cell[
        v3_per_cell["method"].eq("H2 Hybrid v3")
        & v3_per_cell["checkpoint_scope"].eq("all_available")
    ].copy()
    candidate_summary = _aggregate_trajectories(candidate_per_cell)
    v3_summary = _aggregate_trajectories(v3)
    candidate_summary["method"] = "H2 Hybrid v4 candidate"
    comparison = pd.concat([v3_summary, candidate_summary], ignore_index=True)
    comparison = comparison[
        comparison["metric"].isin(["LLI", "Anode_LAM", "Cathode_LAM", "OVERALL"])
    ]
    return candidate_per_cell, candidate_transitions, comparison, v3


def _write_report(
    report_path: Path,
    audit: pd.DataFrame,
    h2_v4: pd.DataFrame,
    coverage: pd.DataFrame,
    comparison: pd.DataFrame,
    transitions: pd.DataFrame,
) -> None:
    old_count = int(h2_v4["h2_v3_selected_accepted"].map(_truth).sum())
    new_count = int(h2_v4["h2_v4_candidate_selected_accepted"].map(_truth).sum())
    new_rows = int(
        (
            ~h2_v4["h2_v3_selected_accepted"].map(_truth)
            & h2_v4["h2_v4_candidate_selected_accepted"].map(_truth)
        ).sum()
    )
    lost_rows = int(
        (
            h2_v4["h2_v3_selected_accepted"].map(_truth)
            & ~h2_v4["h2_v4_candidate_selected_accepted"].map(_truth)
        ).sum()
    )
    lines = [
        "# H2 v4 candidate - single-direction availability audit",
        "",
        "This is a separate, non-production candidate. H2 v3 outputs are read-only and unchanged. "
        "No PyDMA optimization, SoH inference, model training, label editing, or manifest editing was performed.",
        "",
        "Candidate rule: only a cell-metric policy with zero matched accepted checkpoints may use a "
        "direction if that direction has accepted fits and the other has none. If both directions have "
        "accepted unpaired fits, the candidate remains explicitly ambiguous; if neither has fits, it stays unavailable. "
        "At each RPT the existing signed positive-fallback selection logic is reused.",
        "",
        "## Six undetermined policies",
        "",
        "| Domain | Cell | Metric | Accepted Charge checkpoints | Accepted Discharge checkpoints | Paired accepted | Candidate direction | Candidate status |",
        "|---|---|---|---:|---:|---:|---|---|",
    ]
    for row in audit.to_dict("records"):
        lines.append(
            f"| {row['evaluation_domain']} | {row['physical_cell_id']} | {row['metric']} | "
            f"{row['accepted_charge_fit_checkpoints']} | {row['accepted_discharge_fit_checkpoints']} | "
            f"{row['matched_accepted_checkpoints']} | {row['h2_v4_candidate_direction']} | "
            f"{row['h2_v4_candidate_status']} |"
        )
    lines.extend(
        [
            "",
            "G2-Cell8 has nine accepted Charge-only checkpoints per metric (27 estimate rows total) "
            "and no accepted Discharge fits. G10-Cell2 has no accepted fits in either direction and "
            "Step 4 remains unaudited; no degradation estimates are created for it.",
            "",
            "## Coverage change",
            "",
            f"- H2 v3 accepted estimate rows: **{old_count}**.",
            f"- H2 v4 candidate accepted estimate rows: **{new_count}**.",
            f"- Newly selected: **{new_rows}**; lost selections: **{lost_rows}**.",
            "- The only new coverage is the G2-Cell8 Charge-only evidence. This is a candidate "
            "availability policy, not a population-wide direction preference.",
            "",
            "| Domain | Metric | Scope | H2 v3 | H2 v4 candidate | Newly selected | Negative candidate estimates | Cells v3 | Cells v4 | Unavailable |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in coverage.to_dict("records"):
        lines.append(
            f"| {row['evaluation_domain']} | {row['metric']} | {row['checkpoint_scope']} | "
            f"{row['h2_v3_selected_estimates']} | {row['h2_v4_candidate_selected_estimates']} | "
            f"{row['newly_selected_estimates']} | {row['negative_h2_v4_candidate_estimates']} | "
            f"{row['cells_with_h2_v3_estimates']} | {row['cells_with_h2_v4_candidate_estimates']} | "
            f"{row['unavailable_h2_v4_candidate_estimates']} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory and temporal diagnostics",
            "",
            "Trend classification requires at least three valid checkpoints. No monotonicity correction "
            "is applied. See `trajectory_comparison.csv` and `transition_details.csv` for each domain, "
            "metric, and regular/EoL-inclusive scope.",
            "",
        ]
    )
    for row in comparison[
        comparison["metric"].eq("OVERALL")
        & comparison["sensitivity"].eq("regular_only")
    ].to_dict("records"):
        lines.append(
            f"- {row['evaluation_domain']} / {row['method']}: "
            f"{row['increasing_trend_trajectories']}/"
            f"{row['eligible_cell_metric_trajectories']} increasing; "
            f"{row['fully_non_decreasing_trajectories']} fully non-decreasing trajectories; "
            f"{row['adjacent_observed_decreases']} adjacent observed decreases."
        )
    v3_decreases = int(
        transitions["transition_direction"].eq("decrease").sum()
    ) if not transitions.empty else 0
    lines.extend(
        [
            "",
            f"Candidate adjacent-observation decreases: **{v3_decreases}**. These are descriptive "
            "transition counts, not independent samples and not a selection objective.",
            "",
            "No policy was selected because it made a trajectory more monotonic. Negative values, when "
            "present, remain signed. H2 v4 is not installed as the dashboard default.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    missing = [str(path) for path in INPUTS.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required H2 v3 inputs are missing: " + ", ".join(missing))
    input_hashes = {name: _sha256(path) for name, path in INPUTS.items()}
    h2_v3 = pd.read_csv(INPUTS["h2_hybrid_v3.csv"], low_memory=False)
    policies = pd.read_csv(INPUTS["h2_direction_policy_v3.csv"], low_memory=False)
    method_values = pd.read_csv(INPUTS["trajectory_method_values.csv"], low_memory=False)
    prior_per_cell = pd.read_csv(INPUTS["trajectory_trend_per_cell.csv"], low_memory=False)

    audit = _audit_undetermined(policies)
    expected = {
        ("G2-Cell8", metric): (9, 0)
        for metric in ("LLI", "Anode_LAM", "Cathode_LAM")
    } | {
        ("G10-Cell2", metric): (0, 0)
        for metric in ("LLI", "Anode_LAM", "Cathode_LAM")
    }
    observed = {
        (row["physical_cell_id"], row["metric"]): (
            int(row["accepted_charge_fit_checkpoints"]),
            int(row["accepted_discharge_fit_checkpoints"]),
        )
        for row in audit.to_dict("records")
    }
    if observed != expected:
        raise ValueError(f"Undetermined-policy evidence changed: {observed}")

    h2_v4 = _make_h2_v4(h2_v3, audit)
    values = _candidate_method_values(h2_v4, method_values)
    candidate_per_cell, transitions, trajectory_summary, _ = _trajectory_outputs(
        values, prior_per_cell
    )
    coverage = _coverage_summary(h2_v3, h2_v4)
    if int(h2_v4["h2_v4_candidate_selected_accepted"].map(_truth).sum()) != 714:
        raise ValueError("H2 v4 candidate selected count differs from the audited +27 estimate expectation.")
    if int(
        (
            h2_v4["physical_cell_id"].eq("G10-Cell2")
            & h2_v4["h2_v4_candidate_selected_accepted"].map(_truth)
        ).sum()
    ):
        raise ValueError("H2 v4 candidate must not fabricate G10-Cell2 estimates.")

    outputs = {
        "undetermined_policy_audit.csv": audit,
        "h2_v4_candidate.csv": h2_v4,
        "coverage_comparison.csv": coverage,
        "trajectory_per_cell.csv": candidate_per_cell,
        "transition_details.csv": transitions,
        "trajectory_comparison.csv": trajectory_summary,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUT / name, index=False)
    report = OUT / "h2_v4_candidate_report.md"
    _write_report(report, audit, h2_v4, coverage, trajectory_summary, transitions)

    after_hashes = {name: _sha256(path) for name, path in INPUTS.items()}
    if input_hashes != after_hashes:
        raise RuntimeError("H2 v3 inputs changed while building the candidate experiment.")
    manifest = {
        "experiment": "pydma_hybrid_direction_v4_candidate",
        "optimizer_rerun": False,
        "h2_v3_modified": False,
        "model_retrained": False,
        "input_sha256": input_hashes,
        "input_hashes_unchanged": True,
        "output_files": sorted([*outputs, report.name]),
        "counts": {
            "undetermined_policies": len(audit),
            "newly_selected_estimates": int(
                (
                    ~h2_v4["h2_v3_selected_accepted"].map(_truth)
                    & h2_v4["h2_v4_candidate_selected_accepted"].map(_truth)
                ).sum()
            ),
            "h2_v3_selected_estimates": int(
                h2_v4["h2_v3_selected_accepted"].map(_truth).sum()
            ),
            "h2_v4_candidate_selected_estimates": int(
                h2_v4["h2_v4_candidate_selected_accepted"].map(_truth).sum()
            ),
        },
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
