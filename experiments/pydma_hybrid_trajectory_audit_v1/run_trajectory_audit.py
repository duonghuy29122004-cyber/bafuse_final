from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "experiments" / "pydma_hybrid_direction_v2"
OUT = Path(__file__).resolve().parent
METHODS = ("Charge-only", "Discharge-only", "H1 Hybrid", "H2 Hybrid")
METRICS = ("LLI", "Anode_LAM", "Cathode_LAM")
DOMAIN_ORDER = ("parts1_2_internal", "external_part3")
INPUT_FILES = ("h1_hybrid_v2.csv", "h2_hybrid_v2.csv")
CELL_GROUP_COLUMNS = ["evaluation_domain", "group", "physical_cell_id", "metric"]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    return _text(value).casefold() in {"true", "1", "yes"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_key(row: dict[str, Any]) -> str:
    part = _number(row.get("part"))
    part_text = str(int(part)) if part is not None else _text(row.get("part"))
    rpt = _number(row.get("rpt_index"))
    cell = _text(row.get("physical_cell_id"))
    if rpt is not None:
        return f"P{part_text}|{cell}|RPT{int(rpt)}"
    checkpoint = _text(row.get("checkpoint_id")) or _text(row.get("rpt_or_checkpoint"))
    if not checkpoint:
        raise ValueError(f"Unnumbered checkpoint lacks an identifier: {row}")
    return f"P{part_text}|{cell}|CHECKPOINT:{checkpoint}"


def _source_key(row: dict[str, Any]) -> tuple[Any, ...]:
    part = _number(row.get("part"))
    rpt = _number(row.get("rpt_index"))
    return (
        int(part) if part is not None else _text(row.get("part")),
        _text(row.get("physical_cell_id")),
        int(rpt) if rpt is not None else None,
        _text(row.get("rpt_or_checkpoint")),
        _text(row.get("checkpoint_id")),
        _text(row.get("metric")),
    )


def _input_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, str]]:
    missing = [INPUT_DIR / name for name in INPUT_FILES if not (INPUT_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Required v2 hybrid results are missing: " + ", ".join(map(str, missing)))
    hashes = {name: _sha256(INPUT_DIR / name) for name in INPUT_FILES}
    h1 = pd.read_csv(INPUT_DIR / INPUT_FILES[0], low_memory=False)
    h2 = pd.read_csv(INPUT_DIR / INPUT_FILES[1], low_memory=False)
    for frame in (h1, h2):
        frame["part"] = pd.to_numeric(frame["part"], errors="coerce")
        frame["group"] = pd.to_numeric(frame["group"], errors="coerce")
        frame["rpt_index"] = pd.to_numeric(frame["rpt_index"], errors="coerce")
        frame["chronological_position"] = pd.to_numeric(frame["chronological_position"], errors="coerce")
        frame["position"] = pd.to_numeric(frame["position"], errors="coerce")
    h1 = h1[h1["mapping_verified"].map(_truth)].copy()
    h2 = h2[h2["mapping_verified"].map(_truth)].copy()
    if h1["group"].isin({5, 6}).any() or h2["group"].isin({5, 6}).any():
        raise ValueError("Part 2 Groups 5–6 must not be present in the v2 analysis inputs.")
    if h1["evaluation_domain"].eq("external_part3").any() != h2["evaluation_domain"].eq("external_part3").any():
        raise ValueError("H1 and H2 inputs do not have consistent domain coverage.")
    key_columns = ["part", "group", "physical_cell_id", "rpt_index", "rpt_or_checkpoint", "checkpoint_id", "metric"]
    h1_keys = {_source_key(row) for row in h1.to_dict("records")}
    h2_keys = {_source_key(row) for row in h2.to_dict("records")}
    if h1_keys != h2_keys or len(h1_keys) != len(h1) or len(h2_keys) != len(h2):
        raise ValueError("H1/H2 input checkpoint keys are not a one-to-one match.")
    for frame in (h1, h2):
        if frame["chronological_position"].isna().any() or frame["position"].isna().any():
            raise ValueError("A verified checkpoint is missing its saved sequence-order position.")
        if frame.duplicated(key_columns).any():
            raise ValueError("A v2 input contains duplicated cell/checkpoint/metric keys.")
    return h1, h2, hashes


def _method_value(row: dict[str, Any], method: str) -> dict[str, Any]:
    metric = row["metric"]
    if method == "Charge-only":
        direction = "charge"
        candidate = _number(row.get(f"charge_{metric}_percent"))
        accepted = _truth(row.get("charge_accepted"))
        status = row.get("charge_status", "")
        rmse = row.get("charge_rmse_mV")
        runs = row.get("charge_accepted_runs")
        source = "charge"
    elif method == "Discharge-only":
        direction = "discharge"
        candidate = _number(row.get(f"discharge_{metric}_percent"))
        accepted = _truth(row.get("discharge_accepted"))
        status = row.get("discharge_status", "")
        rmse = row.get("discharge_rmse_mV")
        runs = row.get("discharge_accepted_runs")
        source = "discharge"
    else:
        candidate = _number(row.get("selected_percent"))
        accepted = _truth(row.get("selected_accepted"))
        status = row.get("selected_fit_status", "")
        rmse = row.get("selected_fit_rmse_mV")
        runs = row.get("selected_fit_accepted_runs")
        source = _text(row.get("source_direction"))
        direction = source
    valid = accepted and candidate is not None and _truth(row.get("mapping_verified"))
    return {
        "method": method,
        "candidate_value_percent": candidate,
        "value_percent": candidate if valid else np.nan,
        "valid_value": valid,
        "method_fit_accepted": accepted,
        "method_fit_status": status,
        "method_fit_rmse_mV": _number(rmse),
        "method_fit_accepted_runs": _number(runs),
        "method_source_direction": source,
        "method_primary_direction": _text(
            row.get("h1_primary_direction") if method == "H1 Hybrid" else
            row.get("h2_primary_direction") if method == "H2 Hybrid" else direction
        ),
        "method_fallback_applied": _truth(row.get("fallback_applied")) if method in {"H1 Hybrid", "H2 Hybrid"} else False,
    }


def _build_checkpoint_values(h1: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sources = (
        ("h1_hybrid_v2.csv", h1, ("Charge-only", "Discharge-only", "H1 Hybrid")),
        ("h2_hybrid_v2.csv", h2, ("H2 Hybrid",)),
    )
    for source_name, frame, methods in sources:
        for source_row in frame.to_dict("records"):
            for method in methods:
                row = {
                    "checkpoint_key": _checkpoint_key(source_row),
                    "source_input_file": source_name,
                    "evaluation_domain": source_row["evaluation_domain"],
                    "part": int(source_row["part"]),
                    "group": int(source_row["group"]),
                    "physical_cell_id": source_row["physical_cell_id"],
                    "metric": source_row["metric"],
                    "is_eol": _truth(source_row.get("is_eol")),
                    "is_unnumbered_eol": _truth(source_row.get("is_unnumbered_eol")),
                    "mapping_verified": _truth(source_row.get("mapping_verified")),
                    "rpt_index": _number(source_row.get("rpt_index")),
                    "rpt_or_checkpoint": source_row.get("rpt_or_checkpoint"),
                    "checkpoint_id": source_row.get("checkpoint_id"),
                    "position": _number(source_row.get("position")),
                    "chronological_position": _number(source_row.get("chronological_position")),
                    **_method_value(source_row, method),
                }
                row.update({f"source__{name}": value for name, value in source_row.items()})
                rows.append(row)
    values = pd.DataFrame(rows)
    if values.duplicated(["method", "checkpoint_key", "metric"]).any():
        raise ValueError("Derived checkpoint-value rows contain duplicate method/checkpoint/metric keys.")
    return values


def _group_key(row: pd.Series | dict[str, Any]) -> tuple[Any, ...]:
    return tuple(row[name] for name in CELL_GROUP_COLUMNS)


def _mark_common_checkpoints(values: pd.DataFrame) -> pd.DataFrame:
    result = values.copy()
    result["common_regular_checkpoint"] = False
    result["common_eol_inclusive_checkpoint"] = False
    for include_eol, flag in (
        (False, "common_regular_checkpoint"),
        (True, "common_eol_inclusive_checkpoint"),
    ):
        subset = result[result["valid_value"]]
        if not include_eol:
            subset = subset[~subset["is_eol"]]
        for key, group in subset.groupby(CELL_GROUP_COLUMNS, sort=False, dropna=False):
            per_method = []
            for method in METHODS:
                per_method.append(
                    set(group.loc[group["method"].eq(method), "checkpoint_key"].astype(str))
                )
            common = set.intersection(*per_method) if per_method else set()
            if common:
                mask = (
                    result["evaluation_domain"].eq(key[0])
                    & result["group"].eq(key[1])
                    & result["physical_cell_id"].eq(key[2])
                    & result["metric"].eq(key[3])
                    & result["checkpoint_key"].isin(common)
                )
                if not include_eol:
                    mask &= ~result["is_eol"]
                result.loc[mask, flag] = True
    return result


def _classify_delta(delta: float) -> str:
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return "increase" if delta > 0 else "decrease"


def _transition_row(left: dict[str, Any], right: dict[str, Any], sensitivity: str, checkpoint_scope: str) -> dict[str, Any]:
    delta = float(right["value_percent"]) - float(left["value_percent"])
    left_rpt = _number(left.get("rpt_index"))
    right_rpt = _number(right.get("rpt_index"))
    is_eol_transition = _truth(left.get("is_eol")) or _truth(right.get("is_eol"))
    rpt_gap = right_rpt - left_rpt if left_rpt is not None and right_rpt is not None else None
    consecutive = (
        not is_eol_transition
        and _truth(left.get("mapping_verified"))
        and _truth(right.get("mapping_verified"))
        and rpt_gap is not None
        and math.isclose(rpt_gap, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )
    if is_eol_transition:
        transition_type = "eol_adjacent"
    elif consecutive:
        transition_type = "consecutive_verified_rpt"
    else:
        transition_type = "observed_gap"
    common = {
        "evaluation_domain": left["evaluation_domain"],
        "group": left["group"],
        "physical_cell_id": left["physical_cell_id"],
        "method": left["method"],
        "metric": left["metric"],
        "sensitivity": sensitivity,
        "checkpoint_scope": checkpoint_scope,
        "left_checkpoint_key": left["checkpoint_key"],
        "right_checkpoint_key": right["checkpoint_key"],
        "left_part": left["part"],
        "right_part": right["part"],
        "left_rpt_index": left_rpt,
        "right_rpt_index": right_rpt,
        "left_checkpoint": left["rpt_or_checkpoint"],
        "right_checkpoint": right["rpt_or_checkpoint"],
        "left_is_eol": _truth(left.get("is_eol")),
        "right_is_eol": _truth(right.get("is_eol")),
        "left_value_percent": float(left["value_percent"]),
        "right_value_percent": float(right["value_percent"]),
        "delta_percent": delta,
        "transition_direction": _classify_delta(delta),
        "transition_type": transition_type,
        "rpt_gap": rpt_gap,
        "missing_intermediate_rpt_count": max(int(rpt_gap) - 1, 0) if rpt_gap is not None else None,
        "is_consecutive_verified_rpt_transition": consecutive,
        "left_source_direction": left["method_source_direction"],
        "right_source_direction": right["method_source_direction"],
        "left_primary_direction": left["method_primary_direction"],
        "right_primary_direction": right["method_primary_direction"],
        "left_fallback_applied": left["method_fallback_applied"],
        "right_fallback_applied": right["method_fallback_applied"],
        "left_fit_status": left["method_fit_status"],
        "right_fit_status": right["method_fit_status"],
        "left_fit_rmse_mV": left["method_fit_rmse_mV"],
        "right_fit_rmse_mV": right["method_fit_rmse_mV"],
        "left_accepted_runs": left["method_fit_accepted_runs"],
        "right_accepted_runs": right["method_fit_accepted_runs"],
    }
    return common


def _trajectory_stats(
    trajectory: pd.DataFrame,
    sensitivity: str,
    checkpoint_scope: str,
    identity: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = trajectory.sort_values(
        ["chronological_position", "position", "part", "rpt_index"],
        kind="stable",
        na_position="last",
    )
    records = ordered.to_dict("records")
    transitions = [
        _transition_row(left, right, sensitivity, checkpoint_scope)
        for left, right in zip(records, records[1:])
    ]
    n = len(records)
    observed_directions = [row["transition_direction"] for row in transitions]
    consecutive_directions = [
        row["transition_direction"]
        for row in transitions
        if row["is_consecutive_verified_rpt_transition"]
    ]
    values = [float(row["value_percent"]) for row in records]
    positions = [float(row["chronological_position"]) for row in records]
    delta = values[-1] - values[0] if n else np.nan
    rho = pd.Series(positions).corr(pd.Series(values), method="spearman") if n >= 2 else np.nan
    eligible = n >= 3
    increasing = bool(delta > 0 and rho > 0) if eligible and pd.notna(rho) else False
    decreases = sum(direction == "decrease" for direction in observed_directions)
    increases = sum(direction == "increase" for direction in observed_directions)
    ties = sum(direction == "tie" for direction in observed_directions)
    consecutive_increases = sum(direction == "increase" for direction in consecutive_directions)
    consecutive_decreases = sum(direction == "decrease" for direction in consecutive_directions)
    consecutive_ties = sum(direction == "tie" for direction in consecutive_directions)
    first, last = (records[0], records[-1]) if n else ({}, {})
    stats = {
        "evaluation_domain": records[0]["evaluation_domain"] if n else identity["evaluation_domain"],
        "group": records[0]["group"] if n else identity["group"],
        "physical_cell_id": records[0]["physical_cell_id"] if n else identity["physical_cell_id"],
        "method": records[0]["method"] if n else identity["method"],
        "metric": records[0]["metric"] if n else identity["metric"],
        "sensitivity": sensitivity,
        "checkpoint_scope": checkpoint_scope,
        "valid_checkpoint_count": n,
        "eligible_for_trend_classification": eligible,
        "first_checkpoint": first.get("rpt_or_checkpoint"),
        "first_rpt_index": _number(first.get("rpt_index")),
        "last_checkpoint": last.get("rpt_or_checkpoint"),
        "last_rpt_index": _number(last.get("rpt_index")),
        "first_to_last_change_percent": delta,
        "spearman_chronological_order_vs_degradation": rho,
        "overall_increasing_trend": increasing if eligible else None,
        "overall_decrease": bool(delta < 0) if eligible else None,
        "overall_flat": bool(math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12)) if eligible else None,
        "adjacent_observed_transition_count": len(transitions),
        "adjacent_observed_increases": increases,
        "adjacent_observed_decreases": decreases,
        "adjacent_observed_ties": ties,
        "percentage_non_decreasing_adjacent_observed_transitions": (
            100.0 * (increases + ties) / len(transitions) if transitions else np.nan
        ),
        "all_observed_adjacent_transitions_non_decreasing": (
            bool(decreases == 0) if transitions else None
        ),
        "consecutive_verified_rpt_transition_count": len(consecutive_directions),
        "consecutive_verified_rpt_increases": consecutive_increases,
        "consecutive_verified_rpt_decreases": consecutive_decreases,
        "consecutive_verified_rpt_ties": consecutive_ties,
        "percentage_non_decreasing_consecutive_verified_rpt_transitions": (
            100.0 * (consecutive_increases + consecutive_ties) / len(consecutive_directions)
            if consecutive_directions
            else np.nan
        ),
        "nonconsecutive_observed_transition_count": sum(
            row["transition_type"] == "observed_gap" for row in transitions
        ),
        "eol_adjacent_transition_count": sum(
            row["transition_type"] == "eol_adjacent" for row in transitions
        ),
        "mixed_increases_and_decreases": bool(increases > 0 and decreases > 0),
    }
    return stats, transitions


def _build_trajectories(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    identities = values[CELL_GROUP_COLUMNS].drop_duplicates().to_dict("records")
    per_cell_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    scopes = (
        ("regular_only", "all_available", "common_regular_checkpoint", False),
        ("regular_only", "common_all_methods", "common_regular_checkpoint", False),
        ("eol_inclusive", "all_available", "common_eol_inclusive_checkpoint", True),
        ("eol_inclusive", "common_all_methods", "common_eol_inclusive_checkpoint", True),
    )
    for identity in identities:
        identity_mask = pd.Series(True, index=values.index)
        for col in CELL_GROUP_COLUMNS:
            identity_mask &= values[col].eq(identity[col])
        for method in METHODS:
            for sensitivity, checkpoint_scope, common_flag, include_eol in scopes:
                selected = values[identity_mask & values["method"].eq(method) & values["valid_value"]]
                if not include_eol:
                    selected = selected[~selected["is_eol"]]
                if checkpoint_scope == "common_all_methods":
                    selected = selected[selected[common_flag]]
                stats, transitions = _trajectory_stats(
                    selected,
                    sensitivity,
                    checkpoint_scope,
                    {**identity, "method": method},
                )
                per_cell_rows.append(stats)
                transition_rows.extend(transitions)
    per_cell = pd.DataFrame(per_cell_rows)
    transitions = pd.DataFrame(transition_rows)
    _mark_representatives(per_cell)
    return per_cell, transitions


def _mark_representatives(per_cell: pd.DataFrame) -> None:
    per_cell["representative_increasing_rank"] = np.nan
    per_cell["representative_decreasing_rank"] = np.nan
    groups = ["evaluation_domain", "sensitivity", "checkpoint_scope", "method", "metric"]
    for _, group in per_cell[per_cell["eligible_for_trend_classification"]].groupby(groups, sort=False):
        increasing = group[group["first_to_last_change_percent"] > 0].sort_values(
            ["first_to_last_change_percent", "spearman_chronological_order_vs_degradation"],
            ascending=[False, False],
        )
        decreasing = group[group["first_to_last_change_percent"] < 0].sort_values(
            ["first_to_last_change_percent", "spearman_chronological_order_vs_degradation"],
            ascending=[True, True],
        )
        per_cell.loc[increasing.head(3).index, "representative_increasing_rank"] = np.arange(
            1, min(len(increasing), 3) + 1
        )
        per_cell.loc[decreasing.head(3).index, "representative_decreasing_rank"] = np.arange(
            1, min(len(decreasing), 3) + 1
        )


def _summary_row(group: pd.DataFrame, domain: str, sensitivity: str, checkpoint_scope: str, method: str, metric: str) -> dict[str, Any]:
    eligible = group[group["eligible_for_trend_classification"]]
    n_eligible = len(eligible)
    n_increasing = sum(_truth(value) for value in eligible["overall_increasing_trend"])
    adjacent_total = int(eligible["adjacent_observed_transition_count"].sum())
    adjacent_nondec = int(
        (eligible["adjacent_observed_increases"] + eligible["adjacent_observed_ties"]).sum()
    )
    consecutive_total = int(eligible["consecutive_verified_rpt_transition_count"].sum())
    consecutive_nondec = int(
        (
            eligible["consecutive_verified_rpt_increases"]
            + eligible["consecutive_verified_rpt_ties"]
        ).sum()
    )
    fully_non_decreasing = sum(
        _truth(value) for value in eligible["all_observed_adjacent_transitions_non_decreasing"]
    )
    return {
        "evaluation_domain": domain,
        "sensitivity": sensitivity,
        "checkpoint_scope": checkpoint_scope,
        "method": method,
        "metric": metric,
        "valid_cell_metric_checkpoints": int(group["valid_checkpoint_count"].sum()),
        "cell_metric_trajectories_with_any_valid_checkpoint": int((group["valid_checkpoint_count"] > 0).sum()),
        "eligible_cell_metric_trajectories": n_eligible,
        "increasing_trend_trajectories": n_increasing,
        "percentage_increasing_trend_trajectories": 100.0 * n_increasing / n_eligible if n_eligible else np.nan,
        "increasing_share_exceeds_50_percent": bool(100.0 * n_increasing / n_eligible > 50.0) if n_eligible else False,
        "fully_non_decreasing_trajectories": fully_non_decreasing,
        "percentage_fully_non_decreasing_trajectories": (
            100.0 * fully_non_decreasing / n_eligible if n_eligible else np.nan
        ),
        "decreasing_overall_trajectories": sum(
            _truth(value) for value in eligible["overall_decrease"]
        ),
        "mixed_increase_decrease_trajectories": sum(
            _truth(value) for value in eligible["mixed_increases_and_decreases"]
        ),
        "adjacent_observed_transition_count": adjacent_total,
        "adjacent_observed_increases": int(eligible["adjacent_observed_increases"].sum()),
        "adjacent_observed_decreases": int(eligible["adjacent_observed_decreases"].sum()),
        "adjacent_observed_ties": int(eligible["adjacent_observed_ties"].sum()),
        "percentage_non_decreasing_adjacent_observed_transitions": (
            100.0 * adjacent_nondec / adjacent_total if adjacent_total else np.nan
        ),
        "consecutive_verified_rpt_transition_count": consecutive_total,
        "consecutive_verified_rpt_increases": int(eligible["consecutive_verified_rpt_increases"].sum()),
        "consecutive_verified_rpt_decreases": int(eligible["consecutive_verified_rpt_decreases"].sum()),
        "consecutive_verified_rpt_ties": int(eligible["consecutive_verified_rpt_ties"].sum()),
        "percentage_non_decreasing_consecutive_verified_rpt_transitions": (
            100.0 * consecutive_nondec / consecutive_total if consecutive_total else np.nan
        ),
    }


def _build_summary(per_cell: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_fields = ["evaluation_domain", "sensitivity", "checkpoint_scope", "method"]
    for keys, group in per_cell.groupby(group_fields, sort=False):
        domain, sensitivity, checkpoint_scope, method = keys
        for metric in METRICS:
            rows.append(
                _summary_row(
                    group[group["metric"].eq(metric)],
                    domain,
                    sensitivity,
                    checkpoint_scope,
                    method,
                    metric,
                )
            )
        rows.append(_summary_row(group, domain, sensitivity, checkpoint_scope, method, "OVERALL"))
    return pd.DataFrame(rows)


def _representative_text(per_cell: pd.DataFrame, domain: str, method: str, metric: str, rank_column: str) -> str:
    chosen = per_cell[
        per_cell["evaluation_domain"].eq(domain)
        & per_cell["sensitivity"].eq("regular_only")
        & per_cell["checkpoint_scope"].eq("all_available")
        & per_cell["method"].eq(method)
        & per_cell["metric"].eq(metric)
        & per_cell[rank_column].notna()
    ].sort_values(rank_column)
    if chosen.empty:
        return "None eligible."
    return "; ".join(
        f"{row.physical_cell_id} (change {row.first_to_last_change_percent:+.3f} pp, "
        f"Spearman {row.spearman_chronological_order_vs_degradation:+.3f}, "
        f"n={int(row.valid_checkpoint_count)})"
        for row in chosen.itertuples()
    )


def _trend_table(summary: pd.DataFrame, domain: str, sensitivity: str, scope: str) -> pd.DataFrame:
    return summary[
        summary["evaluation_domain"].eq(domain)
        & summary["sensitivity"].eq(sensitivity)
        & summary["checkpoint_scope"].eq(scope)
        & summary["method"].eq("H1 Hybrid")
        & summary["metric"].isin((*METRICS, "OVERALL"))
    ][
        [
            "metric",
            "increasing_trend_trajectories",
            "eligible_cell_metric_trajectories",
            "percentage_increasing_trend_trajectories",
            "increasing_share_exceeds_50_percent",
            "fully_non_decreasing_trajectories",
            "percentage_fully_non_decreasing_trajectories",
            "percentage_non_decreasing_adjacent_observed_transitions",
        ]
    ]


def _md_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No eligible trajectories._"
    def format_cell(column: str, value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (bool, np.bool_)):
            return "Yes" if value else "No"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)):
            if column.startswith("percentage_") or column.endswith("_percent"):
                return f"{float(value):.2f}%"
            return f"{float(value):.2f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(format_cell(column, value) for column, value in zip(headers, row)) + " |"
        )
    return "\n".join(lines)


def _write_report(
    per_cell: pd.DataFrame,
    summary: pd.DataFrame,
    hashes: dict[str, str],
    valid: pd.DataFrame,
) -> None:
    lines = [
        "# Hybrid PyDMA Increasing-Trajectory Audit — v1",
        "",
        "## Scope and method",
        "",
        "- This is a read-only analysis of the saved v2 hybrid CSVs. No PyDMA fitting or result selection was rerun, and no existing result was modified.",
        "- Charge-only and Discharge-only include only records with the corresponding saved v1 acceptance flag and a finite signed estimate. H1/H2 include only saved `selected_accepted` values. No values are clipped, made absolute, interpolated, or imputed.",
        "- Only `mapping_verified` records are included. Part 2 Groups 5–6 are excluded. Parts 1–2 are evaluated as the internal domain; Part 3 remains a separate external/domain-shift analysis.",
        "- Primary conclusions use regular RPTs only. EoL checkpoints are analyzed separately as an EoL-inclusive sensitivity. Part 2 unnumbered EoL records remain unnumbered; their saved chronological sequence position is used only to order the sensitivity, not to create an RPT number.",
        "- Sequence order uses the saved `chronological_position`. Spearman is an ordinal association with that order, not a cycle-age or causal estimate.",
        "- Adjacent observed transitions connect successive available valid values. Every such pair is marked with its RPT gap. Consecutive verified RPT transitions require both endpoints to be regular mapped RPTs and `right_rpt_index - left_rpt_index == 1`. Gaps are not bridged silently.",
        "- Trend classification requires at least 3 valid checkpoints and is `increasing` only when both first-to-last change and Spearman correlation are positive. Full-sequence monotonicity uses all adjacent observed transitions (including marked gaps).",
        "- Common-checkpoint comparisons use only checkpoints with valid values for all four methods at that same cell and metric. The common set is computed separately for regular-only and EoL-inclusive sensitivity.",
        "",
        "## Inputs and coverage",
        "",
        f"- Verified source coverage: {valid['checkpoint_key'].nunique()} unique checkpoints, "
        f"{len(valid.drop_duplicates(['checkpoint_key', 'metric']))} checkpoint-metric records, "
        f"and {int(valid['physical_cell_id'].nunique())} physical cells across both domains.",
        "- Source files and SHA-256:",
    ]
    lines.extend(f"  - `{name}`: `{digest}`" for name, digest in sorted(hashes.items()))
    lines.extend(
        [
            "",
            "## Primary regular-RPT method/metric summary",
            "",
            "Transition percentages pool observed transitions within eligible trajectories; trajectory prevalence uses cell–metric trajectories as the unit.",
            "",
        ]
    )
    for domain, title in (
        ("parts1_2_internal", "Parts 1–2 internal"),
        ("external_part3", "External Part 3"),
    ):
        primary = summary[
            summary["evaluation_domain"].eq(domain)
            & summary["sensitivity"].eq("regular_only")
            & summary["checkpoint_scope"].eq("all_available")
        ][
            [
                "method",
                "metric",
                "valid_cell_metric_checkpoints",
                "eligible_cell_metric_trajectories",
                "increasing_trend_trajectories",
                "percentage_increasing_trend_trajectories",
                "fully_non_decreasing_trajectories",
                "decreasing_overall_trajectories",
                "mixed_increase_decrease_trajectories",
                "adjacent_observed_increases",
                "adjacent_observed_decreases",
                "adjacent_observed_ties",
                "percentage_non_decreasing_adjacent_observed_transitions",
                "consecutive_verified_rpt_transition_count",
                "percentage_non_decreasing_consecutive_verified_rpt_transitions",
            ]
        ]
        lines.extend([f"### {title}", "", _md_table(primary), ""])
    lines.extend(
        [
            "",
            "## H1 Hybrid: increasing overall trend (>50% question)",
            "",
            "Numerator and denominator count eligible cell–metric trajectories, not individual transitions. Parts 1–2 and external Part 3 are reported separately.",
            "",
            "### Parts 1–2 internal, regular RPTs",
            "",
            _md_table(_trend_table(summary, "parts1_2_internal", "regular_only", "all_available")),
            "",
            "### External Part 3, regular RPTs",
            "",
            _md_table(_trend_table(summary, "external_part3", "regular_only", "all_available")),
            "",
            "## Common-checkpoint comparison",
            "",
            "All four methods are compared on the per-cell, per-metric intersection of their valid regular-RPT checkpoints. The denominator remains eligible cell–metric trajectories (at least 3 common checkpoints).",
            "",
        ]
    )
    for domain, title in (
        ("parts1_2_internal", "Parts 1–2 internal"),
        ("external_part3", "External Part 3"),
    ):
        common = summary[
            summary["evaluation_domain"].eq(domain)
            & summary["sensitivity"].eq("regular_only")
            & summary["checkpoint_scope"].eq("common_all_methods")
            & summary["metric"].isin((*METRICS, "OVERALL"))
        ][
            [
                "method",
                "metric",
                "increasing_trend_trajectories",
                "eligible_cell_metric_trajectories",
                "percentage_increasing_trend_trajectories",
                "fully_non_decreasing_trajectories",
                "percentage_non_decreasing_adjacent_observed_transitions",
            ]
        ]
        lines.extend([f"### {title}", "", _md_table(common), ""])
    lines.extend(["## EoL-inclusive sensitivity", ""])
    for domain, title in (
        ("parts1_2_internal", "Parts 1–2 internal"),
        ("external_part3", "External Part 3"),
    ):
        eol = summary[
            summary["evaluation_domain"].eq(domain)
            & summary["sensitivity"].eq("eol_inclusive")
            & summary["checkpoint_scope"].eq("all_available")
            & summary["metric"].isin((*METRICS, "OVERALL"))
        ][
            [
                "method",
                "metric",
                "increasing_trend_trajectories",
                "eligible_cell_metric_trajectories",
                "percentage_increasing_trend_trajectories",
                "decreasing_overall_trajectories",
                "mixed_increase_decrease_trajectories",
            ]
        ]
        lines.extend([f"### {title}", "", _md_table(eol), ""])
    lines.extend(["## Decreasing and mixed trajectories", ""])
    for domain, title in (
        ("parts1_2_internal", "Parts 1–2 internal"),
        ("external_part3", "External Part 3"),
    ):
        table = summary[
            summary["evaluation_domain"].eq(domain)
            & summary["sensitivity"].eq("regular_only")
            & summary["checkpoint_scope"].eq("all_available")
            & summary["metric"].eq("OVERALL")
        ][
            [
                "method",
                "eligible_cell_metric_trajectories",
                "decreasing_overall_trajectories",
                "mixed_increase_decrease_trajectories",
                "fully_non_decreasing_trajectories",
                "percentage_non_decreasing_adjacent_observed_transitions",
            ]
        ]
        lines.extend([f"### {title}", "", _md_table(table), ""])
    lines.extend(["## Representative regular-RPT trajectories", ""])
    for domain, title in (
        ("parts1_2_internal", "Parts 1–2 internal"),
        ("external_part3", "External Part 3"),
    ):
        lines.extend([f"### {title}", ""])
        for method in ("H1 Hybrid", "Discharge-only"):
            lines.append(f"**{method}**")
            for metric in METRICS:
                lines.append(
                    f"- **{metric} — largest increases:** "
                    + _representative_text(per_cell, domain, method, metric, "representative_increasing_rank")
                )
                lines.append(
                    f"- **{metric} — largest decreases:** "
                    + _representative_text(per_cell, domain, method, metric, "representative_decreasing_rank")
                )
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "An increasing trajectory is a descriptive property of the selected model-derived degradation estimates. It does not establish physical correctness, independent ground truth, or a valid causal degradation mechanism. Directional fitting outputs can disagree; the original estimates, accepted-fit flags, selected source directions, and fallback flags are preserved in `trajectory_checkpoint_values.csv`.",
            "",
            "## Reproduction",
            "",
            "From the repository root on Windows: `powershell -ExecutionPolicy Bypass -File experiments\\pydma_hybrid_trajectory_audit_v1\\reproduce.ps1`.",
            "",
        ]
    )
    (OUT / "trajectory_trend_report.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_shares(summary: pd.DataFrame, domain: str) -> None:
    methods = METHODS
    fig, ax = plt.subplots(figsize=(11, 5.5))
    data = summary[
        summary["evaluation_domain"].eq(domain)
        & summary["sensitivity"].eq("regular_only")
        & summary["checkpoint_scope"].eq("all_available")
        & summary["metric"].isin(METRICS)
    ]
    x = np.arange(len(METRICS), dtype=float)
    width = 0.18
    for index, method in enumerate(methods):
        subset = data[data["method"].eq(method)].set_index("metric")
        vals = [subset.loc[metric, "percentage_increasing_trend_trajectories"] if metric in subset.index else np.nan for metric in METRICS]
        ax.bar(x + (index - 1.5) * width, vals, width, label=method)
    ax.axhline(50, color="#a33", linestyle="--", linewidth=1, label="50% reference")
    ax.set_xticks(x, METRICS)
    ax.set_ylabel("Eligible trajectories with increasing overall trend (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{domain}: regular-RPT increasing-trend share")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"increasing_trend_share_{domain}.png", dpi=150)
    plt.close(fig)


def _plot_h1_examples(values: pd.DataFrame, per_cell: pd.DataFrame, domain: str) -> None:
    regular = values[
        values["evaluation_domain"].eq(domain)
        & values["method"].eq("H1 Hybrid")
        & values["valid_value"]
        & ~values["is_eol"]
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    for ax, metric in zip(axes, METRICS):
        selected = per_cell[
            per_cell["evaluation_domain"].eq(domain)
            & per_cell["sensitivity"].eq("regular_only")
            & per_cell["checkpoint_scope"].eq("all_available")
            & per_cell["method"].eq("H1 Hybrid")
            & per_cell["metric"].eq(metric)
        ]
        chosen = selected[
            selected["representative_increasing_rank"].notna()
            | selected["representative_decreasing_rank"].notna()
        ]
        for trajectory in chosen.itertuples():
            cell_rows = regular[
                regular["physical_cell_id"].eq(trajectory.physical_cell_id)
                & regular["metric"].eq(metric)
            ].sort_values("chronological_position")
            ax.scatter(
                cell_rows["rpt_index"],
                cell_rows["value_percent"],
                s=18,
                label=f"{trajectory.physical_cell_id} ({trajectory.first_to_last_change_percent:+.2f})",
            )
            records = cell_rows.to_dict("records")
            for left, right in zip(records, records[1:]):
                gap = _number(right.get("rpt_index")) - _number(left.get("rpt_index"))
                if math.isclose(gap, 1.0, rel_tol=0.0, abs_tol=1e-12):
                    ax.plot(
                        [left["rpt_index"], right["rpt_index"]],
                        [left["value_percent"], right["value_percent"]],
                        linewidth=0.9,
                        alpha=0.65,
                    )
        ax.set_title(metric)
        ax.set_xlabel("Verified RPT number")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel("Selected estimate (%)")
    fig.suptitle(f"{domain} — H1 Hybrid representative regular-RPT trajectories")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / f"h1_examples_{domain}.png", dpi=150)
    plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    hashes_before = {name: _sha256(INPUT_DIR / name) for name in INPUT_FILES}
    h1, h2, hashes = _input_rows()
    if hashes != hashes_before:
        raise RuntimeError("Hybrid v2 inputs changed while loading.")
    values = _mark_common_checkpoints(_build_checkpoint_values(h1, h2))
    per_cell, transitions = _build_trajectories(values)
    summary = _build_summary(per_cell)
    values.to_csv(OUT / "trajectory_checkpoint_values.csv", index=False)
    per_cell.to_csv(OUT / "trajectory_trend_per_cell.csv", index=False)
    transitions.to_csv(OUT / "trajectory_transition_details.csv", index=False)
    summary.to_csv(OUT / "trajectory_trend_summary.csv", index=False)
    for domain in DOMAIN_ORDER:
        _plot_shares(summary, domain)
        _plot_h1_examples(values, per_cell, domain)
    hashes_after = {name: _sha256(INPUT_DIR / name) for name in INPUT_FILES}
    if hashes_before != hashes_after:
        raise RuntimeError("A v2 source file changed during the trajectory audit.")
    _write_report(per_cell, summary, hashes_before, values)
    manifest = {
        "input_directory": str(INPUT_DIR),
        "input_sha256": hashes_before,
        "v2_inputs_unchanged_after_run": hashes_before == hashes_after,
        "verified_checkpoint_metric_rows": int(values[["checkpoint_key", "metric"]].drop_duplicates().shape[0]),
        "method_checkpoint_rows": len(values),
        "regular_rpt_checkpoint_rows": int((~values["is_eol"]).sum()),
        "eol_checkpoint_rows": int(values["is_eol"].sum()),
        "trajectory_rows": len(per_cell),
        "transition_rows": len(transitions),
        "pydma_optimizer_rerun": False,
        "source_results_modified": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
