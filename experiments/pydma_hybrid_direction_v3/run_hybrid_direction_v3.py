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
V1_DIR = ROOT / "experiments" / "pydma_hybrid_direction_v1"
V2_DIR = ROOT / "experiments" / "pydma_hybrid_direction_v2"
TRAJECTORY_V1_DIR = ROOT / "experiments" / "pydma_hybrid_trajectory_audit_v1"
REVIEW_DIR = ROOT / "experiments" / "dataset_review"
MANIFEST_DIR = ROOT / "data" / "manifests"
OUT = Path(__file__).resolve().parent

METRICS = ("LLI", "Anode_LAM", "Cathode_LAM")
DIRECTIONS = ("charge", "discharge")
METHODS = (
    "Charge-only",
    "Discharge-only",
    "H1 Hybrid v2",
    "H2 Hybrid v2",
    "H2 Hybrid v3",
)
DOMAINS = ("parts1_2_internal", "external_part3")
CELL_GROUP_COLUMNS = ["evaluation_domain", "group", "physical_cell_id", "metric"]

INPUTS = {
    "matched_charge_discharge_fits.csv": V1_DIR / "matched_charge_discharge_fits.csv",
    "direction_positive_rates.csv": V1_DIR / "direction_positive_rates.csv",
    "h1_hybrid_v2.csv": V2_DIR / "h1_hybrid_v2.csv",
    "h2_hybrid_v2.csv": V2_DIR / "h2_hybrid_v2.csv",
    "trajectory_trend_per_cell.csv": TRAJECTORY_V1_DIR / "trajectory_trend_per_cell.csv",
    "trajectory_trend_summary.csv": TRAJECTORY_V1_DIR / "trajectory_trend_summary.csv",
    "trajectory_checkpoint_values.csv": TRAJECTORY_V1_DIR / "trajectory_checkpoint_values.csv",
    "source_file_index.csv": REVIEW_DIR / "source_file_index.csv",
    "guide_cell_coverage.csv": REVIEW_DIR / "guide_cell_coverage.csv",
    "canonical_observation_manifest_v3.csv": MANIFEST_DIR / "canonical_observation_manifest_v3.csv",
    "unified_observation_manifest.csv": MANIFEST_DIR / "unified_observation_manifest.csv",
}


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


def _part3_cell_id(group: Any, cell: Any) -> str:
    group_value = _number(group)
    cell_value = _number(cell)
    if group_value is None or cell_value is None:
        raise ValueError(f"Part 3 inventory row has no numeric group/cell identity: {group}, {cell}")
    return f"G{int(group_value)}-Cell{int(cell_value)}"


def _observation_key(row: dict[str, Any]) -> str:
    part = _number(row.get("part"))
    rpt = _number(row.get("rpt_index"))
    physical_cell = _text(row.get("physical_cell_id"))
    if part is None:
        raise ValueError(f"Observation has no dataset part: {row}")
    if rpt is not None:
        return f"P{int(part)}|{physical_cell}|RPT{int(rpt)}"
    checkpoint = _text(row.get("checkpoint_id")) or _text(row.get("rpt_or_checkpoint"))
    if not checkpoint:
        raise ValueError(f"Unnumbered checkpoint lacks an identifier: {row}")
    return f"P{int(part)}|{physical_cell}|CHECKPOINT:{checkpoint}"


def _load_inputs() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    missing = [name for name, path in INPUTS.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required saved audit inputs are missing: " + ", ".join(missing))
    hashes = {name: _sha256(path) for name, path in INPUTS.items()}
    frames: dict[str, pd.DataFrame] = {}
    for name in (
        "matched_charge_discharge_fits.csv",
        "direction_positive_rates.csv",
        "h1_hybrid_v2.csv",
        "h2_hybrid_v2.csv",
        "trajectory_trend_per_cell.csv",
        "trajectory_trend_summary.csv",
        "trajectory_checkpoint_values.csv",
        "source_file_index.csv",
        "guide_cell_coverage.csv",
        "canonical_observation_manifest_v3.csv",
        "unified_observation_manifest.csv",
    ):
        frames[name] = pd.read_csv(INPUTS[name], low_memory=False)
    return frames, hashes


def _verified_hybrid_rows(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[frame["mapping_verified"].map(_truth)].copy()
    result["part"] = pd.to_numeric(result["part"], errors="coerce")
    result["group"] = pd.to_numeric(result["group"], errors="coerce")
    result["rpt_index"] = pd.to_numeric(result["rpt_index"], errors="coerce")
    if result["group"].isin({5, 6}).any():
        raise ValueError("Part 2 Groups 5-6 appeared in the v2 inputs; v3 must preserve the v2 scope.")
    key_columns = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "checkpoint_id",
    ]
    if "metric" in result.columns:
        key_columns.append("metric")
    if result.duplicated(key_columns).any():
        raise ValueError("A v2 input contains duplicated verified checkpoint/metric keys.")
    return result


def _accepted_metric_value(row: dict[str, Any], direction: str, metric: str) -> float | None:
    if not _truth(row.get(f"{direction}_accepted")):
        return None
    return _number(row.get(f"{direction}_{metric}_percent"))


def _rate_policy(h2_rows: pd.DataFrame, previous_rates: pd.DataFrame) -> pd.DataFrame:
    policy_rows = []
    for (domain, group, cell), cell_rows in h2_rows.groupby(
        ["evaluation_domain", "group", "physical_cell_id"], sort=True
    ):
        for metric in METRICS:
            metric_rows = cell_rows[cell_rows["metric"].eq(metric)].copy()
            rated_rows = metric_rows[~metric_rows["is_unnumbered_eol"].map(_truth)]
            paired: list[tuple[float, float]] = []
            charge_accepted = 0
            discharge_accepted = 0
            charge_single = 0
            discharge_single = 0
            for row in rated_rows.to_dict("records"):
                c_value = _accepted_metric_value(row, "charge", metric)
                d_value = _accepted_metric_value(row, "discharge", metric)
                if c_value is not None:
                    charge_accepted += 1
                if d_value is not None:
                    discharge_accepted += 1
                if c_value is not None and d_value is not None:
                    paired.append((c_value, d_value))
                elif c_value is not None:
                    charge_single += 1
                elif d_value is not None:
                    discharge_single += 1
            n = len(paired)
            c_nonnegative = sum(c_value >= 0 for c_value, _ in paired)
            d_nonnegative = sum(d_value >= 0 for _, d_value in paired)
            if n == 0:
                c_rate = d_rate = np.nan
                primary = "undetermined"
                reason = "no_paired_accepted_checkpoints"
            else:
                c_rate = c_nonnegative / n
                d_rate = d_nonnegative / n
                if c_nonnegative > d_nonnegative:
                    primary, reason = "charge", "higher_charge_nonnegative_rate"
                elif d_nonnegative > c_nonnegative:
                    primary, reason = "discharge", "higher_discharge_nonnegative_rate"
                else:
                    primary, reason = "charge", "tie_prefer_charge"
            policy_rows.append(
                {
                    "evaluation_domain": domain,
                    "group": int(group),
                    "physical_cell_id": cell,
                    "metric": metric,
                    "paired_accepted_checkpoints": n,
                    "charge_nonnegative_count": c_nonnegative,
                    "charge_negative_count": n - c_nonnegative,
                    "charge_positive_rate": c_rate,
                    "discharge_nonnegative_count": d_nonnegative,
                    "discharge_negative_count": n - d_nonnegative,
                    "discharge_positive_rate": d_rate,
                    "charge_accepted_metric_checkpoints": charge_accepted,
                    "discharge_accepted_metric_checkpoints": discharge_accepted,
                    "charge_single_direction_only_checkpoints": charge_single,
                    "discharge_single_direction_only_checkpoints": discharge_single,
                    "primary_direction": primary,
                    "selection_reason": reason,
                }
            )
    policies = pd.DataFrame(policy_rows)
    _validate_rate_parity(policies, previous_rates)
    return policies


def _validate_rate_parity(policies: pd.DataFrame, previous_rates: pd.DataFrame) -> None:
    saved = previous_rates.copy()
    key_columns = ["physical_cell_id", "metric"]
    if saved.duplicated(key_columns).any() or policies.duplicated(key_columns).any():
        raise ValueError("Direction-rate policy keys are not unique.")
    joined = policies.merge(saved, on=key_columns, how="left", suffixes=("_v3", "_v1"), validate="one_to_one")
    if joined["paired_accepted_checkpoints_v1"].isna().any():
        raise ValueError("V1 direction rates are missing for a v3 cell/metric policy.")
    for row in joined.to_dict("records"):
        if int(row["paired_accepted_checkpoints_v3"]) != int(row["paired_accepted_checkpoints_v1"]):
            raise ValueError(f"V3 paired-fit denominator differs from v1 for {row['physical_cell_id']} {row['metric']}.")
        if int(row["charge_nonnegative_count_v3"]) != int(row["charge_nonnegative_count_v1"]):
            raise ValueError(f"V3 Charge positive count differs from v1 for {row['physical_cell_id']} {row['metric']}.")
        if int(row["discharge_nonnegative_count_v3"]) != int(row["discharge_nonnegative_count_v1"]):
            raise ValueError(f"V3 Discharge positive count differs from v1 for {row['physical_cell_id']} {row['metric']}.")
        for new_col, old_col in (
            ("charge_positive_rate", "charge_nonnegative_rate"),
            ("discharge_positive_rate", "discharge_nonnegative_rate"),
        ):
            new_value = _number(row.get(new_col))
            old_value = _number(row.get(old_col))
            if (new_value is None) != (old_value is None) or (
                new_value is not None
                and old_value is not None
                and not math.isclose(new_value, old_value, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise ValueError(f"V3 rate differs from v1 for {row['physical_cell_id']} {row['metric']} {new_col}.")
        old_direction = _text(row.get("h2_primary_direction"))
        expected = "charge" if old_direction == "tie" else old_direction
        if row["primary_direction"] != expected:
            raise ValueError(f"V3 changed more than the tie-break rule for {row['physical_cell_id']} {row['metric']}.")


def _select_v3_row(row: dict[str, Any], primary: str) -> dict[str, Any]:
    metric = row["metric"]
    charge = _accepted_metric_value(row, "charge", metric)
    discharge = _accepted_metric_value(row, "discharge", metric)
    if primary == "undetermined":
        return {
            "h2_v3_selected_percent": np.nan,
            "h2_v3_selected_accepted": False,
            "h2_v3_source_direction": "",
            "h2_v3_fallback_applied": False,
            "h2_v3_fallback_reason": "no_paired_accepted_checkpoints",
            "h2_v3_selected_fit_status": "",
            "h2_v3_selected_fit_rmse_mV": np.nan,
            "h2_v3_selected_fit_accepted_runs": np.nan,
        }
    primary_value = charge if primary == "charge" else discharge
    opposite_direction = "discharge" if primary == "charge" else "charge"
    opposite_value = discharge if primary == "charge" else charge
    direction = primary
    value = primary_value
    fallback = False
    if primary_value is None:
        value = None
        reason = "primary_fit_missing_or_rejected"
        direction = ""
    elif primary_value >= 0:
        reason = "primary_nonnegative"
    elif opposite_value is not None and opposite_value >= 0:
        value = opposite_value
        direction = opposite_direction
        fallback = True
        reason = "negative_primary_positive_opposite"
    elif opposite_value is not None:
        reason = "unresolved_negative"
    else:
        reason = "fallback_unavailable"
    status = row.get(f"{direction}_status", "") if direction else ""
    rmse = row.get(f"{direction}_rmse_mV") if direction else np.nan
    runs = row.get(f"{direction}_accepted_runs") if direction else np.nan
    selected_value = _number(value)
    return {
        "h2_v3_selected_percent": selected_value if selected_value is not None else np.nan,
        "h2_v3_selected_accepted": selected_value is not None,
        "h2_v3_source_direction": direction,
        "h2_v3_fallback_applied": fallback,
        "h2_v3_fallback_reason": reason,
        "h2_v3_selected_fit_status": status,
        "h2_v3_selected_fit_rmse_mV": _number(rmse),
        "h2_v3_selected_fit_accepted_runs": _number(runs),
    }


def _build_v3_results(h2_rows: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    lookup = {
        (row["physical_cell_id"], row["metric"]): row
        for row in rates.to_dict("records")
    }
    rows = []
    for row in h2_rows.to_dict("records"):
        policy = lookup[(row["physical_cell_id"], row["metric"])]
        selected = _select_v3_row(row, policy["primary_direction"])
        rows.append(
            {
                **row,
                "h2_v3_primary_direction": policy["primary_direction"],
                "h2_v3_selection_reason": policy["selection_reason"],
                "h2_v3_charge_positive_rate": policy["charge_positive_rate"],
                "h2_v3_discharge_positive_rate": policy["discharge_positive_rate"],
                "h2_v3_paired_accepted_checkpoints": policy["paired_accepted_checkpoints"],
                **selected,
            }
        )
    result = pd.DataFrame(rows)
    key_columns = ["part", "group", "physical_cell_id", "rpt_index", "rpt_or_checkpoint", "checkpoint_id", "metric"]
    if result.duplicated(key_columns).any():
        raise ValueError("H2 v3 produced duplicate checkpoint/metric keys.")
    result["h2_v3_temporal_decrease_warning"] = False
    result["h2_v3_previous_selected_checkpoint"] = ""
    result["h2_v3_previous_selected_percent"] = np.nan
    for _, indices in result.groupby(["physical_cell_id", "metric"], sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            ["chronological_position", "position", "part", "rpt_index"],
            kind="stable",
            na_position="last",
        )
        previous: dict[str, Any] | None = None
        for index, row in ordered.iterrows():
            current_value = _number(row.get("h2_v3_selected_percent"))
            if previous is not None and current_value is not None:
                delta = current_value - float(previous["value"])
                result.at[index, "h2_v3_temporal_decrease_warning"] = delta < -1e-12
                result.at[index, "h2_v3_previous_selected_checkpoint"] = previous["checkpoint"]
                result.at[index, "h2_v3_previous_selected_percent"] = previous["value"]
            if current_value is not None:
                previous = {
                    "checkpoint": row.get("rpt_or_checkpoint"),
                    "value": current_value,
                }
    return result


def _selection_audits(v3: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = v3.to_dict("records")
    fallback_rows = []
    eol_rows = []
    for row in rows:
        metric = row["metric"]
        primary = row["h2_v3_primary_direction"]
        primary_value = _number(row.get(f"{primary}_{metric}_percent")) if primary in DIRECTIONS else None
        opposite = "discharge" if primary == "charge" else "charge" if primary == "discharge" else ""
        if _truth(row["is_eol"]):
            eol_rows.append(
                {
                    "evaluation_domain": row["evaluation_domain"],
                    "part": row["part"],
                    "group": row["group"],
                    "physical_cell_id": row["physical_cell_id"],
                    "checkpoint_id": row["checkpoint_id"],
                    "rpt_index": row["rpt_index"],
                    "rpt_or_checkpoint": row["rpt_or_checkpoint"],
                    "is_unnumbered_eol": row["is_unnumbered_eol"],
                    "metric": metric,
                    "primary_direction": primary,
                    "selected_percent": row["h2_v3_selected_percent"],
                    "selected_accepted": row["h2_v3_selected_accepted"],
                    "source_direction": row["h2_v3_source_direction"],
                    "fallback_applied": row["h2_v3_fallback_applied"],
                    "fallback_reason": row["h2_v3_fallback_reason"],
                    "charge_accepted": row["charge_accepted"],
                    "charge_status": row["charge_status"],
                    "charge_rmse_mV": row["charge_rmse_mV"],
                    "discharge_accepted": row["discharge_accepted"],
                    "discharge_status": row["discharge_status"],
                    "discharge_rmse_mV": row["discharge_rmse_mV"],
                    "previous_selected_checkpoint": row.get("h2_v3_previous_selected_checkpoint"),
                    "previous_selected_percent": row.get("h2_v3_previous_selected_percent"),
                    "temporal_decrease_warning": row.get("h2_v3_temporal_decrease_warning"),
                }
            )
        if primary_value is not None and primary_value < 0:
            fallback_rows.append(
                {
                    "evaluation_domain": row["evaluation_domain"],
                    "part": row["part"],
                    "group": row["group"],
                    "physical_cell_id": row["physical_cell_id"],
                    "checkpoint_id": row["checkpoint_id"],
                    "rpt_index": row["rpt_index"],
                    "rpt_or_checkpoint": row["rpt_or_checkpoint"],
                    "is_eol": row["is_eol"],
                    "metric": metric,
                    "primary_direction": primary,
                    "primary_value_percent": primary_value,
                    "primary_fit_status": row.get(f"{primary}_status"),
                    "primary_fit_accepted": row.get(f"{primary}_accepted"),
                    "primary_fit_rmse_mV": row.get(f"{primary}_rmse_mV"),
                    "opposite_direction": opposite,
                    "opposite_value_percent": _number(row.get(f"{opposite}_{metric}_percent")) if opposite else None,
                    "opposite_fit_status": row.get(f"{opposite}_status") if opposite else "",
                    "opposite_fit_accepted": row.get(f"{opposite}_accepted") if opposite else False,
                    "selected_percent": row["h2_v3_selected_percent"],
                    "selected_source_direction": row["h2_v3_source_direction"],
                    "fallback_applied": row["h2_v3_fallback_applied"],
                    "fallback_reason": row["h2_v3_fallback_reason"],
                }
            )
    audit_columns = [
        "evaluation_domain",
        "part",
        "group",
        "physical_cell_id",
        "checkpoint_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "is_eol",
        "metric",
        "primary_direction",
        "primary_value_percent",
        "primary_fit_status",
        "primary_fit_accepted",
        "primary_fit_rmse_mV",
        "opposite_direction",
        "opposite_value_percent",
        "opposite_fit_status",
        "opposite_fit_accepted",
        "selected_percent",
        "selected_source_direction",
        "fallback_applied",
        "fallback_reason",
    ]
    audit = pd.DataFrame(fallback_rows, columns=audit_columns)
    eol_audit = pd.DataFrame(eol_rows)
    policy_audit = _build_direction_audit(v3)
    return policy_audit, audit, eol_audit


def _build_direction_audit(v3: pd.DataFrame) -> pd.DataFrame:
    fields = [
        "evaluation_domain",
        "group",
        "physical_cell_id",
        "metric",
        "h2_v3_paired_accepted_checkpoints",
        "h2_v3_charge_positive_rate",
        "h2_v3_discharge_positive_rate",
        "h2_v3_primary_direction",
        "h2_v3_selection_reason",
    ]
    result = v3[fields].drop_duplicates().rename(
        columns={
            "h2_v3_paired_accepted_checkpoints": "paired_accepted_checkpoints",
            "h2_v3_charge_positive_rate": "charge_positive_rate",
            "h2_v3_discharge_positive_rate": "discharge_positive_rate",
            "h2_v3_primary_direction": "primary_direction",
            "h2_v3_selection_reason": "selection_reason",
        }
    )
    accepted_counts = []
    for row in result.to_dict("records"):
        cell_rows = v3[
            v3["physical_cell_id"].eq(row["physical_cell_id"])
            & v3["metric"].eq(row["metric"])
        ]
        regular = cell_rows[~cell_rows["is_unnumbered_eol"].map(_truth)]
        charge_count = sum(
            _accepted_metric_value(record, "charge", row["metric"]) is not None
            for record in regular.to_dict("records")
        )
        discharge_count = sum(
            _accepted_metric_value(record, "discharge", row["metric"]) is not None
            for record in regular.to_dict("records")
        )
        accepted_counts.append(
            {
                "charge_accepted_metric_checkpoints_in_rate_scope": charge_count,
                "discharge_accepted_metric_checkpoints_in_rate_scope": discharge_count,
                "single_direction_only_checkpoint_count": max(charge_count + discharge_count - 2 * row["paired_accepted_checkpoints"], 0),
            }
        )
    for name in accepted_counts[0] if accepted_counts else ():
        result[name] = [item[name] for item in accepted_counts]
    return result


def _h2_v2_v3_comparison(h2: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    key_columns = ["part", "group", "physical_cell_id", "rpt_index", "rpt_or_checkpoint", "checkpoint_id", "metric"]
    left = h2[
        key_columns
        + [
            "evaluation_domain",
            "is_eol",
            "is_unnumbered_eol",
            "selected_percent",
            "selected_accepted",
            "source_direction",
            "fallback_applied",
            "fallback_reason",
        ]
    ].rename(
        columns={
            "selected_percent": "h2_v2_selected_percent",
            "selected_accepted": "h2_v2_selected_accepted",
            "source_direction": "h2_v2_source_direction",
            "fallback_applied": "h2_v2_fallback_applied",
            "fallback_reason": "h2_v2_selection_reason",
        }
    )
    right = v3[
        key_columns
        + [
            "h2_v3_selected_percent",
            "h2_v3_selected_accepted",
            "h2_v3_source_direction",
            "h2_v3_fallback_applied",
            "h2_v3_fallback_reason",
            "h2_v3_primary_direction",
        ]
    ]
    result = left.merge(right, on=key_columns, how="outer", validate="one_to_one")
    result["newly_recovered"] = (
        ~result["h2_v2_selected_accepted"].map(_truth)
        & result["h2_v3_selected_accepted"].map(_truth)
    )
    result["selection_lost"] = (
        result["h2_v2_selected_accepted"].map(_truth)
        & ~result["h2_v3_selected_accepted"].map(_truth)
    )
    v2_value = pd.to_numeric(result["h2_v2_selected_percent"], errors="coerce")
    v3_value = pd.to_numeric(result["h2_v3_selected_percent"], errors="coerce")
    result["numeric_value_changed"] = ~(
        (v2_value.isna() & v3_value.isna())
        | (
            v2_value.notna()
            & v3_value.notna()
            & np.isclose(v2_value.fillna(0), v3_value.fillna(0), atol=1e-10, rtol=0)
        )
    )
    result["v3_minus_v2_percent"] = v3_value - v2_value
    return result


def _inventory_audit(
    frames: dict[str, pd.DataFrame],
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    v3: pd.DataFrame,
    prior_trajectories: pd.DataFrame,
    v3_trajectories: pd.DataFrame,
) -> pd.DataFrame:
    source = frames["source_file_index.csv"]
    source = source[(source["part"] == 3) & source["group"].notna() & source["cell"].notna()].copy()
    source["physical_cell_id"] = [
        _part3_cell_id(group, cell) for group, cell in zip(source["group"], source["cell"])
    ]
    cells = source[["group", "cell", "physical_cell_id"]].drop_duplicates().sort_values(
        ["group", "cell"]
    )
    expected_cells = {
        "G7-Cell4",
        "G7-Cell19",
        "G7-Cell27",
        "G8-Cell5",
        "G8-Cell18",
        "G8-Cell20",
        "G9-Cell1",
        "G9-Cell8",
        "G9-Cell9",
        "G10-Cell2",
        "G10-Cell22",
        "G10-Cell25",
    }
    if set(cells["physical_cell_id"]) != expected_cells:
        raise ValueError("Part 3 source-inventory cell identities differ from the 12-cell audited inventory.")

    guide = frames["guide_cell_coverage.csv"]
    guide = guide[guide["part"].eq(3)].copy()
    canonical = frames["canonical_observation_manifest_v3.csv"]
    canonical = canonical[
        canonical["dataset_part"].map(lambda part: "3" in _text(part).split(";"))
        & canonical["record_type"].eq("cycling_guide_record")
    ].copy()
    if set(canonical["physical_cell_id"].dropna()) != expected_cells:
        raise ValueError("Canonical Part 3 physical-cell identities do not reconcile with the source inventory.")
    unified = frames["unified_observation_manifest.csv"]
    unified = unified[unified["dataset_part"].eq(3)].copy()
    matched = frames["matched_charge_discharge_fits.csv"]
    matched = matched[(matched["part"] == 3)].copy()
    matched_identities = set(
        matched.loc[matched["mapping_verified"].map(_truth), "physical_cell_id"].dropna()
    )
    if matched_identities != expected_cells:
        raise ValueError("Saved-fit mappings do not reconcile with all 12 Part 3 inventory identities.")
    guide_identities = {
        _part3_cell_id(row["group"], row["cell"])
        for row in guide.to_dict("records")
        if _number(row.get("group")) is not None and _number(row.get("cell")) is not None
    }
    if guide_identities != expected_cells - {"G10-Cell2"}:
        raise ValueError("Part 3 Guide identities differ from the audited 11-cell Guide coverage.")

    v2_h1 = h1[h1["part"].eq(3)].copy()
    v2_h2 = h2[h2["part"].eq(3)].copy()
    v2_h1["is_eol_bool"] = v2_h1["is_eol"].map(_truth)
    v2_h2["is_eol_bool"] = v2_h2["is_eol"].map(_truth)
    v3_p3 = v3[v3["part"].eq(3)].copy()
    v3_p3["is_eol_bool"] = v3_p3["is_eol"].map(_truth)

    rows = []
    for item in cells.to_dict("records"):
        cell = item["physical_cell_id"]
        group_id = int(item["group"])
        cell_id = int(item["cell"])
        inventory_rows = source[source["physical_cell_id"].eq(cell)]
        guide_rows = guide[(guide["group"] == group_id) & (guide["cell"] == cell_id)]
        canonical_rows = canonical[canonical["physical_cell_id"].eq(cell)]
        canonical_rpts = canonical_rows[pd.to_numeric(canonical_rows["rpt_number"], errors="coerce").notna()]
        unified_rows = unified[unified["physical_cell_id"].eq(cell)]
        matched_rows = matched[matched["physical_cell_id"].eq(cell)]
        mapped = matched_rows[matched_rows["mapping_verified"].map(_truth)]
        regular_mapped = mapped[mapped["rpt_index"].notna() & ~mapped["is_eol"].map(_truth)]
        eol_mapped = mapped[mapped["is_eol"].map(_truth)]

        h1_cell = v2_h1[v2_h1["physical_cell_id"].eq(cell)]
        h2_cell = v2_h2[v2_h2["physical_cell_id"].eq(cell)]
        v3_cell = v3_p3[v3_p3["physical_cell_id"].eq(cell)]
        trajectory_rows = prior_trajectories[
            prior_trajectories["evaluation_domain"].eq("external_part3")
            & prior_trajectories["physical_cell_id"].eq(cell)
            & prior_trajectories["sensitivity"].eq("regular_only")
            & prior_trajectories["checkpoint_scope"].eq("all_available")
        ]
        v3_traj_cell = v3_trajectories[
            v3_trajectories["evaluation_domain"].eq("external_part3")
            & v3_trajectories["physical_cell_id"].eq(cell)
            & v3_trajectories["sensitivity"].eq("regular_only")
            & v3_trajectories["checkpoint_scope"].eq("all_available")
        ]
        row = {
            "physical_cell_id": cell,
            "group": group_id,
            "cell_number": cell_id,
            "source_inventory_record_count": len(inventory_rows),
            "source_inventory_csv_count": int(inventory_rows["extension"].str.casefold().eq(".csv").sum()),
            "source_inventory_mat_count": int(inventory_rows["extension"].str.casefold().eq(".mat").sum()),
            "source_inventory_procedure_count": int(inventory_rows["procedure"].dropna().nunique()),
            "guide_cell_entry_present": not guide_rows.empty,
            "guide_rpt_record_count": int(guide_rows["rpt_records"].sum()) if not guide_rows.empty else 0,
            "canonical_cycling_record_count": len(canonical_rows),
            "canonical_numbered_rpt_count": len(canonical_rpts),
            "canonical_verified_unique_source_rpt_count": int(canonical_rpts["mapping_status"].eq("verified_unique_source").sum()),
            "canonical_user_confirmed_identity_rpt_count": int(canonical_rpts["mapping_status"].eq("user_confirmed_identity_by_group_membership").sum()),
            "canonical_step4_not_audited_rpt_count": int(canonical_rpts["step4_discharge_status"].eq("not_audited").sum()),
            "canonical_mapping_statuses": ";".join(sorted(set(canonical_rpts["mapping_status"].dropna().astype(str)))),
            "canonical_cell_identity_statuses": ";".join(sorted(set(canonical_rpts["cell_identity_status"].dropna().astype(str)))),
            "unified_manifest_measurement_count": int(unified_rows["measurement_id"].notna().sum()),
            "v1_mapping_verified_regular_rpt_count": len(regular_mapped),
            "v1_mapping_verified_eol_count": len(eol_mapped),
            "v1_all_matched_regular_rpt_count": int((matched_rows["rpt_index"].notna() & ~matched_rows["is_eol"].map(_truth)).sum()),
            "v1_accepted_charge_fit_count": int(regular_mapped["charge_accepted"].map(_truth).sum()),
            "v1_accepted_discharge_fit_count": int(regular_mapped["discharge_accepted"].map(_truth).sum()),
            "v1_accepted_charge_eol_fit_count": int(eol_mapped["charge_accepted"].map(_truth).sum()),
            "v1_accepted_discharge_eol_fit_count": int(eol_mapped["discharge_accepted"].map(_truth).sum()),
        }
        for metric in METRICS:
            row[f"charge_accepted_{metric}_regular_count"] = sum(
                _accepted_metric_value(record, "charge", metric) is not None
                for record in regular_mapped.to_dict("records")
            )
            row[f"discharge_accepted_{metric}_regular_count"] = sum(
                _accepted_metric_value(record, "discharge", metric) is not None
                for record in regular_mapped.to_dict("records")
            )
            for label, frame in (("h1_v2", h1_cell), ("h2_v2", h2_cell), ("h2_v3", v3_cell)):
                metric_rows = frame[frame["metric"].eq(metric)]
                accepted_column = "selected_accepted" if label.endswith("_v2") else "h2_v3_selected_accepted"
                value_column = "selected_percent" if label.endswith("_v2") else "h2_v3_selected_percent"
                row[f"{label}_{metric}_selected_regular_count"] = int(
                    (
                        metric_rows[accepted_column].map(_truth)
                        & ~metric_rows["is_eol_bool"]
                        & pd.to_numeric(metric_rows[value_column], errors="coerce").notna()
                    ).sum()
                )
                row[f"{label}_{metric}_selected_eol_count"] = int(
                    (
                        metric_rows[accepted_column].map(_truth)
                        & metric_rows["is_eol_bool"]
                        & pd.to_numeric(metric_rows[value_column], errors="coerce").notna()
                    ).sum()
                )
            for label, trajectory_frame, method in (
                ("h1_v2", trajectory_rows, "H1 Hybrid"),
                ("h2_v2", trajectory_rows, "H2 Hybrid"),
                ("h2_v3", v3_traj_cell, "H2 Hybrid v3"),
            ):
                metric_traj = trajectory_frame[
                    trajectory_frame["metric"].eq(metric) & trajectory_frame["method"].eq(method)
                ]
                row[f"{label}_{metric}_valid_regular_checkpoints"] = (
                    int(metric_traj["valid_checkpoint_count"].iloc[0]) if not metric_traj.empty else 0
                )
                row[f"{label}_{metric}_trajectory_eligible"] = bool(
                    not metric_traj.empty and _truth(metric_traj["eligible_for_trend_classification"].iloc[0])
                )
        row["guide_identity_status"] = (
            "guide_cell_listed"
            if not guide_rows.empty
            else "no_cell_entry_in_guide_user_confirmed_in_canonical"
        )
        row["h1_v2_eligible_metric_trajectory_count"] = sum(
            _truth(row[f"h1_v2_{metric}_trajectory_eligible"]) for metric in METRICS
        )
        row["h2_v2_eligible_metric_trajectory_count"] = sum(
            _truth(row[f"h2_v2_{metric}_trajectory_eligible"]) for metric in METRICS
        )
        row["h2_v3_eligible_metric_trajectory_count"] = sum(
            _truth(row[f"h2_v3_{metric}_trajectory_eligible"]) for metric in METRICS
        )
        row["h2_v2_selected_any_metric"] = any(
            row[f"h2_v2_{metric}_selected_regular_count"] > 0
            or row[f"h2_v2_{metric}_selected_eol_count"] > 0
            for metric in METRICS
        )
        row["h2_v3_selected_any_metric"] = any(
            row[f"h2_v3_{metric}_selected_regular_count"] > 0
            or row[f"h2_v3_{metric}_selected_eol_count"] > 0
            for metric in METRICS
        )
        if cell == "G10-Cell2":
            row["exclusion_reason"] = (
                "No accepted saved PyDMA fit exists: all 10 mapped RPT rows have "
                "charge_status=discharge_status=no_saved_fit_record and both acceptance flags false. "
                "The Guide has no cell-number entry for the TPG10 chain; canonical identity is user-confirmed "
                "by Group 10 membership, and Step 4 is not audited. Therefore H2 v3 has no accepted paired "
                "or single-direction estimate and no trajectory values."
            )
        else:
            row["exclusion_reason"] = ""
        rows.append(row)
    result = pd.DataFrame(rows)
    if set(result["physical_cell_id"]) != expected_cells:
        raise ValueError("Part 3 inventory audit did not account for every expected physical cell.")
    user_confirmed = result[result["physical_cell_id"].eq("G10-Cell2")].iloc[0]
    if (
        int(user_confirmed["canonical_user_confirmed_identity_rpt_count"]) != 10
        or int(user_confirmed["canonical_step4_not_audited_rpt_count"]) != 10
        or int(user_confirmed["v1_mapping_verified_regular_rpt_count"]) != 10
    ):
        raise ValueError("G10-Cell2 canonical identity, Step 4 audit, or saved mapping counts changed.")
    return result


def _method_value(row: dict[str, Any], method: str) -> dict[str, Any]:
    metric = row["metric"]
    if method == "Charge-only":
        source = "charge"
        candidate = _number(row.get(f"charge_{metric}_percent"))
        accepted = _truth(row.get("charge_accepted"))
        fit_status = row.get("charge_status", "")
        fit_rmse = _number(row.get("charge_rmse_mV"))
        accepted_runs = _number(row.get("charge_accepted_runs"))
        primary = "charge"
        fallback = False
    elif method == "Discharge-only":
        source = "discharge"
        candidate = _number(row.get(f"discharge_{metric}_percent"))
        accepted = _truth(row.get("discharge_accepted"))
        fit_status = row.get("discharge_status", "")
        fit_rmse = _number(row.get("discharge_rmse_mV"))
        accepted_runs = _number(row.get("discharge_accepted_runs"))
        primary = "discharge"
        fallback = False
    elif method == "H1 Hybrid v2":
        source = _text(row.get("source_direction"))
        candidate = _number(row.get("selected_percent"))
        accepted = _truth(row.get("selected_accepted"))
        fit_status = row.get("selected_fit_status", "")
        fit_rmse = _number(row.get("selected_fit_rmse_mV"))
        accepted_runs = _number(row.get("selected_fit_accepted_runs"))
        primary = _text(row.get("h1_primary_direction"))
        fallback = _truth(row.get("fallback_applied"))
    elif method == "H2 Hybrid v2":
        source = _text(row.get("source_direction"))
        candidate = _number(row.get("selected_percent"))
        accepted = _truth(row.get("selected_accepted"))
        fit_status = row.get("selected_fit_status", "")
        fit_rmse = _number(row.get("selected_fit_rmse_mV"))
        accepted_runs = _number(row.get("selected_fit_accepted_runs"))
        primary = _text(row.get("h2_primary_direction"))
        fallback = _truth(row.get("fallback_applied"))
    else:
        source = _text(row.get("h2_v3_source_direction"))
        candidate = _number(row.get("h2_v3_selected_percent"))
        accepted = _truth(row.get("h2_v3_selected_accepted"))
        fit_status = row.get("h2_v3_selected_fit_status", "")
        fit_rmse = _number(row.get("h2_v3_selected_fit_rmse_mV"))
        accepted_runs = _number(row.get("h2_v3_selected_fit_accepted_runs"))
        primary = _text(row.get("h2_v3_primary_direction"))
        fallback = _truth(row.get("h2_v3_fallback_applied"))
    valid = accepted and candidate is not None and _truth(row.get("mapping_verified"))
    return {
        "method": method,
        "candidate_value_percent": candidate,
        "value_percent": candidate if valid else np.nan,
        "valid_value": valid,
        "fit_accepted": accepted,
        "fit_status": fit_status,
        "fit_rmse_mV": fit_rmse,
        "fit_accepted_runs": accepted_runs,
        "source_direction": source,
        "primary_direction": primary,
        "fallback_applied": fallback,
    }


def _build_method_values(
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    v3: pd.DataFrame,
) -> pd.DataFrame:
    sources = (
        ("h1_hybrid_v2.csv", h1, ("Charge-only", "Discharge-only", "H1 Hybrid v2")),
        ("h2_hybrid_v2.csv", h2, ("H2 Hybrid v2",)),
        ("h2_v3_results.csv", v3, ("H2 Hybrid v3",)),
    )
    rows = []
    for source_name, frame, methods in sources:
        for source_row in frame.to_dict("records"):
            for method in methods:
                row = {
                    "checkpoint_key": _observation_key(source_row),
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
                    "temporal_decrease_warning": (
                        source_row.get("h2_v3_temporal_decrease_warning")
                        if method == "H2 Hybrid v3"
                        else source_row.get("temporal_decrease_warning")
                    ),
                    **_method_value(source_row, method),
                }
                for direction in DIRECTIONS:
                    row[f"{direction}_accepted"] = _truth(source_row.get(f"{direction}_accepted"))
                    row[f"{direction}_status"] = source_row.get(f"{direction}_status", "")
                    row[f"{direction}_rmse_mV"] = _number(source_row.get(f"{direction}_rmse_mV"))
                    row[f"{direction}_accepted_runs"] = _number(source_row.get(f"{direction}_accepted_runs"))
                    row[f"original_{direction}_estimate_percent"] = _number(
                        source_row.get(f"{direction}_{source_row['metric']}_percent")
                    )
                row["h1_primary_direction"] = _text(source_row.get("h1_primary_direction"))
                row["h2_primary_direction"] = _text(source_row.get("h2_primary_direction"))
                row["h2_v3_selection_reason"] = _text(source_row.get("h2_v3_selection_reason"))
                row["h2_v3_fallback_reason"] = _text(source_row.get("h2_v3_fallback_reason"))
                rows.append(row)
    values = pd.DataFrame(rows)
    if values.duplicated(["method", "checkpoint_key", "metric"]).any():
        raise ValueError("Method comparison rows contain duplicate method/checkpoint/metric keys.")
    return values


def _mark_common(values: pd.DataFrame) -> pd.DataFrame:
    result = values.copy()
    result["common_regular_checkpoint"] = False
    result["common_eol_inclusive_checkpoint"] = False
    for include_eol, column in (
        (False, "common_regular_checkpoint"),
        (True, "common_eol_inclusive_checkpoint"),
    ):
        valid = result[result["valid_value"]]
        if not include_eol:
            valid = valid[~valid["is_eol"]]
        for key, group in valid.groupby(CELL_GROUP_COLUMNS, sort=False, dropna=False):
            per_method = [
                set(group.loc[group["method"].eq(method), "checkpoint_key"].astype(str))
                for method in METHODS
            ]
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
                result.loc[mask, column] = True
    return result


def _delta_classification(delta: float) -> str:
    if math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
        return "tie"
    return "increase" if delta > 0 else "decrease"


def _make_transition(left: dict[str, Any], right: dict[str, Any], sensitivity: str, scope: str) -> dict[str, Any]:
    delta = float(right["value_percent"]) - float(left["value_percent"])
    left_rpt = _number(left.get("rpt_index"))
    right_rpt = _number(right.get("rpt_index"))
    eol_transition = _truth(left.get("is_eol")) or _truth(right.get("is_eol"))
    rpt_gap = right_rpt - left_rpt if left_rpt is not None and right_rpt is not None else None
    consecutive = (
        not eol_transition
        and _truth(left.get("mapping_verified"))
        and _truth(right.get("mapping_verified"))
        and rpt_gap is not None
        and math.isclose(rpt_gap, 1.0, rel_tol=0.0, abs_tol=1e-12)
    )
    transition_type = (
        "eol_adjacent"
        if eol_transition
        else "consecutive_verified_rpt"
        if consecutive
        else "observed_gap"
    )
    return {
        "evaluation_domain": left["evaluation_domain"],
        "group": left["group"],
        "physical_cell_id": left["physical_cell_id"],
        "method": left["method"],
        "metric": left["metric"],
        "sensitivity": sensitivity,
        "checkpoint_scope": scope,
        "left_checkpoint_key": left["checkpoint_key"],
        "right_checkpoint_key": right["checkpoint_key"],
        "left_checkpoint": left["rpt_or_checkpoint"],
        "right_checkpoint": right["rpt_or_checkpoint"],
        "left_part": left["part"],
        "right_part": right["part"],
        "left_rpt_index": left_rpt,
        "right_rpt_index": right_rpt,
        "rpt_gap": rpt_gap,
        "missing_intermediate_rpt_count": max(int(rpt_gap) - 1, 0) if rpt_gap is not None else np.nan,
        "left_is_eol": _truth(left.get("is_eol")),
        "right_is_eol": _truth(right.get("is_eol")),
        "is_eol_transition": eol_transition,
        "left_value_percent": float(left["value_percent"]),
        "right_value_percent": float(right["value_percent"]),
        "delta_percent": delta,
        "transition_direction": _delta_classification(delta),
        "transition_type": transition_type,
        "is_consecutive_verified_rpt_transition": consecutive,
        "left_source_direction": left["source_direction"],
        "right_source_direction": right["source_direction"],
        "left_primary_direction": left["primary_direction"],
        "right_primary_direction": right["primary_direction"],
        "left_fallback_applied": left["fallback_applied"],
        "right_fallback_applied": right["fallback_applied"],
        "left_fit_status": left["fit_status"],
        "right_fit_status": right["fit_status"],
        "left_fit_rmse_mV": left["fit_rmse_mV"],
        "right_fit_rmse_mV": right["fit_rmse_mV"],
        "left_temporal_decrease_warning": left["temporal_decrease_warning"],
        "right_temporal_decrease_warning": right["temporal_decrease_warning"],
    }


def _trajectory_stats(
    trajectory: pd.DataFrame,
    sensitivity: str,
    scope: str,
    identity: dict[str, Any],
    verified_rpts: set[int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ordered = trajectory.sort_values(
        ["chronological_position", "position", "part", "rpt_index"],
        kind="stable",
        na_position="last",
    )
    records = ordered.to_dict("records")
    transitions = [
        _make_transition(left, right, sensitivity, scope)
        for left, right in zip(records, records[1:])
    ]
    n = len(records)
    values = [float(row["value_percent"]) for row in records]
    sequence_positions = [float(row["chronological_position"]) for row in records]
    directions = [row["source_direction"] for row in records]
    observed_directions = [row["transition_direction"] for row in transitions]
    consecutive = [
        row["transition_direction"]
        for row in transitions
        if row["is_consecutive_verified_rpt_transition"]
    ]
    increases = sum(item == "increase" for item in observed_directions)
    decreases = sum(item == "decrease" for item in observed_directions)
    ties = sum(item == "tie" for item in observed_directions)
    consecutive_inc = sum(item == "increase" for item in consecutive)
    consecutive_dec = sum(item == "decrease" for item in consecutive)
    consecutive_tie = sum(item == "tie" for item in consecutive)
    change = values[-1] - values[0] if n else np.nan
    rho = pd.Series(sequence_positions).corr(pd.Series(values), method="spearman") if n >= 2 else np.nan
    eligible = n >= 3
    increasing = bool(change > 0 and rho > 0) if eligible and pd.notna(rho) else False
    observed_rpts = {
        int(rpt)
        for row in records
        if (rpt := _number(row.get("rpt_index"))) is not None and not _truth(row.get("is_eol"))
    }
    if observed_rpts:
        lo, hi = min(observed_rpts), max(observed_rpts)
        span_missing = max(hi - lo + 1 - len(observed_rpts), 0)
        verified_missing = len({rpt for rpt in verified_rpts if lo <= rpt <= hi} - observed_rpts)
        missing_indices = sorted({rpt for rpt in verified_rpts if lo <= rpt <= hi} - observed_rpts)
    else:
        span_missing = np.nan
        verified_missing = len(verified_rpts) if n == 0 else 0
        missing_indices = sorted(verified_rpts) if n == 0 else []
    direction_switches = sum(
        bool(left and right and left != right)
        for left, right in zip(directions, directions[1:])
    )
    first = records[0] if n else {}
    last = records[-1] if n else {}
    stats = {
        **identity,
        "sensitivity": sensitivity,
        "checkpoint_scope": scope,
        "valid_checkpoint_count": n,
        "eligible_for_trend_classification": eligible,
        "first_checkpoint": first.get("rpt_or_checkpoint"),
        "first_rpt_index": _number(first.get("rpt_index")),
        "last_checkpoint": last.get("rpt_or_checkpoint"),
        "last_rpt_index": _number(last.get("rpt_index")),
        "first_to_last_change_percent": change,
        "spearman_chronological_order_vs_degradation": rho,
        "overall_increasing_trend": increasing if eligible else None,
        "overall_decrease": bool(change < 0) if eligible else None,
        "overall_flat": bool(math.isclose(change, 0.0, rel_tol=0.0, abs_tol=1e-12)) if eligible else None,
        "adjacent_observed_transition_count": len(transitions),
        "adjacent_observed_increases": increases,
        "adjacent_observed_decreases": decreases,
        "adjacent_observed_ties": ties,
        "percentage_non_decreasing_adjacent_observed_transitions": (
            100.0 * (increases + ties) / len(transitions) if transitions else np.nan
        ),
        "fully_non_decreasing_observed_sequence": bool(decreases == 0) if transitions else None,
        "consecutive_verified_rpt_transition_count": len(consecutive),
        "consecutive_verified_rpt_increases": consecutive_inc,
        "consecutive_verified_rpt_decreases": consecutive_dec,
        "consecutive_verified_rpt_ties": consecutive_tie,
        "percentage_non_decreasing_consecutive_verified_rpt_transitions": (
            100.0 * (consecutive_inc + consecutive_tie) / len(consecutive) if consecutive else np.nan
        ),
        "nonconsecutive_observed_transition_count": sum(
            item["transition_type"] == "observed_gap" for item in transitions
        ),
        "eol_adjacent_transition_count": sum(item["transition_type"] == "eol_adjacent" for item in transitions),
        "direction_switch_count": direction_switches,
        "missing_verified_rpt_checkpoint_count_in_span": verified_missing,
        "missing_verified_rpt_indices_in_span": ";".join(map(str, missing_indices)),
        "missing_rpt_index_count_in_observed_span": span_missing,
        "mixed_increases_and_decreases": bool(increases > 0 and decreases > 0),
    }
    return stats, transitions


def _build_trajectory_outputs(values: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    identities = values[CELL_GROUP_COLUMNS].drop_duplicates().to_dict("records")
    base = values[values["method"].eq("H1 Hybrid v2")]
    verified_rpt_sets: dict[tuple[Any, ...], set[int]] = {}
    for key, rows in base[base["rpt_index"].notna() & ~base["is_eol"]].groupby(
        CELL_GROUP_COLUMNS, sort=False, dropna=False
    ):
        verified_rpt_sets[key] = set(pd.to_numeric(rows["rpt_index"], errors="coerce").dropna().astype(int))
    per_cell = []
    transitions = []
    scopes = (
        ("regular_only", "all_available", "common_regular_checkpoint", False),
        ("regular_only", "common_all_methods", "common_regular_checkpoint", False),
        ("eol_inclusive", "all_available", "common_eol_inclusive_checkpoint", True),
        ("eol_inclusive", "common_all_methods", "common_eol_inclusive_checkpoint", True),
    )
    for identity in identities:
        mask = pd.Series(True, index=values.index)
        for column in CELL_GROUP_COLUMNS:
            mask &= values[column].eq(identity[column])
        group_key = tuple(identity[column] for column in CELL_GROUP_COLUMNS)
        verified_rpts = verified_rpt_sets.get(group_key, set())
        for method in METHODS:
            for sensitivity, scope, common_flag, include_eol in scopes:
                selected = values[mask & values["method"].eq(method) & values["valid_value"]]
                if not include_eol:
                    selected = selected[~selected["is_eol"]]
                if scope == "common_all_methods":
                    selected = selected[selected[common_flag]]
                stats, details = _trajectory_stats(
                    selected,
                    sensitivity,
                    scope,
                    {**identity, "method": method},
                    verified_rpts,
                )
                per_cell.append(stats)
                transitions.extend(details)
    per_cell_frame = pd.DataFrame(per_cell)
    transition_frame = pd.DataFrame(transitions)
    summary = _aggregate_trajectories(per_cell_frame)
    return per_cell_frame, transition_frame, summary


def _count_true(series: pd.Series) -> int:
    return sum(_truth(value) for value in series)


def _aggregate_row(group: pd.DataFrame, keys: dict[str, Any], metric: str) -> dict[str, Any]:
    eligible = group[group["eligible_for_trend_classification"]]
    eligible_n = len(eligible)
    increasing = _count_true(eligible["overall_increasing_trend"])
    fully = _count_true(eligible["fully_non_decreasing_observed_sequence"])
    adjacent_n = int(eligible["adjacent_observed_transition_count"].sum())
    adjacent_nondec = int(
        (eligible["adjacent_observed_increases"] + eligible["adjacent_observed_ties"]).sum()
    )
    consecutive_n = int(eligible["consecutive_verified_rpt_transition_count"].sum())
    consecutive_nondec = int(
        (eligible["consecutive_verified_rpt_increases"] + eligible["consecutive_verified_rpt_ties"]).sum()
    )
    return {
        **keys,
        "metric": metric,
        "valid_cell_metric_checkpoints": int(group["valid_checkpoint_count"].sum()),
        "eligible_cell_metric_trajectories": eligible_n,
        "increasing_trend_trajectories": increasing,
        "percentage_increasing_overall": 100.0 * increasing / eligible_n if eligible_n else np.nan,
        "increasing_percentage_exceeds_50": bool(100.0 * increasing / eligible_n > 50.0) if eligible_n else False,
        "fully_non_decreasing_trajectories": fully,
        "decreasing_overall_trajectories": _count_true(eligible["overall_decrease"]),
        "mixed_increase_decrease_trajectories": _count_true(eligible["mixed_increases_and_decreases"]),
        "adjacent_observed_transition_count": adjacent_n,
        "adjacent_observed_increases": int(eligible["adjacent_observed_increases"].sum()),
        "adjacent_observed_decreases": int(eligible["adjacent_observed_decreases"].sum()),
        "adjacent_observed_ties": int(eligible["adjacent_observed_ties"].sum()),
        "percentage_non_decreasing_adjacent_observed_transitions": (
            100.0 * adjacent_nondec / adjacent_n if adjacent_n else np.nan
        ),
        "consecutive_verified_rpt_transition_count": consecutive_n,
        "consecutive_verified_rpt_increases": int(eligible["consecutive_verified_rpt_increases"].sum()),
        "consecutive_verified_rpt_decreases": int(eligible["consecutive_verified_rpt_decreases"].sum()),
        "consecutive_verified_rpt_ties": int(eligible["consecutive_verified_rpt_ties"].sum()),
        "percentage_non_decreasing_consecutive_verified_rpt_transitions": (
            100.0 * consecutive_nondec / consecutive_n if consecutive_n else np.nan
        ),
        "direction_switch_count": int(eligible["direction_switch_count"].sum()),
        "missing_verified_rpt_checkpoint_count_in_span": int(
            pd.to_numeric(eligible["missing_verified_rpt_checkpoint_count_in_span"], errors="coerce").fillna(0).sum()
        ),
    }


def _aggregate_trajectories(per_cell: pd.DataFrame) -> pd.DataFrame:
    group_fields = ["evaluation_domain", "sensitivity", "checkpoint_scope", "method"]
    rows = []
    for key, group in per_cell.groupby(group_fields, sort=False):
        keys = dict(zip(group_fields, key))
        for metric in METRICS:
            rows.append(_aggregate_row(group[group["metric"].eq(metric)], keys, metric))
        rows.append(_aggregate_row(group, keys, "OVERALL"))
    return pd.DataFrame(rows)


def _coverage_summary(v2: pd.DataFrame, v3: pd.DataFrame) -> pd.DataFrame:
    merged = _h2_v2_v3_comparison(v2, v3)
    rows = []
    for domain in (*DOMAINS, "ALL"):
        domain_rows = merged if domain == "ALL" else merged[merged["evaluation_domain"].eq(domain)]
        for metric in (*METRICS, "OVERALL"):
            metric_rows = domain_rows if metric == "OVERALL" else domain_rows[domain_rows["metric"].eq(metric)]
            for scope in ("regular", "eol"):
                selected = metric_rows[
                    metric_rows["is_eol"].map(_truth).eq(scope == "eol")
                ]
                v2_ok = selected["h2_v2_selected_accepted"].map(_truth)
                v3_ok = selected["h2_v3_selected_accepted"].map(_truth)
                rows.append(
                    {
                        "evaluation_domain": domain,
                        "metric": metric,
                        "checkpoint_scope": scope,
                        "verified_checkpoint_metric_rows": len(selected),
                        "h2_v2_selected_estimates": int(v2_ok.sum()),
                        "h2_v3_selected_estimates": int(v3_ok.sum()),
                        "newly_recovered_estimates": int((~v2_ok & v3_ok).sum()),
                        "selection_lost_estimates": int((v2_ok & ~v3_ok).sum()),
                        "h2_v3_negative_selected_estimates": int(
                            (v3_ok & pd.to_numeric(selected["h2_v3_selected_percent"], errors="coerce").lt(0)).sum()
                        ),
                        "h2_v3_unavailable_estimates": int((~v3_ok).sum()),
                        "h2_v2_selected_cells": int(
                            selected.loc[v2_ok, "physical_cell_id"].nunique()
                        ),
                        "h2_v3_selected_cells": int(
                            selected.loc[v3_ok, "physical_cell_id"].nunique()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _common_checkpoint_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    selected = summary[
        summary["checkpoint_scope"].eq("common_all_methods")
        & summary["sensitivity"].eq("regular_only")
        & summary["method"].isin({"H1 Hybrid v2", "H2 Hybrid v3"})
    ].copy()
    fields = [
        "evaluation_domain",
        "metric",
        "method",
        "eligible_cell_metric_trajectories",
        "increasing_trend_trajectories",
        "percentage_increasing_overall",
        "fully_non_decreasing_trajectories",
        "percentage_non_decreasing_adjacent_observed_transitions",
        "adjacent_observed_transition_count",
        "adjacent_observed_increases",
        "adjacent_observed_decreases",
        "adjacent_observed_ties",
    ]
    methods = []
    for method in ("H1 Hybrid v2", "H2 Hybrid v3"):
        subset = selected[selected["method"].eq(method)].drop(columns="method").rename(
            columns={field: f"{method}_{field}" for field in fields[3:]}
        )
        methods.append(subset)
    if not methods or all(method.empty for method in methods):
        return pd.DataFrame(columns=["evaluation_domain", "metric"])
    return methods[0].merge(
        methods[1],
        on=["evaluation_domain", "sensitivity", "checkpoint_scope", "metric"],
        how="outer",
        validate="one_to_one",
    )


def _make_figures(values: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    method_order = ["Charge-only", "Discharge-only", "H1 Hybrid v2", "H2 Hybrid v2", "H2 Hybrid v3"]
    colors = {
        "Charge-only": "#4472C4",
        "Discharge-only": "#ED7D31",
        "H1 Hybrid v2": "#70AD47",
        "H2 Hybrid v2": "#A5A5A5",
        "H2 Hybrid v3": "#7030A0",
    }
    for domain in DOMAINS:
        for metric in METRICS:
            figure, axis = plt.subplots(figsize=(10, 5.5))
            subset = values[
                values["evaluation_domain"].eq(domain)
                & values["metric"].eq(metric)
                & ~values["is_eol"]
                & values["valid_value"]
            ]
            for method in method_order:
                method_rows = subset[subset["method"].eq(method)]
                for _, cell_rows in method_rows.groupby("physical_cell_id", sort=True):
                    ordered = cell_rows.sort_values(
                        ["chronological_position", "position"], kind="stable"
                    )
                    axis.plot(
                        ordered["rpt_index"],
                        ordered["value_percent"],
                        color=colors[method],
                        alpha=0.16 if method != "H2 Hybrid v3" else 0.24,
                        linewidth=0.9,
                    )
            # Add one legend handle per method without obscuring individual cell paths.
            for method in method_order:
                axis.plot([], [], color=colors[method], label=method)
            axis.set_title(f"{domain.replace('_', ' ').title()} - {metric} regular RPT trajectories")
            axis.set_xlabel("Verified RPT number")
            axis.set_ylabel("Signed estimate (%)")
            axis.grid(alpha=0.2)
            axis.legend(ncol=2, fontsize=8)
            figure.tight_layout()
            path = figure_dir / f"{domain}_{metric}_trajectories.png"
            figure.savefig(path, dpi=160)
            plt.close(figure)
            created.append(path)

    regular = summary[
        summary["checkpoint_scope"].eq("all_available")
        & summary["sensitivity"].eq("regular_only")
        & summary["metric"].isin(METRICS)
        & summary["method"].isin({"H1 Hybrid v2", "H2 Hybrid v2", "H2 Hybrid v3"})
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for axis, domain in zip(axes, DOMAINS):
        plot_data = regular[regular["evaluation_domain"].eq(domain)]
        positions = np.arange(len(METRICS))
        width = 0.24
        for offset, method in enumerate(("H1 Hybrid v2", "H2 Hybrid v2", "H2 Hybrid v3")):
            rows = plot_data[plot_data["method"].eq(method)].set_index("metric")
            heights = [
                _number(rows.at[metric, "percentage_non_decreasing_adjacent_observed_transitions"])
                if metric in rows.index
                else None
                for metric in METRICS
            ]
            axes_index = offset - 1
            axis.bar(
                positions + axes_index * width,
                [0.0 if value is None else value for value in heights],
                width,
                label=method,
            )
        axis.set_title(domain.replace("_", " ").title())
        axis.set_xticks(positions, METRICS)
        axis.set_ylim(0, 100)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Non-decreasing adjacent observed transitions (%)")
    axes[1].legend(fontsize=8)
    figure.suptitle("Hybrid trajectory transition consistency (regular RPTs)")
    figure.tight_layout()
    path = figure_dir / "hybrid_transition_consistency.png"
    figure.savefig(path, dpi=160)
    plt.close(figure)
    created.append(path)
    return created


def _format_count(row: pd.Series, column: str) -> str:
    value = _number(row.get(column))
    return str(int(value)) if value is not None else "0"


def _write_report(
    path: Path,
    inventory: pd.DataFrame,
    policies: pd.DataFrame,
    v3: pd.DataFrame,
    coverage: pd.DataFrame,
    trajectories: pd.DataFrame,
    common_comparison: pd.DataFrame,
    figures: list[Path],
) -> None:
    v2_selected = v3["selected_accepted"].map(_truth)
    v3_selected = v3["h2_v3_selected_accepted"].map(_truth)
    v2_values = pd.to_numeric(v3["selected_percent"], errors="coerce")
    v3_values = pd.to_numeric(v3["h2_v3_selected_percent"], errors="coerce")
    shared_selected = v2_selected & v3_selected
    changed_shared_values = shared_selected & ~np.isclose(
        v2_values.fillna(0), v3_values.fillna(0), rtol=0.0, atol=1e-10
    )
    recovered = ~v2_selected & v3_selected
    selected_sources = v3.loc[v3_selected, "h2_v3_source_direction"].value_counts().to_dict()
    primary_counts = policies["primary_direction"].value_counts().to_dict()
    selected_source_summary = (
        "Every selected value matches the saved accepted Charge estimate in this cohort."
        if selected_sources == {"charge": int(v3_selected.sum())}
        else f"Selected source-direction counts are {selected_sources}."
    )
    lines = [
        "# BaFuse H2 Hybrid Direction v3 - Charge Tie-Breaker",
        "",
        "## Scope and safeguards",
        "",
        "This isolated experiment reuses saved accepted charge/discharge PyDMA records and v1/v2 selections. "
        "It does not rerun PyDMA, change original signed estimates, modify labels/manifests/models, or use "
        "Part 3 to tune the internal method. A tied paired-positive rate selects Charge in v3; all other "
        "majority-positive direction rules and the signed per-checkpoint fallback rule are retained.",
        "",
        "Part 3 is reported as external/domain-shift evaluation. H2 v3 selection is a diagnostic policy, "
        "not physical-ground-truth correction.",
        "",
        "## Part 3 inventory reconciliation",
        "",
        f"- Source inventory identities audited: **{inventory['physical_cell_id'].nunique()}**.",
        f"- Guide-listed identities: **{int(inventory['guide_cell_entry_present'].sum())}**.",
        f"- Identities without a Guide cell entry but user-confirmed in the canonical manifest: "
        f"**{int((~inventory['guide_cell_entry_present']).sum())}**.",
        "- G10-Cell2 is retained as a user-confirmed identity only; its ten mapped RPT rows have no saved "
        "accepted PyDMA fit and Step 4 is unaudited, so it remains excluded from estimates and trajectories.",
        "",
        "## Direction policy and coverage",
        "",
        f"- Cell/metric direction policies: **{len(policies)}**; Charge-preferred ties: "
        f"**{int(policies['selection_reason'].eq('tie_prefer_charge').sum())}**; no paired accepted "
        f"checkpoints (undetermined): **{int(policies['primary_direction'].eq('undetermined').sum())}**; "
        f"primary directions: `{json.dumps(primary_counts, sort_keys=True)}`.",
        f"- Selected H2 v3 estimate rows: **{int(v3['h2_v3_selected_accepted'].map(_truth).sum())}**; "
        f"positive-fallback events: **{int(v3['h2_v3_fallback_applied'].map(_truth).sum())}**.",
        f"- Relative to H2 v2: **{int(recovered.sum())}** estimates newly selected, "
        f"**{int((v2_selected & ~v3_selected).sum())}** selections lost, and "
        f"**{int(changed_shared_values.sum())}** numeric values changed among previously selected rows.",
        f"- H2 v3 selected source directions: `{json.dumps(selected_sources, sort_keys=True)}`. "
        f"{selected_source_summary}",
        "",
        "| Domain | Metric | Scope | H2 v2 selected | H2 v3 selected | Newly recovered | Negative selected | Unavailable |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in coverage[
        coverage["metric"].isin(METRICS + ("OVERALL",))
        & coverage["checkpoint_scope"].eq("regular")
    ].iterrows():
        lines.append(
            f"| {row['evaluation_domain']} | {row['metric']} | regular | "
            f"{_format_count(row, 'h2_v2_selected_estimates')} | "
            f"{_format_count(row, 'h2_v3_selected_estimates')} | "
            f"{_format_count(row, 'newly_recovered_estimates')} | "
            f"{_format_count(row, 'h2_v3_negative_selected_estimates')} | "
            f"{_format_count(row, 'h2_v3_unavailable_estimates')} |"
        )
    for _, row in coverage[
        coverage["metric"].isin((*METRICS, "OVERALL"))
        & coverage["checkpoint_scope"].eq("eol")
    ].iterrows():
        lines.append(
            f"| {row['evaluation_domain']} | {row['metric']} | EoL | "
            f"{_format_count(row, 'h2_v2_selected_estimates')} | "
            f"{_format_count(row, 'h2_v3_selected_estimates')} | "
            f"{_format_count(row, 'newly_recovered_estimates')} | "
            f"{_format_count(row, 'h2_v3_negative_selected_estimates')} | "
            f"{_format_count(row, 'h2_v3_unavailable_estimates')} |"
        )

    lines.extend(["", "## Part 2 unnumbered EoL checks", ""])
    for cell in ("G1-Cell20", "G3-Cell14"):
        selected = v3[
            v3["physical_cell_id"].eq(cell)
            & v3["is_unnumbered_eol"].map(_truth)
            & v3["metric"].eq("Anode_LAM")
        ]
        if selected.empty:
            lines.append(f"- **{cell}:** no matching saved unnumbered EoL row.")
            continue
        row = selected.iloc[0]
        lines.append(
            f"- **{cell}:** selected {row['h2_v3_selected_percent']:.4f}% from "
            f"{row['h2_v3_source_direction']} (primary {row['h2_v3_primary_direction']}; "
            f"Charge accepted={_truth(row['charge_accepted'])}, "
            f"Discharge accepted={_truth(row['discharge_accepted'])}; "
            f"fallback={_truth(row['h2_v3_fallback_applied'])})."
        )

    regular_summary = trajectories[
        trajectories["checkpoint_scope"].eq("all_available")
        & trajectories["sensitivity"].eq("regular_only")
        & trajectories["metric"].isin((*METRICS, "OVERALL"))
    ]
    lines.extend(
        [
            "",
            "## Regular-RPT trajectory comparison",
            "",
            "Trajectories require at least three valid checkpoints for trend classification. The percentage "
            "of non-decreasing transitions is weighted by transitions for descriptive summary; the "
            "increasing-trend percentage is cell-trajectory based. No transition is treated as an "
            "independent cell.",
            "",
            "| Domain | Method | Metric | Eligible trajectories | Increasing trend | Increasing (%) | Fully non-decreasing | Non-decreasing transitions (%) |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in regular_summary.iterrows():
        increasing_rate = _number(row.get("percentage_increasing_overall"))
        transition_rate = _number(row.get("percentage_non_decreasing_adjacent_observed_transitions"))
        lines.append(
            f"| {row['evaluation_domain']} | {row['method']} | {row['metric']} | "
            f"{_format_count(row, 'eligible_cell_metric_trajectories')} | "
            f"{_format_count(row, 'increasing_trend_trajectories')} | "
            f"{increasing_rate:.2f}% | {_format_count(row, 'fully_non_decreasing_trajectories')} | "
            f"{transition_rate:.2f}% |"
            if increasing_rate is not None and transition_rate is not None
            else f"| {row['evaluation_domain']} | {row['method']} | {row['metric']} | "
            f"{_format_count(row, 'eligible_cell_metric_trajectories')} | "
            f"{_format_count(row, 'increasing_trend_trajectories')} | n/a | "
            f"{_format_count(row, 'fully_non_decreasing_trajectories')} | n/a |"
        )
    lines.extend(
        [
            "",
            "### Same common checkpoints",
            "",
            "H1 and H2 v3 are compared on their identical intersection with Charge-only, Discharge-only, "
            "and H2 v2, separately by cell and metric.",
            "",
        ]
    )
    if common_comparison.empty:
        lines.append("No common regular cell-metric checkpoint trajectories were available.")
    else:
        lines.append("See `common_checkpoint_comparison.csv` for paired H1/H2 v3 rows and exact counts.")
        for _, row in common_comparison[
            common_comparison["metric"].eq("OVERALL")
        ].iterrows():
            lines.append(
                f"- {row['evaluation_domain']}: "
                f"H1 eligible={_format_count(row, 'H1 Hybrid v2_eligible_cell_metric_trajectories')}, "
                f"non-decreasing transitions="
                f"{_number(row.get('H1 Hybrid v2_percentage_non_decreasing_adjacent_observed_transitions'))}; "
                f"H2 v3 eligible={_format_count(row, 'H2 Hybrid v3_eligible_cell_metric_trajectories')}, "
                f"non-decreasing transitions="
                f"{_number(row.get('H2 Hybrid v3_percentage_non_decreasing_adjacent_observed_transitions'))}."
            )
    eol_summary = trajectories[
        trajectories["checkpoint_scope"].eq("all_available")
        & trajectories["sensitivity"].eq("eol_inclusive")
        & trajectories["metric"].eq("OVERALL")
        & trajectories["method"].isin(("H1 Hybrid v2", "H2 Hybrid v2", "H2 Hybrid v3"))
    ]
    lines.extend(
        [
            "",
            "### EoL-inclusive sensitivity",
            "",
            "EoL checkpoints are excluded from the primary regular-RPT conclusion; these rows are a "
            "separate sensitivity analysis.",
            "",
            "| Domain | Method | Eligible cell-metric trajectories | Increasing trend | Increasing (%) | Fully non-decreasing | Non-decreasing adjacent transitions (%) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for _, row in eol_summary.iterrows():
        increasing_rate = _number(row.get("percentage_increasing_overall"))
        transition_rate = _number(row.get("percentage_non_decreasing_adjacent_observed_transitions"))
        lines.append(
            f"| {row['evaluation_domain']} | {row['method']} | "
            f"{_format_count(row, 'eligible_cell_metric_trajectories')} | "
            f"{_format_count(row, 'increasing_trend_trajectories')} | "
            f"{increasing_rate:.2f}% | {_format_count(row, 'fully_non_decreasing_trajectories')} | "
            f"{transition_rate:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "H2 v3 changes direction selection only for tied paired-positive rates. Coverage changes "
            "therefore reflect estimates recoverable under Charge tie-breaking plus the unchanged "
            "checkpoint-level positive fallback. Negative selected estimates remain signed and visible; "
            "unavailable estimates are not imputed. An increasing trajectory is not evidence that the "
            "estimate is physically correct.",
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{figure.relative_to(OUT).as_posix()}`" for figure in figures)
    lines.extend(["", "## Outputs", ""])
    for name in (
        "h2_hybrid_v3.csv",
        "h2_direction_policy_v3.csv",
        "h2_v2_vs_v3.csv",
        "part3_inventory_audit.csv",
        "h2_v3_coverage_summary.csv",
        "positive_fallback_audit.csv",
        "eol_selection_audit.csv",
        "trajectory_method_values.csv",
        "trajectory_trend_per_cell.csv",
        "trajectory_transition_details.csv",
        "trajectory_trend_summary.csv",
        "common_checkpoint_comparison.csv",
        "run_manifest.json",
    ):
        lines.append(f"- `{name}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _validate_known_eol(v3: pd.DataFrame) -> None:
    expected = {"G1-Cell20": 16.8616, "G3-Cell14": 19.1026}
    for cell, expected_value in expected.items():
        rows = v3[
            v3["physical_cell_id"].eq(cell)
            & v3["is_unnumbered_eol"].map(_truth)
            & v3["metric"].eq("Anode_LAM")
        ]
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one unnumbered EoL Anode LAM row for {cell}.")
        row = rows.iloc[0]
        actual = _number(row["h2_v3_selected_percent"])
        if (
            actual is None
            or not math.isclose(actual, expected_value, rel_tol=0.0, abs_tol=1e-4)
            or _text(row["h2_v3_source_direction"]) != "charge"
            or not _truth(row["charge_accepted"])
        ):
            raise ValueError(
                f"{cell} EoL did not select the expected accepted Charge fit: "
                f"value={actual}, source={row['h2_v3_source_direction']}."
            )
        if _number(row["rpt_index"]) is not None:
            raise ValueError(f"{cell} unnumbered EoL checkpoint was assigned an RPT number.")


def _write_csv(frame: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    frame.to_csv(path, index=False)
    return path


def main() -> None:
    frames, input_hashes = _load_inputs()
    base = _verified_hybrid_rows(frames["matched_charge_discharge_fits.csv"])
    h1 = _verified_hybrid_rows(frames["h1_hybrid_v2.csv"])
    h2 = _verified_hybrid_rows(frames["h2_hybrid_v2.csv"])
    previous_rates = frames["direction_positive_rates.csv"]
    if base.duplicated(
        ["part", "group", "physical_cell_id", "rpt_index", "rpt_or_checkpoint", "checkpoint_id"]
    ).any():
        raise ValueError("Verified saved fit records contain duplicate physical cell/checkpoint keys.")
    if len(base) * len(METRICS) != len(h2):
        raise ValueError("H2 v2 checkpoint/metric rows do not match the saved fit records.")
    observation_columns = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "checkpoint_id",
    ]
    base_observations = base[observation_columns + ["is_unnumbered_eol"]].drop_duplicates()
    h2_observations = h2[observation_columns + ["is_unnumbered_eol"]].drop_duplicates()
    marker_comparison = base_observations.merge(
        h2_observations,
        on=observation_columns,
        how="outer",
        suffixes=("_base", "_h2"),
        indicator=True,
        validate="one_to_one",
    )
    if (
        not marker_comparison["_merge"].eq("both").all()
        or any(
            _truth(row["is_unnumbered_eol_base"]) != _truth(row["is_unnumbered_eol_h2"])
            for row in marker_comparison.to_dict("records")
        )
    ):
        raise ValueError("H2 v2 checkpoint identities or unnumbered-EoL markers differ from saved fits.")

    policies = _rate_policy(h2, previous_rates)
    v3 = _build_v3_results(h2, policies)
    _validate_known_eol(v3)
    policy_audit, fallback_audit, eol_audit = _selection_audits(v3)
    comparison = _h2_v2_v3_comparison(h2, v3)

    values = _mark_common(_build_method_values(h1, h2, v3))
    per_cell, transitions, trajectory_summary = _build_trajectory_outputs(values)
    inventory = _inventory_audit(
        frames,
        h1,
        h2,
        v3,
        frames["trajectory_trend_per_cell.csv"],
        per_cell,
    )
    coverage = _coverage_summary(h2, v3)
    common_comparison = _common_checkpoint_comparison(trajectory_summary)
    figures = _make_figures(values, trajectory_summary)

    outputs = [
        _write_csv(v3, "h2_hybrid_v3.csv"),
        _write_csv(policy_audit, "h2_direction_policy_v3.csv"),
        _write_csv(fallback_audit, "positive_fallback_audit.csv"),
        _write_csv(eol_audit, "eol_selection_audit.csv"),
        _write_csv(comparison, "h2_v2_vs_v3.csv"),
        _write_csv(inventory, "part3_inventory_audit.csv"),
        _write_csv(coverage, "h2_v3_coverage_summary.csv"),
        _write_csv(values, "trajectory_method_values.csv"),
        _write_csv(per_cell, "trajectory_trend_per_cell.csv"),
        _write_csv(transitions, "trajectory_transition_details.csv"),
        _write_csv(trajectory_summary, "trajectory_trend_summary.csv"),
        _write_csv(common_comparison, "common_checkpoint_comparison.csv"),
    ]
    report_path = OUT / "hybrid_direction_v3_report.md"
    _write_report(
        report_path,
        inventory,
        policy_audit,
        v3,
        coverage,
        trajectory_summary,
        common_comparison,
        figures,
    )
    outputs.append(report_path)
    after_hashes = {name: _sha256(path) for name, path in INPUTS.items()}
    if input_hashes != after_hashes:
        raise RuntimeError("One or more saved source inputs changed during v3 generation.")
    output_hashes = {
        path.relative_to(OUT).as_posix(): _sha256(path)
        for path in [*outputs, *figures]
    }
    manifest = {
        "experiment": "pydma_hybrid_direction_v3",
        "policy": "H2 majority-positive with Charge tie-breaker and signed per-checkpoint positive fallback",
        "optimizer_rerun": False,
        "soh_model_retrained": False,
        "source_inputs_sha256": input_hashes,
        "source_inputs_unchanged_after_run": True,
        "output_sha256": output_hashes,
        "summary": {
            "part3_source_inventory_cells": int(inventory["physical_cell_id"].nunique()),
            "part3_guide_listed_cells": int(inventory["guide_cell_entry_present"].sum()),
            "part3_user_confirmed_non_guide_cells": int((~inventory["guide_cell_entry_present"]).sum()),
            "h2_v3_selected_estimates": int(v3["h2_v3_selected_accepted"].map(_truth).sum()),
            "positive_fallback_events": int(v3["h2_v3_fallback_applied"].map(_truth).sum()),
            "direction_policies": len(policy_audit),
            "charge_tie_break_policies": int(policy_audit["selection_reason"].eq("tie_prefer_charge").sum()),
            "undetermined_direction_policies": int(policy_audit["primary_direction"].eq("undetermined").sum()),
        },
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], indent=2))


if __name__ == "__main__":
    main()
