from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HISTORICAL_PATH = (
    ROOT
    / "experiments/results/data_audits/pydma_eol_audit_v1/"
    / "pydma_negative_estimate_audit.csv"
)
FIT_AUDIT_PATH = ROOT / "reports/final_mapping_label_audit/pydma_fit_audit.csv"
EOL_FITS_PATH = ROOT / "experiments/results/eol_coverage_v1/eol_pydma_fits.csv"
EOL_MANIFEST_PATH = (
    ROOT / "experiments/results/eol_coverage_v1/eol_checkpoint_manifest.csv"
)
STAGE1_BOL_PATH = ROOT / "reports/bafuse_stage_validation_v2/stage1_bol_reference_audit.csv"
MAPPING_PATH = ROOT / "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv"
METRICS = {
    "LLI": "LLI_percent_vs_BoL",
    "Anode_LAM": "LAM_anode_percent_vs_BoL",
    "Cathode_LAM": "LAM_cathode_percent_vs_BoL",
}


def _part(value: Any) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def _rpt(value: Any) -> int | None:
    match = re.search(r"RPT\s*(\d+)", str(value), re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"\d+", str(value).strip()):
        return int(value)
    return None


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _key(part: Any, cell: Any, rpt: Any, direction: Any) -> tuple[str, str, int | None, str]:
    return (
        _part(part),
        str(cell).strip(),
        _rpt(rpt),
        str(direction).strip().casefold(),
    )


def _load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for path in (HISTORICAL_PATH, FIT_AUDIT_PATH, EOL_FITS_PATH, EOL_MANIFEST_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required audit input is missing: {path}")

    history = pd.read_csv(HISTORICAL_PATH, dtype=str, keep_default_na=False)
    fit_audit = pd.read_csv(FIT_AUDIT_PATH, dtype=str, keep_default_na=False)
    eol = pd.read_csv(EOL_FITS_PATH, dtype=str, keep_default_na=False)
    eol_manifest = pd.read_csv(EOL_MANIFEST_PATH, dtype=str, keep_default_na=False)

    history["_part"] = history["part"].map(_part)
    history["_rpt_num"] = history["rpt_number"].map(_rpt)
    history["_cell"] = history["physical_cell_id"].str.strip()
    history["_direction"] = history["direction"].str.strip().str.casefold()
    for metric, column in METRICS.items():
        history[metric] = pd.to_numeric(history[column], errors="coerce")

    mapping = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)
    mapping["_part"] = mapping["part"].map(_part)
    mapping["_rpt_num"] = mapping["rpt_index"].map(_rpt)
    mapping_lookup: dict[tuple[str, str, int | None], list[dict[str, Any]]] = {}
    for _, row in mapping.iterrows():
        map_key = (
            row["_part"],
            str(row["physical_cell_id"]).strip(),
            row["_rpt_num"],
        )
        mapping_lookup.setdefault(map_key, []).append(row.to_dict())
    mapping_statuses: list[str] = []
    mapping_verified: list[bool] = []
    for _, row in history.iterrows():
        map_key = (row["_part"], row["_cell"], row["_rpt_num"])
        matches = mapping_lookup.get(map_key, [])
        if len(matches) != 1:
            mapping_statuses.append(
                "ambiguous_duplicate_mapping_key" if matches else "missing_mapping"
            )
            mapping_verified.append(False)
            continue
        mapped = matches[0]
        status = mapped["final_mapping_status"]
        step_col = "step10_valid" if row["_direction"] == "charge" else "step8_valid"
        step_valid = str(mapped[step_col]).strip().casefold() in {"true", "1"}
        is_verified = status in {"verified", "verified_from_ordered_raw_blocks"} and step_valid
        mapping_statuses.append(
            status if step_valid else f"{status};{step_col}=false"
        )
        mapping_verified.append(is_verified)
    history["_mapping_status"] = mapping_statuses
    history["_mapping_verified"] = mapping_verified

    # This selects the current audited tables only; older snapshots and the
    # Group 5 candidate table are deliberately not mixed into this cohort.
    preferred = fit_audit[
        fit_audit["result_table"].isin(
            {"part1_part2_rmse15", "part3_rmse15"}
        )
    ].copy()
    preferred["_part"] = preferred["dataset_part"].map(_part)
    preferred["_rpt_num"] = preferred["checkup"].map(_rpt)
    preferred["_cell"] = preferred["physical_cell_id"].str.strip()
    preferred["_direction"] = preferred["direction"].str.strip().str.casefold()
    preferred = preferred[
        ~(
            preferred["_part"].eq("2")
            & preferred["_cell"].str.match(r"^G[56]-", case=False)
        )
    ]

    eol["part_key"] = eol["part"].map(_part)
    for column in ("lli_percent", "lam_anode_percent", "lam_cathode_percent"):
        eol[column] = pd.to_numeric(eol[column], errors="coerce")
    eol["rmse_mV"] = pd.to_numeric(eol["rmse_mV"], errors="coerce")
    eol["fit_rmse_mV"] = pd.to_numeric(eol["fit_rmse_mV"], errors="coerce")
    eol["accepted_runs"] = pd.to_numeric(eol["accepted_runs"], errors="coerce")
    eol["total_runs"] = pd.to_numeric(eol["total_runs"], errors="coerce")
    eol["fit_gate_passed_bool"] = eol["fit_gate_passed"].map(_truth)

    return history, preferred, eol, eol_manifest


def _audit_status(preferred: pd.DataFrame, key: tuple[str, str, int | None, str]) -> dict[str, Any]:
    if key[2] is None:
        return {"status": "not_mapped", "matches": 0}
    found = preferred[
        preferred["_part"].eq(key[0])
        & preferred["_cell"].eq(key[1])
        & preferred["_rpt_num"].eq(key[2])
        & preferred["_direction"].eq(key[3])
    ]
    if found.empty:
        return {"status": "not_found_in_selected_audit", "matches": 0}
    if len(found) != 1:
        return {"status": "ambiguous_duplicate_audit_key", "matches": len(found)}
    row = found.iloc[0]
    accepted = _truth(row["fit_accepted_flag"]) and _truth(row["passes_recorded_gate"])
    return {
        "status": "accepted" if accepted else "executed_rejected_or_gate_failed",
        "matches": 1,
        "original_fit_status": row["fit_status_original"],
        "rmse_mV": _number(row["fit_rmse_mV"]),
        "accepted_runs": _number(row["accepted_runs"]),
        "total_runs": _number(row["total_runs"]),
        "source_file": row["source_file"],
        "parameter_bound_status": "not_recorded_in_historical_fit_audit",
    }


def _history_lookup(history: pd.DataFrame) -> dict[tuple[str, str, int | None, str], dict[str, Any]]:
    lookup: dict[tuple[str, str, int | None, str], dict[str, Any]] = {}
    for _, row in history.iterrows():
        key = _key(row["_part"], row["_cell"], row["_rpt_num"], row["_direction"])
        if key in lookup:
            raise ValueError(f"Historical accepted cohort has duplicate fit key: {key}")
        lookup[key] = row.to_dict()
    return lookup


def _charge_fit(
    history: pd.DataFrame,
    preferred: pd.DataFrame,
    lookup: dict[tuple[str, str, int | None, str], dict[str, Any]],
    key: tuple[str, str, int | None, str],
    metric: str,
) -> dict[str, Any]:
    charge_key = (key[0], key[1], key[2], "charge")
    if charge_key in lookup:
        row = lookup[charge_key]
        if not bool(row.get("_mapping_verified")):
            return {
                "status": "not_evaluated_unverified_charge_mapping",
                "estimate": None,
                "parameter_bound_status": row.get("_mapping_status", "unverified"),
            }
        return {
            "status": "accepted",
            "estimate": _number(row.get(metric)),
            "rmse_mV": _number(row.get("fit_rmse_mV")),
            "accepted_runs": _number(row.get("accepted_runs")),
            "total_runs": _number(row.get("total_runs")),
            "source_file": row.get("source_file", ""),
            "parameter_bound_status": "not_recorded_in_historical_fit_audit",
        }
    return _audit_status(preferred, charge_key)


def _neighbors(
    history: pd.DataFrame,
    part: str,
    cell: str,
    direction: str,
    rpt: int,
    metric: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    parts = {"1", "2"} if part in {"1", "2"} and re.match(r"^G[1-4]-", cell) else {part}
    subset = history[
        history["_part"].isin(parts)
        & history["_cell"].eq(cell)
        & history["_direction"].eq(direction)
        & history["_rpt_num"].notna()
        & history["_mapping_verified"]
    ].copy()
    subset = subset[subset["_rpt_num"].ne(rpt) & subset[metric].notna()]
    previous = subset[subset["_rpt_num"].lt(rpt)].sort_values("_rpt_num")
    following = subset[subset["_rpt_num"].gt(rpt)].sort_values("_rpt_num")
    return (
        previous.iloc[-1].to_dict() if not previous.empty else None,
        following.iloc[0].to_dict() if not following.empty else None,
    )


def _temporal_result(
    current: float | None,
    previous: dict[str, Any] | None,
    following: dict[str, Any] | None,
    metric: str,
    current_rpt: int | str,
    eol: bool = False,
) -> dict[str, Any]:
    previous_value = _number(previous.get(metric)) if previous else None
    following_value = _number(following.get(metric)) if following else None
    if current is None:
        status = "not_evaluable_missing_current"
        inclusive = strict = None
    elif previous_value is not None and following_value is not None:
        inclusive = previous_value <= current <= following_value
        strict = previous_value < current < following_value
        status = "full_between_neighbors_pass" if inclusive else "full_between_neighbors_violation"
    elif previous_value is not None:
        inclusive = current >= previous_value
        strict = current > previous_value
        status = "one_sided_previous_pass" if inclusive else "one_sided_previous_violation"
    elif following_value is not None:
        inclusive = current <= following_value
        strict = current < following_value
        status = "one_sided_next_pass" if inclusive else "one_sided_next_violation"
    else:
        inclusive = strict = None
        status = "not_evaluable_no_neighbors"
    previous_rpt = previous.get("_rpt_num") if previous else ""
    following_rpt = following.get("_rpt_num") if following else ""
    return {
        "current_rpt_or_checkpoint": current_rpt,
        "previous_part": previous.get("_part") if previous else "",
        "previous_rpt": previous_rpt,
        "previous_value": previous_value,
        "next_part": following.get("_part") if following else "",
        "next_rpt": following_rpt,
        "next_value": following_value,
        "previous_gap_rpts": (current_rpt - previous_rpt) if previous_rpt != "" and isinstance(current_rpt, int) else "",
        "next_gap_rpts": (following_rpt - current_rpt) if following_rpt != "" and isinstance(current_rpt, int) else "",
        "delta_prev": current - previous_value if current is not None and previous_value is not None else "",
        "delta_next": following_value - current if current is not None and following_value is not None else "",
        "inclusive_consistency": inclusive,
        "strict_consistency": strict,
        "temporal_status": status,
        "is_eol_one_sided_check": bool(eol and status.startswith("one_sided")),
    }


def _stage_bol_status() -> dict[tuple[str, str], str]:
    if not STAGE1_BOL_PATH.is_file():
        return {}
    frame = pd.read_csv(STAGE1_BOL_PATH, dtype=str, keep_default_na=False)
    results: dict[tuple[str, str], str] = {}
    for _, row in frame.iterrows():
        key = (_part(row["dataset_part"]), str(row["physical_cell_id"]).strip())
        if key in results:
            results[key] = "ambiguous_cell_level_reference_audit"
        else:
            results[key] = (
                "verified_cell_level_directional_BoL"
                if _truth(row["bol_verified"])
                else str(row.get("bol_policy_status", "not_verified"))
            )
    return results


def _eol_metric_columns(metric: str) -> str:
    return {
        "LLI": "lli_percent",
        "Anode_LAM": "lam_anode_percent",
        "Cathode_LAM": "lam_cathode_percent",
    }[metric]


def _eol_records(
    history: pd.DataFrame,
    preferred: pd.DataFrame,
    eol: pd.DataFrame,
    eol_manifest: pd.DataFrame,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    results: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    negative_rows = eol[
        eol["fit_gate_passed_bool"]
        & eol["direction"].str.casefold().eq("discharge")
    ]
    for _, row in negative_rows.iterrows():
        values = {
            metric: _number(row[_eol_metric_columns(metric)])
            for metric in METRICS
        }
        negative_metrics = [metric for metric, value in values.items() if value is not None and value < 0]
        if not negative_metrics:
            continue
        part = _part(row["part"])
        cell = str(row["physical_cell_id"]).strip()
        checkpoint = str(row["checkpoint_id"])
        manifest_rows = eol_manifest[eol_manifest["checkpoint_id"].eq(checkpoint)]
        if len(manifest_rows) != 1:
            prior_rpt = None
            prior_status = "ambiguous_or_missing_manifest"
        else:
            prior = manifest_rows.iloc[0]["prior_audit_last_mapped_rpt"]
            prior_rpt = _rpt(prior)
            prior_status = "guide_manifest_prior_rpt"
        for metric in negative_metrics:
            current_value = values[metric]
            inventory = {
                "cohort": "additional_eol",
                "part": part,
                "physical_cell_id": cell,
                "rpt_or_checkpoint": checkpoint,
                "verified_rpt_assignment": row["guide_rpt_assignment"],
                "direction": "discharge",
                "negative_metric": metric,
                "estimate_percent": current_value,
                "rmse_mV": _number(row["rmse_mV"]) or _number(row["fit_rmse_mV"]),
                "rmse_threshold_mV": 15.0,
                "accepted_runs": _number(row["accepted_runs"]),
                "total_runs": _number(row["total_runs"]),
                "fit_gate_passed": True,
                "source_file": row["source_file"],
                "source_fit_id": row["fit_id"],
                "same_cell_direction_matched_bol_status": row["baseline_reference_status"],
                "baseline_fit_status": row["baseline_fit_status"],
                "baseline_fit_rmse_mV": _number(row["baseline_fit_rmse_mV"]),
                "parameter_bound_status": row["parameter_at_bound_flags"] or "not_recorded_on_eol_fit_row",
            }
            results.append(inventory)

            charge_rows = eol[
                eol["checkpoint_id"].eq(checkpoint)
                & eol["direction"].str.casefold().eq("charge")
            ]
            if len(charge_rows) == 1 and _truth(charge_rows.iloc[0]["fit_gate_passed"]):
                charge_row = charge_rows.iloc[0]
                charge_status = "accepted"
                charge_value = _number(charge_row[_eol_metric_columns(metric)])
                charge_rmse = _number(charge_row["rmse_mV"]) or _number(charge_row["fit_rmse_mV"])
                charge_bound = charge_row["parameter_at_bound_flags"] or "not_recorded_on_eol_fit_row"
                charge_bol = charge_row["baseline_reference_status"]
            elif len(charge_rows) > 1:
                charge_status, charge_value, charge_rmse = "ambiguous_duplicate_eol_charge_fit", None, None
                charge_bound, charge_bol = "unavailable", "unavailable"
            elif len(charge_rows) == 1:
                charge_row = charge_rows.iloc[0]
                charge_status = "executed_rejected_or_gate_failed"
                charge_value = _number(charge_row[_eol_metric_columns(metric)])
                charge_rmse = _number(charge_row["rmse_mV"]) or _number(charge_row["fit_rmse_mV"])
                charge_bound = charge_row["parameter_at_bound_flags"] or "not_recorded_on_eol_fit_row"
                charge_bol = charge_row["baseline_reference_status"]
            else:
                charge_status, charge_value, charge_rmse = "not_executed_or_no_record", None, None
                charge_bound, charge_bol = "unavailable", "unavailable"

            temporal_parts = (
                {"1", "2"}
                if part == "2" and re.match(r"^G[1-4]-", cell)
                else {part}
            )
            previous_candidates = history[
                history["_part"].isin(temporal_parts)
                & history["_cell"].eq(cell)
                & history["_direction"].eq("charge")
                & history["_rpt_num"].notna()
                & history["_rpt_num"].le(prior_rpt if prior_rpt is not None else -1)
                & history[metric].notna()
                & history["_mapping_verified"]
            ].sort_values("_rpt_num")
            previous = previous_candidates.iloc[-1].to_dict() if not previous_candidates.empty else None
            temporal = _temporal_result(
                charge_value,
                previous,
                None,
                metric,
                checkpoint,
                eol=True,
            )
            temporal["previous_gap_rpts"] = (
                f"checkpoint follows verified {prior_status} RPT{prior_rpt}"
                if prior_rpt is not None
                else "unavailable"
            )
            pairs.append(
                {
                    "cohort": "additional_eol",
                    "part": part,
                    "physical_cell_id": cell,
                    "rpt_or_checkpoint": checkpoint,
                    "negative_metric": metric,
                    "discharge_estimate_percent": current_value,
                    "discharge_rmse_mV": inventory["rmse_mV"],
                    "discharge_fit_status": "accepted",
                    "charge_estimate_percent": charge_value,
                    "charge_fit_status": charge_status,
                    "charge_rmse_mV": charge_rmse,
                    "charge_positive": charge_value > 0 if charge_value is not None else "",
                    "charge_parameter_bound_status": charge_bound,
                    "charge_bol_reference_status": charge_bol,
                    **temporal,
                }
            )
    return results, pairs


def _classify_neighbors(
    history: pd.DataFrame,
    row: pd.Series,
    metric: str,
    direction: str,
) -> dict[str, Any]:
    previous, following = _neighbors(
        history,
        row["_part"],
        row["_cell"],
        direction,
        int(row["_rpt_num"]),
        metric,
    )
    return _temporal_result(
        _number(row[metric]),
        previous,
        following,
        metric,
        int(row["_rpt_num"]),
    )


def _phase_a(history: pd.DataFrame, preferred: pd.DataFrame, eol: pd.DataFrame, eol_manifest: pd.DataFrame) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    if history.duplicated(["_part", "_cell", "_rpt_num", "_direction"]).any():
        raise ValueError("The accepted historical cohort contains duplicate fit keys.")
    lookup = _history_lookup(history)

    negative_rows = history[
        history["_direction"].eq("discharge")
        & history[list(METRICS)].lt(0).any(axis=1)
    ].copy()
    analysis_history = history[history["_mapping_verified"]].copy()
    negative_inventory: list[dict[str, Any]] = []
    matched_pairs: list[dict[str, Any]] = []
    neighbor_records: list[dict[str, Any]] = []
    control_records: list[dict[str, Any]] = []
    comparable_records: list[dict[str, Any]] = []

    for _, row in history[history["_direction"].eq("discharge")].iterrows():
        key = _key(row["_part"], row["_cell"], row["_rpt_num"], "discharge")
        if not bool(row["_mapping_verified"]):
            for metric in METRICS:
                value = _number(row[metric])
                if value is None or value >= 0:
                    continue
                negative_inventory.append(
                    {
                        "cohort": "historical_accepted",
                        "part": row["_part"],
                        "group": row["group"],
                        "physical_cell_id": row["_cell"],
                        "rpt_or_checkpoint": row["rpt"],
                        "rpt_number": int(row["_rpt_num"]),
                        "direction": "discharge",
                        "negative_metric": metric,
                        "estimate_percent": value,
                        "fit_rmse_mV": _number(row["fit_rmse_mV"]),
                        "rmse_threshold_mV": _number(row["rmse_threshold_mV"]),
                        "accepted_runs": _number(row["accepted_runs"]),
                        "total_runs": _number(row["total_runs"]),
                        "fit_gate_passed": True,
                        "source_file": row["source_file"],
                        "source_run": row["source_run"],
                        "mapping_status": row["_mapping_status"],
                        "mapping_verified": False,
                    }
                )
                matched_pairs.append(
                    {
                        "cohort": "historical_negative_discharge",
                        "part": row["_part"],
                        "group": row["group"],
                        "physical_cell_id": row["_cell"],
                        "rpt_or_checkpoint": row["rpt"],
                        "rpt_number": int(row["_rpt_num"]),
                        "negative_metric": metric,
                        "discharge_estimate_percent": value,
                        "discharge_rmse_mV": _number(row["fit_rmse_mV"]),
                        "discharge_fit_status": "accepted_but_mapping_unverified",
                        "mapping_status": row["_mapping_status"],
                        "mapping_verified": False,
                        "charge_estimate_percent": "",
                        "charge_fit_status": "not_evaluated_unverified_mapping",
                        "charge_rmse_mV": "",
                        "charge_accepted_runs": "",
                        "charge_total_runs": "",
                        "charge_positive": "",
                        "charge_parameter_bound_status": "unavailable",
                        "charge_source_file": "",
                        "charge_bol_reference_status": "not_evaluated_unverified_mapping",
                        "current_rpt_or_checkpoint": int(row["_rpt_num"]),
                        "previous_part": "",
                        "previous_rpt": "",
                        "previous_value": "",
                        "next_part": "",
                        "next_rpt": "",
                        "next_value": "",
                        "previous_gap_rpts": "",
                        "next_gap_rpts": "",
                        "delta_prev": "",
                        "delta_next": "",
                        "inclusive_consistency": "",
                        "strict_consistency": "",
                        "temporal_status": "not_evaluable_unverified_mapping",
                        "is_eol_one_sided_check": False,
                    }
                )
            continue
        all_nonnegative = all(_number(row[metric]) is not None and _number(row[metric]) >= 0 for metric in METRICS)
        for metric in METRICS:
            value = _number(row[metric])
            if value is None:
                continue
            cohort = "negative_discharge" if value < 0 else ("nonnegative_discharge_control" if all_nonnegative else "mixed_sign_discharge")
            charge = _charge_fit(history, preferred, lookup, key, metric)
            cval = _number(charge.get("estimate"))
            if charge["status"] == "accepted":
                charge_current_row = row.copy()
                charge_current_row[metric] = cval
                temporal_charge = _classify_neighbors(
                    history, charge_current_row, metric, "charge"
                )
            else:
                temporal_charge = {
                    "current_rpt_or_checkpoint": int(row["_rpt_num"]),
                    "previous_part": "",
                    "previous_rpt": "",
                    "previous_value": "",
                    "next_part": "",
                    "next_rpt": "",
                    "next_value": "",
                    "previous_gap_rpts": "",
                    "next_gap_rpts": "",
                    "delta_prev": "",
                    "delta_next": "",
                    "inclusive_consistency": "",
                    "strict_consistency": "",
                    "temporal_status": "not_evaluable_charge_fit_unavailable",
                    "is_eol_one_sided_check": False,
                }
            discharge_temporal = _classify_neighbors(history, row, metric, "discharge")
            if value < 0:
                negative_inventory.append(
                    {
                        "cohort": "historical_accepted",
                        "part": row["_part"],
                        "group": row["group"],
                        "physical_cell_id": row["_cell"],
                        "rpt_or_checkpoint": row["rpt"],
                        "rpt_number": int(row["_rpt_num"]),
                        "direction": "discharge",
                        "negative_metric": metric,
                        "estimate_percent": value,
                        "fit_rmse_mV": _number(row["fit_rmse_mV"]),
                        "rmse_threshold_mV": _number(row["rmse_threshold_mV"]),
                        "accepted_runs": _number(row["accepted_runs"]),
                        "total_runs": _number(row["total_runs"]),
                        "fit_gate_passed": True,
                        "source_file": row["source_file"],
                        "source_run": row["source_run"],
                        "mapping_status": row["_mapping_status"],
                        "mapping_verified": True,
                    }
                )
                matched_pairs.append(
                    {
                        "cohort": "historical_negative_discharge",
                        "part": row["_part"],
                        "group": row["group"],
                        "physical_cell_id": row["_cell"],
                        "rpt_or_checkpoint": row["rpt"],
                        "rpt_number": int(row["_rpt_num"]),
                        "negative_metric": metric,
                        "discharge_estimate_percent": value,
                        "discharge_rmse_mV": _number(row["fit_rmse_mV"]),
                        "discharge_fit_status": "accepted",
                        "charge_estimate_percent": cval if charge["status"] == "accepted" else "",
                        "charge_fit_status": charge["status"],
                        "charge_rmse_mV": charge.get("rmse_mV", ""),
                        "charge_accepted_runs": charge.get("accepted_runs", ""),
                        "charge_total_runs": charge.get("total_runs", ""),
                        "charge_positive": cval > 0 if cval is not None and charge["status"] == "accepted" else "",
                        "charge_parameter_bound_status": charge.get("parameter_bound_status", "unavailable"),
                        "charge_source_file": charge.get("source_file", ""),
                        "mapping_status": row["_mapping_status"],
                        "mapping_verified": True,
                        "charge_bol_reference_status": (
                            "not_stored_per_fit_in_historical_artifact"
                        ),
                        **temporal_charge,
                        "discharge_temporal_status": discharge_temporal["temporal_status"],
                        "discharge_previous_rpt": discharge_temporal["previous_rpt"],
                        "discharge_previous_value": discharge_temporal["previous_value"],
                        "discharge_next_rpt": discharge_temporal["next_rpt"],
                        "discharge_next_value": discharge_temporal["next_value"],
                        "discharge_inclusive_consistency": discharge_temporal["inclusive_consistency"],
                        "discharge_strict_consistency": discharge_temporal["strict_consistency"],
                    }
                )

            if value < 0 or all_nonnegative:
                cohort_row = {
                    "cohort": cohort,
                    "part": row["_part"],
                    "group": row["group"],
                    "physical_cell_id": row["_cell"],
                    "rpt_number": int(row["_rpt_num"]),
                    "metric": metric,
                    "discharge_estimate_percent": value,
                    "charge_estimate_percent": cval if charge["status"] == "accepted" else "",
                    "charge_fit_status": charge["status"],
                    "charge_positive": cval > 0 if cval is not None and charge["status"] == "accepted" else "",
                    **temporal_charge,
                    "charge_temporal_status": temporal_charge["temporal_status"],
                    "charge_inclusive_consistency": temporal_charge["inclusive_consistency"],
                    "discharge_temporal_status": discharge_temporal["temporal_status"],
                    "discharge_previous_part": discharge_temporal["previous_part"],
                    "discharge_previous_rpt": discharge_temporal["previous_rpt"],
                    "discharge_next_part": discharge_temporal["next_part"],
                    "discharge_next_rpt": discharge_temporal["next_rpt"],
                    "discharge_inclusive_consistency": discharge_temporal["inclusive_consistency"],
                }
                neighbor_records.append(cohort_row)
                if all_nonnegative:
                    control_records.append(cohort_row)

            if charge["status"] == "accepted":
                charge_prev, charge_next = _neighbors(
                    history, row["_part"], row["_cell"], "charge", int(row["_rpt_num"]), metric
                )
                discharge_prev, discharge_next = _neighbors(
                    history, row["_part"], row["_cell"], "discharge", int(row["_rpt_num"]), metric
                )
                if all((charge_prev, charge_next, discharge_prev, discharge_next)):
                    charge_temporal = _temporal_result(
                        cval, charge_prev, charge_next, metric, int(row["_rpt_num"])
                    )
                    discharge_temporal_common = _temporal_result(
                        value, discharge_prev, discharge_next, metric, int(row["_rpt_num"])
                    )
                    comparable_records.append(
                        {
                            "part": row["_part"],
                            "physical_cell_id": row["_cell"],
                            "rpt_number": int(row["_rpt_num"]),
                            "metric": metric,
                            "discharge_cohort": cohort,
                            "charge_inclusive_consistency": charge_temporal["inclusive_consistency"],
                            "discharge_inclusive_consistency": discharge_temporal_common["inclusive_consistency"],
                            "charge_strict_consistency": charge_temporal["strict_consistency"],
                            "discharge_strict_consistency": discharge_temporal_common["strict_consistency"],
                        }
                    )

    eol_inventory, eol_pairs = _eol_records(history, preferred, eol, eol_manifest)
    negative_inventory.extend(eol_inventory)
    matched_pairs.extend(eol_pairs)

    pd.DataFrame(negative_inventory).to_csv(OUT / "negative_discharge_inventory.csv", index=False)
    historical_negative_frame = pd.DataFrame(
        [row for row in negative_inventory if row["cohort"] == "historical_accepted"]
    )
    if not historical_negative_frame.empty:
        counts = (
            historical_negative_frame.groupby(
                [
                    "part",
                    "group",
                    "physical_cell_id",
                    "rpt_number",
                    "rpt_or_checkpoint",
                    "mapping_status",
                    "mapping_verified",
                ],
                dropna=False,
            )
            .agg(
                negative_metric_count=("negative_metric", "nunique"),
                negative_metrics=("negative_metric", lambda values: ";".join(sorted(set(values)))),
            )
            .reset_index()
        )
        counts.to_csv(OUT / "negative_fit_counts_by_part_cell_rpt.csv", index=False)
    pd.DataFrame(matched_pairs).to_csv(OUT / "charge_discharge_matched_pairs.csv", index=False)
    pd.DataFrame(neighbor_records).to_csv(OUT / "charge_neighbor_consistency.csv", index=False)

    controls = pd.DataFrame(control_records)
    comparisons: list[dict[str, Any]] = []
    for cohort_name, data in (
        ("negative_discharge_metric_occurrences", pd.DataFrame(neighbor_records).query("cohort == 'negative_discharge'")),
        ("fully_nonnegative_discharge_fit_control", controls),
    ):
        for metric, subset in data.groupby("metric", dropna=False):
            accepted = subset[subset["charge_fit_status"].eq("accepted")]
            positive = accepted[accepted["charge_positive"].eq(True)]
            full = accepted[accepted["charge_temporal_status"].astype(str).str.startswith("full_between_neighbors_pass")]
            full_violations = accepted[accepted["charge_temporal_status"].astype(str).str.startswith("full_between_neighbors_violation")]
            strict_full_pass = accepted[
                accepted["charge_temporal_status"].astype(str).str.startswith("full_between_neighbors_")
                & accepted["strict_consistency"].eq(True)
            ]
            one_sided = accepted[accepted["charge_temporal_status"].astype(str).str.startswith("one_sided_")]
            one_sided_passes = one_sided[one_sided["charge_temporal_status"].astype(str).str.endswith("_pass")]
            one_sided_violations = one_sided[one_sided["charge_temporal_status"].astype(str).str.endswith("_violation")]
            strict_one_sided_passes = one_sided[one_sided["strict_consistency"].eq(True)]
            not_eval = accepted[accepted["charge_temporal_status"].astype(str).eq("not_evaluable_no_neighbors")]
            unavailable = subset[~subset["charge_fit_status"].eq("accepted")]
            d_full = subset[subset["discharge_temporal_status"].astype(str).str.startswith("full_between_neighbors_pass")]
            d_full_violations = subset[subset["discharge_temporal_status"].astype(str).str.startswith("full_between_neighbors_violation")]
            d_one_sided = subset[subset["discharge_temporal_status"].astype(str).str.startswith("one_sided_")]
            d_one_sided_pass = d_one_sided[d_one_sided["discharge_temporal_status"].astype(str).str.endswith("_pass")]
            d_one_sided_violation = d_one_sided[d_one_sided["discharge_temporal_status"].astype(str).str.endswith("_violation")]
            comparisons.append(
                {
                    "cohort": cohort_name,
                    "metric": metric,
                    "discharge_metric_occurrences": len(subset),
                    "matching_accepted_charge_fits": len(accepted),
                    "positive_charge_estimates": len(positive),
                    "positive_charge_rate_pct": 100 * len(positive) / len(accepted) if len(accepted) else "",
                    "full_neighbor_passes": len(full),
                    "full_neighbor_violations": len(full_violations),
                    "strict_full_neighbor_passes": len(strict_full_pass),
                    "one_sided_checks": len(one_sided),
                    "one_sided_passes": len(one_sided_passes),
                    "one_sided_violations": len(one_sided_violations),
                    "strict_one_sided_passes": len(strict_one_sided_passes),
                    "not_evaluable_no_neighbors": len(not_eval),
                    "charge_fit_missing_or_rejected": len(unavailable),
                    "discharge_full_neighbor_passes": len(d_full),
                    "discharge_full_neighbor_violations": len(d_full_violations),
                    "discharge_one_sided_passes": len(d_one_sided_pass),
                    "discharge_one_sided_violations": len(d_one_sided_violation),
                }
            )
    comp = pd.DataFrame(comparisons)
    comparable = pd.DataFrame(comparable_records)
    comp.to_csv(OUT / "control_cohort_comparison.csv", index=False)
    comparable.to_csv(OUT / "charge_discharge_comparable_support.csv", index=False)

    _write_figures(analysis_history, negative_rows, eol, preferred)
    _write_summary(
        history,
        preferred,
        eol,
        negative_inventory,
        matched_pairs,
        neighbor_records,
        control_records,
        comparable_records,
    )


def _write_figures(
    history: pd.DataFrame,
    negative_rows: pd.DataFrame,
    eol: pd.DataFrame,
    preferred: pd.DataFrame,
) -> None:
    figures = OUT / "figures"
    cells = sorted(negative_rows["_cell"].drop_duplicates())
    colors = {"charge": "#2274A5", "discharge": "#D1495B"}
    markers = {"1": "o", "2": "^", "3": "s"}
    mapping_path = ROOT / "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv"
    mapping = pd.read_csv(mapping_path, dtype=str, keep_default_na=False)
    mapping["_part"] = mapping["part"].map(_part)
    mapping["_rpt_num"] = mapping["rpt_index"].map(_rpt)
    for cell in cells:
        group_match = re.match(r"^G(\d+)-", cell)
        group = int(group_match.group(1)) if group_match else 0
        parts = {"1", "2"} if group in {1, 2, 3, 4} else {"3"}
        data = history[history["_part"].isin(parts) & history["_cell"].eq(cell)]
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, (metric, _) in zip(axes, METRICS.items()):
            for direction in ("charge", "discharge"):
                direction_data = data[data["_direction"].eq(direction)]
                for source_part, points in direction_data.groupby("_part"):
                    points = points.sort_values("_rpt_num")
                    ax.plot(
                        points["_rpt_num"],
                        points[metric],
                        marker=markers.get(source_part, "o"),
                        linestyle="-",
                        color=colors[direction],
                        label=f"{direction}, Part {source_part}",
                    )
            negatives = data[
                data["_direction"].eq("discharge") & data[metric].lt(0)
            ]
            if not negatives.empty:
                ax.scatter(
                    negatives["_rpt_num"],
                    negatives[metric],
                    marker="x",
                    s=85,
                    color="#8B0000",
                    zorder=4,
                    label="negative discharge",
                )
            seen_labels: set[str] = set()
            expected = mapping[
                mapping["physical_cell_id"].eq(cell)
                & mapping["_part"].isin(parts)
                & mapping["_rpt_num"].notna()
            ]
            for _, mapped in expected.iterrows():
                part_key = mapped["_part"]
                rpt_key = int(mapped["_rpt_num"])
                if (
                    mapped["final_mapping_status"]
                    not in {"verified", "verified_from_ordered_raw_blocks"}
                    or str(mapped["step10_valid"]).strip().casefold() not in {"true", "1"}
                ):
                    marker, label = "x", "unverified/invalid RPT mapping"
                    color = "#777777"
                else:
                    key = (part_key, cell, rpt_key, "charge")
                    status = _audit_status(preferred, key)
                    accepted_charge = (
                        history["_part"].eq(part_key)
                        & history["_cell"].eq(cell)
                        & history["_rpt_num"].eq(rpt_key)
                        & history["_direction"].eq("charge")
                    ).any()
                    if accepted_charge:
                        continue
                    if status["status"] == "executed_rejected_or_gate_failed":
                        marker, label, color = "X", "charge fit rejected/not accepted", "#6A4C93"
                    elif status["status"] == "ambiguous_duplicate_audit_key":
                        marker, label, color = "P", "ambiguous charge-fit audit key", "#777777"
                    else:
                        marker, label, color = "v", "no accepted charge-fit record", "#E07A5F"
                if label in seen_labels:
                    label = "_nolegend_"
                seen_labels.add(label)
                ax.scatter(
                    [rpt_key],
                    [0.035],
                    marker=marker,
                    color=color,
                    s=45,
                    transform=ax.get_xaxis_transform(),
                    clip_on=False,
                    label=label,
                )
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(f"{metric} (%)")
            ax.grid(alpha=0.25)
        axes[-1].set_xlabel("Verified RPT number (ordinal; no cycle spacing implied)")
        axes[0].legend(loc="best", ncols=3, fontsize=7)
        fig.suptitle(
            f"{cell} — PyDMA degradation by direction "
            "(Parts 1–2 combined where continuous)"
        )
        fig.tight_layout()
        fig.savefig(figures / f"{cell}_charge_discharge.png", dpi=150)
        plt.close(fig)

    for _, row in eol[eol["fit_gate_passed_bool"] & eol["direction"].str.casefold().eq("discharge")].iterrows():
        if not any(_number(row[_eol_metric_columns(metric)]) is not None and _number(row[_eol_metric_columns(metric)]) < 0 for metric in METRICS):
            continue
        manifest_file = ROOT / "experiments/results/eol_coverage_v1/eol_checkpoint_manifest.csv"
        manifest = pd.read_csv(manifest_file, dtype=str, keep_default_na=False)
        match = manifest[manifest["checkpoint_id"].eq(row["checkpoint_id"])]
        if len(match) != 1:
            continue
        part, cell = _part(row["part"]), row["physical_cell_id"]
        temporal_parts = (
            {"1", "2"}
            if part == "2" and re.match(r"^G[1-4]-", cell)
            else {part}
        )
        data = history[history["_part"].isin(temporal_parts) & history["_cell"].eq(cell)]
        prior_rpt = _rpt(match.iloc[0]["prior_audit_last_mapped_rpt"])
        eol_x = (
            prior_rpt + 0.5
            if prior_rpt is not None
            else float(data["_rpt_num"].max()) + 0.5
        )
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        for ax, metric in zip(axes, METRICS):
            col = _eol_metric_columns(metric)
            for direction, color in colors.items():
                points = data[data["_direction"].eq(direction)].sort_values("_rpt_num")
                ax.plot(points["_rpt_num"], points[metric], marker="o", color=color, label=direction)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_ylabel(f"{metric} (%)")
            ax.grid(alpha=0.25)
            value = _number(row[col])
            if value is not None:
                ax.scatter([eol_x], [value], marker="*", s=120, color=colors[row["direction"]], zorder=5)
        ticks = sorted(set(data["_rpt_num"].dropna().astype(int).tolist()) | {eol_x})
        tick_labels = [
            "EoL (unnumbered)" if tick == eol_x else str(tick)
            for tick in ticks
        ]
        axes[-1].set_xticks(ticks, tick_labels)
        axes[-1].set_xlabel(
            "Verified RPT number; final star is unnumbered EoL "
            "(layout position only)"
        )
        axes[0].legend(loc="best", ncols=2, fontsize=8)
        fig.suptitle(f"{cell} — Part {part}, separate unnumbered EoL checkpoint")
        fig.tight_layout()
        fig.savefig(figures / f"part{part}_{cell}_eol_charge_discharge.png", dpi=150)
        plt.close(fig)


def _write_summary(
    history: pd.DataFrame,
    preferred: pd.DataFrame,
    eol: pd.DataFrame,
    negative_inventory: list[dict[str, Any]],
    matched_pairs: list[dict[str, Any]],
    neighbor_records: list[dict[str, Any]],
    control_records: list[dict[str, Any]],
    comparable_records: list[dict[str, Any]],
) -> None:
    hist_negative = [r for r in negative_inventory if r["cohort"] == "historical_accepted"]
    hist_unique = {
        (r["part"], r["physical_cell_id"], r["rpt_number"], r["direction"])
        for r in hist_negative
    }
    metric_counts = pd.Series([r["negative_metric"] for r in hist_negative]).value_counts().to_dict()
    negative_pairs = pd.DataFrame([r for r in matched_pairs if r["cohort"] == "historical_negative_discharge"])
    unique_negative_frame = pd.DataFrame(
        list(
            {
                (r["part"], r["physical_cell_id"], r["rpt_number"], r["rpt_or_checkpoint"])
                for r in hist_negative
            }
        ),
        columns=["part", "physical_cell_id", "rpt_number", "rpt_or_checkpoint"],
    )
    by_part = unique_negative_frame.groupby("part").size().to_dict()
    by_cell = (
        unique_negative_frame.groupby(["part", "physical_cell_id"])
        .size()
        .sort_values(ascending=False)
    )
    lines = [
        "# Phase A — Charge-Fit Neighbor Consistency Audit",
        "",
        "## Scope and method",
        "",
        f"- Historical accepted cohort: {len(history)} fit rows; {int(history['_mapping_verified'].sum())} fit rows have a strict verified RPT identity and valid direction-specific curve mapping. Temporal comparisons use only those verified rows.",
        "- Historical chronology is verified RPT ordinal order within each cell. Part 1 and Part 2 are joined for Groups 1–4 because those are continuous cell sequences; Part 3 remains separate. RPT gaps are not interpreted as exact cycle intervals.",
        "- Part 2 Groups 5–6 were excluded. Part 3 is included only as a descriptive audit cohort, not for tuning.",
        "- Additional EoL fits are kept separate from numbered RPTs; the two relevant Part 2 checkpoints remain unnumbered.",
        "- The historical fit audit has no per-fit optimizer parameter vectors/bound flags or exact BoL reference ID. Those fields are reported unavailable rather than inferred.",
        "- Charge positivity or monotonicity is not treated as physical validation.",
        "",
        "## Negative discharge inventory",
        "",
        f"- Historical unique negative discharge fit records: **{len(hist_unique)}**.",
        f"- Historical metric-level negative occurrences: **{len(hist_negative)}** (LLI {metric_counts.get('LLI',0)}, Anode LAM {metric_counts.get('Anode_LAM',0)}, Cathode LAM {metric_counts.get('Cathode_LAM',0)}).",
        f"- Additional accepted EoL negative occurrences: **{sum(r['cohort']=='additional_eol' for r in negative_inventory)}**, kept separate.",
        f"- Unique negative fit records by Part: {', '.join(f'Part {part}: {count}' for part, count in sorted(by_part.items()))}.",
        "- Unique negative fit records by physical cell: "
        + "; ".join(
            f"Part {part} {cell}: {count}"
            for (part, cell), count in by_cell.items()
        )
        + ".",
        "- Exact negative RPT identities and which metrics were negative are listed in `negative_fit_counts_by_part_cell_rpt.csv`; metric-level signed values and RMSEs are in `negative_discharge_inventory.csv`.",
        "",
        "## Matching and temporal consistency",
        "",
    ]
    if not negative_pairs.empty:
        for metric in METRICS:
            subset = negative_pairs[negative_pairs["negative_metric"].eq(metric)]
            eligible_subset = subset[subset["mapping_verified"].eq(True)]
            unverified_subset = subset[subset["mapping_verified"].eq(False)]
            accepted = subset[subset["charge_fit_status"].eq("accepted")]
            positive = accepted[accepted["charge_positive"].eq(True)]
            full = accepted[accepted["temporal_status"].astype(str).str.startswith("full_between_neighbors_pass")]
            full_violations = accepted[accepted["temporal_status"].astype(str).str.startswith("full_between_neighbors_violation")]
            strict_full = accepted[
                accepted["temporal_status"].astype(str).str.startswith("full_between_neighbors_")
                & accepted["strict_consistency"].eq(True)
            ]
            one_sided = accepted[accepted["temporal_status"].astype(str).str.startswith("one_sided_")]
            one_sided_pass = one_sided[one_sided["temporal_status"].astype(str).str.endswith("_pass")]
            one_sided_violation = one_sided[one_sided["temporal_status"].astype(str).str.endswith("_violation")]
            strict_one_sided = one_sided[one_sided["strict_consistency"].eq(True)]
            not_eval = accepted[accepted["temporal_status"].astype(str).eq("not_evaluable_no_neighbors")]
            unavailable = subset[~subset["charge_fit_status"].eq("accepted")]
            discharge_subset = pd.DataFrame(neighbor_records)
            discharge_subset = discharge_subset[
                discharge_subset["cohort"].eq("negative_discharge")
                & discharge_subset["metric"].eq(metric)
            ]
            discharge_full_pass = discharge_subset[
                discharge_subset["discharge_temporal_status"].astype(str).str.startswith("full_between_neighbors_pass")
            ]
            discharge_full_violation = discharge_subset[
                discharge_subset["discharge_temporal_status"].astype(str).str.startswith("full_between_neighbors_violation")
            ]
            discharge_one_sided = discharge_subset[
                discharge_subset["discharge_temporal_status"].astype(str).str.startswith("one_sided_")
            ]
            discharge_one_sided_pass = discharge_one_sided[
                discharge_one_sided["discharge_temporal_status"].astype(str).str.endswith("_pass")
            ]
            discharge_one_sided_violation = discharge_one_sided[
                discharge_one_sided["discharge_temporal_status"].astype(str).str.endswith("_violation")
            ]
            lines.append(
                f"- **{metric} negative historical occurrences:** {len(subset)} total, of which {len(eligible_subset)} have strict verified mappings and {len(unverified_subset)} are excluded from exact matching/chronology; matching accepted charge fits {len(accepted)} of the verified subset; positive charge estimates {len(positive)}; charge full inclusive passes {len(full)} (strict passes {len(strict_full)}), full violations {len(full_violations)}, one-sided passes {len(one_sided_pass)}/{len(one_sided)} (strict {len(strict_one_sided)}) and one-sided violations {len(one_sided_violation)}; no-neighbor cases {len(not_eval)}; missing/rejected/ambiguous/unverified charge fits {len(unavailable)}. Discharge trajectory on eligible negative occurrences: full passes {len(discharge_full_pass)}, full violations {len(discharge_full_violation)}, one-sided passes {len(discharge_one_sided_pass)}, one-sided violations {len(discharge_one_sided_violation)}."
            )
        lines += [
            "",
            "The exact negative occurrence-to-charge records and estimates are in `charge_discharge_matched_pairs.csv`; previous/next RPT IDs, gaps, deltas, and strict/inclusive results are in `charge_neighbor_consistency.csv`.",
            "",
        ]
    comp = pd.DataFrame(comparable_records)
    if comp.empty:
        lines.append("- Same-key charge/discharge common support with both neighboring RPTs: **0**; a matched directional consistency-rate comparison is not estimable.")
    else:
        lines.append(f"- Same-key comparison rows with both-direction complete neighbor support: **{len(comp)}**.")
        lines.append(
            f"- Charge inclusive pass rate on common support: {100*comp['charge_inclusive_consistency'].astype(bool).mean():.1f}% ({int(comp['charge_inclusive_consistency'].astype(bool).sum())}/{len(comp)}); discharge: {100*comp['discharge_inclusive_consistency'].astype(bool).mean():.1f}% ({int(comp['discharge_inclusive_consistency'].astype(bool).sum())}/{len(comp)})."
        )
        for cohort_name in ("negative_discharge", "nonnegative_discharge_control"):
            cohort_support = comp[comp["discharge_cohort"].eq(cohort_name)]
            if cohort_support.empty:
                continue
            charge_passes = int(cohort_support["charge_inclusive_consistency"].astype(bool).sum())
            discharge_passes = int(cohort_support["discharge_inclusive_consistency"].astype(bool).sum())
            lines.append(
                f"- {cohort_name.replace('_', ' ').capitalize()} common support: charge {charge_passes}/{len(cohort_support)}; discharge {discharge_passes}/{len(cohort_support)}."
            )
    lines += [
        "",
        "## Nonnegative-discharge control",
        "",
        f"- Fully nonnegative accepted discharge fit controls with strict verified mappings: {history[(history['_direction'].eq('discharge')) & history['_mapping_verified'] & history[list(METRICS)].ge(0).all(axis=1)].shape[0]} fit records ({len(control_records)} metric-level observations).",
        "",
        "| Control metric | Charge positive | Charge full inclusive / evaluated (strict) | Charge one-sided pass / checks | Discharge full pass / evaluated | Discharge one-sided pass / checks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    control_frame = pd.read_csv(OUT / "control_cohort_comparison.csv")
    for _, row in control_frame[
        control_frame["cohort"].eq("fully_nonnegative_discharge_fit_control")
    ].iterrows():
        charge_full_total = row["full_neighbor_passes"] + row["full_neighbor_violations"]
        discharge_full_total = (
            row["discharge_full_neighbor_passes"]
            + row["discharge_full_neighbor_violations"]
        )
        lines.append(
            f"| {row['metric']} | {int(row['positive_charge_estimates'])}/{int(row['matching_accepted_charge_fits'])} | {int(row['full_neighbor_passes'])}/{int(charge_full_total)} ({int(row['strict_full_neighbor_passes'])} strict) | {int(row['one_sided_passes'])}/{int(row['one_sided_checks'])} ({int(row['strict_one_sided_passes'])} strict) | {int(row['discharge_full_neighbor_passes'])}/{int(discharge_full_total)} | {int(row['discharge_one_sided_passes'])}/{int(row['discharge_one_sided_passes'] + row['discharge_one_sided_violations'])} |"
        )
    lines += [
        "",
        "Control metric rates and all missing/rejected-fit counts are also in `control_cohort_comparison.csv`.",
        "- Original discharge trajectories are evaluated on the same within-cell chronological RPT axis as charge, including the verified Part 1–2 continuation; the plots are unsmoothed.",
        "- Charge is not uniformly more ordered for every metric: use the common-support counts by cohort above and the full per-metric table; these are consistency diagnostics, not physical validation.",
        "",
        "## EoL checkpoints",
        "",
    ]
    for row in [r for r in matched_pairs if r["cohort"] == "additional_eol"]:
        prior_value = row["previous_value"]
        comparison = (
            f"current charge {row['charge_estimate_percent']:.3f}% vs previous {prior_value:.3f}%"
            if isinstance(prior_value, (int, float))
            and isinstance(row["charge_estimate_percent"], (int, float))
            else "prior charge comparison unavailable"
        )
        lines.append(
            f"- {row['physical_cell_id']} {row['rpt_or_checkpoint']} {row['negative_metric']}: discharge {row['discharge_estimate_percent']:.3f}%; same-checkpoint charge {row['charge_fit_status']} at {row['charge_estimate_percent'] if row['charge_estimate_percent'] != '' else 'unavailable'}; previous accepted charge RPT {row['previous_rpt'] or 'not available'}; {comparison}; one-sided result `{row['temporal_status']}`."
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Positive charge estimates and temporal ordering are descriptive evidence only. A positive or monotone sequence does not establish that the charge fit is physically correct. The historical records lack the parameters needed to test boundary saturation or optimizer variability, so the charge-direction comparison cannot resolve the cause of the historical discharge negatives.",
        "",
        "## Reproducibility",
        "",
        "From the repository root in PowerShell:",
        "",
        "```powershell",
        "$env:PYTHONPATH = 'src'",
        ".\\.venv-torch-cuda\\Scripts\\python.exe experiments\\pydma_charge_consistency_v1\\analyze_charge_consistency.py",
        "```",
        "",
    ]
    (OUT / "charge_consistency_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    history, preferred, eol, eol_manifest = _load_inputs()
    _phase_a(history, preferred, eol, eol_manifest)
    print(f"Phase A outputs written to {OUT}")


if __name__ == "__main__":
    main()
