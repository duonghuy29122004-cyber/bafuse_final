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
OUT = Path(__file__).resolve().parent
METRICS = ("LLI", "Anode_LAM", "Cathode_LAM")
KEY_COLUMNS = ["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id", "metric"]
ALLOWED_GROUPS = {1, 2, 3, 4, 7, 8, 9, 10}
INPUTS = (
    "matched_charge_discharge_fits.csv",
    "h1_charge_first_results.csv",
    "h2_majority_positive_results.csv",
)


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


def _record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    part = _number(row.get("part"))
    part_key = int(part) if part is not None else _text(row.get("part"))
    return (
        part_key,
        _text(row.get("physical_cell_id")),
        _text(row.get("rpt_or_checkpoint")),
        _text(row.get("checkpoint_id")),
        _text(row.get("metric")),
    )


def assert_unique_selection_keys(frame: pd.DataFrame) -> None:
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        keys = frame.loc[duplicated, KEY_COLUMNS].to_dict("records")
        raise ValueError(f"Duplicate cell/checkpoint/metric selection keys: {keys[:10]}")


def select_positive_fallback(
    primary_direction: str,
    charge_value: float | None,
    discharge_value: float | None,
    charge_accepted: bool,
    discharge_accepted: bool,
) -> dict[str, Any]:
    primary = _text(primary_direction).casefold()
    if primary not in {"charge", "discharge"}:
        reason = "primary_direction_tie" if primary == "tie" else "primary_direction_undetermined"
        return {
            "selected_percent": None,
            "source_direction": "",
            "fallback_applied": False,
            "fallback_reason": reason,
            "selected_accepted": False,
        }

    values = {
        "charge": charge_value if charge_accepted else None,
        "discharge": discharge_value if discharge_accepted else None,
    }
    primary_value = _number(values[primary])
    if primary_value is None:
        return {
            "selected_percent": None,
            "source_direction": "",
            "fallback_applied": False,
            "fallback_reason": "primary_fit_missing_or_rejected",
            "selected_accepted": False,
        }
    if primary_value >= 0:
        return {
            "selected_percent": primary_value,
            "source_direction": primary,
            "fallback_applied": False,
            "fallback_reason": "primary_nonnegative",
            "selected_accepted": True,
        }

    opposite = "discharge" if primary == "charge" else "charge"
    opposite_value = _number(values[opposite])
    if opposite_value is not None and opposite_value >= 0:
        return {
            "selected_percent": opposite_value,
            "source_direction": opposite,
            "fallback_applied": True,
            "fallback_reason": "negative_primary_positive_opposite",
            "selected_accepted": True,
        }
    if opposite_value is not None and opposite_value < 0:
        reason = "unresolved_negative"
    else:
        reason = "fallback_unavailable"
    return {
        "selected_percent": primary_value,
        "source_direction": primary,
        "fallback_applied": False,
        "fallback_reason": reason,
        "selected_accepted": True,
    }


def _validate_v1_acceptance(frame: pd.DataFrame) -> None:
    for direction in ("charge", "discharge"):
        accepted = frame[frame[f"{direction}_accepted"].map(_truth)]
        for row in accepted.to_dict("records"):
            rmse = _number(row.get(f"{direction}_rmse_mV"))
            runs = _number(row.get(f"{direction}_accepted_runs"))
            if rmse is None or rmse >= 15.0 or runs is None or runs < 3:
                raise ValueError(
                    f"V1 marked a fit accepted without the recorded RMSE/run gate: "
                    f"{row.get('physical_cell_id')} {row.get('rpt_or_checkpoint')} {direction}"
                )


def _load_v1() -> tuple[pd.DataFrame, dict[str, str]]:
    missing = [V1_DIR / name for name in INPUTS if not (V1_DIR / name).is_file()]
    if missing:
        raise FileNotFoundError("Required v1 results are missing: " + ", ".join(map(str, missing)))
    hashes = {name: _sha256(V1_DIR / name) for name in INPUTS}
    base = pd.read_csv(V1_DIR / INPUTS[0], low_memory=False)
    h1 = pd.read_csv(V1_DIR / INPUTS[1], low_memory=False)
    h2 = pd.read_csv(V1_DIR / INPUTS[2], low_memory=False)

    base["part"] = pd.to_numeric(base["part"], errors="coerce")
    base["group"] = pd.to_numeric(base["group"], errors="coerce")
    base["rpt_index"] = pd.to_numeric(base["rpt_index"], errors="coerce")
    base = base[
        base["mapping_verified"].map(_truth)
        & base["part"].isin({1, 2, 3})
        & base["group"].isin(ALLOWED_GROUPS)
    ].copy()
    base["part"] = base["part"].astype(int)
    base["group"] = base["group"].astype(int)
    base["evaluation_domain"] = np.where(
        base["part"].eq(3), "external_part3", "parts1_2_internal"
    )
    if base["is_unnumbered_eol"].map(_truth).any():
        invalid_eol_ids = base[
            base["is_unnumbered_eol"].map(_truth) & base["rpt_index"].notna()
        ]
        if not invalid_eol_ids.empty:
            raise ValueError("A Part 2 unnumbered EoL checkpoint has an RPT number.")
    part3_eol = base[base["part"].eq(3) & base["is_eol"].map(_truth)]
    if not part3_eol.empty and not part3_eol["rpt_index"].isin({6, 7}).all():
        raise ValueError("A Part 3 EoL checkpoint does not use its verified RPT6/RPT7 mapping.")

    if base.duplicated(["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id"]).any():
        raise ValueError("V1 matched-fit table contains duplicate verified checkpoint keys.")
    h1 = h1[h1["mapping_verified"].map(_truth)].copy()
    h2 = h2[h2["mapping_verified"].map(_truth)].copy()
    for name, result in (("H1", h1), ("H2", h2)):
        if result.duplicated(KEY_COLUMNS).any():
            raise ValueError(f"V1 {name} rows contain duplicate cell/checkpoint/metric keys.")
    h1_lookup = {_record_key(row): row for row in h1.to_dict("records")}
    h2_lookup = {_record_key(row): row for row in h2.to_dict("records")}

    rows: list[dict[str, Any]] = []
    for base_row in base.to_dict("records"):
        for metric in METRICS:
            key_row = {**base_row, "metric": metric}
            key = _record_key(key_row)
            if key not in h1_lookup or key not in h2_lookup:
                raise ValueError(f"V1 policy result missing for verified checkpoint key {key}")
            row = dict(key_row)
            row.update({name: value for name, value in h1_lookup[key].items() if name.startswith("h1_")})
            row.update({name: value for name, value in h2_lookup[key].items() if name.startswith("h2_")})
            rows.append(row)
    expanded = pd.DataFrame(rows)
    if len(expanded) != len(base) * len(METRICS):
        raise ValueError("V1 policy assembly did not produce exactly one row per checkpoint and metric.")
    _validate_v1_acceptance(base)
    return expanded, hashes


def _make_method_results(expanded: pd.DataFrame, method: str) -> pd.DataFrame:
    primary_column = "h1_primary_direction" if method == "H1" else "h2_primary_direction"
    rows: list[dict[str, Any]] = []
    subset = expanded[expanded["policy_method"].eq(method)] if "policy_method" in expanded else expanded
    if "policy_method" not in expanded:
        subset = expanded[expanded["h1_primary_direction"].notna()] if method == "H1" else expanded[expanded["h2_primary_direction"].notna()]
    for row in subset.to_dict("records"):
        charge_value = _number(row.get(f"charge_{row['metric']}_percent"))
        discharge_value = _number(row.get(f"discharge_{row['metric']}_percent"))
        charge_accepted = _truth(row.get("charge_accepted"))
        discharge_accepted = _truth(row.get("discharge_accepted"))
        selection = select_positive_fallback(
            _text(row.get(primary_column)),
            charge_value,
            discharge_value,
            charge_accepted,
            discharge_accepted,
        )
        selected_direction = selection["source_direction"]
        row.update(
            {
                "method": method,
                "primary_direction": _text(row.get(primary_column)),
                "original_charge_estimate_percent": charge_value,
                "original_discharge_estimate_percent": discharge_value,
                **selection,
                "selected_fit_status": row.get(f"{selected_direction}_status", "") if selected_direction else "",
                "selected_fit_rmse_mV": row.get(f"{selected_direction}_rmse_mV") if selected_direction else None,
                "selected_fit_accepted_runs": row.get(f"{selected_direction}_accepted_runs") if selected_direction else None,
                "selected_fit_total_runs": row.get(f"{selected_direction}_total_runs") if selected_direction else None,
                "selected_parameter_bound_status": row.get(f"{selected_direction}_parameter_bound_status", "") if selected_direction else "",
                "selected_parameter_bound_any": row.get(f"{selected_direction}_parameter_bound_any") if selected_direction else None,
                "temporal_decrease_warning": None,
                "temporal_delta_from_previous_pp": None,
                "temporal_previous_checkpoint": "",
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError(f"No verified {method} results to process.")
    assert_unique_selection_keys(result)
    return _add_temporal_warnings(result)


def _add_temporal_warnings(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for _, indices in result.groupby(["physical_cell_id", "metric"], sort=False).groups.items():
        ordered = result.loc[list(indices)].sort_values(
            ["chronological_position", "position"], kind="stable"
        )
        previous_row: dict[str, Any] | None = None
        for index, row in ordered.iterrows():
            current_value = _number(row.get("selected_percent"))
            previous_value = _number(previous_row.get("selected_percent")) if previous_row else None
            if previous_row is not None and current_value is not None and previous_value is not None:
                delta = current_value - previous_value
                result.at[index, "temporal_delta_from_previous_pp"] = delta
                result.at[index, "temporal_decrease_warning"] = delta < 0
                result.at[index, "temporal_previous_checkpoint"] = previous_row.get("rpt_or_checkpoint", "")
            previous_row = row.to_dict()
    return result


def _validate_known_eol_cases(h1: pd.DataFrame, h2: pd.DataFrame) -> None:
    expected = {
        "G1-Cell20": (-50.0482, 16.8616),
        "G3-Cell14": (-41.1591, 19.1026),
    }
    for cell, (expected_discharge, expected_charge) in expected.items():
        for method, result in (("H1", h1), ("H2", h2)):
            matches = result[
                result["physical_cell_id"].eq(cell)
                & result["is_unnumbered_eol"].map(_truth)
                & result["metric"].eq("Anode_LAM")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected exactly one verified {cell} unnumbered EoL row for {method}.")
            row = matches.iloc[0]
            discharge = _number(row["original_discharge_estimate_percent"])
            charge = _number(row["original_charge_estimate_percent"])
            if (
                not _truth(row["discharge_accepted"])
                or not _truth(row["charge_accepted"])
                or discharge is None
                or charge is None
                or not math.isclose(discharge, expected_discharge, abs_tol=0.0001, rel_tol=0)
                or not math.isclose(charge, expected_charge, abs_tol=0.0001, rel_tol=0)
            ):
                raise ValueError(f"Saved fit values or acceptance flags do not match the known {cell} EoL case.")
            if row["primary_direction"] == "discharge":
                selected_value = _number(row["selected_percent"])
                if (
                    selected_value is None
                    or not math.isclose(selected_value, expected_charge, abs_tol=0.0001, rel_tol=0)
                    or row["source_direction"] != "charge"
                    or not _truth(row["fallback_applied"])
                ):
                    raise ValueError(f"{method} did not apply the expected accepted Charge fallback for {cell}.")
            elif row["primary_direction"] == "charge":
                selected_value = _number(row["selected_percent"])
                if (
                    selected_value is None
                    or not math.isclose(selected_value, expected_charge, abs_tol=0.0001, rel_tol=0)
                    or row["source_direction"] != "charge"
                    or _truth(row["fallback_applied"])
                ):
                    raise ValueError(f"{method} did not retain the primary Charge fit for {cell}.")


def _audit_rows(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in results.to_dict("records"):
        primary_value = _number(row.get(f"{row.get('primary_direction', '')}_{row['metric']}_percent")) if row.get("primary_direction") in {"charge", "discharge"} else None
        if not (
            row.get("fallback_applied")
            or (primary_value is not None and primary_value < 0)
            or row.get("fallback_reason") in {"primary_direction_tie", "primary_direction_undetermined"}
        ):
            continue
        opposite = "discharge" if row.get("primary_direction") == "charge" else "charge"
        rows.append(
            {
                "method": row["method"],
                "evaluation_domain": row["evaluation_domain"],
                "part": row["part"],
                "group": row["group"],
                "physical_cell_id": row["physical_cell_id"],
                "checkpoint_id": row["checkpoint_id"],
                "rpt_index": row["rpt_index"],
                "rpt_or_checkpoint": row["rpt_or_checkpoint"],
                "is_eol": row["is_eol"],
                "is_unnumbered_eol": row["is_unnumbered_eol"],
                "metric": row["metric"],
                "primary_direction": row["primary_direction"],
                "original_discharge_estimate_percent": row["original_discharge_estimate_percent"],
                "original_charge_estimate_percent": row["original_charge_estimate_percent"],
                "discharge_fit_status": row.get("discharge_status"),
                "discharge_fit_accepted": row.get("discharge_accepted"),
                "discharge_fit_rmse_mV": row.get("discharge_rmse_mV"),
                "discharge_fit_accepted_runs": row.get("discharge_accepted_runs"),
                "charge_fit_status": row.get("charge_status"),
                "charge_fit_accepted": row.get("charge_accepted"),
                "charge_fit_rmse_mV": row.get("charge_rmse_mV"),
                "charge_fit_accepted_runs": row.get("charge_accepted_runs"),
                "primary_estimate_percent": primary_value,
                "opposite_direction": opposite,
                "opposite_fit_status": row.get(f"{opposite}_status"),
                "opposite_fit_accepted": row.get(f"{opposite}_accepted"),
                "opposite_fit_rmse_mV": row.get(f"{opposite}_rmse_mV"),
                "selected_hybrid_percent": row["selected_percent"],
                "selected_source_direction": row["source_direction"],
                "fallback_applied": row["fallback_applied"],
                "fallback_reason": row["fallback_reason"],
                "selected_fit_status": row["selected_fit_status"],
                "selected_fit_rmse_mV": row["selected_fit_rmse_mV"],
                "selected_parameter_bound_status": row["selected_parameter_bound_status"],
                "selected_parameter_bound_any": row["selected_parameter_bound_any"],
                "temporal_decrease_warning": row["temporal_decrease_warning"],
            }
        )
    return pd.DataFrame(rows)


def _compare_v1_v2(
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    original_h1: pd.DataFrame,
    original_h2: pd.DataFrame,
) -> pd.DataFrame:
    outputs = []
    for method, current, previous, old_value, old_source, old_reason in (
        ("H1", h1, original_h1, "h1_selected_percent", "h1_source_direction", "h1_selection_status"),
        ("H2", h2, original_h2, "h2_selected_percent", "h2_source_direction", "h2_selection_status"),
    ):
        key_columns = KEY_COLUMNS
        old = previous[key_columns + [old_value, old_source, old_reason]].copy()
        old = old.rename(
            columns={
                old_value: "v1_selected_percent",
                old_source: "v1_source_direction",
                old_reason: "v1_selection_status",
            }
        )
        new = current[
            key_columns
            + [
                "evaluation_domain",
                "is_eol",
                "primary_direction",
                "selected_percent",
                "source_direction",
                "fallback_applied",
                "fallback_reason",
            ]
        ].rename(
            columns={
                "selected_percent": "v2_selected_percent",
                "source_direction": "v2_source_direction",
                "fallback_reason": "v2_selection_status",
            }
        )
        merged = old.merge(new, on=key_columns, how="outer", validate="one_to_one")
        merged["method"] = method
        v1 = pd.to_numeric(merged["v1_selected_percent"], errors="coerce")
        v2 = pd.to_numeric(merged["v2_selected_percent"], errors="coerce")
        both_missing = v1.isna() & v2.isna()
        equal = both_missing | (v1.notna() & v2.notna() & np.isclose(v1.fillna(0), v2.fillna(0), atol=1e-10, rtol=0))
        merged["estimate_changed"] = ~equal
        merged["v2_minus_v1_pp"] = v2 - v1
        outputs.append(merged)
    result = pd.concat(outputs, ignore_index=True)
    if result.duplicated(KEY_COLUMNS + ["method"]).any():
        raise ValueError("V1/V2 comparison contains duplicate keys.")
    return result


def _checkpoint_counts(frame: pd.DataFrame, mask: pd.Series) -> int:
    selected = frame[mask]
    return int(
        selected[["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id"]]
        .drop_duplicates()
        .shape[0]
    )


def _primary_positive_opposite_negative(frame: pd.DataFrame) -> pd.DataFrame:
    indices = []
    for index, row in frame.iterrows():
        primary = _text(row.get("primary_direction")).casefold()
        if primary not in {"charge", "discharge"}:
            continue
        opposite = "discharge" if primary == "charge" else "charge"
        primary_value = _number(row.get(f"original_{primary}_estimate_percent"))
        opposite_value = _number(row.get(f"original_{opposite}_estimate_percent"))
        if (
            _truth(row.get(f"{primary}_accepted"))
            and _truth(row.get(f"{opposite}_accepted"))
            and primary_value is not None
            and primary_value >= 0
            and opposite_value is not None
            and opposite_value < 0
        ):
            indices.append(index)
    return frame.loc[indices]


def _negative_counterpart_records(base: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in base.to_dict("records"):
        for metric in METRICS:
            for negative_direction, opposite in (("discharge", "charge"), ("charge", "discharge")):
                negative_value = _number(row.get(f"{negative_direction}_{metric}_percent"))
                opposite_value = _number(row.get(f"{opposite}_{metric}_percent"))
                if (
                    _truth(row.get(f"{negative_direction}_accepted"))
                    and negative_value is not None
                    and negative_value < 0
                    and _truth(row.get(f"{opposite}_accepted"))
                    and opposite_value is not None
                    and opposite_value >= 0
                ):
                    records.append(
                        {
                            "evaluation_domain": row["evaluation_domain"],
                            "part": row["part"],
                            "group": row["group"],
                            "physical_cell_id": row["physical_cell_id"],
                            "checkpoint_id": row["checkpoint_id"],
                            "rpt_or_checkpoint": row["rpt_or_checkpoint"],
                            "is_eol": row["is_eol"],
                            "metric": metric,
                            "negative_direction": negative_direction,
                            "negative_estimate_percent": negative_value,
                            "opposite_direction": opposite,
                            "opposite_estimate_percent": opposite_value,
                        }
                    )
    return pd.DataFrame(records)


def _summary_lines(
    base: pd.DataFrame,
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    audit: pd.DataFrame,
    comparison: pd.DataFrame,
) -> list[str]:
    lines: list[str] = []
    counterpart = _negative_counterpart_records(base)
    lines.extend(["## Negative estimates with accepted nonnegative counterparts", ""])
    for domain in ("parts1_2_internal", "external_part3"):
        subset = counterpart[counterpart["evaluation_domain"].eq(domain)]
        lines.append(
            f"- {domain}: negative Discharge with accepted nonnegative Charge: {len(subset[subset.negative_direction == 'discharge'])} estimate occurrences / "
            f"{_checkpoint_counts(subset, subset.negative_direction.eq('discharge'))} unique checkpoints; "
            f"negative Charge with accepted nonnegative Discharge: {len(subset[subset.negative_direction == 'charge'])} / "
            f"{_checkpoint_counts(subset, subset.negative_direction.eq('charge'))}."
        )
    lines.extend(["", "## Fallback events and residual negatives", ""])
    for domain in ("parts1_2_internal", "external_part3"):
        lines.extend([f"### {domain}", ""])
        for method, result in (("H1", h1), ("H2", h2)):
            result = result[result["evaluation_domain"].eq(domain)]
            selected = result[result["selected_percent"].notna()]
            negatives = selected[pd.to_numeric(selected["selected_percent"], errors="coerce") < 0]
            fallbacks = result[result["fallback_applied"].map(_truth)]
            unresolved = result[result["fallback_reason"].isin({"unresolved_negative", "fallback_unavailable"})]
            lines.append(
                f"- {method}: {len(fallbacks)} fallback estimate events / "
                f"{_checkpoint_counts(fallbacks, pd.Series(True, index=fallbacks.index))} unique checkpoints; "
                f"{len(negatives)} selected estimates remain negative / "
                f"{_checkpoint_counts(negatives, pd.Series(True, index=negatives.index))} checkpoints; "
                f"{len(unresolved)} negative-primary records remain unresolved or lack an accepted opposite fit."
            )
            primary_kept = _primary_positive_opposite_negative(result)
            lines.append(
                f"  - Existing accepted nonnegative primary retained despite an accepted negative opposite fit "
                f"(not a fallback): {len(primary_kept)} estimate cases / "
                f"{_checkpoint_counts(primary_kept, pd.Series(True, index=primary_kept.index))} unique checkpoints."
            )
            for is_eol, label in ((False, "regular RPT"), (True, "EoL")):
                part = fallbacks[fallbacks["is_eol"].map(_truth).eq(is_eol)]
                lines.append(
                    f"  - {label}: {len(part)} fallback estimates / "
                    f"{_checkpoint_counts(part, pd.Series(True, index=part.index))} unique checkpoints."
                )

    lines.extend(["", "## H2 ties and undetermined direction", ""])
    for domain in ("parts1_2_internal", "external_part3"):
        domain_h2 = h2[h2["evaluation_domain"].eq(domain)]
        h2_ties = domain_h2[domain_h2["primary_direction"].eq("tie") & domain_h2["selected_percent"].isna()]
        h2_undetermined = domain_h2[
            domain_h2["primary_direction"].eq("undetermined") & domain_h2["selected_percent"].isna()
        ]
        tie_policies = h2_ties[["physical_cell_id", "metric"]].drop_duplicates()
        undetermined_policies = h2_undetermined[["physical_cell_id", "metric"]].drop_duplicates()
        lines.append(
            f"- {domain}: {len(tie_policies)} tied cell/metric policies covering {len(h2_ties)} checkpoint/metric rows and {_checkpoint_counts(h2_ties, pd.Series(True, index=h2_ties.index))} unique checkpoints; "
            f"{len(undetermined_policies)} undetermined policies covering {len(h2_undetermined)} rows and {_checkpoint_counts(h2_undetermined, pd.Series(True, index=h2_undetermined.index))} unique checkpoints."
        )
    lines.extend(
        [
            "",
            "## V1 versus V2",
            "",
        ]
    )
    for domain in ("parts1_2_internal", "external_part3"):
        for method in ("H1", "H2"):
            subset = comparison[
                comparison["method"].eq(method) & comparison["evaluation_domain"].eq(domain)
            ]
            changed = subset[subset["estimate_changed"].map(_truth)]
            lines.append(
                f"- {domain} {method}: {len(changed)} changed estimate rows across "
                f"{_checkpoint_counts(changed, pd.Series(True, index=changed.index))} unique checkpoints."
            )
    lines.extend(
        [
            "- If the numeric outputs match v1, the reason is that v1 already used the same primary-direction and accepted-nonnegative fallback choices for the verified records. V2 makes the policy reasons explicit and audits missing/rejected/both-negative cases separately; it does not add a temporal eligibility rule.",
            "",
            "## Part 2 EoL examples",
            "",
        ]
    )
    for cell in ("G1-Cell20", "G3-Cell14"):
        for method, result in (("H1", h1), ("H2", h2)):
            case = result[
                result["physical_cell_id"].eq(cell)
                & result["is_unnumbered_eol"].map(_truth)
                & result["metric"].eq("Anode_LAM")
            ]
            if case.empty:
                lines.append(f"- {cell} {method}: verified unnumbered EoL result not available.")
                continue
            row = case.iloc[0]
            lines.append(
                f"- {cell} {method}: primary {row['primary_direction']}; "
                f"Discharge {row['original_discharge_estimate_percent']:.6f}% "
                f"({row['discharge_status']}, RMSE {row['discharge_rmse_mV']:.3f} mV); "
                f"Charge {row['original_charge_estimate_percent']:.6f}% "
                f"({row['charge_status']}, RMSE {row['charge_rmse_mV']:.3f} mV); "
                f"selected {row['selected_percent']:.6f}% from {row['source_direction']}; "
                f"fallback={_truth(row['fallback_applied'])}, reason `{row['fallback_reason']}`. "
                f"Previous-RPT comparison is warning-only (decrease={_truth(row['temporal_decrease_warning'])})."
            )
    return lines


def _write_summary(
    base: pd.DataFrame,
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    audit: pd.DataFrame,
    comparison: pd.DataFrame,
    v1_hashes: dict[str, str],
) -> None:
    lines = [
        "# Hybrid PyDMA Positive Fallback Experiment — v2",
        "",
        "## Scope",
        "",
        "- V2 is an isolated post-processing experiment. It reads the saved v1 accepted-fit table and v1 H1/H2 primary-direction decisions; it does not rerun the optimizer.",
        "- Only verified mappings are included. Part 2 Groups 5–6 and unverified/ambiguous mappings are excluded. Part 2 unnumbered EoL checkpoints retain no RPT number; Part 3 EoL uses its Guide-verified RPT6/RPT7 assignment.",
        "- Parts 1–2 and External Part 3 are reported separately. The existing per-cell H1/H2 rules are applied as specified; Part 3 aggregate performance is not used to select or tune a method.",
        "- An estimate is selectable only when the original v1 fit-acceptance flag is true. Every such flag was checked against saved RMSE <15 mV and at least 3 accepted runs; EoL's same-cell, same-direction BoL gate remains encoded in the original flag.",
        "- Temporal decreases are warnings only. They never block a positive fallback. No smoothing, interpolation, cumulative maximum, clipping, or absolute value is used.",
        "",
        "## Inputs and preservation",
        "",
        f"- Verified observations: {len(base)} checkpoints; {int(base['physical_cell_id'].nunique())} physical cells; {len(h1)} H1 metric rows and {len(h2)} H2 metric rows.",
        f"- Part 2: {int((base['part'] == 2).sum())} verified RPT/EoL observations; External Part 3: {int((base['part'] == 3).sum())} verified RPT/EoL observations.",
        f"- No duplicated cell/checkpoint/metric keys in either v2 output. The original signed inputs are carried in both hybrid CSVs and `positive_fallback_audit.csv`.",
        "- V1 source hashes used for this run:",
    ]
    lines.extend(f"  - `{name}`: `{digest}`" for name, digest in sorted(v1_hashes.items()))
    lines.extend(_summary_lines(base, h1, h2, audit, comparison))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A selected nonnegative value is a model-derived diagnostic from one of two direction-specific fitting branches. It is not independently validated physical degradation ground truth. A fallback can create a discontinuity between adjacent checkpoints; the source direction and temporal warning are retained for review.",
            "",
            "## Reproduction",
            "",
            "From the repository root on Windows: `powershell -ExecutionPolicy Bypass -File experiments\\pydma_hybrid_direction_v2\\reproduce.ps1`.",
            "",
        ]
    )
    (OUT / "hybrid_v2_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_cell(cell: str, h1: pd.DataFrame, h2: pd.DataFrame) -> None:
    cell_rows = pd.concat([h1[h1.physical_cell_id.eq(cell)], h2[h2.physical_cell_id.eq(cell)]])
    if cell_rows.empty:
        return
    domain = "External Part 3" if cell_rows["evaluation_domain"].eq("external_part3").any() else "Parts 1–2"
    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        group = cell_rows[cell_rows.metric.eq(metric)].copy()
        keys = ["part", "rpt_or_checkpoint", "checkpoint_id", "method"]
        group = group.drop_duplicates(keys)
        for direction, color, marker, label in (
            ("discharge", "#145da0", "o", "Original Discharge"),
            ("charge", "#e18b00", "^", "Original Charge"),
        ):
            val_col = f"original_{direction}_estimate_percent"
            fit_col = f"{direction}_accepted"
            for is_eol, eol_marker, eol_label in (
                (False, marker, label),
                (True, "*", f"{label} EoL"),
            ):
                points = group[
                    group[fit_col].map(_truth)
                    & group["is_eol"].map(_truth).eq(is_eol)
                    & group[val_col].notna()
                ]
                if not points.empty:
                    ax.scatter(
                        points["position"],
                        points[val_col],
                        c=color,
                        marker=eol_marker,
                        s=75 if is_eol else 42,
                        label=eol_label,
                        alpha=0.85,
                    )
        for method, color, marker in (("H1", "#218739", "s"), ("H2", "#8d52a1", "D")):
            points = group[
                group["method"].eq(method) & group["selected_percent"].notna()
            ]
            if not points.empty:
                ax.scatter(
                    points["position"],
                    points["selected_percent"],
                    c=color,
                    marker=marker,
                    s=48,
                    label=f"{method} selected",
                )
            fallback = points[points["fallback_applied"].map(_truth)]
            if not fallback.empty:
                ax.scatter(
                    fallback["position"],
                    fallback["selected_percent"],
                    marker="*",
                    s=150,
                    facecolors="none",
                    edgecolors="#111111",
                    linewidths=1.2,
                    label=f"{method} positive fallback",
                    zorder=5,
                )
        eol = group[group["is_eol"].map(_truth)].drop_duplicates(["position", "rpt_or_checkpoint"])
        for _, row in eol.iterrows():
            ax.axvline(float(row["position"]), color="#777777", linewidth=0.7, linestyle=":")
        ticks = group[["position", "rpt_or_checkpoint", "is_unnumbered_eol"]].drop_duplicates().sort_values("position")
        if len(ticks) <= 24:
            labels = [
                "EoL (unnumbered)" if _truth(flag) else str(label)
                for label, flag in zip(ticks["rpt_or_checkpoint"], ticks["is_unnumbered_eol"])
            ]
            ax.set_xticks(ticks["position"], labels, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Chronological position (RPT ordinal; not cycle count)")
        ax.set_title(f"{domain} — {cell} — {metric}: original directions and v2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncols=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"{cell}_{metric}_v2.png", dpi=150)
        plt.close(fig)


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    v1_hashes_before = {name: _sha256(V1_DIR / name) for name in INPUTS}
    expanded, recorded_hashes = _load_v1()
    if recorded_hashes != v1_hashes_before:
        raise RuntimeError("V1 source hashes changed while loading.")
    h1_input = expanded.copy()
    h1_input["policy_method"] = "H1"
    h2_input = expanded.copy()
    h2_input["policy_method"] = "H2"
    h1 = _make_method_results(h1_input, "H1")
    h2 = _make_method_results(h2_input, "H2")
    _validate_known_eol_cases(h1, h2)
    audit_frames = (_audit_rows(h1), _audit_rows(h2))
    audit = pd.DataFrame(
        [record for frame in audit_frames for record in frame.to_dict("records")]
    )
    eol_audit = audit[audit["is_eol"].map(_truth)].copy()
    original_h1 = pd.read_csv(V1_DIR / INPUTS[1], low_memory=False)
    original_h2 = pd.read_csv(V1_DIR / INPUTS[2], low_memory=False)
    original_h1 = original_h1[original_h1["mapping_verified"].map(_truth)].copy()
    original_h2 = original_h2[original_h2["mapping_verified"].map(_truth)].copy()
    comparison = _compare_v1_v2(h1, h2, original_h1, original_h2)

    h1.to_csv(OUT / "h1_hybrid_v2.csv", index=False)
    h2.to_csv(OUT / "h2_hybrid_v2.csv", index=False)
    audit.to_csv(OUT / "positive_fallback_audit.csv", index=False)
    eol_audit.to_csv(OUT / "eol_fallback_audit.csv", index=False)
    comparison.to_csv(OUT / "hybrid_v1_vs_v2.csv", index=False)
    for cell in ("G1-Cell20", "G3-Cell14", "G7-Cell19"):
        _plot_cell(cell, h1, h2)

    v1_hashes_after = {name: _sha256(V1_DIR / name) for name in INPUTS}
    if v1_hashes_before != v1_hashes_after:
        raise RuntimeError("A v1 result file changed during the v2 run.")
    _write_summary(expanded.drop_duplicates(["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id"]), h1, h2, audit, comparison, v1_hashes_before)
    manifest = {
        "input_directory": str(V1_DIR),
        "input_sha256": v1_hashes_before,
        "v1_inputs_unchanged_after_run": v1_hashes_before == v1_hashes_after,
        "verified_checkpoint_count": int(expanded[["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id"]].drop_duplicates().shape[0]),
        "h1_metric_rows": len(h1),
        "h2_metric_rows": len(h2),
        "positive_fallback_audit_rows": len(audit),
        "eol_fallback_audit_rows": len(eol_audit),
        "external_part3_kept_separate": True,
        "optimizer_rerun": False,
        "production_artifacts_modified": False,
    }
    (OUT / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
