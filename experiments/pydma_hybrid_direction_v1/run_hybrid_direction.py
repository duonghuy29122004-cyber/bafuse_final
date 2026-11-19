from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
HISTORY_PATH = ROOT / "experiments/results/data_audits/pydma_eol_audit_v1/pydma_negative_estimate_audit.csv"
FIT_AUDIT_PATH = ROOT / "reports/final_mapping_label_audit/pydma_fit_audit.csv"
MAPPING_PATH = ROOT / "reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv"
EOL_FITS_PATH = ROOT / "experiments/results/eol_coverage_v1/eol_pydma_fits.csv"
EOL_MANIFEST_PATH = ROOT / "experiments/results/eol_coverage_v1/eol_checkpoint_manifest.csv"

HISTORY_METRICS = {
    "LLI": "LLI_percent_vs_BoL",
    "Anode_LAM": "LAM_anode_percent_vs_BoL",
    "Cathode_LAM": "LAM_cathode_percent_vs_BoL",
}
EOL_METRICS = {
    "LLI": "lli_percent",
    "Anode_LAM": "lam_anode_percent",
    "Cathode_LAM": "lam_cathode_percent",
}
VERIFIED_MAPPING_STATUSES = {"verified", "verified_from_ordered_raw_blocks"}
ALLOWED_GROUPS = {1, 2, 3, 4, 7, 8, 9, 10}
JUMP_THRESHOLDS_PP = (5.0, 10.0, 20.0)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _part(value: Any) -> int | None:
    match = re.search(r"\d+", _text(value))
    return int(match.group()) if match else None


def _rpt(value: Any) -> int | None:
    text = _text(value)
    match = re.search(r"\bRPT\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return int(float(text))
    return None


def _number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _truth(value: Any) -> bool:
    return _text(value).casefold() in {"true", "1", "yes"}


def _valid_segment(value: Any) -> bool:
    status = _text(value).casefold()
    return status.startswith("verified") and not any(
        word in status for word in ("invalid", "ambiguous", "no_valid")
    )


def _source_key(value: Any) -> str:
    return _text(value).replace("\\", "/").strip("/").casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_inputs() -> None:
    missing = [path for path in (HISTORY_PATH, FIT_AUDIT_PATH, MAPPING_PATH, EOL_FITS_PATH, EOL_MANIFEST_PATH) if not path.is_file()]
    if missing:
        raise FileNotFoundError("Required saved audit inputs are missing: " + ", ".join(map(str, missing)))


def _history_accepts(row: dict[str, Any]) -> bool:
    return (
        _text(row.get("diagnostic_label_status")) == "diagnostic_estimate_accepted"
        and (_number(row.get("fit_rmse_mV")) or math.inf) < 15.0
        and (_number(row.get("accepted_runs")) or 0) >= 3
    )


def _audit_rows(fit_audit: pd.DataFrame) -> dict[tuple[int, str, int, str], list[dict[str, Any]]]:
    selected = fit_audit[
        fit_audit["result_table"].isin({"part1_part2_rmse15", "part3_rmse15"})
    ].copy()
    selected["_part"] = selected["dataset_part"].map(_part)
    selected["_rpt"] = selected["checkup"].map(_rpt)
    selected["_cell"] = selected["physical_cell_id"].map(_text)
    selected["_direction"] = selected["direction"].map(lambda v: _text(v).casefold())
    selected = selected[
        selected["_part"].notna()
        & selected["_rpt"].notna()
        & selected["_cell"].ne("")
        & selected["_direction"].isin({"charge", "discharge"})
    ]
    selected = selected[
        ~(
            selected["_part"].eq(2)
            & selected["_cell"].str.match(r"^G[56]-", case=False)
        )
    ]
    result: dict[tuple[int, str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in selected.to_dict("records"):
        key = (int(row["_part"]), row["_cell"], int(row["_rpt"]), row["_direction"])
        result[key].append(row)
    return result


def _mapping_records(mapping: pd.DataFrame) -> list[dict[str, Any]]:
    frame = mapping.copy()
    frame["_part"] = frame["part"].map(_part)
    frame["_group"] = frame["group"].map(_part)
    frame["_rpt"] = frame["rpt_index"].map(_rpt)
    frame["_cell"] = frame["physical_cell_id"].map(_text)
    frame["_status"] = frame["final_mapping_status"].map(_text)
    frame = frame[
        frame["_part"].isin({1, 2, 3})
        & frame["_group"].isin(ALLOWED_GROUPS)
        & frame["_rpt"].notna()
        & frame["_cell"].ne("")
    ]
    records: list[dict[str, Any]] = []
    for key, group in frame.groupby(["_part", "_cell", "_rpt"], sort=False):
        rows = group.to_dict("records")
        row = rows[0]
        status = row["_status"] if len(rows) == 1 else "ambiguous_duplicate_mapping_key"
        records.append(
            {
                "observation_key": (int(key[0]), str(key[1]), int(key[2])),
                "part": int(key[0]),
                "group": int(row["_group"]),
                "physical_cell_id": str(key[1]),
                "rpt_index": int(key[2]),
                "rpt_or_checkpoint": f"RPT{int(key[2])}",
                "checkpoint_id": "",
                "source_file": _text(row.get("source_file")),
                "mapping_status": status,
                "mapping_verified": len(rows) == 1 and status in VERIFIED_MAPPING_STATUSES,
                "step8_valid": len(rows) == 1 and _truth(row.get("step8_valid")),
                "step10_valid": len(rows) == 1 and _truth(row.get("step10_valid")),
                "is_eol": False,
                "is_unnumbered_eol": False,
                "prior_rpt_for_eol": None,
                "sequence_key": str(key[1]),
                "position": float(key[2]),
                "mapping_rows": len(rows),
            }
        )
    return records


def _history_lookup(history: pd.DataFrame) -> tuple[dict[tuple[int, str, int, str], dict[str, Any]], list[dict[str, Any]]]:
    frame = history.copy()
    frame["_part"] = frame["part"].map(_part)
    frame["_rpt"] = frame["rpt_number"].map(_rpt)
    frame["_cell"] = frame["physical_cell_id"].map(_text)
    frame["_direction"] = frame["direction"].map(lambda v: _text(v).casefold())
    frame["_group"] = frame["group"].map(_part)
    lookup: dict[tuple[int, str, int, str], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        if row["_part"] not in {1, 2, 3} or row["_group"] not in ALLOWED_GROUPS or row["_rpt"] is None:
            continue
        key = (int(row["_part"]), row["_cell"], int(row["_rpt"]), row["_direction"])
        if key in lookup:
            duplicates.append({"key": key, "reason": "duplicate_accepted_history_key"})
        else:
            lookup[key] = row
    return lookup, duplicates


def _eol_fit_is_accepted(row: dict[str, Any]) -> bool:
    rmse = _number(row.get("rmse_mV"))
    if rmse is None:
        rmse = _number(row.get("fit_rmse_mV"))
    target_ok = (
        _truth(row.get("fit_gate_passed"))
        and rmse is not None
        and rmse < 15.0
        and (_number(row.get("accepted_runs")) or 0) >= 3
    )
    baseline_rmse = _number(row.get("baseline_fit_rmse_mV"))
    baseline_ok = (
        _text(row.get("baseline_fit_status")).casefold() == "accepted"
        and baseline_rmse is not None
        and baseline_rmse < 15.0
    )
    return target_ok and baseline_ok


def _bound_details(row: dict[str, Any] | None) -> tuple[str, int | None, bool | None]:
    if not row:
        return "not_recorded_for_historical_fit", None, None
    raw = (
        _text(row.get("parameter_at_bound_flags"))
        or _text(row.get("parameter_bounds_json"))
        or _text(row.get("target_parameter_bounds_json"))
    )
    if not raw:
        return "not_recorded", None, None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return "unparseable", None, None
    if isinstance(parsed, list):
        flagged = [
            item for item in parsed
            if isinstance(item, dict) and _truth(item.get("exact_or_near_bound"))
        ]
        return raw, len(flagged), bool(flagged)
    if isinstance(parsed, dict):
        flagged = [key for key, value in parsed.items() if _truth(value)]
        return raw, len(flagged), bool(flagged)
    return raw, None, None


def _fit_record(
    observation: dict[str, Any],
    direction: str,
    history_lookup: dict[tuple[int, str, int, str], dict[str, Any]],
    audit_lookup: dict[tuple[int, str, int, str], list[dict[str, Any]]],
    eol_lookup: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    step_valid = observation["step10_valid"] if direction == "charge" else observation["step8_valid"]
    key = (
        observation.get("part"),
        observation["physical_cell_id"],
        observation.get("rpt_index"),
        direction,
    )
    hist = history_lookup.get(key) if observation.get("rpt_index") is not None else None
    eol_key = (
        observation.get("part"),
        observation["physical_cell_id"],
        observation.get("rpt_index"),
        observation.get("checkpoint_id", ""),
        direction,
    )
    eol = eol_lookup.get(eol_key)
    accepted = False
    fit_status = ""
    source: dict[str, Any] | None = None
    if hist is not None:
        source = hist
        accepted = _history_accepts(hist)
    elif eol is not None:
        source = eol
        accepted = _eol_fit_is_accepted(eol)
    elif observation.get("rpt_index") is not None:
        audits = audit_lookup.get(key, [])
        if len(audits) == 1:
            source = audits[0]
            audit_accepted = _truth(source.get("fit_accepted_flag")) and _truth(source.get("passes_recorded_gate"))
            fit_status = "accepted_not_in_primary_cohort" if audit_accepted else "attempted_rejected_or_gate_failed"
        elif len(audits) > 1:
            fit_status = "ambiguous_fit_audit_records"
        else:
            fit_status = "no_saved_fit_record"
    else:
        fit_status = "no_eol_fit_record"

    if not observation["mapping_verified"]:
        status = "excluded_unverified_or_ambiguous_mapping"
        accepted = False
    elif not step_valid:
        status = "missing_or_invalid_direction_step"
        accepted = False
    elif hist is not None:
        status = "accepted_historical" if _history_accepts(hist) else "historical_gate_not_confirmed"
    elif eol is not None:
        status = "accepted_eol" if _eol_fit_is_accepted(eol) else "eol_fit_or_bol_gate_failed"
    else:
        status = fit_status

    raw_values: dict[str, float | None] = {}
    for metric, column in HISTORY_METRICS.items():
        if source is None:
            value = None
        elif column in source:
            value = _number(source.get(column))
        else:
            value = _number(source.get(EOL_METRICS[metric]))
        raw_values[metric] = value
    if source is not None and _text(source.get("result_table")):
        rmse = _number(source.get("fit_rmse_mV"))
        accepted_runs = _number(source.get("accepted_runs"))
        total_runs = _number(source.get("total_runs"))
        source_file = _text(source.get("source_file"))
    elif source is not None:
        rmse = _number(source.get("fit_rmse_mV"))
        if rmse is None:
            rmse = _number(source.get("rmse_mV"))
        accepted_runs = _number(source.get("accepted_runs"))
        total_runs = _number(source.get("total_runs"))
        source_file = _text(source.get("source_file"))
    else:
        rmse = accepted_runs = total_runs = None
        source_file = ""
    bounds, bound_count, bound_any = _bound_details(source)
    result: dict[str, Any] = {
        "status": status,
        "accepted": bool(accepted),
        "rmse_mV": rmse,
        "accepted_runs": accepted_runs,
        "total_runs": total_runs,
        "source_file": source_file or observation["source_file"],
        "parameter_bound_status": bounds,
        "parameter_bound_count": bound_count,
        "parameter_bound_any": bound_any,
    }
    for metric, value in raw_values.items():
        result[metric] = value
    return result


def _eol_observations(
    base_observations: list[dict[str, Any]],
    eol_manifest: pd.DataFrame,
    eol_fits: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[tuple[Any, ...], dict[str, Any]], list[dict[str, Any]]]:
    by_rpt: dict[tuple[int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for observation in base_observations:
        if observation.get("rpt_index") is not None:
            by_rpt[(observation["part"], observation["physical_cell_id"], observation["rpt_index"])].append(observation)
    fits_by_checkpoint = {
        str(key): group.to_dict("records")
        for key, group in eol_fits.groupby("checkpoint_id", dropna=False)
    }
    extras: list[dict[str, Any]] = []
    eol_lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    conflicts: list[dict[str, Any]] = []
    for row in eol_manifest.to_dict("records"):
        part = _part(row.get("part"))
        group = _part(row.get("group"))
        cell = _text(row.get("physical_cell_id"))
        if part not in {2, 3} or group not in ALLOWED_GROUPS or not cell:
            continue
        checkpoint = _text(row.get("checkpoint_id"))
        manifest_status = _text(row.get("mapping_status"))
        verified = "verified" in manifest_status.casefold() and not any(
            word in manifest_status.casefold() for word in ("ambiguous", "mismatch", "unverified")
        )
        rpt_idx = _rpt(row.get("rpt_index"))
        assignment = _rpt(row.get("guide_rpt_assignment"))
        if part == 3 and (not verified or rpt_idx is None or assignment != rpt_idx):
            conflicts.append({"checkpoint_id": checkpoint, "physical_cell_id": cell, "reason": "Part 3 Guide RPT assignment missing, unverified, or inconsistent"})
            continue
        checkpoint_fits = fits_by_checkpoint.get(checkpoint, [])
        fit_rows = {
            _text(fit.get("direction")).casefold(): fit
            for fit in checkpoint_fits
            if _text(fit.get("direction")).casefold() in {"charge", "discharge"}
        }
        observation: dict[str, Any] | None = None
        if part == 3 and rpt_idx is not None:
            candidates = by_rpt.get((part, cell, rpt_idx), [])
            if len(candidates) > 1:
                conflicts.append({"checkpoint_id": checkpoint, "physical_cell_id": cell, "reason": "duplicate canonical mapping key"})
                continue
            if candidates:
                candidate = candidates[0]
                if _source_key(candidate["source_file"]) != _source_key(row.get("source_file")):
                    conflicts.append({"checkpoint_id": checkpoint, "physical_cell_id": cell, "reason": "Guide EoL source differs from canonical mapping at the assigned RPT"})
                    continue
                observation = candidate
                observation["checkpoint_id"] = checkpoint
                observation["is_eol"] = True
                observation["source_type"] = "Guide-mapped Part 3 EoL RPT"
            else:
                observation = {
                    "observation_key": (part, cell, rpt_idx),
                    "part": part,
                    "group": group,
                    "physical_cell_id": cell,
                    "rpt_index": rpt_idx,
                    "rpt_or_checkpoint": f"RPT{rpt_idx}",
                    "checkpoint_id": checkpoint,
                    "source_file": _text(row.get("source_file")),
                    "mapping_status": manifest_status,
                    "mapping_verified": verified,
                    "step8_valid": _valid_segment(row.get("step8_status")),
                    "step10_valid": _valid_segment(row.get("step10_status")),
                    "is_eol": True,
                    "is_unnumbered_eol": False,
                    "prior_rpt_for_eol": _rpt(row.get("prior_audit_last_mapped_rpt")),
                    "sequence_key": cell,
                    "position": float(rpt_idx),
                    "mapping_rows": 1,
                    "source_type": "Guide-mapped Part 3 EoL RPT",
                }
                base_observations.append(observation)
                by_rpt[(part, cell, rpt_idx)].append(observation)
        elif part == 2:
            prior_rpt = _rpt(row.get("prior_audit_last_mapped_rpt"))
            observation = {
                "observation_key": ("EOL", checkpoint),
                "part": part,
                "group": group,
                "physical_cell_id": cell,
                "rpt_index": None,
                "rpt_or_checkpoint": checkpoint or "Part 2 unnumbered EoL",
                "checkpoint_id": checkpoint,
                "source_file": _text(row.get("source_file")),
                "mapping_status": manifest_status,
                "mapping_verified": verified,
                "step8_valid": _valid_segment(row.get("step8_status")),
                "step10_valid": _valid_segment(row.get("step10_status")),
                "is_eol": True,
                "is_unnumbered_eol": True,
                "prior_rpt_for_eol": prior_rpt,
                "sequence_key": cell,
                "position": float(prior_rpt) + 0.5 if prior_rpt is not None else math.inf,
                "mapping_rows": 1,
                "source_type": "Part 2 unnumbered Guide EoL checkpoint",
            }
            extras.append(observation)
        if observation is None:
            continue
        for direction, fit in fit_rows.items():
            eol_lookup[(part, cell, rpt_idx, checkpoint, direction)] = fit
            if observation.get("rpt_index") is None:
                eol_lookup[(part, cell, None, checkpoint, direction)] = fit
    return extras, eol_lookup, conflicts


def _fit_audit_status(
    key: tuple[int, str, int, str],
    audit_lookup: dict[tuple[int, str, int, str], list[dict[str, Any]]],
) -> str:
    rows = audit_lookup.get(key, [])
    if not rows:
        return "no_saved_fit_record"
    if len(rows) > 1:
        return "ambiguous_fit_audit_records"
    row = rows[0]
    if _truth(row.get("fit_accepted_flag")) and _truth(row.get("passes_recorded_gate")):
        return "accepted_not_in_primary_cohort"
    return "attempted_rejected_or_gate_failed"


def _make_observation_table() -> tuple[pd.DataFrame, dict[str, Any]]:
    _assert_inputs()
    history = pd.read_csv(HISTORY_PATH, dtype=str, keep_default_na=False)
    fit_audit = pd.read_csv(FIT_AUDIT_PATH, dtype=str, keep_default_na=False)
    mapping = pd.read_csv(MAPPING_PATH, dtype=str, keep_default_na=False)
    eol_fits = pd.read_csv(EOL_FITS_PATH, dtype=str, keep_default_na=False)
    eol_manifest = pd.read_csv(EOL_MANIFEST_PATH, dtype=str, keep_default_na=False)
    audit_lookup = _audit_rows(fit_audit)
    history_lookup, duplicate_history = _history_lookup(history)
    observations = _mapping_records(mapping)
    extras, eol_lookup, eol_conflicts = _eol_observations(observations, eol_manifest, eol_fits)
    observations.extend(extras)
    observations.sort(
        key=lambda item: (
            item["physical_cell_id"],
            item["position"],
            item["part"],
            item["checkpoint_id"],
        )
    )

    accepted_history_keys = set(history_lookup)
    mapped_history_keys: set[tuple[int, str, int, str]] = set()
    for observation in observations:
        if observation.get("rpt_index") is not None:
            for direction in ("charge", "discharge"):
                hist_key = (
                    observation["part"],
                    observation["physical_cell_id"],
                    observation["rpt_index"],
                    direction,
                )
                if hist_key in history_lookup:
                    mapped_history_keys.add(hist_key)
        observation["chronological_position"] = 0
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for observation in observations:
        if observation["mapping_verified"]:
            by_cell[observation["physical_cell_id"]].append(observation)
    for cell_records in by_cell.values():
        cell_records.sort(key=lambda item: (item["position"], item["part"], item["checkpoint_id"]))
        for index, observation in enumerate(cell_records, start=1):
            observation["chronological_position"] = index

    result_rows: list[dict[str, Any]] = []
    for observation in observations:
        record = dict(observation)
        for direction in ("charge", "discharge"):
            fit = _fit_record(observation, direction, history_lookup, audit_lookup, eol_lookup)
            if not fit["status"].startswith(("accepted_historical", "accepted_eol")) and observation.get("rpt_index") is not None:
                key = (observation["part"], observation["physical_cell_id"], observation["rpt_index"], direction)
                if fit["status"] in {"no_saved_fit_record", "accepted_not_in_primary_cohort", "attempted_rejected_or_gate_failed", "ambiguous_fit_audit_records"}:
                    fit["status"] = _fit_audit_status(key, audit_lookup)
            record[f"{direction}_status"] = fit["status"]
            record[f"{direction}_accepted"] = fit["accepted"]
            record[f"{direction}_rmse_mV"] = fit["rmse_mV"]
            record[f"{direction}_accepted_runs"] = fit["accepted_runs"]
            record[f"{direction}_total_runs"] = fit["total_runs"]
            record[f"{direction}_source_file"] = fit["source_file"]
            record[f"{direction}_parameter_bound_status"] = fit["parameter_bound_status"]
            record[f"{direction}_parameter_bound_count"] = fit["parameter_bound_count"]
            record[f"{direction}_parameter_bound_any"] = fit["parameter_bound_any"]
            for metric in HISTORY_METRICS:
                record[f"{direction}_{metric}_percent"] = fit[metric]
        result_rows.append(record)

    unmatched_history = [
        {
            "part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "direction": key[3],
            "reason": "accepted history fit did not resolve to a unique verified mapped RPT",
        }
        for key in sorted(accepted_history_keys - mapped_history_keys)
    ]
    excluded_history = []
    for key in sorted(accepted_history_keys & mapped_history_keys):
        part, cell, rpt_idx, direction = key
        matching = [
            item for item in observations
            if item.get("rpt_index") == rpt_idx
            and item.get("part") == part
            and item.get("physical_cell_id") == cell
        ]
        if len(matching) != 1 or not matching[0]["mapping_verified"]:
            excluded_history.append({"part": part, "physical_cell_id": cell, "rpt_index": rpt_idx, "direction": direction, "reason": "mapping not uniquely verified"})
            continue
        valid_step = matching[0]["step10_valid"] if direction == "charge" else matching[0]["step8_valid"]
        if not valid_step:
            excluded_history.append({"part": part, "physical_cell_id": cell, "rpt_index": rpt_idx, "direction": direction, "reason": "direction-specific Step 8/10 mapping is invalid"})
    result = pd.DataFrame(result_rows)
    if not result.empty:
        result = result.sort_values(
            ["physical_cell_id", "chronological_position", "part", "checkpoint_id"],
            kind="stable",
        ).reset_index(drop=True)
    metadata = {
        "history_rows": len(history),
        "history_accepted_rows": sum(_history_accepts(row) for row in history_lookup.values()),
        "verified_mapped_rpt_rows": sum(bool(item["mapping_verified"]) for item in observations),
        "mapped_observation_rows": len(observations),
        "unmatched_history_rows": unmatched_history,
        "history_rows_excluded_by_mapping_or_step": excluded_history,
        "duplicate_history_keys": duplicate_history,
        "eol_mapping_conflicts": eol_conflicts,
        "fit_audit_ambiguous_keys": sum(len(rows) > 1 for rows in audit_lookup.values()),
        "part2_groups_5_6_excluded": True,
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in (HISTORY_PATH, FIT_AUDIT_PATH, MAPPING_PATH, EOL_FITS_PATH, EOL_MANIFEST_PATH)
        },
    }
    return result, metadata


def _accepted_value(row: dict[str, Any], direction: str, metric: str) -> float | None:
    if not _truth(row.get(f"{direction}_accepted")):
        return None
    return _number(row.get(f"{direction}_{metric}_percent"))


def _direction_rates(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numbered = frame[
        frame["mapping_verified"]
        & ~frame["is_unnumbered_eol"].map(_truth)
    ]
    for cell, group in numbered.groupby("physical_cell_id"):
        for metric in HISTORY_METRICS:
            paired: list[tuple[float, float]] = []
            for _, row in group.iterrows():
                charge = _accepted_value(row.to_dict(), "charge", metric)
                discharge = _accepted_value(row.to_dict(), "discharge", metric)
                if charge is not None and discharge is not None:
                    paired.append((charge, discharge))
            charge_positive = sum(charge >= 0 for charge, _ in paired)
            discharge_positive = sum(discharge >= 0 for _, discharge in paired)
            n = len(paired)
            charge_rate = charge_positive / n if n else None
            discharge_rate = discharge_positive / n if n else None
            if not n:
                primary = "undetermined"
            elif charge_rate > discharge_rate:
                primary = "charge"
            elif discharge_rate > charge_rate:
                primary = "discharge"
            else:
                primary = "tie"
            rows.append(
                {
                    "physical_cell_id": cell,
                    "metric": metric,
                    "paired_accepted_checkpoints": n,
                    "charge_nonnegative_count": charge_positive,
                    "charge_negative_count": n - charge_positive,
                    "charge_nonnegative_rate": charge_rate,
                    "discharge_nonnegative_count": discharge_positive,
                    "discharge_negative_count": n - discharge_positive,
                    "discharge_nonnegative_rate": discharge_rate,
                    "h2_primary_direction": primary,
                }
            )
    return pd.DataFrame(rows)


def _select_h1(sequence: list[dict[str, Any]], metric: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    charge_values = [_accepted_value(row, "charge", metric) for row in sequence]
    full_charge = bool(sequence) and all(value is not None for value in charge_values)
    all_charge_nonnegative = full_charge and all(value >= 0 for value in charge_values if value is not None)
    primary = "charge" if all_charge_nonnegative else "discharge"
    selected: list[dict[str, Any]] = []
    for row in sequence:
        d_value = _accepted_value(row, "discharge", metric)
        c_value = _accepted_value(row, "charge", metric)
        source = ""
        value: float | None = None
        reason = ""
        fallback = False
        if all_charge_nonnegative:
            source, value, reason = "charge", c_value, "complete_nonnegative_charge_sequence"
        elif d_value is not None and d_value >= 0:
            source, value, reason = "discharge", d_value, "default_discharge_nonnegative"
        elif d_value is not None and d_value < 0 and c_value is not None and c_value >= 0:
            source, value, reason, fallback = "charge", c_value, "fallback_for_negative_discharge", True
        elif d_value is not None and d_value < 0:
            source, value, reason = "discharge", d_value, "negative_discharge_unresolved"
        else:
            reason = "default_discharge_missing_or_rejected"
        selected.append(
            {
                **row,
                "metric": metric,
                "h1_primary_direction": primary,
                "h1_charge_complete_coverage": full_charge,
                "h1_all_charge_nonnegative": all_charge_nonnegative,
                "h1_selected_percent": value,
                "h1_source_direction": source,
                "h1_fallback_used": fallback,
                "h1_selection_status": reason,
            }
        )
    return selected, {
        "primary_direction": primary,
        "charge_complete_coverage": full_charge,
        "all_charge_nonnegative": all_charge_nonnegative,
    }


def _select_h2(sequence: list[dict[str, Any]], metric: str, primary: str) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in sequence:
        c_value = _accepted_value(row, "charge", metric)
        d_value = _accepted_value(row, "discharge", metric)
        value: float | None = None
        source = ""
        reason = ""
        fallback = False
        if primary in {"tie", "undetermined"}:
            reason = f"primary_{primary}_no_direction_selected"
        else:
            primary_value = c_value if primary == "charge" else d_value
            other_direction = "discharge" if primary == "charge" else "charge"
            other_value = d_value if primary == "charge" else c_value
            if primary_value is None:
                reason = "primary_fit_missing_or_rejected"
            elif primary_value >= 0:
                value, source, reason = primary_value, primary, "primary_nonnegative"
            elif other_value is not None and other_value >= 0:
                value, source, reason, fallback = other_value, other_direction, "fallback_for_negative_primary", True
            else:
                value, source, reason = primary_value, primary, "negative_primary_unresolved"
        selected.append(
            {
                **row,
                "metric": metric,
                "h2_primary_direction": primary,
                "h2_charge_candidate_percent": c_value,
                "h2_discharge_candidate_percent": d_value,
                "h2_selected_percent": value,
                "h2_source_direction": source,
                "h2_fallback_used": fallback,
                "h2_selection_status": reason,
            }
        )
    return selected


def _long_trajectories(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rate_lookup = {
        (row["physical_cell_id"], row["metric"]): row["h2_primary_direction"]
        for row in rates.to_dict("records")
    }
    h1_rows: list[dict[str, Any]] = []
    h2_rows: list[dict[str, Any]] = []
    sequences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame.to_dict("records"):
        if row.get("mapping_verified"):
            sequences[row["physical_cell_id"]].append(row)
    for cell, sequence in sequences.items():
        sequence.sort(key=lambda row: (row["position"], row["part"], row["checkpoint_id"]))
        for metric in HISTORY_METRICS:
            h1_selected, _ = _select_h1(sequence, metric)
            h1_rows.extend(h1_selected)
            primary = rate_lookup.get((cell, metric), "undetermined")
            h2_rows.extend(_select_h2(sequence, metric, primary))
    return pd.DataFrame(h1_rows), pd.DataFrame(h2_rows)


def _method_rows(frame: pd.DataFrame, h1: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        base = row.to_dict()
        for metric in HISTORY_METRICS:
            for method, direction in (("H0_Discharge", "discharge"), ("H0_Charge", "charge")):
                records.append(
                    {
                        **base,
                        "metric": metric,
                        "method": method,
                        "estimate_percent": _accepted_value(base, direction, metric),
                        "source_direction": direction if _accepted_value(base, direction, metric) is not None else "",
                        "fallback_used": False,
                        "selection_status": base[f"{direction}_status"],
                    }
                )
    for df, method, value_col, source_col, fallback_col, status_col in (
        (h1, "H1", "h1_selected_percent", "h1_source_direction", "h1_fallback_used", "h1_selection_status"),
        (h2, "H2", "h2_selected_percent", "h2_source_direction", "h2_fallback_used", "h2_selection_status"),
    ):
        for row in df.to_dict("records"):
            records.append(
                {
                    **row,
                    "method": method,
                    "estimate_percent": _number(row.get(value_col)),
                    "source_direction": row.get(source_col, ""),
                    "fallback_used": bool(row.get(fallback_col, False)),
                    "selection_status": row.get(status_col, ""),
                }
            )
    return pd.DataFrame(records)


def _temporal_consistency(methods: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (cell, metric, method), group in methods.groupby(["physical_cell_id", "metric", "method"], sort=False):
        group = group[group["mapping_verified"]].sort_values(
            ["position", "part", "checkpoint_id"], kind="stable"
        )
        estimates = [_number(v) for v in group["estimate_percent"].tolist()]
        sources = [_text(v) for v in group["source_direction"].tolist()]
        dec_magnitudes: list[float] = []
        abs_deltas: list[float] = []
        switches = 0
        eol_checks = 0
        eol_passes = 0
        eol_violations = 0
        eol_not_evaluable = 0
        transition_details: list[dict[str, Any]] = []
        eol_details: list[dict[str, Any]] = []
        for index in range(1, len(group)):
            previous, current = estimates[index - 1], estimates[index]
            previous_row = group.iloc[index - 1]
            current_row = group.iloc[index]
            delta = current - previous if previous is not None and current is not None else None
            transition_details.append(
                {
                    "from_checkpoint": previous_row["rpt_or_checkpoint"],
                    "to_checkpoint": current_row["rpt_or_checkpoint"],
                    "from_estimate_percent": previous,
                    "to_estimate_percent": current,
                    "delta_pp": delta,
                    "from_direction": sources[index - 1],
                    "to_direction": sources[index],
                    "evaluable": delta is not None,
                }
            )
            if previous is not None and current is not None:
                assert delta is not None
                abs_deltas.append(abs(delta))
                if delta < 0:
                    dec_magnitudes.append(-delta)
                if sources[index - 1] and sources[index] and sources[index - 1] != sources[index]:
                    switches += 1
            if _truth(current_row.get("is_unnumbered_eol")):
                eol_checks += 1
                if previous is not None and current is not None:
                    passed = current >= previous
                    eol_passes += int(passed)
                    eol_violations += int(not passed)
                    eol_status = "pass" if passed else "violation"
                else:
                    eol_not_evaluable += 1
                    eol_status = "not_evaluable_missing_adjacent_estimate"
                eol_details.append(
                    {
                        "checkpoint_id": current_row["checkpoint_id"],
                        "previous_checkpoint": previous_row["rpt_or_checkpoint"],
                        "previous_estimate_percent": previous,
                        "eol_estimate_percent": current,
                        "delta_pp": delta,
                        "status": eol_status,
                    }
                )
        estimate_count = sum(value is not None for value in estimates)
        record: dict[str, Any] = {
            "physical_cell_id": cell,
            "metric": metric,
            "method": method,
            "verified_checkpoints": len(group),
            "available_estimates": estimate_count,
            "missing_value_count": len(group) - estimate_count,
            "negative_estimates": sum(value < 0 for value in estimates if value is not None),
            "adjacent_evaluable_transitions": len(abs_deltas),
            "decrease_count": len(dec_magnitudes),
            "total_decrease_magnitude_pp": sum(dec_magnitudes),
            "maximum_decrease_magnitude_pp": max(dec_magnitudes, default=None),
            "mean_absolute_adjacent_delta_pp": float(np.mean(abs_deltas)) if abs_deltas else None,
            "maximum_absolute_adjacent_delta_pp": max(abs_deltas, default=None),
            "direction_switches": switches,
            "eol_one_sided_checks": eol_checks,
            "eol_one_sided_passes": eol_passes,
            "eol_one_sided_violations": eol_violations,
            "eol_one_sided_not_evaluable": eol_not_evaluable,
            "adjacent_transition_details_json": json.dumps(transition_details, separators=(",", ":")),
            "eol_one_sided_details_json": json.dumps(eol_details, separators=(",", ":")),
        }
        for threshold in JUMP_THRESHOLDS_PP:
            record[f"abrupt_jumps_ge_{int(threshold)}pp"] = sum(delta >= threshold for delta in abs_deltas)
        rows.append(record)
    return pd.DataFrame(rows)


def _metric_summary(
    methods: pd.DataFrame,
    temporal: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = methods.copy()
    methods["is_common_key"] = False
    for metric, metric_group in methods.groupby("metric"):
        pivot = metric_group.pivot_table(
            index=["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id"],
            columns="method",
            values="estimate_percent",
            aggfunc="first",
        )
        needed = ["H0_Discharge", "H0_Charge", "H1", "H2"]
        common_keys = set(pivot.index[pivot.reindex(columns=needed).notna().all(axis=1)])
        metric_group["key_tuple"] = list(
            zip(metric_group["physical_cell_id"], metric_group["part"], metric_group["rpt_or_checkpoint"], metric_group["checkpoint_id"])
        )
        all_rows = metric_group[metric_group["mapping_verified"]]
        for scope, subset in (
            ("full_available", all_rows),
            ("common_all_four_methods", all_rows[all_rows["key_tuple"].isin(common_keys)]),
        ):
            for method, g in subset.groupby("method"):
                available = g[g["estimate_percent"].notna()]
                values = available["estimate_percent"].astype(float)
                cells = available["physical_cell_id"].nunique()
                temporal_subset = temporal[
                    temporal["metric"].eq(metric) & temporal["method"].eq(method)
                ]
                t = temporal_subset if scope == "full_available" else pd.DataFrame()
                rows.append(
                    {
                        "metric": metric,
                        "method": method,
                        "scope": scope,
                        "verified_checkpoints_in_scope": g.drop_duplicates(
                            ["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id"]
                        ).shape[0],
                        "available_estimates": len(available),
                        "missing_estimates": max(
                            0,
                            g.drop_duplicates(
                                ["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id"]
                            ).shape[0] - len(available),
                        ),
                        "physical_cells": cells,
                        "negative_estimates": int((values < 0).sum()),
                        "nonnegative_rate": float((values >= 0).mean()) if len(values) else None,
                        "mean_estimate_percent": float(values.mean()) if len(values) else None,
                        "mean_selected_fit_rmse_mV": float(pd.to_numeric(available["charge_rmse_mV"], errors="coerce").mean()) if method in {"H0_Charge"} and len(available) else (
                            float(pd.to_numeric(available["discharge_rmse_mV"], errors="coerce").mean()) if method == "H0_Discharge" and len(available) else (
                                float(pd.to_numeric(available.apply(lambda r: r.get(f"{_text(r.get('source_direction'))}_rmse_mV"), axis=1), errors="coerce").mean()) if len(available) else None
                            )
                        ),
                        "temporal_decreases_full_coverage": int(t["decrease_count"].sum()) if not t.empty else None,
                        "abrupt_jumps_ge_10pp_full_coverage": int(t["abrupt_jumps_ge_10pp"].sum()) if not t.empty else None,
                    }
                )
            for scope, subset in (
                ("per_cell_full_available", all_rows),
                ("per_cell_common_all_four_methods", all_rows[all_rows["key_tuple"].isin(common_keys)]),
            ):
                for cell, cell_rows in subset.groupby("physical_cell_id"):
                    for method, g in cell_rows.groupby("method"):
                        available = g[g["estimate_percent"].notna()]
                        values = pd.to_numeric(available["estimate_percent"], errors="coerce")
                        rows.append(
                            {
                                "metric": metric,
                                "method": method,
                                "scope": scope,
                                "physical_cell_id": cell,
                                "verified_checkpoints_in_scope": g.drop_duplicates(
                                    ["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id"]
                                ).shape[0],
                                "available_estimates": len(available),
                                "missing_estimates": max(
                                    0,
                                    g.drop_duplicates(
                                        ["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id"]
                                    ).shape[0] - len(available),
                                ),
                                "physical_cells": int(len(available) > 0),
                                "negative_estimates": int((values < 0).sum()),
                                "nonnegative_rate": float((values >= 0).mean()) if len(values) else None,
                                "mean_estimate_percent": float(values.mean()) if len(values) else None,
                                "mean_selected_fit_rmse_mV": float(
                                    pd.to_numeric(
                                        available.apply(
                                            lambda r: r.get(f"{_text(r.get('source_direction'))}_rmse_mV"),
                                            axis=1,
                                        ),
                                        errors="coerce",
                                    ).mean()
                                ) if len(available) else None,
                                "temporal_decreases_full_coverage": None,
                                "abrupt_jumps_ge_10pp_full_coverage": None,
                            }
                        )

        h1_h2 = metric_group[
            metric_group["mapping_verified"]
            & metric_group["method"].isin({"H1", "H2"})
        ].pivot_table(
            index=["physical_cell_id", "position", "part", "rpt_or_checkpoint", "checkpoint_id"],
            columns="method",
            values="estimate_percent",
            aggfunc="first",
        ).reset_index()
        transition_deltas: dict[str, list[float]] = {"H1": [], "H2": []}
        transition_cells: dict[str, set[str]] = {"H1": set(), "H2": set()}
        for cell, cell_rows in h1_h2.groupby("physical_cell_id", sort=False):
            cell_rows = cell_rows.sort_values("position", kind="stable")
            for index in range(1, len(cell_rows)):
                previous = cell_rows.iloc[index - 1]
                current = cell_rows.iloc[index]
                if any(pd.isna(value) for value in (previous.get("H1"), current.get("H1"), previous.get("H2"), current.get("H2"))):
                    continue
                for method in ("H1", "H2"):
                    transition_deltas[method].append(float(current[method]) - float(previous[method]))
                    transition_cells[method].add(str(cell))
        pair_count = len(transition_deltas["H1"])
        for method in ("H1", "H2"):
            deltas = transition_deltas[method]
            row = {
                "metric": metric,
                "method": method,
                "scope": "common_H1_H2_adjacent_transitions",
                "physical_cell_id": "ALL",
                "verified_checkpoints_in_scope": pair_count,
                "available_estimates": pair_count * 2,
                "missing_estimates": 0,
                "physical_cells": len(transition_cells[method]),
                "negative_estimates": None,
                "nonnegative_rate": None,
                "mean_estimate_percent": None,
                "mean_selected_fit_rmse_mV": None,
                "temporal_decreases_full_coverage": sum(delta < 0 for delta in deltas),
                "abrupt_jumps_ge_10pp_full_coverage": sum(abs(delta) >= 10 for delta in deltas),
                "mean_absolute_adjacent_delta_pp": float(np.mean(np.abs(deltas))) if deltas else None,
                "maximum_absolute_adjacent_delta_pp": max((abs(delta) for delta in deltas), default=None),
                "matched_transition_pairs": pair_count,
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _fallback_audit(h1: pd.DataFrame, h2: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for data, method in ((h1, "H1"), (h2, "H2")):
        for row in data.to_dict("records"):
            is_fallback = _truth(row.get(f"{method.lower()}_fallback_used"))
            default_direction = "discharge" if method == "H1" else _text(row.get("h2_primary_direction"))
            default_value = _accepted_value(row, default_direction, row["metric"]) if default_direction in {"charge", "discharge"} else None
            if not is_fallback and not (default_value is not None and default_value < 0):
                continue
            selected_direction = _text(row.get(f"{method.lower()}_source_direction"))
            selected_value = _number(row.get(f"{method.lower()}_selected_percent"))
            parameter_bound_status = row.get(f"{selected_direction}_parameter_bound_status", "unavailable") if selected_direction else "unavailable"
            rows.append(
                {
                    "method": method,
                    "part": row["part"],
                    "group": row["group"],
                    "physical_cell_id": row["physical_cell_id"],
                    "rpt_or_checkpoint": row["rpt_or_checkpoint"],
                    "checkpoint_id": row["checkpoint_id"],
                    "metric": row["metric"],
                    "primary_direction": row.get("h1_primary_direction", row.get("h2_primary_direction", "")),
                    "default_direction": default_direction,
                    "default_estimate_percent": default_value,
                    "default_fit_status": row.get(f"{default_direction}_status", "") if default_direction in {"charge", "discharge"} else "",
                    "selected_direction": selected_direction,
                    "selected_estimate_percent": selected_value,
                    "fallback_used": is_fallback,
                    "fallback_reason": row.get("h1_selection_status", row.get("h2_selection_status", "")),
                    "selected_fit_rmse_mV": row.get(f"{selected_direction}_rmse_mV") if selected_direction else None,
                    "selected_fit_accepted_runs": row.get(f"{selected_direction}_accepted_runs") if selected_direction else None,
                    "selected_fit_total_runs": row.get(f"{selected_direction}_total_runs") if selected_direction else None,
                    "selected_parameter_bound_status": parameter_bound_status,
                    "selected_parameter_bound_count": row.get(f"{selected_direction}_parameter_bound_count") if selected_direction else None,
                    "selected_parameter_bound_any": row.get(f"{selected_direction}_parameter_bound_any") if selected_direction else None,
                    "delta_selected_vs_default_pp": selected_value - default_value if selected_value is not None and default_value is not None else None,
                }
            )
    return pd.DataFrame(rows)


def _plot_cell(cell: str, methods: pd.DataFrame, figures: Path) -> None:
    selected = methods[
        methods["physical_cell_id"].eq(cell)
        & methods["mapping_verified"]
    ]
    if selected.empty:
        return
    for metric in HISTORY_METRICS:
        fig, ax = plt.subplots(figsize=(12, 5.5))
        group = selected[selected["metric"].eq(metric)].copy()
        records = (
            group[["physical_cell_id", "part", "rpt_or_checkpoint", "checkpoint_id", "position", "method", "estimate_percent", "source_direction", "fallback_used", "selection_status", "is_eol", "is_unnumbered_eol"]]
            .drop_duplicates(["part", "rpt_or_checkpoint", "checkpoint_id", "method"])
        )
        styles = {
            ("H0_Charge", "charge"): ("#d98200", "o", "H0 Charge"),
            ("H0_Discharge", "discharge"): ("#1464a5", "^", "H0 Discharge"),
            ("H1", "charge"): ("#07843b", "D", "H1 from Charge"),
            ("H1", "discharge"): ("#07843b", "o", "H1 from Discharge"),
            ("H2", "charge"): ("#a04db3", "s", "H2 from Charge"),
            ("H2", "discharge"): ("#a04db3", "^", "H2 from Discharge"),
        }
        for (method, source), (color, marker, label) in styles.items():
            points = records[
                records["method"].eq(method)
                & records["source_direction"].eq(source)
                & records["estimate_percent"].notna()
            ]
            if points.empty:
                continue
            ax.scatter(points["position"], points["estimate_percent"], c=color, marker=marker, s=48, label=label, alpha=0.85)
        fallbacks = records[records["fallback_used"].map(_truth) & records["estimate_percent"].notna()]
        if not fallbacks.empty:
            ax.scatter(fallbacks["position"], fallbacks["estimate_percent"], marker="*", s=145, facecolors="none", edgecolors="#111111", linewidths=1.2, label="Fallback replacement", zorder=5)
        unresolved = records[
            records["selection_status"].astype(str).str.contains("unresolved", case=False, na=False)
            & records["estimate_percent"].notna()
        ]
        if not unresolved.empty:
            ax.scatter(unresolved["position"], unresolved["estimate_percent"], marker="x", s=75, c="#c00000", label="Unresolved negative", zorder=6)
        eol = group[group["is_eol"].map(_truth)].drop_duplicates(["position", "rpt_or_checkpoint"])
        for _, row in eol.iterrows():
            ax.axvline(float(row["position"]), color="#777777", linewidth=0.7, linestyle=":")
        tick_rows = group[["position", "rpt_or_checkpoint", "is_unnumbered_eol"]].drop_duplicates().sort_values("position")
        ticks = tick_rows["position"].tolist()
        labels = [
            "EoL (unnumbered)" if _truth(flag) else label
            for label, flag in zip(tick_rows["rpt_or_checkpoint"], tick_rows["is_unnumbered_eol"])
        ]
        if len(ticks) <= 24:
            ax.set_xticks(ticks, labels, rotation=45, ha="right")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(f"{metric} (%)")
        ax.set_xlabel("Chronological position (RPT ordinal; not cycle count)")
        ax.set_title(f"{cell} — {metric}: original directions and H1/H2")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", ncols=3, fontsize=8)
        fig.tight_layout()
        fig.savefig(figures / f"{cell}_{metric}.png", dpi=150)
        plt.close(fig)


def _write_report(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
    h1: pd.DataFrame,
    h2: pd.DataFrame,
    comparisons: pd.DataFrame,
    temporal: pd.DataFrame,
    fallback: pd.DataFrame,
    metadata: dict[str, Any],
) -> None:
    historical = frame[frame["mapping_verified"] & ~frame["is_eol"].map(_truth)]
    negative_candidates: list[dict[str, Any]] = []
    for row in frame[frame["mapping_verified"]].to_dict("records"):
        for metric in HISTORY_METRICS:
            discharge = _accepted_value(row, "discharge", metric)
            charge = _accepted_value(row, "charge", metric)
            if discharge is not None and discharge < 0:
                negative_candidates.append(
                    {
                        "part": row["part"],
                        "cell": row["physical_cell_id"],
                        "checkpoint": row["rpt_or_checkpoint"],
                        "metric": metric,
                        "discharge": discharge,
                        "charge": charge,
                        "charge_nonnegative": charge is not None and charge >= 0,
                    }
                )
    negative_frame = pd.DataFrame(negative_candidates)
    correspond_nonnegative = int(negative_frame["charge_nonnegative"].sum()) if not negative_frame.empty else 0
    h1_negative_replaced = 0
    for row in h1.to_dict("records"):
        d = _accepted_value(row, "discharge", row["metric"])
        if d is not None and d < 0 and _number(row.get("h1_selected_percent")) is not None and _number(row.get("h1_selected_percent")) >= 0:
            h1_negative_replaced += 1
    h2_negative_replaced = 0
    for row in h2.to_dict("records"):
        d = _accepted_value(row, "discharge", row["metric"])
        if d is not None and d < 0 and _number(row.get("h2_selected_percent")) is not None and _number(row.get("h2_selected_percent")) >= 0:
            h2_negative_replaced += 1

    h1_dec = int(temporal[temporal["method"].eq("H1")]["decrease_count"].sum())
    h2_dec = int(temporal[temporal["method"].eq("H2")]["decrease_count"].sum())
    h1_jumps = int(temporal[temporal["method"].eq("H1")]["abrupt_jumps_ge_10pp"].sum())
    h2_jumps = int(temporal[temporal["method"].eq("H2")]["abrupt_jumps_ge_10pp"].sum())
    actual_fallbacks = fallback[fallback["fallback_used"].map(_truth)] if not fallback.empty else pd.DataFrame()
    bound_known = actual_fallbacks[actual_fallbacks["selected_parameter_bound_any"].notna()] if not actual_fallbacks.empty else pd.DataFrame()
    bound_fallbacks = int(bound_known["selected_parameter_bound_any"].map(_truth).sum()) if not bound_known.empty else 0
    direction_counts = rates["h2_primary_direction"].value_counts().to_dict()
    h2_source_counts = (
        h2[h2["h2_selected_percent"].notna()]["h2_source_direction"].value_counts().to_dict()
        if not h2.empty else {}
    )
    common_directional_results: dict[str, tuple[int, int]] = {}
    join_keys = ["part", "physical_cell_id", "rpt_or_checkpoint", "checkpoint_id", "metric"]
    for metric in HISTORY_METRICS:
        candidates = h2[h2["metric"].eq(metric)].merge(
            h1[h1["metric"].eq(metric)][join_keys + ["h1_selected_percent"]],
            on=join_keys,
            validate="one_to_one",
        )
        discharge_col = f"discharge_{metric}_percent"
        charge_col = f"charge_{metric}_percent"
        common = candidates[
            candidates[discharge_col].notna()
            & candidates[charge_col].notna()
            & candidates["h1_selected_percent"].notna()
            & candidates["h2_selected_percent"].notna()
        ]
        same_as_charge = (
            int(
                np.isclose(
                    pd.to_numeric(common["h2_selected_percent"], errors="coerce"),
                    pd.to_numeric(common[charge_col], errors="coerce"),
                ).sum()
            )
            if not common.empty
            else 0
        )
        common_directional_results[metric] = (len(common), same_as_charge)
    common_temporal = comparisons[
        comparisons["scope"].eq("common_H1_H2_adjacent_transitions")
    ]

    lines = [
        "# Hybrid Charge/Discharge PyDMA Diagnostic Experiment",
        "",
        "## Scope and safeguards",
        "",
        "- This is a separate post-processing experiment on saved PyDMA fits; no PyDMA was rerun.",
        "- Original signed charge and discharge estimates are preserved. No value was clipped, made absolute, or selected for being positive except where the explicitly requested conditional fallback rule applies.",
        "- Parts 2 Groups 5–6 are excluded. Parts 1–2 Groups 1–4 share a cell sequence; Part 3 remains a separate domain and is only descriptively combined within its own physical cells.",
        "- Part 2 EoL observations remain unnumbered and are placed after the Guide-recorded prior RPT only for chronological comparison. Part 3 EoL observations use only the Guide-verified RPT6/RPT7 assignment.",
        "- A fit is eligible only when its saved fit passes RMSE <15 mV and at least 3 accepted runs; EoL additionally requires the saved same-cell, same-direction BoL fit to be accepted with RMSE <15 mV.",
        "- The 446-row historical accepted-fit source contains no fitted parameter vectors or bound flags. Those diagnostics are therefore marked unavailable for historical fits. EoL parameter-bound flags are carried from the saved EoL table.",
        "- Abrupt jumps are reported at 5, 10, and 20 percentage-point absolute change per adjacent verified checkpoint. The 10 pp threshold is a descriptive convention, not a physical limit; missing checkpoints are not bridged or interpolated.",
        "",
        "## Verified input coverage",
        "",
        f"- Historical accepted cohort source rows: {metadata['history_rows']}; verified mapped checkpoint rows in the analysis table: {metadata['verified_mapped_rpt_rows']}; total mapped and EoL observation rows: {metadata['mapped_observation_rows']}.",
        f"- Accepted historical fit keys that did not resolve to a unique verified mapping: {len(metadata['unmatched_history_rows'])}.",
        f"- Accepted historical fit rows excluded from direction selection due to an unverified mapping or invalid Step 8/10: {len(metadata['history_rows_excluded_by_mapping_or_step'])}. Their source estimates remain in the matched-fit table when available.",
        f"- EoL mapping conflicts excluded from automatic selection: {len(metadata['eol_mapping_conflicts'])}.",
        f"- EoL coverage: {metadata['part2_unnumbered_eol_observations']} Part 2 unnumbered checkpoints with {metadata['part2_unnumbered_eol_eligible_direction_fits']} accepted direction fits; {metadata['part3_guide_mapped_eol_observations']} Guide-mapped Part 3 RPT6/RPT7 checkpoints with {metadata['part3_guide_mapped_eol_eligible_direction_fits']} accepted direction fits. Duplicate numbered cell/RPT keys: {metadata['duplicate_numbered_rpt_keys']}.",
        f"- Duplicate accepted-history keys: {len(metadata['duplicate_history_keys'])}; ambiguous fit-audit keys: {metadata['fit_audit_ambiguous_keys']}.",
        "- Input file hashes are recorded in `run_manifest.json`.",
        "",
        "## Negative discharge matches and fallback",
        "",
        f"- Accepted negative discharge metric observations with a corresponding accepted nonnegative charge estimate: **{correspond_nonnegative} / {len(negative_frame)}**.",
        f"- Negative discharge metric observations replaced by H1 with a nonnegative estimate: **{h1_negative_replaced}**.",
        f"- Negative discharge metric observations replaced by H2 with a nonnegative estimate: **{h2_negative_replaced}**.",
        f"- H2 numbered-RPT cell/metric primary-direction selections: {direction_counts}. Unnumbered Part 2 EoL checkpoints do not affect these positive-rate denominators. Tied and undetermined cases remain unselected; no implicit tie preference is used.",
        f"- H2 selected estimate source directions: {h2_source_counts}.",
        f"- Fallback selections with recorded parameter-bound warnings: **{bound_fallbacks} / {len(bound_known)}** where bound flags are available; {len(actual_fallbacks) - len(bound_known)} fallback selections lack parameter histories.",
        "",
        "## H0, H1, H2 comparison",
        "",
        "See `hybrid_comparison.csv` for full-coverage and common-key summaries, including per-cell rows, missingness, nonnegative rates, fit RMSE, and temporal diagnostics.",
        "",
        f"- H1 decrease count across cell/metric sequences: {h1_dec}; H2: {h2_dec}.",
        f"- Adjacent jumps >=10 pp: H1 {h1_jumps}; H2 {h2_jumps}. Also inspect the 5 pp and 20 pp columns in `temporal_consistency.csv`.",
        "- For a fair H1/H2 temporal comparison, `hybrid_comparison.csv` also reports only adjacent transitions where both methods have estimates at both endpoints (`common_H1_H2_adjacent_transitions`).",
        f"- Fallback audit rows with selected-vs-default discontinuity >=10 pp: {int((pd.to_numeric(fallback.get('delta_selected_vs_default_pp', pd.Series(dtype=float)), errors='coerce').abs() >= 10).sum()) if not fallback.empty else 0}.",
        "- Matched H1/H2 adjacent transitions: "
        + "; ".join(
            f"{metric}: {int(common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H1')]['matched_transition_pairs'].iloc[0]) if not common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H1')].empty else 0} pairs, "
            f"H1/H2 decreases {int(common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H1')]['temporal_decreases_full_coverage'].iloc[0]) if not common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H1')].empty else 0}/"
            f"{int(common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H2')]['temporal_decreases_full_coverage'].iloc[0]) if not common_temporal[(common_temporal['metric'] == metric) & (common_temporal['method'] == 'H2')].empty else 0}"
            for metric in HISTORY_METRICS
        )
        + ".",
        "- On the common four-method subset, H2 equals charge-only on "
        + "; ".join(
            f"{metric} {same}/{total}"
            for metric, (total, same) in common_directional_results.items()
        )
        + " observations. Cathode LAM has no common H2 estimates because the positive-rate comparison ties everywhere.",
        "- Therefore H2 does not demonstrate a prediction or physical-validity advantage over charge-only; it has lower coverage, and the hybrid values are not independent ground truth.",
        "- These counts do not establish which direction is physically correct. A lower negative count alone is not a selection criterion.",
        "",
        "## Specified Part 2 EoL cases",
        "",
    ]
    for cell in ("G1-Cell20", "G3-Cell14"):
        group = frame[
            frame["physical_cell_id"].eq(cell)
            & frame["is_unnumbered_eol"].map(_truth)
        ]
        if group.empty:
            lines.append(f"- {cell}: no unique verified unnumbered Part 2 EoL record was eligible.")
            continue
        row = group.iloc[0].to_dict()
        lines.append(
            f"- {cell} ({row['checkpoint_id']}): prior Guide-mapped RPT {row['prior_rpt_for_eol']}; "
            f"charge status `{row['charge_status']}`, discharge status `{row['discharge_status']}`."
        )
        for metric in HISTORY_METRICS:
            charge = _accepted_value(row, "charge", metric)
            discharge = _accepted_value(row, "discharge", metric)
            h1_row = h1[(h1["physical_cell_id"] == cell) & h1["checkpoint_id"].eq(row["checkpoint_id"]) & h1["metric"].eq(metric)]
            h2_row = h2[(h2["physical_cell_id"] == cell) & h2["checkpoint_id"].eq(row["checkpoint_id"]) & h2["metric"].eq(metric)]
            c_temporal = temporal[
                temporal["physical_cell_id"].eq(cell)
                & temporal["metric"].eq(metric)
                & temporal["method"].eq("H0_Charge")
            ]
            if not h1_row.empty and not h2_row.empty:
                lines.append(
                    f"  - {metric}: Charge {charge if charge is not None else 'unavailable'}%, Discharge {discharge if discharge is not None else 'unavailable'}%; "
                    f"H1 {h1_row.iloc[0]['h1_selection_status']} ({h1_row.iloc[0]['h1_selected_percent']}%), "
                    f"H2 primary {h2_row.iloc[0]['h2_primary_direction']} ({h2_row.iloc[0]['h2_selection_status']}; {h2_row.iloc[0]['h2_selected_percent']}%)."
                )
            if not c_temporal.empty:
                t = c_temporal.iloc[0]
                details = json.loads(t["eol_one_sided_details_json"])
                case_detail = next(
                    (item for item in details if item["checkpoint_id"] == row["checkpoint_id"]),
                    None,
                )
                if case_detail:
                    charge_check = (
                        f"Charge EoL {case_detail['eol_estimate_percent']}%, prior accepted checkpoint "
                        f"{case_detail['previous_checkpoint']} {case_detail['previous_estimate_percent']}%, "
                        f"delta {case_detail['delta_pp']} pp ({case_detail['status']})."
                    )
                else:
                    charge_check = "Charge EoL one-sided comparison was not evaluable."
                lines.append(
                    f"    - {charge_check} Selected-fit boundary warnings are recorded in `fallback_audit.csv`."
                )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Hybrid selection is an exploratory diagnostic visualization, not validated physical degradation labeling. H1 and H2 can switch between direction-specific BoL/reference branches, creating discontinuities even when the selected value is nonnegative. The common-key comparison and the recorded source direction, RMSE, run counts, bounds, and missingness should be considered together. No method is declared physically superior from positivity or temporal smoothness alone.",
            "",
            "## Reproduction",
            "",
            "From the repository root on Windows, run `powershell -ExecutionPolicy Bypass -File experiments\\pydma_hybrid_direction_v1\\reproduce.ps1`. The experiment reads only saved reports, manifests, and PyDMA outputs; it does not fit PyDMA or modify canonical artifacts.",
            "",
        ]
    )
    (OUT / "hybrid_diagnostic_report.md").write_text("\n".join(lines), encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    frame, metadata = _make_observation_table()
    rates = _direction_rates(frame)
    h1, h2 = _long_trajectories(frame, rates)
    methods = _method_rows(frame, h1, h2)
    temporal = _temporal_consistency(methods)
    comparison = _metric_summary(methods, temporal)
    fallback = _fallback_audit(h1, h2)

    frame.to_csv(OUT / "matched_charge_discharge_fits.csv", index=False)
    rates.to_csv(OUT / "direction_positive_rates.csv", index=False)
    h1.to_csv(OUT / "h1_charge_first_results.csv", index=False)
    h2.to_csv(OUT / "h2_majority_positive_results.csv", index=False)
    comparison.to_csv(OUT / "hybrid_comparison.csv", index=False)
    temporal.to_csv(OUT / "temporal_consistency.csv", index=False)
    fallback.to_csv(OUT / "fallback_audit.csv", index=False)
    for cell in ("G1-Cell20", "G3-Cell14", "G7-Cell19"):
        _plot_cell(cell, methods, figures)
    metadata["output_observations"] = len(frame)
    metadata["output_h1_rows"] = len(h1)
    metadata["output_h2_rows"] = len(h2)
    metadata["unique_cells"] = int(frame.loc[frame["mapping_verified"], "physical_cell_id"].nunique())
    part2_eol = frame[frame["part"].eq(2) & frame["is_unnumbered_eol"].map(_truth)]
    part3_eol = frame[frame["part"].eq(3) & frame["is_eol"].map(_truth)]
    metadata["part2_unnumbered_eol_observations"] = len(part2_eol)
    metadata["part2_unnumbered_eol_eligible_direction_fits"] = sum(
        _truth(value)
        for value in part2_eol[["charge_accepted", "discharge_accepted"]].to_numpy().ravel()
    )
    metadata["part3_guide_mapped_eol_observations"] = len(part3_eol)
    metadata["part3_guide_mapped_eol_eligible_direction_fits"] = sum(
        _truth(value)
        for value in part3_eol[["charge_accepted", "discharge_accepted"]].to_numpy().ravel()
    )
    numbered = frame[frame["rpt_index"].notna()]
    metadata["duplicate_numbered_rpt_keys"] = int(
        numbered.duplicated(["part", "physical_cell_id", "rpt_index"]).sum()
    )
    metadata["historical_fit_rows_used"] = int(
        frame[["charge_status", "discharge_status"]].isin(["accepted_historical"]).sum().sum()
    )
    metadata["jump_thresholds_pp"] = list(JUMP_THRESHOLDS_PP)
    metadata["production_artifacts_modified"] = False
    (OUT / "run_manifest.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    _write_report(frame, rates, h1, h2, comparison, temporal, fallback, metadata)


if __name__ == "__main__":
    run()
