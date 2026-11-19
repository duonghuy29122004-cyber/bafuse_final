from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bafuse.data.pydma_stage2 import (
    CurvePreflightError,
    _fit_record,
    _independent_degradation_labels,
    _load_references,
    _read_verified_curve,
    _resolve_source,
    _source_relative,
)

_AUDIT_DIR = Path("reports/bafuse_negative_root_cause_v1")
_MAPPING_PATH = Path("reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv")
_STAGE1_DIR = Path("reports/bafuse_stage_validation_v1")
_STAGE2_DIR = Path("reports/bafuse_stage_validation_v2/stage2_pydma_pilot_v1")
_HISTORICAL_AUDIT = Path("reports/final_mapping_label_audit/pydma_fit_audit.csv")
_RESIDUAL_AUDIT = _AUDIT_DIR / "curve_residual_audit.csv"
_SEEDS = (211, 212, 213)
_RMSE_GATE_MV = 15.0
_MIN_ACCEPTED_RUNS = 3
_STRICT_MAPPING = {"verified", "verified_from_ordered_raw_blocks"}
_PILOT_KEYS = (
    ("1", "G1-Cell9", "8"),
    ("1", "G3-Cell11", "3"),
)
_REVIEW_KEYS = (
    ("1", "G1-Cell9", "2"),
    *_PILOT_KEYS,
)
_METRICS = ("lli_percent", "lam_anode_percent", "lam_cathode_percent")


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _read_table(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required correction input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _json_value(value: Any) -> Any:
    if isinstance(value, str):
        if not value:
            return {}
        return json.loads(value)
    return value


def _bounds_valid(record: dict[str, Any]) -> bool:
    try:
        bounds = _json_value(record.get("parameter_bounds_json", ""))
        parameters = _json_value(record.get("fitted_parameters_json", ""))
        if not isinstance(bounds, list) or not isinstance(parameters, dict):
            return False
        checks = {
            str(item["parameter"]): (
                float(item["lower_bound"]),
                float(item["upper_bound"]),
            )
            for item in bounds
        }
        names = ("alpha_an", "beta_an", "alpha_ca", "beta_ca")
        return all(
            name in checks
            and name in parameters
            and parameters[name] is not None
            and checks[name][0] - 1e-10
            <= float(parameters[name])
            <= checks[name][1] + 1e-10
            for name in names
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _gate_passed(record: dict[str, Any]) -> bool:
    try:
        return (
            _truth(record.get("optimizer_accepted", False))
            and float(record["rmse_mV"]) < _RMSE_GATE_MV
            and int(float(record["accepted_runs"])) >= _MIN_ACCEPTED_RUNS
        )
    except (KeyError, TypeError, ValueError):
        return False


def select_best_fit(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        candidate
        for candidate in candidates
        if candidate.get("candidate_valid")
        and candidate.get("labels")
        and np.isfinite(float(candidate.get("rmse_mV", np.nan)))
    ]
    return min(eligible, key=lambda item: float(item["rmse_mV"])) if eligible else None


def _sign_instability(candidates: list[dict[str, Any]]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    eligible = [
        item for item in candidates if item.get("candidate_valid") and item.get("labels")
    ]
    for metric in _METRICS:
        signs = {
            float(item["labels"][metric]) < 0
            for item in eligible
            if metric in item["labels"]
        }
        result[metric] = len(signs) > 1
    return result


def _pydma_degradation_labels(
    reference_params: dict[str, Any],
    current_params: dict[str, Any],
    reference_capacity: float,
    current_capacity: float,
    anode: Any,
    cathode: Any,
) -> dict[str, float]:
    from pydma import DMAAnalyzer, DMAConfig
    from pydma.utils.results import FittedParams

    def fitted_params(values: dict[str, Any]) -> Any:
        return FittedParams(
            alpha_an=float(values["alpha_an"]),
            beta_an=float(values["beta_an"]),
            alpha_ca=float(values["alpha_ca"]),
            beta_ca=float(values["beta_ca"]),
            gamma_blend2_an=values.get("gamma_blend2_an"),
            gamma_blend2_ca=values.get("gamma_blend2_ca"),
            inhom_an=values.get("inhom_an"),
            inhom_ca=values.get("inhom_ca"),
            r_offset_ohm=values.get("r_offset_ohm"),
        )

    result = DMAAnalyzer(
        config=DMAConfig(direction="discharge"),
        anode=anode,
        cathode=cathode,
    ).compare_with_reference(
        reference_params=fitted_params(reference_params),
        current_params=fitted_params(current_params),
        reference_capacity=reference_capacity,
        current_capacity=current_capacity,
    )
    return {
        "lli_percent": float(result.lli * 100),
        "lam_anode_percent": float(result.lam_anode * 100),
        "lam_cathode_percent": float(result.lam_cathode * 100),
    }


def _output_status(
    candidates: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    *,
    existing_only: bool = False,
) -> str:
    if selected is None:
        return "NEGATIVE_BEST_FIT_REVIEW"
    if any(_sign_instability(candidates).values()):
        return "OPTIMIZER_UNSTABLE"
    if any(float(value) < 0 for value in selected["labels"].values()):
        return "NEGATIVE_BEST_FIT_REVIEW"
    return "RESOLVED_EXISTING_FIT" if existing_only else "RESOLVED_TARGETED_REFIT"


def _key(part: str, cell: str, rpt: str, direction: str = "discharge") -> tuple[str, ...]:
    return part, cell, rpt, direction


def _source_matches(record: dict[str, Any], source_file: str) -> bool:
    saved = str(record.get("source_file", "")).replace("\\", "/").casefold()
    expected = source_file.replace("\\", "/").casefold()
    return saved.endswith(expected) or expected.endswith(saved)


def _source_rows_match(
    record: dict[str, Any], curve_audit: dict[str, Any]
) -> bool:
    try:
        return (
            int(record["source_row_start"]) == int(curve_audit["source_row_start"])
            and int(record["source_row_end"]) == int(curve_audit["source_row_end"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _curve_residual_summary(result: Any | None) -> dict[str, Any]:
    if result is None:
        return {}
    measured = getattr(result, "ocv_measured", None)
    reconstructed = getattr(result, "ocv_reconstructed", None)
    soc = getattr(result, "soc_measured", None)
    if measured is None or reconstructed is None:
        return {}
    measured_array = np.asarray(measured, dtype=np.float64).reshape(-1)
    reconstructed_array = np.asarray(reconstructed, dtype=np.float64).reshape(-1)
    if measured_array.shape != reconstructed_array.shape:
        return {"status": "shape_mismatch"}
    finite = np.isfinite(measured_array) & np.isfinite(reconstructed_array)
    residual_mv = (measured_array[finite] - reconstructed_array[finite]) * 1000.0
    if residual_mv.size == 0:
        return {"status": "no_finite_residuals"}
    summary: dict[str, Any] = {
        "status": "computed_from_saved_fit_arrays",
        "sample_count": int(residual_mv.size),
        "rmse_mV": float(np.sqrt(np.mean(np.square(residual_mv)))),
        "max_abs_residual_mV": float(np.max(np.abs(residual_mv))),
        "p95_abs_residual_mV": float(np.percentile(np.abs(residual_mv), 95)),
    }
    if soc is not None:
        soc_array = np.asarray(soc, dtype=np.float64).reshape(-1)
        if soc_array.shape == measured_array.shape:
            soc_array = soc_array[finite]
            edges = np.quantile(soc_array, [1 / 3, 2 / 3])
            bands = (
                ("low_soc", soc_array <= edges[0]),
                ("mid_soc", (soc_array > edges[0]) & (soc_array <= edges[1])),
                ("high_soc", soc_array > edges[1]),
            )
            for name, mask in bands:
                if mask.any():
                    summary[f"{name}_rmse_mV"] = float(
                        np.sqrt(np.mean(np.square(residual_mv[mask])))
                    )
    return summary


def _historical_context(
    historical: pd.DataFrame,
    mapping: pd.DataFrame,
    part: str,
    cell: str,
    rpt: str,
) -> tuple[str, str]:
    rpt_number = int(rpt)
    accepted = historical.loc[
        historical.result_table.eq("part1_part2_rmse15")
        & historical.dataset_part.eq(part)
        & historical.physical_cell_id.eq(cell)
        & historical.direction.eq("charge")
        & historical.diagnostic_label_accepted.str.casefold().isin(
            {"true", "1", "yes"}
        )
        & historical.checkup.eq(f"RPT{rpt_number}")
    ]
    charge_columns = [
        "fit_rmse_mV",
        "lli_percent",
        "lam_anode_percent",
        "lam_cathode_percent",
        "source_file",
    ]
    charge_rows = accepted[charge_columns].drop_duplicates().to_dict("records")
    neighbors = []
    for neighbor_rpt in (rpt_number - 1, rpt_number + 1):
        if neighbor_rpt < 1:
            continue
        row = historical.loc[
            historical.result_table.eq("part1_part2_rmse15")
            & historical.dataset_part.eq(part)
            & historical.physical_cell_id.eq(cell)
            & historical.direction.eq("discharge")
            & historical.diagnostic_label_accepted.str.casefold().isin(
                {"true", "1", "yes"}
            )
            & historical.checkup.eq(f"RPT{neighbor_rpt}")
        ]
        mapping_rows = mapping.loc[
            mapping.part.eq(part)
            & mapping.physical_cell_id.eq(cell)
            & mapping.rpt_index.eq(str(neighbor_rpt))
        ]
        mapping_status = (
            str(mapping_rows.iloc[0]["mapping_status_v2"])
            if len(mapping_rows) == 1
            else "no_unique_mapping_row"
        )
        columns = [
            "fit_rmse_mV",
            "lli_percent",
            "lam_anode_percent",
            "lam_cathode_percent",
            "source_file",
        ]
        neighbor_rows = row[columns].drop_duplicates().to_dict("records")
        neighbors.append(
            {
                "rpt_index": neighbor_rpt,
                "mapping_status": mapping_status,
                "accepted_historical_outputs": neighbor_rows,
            }
        )
    return json.dumps(charge_rows, sort_keys=True), json.dumps(neighbors, sort_keys=True)


def _find_mapping(
    mapping: pd.DataFrame, part: str, cell: str, rpt: str
) -> dict[str, Any]:
    rows = mapping.loc[
        mapping.part.eq(part)
        & mapping.physical_cell_id.eq(cell)
        & mapping.rpt_index.eq(rpt)
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one mapping row for Part {part} {cell} RPT{rpt}.")
    return rows.iloc[0].to_dict()


def _find_bol(bol: pd.DataFrame, part: str, cell: str) -> dict[str, Any]:
    rows = bol.loc[
        bol.dataset_part.eq(part)
        & bol.physical_cell_id.eq(cell)
        & bol.observation_type.eq("BOL_REFERENCE")
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one BoL audit row for Part {part} {cell}.")
    return rows.iloc[0].to_dict()


def _read_bol_reference_overrides(
    root: Path, manifest_path: Path | None
) -> dict[tuple[str, str], dict[str, str]]:
    if manifest_path is None:
        return {}
    path = manifest_path if manifest_path.is_absolute() else root / manifest_path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"BoL reference manifest is missing: {path}")
    manifest = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {
        "dataset_part",
        "physical_cell_id",
        "original_selected_source",
        "selected_step8_reference",
        "selected_step10_reference",
        "selection_status",
    }
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            "BoL reference manifest is missing required columns: "
            + ", ".join(sorted(missing))
        )
    overrides: dict[tuple[str, str], dict[str, str]] = {}
    for row in manifest.to_dict("records"):
        if row["selection_status"] not in {"selected_provisional", "selected_verified"}:
            continue
        key = (str(row["dataset_part"]), str(row["physical_cell_id"]))
        if key in overrides:
            raise ValueError(f"Duplicate selected BoL reference for {key}.")
        for field in (
            "original_selected_source",
            "selected_step8_reference",
            "selected_step10_reference",
        ):
            if not str(row[field]).strip():
                raise ValueError(f"Selected BoL reference for {key} lacks {field}.")
        source8 = str(row["selected_step8_reference"]).replace("\\", "/").split("#", 1)[0]
        source10 = str(row["selected_step10_reference"]).replace("\\", "/").split("#", 1)[0]
        original_source = str(row["original_selected_source"]).replace("\\", "/")
        if source8 != original_source or source10 != original_source:
            raise ValueError(
                f"Selected BoL Step 8/10 sources do not match the Guide-selected source for {key}."
            )
        overrides[key] = row
    return overrides


def _pilot_preflight(
    root: Path,
    mapping: pd.DataFrame,
    bol_audit: pd.DataFrame,
    requested_keys: tuple[tuple[str, str, str], ...],
    allowed_partitions: set[str],
    bol_reference_overrides: dict[tuple[str, str], dict[str, str]] | None = None,
) -> tuple[
    dict[tuple[str, str, str], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, str],
]:
    targets: dict[tuple[str, str, str], dict[str, Any]] = {}
    baselines: dict[str, dict[str, Any]] = {}
    baseline_issues: dict[str, str] = {}
    reference_overrides = bol_reference_overrides or {}
    for part, cell, rpt in sorted(set(requested_keys)):
        row = _find_mapping(mapping, part, cell, rpt)
        if (
            row["mapping_status_v2"] not in _STRICT_MAPPING
            or row["step8_status_exact"] != "verified"
            or row["step10_status_exact"] != "verified"
        ):
            raise ValueError(
                f"Review target lacks verified Step 8/10 mapping: {cell} RPT{rpt}."
            )
        partition = str(row["partition_v2"]).casefold()
        if partition not in allowed_partitions:
            raise ValueError(
                f"Review target is outside the allowed locked partitions: "
                f"{cell} RPT{rpt} is {row['partition_v2']}."
            )
        source = _resolve_source(root, part, str(row["source_file"]))
        if cell not in baselines and cell not in baseline_issues:
            bol_row = _find_bol(bol_audit, part, cell)
            if (
                not _truth(bol_row["bol_verified"])
                or bol_row["bol_policy_status"]
                != "part1_provisional_earliest_complete_bol_selected"
            ):
                raise ValueError(f"BoL policy is not verified for {cell}.")
            bol_source = _resolve_source(
                root, part, str(bol_row["primary_bol_reference_file"])
            )
            override = reference_overrides.get((part, cell))
            if override is not None:
                bol_source_relative = _source_relative(root, bol_source)
                if bol_source_relative != str(override["original_selected_source"]).replace(
                    "\\", "/"
                ):
                    raise ValueError(
                        f"BoL override source does not match the existing Guide source for {cell}."
                    )
                capacity, voltage, step8_bol_audit = _read_verified_curve(
                    root,
                    bol_source,
                    8,
                    "discharge",
                    str(override["selected_step8_reference"]),
                )
                _, _, step10_bol_audit = _read_verified_curve(
                    root,
                    bol_source,
                    10,
                    "charge",
                    str(override["selected_step10_reference"]),
                )
            else:
                try:
                    capacity, voltage, step8_bol_audit = _read_verified_curve(
                        root, bol_source, 8, "discharge"
                    )
                except CurvePreflightError as error:
                    baseline_issues[cell] = (
                        f"BoL Step 8 discharge is not uniquely verified: {error}"
                    )
                    continue
                try:
                    _, _, step10_bol_audit = _read_verified_curve(
                        root, bol_source, 10, "charge"
                    )
                except CurvePreflightError as error:
                    step10_bol_audit = {
                        "status": "not_used_for_discharge_fit",
                        "preflight_error": str(error),
                    }
            baselines[cell] = {
                "audit": bol_row,
                "source": bol_source,
                "capacity": capacity,
                "voltage": voltage,
                "step8_audit": step8_bol_audit,
                "step10_audit": step10_bol_audit,
            }
        if cell in baseline_issues:
            continue
        step8, voltage8, audit8 = _read_verified_curve(
            root, source, 8, "discharge", str(row["step8_segment_ref_exact"])
        )
        _, _, audit10 = _read_verified_curve(
            root, source, 10, "charge", str(row["step10_segment_ref_exact"])
        )
        targets[(part, cell, rpt)] = {
            "mapping": row,
            "source": source,
            "step8": step8,
            "voltage8": voltage8,
            "step8_audit": audit8,
            "step10_audit": audit10,
        }
    return targets, baselines, baseline_issues


def _saved_fit_candidate(
    record: dict[str, Any],
    baseline_record: dict[str, Any] | None,
    current_capacity: np.ndarray,
    baseline_capacity: np.ndarray,
    expected_source: str,
    expected_curve_audit: dict[str, Any],
    saved_labels: dict[str, dict[str, Any]],
    references: tuple[Any, Any],
) -> dict[str, Any]:
    params = _json_value(record.get("fitted_parameters_json", ""))
    baseline_params = (
        _json_value(baseline_record.get("fitted_parameters_json", ""))
        if baseline_record
        else {}
    )
    labels: dict[str, float] = {}
    formula_parity = False
    formula_error = ""
    try:
        labels = _independent_degradation_labels(
            baseline_params,
            params,
            float(np.ptp(baseline_capacity)),
            float(np.ptp(current_capacity)),
        )
        library_labels = _pydma_degradation_labels(
            baseline_params,
            params,
            float(np.ptp(baseline_capacity)),
            float(np.ptp(current_capacity)),
            references[0],
            references[1],
        )
        prior = saved_labels.get(str(record.get("fit_id", "")))
        library_parity = all(
            np.isclose(
                labels[metric],
                library_labels[metric],
                rtol=1e-10,
                atol=1e-10,
            )
            for metric in _METRICS
        )
        saved_label_parity = bool(
            prior
            and all(
                metric in prior
                and np.isclose(
                    labels[metric],
                    float(prior[metric]),
                    rtol=1e-10,
                    atol=1e-10,
                )
                for metric in _METRICS
            )
        )
        formula_parity = library_parity and saved_label_parity
        if not library_parity:
            formula_error = "Independent labels differ from PyDMA compare_with_reference."
        elif not saved_label_parity:
            formula_error = "Recomputed labels do not match the saved Stage 2 labels."
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        formula_error = f"{type(error).__name__}: {error}"
    valid = bool(
        baseline_record
        and _gate_passed(record)
        and _bounds_valid(record)
        and _gate_passed(baseline_record)
        and _bounds_valid(baseline_record)
        and _source_matches(record, expected_source)
        and _source_rows_match(record, expected_curve_audit)
        and formula_parity
    )
    return {
        "fit_id": str(record.get("fit_id", "")),
        "seed": int(record.get("seed", 0)),
        "source": "saved_existing_fit",
        "rmse_mV": float(record.get("rmse_mV", "nan")),
        "accepted_runs": int(float(record.get("accepted_runs", 0))),
        "total_runs": int(float(record.get("total_runs", 0))),
        "gate_passed": _gate_passed(record),
        "bounds_valid": _bounds_valid(record),
        "baseline_fit_id": (
            str(baseline_record.get("fit_id", "")) if baseline_record else ""
        ),
        "formula_parity": formula_parity,
        "formula_error": formula_error,
        "labels": labels,
        "parameters": params,
        "parameter_bounds_json": record.get("parameter_bounds_json", ""),
        "optimizer_config_json": record.get("optimizer_config_json", ""),
        "candidate_valid": valid,
        "record": record,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, encoding="utf-8-sig")
    temporary.replace(path)


def _format_original_values(rows: pd.DataFrame) -> str:
    records = []
    for _, row in rows.iterrows():
        records.append(
            {
                "record_type": row.get("record_type", ""),
                "source_artifact": row.get("source_artifact", ""),
                "fit_rmse_mV": row.get("fit_rmse_mV", ""),
                "accepted_runs": row.get("accepted_runs", ""),
                "total_runs": row.get("total_runs", ""),
                "lli_percent": row.get("lli_percent", ""),
                "lam_anode_percent": row.get("lam_anode_percent", ""),
                "lam_cathode_percent": row.get("lam_cathode_percent", ""),
                "fit_id": row.get("fit_id", ""),
                "seed": row.get("seed", ""),
            }
        )
    return json.dumps(records, sort_keys=True)


def _classify_worklist(
    audit: pd.DataFrame,
    mapping: pd.DataFrame,
    bol_audit: pd.DataFrame,
    inventory: pd.DataFrame,
    selected_rows: dict[tuple[str, ...], dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, item in audit.iterrows():
        part = str(item["dataset_part"])
        cell = str(item["physical_cell_id"])
        rpt = str(item["rpt_index"])
        direction = str(item["direction"])
        key = _key(part, cell, rpt, direction)
        matched = mapping.loc[
            mapping.part.eq(part)
            & mapping.physical_cell_id.eq(cell)
            & mapping.rpt_index.eq(rpt)
        ]
        map_row = matched.iloc[0].to_dict() if len(matched) == 1 else {}
        bol_rows = bol_audit.loc[
            bol_audit.dataset_part.eq(part) & bol_audit.physical_cell_id.eq(cell)
        ]
        bol_row = bol_rows.iloc[0].to_dict() if len(bol_rows) == 1 else {}
        records = inventory.loc[
            inventory.dataset_part.eq(part)
            & inventory.physical_cell_id.eq(cell)
            & inventory.rpt_index.eq(rpt)
            & inventory.direction.eq(direction)
        ]
        selected = selected_rows.get(key, {})
        if map_row and map_row.get("mapping_status_v2") not in _STRICT_MAPPING:
            status = "MAPPING_UNRESOLVED"
            reason = "Mapping status is not in the strict verified set."
        elif map_row and (
            map_row.get("step8_status_exact") != "verified"
            or map_row.get("step10_status_exact") != "verified"
        ):
            status = "MAPPING_UNRESOLVED"
            reason = "Step 8/10 source boundary is missing or not verified."
        elif not bol_row or not _truth(bol_row.get("bol_verified")):
            status = "BOL_UNRESOLVED"
            reason = "No verified BoL reference is available for this cell."
        elif selected:
            status = str(selected["status"])
            reason = str(selected["selection_reason"])
        else:
            status = "REFIT_NOT_ATTEMPTED"
            reason = "Eligible negative key deferred beyond the bounded pilot."
        rows.append(
            {
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "direction": direction,
                "negative_metrics": item["negative_metrics"],
                "historical_negative_rows": item["historical_negative_rows"],
                "historical_accepted_negative_rows": item[
                    "historical_accepted_negative_rows"
                ],
                "mapping_status": map_row.get("mapping_status_v2", item["mapping_status"]),
                "partition": map_row.get("partition_v2", item["evaluation_role"]),
                "source_file": map_row.get("source_file", ""),
                "step8_segment_ref": map_row.get("step8_segment_ref_exact", ""),
                "step10_segment_ref": map_row.get("step10_segment_ref_exact", ""),
                "bol_source_file": bol_row.get("primary_bol_reference_file", ""),
                "bol_policy_status": bol_row.get("bol_policy_status", item["bol_policy_status"]),
                "original_degradation_values_json": _format_original_values(records),
                "selected_fit_id": selected.get("selected_fit_id", ""),
                "selected_rmse_mV": selected.get("selected_rmse_mV", ""),
                "status": status,
                "worklist_reason": reason,
            }
        )
    return rows


def _select_run_targets(
    scope: str,
    worklist: list[dict[str, Any]],
    prior_worklist: list[dict[str, Any]] | None = None,
    explicit_target_keys: tuple[tuple[str, str, str], ...] | None = None,
) -> tuple[tuple[tuple[str, str, str], ...], tuple[tuple[str, str, str], ...]]:
    if explicit_target_keys is not None:
        if scope != "pilot":
            raise ValueError("Explicit target keys are restricted to pilot scope.")
        normalized = tuple(sorted(set(explicit_target_keys)))
        if len(normalized) != len(explicit_target_keys):
            raise ValueError("Explicit target keys contain duplicates.")
        worklist_by_key = {
            _key(
                str(row["dataset_part"]),
                str(row["physical_cell_id"]),
                str(row["rpt_index"]),
            ): row
            for row in worklist
        }
        for key in normalized:
            row = worklist_by_key.get(_key(*key))
            if row is None:
                raise ValueError(f"Explicit target is not in the negative worklist: {key}.")
            if key[0] != "1" or str(row["partition"]).casefold() != "train":
                raise ValueError(
                    "Explicit pilot targets must remain in the locked Part 1 train split."
                )
        return normalized, normalized
    if scope == "pilot":
        return _REVIEW_KEYS, _PILOT_KEYS
    if scope != "deferred":
        raise ValueError(f"Unknown negative-correction scope: {scope!r}")
    if prior_worklist is None:
        raise ValueError("Deferred refits require the v1 pilot worklist.")

    current_by_key = {
        _key(
            str(row["dataset_part"]),
            str(row["physical_cell_id"]),
            str(row["rpt_index"]),
        ): row["status"]
        for row in worklist
    }
    prior_deferred = [
        row for row in prior_worklist if row["status"] == "REFIT_NOT_ATTEMPTED"
    ]
    current_deferred = {
        key for key, status in current_by_key.items() if status == "REFIT_NOT_ATTEMPTED"
    }
    deferred = []
    for row in prior_deferred:
        key = _key(
            str(row["dataset_part"]),
            str(row["physical_cell_id"]),
            str(row["rpt_index"]),
        )
        if current_by_key.get(key) != "REFIT_NOT_ATTEMPTED":
            raise ValueError(f"Previously deferred target is no longer eligible: {key}.")
        deferred.append(row)
    prior_keys = {
        _key(
            str(row["dataset_part"]),
            str(row["physical_cell_id"]),
            str(row["rpt_index"]),
        )
        for row in prior_deferred
    }
    unexpected = current_deferred - prior_keys - {_key(*item) for item in _REVIEW_KEYS}
    if unexpected:
        raise ValueError(
            f"New eligible negative keys require a separate review: {sorted(unexpected)}."
        )
    if not deferred:
        raise ValueError("No eligible deferred negative keys remain.")
    if any(
        row["dataset_part"] != "1"
        or row["partition"] not in {"train", "validation"}
        for row in deferred
    ):
        raise ValueError(
            "Deferred refits are restricted to Part 1 train/validation keys."
        )
    targets = tuple(
        sorted(
            (
                row["dataset_part"],
                row["physical_cell_id"],
                row["rpt_index"],
            )
            for row in deferred
        )
    )
    if len(set(targets)) != len(targets):
        raise ValueError("Deferred worklist contains duplicate target keys.")
    return targets, targets


def _summary_markdown(
    *,
    audit: pd.DataFrame,
    worklist: list[dict[str, Any]],
    existing_fits: list[dict[str, Any]],
    attempts: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    device: str,
    input_hashes: dict[str, str],
    scope: str,
    run_id: str,
) -> str:
    status_counts = pd.Series([row["status"] for row in worklist]).value_counts()
    unresolved = [row for row in worklist if row["status"] not in {"RESOLVED_EXISTING_FIT", "RESOLVED_TARGETED_REFIT", "REUSED_VALID_FIT"}]
    still_negative = [
        row
        for row in selected
        if row.get("selected_labels_json")
        and any(
            float(value) < 0
            for value in json.loads(row["selected_labels_json"]).values()
        )
    ]
    target_attempts = [row for row in attempts if row["fit_role"] == "target_rpt"]
    baseline_attempts = [row for row in attempts if row["fit_role"] == "bol_reference"]
    deferred = [row for row in worklist if row["status"] == "REFIT_NOT_ATTEMPTED"]
    bol_step8_unresolved = [
        row for row in worklist if row["status"] == "BOL_STEP8_UNRESOLVED"
    ]
    deferred_cells = {row["physical_cell_id"] for row in deferred}
    pilot_cells = {cell for _, cell, _ in _PILOT_KEYS}
    new_baseline_cells = deferred_cells - pilot_cells
    target_fit_seconds = [
        float(row["runtime_seconds"])
        for row in target_attempts
        if str(row.get("runtime_seconds", "")).strip()
    ]
    baseline_fit_seconds = [
        float(row["runtime_seconds"])
        for row in baseline_attempts
        if str(row.get("runtime_seconds", "")).strip()
    ]
    estimated_remaining_seconds = (
        len(deferred) * len(_SEEDS) * float(np.mean(target_fit_seconds))
        + len(new_baseline_cells)
        * len(_SEEDS)
        * float(np.mean(baseline_fit_seconds))
        if deferred and target_fit_seconds and baseline_fit_seconds
        else 0.0
    )
    deferred_run = scope == "deferred"
    title = "Deferred Negative PyDMA Refits" if deferred_run else "Negative PyDMA Correction Pilot"
    scope_description = (
        "Approved Part 1 deferred-key expansion across locked train/validation cells. "
        "The original pilot outcomes are carried forward for worklist status only; "
        "pilot fits were not rerun. No historical output, label, model, scaler, split, "
        "or source file was modified."
        if deferred_run
        else "Existing-fit comparison plus a bounded Part 1 training-cell pilot. "
        "No historical output, label, model, scaler, split, or source file was modified. "
        "Part 2 cases with unresolved BoL policy and all Part 3 external cases were not refit."
    )
    lines = [
        f"# {title}",
        "",
        f"**Scope:** {scope_description}",
        "",
        "## Summary",
        "",
        f"- Unique negative observation/direction keys inventoried: **{len(audit)}**.",
        f"- Existing parameterized target fits compared: **{sum(row.get('fit_role') == 'existing_target_fit' for row in existing_fits)}**.",
        f"- New target refit attempts: **{len(target_attempts)}**; new BoL reference attempts: **{len(baseline_attempts)}**.",
        (
            f"- Deferred target observations processed: **{len(selected)}**; "
            "train/validation assignments were retained and not used for model fitting."
            if deferred_run
            else f"- Pilot target observations: **{len(_PILOT_KEYS)}** (G1-Cell9 RPT8 and G3-Cell11 RPT3, both locked `train`)."
        ),
        f"- Existing fit selected by RMSE policy: **{sum(row.get('selection_source') == 'saved_existing_fit' for row in selected)}**; any seed/sign instability remains explicitly flagged.",
        f"- Selected observations still carrying at least one negative estimate: **{len(still_negative)}**.",
        f"- Unresolved or deferred negative keys: **{len(unresolved)}**.",
        "",
        "### Worklist status counts",
        "",
        "| Status | Cases |",
        "|---|---:|",
    ]
    lines.extend(
        f"| `{status}` | {int(count)} |" for status, count in status_counts.items()
    )
    if bol_step8_unresolved:
        lines.extend(
            [
                "",
                "### BoL Step 8 cases withheld",
                "",
                "These targets were not fit because their BoL Step 8 discharge baseline was ambiguous; the worklist records the cell, RPT, and preflight error.",
            ]
        )
        lines.extend(
            f"- {row['physical_cell_id']} RPT{row['rpt_index']}: {row['worklist_reason']}"
            for row in bol_step8_unresolved
        )
    lines.extend(
        [
            "",
            "### Deferred worklist status",
            "",
            f"- Eligible keys still deferred after this run: **{len(deferred)}** across {len(deferred_cells)} cells.",
            f"- Estimated additional optimizer calls at the 3-seed policy: **{len(deferred) * len(_SEEDS)} target fits** plus **{len(new_baseline_cells) * len(_SEEDS)} new BoL fits** for cells without pilot baselines.",
            (
                f"- Based on this run's observed mean runtimes ({np.mean(target_fit_seconds):.1f} s/target fit; "
                f"{np.mean(baseline_fit_seconds):.1f} s/BoL fit), remaining optimizer time is approximately "
                f"**{estimated_remaining_seconds / 60:.0f} minutes**, excluding preflight and file-read time."
                if target_fit_seconds and baseline_fit_seconds
                else "- No runtime estimate is available for remaining work."
            ),
            "",
        ]
    )
    lines.extend(
        [
            "",
            "## Selection and fit policy",
            "",
            f"- Deterministic seed set: `{list(_SEEDS)}`; CUDA device: `{device}`.",
            f"- Existing acceptance gate: RMSE < {_RMSE_GATE_MV:g} mV and at least {_MIN_ACCEPTED_RUNS} accepted runs, with all four fitted parameters inside saved bounds.",
            "- Candidate ranking uses target-fit RMSE only after the fit, baseline, source-boundary, parameter-bound, and formula-parity checks pass. Degradation sign is never a ranking criterion.",
            "- No numeric cross-seed stability tolerance was introduced. Sign changes are reported categorically; numeric ranges remain in the selected-results output.",
            "- Step 8 discharge was fitted. The corresponding Step 10 charge segment was independently preflighted and retained as provenance; no charge refit was launched.",
            "- Historical accepted charge-direction outputs and adjacent RPT outputs are included as context only; they were not used to select a discharge fit. Reconstructed/measured residual summaries and arrays are retained per new attempt.",
            "- BoL uses the repository's existing Part 1 provisional earliest-complete reference policy. It is not promoted to an official dataset rule.",
            "",
            "## Selected observation outcomes",
            "",
            "| Cell / RPT | Selected fit | Historical best RMSE | Selected RMSE | Delta (mV) | LLI (%) | Anode LAM (%) | Cathode LAM (%) | Status |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in selected:
        values = json.loads(row["selected_labels_json"]) if row.get("selected_labels_json") else {}
        lines.append(
            "| {cell} RPT{rpt} | `{fit}` | {original_rmse} | {rmse} | {delta} | {lli} | {anode} | {cathode} | `{status}` |".format(
                cell=row["physical_cell_id"],
                rpt=row["rpt_index"],
                fit=row.get("selected_fit_id", ""),
                original_rmse=row.get("best_historical_accepted_rmse_mV", ""),
                rmse=row.get("selected_rmse_mV", ""),
                delta=row.get("selected_vs_historical_rmse_delta_mV", ""),
                lli=values.get("lli_percent", ""),
                anode=values.get("lam_anode_percent", ""),
                cathode=values.get("lam_cathode_percent", ""),
                status=row["status"],
            )
        )
    lines.extend(
        [
            "",
            "A selected positive diagnostic is not experimentally validated ground truth. Historical negative values remain in their original artifacts; this output records a non-destructive candidate selection only.",
            "",
            "## Reproducibility",
            "",
            "Rerun from the repository root with:",
            "",
            "```powershell",
            f"python -m bafuse.data.negative_correction --root . --run-id {run_id} --scope {scope}",
            "```",
            "",
            "The script refuses to overwrite an existing versioned output directory. New fit attempts and their measured/reconstructed arrays are saved in this directory. Input SHA-256 values captured for this run:",
            "",
        ]
    )
    lines.extend(f"- `{name}`: `{digest}`" for name, digest in input_hashes.items())
    lines.extend(
        [
            "",
            "## Output files",
            "",
            "- `negative_worklist.csv`: all 52 unique historical negative keys and their mapping/BoL disposition.",
            "- `existing_fit_comparison.csv`: historical values plus all saved parameterized target fits examined.",
            "- `targeted_refit_attempts.csv`: every new target and baseline fit attempt, including errors and quality fields.",
            "- `selected_degradation_results.csv`: preferred candidate per reviewed observation with original values retained.",
            "- `unresolved_negative_cases.csv`: blocked, deferred, negative-best, or seed-unstable cases.",
            "- `targeted_refit_curve_arrays.npz`: measured and reconstructed curves keyed by fit ID.",
            "",
            "Selected fits are diagnostic candidates, not independently measured ground truth. Historical labels and all existing model artifacts remain unchanged.",
            "",
        ]
    )
    return "\n".join(lines)


def run_negative_correction(
    root: Path | None = None,
    run_id: str = "v1",
    scope: str = "pilot",
    target_keys: tuple[tuple[str, str, str], ...] | None = None,
    bol_reference_manifest: Path | None = None,
    output_dir: Path | None = None,
) -> Path:
    repo = (root or Path(__file__).resolve().parents[3]).resolve()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise ValueError(f"Invalid correction run ID: {run_id!r}")
    output = (
        repo / "reports" / f"bafuse_negative_correction_{run_id}"
        if output_dir is None
        else output_dir if output_dir.is_absolute() else repo / output_dir
    ).resolve()
    try:
        output.relative_to(repo)
    except ValueError as error:
        raise ValueError("Correction output must remain inside the repository.") from error
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite correction artifacts: {output}")

    if scope not in {"pilot", "deferred"}:
        raise ValueError(f"Unknown negative-correction scope: {scope!r}")

    audit = _read_table(repo, _AUDIT_DIR / "negative_root_cause_audit.csv")
    inventory = _read_table(repo, _AUDIT_DIR / "negative_inventory.csv")
    mapping = _read_table(repo, _MAPPING_PATH)
    bol_audit = _read_table(repo, _STAGE1_DIR / "stage1_bol_reference_audit.csv")
    old_attempts = _read_table(repo, _STAGE2_DIR / "stage2_pydma_fit_attempts.csv")
    old_labels = _read_table(repo, _STAGE2_DIR / "stage2_pydma_label_comparison.csv")
    prior_output = repo / "reports" / "bafuse_negative_correction_v1"
    prior_worklist: list[dict[str, Any]] | None = None
    prior_selected: list[dict[str, Any]] = []
    prior_attempts: list[dict[str, Any]] = []
    if scope == "deferred":
        prior_worklist = _read_table(
            repo, prior_output.relative_to(repo) / "negative_worklist.csv"
        ).to_dict("records")
        prior_selected = _read_table(
            repo, prior_output.relative_to(repo) / "selected_degradation_results.csv"
        ).to_dict("records")
        prior_attempts = _read_table(
            repo, prior_output.relative_to(repo) / "targeted_refit_attempts.csv"
        ).to_dict("records")
    initial_worklist = _classify_worklist(
        audit, mapping, bol_audit, inventory, {}
    )
    review_keys, target_fit_keys = _select_run_targets(
        scope, initial_worklist, prior_worklist, target_keys
    )
    allowed_partitions = {"train"} if scope == "pilot" else {"train", "validation"}
    requested_keys = review_keys
    bol_reference_overrides = _read_bol_reference_overrides(
        repo, bol_reference_manifest
    )
    target_curves, baseline_curves, baseline_issues = _pilot_preflight(
        repo,
        mapping,
        bol_audit,
        review_keys,
        allowed_partitions,
        bol_reference_overrides,
    )
    blocked_target_keys = tuple(
        key for key in requested_keys if key[1] in baseline_issues
    )
    review_keys = tuple(key for key in requested_keys if key[1] in baseline_curves)
    target_fit_keys = tuple(key for key in target_fit_keys if key[1] in baseline_curves)
    historical_audit = _read_table(repo, _HISTORICAL_AUDIT)
    residual_audit = _read_table(repo, _RESIDUAL_AUDIT)
    old_residual_by_id = {
        str(row["fit_id"]): row.to_dict()
        for _, row in residual_audit.iterrows()
        if str(row.get("fit_id", ""))
    }

    from bafuse.labeling.pydma_runner import _fit_one
    import torch

    anode_reference, cathode_reference, reference_metadata = _load_references(
        repo, "discharge"
    )
    references = (anode_reference, cathode_reference)
    use_cuda = bool(torch.cuda.is_available())
    device = torch.cuda.get_device_name(0) if use_cuda else "CPU"

    old_rows = [row.to_dict() for _, row in old_attempts.iterrows()]
    saved_by_id = {str(row["fit_id"]): row for row in old_rows}
    old_label_by_id = {
        str(row["rpt_fit_id"]): row
        for _, row in old_labels.iterrows()
        if str(row.get("rpt_fit_id", ""))
    }

    source_keys = {
        "negative_root_cause_audit.csv": _AUDIT_DIR / "negative_root_cause_audit.csv",
        "negative_inventory.csv": _AUDIT_DIR / "negative_inventory.csv",
        "rpt_mapping_validation.csv": _MAPPING_PATH,
        "stage1_bol_reference_audit.csv": _STAGE1_DIR / "stage1_bol_reference_audit.csv",
        "stage2_pydma_fit_attempts.csv": _STAGE2_DIR / "stage2_pydma_fit_attempts.csv",
        "stage2_pydma_label_comparison.csv": _STAGE2_DIR / "stage2_pydma_label_comparison.csv",
        "pydma_fit_audit.csv": _HISTORICAL_AUDIT,
        "curve_residual_audit.csv": _RESIDUAL_AUDIT,
        "negative_correction.py": Path("src/bafuse/data/negative_correction.py"),
    }
    if bol_reference_manifest is not None:
        reference_manifest = (
            bol_reference_manifest
            if bol_reference_manifest.is_absolute()
            else repo / bol_reference_manifest
        ).resolve()
        try:
            reference_manifest_name = reference_manifest.relative_to(repo).as_posix()
        except ValueError as error:
            raise ValueError("BoL reference manifest must remain inside the repository.") from error
        source_keys["bol_reference_manifest.csv"] = Path(reference_manifest_name)
    if scope == "deferred":
        source_keys.update(
            {
                "pilot_selected_degradation_results.csv": prior_output.relative_to(repo)
                / "selected_degradation_results.csv",
                "pilot_negative_worklist.csv": prior_output.relative_to(repo)
                / "negative_worklist.csv",
                "pilot_targeted_refit_attempts.csv": prior_output.relative_to(repo)
                / "targeted_refit_attempts.csv",
            }
        )
    input_hashes = {
        name: hashlib.sha256((repo / path).read_bytes()).hexdigest()
        for name, path in source_keys.items()
    }

    bol_cache: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in baseline_curves:
        for seed in _SEEDS:
            fit_id = f"bol-p1-{cell.lower()}-rpt0-discharge-seed{seed}"
            saved = saved_by_id.get(fit_id)
            if (
                cell == "G1-Cell9"
                and seed in {211, 212}
                and saved
                and _source_matches(
                    saved,
                    _source_relative(repo, baseline_curves[cell]["source"]),
                )
                and _source_rows_match(saved, baseline_curves[cell]["step8_audit"])
            ):
                bol_cache[(cell, seed)] = {
                    "record": saved,
                    "fit_id": fit_id,
                    "source": "saved_existing_fit",
                    "result": None,
                    "accepted": _gate_passed(saved) and _bounds_valid(saved),
                }
    for saved in prior_attempts:
        if saved.get("fit_role") != "bol_reference":
                continue
        cell = str(saved.get("physical_cell_id", ""))
        try:
                seed = int(saved["seed"])
        except (KeyError, TypeError, ValueError):
                continue
        baseline = baseline_curves.get(cell)
        if (
                baseline
                and _source_matches(
                    saved, _source_relative(repo, baseline["source"])
                )
                and _source_rows_match(saved, baseline["step8_audit"])
                and (cell, seed) not in bol_cache
        ):
                bol_cache[(cell, seed)] = {
                    "record": saved,
                    "fit_id": str(saved["fit_id"]),
                    "source": "saved_pilot_fit",
                    "result": None,
                    "accepted": _gate_passed(saved) and _bounds_valid(saved),
                }

    new_attempts: list[dict[str, Any]] = []
    new_results: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}

    def run_fit(
        *,
        fit_id: str,
        fit_role: str,
        cell: str,
        rpt: str,
        seed: int,
        curve: np.ndarray,
        voltage: np.ndarray,
        curve_audit: dict[str, Any],
        source_path: Path,
        observation_type: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        result = None
        error = ""
        try:
            result = _fit_one(
                "discharge",
                curve,
                voltage,
                references[0],
                references[1],
                seed=seed,
                context=f"negative correction {cell} RPT{rpt} {fit_role}",
                use_cuda=use_cuda,
                maxiter=50,
                popsize=30,
                max_tries=10,
            )
            new_results[fit_id] = result
            for field, value in (
                ("measured_capacity", curve),
                ("measured_voltage", voltage),
            ):
                arrays[f"{fit_id}__{field}"] = np.asarray(value)
            for field in (
                "soc_measured",
                "ocv_measured",
                "soc_reconstructed",
                "ocv_reconstructed",
            ):
                value = getattr(result, field, None)
                if value is not None:
                    arrays[f"{fit_id}__{field}"] = np.asarray(value)
            measured_ocv = getattr(result, "ocv_measured", None)
            reconstructed_ocv = getattr(result, "ocv_reconstructed", None)
            if measured_ocv is not None and reconstructed_ocv is not None:
                measured_values = np.asarray(measured_ocv)
                reconstructed_values = np.asarray(reconstructed_ocv)
                if measured_values.shape == reconstructed_values.shape:
                    arrays[f"{fit_id}__residual"] = (
                        measured_values - reconstructed_values
                    )
        except (
            ArithmeticError,
            FloatingPointError,
            IndexError,
            MemoryError,
            RuntimeError,
            TypeError,
            ValueError,
        ) as fit_error:
            error = f"{type(fit_error).__name__}: {fit_error}"
            print(f"PyDMA correction attempt failed {fit_id}: {error}", flush=True)
        row = _fit_record(
            fit_id=fit_id,
            observation_type=observation_type,
            part="1",
            cell=cell,
            rpt="0" if observation_type == "BOL_REFERENCE" else rpt,
            direction="discharge",
            seed=seed,
            curve=curve,
            voltage=voltage,
            curve_audit=curve_audit,
            references=reference_metadata,
            source_file=_source_relative(repo, source_path),
            baseline_reference_id="",
            result=result,
            runtime_seconds=time.perf_counter() - started,
            error=error,
        )
        row["fit_role"] = fit_role
        row["seed"] = seed
        row["candidate_valid"] = bool(
            not error and _gate_passed(row) and _bounds_valid(row)
        )
        row["baseline_fit_id"] = ""
        row["formula_parity"] = ""
        row["formula_error"] = ""
        row["lli_percent"] = ""
        row["lam_anode_percent"] = ""
        row["lam_cathode_percent"] = ""
        row["residual_summary_json"] = json.dumps(
            _curve_residual_summary(result), sort_keys=True
        )
        new_attempts.append(row)
        return row

    output.mkdir(parents=True)
    for cell, baseline in baseline_curves.items():
        for seed in _SEEDS:
            if (cell, seed) in bol_cache:
                continue
            fit_id = f"negcorr-bol-p1-{cell.lower()}-rpt0-discharge-seed{seed}"
            row = run_fit(
                fit_id=fit_id,
                fit_role="bol_reference",
                cell=cell,
                rpt="0",
                seed=seed,
                curve=baseline["capacity"],
                voltage=baseline["voltage"],
                curve_audit=baseline["step8_audit"],
                source_path=baseline["source"],
                observation_type="BOL_REFERENCE",
            )
            bol_cache[(cell, seed)] = {
                "record": row,
                "fit_id": fit_id,
                "source": "targeted_refit",
                "result": new_results.get(fit_id),
                "accepted": bool(row["candidate_valid"]),
            }

    for part, cell, rpt in target_fit_keys:
        target = target_curves[(part, cell, rpt)]
        for seed in _SEEDS:
            existing_id = f"rpt-p{part}-{cell.lower()}-rpt{rpt}-discharge-seed{seed}"
            if existing_id in saved_by_id:
                continue
            fit_id = f"negcorr-rpt-p{part}-{cell.lower()}-rpt{rpt}-discharge-seed{seed}"
            run_fit(
                fit_id=fit_id,
                fit_role="target_rpt",
                cell=cell,
                rpt=rpt,
                seed=seed,
                curve=target["step8"],
                voltage=target["voltage8"],
                curve_audit=target["step8_audit"],
                source_path=target["source"],
                observation_type="RPT",
            )

    candidates_by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    comparison_rows: list[dict[str, Any]] = []
    for part, cell, rpt in review_keys:
        target = target_curves[(part, cell, rpt)]
        key = _key(part, cell, rpt)
        candidates: list[dict[str, Any]] = []
        prior_target_rows = [
            row
            for row in old_rows
            if row["observation_type"] == "RPT"
            and row["dataset_part"] == part
            and row["physical_cell_id"] == cell
            and row["rpt_index"] == rpt
            and row["direction"] == "discharge"
        ]
        if key in {_key(*item) for item in review_keys}:
            for record in prior_target_rows:
                source_rel = _source_relative(repo, target["source"])
                if (
                    not _source_matches(record, source_rel)
                    or not _source_rows_match(record, target["step8_audit"])
                ):
                    continue
                seed = int(record["seed"])
                baseline = bol_cache.get((cell, seed))
                candidate = _saved_fit_candidate(
                    record,
                    baseline["record"] if baseline else None,
                    target["step8"],
                    baseline_curves[cell]["capacity"],
                    source_rel,
                    target["step8_audit"],
                    {
                        fit_id: {
                            metric: float(label[metric])
                            for metric in _METRICS
                        }
                        for fit_id, label in old_label_by_id.items()
                    },
                    references,
                )
                candidates.append(candidate)
                candidate["residual_summary"] = old_residual_by_id.get(
                    candidate["fit_id"], {}
                )
                comparison_rows.append(
                    {
                        "fit_role": "existing_target_fit",
                        "dataset_part": part,
                        "physical_cell_id": cell,
                        "rpt_index": rpt,
                        "fit_id": candidate["fit_id"],
                        "baseline_fit_id": candidate["baseline_fit_id"],
                        "seed": seed,
                        "rmse_mV": candidate["rmse_mV"],
                        "accepted_runs": candidate["accepted_runs"],
                        "total_runs": candidate["total_runs"],
                        "gate_passed": candidate["gate_passed"],
                        "bounds_valid": candidate["bounds_valid"],
                        "formula_parity": candidate["formula_parity"],
                        "candidate_valid": candidate["candidate_valid"],
                        "labels_json": json.dumps(candidate["labels"], sort_keys=True),
                        "parameters_json": json.dumps(candidate["parameters"], sort_keys=True),
                        "parameter_bounds_json": candidate["parameter_bounds_json"],
                        "optimizer_config_json": candidate["optimizer_config_json"],
                        "residual_summary_json": json.dumps(
                            candidate["residual_summary"], sort_keys=True
                        ),
                        "selection_basis": "existing accepted fit; target RMSE ranking only",
                        "error": candidate["formula_error"],
                    }
                )

        for attempt in new_attempts:
            if (
                attempt["fit_role"] != "target_rpt"
                or attempt["dataset_part"] != part
                or attempt["physical_cell_id"] != cell
                or attempt["rpt_index"] != rpt
            ):
                continue
            baseline = bol_cache.get((cell, int(attempt["seed"])))
            labels: dict[str, float] = {}
            formula_parity = False
            formula_error = ""
            if baseline and baseline["accepted"] and attempt["candidate_valid"]:
                try:
                    labels = _independent_degradation_labels(
                        _json_value(baseline["record"]["fitted_parameters_json"]),
                        _json_value(attempt["fitted_parameters_json"]),
                        float(np.ptp(baseline_curves[cell]["capacity"])),
                        float(np.ptp(target["step8"])),
                    )
                    library_labels = _pydma_degradation_labels(
                        _json_value(baseline["record"]["fitted_parameters_json"]),
                        _json_value(attempt["fitted_parameters_json"]),
                        float(np.ptp(baseline_curves[cell]["capacity"])),
                        float(np.ptp(target["step8"])),
                        references[0],
                        references[1],
                    )
                    formula_parity = all(
                        np.isclose(
                            labels[metric],
                            library_labels[metric],
                            rtol=1e-10,
                            atol=1e-10,
                        )
                        for metric in _METRICS
                    )
                    if not formula_parity:
                        formula_error = (
                            "Independent labels differ from PyDMA compare_with_reference."
                        )
                except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
                    formula_error = f"{type(error).__name__}: {error}"
            attempt["baseline_fit_id"] = baseline["fit_id"] if baseline else ""
            attempt["formula_parity"] = formula_parity
            attempt["formula_error"] = formula_error
            attempt["lli_percent"] = labels.get("lli_percent", "")
            attempt["lam_anode_percent"] = labels.get("lam_anode_percent", "")
            attempt["lam_cathode_percent"] = labels.get("lam_cathode_percent", "")
            candidate = {
                "fit_id": attempt["fit_id"],
                "seed": int(attempt["seed"]),
                "source": "targeted_refit",
                "rmse_mV": float(attempt["rmse_mV"])
                if attempt["rmse_mV"] != ""
                else float("nan"),
                "accepted_runs": int(attempt["accepted_runs"]),
                "total_runs": int(attempt["total_runs"]),
                "gate_passed": _gate_passed(attempt),
                "bounds_valid": _bounds_valid(attempt),
                "baseline_fit_id": attempt["baseline_fit_id"],
                "formula_parity": bool(labels),
                "formula_error": formula_error,
                "labels": labels,
                "parameters": _json_value(attempt["fitted_parameters_json"]),
                "parameter_bounds_json": attempt["parameter_bounds_json"],
                "optimizer_config_json": attempt["optimizer_config_json"],
                "candidate_valid": bool(
                    baseline
                    and baseline["accepted"]
                    and _gate_passed(attempt)
                    and _bounds_valid(attempt)
                    and formula_parity
                ),
                "record": attempt,
                "residual_summary": _json_value(attempt["residual_summary_json"]),
            }
            candidates.append(candidate)
            comparison_rows.append(
                {
                    "fit_role": "targeted_target_fit",
                    "dataset_part": part,
                    "physical_cell_id": cell,
                    "rpt_index": rpt,
                    "fit_id": candidate["fit_id"],
                    "baseline_fit_id": candidate["baseline_fit_id"],
                    "seed": candidate["seed"],
                    "rmse_mV": candidate["rmse_mV"],
                    "accepted_runs": candidate["accepted_runs"],
                    "total_runs": candidate["total_runs"],
                    "gate_passed": candidate["gate_passed"],
                    "bounds_valid": candidate["bounds_valid"],
                    "formula_parity": candidate["formula_parity"],
                    "candidate_valid": candidate["candidate_valid"],
                    "labels_json": json.dumps(candidate["labels"], sort_keys=True),
                    "parameters_json": json.dumps(candidate["parameters"], sort_keys=True),
                    "parameter_bounds_json": candidate["parameter_bounds_json"],
                    "optimizer_config_json": candidate["optimizer_config_json"],
                    "residual_summary_json": json.dumps(
                        candidate["residual_summary"], sort_keys=True
                    ),
                    "selection_basis": "target RMSE ranking only; sign is not considered",
                    "error": candidate["formula_error"],
                }
            )
        candidates_by_key[key] = candidates

    for _, item in audit.iterrows():
        part, cell, rpt, direction = (
            str(item["dataset_part"]),
            str(item["physical_cell_id"]),
            str(item["rpt_index"]),
            str(item["direction"]),
        )
        key = _key(part, cell, rpt, direction)
        rows = inventory.loc[
            inventory.dataset_part.eq(part)
            & inventory.physical_cell_id.eq(cell)
            & inventory.rpt_index.eq(rpt)
            & inventory.direction.eq(direction)
        ]
        historical = rows.loc[rows.record_type.eq("historical_output_signature")]
        comparison_rows.append(
            {
                "fit_role": "historical_negative_output",
                "dataset_part": part,
                "physical_cell_id": cell,
                "rpt_index": rpt,
                "fit_id": "",
                "baseline_fit_id": "",
                "seed": "",
                "rmse_mV": "",
                "accepted_runs": "",
                "total_runs": "",
                "gate_passed": "",
                "bounds_valid": "",
                "formula_parity": "",
                "candidate_valid": "",
                "labels_json": _format_original_values(historical),
                "parameters_json": "",
                "parameter_bounds_json": "",
                "optimizer_config_json": "",
                "selection_basis": "Historical value preserved; no saved accepted parameter vector for this key.",
                "error": "",
            }
        )

    selected_rows: list[dict[str, Any]] = []
    selected_by_key: dict[tuple[str, ...], dict[str, Any]] = {}
    for part, cell, rpt in review_keys:
        key = _key(part, cell, rpt)
        candidates = candidates_by_key.get(key, [])
        best = select_best_fit(candidates)
        status = _output_status(
            candidates,
            best,
            existing_only=bool(best and best["source"] == "saved_existing_fit"),
        )
        original = inventory.loc[
            inventory.dataset_part.eq(part)
            & inventory.physical_cell_id.eq(cell)
            & inventory.rpt_index.eq(rpt)
            & inventory.direction.eq("discharge")
            & inventory.label_accepted.str.casefold().isin({"true", "1", "yes"})
        ]
        original_values = _format_original_values(original)
        historical_accepted = original.loc[
            original.record_type.eq("historical_output_signature")
        ]
        historical_rmses = [
            float(value)
            for value in historical_accepted.fit_rmse_mV
            if str(value).strip()
        ]
        best_historical_rmse = min(historical_rmses) if historical_rmses else ""
        original_fit_ids = [
            str(value)
            for value in original.fit_id
            if str(value).strip()
        ]
        valid = [item for item in candidates if item.get("candidate_valid")]
        metric_ranges = {}
        for metric in _METRICS:
            values = [
                float(item["labels"][metric])
                for item in valid
                if metric in item.get("labels", {})
            ]
            metric_ranges[metric] = (
                {"min": min(values), "max": max(values)} if values else {}
            )
        if best:
            reason = (
                "Lowest RMSE among candidates passing mapping/source, baseline, "
                "formula-parity, parameter-bound, and existing fit-quality checks; "
                "degradation sign was not considered."
            )
            selected_id = best["fit_id"]
            selected_rmse = best["rmse_mV"]
            selected_labels = best["labels"]
            selection_source = best["source"]
            selected_parameters = best["parameters"]
            selected_bounds = best["parameter_bounds_json"]
            residual_summary = best.get("residual_summary", {})
        else:
            reason = "No candidate passed all required fit and baseline checks."
            selected_id = ""
            selected_rmse = ""
            selected_labels = {}
            selection_source = ""
            selected_parameters = {}
            selected_bounds = ""
            residual_summary = {}
        charge_context, neighbor_context = _historical_context(
            historical_audit, mapping, part, cell, rpt
        )
        charge_rows = json.loads(charge_context)
        charge_sign_agreement = {
            metric: (
                all((float(row[metric]) < 0) == (float(selected_labels[metric]) < 0) for row in charge_rows)
                if charge_rows and metric in selected_labels
                else ""
            )
            for metric in _METRICS
        }
        try:
            selected_bound_rows = _json_value(selected_bounds)
            near_bound_parameters = [
                row["parameter"]
                for row in selected_bound_rows
                if float(row.get("normalized_distance_to_nearest_bound", 1.0)) <= 0.01
            ]
        except (TypeError, ValueError, KeyError):
            near_bound_parameters = []
        historical_value_rows = json.loads(original_values)
        original_fit_ids.extend(
            str(row["fit_id"])
            for row in historical_value_rows
            if row.get("fit_id") and row["fit_id"] not in original_fit_ids
        )
        delta = (
            float(selected_rmse) - best_historical_rmse
            if selected_rmse != "" and best_historical_rmse != ""
            else ""
        )
        result_row = {
            "dataset_part": part,
            "physical_cell_id": cell,
            "rpt_index": rpt,
            "direction": "discharge",
            "source_file": target_curves[(part, cell, rpt)]["mapping"]["source_file"],
            "step8_segment_ref": target_curves[(part, cell, rpt)]["mapping"][
                "step8_segment_ref_exact"
            ],
            "step10_segment_ref": target_curves[(part, cell, rpt)]["mapping"][
                "step10_segment_ref_exact"
            ],
            "bol_source_file": baseline_curves[cell]["audit"][
                "primary_bol_reference_file"
            ],
            "bol_policy_status": baseline_curves[cell]["audit"]["bol_policy_status"],
            "bol_step8_reference": baseline_curves[cell]["step8_audit"].get(
                "source_reference", ""
            ),
            "bol_step10_reference": baseline_curves[cell]["step10_audit"].get(
                "source_reference", ""
            ),
            "original_degradation_values_json": original_values,
            "original_fit_ids_json": json.dumps(original_fit_ids),
            "best_historical_accepted_rmse_mV": best_historical_rmse,
            "selected_fit_id": selected_id,
            "selected_baseline_fit_id": best["baseline_fit_id"] if best else "",
            "selected_seed": best["seed"] if best else "",
            "selected_rmse_mV": selected_rmse,
            "selected_vs_historical_rmse_delta_mV": delta,
            "selected_labels_json": json.dumps(selected_labels, sort_keys=True),
            "selected_parameters_json": json.dumps(selected_parameters, sort_keys=True),
            "selected_parameter_bounds_json": (
                selected_bounds
                if isinstance(selected_bounds, str)
                else json.dumps(selected_bounds, sort_keys=True)
            ),
            "selected_near_bound_parameters_json": json.dumps(near_bound_parameters),
            "selected_residual_summary_json": json.dumps(
                residual_summary, sort_keys=True
            ),
            "fitted_current_exceeds_reference_json": json.dumps(
                {metric: float(value) < 0 for metric, value in selected_labels.items()},
                sort_keys=True,
            ),
            "same_rpt_historical_charge_outputs_json": charge_context,
            "same_rpt_charge_sign_agreement_json": json.dumps(
                charge_sign_agreement, sort_keys=True
            ),
            "neighboring_rpt_historical_outputs_json": neighbor_context,
            "valid_candidate_fit_ids_json": json.dumps(
                [item["fit_id"] for item in valid]
            ),
            "metric_ranges_json": json.dumps(metric_ranges, sort_keys=True),
            "metric_sign_instability_json": json.dumps(
                _sign_instability(candidates), sort_keys=True
            ),
            "selection_source": selection_source,
            "optimization_rerun": selection_source == "targeted_refit",
            "selection_reason": reason,
            "status": status,
        }
        selected_rows.append(result_row)
        selected_by_key[key] = result_row

    worklist = _classify_worklist(
        audit, mapping, bol_audit, inventory, selected_by_key
    )
    blocked_by_key = {
        _key(part, cell, rpt): baseline_issues[cell]
        for part, cell, rpt in blocked_target_keys
    }
    for row in worklist:
        issue = blocked_by_key.get(
            _key(
                str(row["dataset_part"]),
                str(row["physical_cell_id"]),
                str(row["rpt_index"]),
            )
        )
        if issue:
            row["status"] = "BOL_STEP8_UNRESOLVED"
            row["worklist_reason"] = issue
    if scope == "deferred":
        prior_by_key = {
            _key(
                str(row["dataset_part"]),
                str(row["physical_cell_id"]),
                str(row["rpt_index"]),
            ): row
            for row in prior_selected
        }
        for row in worklist:
            prior = prior_by_key.get(
                _key(
                    str(row["dataset_part"]),
                    str(row["physical_cell_id"]),
                    str(row["rpt_index"]),
                )
            )
            if prior:
                row["status"] = prior["status"]
                row["selected_fit_id"] = prior["selected_fit_id"]
                row["selected_rmse_mV"] = prior["selected_rmse_mV"]
                row["worklist_reason"] = (
                    "Carried forward from the v1 pilot; this target was not rerun."
                )
    unresolved_rows = [
        row
        for row in worklist
        if row["status"]
        not in {"RESOLVED_EXISTING_FIT", "RESOLVED_TARGETED_REFIT", "REUSED_VALID_FIT"}
    ]
    for attempt in new_attempts:
        cell = str(attempt["physical_cell_id"])
        baseline = bol_cache.get((cell, int(attempt["seed"])))
        baseline_curve = baseline_curves.get(cell)
        if attempt["fit_role"] == "target_rpt" and baseline and baseline_curve:
            attempt["baseline_fit_id"] = baseline["fit_id"]
            attempt["baseline_reference_id"] = baseline_curve["step8_audit"].get(
                "source_reference", ""
            )
            attempt["baseline_step10_reference_id"] = baseline_curve[
                "step10_audit"
            ].get("source_reference", "")

    _write_csv(output / "negative_worklist.csv", worklist)
    _write_csv(output / "existing_fit_comparison.csv", comparison_rows)
    _write_csv(output / "targeted_refit_attempts.csv", new_attempts)
    _write_csv(output / "selected_degradation_results.csv", selected_rows)
    _write_csv(output / "unresolved_negative_cases.csv", unresolved_rows)
    array_path = output / "targeted_refit_curve_arrays.npz"
    temporary = output / "targeted_refit_curve_arrays.npz.tmp"
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(array_path)
    (output / "negative_correction_summary.md").write_text(
        _summary_markdown(
            audit=audit,
            worklist=worklist,
            existing_fits=comparison_rows,
            attempts=new_attempts,
            selected=selected_rows,
            device=device,
            input_hashes=input_hashes,
            scope=scope,
            run_id=run_id,
        ),
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the bounded, versioned negative PyDMA correction pilot."
    )
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--run-id", default="v1")
    parser.add_argument("--scope", choices=("pilot", "deferred"), default="pilot")
    parser.add_argument(
        "--target-key",
        action="append",
        default=[],
        help="Explicit pilot target as PART,CELL,RPT; may be supplied more than once.",
    )
    parser.add_argument("--bol-reference-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    parsed_targets = []
    for value in args.target_key:
        pieces = value.split(",")
        if len(pieces) != 3 or not all(piece.strip() for piece in pieces):
            raise ValueError(f"Invalid --target-key value: {value!r}")
        parsed_targets.append(tuple(piece.strip() for piece in pieces))
    print(
        run_negative_correction(
            args.root,
            args.run_id,
            args.scope,
            tuple(parsed_targets) if parsed_targets else None,
            args.bol_reference_manifest,
            args.output_dir,
        )
    )


if __name__ == "__main__":
    main()
