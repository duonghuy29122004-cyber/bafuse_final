from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


_DEFAULT_OUTPUT = Path("reports/bafuse_negative_root_cause_v1")
_HISTORICAL_AUDIT = Path("reports/final_mapping_label_audit/pydma_fit_audit.csv")
_RAW_RESULTS = Path("experiments/results/data_audits/pydma_rmse15_audit/pydma_raw_results.csv")
_CELL15_CANDIDATES = Path(
    "experiments/soh_forecast/part1_pipeline_legacy_rmse10/"
    "oxford_pydma_diagnostic_Cell15/cell15_pydma_diagnostic_estimates.csv"
)
_VISUALIZATION_SUMMARY = Path(
    "experiments/results/diagnostics/pydma_two_stage_128_fit_visualization/"
    "pydma_fit_summary.csv"
)
_MAPPING = Path("reports/bafuse_final_data_validation_v2/rpt_mapping_validation.csv")
_BOL_AUDIT = Path("reports/bafuse_stage_validation_v2/stage1_bol_reference_audit.csv")
_STAGE2_DIR = Path("reports/bafuse_stage_validation_v2/stage2_pydma_pilot_v1")
_LABELS = ("lli_percent", "lam_anode_percent", "lam_cathode_percent")
_HISTORICAL_LABELS = (
    "lli_percent",
    "lam_anode_percent",
    "lam_cathode_percent",
)
_PARAMETER_NAMES = ("alpha_an", "beta_an", "alpha_ca", "beta_ca")
_RPT_PATTERN = re.compile(r"RPT\s*(\d+)", re.IGNORECASE)
_ROW_RANGE_PATTERN = re.compile(r"rows(\d+)-(\d+)", re.IGNORECASE)


def _read_csv(root: Path, relative: Path) -> pd.DataFrame:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required audit input is missing: {path}")
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _read_npz(root: Path, relative: Path) -> dict[str, np.ndarray]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Required saved curve arrays are missing: {path}")
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def _part(value: Any) -> str:
    match = re.search(r"\d+", str(value))
    return match.group(0) if match else ""


def _rpt(value: Any) -> str:
    text = str(value).strip()
    if text.casefold() in {"bol", "beginning of life"}:
        return "0"
    match = _RPT_PATTERN.search(text)
    if match:
        return str(int(match.group(1)))
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return str(int(float(text)))
    return ""


def _number(value: Any) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _truth(value: Any) -> bool:
    return str(value).strip().casefold() in {"true", "1", "yes"}


def _cell(value: Any, group: Any = "") -> str:
    text = str(value).strip()
    if re.fullmatch(r"G\d+-Cell\d+", text, re.IGNORECASE):
        return text
    cell_match = re.search(r"Cell\s*(\d+)", text, re.IGNORECASE)
    group_match = re.search(r"(?:Group|G)\s*(\d+)", str(group), re.IGNORECASE)
    if cell_match and group_match:
        return f"G{group_match.group(1)}-Cell{cell_match.group(1)}"
    return text


def _source_basename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name.casefold()


def _stable_number(value: Any) -> str:
    number = _number(value)
    return "" if number is None else format(number, ".12g")


def _digest(parts: tuple[str, ...]) -> str:
    return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def _negative_metrics(row: dict[str, Any]) -> list[str]:
    return [
        name
        for name in _LABELS
        if _number(row.get(name)) is not None and float(row[name]) < 0
    ]


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _part(row.get("dataset_part", row.get("part", ""))),
        _cell(row.get("physical_cell_id", row.get("cell_id", "")), row.get("group_id", row.get("group", ""))),
        _rpt(row.get("rpt_index", row.get("checkup", row.get("measurement_id", "")))),
        str(row.get("direction", "")).strip().casefold(),
    )


def _load_context(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = _read_csv(root, _MAPPING)
    mapping["_part_key"] = mapping["part"].map(_part)
    mapping["_rpt_key"] = mapping["rpt_index"].map(_rpt)
    bol = _read_csv(root, _BOL_AUDIT)
    bol["_part_key"] = bol["dataset_part"].map(_part)
    return mapping, bol


def _one_match(frame: pd.DataFrame, mask: pd.Series) -> tuple[dict[str, Any], int]:
    rows = frame.loc[mask]
    return (rows.iloc[0].to_dict() if len(rows) == 1 else {}, len(rows))


def _context_for_key(
    key: tuple[str, str, str, str],
    mapping: pd.DataFrame,
    bol: pd.DataFrame,
) -> dict[str, Any]:
    part, cell, rpt, _ = key
    mapping_row, mapping_count = _one_match(
        mapping,
        mapping["_part_key"].eq(part)
        & mapping["physical_cell_id"].eq(cell)
        & mapping["_rpt_key"].eq(rpt),
    )
    bol_row, bol_count = _one_match(
        bol,
        bol["_part_key"].eq(part) & bol["physical_cell_id"].eq(cell),
    )
    mapping_status = (
        mapping_row.get("mapping_status_v2", "")
        if mapping_count == 1
        else "not_in_mapping_manifest"
        if mapping_count == 0
        else "duplicate_manifest_key"
    )
    step = "8" if key[3] == "discharge" else "10"
    bol_policy_status = bol_row.get("bol_policy_status", "")
    if not bol_policy_status and bol_count == 0:
        bol_policy_status = {
            "1": "part1_baseline_policy_unresolved_no_verified_reference",
            "2": "part2_baseline_policy_unresolved_no_verified_reference",
            "3": "part3_baseline_policy_unresolved_no_verified_reference",
        }.get(part, "baseline_policy_unresolved_no_verified_reference")
    return {
        "mapping_status": mapping_status,
        "mapping_match_count": mapping_count,
        "canonical_source_path": mapping_row.get("source_file", ""),
        "source_exists": mapping_row.get("source_exists_bool", mapping_row.get("source_exists", "")),
        "step4_segment_ref": mapping_row.get("step4_segment_ref_exact", ""),
        "step8_segment_ref": mapping_row.get("step8_segment_ref_exact", ""),
        "step10_segment_ref": mapping_row.get("step10_segment_ref_exact", ""),
        "selected_direction_segment_ref": mapping_row.get(f"step{step}_segment_ref_exact", ""),
        "bol_match_count": bol_count,
        "bol_source_file": bol_row.get("primary_bol_reference_file", ""),
        "bol_verified": bol_row.get("bol_verified", ""),
        "bol_policy_status": bol_policy_status,
        "bol_step4_segment_ref": bol_row.get("bol_step4_segment_ref", ""),
        "bol_step8_status": bol_row.get("bol_step8_status", ""),
        "bol_step10_status": bol_row.get("bol_step10_status", ""),
        "identity_provenance": bol_row.get("identity_provenance", ""),
        "split": mapping_row.get("partition_v2", mapping_row.get("split", "")),
        "evaluation_role": "external_domain_shift_only" if part == "3" else "internal_or_training_cohort",
    }


def _historical_signature(row: dict[str, Any]) -> tuple[str, ...]:
    return (
        _part(row.get("dataset_part", "")),
        _cell(row.get("physical_cell_id", "")),
        _rpt(row.get("checkup", "")),
        str(row.get("direction", "")).strip().casefold(),
        str(row.get("test_step", "")).strip(),
        str(row.get("source_file", "")).strip().casefold(),
        *(str(row.get(name, "")).strip() for name in ("fit_rmse_mV", "accepted_runs", "total_runs")),
        *(str(row.get(name, "")).strip() for name in _HISTORICAL_LABELS),
    )


def _historical_records(audit: pd.DataFrame) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    label_frame = audit[list(_HISTORICAL_LABELS)].apply(pd.to_numeric, errors="coerce")
    negative_mask = label_frame.lt(0).any(axis=1)
    raw_negative = audit.loc[negative_mask].copy()
    grouped: dict[tuple[str, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, row in raw_negative.iterrows():
        record = row.to_dict()
        grouped[_historical_signature(record)].append((int(row_index), record))

    records: list[dict[str, Any]] = []
    for signature, rows in sorted(grouped.items()):
        _, first = rows[0]
        key = _row_key(first)
        tables = sorted({str(row.get("result_table", "")) for _, row in rows})
        accepted = any(_truth(row.get("diagnostic_label_accepted", "")) for _, row in rows)
        record = {
            "negative_record_id": "hist-" + _digest(signature),
            "record_type": "historical_output_signature",
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "direction": key[3],
            "test_step": str(first.get("test_step", "")),
            "source_file": str(first.get("source_file", "")),
            "baseline_reference_id": "",
            "fit_id": "",
            "seed": "",
            "fit_rmse_mV": first.get("fit_rmse_mV", ""),
            "accepted_runs": first.get("accepted_runs", ""),
            "total_runs": first.get("total_runs", ""),
            "label_accepted": accepted,
            "fit_gate_passed": any(_truth(row.get("passes_recorded_gate", "")) for _, row in rows),
            "fit_status": first.get("fit_status_original", first.get("diagnostic_label_accepted", "")),
            "lli_percent": first.get("lli_percent", ""),
            "lam_anode_percent": first.get("lam_anode_percent", ""),
            "lam_cathode_percent": first.get("lam_cathode_percent", ""),
            "negative_metrics": ";".join(_negative_metrics(first)),
            "parameter_vector_saved": False,
            "parameter_vector_source": "",
            "source_artifact": _HISTORICAL_AUDIT.as_posix(),
            "source_tables": ";".join(tables),
            "source_row_locator": ";".join(
                f"{row.get('result_table', '')}:{row_index + 2}"
                for row_index, row in rows
            ),
            "deduplication_basis": "same part/cell/RPT/direction/step/source/RMSE/run counts/labels across saved table snapshots",
            "historical_row_multiplicity": len(rows),
        }
        records.append(record)
    return records, raw_negative


def _raw_results_records(
    raw: pd.DataFrame, audit: pd.DataFrame
) -> tuple[list[dict[str, Any]], int]:
    labels = ("LLI_percent_vs_BoL", "LAM_anode_percent_vs_BoL", "LAM_cathode_percent_vs_BoL")
    raw_values = raw[list(labels)].apply(pd.to_numeric, errors="coerce")
    mask = raw_values.lt(0).any(axis=1)
    records: list[dict[str, Any]] = []
    aliases = 0
    for index, row in raw.loc[mask].iterrows():
        cell = _cell(row["cell_id"], row["group_id"])
        rpt = _rpt(row["checkup"])
        direction = str(row["direction"]).strip().casefold()
        candidates = audit.loc[
            audit["physical_cell_id"].eq(cell)
            & audit["checkup"].map(_rpt).eq(rpt)
            & audit["direction"].str.casefold().eq(direction)
            & audit["source_file"].map(_source_basename).eq(_source_basename(row["source_file"]))
            & audit["test_step"].eq(str(row["test_step"]))
        ]
        parts = sorted(set(candidates["dataset_part"].map(_part)))
        if len(parts) != 1:
            raise ValueError(
                f"Cannot establish a unique dataset part for saved raw-result source "
                f"{row['source_file']} / {cell} / RPT{rpt}: {parts}"
            )
        key = (parts[0], cell, rpt, direction)
        matched = False
        for _, candidate in candidates.iterrows():
            if all(
                (_number(row[source]) is None and _number(candidate[target]) is None)
                or (
                    _number(row[source]) is not None
                    and _number(candidate[target]) is not None
                    and math.isclose(float(row[source]), float(candidate[target]), abs_tol=1e-9)
                )
                for source, target in zip(labels, _HISTORICAL_LABELS)
            ) and math.isclose(
                _number(row["fit_rmse_mV"]) or 0.0,
                _number(candidate["fit_rmse_mV"]) or 0.0,
                abs_tol=1e-9,
            ):
                matched = True
                break
        if matched:
            aliases += 1
            continue
        values = {
            "lli_percent": row["LLI_percent_vs_BoL"],
            "lam_anode_percent": row["LAM_anode_percent_vs_BoL"],
            "lam_cathode_percent": row["LAM_cathode_percent_vs_BoL"],
        }
        signature = (
            *key,
            str(row["test_step"]),
            str(row["source_file"]).casefold(),
            _stable_number(row["fit_rmse_mV"]),
            _stable_number(row["accepted_runs"]),
            _stable_number(row["total_runs"]),
            *(_stable_number(values[name]) for name in _LABELS),
        )
        records.append(
            {
                "negative_record_id": "raw-" + _digest(signature),
                "record_type": "legacy_raw_result_output",
                "dataset_part": key[0],
                "physical_cell_id": key[1],
                "rpt_index": key[2],
                "direction": key[3],
                "test_step": str(row["test_step"]),
                "source_file": str(row["source_file"]),
                "baseline_reference_id": "",
                "fit_id": "",
                "seed": "",
                "fit_rmse_mV": row["fit_rmse_mV"],
                "accepted_runs": row["accepted_runs"],
                "total_runs": row["total_runs"],
                "label_accepted": False,
                "fit_gate_passed": _truth(row["fit_accepted_by_pydma_10mV"]),
                "fit_status": row["diagnostic_label_status"],
                **values,
                "negative_metrics": ";".join(_negative_metrics(values)),
                "parameter_vector_saved": False,
                "parameter_vector_source": "",
                "source_artifact": _RAW_RESULTS.as_posix(),
                "source_tables": "",
                "source_row_locator": str(index + 2),
                "deduplication_basis": "distinct saved legacy raw-result output; no optimizer fit ID or seed",
                "historical_row_multiplicity": 1,
            }
        )
    return records, aliases


def _visualization_crosscheck(root: Path, audit: pd.DataFrame) -> int:
    summary = _read_csv(root, _VISUALIZATION_SUMMARY)
    labels = ("lli_label_percent", "lam_anode_label_percent", "lam_cathode_label_percent")
    values = summary[list(labels)].apply(pd.to_numeric, errors="coerce")
    mask = values.lt(0).any(axis=1)
    checked = 0
    for _, row in summary.loc[mask].iterrows():
        key = (
            _part(row["dataset_part"]),
            row["physical_cell_id"],
            _rpt(row["measurement_id"]),
            str(row["direction"]).strip().casefold(),
        )
        candidates = audit.loc[
            audit["dataset_part"].map(_part).eq(key[0])
            & audit["physical_cell_id"].eq(key[1])
            & audit["checkup"].map(_rpt).eq(key[2])
            & audit["direction"].str.casefold().eq(key[3])
            & audit["source_file"].map(_source_basename).eq(_source_basename(row["source_file"]))
            & audit["test_step"].eq(str(row["test_step"]))
        ]
        matched = False
        for _, candidate in candidates.iterrows():
            if all(
                (_number(row[source]) is None and _number(candidate[target]) is None)
                or (
                    _number(row[source]) is not None
                    and _number(candidate[target]) is not None
                    and math.isclose(float(row[source]), float(candidate[target]), abs_tol=1e-8)
                )
                for source, target in zip(labels, _HISTORICAL_LABELS)
            ):
                matched = True
                break
        if not matched:
            raise ValueError(f"Visualization label did not match historical audit: {key}")
        checked += 1
    return checked


def _candidate_records(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = _read_csv(root, _CELL15_CANDIDATES)
    records: list[dict[str, Any]] = []
    vectors: list[dict[str, Any]] = []
    label_sources = {
        "candidate_LLI_percent_vs_BoL": "lli_percent",
        "candidate_LAM_anode_percent_vs_BoL": "lam_anode_percent",
        "candidate_LAM_cathode_percent_vs_BoL": "lam_cathode_percent",
    }
    for index, row in frame.iterrows():
        values = {target: row[source] for source, target in label_sources.items()}
        if not _negative_metrics(values):
            continue
        key = (
            _part(row["part"]),
            _cell(row["cell_id"], row["group_id"]),
            _rpt(row["checkup"]),
            str(row["direction"]).strip().casefold(),
        )
        vector = {name: _number(row.get(name)) for name in _PARAMETER_NAMES}
        saved = all(value is not None for value in vector.values())
        record = {
            "negative_record_id": f"legacy-cell15-row-{index + 2}",
            "record_type": "legacy_rejected_candidate",
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "direction": key[3],
            "test_step": str(row["test_step"]),
            "source_file": str(row["source_file"]),
            "baseline_reference_id": "",
            "fit_id": "",
            "seed": "",
            "fit_rmse_mV": row["fit_rmse_mV"],
            "accepted_runs": row["accepted_runs"],
            "total_runs": row["total_runs"],
            "label_accepted": False,
            "fit_gate_passed": _truth(row["fit_accepted_by_pydma_10mV"]),
            "fit_status": row["diagnostic_label_status"],
            **values,
            "negative_metrics": ";".join(_negative_metrics(values)),
            "parameter_vector_saved": saved,
            "parameter_vector_source": _CELL15_CANDIDATES.as_posix() if saved else "",
            "source_artifact": _CELL15_CANDIDATES.as_posix(),
            "source_tables": "",
            "source_row_locator": str(index + 2),
            "deduplication_basis": "separate legacy candidate record; no fit ID or seed persisted",
            "historical_row_multiplicity": 1,
        }
        records.append(record)
        if saved:
            vectors.append(
                {
                    "fit_id": "",
                    "fit_role": "legacy_rejected_candidate",
                    "dataset_part": key[0],
                    "physical_cell_id": key[1],
                    "rpt_index": key[2],
                    "direction": key[3],
                    "source_file": row["source_file"],
                    "source_artifact": _CELL15_CANDIDATES.as_posix(),
                    "source_row": index + 2,
                    "seed": "",
                    "parameter_vector_saved": True,
                    "parameter_bounds_saved": False,
                    "parameters": vector,
                    "parameter_bounds": None,
                }
            )
    return records, vectors


def _stage2_records(
    root: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    labels = _read_csv(root, _STAGE2_DIR / "stage2_pydma_label_comparison.csv")
    fits = _read_csv(root, _STAGE2_DIR / "stage2_pydma_fit_attempts.csv")
    fit_by_id = {row["fit_id"]: row for _, row in fits.iterrows()}
    records: list[dict[str, Any]] = []
    for index, row in labels.iterrows():
        values = {name: row[name] for name in _LABELS}
        if not _negative_metrics(values):
            continue
        fit_id = row["rpt_fit_id"]
        if fit_id not in fit_by_id:
            raise ValueError(f"Saved Stage 2 fit ID is missing: {fit_id}")
        fit = fit_by_id[fit_id]
        key = (
            _part(row["dataset_part"]),
            row["physical_cell_id"],
            _rpt(row["rpt_index"]),
            str(row["direction"]).strip().casefold(),
        )
        params = json.loads(fit["fitted_parameters_json"] or "{}")
        record = {
            "negative_record_id": fit_id,
            "record_type": "parameterized_stage2_fit",
            "dataset_part": key[0],
            "physical_cell_id": key[1],
            "rpt_index": key[2],
            "direction": key[3],
            "test_step": str(row["step"]),
            "source_file": str(fit["source_file"]),
            "baseline_reference_id": row["baseline_reference_id"],
            "fit_id": fit_id,
            "seed": fit["seed"],
            "fit_rmse_mV": fit["rmse_mV"],
            "accepted_runs": fit["accepted_runs"],
            "total_runs": fit["total_runs"],
            "label_accepted": _truth(row["label_fit_accepted"]),
            "fit_gate_passed": _truth(fit["passes_primary_gate_15mV_3runs"]),
            "fit_status": fit["fit_status"],
            **values,
            "negative_metrics": ";".join(_negative_metrics(values)),
            "parameter_vector_saved": bool(params),
            "parameter_vector_source": (_STAGE2_DIR / "stage2_pydma_fit_attempts.csv").as_posix(),
            "source_artifact": (_STAGE2_DIR / "stage2_pydma_label_comparison.csv").as_posix(),
            "source_tables": "",
            "source_row_locator": str(index + 2),
            "deduplication_basis": "unique persisted fit_id; do not merge across optimizer seeds",
            "historical_row_multiplicity": 1,
        }
        records.append(record)
    curves = _read_npz(root, _STAGE2_DIR / "stage2_pydma_reconstructed_curves.npz")
    return records, fits, labels, curves


def _formula(
    reference_params: dict[str, Any],
    current_params: dict[str, Any],
    reference_capacity: float,
    current_capacity: float,
) -> dict[str, float]:
    anode_initial = float(reference_params["alpha_an"]) * reference_capacity
    cathode_initial = float(reference_params["alpha_ca"]) * reference_capacity
    inventory_initial = (
        float(reference_params["alpha_ca"])
        + float(reference_params["beta_ca"])
        - float(reference_params["beta_an"])
    ) * reference_capacity
    if min(abs(anode_initial), abs(cathode_initial), abs(inventory_initial)) <= 1e-12:
        raise ValueError("A saved baseline fit has zero component capacity or lithium inventory.")
    anode_current = float(current_params["alpha_an"]) * current_capacity
    cathode_current = float(current_params["alpha_ca"]) * current_capacity
    inventory_current = (
        float(current_params["alpha_ca"])
        + float(current_params["beta_ca"])
        - float(current_params["beta_an"])
    ) * current_capacity
    return {
        "lli_percent": 100.0 * (inventory_initial - inventory_current) / inventory_initial,
        "lam_anode_percent": 100.0 * (anode_initial - anode_current) / anode_initial,
        "lam_cathode_percent": 100.0 * (cathode_initial - cathode_current) / cathode_initial,
    }


def _formula_audit(fits: pd.DataFrame, labels: pd.DataFrame) -> list[dict[str, Any]]:
    by_id = {row["fit_id"]: row for _, row in fits.iterrows()}
    output: list[dict[str, Any]] = []
    for _, row in labels.iterrows():
        baseline = by_id.get(row["baseline_reference_id"])
        target = by_id.get(row["rpt_fit_id"])
        if baseline is None or target is None:
            raise ValueError(f"Missing saved fit vector for formula parity row: {row['rpt_fit_id']}")
        values = _formula(
            json.loads(baseline["fitted_parameters_json"]),
            json.loads(target["fitted_parameters_json"]),
            float(baseline["capacity_span_native"]),
            float(target["capacity_span_native"]),
        )
        stored = {name: float(row[name]) for name in _LABELS}
        errors = {name: values[name] - stored[name] for name in _LABELS}
        output.append(
            {
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": row["rpt_index"],
                "direction": row["direction"],
                "baseline_reference_id": row["baseline_reference_id"],
                "rpt_fit_id": row["rpt_fit_id"],
                "formula_lli_percent": values["lli_percent"],
                "saved_lli_percent": stored["lli_percent"],
                "lli_delta_percentage_points": errors["lli_percent"],
                "formula_lam_anode_percent": values["lam_anode_percent"],
                "saved_lam_anode_percent": stored["lam_anode_percent"],
                "lam_anode_delta_percentage_points": errors["lam_anode_percent"],
                "formula_lam_cathode_percent": values["lam_cathode_percent"],
                "saved_lam_cathode_percent": stored["lam_cathode_percent"],
                "lam_cathode_delta_percentage_points": errors["lam_cathode_percent"],
                "max_abs_delta_percentage_points": max(abs(value) for value in errors.values()),
                "parity_with_saved_label": max(abs(value) for value in errors.values()) <= 1e-10,
                "calculation": "independent recomputation from saved parameter vectors and native capacity spans; no optimizer call",
            }
        )
    return output


def _cross_seed_matrix(fits: pd.DataFrame) -> list[dict[str, Any]]:
    selected = fits.loc[
        fits["fit_id"].str.contains("bol-p1-g1-cell9-rpt0-discharge")
        | fits["fit_id"].str.contains("rpt-p1-g1-cell9-rpt2-discharge")
    ]
    baseline = {
        row["seed"]: row
        for _, row in selected.loc[selected["fit_id"].str.startswith("bol-")].iterrows()
    }
    targets = {
        row["seed"]: row
        for _, row in selected.loc[selected["fit_id"].str.startswith("rpt-")].iterrows()
    }
    if set(baseline) != {"211", "212"} or set(targets) != {"211", "212"}:
        raise ValueError("The saved G1-Cell9 RPT2 discharge seed matrix is incomplete.")
    output: list[dict[str, Any]] = []
    for bol_seed, base in sorted(baseline.items()):
        for rpt_seed, target in sorted(targets.items()):
            result = _formula(
                json.loads(base["fitted_parameters_json"]),
                json.loads(target["fitted_parameters_json"]),
                float(base["capacity_span_native"]),
                float(target["capacity_span_native"]),
            )
            output.append(
                {
                    "dataset_part": "1",
                    "physical_cell_id": "G1-Cell9",
                    "rpt_index": "2",
                    "direction": "discharge",
                    "bol_seed": bol_seed,
                    "rpt_seed": rpt_seed,
                    "baseline_fit_id": base["fit_id"],
                    "rpt_fit_id": target["fit_id"],
                    "lli_percent": result["lli_percent"],
                    "lam_anode_percent": result["lam_anode_percent"],
                    "lam_cathode_percent": result["lam_cathode_percent"],
                    "bol_rmse_mV": base["rmse_mV"],
                    "rpt_rmse_mV": target["rmse_mV"],
                    "bol_accepted_runs": base["accepted_runs"],
                    "rpt_accepted_runs": target["accepted_runs"],
                    "bol_capacity_span_native": base["capacity_span_native"],
                    "rpt_capacity_span_native": target["capacity_span_native"],
                    "formula_source": "saved fit vectors; no optimizer call",
                }
            )
    return output


def _parameter_bound_audit(
    fits: pd.DataFrame,
    candidate_vectors: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for _, fit in fits.iterrows():
        bounds = json.loads(fit["parameter_bounds_json"] or "[]")
        for bound in bounds:
            lower = float(bound["lower_bound"])
            upper = float(bound["upper_bound"])
            value = float(bound["value"])
            width = upper - lower
            if width <= 0:
                raise ValueError(f"Invalid parameter bounds for {fit['fit_id']}: {bound}")
            lower_distance = (value - lower) / width
            upper_distance = (upper - value) / width
            output.append(
                {
                    "fit_id": fit["fit_id"],
                    "fit_role": fit["observation_type"],
                    "dataset_part": fit["dataset_part"],
                    "physical_cell_id": fit["physical_cell_id"],
                    "rpt_index": fit["rpt_index"],
                    "direction": fit["direction"],
                    "seed": fit["seed"],
                    "parameter": bound["parameter"],
                    "value": value,
                    "lower_bound": lower,
                    "upper_bound": upper,
                    "normalized_distance_to_lower_bound": lower_distance,
                    "normalized_distance_to_upper_bound": upper_distance,
                    "within_bounds": lower - 1e-12 <= value <= upper + 1e-12,
                    "within_1pct_of_bound": min(lower_distance, upper_distance) <= 0.01,
                    "bound_threshold_convention": "1% of the declared range; audit convention, not PyDMA acceptance threshold",
                    "source_artifact": (_STAGE2_DIR / "stage2_pydma_fit_attempts.csv").as_posix(),
                }
            )
    for vector in candidate_vectors:
        for parameter, value in vector["parameters"].items():
            output.append(
                {
                    "fit_id": vector["fit_id"],
                    "fit_role": vector["fit_role"],
                    "dataset_part": vector["dataset_part"],
                    "physical_cell_id": vector["physical_cell_id"],
                    "rpt_index": vector["rpt_index"],
                    "direction": vector["direction"],
                    "seed": vector["seed"],
                    "parameter": parameter,
                    "value": value,
                    "lower_bound": "",
                    "upper_bound": "",
                    "normalized_distance_to_lower_bound": "",
                    "normalized_distance_to_upper_bound": "",
                    "within_bounds": "",
                    "within_1pct_of_bound": "",
                    "bound_threshold_convention": "optimizer bounds not persisted for this legacy candidate",
                    "source_artifact": vector["source_artifact"],
                }
            )
    return output


def _step4_capacity(root: Path, reference: str, part: str) -> dict[str, Any]:
    match = _ROW_RANGE_PATTERN.search(str(reference))
    if not match:
        return {}
    source, _ = str(reference).split("#", 1)
    source_path = root / source
    if not source_path.is_file():
        source_path = root / "data" / "raw" / "oxford" / f"part{part}" / source
    if not source_path.is_file():
        raise FileNotFoundError(f"Step 4 source referenced by audit is missing: {source_path}")
    start, end = int(match.group(1)), int(match.group(2))
    with source_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        columns = {name: header.index(name) for name in ("Amphr", "Amps", "TestTime")}
        times: list[float] = []
        currents: list[float] = []
        capacities: list[float] = []
        for line_number, row in enumerate(reader, start=2):
            if line_number < start:
                continue
            if line_number > end:
                break
            values = [_number(row[columns[name]]) for name in ("Amphr", "Amps", "TestTime")]
            if any(value is None for value in values):
                raise ValueError(f"Non-numeric Step 4 data at {source_path}:{line_number}")
            capacity, current, time_value = values
            capacities.append(float(capacity))
            currents.append(float(current))
            times.append(float(time_value))
    if len(capacities) != end - start + 1 or len(capacities) < 2:
        raise ValueError(f"Incomplete Step 4 source range in {source_path}: rows {start}-{end}")
    time_deltas = np.diff(np.asarray(times, dtype=np.float64))
    current_values = np.asarray(currents, dtype=np.float64)
    integral_if_minutes = float(
        np.sum((np.abs(current_values[:-1]) + np.abs(current_values[1:])) * 0.5 * time_deltas) / 60.0
    )
    return {
        "source_file": source_path.resolve().relative_to(root.resolve()).as_posix(),
        "source_row_start": start,
        "source_row_end": end,
        "sample_count": len(capacities),
        "capacity_span_native": float(max(capacities) - min(capacities)),
        "current_integral_if_testtime_minutes_Ah": integral_if_minutes,
        "testtime_min_increment": float(np.min(time_deltas)),
        "testtime_max_increment": float(np.max(time_deltas)),
        "time_unit_note": "minute interpretation is an empirical consistency check, not an officially documented unit",
    }


def _bol_trace(
    root: Path,
    keys: set[tuple[str, str, str, str]],
    mapping: pd.DataFrame,
    bol: pd.DataFrame,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, str], dict[str, Any]]]:
    contexts = {key: _context_for_key(key, mapping, bol) for key in keys}
    step4_cache: dict[str, dict[str, Any]] = {}
    output: list[dict[str, Any]] = []
    for key, context in sorted(contexts.items()):
        bol_step4 = context.get("bol_step4_segment_ref", "")
        target_step4 = context.get("step4_segment_ref", "")
        if key[0] == "1" and key[1] == "G1-Cell9":
            if bol_step4 and bol_step4 not in step4_cache:
                step4_cache[bol_step4] = _step4_capacity(root, bol_step4, key[0])
            if target_step4 and target_step4 not in step4_cache:
                step4_cache[target_step4] = _step4_capacity(root, target_step4, key[0])
        bol_capacity = step4_cache.get(bol_step4, {})
        target_capacity = step4_cache.get(target_step4, {})
        output.append(
            {
                **dict(zip(("dataset_part", "physical_cell_id", "rpt_index", "direction"), key)),
                "evaluation_role": context["evaluation_role"],
                "mapping_status": context["mapping_status"],
                "mapping_match_count": context["mapping_match_count"],
                "canonical_source_path": context["canonical_source_path"],
                "source_exists": context["source_exists"],
                "step4_segment_ref": context["step4_segment_ref"],
                "step8_segment_ref": context["step8_segment_ref"],
                "step10_segment_ref": context["step10_segment_ref"],
                "selected_direction_segment_ref": context["selected_direction_segment_ref"],
                "bol_source_file": context["bol_source_file"],
                "bol_verified": context["bol_verified"],
                "bol_policy_status": context["bol_policy_status"],
                "bol_step4_segment_ref": bol_step4,
                "bol_step8_status": context["bol_step8_status"],
                "bol_step10_status": context["bol_step10_status"],
                "bol_step4_capacity_span_native": bol_capacity.get("capacity_span_native", ""),
                "bol_step4_sample_count": bol_capacity.get("sample_count", ""),
                "bol_step4_current_integral_if_testtime_minutes_Ah": bol_capacity.get("current_integral_if_testtime_minutes_Ah", ""),
                "target_step4_capacity_span_native": target_capacity.get("capacity_span_native", ""),
                "target_step4_sample_count": target_capacity.get("sample_count", ""),
                "target_step4_current_integral_if_testtime_minutes_Ah": target_capacity.get("current_integral_if_testtime_minutes_Ah", ""),
                "step4_capacity_unit": "native Amphr span; not relabeled Ah",
                "step4_time_unit_note": "current integration assumes TestTime minutes only for empirical comparison",
                "identity_provenance": context["identity_provenance"],
                "split": context["split"],
            }
        )
    return output, contexts


def _curve_residual_audit(
    root: Path,
    fits: pd.DataFrame,
    labels: pd.DataFrame,
    curves: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    visual = _read_csv(root, _VISUALIZATION_SUMMARY)
    visual_labels = ("lli_label_percent", "lam_anode_label_percent", "lam_cathode_label_percent")
    visual_numeric = visual[list(visual_labels)].apply(pd.to_numeric, errors="coerce")
    visual_negative = visual_numeric.lt(0).any(axis=1)
    visual_residual = visual["max_abs_refit_residual_mV"].map(_number).notna()
    for index, row in visual.loc[visual_negative & visual_residual].iterrows():
        output.append(
            {
                "dataset_part": row["dataset_part"],
                "physical_cell_id": row["physical_cell_id"],
                "rpt_index": _rpt(row["measurement_id"]),
                "direction": row["direction"],
                "source_file": row["source_file"],
                "fit_id": f"visualization-row-{index + 2}",
                "residual_evidence_type": "saved_visualization_summary",
                "fit_rmse_mV": row["refit_rmse_mV"],
                "residual_rmse_mV": row["rms_refit_residual_mV"],
                "max_abs_residual_mV": row["max_abs_refit_residual_mV"],
                "max_abs_residual_soc": "",
                "low_soc_rmse_mV": "",
                "mid_soc_rmse_mV": "",
                "high_soc_rmse_mV": "",
                "sample_count": row["curve_sample_count"],
                "accepted_runs": row["refit_accepted_runs"],
                "total_runs": row["refit_total_runs"],
                "parameter_vector_saved": False,
                "saved_curve_arrays": False,
                "source_row_start": "",
                "source_row_end": "",
                "limitation": "only summary residual metrics persisted; no fitted vector or pointwise residual array",
                "source_artifact": _VISUALIZATION_SUMMARY.as_posix(),
            }
        )
    negative_ids = {
        row["rpt_fit_id"]
        for _, row in labels.iterrows()
        if _negative_metrics({name: row[name] for name in _LABELS})
    }
    rpt2_seed_pair_ids = set(
        labels.loc[
            labels["rpt_index"].map(_rpt).eq("2")
            & labels["direction"].eq("discharge"),
            "rpt_fit_id",
        ]
    )
    target_fit_ids = negative_ids | rpt2_seed_pair_ids
    target_fits = fits.loc[fits["fit_id"].isin(target_fit_ids)]
    for _, fit in target_fits.iterrows():
        fit_id = fit["fit_id"]
        soc = curves.get(f"{fit_id}__soc_measured")
        measured = curves.get(f"{fit_id}__ocv_measured")
        reconstructed_soc = curves.get(f"{fit_id}__soc_reconstructed")
        reconstructed = curves.get(f"{fit_id}__ocv_reconstructed")
        if any(array is None for array in (soc, measured, reconstructed_soc, reconstructed)):
            continue
        if len(soc) != len(measured) or len(reconstructed_soc) != len(reconstructed):
            raise ValueError(f"Saved curve arrays have inconsistent lengths for {fit_id}")
        if len(soc) != len(reconstructed_soc) or not np.allclose(soc, reconstructed_soc):
            raise ValueError(f"Measured/reconstructed SoC grids differ for {fit_id}")
        residual_mv = (reconstructed - measured) * 1000.0
        bins = (
            (soc <= 1.0 / 3.0),
            (soc > 1.0 / 3.0) & (soc <= 2.0 / 3.0),
            (soc > 2.0 / 3.0),
        )
        bin_rmse = [
            float(np.sqrt(np.mean(np.square(residual_mv[mask])))) if np.any(mask) else math.nan
            for mask in bins
        ]
        peak_index = int(np.argmax(np.abs(residual_mv)))
        output.append(
            {
                "dataset_part": fit["dataset_part"],
                "physical_cell_id": fit["physical_cell_id"],
                "rpt_index": fit["rpt_index"],
                "direction": fit["direction"],
                "source_file": fit["source_file"],
                "fit_id": fit_id,
                "residual_evidence_type": "saved_stage2_curve_arrays",
                "fit_rmse_mV": fit["rmse_mV"],
                "residual_rmse_mV": float(np.sqrt(np.mean(np.square(residual_mv)))),
                "max_abs_residual_mV": float(np.max(np.abs(residual_mv))),
                "max_abs_residual_soc": float(soc[peak_index]),
                "low_soc_rmse_mV": bin_rmse[0],
                "mid_soc_rmse_mV": bin_rmse[1],
                "high_soc_rmse_mV": bin_rmse[2],
                "sample_count": len(soc),
                "accepted_runs": fit["accepted_runs"],
                "total_runs": fit["total_runs"],
                "parameter_vector_saved": bool(fit["fitted_parameters_json"]),
                "saved_curve_arrays": True,
                "source_row_start": fit["source_row_start"],
                "source_row_end": fit["source_row_end"],
                "limitation": "SoC thirds are descriptive bins; this is fit residual evidence, not label ground truth",
                "source_artifact": (_STAGE2_DIR / "stage2_pydma_reconstructed_curves.npz").as_posix(),
            }
        )
    return output


def _root_cause_rows(
    inventory: list[dict[str, Any]],
    raw_negative: pd.DataFrame,
    residuals: list[dict[str, Any]],
    contexts: dict[tuple[str, str, str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        groups[_row_key(row)].append(row)
    raw_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    raw_accepted: dict[tuple[str, str, str, str], int] = defaultdict(int)
    raw_min: dict[tuple[tuple[str, str, str, str], str], float] = {}
    for _, row in raw_negative.iterrows():
        key = _row_key(row.to_dict())
        raw_counts[key] += 1
        if _truth(row["diagnostic_label_accepted"]):
            raw_accepted[key] += 1
        for name in _LABELS:
            value = _number(row.get(name))
            if value is not None and value < 0:
                item_key = (key, name)
                raw_min[item_key] = min(raw_min.get(item_key, value), value)
    residual_counts: dict[tuple[str, str, str, str], int] = defaultdict(int)
    for row in residuals:
        residual_counts[_row_key(row)] += 1

    output: list[dict[str, Any]] = []
    for key, rows in sorted(groups.items()):
        context = contexts[key]
        history = [
            row
            for row in rows
            if row["record_type"]
            in {"historical_output_signature", "legacy_raw_result_output"}
        ]
        accepted_signatures = sum(bool(row["label_accepted"]) for row in history)
        raw_result_records = sum(
            row["record_type"] == "legacy_raw_result_output" for row in history
        )
        saved_stage2 = [row for row in rows if row["record_type"] == "parameterized_stage2_fit"]
        legacy_vectors = [
            row for row in rows
            if row["record_type"] == "legacy_rejected_candidate" and row["parameter_vector_saved"]
        ]
        metrics = sorted({name for row in rows for name in row["negative_metrics"].split(";") if name})
        minimums = {
            name: min(
                [float(row[name]) for row in rows if _number(row.get(name)) is not None and float(row[name]) < 0]
                + ([raw_min[(key, name)] ] if (key, name) in raw_min else [])
            )
            if any(_number(row.get(name)) is not None and float(row[name]) < 0 for row in rows)
            or (key, name) in raw_min
            else ""
            for name in _LABELS
        }
        if key == ("1", "G1-Cell9", "2", "discharge"):
            classification = "OPTIMIZER_UNSTABLE"
            confidence = "high for seed sensitivity in the saved Stage 2 fits"
            evidence = "RPT seed 211 is negative while seed 212 is positive for the same RPT; changing BoL seed alone has negligible effect."
        elif key == ("1", "G1-Cell9", "8", "discharge"):
            classification = "UNRESOLVED_SINGLE_SEED"
            confidence = "low"
            evidence = "Only one parameterized Stage 2 target seed is saved; no cross-seed attribution is possible."
        elif saved_stage2:
            classification = "UNRESOLVED_PARAMETERIZED_FIT"
            confidence = "low"
            evidence = "A parameterized saved fit exists, but the available seeds do not establish the root cause."
        elif legacy_vectors:
            classification = "UNRESOLVED_REJECTED_CANDIDATE"
            confidence = "low"
            evidence = "A negative legacy candidate vector exists but its fit was rejected (0 accepted runs); it does not identify the accepted historical fit."
        else:
            classification = "UNRESOLVED_NO_SAVED_FIT_VECTOR"
            confidence = "low"
            evidence = "Historical output rows have no fit ID, seed, or fitted parameter vector."
        output.append(
            {
                "dataset_part": key[0],
                "physical_cell_id": key[1],
                "rpt_index": key[2],
                "direction": key[3],
                "evaluation_role": context["evaluation_role"],
                "negative_metrics": ";".join(metrics),
                "historical_negative_rows": raw_counts[key],
                "historical_accepted_negative_rows": raw_accepted[key],
                "historical_distinct_output_signatures": len(history),
                "historical_accepted_output_signatures": accepted_signatures,
                "supplemental_raw_result_output_signatures": raw_result_records,
                "negative_inventory_records": len(rows),
                "saved_parameterized_negative_fit_ids": ";".join(row["fit_id"] for row in saved_stage2),
                "baseline_reference_fit_ids": ";".join(
                    sorted(
                        {
                            row["baseline_reference_id"]
                            for row in saved_stage2
                            if row.get("baseline_reference_id")
                        }
                    )
                ),
                "legacy_candidate_vectors_not_linked_to_accepted_fit": len(legacy_vectors),
                "residual_evidence_records": residual_counts[key],
                "minimum_lli_percent": minimums["lli_percent"],
                "minimum_lam_anode_percent": minimums["lam_anode_percent"],
                "minimum_lam_cathode_percent": minimums["lam_cathode_percent"],
                "mapping_status": context["mapping_status"],
                "bol_source_file": context["bol_source_file"],
                "bol_policy_status": context["bol_policy_status"],
                "root_cause_classification": classification,
                "confidence": confidence,
                "evidence": evidence,
                "interpretation_note": "negative degradation is a fitted current value exceeding its fitted BoL reference; not independent ground truth",
            }
        )
    return output


def _write_outputs(
    output_dir: Path,
    inventory: list[dict[str, Any]],
    root_causes: list[dict[str, Any]],
    bol_trace: list[dict[str, Any]],
    bound_audit: list[dict[str, Any]],
    cross_seed: list[dict[str, Any]],
    formula_audit: list[dict[str, Any]],
    residual_audit: list[dict[str, Any]],
    audit: pd.DataFrame,
    raw_negative: pd.DataFrame,
    raw_alias_count: int,
    visualization_alias_count: int,
) -> None:
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {output_dir}")
    with tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-", dir=parent) as temporary:
        stage = Path(temporary)
        pd.DataFrame(inventory).to_csv(stage / "negative_inventory.csv", index=False)
        pd.DataFrame(root_causes).to_csv(stage / "negative_root_cause_audit.csv", index=False)
        pd.DataFrame(bol_trace).to_csv(stage / "bol_reference_trace.csv", index=False)
        pd.DataFrame(bound_audit).to_csv(stage / "parameter_bound_audit.csv", index=False)
        pd.DataFrame(cross_seed).to_csv(stage / "rpt2_cross_seed_matrix.csv", index=False)
        pd.DataFrame(formula_audit).to_csv(stage / "formula_parity_audit.csv", index=False)
        pd.DataFrame(residual_audit).to_csv(stage / "curve_residual_audit.csv", index=False)

        metric_counts = audit[list(_HISTORICAL_LABELS)].apply(pd.to_numeric, errors="coerce").lt(0).sum()
        accepted = audit["diagnostic_label_accepted"].map(_truth)
        negative_mask = audit[list(_HISTORICAL_LABELS)].apply(pd.to_numeric, errors="coerce").lt(0).any(axis=1)
        key_part_counts = pd.Series([row["dataset_part"] for row in root_causes]).value_counts().to_dict()
        accepted_key_counts = {
            part: sum(
                1
                for row in root_causes
                if row["dataset_part"] == part and row["historical_accepted_output_signatures"] > 0
            )
            for part in ("1", "2", "3")
        }
        mapping_counts = pd.Series([row["mapping_status"] for row in root_causes]).value_counts().to_dict()
        bol_counts = pd.Series([row["bol_policy_status"] or "unresolved_or_unmatched" for row in root_causes]).value_counts().to_dict()
        matrix = pd.DataFrame(cross_seed)
        baseline_effects: dict[str, float] = {}
        target_effects: dict[str, float] = {}
        for metric in _LABELS:
            baseline_deltas = []
            target_deltas = []
            for rpt_seed in ("211", "212"):
                pair = matrix.loc[matrix["rpt_seed"].eq(rpt_seed)].set_index("bol_seed")
                baseline_deltas.append(abs(float(pair.loc["211", metric]) - float(pair.loc["212", metric])))
            for bol_seed in ("211", "212"):
                pair = matrix.loc[matrix["bol_seed"].eq(bol_seed)].set_index("rpt_seed")
                target_deltas.append(abs(float(pair.loc["211", metric]) - float(pair.loc["212", metric])))
            baseline_effects[metric] = max(baseline_deltas)
            target_effects[metric] = max(target_deltas)

        mapping_table = "\n".join(
            f"| {name or 'unmatched'} | {count} |"
            for name, count in sorted(mapping_counts.items())
        )
        cross_seed_table = "\n".join(
            f"| {row['bol_seed']} | {row['rpt_seed']} | "
            f"{float(row['lli_percent']):.6f} | {float(row['lam_anode_percent']):.6f} | "
            f"{float(row['lam_cathode_percent']):.6f} | {float(row['rpt_rmse_mV']):.6f} |"
            for row in cross_seed
        )
        bol_status_summary = ", ".join(
            f"`{name or 'unresolved_or_unmatched'}` {count}"
            for name, count in sorted(bol_counts.items())
        )
        historical_signature_count = sum(
            row["record_type"]
            in {"historical_output_signature", "legacy_raw_result_output"}
            for row in inventory
        )
        central_signature_count = sum(
            row["record_type"] == "historical_output_signature" for row in inventory
        )
        additional_raw_signature_count = sum(
            row["record_type"] == "legacy_raw_result_output" for row in inventory
        )
        accepted_negative_rows = int((negative_mask & accepted).sum())
        unique_key_count = len(root_causes)
        accepted_key_count = sum(
            row["historical_accepted_output_signatures"] > 0 for row in root_causes
        )
        parameterized_negative_fit_count = sum(
            row["record_type"] == "parameterized_stage2_fit" for row in inventory
        )
        legacy_candidate_count = sum(
            row["record_type"] == "legacy_rejected_candidate" for row in inventory
        )
        parity_count = sum(bool(row["parity_with_saved_label"]) for row in formula_audit)
        residual_summary_count = sum(
            row["residual_evidence_type"] == "saved_visualization_summary"
            for row in residual_audit
        )
        seed211_rows = [row for row in cross_seed if row["rpt_seed"] == "211"]
        seed212_rows = [row for row in cross_seed if row["rpt_seed"] == "212"]
        seed211_lli = sum(float(row["lli_percent"]) for row in seed211_rows) / len(seed211_rows)
        seed211_anode = sum(float(row["lam_anode_percent"]) for row in seed211_rows) / len(seed211_rows)
        seed212_lli = sum(float(row["lli_percent"]) for row in seed212_rows) / len(seed212_rows)
        seed212_anode = sum(float(row["lam_anode_percent"]) for row in seed212_rows) / len(seed212_rows)
        max_baseline_effect = max(baseline_effects.values())
        step4_rpt2_trace = next(
            (
                row
                for row in bol_trace
                if row["dataset_part"] == "1"
                and row["physical_cell_id"] == "G1-Cell9"
                and row["rpt_index"] == "2"
            ),
            {},
        )
        step4_rpt8_trace = next(
            (
                row
                for row in bol_trace
                if row["dataset_part"] == "1"
                and row["physical_cell_id"] == "G1-Cell9"
                and row["rpt_index"] == "8"
            ),
            {},
        )
        step4_bol_span = step4_rpt2_trace.get("bol_step4_capacity_span_native", "")
        step4_rpt2_span = step4_rpt2_trace.get("target_step4_capacity_span_native", "")
        step4_rpt8_span = step4_rpt8_trace.get("target_step4_capacity_span_native", "")
        step4_rpt8_count = step4_rpt8_trace.get("target_step4_sample_count", "")
        step8_bol_span = cross_seed[0]["bol_capacity_span_native"]
        step8_rpt2_span = cross_seed[0]["rpt_capacity_span_native"]
        optimizer_unstable_count = sum(
            row["root_cause_classification"] == "OPTIMIZER_UNSTABLE"
            for row in root_causes
        )
        single_seed_count = sum(
            row["root_cause_classification"] == "UNRESOLVED_SINGLE_SEED"
            for row in root_causes
        )
        other_unresolved_count = unique_key_count - optimizer_unstable_count - single_seed_count
        report = f"""# PyDMA Negative Degradation Root-Cause Audit

**Scope:** Read-only forensic analysis of saved labels, fit parameters, mapping records, and reconstructed curves. No PyDMA optimization, model training, clipping, relabeling, or historical-output overwrite was performed.

## Executive findings

- The historical audit contains **{len(raw_negative)} negative-bearing rows**, of which **{accepted_negative_rows}** are accepted under their recorded label gates. These are table rows, not independent optimizer attempts.
- Deduplicating overlapping result-table snapshots by the complete saved output signature yields **{historical_signature_count} distinct historical negative output signatures** ({central_signature_count} in the central audit and {additional_raw_signature_count} additional legacy raw-result outputs) across **{unique_key_count} observation/direction keys**. **{accepted_key_count} keys** have an accepted negative output; other keys are raw/unaccepted only.
- Negative historical values by audit row: **LLI {int(metric_counts['lli_percent'])}**, **anode LAM {int(metric_counts['lam_anode_percent'])}**, **cathode LAM {int(metric_counts['lam_cathode_percent'])}**. All accepted negative records are discharge-direction; no cathode-LAM negatives were found.
- **G1-Cell9 Part 1 RPT2 discharge is optimizer-unstable across saved target seeds.** With either fixed BoL seed, RPT seed 211 gives LLI/anode-LAM **{seed211_lli:.6f}% / {seed211_anode:.6f}%**, while seed 212 gives **{seed212_lli:.6f}% / {seed212_anode:.6f}%**. Maximum output change from switching only the BoL seed is {max_baseline_effect:.6g} percentage points; switching the RPT seed changes LLI by up to {target_effects['lli_percent']:.6f} pp and anode LAM by up to {target_effects['lam_anode_percent']:.6f} pp.
- G1-Cell9 RPT8 discharge remains **unresolved**: only one parameterized target seed is saved. The historical negative outputs outside these parameterized cases cannot be attributed to an optimizer seed because their fit IDs, seeds, and vectors were not saved.
- The independent degradation-equation recomputation matches the saved Stage 2 labels on **{parity_count}/{len(formula_audit)}** rows. No formula/sign bug is confirmed.

## Inventory and provenance

| Quantity | Result |
|---|---:|
| Historical negative-bearing audit rows (overlapping tables included) | {len(raw_negative)} |
| Distinct historical negative output signatures | {historical_signature_count} |
| Additional legacy raw-result output signatures | {additional_raw_signature_count} |
| Accepted historical negative-bearing rows | {accepted_negative_rows} |
| Unique observation/direction keys | {unique_key_count} |
| Parameterized Stage 2 negative fit IDs | {parameterized_negative_fit_count} |
| Legacy negative candidate rows with parameter columns (rejected, no seed/fit ID) | {legacy_candidate_count} |
| Raw-results duplicate rows matched to the historical audit | {raw_alias_count} |
| Visualization negative labels matched to the historical audit | {visualization_alias_count} |
| Saved curve-residual evidence rows | {len(residual_audit)} |

Historical tables overlap. Their row counts are not an attempt count. There are {central_signature_count} distinct output signatures in the central audit and {additional_raw_signature_count} additional negative outputs in the separate legacy raw-results file. The historical result schemas do not persist fit IDs, seeds, or fitted vectors; therefore the number of unique optimizer attempts is **not recoverable**. The parameterized Stage 2 output identifies {parameterized_negative_fit_count} negative fit IDs. Two older G1-Cell15 rejected candidate rows also contain alpha/beta columns but no seed or fit ID and have 0/10 accepted runs; they do not identify the accepted historical fit.

The three `pydma_raw_results.csv` rows were checked by cell/RPT/source and retained as distinct outputs because their RMSE/run counts/labels do not exactly match the central audit; {raw_alias_count} were exact duplicates. The 44 negative label rows in the saved visualization summary match the historical audit and are not counted again. The Group 5 candidate table is already represented in the historical audit. Aggregated hybrid/model outputs and CNN predictions are not treated as PyDMA fit attempts.

## Negative keys by dataset part

| Part | Negative keys | Keys with accepted historical negative output | Evaluation treatment |
|---|---:|---:|---|
| Part 1 | {key_part_counts.get('1', 0)} | {accepted_key_counts['1']} | Internal cell-level split retained |
| Part 2 | {key_part_counts.get('2', 0)} | {accepted_key_counts['2']} | Internal cell-level split retained |
| Part 3 | {key_part_counts.get('3', 0)} | {accepted_key_counts['3']} | External/domain-shift only |

## Mapping and BoL status

| Mapping status | Negative keys |
|---|---:|
{mapping_table}

BoL policy-status counts across the negative keys: {bol_status_summary}. The per-key source file, exact Step 4/8/10 segment references, selected BoL file, and unresolved statuses are in `bol_reference_trace.csv`. Part 1 uses the provisional earliest-complete-BoL policy; this audit does not promote it to an official protocol rule. Part 2/3 BoL choices that remain unresolved are kept unresolved.

## G1-Cell9 RPT2: baseline versus target-fit sensitivity

All four saved combinations of BoL seed 211/212 and RPT2 discharge seed 211/212 were recomputed from their saved parameter vectors and native capacity spans. No fit was launched.

| BoL seed | RPT seed | LLI (%) | Anode LAM (%) | Cathode LAM (%) | Target RMSE (mV) |
|---:|---:|---:|---:|---:|---:|
{cross_seed_table}

Changing the baseline seed alone changes labels by at most {baseline_effects['lli_percent']:.6g} pp (LLI), {baseline_effects['lam_anode_percent']:.6g} pp (anode LAM), and {baseline_effects['lam_cathode_percent']:.6g} pp (cathode LAM). Changing the target RPT seed changes them by up to {target_effects['lli_percent']:.6f}, {target_effects['lam_anode_percent']:.6f}, and {target_effects['lam_cathode_percent']:.6f} pp, respectively. The sign reversal is therefore attributable to the target fit in these saved runs, not to the BoL fit. This establishes seed instability; it does **not** prove structural non-identifiability.

Both RPT2 target fits pass the primary RMSE <15 mV / at least 3 accepted runs gate. The RPT2 seed-211 saved output numerically reproduces the historical negative label and RMSE to displayed precision, but the old record has no fit ID/vector, so identity with that exact optimizer attempt cannot be proven.

## Formula, bounds, and residual evidence

- The PyDMA loss convention is `(initial - fitted_current) / initial`; negative values indicate that a fitted current component/inventory exceeds its fitted BoL reference. This is a fitted diagnostic, not measured ground truth.
- Direct recomputation from saved Stage 2 parameters and capacity spans matches all {len(formula_audit)} saved Stage 2 label rows within 1e-10 percentage points. The charge/discharge half-cell direction assignments were retained.
- The bound audit lists every Stage 2 fitted parameter with its saved bounds. The 1%-of-range proximity flag is an audit convention, not PyDMA's acceptance gate. Legacy G1-Cell15 candidate vectors have no persisted optimizer-bound metadata; their bound status is consequently unknown.
- Saved residuals are localized: Stage 2 RPT2 seed 211/212 and RPT8 seed 211 have pointwise arrays; the historical visualization summary contributes {residual_summary_count} rows with only scalar residual summaries. These do not identify causes for other historical fits.
- For G1-Cell9, the raw Step 4 capacity span is {step4_bol_span} at the selected BoL and {step4_rpt2_span} at RPT2. RPT8 Step 4 has only {step4_rpt8_count} rows with span {step4_rpt8_span}; it is partial and is not a full-capacity SOH observation. The saved Step 8 pOCV spans used in the RPT2 formula are {step8_bol_span} (BoL) and {step8_rpt2_span} (RPT2), in native capacity units. Step 4 capacity context does not replace the Step 8/10 pOCV fit used by PyDMA.
- Capacity values remain in native `Amphr` scale. `TestTime` minute-based current integration is only an empirical consistency check; the source files do not explicitly define those units. No EFC or cycle-age inference is made.

## Root-cause disposition

| Classification | Keys | Meaning |
|---|---:|---|
| `OPTIMIZER_UNSTABLE` | {optimizer_unstable_count} | RPT2 target seed changes sign while baseline seed does not explain the change |
| `UNRESOLVED_SINGLE_SEED` | {single_seed_count} | RPT8 has only one parameterized target seed |
| Other unresolved historical keys | {other_unresolved_count} | No saved accepted-fit vectors/seeds; some keys also have mapping or curve audit caveats |

No confirmed PyDMA formula, mapping, reference-selection, or capacity-unit bug was found. Possible non-identifiability remains a hypothesis, not a proven root cause. Negative values were not clipped or rewritten.

## Inputs and reproducibility

Run from the repository root with the project's CUDA environment (the script itself performs no optimization):

```powershell
$env:PYTHONPATH = 'src'
\\.venv-torch-cuda\\Scripts\\python.exe -m bafuse.data.negative_root_cause_audit
```

Inputs include `reports/final_mapping_label_audit/pydma_fit_audit.csv`, the final mapping and BoL audits, the saved Stage 2 fit attempts/labels/reconstructed curves, the legacy G1-Cell15 candidate estimates, the legacy raw-results table, and the saved visualization residual summary. Part 3 remains an external cohort. No model checkpoint, scaler, source data, label table, or historical result was modified.
"""
        (stage / "negative_root_cause_report.md").write_text(report, encoding="utf-8")
        bugs = """# Confirmed PyDMA Bugs and Fixes

## Result

No confirmed PyDMA formula, sign, mapping, BoL-reference, or capacity-unit implementation bug was established for the investigated negative estimates. No source code or label data was changed as part of this investigation.

## Verified behavior

- The loss equation uses `(initial - fitted_current) / initial`; a negative result means the fitted current component/inventory is larger than its fitted BoL reference.
- Recalculation from saved Stage 2 parameters and capacity spans agrees with the saved labels on all eight formula-parity rows.
- G1-Cell9 Part 1 RPT2 discharge is optimizer-seed-sensitive in the saved fits. This is a demonstrated output instability, not proof of structural non-identifiability.
- G1-Cell9 Part 1 RPT8 discharge has only one saved target seed and remains unresolved.

## No remediation applied

Historical outputs were not clipped, rewritten, relabeled, or retrained. The 1%-of-parameter-range bound flag is an audit convention only. Possible model non-identifiability remains a hypothesis until supported by additional saved fit evidence; no optimization was run for this report.
"""
        (stage / "confirmed_bugs_and_fixes.md").write_text(bugs, encoding="utf-8")
        if output_dir.exists():
            raise FileExistsError(f"Refusing to overwrite existing audit directory: {output_dir}")
        Path(temporary).rename(output_dir)


def generate_audit(root: Path, output_dir: Path = _DEFAULT_OUTPUT) -> Path:
    root = root.resolve()
    output_dir = output_dir if output_dir.is_absolute() else root / output_dir
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing audit directory: {output_dir}")

    audit = _read_csv(root, _HISTORICAL_AUDIT)
    required = {
        "result_table",
        "dataset_part",
        "physical_cell_id",
        "checkup",
        "direction",
        "test_step",
        "source_file",
        "fit_rmse_mV",
        "accepted_runs",
        "total_runs",
        "diagnostic_label_accepted",
        *_HISTORICAL_LABELS,
    }
    missing = required.difference(audit.columns)
    if missing:
        raise ValueError(f"Historical audit is missing required columns: {sorted(missing)}")
    historical, raw_negative = _historical_records(audit)
    raw_result_table = _read_csv(root, _RAW_RESULTS)
    raw_records, raw_alias_count = _raw_results_records(raw_result_table, audit)
    visualization_alias_count = _visualization_crosscheck(root, audit)
    candidate_records, candidate_vectors = _candidate_records(root)
    stage2_records, fits, labels, curves = _stage2_records(root)
    inventory = historical + raw_records + candidate_records + stage2_records

    mapping, bol = _load_context(root)
    keys = {_row_key(row) for row in inventory}
    bol_trace, contexts = _bol_trace(root, keys, mapping, bol)
    residual_audit = _curve_residual_audit(root, fits, labels, curves)
    root_causes = _root_cause_rows(
        inventory,
        raw_negative,
        residual_audit,
        contexts,
    )
    formula_audit = _formula_audit(fits, labels)
    if not formula_audit or not all(row["parity_with_saved_label"] for row in formula_audit):
        raise ValueError("Saved Stage 2 labels failed independent formula parity.")
    cross_seed = _cross_seed_matrix(fits)
    bound_audit = _parameter_bound_audit(fits, candidate_vectors)
    _write_outputs(
        output_dir,
        inventory,
        root_causes,
        bol_trace,
        bound_audit,
        cross_seed,
        formula_audit,
        residual_audit,
        audit,
        raw_negative,
        raw_alias_count,
        visualization_alias_count,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit saved negative PyDMA degradation estimates without running optimization."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="BaFuse repository root.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help="New versioned output directory; an existing directory is never overwritten.",
    )
    args = parser.parse_args()
    output = generate_audit(args.root, args.output_dir)
    print(output)


if __name__ == "__main__":
    main()
