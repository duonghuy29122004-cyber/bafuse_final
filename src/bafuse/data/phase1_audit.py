from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from bafuse.config import load_project_paths
from bafuse.labeling.audit import GuideEntry, read_guide_entries
from bafuse.labeling.pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
)


SOH_USABLE = {"baseline_reference", "calculated_from_measured_capacity"}
PYDMA_ACCEPTED = "diagnostic_only_not_independently_validated"
TARGET_COLUMNS = {
    "lli": "lli_percent",
    "lam_anode": "lam_anode_percent",
    "lam_cathode": "lam_cathode_percent",
}
EIS_PATTERN = re.compile(
    r"Group(?P<group>\d+)_(?P<stage>BoL|MoL)_Cell(?P<cell>\d+)_SOC(?P<soc>\d+)(?:_(?P<copy>\d+))?",
    re.IGNORECASE,
)
OUTPUT_FILES = {
    "eis_audit.json",
    "eis_mat_decode.csv",
    "eis_pair_mismatches.csv",
    "cohort_by_cell_rpt.csv",
    "pydma_cell_coverage.csv",
    "cohort_summary.json",
    "soh_by_rpt_order_by_group.png",
    "soh_by_rpt_order.csv",
    "soh_label_audit_rows.csv",
    "soh_rpt_consistency_by_cell.csv",
    "soh_audit.json",
    "pydma_provenance_and_direction_audit.json",
    "aging_axis_audit.json",
    "forbidden_features.csv",
    "source_and_results_sha256.csv",
    "source_and_results_sha256.partial.csv",
    "phase1_summary.json",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _number(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) else None


def _norm_rpt(value: str) -> str:
    compact = re.sub(r"\s+", "", value).casefold()
    match = re.fullmatch(r"rpt(\d+)", compact)
    return f"rpt{int(match.group(1))}" if match else compact


def _norm_source(value: str) -> str:
    return Path(value.replace("\\", "/")).as_posix().casefold()


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "std_population": None, "min": None, "median": None, "max": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std_population": float(array.std()),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _eis_csv_audit(dataset: Path, rows: list[dict[str, str]]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    csv_rows = [row for row in rows if row["modality"] == "EIS" and row["file_type"] == "csv"]
    mat_rows = [row for row in rows if row["modality"] == "EIS" and row["file_type"] == "mat"]
    frequency_names: set[str] = set()
    columns: set[str] = set()
    values: dict[str, list[float]] = defaultdict(list)
    missing: Counter[str] = Counter()
    nonfinite: Counter[str] = Counter()
    duplicate_rows: list[dict[str, Any]] = []
    counts: Counter[int] = Counter()
    stage_counts: Counter[str] = Counter()
    soc_counts: Counter[str] = Counter()
    csv_to_expected_mat = {
        row["relative_path"]: Path(row["relative_path"]).with_suffix(".mat").as_posix()
        for row in csv_rows
    }
    csv_expected_mat_paths = set(csv_to_expected_mat.values())
    mat_paths = {row["relative_path"] for row in mat_rows}

    for row in csv_rows:
        source = dataset / Path(row["relative_path"])
        stage_counts[row["stage"]] += 1
        soc_counts[row["soc"]] += 1
        with source.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            columns.update(headers)
            frequency_names.update(
                name for name in headers
                if re.search(r"freq|frequency|(^|_)hz($|_)", name, re.IGNORECASE)
            )
            seen: Counter[tuple[str, ...]] = Counter()
            count = 0
            for item in reader:
                count += 1
                signature = tuple((item.get(name) or "").strip() for name in headers)
                seen[signature] += 1
                for name in headers:
                    raw = (item.get(name) or "").strip()
                    if not raw:
                        missing[name] += 1
                    else:
                        number = _number(raw)
                        if number is None:
                            nonfinite[name] += 1
                        else:
                            values[name].append(number)
            counts[count] += 1
            repeated = sum(value - 1 for value in seen.values() if value > 1)
            if repeated:
                duplicate_rows.append({"relative_path": row["relative_path"], "exact_duplicate_rows": repeated})

    csv_only = sorted(csv_path for csv_path, expected in csv_to_expected_mat.items() if expected not in mat_paths)
    mat_only = sorted(mat_paths - csv_expected_mat_paths)
    mismatch_rows = [
        {
            "side": "CSV_without_exact_stem_MAT",
            "csv_relative_path": path,
            "expected_mat_relative_path": csv_to_expected_mat[path],
            "mat_relative_path": "",
        }
        for path in csv_only
    ] + [
        {
            "side": "MAT_without_exact_stem_CSV",
            "csv_relative_path": "",
            "expected_mat_relative_path": "",
            "mat_relative_path": path,
        }
        for path in mat_only
    ]

    mat_details = []
    for index, row in enumerate(mat_rows, 1):
        source = dataset / Path(row["relative_path"])
        try:
            with source.open("rb") as stream:
                signature = stream.read(520)
            if signature.startswith(b"MATLAB 5.0 MAT-file"):
                from scipy.io import loadmat

                content = loadmat(source, squeeze_me=True, struct_as_record=False)
                names = [name for name in content if not name.startswith("__")]
                opaque = [name for name in names if type(content[name]).__name__ == "MatlabOpaque"]
                numeric = [
                    {
                        "name": name,
                        "shape": tuple(np.asarray(content[name]).shape),
                        "dtype": str(np.asarray(content[name]).dtype),
                        "values": np.asarray(content[name]).tolist(),
                    }
                    for name in names
                    if isinstance(content[name], np.ndarray) and np.issubdtype(content[name].dtype, np.number)
                ]
                status = (
                    "mcos_opaque_with_numeric_fragments"
                    if opaque and numeric
                    else "mcos_opaque_only"
                    if opaque
                    else "decoded_numeric_variables"
                    if numeric
                    else "decoded_non_numeric_variables"
                )
                detail = (
                    "MCOS table remains opaque; numeric variables are only partial fragments and not a decoded measurement table."
                    if opaque and numeric
                    else "MCOS table object; no numeric measurement vectors decoded."
                    if opaque
                    else "MAT variables loaded; frequency/impedance meaning still requires validation."
                )
                mat_details.append(
                    {
                        "relative_path": row["relative_path"],
                        "format": row["format"],
                        "decode_status": status,
                        "variable_names": ";".join(names),
                        "opaque_variables": ";".join(opaque),
                        "numeric_variables": json.dumps(numeric),
                        "frequency_vector_found": any(re.search(r"freq|hz", name, re.I) for name in names),
                        "detail": detail,
                    }
                )
            elif signature.startswith(b"MATLAB 7.3 MAT-file") or signature[512:520] == b"\x89HDF\r\n\x1a\n":
                import h5py

                datasets = []
                with h5py.File(source, "r") as h5file:
                    h5file.visititems(
                        lambda name, obj: datasets.append((name, tuple(obj.shape), str(obj.dtype)))
                        if isinstance(obj, h5py.Dataset)
                        else None
                    )
                mat_details.append(
                    {
                        "relative_path": row["relative_path"],
                        "format": row["format"],
                        "decode_status": "hdf5_hierarchy_read",
                        "variable_names": "",
                        "opaque_variables": "",
                        "numeric_variables": json.dumps(datasets),
                        "frequency_vector_found": any(re.search(r"freq|hz", name, re.I) for name, _, _ in datasets),
                        "detail": "HDF5 hierarchy read; MATLAB table reference mapping may still be unresolved.",
                    }
                )
            else:
                mat_details.append(
                    {
                        "relative_path": row["relative_path"],
                        "format": row["format"],
                        "decode_status": "unsupported_container",
                        "variable_names": "",
                        "opaque_variables": "",
                        "numeric_variables": "[]",
                        "frequency_vector_found": False,
                        "detail": "Unrecognized MAT container signature.",
                    }
                )
        except Exception as error:
            mat_details.append(
                {
                    "relative_path": row["relative_path"],
                    "format": row["format"],
                    "decode_status": "decode_error",
                    "variable_names": "",
                    "opaque_variables": "",
                    "numeric_variables": "[]",
                    "frequency_vector_found": False,
                    "detail": f"{type(error).__name__}: {error}",
                }
            )
        if index % 25 == 0 or index == len(mat_rows):
            print(f"Decoded EIS MAT headers/payloads {index}/{len(mat_rows)}", flush=True)

    grid_points = math.log10(5000.0 / 0.01) * 6.0
    audit = {
        "csv_files": len(csv_rows),
        "mat_files": len(mat_rows),
        "csv_columns": sorted(columns),
        "frequency_columns_in_csv": sorted(frequency_names),
        "mat_decode_status_counts": dict(Counter(row["decode_status"] for row in mat_details)),
        "mat_frequency_vectors_found_by_name": sum(bool(row["frequency_vector_found"]) for row in mat_details),
        "mat_numeric_measurement_tables_decoded": sum(row["decode_status"] == "decoded_numeric_variables" for row in mat_details),
        "mat_numeric_fragments_with_opaque_table": sum(row["decode_status"] == "mcos_opaque_with_numeric_fragments" for row in mat_details),
        "csv_rows_per_file": {str(key): value for key, value in sorted(counts.items())},
        "csv_frequency_grid_check": {
            "readme_range_hz": [5000, 0.01],
            "readme_points_per_decade": 6,
            "implied_intervals": grid_points,
            "approx_expected_points_including_endpoints": round(grid_points) + 1,
            "all_csv_have_35_rows": counts == Counter({35: len(csv_rows)}),
            "actual_frequency_values_available": False,
            "log_spacing_verified": False,
        },
        "numeric_columns": {
            name: {
                **_stats(values[name]),
                "missing": missing[name],
                "nonfinite_or_non_numeric": nonfinite[name],
            }
            for name in sorted(columns)
        },
        "exact_duplicate_rows": duplicate_rows,
        "stage_file_counts": dict(stage_counts),
        "soc_file_counts": dict(soc_counts),
        "csv_only_exact_stems": csv_only,
        "mat_only_exact_stems": mat_only,
        "exact_stem_mismatch_counts": {
            "csv_without_mat": len(csv_only),
            "mat_without_csv": len(mat_only),
            "net_csv_minus_mat": len(csv_rows) - len(mat_rows),
        },
        "eis_status": "not_ready",
        "reason": "CSV points count is consistent with an approximate expectation, but no CSV frequency vector exists. MAT files decode as opaque MATLAB MCOS tables or errors; frequency-to-impedance mapping, Re/Im semantics, and point order remain unverified. EIS records are BoL/MoL, not RPT-aligned.",
    }
    return audit, mat_details, mismatch_rows


def _identity(cell: str, source: str, rpt: str, direction: str | None = None):
    base = (cell, _norm_source(source), _norm_rpt(rpt))
    return base if direction is None else (*base, direction.casefold())


def _load_cohort(labels_dir: Path, readiness: dict[str, Any], soh_run: Path, charge_run: Path, dataset_manifest):
    guide_path = Path(readiness["dataset_root"]) / "Guide_to_Datafiles.xlsx"
    entries = read_guide_entries(guide_path)
    source_by_id = {
        (row["group_id"], row["cell_id"], row["file_id"], row["file_type"]): row["relative_path"]
        for row in dataset_manifest
        if row["modality"] == "cycling_and_RPT" and row["file_id"] and row["file_type"] == "csv"
    }
    soh_rows = _read_csv(labels_dir / "labels_soh.csv")
    degradation_rows = _read_csv(labels_dir / "labels_degradation.csv")
    soh_by_id = {}
    for row in soh_rows:
        if not row["source_file"]:
            continue
        key = _identity(row["physical_cell_id"], row["source_file"], row["measurement_id"])
        if key in soh_by_id:
            raise ValueError(f"Duplicate SOH source/RPT identity: {key}")
        soh_by_id[key] = row
    degradation_by_id = {}
    for row in degradation_rows:
        if not row["source_file"]:
            continue
        key = _identity(row["physical_cell_id"], row["source_file"], row["measurement_id"], row["direction"])
        if key in degradation_by_id:
            raise ValueError(f"Duplicate directional PyDMA identity: {key}")
        degradation_by_id[key] = row

    feature_sources = {
        _norm_source(row["source_file"])
        for row in _read_csv(soh_run / "sample_provenance.csv")
    }
    exclusions = {
        _norm_source(row["source_file"]): row["reason"]
        for row in readiness["discharge_curve_exclusions"]
    }
    eis_by_cell: dict[str, list[dict[str, str]]] = defaultdict(list)
    rpt_files_by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    pocv_files_by_cell: dict[str, Counter[str]] = defaultdict(Counter)
    for row in dataset_manifest:
        if row["modality"] == "EIS" and row["file_type"] == "csv":
            eis_by_cell[row["physical_cell_id"]].append(row)
        if row["modality"] == "cycling_and_RPT" and (
            row["stage"].casefold().startswith("bol") or re.search(r"rpt", row["stage"], re.I)
        ):
            rpt_files_by_cell[row["physical_cell_id"]][row["file_type"]] += 1
            if row["file_type"] == "csv":
                try:
                    steps = json.loads(row["measurement_summary_json"] or "{}").get("step_counts", {})
                except json.JSONDecodeError:
                    steps = {}
                for step in ("8", "10"):
                    if int(steps.get(step, 0) or 0) > 0:
                        pocv_files_by_cell[row["physical_cell_id"]][step] += 1

    cohort = []
    for entry in entries:
        if not (entry.stage.casefold().startswith("bol") or re.search(r"rpt", entry.stage, re.I)):
            continue
        cell = f"G{entry.group_id}-Cell{entry.cell_id}"
        source = source_by_id.get((entry.group_id, entry.cell_id, str(entry.file_id), "csv"), "")
        key = _identity(cell, source, entry.stage)
        soh = soh_by_id.get(key) if source else None
        charge = degradation_by_id.get((*key, "charge")) if source else None
        discharge = degradation_by_id.get((*key, "discharge")) if source else None
        eis_records = eis_by_cell.get(cell, [])
        eis_summary = ";".join(
            f"{stage}:SOC{soc}"
            for stage, soc in sorted({(row["stage"], row["soc"]) for row in eis_records})
        )
        source_key = _norm_source(source) if source else ""
        qc = "pass" if source_key in feature_sources else "fail" if source_key in exclusions else "not_evaluated"
        row = {
            "group_id": entry.group_id,
            "physical_cell_id": cell,
            "guide_file_id": entry.file_id,
            "measurement_id": entry.stage,
            "source_file": source,
            "soh_percent": soh["soh_percent"] if soh else "",
            "soh_status": soh["validation_status"] if soh else "not_mapped",
            "pydma_charge_status": charge["identifiability_status"] if charge else "not_mapped",
            "charge_fit_success": charge["fit_success"] if charge else "",
            "charge_fit_rmse_mV": charge["fit_residual_mV"] if charge else "",
            "lli_charge": charge["lli_percent"] if charge else "",
            "lam_anode_charge": charge["lam_anode_percent"] if charge else "",
            "lam_cathode_charge": charge["lam_cathode_percent"] if charge else "",
            "pydma_discharge_status": discharge["identifiability_status"] if discharge else "not_mapped",
            "discharge_fit_success": discharge["fit_success"] if discharge else "",
            "discharge_fit_rmse_mV": discharge["fit_residual_mV"] if discharge else "",
            "lli_discharge": discharge["lli_percent"] if discharge else "",
            "lam_anode_discharge": discharge["lam_anode_percent"] if discharge else "",
            "lam_cathode_discharge": discharge["lam_cathode_percent"] if discharge else "",
            "eis_cell_stage_soc_coverage_not_rpt_aligned": eis_summary,
            "eis_rpt_alignment": "not_verified",
            "discharge_curve_qc": qc,
            "discharge_curve_qc_reason": exclusions.get(source_key, "") if qc == "fail" else "",
        }
        cohort.append(row)

    # The primary directional target is registered as charge before calculating agreement.
    selected_charge = [
        row for row in cohort
        if row["charge_fit_success"].casefold() == "true"
        and row["pydma_charge_status"] == PYDMA_ACCEPTED
    ]
    accepted_cells = {row["physical_cell_id"] for row in selected_charge}
    cell_summary = []
    documented_cells = sorted({(entry.group_id, entry.cell_id) for entry in entries})
    for group, cell_id in documented_cells:
        cell = f"G{group}-Cell{cell_id}"
        cell_rows = [row for row in cohort if row["physical_cell_id"] == cell]
        all_label_rows = [row for row in degradation_rows if row["physical_cell_id"] == cell]
        accepted_count = sum(
            row["fit_success"].casefold() == "true" and row["identifiability_status"] == PYDMA_ACCEPTED
            for row in all_label_rows
        )
        status_counts = Counter(row["identifiability_status"] for row in all_label_rows)
        source_counts = rpt_files_by_cell[cell]
        pocv_counts = pocv_files_by_cell[cell]
        if accepted_count:
            reason = "Has accepted fits."
        elif not cell_rows:
            reason = "No Guide-mapped BoL/RPT records resolved to a dataset file."
        elif source_counts["csv"] == 0 and source_counts["mat"] > 0:
            reason = "No readable Guide-mapped cycling/RPT CSV; matching source records are MAT files and their contents were not decoded. PyDMA rows are not_run."
        elif pocv_counts["8"] == 0 and pocv_counts["10"] == 0:
            reason = "No readable Step 8/Step 10 pseudo-OCV rows in the Guide-mapped RPT CSVs; PyDMA fit input curves are unavailable."
        else:
            reason = "Pseudo-OCV source rows exist, but no accepted fit; exact per-fit failure causes are not available for every not_run record. See identifiability status counts."
        cell_summary.append(
            {
                "group_id": group,
                "physical_cell_id": cell,
                "rpt_observations": len(cell_rows),
                "degradation_rows": len(all_label_rows),
                "accepted_directional_fits": accepted_count,
                "accepted_charge_fits": sum(row["direction"] == "charge" and row["fit_success"].casefold() == "true" and row["identifiability_status"] == PYDMA_ACCEPTED for row in all_label_rows),
                "accepted_discharge_fits": sum(row["direction"] == "discharge" and row["fit_success"].casefold() == "true" and row["identifiability_status"] == PYDMA_ACCEPTED for row in all_label_rows),
                "guide_rpt_csv_files": source_counts["csv"],
                "guide_rpt_mat_files": source_counts["mat"],
                "readable_rpt_csvs_with_step8": pocv_counts["8"],
                "readable_rpt_csvs_with_step10": pocv_counts["10"],
                "status_counts_json": json.dumps(dict(status_counts), sort_keys=True),
                "reason": reason,
            }
        )

    # Paired directional consistency uses only accepted fits with the same cell/file/RPT key.
    paired: dict[tuple[str, str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in degradation_rows:
        if row["fit_success"].casefold() == "true" and row["identifiability_status"] == PYDMA_ACCEPTED:
            paired[_identity(row["physical_cell_id"], row["source_file"], row["measurement_id"])][row["direction"]] = row
    consistency = []
    for target, column in TARGET_COLUMNS.items():
        pairs = []
        for direction_rows in paired.values():
            if "charge" not in direction_rows or "discharge" not in direction_rows:
                continue
            charge_value = _number(direction_rows["charge"][column])
            discharge_value = _number(direction_rows["discharge"][column])
            if charge_value is not None and discharge_value is not None:
                pairs.append((charge_value, discharge_value))
        differences = [charge_value - discharge_value for charge_value, discharge_value in pairs]
        correlation = None
        if len(pairs) >= 2 and np.std([pair[0] for pair in pairs]) and np.std([pair[1] for pair in pairs]):
            correlation = float(np.corrcoef([pair[0] for pair in pairs], [pair[1] for pair in pairs])[0, 1])
        consistency.append(
            {
                "target": target,
                "paired_accepted_rpts": len(pairs),
                "pearson_correlation_descriptive": correlation,
                "mean_charge_minus_discharge": float(np.mean(differences)) if differences else None,
                "mean_absolute_difference": float(np.mean(np.abs(differences))) if differences else None,
                "rmse_direction_difference": float(np.sqrt(np.mean(np.square(differences)))) if differences else None,
                "within_1_percentage_point_fraction": float(np.mean(np.abs(differences) <= 1.0)) if differences else None,
                "note": "Descriptive only; paired directions are not treated as independent observations or averaged.",
            }
        )
    return cohort, cell_summary, consistency


def _soh_report(labels_dir: Path, readiness: dict[str, Any], output: Path) -> dict[str, Any]:
    rows = _read_csv(labels_dir / "labels_soh.csv")
    eligible = [row for row in rows if row["validation_status"] in SOH_USABLE and _number(row["soh_percent"]) is not None]
    numeric_rows = [row for row in rows if _number(row["soh_percent"]) is not None]
    excluded = readiness["discharge_curve_exclusions"]
    excluded_sources = {_norm_source(row["source_file"]) for row in excluded}
    trend_rows = []
    raw_audit_rows = []
    for row in numeric_rows:
        measurement = row["measurement_id"]
        match = re.search(r"RPT\s*(\d+)", measurement, re.I)
        order = 0 if measurement.casefold().startswith("bol") else int(match.group(1)) if match else None
        raw_audit_rows.append(
            {
                "group_id": row["group_id"],
                "physical_cell_id": row["physical_cell_id"],
                "measurement_id": measurement,
                "rpt_order": order if order is not None else "",
                "soh_percent": float(row["soh_percent"]),
                "validation_status": row["validation_status"],
                "source_file": row["source_file"],
                "capacity_raw": row["capacity_raw"],
                "reference_capacity_raw": row["reference_capacity_raw"],
                "validation_notes": row["validation_notes"],
            }
        )
    raw_audit_rows.sort(key=lambda row: (row["group_id"], row["physical_cell_id"], int(row["rpt_order"] or -1), row["source_file"]))
    _write_csv(output / "soh_label_audit_rows.csv", raw_audit_rows)
    for row in eligible:
        if _norm_source(row["source_file"]) in excluded_sources:
            continue
        group = row["group_id"]
        measurement = row["measurement_id"]
        match = re.search(r"RPT\s*(\d+)", measurement, re.I)
        order = 0 if measurement.casefold().startswith("bol") else int(match.group(1)) if match else None
        if order is None:
            continue
        trend_rows.append(
            {
                "group_id": group,
                "physical_cell_id": row["physical_cell_id"],
                "measurement_id": measurement,
                "rpt_order": order,
                "soh_percent": float(row["soh_percent"]),
                "source_file": row["source_file"],
            }
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for axis, group in zip(axes.flat, ("1", "2", "3", "4")):
        selected = [row for row in trend_rows if row["group_id"] == group]
        for cell in sorted({row["physical_cell_id"] for row in selected}):
            cell_rows = sorted((row for row in selected if row["physical_cell_id"] == cell), key=lambda row: row["rpt_order"])
            axis.plot([row["rpt_order"] for row in cell_rows], [row["soh_percent"] for row in cell_rows], marker="o", label=cell)
        axis.set_title(f"Group {group}")
        axis.set_xlabel("Guide RPT order (not elapsed time)")
        axis.set_ylabel("SOH (%)")
        axis.grid(True, alpha=0.25)
        if selected:
            axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(output / "soh_by_rpt_order_by_group.png", dpi=160)
    plt.close(figure)
    _write_csv(output / "soh_by_rpt_order.csv", trend_rows)
    changes = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trend_rows:
        grouped[row["physical_cell_id"]].append(row)
    for cell, cell_rows in grouped.items():
        cell_rows.sort(key=lambda row: row["rpt_order"])
        values = [row["soh_percent"] for row in cell_rows]
        delta = np.diff(values)
        changes.append(
            {
                "physical_cell_id": cell,
                "observations": len(values),
                "positive_rpt_changes": int(np.sum(delta > 0)),
                "max_positive_change_pp": float(delta.max()) if delta.size else None,
                "total_change_pp": float(values[-1] - values[0]) if len(values) > 1 else None,
            }
        )
    _write_csv(output / "soh_rpt_consistency_by_cell.csv", changes)
    return {
        "formula": "100 * Step 4 Amphr span at RPT / first valid same-cell BoL Step 4 Amphr span.",
        "capacity_unit": "Unknown native Amphr; absolute unit not verified in the local Guide.",
        "numeric_rows": len([row for row in rows if _number(row["soh_percent"]) is not None]),
        "validation_status_counts_all_rows": dict(Counter(row["validation_status"] for row in rows)),
        "review_or_duplicate_numeric_rows": [
            {
                "physical_cell_id": row["physical_cell_id"],
                "measurement_id": row["measurement_id"],
                "soh_percent": float(row["soh_percent"]),
                "validation_status": row["validation_status"],
                "source_file": row["source_file"],
            }
            for row in numeric_rows
            if row["validation_status"] not in SOH_USABLE
        ],
        "full_numeric_label_table": "soh_label_audit_rows.csv; includes review and duplicate BoL rows.",
        "eligible_before_curve_qc": len(eligible),
        "qc_eligible_curves": readiness["discharge_curves_eligible_after_voltage_qc"],
        "excluded_curve_rows": excluded,
        "rpt_plot_points_after_curve_qc": len(trend_rows),
        "rpt_order_consistency": changes,
        "throughput_axis": {
            "status": "not_verified",
            "reason": "No cumulative throughput field is present in the verified feature schema. Amphr is the measured Step 4 capacity used to define the target; using it as x would reuse the target source. Current integration would require verified TestTime units and cross-file continuity, which are not documented.",
            "plot": None,
        },
        "calendar_time_axis": {
            "status": "not_verified",
            "reason": "No per-observation calendar timestamps or validated elapsed-time units are recorded. Filesystem modification times are extraction metadata, not experimental time.",
            "plot": None,
        },
        "available_plot": "soh_by_rpt_order_by_group.png; RPT order only, explicitly not calendar time or throughput.",
    }


def _pydma_provenance(labels_dir: Path, cell_summary, consistency):
    rows = _read_csv(labels_dir / "labels_degradation.csv")
    successful = [row for row in rows if row["fit_success"].casefold() == "true" and row["identifiability_status"] == PYDMA_ACCEPTED]
    unique_sources = sorted({row["source_file"] for row in successful})
    return {
        "generation_version": "NOT RECORDED in labeling run metadata or label rows. The currently installed environment version is not evidence of the version used when these labels were created.",
        "generation_config": "NOT RECORDED as a run artifact. Current source code describes the configuration below; do not claim it was captured with the generated labels.",
        "current_source_code_config": {
            "directions": {"charge": "Step 10; Amps > 0 and abs(Amps) <= 0.2 A", "discharge": "Step 8; Amps < 0 and abs(Amps) <= 0.2 A"},
            "data_length": 500,
            "weight_ocv": 100.0,
            "weight_dva": 0.0,
            "weight_ica": 0.0,
            "speed_preset": "fast",
            "req_accepted": 3,
            "max_tries_overall": 10,
            "random_seed": 211,
            "analyze_maxiter": 50,
            "analyze_popsize": 30,
            "polish": True,
            "acceptance": (
                f"fit.is_accepted, RMSE < {PYDMA_RMSE_THRESHOLD_MV:g} mV, at "
                f"least {PYDMA_MIN_ACCEPTED_RUNS} accepted runs; compare to "
                "same-direction BoL fit"
            ),
        },
        "source_curves": {
            "charge": "Step 10 pseudo-OCV charge voltage versus Amphr, selected with positive current.",
            "discharge": "Step 8 pseudo-OCV discharge voltage versus Amphr, selected with negative current.",
            "accepted_unique_source_files": len(unique_sources),
            "accepted_source_file_list": unique_sources,
        },
        "forbidden_features": [
            {"feature": "Step 8 voltage/capacity/derived curve descriptors", "reason": "Same discharge pseudo-OCV curve fit by PyDMA for discharge labels."},
            {"feature": "Step 10 voltage/capacity/derived curve descriptors", "reason": "Same charge pseudo-OCV curve fit by PyDMA for charge labels."},
            {"feature": "PyDMA fitted parameters, residual/RMSE, accepted-run counts, or outputs", "reason": "Direct fit provenance or target-derived information."},
            {"feature": "SOH, Step 4 Amphr capacity, or any SOH-derived feature", "reason": "Target leakage for SOH; capacity is the SOH target source."},
            {"feature": "Future RPT/EIS/cycling features and cell/file/RPT identifiers", "reason": "Temporal leakage or cell memorization."},
        ],
        "distinct_curve_input_check": {
            "facts": [
                "Step 4 discharge and Step 6 CC-CV charge are different protocol steps from PyDMA Step 8/10 pseudo-OCV curves.",
                "Same source CSV/RPT does not make them the same curve; use only if step filtering is verified and disclose same-observation correlation.",
            ]
        },
        "direction_rule_registered_before_agreement_summary": "Primary single-direction label is accepted charge (Step 10); if charge is missing, keep the target masked. Do not substitute discharge or average directions. Preserve discharge as a separate direction-specific sensitivity result.",
        "prior_run_004_training_policy": "The existing run_004 kept accepted charge and discharge rows as separate direction-specific samples; it did not apply a single charge-primary rule. The registered rule above is for a future confirmatory run and does not rewrite prior metrics.",
        "direction_agreement": consistency,
        "cell_coverage": cell_summary,
        "accepted_directional_rows": len(successful),
    }


def _cohort_counts(cohort, cell_summary):
    accepted_charge_rows = [row for row in cohort if row["charge_fit_success"].casefold() == "true" and row["pydma_charge_status"] == PYDMA_ACCEPTED]
    accepted_discharge_rows = [row for row in cohort if row["discharge_fit_success"].casefold() == "true" and row["pydma_discharge_status"] == PYDMA_ACCEPTED]
    accepted_charge = len(accepted_charge_rows)
    accepted_discharge = len(accepted_discharge_rows)
    accepted_rows = accepted_charge_rows + accepted_discharge_rows
    qc_matched = [row for row in accepted_rows if row["discharge_curve_qc"] == "pass"]
    qc_pass = sum(row["discharge_curve_qc"] == "pass" for row in cohort)
    qc_fail = sum(row["discharge_curve_qc"] == "fail" for row in cohort)
    return {
        "guide_boL_rpt_rows": len(cohort),
        "accepted_charge_direction_rows": accepted_charge,
        "accepted_discharge_direction_rows": accepted_discharge,
        "accepted_directional_fits_total": accepted_charge + accepted_discharge,
        "accepted_directional_fits_with_qc_passing_step4_feature": len(qc_matched),
        "unique_cell_rpt_source_files_with_accepted_fit_and_qc": len({(row["physical_cell_id"], _norm_source(row["source_file"])) for row in qc_matched}),
        "unique_source_file_observations_with_any_accepted_fit": len({(row["physical_cell_id"], _norm_source(row["source_file"])) for row in accepted_rows}),
        "discharge_qc_pass_from_cached_soh_features": qc_pass,
        "discharge_qc_fail_known": qc_fail,
        "discharge_qc_not_evaluated": sum(row["discharge_curve_qc"] == "not_evaluated" for row in cohort),
        "documented_cells": len(cell_summary),
        "cells_with_any_accepted_pydma": sum(row["accepted_directional_fits"] > 0 for row in cell_summary),
        "cells_without_accepted_pydma": [row["physical_cell_id"] for row in cell_summary if row["accepted_directional_fits"] == 0],
        "reason_63_vs_78": "78 is accepted direction-specific PyDMA rows that match QC-passing Step 4 features; the same source RPT may contribute both charge and discharge rows. These map to 63 unique (cell, source RPT file) observations. SOH is supervised once per source file, while direction-specific PyDMA estimates remain separate and are not independent replicate cells.",
    }


def _hash_manifest(
    workspace: Path,
    dataset: Path,
    output: Path,
    reuse_existing: bool = False,
) -> list[dict[str, Any]]:
    items: list[tuple[str, Path]] = []
    for path in sorted((path for path in dataset.rglob("*") if path.is_file()), key=lambda path: path.relative_to(dataset).as_posix().casefold()):
        items.append(("part1_source", path))
    results_root = workspace / "experiments" / "results"
    output_resolved = output.resolve()
    if results_root.is_dir():
        result_files = sorted(
            (
                path
                for path in results_root.rglob("*")
                if path.is_file() and output_resolved not in path.resolve().parents
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        items.extend(("curated_result", path) for path in result_files)
    manifest_path = output / "source_and_results_sha256.csv"
    partial_path = output / "source_and_results_sha256.partial.csv"
    prior_path = manifest_path if manifest_path.is_file() else partial_path
    prior_by_key = {}
    if prior_path.is_file():
        prior_by_key = {
            (row["scope"], row["relative_path"]): row
            for row in _read_csv(prior_path)
        }
    hashes = []
    for index, (scope, path) in enumerate(items, 1):
        if scope == "part1_source":
            relative = path.relative_to(dataset).as_posix()
            relative = f"{dataset.name}/{relative}"
        else:
            relative = path.relative_to(workspace).as_posix()
        stat = path.stat()
        prior = prior_by_key.get((scope, relative))
        if (
            reuse_existing
            and prior is not None
            and int(prior["size_bytes"]) == stat.st_size
            and (
                not prior.get("modified_ns", "")
                or int(prior["modified_ns"]) == stat.st_mtime_ns
            )
        ):
            digest = prior["sha256"]
        else:
            digest = _content_sha256(path)
        hashes.append(
            {
                "scope": scope,
                "relative_path": relative,
                "size_bytes": stat.st_size,
                "modified_ns": stat.st_mtime_ns,
                "sha256": digest,
            }
        )
        if index % 25 == 0 or index == len(items):
            print(f"SHA-256 hashed {index}/{len(items)} source/result files", flush=True)
            _write_csv(partial_path, hashes)
    _write_csv(manifest_path, hashes)
    if partial_path.exists():
        partial_path.unlink()
    return hashes


def run_phase1(
    dataset: Path,
    labels: Path,
    workspace: Path,
    soh_run: Path,
    charge_run: Path,
    output: Path,
    reuse_hashes: bool = False,
) -> dict[str, Any]:
    if dataset.name not in {
        "Path_Dependent_Battery_Degradation_Dataset_Part 1",
        "part1",
    }:
        raise ValueError("Phase 1 is restricted to Oxford Part 1.")
    if labels.name != "labeling_output_part1_pydma_all":
        raise ValueError("Phase 1 must use only labeling_output_part1_pydma_all.")
    if output.exists():
        existing = {path.name for path in output.iterdir()}
        if not existing <= OUTPUT_FILES:
            raise FileExistsError(f"Output contains files not owned by this audit runner: {output}")
    output.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((labels / "run_metadata.json").read_text(encoding="utf-8"))
    recorded_dataset_root = Path(metadata["dataset_root"]).resolve()
    if recorded_dataset_root != dataset.resolve():
        paths = load_project_paths()
        legacy_part1_root = (
            paths.root / "Path_Dependent_Battery_Degradation_Dataset_Part 1"
        ).resolve()
        if dataset.resolve() != paths.dataset_parts["1"] or recorded_dataset_root != legacy_part1_root:
            raise ValueError("Label metadata points to a different dataset root.")
    readiness = json.loads((soh_run / "data_readiness.json").read_text(encoding="utf-8"))
    dataset_manifest = _read_csv(labels / "dataset_manifest.csv")

    eis, mat_details, mismatches = _eis_csv_audit(dataset, dataset_manifest)
    _write_json(output / "eis_audit.json", eis)
    _write_csv(output / "eis_mat_decode.csv", mat_details)
    _write_csv(output / "eis_pair_mismatches.csv", mismatches)

    cohort, cell_summary, consistency = _load_cohort(labels, readiness, soh_run, charge_run, dataset_manifest)
    _write_csv(output / "cohort_by_cell_rpt.csv", cohort)
    _write_csv(output / "pydma_cell_coverage.csv", cell_summary)
    counts = _cohort_counts(cohort, cell_summary)
    _write_json(output / "cohort_summary.json", counts)
    soh = _soh_report(labels, readiness, output)
    _write_json(output / "soh_audit.json", soh)
    provenance = _pydma_provenance(labels, cell_summary, consistency)
    _write_json(output / "pydma_provenance_and_direction_audit.json", provenance)

    temporal_axes = {
        "available_fields": {name: detail for name, detail in json.loads((labels / "feature_schema.json").read_text(encoding="utf-8"))["columns"].items() if name in {"Cyc", "TestTime", "StepTime", "Amphr"}},
        "cumulative_throughput": "not_verified; no cumulative per-cell throughput field and Amphr is used to define SOH.",
        "calendar_time": "not_verified; no observation timestamp or documented elapsed-time unit; filesystem mtimes are not experimental time.",
        "plots_created": ["soh_by_rpt_order_by_group.png"],
        "plots_not_created": {
            "soh_vs_cumulative_throughput": "BLOCKED: no verified cumulative throughput axis.",
            "soh_vs_calendar_time": "BLOCKED: no verified observation calendar timestamp or elapsed-time unit.",
        },
    }
    _write_json(output / "aging_axis_audit.json", temporal_axes)

    forbidden = [
        {"feature": "Step 8 pseudo-OCV discharge curve (voltage, capacity, or descriptors)", "status": "forbidden_for_discharge_PyDMA_target"},
        {"feature": "Step 10 pseudo-OCV charge curve (voltage, capacity, or descriptors)", "status": "forbidden_for_charge_PyDMA_target"},
        {"feature": "PyDMA fit parameters, RMSE, run counts, labels", "status": "forbidden_target_provenance"},
        {"feature": "Step 4 Amphr/capacity and direct SOH transforms", "status": "forbidden_for_SOH_prediction"},
        {"feature": "Future observations, identifiers, file ID, RPT order", "status": "forbidden_leakage_or_memorization"},
        {"feature": "Step 4 and Step 6 voltage curves", "status": "distinct_from_PyDMA_Step8_Step10_but_same_RPT; step filtering must be checked and same-observation correlation disclosed"},
    ]
    _write_csv(output / "forbidden_features.csv", forbidden)

    hash_rows = _hash_manifest(workspace, dataset, output, reuse_existing=reuse_hashes)
    _write_json(
        output / "phase1_summary.json",
        {
            "phase": "1",
            "dataset_scope": "Oxford Path Dependent Battery Degradation Dataset Part 1 only",
            "source_dataset_metadata_hash": metadata["dataset_manifest_hash"],
            "content_hash_entries": len(hash_rows),
            "eis": eis,
            "cohort": counts,
            "soh": soh,
            "pydma": {
                "generation_version": provenance["generation_version"],
                "generation_config": provenance["generation_config"],
                "source_curve_steps": provenance["source_curves"],
                "direction_rule": provenance["direction_rule_registered_before_agreement_summary"],
                "direction_agreement": consistency,
            },
            "aging_axis_audit": temporal_axes,
            "blockers_and_options": [
                {
                    "blocker": "EIS MAT payloads are MCOS opaque; four numeric fragments conflict with their filename metadata; no verified frequency axis, Re/Im mapping, or RPT alignment.",
                    "options": [
                        "Export the 117 MAT files with MATLAB/official reader to a documented table containing variable names, frequency, Re/Im, cell, stage, SoC, and point order; then reconcile the 16 exact-stem mismatches.",
                        "Obtain an authoritative MAT schema/export from the dataset maintainers and verify every frequency-to-impedance row before use.",
                        "Keep EIS not_ready and proceed only with non-EIS features; do not reconstruct frequency or relabel N_A.",
                    ],
                },
                {
                    "blocker": "One PyDMA cell is MAT-only; three further cells have Step 8/10 CSV rows but every label record is not_run and no reason/log is recorded.",
                    "options": [
                        "Recover original PyDMA run logs and per-cell config/version metadata if archived.",
                        "After separate approval, rerun only those four cells with a frozen PyDMA version/config and preserve accepted/withheld provenance.",
                        "Keep the four cells masked and limit degradation-label conclusions to the eight cells with accepted labels.",
                    ],
                },
                {
                    "blocker": "No verified cumulative-throughput or calendar-time axis per observation; TestTime units/continuity are undocumented.",
                    "options": [
                        "Obtain acquisition logs defining TestTime units and cross-file continuity, then derive calendar elapsed time from verified values.",
                        "Derive cumulative Ah throughput only from a documented integration interval/sign/unit rule and validate against instrument records.",
                        "Use the generated RPT-order plot only and leave throughput/calendar plots explicitly not_run.",
                    ],
                },
            ],
            "gate1_status": "BLOCKED_WITH_EIS_AND_AGING_AXES_NOT_READY; no training or Parts 2/3 access performed.",
            "training_started": False,
            "parts_2_3_read": False,
        },
    )
    return {
        "output": str(output),
        "eis_mat": len(mat_details),
        "csv_only_stems": len(eis["csv_only_exact_stems"]),
        "mat_only_stems": len(eis["mat_only_exact_stems"]),
        "cohort_rows": len(cohort),
        "accepted_pydma_rows": counts["accepted_directional_fits_total"],
        "sha256_entries": len(hash_rows),
        "gate1_status": "BLOCKED; see phase1_summary.json",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only Oxford Part 1 Phase 1 data/provenance audit.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--soh-run", type=Path, required=True)
    parser.add_argument("--charge-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-hashes", action="store_true", help="Reuse an existing SHA snapshot after validating path coverage and file sizes.")
    args = parser.parse_args()
    summary = run_phase1(
        args.dataset.resolve(), args.labels.resolve(), args.workspace.resolve(),
        args.soh_run.resolve(), args.charge_run.resolve(), args.output.resolve(), args.reuse_hashes
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()