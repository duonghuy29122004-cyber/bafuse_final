from __future__ import annotations

import csv
from contextlib import contextmanager
from datetime import datetime
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from pydma import DMAAnalyzer, DMAConfig
from pydma.core.optimizer import DMAOptimizer
from pydma.preprocessing.loader import load_ocp

from bafuse.labeling.pydma_policy import (
    PYDMA_MIN_ACCEPTED_RUNS,
    PYDMA_RMSE_THRESHOLD_MV,
    pydma_fit_passes_gate,
)
from bafuse.labeling.pydma_runner import _fit_one, _read_pocv_with_audit
from bafuse.labeling.torch_pydma import use_cuda_optimizer


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = Path(__file__).resolve().parent
PART3_ROOT = ROOT / "data" / "raw" / "oxford" / "part3"
HALF_CELL_ROOT = ROOT / "data" / "raw" / "oxford" / "part1" / "Half_Cell" / "Half Cell"
CANONICAL_MAPPING = ROOT / "data" / "manifests" / "bafuse_rpt_observations.csv"
FINAL2_DATASET = ROOT / "experiments" / "final2" / "v1" / "dataset"
FINAL2_MAPPING_AUDIT = ROOT / "reports" / "final_mapping_label_audit" / "mapping_observation_audit.csv"
VALIDATION_RUN_MANIFEST = ROOT / "reports" / "bafuse_final_data_validation_v2" / "run_manifest.json"
SOURCE_HASH_MANIFEST = FINAL2_DATASET / "g10_cell2_source_sha256.json"
CURVE_ELIGIBILITY = FINAL2_DATASET / "g10_cell2_curve_eligibility.csv"
IDENTITY_PROVENANCE = (
    "Group 10 Cell 2 identity confirmed by project owner; "
    "explicit Cell 2 label omitted from printed Guide."
)
TPG_BY_RPT = {
    1: "TPG10.3",
    2: "TPG10.6",
    3: "TPG10.7",
    4: "TPG10.8",
    5: "TPG10.10",
    6: "TPG10.13",
    7: "TPG10.14",
    8: "TPG10.17",
    9: "TPG10.18",
    10: "TPG10.19",
}
OPTIMIZER_OPTIONS = {"maxiter": 50, "popsize": 30, "max_tries": 10}
SEED = 211
REFERENCE_FILES = {
    "anode": HALF_CELL_ROOT / "CH_HC_anode.csv",
    "cathode": HALF_CELL_ROOT / "CH_HC_cathode.csv",
}
EXISTING_RESULT_FILES = (
    ROOT / "data" / "processed" / "label_pydma_fit_part3" / "run_009_part3_pydma" / "pydma_results.csv",
    ROOT / "reports" / "final_mapping_label_audit" / "pydma_fit_audit.csv",
    ROOT / "reports" / "bafuse_final_data_validation_v2" / "pydma_refit_results.csv",
    ROOT / "reports" / "bafuse_stage_validation_v2" / "stage2_pydma_pilot_v1" / "stage2_pydma_fit_attempts.csv",
)
EXPECTED_OUTPUTS = (
    "charge_curve_audit.csv",
    "existing_charge_fit_audit.csv",
    "charge_fit_results.csv",
    "charge_degradation_results.csv",
    "charge_fit_parameters.csv",
    "charge_fit_attempts.csv",
    "charge_fitted_curves.npz",
    "charge_fit_quality.json",
    "charge_fit_manifest.json",
    "artifact_sha256_manifest.json",
    "phase_c2_charge_report.md",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _normal_path(value: str) -> str:
    return value.replace("\\", "/").casefold()


def _source_path(value: str) -> Path:
    normalized = value.replace("\\", "/")
    if normalized.startswith("data/raw/"):
        relative = Path(*normalized.split("/"))
    else:
        relative = Path("data/raw/oxford/part3", *normalized.split("/"))
    path = (ROOT / relative).resolve()
    path.relative_to(ROOT.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"Mapped source file does not exist: {path}")
    return path


def _segment_bounds(reference: str, expected_step: int) -> tuple[int, int]:
    match = re.search(
        rf"#Step{expected_step}:[^#]*?rows(\d+)-(\d+)",
        reference,
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not parse Step {expected_step} row bounds: {reference}")
    start, end = int(match.group(1)), int(match.group(2))
    if start < 2 or end < start:
        raise ValueError(f"Invalid Step {expected_step} row bounds: {reference}")
    return start, end


def _check_mapping_and_dataset() -> tuple[list[dict[str, str]], dict[str, str]]:
    rows = [
        row
        for row in _read_csv(CANONICAL_MAPPING)
        if row.get("part") == "3" and row.get("physical_cell_id") == "G10-Cell2"
    ]
    rows.sort(key=lambda row: int(row["rpt_index"]))
    if [int(row["rpt_index"]) for row in rows] != list(TPG_BY_RPT):
        raise ValueError("Canonical G10-Cell2 RPT sequence is not RPT1-RPT10.")
    for row in rows:
        rpt = int(row["rpt_index"])
        if (
            row.get("source_tpg") != TPG_BY_RPT[rpt]
            or row.get("mapping_status") != "GUIDE_VERIFIED"
            or row.get("split") != "external_test"
            or row.get("audit_notes") != IDENTITY_PROVENANCE
            or row.get("step10_status") != "verified"
        ):
            raise ValueError(f"Canonical G10-Cell2 mapping changed at RPT{rpt}.")
        if Path(row["source_file"]).name != f"{TPG_BY_RPT[rpt]}-Cell2.csv":
            raise ValueError(f"Unexpected G10-Cell2 source at RPT{rpt}.")

    bol_sources = {row["bol_reference_file"] for row in rows}
    if len(bol_sources) != 1 or Path(next(iter(bol_sources))).name != "TPG10.0-Cell2.csv":
        raise ValueError("G10-Cell2 rows do not share the approved same-cell BoL source.")

    summary = json.loads((FINAL2_DATASET / "dataset_summary.json").read_text(encoding="utf-8"))
    if (
        summary["status_counts"] != {"ACCEPTED": 212, "REVIEW": 259, "REJECTED": 6}
        or summary["accepted_observations_by_partition"]
        != {"train": 99, "validation": 40, "external_test": 73}
        or summary["training_authorized"]
        or summary["external_inference_authorized"]
    ):
        raise ValueError("The current Final2 dataset no longer matches the approved state.")
    observation_rows = _read_csv(FINAL2_DATASET / "observation_manifest.csv")
    g10_observations = [
        row for row in observation_rows if row["physical_cell_id"] == "G10-Cell2"
    ]
    if (
        len(g10_observations) != 10
        or {
            int(row["rpt_index"]): row["status"]
            for row in g10_observations
        }
        != {**{index: "ACCEPTED" for index in range(1, 10)}, 10: "REJECTED"}
    ):
        raise ValueError("Final2 G10-Cell2 observation eligibility changed.")
    eligibility = {
        int(row["rpt_index"]): row
        for row in _read_csv(CURVE_ELIGIBILITY)
        if row["physical_cell_id"] == "G10-Cell2"
    }
    if len(eligibility) != 10 or eligibility[10]["step10_eligible"] != "False":
        raise ValueError("Expected the independent Final2 RPT10 Step 10 exclusion.")
    return rows, eligibility


def _historical_cell_match(row: dict[str, str]) -> bool:
    if row.get("physical_cell_id") == "G10-Cell2":
        return True
    group = row.get("group_id", "").replace("Group ", "").strip()
    cell = row.get("cell_id", "").replace("Cell ", "").strip()
    return group == "10" and cell == "2"


def _audit_existing_fits(
    sources: dict[str, str],
) -> list[dict[str, Any]]:
    matches: dict[str, list[dict[str, Any]]] = {name: [] for name in sources}
    scanned_files = []
    for path in EXISTING_RESULT_FILES:
        if not path.is_file():
            continue
        scanned_files.append(path.relative_to(ROOT).as_posix())
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if not _historical_cell_match(row):
                    continue
                if row.get("direction", "").casefold() != "charge":
                    continue
                source_name = Path(row.get("source_file", "")).name
                for measurement, source_file in sources.items():
                    if source_name == Path(source_file).name:
                        matches[measurement].append(
                            {
                                "result_file": path.relative_to(ROOT).as_posix(),
                                "source_file": source_name,
                                "step": row.get("test_step") or row.get("step") or "",
                                "rmse_mV": row.get("fit_rmse_mV") or row.get("rmse_mV") or "",
                                "accepted_runs": row.get("accepted_runs", ""),
                                "total_runs": row.get("total_runs", ""),
                                "fit_status": (
                                    row.get("diagnostic_label_status")
                                    or row.get("fit_status")
                                    or row.get("fit_status_original")
                                    or ""
                                ),
                                "has_parameters": bool(
                                    row.get("fitted_parameters")
                                    or row.get("fitted_parameters_json")
                                ),
                            }
                        )

    output = []
    for measurement, source_file in sources.items():
        rows = matches[measurement]
        output.append(
            {
                "physical_cell_id": "G10-Cell2",
                "measurement_id": measurement,
                "direction": "charge",
                "expected_step": 10,
                "source_file": source_file,
                "matching_existing_fit_count": len(rows),
                "existing_charge_fit_found": bool(rows),
                "source_reference_match": (
                    all(item["source_file"] == Path(source_file).name for item in rows)
                    if rows
                    else False
                ),
                "quality_gate_passed": any(
                    _row_passes_historical_gate(item) for item in rows
                ),
                "fitted_parameters_available": any(
                    item["has_parameters"] for item in rows
                ),
                "existing_fit_records_json": json.dumps(rows, ensure_ascii=False),
                "searched_result_files_json": json.dumps(scanned_files),
                "reuse_decision": (
                    "no_existing_fit; new fit required" if not rows else "review_existing_records"
                ),
            }
        )
    return output


def _row_passes_historical_gate(row: dict[str, Any]) -> bool:
    try:
        rmse = float(row["rmse_mV"])
        accepted = int(float(row["accepted_runs"]))
    except (TypeError, ValueError):
        return False
    return rmse < PYDMA_RMSE_THRESHOLD_MV and accepted >= PYDMA_MIN_ACCEPTED_RUNS


@contextmanager
def _capture_optimizer_runs() -> Iterator[list[dict[str, Any]]]:
    original = DMAOptimizer._run_single
    runs: list[dict[str, Any]] = []

    def capture(self: DMAOptimizer, seed: int | None = None, **kwargs: Any) -> Any:
        result = original(self, seed=seed, **kwargs)
        is_accepted = bool(result.rmse < PYDMA_RMSE_THRESHOLD_MV / 1000.0)
        runs.append(
            {
                "run_index": len(runs) + 1,
                "seed": seed,
                "rmse_mV": float(result.rmse * 1000.0),
                "cost": float(result.cost),
                "accepted_by_pydma_rmse_loop": is_accepted,
                "optimizer_converged": bool(result.success),
                "iterations": int(result.n_iterations),
                "function_evaluations": int(result.n_function_evals),
                "parameters": np.asarray(result.params, dtype=np.float64).tolist(),
                "parameter_bounds": [
                    [float(lower), float(upper)] for lower, upper in self.bounds
                ],
                "solver_options": self._solver_opts,
                "call_options": {
                    key: _jsonable(value)
                    for key, value in kwargs.items()
                    if key in {"maxiter", "popsize", "polish", "x0"}
                },
            }
        )
        return result

    DMAOptimizer._run_single = capture
    try:
        yield runs
    finally:
        DMAOptimizer._run_single = original


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Cannot serialize non-finite value: {value}")
    return value


def _reference_audit() -> tuple[Any, Any, dict[str, dict[str, Any]]]:
    references = {}
    loaded = {}
    for kind, path in REFERENCE_FILES.items():
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames or []
            raw_rows = list(reader)
        if not {"SoC", "Volts"}.issubset(fields):
            raise ValueError(f"Charge half-cell reference has an unexpected schema: {path}")
        soc = np.asarray([float(row["SoC"]) for row in raw_rows], dtype=np.float64)
        voltage = np.asarray([float(row["Volts"]) for row in raw_rows], dtype=np.float64)
        delta_soc = np.diff(soc)
        if (
            not np.isfinite(soc).all()
            or not np.isfinite(voltage).all()
            or not np.all(delta_soc > 0)
            or soc[-1] <= soc[0]
        ):
            raise ValueError(f"Charge half-cell reference is not finite and SoC-increasing: {path}")
        electrode = load_ocp(
            path,
            electrode_type=kind,
            soc_col="SoC",
            voltage_col="Volts",
        )
        if not (
            np.isclose(float(electrode.soc[0]), 0.0)
            and np.isclose(float(electrode.soc[-1]), 1.0)
        ):
            raise ValueError(f"PyDMA did not normalize the charge reference to [0,1]: {path}")
        loaded[kind] = electrode
        references[kind] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": _sha256(path),
            "schema": fields,
            "rows": len(raw_rows),
            "all_values_finite": True,
            "raw_soc_orientation": "strictly_increasing",
            "raw_soc_min": float(soc[0]),
            "raw_soc_max": float(soc[-1]),
            "raw_voltage_first_v": float(voltage[0]),
            "raw_voltage_last_v": float(voltage[-1]),
            "loaded_electrode_type": electrode.electrode_type,
            "loaded_point_count": int(len(electrode.soc)),
            "loaded_soc_first": float(electrode.soc[0]),
            "loaded_soc_last": float(electrode.soc[-1]),
            "smoothing": False,
            "loader": "pydma.preprocessing.loader.load_ocp(soc_col='SoC', voltage_col='Volts')",
        }
    return loaded["anode"], loaded["cathode"], references


def _safe_key(measurement: str) -> str:
    return "bol_charge" if measurement == "BoL" else f"rpt_{int(measurement[3:]):02d}_charge"


def _report_text(
    fit_rows: list[dict[str, Any]],
    degradation_rows: list[dict[str, Any]],
    quality: dict[str, Any],
) -> str:
    lines = [
        "# Phase C2 — G10-Cell2 PyDMA nạp",
        "",
        "Đây là kết quả PyDMA chẩn đoán từ đường pseudo-OCV nạp Step 10; "
        "LLI/LAM không phải ground truth độc lập. Không thực hiện discharge fitting "
        "hoặc ML training.",
        "",
        "## Mapping và reference",
        "",
        "- Mapping giữ `GUIDE_VERIFIED`, RPT1–RPT10 và External Test; BoL là "
        "TPG10.0-Cell2 cùng cell.",
        "- Chỉ nạp được fit: Step 10, `Amps > 0`, `abs(Amps) <= 0.2 A`; capacity "
        "đơn điệu tăng, chọn segment hợp lệ dài nhất theo selector PyDMA hiện có.",
        "- Tham chiếu half-cell: CH_HC_anode.csv và CH_HC_cathode.csv; xem hash "
        "và schema trong `charge_fit_manifest.json`.",
        "- Gate PyDMA hiện hành: RMSE < 15 mV và ít nhất 3 accepted runs. "
        "Optimizer dừng khi đủ 3 accepted runs hoặc đủ 10 lần thử; accepted run "
        "được xác định bằng RMSE < 15 mV. Kết quả tốt nhất chọn theo RMSE.",
        "- Degradation tính so với fit charge BoL cùng cell, không reset ở RPT1. "
        "Nếu BoL không qua gate, các degradation estimate bị chặn.",
        "",
        "## Kết quả fit",
        "",
        "| Checkpoint | Fit RMSE (mV) | Accepted/total | Gate | Trạng thái |",
        "|---|---:|---:|---|---|",
    ]
    for row in fit_rows:
        rmse = row["fit_rmse_mV"] if row["fit_rmse_mV"] != "" else "—"
        lines.append(
            f"| {row['measurement_id']} | {rmse} | "
            f"{row['accepted_runs']}/{row['total_runs']} | "
            f"{row['quality_gate_passed']} | {row['fit_status']} |"
        )
    lines.extend(
        [
            "",
            "## Ước lượng suy giảm",
            "",
            "| RPT | LLI (%) | LAM anode (%) | LAM cathode (%) | QC |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for row in degradation_rows:
        values = [
            row["lli_percent"],
            row["lam_anode_percent"],
            row["lam_cathode_percent"],
        ]
        cells = [f"{float(value):.6g}" if value != "" else "—" for value in values]
        lines.append(
            f"| {row['rpt_index']} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{row['quality_control_status']} |"
        )
    lines.extend(
        [
            "",
            "## Tóm tắt QC",
            "",
            f"- Số fit pass gate (tính cả BoL): {quality['accepted_fit_count']}.",
            f"- Số input `PYDMA_CHARGE_REVIEW`: {quality['charge_review_count']}.",
            f"- BoL same-cell reference hợp lệ: {quality['bol_reference_accepted']}.",
            f"- RPT10 được xét độc lập cho PyDMA; trạng thái Final2 vẫn "
            f"`REJECTED`: {quality['rpt10_final2_status']}.",
            f"- RPT10 PyDMA input: {quality['rpt10_pydma_input_status']}.",
            "- Giá trị âm không bị clamp; cờ kiểm tra được ghi riêng trong "
            "`charge_degradation_results.csv`.",
            "- Không sửa raw CSV, mapping/split, dataset Final2, nhãn P5 hay "
            "kết quả PyDMA lịch sử.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("The validated Part 3 PyDMA runner requires CUDA.")
    if any((OUTPUT / name).exists() for name in EXPECTED_OUTPUTS):
        raise FileExistsError(
            f"Refusing to overwrite an existing Phase C2 artifact in {OUTPUT}."
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_started_at = datetime.now().astimezone().isoformat()
    run_started_perf = time.perf_counter()
    mapping_rows, final2_eligibility = _check_mapping_and_dataset()
    bol_source = mapping_rows[0]["bol_reference_file"]
    source_by_measurement = {"BoL": _source_path(bol_source).relative_to(ROOT).as_posix()}
    for row in mapping_rows:
        source_by_measurement[f"RPT{row['rpt_index']}"] = row["source_file"]

    existing_fit_rows = _audit_existing_fits(source_by_measurement)
    if any(row["existing_charge_fit_found"] for row in existing_fit_rows):
        raise RuntimeError(
            "Historical exact-identity charge fit records were found. "
            "Review/reuse them before launching new optimizer runs."
        )

    saved_source_hashes = json.loads(SOURCE_HASH_MANIFEST.read_text(encoding="utf-8"))
    source_hashes: dict[str, dict[str, Any]] = {}
    for measurement, relative_path in source_by_measurement.items():
        path = ROOT / Path(*relative_path.split("/"))
        actual = _sha256(path)
        expected = saved_source_hashes["source_files"].get(relative_path, {}).get("sha256")
        if not expected or actual != expected:
            raise ValueError(f"Mapped source SHA-256 differs from the Final2 audit: {relative_path}")
        source_hashes[relative_path] = {
            "sha256": actual,
            "size_bytes": path.stat().st_size,
            "measurement_id": measurement,
        }
        print(f"Verified mapped source {measurement}: {relative_path}", flush=True)

    anode, cathode, half_cell_audit = _reference_audit()
    reference_capacity_row = mapping_rows[0]
    bol_reference = reference_capacity_row["bol_step4_segment_ref"]
    if "TPG10.0-Cell2.csv" not in bol_reference:
        raise ValueError("The mapped same-cell BoL Step 4 source is not TPG10.0-Cell2.")

    curve_rows: list[dict[str, Any]] = []
    inputs: dict[str, dict[str, Any]] = {}
    for measurement, relative_path in source_by_measurement.items():
        path = ROOT / Path(*relative_path.split("/"))
        print(f"Charge Step 10 preflight {measurement}: {relative_path}", flush=True)
        curve, audit = _read_pocv_with_audit(path, step_number=10, direction="charge")
        mapped_bounds: tuple[int, int] | None = None
        mapping_reference = ""
        if measurement != "BoL":
            rpt_index = int(measurement[3:])
            mapped_row = mapping_rows[rpt_index - 1]
            mapping_reference = mapped_row["step10_segment_ref"]
            ref_source = mapping_reference.split("#", 1)[0]
            if _normal_path(ref_source) != _normal_path(relative_path):
                raise ValueError(f"Step 10 reference source differs at {measurement}.")
            mapped_bounds = _segment_bounds(mapping_reference, 10)

        selected = audit.get("segments", [])
        selected_index = audit.get("selected_segment")
        selected_segment = next(
            (item for item in selected if item["segment_index"] == selected_index),
            None,
        )
        source_rows_match = True
        if curve is None or selected_segment is None:
            source_rows_match = False
        elif mapped_bounds is not None:
            source_rows_match = (
                mapped_bounds[0] <= selected_segment["source_row_start"]
                and selected_segment["source_row_end"] <= mapped_bounds[1]
            )
        time_audit = audit.get("selected_testtime_audit", {})
        time_status = time_audit.get("status", "unavailable")
        voltage_increases = bool(
            curve is not None and float(curve[1][-1]) > float(curve[1][0])
        )
        reasons = []
        if curve is None:
            reasons.append(audit.get("status", "curve_unavailable"))
        if not source_rows_match:
            reasons.append("selected_segment_outside_verified_mapping_bounds")
        if time_status != "verified_strictly_increasing":
            reasons.append(f"testtime_{time_status}")
        if curve is not None and not voltage_increases:
            reasons.append("charge_voltage_endpoint_not_increasing")
        pydma_eligible = curve is not None and not reasons
        row = {
            "physical_cell_id": "G10-Cell2",
            "measurement_id": measurement,
            "rpt_index": "" if measurement == "BoL" else measurement[3:],
            "source_file": relative_path,
            "source_sha256": source_hashes[relative_path]["sha256"],
            "direction": "charge",
            "step": 10,
            "mapping_status": "GUIDE_VERIFIED" if measurement != "BoL" else "same_cell_bol",
            "step10_mapping_reference": mapping_reference,
            "final2_step10_eligible": (
                ""
                if measurement == "BoL"
                else final2_eligibility[int(measurement[3:])]["step10_eligible"]
            ),
            "final2_step10_current_sign_counts": (
                ""
                if measurement == "BoL"
                else final2_eligibility[int(measurement[3:])]["current_sign_counts"]
            ),
            "existing_extractor_status": audit.get("status", ""),
            "selected_segment": selected_index or "",
            "selected_source_row_start": (
                audit.get("selected_source_row_start", "")
            ),
            "selected_source_row_end": audit.get("selected_source_row_end", ""),
            "mapped_step10_row_start": mapped_bounds[0] if mapped_bounds else "",
            "mapped_step10_row_end": mapped_bounds[1] if mapped_bounds else "",
            "selected_segment_within_mapping": source_rows_match,
            "filtered_finite_samples": audit.get("filtered_finite_samples", 0),
            "selected_sample_count": audit.get("selected_sample_count", 0),
            "minimum_point_count": 10,
            "minimum_point_count_passed": bool(
                selected_segment and selected_segment["sample_count"] >= 10
            ),
            "capacity_min_native_amphr": (
                selected_segment["capacity_min"] if selected_segment else ""
            ),
            "capacity_max_native_amphr": (
                selected_segment["capacity_max"] if selected_segment else ""
            ),
            "capacity_span_native_amphr": (
                selected_segment["capacity_span"] if selected_segment else ""
            ),
            "capacity_unit": "native Amphr; absolute unit not independently established",
            "capacity_orientation": (
                "strictly increasing" if selected_segment else "unavailable"
            ),
            "voltage_first_v": float(curve[1][0]) if curve is not None else "",
            "voltage_last_v": float(curve[1][-1]) if curve is not None else "",
            "charge_voltage_endpoint_increasing": voltage_increases,
            "selected_testtime_audit_json": json.dumps(time_audit, allow_nan=False),
            "pydma_input_eligible": pydma_eligible,
            "pydma_input_status": "eligible" if pydma_eligible else "PYDMA_CHARGE_REVIEW",
            "review_reasons": ";".join(reasons),
            "curve_audit_json": json.dumps(audit, ensure_ascii=False, allow_nan=False),
            "final2_ml_status": (
                ""
                if measurement == "BoL"
                else (
                    "REJECTED"
                    if measurement == "RPT10"
                    else "ACCEPTED"
                )
            ),
        }
        curve_rows.append(row)
        if pydma_eligible and curve is not None:
            inputs[measurement] = {
                "capacity": curve[0],
                "voltage": curve[1],
                "audit": audit,
                "source_file": relative_path,
            }
        print(
            f"  {row['pydma_input_status']}: points={row['selected_sample_count']} "
            f"capacity_span={row['capacity_span_native_amphr']} "
            f"time={time_status}",
            flush=True,
        )

    _write_csv(
        OUTPUT / "charge_curve_audit.csv",
        curve_rows,
        list(curve_rows[0].keys()),
    )
    _write_csv(
        OUTPUT / "existing_charge_fit_audit.csv",
        existing_fit_rows,
        list(existing_fit_rows[0].keys()),
    )
    if not inputs.get("BoL"):
        print(
            "Same-cell BoL charge curve is unavailable; RPT fits may be diagnostic, "
            "but degradation values will be withheld.",
            flush=True,
        )

    fit_rows: list[dict[str, Any]] = []
    attempt_rows: list[dict[str, Any]] = []
    parameter_rows: list[dict[str, Any]] = []
    fit_objects: dict[str, Any] = {}
    fitted_curves: dict[str, np.ndarray] = {}
    for measurement in ("BoL", *(f"RPT{i}" for i in range(1, 11))):
        if measurement not in inputs:
            fit_rows.append(
                {
                    "physical_cell_id": "G10-Cell2",
                    "measurement_id": measurement,
                    "rpt_index": "" if measurement == "BoL" else measurement[3:],
                    "source_file": source_by_measurement[measurement],
                    "fit_direction": "charge",
                    "step": 10,
                    "fit_rmse_mV": "",
                    "accepted_runs": 0,
                    "total_runs": 0,
                    "quality_gate_passed": False,
                    "optimization_status": "not_run_curve_review",
                    "fit_status": "PYDMA_CHARGE_REVIEW",
                    "failure_reason": "Charge Step 10 input failed existing extraction/time/mapping checks.",
                    "best_parameters_json": "",
                    "parameter_std_json": "",
                }
            )
            continue
        fit_input = inputs[measurement]
        run_details: list[dict[str, Any]] = []
        fit_error = ""
        fit = None
        started = time.perf_counter()
        try:
            with _capture_optimizer_runs() as run_details:
                fit = _fit_one(
                    "charge",
                    fit_input["capacity"],
                    fit_input["voltage"],
                    anode,
                    cathode,
                    seed=SEED,
                    context=f"G10-Cell2 {measurement}",
                    use_cuda=True,
                    **OPTIMIZER_OPTIONS,
                )
        except Exception as error:
            fit_error = f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started

        for attempt in run_details:
            attempt_rows.append(
                {
                    "physical_cell_id": "G10-Cell2",
                    "measurement_id": measurement,
                    "direction": "charge",
                    "step": 10,
                    "run_index": attempt["run_index"],
                    "seed": attempt["seed"],
                    "rmse_mV": attempt["rmse_mV"],
                    "cost": attempt["cost"],
                    "accepted_by_pydma_rmse_loop": attempt[
                        "accepted_by_pydma_rmse_loop"
                    ],
                    "optimizer_converged": attempt["optimizer_converged"],
                    "iterations": attempt["iterations"],
                    "function_evaluations": attempt["function_evaluations"],
                    "parameters_json": json.dumps(attempt["parameters"]),
                    "parameter_bounds_json": json.dumps(attempt["parameter_bounds"]),
                    "solver_options_json": json.dumps(_jsonable(attempt["solver_options"])),
                    "call_options_json": json.dumps(attempt["call_options"]),
                }
            )

        if fit is None:
            fit_rows.append(
                {
                    "physical_cell_id": "G10-Cell2",
                    "measurement_id": measurement,
                    "rpt_index": "" if measurement == "BoL" else measurement[3:],
                    "source_file": fit_input["source_file"],
                    "fit_direction": "charge",
                    "step": 10,
                    "fit_rmse_mV": "",
                    "accepted_runs": len(
                        [item for item in run_details if item["accepted_by_pydma_rmse_loop"]]
                    ),
                    "total_runs": len(run_details),
                    "quality_gate_passed": False,
                    "optimization_status": "exception",
                    "fit_status": "withheld_fit_exception",
                    "failure_reason": fit_error,
                    "best_parameters_json": "",
                    "parameter_std_json": "",
                    "runtime_seconds": elapsed,
                    "optimizer_runs_json": json.dumps(run_details, allow_nan=False),
                }
            )
            continue

        if (
            len(run_details) != int(fit.n_total_runs)
            or sum(item["accepted_by_pydma_rmse_loop"] for item in run_details)
            != int(fit.n_accepted_runs)
        ):
            raise RuntimeError(
                f"Captured optimizer runs do not match PyDMA's run counts for {measurement}."
            )
        best_attempt = min(
            run_details,
            key=lambda item: (
                not item["accepted_by_pydma_rmse_loop"],
                item["rmse_mV"],
            ),
        )
        for attempt_row in attempt_rows:
            if (
                attempt_row["measurement_id"] == measurement
                and attempt_row["run_index"] == best_attempt["run_index"]
            ):
                attempt_row["is_best_run"] = True
        passes_gate = pydma_fit_passes_gate(
            rmse_mV=float(fit.rmse * 1000.0),
            accepted_runs=int(fit.n_accepted_runs),
            is_accepted=bool(fit.is_accepted),
        )
        best_parameters = fit.fitted_params.to_dict()
        parameter_std = (
            np.asarray(fit.param_std, dtype=np.float64).tolist()
            if fit.param_std is not None
            else None
        )
        fit_status = (
            "reference_baseline"
            if measurement == "BoL" and passes_gate
            else (
                "withheld_baseline_fit_failed"
                if measurement == "BoL"
                else (
                    "diagnostic_fit_passed_quality_gate"
                    if passes_gate
                    else "raw_diagnostic_estimate"
                )
            )
        )
        if measurement == "RPT10" and passes_gate:
            fit_status = "diagnostic_only_final2_rejected"
        fit_rows.append(
            {
                "physical_cell_id": "G10-Cell2",
                "measurement_id": measurement,
                "rpt_index": "" if measurement == "BoL" else measurement[3:],
                "source_file": fit_input["source_file"],
                "fit_direction": "charge",
                "step": 10,
                "fit_rmse_mV": float(fit.rmse * 1000.0),
                "accepted_runs": int(fit.n_accepted_runs),
                "total_runs": int(fit.n_total_runs),
                "quality_gate_passed": passes_gate,
                "pydma_is_accepted": bool(fit.is_accepted),
                "optimization_status": fit.status,
                "fit_status": fit_status,
                "failure_reason": "",
                "best_parameters_json": json.dumps(best_parameters, allow_nan=False),
                "parameter_std_json": json.dumps(parameter_std, allow_nan=False),
                "runtime_seconds": elapsed,
                "optimizer_runs_json": json.dumps(run_details, allow_nan=False),
                "final2_ml_status": (
                    ""
                    if measurement == "BoL"
                    else ("REJECTED" if measurement == "RPT10" else "ACCEPTED")
                ),
                "quality_note": "PyDMA estimates are diagnostic, not independent ground truth.",
            }
        )
        fit_objects[measurement] = {
            "fit": fit,
            "input": fit_input,
            "quality_gate_passed": passes_gate,
        }
        parameter_rows.append(
            {
                "physical_cell_id": "G10-Cell2",
                "measurement_id": measurement,
                "fit_direction": "charge",
                "step": 10,
                "source_file": fit_input["source_file"],
                "best_parameters_json": json.dumps(best_parameters, allow_nan=False),
                "parameter_std_json": json.dumps(parameter_std, allow_nan=False),
                "accepted_runs": int(fit.n_accepted_runs),
                "total_runs": int(fit.n_total_runs),
                "fit_rmse_mV": float(fit.rmse * 1000.0),
                "pydma_version": "2.1.0",
            }
        )
        key = _safe_key(measurement)
        fitted_curves[f"{key}__input_capacity_native"] = fit_input["capacity"]
        fitted_curves[f"{key}__input_voltage_v"] = fit_input["voltage"]
        fitted_curves[f"{key}__fit_capacity_fraction"] = np.asarray(
            fit.measured_capacity, dtype=np.float64
        )
        fitted_curves[f"{key}__fit_measured_voltage_v"] = np.asarray(
            fit.measured_voltage, dtype=np.float64
        )
        fitted_curves[f"{key}__reconstructed_capacity_fraction"] = np.asarray(
            fit.soc_reconstructed, dtype=np.float64
        )
        fitted_curves[f"{key}__reconstructed_voltage_v"] = np.asarray(
            fit.ocv_reconstructed, dtype=np.float64
        )

    bol_fit = fit_objects.get("BoL")
    bol_accepted = bool(bol_fit and bol_fit["quality_gate_passed"])
    degradation_rows: list[dict[str, Any]] = []
    for index in range(1, 11):
        measurement = f"RPT{index}"
        fit_item = fit_objects.get(measurement)
        fit_row = next(row for row in fit_rows if row["measurement_id"] == measurement)
        values: tuple[float | None, float | None, float | None] = (None, None, None)
        comparison_error = ""
        if fit_item and bol_accepted:
            try:
                comparison = DMAAnalyzer(
                    config=DMAConfig(direction="charge"),
                    anode=anode,
                    cathode=cathode,
                ).compare_with_reference(
                    reference_params=bol_fit["fit"].fitted_params,
                    current_params=fit_item["fit"].fitted_params,
                    reference_capacity=float(
                        np.ptp(bol_fit["input"]["capacity"])
                    ),
                    current_capacity=float(np.ptp(fit_item["input"]["capacity"])),
                )
                values = (
                    float(comparison.lli * 100.0),
                    float(comparison.lam_anode * 100.0),
                    float(comparison.lam_cathode * 100.0),
                )
            except Exception as error:
                comparison_error = f"{type(error).__name__}: {error}"

        if not fit_item:
            quality_status = "PYDMA_CHARGE_REVIEW"
        elif not bol_accepted:
            quality_status = "BLOCKED_NO_ACCEPTED_SAME_CELL_BOL_REFERENCE"
        elif comparison_error:
            quality_status = "withheld_comparison_exception"
        elif index == 10:
            quality_status = "diagnostic_only_final2_rejected"
        elif fit_item["quality_gate_passed"]:
            quality_status = "diagnostic_estimate_accepted"
        else:
            quality_status = "raw_diagnostic_estimate"
        negative = any(value is not None and value < 0 for value in values)
        outside_range = any(
            value is not None and (value < 0 or value > 100) for value in values
        )
        degradation_rows.append(
            {
                "physical_cell_id": "G10-Cell2",
                "rpt_index": index,
                "measurement_id": measurement,
                "source_file": source_by_measurement[measurement],
                "fit_direction": "charge",
                "step": 10,
                "fit_rmse_mV": fit_row.get("fit_rmse_mV", ""),
                "lli_percent": values[0] if values[0] is not None else "",
                "lam_anode_percent": values[1] if values[1] is not None else "",
                "lam_cathode_percent": values[2] if values[2] is not None else "",
                "accepted_runs": fit_row.get("accepted_runs", 0),
                "total_runs": fit_row.get("total_runs", 0),
                "quality_control_status": quality_status,
                "same_cell_bol_source": source_by_measurement["BoL"],
                "same_cell_bol_fit_accepted": bol_accepted,
                "final2_ml_status": "REJECTED" if index == 10 else "ACCEPTED",
                "negative_estimate_review": negative,
                "outside_0_100_percent_review": outside_range,
                "comparison_error": comparison_error,
                "interpretation": (
                    "Model-derived diagnostic estimate; not independent ground truth."
                ),
            }
        )

    fit_fields = list(dict.fromkeys(key for row in fit_rows for key in row))
    degradation_fields = list(degradation_rows[0].keys())
    parameter_fields = list(parameter_rows[0].keys()) if parameter_rows else [
        "physical_cell_id", "measurement_id", "fit_direction", "step", "source_file",
        "best_parameters_json", "parameter_std_json", "accepted_runs", "total_runs",
        "fit_rmse_mV", "pydma_version",
    ]
    attempt_fields = [
        "physical_cell_id", "measurement_id", "direction", "step", "run_index",
        "seed", "rmse_mV", "cost", "accepted_by_pydma_rmse_loop",
        "is_best_run", "optimizer_converged", "iterations", "function_evaluations",
        "parameters_json", "parameter_bounds_json", "solver_options_json",
        "call_options_json",
    ]
    _write_csv(OUTPUT / "charge_fit_results.csv", fit_rows, fit_fields)
    _write_csv(OUTPUT / "charge_degradation_results.csv", degradation_rows, degradation_fields)
    _write_csv(OUTPUT / "charge_fit_parameters.csv", parameter_rows, parameter_fields)
    _write_csv(OUTPUT / "charge_fit_attempts.csv", attempt_rows, attempt_fields)
    np.savez_compressed(OUTPUT / "charge_fitted_curves.npz", **fitted_curves)

    bol_result_row = next(row for row in fit_rows if row["measurement_id"] == "BoL")
    rpt10_curve = next(row for row in curve_rows if row["measurement_id"] == "RPT10")
    quality = {
        "schema_version": 1,
        "dataset_scope": "Part 3, Group 10, Cell 2; charge-only Step 10",
        "fit_measurement_count": len(fit_rows),
        "fit_attempt_count": len(attempt_rows),
        "accepted_fit_count": sum(bool(row.get("quality_gate_passed")) for row in fit_rows),
        "charge_review_count": sum(
            row["pydma_input_status"] == "PYDMA_CHARGE_REVIEW" for row in curve_rows
        ),
        "bol_reference_accepted": bol_accepted,
        "bol_fit_status": bol_result_row["fit_status"],
        "rpt10_final2_status": "REJECTED",
        "rpt10_pydma_input_status": rpt10_curve["pydma_input_status"],
        "rpt10_pydma_quality_control_status": degradation_rows[-1][
            "quality_control_status"
        ],
        "rmse_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
        "minimum_accepted_runs": PYDMA_MIN_ACCEPTED_RUNS,
        "negative_estimate_count": sum(
            bool(row["negative_estimate_review"]) for row in degradation_rows
        ),
        "outside_0_100_percent_count": sum(
            bool(row["outside_0_100_percent_review"]) for row in degradation_rows
        ),
        "all_degradation_values_are_independently_measured": False,
        "training_authorized": False,
        "external_inference_authorized": False,
    }
    _write_json(OUTPUT / "charge_fit_quality.json", quality)

    report_path = OUTPUT / "phase_c2_charge_report.md"
    report_path.write_text(
        _report_text(fit_rows, degradation_rows, quality),
        encoding="utf-8",
    )

    import pydma

    source_manifest = json.loads(SOURCE_HASH_MANIFEST.read_text(encoding="utf-8"))
    validation_manifest = json.loads(
        VALIDATION_RUN_MANIFEST.read_text(encoding="utf-8")
    )
    inputs_hashes = {
        CANONICAL_MAPPING.relative_to(ROOT).as_posix(): _sha256(CANONICAL_MAPPING),
        FINAL2_MAPPING_AUDIT.relative_to(ROOT).as_posix(): _sha256(FINAL2_MAPPING_AUDIT),
        SOURCE_HASH_MANIFEST.relative_to(ROOT).as_posix(): _sha256(SOURCE_HASH_MANIFEST),
        CURVE_ELIGIBILITY.relative_to(ROOT).as_posix(): _sha256(CURVE_ELIGIBILITY),
        (FINAL2_DATASET / "dataset_summary.json").relative_to(ROOT).as_posix(): _sha256(
            FINAL2_DATASET / "dataset_summary.json"
        ),
        VALIDATION_RUN_MANIFEST.relative_to(ROOT).as_posix(): _sha256(
            VALIDATION_RUN_MANIFEST
        ),
        Path(__file__).resolve().relative_to(ROOT).as_posix(): _sha256(Path(__file__)),
        (ROOT / "src" / "bafuse" / "labeling" / "pydma_runner.py")
        .relative_to(ROOT)
        .as_posix(): _sha256(ROOT / "src" / "bafuse" / "labeling" / "pydma_runner.py"),
        (ROOT / "src" / "bafuse" / "labeling" / "torch_pydma.py")
        .relative_to(ROOT)
        .as_posix(): _sha256(ROOT / "src" / "bafuse" / "labeling" / "torch_pydma.py"),
    }
    for name, item in half_cell_audit.items():
        inputs_hashes[item["path"]] = item["sha256"]
    for relative_path, item in source_hashes.items():
        inputs_hashes[relative_path] = item["sha256"]

    artifact_names = [
        name for name in EXPECTED_OUTPUTS
        if name not in {"charge_fit_manifest.json", "artifact_sha256_manifest.json"}
    ]
    output_hashes = {
        name: _sha256(OUTPUT / name)
        for name in artifact_names
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": "bafuse-final2-phase-c2-g10-cell2-charge-v1",
        "physical_cell_id": "G10-Cell2",
        "mapping_status": "GUIDE_VERIFIED",
        "mapping_provenance": IDENTITY_PROVENANCE,
        "partition": "external_test",
        "rpt_order": [f"RPT{index}" for index in range(1, 11)],
        "final2_dataset_state": {
            "accepted": 212,
            "review": 259,
            "rejected": 6,
            "train_samples_cells": [99, 9],
            "validation_samples_cells": [40, 3],
            "external_test_samples_cells": [73, 12],
            "g10_cell2_accepted_rpts": list(range(1, 10)),
            "g10_cell2_rpt10_status": "REJECTED",
        },
        "charge_only": True,
        "fit_direction": "charge",
        "fit_step": 10,
        "discharge_fitting_performed": False,
        "bol_source_file": source_by_measurement["BoL"],
        "same_cell_bol_reference_policy": (
            "Fit the mapped same-cell Part 3 BoL charge Step 10 once; compare "
            "every RPT directly with that BoL fit. Do not reset at RPT1."
        ),
        "rpt10_policy": {
            "final2_ml_status_unchanged": "REJECTED",
            "pydma_status": rpt10_curve["pydma_input_status"],
            "extraction_rule": (
                "Use the existing PyDMA sign/magnitude filter; zero-current rows "
                "are excluded by Amps > 0. RPT10 fit is diagnostic only."
            ),
            "mapped_step10_reference": rpt10_curve["step10_mapping_reference"],
        },
        "half_cell_references": half_cell_audit,
        "curve_extraction": {
            "implementation": "bafuse.labeling.pydma_runner._read_pocv_with_audit",
            "filter": "Step == 10, Amps > 0, abs(Amps) <= 0.2 A",
            "segment_rule": (
                "Finite Amphr/Volts; split on Amphr non-increase; each selected "
                "segment has at least 10 points and strictly increasing Amphr; "
                "choose longest, earliest within 1% tie."
            ),
            "time_continuity_rule": (
                "Audit selected TestTime for finite values and strictly positive "
                "successive intervals; this audit does not alter curve selection."
            ),
            "orientation": "Keep original charge axis order; do not reverse capacity or voltage.",
            "capacity_unit": "native Amphr; absolute unit not independently established",
        },
        "pydma": {
            "version": version("pydma"),
            "rmse_threshold_mV": PYDMA_RMSE_THRESHOLD_MV,
            "minimum_accepted_runs": PYDMA_MIN_ACCEPTED_RUNS,
            "multi_run_interpretation": (
                "PyDMA accepts each optimizer attempt with OCV RMSE < 15 mV; "
                "stops after 3 accepted attempts or 10 total attempts; best "
                "accepted attempt is lowest RMSE. The repository label gate also "
                "requires the fit to be accepted and at least 3 accepted runs."
            ),
            "fit_config": {
                "direction": "charge",
                "data_length": 500,
                "weight_ocv": 100.0,
                "weight_dva": 0.0,
                "weight_ica": 0.0,
                "speed_preset": "fast",
                "req_accepted": 3,
                "max_tries_overall": 10,
                "rmse_threshold_v": 0.015,
                "random_seed_base": SEED,
                "seed_increment": "base seed + optimizer attempt index starting at zero",
                "allow_first_cycle_inhomogeneity": False,
                "maxiter": OPTIMIZER_OPTIONS["maxiter"],
                "popsize": OPTIMIZER_OPTIONS["popsize"],
                "polish": True,
                "algorithm": "differential_evolution best1bin; existing PyTorch CUDA backend",
                "reference_alignment": "same-cell charge BoL fit and measured capacity spans",
            },
            "parameter_and_run_state": (
                "Best fitted parameters, parameter standard deviation, per-attempt "
                "parameters/seed/RMSE/status, and fitted curves saved. PyDMA does "
                "not expose a resumable internal optimizer population state."
            ),
        },
        "runtime": {
            "started_at": run_started_at,
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "elapsed_seconds_total_fit": sum(
                float(row.get("runtime_seconds", 0) or 0) for row in fit_rows
            ),
            "elapsed_seconds_total_run": time.perf_counter() - run_started_perf,
        },
        "input_source_hashes": source_hashes,
        "saved_validation_source_hash_manifest": {
            "path": VALIDATION_RUN_MANIFEST.relative_to(ROOT).as_posix(),
            "matches_current_sources": True,
        },
        "final2_source_manifest": {
            "path": SOURCE_HASH_MANIFEST.relative_to(ROOT).as_posix(),
            "manifest_source_hashes_checked": True,
            "input_sha256_snapshot": source_manifest,
            "validation_run_manifest_sha256": _sha256(VALIDATION_RUN_MANIFEST),
            "validation_run_manifest_source_entries": len(
                validation_manifest.get("input_sha256", {})
            ),
        },
        "historical_charge_fit_audit": {
            "result_files_searched": [
                path.relative_to(ROOT).as_posix()
                for path in EXISTING_RESULT_FILES
                if path.is_file()
            ],
            "exact_identity_charge_fits_found": 0,
            "reuse_required": False,
        },
        "training_authorized": False,
        "neural_network_checkpoint_created": False,
        "output_artifact_sha256": output_hashes,
    }
    _write_json(OUTPUT / "charge_fit_manifest.json", manifest)
    all_artifacts = [
        path for path in OUTPUT.iterdir()
        if path.is_file() and path.name != "artifact_sha256_manifest.json"
    ]
    artifact_manifest = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "hash_algorithm": "SHA-256",
        "artifacts": {
            path.name: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(all_artifacts, key=lambda item: item.name)
        },
        "manifest_includes_itself": False,
    }
    _write_json(OUTPUT / "artifact_sha256_manifest.json", artifact_manifest)
    return quality


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, ensure_ascii=False))
