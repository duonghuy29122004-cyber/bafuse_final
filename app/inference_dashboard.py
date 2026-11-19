from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import shutil
import sys
import tempfile
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from bafuse.hybrid.soh_fusion import (  # noqa: E402
    INPUT_SCHEMA_VERSION,
    MODEL_VERSION,
    PHYSICS_FEATURE_NAMES,
    ThreeEncoderSoH,
)
from app.dashboard_page import HTML_PAGE as SINGLE_PAGE_HTML
from app.tpg_mat_ingestion import (
    MatlabRuntimeUnavailable,
    TpgIngestionError,
    analyze_tpg_file,
    inspect_tpg_file,
)
from app.model_runtime import Final2ModelRuntime, MODEL_NAMES
from app.upload_pydma import UploadedCurveError, run_uploaded_pydma


CHECKPOINT_PATH = (
    ROOT
    / "experiments"
    / "soh_forecast"
    / "bafuse_three_encoder_v1"
    / "fusion_checkpoint.pt"
)
FEATURE_DIR = ROOT / "experiments" / "soh_forecast" / "bafuse_three_encoder_v1"
H2_V3_PATH = ROOT / "experiments" / "pydma_hybrid_direction_v3" / "h2_hybrid_v3.csv"
H2_V4_CANDIDATE_DIR = ROOT / "experiments" / "pydma_hybrid_direction_v4_candidate"
CURVE_POINTS = 128
IDENTITY_COLUMNS = ("part", "group", "physical_cell_id", "rpt_index")
UPLOAD_METADATA_COLUMNS = (*IDENTITY_COLUMNS, "partition")
MODEL_INPUT_COLUMNS = (
    *(f"step4_voltage_progress_{index:03d}" for index in range(CURVE_POINTS)),
    *(f"step10_voltage_progress_{index:03d}" for index in range(CURVE_POINTS)),
    *PHYSICS_FEATURE_NAMES,
)
UPLOAD_COLUMNS = (*UPLOAD_METADATA_COLUMNS, *MODEL_INPUT_COLUMNS)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_RAW_TPG_CSV_UPLOAD_BYTES = 1024 * 1024 * 1024
MAX_RAW_TPG_MAT_UPLOAD_BYTES = 100 * 1024 * 1024
TRAINING_LABEL_FIELDS = {
    "soh_percent",
    "target",
    "label",
    "label_soh",
    "reference_soh_percent",
}


class UploadValidationError(ValueError):
    pass


class LegacyCheckpointUnavailable(RuntimeError):
    pass


class InferenceEngine:
    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        self.final2_runtime = Final2ModelRuntime()
        self._legacy_checkpoint_path = checkpoint_path
        self._legacy_loaded = False
        self.model: ThreeEncoderSoH | None = None
        self.scalers: dict[str, np.ndarray] = {}
        self.target_mean = math.nan
        self.target_scale = math.nan
        self.training_cells: tuple[Any, ...] = ()
        self.checkpoint_sha256: str | None = None

    def legacy_status(self) -> dict[str, Any]:
        if not self._legacy_checkpoint_path.is_file():
            return {
                "available": False,
                "reason": f"legacy_checkpoint_missing:{self._legacy_checkpoint_path}",
            }
        return {"available": True, "reason": "legacy_checkpoint_not_loaded"}

    def _ensure_legacy_loaded(self) -> None:
        if self._legacy_loaded:
            return
        if not self._legacy_checkpoint_path.is_file():
            raise LegacyCheckpointUnavailable(
                "Legacy /api/infer requires the unrelated three-encoder checkpoint, "
                f"which is missing: {self._legacy_checkpoint_path}. "
                "Use /api/analyze or select a Final2 model instead."
            )
        try:
            payload = torch.load(
                self._legacy_checkpoint_path, map_location="cpu", weights_only=False
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise LegacyCheckpointUnavailable(
                f"Legacy /api/infer checkpoint could not be loaded: {error}"
            ) from error
        if payload.get("model_version") != MODEL_VERSION:
            raise ValueError("Frozen checkpoint model version does not match the inference code.")
        if payload.get("input_schema_version") != INPUT_SCHEMA_VERSION:
            raise ValueError("Frozen checkpoint input schema does not match the inference code.")
        if tuple(payload.get("physics_feature_names", ())) != PHYSICS_FEATURE_NAMES:
            raise ValueError("Frozen checkpoint physics feature order is incompatible.")

        model = ThreeEncoderSoH()
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        self.model = model
        scalers = {}
        for name, length in (
            ("discharge_mean", CURVE_POINTS),
            ("discharge_scale", CURVE_POINTS),
            ("charge_mean", CURVE_POINTS),
            ("charge_scale", CURVE_POINTS),
            ("physics_mean", len(PHYSICS_FEATURE_NAMES)),
            ("physics_scale", len(PHYSICS_FEATURE_NAMES)),
        ):
            values = np.asarray(payload[name], dtype=np.float32).reshape(-1)
            if values.size != length or not np.isfinite(values).all():
                raise ValueError(f"Frozen checkpoint has invalid {name}.")
            if name.endswith("_scale") and np.any(values <= 0):
                raise ValueError(f"Frozen checkpoint has non-positive {name}.")
            scalers[name] = values

        self.target_mean = float(payload["target_mean"])
        self.target_scale = float(payload["target_scale"])
        if not math.isfinite(self.target_mean) or not math.isfinite(self.target_scale):
            raise ValueError("Frozen checkpoint has non-finite target scaling.")
        if self.target_scale <= 0:
            raise ValueError("Frozen checkpoint has non-positive target scaling.")
        self.training_cells = tuple(payload.get("training_cells", ()))
        self.scalers = scalers
        self.checkpoint_sha256 = _sha256(self._legacy_checkpoint_path)
        self._legacy_loaded = True

    def predict_for_model(
        self, rows: list[dict[str, Any]], model_name: str | None = None
    ) -> dict[str, Any]:
        return self.final2_runtime.preprocess_rows(rows, model_name)

    def preprocess(self, rows: list[dict[str, str]]) -> dict[str, np.ndarray]:
        self._ensure_legacy_loaded()
        discharge = np.asarray(
            [
                [float(row[name]) for name in MODEL_INPUT_COLUMNS[:CURVE_POINTS]]
                for row in rows
            ],
            dtype=np.float32,
        )
        charge = np.asarray(
            [
                [
                    float(row[name])
                    for name in MODEL_INPUT_COLUMNS[
                        CURVE_POINTS : 2 * CURVE_POINTS
                    ]
                ]
                for row in rows
            ],
            dtype=np.float32,
        )
        physics = np.asarray(
            [
                [float(row[name]) for name in PHYSICS_FEATURE_NAMES]
                for row in rows
            ],
            dtype=np.float32,
        )
        if not (
            np.isfinite(discharge).all()
            and np.isfinite(charge).all()
            and np.isfinite(physics).all()
        ):
            raise UploadValidationError("Model input contains a non-finite value.")

        return {
            "step4_voltage_normalized": (
                discharge - self.scalers["discharge_mean"]
            )
            / self.scalers["discharge_scale"],
            "step10_voltage_normalized": (
                charge - self.scalers["charge_mean"]
            )
            / self.scalers["charge_scale"],
            "physics_features_normalized": (
                physics - self.scalers["physics_mean"]
            )
            / self.scalers["physics_scale"],
        }

    def predict(self, rows: list[dict[str, str]]) -> list[dict[str, Any]]:
        self._ensure_legacy_loaded()
        inputs = self.preprocess(rows)
        with torch.inference_mode():
            prediction = self.model(
                torch.from_numpy(inputs["step4_voltage_normalized"]),
                torch.from_numpy(inputs["step10_voltage_normalized"]),
                torch.from_numpy(inputs["physics_features_normalized"]),
            ).reshape(-1)
        values = (
            prediction.numpy().astype(np.float64) * self.target_scale
            + self.target_mean
        )
        results = []
        for row, value in zip(rows, values):
            if not math.isfinite(float(value)):
                raise ValueError("The frozen model returned a non-finite SoH estimate.")
            part = str(row["part"]).strip()
            results.append(
                {
                    "part": part,
                    "group": str(row["group"]).strip(),
                    "physical_cell_id": str(row["physical_cell_id"]).strip(),
                    "rpt_index": str(row["rpt_index"]).strip(),
                    "partition": str(row["partition"]).strip(),
                    "target_relationship": "same-time SoH estimation",
                    "evaluation_context": (
                        "external/domain-shift input; no reference label supplied"
                        if part == "3"
                        else "uploaded input; partition is user-provided metadata"
                    ),
                    "predicted_soh_percent": float(value),
                    "prediction_range_status": (
                        "within_0_to_100"
                        if 0 <= float(value) <= 100
                        else "outside_0_to_100_unclipped"
                    ),
                    "model_version": MODEL_VERSION,
                    "input_provenance": (
                        "user-uploaded prepared feature row; upstream source mapping "
                        "is not revalidated by this dashboard"
                    ),
                }
            )
        return results


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_upload_csv(text: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(text))
        headers = reader.fieldnames or []
        if len(headers) != len(set(headers)):
            raise UploadValidationError("CSV contains duplicate column names.")
        missing = set(UPLOAD_COLUMNS).difference(headers)
        if missing:
            raise UploadValidationError(
                "CSV is missing required columns: " + ", ".join(sorted(missing))
            )
        target_columns = TRAINING_LABEL_FIELDS.intersection(headers)
        if target_columns:
            raise UploadValidationError(
                "Do not include target/label columns in inference uploads: "
                + ", ".join(sorted(target_columns))
            )
        unexpected = set(headers).difference(UPLOAD_COLUMNS)
        if unexpected:
            raise UploadValidationError(
                "CSV has unsupported columns: " + ", ".join(sorted(unexpected))
            )

        rows = list(reader)
    except csv.Error as error:
        raise UploadValidationError(f"Could not parse CSV: {error}") from error
    if not rows:
        raise UploadValidationError("CSV must contain at least one data row.")
    if len(rows) > 500:
        raise UploadValidationError("At most 500 observations may be uploaded at once.")

    seen: set[tuple[str, ...]] = set()
    for row_number, row in enumerate(rows, start=2):
        if None in row:
            raise UploadValidationError(f"CSV row {row_number} has extra values.")
        if any(not str(row.get(name, "")).strip() for name in UPLOAD_METADATA_COLUMNS):
            raise UploadValidationError(
                f"CSV row {row_number} is missing required identity metadata."
            )
        if str(row["partition"]).strip() not in {
            "train",
            "validation",
            "external_test",
            "unassigned",
        }:
            raise UploadValidationError(
                f"CSV row {row_number} has an unsupported partition value."
            )
        key = tuple(str(row[name]).strip() for name in IDENTITY_COLUMNS)
        if key in seen:
            raise UploadValidationError(
                f"Duplicate observation key in uploaded CSV at row {row_number}: {key}"
            )
        seen.add(key)
        for name in MODEL_INPUT_COLUMNS:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as error:
                raise UploadValidationError(
                    f"CSV row {row_number} has a non-numeric value in {name}."
                ) from error
            if not math.isfinite(value):
                raise UploadValidationError(
                    f"CSV row {row_number} has a non-finite value in {name}."
                )
    return rows


def _normalized_key(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["part"].astype(str).str.strip()
        + "|"
        + frame["physical_cell_id"].astype(str).str.strip()
        + "|"
        + pd.to_numeric(frame["rpt_index"], errors="raise")
        .astype(int)
        .astype(str)
    )


def _truth(value: Any) -> bool:
    if value is True or isinstance(value, np.bool_) and bool(value):
        return True
    return str(value).strip().casefold() in {"true", "1", "yes"}


def build_sample_upload(feature_dir: Path = FEATURE_DIR) -> str:
    discharge = pd.read_csv(feature_dir / "discharge_features.csv", low_memory=False)
    charge = pd.read_csv(feature_dir / "charge_features.csv", low_memory=False)
    physics = pd.read_csv(feature_dir / "physics_features.csv", low_memory=False)
    frames = [discharge, charge, physics]
    for frame in frames:
        frame["_key"] = _normalized_key(frame)
        if frame["_key"].duplicated().any():
            raise ValueError("Saved feature data contains duplicate sample keys.")
    merged = (
        frames[0]
        .merge(
            frames[1][["_key", *[
                column for column in frames[1].columns
                if column.startswith("voltage_progress_")
            ]]],
            on="_key",
            how="inner",
            validate="one_to_one",
            suffixes=("_step4", "_step10"),
        )
        .merge(
            frames[2][["_key", *PHYSICS_FEATURE_NAMES]],
            on="_key",
            how="inner",
            validate="one_to_one",
        )
    )
    if merged.empty:
        raise ValueError("No complete saved feature observation can be used as a sample.")
    external = merged[
        merged["part"].astype(str).eq("3")
        & merged["partition"].astype(str).eq("external_test")
    ]
    selected = external.iloc[0] if not external.empty else merged.iloc[0]
    output: dict[str, Any] = {
        "part": selected["part"],
        "group": selected["group"],
        "physical_cell_id": selected["physical_cell_id"],
        "rpt_index": int(selected["rpt_index"]),
        "partition": selected["partition"],
    }
    for index in range(CURVE_POINTS):
        name = f"voltage_progress_{index:03d}"
        output[f"step4_{name}"] = selected[f"{name}_step4"]
        output[f"step10_{name}"] = selected[f"{name}_step10"]
    for name in PHYSICS_FEATURE_NAMES:
        output[name] = selected[name]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=UPLOAD_COLUMNS)
    writer.writeheader()
    writer.writerow(output)
    return stream.getvalue()


def lookup_h2_v3(payload: dict[str, Any], path: Path = H2_V3_PATH) -> list[dict[str, Any]]:
    required = ("part", "group", "physical_cell_id", "checkpoint")
    missing = [name for name in required if not str(payload.get(name, "")).strip()]
    if missing:
        raise UploadValidationError(
            "Lookup requires: " + ", ".join(missing)
        )
    if not path.is_file():
        raise FileNotFoundError(f"Saved H2 v3 results are missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    mask = (
        frame["part"].astype(str).eq(str(payload["part"]).strip())
        & frame["group"].astype(str).eq(str(payload["group"]).strip())
        & frame["physical_cell_id"].astype(str).eq(
            str(payload["physical_cell_id"]).strip()
        )
    )
    checkpoint = str(payload["checkpoint"]).strip()
    mask &= (
        frame["checkpoint_id"].fillna("").astype(str).eq(checkpoint)
        | frame["rpt_or_checkpoint"].fillna("").astype(str).str.casefold().eq(
            checkpoint.casefold()
        )
        | pd.to_numeric(frame["rpt_index"], errors="coerce").eq(
            pd.to_numeric(pd.Series([checkpoint]), errors="coerce").iloc[0]
        )
    )
    columns = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_or_checkpoint",
        "checkpoint_id",
        "is_eol",
        "metric",
        "charge_accepted",
        "charge_rmse_mV",
        "charge_accepted_runs",
        "charge_LLI_percent",
        "charge_Anode_LAM_percent",
        "charge_Cathode_LAM_percent",
        "discharge_accepted",
        "discharge_rmse_mV",
        "discharge_accepted_runs",
        "discharge_LLI_percent",
        "discharge_Anode_LAM_percent",
        "discharge_Cathode_LAM_percent",
        "h2_v3_primary_direction",
        "h2_v3_selected_percent",
        "h2_v3_selected_accepted",
        "h2_v3_source_direction",
        "h2_v3_selected_fit_rmse_mV",
        "h2_v3_selection_reason",
    ]
    found = frame.loc[mask, columns].copy()
    return json.loads(found.to_json(orient="records"))


def lookup_h2_v3_trajectory(
    payload: dict[str, Any], path: Path = H2_V3_PATH
) -> list[dict[str, Any]]:
    required = ("part", "group", "physical_cell_id", "metric")
    missing = [name for name in required if not str(payload.get(name, "")).strip()]
    if missing:
        raise UploadValidationError("Trajectory lookup requires: " + ", ".join(missing))
    metric = str(payload["metric"]).strip()
    if metric not in {"LLI", "Anode_LAM", "Cathode_LAM"}:
        raise UploadValidationError("Metric must be LLI, Anode_LAM, or Cathode_LAM.")
    if not path.is_file():
        raise FileNotFoundError(f"Saved H2 v3 results are missing: {path}")
    frame = pd.read_csv(path, low_memory=False)
    mask = (
        frame["part"].astype(str).eq(str(payload["part"]).strip())
        & frame["group"].astype(str).eq(str(payload["group"]).strip())
        & frame["physical_cell_id"].astype(str).eq(
            str(payload["physical_cell_id"]).strip()
        )
        & frame["metric"].eq(metric)
    )
    include_eol = str(payload.get("include_eol", "false")).strip().casefold() in {
        "true",
        "1",
        "yes",
    }
    if not include_eol:
        mask &= ~frame["is_eol"].map(_truth)
    columns = [
        "part",
        "group",
        "physical_cell_id",
        "rpt_index",
        "rpt_or_checkpoint",
        "checkpoint_id",
        "is_eol",
        "metric",
        "h2_v3_primary_direction",
        "h2_v3_selected_percent",
        "h2_v3_selected_accepted",
        "h2_v3_source_direction",
        "h2_v3_selected_fit_rmse_mV",
        "h2_v3_selection_reason",
        "temporal_decrease_warning",
        "temporal_delta_from_previous_pp",
    ]
    found = frame.loc[mask, columns].copy()
    found["_sort_rpt"] = pd.to_numeric(found["rpt_index"], errors="coerce")
    found["_sort_position"] = pd.to_numeric(
        frame.loc[mask, "chronological_position"], errors="coerce"
    )
    found = found.sort_values(
        ["_sort_position", "_sort_rpt", "checkpoint_id"],
        na_position="last",
        kind="stable",
    ).drop(columns=["_sort_rpt", "_sort_position"])
    return json.loads(found.to_json(orient="records"))


def load_h2_v4_candidate_audit(directory: Path = H2_V4_CANDIDATE_DIR) -> dict[str, Any]:
    required = (
        directory / "undetermined_policy_audit.csv",
        directory / "coverage_comparison.csv",
        directory / "run_manifest.json",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Saved H2 v4 candidate audit is incomplete: " + ", ".join(missing)
        )
    policies = pd.read_csv(required[0], low_memory=False)
    coverage = pd.read_csv(required[1], low_memory=False)
    overall_coverage = coverage.loc[coverage["metric"].eq("OVERALL")]
    manifest = json.loads(required[2].read_text(encoding="utf-8"))
    return {
        "candidate_status": "experimental_not_dashboard_default",
        "counts": manifest["counts"],
        "policies": json.loads(policies.to_json(orient="records")),
        "overall_coverage": json.loads(overall_coverage.to_json(orient="records")),
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    engine: InferenceEngine

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[dashboard] {self.address_string()} {format % args}")

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload: Any) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise UploadValidationError("Invalid Content-Length header.") from error
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise UploadValidationError("Request body must be between 1 byte and 5 MB.")
        try:
            payload = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise UploadValidationError("Request body must be valid JSON.") from error
        if not isinstance(payload, dict):
            raise UploadValidationError("Request JSON must be an object.")
        return payload

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(
                HTTPStatus.OK,
                SINGLE_PAGE_HTML.encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return
        if path == "/api/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if path == "/api/status":
            self._send_json(
                HTTPStatus.OK,
                {
                    "model_version": MODEL_VERSION,
                    "input_schema_version": INPUT_SCHEMA_VERSION,
                    "frozen_inference_only": True,
                    "mat_upload_supported": shutil.which("matlab") is not None,
                    "raw_csv_upload_supported": True,
                    "new_pydma_fit_supported": True,
                    "pydma_status": (
                        "A new upload-only diagnostic fit is available from a BoL/target "
                        "curve CSV bundle. Its Guide/procedure/Step identity remains "
                        "user-supplied and unverified."
                    ),
                    "selected_model": self.engine.final2_runtime.default_model(),
                    "model_device": str(self.engine.final2_runtime.device),
                    "legacy_inference": self.engine.legacy_status(),
                },
            )
            return
        if path == "/api/models":
            self._send_json(
                HTTPStatus.OK,
                {
                    "default_model": self.engine.final2_runtime.default_model(),
                    "models": self.engine.final2_runtime.model_catalog(),
                },
            )
            return
        if path == "/api/h2v4-audit":
            try:
                self._send_json(HTTPStatus.OK, load_h2_v4_candidate_audit())
            except (FileNotFoundError, OSError, KeyError, ValueError) as error:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(error).__name__}: {error}"},
                )
            return
        if path == "/sample.csv":
            try:
                body = build_sample_upload().encode("utf-8")
            except (OSError, KeyError, ValueError) as error:
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"Could not construct the saved-feature sample: {error}"},
                )
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header(
                "Content-Disposition",
                'attachment; filename="prepared_model_input_sample.csv"',
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/inspect":
                self._inspect_tpg_file()
            elif path == "/api/analyze":
                self._analyze_tpg_file()
            elif path == "/api/infer":
                self._infer()
            elif path == "/api/lookup":
                rows = lookup_h2_v3(self._read_json())
                self._send_json(HTTPStatus.OK, {"results": rows})
            elif path == "/api/trajectory":
                rows = lookup_h2_v3_trajectory(self._read_json())
                self._send_json(HTTPStatus.OK, {"results": rows})
            elif path == "/api/pydma":
                self._pydma()
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
        except UploadValidationError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except UploadedCurveError as error:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
        except MatlabRuntimeUnavailable as error:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        except LegacyCheckpointUnavailable as error:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(error)})
        except TpgIngestionError as error:
            self._send_json(HTTPStatus.UNPROCESSABLE_ENTITY, {"error": str(error)})
        except FileNotFoundError as error:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(error)})
        except (ImportError, OSError, KeyError, ValueError, RuntimeError) as error:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": f"{type(error).__name__}: {error}"},
            )

    def _read_tpg_upload(self, input_path: Path) -> str:
        filename = self.headers.get("X-Upload-Filename", "").strip()
        filename = Path(filename.replace("\\", "/")).name
        extension = Path(filename).suffix.casefold()
        if extension not in {".mat", ".csv"}:
            raise UploadValidationError("Upload one raw TPG .mat or .csv file.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise UploadValidationError("Invalid Content-Length header.") from error
        maximum = (
            MAX_RAW_TPG_MAT_UPLOAD_BYTES
            if extension == ".mat"
            else MAX_RAW_TPG_CSV_UPLOAD_BYTES
        )
        if length <= 0 or length > maximum:
            size_limit = "100 MB" if extension == ".mat" else "1 GiB"
            raise UploadValidationError(
                f"Raw TPG {extension[1:].upper()} upload must be between 1 byte and {size_limit}."
            )
        input_path.parent.mkdir(parents=True, exist_ok=True)
        remaining = length
        with input_path.open("wb") as stream:
            while remaining:
                block = self.rfile.read(min(1024 * 1024, remaining))
                if not block:
                    self.close_connection = True
                    raise UploadValidationError(
                        "The upload ended before the declared Content-Length was received."
                    )
                stream.write(block)
                remaining -= len(block)
        return filename

    def _inspect_tpg_file(self) -> None:
        payload = self._read_json()
        filename = payload.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            raise UploadValidationError("Inspection requires the selected TPG filename.")
        result = inspect_tpg_file(filename)
        self._send_json(HTTPStatus.OK, result)

    def _analyze_tpg_file(self) -> None:
        filename = self.headers.get("X-Upload-Filename", "").strip()
        extension = Path(filename.replace("\\", "/")).suffix.casefold()
        if extension not in {".mat", ".csv"}:
            raise UploadValidationError("Upload one raw TPG .mat or .csv file.")
        with tempfile.TemporaryDirectory(prefix="bafuse-dashboard-upload-") as directory:
            input_path = Path(directory) / f"uploaded{extension}"
            filename = self._read_tpg_upload(input_path)
            model_name = self.headers.get("X-Model-Name", "").strip() or None
            if model_name is not None and model_name not in MODEL_NAMES:
                raise UploadValidationError(
                    f"Unknown model selection: {model_name}"
                )
            result = analyze_tpg_file(
                input_path, filename, self.engine, model_name=model_name
            )
        self._send_json(HTTPStatus.OK, result)

    def _infer(self) -> None:
        filename = self.headers.get("X-Upload-Filename", "").strip()
        if filename.casefold().endswith(".mat"):
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {
                    "error": (
                        "Raw TPG MAT upload is not supported. The inspected TPG MAT "
                        "is a MATLAB 5.0 opaque MCOS object, not a decoded measurement "
                        "table, and this environment has no validated decoder. Upload "
                        "a prepared feature CSV instead; do not substitute a same-stem "
                        "CSV without source-equivalence validation."
                    )
                },
            )
            return
        if not filename.casefold().endswith(".csv"):
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Upload a prepared feature CSV. Other file formats are unsupported."},
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise UploadValidationError("Invalid Content-Length header.") from error
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise UploadValidationError("CSV upload must be between 1 byte and 5 MB.")
        try:
            text = self.rfile.read(length).decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise UploadValidationError("CSV must use UTF-8 encoding.") from error
        rows = parse_upload_csv(text)
        model_name = self.headers.get("X-Model-Name", "").strip()
        if model_name:
            if model_name not in MODEL_NAMES:
                raise UploadValidationError(f"Unknown model selection: {model_name}")
            inference = self.engine.predict_for_model(rows, model_name)
            self._send_json(
                HTTPStatus.OK,
                {
                    "model_name": model_name,
                    "results": inference["results"],
                },
            )
            return
        self._send_json(
            HTTPStatus.OK,
            {
                "model_version": MODEL_VERSION,
                "checkpoint_sha256": self.engine.checkpoint_sha256,
                "results": self.engine.predict(rows),
            },
        )

    def _pydma(self) -> None:
        filename = self.headers.get("X-Upload-Filename", "").strip()
        if filename.casefold().endswith(".mat"):
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Raw TPG MAT upload is unsupported; use a prepared curve CSV bundle."},
            )
            return
        if not filename.casefold().endswith(".csv"):
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "Upload a CSV with both BoL and target curve rows."},
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as error:
            raise UploadValidationError("Invalid Content-Length header.") from error
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            raise UploadValidationError("PyDMA CSV upload must be between 1 byte and 5 MB.")
        try:
            text = self.rfile.read(length).decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise UploadValidationError("PyDMA CSV must use UTF-8 encoding.") from error
        self._send_json(HTTPStatus.OK, run_uploaded_pydma(text))


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BaFuse local inference demo</title>
<style>
body{font:16px/1.5 Segoe UI,Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#17212b;background:#f5f7f9}
section{background:white;border:1px solid #d6dee5;border-radius:8px;padding:1rem;margin:1rem 0}
h1{color:#14324b}label{display:inline-grid;gap:.3rem;margin:.5rem}input,button{padding:.5rem;font:inherit}
button{cursor:pointer;background:#14324b;color:white;border:0;border-radius:4px}table{border-collapse:collapse;width:100%;font-size:.9rem}
th,td{text-align:left;border-bottom:1px solid #d6dee5;padding:.45rem;vertical-align:top}
.warning{background:#fff5df;border-left:4px solid #e29b26;padding:.7rem}.muted{color:#536574}
.error{color:#a22828}.ok{color:#17633a}.scroll{overflow:auto}
</style>
</head>
<body>
<h1>BaFuse — local inference demo</h1>
<p class="muted">Frozen three-encoder same-time SoH estimator and saved PyDMA lookup. Runs on this computer only.</p>
<div id="status" class="muted">Loading checkpoint status…</div>
<section class="warning">
<strong>Data limitations.</strong> The raw TPG MATLAB file inspected is an opaque MATLAB 5.0 MCOS object, not a decoded measurement table. This demo cannot extract its curves; it will reject MAT uploads. Use the prepared-feature sample to demonstrate frozen inference, or upload two raw CSV curves for the explicitly unverified PyDMA diagnostic below. Part 3 remains external/domain-shift.
</section>
<section>
<h2>Upload prepared feature CSV</h2>
<p>CSV must contain 128-point Step 4 and Step 10 voltage-progress curves, the five exact physics features, and cell/checkpoint metadata. Do not include an SOH target. Identity fields are display-only; they are not model inputs. Missing modalities are rejected, not imputed.</p>
<a href="/sample.csv">Download prepared input sample (no labels)</a>
<p><input id="upload" type="file" accept=".csv,.mat"><button id="infer">Run frozen inference</button></p>
<div id="inferError" class="error"></div><div class="scroll"><table><thead id="inferHead"></thead><tbody id="inferBody"></tbody></table></div>
<p><button id="exportInference" disabled>Download inference results CSV</button></p>
</section>
<section>
<h2>Saved Hybrid H2 v3 lookup</h2>
<p>Read-only lookup of saved signed direction fits and H2 v3 selections. No optimizer is rerun and no selection is changed.</p>
<label>Part <input id="part" value="1"></label>
<label>Group <input id="group" value="1"></label>
<label>Physical cell <input id="cell" value="G1-Cell15"></label>
<label>Checkpoint (e.g. RPT1 or exact EoL checkpoint ID) <input id="checkpoint" value="RPT1"></label>
<button id="lookup">Look up saved fits</button>
<div id="lookupError" class="error"></div><div class="scroll"><table><thead id="lookupHead"></thead><tbody id="lookupBody"></tbody></table></div>
<p><button id="exportLookup" disabled>Download lookup CSV</button></p>
</section>
<section>
<h2>Saved H2 v3 trajectory</h2>
<p>Read-only signed selections by verified checkpoint. Missing estimates remain blank and are not interpolated; points are not connected across gaps. EoL checkpoints are excluded unless explicitly included.</p>
<label>Part <input id="trajectoryPart" value="1"></label>
<label>Group <input id="trajectoryGroup" value="1"></label>
<label>Physical cell <input id="trajectoryCell" value="G1-Cell15"></label>
<label>Metric <select id="trajectoryMetric"><option>LLI</option><option>Anode_LAM</option><option>Cathode_LAM</option></select></label>
<label><input id="includeEol" type="checkbox"> Include EoL sensitivity</label>
<button id="loadTrajectory">Load trajectory</button>
<div id="trajectoryError" class="error"></div>
<canvas id="trajectoryCanvas" width="900" height="280" aria-label="Saved H2 v3 degradation estimates"></canvas>
<div class="scroll"><table><thead id="trajectoryHead"></thead><tbody id="trajectoryBody"></tbody></table></div>
<p><button id="exportTrajectory" disabled>Download trajectory CSV</button></p>
</section>
<section>
<h2>Six undetermined H2 v3 policies</h2>
<p>Read-only audit of the separate H2 v4 single-direction availability candidate. It is not the dashboard default; H2 v3 outputs remain unchanged. Candidate coverage is limited to accepted Charge-only G2-Cell8 fits, while G10-Cell2 remains unavailable.</p>
<div id="h2v4Counts" class="muted"></div>
<div class="scroll"><table><thead id="h2v4Head"></thead><tbody id="h2v4Body"></tbody></table></div>
<h3>Coverage by domain and checkpoint scope</h3>
<div class="scroll"><table><thead id="coverageHead"></thead><tbody id="coverageBody"></tbody></table></div>
</section>
<section>
<h2>Optional uploaded-curve PyDMA diagnostic</h2>
<p>CSV bundle must contain both <code>bol</code> and <code>target</code> roles with columns:
<code>role, part, group, physical_cell_id, checkpoint_id, source_file, direction, Step, Amphr, Amps, Volts</code>.
For each curve, include the documented raw Step and current values (Step 8 discharge or Step 10 charge); extraction uses the existing direction/current filter and monotone-capacity segment rule. Both curves must be the same cell and direction, with explicit BoL and target checkpoint IDs. The source procedure may be the same or different; `role` separates the submitted curve segments. Part 2 Groups 5–6 are excluded.</p>
<p class="warning">This diagnostic reruns two PyDMA fits (BoL and target) and may take several minutes. The fit gate remains RMSE &lt; 15 mV and at least 3 accepted runs. Guide/procedure/physical-cell mappings and capacity units are user-supplied and not independently verified. Estimates are model-derived diagnostics, not labels or validated physical degradation. Negative values remain signed.</p>
<p><input id="pydmaUpload" type="file" accept=".csv"><button id="runPydma">Run two-fit PyDMA diagnostic</button></p>
<div id="pydmaError" class="error"></div><pre id="pydmaResult" class="scroll"></pre>
<p><button id="exportPydma" disabled>Download PyDMA diagnostic JSON</button></p>
</section>
<script>
let inferenceRows=[], lookupRows=[], trajectoryRows=[], pydmaResult=null;
const $=id=>document.getElementById(id);
function table(headId,bodyId,rows){
 const head=$(headId), body=$(bodyId); head.replaceChildren(); body.replaceChildren();
 if(!rows.length)return;
 const columns=Object.keys(rows[0]), tr=document.createElement("tr");
 for(const name of columns){const th=document.createElement("th");th.textContent=name;tr.append(th);}
 head.append(tr);
 for(const row of rows){const line=document.createElement("tr");for(const name of columns){const td=document.createElement("td");td.textContent=row[name]??"";line.append(td);}body.append(line);}
}
async function saveCsv(rows,name,button){
 if(!rows.length)return;
 const columns=Object.keys(rows[0]), quote=v=>'"'+String(v??"").replaceAll('"','""')+'"';
 const csv=[columns.map(quote).join(","),...rows.map(r=>columns.map(c=>quote(r[c])).join(","))].join("\r\n");
 const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([csv],{type:"text/csv"}));link.download=name;link.click();URL.revokeObjectURL(link.href);
}
async function status(){
 try{const r=await fetch("/api/status");const s=await r.json();$("status").textContent=`Model: ${s.model_version}; input schema: ${s.input_schema_version}; frozen inference only. Checkpoint SHA-256: ${s.checkpoint_sha256}`;}
 catch(e){$("status").textContent=`Status error: ${e}`;}
}
async function loadH2V4Audit(){
 try{
  const r=await fetch("/api/h2v4-audit"),d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);
  $("h2v4Counts").textContent=`H2 v3 accepted estimates: ${d.counts.h2_v3_selected_estimates}; H2 v4 candidate: ${d.counts.h2_v4_candidate_selected_estimates}; newly selected: ${d.counts.newly_selected_estimates}.`;
  table("h2v4Head","h2v4Body",d.policies);
  const overall=d.overall_coverage.map(row=>({evaluation_domain:row.evaluation_domain,checkpoint_scope:row.checkpoint_scope,h2_v3_selected_estimates:row.h2_v3_selected_estimates,h2_v4_candidate_selected_estimates:row.h2_v4_candidate_selected_estimates,newly_selected_estimates:row.newly_selected_estimates,unavailable_h2_v4_candidate_estimates:row.unavailable_h2_v4_candidate_estimates}));
  table("coverageHead","coverageBody",overall);
 }catch(e){$("h2v4Counts").textContent=`Saved H2 v4 audit unavailable: ${e}`;}
}
$("infer").onclick=async()=>{
 $("inferError").textContent="";const file=$("upload").files[0];if(!file){$("inferError").textContent="Choose a CSV or MAT file.";return;}
 try{const r=await fetch("/api/infer",{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Upload-Filename":file.name},body:file});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);inferenceRows=d.results;table("inferHead","inferBody",inferenceRows);$("exportInference").disabled=!inferenceRows.length;}
 catch(e){$("inferError").textContent=String(e);}
};
$("lookup").onclick=async()=>{
 $("lookupError").textContent="";
 try{const payload={part:$("part").value,group:$("group").value,physical_cell_id:$("cell").value,checkpoint:$("checkpoint").value};const r=await fetch("/api/lookup",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);lookupRows=d.results;table("lookupHead","lookupBody",lookupRows);$("exportLookup").disabled=!lookupRows.length;if(!lookupRows.length)$("lookupError").textContent="No saved H2 v3 rows match this exact checkpoint.";}
 catch(e){$("lookupError").textContent=String(e);}
};
$("loadTrajectory").onclick=async()=>{
 $("trajectoryError").textContent="";
 try{
  const payload={part:$("trajectoryPart").value,group:$("trajectoryGroup").value,physical_cell_id:$("trajectoryCell").value,metric:$("trajectoryMetric").value,include_eol:$("includeEol").checked};
  const r=await fetch("/api/trajectory",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});
  const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);
  trajectoryRows=d.results;table("trajectoryHead","trajectoryBody",trajectoryRows);$("exportTrajectory").disabled=!trajectoryRows.length;
  const canvas=$("trajectoryCanvas"),ctx=canvas.getContext("2d"),left=55,right=20,top=20,bottom=45,width=canvas.width,height=canvas.height;
  ctx.clearRect(0,0,width,height);ctx.strokeStyle="#667";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,height-bottom);ctx.lineTo(width-right,height-bottom);ctx.stroke();
  const plotted=trajectoryRows.map((row,index)=>({row,index,value:Number(row.h2_v3_selected_percent)})).filter(item=>item.row.h2_v3_selected_accepted&&Number.isFinite(item.value));
  if(!plotted.length){ctx.fillStyle="#536574";ctx.fillText("No accepted saved estimates for this trajectory.",left+8,top+22);}
  else{
   const values=plotted.map(item=>item.value),min=Math.min(...values),max=Math.max(...values),pad=(max-min||1)*.08;
   const x=i=>left+i*(width-left-right)/Math.max(trajectoryRows.length-1,1),y=v=>top+(max+pad-v)*(height-top-bottom)/(max-min+2*pad);
   for(const item of plotted){const cx=x(item.index),cy=y(item.value);ctx.fillStyle=item.row.is_eol?"#d77b00":"#17699b";if(item.row.is_eol){ctx.fillRect(cx-5,cy-5,10,10);}else{ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fill();}}
   ctx.fillStyle="#17212b";ctx.fillText(`${min.toFixed(2)}–${max.toFixed(2)}%`,left+8,top+14);
   ctx.fillText(trajectoryRows[0].rpt_or_checkpoint||"",left,height-12);
   ctx.fillText(trajectoryRows[trajectoryRows.length-1].rpt_or_checkpoint||"",width-right-80,height-12);
  }
 }catch(e){$("trajectoryError").textContent=String(e);}
};
$("exportInference").onclick=()=>saveCsv(inferenceRows,"soh_inference_results.csv",$("exportInference"));
$("exportLookup").onclick=()=>saveCsv(lookupRows,"saved_h2_v3_lookup.csv",$("exportLookup"));
$("exportTrajectory").onclick=()=>saveCsv(trajectoryRows,"saved_h2_v3_trajectory.csv",$("exportTrajectory"));
$("runPydma").onclick=async()=>{
 $("pydmaError").textContent="";$("pydmaResult").textContent="";const file=$("pydmaUpload").files[0];if(!file){$("pydmaError").textContent="Choose a BoL/target curve CSV bundle.";return;}
 try{const r=await fetch("/api/pydma",{method:"POST",headers:{"Content-Type":"application/octet-stream","X-Upload-Filename":file.name},body:file});const d=await r.json();if(!r.ok)throw Error(d.error||r.statusText);pydmaResult=d;$("pydmaResult").textContent=JSON.stringify(d,null,2);$("exportPydma").disabled=false;}
 catch(e){$("pydmaError").textContent=String(e);}
};
$("exportPydma").onclick=()=>{if(!pydmaResult)return;const link=document.createElement("a");link.href=URL.createObjectURL(new Blob([JSON.stringify(pydmaResult,null,2)],{type:"application/json"}));link.download="uploaded_pydma_diagnostic.json";link.click();URL.revokeObjectURL(link.href);};
status();
loadH2V4Audit();
</script>
</body>
</html>"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local BaFuse inference demo.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise ValueError("The dashboard is local-only; host must be 127.0.0.1 or localhost.")
    engine = InferenceEngine()
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"engine": engine})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"BaFuse inference demo: http://{args.host}:{args.port}/", flush=True)
    print("Press Ctrl+C to stop the local server.", flush=True)
    serve_thread = threading.Thread(target=server.serve_forever, daemon=True)
    serve_thread.start()
    try:
        if args.open_browser:
            webbrowser.open(f"http://{args.host}:{args.port}/")
        serve_thread.join()
    except KeyboardInterrupt:
        print("\nStopping BaFuse inference demo.", flush=True)
    finally:
        if serve_thread.is_alive():
            server.shutdown()
        server.server_close()
        serve_thread.join()


if __name__ == "__main__":
    main()
