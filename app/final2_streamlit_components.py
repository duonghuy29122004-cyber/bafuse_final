from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, MutableMapping

from app.model_runtime import MODEL_NAMES
from benchmark import MODELS
from app.tpg_mat_ingestion import (
    TpgIngestionError,
    frozen_soh_label_for_identity,
    guide_mapping_for_file,
)


MODEL_OPTIONS = MODELS
DEFAULT_MODEL = "TCN-2M"
MAX_CSV_UPLOAD_BYTES = 1024 * 1024 * 1024
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


class DashboardInputError(ValueError):
    pass


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _filename(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).name


def _rpt_key(value: Any) -> str:
    match = re.fullmatch(r"(?:RPT\s*)?(\d+)", str(value).strip(), re.IGNORECASE)
    return str(int(match.group(1))) if match else str(value).strip().casefold()


def _same_source_file(source: Any, expected_source: Any) -> bool:
    source_path = str(source).strip().replace("\\", "/").casefold().lstrip("./")
    expected_path = (
        str(expected_source).strip().replace("\\", "/").casefold().lstrip("./")
    )
    return bool(
        source_path
        and expected_path
        and (
            source_path == expected_path
            or source_path.endswith("/" + expected_path)
            or expected_path.endswith("/" + source_path)
        )
    )


def validate_csv_upload(filename: str, size_bytes: int) -> str:
    safe_name = _filename(filename)
    if Path(safe_name).suffix.casefold() != ".csv":
        raise DashboardInputError("Upload one raw RPT CSV file; other formats are unsupported.")
    if size_bytes <= 0 or size_bytes > MAX_CSV_UPLOAD_BYTES:
        raise DashboardInputError("CSV upload must be between 1 byte and 1 GiB.")
    return safe_name


def resolve_single_rpt(
    filename: str,
    mapping: dict[str, Any],
    observations: list[dict[str, Any]],
    rpt_index: str | None = None,
) -> dict[str, Any]:
    safe_name = validate_csv_upload(filename, 1)
    if rpt_index is None:
        selected_observations = observations
    else:
        selected_key = _rpt_key(rpt_index)
        selected_observations = [
            observation
            for observation in observations
            if _rpt_key(observation.get("rpt_index", "")) == selected_key
        ]
    if len(selected_observations) != 1:
        raise DashboardInputError(
            "This source maps to "
            f"{len(observations)} RPT observations; select exactly one mapped RPT."
        )
    if str(mapping.get("source_present", "")).casefold() != "true":
        raise DashboardInputError("The mapped raw source is not present in the verified inventory.")
    expected_source = str(mapping.get("repository_path", "")).strip()
    expected_hash = str(mapping.get("source_sha256", "")).strip()
    if not expected_source or _filename(expected_source).casefold() != safe_name.casefold():
        raise DashboardInputError("Uploaded filename does not match the verified source mapping.")
    if not _SHA256_RE.fullmatch(expected_hash):
        raise DashboardInputError("The source mapping has no valid SHA-256 provenance.")

    observation = selected_observations[0]
    identity = {
        "part": str(observation.get("part", mapping.get("dataset_part", ""))).strip(),
        "group": str(observation.get("group", mapping.get("group_id", ""))).strip(),
        "physical_cell_id": str(
            observation.get("physical_cell_id", mapping.get("physical_cell_id", ""))
        ).strip(),
        "rpt_index": str(observation.get("rpt_index", "")).strip(),
    }
    if any(not value for value in identity.values()):
        raise DashboardInputError("The Guide mapping does not resolve a complete RPT identity.")
    if (
        identity["part"] != str(mapping.get("dataset_part", "")).strip()
        or identity["group"] != str(mapping.get("group_id", "")).strip()
        or identity["physical_cell_id"]
        != str(mapping.get("physical_cell_id", "")).strip()
        or _filename(observation.get("source_file", "")).casefold()
        != safe_name.casefold()
    ):
        raise DashboardInputError("Guide mapping and RPT identity/source provenance disagree.")
    if (
        str(mapping.get("is_eol_procedure", "false")).casefold() == "true"
        or str(mapping.get("inventory_scope", "")).casefold() == "verified_demo_only"
    ):
        raise DashboardInputError(
            "This upload is a demo-only EoL procedure, not a saved RPT observation."
        )
    return {
        **identity,
        "source_file": expected_source,
        "source_sha256": expected_hash,
        "source_sha256_provenance": str(
            mapping.get("source_sha256_provenance", "")
        ),
        "procedure": str(mapping.get("procedure", "")),
        "mapping_status": str(
            mapping.get("mapping_resolution_status")
            or mapping.get("guide_source_mapping_status", "")
        ),
    }


def resolve_uploaded_rpts(filename: str) -> list[dict[str, Any]]:
    safe_name = validate_csv_upload(filename, 1)
    try:
        mapping, observations = guide_mapping_for_file(
            safe_name,
            require_source_sha256=True,
        )
    except (TpgIngestionError, FileNotFoundError) as error:
        raise DashboardInputError(str(error)) from error
    return [
        resolve_single_rpt(safe_name, mapping, [observation])
        for observation in observations
    ]


def resolve_uploaded_rpt(
    filename: str, rpt_index: str | None = None
) -> dict[str, Any]:
    resolved = resolve_uploaded_rpts(filename)
    if rpt_index is None:
        if len(resolved) != 1:
            raise DashboardInputError(
                f"This source maps to {len(resolved)} RPT observations; select an RPT."
            )
        return resolved[0]
    selected_key = _rpt_key(rpt_index)
    selected = [
        identity
        for identity in resolved
        if _rpt_key(identity["rpt_index"]) == selected_key
    ]
    if len(selected) != 1:
        raise DashboardInputError(
            f"Selected RPT{rpt_index} is not a unique observation in this Guide mapping."
        )
    return selected[0]


def _identity_matches(row: dict[str, Any], identity: dict[str, Any]) -> bool:
    row_identity = {
        "part": str(row.get("part", "")).strip(),
        "group": str(row.get("group_id", row.get("group", ""))).strip(),
        "physical_cell_id": str(
            row.get("physical_cell_id", row.get("cell_id", ""))
        ).strip(),
        "rpt_index": str(row.get("rpt_index", "")).strip(),
    }
    return row_identity == {
        name: str(identity[name]).strip()
        for name in ("part", "group", "physical_cell_id", "rpt_index")
    }


def _verified_label(
    row: dict[str, Any], identity: dict[str, Any]
) -> tuple[float | None, str, dict[str, Any]]:
    reference = row.get("reference_soh")
    if not isinstance(reference, dict):
        return None, "saved_label_unavailable", {}
    value = _finite_number(reference.get("value_percent"))
    status = str(reference.get("status", "saved_label_unavailable"))
    source = str(reference.get("source_file", "")).strip()
    if value is None:
        return None, status, reference
    if status != "calculated_from_measured_capacity":
        return None, f"label_status_not_approved:{status}", reference
    if not _same_source_file(source, identity["source_file"]):
        return None, "label_source_does_not_match_verified_rpt", reference
    source_hash = str(
        reference.get("source_sha256")
        or reference.get("source_file_sha256")
        or ""
    ).strip()
    expected_hash = str(identity.get("source_sha256", "")).strip()
    if source_hash and source_hash.casefold() != expected_hash.casefold():
        return None, "label_source_sha256_does_not_match_verified_rpt", reference
    return value, status, reference


def _verified_degradation(
    row: dict[str, Any], identity: dict[str, Any]
) -> tuple[dict[str, float | None], dict[str, Any]]:
    raw = row.get("charge_degradation")
    if not isinstance(raw, dict):
        raw = {}
    source = str(raw.get("charge_source_file") or raw.get("source_file") or "").strip()
    accepted_status = str(raw.get("status", "")).startswith("accepted_charge_")
    source_matches = (
        bool(source)
        and _same_source_file(source, identity["source_file"])
    )
    verified = (
        accepted_status
        and str(raw.get("source_direction", "")).casefold() == "charge"
        and source_matches
    )
    values = {
        "LLI (%)": _finite_number(raw.get("LLI_percent")) if verified else None,
        "LAM Anode (%)": (
            _finite_number(raw.get("Anode_LAM_percent")) if verified else None
        ),
        "LAM Cathode (%)": (
            _finite_number(raw.get("Cathode_LAM_percent")) if verified else None
        ),
    }
    if verified and any(value is None for value in values.values()):
        verified = False
        values = {name: None for name in values}
    details = {
        "status": raw.get("status", "saved_charge_fit_unavailable"),
        "reason": raw.get("reason", ""),
        "source_file": source or None,
        "same_cell_bol_source": raw.get("same_cell_bol_source"),
        "same_cell_bol_status": raw.get("same_cell_bol_status"),
        "fit_rmse_mV": _finite_number(raw.get("fit_rmse_mV")),
        "accepted_runs": raw.get("accepted_runs"),
        "verified_saved_charge_result": verified,
    }
    if not verified and not details["reason"]:
        details["reason"] = "No accepted saved Charge fit with matching source provenance."
    return values, details


def build_prediction_result(
    analysis: dict[str, Any],
    identity: dict[str, Any],
    model_name: str,
    runtime: Any,
) -> dict[str, Any]:
    if model_name not in MODEL_OPTIONS or model_name not in MODEL_NAMES:
        raise DashboardInputError(f"Unsupported model selection: {model_name}")
    if analysis.get("selected_model") != model_name:
        raise DashboardInputError("Analysis result belongs to a different selected model.")
    rows = analysis.get("rpts")
    if not isinstance(rows, list) or len(rows) != 1:
        raise DashboardInputError("Analysis did not return exactly one RPT observation.")
    row = rows[0]
    if not _identity_matches(row, identity):
        raise DashboardInputError("Analyzed RPT identity differs from the verified Guide mapping.")
    if not _same_source_file(row.get("source_file", ""), identity["source_file"]):
        raise DashboardInputError("Analyzed source file differs from the verified mapping.")

    reference_row = row
    provenance = analysis.get("provenance", {})
    if isinstance(provenance, dict) and provenance.get("source_checksum_verified") is True:
        analysis_hash = str(provenance.get("source_file_sha256", "")).strip()
        identity_hash = str(identity.get("source_sha256", "")).strip()
        if not analysis_hash or analysis_hash.casefold() != identity_hash.casefold():
            raise DashboardInputError(
                "Analyzed source SHA-256 differs from the verified upload mapping."
            )
        frozen_reference = frozen_soh_label_for_identity(identity)
        if frozen_reference is not None:
            reference_row = {"reference_soh": frozen_reference}
    label, label_status, label_details = _verified_label(reference_row, identity)
    prediction = _finite_number(row.get("predicted_soh_percent"))
    prediction_status = str(row.get("soh_status", "prediction_unavailable"))
    prediction_reason = str(
        row.get("model_prediction", {}).get("reason", "")
        if isinstance(row.get("model_prediction"), dict)
        else ""
    )
    absolute_error = (
        abs(prediction - label)
        if prediction is not None and label is not None
        else None
    )
    inputs = row.get("inputs", {})
    step4 = inputs.get("step4", {}) if isinstance(inputs, dict) else {}
    step10 = inputs.get("step10", {}) if isinstance(inputs, dict) else {}
    step4_curve = [
        _finite_number(value)
        for value in step4.get("processed_128_voltage", [])
    ]
    step10_curve = [
        _finite_number(value)
        for value in step10.get("processed_128_voltage", [])
    ]
    if len(step4_curve) != 128 or any(value is None for value in step4_curve):
        step4_curve = []
    if len(step10_curve) != 128 or any(value is None for value in step10_curve):
        step10_curve = []

    degradation, degradation_details = _verified_degradation(row, identity)
    validation = runtime.validation_metrics.get(model_name)
    external = runtime.external_metrics.get(model_name)
    return {
        "identity": dict(identity),
        "model_name": model_name,
        "step4_curve": step4_curve,
        "step10_curve": step10_curve,
        "step4_status": step4.get("status", "unavailable"),
        "step10_status": step10.get("status", "unavailable"),
        "predicted_soh_percent": prediction,
        "prediction_status": prediction_status,
        "prediction_reason": prediction_reason,
        "label_soh_percent": label,
        "label_status": label_status,
        "label_details": label_details,
        "mae_percentage_points": absolute_error,
        "rmse_percentage_points": absolute_error,
        "validation_metrics": dict(validation) if isinstance(validation, dict) else None,
        "external_metrics": dict(external) if isinstance(external, dict) else None,
        "degradation": degradation,
        "degradation_details": degradation_details,
        "checkpoint_sha256": (
            row.get("model_prediction", {}).get("checkpoint_sha256")
            if isinstance(row.get("model_prediction"), dict)
            else None
        ),
        "dataset_identifier": analysis.get("provenance", {}).get(
            "dataset_identifier", ""
        ),
    }


def clear_battery_result(state: MutableMapping[str, Any]) -> None:
    if state.get("rag_battery_context") is not None:
        state.pop("rag_messages", None)
        state.pop("rag_retrieved_chunks", None)
    for key in (
        "battery_result",
        "battery_result_signature",
        "rag_battery_context",
        "rag_last_context",
    ):
        state.pop(key, None)


def invalidate_stale_result(
    state: MutableMapping[str, Any],
    current_signature: tuple[Any, ...] | None,
) -> bool:
    if "battery_result" not in state:
        return False
    if current_signature is None or state.get("battery_result_signature") != current_signature:
        clear_battery_result(state)
        return True
    return False


def make_upload_signature(upload_id: Any, filename: str, size_bytes: int, model: str) -> tuple[Any, ...]:
    return (str(upload_id or ""), _filename(filename), int(size_bytes), model)


def build_rag_battery_context(result: dict[str, Any]) -> dict[str, Any]:
    identity = result["identity"]
    cell = str(identity["physical_cell_id"])
    prefix = f"G{identity['group']}-"
    if cell.startswith(prefix):
        cell = cell[len(prefix):]
    degradation = result.get("degradation", {})
    return {
        "part": str(identity["part"]),
        "group": str(identity["group"]),
        "cell": cell,
        "rpt": int(identity["rpt_index"]),
        "model": str(result["model_name"]),
        "predicted_soh": _finite_number(result.get("predicted_soh_percent")),
        "saved_label_soh": _finite_number(result.get("label_soh_percent")),
        "lli_percent": _finite_number(degradation.get("LLI (%)")),
        "lam_anode_percent": _finite_number(degradation.get("LAM Anode (%)")),
        "lam_cathode_percent": _finite_number(degradation.get("LAM Cathode (%)")),
    }
