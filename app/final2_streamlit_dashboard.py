from __future__ import annotations

import hashlib
from io import BytesIO
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from app.final2_streamlit_components import (
    DEFAULT_MODEL,
    MODEL_OPTIONS,
    DashboardInputError,
    build_prediction_result,
    build_rag_battery_context,
    clear_battery_result,
    resolve_uploaded_rpt,
    resolve_uploaded_rpts,
    validate_csv_upload,
)
from app.final2_streamlit_styles import CSS
from app.inference_dashboard import InferenceEngine
from app.model_runtime import MODEL_NAMES, load_shared_charge_degradation
from app.tpg_mat_ingestion import (
    FROZEN_BUILD_MANIFEST,
    FROZEN_FINAL2_DIR,
    TpgIngestionError,
    _verified_frozen_observation_rows,
    analyze_tpg_file,
    frozen_soh_label_for_identity,
)
from rag.rag_ui import render_rag_page, reset_rag_conversation


ROOT = Path(__file__).resolve().parents[1]


@st.cache_resource
def _engine() -> InferenceEngine:
    return InferenceEngine()


def _apply_pending_navigation() -> None:
    main = st.session_state.pop("pending_main_navigation", None)
    if main:
        st.session_state["main_navigation"] = main
    section = st.session_state.pop("pending_battery_section", None)
    if section:
        st.session_state["battery_section"] = section


def _clear_result_callback() -> None:
    clear_battery_result(st.session_state)


def _uploaded_id(uploaded: Any) -> str:
    return str(getattr(uploaded, "file_id", "") or "")


def _stage_upload(uploaded: Any, path: Path) -> None:
    uploaded.seek(0)
    with path.open("wb") as stream:
        shutil.copyfileobj(uploaded, stream, length=1024 * 1024)
    uploaded.seek(0)


def _run_prediction(
    uploaded: Any,
    model_name: str,
    engine: InferenceEngine,
    selected_rpt_index: str | None = None,
) -> dict[str, Any]:
    filename = validate_csv_upload(uploaded.name, int(uploaded.size))
    identity = resolve_uploaded_rpt(filename, selected_rpt_index)
    if model_name not in MODEL_OPTIONS or model_name not in MODEL_NAMES:
        raise ValueError(f"Model unavailable in the dashboard: {model_name}")
    with tempfile.TemporaryDirectory(prefix="bafuse-final2-streamlit-") as directory:
        input_path = Path(directory) / filename
        _stage_upload(uploaded, input_path)
        return _predict_verified_source(input_path, identity, model_name, engine)


def _predict_verified_source(
    input_path: Path, identity: dict[str, Any], model_name: str, engine: InferenceEngine,
) -> dict[str, Any]:
    if model_name not in MODEL_OPTIONS or model_name not in MODEL_NAMES:
        raise DashboardInputError(f"Model unavailable in the dashboard: {model_name}")
    analysis = analyze_tpg_file(
        input_path, input_path.name, engine, model_name=model_name,
        selected_rpt_index=identity["rpt_index"], require_source_sha256=True,
    )
    return build_prediction_result(
        analysis,
        identity,
        model_name,
        engine.final2_runtime,
    )


def _run_dataset_prediction(row: dict[str, str], model_name: str, engine: InferenceEngine) -> dict[str, Any]:
    if row["status"] != "ACCEPTED" or model_name not in MODEL_OPTIONS:
        raise DashboardInputError("The frozen observation or model is unavailable.")
    filename = {"train": "train.npz", "validation": "validation.npz",
                "external_test": "external_test.npz"}[row["final2_partition"]]
    manifest = json.loads(FROZEN_BUILD_MANIFEST.read_bytes())
    path = FROZEN_FINAL2_DIR / filename
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != manifest["dataset_artifact_sha256"][filename]:
        raise DashboardInputError("Frozen input arrays failed checksum verification.")
    identity = {
        "part": row["dataset_part"], "group": row["group"],
        "physical_cell_id": row["physical_cell_id"], "rpt_index": row["rpt_index"],
        "source_file": row["source_file"], "source_sha256": row["step4_source_sha256_snapshot"],
    }
    with np.load(BytesIO(payload), allow_pickle=False) as arrays:
        matches = np.flatnonzero(
            (arrays["dataset_part"] == int(identity["part"]))
            & (arrays["physical_cell_id"] == identity["physical_cell_id"])
            & (arrays["rpt_index"] == int(identity["rpt_index"]))
        )
        if len(matches) != 1:
            raise DashboardInputError("The selected RPT does not identify one frozen input row.")
        index = matches[0]
        discharge = arrays["X_discharge"][index].tolist()
        charge = arrays["X_charge"][index].tolist()
        if len(discharge) != 128 or len(charge) != 128:
            raise DashboardInputError("Frozen voltage curves must each contain 128 points.")
        features = {**identity, "source_checksum_verified": True,
                    "p5_physics_features": arrays["X_physics_raw"][index].tolist()}
    features.update({f"step4_voltage_progress_{i:03d}": value for i, value in enumerate(discharge)})
    features.update({f"step10_voltage_progress_{i:03d}": value for i, value in enumerate(charge)})
    prediction = engine.predict_for_model([features], model_name)["results"][0]
    observation = {
        **prediction, **identity, "model_prediction": prediction,
        "inputs": {
            "step4": {"processed_128_voltage": discharge, "status": "verified_frozen"},
            "step10": {"processed_128_voltage": charge, "status": "verified_frozen"},
        },
        "reference_soh": frozen_soh_label_for_identity(identity),
        "charge_degradation": load_shared_charge_degradation(
            identity["part"], identity["group"], identity["physical_cell_id"], identity["rpt_index"],
        ),
    }
    analysis = {"selected_model": model_name, "rpts": [observation],
                "provenance": {"dataset_identifier": manifest["dataset_identifier"]}}
    return build_prediction_result(analysis, identity, model_name, engine.final2_runtime)


def _select_frozen_observation() -> dict[str, str] | None:
    try:
        rows = [row for row in _verified_frozen_observation_rows() if row["status"] == "ACCEPTED"]
    except (TpgIngestionError, OSError, ValueError) as error:
        st.error(f"Frozen dataset unavailable: {error}")
        return None
    if not rows:
        st.info("No accepted observations are available.")
        return None
    part = st.selectbox(
        "Dataset", sorted({row["dataset_part"] for row in rows}),
        format_func=lambda value: f"Oxford Part {value}",
        key="frozen_part", on_change=_clear_result_callback,
    )
    rows = [row for row in rows if row["dataset_part"] == part]
    cell = st.selectbox(
        "Cell", sorted({row["physical_cell_id"] for row in rows}),
        key=f"frozen_cell_{part}", on_change=_clear_result_callback,
    )
    rows = [row for row in rows if row["physical_cell_id"] == cell]
    rpt = st.selectbox(
        "RPT", sorted({row["rpt_index"] for row in rows}, key=int),
        format_func=lambda value: f"RPT{value}",
        key=f"frozen_rpt_{part}_{cell}", on_change=_clear_result_callback,
    )
    selected = [row for row in rows if row["rpt_index"] == rpt]
    if len(selected) != 1:
        st.error("The selected frozen observation is ambiguous.")
        return None
    return selected[0]


def _metric_card(label: str, value: str, help_text: str | None = None) -> None:
    with st.container(border=True):
        st.metric(label, value, help=help_text)


def _plot_curve(title: str, values: list[float], color: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if not values:
        st.info("N/A — no valid, verified curve is available for this RPT.")
        return
    progress = np.linspace(0.0, 100.0, len(values))
    figure = go.Figure(
        go.Scatter(
            x=progress,
            y=values,
            mode="lines",
            line={"color": color, "width": 2.6},
            name=title,
            hovertemplate="Progress: %{x:.1f}%<br>Voltage: %{y:.4f} V<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 18, "r": 18, "t": 18, "b": 18},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        showlegend=False,
        xaxis_title="Normalized progress (%)",
        yaxis_title="Voltage (V)",
        font={"color": "#334155"},
        xaxis={"gridcolor": "#edf1f5"},
        yaxis={"gridcolor": "#edf1f5"},
    )
    st.plotly_chart(figure, use_container_width=True, config={"displaylogo": False})


def _render_input(engine: InferenceEngine) -> None:
    mode = st.radio(
        "RPT source", ("Frozen dataset", "Upload RPT CSV"), horizontal=True,
        key="rpt_source_mode", on_change=_clear_result_callback,
    )
    frozen_row = _select_frozen_observation() if mode == "Frozen dataset" else None
    uploaded = None
    if mode == "Upload RPT CSV":
        uploaded = st.file_uploader(
            "Upload RPT CSV", type=["csv"], accept_multiple_files=False,
            key="rpt_csv_upload", on_change=_clear_result_callback,
            help="Upload an original Guide-mapped CSV. Its source checksum is verified.",
        )
    st.selectbox(
        "Model",
        options=list(MODEL_OPTIONS),
        index=MODEL_OPTIONS.index(DEFAULT_MODEL),
        key="selected_model",
        on_change=_clear_result_callback,
    )
    selected_model = st.session_state["selected_model"]
    if selected_model == "CNN-3M":
        st.caption("Approved CNN-3M runtime uses the historical Physics schema.")
    selected_rpt_index: str | None = None
    if uploaded is not None:
        st.caption(f"Selected source: {uploaded.name} · {int(uploaded.size):,} bytes")
        try:
            rpt_identities = resolve_uploaded_rpts(uploaded.name)
        except DashboardInputError as error:
            st.error(f"Source mapping unavailable: {error}")
        else:
            if len(rpt_identities) > 1:
                options = [identity["rpt_index"] for identity in rpt_identities]
                upload_key = _uploaded_id(uploaded) or f"{uploaded.name}-{uploaded.size}"
                selected_rpt_index = st.selectbox(
                    "Select RPT",
                    options=options,
                    format_func=lambda value: f"RPT{value}",
                    key=f"selected_rpt_{upload_key}",
                    on_change=_clear_result_callback,
                )
            elif rpt_identities:
                selected_rpt_index = rpt_identities[0]["rpt_index"]
    if st.button("Run Prediction", type="primary", use_container_width=True):
        if uploaded is None and frozen_row is None:
            st.error("Select an accepted observation or upload a verified RPT CSV.")
            return
        try:
            with st.spinner("Verifying source, extracting eligible curves, and running the frozen model..."):
                result = (
                    _run_dataset_prediction(frozen_row, selected_model, engine)
                    if frozen_row is not None
                    else _run_prediction(uploaded, selected_model, engine, selected_rpt_index)
                )
        except (DashboardInputError, TpgIngestionError, FileNotFoundError, OSError, KeyError, ValueError, RuntimeError) as error:
            clear_battery_result(st.session_state)
            st.error(f"Prediction unavailable: {type(error).__name__}: {error}")
            return
        if st.session_state.get("rag_battery_context") != build_rag_battery_context(result):
            clear_battery_result(st.session_state)
        st.session_state["battery_result"] = result
        st.session_state["pending_battery_section"] = "Prediction"
        st.rerun()


def _render_model_evaluation(result: dict[str, Any]) -> None:
    st.markdown('<div class="section-title">Model Evaluation</div>', unsafe_allow_html=True)
    validation = result.get("validation_metrics")
    external = result.get("external_metrics")
    scopes = (
        ("Validation · frozen selected checkpoint", validation, "validation"),
        ("External Test · one-time frozen evaluation", external, "external"),
    )
    for title, metrics, prefix in scopes:
        st.markdown(f"**{title}**")
        columns = st.columns(3)
        if not isinstance(metrics, dict):
            values = ("N/A", "N/A", "N/A")
        elif prefix == "validation":
            values = tuple(
                _format_metric(metrics.get(key))
                for key in ("validation_mae_pp", "validation_rmse_pp", "validation_r2")
            )
        else:
            values = tuple(
                _format_metric(metrics.get(key))
                for key in ("external_mae_pp", "external_rmse_pp", "external_r2")
            )
        for column, label, value in zip(columns, ("MAE (pp)", "RMSE (pp)", "R²"), values):
            with column:
                _metric_card(label, value)
        if isinstance(metrics, dict):
            count = metrics.get("external_sample_count") if prefix == "external" else None
            cells = metrics.get("external_cell_count") if prefix == "external" else None
            if count is not None and cells is not None:
                st.caption(f"Saved evaluation cohort: {int(count)} observations · {int(cells)} cells.")


def _format_metric(value: Any) -> str:
    number = _finite_number(value)
    return f"{number:.3f}" if number is not None else "N/A"


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _render_prediction(result: dict[str, Any]) -> None:
    identity = result["identity"]
    st.caption(
        f"Verified RPT: Part {identity['part']} · Group {identity['group']} · "
        f"{identity['physical_cell_id']} · RPT{identity['rpt_index']} · "
        f"Model: {result['model_name']}"
    )
    if result.get("prediction_status") != "frozen_checkpoint_inference":
        reason = result.get("prediction_reason") or result.get("prediction_status")
        st.warning(f"Prediction unavailable for this model/input: {reason}")

    st.markdown('<div class="section-title">Input Visualization</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        _plot_curve("Discharge Curve — Step 4", result["step4_curve"], "#168c91")
    with right:
        _plot_curve("Charge Curve — Step 10", result["step10_curve"], "#3276b1")

    st.markdown('<div class="section-title">SoH Results</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        _metric_card(
            "Predicted SoH (%)",
            _format_metric(result.get("predicted_soh_percent")),
        )
    with right:
        _metric_card(
            "Label SoH (%)",
            _format_metric(result.get("label_soh_percent")),
            help_text=(
                "Saved identity-matched capacity-ratio label; provisional and not "
                "Oxford ground truth."
            ),
        )
    if result.get("label_soh_percent") is None:
        st.caption(f"Label unavailable: {result.get('label_status', 'no exact saved label')}.")
    else:
        st.caption(
            "Label source was matched to this exact verified RPT. "
            "Capacity-ratio labels are provisional, not Oxford ground truth."
        )

    st.markdown('<div class="section-title">Prediction Error</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    for column, metric in zip((left, right), ("MAE", "RMSE")):
        with column:
            _metric_card(
                f"{metric} (percentage points)",
                _format_metric(result.get(f"{metric.lower()}_percentage_points")),
            )

    _render_model_evaluation(result)

    st.markdown('<div class="section-title">Degradation Diagnosis</div>', unsafe_allow_html=True)
    degradation = result["degradation"]
    columns = st.columns(3)
    for column, name in zip(columns, ("LLI (%)", "LAM Anode (%)", "LAM Cathode (%)")):
        with column:
            _metric_card(name, _format_metric(degradation.get(name)))
    details = result.get("degradation_details", {})
    if not details.get("verified_saved_charge_result"):
        st.caption(f"Saved accepted Charge estimate unavailable: {details.get('reason') or details.get('status')}.")
    else:
        st.caption(
            "Saved Charge-only PyDMA diagnostic; not independent ground truth. "
            f"Fit RMSE: {_format_metric(details.get('fit_rmse_mV'))} mV."
        )
        if details.get("same_cell_bol_status") not in {"same_cell_bol_fit_accepted"}:
            st.caption("Same-cell BoL fit provenance is not asserted by this saved artifact.")

    if (
        result.get("prediction_status") == "frozen_checkpoint_inference"
        and _finite_number(result.get("predicted_soh_percent")) is not None
        and st.button("Explain with RAG", type="primary")
    ):
        context = build_rag_battery_context(result)
        if st.session_state.get("rag_battery_context") != context:
            reset_rag_conversation(st.session_state)
        st.session_state["rag_battery_context"] = context
        st.session_state["pending_main_navigation"] = "RAG Assistant"
        st.rerun()


def _render_battery_analysis(engine: InferenceEngine) -> None:
    section = st.radio(
        "Battery Analysis sections",
        options=("Input", "Prediction"),
        horizontal=True,
        label_visibility="collapsed",
        key="battery_section",
    )
    if section == "Input":
        _render_input(engine)
        return
    result = st.session_state.get("battery_result")
    if result is None:
        st.info("Select an accepted observation or upload a verified RPT CSV in Input.")
        return
    _render_prediction(result)


def _render_rag() -> None:
    context = st.session_state.get("rag_battery_context")
    render_rag_page(context)


def main() -> None:
    _apply_pending_navigation()
    # Keep ordinary widget state when its page is not rendered.
    for key in ("battery_section", "selected_model"):
        if key in st.session_state:
            st.session_state[key] = st.session_state[key]
    st.set_page_config(
        page_title="BaFuse Final2",
        page_icon="🔋",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
          <h1>BaFuse Final2</h1>
          <p>Battery Health Estimation &amp; Degradation Diagnosis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    navigation = st.radio(
        "Main navigation",
        options=("Battery Analysis", "RAG Assistant"),
        horizontal=True,
        label_visibility="collapsed",
        key="main_navigation",
    )
    if navigation == "Battery Analysis":
        engine = _engine()
        _render_battery_analysis(engine)
    else:
        _render_rag()


if __name__ == "__main__":
    main()
