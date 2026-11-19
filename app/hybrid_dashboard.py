from __future__ import annotations

import html
import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bafuse.config import load_project_paths
from bafuse.hybrid.hybrid_battery_health import KEY_COLUMNS, _basename


def _dashboard_href(target: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:
        return target.as_uri()


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    clean = frame.copy().replace({np.nan: None})
    return clean.to_dict(orient="records")


def _audit_html_table(
    frame: pd.DataFrame, columns: list[tuple[str, str]]
) -> str:
    headers = "".join(f"<th>{html.escape(label)}</th>" for _, label in columns)
    rows = []
    for record in frame.to_dict(orient="records"):
        cells = []
        for column, _ in columns:
            value = record.get(column, "")
            if value is None or pd.isna(value):
                rendered = ""
            elif isinstance(value, (int, np.integer)):
                rendered = str(int(value))
            elif isinstance(value, (float, np.floating)):
                rendered = f"{float(value):.4f}"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class=\"table-wrap\"><table><thead><tr>"
        + headers
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _render_soh_audit_section(
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    exclusions: pd.DataFrame,
    full_cnn1d: pd.DataFrame,
    report_href: str,
) -> str:
    if metrics.empty or coverage.empty:
        raise ValueError("SoH audit dashboard data must include metrics and coverage.")
    overall = metrics[metrics["metric_scope"] == "overall"].copy()
    full_overall = full_cnn1d[
        full_cnn1d["metric_scope"] == "overall"
    ].copy()
    evaluated_cells = int((coverage["eligible_fusion_predictions"] > 0).sum())
    fusion_observations = int(coverage["eligible_fusion_predictions"].sum())
    complete_modality_observations = int(
        coverage["complete_modality_observations"].sum()
    )
    cnn_observations = int(
        coverage["eligible_full_coverage_cnn1d_predictions"].sum()
    )
    reason_counts = (
        exclusions.loc[
            exclusions["exact_exclusion_reason"].astype(str).str.len() > 0,
            "exact_exclusion_reason",
        ]
        .value_counts()
        .rename_axis("exact_exclusion_reason")
        .reset_index(name="observation_count")
    )
    metrics_table = _audit_html_table(
        overall,
        [
            ("partition", "Partition"),
            ("model", "Model"),
            ("observation_count", "Observations"),
            ("physical_cell_count", "Cells"),
            ("mae_soh_percentage_points", "MAE (SOH pp)"),
            ("rmse_soh_percentage_points", "RMSE (SOH pp)"),
            ("r2", "R²"),
        ],
    )
    full_cnn_table = _audit_html_table(
        full_overall,
        [
            ("observation_count", "Observations"),
            ("physical_cell_count", "Cells"),
            ("mae_soh_percentage_points", "MAE (SOH pp)"),
            ("rmse_soh_percentage_points", "RMSE (SOH pp)"),
            ("r2", "R²"),
        ],
    )
    coverage_table = _audit_html_table(
        coverage,
        [
            ("physical_cell_id", "Physical cell"),
            ("available_step4_observations", "Valid Step 4"),
            ("valid_soh_reference_observations", "Valid references"),
            ("verified_step10_charge_observations", "Mapped Step 10"),
            ("valid_physics_feature_observations", "Physics available"),
            ("complete_modality_observations", "Complete modalities"),
            ("eligible_fusion_predictions", "Fusion predictions"),
            ("eligible_full_coverage_cnn1d_predictions", "CNN1D predictions"),
        ],
    )
    reasons_table = (
        _audit_html_table(
            reason_counts,
            [
                ("exact_exclusion_reason", "Exact exclusion reason"),
                ("observation_count", "Observations"),
            ],
        )
        if not reason_counts.empty
        else "<p>No external fusion exclusions remain.</p>"
    )
    return (
        '<section id="soh-evaluation-audit">'
        "<h2>Corrected overall SoH evaluation</h2>"
        '<p class="muted">Headline metrics are computed over all eligible observations in each partition. '
        "CNN1D and fusion shared-subset metrics use identical keys; the separate full-coverage "
        "CNN1D result has a different population.</p>"
        '<p class="warning">Part 3 is an external/domain-shift cohort. Its capacity-ratio labels '
        "remain provisional; do not treat these metrics as official-label validation.</p>"
        '<div class="cards">'
        f'<div class="card">External fusion cells<strong>{evaluated_cells}/12</strong></div>'
        f'<div class="card">Complete-modality observations<strong>{complete_modality_observations}</strong></div>'
        f'<div class="card">Fusion predictions<strong>{fusion_observations}</strong></div>'
        f'<div class="card">Full-coverage CNN1D observations<strong>{cnn_observations}</strong></div>'
        "</div>"
        "<h3>Overall metrics</h3>"
        + metrics_table
        + "<h3>Full-coverage historical CNN1D</h3>"
        + full_cnn_table
        + "<h3>External 12-cell coverage</h3>"
        + coverage_table
        + "<h3>External fusion exclusions</h3>"
        + reasons_table
        + f'<p><a href="{html.escape(report_href, quote=True)}">Corrected evaluation report</a></p>'
        + "</section>"
    )


def _render_eol_checkpoint_section(
    records: pd.DataFrame,
    csv_href: str,
) -> str:
    columns = [
        ("checkpoint_id", "EoL checkpoint"),
        ("part_label", "Part"),
        ("physical_cell_id", "Physical cell"),
        ("guide_rpt_assignment", "Guide RPT"),
        ("direction", "PyDMA direction"),
        ("modality_status", "SoH modalities"),
        ("predicted_soh_percent", "Predicted SoH (%)"),
        ("raw_model_output_soh_percent", "Raw model output (%)"),
        ("provisional_reference_soh_percent", "Provisional reference (%)"),
        ("prediction_plausibility_status", "Prediction status"),
        ("soh_status", "SoH status"),
        ("fit_status", "PyDMA fit status"),
        ("fit_rmse_mV", "Fit RMSE (mV)"),
        ("accepted_runs", "Accepted runs"),
        ("lli_percent", "LLI (%)"),
        ("lam_anode_percent", "Anode LAM (%)"),
        ("lam_cathode_percent", "Cathode LAM (%)"),
        ("negative_estimate_flags", "Negative estimates"),
        ("parameter_at_bound_flags", "Parameter bounds"),
        ("baseline_reference_status", "BoL reference"),
        ("mapping_status", "Mapping status"),
    ]
    missing = {name for name, _ in columns}.difference(records.columns)
    if missing:
        raise ValueError(
            "EoL dashboard records are missing required columns: "
            + ", ".join(sorted(missing))
        )
    display = records.loc[:, [name for name, _ in columns]].copy()
    for column in (
        "predicted_soh_percent",
        "raw_model_output_soh_percent",
        "provisional_reference_soh_percent",
        "fit_rmse_mV",
        "lli_percent",
        "lam_anode_percent",
        "lam_cathode_percent",
    ):
        display[column] = display[column].map(
            lambda value: (
                ""
                if value is None or pd.isna(value) or str(value).strip() == ""
                else f"{float(value):.3f}"
            )
        )
    table = _audit_html_table(display, columns)
    return (
        '<section id="eol-checkpoints">'
        "<h2>Separate EoL checkpoints</h2>"
        '<p class="warning">These checkpoints are displayed separately from ordinary numbered-RPT '
        "trajectories. Part 2 EoL procedures remain unnumbered; Part 3 RPT assignments follow "
        "the Guide. Part 3 capacity-ratio references are provisional. SoH inference and PyDMA "
        "eligibility are independent; blank fields mean unavailable, not zero.</p>"
        "<p class=\"muted\">One row per checkpoint and PyDMA direction. Predictions use the frozen "
        "three-encoder checkpoint without retraining; no EoL result was used for model selection.</p>"
        + table
        + f'<p><a href="{html.escape(csv_href, quote=True)}">EoL checkpoint records (CSV)</a></p>'
        + "</section>"
    )


def _visualization_index(
    visualization_dir: Path,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    summary_path = visualization_dir / "pydma_fit_summary.csv"
    if not summary_path.is_file():
        return {}
    summary = pd.read_csv(summary_path)
    index = {}
    for row in summary.to_dict(orient="records"):
        key = (
            str(int(row["dataset_part"])),
            str(row["physical_cell_id"]),
            str(row["measurement_id"]),
            str(row["direction"]).casefold(),
            str(int(row["test_step"])),
            _basename(row["source_file"]),
        )
        index[key] = row
    return index


def _add_diagnostic_links(root: Path, output_dir: Path, curve_rows: pd.DataFrame) -> pd.DataFrame:
    rows = curve_rows.copy()
    visualization_dir = load_project_paths(root).pydma_visualization
    visualization = _visualization_index(visualization_dir)
    pdf_path = visualization_dir / "pydma_fit_overlays.pdf"
    href_text = _dashboard_href(pdf_path, output_dir)
    pages = []
    overlays = []
    links = []
    for _, row in rows.iterrows():
        key = (
            str(int(row["dataset_part"])),
            str(row["physical_cell_id"]),
            str(row["rpt_id"]),
            str(row["direction"]).casefold(),
            str(row["test_step"]),
            str(row["source_file_basename"]),
        )
        record = visualization.get(key)
        if record is None or not pdf_path.is_file():
            pages.append(None)
            overlays.append(False)
            links.append("")
            continue
        page = int(record["fit_id"]) + 1
        overlay = pd.notna(record.get("refit_rmse_mV"))
        pages.append(page)
        overlays.append(bool(overlay))
        links.append(f"{href_text}#page={page}")
    rows["diagnostic_pdf_page"] = pages
    rows["reconstructed_overlay_available"] = overlays
    rows["diagnostic_pdf_href"] = links
    return rows


def write_dashboard(
    *,
    root: Path,
    output_dir: Path,
    feature_names: list[str],
    feature_frame: pd.DataFrame,
    checkpoint_manifest: pd.DataFrame,
    curve_rows: pd.DataFrame,
    hybrid: pd.DataFrame,
    aggregates: pd.DataFrame,
) -> None:
    curve_rows = _add_diagnostic_links(root, output_dir, curve_rows)
    feature_curves = {}
    for _, row in feature_frame.iterrows():
        key = (
            f"{row['dataset_part']}|{row['physical_cell_id']}|"
            f"{row['measurement_id']}"
        )
        feature_curves[key] = [
            float(value) for value in row[feature_names].to_numpy(dtype=float)
        ]
    payload = {
        "checkpoints": _records(checkpoint_manifest),
        "curves": _records(curve_rows),
        "hybrid": _records(hybrid),
        "aggregates": _records(aggregates),
        "featureCurves": feature_curves,
        "pdfNote": (
            "The linked PDF is a prior diagnostic: measured curves are shown for "
            "481 fits, with reconstructed overlays only for 12 selected fits."
        ),
    }
    audit_dir = (
        root
        / "experiments"
        / "soh_forecast"
        / "bafuse_three_encoder_v1"
        / "evaluation_audit_v1"
    )
    if audit_dir.is_dir():
        metrics = pd.read_csv(audit_dir / "soh_overall_metrics.csv")
        coverage = pd.read_csv(audit_dir / "external_12_cell_coverage.csv")
        exclusions = pd.read_csv(audit_dir / "external_exclusion_reasons.csv")
        full_cnn1d = pd.read_csv(
            audit_dir / "full_coverage_cnn1d_evaluation.csv"
        )
        soh_audit_html = _render_soh_audit_section(
            metrics,
            coverage,
            exclusions,
            full_cnn1d,
            _dashboard_href(
                audit_dir / "corrected_soh_evaluation_report.md", output_dir
            ),
        )
    else:
        soh_audit_html = ""
    eol_records_path = (
        root
        / "experiments"
        / "results"
        / "eol_coverage_v1"
        / "eol_dashboard_records.csv"
    )
    if eol_records_path.is_file():
        eol_records = pd.read_csv(eol_records_path, keep_default_na=False)
        eol_checkpoint_html = _render_eol_checkpoint_section(
            eol_records,
            _dashboard_href(eol_records_path, output_dir),
        )
    else:
        eol_checkpoint_html = ""
    data_json = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace(
        "</", "<\\/"
    )
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hybrid CNN1D + PyDMA Battery Health</title>
<style>
:root {{ color-scheme: light; font-family: Segoe UI, Arial, sans-serif; color: #17212b; background: #f5f7f9; }}
body {{ margin: 0; }}
header {{ background: #14324b; color: white; padding: 1.1rem 1.5rem; }}
header h1 {{ margin: 0 0 .35rem; font-size: 1.35rem; }}
header p {{ margin: 0; color: #d7e5ef; }}
main {{ max-width: 1280px; margin: auto; padding: 1rem; }}
section {{ background: white; border: 1px solid #d6dee5; border-radius: 8px; padding: 1rem; margin: 0 0 1rem; }}
.controls {{ display: flex; flex-wrap: wrap; gap: .8rem; align-items: end; }}
label {{ display: grid; gap: .3rem; font-weight: 600; }}
select, button {{ border: 1px solid #8da1b2; border-radius: 4px; background: white; padding: .5rem; font-size: .95rem; }}
button {{ cursor: pointer; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit,minmax(180px,1fr)); gap: .65rem; }}
.card {{ border: 1px solid #d6dee5; border-radius: 6px; padding: .7rem; }}
.card strong {{ display: block; font-size: 1.2rem; margin-top: .2rem; }}
.muted {{ color: #536574; font-size: .9rem; }}
.warning {{ color: #8a4c00; background: #fff5df; border-left: 4px solid #e29b26; padding: .65rem; }}
.chart {{ width: 100%; min-height: 200px; }}
svg {{ width: 100%; height: auto; overflow: visible; }}
table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
th, td {{ text-align: left; vertical-align: top; border-bottom: 1px solid #e1e6ea; padding: .45rem; }}
th {{ background: #edf2f6; position: sticky; top: 0; }}
.table-wrap {{ max-height: 420px; overflow: auto; }}
.actions {{ display: flex; gap: .5rem; flex-wrap: wrap; margin-top: .6rem; }}
a {{ color: #145d91; }}
.ok {{ color: #17633a; }}
.bad {{ color: #a22828; }}
@media (max-width: 650px) {{ main {{ padding: .5rem; }} section {{ padding: .7rem; }} }}
</style>
</head>
<body>
<header>
<h1>Hybrid CNN1D + PyDMA Battery Health</h1>
<p>Parallel branches: CNN1D estimates same-RPT SoH; PyDMA provides direction-specific degradation diagnostics.</p>
</header>
<main>
<section>
<div class="controls">
  <label>Dataset part<select id="part"></select></label>
  <label>Physical cell<select id="cell"></select></label>
  <label>RPT<select id="rpt"></select></label>
  <label>Fit rows<select id="fitFilter"><option value="all">All</option><option value="accepted">Strict-gate accepted</option><option value="rejected">Rejected / raw</option><option value="review">Review flags</option></select></label>
</div>
<p id="mapping" class="muted"></p>
<div id="cards" class="cards"></div>
<p id="limits" class="warning"></p>
</section>
{soh_audit_html}
{eol_checkpoint_html}
<section>
<h2>CNN1D Step 4 input</h2>
<p class="muted">128-point voltage-progress input used by the saved checkpoint; capacity-derived target values are display-only and are not model inputs.</p>
<div id="step4" class="chart"></div>
<div class="actions"><button id="exportStep4">Download Step 4 plot (SVG)</button></div>
</section>
<section>
<h2>PyDMA curve-level results</h2>
<p class="muted">Charge and discharge remain separate. Fit acceptance requires RMSE &lt; 15 mV and at least three accepted optimization runs. A fit summary is not a saved parameter vector.</p>
<div class="table-wrap"><table><thead><tr>
<th>Curve ID</th><th>Direction / Step</th><th>RMSE (mV)</th><th>LLI / Anode LAM / Cathode LAM (%)</th><th>Accepted runs</th><th>Gate / status</th><th>Mapping / QC</th><th>Fit plot</th>
</tr></thead><tbody id="curves"></tbody></table></div>
<div class="actions">
<button id="exportCheckpoint">Download selected checkpoint CSV</button>
<button id="exportTrajectory">Download trajectory plot (SVG)</button>
<a href="pydma_curve_level_results.csv" download>Export all curve-level rows</a>
<a href="hybrid_battery_health_results.csv" download>Export joined checkpoints</a>
<a href="checkpoint_manifest.csv" download>Export checkpoint manifest</a>
</div>
</section>
<section>
<h2>Cell trajectories</h2>
<p class="muted">No monotonic smoothing is applied. Missing values and review flags remain visible. Cycle-age values are nominal proxies where documented.</p>
<div id="trajectories"></div>
</section>
<section>
<h2>Exports and evaluation</h2>
<p><a href="soh_evaluation.csv" download>Download SoH evaluation metrics</a> ·
<a href="pydma_rpt_aggregated.csv" download>Download direction-specific PyDMA aggregates</a> ·
<a href="pydma_quality_report.csv" download>Download PyDMA QC summary</a></p>
<p class="muted">Part 3 is an external/domain-shift cohort; its capacity-ratio reference and metrics remain provisional. PyDMA LLI/LAM are model-derived diagnostics, not independent ground truth.</p>
</section>
</main>
<script>
const DATA = {data_json};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}}[c]));
const number = value => value === null || value === undefined || value === "" || !Number.isFinite(Number(value)) ? "Unavailable" : Number(value).toFixed(2);
const key = (part, cell, rpt) => `${{part}}|${{cell}}|${{rpt}}`;
function options(element, values, selected) {{
  element.innerHTML = values.map(v => `<option value="${{esc(v)}}">${{esc(v)}}</option>`).join("");
  if (selected !== undefined && values.includes(String(selected))) element.value = String(selected);
}}
function selectedCheckpoint() {{
  return DATA.checkpoints.find(r => String(r.dataset_part) === $("part").value &&
    r.physical_cell_id === $("cell").value && r.rpt_id === $("rpt").value);
}}
function selectedHybrid() {{
  return DATA.hybrid.find(r => String(r.dataset_part) === $("part").value &&
    r.physical_cell_id === $("cell").value && r.measurement_id === $("rpt").value);
}}
function selectedAggregate(direction) {{
  return DATA.aggregates.find(r => String(r.dataset_part) === $("part").value &&
    r.physical_cell_id === $("cell").value && r.rpt_id === $("rpt").value &&
    r.direction === direction);
}}
function drawLineSvg(values, label, color) {{
  const width=900, height=260, left=55, right=18, top=18, bottom=38;
  const finite=values.filter(v=>Number.isFinite(v));
  if (!finite.length) return `<p class="muted">No values available for this checkpoint.</p>`;
  const min=Math.min(...finite), max=Math.max(...finite), pad=(max-min || 1)*.08;
  const x=i=>left+i*(width-left-right)/(values.length-1 || 1);
  const y=v=>top+(max+pad-v)*(height-top-bottom)/(max-min+2*pad);
  const points=values.map((v,i)=>Number.isFinite(v)?`${{x(i).toFixed(1)}} ${{y(v).toFixed(1)}}`:null).filter(Boolean).join(" ");
  return `<svg id="step4Svg" viewBox="0 0 ${{width}} ${{height}}" role="img" aria-label="${{esc(label)}}">
    <line x1="${{left}}" y1="${{height-bottom}}" x2="${{width-right}}" y2="${{height-bottom}}" stroke="#667"/>
    <line x1="${{left}}" y1="${{top}}" x2="${{left}}" y2="${{height-bottom}}" stroke="#667"/>
    <polyline points="${{points}}" fill="none" stroke="${{color}}" stroke-width="2.5"/>
    <text x="${{left}}" y="${{height-8}}" font-size="12">Discharge progress (0–100%)</text>
    <text x="8" y="16" font-size="12">${{esc(label)}}</text>
    <text x="${{left+5}}" y="${{top+14}}" font-size="11">${{max.toFixed(3)}} V</text>
    <text x="${{left+5}}" y="${{height-bottom-5}}" font-size="11">${{min.toFixed(3)}} V</text>
  </svg>`;
}}
function updateCells(preferred) {{
  const cells=[...new Set(DATA.checkpoints.filter(r=>String(r.dataset_part)===$("part").value).map(r=>r.physical_cell_id))].sort();
  options($("cell"),cells,preferred || cells[0]);
  updateRpts();
}}
function updateRpts(preferred) {{
  const rpts=DATA.checkpoints.filter(r=>String(r.dataset_part)===$("part").value && r.physical_cell_id===$("cell").value)
    .sort((a,b)=>(Number(a.rpt_order)||0)-(Number(b.rpt_order)||0)).map(r=>r.rpt_id);
  options($("rpt"),rpts,preferred || rpts[0]);
  render();
}}
function metricCard(title,value,sub) {{
  return `<div class="card"><span class="muted">${{esc(title)}}</span><strong>${{esc(value)}}</strong><span class="muted">${{esc(sub||"")}}</span></div>`;
}}
function renderCards(cp, hybrid) {{
  const discharge=selectedAggregate("discharge"), charge=selectedAggregate("charge");
  $("cards").innerHTML=[
    metricCard("Predicted SoH",number(hybrid?.predicted_soh_percent),"CNN1D (%)"),
    metricCard("Fusion SoH",
      number(hybrid?.fusion_v2_predicted_soh_percent),
      hybrid?.fusion_v2_model_version || "Three-encoder v1 (%)"),
    metricCard("Estimated LLI",number(discharge?.lli_estimated_pct),"PyDMA discharge (%)"),
    metricCard("Anode LAM",number(discharge?.lam_anode_estimated_pct),"PyDMA discharge (%)"),
    metricCard("Cathode LAM",number(discharge?.lam_cathode_estimated_pct),"PyDMA discharge (%)"),
    metricCard("Discharge fit quality",discharge?.quality_flag || "Unavailable",
      `${{discharge?.n_accepted_curves ?? 0}} gate-accepted; ${{discharge?.n_mapping_eligible_accepted_curves ?? 0}} step-mapped; median RMSE ${{number(discharge?.median_voltage_fit_rmse_mV)}} mV`),
    metricCard("Charge comparison",charge?.quality_flag || "Unavailable",
      "Separate direction; difference is reported, not averaged")
  ].join("");
  const ref = hybrid?.reference_soh_percent;
  $("mapping").textContent=`${{cp.checkpoint_id}} · ${{cp.mapping_status}} · age: ${{cp.cycle_age_status}}`+
    (ref === null || ref === undefined ? "" : ` · SoH reference scope: ${{hybrid.reference_label_scope}}`)+
    ` · observation: ${{cp.observation_mapping_status || "unavailable"}}`+
    ` · Step 4: ${{cp.observation_step4_status || "unavailable"}}`+
    ` · degradation reference: ${{cp.degradation_reference_status || "unavailable"}}`;
  const notes=[];
  if (!cp.cnn1d_input_available) notes.push("No Step 4 CNN1D input is available for this RPT.");
  if (cp.mapping_status.includes("provisional")) notes.push("Provisional Group 5 file-order RPT identity is quarantined.");
  if (String(cp.observation_needs_audit).toLowerCase()==="true") notes.push(
    "Observation mapping audit: " + (cp.observation_audit_notes || "additional curve review remains"));
  if (String(cp.degradation_reference_status || "").match(/provisional|unresolved|candidate/i)) {{
    notes.push("Degradation reference is not fully resolved: " +
      (cp.degradation_reference_limitations || cp.degradation_reference_status));
  }}
  if (discharge?.quality_flags) notes.push(`Discharge QC: ${{discharge.quality_flags}}.`);
  if (String(hybrid?.dataset_part)==="3") notes.push("Part 3 is external/domain shift and the SOH reference is provisional.");
  notes.push(DATA.pdfNote);
  $("limits").textContent=notes.join(" ");
}}
function renderStep4(cp) {{
  const values=DATA.featureCurves[key(cp.dataset_part,cp.physical_cell_id,cp.rpt_id)];
  $("step4").innerHTML=values ? drawLineSvg(values,"Step 4 voltage input","#1261a0") :
    `<p class="muted">No validated Step 4 input is available for this RPT.</p>`;
}}
function fitRows(cp) {{
  let rows=DATA.curves.filter(r=>String(r.dataset_part)===String(cp.dataset_part) &&
    r.physical_cell_id===cp.physical_cell_id && r.rpt_id===cp.rpt_id);
  const filter=$("fitFilter").value;
  if (filter==="accepted") rows=rows.filter(r=>r.strict_gate_pass);
  if (filter==="rejected") rows=rows.filter(r=>!r.strict_gate_pass);
  if (filter==="review") rows=rows.filter(r=>r.quality_flags || r.feature_mapping_status.includes("provisional") ||
    (r.observation_mapping_status && r.observation_mapping_status!=="verified") ||
    (r.observation_curve_step_status && r.observation_curve_step_status!=="verified") ||
    String(r.degradation_reference_status || "").match(/provisional|unresolved|candidate/i));
  return rows;
}}
function renderCurves(cp) {{
  const rows=fitRows(cp);
  $("curves").innerHTML=rows.length ? rows.map(r=>{{
    const fitLink=r.diagnostic_pdf_href ? `<a href="${{esc(r.diagnostic_pdf_href)}}" target="_blank" rel="noopener">${{r.reconstructed_overlay_available ? "Open prior overlay" : "Measured-only page"}}</a>` : "Unavailable";
    const values=[r.lli_estimated_pct,r.lam_anode_estimated_pct,r.lam_cathode_estimated_pct].map(number).join(" / ");
    const gate=r.strict_gate_pass ? "PASS" : "REVIEW / REJECT";
    return `<tr><td>${{esc(r.curve_id)}}</td><td>${{esc(r.direction)}} / Step ${{esc(r.test_step)}}</td>
      <td>${{number(r.fit_rmse_mV)}}</td><td>${{values}}</td>
      <td>${{esc(r.accepted_runs ?? "n/a")}} / ${{esc(r.total_runs ?? "n/a")}}</td>
      <td>${{gate}}<br>${{esc(r.diagnostic_label_status)}}</td>
      <td>${{esc(r.feature_mapping_status)}}<br>Observation: ${{esc(r.observation_mapping_status || "not mapped")}}
        <br>Curve step: ${{esc(r.observation_curve_step_status || "not mapped")}}
        <br>Reference: ${{esc(r.degradation_reference_status || "unavailable")}}
        <br>${{esc(r.quality_flags || "No additional flag")}}</td>
      <td>${{fitLink}}${{r.reconstructed_overlay_available ? "" : (r.diagnostic_pdf_href ? "<br><span class='muted'>No saved parameters for overlay</span>" : "")}}</td></tr>`;
  }}).join("") : `<tr><td colspan="8">No PyDMA result rows for this checkpoint.</td></tr>`;
}}
function trajectorySvg(cp) {{
  const rows=DATA.checkpoints.filter(r=>r.physical_cell_id===cp.physical_cell_id)
    .sort((a,b)=>(Number(a.rpt_order)||0)-(Number(b.rpt_order)||0));
  const width=900,height=560,left=70,right=20,top=15,panel=125,gap=12;
  const series=[
    {{name:"Predicted SoH (%)",hybrid:"predicted_soh_percent",aggregate:null,color:"#1261a0"}},
    {{name:"LLI (%)",hybrid:null,aggregate:"lli_estimated_pct",color:"#6a3d9a"}},
    {{name:"Anode LAM (%)",hybrid:null,aggregate:"lam_anode_estimated_pct",color:"#20805d"}},
    {{name:"Cathode LAM (%)",hybrid:null,aggregate:"lam_cathode_estimated_pct",color:"#bd5a18"}}
  ];
  const hybridByKey=new Map(DATA.hybrid.map(r=>[key(r.dataset_part,r.physical_cell_id,r.measurement_id),r]));
  const aggs=new Map(DATA.aggregates.filter(r=>r.direction==="discharge").map(r=>[key(r.dataset_part,r.physical_cell_id,r.rpt_id),r]));
  const xs=rows.map(r=>Number(r.nominal_cycle_age ?? r.rpt_order));
  const minX=Math.min(...xs),maxX=Math.max(...xs);
  const x=i=>left+(xs[i]-minX)*(width-left-right)/(maxX-minX||1);
  let body="";
  series.forEach((s,j)=>{{
    const y0=top+j*(panel+gap), y1=y0+panel-25;
    const vals=rows.map(r=>{{
      const h=hybridByKey.get(key(r.dataset_part,r.physical_cell_id,r.rpt_id));
      const a=aggs.get(key(r.dataset_part,r.physical_cell_id,r.rpt_id));
      return Number(s.hybrid ? h?.[s.hybrid] : a?.[s.aggregate]);
    }});
    const finite=vals.filter(Number.isFinite), lo=finite.length?Math.min(...finite):0,hi=finite.length?Math.max(...finite):1,pad=(hi-lo||1)*.12;
    const y=v=>y0+8+(hi+pad-v)*(panel-38)/(hi-lo+2*pad);
    const pts=vals.map((v,i)=>Number.isFinite(v)?`${{x(i).toFixed(1)}} ${{y(v).toFixed(1)}}`:null).filter(Boolean).join(" ");
    body+=`<line x1="${{left}}" y1="${{y1}}" x2="${{width-right}}" y2="${{y1}}" stroke="#aab"/>`;
    body+=`<text x="4" y="${{y0+18}}" font-size="12">${{esc(s.name)}}</text>`;
    if(pts) body+=`<polyline points="${{pts}}" fill="none" stroke="${{s.color}}" stroke-width="2"/>`;
    vals.forEach((v,i)=>{{if(Number.isFinite(v)) body+=`<circle cx="${{x(i)}}" cy="${{y(v)}}" r="3.5" fill="${{s.color}}"/>`;else body+=`<path d="M ${{x(i)-4}} ${{y1-8}} l 8 8 m -8 0 l 8 -8" stroke="#b22222"/>`;}});
    if(j===series.length-1) rows.forEach((r,i)=>body+=`<text x="${{x(i)}}" y="${{height-4}}" text-anchor="middle" font-size="10">${{esc(r.rpt_id)}}</text>`);
  }});
  return `<svg id="trajectorySvg" viewBox="0 0 ${{width}} ${{height}}" aria-label="Cell degradation trajectories">${{body}}</svg>`;
}}
function render() {{
  const cp=selectedCheckpoint(); if(!cp) return;
  const hybrid=selectedHybrid();
  renderCards(cp,hybrid);
  renderStep4(cp);
  renderCurves(cp);
  $("trajectories").innerHTML=trajectorySvg(cp);
}}
function downloadSvg(id,name) {{
  const svg=$(id); if(!svg) return;
  const blob=new Blob([svg.outerHTML],{{type:"image/svg+xml"}});
  const url=URL.createObjectURL(blob),a=document.createElement("a");
  a.href=url;a.download=name;a.click();URL.revokeObjectURL(url);
}}
function downloadCsv(rows,name) {{
  if(!rows.length) return;
  const cols=[...new Set(rows.flatMap(r=>Object.keys(r)))];
  const quote=v=>`"${{String(v??"").replace(/"/g,'""')}}"`;
  const csv=[cols.map(quote).join(","),...rows.map(r=>cols.map(c=>quote(r[c])).join(","))].join("\\r\\n");
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([csv],{{type:"text/csv"}}));a.download=name;a.click();
}}
const parts=[...new Set(DATA.checkpoints.map(r=>String(r.dataset_part)))].sort((a,b)=>Number(a)-Number(b));
options($("part"),parts,parts[0]);
updateCells();
$("part").addEventListener("change",()=>updateCells());
$("cell").addEventListener("change",()=>updateRpts());
$("rpt").addEventListener("change",render);
$("fitFilter").addEventListener("change",render);
$("exportStep4").addEventListener("click",()=>downloadSvg("step4Svg","step4_voltage_input.svg"));
$("exportTrajectory").addEventListener("click",()=>downloadSvg("trajectorySvg","cell_trajectory.svg"));
$("exportCheckpoint").addEventListener("click",()=>{{
  const cp=selectedCheckpoint(); if(!cp)return;
  const checkpoint=DATA.hybrid.filter(r=>String(r.dataset_part)===String(cp.dataset_part)&&r.physical_cell_id===cp.physical_cell_id&&r.measurement_id===cp.rpt_id);
  const curves=DATA.curves.filter(r=>String(r.dataset_part)===String(cp.dataset_part)&&r.physical_cell_id===cp.physical_cell_id&&r.rpt_id===cp.rpt_id);
  downloadCsv([...checkpoint,...curves],`${{cp.physical_cell_id}}_${{cp.rpt_id}}_hybrid.csv`);
}});
render();
</script>
</body>
</html>
"""
    (output_dir / "dashboard.html").write_text(page, encoding="utf-8")
