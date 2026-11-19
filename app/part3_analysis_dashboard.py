from __future__ import annotations

import argparse
import json
import threading
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from app.part3_analysis_data import DashboardDataError, Part3AnalysisData
from app.tpg_mat_ingestion import TpgIngestionError


HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BaFuse | Part 3 Analysis</title>
<style>
:root{font-family:Inter,"Segoe UI",Arial,sans-serif;color:#142737;background:#f1f5f7;--navy:#123149;--teal:#147b83;--line:#dce5e9;--muted:#607482}
*{box-sizing:border-box}body{margin:0}header{background:linear-gradient(115deg,#102b40,#174a61);color:#fff;padding:1.55rem max(1rem,calc((100vw - 1440px)/2))}
header h1{font-size:clamp(1.45rem,3vw,2rem);margin:0 0 .4rem}header p{margin:0;color:#d4e4eb}
main{max-width:1440px;margin:auto;padding:1rem}section,.panel{background:#fff;border:1px solid var(--line);border-radius:13px;padding:1rem;margin-bottom:.9rem;box-shadow:0 3px 14px #17374b0a}
h2{font-size:1.08rem;margin:.1rem 0 .8rem}h3{font-size:.98rem;margin:.25rem 0 .6rem}
.layout{display:grid;grid-template-columns:280px minmax(0,1fr);gap:.9rem;align-items:start}
.sidebar{position:sticky;top:.7rem}.control{margin:.65rem 0}.control label{display:block;font-weight:650;font-size:.88rem;margin-bottom:.3rem}
select,input{font:inherit;border:1px solid #c9d6dd;border-radius:7px;background:#fff;padding:.55rem .6rem;color:#203849;width:100%}
select[multiple]{height:120px}.check{display:flex;align-items:center;gap:.5rem;font-size:.88rem;margin:.55rem 0}.check input{width:auto}
button{font:inherit;border:0;border-radius:7px;background:var(--teal);color:white;padding:.62rem .95rem;font-weight:650;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.secondary{background:#edf4f6;color:#204454;border:1px solid #d6e2e7}
.status{color:var(--muted);font-size:.9rem;min-height:1.4rem}.error{color:#a32b31}.success{color:#216743}.loading{display:inline-flex;align-items:center;gap:.5rem}
.spinner{width:16px;height:16px;border:2px solid #c9d8df;border-top-color:var(--teal);border-radius:50%;animation:spin .8s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:.6rem}.card{border:1px solid #e1e9ed;border-radius:9px;padding:.7rem .8rem;background:#fbfdfe}
.card span{display:block;color:var(--muted);font-size:.8rem}.card strong{display:block;font-size:1.25rem;margin-top:.25rem;color:var(--navy)}
.notice{border-left:4px solid #d28a22;background:#fff8e9;padding:.65rem .8rem;border-radius:5px;color:#684a16;font-size:.87rem;margin:.65rem 0}
.chart-grid{display:grid;grid-template-columns:1fr;gap:.75rem}.chart-box{border:1px solid #e3eaee;border-radius:9px;padding:.55rem;min-width:0}
.chart{width:100%;height:330px;display:block}.chart.small{height:250px}.legend{display:flex;flex-wrap:wrap;gap:.5rem .85rem;font-size:.83rem;color:#435b69;margin:.3rem 0}.legend span{display:inline-flex;align-items:center;gap:.35rem}
.swatch{width:17px;height:3px;display:inline-block;background:var(--swatch);border-radius:2px}.swatch.dash{background:repeating-linear-gradient(90deg,var(--swatch) 0 5px,transparent 5px 8px)}
.table-tools{display:flex;gap:.55rem;flex-wrap:wrap;margin:.45rem 0}.table-tools input{max-width:280px}.table-tools select{max-width:190px}
.table-wrap{overflow:auto;max-height:520px;border:1px solid var(--line);border-radius:8px}table{border-collapse:collapse;width:100%;min-width:1050px;font-size:.83rem}
th,td{text-align:left;padding:.48rem .52rem;border-bottom:1px solid #e5ecef;vertical-align:top}th{position:sticky;top:0;background:#f2f6f8;color:#324d5c;z-index:1}
th button{padding:0;background:transparent;color:inherit;font-weight:700;text-align:left}.observation-row{cursor:pointer}.observation-row:hover{background:#f4fafb}.selected-row{background:#e8f4f4}
.badge{display:inline-block;border-radius:20px;padding:.15rem .45rem;font-size:.75rem;font-weight:700;white-space:nowrap}.accepted{background:#e7f5ec;color:#24623b}.review{background:#fff3d8;color:#795515}.rejected{background:#fde9e8;color:#8f2f2c}.demo{background:#efe8ff;color:#594099}.muted{color:var(--muted);font-size:.86rem}.gap{color:#8f3030}.good{color:#236943}
.detail-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:.7rem}.detail-card{padding:.75rem;border:1px solid var(--line);border-radius:8px;background:#fbfdfe}
.detail-card p{margin:.3rem 0;overflow-wrap:anywhere;font-size:.86rem}.curve-grid{display:grid;grid-template-columns:1fr 1fr;gap:.65rem}.curve-svg{height:220px;width:100%;display:block}
.footer{color:var(--muted);font-size:.82rem;padding:.2rem .3rem 1rem}.hidden{display:none!important}
@media(max-width:900px){.layout{grid-template-columns:1fr}.sidebar{position:static}.chart{height:290px}}
@media(max-width:620px){main{padding:.55rem}.curve-grid{grid-template-columns:1fr}.chart{height:260px}section,.panel{padding:.75rem}}
</style>
</head>
<body>
<header>
<h1>Battery Health Analysis - Part 3</h1>
<p>Read-only exploration of verified Part 3 observations and frozen Final2 checkpoints.</p>
</header>
<main>
<div class="layout">
<aside class="sidebar">
<section>
<h2>Selection</h2>
<div class="control"><label for="groupSelect">Group</label><select id="groupSelect" disabled></select></div>
<div class="control"><label for="cellSelect">Physical cell</label><select id="cellSelect" disabled></select></div>
<div class="control"><label for="modelSelect">SoH model</label><select id="modelSelect" disabled></select></div>
<div class="control"><label for="compareSelect">Compare additional models</label><select id="compareSelect" multiple disabled></select></div>
<label class="check"><input id="showReference" type="checkbox" checked>Show reconstructed reference</label>
<label class="check"><input id="showPoints" type="checkbox" checked>Show individual points</label>
<label class="check"><input id="includeEol" type="checkbox" checked>Include verified EoL RPTs</label>
<button id="loadCell" disabled>Load selected cell</button>
<p id="status" class="status" role="status" aria-live="polite">Checking local analysis service...</p>
</section>
<section>
<h2>Model contract</h2>
<p id="modelInfo" class="muted">Loading frozen model metadata...</p>
<div class="notice">The default is selected by saved Validation RMSE only. External Test results are never used for model selection.</div>
</section>
</aside>
<div>
<section>
<h2>Coverage</h2>
<div class="cards">
<div class="card"><span>Mapped observations</span><strong id="mappedCount">-</strong></div>
<div class="card"><span>Inference eligible</span><strong id="eligibleCount">-</strong></div>
<div class="card"><span>Successful predictions</span><strong id="predictedCount">-</strong></div>
<div class="card"><span>Latest eligible canonical prediction</span><strong id="latestPrediction">N/A</strong></div>
<div class="card"><span>Latest eligible canonical reference</span><strong id="latestReference">N/A</strong></div>
</div>
<div class="notice">SoH references are reconstructed capacity-retention values, not Oxford official ground truth. EoL rows are demo-only and excluded from all evaluation metrics.</div>
</section>
<section>
<h2>SoH Evolution Across RPTs</h2>
<div id="sohLegend" class="legend"></div>
<div id="sohChart" class="chart" role="img" aria-label="SoH predictions and reconstructed references by RPT"></div>
<h3>Exploratory per-cell error summary</h3>
<p id="metricScope" class="muted">Frozen eligible observations only; not official Final2 External Test metrics.</p>
<div class="cards">
<div class="card"><span>MAE (pp)</span><strong id="maeMetric">N/A</strong></div>
<div class="card"><span>RMSE (pp)</span><strong id="rmseMetric">N/A</strong></div>
<div class="card"><span>R²</span><strong id="r2Metric">N/A</strong></div>
<div class="card"><span>Valid pairs</span><strong id="pairCount">0</strong></div>
</div>
</section>
<section>
<h2>Degradation Evolution Across RPTs</h2>
<div id="degradationLegend" class="legend"></div>
<div id="degradationChart" class="chart" role="img" aria-label="Saved Charge PyDMA LLI and LAM by RPT"></div>
<p class="muted">Saved Charge-only PyDMA estimates are diagnostic, not independent ground truth. The chart includes only fits passing RMSE &lt; 15 mV, at least 3 accepted runs, and same-cell BoL QC. Missing and diagnostic-only fits remain gaps.</p>
</section>
<section>
<h2>RPT Observations</h2>
<div class="table-tools">
<input id="tableSearch" type="search" placeholder="Filter source, RPT, or reason">
<select id="statusFilter"><option value="">All statuses</option><option>ACCEPTED</option><option>REVIEW</option><option>REJECTED</option><option>DEMO_ONLY</option></select>
</div>
<div class="table-wrap">
<table id="observationTable">
<thead><tr>
<th><button data-sort="rpt_index">RPT</button></th><th><button data-sort="procedure">Procedure</button></th><th><button data-sort="source_file">Source</button></th><th><button data-sort="status">Original / data QC</button></th>
<th><button data-sort="reference_soh_percent">SoH reference</button></th><th><button data-sort="predicted_soh_percent">Predicted SoH</button></th><th>Abs. error</th>
<th>LLI</th><th>LAM Anode</th><th>LAM Cathode</th><th>Modality / PyDMA status</th>
</tr></thead><tbody id="observationBody"></tbody>
</table></div>
<p id="tableSummary" class="muted"></p>
</section>
<section id="detailSection">
<h2>Individual RPT Inspection</h2>
<p id="detailPrompt" class="muted">Select a row to inspect mapping provenance, inference eligibility, saved PyDMA QC and source curves.</p>
<div id="detailContent" class="hidden"></div>
</section>
</div>
</div>
<p class="footer">This standalone application reads frozen local artifacts. It does not train models, fit scalers, run PyDMA, modify datasets, or update official evaluation metrics.</p>
</main>
<script>
const $=id=>document.getElementById(id);
let meta=null,cellData=null,sortKey="rpt_index",sortAscending=true,selectedKey="";
const modelColors={"CNN1D-Step4-Baseline":"#2270ad","CNN-3M":"#7651b5","CNN-2M":"#e58b24","TCN-2M":"#116e75","ResNet-2M":"#d14e5a","DenseNet-2M":"#4d8c46","ConvNeXt-2M":"#765c38"};
const degradationSeries=[["lli_percent","LLI","#16858b"],["lam_anode_percent","LAM Anode","#d37a1c"],["lam_cathode_percent","LAM Cathode","#7251a1"]];
function escapeHtml(value){return String(value??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));}
function finite(value){return value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value));}
function fmt(value,digits=2){return finite(value)?Number(value).toFixed(digits):"—";}
function selectedModels(){const chosen=Array.from($("compareSelect").selectedOptions).map(o=>o.value);return [...new Set([ $("modelSelect").value,...chosen ].filter(Boolean))];}
function setStatus(message,state=""){ $("status").textContent=message;$("status").className=`status ${state}`; }
function selectedModelInfo(){const model=meta?.model_catalog.find(m=>m.name===$("modelSelect").value);if(!model)return;
 const v=model.validation_metrics;const available=model.checkpoint_status?.available;
 $("modelInfo").textContent=`${model.name} | ${model.architecture}\nInputs: ${model.modalities.join(", ")}\nValidation RMSE: ${finite(v?.validation_rmse_pp)?Number(v.validation_rmse_pp).toFixed(3)+" pp":"not recorded"}\nCheckpoint: ${available?"verified frozen checkpoint":model.checkpoint_status?.reason||"unavailable"}`;
}
async function loadMeta(){
 try{
  const [healthResponse,metaResponse]=await Promise.all([fetch("/api/health",{cache:"no-store"}),fetch("/api/meta",{cache:"no-store"})]);
  const health=await healthResponse.json(),payload=await metaResponse.json();if(!healthResponse.ok||!metaResponse.ok)throw Error(payload.error||health.error||"Metadata unavailable.");
  meta=payload;$("groupSelect").replaceChildren(...meta.groups.map(g=>new Option(`Group ${g}`,g)));
  const available=meta.model_catalog.filter(m=>m.checkpoint_status?.available);
  $("modelSelect").replaceChildren(...available.map(m=>new Option(m.name,m.name)));
  $("compareSelect").replaceChildren(...available.map(m=>new Option(m.name,m.name)));
  $("modelSelect").value=meta.default_model;
  for(const option of $("compareSelect").options)option.selected=false;
  updateCells();$("groupSelect").disabled=false;$("cellSelect").disabled=false;$("modelSelect").disabled=false;$("compareSelect").disabled=false;$("loadCell").disabled=false;
  $("modelInfo").textContent=`${meta.cell_count} verified cells; ${meta.canonical_rpt_count} mapped RPTs, ${meta.demo_only_eol_count} verified demo-only EoL records, and ${meta.frozen_external_count} frozen External Test observations.`;
  selectedModelInfo();setStatus(`Local data ready (${health.available_model_count} frozen checkpoints available).`,"success");await loadCellData();
 }catch(error){setStatus(`Could not load Part 3 data: ${error.message||error}`,"error");}
}
function updateCells(){
 const cells=meta?.cells_by_group[$("groupSelect").value]||[];$("cellSelect").replaceChildren(...cells.map(c=>new Option(c,c)));
}
async function loadCellData(){
 if(!meta)return;const group=$("groupSelect").value,cell=$("cellSelect").value;if(!group||!cell)return;
 const models=selectedModels();$("loadCell").disabled=true;setStatus(`Loading ${cell} observations and validated model inputs...`);
 $("sohChart").innerHTML='<div class="loading"><span class="spinner"></span>Loading observations and running frozen inference...</div>';
 try{
  const query=new URLSearchParams({group,cell,models:models.join(","),include_eol:String($("includeEol").checked)});
  const response=await fetch(`/api/cell?${query}`,{cache:"no-store"});const payload=await response.json();if(!response.ok)throw Error(payload.error||"Cell data unavailable.");
  cellData=payload;selectedKey="";renderCell();setStatus(`${cell}: ${payload.observation_count} mapped observations loaded; predictions use ${models.join(", ")}.`,"success");
 }catch(error){cellData=null;setStatus(`Cell analysis failed: ${error.message||error}`,"error");$("sohChart").textContent="No chart data.";clearSummary();}
 finally{$("loadCell").disabled=false;}
}
function clearSummary(){for(const id of ["mappedCount","eligibleCount","predictedCount","latestPrediction","latestReference","maeMetric","rmseMetric","r2Metric","pairCount"])$(id).textContent=id==="pairCount"?"0":"—";$("observationBody").replaceChildren();}
function badge(status,isEol){const cls=isEol?"demo":status==="ACCEPTED"?"accepted":status==="REVIEW"?"review":status==="REJECTED"?"rejected":"demo";return `<span class="badge ${cls}">${escapeHtml(isEol?"DEMO-ONLY EoL":status)}</span>`;}
function currentPrediction(row){return row.predictions?.[cellData.selected_model]||{};}
function currentValue(row){return currentPrediction(row).predicted_soh_percent;}
function renderCell(){
 const rows=cellData.observations,summary=cellData.summary,metrics=cellData.metrics;
 $("mappedCount").textContent=summary.mapped_rpt_count;$("eligibleCount").textContent=summary.inference_eligible_count;$("predictedCount").textContent=summary.successful_prediction_count;
 $("latestPrediction").textContent=finite(summary.latest_predicted_soh_percent)?`${fmt(summary.latest_predicted_soh_percent)}%`:"N/A";
 $("latestReference").textContent=finite(summary.latest_reference_soh_percent)?`${fmt(summary.latest_reference_soh_percent)}%`:"N/A";
 $("maeMetric").textContent=finite(metrics.mae_pp)?fmt(metrics.mae_pp):"N/A";$("rmseMetric").textContent=finite(metrics.rmse_pp)?fmt(metrics.rmse_pp):"N/A";
 $("r2Metric").textContent=finite(metrics.r2)?fmt(metrics.r2,3):"N/A";$("pairCount").textContent=metrics.n;
 $("metricScope").textContent=`${metrics.scope}. Valid pairs: ${metrics.n}; EoL, REVIEW and REJECTED rows are excluded. ${metrics.r2_reason?`R² withheld: ${metrics.r2_reason}.`:""}`;
 renderSohChart();renderDegradationChart();renderTable();
}
function chartSvg(containerId,rows,series,{showPoints=true,domain=null,markers=false}={}){
 const container=$(containerId),width=Math.max(container.clientWidth||760,320),height=container.clientHeight||330;
 const left=48,right=15,top=14,bottom=43,plotW=width-left-right,plotH=height-top-bottom;
 const values=[];for(const s of series)for(const row of rows){const v=s.value(row);if(finite(v))values.push(Number(v));}
 if(!values.length){container.innerHTML='<div class="muted">No eligible values. Missing observations are shown as gaps and are not interpolated.</div>';return;}
 let min=domain?domain[0]:Math.min(...values),max=domain?domain[1]:Math.max(...values);if(min===max){min-=1;max+=1;}const pad=(max-min)*.08;min=domain?min:min-pad;max=domain?max:max+pad;
 const x=i=>left+(rows.length<=1?plotW/2:i*plotW/(rows.length-1)),y=v=>top+(max-Number(v))*plotH/(max-min);
 let svg=`<svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" role="img">`;
 for(let i=0;i<5;i++){const value=min+(max-min)*i/4,yy=y(value);svg+=`<line x1="${left}" y1="${yy}" x2="${width-right}" y2="${yy}" stroke="#e5ecef"/><text x="${left-7}" y="${yy+4}" text-anchor="end" fill="#617580" font-size="11">${value.toFixed(1)}</text>`;}
 svg+=`<line x1="${left}" y1="${top+plotH}" x2="${width-right}" y2="${top+plotH}" stroke="#8095a0"/>`;
 rows.forEach((row,i)=>{const label=row.chart_rpt||`RPT${row.rpt_index}`;if(rows.length<=12||i%Math.ceil(rows.length/12)===0||i===rows.length-1)svg+=`<text x="${x(i)}" y="${height-12}" text-anchor="middle" fill="#536976" font-size="10">${escapeHtml(label)}</text>`;});
 for(const s of series){
  let d="",pen=false;
  rows.forEach((row,i)=>{const value=s.value(row);if(!finite(value)){pen=false;return;}d+=`${pen?"L":"M"}${x(i).toFixed(1)},${y(value).toFixed(1)} `;pen=true;});
  svg+=`<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width||2.4}" ${s.dash?'stroke-dasharray="7 5"':""}/>`;
  if(showPoints)rows.forEach((row,i)=>{const value=s.value(row);if(!finite(value))return;const title=`${row.chart_rpt||"RPT"+row.rpt_index}: ${Number(value).toFixed(3)}${s.unit||"%"}`;const fill=markers?(row.is_eol?"#7651b5":row.status==="REVIEW"?"#d28a22":row.status==="REJECTED"?"#9aa8ae":s.color):s.color;
   if(markers&&row.is_eol){svg+=`<path d="M${x(i)},${y(value)-5} L${x(i)+5},${y(value)} L${x(i)},${y(value)+5} L${x(i)-5},${y(value)} Z" fill="${fill}" stroke="#fff" stroke-width="1"><title>${escapeHtml(title)}</title></path>`;}
   else svg+=`<circle cx="${x(i)}" cy="${y(value)}" r="${markers?4:3}" fill="${fill}" stroke="#fff" stroke-width="1"><title>${escapeHtml(title)}</title></circle>`;
  });
 }
 svg+="</svg>";container.innerHTML=svg;
}
function renderLegend(containerId,series){$(containerId).innerHTML=series.map(s=>`<span><i class="swatch ${s.dash?"dash":""}" style="--swatch:${s.color}"></i>${escapeHtml(s.label)}</span>`).join("");}
function renderSohChart(){
 const rows=cellData.observations,models=cellData.models,series=[];
 for(const name of models)series.push({label:`${name} prediction`,color:modelColors[name]||"#333",value:r=>r.predictions?.[name]?.predicted_soh_percent});
 if($("showReference").checked)series.push({label:"Reconstructed reference",color:"#32824b",dash:true,value:r=>r.reference_soh_percent});
 renderLegend("sohLegend",series);chartSvg("sohChart",rows,series,{showPoints:$("showPoints").checked,markers:true});
}
function renderDegradationChart(){
 const rows=cellData.observations,series=degradationSeries.map(([key,label,color])=>({label,color,value:r=>r.degradation?.chart_eligible?r.degradation[key]:null}));
 renderLegend("degradationLegend",series);chartSvg("degradationChart",rows,series,{showPoints:$("showPoints").checked});
}
function escapeCellReason(row){const p=currentPrediction(row);return p.reason||row.failure_reason||row.status_reason||row.recommended_action||"";}
function renderTable(){
 const search=$("tableSearch").value.trim().toLowerCase(),filter=$("statusFilter").value;
 const rows=[...cellData.observations].sort((a,b)=>{let av=a[sortKey],bv=b[sortKey];if(sortKey==="predicted_soh_percent"){av=currentValue(a);bv=currentValue(b);}if(sortKey==="rpt_index"){av=Number(a.rpt_index)+(a.is_eol?.25:0);bv=Number(b.rpt_index)+(b.is_eol?.25:0);}
  if(av===null||av===undefined)return 1;if(bv===null||bv===undefined)return -1;const compare=typeof av==="number"&&typeof bv==="number"?av-bv:String(av).localeCompare(String(bv));return sortAscending?compare:-compare;});
 const visible=rows.filter(row=>(!filter||(row.is_eol?filter==="DEMO_ONLY":row.status===filter))&&(!search||`${row.chart_rpt} ${row.procedure} ${row.source_file} ${row.status_reason} ${row.data_qc_status||""} ${row.original_review_reason||""} ${row.resolved_issue||""} ${row.remaining_issue||""} ${row.evaluation_membership||""} ${escapeCellReason(row)}`.toLowerCase().includes(search)));
 $("observationBody").innerHTML=visible.map(row=>{
  const prediction=currentPrediction(row),pred=prediction.predicted_soh_percent,reference=row.reference_soh_percent;
  const error=finite(pred)&&finite(reference)&&row.evaluation_eligible?Math.abs(Number(pred)-Number(reference)):null;
  const d=row.degradation||{},diag=d.diagnostic_values;
  const lli=d.chart_eligible?d.lli_percent:diag?.lli_percent,anode=d.chart_eligible?d.lam_anode_percent:diag?.lam_anode_percent,cathode=d.chart_eligible?d.lam_cathode_percent:diag?.lam_cathode_percent;
  const modality=`Step4 ${row.step4_eligible?"yes":"no"} / Step10 ${row.step10_eligible?"yes":"no"} / P5 ${row.p5_eligible?"yes":"no"}`;
  const statusText=row.is_eol?"DEMO_ONLY":row.status;
  return `<tr class="observation-row ${selectedKey===JSON.stringify(row.key)?"selected-row":""}" data-key="${escapeHtml(JSON.stringify(row.key))}">
  <td>${escapeHtml(row.chart_rpt)}</td><td>${escapeHtml(row.procedure||"—")}</td><td title="${escapeHtml(row.source_file)}">${escapeHtml(row.source_file.split("/").pop())}</td>
  <td>${badge(statusText,row.is_eol)}${row.data_qc_status?`<br><small>${escapeHtml(row.data_qc_status)}</small>`:""}</td><td>${fmt(reference)}${finite(reference)?"%":""}<br><small>${escapeHtml(row.reference_status||"")}</small></td>
  <td>${fmt(pred)}${finite(pred)?"%":""}<br><small>${escapeHtml(prediction.status||"")}</small></td><td>${fmt(error)}</td>
  <td>${fmt(lli)}</td><td>${fmt(anode)}</td><td>${fmt(cathode)}</td>
  <td title="${escapeHtml([escapeCellReason(row),d.reason,d.same_cell_bol_note].filter(Boolean).join(" | "))}">${escapeHtml(modality)}<br><small>${escapeHtml(d.status||"PyDMA unavailable")}</small></td></tr>`;
 }).join("");
 $("tableSummary").textContent=`Showing ${visible.length} of ${cellData.observations.length} mapped observations. Click a row for detailed provenance and source curves.`;
 document.querySelectorAll(".observation-row").forEach(row=>row.addEventListener("click",()=>inspectRPT(JSON.parse(row.dataset.key))));
}
async function inspectRPT(key){
 selectedKey=JSON.stringify(key);renderTable();$("detailPrompt").textContent="Loading exact RPT provenance and source curves...";
 $("detailContent").classList.add("hidden");
 try{
  const query=new URLSearchParams({group:key[1],cell:key[2],rpt:key[3],include_eol:String($("includeEol").checked),model:cellData.selected_model});
  const response=await fetch(`/api/rpt?${query}`,{cache:"no-store"});const payload=await response.json();if(!response.ok)throw Error(payload.error||"RPT detail unavailable.");
  renderDetail(payload);
 }catch(error){$("detailPrompt").textContent=`RPT inspection failed: ${error.message||error}`;}
}
function renderDetail(payload){
 const row=payload.observation,curves=payload.curves,d=row.degradation||{},pred=row.predictions?.[cellData.selected_model]||{};
 const diag=d.diagnostic_values;const shown=v=>finite(v)?`${fmt(v)}%`:"Unavailable";
 const values=[
  ["Identity",`Group ${row.group} / ${row.cell} / ${row.chart_rpt}`],
  ["Procedure / source",`${row.procedure||"—"} | ${row.source_file}`],
  ["Mapping status",`Original audit: ${row.mapping_status||"—"} | Review resolution: ${row.review_mapping_status||"—"} | Canonical: ${row.canonical_mapping_status||"—"} | Guide: ${row.canonical_guide_mapping_status||"—"}`],
  ["Mapping provenance",`Audit: ${row.identity_provenance||"—"} | Cell identity: ${row.canonical_cell_identity_status||"—"} | Guide note: ${row.guide_mapping_status||"—"}`],
  ["Status / partition",`${row.status} | ${row.partition}`],
  ["Data-QC resolution",`${row.original_review_reason||"—"} | ${row.data_qc_status||"not reviewed"} | ${row.resolved_issue||row.remaining_issue||"No unresolved issue."}`],
  ["Model input eligibility",`Step4=${row.step4_eligible}; Step10=${row.step10_eligible}; P5=${row.p5_eligible}`],
  ["SoH prediction",`${shown(pred.predicted_soh_percent)} | ${pred.reason||pred.status||""}`],
  ["SoH reference",`${shown(row.reference_soh_percent)} | ${row.reference_status||"—"} | available=${finite(row.reference_soh_percent)} | evaluation eligible=${row.evaluation_eligible}`],
  ["Frozen evaluation membership",`${row.evaluation_membership||"—"} | ${row.evaluation_eligible?"included":"excluded"}`],
  ["Charge PyDMA QC",`${d.status||"unavailable"} | RMSE=${finite(d.fit_rmse_mV)?fmt(d.fit_rmse_mV,3)+" mV":"—"} | runs=${d.accepted_runs??"—"}/${d.total_runs??"—"}`],
  ["LLI / LAM Anode / LAM Cathode",`${shown(d.lli_percent??diag?.lli_percent)} / ${shown(d.lam_anode_percent??diag?.lam_anode_percent)} / ${shown(d.lam_cathode_percent??diag?.lam_cathode_percent)}`],
  ["Same-cell BoL",`${d.same_cell_bol_source||"—"} | ${d.same_cell_bol_status||d.reason||"not recorded"}`],
  ["Availability reason",[escapeCellReason(row),d.reason].filter(Boolean).join(" | ")||"No blocking reason."]
 ];
 $("detailContent").innerHTML=`<div class="detail-grid">${values.map(([k,v])=>`<div class="detail-card"><strong>${escapeHtml(k)}</strong><p>${escapeHtml(v)}</p></div>`).join("")}</div>
 <div class="curve-grid"><div class="detail-card"><strong>Step 4 discharge voltage</strong><div id="step4Curve"></div></div><div class="detail-card"><strong>Step 10 charge voltage</strong><div id="step10Curve"></div></div></div>
 <p class="muted">${payload.curve_error?`Curve unavailable: ${escapeHtml(payload.curve_error)}`:"Curves are read lazily from the exact mapped source; processed traces use the repository's 128-point row-progress interpolation."}</p>`;
 $("detailPrompt").textContent="";$("detailContent").classList.remove("hidden");
 drawCurve("step4Curve",curves?.step4,"Step 4");drawCurve("step10Curve",curves?.step10,"Step 10");
}
function drawCurve(id,curve,label){
 const host=$(id);if(!curve?.available||!curve.raw_voltage?.length){host.innerHTML=`<p class="gap">Unavailable: ${escapeHtml(curve?.reason||"no verified contiguous segment")}</p>`;return;}
 const values=curve.raw_voltage.map(Number),processed=(curve.processed_128_voltage||[]).map(Number),all=values.filter(Number.isFinite).concat(processed.filter(Number.isFinite));
 if(!all.length){host.textContent="No finite voltage values.";return;}
 let min=Math.min(...all),max=Math.max(...all);if(min===max){min-=.1;max+=.1;}const w=480,h=190,p=22,x=(i,n)=>p+i*(w-2*p)/Math.max(1,n-1),y=v=>h-p-(v-min)*(h-2*p)/(max-min);
 const path=a=>a.map((v,i)=>`${i?"L":"M"}${x(i,a.length).toFixed(1)},${y(v).toFixed(1)}`).join(" ");
 host.innerHTML=`<svg class="curve-svg" viewBox="0 0 ${w} ${h}" role="img" aria-label="${label} raw and 128 point voltage">${[0,1,2,3].map(i=>`<line x1="${p}" y1="${p+i*(h-2*p)/3}" x2="${w-p}" y2="${p+i*(h-2*p)/3}" stroke="#e3eaee"/>`).join("")}<path d="${path(values)}" fill="none" stroke="#286ca0" stroke-width="1.5"/><path d="${path(processed)}" fill="none" stroke="#d27a1b" stroke-width="2" stroke-dasharray="6 4"/></svg><div class="legend"><span><i class="swatch" style="--swatch:#286ca0"></i>Raw (${values.length})</span><span><i class="swatch dash" style="--swatch:#d27a1b"></i>128-point model input</span></div>`;
}
$("groupSelect").addEventListener("change",()=>{updateCells();loadCellData();});
$("cellSelect").addEventListener("change",loadCellData);$("modelSelect").addEventListener("change",()=>{selectedModelInfo();loadCellData();});
$("compareSelect").addEventListener("change",loadCellData);$("includeEol").addEventListener("change",loadCellData);
$("loadCell").addEventListener("click",loadCellData);$("showReference").addEventListener("change",()=>cellData&&renderSohChart());$("showPoints").addEventListener("change",()=>cellData&&(renderSohChart(),renderDegradationChart()));
$("tableSearch").addEventListener("input",()=>cellData&&renderTable());$("statusFilter").addEventListener("change",()=>cellData&&renderTable());
document.querySelectorAll("th button[data-sort]").forEach(button=>button.addEventListener("click",()=>{const next=button.dataset.sort;sortAscending=sortKey===next?!sortAscending:true;sortKey=next;cellData&&renderTable();}));
window.addEventListener("resize",()=>{if(cellData){renderSohChart();renderDegradationChart();}});
loadMeta();
</script>
</body>
</html>"""


class _DashboardServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _handler_for(service: Part3AnalysisData) -> type[BaseHTTPRequestHandler]:
    class DashboardHandler(BaseHTTPRequestHandler):
        server_version = "BaFusePart3Dashboard/1.0"

        def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=True, allow_nan=False).encode("utf-8")
            self._send(status, body, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(
                    HTTPStatus.OK,
                    HTML_PAGE.encode("utf-8"),
                    "text/html; charset=utf-8",
                )
                return
            try:
                if parsed.path == "/api/health":
                    self._json(HTTPStatus.OK, service.health())
                    return
                if parsed.path == "/api/meta":
                    self._json(HTTPStatus.OK, service.metadata())
                    return
                query = parse_qs(parsed.query, keep_blank_values=True)
                if parsed.path == "/api/cell":
                    group = query.get("group", [""])[0]
                    cell = query.get("cell", [""])[0]
                    if not group or not cell:
                        self._json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Both group and cell are required."},
                        )
                        return
                    model_names = [
                        name
                        for item in query.get("models", [])
                        for name in item.split(",")
                        if name.strip()
                    ]
                    include_eol = query.get("include_eol", ["true"])[0].casefold() not in {
                        "false",
                        "0",
                        "no",
                    }
                    self._json(
                        HTTPStatus.OK,
                        service.cell_data(
                            group,
                            cell,
                            model_names=model_names or None,
                            include_eol=include_eol,
                        ),
                    )
                    return
                if parsed.path == "/api/rpt":
                    group = query.get("group", [""])[0]
                    cell = query.get("cell", [""])[0]
                    rpt = query.get("rpt", [""])[0]
                    if not group or not cell or not rpt:
                        self._json(
                            HTTPStatus.BAD_REQUEST,
                            {"error": "Group, cell, and RPT are required."},
                        )
                        return
                    include_eol = query.get("include_eol", ["true"])[0].casefold() not in {
                        "false",
                        "0",
                        "no",
                    }
                    self._json(
                        HTTPStatus.OK,
                        service.rpt_detail(
                            group,
                            cell,
                            rpt,
                            include_eol=include_eol,
                            model_name=query.get("model", [""])[0] or None,
                        ),
                    )
                    return
                self._json(HTTPStatus.NOT_FOUND, {"error": "Not found."})
            except DashboardDataError as error:
                self._json(HTTPStatus.BAD_REQUEST, {"error": str(error)})
            except (TpgIngestionError, OSError, UnicodeDecodeError, RuntimeError) as error:
                self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": f"{type(error).__name__}: {error}"},
                )

        def log_message(self, format_string: str, *args: Any) -> None:
            super().log_message(format_string, *args)

    return DashboardHandler


def create_server(
    host: str = "127.0.0.1",
    port: int = 8766,
    service: Part3AnalysisData | None = None,
) -> _DashboardServer:
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("The research dashboard must bind to loopback only.")
    data_service = service or Part3AnalysisData()
    return _DashboardServer((host, port), _handler_for(data_service))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone read-only Part 3 Battery Health Analysis Dashboard."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--open-browser", action="store_true")
    arguments = parser.parse_args(argv)
    server = create_server(arguments.host, arguments.port)
    address = f"http://{arguments.host}:{arguments.port}"
    print(f"Part 3 analysis dashboard: {address}")
    print("Read-only mode: no training, scaler fitting, PyDMA, or artifact writes.")
    if arguments.open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Part 3 analysis dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
