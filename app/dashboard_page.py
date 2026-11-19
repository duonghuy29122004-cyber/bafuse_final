HTML_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BaFuse — Battery Health &amp; Degradation Analysis</title>
<style>
:root{font-family:Inter,"Segoe UI",Arial,sans-serif;color:#142333;background:#f3f6f8}
*{box-sizing:border-box}body{margin:0}header{background:#123149;color:#fff;padding:2rem max(1.25rem,calc((100vw - 1180px)/2))}
header h1{font-size:clamp(1.6rem,3vw,2.2rem);margin:0 0 .45rem}header p{margin:0;color:#d7e5ef}
main{max-width:1180px;margin:auto;padding:1.2rem}section{background:#fff;border:1px solid #dce5eb;border-radius:14px;padding:1.25rem;margin:0 0 1rem;box-shadow:0 3px 14px #14324b0a}
h2{font-size:1.2rem;margin:.1rem 0 1rem}.upload-zone{border:2px dashed #9bb1c1;border-radius:12px;padding:1.5rem;text-align:center;background:#f8fbfd}
.upload-zone strong{display:block;font-size:1.1rem;margin-bottom:.3rem}.upload-zone input{max-width:100%;margin-top:.7rem}
button{background:#126b85;color:white;border:0;border-radius:8px;padding:.7rem 1.3rem;font-size:1rem;font-weight:650;cursor:pointer;margin-top:.85rem}
button:disabled{opacity:.55;cursor:not-allowed}.summary-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:.75rem}
.card{border:1px solid #e0e8ed;border-radius:10px;padding:.8rem 1rem;background:#fbfdfe}.card span{display:block;color:#607482;font-size:.88rem}
.card strong{display:block;font-size:1.55rem;margin-top:.3rem;color:#123149}.status{min-height:1.4rem;margin:.65rem 0 0;color:#536976}.error{color:#a4262c}.success{color:#20734b}[hidden]{display:none!important}
.chart-wrap{position:relative;margin-top:.7rem}.chart{display:block;width:100%;height:330px}.tooltip{display:none;position:absolute;z-index:2;background:#102738;color:white;border-radius:6px;padding:.45rem .6rem;font-size:.83rem;pointer-events:none;white-space:pre-line}
.legend{display:flex;flex-wrap:wrap;gap:.55rem;margin:.25rem 0 .5rem}.legend button{margin:0;background:#f3f6f8;color:#233c4d;border:1px solid #d8e2e8;padding:.35rem .65rem;font-size:.88rem}
.legend button[aria-pressed=false]{opacity:.45;text-decoration:line-through}.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:.35rem}
.plot-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:1rem}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem}
.diagnostic{border-radius:8px;padding:.65rem .8rem;margin:.5rem 0;background:#f2f6f8;color:#354c5a;font-size:.9rem}
.diagnostic.error{background:#fff1f0;color:#8f252b}.diagnostic.ok{background:#edf8f1;color:#23613d}
details{margin:.75rem 0}summary{cursor:pointer;font-weight:650;color:#234357}
.table-wrap{overflow:auto;margin-top:.7rem}table{border-collapse:collapse;width:100%;font-size:.9rem;min-width:620px}
th,td{text-align:left;padding:.55rem .65rem;border-bottom:1px solid #e0e8ed;vertical-align:top}th{background:#f5f8fa;color:#354d5c}
.chart-title{font-size:1rem;margin:.5rem 0}.reason{color:#8f252b}.available{color:#23613d}
.muted{color:#637784;font-size:.9rem}.loading{display:inline-flex;align-items:center;gap:.55rem}.spinner{width:17px;height:17px;border:2px solid #c8d8e1;border-top-color:#126b85;border-radius:50%;animation:spin .8s linear infinite}
.model-info{white-space:pre-line;text-align:left;max-width:760px;margin:.65rem auto 0}.model-info strong{color:#234357}
@keyframes spin{to{transform:rotate(360deg)}}footer{color:#637784;text-align:center;font-size:.85rem;padding:0 1rem 1.5rem}
@media(max-width:650px){main{padding:.7rem}section{padding:1rem}.chart{height:280px}}
</style>
</head>
<body>
<header>
<h1>BaFuse — Battery Health &amp; Degradation Analysis</h1>
<p>Upload battery test data to estimate State of Health and analyze degradation across RPTs.</p>
</header>
<main>
<section aria-labelledby="uploadTitle">
<h2 id="uploadTitle">Upload TPG File</h2>
<div class="upload-zone">
<strong>Upload TPG File</strong>
<span class="muted">Choose one raw TPG <code>.mat</code> or <code>.csv</code> file for a battery cell.</span><br>
<input id="tpgFile" type="file" accept=".mat,.csv">
<p><label for="modelSelect">Frozen SOH model</label>
<select id="modelSelect" disabled aria-describedby="modelInfo"></select></p>
<div id="modelInfo" class="muted model-info">Loading validation-selected models…</div>
</div>
<button id="analyze" disabled>Analyze Battery</button>
<div id="status" class="status" role="status" aria-live="polite">Checking local analysis service…</div>
<div id="cellSummary" class="summary-grid" hidden>
<div class="card"><span>Cell ID</span><strong id="cellId">—</strong></div>
</div>
</section>

<section aria-labelledby="inputsTitle">
<h2 id="inputsTitle">Input Visualization</h2>
<label for="rptSelect">Resolved RPT checkpoint</label>
<select id="rptSelect" disabled></select>
<div id="rptProvenance" class="diagnostic" role="status" hidden></div>
<div id="mappingNotice" class="diagnostic" hidden></div>
<div id="inputStatuses" aria-live="polite"></div>
<div class="plot-grid">
<div><h3 class="chart-title">Step 4 — Discharge Voltage</h3><canvas id="step4Chart" class="chart" aria-label="Step 4 original and resampled discharge voltage"></canvas><h3 class="chart-title">Step 4 — Normalized frozen-model input</h3><canvas id="step4ModelChart" class="chart" aria-label="Normalized Step 4 model input"></canvas></div>
<div><h3 class="chart-title">Step 10 — Charge Voltage</h3><canvas id="step10Chart" class="chart" aria-label="Step 10 original and resampled charge voltage"></canvas><h3 class="chart-title">Step 10 — Normalized frozen-model input</h3><canvas id="step10ModelChart" class="chart" aria-label="Normalized Step 10 model input"></canvas></div>
</div>
<div class="legend muted"><span>Blue: source samples</span><span>Orange: 128-point resampling</span></div>
<details open><summary>Five P5 electrical features (required by CNN-3M only)</summary>
<div class="table-wrap"><table><thead><tr><th>Feature</th><th>Measured feature value</th><th>Normalized model input</th><th>Status</th></tr></thead><tbody id="physicsTable"></tbody></table></div>
</details>
<p class="muted">Solid line: original verified voltage samples. Dashed line: the repository's 128-point resampling. The frozen checkpoint then applies its saved mean/scale normalization.</p>
</section>

<section aria-labelledby="sohTitle">
<h2 id="sohTitle">SoH Prediction vs Reference</h2>
<div class="summary-grid">
<div class="card"><span>Latest predicted SoH</span><strong id="latestSoh">N/A</strong></div>
<div class="card"><span>Latest reference SoH</span><strong id="latestReference">N/A</strong></div>
<div class="card"><span>Model used for current results</span><strong id="activeModel">N/A</strong></div>
</div>
<p class="muted">Reference values are provisional same-cell capacity-ratio estimates, not official dataset ground truth. Part 3 is external/domain-shift evaluation. Changing the selected frozen model changes SOH inference only; the same-RPT reference and saved degradation joins are unchanged.</p>
<div class="legend muted" id="sohLegend"><span>Blue: predicted SoH</span><span>Orange: provisional reference</span></div>
<div class="chart-wrap"><canvas id="sohChart" class="chart" aria-label="Predicted and reference SoH by RPT"></canvas></div>
<div class="table-wrap"><table><thead><tr><th>RPT</th><th>Reference SoH (%)</th><th>Predicted SoH (%)</th><th>Signed Error (pp)</th><th>Absolute Error (pp)</th><th>Availability / reason</th></tr></thead><tbody id="sohTable"></tbody></table></div>
</section>

<section aria-labelledby="evaluationTitle">
<h2 id="evaluationTitle">Evaluation Metrics and Error Analysis</h2>
<p class="muted">Metrics use only finite predictions matched to an eligible reference for the exact Part, cell, and RPT. MAE/RMSE are percentage points.</p>
<div class="metric-grid">
<div class="card"><span>MAE (pp)</span><strong id="maeMetric">N/A</strong><small id="maeReason"></small></div>
<div class="card"><span>RMSE (pp)</span><strong id="rmseMetric">N/A</strong><small id="rmseReason"></small></div>
<div class="card"><span>R²</span><strong id="r2Metric">N/A</strong><small id="r2Reason"></small></div>
<div class="card"><span>Evaluated RPTs (N)</span><strong id="nMetric">0</strong></div>
</div>
<div class="plot-grid">
<div><h3 class="chart-title">Predicted vs Reference SoH</h3><canvas id="scatterChart" class="chart" aria-label="Scatter plot of predicted versus reference SoH"></canvas></div>
<div><h3 class="chart-title">Signed Error by RPT</h3><canvas id="errorChart" class="chart" aria-label="Signed SoH prediction error by RPT"></canvas></div>
</div>
</section>

<section aria-labelledby="degradationTitle">
<h2 id="degradationTitle">Degradation Analysis</h2>
<h3 class="chart-title">Accepted Charge-only PyDMA degradation (primary display)</h3>
<p class="muted">Existing accepted Charge fits only (RMSE &lt; 15 mV and at least three accepted runs), joined by exact Part/Group/Cell/RPT. These diagnostic estimates are shared across SOH model selections; unavailable fits are not filled or interpolated.</p>
<div id="chargeFitMessage" class="diagnostic" hidden></div>
<div class="summary-grid">
<div class="card"><span>Latest accepted Charge LLI (%)</span><strong id="latestLli">N/A</strong></div>
<div class="card"><span>Latest accepted Charge Anode LAM (%)</span><strong id="latestAnode">N/A</strong></div>
<div class="card"><span>Latest accepted Charge Cathode LAM (%)</span><strong id="latestCathode">N/A</strong></div>
</div>
<h3 class="chart-title">Accepted Charge PyDMA trends across RPTs</h3>
<div id="degradationLegend" class="legend"></div>
<div class="chart-wrap"><canvas id="degradationChart" class="chart" aria-label="LLI and LAM estimates by RPT"></canvas></div>
<details><summary>H2 v3 supplemental results and per-RPT fit provenance</summary>
<p class="muted">H2 v3 estimates are supplemental only and are not used in the primary cards or trend above.</p>
<div class="table-wrap"><table><thead><tr><th>RPT</th><th>Accepted Charge PyDMA LLI (%)</th><th>Accepted Charge Anode LAM (%)</th><th>Accepted Charge Cathode LAM (%)</th><th>Charge fit status / reason</th><th>Charge / same-cell BoL provenance</th><th>Supplemental H2 v3 LLI (%)</th><th>Supplemental H2 v3 Anode LAM (%)</th><th>Supplemental H2 v3 Cathode LAM (%)</th><th>H2 fit status / direction</th></tr></thead><tbody id="degradationTable"></tbody></table></div>
</details>
</section>
</main>
<footer>Degradation values are model-derived estimates, not independently verified physical ground truth.</footer>
<script>
const $=id=>document.getElementById(id);
let analysis=null, inspection=null, inspectionToken=0, serviceReady=false;
let selectedIndex=-1, sohPoints=[], degradationPoints=[];
const series=[
 {key:"LLI",chargeKey:"LLI_percent",label:"Charge LLI",color:"#138a8a"},
 {key:"Anode_LAM",chargeKey:"Anode_LAM_percent",label:"Charge Anode LAM",color:"#dc7a12"},
 {key:"Cathode_LAM",chargeKey:"Cathode_LAM_percent",label:"Charge Cathode LAM",color:"#7651b5"}
];
const visible={LLI:true,Anode_LAM:true,Cathode_LAM:true};
function finite(value){return value!==null&&value!==undefined&&value!==""&&Number.isFinite(Number(value));}
function fmt(value){return finite(value)?`${Number(value).toFixed(2)}%`:"N/A";}
function rptX(row,index){const n=Number(row.rpt_index);return Number.isFinite(n)?n:index+1;}
async function checkHealth(){
 try{const response=await fetch("/api/health",{cache:"no-store"});if(!response.ok)throw Error();serviceReady=true;$("status").textContent="Analysis service ready."; $("status").className="status success";}
 catch{serviceReady=false;$("status").textContent="Local analysis service is unavailable. Start the dashboard with the documented PowerShell command, then reload this page.";$("status").className="status error";}
}
async function loadModels(){
 const select=$("modelSelect");
 try{
  const response=await fetch("/api/models",{cache:"no-store"}),payload=await response.json();
  if(!response.ok)throw Error(payload.error||"Model metadata unavailable.");
  select.replaceChildren();
  for(const model of payload.models){
   const option=document.createElement("option");option.value=model.name;
   const metrics=model.validation_metrics;
   const rmse=finite(metrics?.validation_rmse_pp)?` · Val RMSE ${Number(metrics.validation_rmse_pp).toFixed(3)} pp`:"";
   const available=model.checkpoint_status?.available===true;
   const unavailable=available?``:` · unavailable: ${model.checkpoint_status?.reason||"unknown"}`;
   option.disabled=!available;option.title=available?"":String(model.checkpoint_status?.reason||"Checkpoint unavailable");
   option.textContent=`${model.name}${rmse}${unavailable}`;select.appendChild(option);
  }
  const availableModels=payload.models.filter(model=>model.checkpoint_status?.available===true);
  select.disabled=!availableModels.length;
  const selected=payload.models.find(model=>model.name===payload.default_model&&model.checkpoint_status?.available===true)||availableModels[0];
  if(selected){select.value=selected.name;renderModelInfo(selected,payload.default_model===selected.name);}
  else $("modelInfo").textContent="No frozen Final2 checkpoint is currently available. Each model option shows its failure reason.";
  select.onchange=()=>renderModelInfo(payload.models.find(model=>model.name===select.value),false);
 }catch(error){$("modelInfo").textContent=`Frozen model list unavailable: ${String(error.message||error)}`;}
}
function metricLine(label,metrics,maeKey,rmseKey,r2Key){
 if(!metrics)return `${label}: unavailable in saved artifacts`;
 const value=key=>finite(metrics[key])?Number(metrics[key]).toFixed(3):"N/A";
 return `${label}: MAE ${value(maeKey)} pp · RMSE ${value(rmseKey)} pp · R² ${value(r2Key)}`;
}
function renderModelInfo(model,isDefault){
 if(!model){$("modelInfo").textContent="Selected frozen model metadata is unavailable.";return;}
 const checkpoint=model.checkpoint_status||{};
 const validation=model.validation_metrics;
 const lines=[
  `${isDefault?"Validation-selected default":"Selected model"}: ${model.name}`,
  `Architecture: ${model.architecture||"not recorded"}`,
  `Required modalities: ${(model.modalities||[]).join(", ")||"not recorded"}`,
  `Checkpoint ID: ${model.checkpoint_id||"not recorded"}`,
  `Checkpoint path: ${model.checkpoint_path||checkpoint.reason||"not available"}`,
  `Dataset version: ${model.dataset_identifier||"not recorded"}`,
  metricLine("Validation",validation,"validation_mae_pp","validation_rmse_pp","validation_r2"),
  metricLine("External (display only; not used for selection)",model.external_metrics,"external_mae_pp","external_rmse_pp","external_r2"),
  `Checkpoint status: ${checkpoint.available?"available":checkpoint.reason||"unavailable"}`,
  "Switching models changes SOH inference only; reference and Charge PyDMA degradation remain unchanged."
 ];
 $("modelInfo").textContent=lines.join("\n");
}
function clearAnalysis(){
 analysis=null;inspection=null;selectedIndex=-1;$("cellSummary").hidden=true;
 updateSummaries();drawCharts();
}
$("tpgFile").addEventListener("change",async()=>{
 const token=++inspectionToken,file=$("tpgFile").files[0];$("analyze").disabled=true;clearAnalysis();
 $("mappingNotice").hidden=true;
 if(!file)return;
 if(!/\.(mat|csv)$/i.test(file.name)){$("status").textContent="Select a raw TPG .mat or .csv file.";$("status").className="status error";return;}
 if(!serviceReady){$("status").textContent="Local analysis service is unavailable. Start the dashboard with the documented PowerShell command, then reload this page.";$("status").className="status error";return;}
 $("status").textContent="Checking the file against Guide mappings…";$("status").className="status";
 try{
  const response=await fetch("/api/inspect",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filename:file.name}),signal:AbortSignal.timeout(120000)});
  const payload=await response.json();if(!response.ok)throw Error(payload.error||"Could not inspect this TPG file.");
  if(token!==inspectionToken)return;
  inspection=payload;$("cellSummary").hidden=false;$("cellId").textContent=payload.cell_id;
  $("analyze").disabled=false;
  if(payload.inventory_scope==="VERIFIED_DEMO_ONLY"){
   $("mappingNotice").hidden=false;
   $("mappingNotice").textContent=`Guide-verified End-of-Life observation; demo-only and excluded from frozen evaluation. Step 10 is unavailable under the existing contiguous-segment extractor. ${payload.mapping_provenance||""} Source SHA-256: ${payload.source_sha256||"not available"}.`;
   $("mappingNotice").className="diagnostic";
   $("status").textContent=`Detected ${payload.detected_rpt_count} Guide-verified demo-only End-of-Life RPT checkpoint.`;$("status").className="status success";
  }else if(payload.mapping_resolution_status==="owner_confirmed_frozen_final2_exception"){
   $("mappingNotice").hidden=false;
   $("mappingNotice").textContent=`Owner-confirmed frozen Final2 mapping; not independently Guide-mapped. The raw inventory remains ambiguous. ${payload.mapping_provenance||""} Source SHA-256: ${payload.source_sha256||"not available"}.`;
   $("mappingNotice").className="diagnostic";
   $("status").textContent=`Detected ${payload.detected_rpt_count} owner-confirmed RPT checkpoint.`;$("status").className="status success";
  }else{
   $("mappingNotice").hidden=true;
   $("status").textContent=`Detected ${payload.detected_rpt_count} Guide-mapped RPT checkpoint(s).`;$("status").className="status success";
  }
 }catch(error){
  if(token!==inspectionToken)return;
  const message=error instanceof TypeError?"Could not reach the local analysis service. Check that the dashboard is running and retry.":String(error.message||error);
  $("status").textContent=message;$("status").className="status error";
 }
});
function latestChargeFor(metric){
 if(!analysis)return null;
 for(let i=analysis.rpts.length-1;i>=0;i--){const entry=analysis.rpts[i].charge_degradation?.[metric];if(entry!==null&&entry!==undefined&&finite(entry))return {value:entry,rpt:analysis.rpts[i]};}
 return null;
}
function updateChargeFitMessage(){
 const target=$("chargeFitMessage"),rows=analysis?.rpts||[];
 if(!rows.length||selectedIndex<0||selectedIndex>=rows.length){target.hidden=true;return;}
 const row=rows[selectedIndex],charge=row.charge_degradation||{};
 const valid=["LLI_percent","Anode_LAM_percent","Cathode_LAM_percent"].every(key=>finite(charge[key]));
 target.hidden=false;
 if(!valid){
  if(charge.status==="charge_artifact_parse_failed"){
   target.textContent=`Charge PyDMA artifact parse/schema failure: ${charge.reason||"details unavailable"}; ${JSON.stringify(charge.artifact_errors||[])}`;
  }else if(charge.status==="charge_artifact_source_mismatch"){
   target.textContent=`Charge PyDMA source mismatch: ${charge.reason||"source validation failed"}; ${JSON.stringify(charge.source_errors||[])}`;
  }else if(charge.status==="charge_artifact_missing"){
   target.textContent=`Charge PyDMA saved artifact is missing: ${charge.reason||"expected artifact unavailable"}.`;
  }else{
   target.textContent="Chưa có kết quả PyDMA Charge hợp lệ cho RPT này.";
  }
  target.className="diagnostic error";return;
 }
 const warnings=[...(charge.artifact_warnings||[])].map(item=>item.reason||"saved artifact warning");
 target.textContent=`${row.rpt}: accepted Charge PyDMA fit · RMSE ${finite(charge.fit_rmse_mV)?Number(charge.fit_rmse_mV).toFixed(3):"N/A"} mV · accepted runs ${finite(charge.accepted_runs)?charge.accepted_runs:"N/A"} · ${charge.source_artifact||"saved artifact"}${warnings.length?` · Artifact warning: ${warnings.join("; ")}`:""}.`;
 target.className="diagnostic ok";
}
function updateSummaries(){
 const rows=analysis?.rpts||[], valid=rows.filter(row=>finite(row.predicted_soh_percent));
 $("cellSummary").hidden=!analysis&&!inspection;
 $("cellId").textContent=analysis?.cell_id||inspection?.cell_id||"—";
 $("activeModel").textContent=analysis?.selected_model||"N/A";
 const latest=valid.length?valid[valid.length-1]:null;
 $("latestSoh").textContent=latest?fmt(latest.predicted_soh_percent):"N/A";
 const latestReference=[...rows].reverse().find(row=>finite(row.reference_soh?.value_percent));
 $("latestReference").textContent=latestReference?fmt(latestReference.reference_soh.value_percent):"N/A";
 if(selectedIndex<0||selectedIndex>=rows.length)selectedIndex=latest?rows.indexOf(latest):-1;
 $("latestLli").textContent=fmt(latestChargeFor("LLI_percent")?.value);
 $("latestAnode").textContent=fmt(latestChargeFor("Anode_LAM_percent")?.value);
 $("latestCathode").textContent=fmt(latestChargeFor("Cathode_LAM_percent")?.value);
 updateChargeFitMessage();
}
function reportModelSelectionChange(){
 if(analysis&&$("modelSelect").value!==analysis.selected_model){
  $("status").textContent=`Current results use ${analysis.selected_model}. Click Analyze Battery to run ${$("modelSelect").value}; the reference and saved degradation joins remain unchanged.`;
  $("status").className="status";
 }
}
function drawLinePlot(canvas,plots,xLabel,yLabel){
 const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1,width=Math.max(rect.width,320),height=rect.height||320;
 canvas.width=width*dpr;canvas.height=height*dpr;const ctx=canvas.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
 const left=62,right=18,top=18,bottom=48,plotW=width-left-right,plotH=height-top-bottom;
 const points=plots.flatMap(plot=>plot.points.filter(p=>finite(p.x)&&finite(p.y)));
 const xs=points.map(p=>Number(p.x)),ys=points.map(p=>Number(p.y));
 let xmin=xs.length?Math.min(...xs):0,xmax=xs.length?Math.max(...xs):1,ymin=ys.length?Math.min(...ys):0,ymax=ys.length?Math.max(...ys):1;
 if(xmin===xmax){xmin-=.5;xmax+=.5;}if(ymin===ymax){ymin-=1;ymax+=1;}const pad=(ymax-ymin)*.08;ymin-=pad;ymax+=pad;
 const x=v=>left+(v-xmin)/(xmax-xmin)*plotW,y=v=>top+(ymax-v)/(ymax-ymin)*plotH;
 ctx.font="12px Segoe UI,Arial";ctx.strokeStyle="#dbe4e9";ctx.fillStyle="#607482";ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const value=ymin+(ymax-ymin)*i/4,py=y(value);ctx.beginPath();ctx.moveTo(left,py);ctx.lineTo(width-right,py);ctx.stroke();ctx.fillText(value.toFixed(2),5,py+4);}
 ctx.strokeStyle="#617684";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,height-bottom);ctx.lineTo(width-right,height-bottom);ctx.stroke();
 for(let i=0;i<=Math.min(5,Math.max(1,Math.floor(xmax-xmin)));i++){const value=xmin+(xmax-xmin)*i/Math.min(5,Math.max(1,Math.floor(xmax-xmin)));ctx.fillStyle="#607482";ctx.fillText(Number.isInteger(value)?String(value):value.toFixed(0),x(value)-5,height-bottom+19);}
 ctx.fillStyle="#344e5e";ctx.fillText(xLabel,width/2-35,height-8);ctx.save();ctx.translate(14,height/2+20);ctx.rotate(-Math.PI/2);ctx.fillText(yLabel,0,0);ctx.restore();
 for(const plot of plots){const valid=plot.points.filter(p=>finite(p.x)&&finite(p.y));ctx.strokeStyle=plot.color;ctx.fillStyle=plot.color;ctx.lineWidth=2;
  for(let i=0;i<valid.length;i++){const point=valid[i],px=x(Number(point.x)),py=y(Number(point.y));
   if(plot.connect!==false&&i>0&&(!plot.requireAdjacent||Number(point.x)-Number(valid[i-1].x)===1)){ctx.beginPath();ctx.moveTo(x(Number(valid[i-1].x)),y(Number(valid[i-1].y)));ctx.lineTo(px,py);ctx.stroke();}
   ctx.beginPath();ctx.arc(px,py,3.8,0,Math.PI*2);ctx.fill();
  }
 }
 return {ctx,x,y,left,right,top,bottom,width,height};
}
function rowsForMetric(field){return (analysis?.rpts||[]).map((row,index)=>({x:Number(row.rpt_index)||index+1,y:finite(row[field])?Number(row[field]):null,row}));}
function drawSoh(){
 const rows=analysis?.rpts||[];
 drawLinePlot($("sohChart"),[
  {name:"Reference SoH",color:"#dc7a12",points:rows.map((row,i)=>({x:Number(row.rpt_index)||i+1,y:finite(row.reference_soh?.value_percent)?Number(row.reference_soh.value_percent):null})),requireAdjacent:true},
  {name:"Predicted SoH",color:"#15759a",points:rows.map((row,i)=>({x:Number(row.rpt_index)||i+1,y:finite(row.predicted_soh_percent)?Number(row.predicted_soh_percent):null})),requireAdjacent:true}
 ],"RPT","SoH (%)");
 renderSohTable();
 drawEvaluationPlots();
}
function drawEvaluationPlots(){
 const rows=analysis?.rpts||[],matched=rows.filter(row=>finite(row.reference_soh?.value_percent)&&finite(row.predicted_soh_percent));
 drawLinePlot($("errorChart"),[{color:"#b63b47",points:rows.map((row,i)=>({x:Number(row.rpt_index)||i+1,y:finite(row.signed_error_percentage_points)?Number(row.signed_error_percentage_points):null})),requireAdjacent:true}],"RPT","Signed error (pp)");
 const scatter=$("scatterChart"),rect=scatter.getBoundingClientRect(),dpr=window.devicePixelRatio||1,width=Math.max(rect.width,320),height=rect.height||320;
 scatter.width=width*dpr;scatter.height=height*dpr;const ctx=scatter.getContext("2d");ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
 const left=62,right=18,top=18,bottom=48,plotW=width-left-right,plotH=height-top-bottom;
 if(!matched.length){ctx.fillStyle="#607482";ctx.font="14px Segoe UI,Arial";ctx.fillText("No finite matched prediction/reference pairs.",left,top+24);return;}
 const refs=matched.map(row=>Number(row.reference_soh.value_percent)),preds=matched.map(row=>Number(row.predicted_soh_percent));
 const min=Math.min(...refs,...preds),max=Math.max(...refs,...preds),span=max-min||1,low=min-span*.08,high=max+span*.08;
 const x=v=>left+(v-low)/(high-low)*plotW,y=v=>top+(high-v)/(high-low)*plotH;
 ctx.font="12px Segoe UI,Arial";ctx.strokeStyle="#dbe4e9";ctx.fillStyle="#607482";
 for(let i=0;i<=4;i++){const value=low+(high-low)*i/4,pos=x(value);ctx.beginPath();ctx.moveTo(pos,top);ctx.lineTo(pos,height-bottom);ctx.stroke();ctx.fillText(value.toFixed(1),pos-12,height-bottom+19);ctx.beginPath();ctx.moveTo(left,y(value));ctx.lineTo(width-right,y(value));ctx.stroke();ctx.fillText(value.toFixed(1),5,y(value)+4);}
 ctx.strokeStyle="#617684";ctx.beginPath();ctx.moveTo(left,top);ctx.lineTo(left,height-bottom);ctx.lineTo(width-right,height-bottom);ctx.stroke();
 ctx.fillStyle="#344e5e";ctx.fillText("Reference SoH (%)",width/2-48,height-8);ctx.save();ctx.translate(14,height/2+20);ctx.rotate(-Math.PI/2);ctx.fillText("Predicted SoH (%)",0,0);ctx.restore();
 ctx.save();ctx.strokeStyle="#647987";ctx.setLineDash([6,4]);ctx.beginPath();ctx.moveTo(x(low),y(low));ctx.lineTo(x(high),y(high));ctx.stroke();ctx.restore();
 ctx.fillStyle="#15759a";for(const row of matched){ctx.beginPath();ctx.arc(x(Number(row.reference_soh.value_percent)),y(Number(row.predicted_soh_percent)),4.5,0,Math.PI*2);ctx.fill();}
}
function cell(row,text,className){
 const td=document.createElement("td");td.textContent=text==null?"":String(text);if(className)td.className=className;row.appendChild(td);return td;
}
function renderSohTable(){
 const body=$("sohTable");body.replaceChildren();
 for(const row of analysis?.rpts||[]){const tr=document.createElement("tr");cell(tr,row.rpt);
  cell(tr,fmt(row.reference_soh?.value_percent));cell(tr,fmt(row.predicted_soh_percent));
  cell(tr,finite(row.signed_error_percentage_points)?Number(row.signed_error_percentage_points).toFixed(3):"N/A");
  cell(tr,finite(row.absolute_error_percentage_points)?Number(row.absolute_error_percentage_points).toFixed(3):"N/A");
  const reason=row.model_prediction?.reason||row.error_status||row.reference_soh?.status||row.soh_status;cell(tr,reason,finite(row.signed_error_percentage_points)?"available":"reason");body.appendChild(tr);
 }
}
function renderRptProvenance(){
 const target=$("rptProvenance"),row=analysis?.rpts?.[selectedIndex];
 if(!row){target.hidden=true;return;}
 const modalities=row.modality_eligibility||{};
 const hasDischarge=modalities.step4_voltage===true;
 const hasCharge=modalities.step10_voltage===true;
 const hasPhysics=modalities.p5_physics===true;
 const eol=row.is_eol_procedure===true||String(row.guide_stage||"").toLowerCase().includes("eol");
 const lines=[
  `Part ${row.part||"?"} · Group ${row.group_id||row.group||"?"} · Cell ${row.cell_id||row.physical_cell_id||"?"} · ${row.rpt||`RPT${row.rpt_index||"?"}`}`,
  `Procedure: ${row.procedure||row.source_tpg||"not recorded"}${eol?" · End-of-Life":""}`,
  `Inventory: ${row.inventory_scope||"not recorded"} · Mapping: ${row.mapping_status||row.mapping_resolution_status||"not recorded"}`,
  `Mapping evidence: ${row.mapping_provenance||row.mapping_evidence||"not recorded"}`,
  `Modalities — discharge: ${hasDischarge?"eligible":"unavailable"}; 2M: ${hasDischarge&&hasCharge?"eligible":"unavailable"}; 3M: ${hasDischarge&&hasCharge&&hasPhysics?"eligible":"unavailable"}`,
  `Step 10 review: ${row.demo_step10_review_reason||"none recorded"}`,
  `Source: ${row.source_file||"not recorded"}`,
  `Source SHA-256: ${row.source_sha256||"not recorded"}`,
  `Reference: ${row.reference_soh?.status||"unavailable"}${finite(row.reference_soh?.value_percent)?` · ${fmt(row.reference_soh.value_percent)}`:""}`,
  `Label basis: ${row.reference_soh?.label_method||"not recorded"}${row.reference_soh?.bol_source_file?` · BoL: ${row.reference_soh.bol_source_file}`:""}`
 ];
 target.textContent=lines.join("\n");target.hidden=false;
}
function renderMetrics(){
 const metrics=analysis?.evaluation_metrics||{};
 for(const [key,valueId,reasonId] of [["mae_percentage_points","maeMetric","maeReason"],["rmse_percentage_points","rmseMetric","rmseReason"],["r2","r2Metric","r2Reason"]]){
  const metric=metrics[key];$(valueId).textContent=finite(metric?.value)?Number(metric.value).toFixed(4):"N/A";$(reasonId).textContent=metric?.reason||"";
 }
 $("nMetric").textContent=metrics.n??0;
}
function showInputStatus(label,status){
 const item=document.createElement("div");item.className=`diagnostic ${status.available?"ok":"error"}`;
 item.textContent=`${label}: ${status.available?"available ("+status.status+")":"unavailable — "+status.reason}`;$("inputStatuses").appendChild(item);
}
function drawSelectedInputs(){
 const row=analysis?.rpts?.[selectedIndex];if(!row)return;
 const inputs=row.inputs||{},step4=inputs.step4||{},step10=inputs.step10||{};
 $("inputStatuses").replaceChildren();
 showInputStatus("Step 4 discharge",step4);
 const temperature={available:!!step4.temperature_available,status:step4.temperature_status||"",reason:step4.temperature_available?"":(step4.temperature_status||step4.reason||"Step 4 temperature unavailable")};
 showInputStatus("Step 4 temperature",temperature);showInputStatus("Step 10 charge",step10);
 if(inputs.step8)showInputStatus("Optional Step 8 discharge",inputs.step8);
 const step4Raw=step4.raw_voltage||[],step10Raw=step10.raw_voltage||[];
 const sampled=(values)=>{const stride=Math.max(1,Math.ceil(values.length/1500));return values.filter((_,i)=>i%stride===0||i===values.length-1);};
 drawLinePlot($("step4Chart"),[
  {color:"#15759a",points:sampled(step4Raw).map((v,i)=>({x:Math.min(i*Math.max(1,Math.ceil(step4Raw.length/1500)),step4Raw.length-1),y:v})),connect:true},
  {color:"#dc7a12",points:(step4.processed_128_voltage||[]).map((v,i)=>({x:i*Math.max(0,step4Raw.length-1)/127,y:v})),connect:true}
 ],"Sample index","Voltage (V)");
 drawLinePlot($("step10Chart"),[
  {color:"#15759a",points:sampled(step10Raw).map((v,i)=>({x:Math.min(i*Math.max(1,Math.ceil(step10Raw.length/1500)),step10Raw.length-1),y:v})),connect:true},
  {color:"#dc7a12",points:(step10.processed_128_voltage||[]).map((v,i)=>({x:i*Math.max(0,step10Raw.length-1)/127,y:v})),connect:true}
 ],"Sample index","Voltage (V)");
 const normalized=inputs.model_normalized||{};
 drawLinePlot($("step4ModelChart"),[{color:"#7651b5",points:(normalized.step4_voltage_normalized||[]).map((v,i)=>({x:i,y:v}))}],"Progress index","Normalized input");
 drawLinePlot($("step10ModelChart"),[{color:"#138a8a",points:(normalized.step10_voltage_normalized||[]).map((v,i)=>({x:i,y:v}))}],"Progress index","Normalized input");
 const featureBody=$("physicsTable");featureBody.replaceChildren();
 for(const [name,item] of Object.entries(inputs.p5_physics_features||{})){const tr=document.createElement("tr");cell(tr,name);cell(tr,finite(item.value)?Number(item.value).toPrecision(6):"N/A");cell(tr,finite(item.normalized_value)?Number(item.normalized_value).toPrecision(6):"N/A");cell(tr,item.status,finite(item.value)?"available":"reason");featureBody.appendChild(tr);}
}
function makeLegend(){
 const root=$("degradationLegend");root.replaceChildren();
 for(const s of series){const button=document.createElement("button");button.type="button";button.setAttribute("aria-pressed",String(visible[s.key]));const dot=document.createElement("i");dot.style.background=s.color;button.append(dot,document.createTextNode(s.label));button.onclick=()=>{visible[s.key]=!visible[s.key];button.setAttribute("aria-pressed",String(visible[s.key]));drawDegradation();};root.append(button);}
}
function drawDegradation(){
 const rows=analysis?.rpts||[];
 const plots=series.filter(s=>visible[s.key]).map(s=>({
  color:s.color,
  requireAdjacent:true,
  points:rows.map((row,index)=>({x:Number(row.rpt_index)||index+1,y:finite(row.charge_degradation?.[s.chargeKey])?Number(row.charge_degradation[s.chargeKey]):null}))
 }));
 drawLinePlot($("degradationChart"),plots,"RPT","Estimated degradation (%)");
 renderDegradationTable();
}
function renderDegradationTable(){
 const body=$("degradationTable");body.replaceChildren();
 for(const row of analysis?.rpts||[]){const tr=document.createElement("tr");cell(tr,row.rpt);
  const charge=row.charge_degradation||{};
  const chargeValues=["LLI_percent","Anode_LAM_percent","Cathode_LAM_percent"].map(key=>finite(charge[key])?Number(charge[key]).toFixed(3):"N/A");
  for(const value of chargeValues)cell(tr,value);
  const chargeAvailable=chargeValues.every(value=>value!=="N/A");
  const chargeFailure=charge.status==="charge_artifact_parse_failed"?`Charge artifact parse/schema failure: ${charge.reason||"details unavailable"}`:
   charge.status==="charge_artifact_source_mismatch"?`Charge source mismatch: ${charge.reason||"source validation failed"}`:
   charge.status==="charge_artifact_missing"?`Charge saved artifact missing: ${charge.reason||"expected artifact unavailable"}`:
   "Chưa có kết quả PyDMA Charge hợp lệ cho RPT này.";
  cell(tr,chargeAvailable?`${charge.status||"accepted Charge PyDMA"}; RMSE ${charge.fit_rmse_mV??"N/A"} mV; n=${charge.accepted_runs??"N/A"}`:chargeFailure,chargeAvailable?"available":"reason");
  const h2Entry=Object.values(row.degradation||{}).find(item=>finite(item.value_percent));
  const bolProvenance=charge.same_cell_bol_note||h2Entry?.same_cell_bol_note||"H2 v3 does not record same-cell BoL source or acceptance; same-cell BoL provenance is not asserted.";
  const bolFit=charge.same_cell_bol_fit_accepted===true||charge.same_cell_bol_fit_accepted===false?`same_cell_bol_fit_accepted=${charge.same_cell_bol_fit_accepted}`:"same_cell_bol_fit_accepted=not recorded";
  const chargeSource=charge.charge_source_file||charge.source_file;
  const chargeQc=`Charge source=${chargeSource||"not recorded"}; QC=${charge.quality_control_status||"not recorded"}; accepted=${charge.fit_accepted_by_pydma_15mV??"not recorded"}; RMSE=${finite(charge.fit_rmse_mV)?Number(charge.fit_rmse_mV).toFixed(3):"N/A"} mV; runs=${charge.accepted_runs??"N/A"}/${charge.total_runs??"N/A"}${finite(charge.fit_threshold_mV)?`; threshold=${Number(charge.fit_threshold_mV).toFixed(1)} mV`:""}`;
  const bolQc=`BoL QC=${charge.same_cell_bol_quality_control_status||"not recorded"}; RMSE=${finite(charge.same_cell_bol_fit_rmse_mV)?Number(charge.same_cell_bol_fit_rmse_mV).toFixed(3):"N/A"} mV; runs=${charge.same_cell_bol_accepted_runs??"N/A"}/${charge.same_cell_bol_total_runs??"N/A"}${finite(charge.same_cell_bol_fit_threshold_mV)?`; threshold=${Number(charge.same_cell_bol_fit_threshold_mV).toFixed(1)} mV`:""}`;
  cell(tr,`${chargeQc}; ${bolProvenance}; ${bolFit}${charge.same_cell_bol_source?`; same_cell_bol_source=${charge.same_cell_bol_source}`:""}; ${bolQc}`,charge.same_cell_bol_fit_accepted===true?"available":"reason");
  const h2Reasons=[];
  for(const key of ["LLI","Anode_LAM","Cathode_LAM"]){const item=row.degradation?.[key];cell(tr,fmt(item?.value_percent));if(!finite(item?.value_percent))h2Reasons.push(`${key}: ${item?.status||"no accepted fit"}`);}
  const h2Provenance=Object.values(row.degradation||{}).filter(item=>finite(item.value_percent)).map(item=>`${item.source_direction||"direction unavailable"}; RMSE ${item.fit_rmse_mV??"N/A"} mV`).join(" | ");
  cell(tr,h2Reasons.length?h2Reasons.join(" | "):h2Provenance,h2Reasons.length?"reason":"available");
  body.appendChild(tr);
 }
}
function populateRpts(){
 const select=$("rptSelect"),rows=analysis?.rpts||[];select.replaceChildren();
 rows.forEach((row,index)=>{const option=document.createElement("option");option.value=String(index);const eol=row.is_eol_procedure===true||String(row.guide_stage||"").toLowerCase().includes("eol");option.textContent=`${row.rpt}${eol?" · EoL":""}${row.inventory_scope?` · ${row.inventory_scope}`:""}`;select.appendChild(option);});
 select.disabled=!rows.length;selectedIndex=rows.length?0:-1;
 select.onchange=()=>{selectedIndex=Number(select.value);updateChargeFitMessage();drawSelectedInputs();renderRptProvenance();};
}
function drawCharts(){drawSoh();drawDegradation();drawSelectedInputs();renderRptProvenance();renderMetrics();}
async function analyze(){
 const file=$("tpgFile").files[0];if(!file||!inspection)return;
 $("analyze").disabled=true;$("tpgFile").disabled=true;$("status").className="status";$("status").innerHTML='<span class="loading"><i class="spinner"></i>Validating raw data and processing verified RPTs…</span>';
 try{
  const modelName=$("modelSelect").value;
  const response=await fetch("/api/analyze",{method:"POST",headers:{"X-Upload-Filename":file.name,"X-Model-Name":modelName},body:file,signal:AbortSignal.timeout(1800000)});
  const payload=await response.json();if(!response.ok)throw Error(payload.error||"Battery analysis failed.");
  analysis=payload;selectedIndex=-1;populateRpts();updateSummaries();makeLegend();drawCharts();
  const mappingLabel=payload.provenance?.mapping_resolution_status==="owner_confirmed_frozen_final2_exception"?" Owner-confirmed mapping; not independently Guide-mapped.":"";
  $("status").textContent=`Analysis complete with ${payload.selected_model}.${mappingLabel} ${payload.processed_rpt_count} of ${payload.detected_rpt_count} mapped RPTs produced SoH predictions. Missing inputs, checkpoint, or eligibility are shown per RPT.`;$("status").className="status success";
 }catch(error){
  const message=error.name==="TimeoutError"?"Analysis timed out. Check the MATLAB conversion runtime and retry.":error instanceof TypeError?"Could not reach the local analysis service. Check that the dashboard is running and retry.":String(error.message||error);
  $("status").textContent=message;$("status").className="status error";analysis=null;updateSummaries();drawCharts();
 }finally{$("tpgFile").disabled=false;$("analyze").disabled=!inspection;}
}
$("analyze").onclick=analyze;
$("modelSelect").onchange=reportModelSelectionChange;
window.addEventListener("resize",()=>{if(analysis)drawCharts();});
makeLegend();checkHealth();
loadModels();
</script>
</body>
</html>"""
