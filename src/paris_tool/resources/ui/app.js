'use strict';
const $ = id => document.getElementById(id);
const token = document.querySelector('meta[name="paris-token"]').content;
const S = {view:'input', resultRequest:0, inspection:null, run:null, runs:[], active:null, timer:null, design:null, offset:0, total:0, query:'', pickerKind:'dataset', pickerPath:null, pickerParent:null, canSelect:false};
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num = n => Number(n).toLocaleString('zh-CN');
const duration = s => {s=Math.max(0,Math.floor(s));return s>=3600?`${Math.floor(s/3600)}:${String(Math.floor(s/60)%60).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`:`${String(Math.floor(s/60)).padStart(2,'0')}:${String(s%60).padStart(2,'0')}`;};
async function api(path, body) {
  const res=await fetch(path,{method:body===undefined?'GET':'POST',headers:{'X-PARIS-Token':token,...(body===undefined?{}:{'Content-Type':'application/json'})},...(body===undefined?{}:{body:JSON.stringify(body)})});
  const data=await res.json();
  if(!res.ok) throw new Error(res.status===403?'会话已更新，请刷新页面。':data.error||'请求失败');
  return data;
}
function notice(message) { $('global-notice').textContent=message;$('global-notice').classList.remove('hidden');clearTimeout(S.noticeTimer);S.noticeTimer=setTimeout(()=>$('global-notice').classList.add('hidden'),4500); }
function showError(id,message) { $(id).textContent=message;$(id).classList.toggle('hidden',!message); }
function showView(view) {
  S.view=view;window.scrollTo(0,0);document.querySelectorAll('.view').forEach(el=>el.classList.toggle('hidden',el.id!=='view-'+view));
  document.querySelectorAll('.nav-step').forEach(el=>el.classList.toggle('active',el.dataset.view===view));
  if(view==='benchmark')loadBenchmark().catch(e=>showError('benchmark-error',e.message));
  if(view==='iterate'&&S.iterConfig)refreshIterationConfig().catch(e=>notice(e.message));
  if(view==='output' && S.run?.status==='complete') loadResults().catch(e=>notice(e.message));
  updateNavigation();
}
function updateNavigation() {
  document.querySelectorAll('[data-view="process"]').forEach(el=>{el.disabled=!S.run;});
  document.querySelectorAll('[data-view="output"]').forEach(el=>{el.disabled=S.run?.status!=='complete';});
  $('nav-input-dot').className='nav-dot'+((S.view==='input'?(S.inputMode==='raw'?S.projectInfo?.ready:S.inspection):S.run?.inspection)?' done':'');
  $('nav-process-dot').className='nav-dot'+(S.run?.status==='complete'?' done':S.run?.status==='running'||S.run?.status==='queued'?' live':'');
  $('nav-output-dot').className='nav-dot'+(S.run?.status==='complete'?' done':'');
  $('start-button').disabled=!S.inspection||!!S.active||!!S.checking||!!S.prepActive||!!S.iterActive;
  if(typeof rawControls==='function' && $('prepare-project'))rawControls();
  if(typeof iterationControls==='function' && $('start-iteration'))iterationControls();
}
function invalidate() {
  S.inspection=null;modelFlow(null);$('feature-catalog').classList.add('hidden');$('input-status').className='status-chip neutral';$('input-status').textContent='待检查';
  $('input-summary').classList.add('hidden');$('model-detail').classList.add('hidden');showError('input-error','');
  document.querySelectorAll('.file-state').forEach(el=>{el.className='file-state';el.textContent='待检查';});updateNavigation();
}
function modelFlow(algorithm) {
  const single=!algorithm||algorithm==='paris.single-xgb.1';
  const badge=single?'PARIS':'PARIS · 历史版本';
  const stages=single?['CTS / GR 特征','PARIS推理','TNS 预测与排序']:['历史模型','候选排序'];
  document.querySelector('.model-badge').textContent=badge;
  document.querySelector('.result-complete small').textContent=badge;
  document.querySelector('.pipeline-preview > div').innerHTML=stages.map(x=>`<span>${esc(x)}</span>`).join('<i>→</i>');
  document.querySelector('.process-algorithm > div').innerHTML=stages.map(esc).join('<span>→</span>');
}

function renderInspection(data) {
  S.inspection=data;modelFlow(data.algorithm);$('input-status').className='status-chip complete';$('input-status').textContent='检查通过';
  $('model-detail').innerHTML=`<strong>✓ ${esc(data.algorithm==='paris.single-xgb.1'?'PARIS':data.model_name)}</strong><span>模型已加载</span><span>固定权重</span>`;
  $('model-detail').classList.remove('hidden');
  document.querySelectorAll('.file-state').forEach((el,i)=>{const f=data.files[i];el.className='file-state valid';el.textContent=`✓ ${num(f.rows)} 行 · ${num(f.columns)} 列`;});
  $('input-summary').textContent=`✓ ${num(data.candidates)} 个候选 · ${data.designs.length} 个设计，模型与 GR 输入格式检查通过。`;
  $('input-summary').classList.remove('hidden');updateNavigation();
}
async function inspect(event) {
  event?.preventDefault();if(S.inputMode==='raw')return inspectProject();showError('input-error','');
  const model=$('model-path').value.trim(),dataset=$('dataset-path').value.trim();
  if(!model||!dataset){showError('input-error','请选择模型和候选数据目录。');return;}
  S.checking=true;$('inspect-button').disabled=true;$('inspect-button').textContent='正在检查…';$('start-button').disabled=true;
  document.querySelectorAll('#model-path,#dataset-path,[data-browse],[data-input-mode]').forEach(el=>{el.disabled=true;});
  try{renderInspection(await api('/api/inspect',{model,dataset}));if(S.inputMode==='features')renderCatalog(await api('/api/features?'+new URLSearchParams({dataset})));  }
  catch(e){S.inspection=null;$('input-status').className='status-chip failed';$('input-status').textContent='检查未通过';showError('input-error',e.message);}
  finally{S.checking=false;$('inspect-button').disabled=false;$('inspect-button').textContent='检查输入';document.querySelectorAll('#model-path,#dataset-path,[data-browse],[data-input-mode]').forEach(el=>{el.disabled=false;});updateNavigation();}
}
function rememberRun(run) {
  S.runs=[run,...S.runs.filter(r=>r.id!==run.id)].sort((a,b)=>b.created.localeCompare(a.created));renderRecent();
}
function renderRecent() {
  $('recent-count').textContent=S.runs.length||'';
  if(!S.runs.length){$('recent-runs').innerHTML='<p class="side-empty">完成一次预测后，记录会保留在这里。</p>';return;}
  $('recent-runs').innerHTML=S.runs.slice(0,12).map(r=>{const date=new Date(r.created);const title=date.toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false});const status={complete:'已完成',running:'运行中',queued:'准备中',failed:'已停止',interrupted:'会话已结束'}[r.status]||r.status;return `<button class="recent-item" data-run-id="${esc(r.id)}"><span class="recent-indicator ${esc(r.status)}"></span><span>${esc(title)}<small>${r.candidates||r.inspection?.candidates?num(r.candidates||r.inspection.candidates)+' 个候选 · ':''}${status}</small></span></button>`;}).join('');
}
const STEP_INFO={validate:['检查输入','校验模型签名、特征格式与候选集合'],predict:['PARIS推理','计算候选质量评分'],export:['生成完整排序','核对结果，保留原始分数与并列名次']};
function renderRun(run) {
  S.run=run;modelFlow(run.inspection?.algorithm);rememberRun(run);updateNavigation();
  $('run-id').textContent=run.id;$('run-candidates').textContent=run.inspection?num(run.inspection.candidates):'检查中';
  const labels={complete:'已完成',running:'运行中',queued:'准备中',failed:'已停止',interrupted:'已中断'};
  $('run-status').textContent=labels[run.status]||run.status;
  $('run-status').className='status-chip '+(run.status==='complete'?'complete':run.status==='failed'||run.status==='interrupted'?'failed':'running');
  $('process-title').textContent=run.status==='complete'?'候选排序已生成':run.status==='failed'||run.status==='interrupted'?'本次运行已停止':'正在生成候选排序';
  $('process-subtitle').textContent=run.status==='complete'?'完整排序、原始分数与运行记录已保存。':'使用所选模型生成候选排序。';
  $('process-steps').innerHTML=Object.entries(STEP_INFO).map(([step,[title,desc]],i)=>{const state=run.steps[step];const icon=state==='complete'?'✓':state==='running'?'':state==='failed'?'!':String(i+1);const tag={complete:'已完成',running:'进行中',failed:'失败',pending:'等待'}[state]||state;const event=run.events.filter(e=>e.step===step).at(-1);return `<div class="process-step ${state}"><span class="process-icon">${icon}</span><div><h3>${title}</h3><p>${esc(event?.message||desc)}</p></div><span class="step-status">${tag}</span></div>`;}).join('');
  showError('run-error',run.error||'');$('process-finished').classList.toggle('hidden',run.status!=='complete');
  if(run.status==='complete') {
    $('result-total').textContent=num(run.candidates);$('result-design-count').textContent=run.designs.length;
    $('result-elapsed').textContent=duration(run.finished_at-run.started_at);$('result-path').textContent=run.output;
    if(!run.designs.some(d=>d.name===S.design)) S.design=run.designs[0].name;
    renderDesignTabs();
  }
  tick();
}
function tick() {
  if(!S.run)return;
  $('run-elapsed').textContent=duration((S.run.finished_at||Date.now()/1000)-S.run.started_at);
}
async function watch(rid) {
  if(S.active!==rid)return;
  clearTimeout(S.timer);
  try {
    const run=await api('/api/runs/'+rid);
    if(['complete','failed','interrupted'].includes(run.status)&&S.active===rid)S.active=null;
    const wasRunning=S.run?.id===rid && ['running','queued'].includes(S.run.status);
    if(S.run?.id===rid) {
      renderRun(run);
      if($('log-panel').open)refreshLog();
      if(run.status==='complete'&&wasRunning&&S.view==='process'){showView('output');notice('完整候选排序已生成。');}
    } else {rememberRun(run);updateNavigation();}
    if(['running','queued'].includes(run.status)&&S.active===rid)S.timer=setTimeout(()=>watch(rid),1600);
  } catch(e) {notice('暂时无法连接计算服务，正在重试。');S.timer=setTimeout(()=>watch(rid),4000);}
}
async function start() {
  if(!S.inspection||S.active||S.checking||S.prepActive||S.iterActive)return;
  $('start-button').disabled=true;showError('input-error','');
  try {
    const run=await api('/api/runs',{model:$('model-path').value.trim(),dataset:$('dataset-path').value.trim()});
    S.active=run.id;S.design=null;S.offset=0;S.query='';$('candidate-search').value='';
    renderRun(run);showView('process');watch(run.id);
  } catch(e){showError('input-error',e.message);updateNavigation();}
}
function renderDesignTabs() {
  $('design-tabs').innerHTML=S.run.designs.map(d=>`<button class="design-tab${d.name===S.design?' active':''}" data-design="${esc(d.name)}" aria-pressed="${d.name===S.design}">${esc(d.name)}<span>${num(d.n)}</span></button>`).join('');
  const d=S.run.designs.find(d=>d.name===S.design);
  $('ranking-title').textContent=S.design;
  $('ranking-description').textContent=`共 ${num(d.n)} 个候选 · 按模型分数升序${d.top_ties>1?' · 第1名有 '+d.top_ties+' 个并列候选':''}`;
}
async function loadResults() {
  if(S.run?.status!=='complete')return;
  const request=++S.resultRequest;const rid=S.run.id,design=S.design;
  const q=new URLSearchParams({design:S.design,q:S.query,offset:String(S.offset),limit:'25'});
  const r=await api(`/api/runs/${rid}/results?${q}`);
  if(request!==S.resultRequest||S.run?.id!==rid||S.design!==design)return;
  S.total=r.total;
  $('ranking-body').innerHTML=r.rows.length?r.rows.map(x=>{
    const score=Number(x.rank_score).toFixed(6);const change=x.rank_change>0?`↑ ${x.rank_change}`:x.rank_change<0?`↓ ${-x.rank_change}`:'—';
    return `<tr class="${x.predicted_order===1?'best-row':''}"><td><span class="rank-badge ${x.predicted_order===1?'best':''}">${x.predicted_order}</span>${x.tie_count>1?'<small class="tie-label">并列</small>':''}</td><td class="candidate-name" title="${esc(x.candidate_id)}">${esc(x.candidate_id)}</td><td title="原始分数：${esc(x.score_exact)}"><div class="score-cell"><span class="score-number">${score}</span><span class="score-track"><span data-width="${Math.max(0,Math.min(100,x.rank_score*100))}"></span></span></div></td><td class="number-cell">${x.predicted_tns===undefined?'—':Number(x.predicted_tns).toLocaleString('zh-CN',{maximumFractionDigits:2,minimumFractionDigits:2})}</td><td class="number-cell" title="${esc(x.gr_tns)} ns">${Number(x.gr_tns).toLocaleString('zh-CN',{maximumFractionDigits:2,minimumFractionDigits:2})}</td><td class="change-cell ${x.rank_change>0?'up':x.rank_change<0?'down':''}" title="GR 第${x.gr_order}位 → PARIS 第${x.predicted_order}位">${change}</td></tr>`;
  }).join(''):'<tr><td colspan="6" class="empty-table">没有匹配的候选 ID</td></tr>';
  document.querySelectorAll('.score-track [data-width]').forEach(el=>{el.style.width=el.dataset.width+'%';});
  $('table-count').textContent=r.total?`显示 ${num(S.offset+1)}–${num(Math.min(S.offset+25,r.total))}，共 ${num(r.total)} 个候选`:'0 个候选';
  $('page-number').textContent=`${Math.floor(S.offset/25)+1} / ${Math.max(1,Math.ceil(r.total/25))}`;
  $('prev-page').disabled=S.offset===0;$('next-page').disabled=S.offset+25>=r.total;
}
async function refreshLog() {
  if(!S.run)return;
  try{$('run-log').textContent=(await api('/api/runs/'+S.run.id+'/log')).text||'等待运行记录…';}catch(e){$('run-log').textContent=e.message;}
}
async function download(kind,button) {
  if(!S.run)return;button.disabled=true;
  try {
    const res=await fetch(`/api/runs/${S.run.id}/download?format=${encodeURIComponent(kind)}`,{headers:{'X-PARIS-Token':token}});
    if(!res.ok){const d=await res.json();throw new Error(d.error||'文件无法下载');}
    const blob=await res.blob(),url=URL.createObjectURL(blob),a=document.createElement('a');
    a.href=url;a.download=kind==='log'?'paris-run.log':kind==='receipt'?'paris-RUN.json':'paris-ranking.'+kind;
    document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){notice(e.message);}finally{button.disabled=false;}
}
async function browse(path=null) {
  showError('picker-error','');$('picker-list').textContent='正在读取目录…';$('select-folder').disabled=true;
  try {
    const q=new URLSearchParams({kind:S.pickerKind});if(path)q.set('path',path);
    const r=await api('/api/browse?'+q);S.pickerPath=r.path;S.pickerParent=r.parent;S.canSelect=r.can_select;
    $('picker-path-text').textContent=r.path||'可访问目录';$('picker-up').disabled=!r.parent;$('select-folder').disabled=!r.can_select;
    $('picker-list').innerHTML=r.entries.length?r.entries.map(e=>`<button class="picker-entry" data-entry-path="${esc(e.path)}" data-entry-directory="${e.directory}"><span class="entry-icon">${e.directory?'▱':'▤'}</span><span class="entry-name">${esc(e.name)}</span><span class="entry-hint">${esc(e.hint)}</span></button>`).join(''):(r.can_select?'<div class="empty-table">该目录可以直接使用。点击下方按钮确认。</div>':'<div class="empty-table">此处没有可选目录或模型包</div>');
    $('picker-selection-note').textContent=r.can_select?'此目录符合所选输入类型':S.pickerKind==='project'?'选择含 PARIS_PROJECT.json 的工程目录':'继续浏览，找到模型或完整特征目录';
  }catch(e){$('picker-list').textContent='';showError('picker-error',e.message);}
}
async function openPicker(kind) {
  S.pickerKind=kind;$('picker-title').textContent=kind==='model'?'选择PARIS模型':kind==='project'?'选择原始设计工程':kind==='training'?'选择累计训练样本':kind==='iteration-reference'?'选择PARIS模型':'选择候选特征目录';$('picker').showModal();
  let path=$(kind+'-path').value.trim();if(/\.(tar\.gz|tgz)$/.test(path))path=path.slice(0,path.lastIndexOf('/'));
  await browse(path||null);
}
function choosePath(path) { $(S.pickerKind+'-path').value=path;S.pickerKind==='project'?projectInvalid():['training','iteration-reference'].includes(S.pickerKind)?iterationInvalidate(true):invalidate();$('picker').close(); }
document.querySelectorAll('.nav-step').forEach(el=>el.addEventListener('click',()=>showView(el.dataset.view)));
$('input-form').addEventListener('submit',inspect);$('start-button').addEventListener('click',start);
['model-path','dataset-path'].forEach(id=>$(id).addEventListener('input',invalidate));
document.querySelectorAll('[data-browse]').forEach(el=>el.addEventListener('click',()=>openPicker(el.dataset.browse)));
$('close-picker').addEventListener('click',()=>$('picker').close());$('picker-roots').addEventListener('click',()=>browse());$('picker-up').addEventListener('click',()=>browse(S.pickerParent));
$('select-folder').addEventListener('click',()=>{if(S.canSelect)choosePath(S.pickerPath);});
$('picker-list').addEventListener('click',e=>{const b=e.target.closest('[data-entry-path]');if(!b)return;b.dataset.entryDirectory==='true'?browse(b.dataset.entryPath):choosePath(b.dataset.entryPath);});
$('help-button').addEventListener('click',()=>$('help').showModal());$('close-help').addEventListener('click',()=>$('help').close());
$('recent-runs').addEventListener('click',async e=>{const b=e.target.closest('[data-run-id]');if(!b)return;try{const run=await api('/api/runs/'+b.dataset.runId);S.design=null;S.offset=0;S.query='';$('candidate-search').value='';renderRun(run);showView(run.status==='complete'?'output':'process');refreshLog();}catch(err){notice(err.message);}});
$('view-results').addEventListener('click',()=>showView('output'));
$('design-tabs').addEventListener('click',e=>{const b=e.target.closest('[data-design]');if(!b)return;S.design=b.dataset.design;S.offset=0;S.query='';$('candidate-search').value='';renderDesignTabs();loadResults().catch(err=>notice(err.message));});
$('candidate-search').addEventListener('input',()=>{clearTimeout(S.searchTimer);S.searchTimer=setTimeout(()=>{S.query=$('candidate-search').value.trim();S.offset=0;loadResults().catch(e=>notice(e.message));},180);});
$('prev-page').addEventListener('click',()=>{S.offset=Math.max(0,S.offset-25);loadResults().catch(e=>notice(e.message));});
$('next-page').addEventListener('click',()=>{S.offset+=25;loadResults().catch(e=>notice(e.message));});
$('log-panel').addEventListener('toggle',()=>{if($('log-panel').open)refreshLog();});
$('show-output-log').addEventListener('click',()=>{showView('process');$('log-panel').open=true;refreshLog();});
$('copy-output').addEventListener('click',async()=>{try{await navigator.clipboard.writeText(S.run.output);notice('结果路径已复制');}catch(e){notice('请手动复制下方结果路径。');}});
document.querySelectorAll('[data-download]').forEach(b=>b.addEventListener('click',()=>download(b.dataset.download,b)));
setInterval(tick,1000);
(async()=>{
  try {
    const c=await api('/api/config');S.workspace=c.workspace;$('package-version').textContent='PARIS '+c.version;$('workspace-path').textContent=c.workspace+'/runs/…';
    $('model-path').value=c.defaults.model;$('dataset-path').value=c.defaults.dataset;S.runs=c.model_profile==='single-xgb'?c.runs.filter(r=>r.inspection?.algorithm==='paris.single-xgb.1'):c.runs;S.active=c.active;S.prepConfig=c.preparation;S.prepActive=c.preparation.active;S.prepRuns=c.preparation.runs;$('raw-flow-plan').innerHTML=c.preparation.stages.map((s,i)=>`<span><b>${String(i+1).padStart(2,'0')}</b>${esc(s.title)}</span>`).join('<i>→</i>');renderProfile();S.prepRuns.slice().reverse().forEach(rememberPrep);renderRecent();updateNavigation();
    if(S.prepActive){setInputMode('raw');const r=await api('/api/preparations/'+S.prepActive);renderPrep(r);watchPrep(r.id);}
    if(c.active){const r=await api('/api/runs/'+c.active);renderRun(r);showView('process');watch(c.active);}
    else if(S.runs.length){renderRun(S.runs[0]);}
    initIteration(c.iteration);
  }catch(e){showError('input-error','无法连接 PARIS 服务：'+e.message);}
})();

// Raw projects stay in INPUT until their actual GR tables pass the original model gate.
Object.assign(S,{inputMode:'features',projectInfo:null,preparation:null,prepActive:null,prepRuns:[],catalog:null,family:'all',prepChecking:false});
function setInputMode(mode) {
  S.inputMode=mode;
  document.querySelectorAll('[data-input-mode]').forEach(b=>{b.classList.toggle('active',b.dataset.inputMode===mode);b.setAttribute('aria-selected',String(b.dataset.inputMode===mode));});
  $('prepared-input').classList.toggle('hidden',mode!=='features');$('raw-input').classList.toggle('hidden',mode!=='raw');
  $('prediction-actions').classList.toggle('hidden',mode!=='features');$('dataset-path').required=mode==='features';
  $('input-summary').classList.toggle('hidden',mode!=='features'||!S.inspection);
  showError('input-error','');$('feature-catalog').classList.add('hidden');if(mode==='raw'&&S.preparation?.catalog)renderCatalog(S.preparation.catalog);if(mode==='features'&&S.catalog?.dataset===$('dataset-path').value.trim())renderCatalog(S.catalog);$('workspace-path').textContent=(S.workspace||'')+(mode==='raw'?'/preparations/…':'/runs/…');document.querySelector('.pipeline-label').textContent=mode==='raw'?'特征就绪后':'将执行';renderInputStatus();updateNavigation();
}
function renderInputStatus() {if(S.inputMode==='raw'){$('input-status').textContent=S.projectInfo?.ready?'工程检查通过':'待检查工程';$('input-status').className='status-chip '+(S.projectInfo?.ready?'complete':'neutral');}else{$('input-status').textContent=S.inspection?'检查通过':'待检查';$('input-status').className='status-chip '+(S.inspection?'complete':'neutral');}}
function projectInvalid() {S.projectInfo=null;$('feature-catalog').classList.add('hidden');if(!S.prepActive)$('preparation-panel').classList.add('hidden');$('project-checks').classList.add('hidden');showError('project-blockers','');renderInputStatus();rawControls();}
function rawControls() {
  $('prepare-project').disabled=!S.projectInfo?.ready||!!S.prepActive||!!S.active||!!S.prepChecking||!!S.iterActive;
  $('check-project').disabled=!!S.prepChecking;$('raw-upload').disabled=!!S.prepChecking;
  $('project-path').disabled=!!S.prepChecking;document.querySelector('[data-browse="project"]').disabled=!!S.prepChecking;document.querySelectorAll('[data-input-mode]').forEach(b=>{b.disabled=!!S.prepChecking||!!S.checking;});
  $('use-prepared').disabled=!!S.active||!!S.prepActive;
}
function projectInfo(info) {
  S.projectInfo=info;$('project-path').value=info.project;
  $('project-checks').innerHTML=`<div class="project-identity"><strong>${esc(info.design)}</strong><span>Top: ${esc(info.top)}</span><span>计划 ${num(info.candidates)} 个分割候选</span></div>`+info.checks.map(c=>`<div class="project-check ${c.ok?'ok':'missing'}"><span>${c.ok?'✓':'!'}</span><strong>${esc(c.label)}</strong><small>${esc(c.message)}</small></div>`).join('');
  $('project-checks').classList.remove('hidden');showError('project-blockers',info.blockers.join('；'));
  renderInputStatus();rawControls();
}
async function inspectProject() {
  S.projectInfo=null;S.prepChecking=true;rawControls();showError('project-blockers','');
  try{projectInfo(await api('/api/projects/inspect',{project:$('project-path').value.trim()}));}
  catch(e){S.projectInfo=null;showError('project-blockers',e.message);}
  finally{S.prepChecking=false;rawControls();}
}
async function uploadProject() {
  const file=$('raw-upload').files[0];if(!file)return;
  if(file.size>256*1024**2){showError('project-blockers','文件超过256 MiB，请在服务器上放好工程后选择目录。');return;}
  projectInvalid();S.prepChecking=true;rawControls();showError('project-blockers','');$('raw-profile').textContent='正在上传并检查工程文件…';
  try{
    const res=await fetch('/api/projects/upload?'+new URLSearchParams({filename:file.name}),{method:'POST',headers:{'X-PARIS-Token':token,'Content-Type':'application/octet-stream'},body:file});
    const data=await res.json();if(!res.ok)throw new Error(data.error||'工程上传失败');projectInfo(data);
  }catch(e){showError('project-blockers',e.message);}
  finally{S.prepChecking=false;$('raw-upload').value='';renderProfile();rawControls();}
}
function renderProfile() {$('raw-profile').textContent=S.prepConfig?.configured?'已配置流程：'+S.prepConfig.profile+' · 2 CPU · 运行到 GR':'尚未配置工艺流程。可先导入工程、检查缺少的输入；配置后才能生成 CTS/GR 特征。';}
const PREP_LABELS={queued:'准备中',running:'进行中',complete:'检查通过',failed:'已停止',blocked:'模型未兼容',interrupted:'已中断'};
function rememberPrep(r) {
  S.prepRuns=[r,...S.prepRuns.filter(x=>x.id!==r.id)].sort((a,b)=>b.created.localeCompare(a.created));
  $('prep-history').classList.toggle('hidden',!S.prepRuns.length);
  $('prep-history-select').innerHTML='<option value="">选择已有记录</option>'+S.prepRuns.map(x=>`<option value="${esc(x.id)}">${esc(x.inspection.design)} · ${esc(new Date(x.created).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false}))} · ${esc(PREP_LABELS[x.status]||x.status)}</option>`).join('');
  if(S.preparation)$('prep-history-select').value=S.preparation.id;
}
function renderPrep(r) {
  S.preparation=r;rememberPrep(r);$('preparation-panel').classList.remove('hidden');
  $('prep-title').textContent=`${r.inspection.design} · ${num(r.inspection.candidates)} 个候选`;
  $('prep-status').textContent=PREP_LABELS[r.status]||r.status;$('prep-status').className='status-chip '+(r.status==='complete'?'complete':['failed','blocked','interrupted'].includes(r.status)?'failed':'running');
  $('prep-stages').innerHTML=(S.prepConfig?.stages||[]).map((s,i)=>{const status=r.steps[s.id];const e=r.events.filter(e=>e.stage===s.id).at(-1);return `<div class="process-step ${esc(status)}"><span class="process-icon">${status==='complete'?'✓':status==='running'?'':['failed','blocked'].includes(status)?'!':i+1}</span><div><h3>${esc(s.title)}</h3><p>${esc(e?.message||s.description)}</p></div><span class="step-status">${esc({complete:'已完成',running:'进行中',pending:'等待',failed:'失败',blocked:'待解决'}[status]||status)}</span></div>`;}).join('');
  $('prep-progress').textContent=r.status==='running'&&r.progress?`本阶段实际完成 ${num(r.progress.completed)} / ${num(r.progress.total)} ${r.progress.unit}`:r.status==='running'?'当前阶段运行中，等待工具产物回执；日志可查看详细进度。':`用时 ${duration((r.finished_at||Date.now()/1000)-r.started_at)}`;
  $('prep-output').textContent=r.output;showError('prep-error',r.error||'');
  $('prep-artifacts').innerHTML=Object.entries(r.artifacts||{}).map(([stage,items])=>`<details><summary>${esc(S.prepConfig.stages.find(s=>s.id===stage)?.title||stage)} · ${num(Object.keys(items).length)} 个已核验产物</summary>${Object.entries(items).map(([name,v])=>`<div title="SHA-256: ${esc(v.sha256)}"><code>${esc(name)}</code><span>${num(Math.ceil(v.bytes/1024))} KiB</span></div>`).join('')}</details>`).join('');
  $('use-prepared').classList.toggle('hidden',r.status!=='complete');
  if(r.catalog)renderCatalog(r.catalog);rawControls();updateNavigation();
}
async function prepLog() {if(S.preparation)try{$('prep-log').textContent=(await api('/api/preparations/'+S.preparation.id+'/log')).text||'等待日志…';}catch(e){$('prep-log').textContent=e.message;}}
async function watchPrep(id) {
  if(S.prepActive!==id)return;clearTimeout(S.prepTimer);
  try{
    const r=await api('/api/preparations/'+id);if(!['running','queued'].includes(r.status))S.prepActive=null;
    if(S.preparation?.id===id){renderPrep(r);if(document.querySelector('.prep-log').open)prepLog();}else{rememberPrep(r);rawControls();updateNavigation();}
    if(S.prepActive===id)S.prepTimer=setTimeout(()=>watchPrep(id),1800);
  }catch(e){notice('数据准备服务暂时不可达，正在重试。');S.prepTimer=setTimeout(()=>watchPrep(id),4000);}
}
async function startPrep() {
  if(!S.projectInfo?.ready||S.prepActive||S.active||S.iterActive)return;
  if(!$('model-path').value.trim()){showError('project-blockers','请先选择用于最终兼容性检查的完整模型。');return;}
  S.prepChecking=true;rawControls();
  try{const r=await api('/api/preparations',{project:$('project-path').value.trim(),model:$('model-path').value.trim()});S.prepActive=r.id;renderPrep(r);watchPrep(r.id);}
  catch(e){showError('project-blockers',e.message);}finally{S.prepChecking=false;rawControls();}
}
async function loadCatalog(dataset) {
  if(!dataset){notice('请先选择特征目录。');return;}
  const c=await api('/api/features?'+new URLSearchParams({dataset}));renderCatalog(c);$('feature-catalog').scrollIntoView({behavior:'smooth',block:'start'});
}
function renderCatalog(c) {
  S.catalog=c;S.family='all';$('feature-catalog').classList.remove('hidden');$('feature-search').value='';
  const present=c.tables.filter(t=>t.status==='complete');
  $('catalog-summary').textContent=`${present.length} / 3 张输入表 · ${num(c.columns)} 个实际字段（含标识与连接键）`;
  $('catalog-tables').innerHTML=c.tables.map(t=>`<div class="catalog-table ${t.status==='missing'?'missing':''}"><strong>${esc(t.name)}</strong><span>${t.status==='missing'?'尚未生成':`${num(t.rows)} 行 · ${num(t.columns)} 列`}</span></div>`).join('');
  $('catalog-families').innerHTML='<button type="button" class="active" data-family="all">全部</button>'+c.families.map(g=>`<button type="button" data-family="${esc(g.name)}">${esc(g.name)}<span>${num(g.count)}</span></button>`).join('');
  $('catalog-footnote').textContent='来源：'+c.dataset;filterFeatures();
}
function filterFeatures() {
  if(!S.catalog)return;const q=$('feature-search').value.trim().toLowerCase();
  const fields=S.catalog.tables.flatMap(t=>t.fields||[]).filter(f=>(S.family==='all'||S.family===f.family)&&(!q||f.name.toLowerCase().includes(q)));
  $('catalog-fields').innerHTML=fields.length?fields.slice(0,150).map(f=>`<tr><td><code>${esc(f.name)}</code></td><td>${esc(f.table)}</td><td><code>${esc(f.dtype)}</code></td><td class="sample-value" title="${esc((f.sample||[]).join(' · '))}">${esc((f.sample||[]).join(' · ')||'—')}</td></tr>`).join(''):'<tr><td colspan="4" class="empty-table">没有匹配的特征</td></tr>';
  if(fields.length>150)$('catalog-fields').insertAdjacentHTML('beforeend',`<tr><td colspan="4" class="empty-table">匹配 ${num(fields.length)} 个字段，显示前150个；使用分组或搜索缩小范围。</td></tr>`);
}
document.querySelectorAll('[data-input-mode]').forEach(b=>b.addEventListener('click',()=>setInputMode(b.dataset.inputMode)));
$('project-path').addEventListener('input',projectInvalid);$('check-project').addEventListener('click',inspectProject);$('prepare-project').addEventListener('click',startPrep);$('raw-upload').addEventListener('change',uploadProject);
$('inspect-features').addEventListener('click',()=>loadCatalog($('dataset-path').value.trim()).catch(e=>notice(e.message)));
$('feature-search').addEventListener('input',filterFeatures);
$('catalog-families').addEventListener('click',e=>{const b=e.target.closest('[data-family]');if(!b)return;S.family=b.dataset.family;document.querySelectorAll('[data-family]').forEach(x=>x.classList.toggle('active',x===b));filterFeatures();});
document.querySelector('.prep-log').addEventListener('toggle',()=>{if(document.querySelector('.prep-log').open)prepLog();});
$('prep-history-select').addEventListener('change',async()=>{const id=$('prep-history-select').value;if(id)try{renderPrep(await api('/api/preparations/'+id));prepLog();}catch(e){notice(e.message);}});
$('use-prepared').addEventListener('click',()=>{if(S.preparation?.status!=='complete')return;$('dataset-path').value=S.preparation.dataset;invalidate();setInputMode('features');inspect();});
$('project-template').addEventListener('click',()=>{
  const p={version:1,design:'my_design',top:'my_top',inputs:{netlist:'design.v',lef:['tech.lef','cells.lef'],liberty:['cells.lib'],sdc:'design.sdc'},partition:{count:32,seed:17}};
  const url=URL.createObjectURL(new Blob([JSON.stringify(p,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='PARIS_PROJECT.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});

// Per-design full-architecture refits. No score or increasing learning curve is fabricated.
Object.assign(S,{iterInfo:null,iterRun:null,iterActive:null,iterBusy:false,iterConfig:null});
function iterationInvalidate(reset=false) {
  S.iterInfo=null;$('iteration-data-summary').classList.add('hidden');showError('iteration-error','');
  $('iteration-input-status').className='status-chip neutral';$('iteration-input-status').textContent='待检查';
  if(reset){$('iteration-selectors').classList.add('hidden');$('iteration-design').innerHTML='';$('iteration-template').innerHTML='';}
  iterationControls();
}
function iterationControls() {
  $('check-iteration').disabled=S.iterBusy;
  document.querySelectorAll('#training-path,#iteration-reference-path,#iteration-design,#iteration-template,[data-browse="training"],[data-browse="iteration-reference"]').forEach(b=>{b.disabled=S.iterBusy;});
  $('start-iteration').disabled=!S.iterInfo?.ready||S.iterBusy||!!S.iterActive||!!S.active||!!S.prepActive;
}
async function inspectIteration(event) {
  event?.preventDefault();S.iterBusy=true;iterationControls();showError('iteration-error','');$('check-iteration').textContent='检查中…';
  try {
    const r=await api('/api/iterations/inspect',{dataset:$('training-path').value.trim(),reference:$('iteration-reference-path').value.trim(),design:$('iteration-design').value||null,template:$('iteration-template').value||null});
    S.iterInfo=r;
    $('iteration-selectors').classList.remove('hidden');
    $('iteration-design').innerHTML=r.designs.map(x=>`<option value="${esc(x.name)}">${esc(x.name)} · ${num(x.n)} 个实测候选</option>`).join('');$('iteration-design').value=r.design;
    $('iteration-template').parentElement.classList.toggle('hidden',r.algorithm==='paris.single-xgb.1');
    $('iteration-template').innerHTML=r.templates.map(x=>`<option value="${esc(x)}">默认配方</option>`).join('');$('iteration-template').value=r.template;
    $('iteration-input-status').className='status-chip '+(r.ready?'complete':'neutral');$('iteration-input-status').textContent=r.ready?'检查通过':'暂不可训练';
    $('iteration-data-summary').innerHTML=`<div class="iteration-data-counts"><div><span>累计实测候选</span><strong>${num(r.labeled)}</strong></div><div><span>本轮训练 / 新增</span><strong>${num(r.training)} <small>/ +${num(r.new_training)}</small></strong></div><div><span>固定开发验证</span><strong>${num(r.validation)}</strong></div><div><span>验证协议</span><strong class="protocol-state">${r.protocol_exists?'沿用已冻结划分':'首次训练时冻结'}</strong></div></div><div class="iteration-provenance"><span>PDK：${esc(r.context.pdk_id)}</span><span>流程：${esc(r.context.flow_id)}</span><span>角点：${esc(r.context.corner)}</span><span>标签版本：${esc(r.context.label_version)}</span><span>单位：ns</span></div><details class="iteration-files"><summary>实际输入表与样本来源</summary>${r.files.map(f=>`<div><code>${esc(f.name)}</code><span>全表 ${num(f.rows)} 行 · ${num(f.columns)} 列</span></div>`).join('')}</details>`;
    $('iteration-data-summary').classList.remove('hidden');showError('iteration-error',r.blockers.join('；'));
  }catch(e){S.iterInfo=null;$('iteration-input-status').textContent='检查未通过';$('iteration-input-status').className='status-chip failed';showError('iteration-error',e.message);}
  finally{S.iterBusy=false;$('check-iteration').textContent='检查样本';iterationControls();}
}
function iterMetric(value,digits=4) {return value===null||value===undefined?'—':Number(value).toFixed(digits);}
function renderIteration(r) {
  S.iterRun=r;$('iteration-progress-panel').classList.remove('hidden');
  $('iteration-job-title').textContent=r.display_name||r.inspection.design+(r.inspection.purpose==='engineering_smoke'?' · 测试版本':' · 新设计版本');$('iteration-job-id').textContent=r.id;
  $('iteration-job-status').textContent={queued:'准备中',running:'运行中',complete:'验证已完成',failed:'已停止',interrupted:'已中断'}[r.status]||r.status;
  $('iteration-job-status').className='status-chip '+(r.status==='complete'?'complete':['failed','interrupted'].includes(r.status)?'failed':'running');
  $('iteration-steps').innerHTML=(S.iterConfig?.steps||[]).map((s,i)=>{const state=r.steps[s.id];return `<div class="process-step ${esc(state)}"><span class="process-icon">${state==='complete'?'✓':state==='running'?'':state==='failed'?'!':i+1}</span><div><h3>${esc(s.title)}</h3><p>${esc(state==='running'?'正在执行':state==='complete'?'':state==='failed'?r.error||'已停止':'等待前序阶段')}</p></div><span class="step-status">${esc({pending:'等待',complete:'已完成',running:'进行中',failed:'失败',interrupted:'已中断'}[state]||state)}</span></div>`;}).join('');
  const p=r.training_progress;
  const phases={load_training:'读取训练样本',wire_crossfit:'训练候选隔离的线网交叉拟合',wire_refit:'拟合完整线网模型',candidate_refit:'拟合原始候选模型',candidate_crossfit:'构建候选纠错训练信号',endpoint_refit:'拟合端点残差分支',sealed:'模型已封存'};
  $('iteration-training-progress').textContent=`用时 ${duration((r.finished_at||Date.now()/1000)-r.started_at)}${r.status==='running'&&p?' · '+(phases[p.stage]||p.stage)+(p.fold?' '+p.fold+'/'+p.total:''):''}`;
  $('iteration-output-path').textContent=r.output;showError('iteration-job-error',r.error||'');
  $('iteration-evaluation').classList.toggle('hidden',!r.metrics);
  if(r.metrics) {
    $('iteration-eval-scope').textContent=`${r.inspection.design} · ${num(r.inspection.validation)} 个验证候选`;
    $('iteration-gate').textContent=r.eligible?'可启用':r.example?'未启用':r.inspection.purpose==='engineering_smoke'?'测试版本':'未通过验证';$('iteration-gate').className='status-chip '+(r.eligible?'complete':r.example?'neutral':'failed');
    const names={GR:'同检查点 GR',current:'当前版本',candidate:'本轮新版本',...r.metric_names};
    $('iteration-metrics').innerHTML=['GR','current','candidate'].filter(k=>r.metrics[k]).map(k=>{const m=r.metrics[k];return `<tr class="${k==='candidate'?'iteration-candidate-row':''}"><td>${esc(names[k])}</td><td>${num(m.n)}</td><td>${iterMetric(m.tau_b)}</td></tr>`;}).join('');
    const current=S.iterConfig?.lines.find(x=>x.line===r.inspection.line)?.current;
    $('activate-iteration').disabled=!r.eligible||current?.id===r.id||!!S.iterActive||!!S.active||!!S.prepActive;
    $('activate-iteration').textContent=current?.id===r.id?'该版本已启用':'启用该设计版本';
    $('iteration-use-model').disabled=current?.id!==r.id;$('iteration-export').disabled=!r.eligible;
  }
  iterationControls();
}
function renderIterationVersions() {
  if(!S.iterConfig)return;
  $('iteration-version-count').textContent=S.iterConfig.runs.length+' 个版本';
  $('iteration-current-models').innerHTML=S.iterConfig.lines.filter(l=>l.current).map(l=>`<div><span class="local-dot"></span><strong>${esc(l.design)}</strong><span>${esc(l.context.flow_id)}</span><code>${esc(l.current.id)}</code><small>当前启用</small></div>`).join('');
  $('iteration-versions').innerHTML=S.iterConfig.runs.length?S.iterConfig.runs.map(r=>{
    const current=S.iterConfig.lines.find(l=>l.line===r.inspection.line)?.current;
    const active=current?.id===r.id;
    const label=active?'已启用':r.example?'未启用':r.inspection.purpose==='engineering_smoke'&&r.status==='complete'?'测试版本':r.status==='complete'?(r.eligible?'待启用':'未通过验证'):{running:'运行中',queued:'准备中',failed:'已停止',interrupted:'已中断'}[r.status]||r.status;
    return `<tr><td><strong>${esc(r.inspection.design)}</strong><small class="iteration-version-meta">${esc(r.id)}</small></td><td>${num(r.inspection.training)} / ${num(r.inspection.validation)}</td><td>${iterMetric(r.metrics?.candidate?.tau_b)}</td><td><span class="status-chip ${active?'complete':r.status==='failed'?'failed':'neutral'}">${label}</span></td><td><button class="text-button" data-iteration-detail="${esc(r.id)}">详情</button>${r.eligible&&!active?`<button class="text-button" data-iteration-activate="${esc(r.id)}">${current?'启用 / 回退':'启用'}</button>`:''}</td></tr>`;
  }).join(''):'<tr><td colspan="5" class="empty-table">还没有设计适配版本。先检查累计实测样本，再训练首个版本。</td></tr>';
}
async function refreshIterationConfig() {
  const c=await api('/api/config');S.iterConfig=c.iteration;renderIterationVersions();return c.iteration;
}
async function iterationLog() {if(S.iterRun)try{$('iteration-log').textContent=(await api('/api/iterations/'+S.iterRun.id+'/log')).text||'等待日志…';}catch(e){$('iteration-log').textContent=e.message;}}
async function watchIteration(id) {
  if(S.iterActive!==id)return;clearTimeout(S.iterTimer);
  try {
    const r=await api('/api/iterations/'+id);
    if(!['running','queued'].includes(r.status)){S.iterActive=null;await refreshIterationConfig();}
    if(S.iterRun?.id===id){renderIteration(r);if(document.querySelector('.iteration-log').open)iterationLog();}
    updateNavigation();rawControls();iterationControls();
    if(S.iterActive===id)S.iterTimer=setTimeout(()=>watchIteration(id),2000);
  }catch(e){notice('迭代服务暂时不可达，正在重试。');S.iterTimer=setTimeout(()=>watchIteration(id),4000);}
}
async function startIteration() {
  if(!S.iterInfo?.ready||S.iterActive||S.active||S.prepActive||S.iterBusy)return;
  S.iterBusy=true;iterationControls();
  try{const r=await api('/api/iterations',{dataset:S.iterInfo.dataset,reference:S.iterInfo.reference,design:S.iterInfo.design,template:S.iterInfo.template});S.iterActive=r.id;renderIteration(r);showView('iterate');$('iteration-progress-panel').scrollIntoView({behavior:'smooth'});watchIteration(r.id);await refreshIterationConfig();}
  catch(e){showError('iteration-error',e.message);}finally{S.iterBusy=false;iterationControls();updateNavigation();}
}
async function activateIteration(id) {
  try{await api('/api/iterations/activate',{id});await refreshIterationConfig();if(S.iterRun)renderIteration(S.iterRun);notice('模型版本已启用。');}
  catch(e){notice(e.message);}
}
async function exportIteration() {
  if(!S.iterRun?.eligible)return;$('iteration-export').disabled=true;
  try{
    const res=await fetch('/api/iterations/'+S.iterRun.id+'/download?format=model',{headers:{'X-PARIS-Token':token}});
    if(!res.ok)throw new Error((await res.json()).error||'权重无法导出');
    const url=URL.createObjectURL(await res.blob()),a=document.createElement('a');a.href=url;a.download='PARIS4-'+S.iterRun.inspection.design+'-'+S.iterRun.id+'.tar.gz';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }catch(e){notice(e.message);}finally{$('iteration-export').disabled=false;}
}
function initIteration(config) {
  S.iterConfig=config;S.iterActive=config.active;$('iteration-reference-path').value=config.runs[0]?.inspection.reference||$('model-path').value;$('training-path').value=config.runs[0]?.inspection.dataset||'';renderIterationVersions();
  if(config.active){const r=config.runs.find(r=>r.id===config.active);if(r)renderIteration(r);showView('iterate');watchIteration(config.active);}
  else if(config.runs.length)renderIteration(config.runs[0]);
  iterationControls();updateNavigation();
}
$('iteration-form').addEventListener('submit',inspectIteration);$('start-iteration').addEventListener('click',startIteration);
['training-path','iteration-reference-path'].forEach(id=>$(id).addEventListener('input',()=>iterationInvalidate(true)));
['iteration-design','iteration-template'].forEach(id=>$(id).addEventListener('change',()=>{iterationInvalidate();inspectIteration();}));
$('iteration-versions').addEventListener('click',async e=>{const detail=e.target.closest('[data-iteration-detail]'),release=e.target.closest('[data-iteration-activate]');if(release)return activateIteration(release.dataset.iterationActivate);if(detail)try{renderIteration(await api('/api/iterations/'+detail.dataset.iterationDetail));iterationLog();$('iteration-progress-panel').scrollIntoView({behavior:'smooth'});}catch(err){notice(err.message);}});
$('activate-iteration').addEventListener('click',()=>{if(S.iterRun)activateIteration(S.iterRun.id);});
$('iteration-use-model').addEventListener('click',()=>{if(!S.iterRun)return;$('model-path').value=S.iterRun.model;$('dataset-path').value='';invalidate();setInputMode('features');showView('input');notice('已选用该设计的适配权重；请选择新的候选 GR 特征进行预测。');});
$('iteration-export').addEventListener('click',exportIteration);
document.querySelector('.iteration-log').addEventListener('toggle',()=>{if(document.querySelector('.iteration-log').open)iterationLog();});
$('training-template').addEventListener('click',()=>{
  const j={version:1,pdk_id:'your-pdk-version',flow_id:'your-verified-flow-version',corner:'nominal',label_version:'measured-labels-v1',feature_version:'original-full-gr-v1',time_unit:'ns',label_source:'measured_dr_sta'};
  const url=URL.createObjectURL(new Blob([JSON.stringify(j,null,2)+'\n'],{type:'application/json'})),a=document.createElement('a');a.href=url;a.download='PARIS_TRAINING.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});


async function loadBenchmark() {
  const base=await api('/api/benchmark');S.benchmark=base;
  const multi=S.benchmarkMode==='five',d=multi?base.five_splits:base;
  document.querySelectorAll('[data-benchmark-mode]').forEach(b=>{b.classList.toggle('active',(multi?'five':'single')===b.dataset.benchmarkMode);b.disabled=b.dataset.benchmarkMode==='five'&&!base.five_splits;});
  if(!d)return;
  S.benchmarkDisplay=d;S.benchmarkMulti=multi;
  $('benchmark-split-label').textContent=multi?'5次划分 · 每次1000训练 / 200测试':'既有单次结果';
  $('benchmark-chart-subtitle').textContent=multi?'Kendall τ-b ↑ · 五次均值，误差线为标准差':'Kendall τ-b ↑ · 越高，排序与 DR 实测越一致';
  const names={CTS:'CTS 直接排序',GR:'GR 直接排序',PARIS:'PARIS'},cols=['CTS','GR','PARIS'],colors=['#9ba6b8','#61a5b7','#376ddd'];
  $('benchmark-cards').innerHTML=cols.map(k=>`<article class="panel benchmark-card ${k.toLowerCase()}"><span class="small-label">${names[k]}</span><strong>${(d.macro[k]*100).toFixed(2)}<small>${multi?' ± '+(d.std[k]*100).toFixed(2):' / 100'}</small></strong><span>Macro τ-b = ${d.macro[k].toFixed(4)}${multi?' ± '+d.std[k].toFixed(4):''}</span>${k==='PARIS'?`<b>较 GR ${d.macro.PARIS>=d.macro.GR?'+':''}${((d.macro.PARIS-d.macro.GR)*100).toFixed(2)} 分</b>`:`<b>${multi?'五次划分 · 均值 ± 标准差':'时序阶段基线'}</b>`}</article>`).join('');
  const lower=Math.min(0,...d.rows.flatMap(r=>cols.map(k=>r[k]-(multi?r.std[k]:0)))),upper=Math.max(1,...d.rows.flatMap(r=>cols.map(k=>r[k]+(multi?r.std[k]:0))));
  const min=Math.floor(lower*4)/4,max=Math.ceil(upper*4)/4,W=920,H=280,L=46,R=16,T=16,B=48,plotH=H-T-B,group=(W-L-R)/d.rows.length,Y=v=>T+(max-v)/(max-min)*plotH;
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="七个设计的CTS、GR与PARIS Kendall tau-b对比">`;
  for(let t=min;t<=max+.001;t+=.25){const y=Y(t);svg+=`<line x1="${L}" x2="${W-R}" y1="${y}" y2="${y}" stroke="#e8edf4"/><text x="${L-10}" y="${y+4}" text-anchor="end" fill="#7b8799" font-size="11">${t.toFixed(2)}</text>`;}
  d.rows.forEach((row,i)=>{cols.forEach((k,j)=>{const x=L+i*group+group*.15+j*group*.23,w=group*.19,y=Y(row[k]);svg+=`<rect x="${x}" y="${Math.min(y,Y(0))}" width="${w}" height="${Math.abs(Y(0)-y)}" rx="3" fill="${colors[j]}"><title>${esc(row.design)} · ${k}: ${row[k].toFixed(4)}${multi?' ± '+row.std[k].toFixed(4):''}</title></rect>`;
    if(multi){const cx=x+w/2,a=Y(row[k]+row.std[k]),b=Y(row[k]-row.std[k]);svg+=`<path d="M${cx} ${a}V${b}M${cx-3} ${a}H${cx+3}M${cx-3} ${b}H${cx+3}" fill="none" stroke="#40516c" stroke-width="1.1"/>`;}});svg+=`<text x="${L+(i+.5)*group}" y="${H-20}" text-anchor="middle" font-size="11" fill="#53647e">${esc(row.design)}</text>`;});
  $('benchmark-chart').innerHTML=svg+'</svg>';
  const makeRow=(r,macro=false)=>`<tr class="${macro?'benchmark-macro':''}"><td>${macro?'Macro':esc(r.design)}</td><td class="number-cell">${r.training}</td><td class="number-cell">${r.test}</td>${cols.map(k=>`<td class="number-cell ${k==='PARIS'?'benchmark-value':''}">${r[k].toFixed(4)}${multi?'<small class="benchmark-std"> ± '+r.std[k].toFixed(4)+'</small>':''}</td>`).join('')}<td class="number-cell ${(r.PARIS-r.GR)>=0?'up':'down'}">${(r.PARIS-r.GR)>=0?'+':''}${(r.PARIS-r.GR).toFixed(4)}</td></tr>`;
  $('benchmark-body').innerHTML=d.rows.map(r=>makeRow(r)).join('')+makeRow({...d.macro,std:d.std,training:d.training,test:d.test},true);
  $('benchmark-repeats').classList.toggle('hidden',!multi);
  $('benchmark-seed-body').innerHTML=multi?d.per_seed.map(r=>`<tr><td>${r.display_id ?? r.seed}</td>${cols.map(k=>`<td class="number-cell ${k==='PARIS'?'benchmark-value':''}">${r[k].toFixed(4)}</td>`).join('')}<td class="number-cell">${r.PARIS>=r.GR?'+':''}${(r.PARIS-r.GR).toFixed(4)}</td></tr>`).join(''):'';
  $('benchmark-source').textContent=d.experiment+' · '+d.date;
  $('benchmark-protocol-note').textContent=multi?'划分：'+(d.display_ids||d.seeds).join('、')+'。每次1000训练、200测试、4例未用；模型与选参规则保持一致，标准差使用ddof=1。':'固定1000例训练、200例测试，另4例未使用。同设计候选开发对照，模型参数仅在训练集内选择。';
}
document.querySelectorAll('[data-benchmark-mode]').forEach(b=>b.addEventListener('click',()=>{S.benchmarkMode=b.dataset.benchmarkMode;loadBenchmark().catch(e=>showError('benchmark-error',e.message));}));
$('benchmark-download').addEventListener('click',()=>{
  const d=S.benchmarkDisplay;if(!d)return;let rows;
  if(S.benchmarkMulti){
    const keys=['section','design','display_id','actual_seed','training','test','CTS','CTS_std','GR','GR_std','PARIS','PARIS_std'];
    rows=[keys.join(','),...d.rows.map(r=>['design',r.design,'','',r.training,r.test,r.CTS,r.std.CTS,r.GR,r.std.GR,r.PARIS,r.std.PARIS].join(',')),['macro','Macro','','',d.training,d.test,d.macro.CTS,d.std.CTS,d.macro.GR,d.std.GR,d.macro.PARIS,d.std.PARIS].join(','),...d.per_seed.map(r=>['split','Macro',r.display_id??r.seed,r.seed,d.training,d.test,r.CTS,'',r.GR,'',r.PARIS,''].join(','))];
  }else{const keys=['design','training','test','CTS','GR','PARIS'];rows=[keys.join(','),...d.rows.map(r=>keys.map(k=>r[k]).join(',')),['Macro',d.training,d.test,d.macro.CTS,d.macro.GR,d.macro.PARIS].join(',')];}
  const url=URL.createObjectURL(new Blob(['\ufeff'+rows.join('\n')],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=S.benchmarkMulti?'PARIS_Open3DBench_five_splits.csv':'PARIS_Open3DBench_1000_200.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
});

// Method overview: keep the supplied full-resolution diagram available at native size.
const methodDiagram=$('method-diagram-dialog'),methodDiagramScroll=$('method-diagram-scroll');
document.querySelectorAll('[data-method-zoom]').forEach(button=>button.addEventListener('click',()=>{
  methodDiagramScroll.classList.remove('original');$('method-image-size').textContent='原始尺寸';$('method-image-size').setAttribute('aria-pressed','false');methodDiagram.showModal();
}));
$('method-image-size').addEventListener('click',()=>{
  const original=methodDiagramScroll.classList.toggle('original');$('method-image-size').textContent=original?'适应窗口':'原始尺寸';$('method-image-size').setAttribute('aria-pressed',String(original));methodDiagramScroll.scrollTo(0,0);
});
$('close-method-diagram').addEventListener('click',()=>methodDiagram.close());
