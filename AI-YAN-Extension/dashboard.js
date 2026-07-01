const BACKEND = 'http://localhost:5000';
const REPORT_URL = 'https://AI-YAN.example/report';

const $ = (id) => document.getElementById(id);
const stage = $('stage'), stagePh = $('stage-ph'), stageSpin = $('stage-spinner');
const imgOriginal = $('img-original'), imgHeatmap = $('img-heatmap');
const tabOriginal = $('tab-original'), tabHeatmap = $('tab-heatmap');
const btnUpload = $('btn-upload'), btnOverlay = $('btn-overlay'), fileInput = $('file-input');
const badge = $('badge'), confPct = $('conf-pct'), confBar = $('conf-bar'), procTime = $('proc-time');
const btnExport = $('btn-export'), agree = $('agree');

function normalize(v){
  if(!v) return '';
  if(v.startsWith('data:')||v.startsWith('http://')||v.startsWith('https://')) return v;
  if(v.startsWith('/9j/')||v.startsWith('iVBOR')||v.startsWith('R0lGOD')) return `data:image/png;base64,${v}`;
  return v;
}

function setTab(which){
  const heat = which === 'heatmap';
  tabHeatmap.classList.toggle('active', heat);
  tabOriginal.classList.toggle('active', !heat);
  if(imgOriginal.src) imgOriginal.style.display = heat ? 'none':'block';
  if(imgHeatmap.src)  imgHeatmap.style.display  = heat ? 'block':'none';
  stagePh.style.display = (heat ? imgHeatmap.src : imgOriginal.src) ? 'none':'block';
}

function setPending(){
  badge.textContent = 'ANALYZING…'; badge.className = 'verdict-badge';
  confPct.textContent = '—'; confBar.style.width = '0%'; procTime.textContent = '—';
}

function renderResult(result){
  if(!result) return;
  const isAI = String(result.prediction||'').toLowerCase().includes('ai');
  badge.textContent = isAI ? 'AI-GENERATED' : 'HUMAN-MADE';
  badge.className = 'verdict-badge ' + (isAI ? 'ai':'real');
  const conf = Number(result.confidence ?? 0) * 100;
  confPct.textContent = `${conf.toFixed(0)}%`;
  confBar.style.width = `${Math.max(0,Math.min(100,conf))}%`;
  confBar.style.background = isAI ? 'var(--bad)' : 'var(--good)';
  procTime.textContent = `${Number(result.processing_time_ms ?? 0).toFixed(0)} ms`;
  const heat = result.heatmap_overlay || result.heatmap;
  if(heat) imgHeatmap.src = normalize(heat);
}

async function analyze(src){
  setPending();
  imgOriginal.src = normalize(src); imgOriginal.style.display='block';
  imgHeatmap.src=''; imgHeatmap.style.display='none'; stagePh.style.display='none';
  setTab('original'); stageSpin.classList.add('active');
  try{
    const resp = await fetch(`${BACKEND}/detect`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image: imgOriginal.src, generate_heatmap: true })
    });
    if(!resp.ok) throw new Error((await resp.text().catch(()=>'')) || `HTTP ${resp.status}`);
    const result = await resp.json();
    renderResult(result);
    chrome.storage.local.set({ lastImage: imgOriginal.src, lastResult: result });
  }catch(e){
    badge.textContent = 'ERROR'; badge.className = 'verdict-badge ai';
    procTime.textContent = e.message;
  }finally{
    stageSpin.classList.remove('active');
  }
}

function refreshGate(){
  const ok = agree.checked;
  [btnUpload, btnOverlay].forEach(b => { b.disabled = !ok; });
}

// ── interactions ──────────────────────────────────────────────
tabOriginal.addEventListener('click', () => setTab('original'));
tabHeatmap.addEventListener('click', () => setTab('heatmap'));
btnUpload.addEventListener('click', () => fileInput.click());
btnOverlay.addEventListener('click', () => chrome.runtime.sendMessage({ action:'open_overlay' }));
agree.addEventListener('change', refreshGate);
$('report-link').addEventListener('click', () => chrome.tabs?.create ? chrome.tabs.create({url:REPORT_URL}) : window.open(REPORT_URL,'_blank'));
btnExport.addEventListener('click', () => window.print());

fileInput.addEventListener('change', (e) => {
  const file = e.target.files?.[0];
  if(!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => analyze(ev.target.result);
  reader.readAsDataURL(file);
});

// drag-drop onto the stage
stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.style.outline='2px dashed var(--accent)'; });
stage.addEventListener('dragleave', () => stage.style.outline='none');
stage.addEventListener('drop', (e) => {
  e.preventDefault(); stage.style.outline='none';
  const file = e.dataTransfer?.files?.[0];
  if(file){ const r=new FileReader(); r.onload=(ev)=>analyze(ev.target.result); r.readAsDataURL(file); return; }
  const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
  if(url) analyze(url);
});

// ── restore last state + live sync with popup/overlay ─────────
refreshGate();
chrome.storage.local.get(['lastImage','lastResult'], ({lastImage,lastResult}) => {
  if(lastImage){ imgOriginal.src = normalize(lastImage); imgOriginal.style.display='block'; stagePh.style.display='none'; }
  if(lastResult) renderResult(lastResult);
  setTab('original');
});
chrome.storage.onChanged.addListener((changes, area) => {
  if(area!=='local') return;
  if(changes.lastImage?.newValue){ imgOriginal.src = normalize(changes.lastImage.newValue); imgOriginal.style.display='block'; stagePh.style.display='none'; setTab('original'); }
  if(changes.lastResult?.newValue) renderResult(changes.lastResult.newValue);
});

// ── feedback form ─────────────────────────────────────────────
const fbForm = $('fb-form'), fbStatus = $('fb-status');
fbForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const type = $('fb-type').value, email = $('fb-email').value, message = $('fb-msg').value.trim();
  if(!type || !message){ fbStatus.textContent='Please fill out the required fields.'; fbStatus.className='fb-status err'; return; }
  fbStatus.textContent='Submitting…'; fbStatus.className='fb-status';
  try{
    const r = await fetch(`${BACKEND}/feedback`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ type, email: email||null, message })
    });
    if(!r.ok) throw new Error(`HTTP ${r.status}`);
    fbStatus.textContent='✓ Thank you! Your feedback was submitted.'; fbStatus.className='fb-status ok';
    fbForm.reset();
    setTimeout(()=> fbStatus.textContent='', 5000);
  }catch(err){
    fbStatus.textContent='✗ Failed to submit feedback. Please try again.'; fbStatus.className='fb-status err';
  }
});
