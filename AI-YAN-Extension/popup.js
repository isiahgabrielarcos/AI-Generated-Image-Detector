const BACKEND_URL = 'http://localhost:5000';
const REPORT_URL = 'https://AI-YAN.example/report'; // change to your report form

const $ = (id) => document.getElementById(id);
const dropzone = $('dropzone');
const fileInput = $('file-input');
const stage = $('stage');
const placeholder = $('placeholder');
const spinner = $('spinner');
const imgOriginal = $('img-original');
const imgHeatmap = $('img-heatmap');
const tabOriginal = $('tab-original');
const tabHeatmap = $('tab-heatmap');
const verdict = $('verdict');
const err = $('err');

function normalizeSrc(v){
  if(!v) return '';
  if(v.startsWith('data:')||v.startsWith('http://')||v.startsWith('https://')) return v;
  if(v.startsWith('/9j/')||v.startsWith('iVBOR')||v.startsWith('R0lGOD')) return `data:image/png;base64,${v}`;
  return v;
}
function showError(m){ err.textContent = m; err.classList.add('show'); }
function hideError(){ err.classList.remove('show'); }

function setActiveTab(which){
  const heat = which === 'heatmap';
  tabHeatmap.classList.toggle('active', heat);
  tabOriginal.classList.toggle('active', !heat);
  if (imgOriginal.src) imgOriginal.style.display = heat ? 'none' : 'block';
  if (imgHeatmap.src)  imgHeatmap.style.display  = heat ? 'block' : 'none';
  // keep placeholder only if nothing loaded for that view
  placeholder.style.display = (heat ? imgHeatmap.src : imgOriginal.src) ? 'none' : 'block';
}

function renderResult(result){
  if(!result) return;
  const isAI = String(result.prediction || '').toLowerCase().includes('ai');
  verdict.className = 'verdict show ' + (isAI ? 'ai' : 'real');
  const conf = Number(result.confidence ?? 0) * 100;
  verdict.textContent = `${isAI ? 'AI-GENERATED' : 'HUMAN-MADE'} · ${conf.toFixed(1)}%`;
  const heat = result.heatmap_overlay || result.heatmap;
  if(heat){ imgHeatmap.src = normalizeSrc(heat); }
}

async function analyze(dataUrl){
  hideError();
  verdict.classList.remove('show');
  imgOriginal.src = normalizeSrc(dataUrl);
  imgOriginal.style.display = 'block';
  imgHeatmap.src = '';
  imgHeatmap.style.display = 'none';
  placeholder.style.display = 'none';
  setActiveTab('original');
  spinner.classList.add('active');
  try{
    const resp = await fetch(`${BACKEND_URL}/detect`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ image: imgOriginal.src, generate_heatmap: true })
    });
    if(!resp.ok){ throw new Error((await resp.text().catch(()=>'')) || `HTTP ${resp.status}`); }
    const result = await resp.json();
    renderResult(result);
    chrome.storage.local.set({ lastImage: imgOriginal.src, lastResult: result });
    chrome.runtime.sendMessage({ action: 'open_overlay' }); // sync the on-page overlay
  }catch(e){
    showError(`Failed to analyze image: ${e.message}`);
  }finally{
    spinner.classList.remove('active');
  }
}

function handleFile(file){
  if(!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => analyze(ev.target.result);
  reader.readAsDataURL(file);
}

// ── interactions ──────────────────────────────────────────────
dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', (e) => handleFile(e.target.files?.[0]));

['dragover','dragenter'].forEach(ev => dropzone.addEventListener(ev, (e)=>{ e.preventDefault(); dropzone.classList.add('hover'); }));
['dragleave','dragend'].forEach(ev => dropzone.addEventListener(ev, ()=> dropzone.classList.remove('hover')));
dropzone.addEventListener('drop', (e) => {
  e.preventDefault(); dropzone.classList.remove('hover');
  const file = e.dataTransfer?.files?.[0];
  if(file){ handleFile(file); return; }
  const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
  if(url) analyze(url);
});

tabOriginal.addEventListener('click', () => setActiveTab('original'));
tabHeatmap.addEventListener('click', () => setActiveTab('heatmap'));

$('dashboard-btn').addEventListener('click', () => chrome.runtime.sendMessage({ action:'open_dashboard' }));
$('close-btn').addEventListener('click', () => window.close());
$('report-link').addEventListener('click', () => chrome.tabs.create({ url: REPORT_URL }));

// ── health + restore last state ───────────────────────────────
(async function health(){
  try{
    const r = await fetch(`${BACKEND_URL}/health`, { signal: AbortSignal.timeout(3000) });
    if(!r.ok) throw 0;
  }catch{ showError('⚠️ Backend not reachable. Start the detection server.'); }
})();

chrome.storage.local.get(['lastImage','lastResult'], ({lastImage,lastResult}) => {
  if(lastImage){ imgOriginal.src = normalizeSrc(lastImage); imgOriginal.style.display='block'; placeholder.style.display='none'; }
  if(lastResult){ renderResult(lastResult); setActiveTab('original'); }
});
