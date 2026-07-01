// AI-YAN content script — floating pill (closed) + analysis panel (open)
const BACKEND_URL = 'http://localhost:5000';
const REPORT_URL = 'https://AI-YAN.example/report';

if (!chrome.runtime?.id) {
  console.warn('AI-YAN: extension context invalidated. Refresh the page.');
} else {
  const host = document.createElement('div');
  host.id = 'AI-YAN-overlay-root';
  document.documentElement.appendChild(host);
  const shadow = host.attachShadow({ mode: 'open' });

  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      * { box-sizing:border-box; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif; }

      /* ── Closed pill (Image 2) ── */
      #pill {
        position: fixed; right: 0; top: 140px; z-index: 2147483647;
        display: flex; flex-direction: column; align-items: center; gap: 10px;
        padding: 14px 8px; cursor: pointer;
        background: linear-gradient(180deg, #49a7d6 0%, #7a6df0 60%, #8b7bf2 100%);
        color: #fff; border-radius: 14px 0 0 14px;
        box-shadow: -6px 8px 22px rgba(0,0,0,.28);
      }
      #pill .logo { width: 20px; height: 20px; }
      #pill .label {
        writing-mode: vertical-rl; text-orientation: mixed;
        font-weight: 800; font-size: 14px; letter-spacing: .08em; transform: rotate(180deg);
      }
      #pill .x {
        width: 22px; height: 22px; border-radius: 50%; border:none; cursor:pointer;
        background: rgba(255,255,255,.22); color:#fff; font-size:12px; line-height:1;
      }

      /* ── Open panel (Image 1) ── */
      #panel {
        position: fixed; top: 16px; right: 16px; width: 340px; z-index: 2147483647;
        max-height: calc(100vh - 32px); overflow:auto; color:#fff; padding:14px;
        border-radius: 18px;
        background: linear-gradient(165deg, #8b7bf2 0%, #7a6df0 50%, #49a7d6 135%);
        box-shadow: 0 20px 50px rgba(0,0,0,.45);
        transform: translateX(calc(100% + 30px)); opacity:0; transition: transform .24s ease, opacity .24s ease;
      }
      #panel.open { transform: translateX(0); opacity:1; }

      .topbar { display:flex; align-items:center; justify-content:center; position:relative; margin-bottom:14px; }
      .close-x { position:absolute; left:0; top:50%; transform:translateY(-50%); width:24px; height:24px; border-radius:50%; border:none; cursor:pointer; background:rgba(255,255,255,.18); color:#fff; font-size:13px; }
      .brand { display:flex; align-items:center; gap:8px; font-weight:800; font-size:18px; }
      .brand svg { width:22px; height:22px; }
      .btn { width:100%; border:none; cursor:pointer; color:#fff; font-weight:700; font-size:14px; background:rgba(255,255,255,.16); border-radius:14px; padding:13px 14px; }
      .btn:hover { background:rgba(255,255,255,.26); }
      .dropzone { width:100%; margin-top:12px; border:2px dashed rgba(255,255,255,.45); border-radius:18px; background:rgba(255,255,255,.07); font-size:13px; font-weight:600; padding:20px 14px; text-align:center; cursor:pointer; }
      .dropzone.hover { background:rgba(255,255,255,.18); border-color:#fff; }
      .toggle { display:flex; gap:10px; margin-top:12px; }
      .seg { flex:1; border:none; cursor:pointer; color:#fff; font-weight:700; font-size:13px; border-radius:14px; padding:11px; background:rgba(255,255,255,.10); }
      .seg.active { background:rgba(255,255,255,.30); }
      .stage { margin-top:12px; border-radius:16px; background:rgba(20,28,48,.32); min-height:150px; display:flex; align-items:center; justify-content:center; overflow:hidden; position:relative; }
      .stage img { max-width:100%; max-height:230px; object-fit:contain; display:none; }
      .ph svg { width:58px; height:58px; opacity:.7; }
      .spinner { position:absolute; width:30px; height:30px; border-radius:50%; border:3px solid rgba(255,255,255,.3); border-top-color:#fff; animation:spin 1s linear infinite; display:none; }
      .spinner.active { display:block; }
      @keyframes spin { to { transform:rotate(360deg); } }
      .verdict { display:none; margin-top:10px; text-align:center; font-weight:800; font-size:14px; padding:9px; border-radius:12px; }
      .verdict.show { display:block; }
      .verdict.real { background:linear-gradient(135deg,#10b981,#34d399); }
      .verdict.ai { background:linear-gradient(135deg,#ef4444,#f97316); }
      .footer { margin-top:14px; text-align:center; }
      .footer .l1 { font-size:12.5px; }
      .footer .l1 a { color:#bfe3ff; font-weight:700; text-decoration:underline; cursor:pointer; }
      .footer .l2 { font-size:11.5px; font-style:italic; opacity:.85; margin-top:6px; }
    </style>

    <div id="pill" title="Open AI-YAN">
      <svg class="logo" viewBox="0 0 24 24" fill="none"><path d="M12 3 L21 20 H3 Z" stroke="#fff" stroke-width="2" stroke-linejoin="round"/><path d="M9 20 l3-6 3 6" stroke="#fff" stroke-width="2" stroke-linejoin="round"/></svg>
      <div class="label">AI-YAN</div>
      <button class="x" id="pill-x" title="Hide">✕</button>
    </div>

    <section id="panel">
      <div class="topbar">
        <button class="close-x" id="panel-close" title="Collapse">✕</button>
        <div class="brand">
          <svg viewBox="0 0 24 24" fill="none"><path d="M12 3 L21 20 H3 Z" stroke="#fff" stroke-width="2" stroke-linejoin="round"/><path d="M9 20 l3-6 3 6" stroke="#fff" stroke-width="2" stroke-linejoin="round"/></svg>
          AI-YAN
        </div>
      </div>
      <button class="btn" id="go-dashboard">Go to Dashboard</button>
      <div class="dropzone" id="dropzone">Drop an image here ...</div>
      <div class="toggle">
        <button class="seg active" id="tab-original">Original</button>
        <button class="seg" id="tab-heatmap">Heatmap</button>
      </div>
      <div class="stage" id="stage">
        <div class="ph" id="placeholder"><svg viewBox="0 0 24 24" fill="#fff"><path d="M21 19V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2zM8.5 11l2.5 3 3.5-4.5L19 17H5z"/></svg></div>
        <div class="spinner" id="spinner"></div>
        <img id="img-original" alt="Original" />
        <img id="img-heatmap" alt="Heatmap" />
      </div>
      <div class="verdict" id="verdict"></div>
      <div class="footer">
        <div class="l1">System had a wrong prediction? <a id="report">Report it here.</a></div>
        <div class="l2">Right-click an image on a page to analyze it here.</div>
      </div>
    </section>
  `;

  const $ = (id) => shadow.getElementById(id);
  const pill = $('pill'), panel = $('panel');
  const stage = $('stage'), placeholder = $('placeholder'), spinner = $('spinner');
  const imgOriginal = $('img-original'), imgHeatmap = $('img-heatmap');
  const tabOriginal = $('tab-original'), tabHeatmap = $('tab-heatmap');
  const verdict = $('verdict'), dropzone = $('dropzone');

  const normalize = (v) => {
    if(!v) return '';
    if(v.startsWith('data:')||v.startsWith('http://')||v.startsWith('https://')) return v;
    if(v.startsWith('/9j/')||v.startsWith('iVBOR')||v.startsWith('R0lGOD')) return `data:image/png;base64,${v}`;
    return v;
  };

  function openPanel(){ host.style.display='block'; pill.style.display='none'; panel.classList.add('open'); }
  function collapse(){ panel.classList.remove('open'); pill.style.display='flex'; }
  function hideAll(){ panel.classList.remove('open'); host.style.display='none'; }

  function setTab(which){
    const heat = which === 'heatmap';
    tabHeatmap.classList.toggle('active', heat);
    tabOriginal.classList.toggle('active', !heat);
    if(imgOriginal.src) imgOriginal.style.display = heat ? 'none':'block';
    if(imgHeatmap.src)  imgHeatmap.style.display  = heat ? 'block':'none';
    placeholder.style.display = (heat ? imgHeatmap.src : imgOriginal.src) ? 'none':'block';
  }

  function render(result){
    if(!result) return;
    const isAI = String(result.prediction||'').toLowerCase().includes('ai');
    verdict.className = 'verdict show ' + (isAI?'ai':'real');
    verdict.textContent = `${isAI?'AI-GENERATED':'HUMAN-MADE'} · ${(Number(result.confidence??0)*100).toFixed(1)}%`;
    const heat = result.heatmap_overlay || result.heatmap;
    if(heat) imgHeatmap.src = normalize(heat);
  }

  async function toDataUrl(src){
    if(!src) throw new Error('no image');
    if(src.startsWith('data:')) return src;
    const r = await fetch(src, { mode:'cors', credentials:'omit' });
    if(!r.ok) throw new Error(`fetch ${r.status}`);
    const blob = await r.blob();
    return await new Promise((res,rej)=>{ const fr=new FileReader(); fr.onloadend=()=>res(fr.result); fr.onerror=rej; fr.readAsDataURL(blob); });
  }

  async function analyze(src){
    openPanel();
    verdict.classList.remove('show');
    imgOriginal.src = normalize(src); imgOriginal.style.display='block';
    imgHeatmap.src=''; imgHeatmap.style.display='none'; placeholder.style.display='none';
    setTab('original'); spinner.classList.add('active');
    try{
      let payload = imgOriginal.src;
      if(!payload.startsWith('data:')){ try{ payload = await toDataUrl(payload); }catch{} }
      const resp = await fetch(`${BACKEND_URL}/detect`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ image: payload, generate_heatmap: true })
      });
      if(!resp.ok) throw new Error((await resp.text().catch(()=>'')) || `HTTP ${resp.status}`);
      const result = await resp.json();
      render(result);
      chrome.storage.local.set({ lastImage: payload, lastResult: result });
    }catch(e){ verdict.className='verdict show ai'; verdict.textContent = `Error: ${e.message}`; }
    finally{ spinner.classList.remove('active'); }
  }

  // interactions
  pill.addEventListener('click', (e)=>{ if(e.target.id!=='pill-x') openPanel(); });
  $('pill-x').addEventListener('click', (e)=>{ e.stopPropagation(); hideAll(); });
  $('panel-close').addEventListener('click', collapse);
  $('go-dashboard').addEventListener('click', ()=> chrome.runtime.sendMessage({action:'open_dashboard'}));
  $('report').addEventListener('click', ()=> window.open(REPORT_URL,'_blank'));
  tabOriginal.addEventListener('click', ()=> setTab('original'));
  tabHeatmap.addEventListener('click', ()=> setTab('heatmap'));

  dropzone.addEventListener('dragover', (e)=>{ e.preventDefault(); dropzone.classList.add('hover'); });
  dropzone.addEventListener('dragleave', ()=> dropzone.classList.remove('hover'));
  dropzone.addEventListener('drop', async (e)=>{
    e.preventDefault(); dropzone.classList.remove('hover');
    const file = e.dataTransfer?.files?.[0];
    if(file){ const fr=new FileReader(); fr.onload=(ev)=>analyze(ev.target.result); fr.readAsDataURL(file); return; }
    const url = e.dataTransfer.getData('text/uri-list') || e.dataTransfer.getData('text/plain');
    if(url) analyze(url);
  });

  // messages from background / popup
  chrome.runtime.onMessage.addListener((msg)=>{
    if(msg.action==='OPEN_OVERLAY' || msg.action==='WAKE_UP'){
      openPanel();
      chrome.storage.local.get(['lastImage','lastResult'], ({lastImage,lastResult})=>{
        if(lastImage){ imgOriginal.src=normalize(lastImage); imgOriginal.style.display='block'; placeholder.style.display='none'; }
        if(lastResult) render(lastResult);
        setTab('original');
      });
    }
    if(msg.action==='OPEN_OVERLAY_AND_ANALYZE' && msg.imageUrl){ analyze(msg.imageUrl); }
  });

  chrome.storage.onChanged.addListener((changes, area)=>{
    if(area!=='local') return;
    if(changes.lastImage?.newValue){ imgOriginal.src=normalize(changes.lastImage.newValue); imgOriginal.style.display='block'; placeholder.style.display='none'; }
    if(changes.lastResult?.newValue) render(changes.lastResult.newValue);
  });

  // start collapsed (pill visible)
  host.style.display='block';
  collapse();
  console.log('AI-YAN overlay ready — look for the pill on the right edge.');
}
