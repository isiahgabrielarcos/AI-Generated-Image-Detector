// Content script for AI Image Detector Extension
// Floating overlay with original image / heatmap / results

const BACKEND_URL = 'http://localhost:5000';

if (!chrome.runtime?.id) {
  console.warn('Extension context invalidated. Please refresh the page.');
} else {
  const host = document.createElement('div');
  host.id = 'ai-detector-overlay-root';
  document.documentElement.appendChild(host);

  const shadow = host.attachShadow({ mode: 'open' });

  shadow.innerHTML = `
    <style>
      :host { all: initial; }
      #overlay-root {
        position: fixed;
        inset: 0;
        z-index: 2147483647;
        pointer-events: none;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }

      #panel {
        pointer-events: auto;
        position: fixed;
        top: 18px;
        right: 18px;
        width: 360px;
        max-width: calc(100vw - 36px);
        max-height: calc(100vh - 36px);
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 18px;
        box-shadow: 0 20px 50px rgba(0,0,0,0.45);
        overflow: hidden;
        transform: translateX(calc(100% + 24px));
        transition: transform 240ms ease, opacity 240ms ease;
        opacity: 0;
        display: flex;
        flex-direction: column;
      }

      #panel.open {
        transform: translateX(0);
        opacity: 1;
      }

      #dock {
        pointer-events: auto;
        position: fixed;
        right: 18px;
        top: 130px;
        width: 44px;
        height: 140px;
        border: none;
        border-radius: 14px 0 0 14px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 12px 25px rgba(0,0,0,0.28);
        writing-mode: vertical-rl;
        text-orientation: mixed;
        letter-spacing: 0.12em;
      }

      .panel-header {
        padding: 16px 16px 12px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }

      .title-wrap { display: flex; flex-direction: column; gap: 2px; }
      .title { font-size: 16px; font-weight: 800; }
      .subtitle { font-size: 12px; opacity: 0.88; }

      .icon-btn {
        border: none;
        background: rgba(255,255,255,0.15);
        color: white;
        border-radius: 10px;
        width: 34px;
        height: 34px;
        cursor: pointer;
      }

      .panel-body {
        padding: 0 16px 16px;
        overflow: auto;
      }

      .toolbar {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
      }

      .toolbar button, .toolbar a {
        flex: 1;
        border: none;
        border-radius: 10px;
        padding: 10px 12px;
        cursor: pointer;
        font-weight: 700;
        font-size: 12px;
        text-align: center;
        text-decoration: none;
        color: white;
        background: rgba(255,255,255,0.12);
      }

      .drop-zone {
        border: 2px dashed rgba(255,255,255,0.45);
        border-radius: 14px;
        padding: 14px;
        text-align: center;
        font-size: 12px;
        margin-bottom: 12px;
        background: rgba(255,255,255,0.06);
        transition: 180ms ease;
      }

      .drop-zone.hover {
        border-color: white;
        background: rgba(255,255,255,0.14);
      }

      .tabs {
        display: flex;
        gap: 8px;
        margin-bottom: 10px;
      }

      .tab {
        flex: 1;
        border: none;
        border-radius: 10px;
        padding: 9px 10px;
        cursor: pointer;
        font-weight: 700;
        font-size: 12px;
        background: rgba(255,255,255,0.12);
        color: white;
      }

      .tab.active { background: rgba(255,255,255,0.30); }

      .preview-wrap {
        border-radius: 14px;
        overflow: hidden;
        background: rgba(0,0,0,0.28);
        min-height: 180px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 12px;
      }

      .preview-wrap img {
        display: none;
        max-width: 100%;
        max-height: 240px;
        object-fit: contain;
      }

      .loading {
        border: 3px solid rgba(255,255,255,0.28);
        border-top: 3px solid white;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        margin: 18px auto;
        animation: spin 1s linear infinite;
        display: none;
      }

      .loading.active { display: block; }

      .prediction {
        display: none;
        padding: 14px;
        border-radius: 14px;
        background: rgba(255,255,255,0.10);
        text-align: center;
        margin-bottom: 12px;
      }

      .prediction.active { display: block; }

      .badge {
        display: inline-block;
        padding: 10px 16px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.26);
      }

      .badge.real { background: linear-gradient(135deg, #11998e, #38ef7d); }
      .badge.ai { background: linear-gradient(135deg, #eb3349, #f45c43); }

      .metric {
        display: none;
        background: rgba(255,255,255,0.10);
        padding: 10px 12px;
        border-radius: 12px;
        margin-bottom: 10px;
      }

      .metric.active { display: block; }

      .metric label {
        display: block;
        font-size: 11px;
        opacity: 0.85;
        margin-bottom: 4px;
      }

      .metric .value {
        font-size: 18px;
        font-weight: 800;
      }

      .hint {
        font-size: 11px;
        opacity: 0.9;
        text-align: center;
        margin-top: 8px;
      }

      @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
      }

      @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-6px); }
        to { opacity: 1; transform: translateY(0); }
      }

      @keyframes fadeOut {
        from { opacity: 1; transform: translateY(0); }
        to { opacity: 0; transform: translateY(-6px); }
      }

      .toast {
        position: fixed;
        left: 16px;
        bottom: 16px;
        padding: 12px 14px;
        border-radius: 12px;
        background: rgba(20, 20, 20, 0.92);
        color: white;
        font-size: 13px;
        box-shadow: 0 12px 28px rgba(0,0,0,0.35);
        z-index: 2147483646;
      }
    </style>

    <div id="overlay-root">
      <button id="dock">AI DETECTOR</button>

      <section id="panel" aria-label="AI Detector Overlay">
        <div class="panel-header">
          <div class="title-wrap">
            <div class="title">🔍 AI Image Scanner</div>
            <div class="subtitle">Overlay mode with popup + dashboard flow</div>
          </div>
          <button class="icon-btn" id="close-btn" title="Close">✕</button>
        </div>

        <div class="panel-body">
          <div class="toolbar">
            <button id="home-btn" type="button">Home</button>
            <button id="open-dashboard-btn" type="button">Dashboard</button>
          </div>

          <div class="drop-zone" id="drop-zone">Drop an image here, or use the popup to upload</div>

          <div class="tabs">
            <button class="tab active" id="tab-original" type="button">Original</button>
            <button class="tab" id="tab-heatmap" type="button">Heatmap</button>
          </div>

          <div class="preview-wrap" id="preview-container">
            <img id="preview-img" alt="Original preview">
          </div>

          <div class="preview-wrap" id="heatmap-container" style="display:none">
            <img id="heatmap-img" alt="Heatmap preview">
          </div>

          <div class="loading" id="loading-spinner"></div>

          <div class="prediction" id="prediction-display"></div>

          <div class="metric" id="confidence-metric">
            <label>Confidence</label>
            <div class="value" id="confidence-value">—</div>
          </div>

          <div class="metric" id="probability-metric">
            <label>AI Probability</label>
            <div class="value" id="probability-value">—</div>
          </div>

          <div class="metric" id="real-metric">
            <label>Real Probability</label>
            <div class="value" id="real-value">—</div>
          </div>

          <div class="metric" id="time-metric">
            <label>Processing Time</label>
            <div class="value" id="time-value">—</div>
          </div>

          <div class="hint">Right-click an image on the page to analyze it here.</div>
        </div>
      </section>
    </div>
  `;

  const panel = shadow.getElementById('panel');
  const dock = shadow.getElementById('dock');
  const closeBtn = shadow.getElementById('close-btn');
  const homeBtn = shadow.getElementById('home-btn');
  const openDashboardBtn = shadow.getElementById('open-dashboard-btn');
  const dropZone = shadow.getElementById('drop-zone');
  const tabOriginal = shadow.getElementById('tab-original');
  const tabHeatmap = shadow.getElementById('tab-heatmap');
  const previewContainer = shadow.getElementById('preview-container');
  const heatmapContainer = shadow.getElementById('heatmap-container');
  const previewImg = shadow.getElementById('preview-img');
  const heatmapImg = shadow.getElementById('heatmap-img');
  const loadingSpinner = shadow.getElementById('loading-spinner');
  const predictionDisplay = shadow.getElementById('prediction-display');
  const confidenceMetric = shadow.getElementById('confidence-metric');
  const probabilityMetric = shadow.getElementById('probability-metric');
  const realMetric = shadow.getElementById('real-metric');
  const timeMetric = shadow.getElementById('time-metric');
  const confidenceValue = shadow.getElementById('confidence-value');
  const probabilityValue = shadow.getElementById('probability-value');
  const realValue = shadow.getElementById('real-value');
  const timeValue = shadow.getElementById('time-value');

  let currentImage = '';
  let lastResult = null;
  let isDragging = false;
  let dragOffsetY = 0;

  function openPanel() {
    panel.classList.add('open');
    dock.innerText = 'CLOSE';
  }

  function closePanel() {
    panel.classList.remove('open');
    dock.innerText = 'AI DETECTOR';
  }

  function ensurePanelVisible() {
    host.style.display = 'block';
    openPanel();
  }

  function resetMetrics() {
    predictionDisplay.classList.remove('active');
    confidenceMetric.classList.remove('active');
    probabilityMetric.classList.remove('active');
    realMetric.classList.remove('active');
    timeMetric.classList.remove('active');
  }

  function setPreview(imageUrl) {
    currentImage = imageUrl;
    previewImg.src = imageUrl;
    previewImg.style.display = 'block';
    previewContainer.style.display = 'flex';
    heatmapContainer.style.display = 'none';
    tabOriginal.classList.add('active');
    tabHeatmap.classList.remove('active');
  }

  function setHeatmap(imageUrl) {
    heatmapImg.src = `data:image/png;base64,${imageUrl}`;
  }

  function toggleTabs(mode) {
    if (mode === 'heatmap') {
      tabHeatmap.classList.add('active');
      tabOriginal.classList.remove('active');
      previewContainer.style.display = 'none';
      heatmapContainer.style.display = 'flex';
    } else {
      tabOriginal.classList.add('active');
      tabHeatmap.classList.remove('active');
      previewContainer.style.display = 'flex';
      heatmapContainer.style.display = 'none';
    }
  }

  tabOriginal.addEventListener('click', () => toggleTabs('original'));
  tabHeatmap.addEventListener('click', () => toggleTabs('heatmap'));

  dock.addEventListener('click', () => {
    if (panel.classList.contains('open')) closePanel();
    else ensurePanelVisible();
  });

  closeBtn.addEventListener('click', closePanel);
  homeBtn.addEventListener('click', () => chrome.runtime.sendMessage({ action: 'open_dashboard' }));
  openDashboardBtn.addEventListener('click', () => chrome.runtime.sendMessage({ action: 'open_dashboard' }));

  dock.addEventListener('mousedown', (e) => {
    isDragging = true;
    dragOffsetY = e.clientY - dock.getBoundingClientRect().top;
    dock.style.transition = 'none';
    panel.style.transition = 'none';
    e.preventDefault();
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const newTop = Math.max(16, Math.min(e.clientY - dragOffsetY, window.innerHeight - 160));
    dock.style.top = `${newTop}px`;
    panel.style.top = `${Math.max(16, Math.min(newTop - 112, window.innerHeight - 40 - panel.offsetHeight))}px`;
  });

  window.addEventListener('mouseup', () => {
    if (!isDragging) return;
    isDragging = false;
    dock.style.transition = '';
    panel.style.transition = '';
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('hover');
  });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('hover'));
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('hover');

    let url = e.dataTransfer.getData('text/uri-list');

    // fallback for images
    if (!url && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      const reader = new FileReader();
      reader.onload = (event) => {
        ensurePanelVisible();
        analyzeImage(event.target.result);
      };
      reader.readAsDataURL(file);
      return;
    }

    if (url) {
      ensurePanelVisible();
      analyzeImage(url);
    } else {
      showToast('Could not read image. Try right-click → Analyze instead.');
    }
  });

  async function urlToBase64(url) {
    try {
      const response = await fetch(url, { mode: 'cors' });
      const blob = await response.blob();
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    } catch (error) {
      console.error('Error converting URL to base64:', error);
      throw error;
    }
  }

  async function analyzeImage(imageUrl) {
    ensurePanelVisible();
    resetMetrics();
    loadingSpinner.classList.add('active');
    setPreview(imageUrl);

    try {
      // 🔥 FIX: convert URL → base64 if needed
      let base64Image = imageUrl;
      if (!imageUrl.startsWith('data:')) {
        base64Image = await urlToBase64(imageUrl);
      }

      const response = await fetch(`${BACKEND_URL}/detect`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64Image, generate_heatmap: true })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      lastResult = result;
      loadingSpinner.classList.remove('active');
      displayResults(result);

      chrome.storage.local.set({ lastImage: base64Image, lastResult: result });

    } catch (error) {
      console.error('Error:', error);
      loadingSpinner.classList.remove('active');
      showToast(`Failed to analyze image: ${error.message}`);
    }
  }

  function displayResults(result) {
    const isAI = (result.prediction || '').toLowerCase().includes('ai');
    const badgeClass = isAI ? 'ai' : 'real';
    const badgeEmoji = isAI ? '🤖' : '✅';

    predictionDisplay.innerHTML = `<div class="badge ${badgeClass}">${badgeEmoji} ${result.prediction || 'Unknown'}</div>`;
    predictionDisplay.classList.add('active');

    const confidence = Number(result.confidence ?? 0);
    const aiProbability = Number(result.probability_ai ?? (isAI ? confidence : 1 - confidence));
    const realProbability = Number(result.probability_real ?? (1 - aiProbability));
    const processingMs = Number(result.processing_time_ms ?? 0);

    confidenceValue.textContent = `${(confidence * 100).toFixed(2)}%`;
    probabilityValue.textContent = `${(aiProbability * 100).toFixed(2)}%`;
    realValue.textContent = `${(realProbability * 100).toFixed(2)}%`;
    timeValue.textContent = `${processingMs.toFixed(1)} ms`;

    confidenceMetric.classList.add('active');
    probabilityMetric.classList.add('active');
    realMetric.classList.add('active');
    timeMetric.classList.add('active');

    if (result.heatmap) {
      setHeatmap(result.heatmap);
    }
  }

  function showToast(message) {
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    host.appendChild(toast);
    requestAnimationFrame(() => toast.style.animation = 'fadeIn 180ms ease');
    setTimeout(() => {
      toast.style.animation = 'fadeOut 180ms ease';
      setTimeout(() => toast.remove(), 200);
    }, 2400);
  }

  function loadStoredState() {
    chrome.storage.local.get(['lastImage', 'lastResult'], ({ lastImage, lastResult: storedResult }) => {
      if (lastImage) {
        setPreview(lastImage);
      }
      if (storedResult) {
        displayResults(storedResult);
      }
    });
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.action === 'WAKE_UP' || msg.action === 'OPEN_OVERLAY') {
      ensurePanelVisible();
      loadStoredState();
    }

    if (msg.action === 'OPEN_OVERLAY_AND_ANALYZE' && msg.imageUrl) {
      ensurePanelVisible();
      analyzeImage(msg.imageUrl);
    }

    if (msg.action === 'ANALYZE_STORED_IMAGE') {
      chrome.storage.local.get(['lastImage'], ({ lastImage }) => {
        if (lastImage) analyzeImage(lastImage);
      });
    }
  });

  chrome.storage.onChanged.addListener((changes) => {
    if (changes.lastImage?.newValue) {
      setPreview(changes.lastImage.newValue);
    }
    if (changes.lastResult?.newValue) {
      displayResults(changes.lastResult.newValue);
    }
  });

  loadStoredState();
}
