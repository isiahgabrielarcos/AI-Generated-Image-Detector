const BACKEND = 'http://localhost:5000';

const container = document.getElementById('image-container');
const fileInput = document.getElementById('debug-file-input');
const uploadBtn = document.getElementById('manual-upload-btn');
const openOverlayBtn = document.getElementById('open-overlay-btn');
const predLabel = document.getElementById('prediction-label');
const confidencePct = document.getElementById('confidence-pct');
const progressFill = document.getElementById('progress-fill');
const probabilityValue = document.getElementById('probability-value');
const realProbabilityValue = document.getElementById('real-probability-value');
const modelName = document.getElementById('model-name');
const processingTime = document.getElementById('processing-time');
const exportReportBtn = document.getElementById('export-report');

let currentImage = null;
let currentResult = null;

function renderImage(src) {
  container.innerHTML = '';
  const img = document.createElement('img');
  img.src = src;
  img.alt = 'Source image';
  img.style.cssText = 'max-width:100%;display:block';
  img.onerror = () => {
    container.innerHTML = '<p style="color:#f87171;font-size:12px">Image failed to load</p>';
  };
  container.appendChild(img);
  return img;
}

function setPendingState(message = '—') {
  predLabel.textContent = message;
  predLabel.style.color = '#38bdf8';
  confidencePct.textContent = '—';
  progressFill.style.width = '0%';
  probabilityValue.textContent = '—';
  realProbabilityValue.textContent = '—';
  processingTime.textContent = '—';
}

function renderResults(result) {
  currentResult = result;
  const prediction = result?.prediction || 'Unknown';
  const isAI = prediction.toLowerCase().includes('ai');

  predLabel.textContent = prediction;
  predLabel.style.color = isAI ? '#f44336' : '#4caf50';

  const confidence = Number(result.confidence ?? 0);
  const aiProbability = Number(result.probability_ai ?? 0);
  const realProbability = Number(result.probability_real ?? (1 - aiProbability));
  const timeMs = Number(result.processing_time_ms ?? 0);

  confidencePct.textContent = `${(confidence * 100).toFixed(1)}%`;
  progressFill.style.width = `${Math.max(0, Math.min(100, confidence * 100))}%`;
  progressFill.style.background = isAI ? '#f44336' : '#4caf50';
  probabilityValue.textContent = `${(aiProbability * 100).toFixed(2)}%`;
  realProbabilityValue.textContent = `${(realProbability * 100).toFixed(2)}%`;
  processingTime.textContent = `${timeMs.toFixed(1)} ms`;
  modelName.textContent = 'Hybrid Detector';
}

async function analyze(imageSource) {
  currentImage = imageSource;
  setPendingState('Analyzing…');
  renderImage(imageSource);

  const body = { image: imageSource, generate_heatmap: true };

  const resp = await fetch(`${BACKEND}/detect`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body)
  });

  if (!resp.ok) {
    const text = await resp.text().catch(() => '');
    throw new Error(text || `HTTP ${resp.status}`);
  }

  const result = await resp.json();
  renderResults(result);
  chrome.storage.local.set({ lastImage: imageSource, lastResult: result });
}

function openOverlay() {
  chrome.runtime.sendMessage({ action: 'open_overlay' });
}

uploadBtn.addEventListener('click', () => fileInput.click());
openOverlayBtn.addEventListener('click', openOverlay);

fileInput.addEventListener('change', async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = async (event) => {
    try {
      await analyze(event.target.result);
      openOverlay();
    } catch (err) {
      setPendingState('—');
      alert(`Failed to analyze image: ${err.message}`);
    }
  };
  reader.readAsDataURL(file);
});

chrome.storage.local.get(['lastImage', 'lastResult'], ({ lastImage, lastResult }) => {
  if (lastImage) {
    currentImage = lastImage;
    renderImage(lastImage);
  } else {
    container.innerHTML = '<p style="color:#94a3b8;font-size:12px">No image stored yet.</p>';
  }
  if (lastResult) {
    currentResult = lastResult;
    renderResults(lastResult);
  } else {
    setPendingState();
  }
});

chrome.storage.onChanged.addListener((changes) => {
  if (changes.lastImage?.newValue) {
    currentImage = changes.lastImage.newValue;
    renderImage(currentImage);
  }
  if (changes.lastResult?.newValue) {
    currentResult = changes.lastResult.newValue;
    renderResults(currentResult);
  }
});

exportReportBtn.addEventListener('click', () => window.print());
