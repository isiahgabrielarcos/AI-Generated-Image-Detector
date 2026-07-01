const BACKEND_URL = 'http://localhost:5000';

const fileUpload = document.getElementById('file-upload');
const loading = document.getElementById('loading');
const previewSection = document.getElementById('preview-section');
const resultsSection = document.getElementById('results-section');
const imagePreview = document.getElementById('image-preview');
const heatmapPreview = document.getElementById('heatmap-preview');
const errorDisplay = document.getElementById('error-display');
const tabOriginal = document.getElementById('tab-original');
const tabHeatmap = document.getElementById('tab-heatmap');
const analyzeAnother = document.getElementById('analyze-another');
const openOverlayBtn = document.getElementById('open-overlay');
const openDashboardBtn = document.getElementById('open-dashboard');

let currentResult = null;

function showError(message) {
  errorDisplay.textContent = message;
  errorDisplay.classList.add('active');
}

function hideError() {
  errorDisplay.classList.remove('active');
}

function displayResults(result) {
  if (!result) return;

  resultsSection.classList.add('active');
  previewSection.classList.add('active');

  const isAI = result.prediction === "AI-Generated";
  const badgeClass = isAI ? 'ai' : 'real';
  const badgeEmoji = isAI ? '🤖' : '✅';

  document.getElementById('prediction-display').innerHTML = `
    <div class="badge ${badgeClass}">
      ${badgeEmoji} ${result.prediction}
    </div>
  `;

  document.getElementById('confidence-value').textContent =
    `${(result.confidence * 100).toFixed(2)}%`;

  document.getElementById('ai-prob-value').textContent =
    `${(result.probability_ai * 100).toFixed(2)}%`;

  document.getElementById('real-prob-value').textContent =
    `${(result.probability_real * 100).toFixed(2)}%`;

  document.getElementById('time-value').textContent =
    `${result.processing_time_ms.toFixed(1)} ms`;

  // 🔥 Fix heatmap handling
  if (result.heatmap) {
    heatmapPreview.src = `data:image/png;base64,${result.heatmap}`;
    heatmapPreview.style.display = 'none'; // default
  }
}

tabOriginal.addEventListener('click', () => {
  tabOriginal.classList.add('active');
  tabHeatmap.classList.remove('active');
  imagePreview.style.display = 'block';
  heatmapPreview.style.display = 'none';
});

tabHeatmap.addEventListener('click', () => {
  tabHeatmap.classList.add('active');
  tabOriginal.classList.remove('active');
  imagePreview.style.display = 'none';
  heatmapPreview.style.display = 'block';
});

fileUpload.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;

  hideError();

  const reader = new FileReader();
  reader.onload = (event) => {
    const base64Image = event.target.result;

    // 🔥 Send to overlay for processing (launcher only)
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, {
          action: 'OPEN_OVERLAY_AND_ANALYZE',
          imageUrl: base64Image
        });
      }
    });

    // Show feedback
    previewSection.classList.add('active');
    imagePreview.src = base64Image;
    imagePreview.style.display = 'block';
    showError('📤 Opening overlay for analysis...');
  };
  reader.readAsDataURL(file);
});

openOverlayBtn.addEventListener('click', () => chrome.runtime.sendMessage({ action: 'open_overlay' }));
openDashboardBtn.addEventListener('click', () => chrome.runtime.sendMessage({ action: 'open_dashboard' }));

analyzeAnother.addEventListener('click', () => {
  fileUpload.value = '';
  previewSection.classList.add('active');
  resultsSection.classList.remove('active');
  hideError();
});

async function checkBackendHealth() {
  try {
    const response = await fetch(`${BACKEND_URL}/health`, {
      method: 'GET',
      signal: AbortSignal.timeout(3000)
    });

    if (!response.ok) throw new Error('Backend unhealthy');
  } catch (error) {
    showError('⚠️ Backend server not reachable. Please start the server.');
  }
}

// Load last analysis when popup opens
document.addEventListener('DOMContentLoaded', () => {
  // Check backend health
  checkBackendHealth();

  // Restore last results from storage
  chrome.storage.local.get(['lastImage', 'lastResult'], (data) => {
    if (data.lastImage) {
      imagePreview.src = data.lastImage;
      previewSection.classList.add('active');
    }
    if (data.lastResult) {
      displayResults(data.lastResult);
    }
  });
});

// Listen for real-time storage changes (when overlay/context menu runs analysis)
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && changes.lastResult) {
    const result = changes.lastResult.newValue;
    const image = changes.lastImage?.newValue;

    if (image) {
      imagePreview.src = image;
      previewSection.classList.add('active');
    }

    displayResults(result);
  }
});


