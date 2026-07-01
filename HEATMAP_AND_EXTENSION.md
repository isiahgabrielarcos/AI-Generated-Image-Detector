# Heatmap, Server, and Browser Extension — How They Work

---

## 1. Heatmap Visualisations

Two distinct visualisation methods are used: **GradCAM** (model attention) and **FFT spectrum** (frequency-domain artifacts). They answer different questions.

### 1.1 GradCAM — Where the Model Looks

GradCAM (Gradient-weighted Class Activation Map) shows the spatial regions of the image that most influenced the model's real/fake decision. A hot (red/yellow) region means the model weighted that area heavily.

**How it is computed** (`utils/visualization.py → GradCAM`):

1. The frozen CLIP ViT runs once under `torch.no_grad()` to extract patch tokens — no autograd graph is built through the 304 M frozen parameters.
2. Forward hooks are placed on `backbone.input_proj`, the layer that projects the fused spatial-frequency feature map `F_fused` to the Swin input resolution. Activations are captured here.
3. A full forward pass runs from `F_fused` onward, producing a scalar probability via `sigmoid(logit)`.
4. `.backward()` is called on that scalar — gradients flow only through the ~34.7 M trainable head, not the frozen ViT.
5. The gradient tensor `[1, C, G, G]` is globally average-pooled over the spatial axes to get per-channel importance weights.
6. Weights multiply activations channel-wise and the result is ReLU'd and summed → a `[G, G]` raw CAM grid.
7. The CAM is bilinearly upsampled to `224 × 224`, normalised to `[0, 1]`, and blended with the original image using OpenCV's `COLORMAP_JET`.

**Why `backbone.input_proj`?**  
The CAM is taken at the point where spatial (CLIP) and frequency (wavelet) features have already been fused by the SFDF cross-attention module. The heatmap therefore reflects *combined* spatial+frequency evidence, not just spatial appearance — matching Man & Cho (2026) Figure 6.

**Speed shortcuts:**
- If pre-cached CLIP tokens are available, the ViT is skipped entirely; only the head forward+backward runs.
- `activation_heatmap()` is a faster forward-only alternative (L2 norm of `F_fused` per token, no backprop), roughly 2× faster at the cost of being less faithful.

---

### 1.2 FFT Spectrum — Frequency-Domain Artifacts

The FFT spectrum is a signal-level analysis, completely independent of the model. It reveals periodic or structural artifacts that AI generators introduce in the frequency domain: GAN upsampling grids, spectral peaks from specific architectures, and compression patterns invisible to the naked eye.

**How it is computed** (`generate_sample_heatmaps.py → compute_fft_spectrum`):

1. The image is converted to grayscale.
2. A 2D Fast Fourier Transform is applied: `np.fft.fft2(gray)`.
3. The zero-frequency component (DC) is shifted to the centre of the array: `np.fft.fftshift(...)`.
4. The magnitude spectrum is log-scaled: `log(1 + |FFT|)` — compresses the very wide dynamic range so mid-frequency content is visible.
5. The result is normalised to `[0, 1]` and rendered with the `inferno` colormap.

**What to look for:**
- **Real images**: energy falls off smoothly from the bright DC centre outward; no strong directional ridges.
- **AI-generated images**: grid-like periodic peaks, cross-shaped ridges along horizontal/vertical axes, or concentric rings — artifacts from upsampling convolutions repeated spatially across the image.

---

### 1.3 Composite Figure

`generate_sample_heatmaps.py` produces a 2×3 panel (matching Man & Cho Fig. 5) for a real/fake pair:

```
              Original     |  Frequency Spectrum  |  Attention Heatmap
Real image    face photo   |  FFT log-magnitude   |  GradCAM overlay
Fake image    face photo   |  FFT log-magnitude   |  GradCAM overlay
```

All outputs land in `heatmap/output/`.

| File | Contents |
|------|----------|
| `heatmap_image1-6.png` | GradCAM overlay on the original image |
| `freq_image7_real.png` | Original + FFT spectrum side-by-side |
| `freq_image8_fake.png` | Original + FFT spectrum side-by-side |
| `frequency_composite.png` | Full 2×3 composite panel |

---

## 2. The Flask Server (`server.py`)

The server is a local Flask REST API that the browser extension calls. It holds the model in memory across requests so the expensive model load only happens once at startup.

### Startup

```
python server.py --checkpoint checkpoints/best_model.pt
```

At startup the server:
1. Loads the YAML config (`configs/default.yaml`).
2. Builds the full `AIGCDetector` with `force_load_visual=True` — the CLIP ViT is loaded because incoming browser images are not in any CLIP cache.
3. Loads the trained head weights from the checkpoint (`strict=False`, since the frozen ViT keys are not stored in checkpoints saved in cache-mode).
4. Instantiates `GradCAM(model)` and `build_transforms(augment=False)`.
5. Starts Flask on `http://127.0.0.1:5000` with `threaded=True`.

A `--demo` flag skips the model load and returns random placeholder responses — useful for UI development without a GPU.

### Endpoints

#### `GET /health`
Returns the server status. The extension's popup calls this on load to show a warning if the backend is unreachable.

```json
{ "status": "ok", "model": "loaded", "device": "cpu" }
```

#### `POST /detect`
Main inference endpoint. Request body:

```json
{
  "image": "<data-URL or raw base64 string or https:// URL>",
  "generate_heatmap": true
}
```

**Image decoding** (`_decode_image`):
- `data:image/...;base64,...` — split at the comma, base64-decode, open with PIL.
- `https://...` URL — fetched server-side with `requests.get` (User-Agent: `ArtifyBot/1.0`).
- Raw base64 string — decoded directly.

**Inference pipeline:**
1. PIL image → `build_transforms` (resize 224×224, CLIP normalise) → `[1, 3, 224, 224]` tensor.
2. `model(tensor)` → binary logit → `sigmoid` → probability `p ∈ [0, 1]`.
3. Threshold at 0.5: `"AI-Generated"` if `p ≥ 0.5`, else `"Real"`.
4. Confidence = `max(p, 1 − p)`.
5. If `generate_heatmap=true`: `GradCAM(tensor)` → `heatmap_to_overlay(pil_img, cam)` → PNG base64 data-URL.

Response:

```json
{
  "prediction":         "AI-Generated",
  "confidence":         0.9912,
  "probability":        0.9912,
  "processing_time_ms": 1842.3,
  "heatmap_overlay":    "data:image/png;base64,..."
}
```

#### `POST /batch_detect`
Accepts an array of images and returns predictions for each, without heatmaps. Used by the dashboard for bulk analysis.

```json
{ "images": ["<data-URL>", "<data-URL>", ...] }
```

---

## 3. The Browser Extension (ArtifyAD)

The extension is a Chrome Manifest V3 extension with four components.

```
manifest.json   — declares permissions, scripts, and entry points
background.js   — service worker; handles context menu and routing
content.js      — injects the floating overlay into every page
popup.js        — drives the popup opened from the toolbar icon
```

### 3.1 `manifest.json`

Key declarations:
- **Permissions**: `storage` (persist last result), `activeTab`, `contextMenus` (right-click menu on images).
- **Host permissions**: `http://localhost:5000/*` (to call the local Flask server) and `<all_urls>` (to fetch images from any site).
- **Content script**: `content.js` injected at `document_idle` into every page.
- **Background**: `background.js` runs as a persistent service worker.
- **Action popup**: `popup.html` opens when the toolbar icon is clicked.

### 3.2 `background.js` — Service Worker

The service worker has no UI of its own. It handles three jobs:

| Trigger | Action |
|---------|--------|
| Extension installed | Registers a **right-click context menu** item "Analyze with ArtifyAD" on images |
| Context menu clicked | Stores the image URL in `chrome.storage.local`, then sends `OPEN_OVERLAY_AND_ANALYZE` to the active tab's content script |
| Message `open_dashboard` | Opens `dashboard.html` in a new tab |
| Message `open_overlay` | Forwards `OPEN_OVERLAY` to the active tab's content script (used by the popup to sync the on-page panel) |

### 3.3 `content.js` — On-Page Overlay

`content.js` injects a Shadow DOM host (`<div id="artifyad-overlay-root">`) into `document.documentElement`. Using Shadow DOM means the extension's styles cannot clash with the host page's CSS.

The overlay has two states:

**Closed (pill):** A small vertical pill fixed to the right edge of the viewport. Clicking it opens the panel; the ✕ button dismisses it entirely.

**Open (panel):** A 340 px side panel slides in from the right. It contains:
- A drop zone — accepts drag-and-drop files or image URLs.
- Original / Heatmap tab toggle — switch between the raw image and the GradCAM overlay.
- A verdict badge — shows "AI-GENERATED" (red/orange) or "HUMAN-MADE" (green) with the confidence percentage.
- A "Report it here" link for incorrect predictions.

**`analyze(src)` — the core function:**

1. Opens the panel and shows a spinner.
2. If `src` is a URL (not a data-URL), fetches the image bytes and converts to a data-URL with `FileReader` so the base64 payload can be sent to the server (cross-origin images cannot be read directly by canvas).
3. POSTs `{ image: dataUrl, generate_heatmap: true }` to `http://localhost:5000/detect`.
4. On success: calls `render(result)` which sets the verdict badge and loads `heatmap_overlay` into the hidden `<img id="img-heatmap">` element.
5. Persists `lastImage` and `lastResult` to `chrome.storage.local` so state survives a panel close/reopen.

**Message listeners:**
- `OPEN_OVERLAY` — opens the panel and restores the last image and result from storage.
- `OPEN_OVERLAY_AND_ANALYZE` — opens the panel *and* immediately calls `analyze(imageUrl)` (triggered by the right-click menu).

### 3.4 `popup.js` — Toolbar Popup

The popup (`popup.html` + `popup.js`) provides a standalone alternative interface inside the extension's popup window. It mirrors the overlay panel's functionality:
- Drag-and-drop or file-picker to load an image.
- Same `fetch → /detect` call with `generate_heatmap: true`.
- Same Original / Heatmap tab toggle.
- On load, calls `GET /health` to check if the server is running; shows an error banner if not.
- Persists results to `chrome.storage.local` and sends `open_overlay` to the background worker so the on-page overlay syncs.

---

## 4. End-to-End Flow

```
User right-clicks image on any page
        │
        ▼
background.js  ── OPEN_OVERLAY_AND_ANALYZE ──▶  content.js (overlay)
                                                       │
                                               fetch image as data-URL
                                                       │
                                               POST /detect  ─────────▶  server.py
                                                                              │
                                                                    decode image (PIL)
                                                                    transform → [1,3,224,224]
                                                                    model(tensor) → sigmoid → prob
                                                                    GradCAM → heatmap_to_overlay
                                                                    return JSON
                                                       │
                                               render verdict badge
                                               load heatmap into <img>
                                               save to chrome.storage.local
```

---

## 5. Running Locally

```powershell
# 1. Start the detection server
.\.venv\Scripts\python.exe server.py --checkpoint checkpoints\best_model.pt

# 2. Load the extension in Chrome
#    chrome://extensions  →  Load unpacked  →  select the extension\ folder

# 3. Generate sample heatmaps (offline, no server needed)
.\.venv\Scripts\python.exe generate_sample_heatmaps.py
#    outputs → heatmap\output\
```
