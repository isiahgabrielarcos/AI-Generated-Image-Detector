# AIGC Detector — Man & Cho (2026)
**Transformer Based on Multi-Domain Feature Fusion for AI-Generated Image Detection**

Replication of the hybrid CLIP + Wavelet + Swin-Transformer architecture from:
> Qiaoyue Man and Young-Im Cho. *Electronics 2026, 15, 716.*
> https://doi.org/10.3390/electronics15030716

Integrated with the **Artify** browser extension via a local Flask API.

---

## Project Structure

```
aigc_detector/
├── configs/
│   └── default.yaml          ← all hyperparameters & dataset paths
├── data/
│   └── dataset.py            ← DFDC / ForenSynths / GenImage loaders
├── models/
│   ├── clip_extractor.py     ← frozen CLIP ViT-L/14 spatial features
│   ├── wavelet_extractor.py  ← DWT-db4 + lightweight CNN frequency features
│   ├── sfdf.py               ← cross-attention + gated fusion
│   ├── swin_backbone.py      ← Swin-Tiny global modelling
│   └── detector.py           ← full AIGCDetector model
├── losses/
│   └── focal_loss.py         ← binary focal loss (γ=2)
├── utils/
│   ├── metrics.py            ← ACC, AP, Recall, F1, AUC-ROC
│   └── visualization.py      ← GradCAM, heatmap overlay, ROC / PR plots
├── train.py                  ← training loop
├── evaluate.py               ← full evaluation with plots
├── inference.py              ← single-image / folder inference
└── server.py                 ← Flask API for the Artify extension
```

---

## 1. Installation

```bash
# Clone / enter project
cd aigc_detector

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

.\venv\Scripts\Activate.ps1  # Working

# Install dependencies
pip install -r requirements.txt
```

> **GPU note:** Install the CUDA-compatible PyTorch first if you have a GPU:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

## 2. Dataset Download

Download the datasets and unzip them. Each must have the structure:
```
<dataset_root>/
  real/   ← real/authentic images
  fake/   ← AI-generated images
```

| Dataset | Link | Notes |
|---------|------|-------|
| **DFDC** (Deepfake Detection Challenge) | https://www.kaggle.com/c/deepfake-detection-challenge/data | Requires Kaggle login. Extract video frames with `ffmpeg` (see §2.1) |
| **ForenSynths** | https://github.com/peterwang512/CNNDetection | Download `wang2020_database.zip` |
| **GenImage** | https://github.com/GenImage-Dataset/GenImage | ~1M images from Midjourney, SD, DALL·E, etc. |

After downloading, set the paths in `configs/default.yaml`:
```yaml
data:
  dfdc_root:        "datasets/DFDC"
  forensynths_root: "datasets/ForenSynths"
  genimage_root:    "datasets/GenImage"
```

### 2.1 Extracting DFDC Frames

```bash
# Extract 1 frame per second from each video
mkdir -p datasets/DFDC/real datasets/DFDC/fake

# Real videos
for f in datasets/DFDC_raw/train_sample_videos/*.mp4; do
  name=$(basename "$f" .mp4)
  ffmpeg -i "$f" -vf fps=1 "datasets/DFDC/real/${name}_%04d.jpg" -hide_banner -loglevel error
done

# Fake videos (check metadata.json for labels)
python scripts/split_dfdc_by_label.py   # see helper script below
```

---

## 3. Training

```bash
# Train with default config (auto-detects GPU)
python train.py --config configs/default.yaml

# Resume from a checkpoint
python train.py --config configs/default.yaml --resume checkpoints/epoch_050.pt

# Force CPU
python train.py --config configs/default.yaml --device cpu
```

### What happens during training
- **CLIP ViT-L/14** is frozen — only the projection layer + CNN + SFDF + Swin are trained.
- Two learning rates: CNN branch `1e-4`, Swin backbone `1e-5` (paper §4.2).
- Loss: Binary Focal Loss with γ=2.
- Checkpoints saved every N epochs (set `logging.save_every` in config).
- **Best model** (by AUC) saved to `checkpoints/best_model.pt` automatically.
- TensorBoard logs written to `runs/`. View with:
  ```bash
  tensorboard --logdir runs/
  ```

### Training output example
```
Epoch [  1/100]  loss=0.4821  (312.4s)
  [val ep1]  ACC=0.7234  AP=0.7891  Recall=0.6523 S F1=0.6981  AUC=0.8102
  ★ New best AUC=0.8102 → saved best_model.pt
Epoch [  2/100]  loss=0.3614  (298.1s)
  [val ep2]  ACC=0.8156  AP=0.8423  Recall=0.7812  F1=0.8044  AUC=0.8897
  ★ New best AUC=0.8897 → saved best_model.pt
...
```

---

## 4. Evaluation

Evaluation computes **Accuracy, Average Precision (AP), Recall, F1-Score, and AUC-ROC**
and saves ROC curve, PR curve, and Confusion Matrix as PNG.

```bash
# Evaluate on all datasets configured in default.yaml
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on a single custom dataset folder
python evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --data_dir   datasets/MyDataset \
    --output_dir results/my_eval

# Adjust decision threshold (default 0.5)
python evaluate.py --checkpoint checkpoints/best_model.pt --threshold 0.4
```

### Evaluation output

Terminal:
```
────────────────────────────────────────────────────────────
 Dataset : DFDC
────────────────────────────────────────────────────────────
[DFDC]  ACC=0.9810  AP=0.9680  Recall=0.9750  F1=0.9720  AUC=0.9930
  Confusion Matrix:
  [[4821  179]
   [ 123 4877]]

────────────────────────────────────────────────────────────
 Dataset : ForenSynths
...
════════════════════════════════════════════════════════════
 SUMMARY
════════════════════════════════════════════════════════════
Dataset           ACC      AP    Recall      F1     AUC
────────────────────────────────────────────────────────────
DFDC            0.9810  0.9680  0.9750  0.9720  0.9930
ForenSynths     0.9690  0.9890  0.9630  0.9650  0.9890
GenImage        0.9510  0.9690  0.9470  0.9480  0.9790
```

Saved files (in `results/eval/<dataset>/`):
```
roc_curve.png
pr_curve.png
confusion_matrix.png
metrics.csv
```

### Metric definitions

| Metric | Description |
|--------|-------------|
| **ACC** | Fraction of correctly classified images |
| **AP** | Area under the Precision-Recall curve (average precision) |
| **Recall** | True positive rate at threshold 0.5 |
| **F1** | Harmonic mean of precision and recall |
| **AUC** | Area under the ROC curve (threshold-independent) |

---

## 5. Single-Image / Folder Inference

```bash
# Analyze one image
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image path/to/image.jpg

# Analyze a folder
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image_dir path/to/folder/

# Generate GradCAM heatmap overlay
python inference.py \
    --checkpoint checkpoints/best_model.pt \
    --image     path/to/image.jpg \
    --heatmap \
    --output_dir results/inference/
```

Output:
```
  sunset.jpg                            ✅ Real            p=0.1234  conf=0.8766  (42.3 ms)
  ai_face.png                           🤖 AI-Generated    p=0.9821  conf=0.9821  (41.1 ms)
  portrait.jpg                          ✅ Real            p=0.0891  conf=0.9109  (40.8 ms)

[inference] predictions saved → results/inference/predictions.csv
```

---

## 6. Artify Extension Integration

The Artify extension already calls `http://localhost:5000` — you just need to
start the server.

### 6.1 Start the server

```bash
# After training – real detection
python server.py --checkpoint checkpoints/best_model.pt

# With GPU
python server.py --checkpoint checkpoints/best_model.pt --device cuda

# UI testing without a trained model
python server.py --demo
```

### 6.2 Load the extension in Chrome

1. Open `chrome://extensions/`
2. Enable **Developer Mode** (top right)
3. Click **Load unpacked**
4. Select the `extension/` folder from your Artify project
5. The extension icon appears in the toolbar

### 6.3 How the extension calls the server

```
Extension                           Server (server.py)
────────────────────────────────────────────────────────────
POST /detect                    →   decode image
  { image: "<data-URL>",            run CLIP + Wavelet + Swin
    generate_heatmap: true }        compute probability
                                    generate GradCAM overlay
                               ←   { prediction, confidence,
                                      probability,
                                      processing_time_ms,
                                      heatmap_overlay }

GET  /health                   →   { status: "ok", model: "loaded" }
```

### 6.4 API endpoints reference

#### `POST /detect`
```json
// Request
{ "image": "data:image/jpeg;base64,...", "generate_heatmap": true }

// Response
{
  "prediction":          "AI-Generated",   // or "Real"
  "confidence":          0.9821,
  "probability":         0.9821,           // raw P(AI-generated)
  "processing_time_ms":  42.3,
  "heatmap_overlay":     "data:image/png;base64,..."  // null if not requested
}
```

#### `POST /batch_detect`
```json
// Request
{ "images": ["data:image/jpeg;base64,...", "data:image/png;base64,..."] }

// Response
{ "results": [
    { "prediction": "Real",         "confidence": 0.89, "probability": 0.11 },
    { "prediction": "AI-Generated", "confidence": 0.97, "probability": 0.97 }
  ]
}
```

#### `GET /health`
```json
{ "status": "ok", "model": "loaded", "device": "cuda:0" }
```

---

## 7. Configuration Reference (`configs/default.yaml`)

```yaml
model:
  clip_model: "ViT-L/14"         # CLIP backbone (frozen)
  feature_dim: 768               # shared embedding dimension
  swin_window_size: 4            # Swin window (must divide 16)
  swin_depths: [2, 2, 6, 2]      # Swin-Tiny stage depths
  swin_num_heads: [3, 6, 12, 24] # attention heads per stage
  swin_embed_dim: 96             # Swin-Tiny base channels
  dropout: 0.1

data:
  dfdc_root:        "datasets/DFDC"
  forensynths_root: "datasets/ForenSynths"
  genimage_root:    "datasets/GenImage"
  train_split: 0.8
  num_workers: 4
  augmentation:
    random_flip: true
    random_crop: true
    jpeg_compression: true       # q ∈ [70,100]

training:
  batch_size: 16
  epochs: 100
  cnn_lr: 1.0e-4                 # lightweight CNN branch
  backbone_lr: 1.0e-5            # Swin backbone
  weight_decay: 1.0e-4
  focal_gamma: 2.0               # focal loss γ
  scheduler: "cosine"
  warmup_epochs: 5
  gradient_clip: 1.0

logging:
  log_dir:    "runs/"
  save_dir:   "checkpoints/"
  save_every: 5
  eval_every: 1
```

---

## 8. Hardware Requirements

| Component | Minimum | Recommended (paper) |
|-----------|---------|---------------------|
| GPU | 8 GB VRAM (RTX 3060) | 24 GB VRAM × 2 (RTX 3090Ti) |
| RAM | 16 GB | 64 GB |
| Storage | 50 GB | 500 GB (for full GenImage) |
| CUDA | 11.8 | 12.1 |

For CPU-only training reduce `batch_size` to 2–4 and expect ~10× longer training.

---

## 9. Model Architecture Summary

```
Input [B, 3, 224, 224]
        │
        ├──► CLIP ViT-L/14 (frozen) ──► Linear ──► Fs [B, 256, 768]
        │      spatial semantic features
        │
        ├──► DWT-db4 (GPU conv) ──► 9-ch map ──► CNN ──► Ff [B, 256, 768]
        │      frequency texture features
        │
        ├──► SFDF ─────────────────────────────────────► F_fused [B, 256, 768]
        │      cross-attention + gated integration
        │
        ├──► reshape → [B, 768, 16, 16]
        │
        ├──► Swin-Tiny (window=4, depths=[2,2,6,2]) ──► z [B, 768]
        │      global contextual modelling
        │
        └──► MLP + Sigmoid ──► ŷ ∈ (0,1)
               0 = Real,  1 = AI-Generated
```

---

## 10. Troubleshooting

| Problem | Solution |
|---------|----------|
| `Extension context invalidated` | Refresh the page after reloading the extension |
| `Backend server not reachable` | Make sure `python server.py` is running |
| `CUDA out of memory` | Reduce `batch_size` in config |
| `No datasets found` | Check paths in `configs/default.yaml` |
| `open_clip not found` | `pip install open-clip-torch` |
| Heatmap not showing | Confirm `generate_heatmap: true` sent in request |
| CORS error in extension | Server already sets `Access-Control-Allow-Origin: *` |

---

## Citation

```bibtex
@article{man2026transformer,
  title   = {Transformer Based on Multi-Domain Feature Fusion for AI-Generated Image Detection},
  author  = {Man, Qiaoyue and Cho, Young-Im},
  journal = {Electronics},
  volume  = {15},
  number  = {3},
  pages   = {716},
  year    = {2026},
  doi     = {10.3390/electronics15030716}
}
```



















































Set-Location .\datasets\ForenSynths

# Download the 7z split archive
1..7 | ForEach-Object {
    $part = "{0:D3}" -f $_
    $url = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.$part"
    $out = "progan_train.7z.$part"

    if (-Not (Test-Path $out)) {
        Write-Host "Downloading $out ..."
        Invoke-WebRequest -Uri $url -OutFile $out -ErrorAction Stop
    } else {
        Write-Host "$out already exists, skipping."
    }
}

# Extract the archive
Write-Host "Extracting archive..."
7z x progan_train.7z.001 -y

# Remove the downloaded parts if extraction succeeds
Remove-Item progan_train.7z.* -Force

Expand-Archive -Path .\progan_train.zip -DestinationPath . -Force
Remove-Item .\progan_train.zip -Force

$RealDir = ".\real"
$FakeDir = ".\fake"
New-Item -ItemType Directory -Force -Path $RealDir | Out-Null
New-Item -ItemType Directory -Force -Path $FakeDir | Out-Null

$realSrc = ".\progan\0_real"
$fakeSrc = ".\progan\1_fake"

Get-ChildItem -Path $realSrc -Filter *.png | Sort-Object Name | Select-Object -First 2500 |
    ForEach-Object { Copy-Item $_.FullName -Destination $RealDir }

Get-ChildItem -Path $fakeSrc -Filter *.png | Sort-Object Name | Select-Object -First 2500 |
    ForEach-Object { Copy-Item $_.FullName -Destination $FakeDir }

Write-Host "Done. 2500 real and 2500 fake images are now in $RealDir and $FakeDir." -ForegroundColor Green


