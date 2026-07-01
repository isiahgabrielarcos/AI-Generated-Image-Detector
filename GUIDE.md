# Man & Cho (2026) AIGC Detector — Practical Guide

A complete reference for training, evaluating, running inference, and deploying the detector described in the paper.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Environment Setup](#2-environment-setup)
3. [Dataset Preparation](#3-dataset-preparation)
4. [CLIP Feature Cache](#4-clip-feature-cache)
5. [Configuration](#5-configuration)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Inference](#8-inference)
9. [Serving via Flask API](#9-serving-via-flask-api)
10. [Key Things to Keep in Mind](#10-key-things-to-keep-in-mind)

---

## 1. Architecture Overview

```
Image [B, 3, 224, 224]
        │
        ├─► CLIPExtractor  (frozen ViT-L/14)  →  Fs [B, N, D]   ← spatial/semantic
        ├─► WaveletExtractor (DWT-db4 + CNN)  →  Ff [B, N, D]   ← frequency/texture
        │
        ├─► SFDF (cross-domain fusion)        →  F_fused [B, N, D]
        │       Stage 1: Cross-Attention  (Fs queries Ff)
        │       Stage 2: Gated Integration  G = σ(MLP([Fs ; A]))
        │                F_fused = G ⊙ Fs + (1-G) ⊙ A
        │
        ├─► Reshape [B, D, G, G]  (G = 16 for ViT-L/14)
        │
        ├─► SwinBackbone  →  z [B, D]         ← global context
        │
        └─► MLP head → sigmoid → ŷ ∈ (0,1)
                0 = Real,  1 = AI-Generated
```

**Why two branches?**
- CLIP catches *semantic* artifacts (unnatural object relationships, style inconsistencies).
- Wavelets catch *frequency* artifacts (GAN/diffusion high-freq noise patterns invisible to the eye).
- SFDF lets the spatial stream selectively pull in frequency evidence via cross-attention.

---

## 2. Environment Setup

### Virtual environment and dependencies

```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

> **CPU-only install:** The `requirements.txt` installs the CPU build of PyTorch by default.
> If you later get a GPU, reinstall PyTorch with the matching CUDA build:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Hugging Face token (speeds up model downloads)

Create a `.env` file in the project root:

```
HF_TOKEN=your_token_here
```

Both `cache_clip_features.py` and `train.py` call `setup_hf_auth()` at startup, which reads this file and authenticates with Hugging Face Hub automatically. This removes rate-limiting on model downloads (`open_clip`, `timm`) and makes the initial CLIP weight download more reliable.

> **The `.env` file is listed in `.gitignore` and will never be committed.**

---

## 3. Dataset Preparation

The model expects datasets structured with two sub-folders:

```
datasets/
  MyDataset/
    real/        ← real images (.jpg, .png, ...)
    fake/        ← AI-generated images
```

Update `configs/default.yaml` to point to your dataset roots:

```yaml
data:
  dfdc_root:        "datasets/DFDC"
  forensynths_root: "datasets/ForenSynths"
  genimage_root:    "datasets/GenImage"
```

The code splits **80% train / 20% val** using a fixed random seed (42), so the split is reproducible across runs. The full dataset used in the paper is ~30 k images total (5 k real + 5 k fake per dataset).

### Supported benchmark datasets

| Dataset | What it covers |
|---|---|
| DFDC | Deepfake video frames (face swaps) |
| ForenSynths | Classic GAN-generated images (ProGAN, StyleGAN, etc.) |
| GenImage | Modern diffusion-model images (SD, Midjourney, VQDM, etc.) |

You can swap in any dataset that follows the `real/` + `fake/` folder layout.

### Reducing dataset size for faster iteration

Set `max_samples_per_dataset` in the config to cap images per class per dataset. For example, `3340` gives ~10 k total (1 670 real + 1 670 fake × 3 datasets):

```yaml
data:
  max_samples_per_dataset: 3340   # ~10 k total; null = use all (~30 k)
```

---

## 4. CLIP Feature Cache

> **This step is not in the original paper.** Man & Cho used a GPU where running CLIP every batch costs ~5 ms. On CPU it costs ~30–60 seconds per batch. The cache eliminates that cost by running CLIP once per image and saving the result.

### Why it is safe

CLIP is **completely frozen** — its weights never change across all training epochs. This means it produces the exact same tokens for the same image every single time. Running it 50 × (once per epoch) is pure waste. The cache runs it once, saves the result, and training loads the saved tokens instead.

The model trains on **identical values** either way. The only difference is where the tokens come from.

### Run the cache script once before training

```powershell
python cache_clip_features.py
```

Optional flags:

```powershell
# Larger batch = faster, but uses more RAM during caching
python cache_clip_features.py --batch-size 8

# Re-cache a dataset that already has a cache file
python cache_clip_features.py --overwrite
```

### What it saves

For each dataset a file is written to `datasets/clip_cache/`:

| File | Contents | Size (float16) |
|---|---|---|
| `dfdc_clip.pt` | `paths` list + `features` Tensor [N, 256, 1024] | ~5.2 GB for 10 k images |
| `forensynths_clip.pt` | same format | ~5.2 GB for 10 k images |
| `genimage_clip.pt` | same format | ~5.2 GB for 10 k images |

> **Total for 30 k images: ~15.7 GB disk + RAM.** Your machine has 31.4 GB RAM so all three caches load comfortably alongside the model and training buffers.

### Expected run time (CPU, batch-size 4)

| Dataset size | Estimated time |
|---|---|
| 10 k images | 30 min – 2 h |
| 30 k images | 1.5 h – 6 h |

This is a **one-time cost**. After the cache exists, `train.py` and `evaluate.py` detect it automatically and skip the CLIP ViT entirely every batch.

### How auto-detection works

`build_detector()` checks whether `datasets/clip_cache/*_clip.pt` files exist. If they do, it sets `load_visual=False` inside `CLIPExtractor`, which prevents the 1.2 GB frozen ViT from being loaded into RAM at all. The trainable 1024→768 projection adapter is still loaded and trained normally.

---

## 5. Configuration

All hyperparameters live in [`configs/default.yaml`](configs/default.yaml).

### Full reference

| Section | Key | Default | What it does |
|---|---|---|---|
| `model` | `clip_model` | `ViT-L/14` | CLIP backbone variant |
| `model` | `feature_dim` | `768` | Shared embedding dimension D |
| `model` | `swin_window_size` | `4` | Swin local attention window (must divide 16) |
| `data` | `image_size` | `224` | Input resolution |
| `data` | `clip_cache_dir` | `datasets/clip_cache` | Where cache files are stored / read from. Set to `null` to disable caching and compute CLIP live (very slow on CPU). |
| `data` | `max_samples_per_dataset` | `null` | Cap images per dataset. `null` = use all. |
| `data` | `num_workers` | `0` | DataLoader worker processes. Keep at `0` on Windows/CPU — worker processes would each reload the full cache, multiplying RAM usage. |
| `data` | `jpeg_compression` | `true` | Randomly augment with JPEG quality 70–100 |
| `training` | `batch_size` | `8` | Per-step batch size |
| `training` | `accumulation_steps` | `4` | Gradient accumulation steps. Effective batch = `batch_size × accumulation_steps` = **32** |
| `training` | `epochs` | `50` | Maximum training epochs (early stopping may end sooner) |
| `training` | `early_stop_patience` | `10` | Stop if val AUC has not improved for this many epochs |
| `training` | `cnn_lr` | `1e-4` | LR for the wavelet CNN branch |
| `training` | `backbone_lr` | `1e-5` | LR for Swin backbone + CLIP projection adapter |
| `training` | `focal_gamma` | `2.0` | Focal loss γ — higher = more focus on hard examples |
| `training` | `warmup_epochs` | `5` | Linear LR warm-up before cosine decay |
| `training` | `gradient_clip` | `1.0` | Max gradient norm |
| `training` | `use_amp` | `false` | bfloat16 autocast. **Disabled by default.** On AMD CPUs without AVX-512 BF16, Intel oneDNN routes these ops through an Intel-optimised path that is **5–10× slower** than plain FP32. Enable only on Intel CPUs with VNNI/BF16 (Alder Lake+, Ice Lake Server+). |
| `training` | `use_compile` | `false` | `torch.compile` — adds ~5–10 min compilation overhead on the first epoch, then gives 20–50% faster iterations. Set `true` to enable. Requires PyTorch 2.0+. |
| `logging` | `save_every` | `5` | Checkpoint every N epochs |
| `logging` | `eval_every` | `1` | Validate every N epochs |

---

## 6. Training

### Full workflow

```powershell
# Step 1 — cache CLIP features (one-time, ~2–6 h on CPU)
python cache_clip_features.py

# Step 2 — train
#   Option A: standard launch
python train.py

#   Option B (AMD CPU — RECOMMENDED): use the launcher script which sets
#   MKL_DEBUG_CPU_TYPE=5 before Python starts, enabling AVX2 BLAS paths on AMD.
#   This must be set BEFORE import torch (MKL initialises at import time).
.\run_train.ps1

#   Option B with extra args:
.\run_train.ps1 --resume checkpoints/epoch_10.pt
```

### What happens during training

1. **HF authentication** — reads `.env`, logs in to HF Hub for fast model downloads.
2. **CPU thread tuning** — sets `torch.set_num_threads` to available cores for best matrix-op performance.
3. **CLIP cache auto-detected** — if `datasets/clip_cache/` has `.pt` files, the ViT is not loaded. The DataLoader returns `(image, clip_tokens, label)` 3-tuples instead of 2-tuples; the model receives cached tokens each batch.
4. **Two-group AdamW optimizer** — CNN branch uses `cnn_lr`, everything else uses `backbone_lr`. CLIP visual is frozen and contributes zero learnable parameters.
5. **Gradient accumulation** — gradients are accumulated over `accumulation_steps=4` mini-batches before each optimizer step, giving an effective batch size of 32 while keeping per-step memory low.
6. **LR schedule** — 5-epoch linear warm-up followed by cosine annealing down to `1e-7`. The scheduler steps once per optimizer update (not per mini-batch).
7. **bfloat16 autocast** — disabled by default (`use_amp: false`). On AMD CPUs without hardware BF16 support, enabling this routes Linear/Conv ops through Intel's oneDNN library and is 5–10× slower than plain FP32. Only enable on Intel CPUs with VNNI/BF16 hardware.
8. **Focal loss** (`gamma=2.0`) down-weights easy examples and focuses training on hard ones.
9. **Gradient clipping** — capped at `1.0` to stabilise Swin attention layers.
10. **Early stopping** — training stops automatically if val AUC does not improve for `early_stop_patience=10` consecutive epochs.
11. **Best model saved** by AUC-ROC → `checkpoints/best_model.pt`.
12. **ETA display** — each epoch prints estimated time remaining based on the average of the last 3 epochs.
13. **TensorBoard logs** written to `runs/`.

### Expected training time on CPU (with cache, AMP disabled, lightweight CNN)

| Dataset size | Iterations/epoch | Approx. time per epoch | 20 epochs (early stop ~ep20) |
|---|---|---|---|
| 10 k (`max_samples_per_dataset: 3340`) | 1 000 | 1–3 h | **1–3 days** ← recommended for CPU |
| 30 k (full, `null`) | 3 000 | 3–8 h | 3–7 days |

> **For CPU training, strongly consider setting `max_samples_per_dataset: 3340`** to keep each epoch manageable (~3× fewer iterations). The model still trains on all three dataset types; it just uses a random 3 340-sample cap per dataset.

> **Prior to the AMP fix**, the `use_amp: true` default caused oneDNN overhead on AMD CPUs, resulting in ~36 s/it (~30 hours per epoch). With `use_amp: false` the expected speed is **3–8 s/it** (~3–8 h/epoch at full 30 k).

### Monitor training

```powershell
tensorboard --logdir runs/
# then open http://localhost:6006 in a browser
```

---

### Checkpointing

| File | When saved | Contents |
|---|---|---|
| `best_model.pt` | Every time val AUC beats the previous best | weights, optimizer state, epoch, best AUC, full metrics, config |
| `epoch_NNN.pt` | Every `save_every` epochs (default: 5) | weights, optimizer state, epoch, best AUC |

> **Always use `best_model.pt` for evaluation and deployment.**

#### What is stored in each `.pt` file

```python
{
    "epoch":     int,    # 0-indexed epoch number
    "model":     dict,   # model.state_dict()
    "optimizer": dict,   # optimizer.state_dict()
    "best_auc":  float,  # best AUC seen so far
    "metrics":   dict,   # (best_model.pt only) full val metrics at that epoch
    "cfg":       dict,   # (best_model.pt only) config dict used during training
}
```

#### Resuming a paused or crashed run

```powershell
python train.py --resume checkpoints/epoch_050.pt
python train.py --resume checkpoints/best_model.pt   # fine-tune further
```

Epoch counter, optimizer momentum, and best AUC are all restored exactly.

#### Inspecting a checkpoint

```python
import torch
ckpt = torch.load("checkpoints/best_model.pt", map_location="cpu")
print(f"Epoch  : {ckpt['epoch'] + 1}")
print(f"AUC    : {ckpt['best_auc']:.4f}")
print(f"Metrics: {ckpt.get('metrics', 'n/a')}")
```

---

## 7. Evaluation

### Evaluate on all three datasets

```powershell
python evaluate.py --checkpoint checkpoints/best_model.pt
```

### Evaluate on a custom dataset folder

```powershell
python evaluate.py --checkpoint checkpoints/best_model.pt `
                   --data_dir datasets/MyDataset `
                   --output_dir results/my_eval
```

> When `--data_dir` is used, the images are not guaranteed to be in the CLIP cache. `evaluate.py` detects this and automatically reloads the full CLIP ViT for live feature extraction on that folder.

### Adjust decision threshold

```powershell
python evaluate.py --checkpoint checkpoints/best_model.pt --threshold 0.4
```

### Console output

For each dataset the terminal prints:

```
────────────────────────────────────────────────────────────
 Dataset : DFDC
────────────────────────────────────────────────────────────
[DFDC] ACC=0.9120  AP=0.9340  Recall=0.8950  F1=0.9100  AUC=0.9580

  Confusion Matrix (n=2000):
                            Pred: Real  Pred: AI-Gen
  Actual: Real                     921           79   (TN / FP)
  Actual: AI-Gen                    97          903   (FN / TP)

  True  Positive (TP) = 903   |  correctly caught AI
  True  Negative (TN) = 921   |  correctly passed Real
  False Positive (FP) =  79   |  Real wrongly flagged as AI
  False Negative (FN) =  97   |  AI missed as Real
```

Followed by a cross-dataset summary table at the end.

### Metrics reported

| Metric | Description |
|---|---|
| **ACC** | Accuracy at the chosen threshold |
| **AP** | Average Precision (area under PR curve) — threshold-free |
| **Recall** | True positive rate at threshold |
| **F1** | Harmonic mean of precision & recall |
| **AUC-ROC** | Area under ROC curve — primary model selection metric |

### Output files

For each dataset, `evaluate.py` saves to `results/eval/<DatasetName>/`:

```
roc_curve.png
pr_curve.png
confusion_matrix.png      ← heatmap version of the confusion matrix
metrics.csv
```

A cross-dataset `results/eval/summary.csv` is also written.

---

## 8. Inference

Inference always loads the **full CLIP ViT** (`force_load_visual=True`) regardless of cache, because the input images are arbitrary and are not guaranteed to be in the pre-computed cache.

### Single image

```powershell
python inference.py --checkpoint checkpoints/best_model.pt --image path/to/image.jpg
```

### Entire folder

```powershell
python inference.py --checkpoint checkpoints/best_model.pt --image_dir path/to/folder/
```

### Generate GradCAM heatmap overlay

```powershell
python inference.py --checkpoint checkpoints/best_model.pt `
                    --image path/to/image.jpg `
                    --heatmap `
                    --output_dir results/inference
```

The heatmap PNG is saved as `<image_stem>_heatmap.png` in `--output_dir`.

### Output columns in `predictions.csv`

| Column | Meaning |
|---|---|
| `image` | Source file path |
| `prediction` | `AI-Generated` or `Real` |
| `probability` | Raw P(AI-generated) ∈ [0, 1] |
| `confidence` | `max(p, 1-p)` — how certain the model is |
| `time_ms` | Inference latency in milliseconds |

---

## 9. Serving via Flask API

The [`server.py`](server.py) powers the Artify browser extension.

### Start the server

```powershell
# Real model
python server.py --checkpoint checkpoints/best_model.pt --port 5000

# Demo mode (no checkpoint needed — returns random results for UI testing)
python server.py --demo
```

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check — returns model status and device |
| `POST` | `/detect` | Single-image detection |
| `POST` | `/batch_detect` | Batch detection |

### `/detect` request/response

```json
// Request
{ "image": "<data-URL or base64>", "generate_heatmap": false }

// Response
{
  "prediction": "AI-Generated",
  "confidence": 0.921,
  "probability": 0.921,
  "processing_time_ms": 42.3,
  "heatmap_overlay": null
}
```

---

## 10. Key Things to Keep in Mind

### Model & Architecture

- **CLIP is frozen** — its weights never update. Only the linear projection adapter (1024→768), wavelet CNN, SFDF, Swin backbone, and classifier head are trained. Do not accidentally unfreeze `clip_extractor.visual`; it would massively increase memory and destabilise training.
- **Grid size must stay at 16** when using `ViT-L/14`. The Swin window size must evenly divide 16 (valid: 1, 2, 4, 8, 16). Changing `clip_model` requires recalculating the grid and adjusting `swin_window_size`.
- **SFDF `dim % num_heads == 0`** is a hard constraint. With `feature_dim=768` and `num_heads=8`, each head gets 96 dims. If you change `feature_dim`, adjust `num_heads` accordingly.

### Data

- Images must be **224×224**. The transforms handle resizing automatically.
- `ImageFolderBinary` expects exactly two sub-folders: `real/` (label 0) and `fake/` (label 1).
- JPEG compression augmentation (`jpeg_compression: true`) is important for real-world robustness — compressed images are common online and can confuse detectors trained only on pristine images.
- When the CLIP cache is active, the cache is loaded with `num_workers=0` to prevent each worker subprocess from duplicating the full cache in RAM (a Windows multiprocessing constraint).

### Training

- **Two learning rates are intentional** (paper §4.2). The Swin backbone is a large pretrained model needing a small LR; the wavelet CNN starts from scratch and can take a larger step.
- **Gradient accumulation** (`accumulation_steps=4`) gives an effective batch size of 32 while keeping per-step memory at 8 samples. The scheduler steps once per optimizer update, not per mini-batch.
- **Early stopping** halts training when val AUC does not improve for `early_stop_patience=10` epochs, saving wasted compute.
- **Focal loss** handles class imbalance. If your dataset is perfectly balanced, `focal_gamma=0` reduces to standard binary cross-entropy.
- **Best model is saved by AUC**, not by loss. A lower loss does not guarantee a better detector.
- The CLIP cache is a **CPU-only optimisation** — it is not in the original paper. On a GPU the CLIP ViT forward pass is fast enough that caching gives negligible benefit.

### Evaluation

- **AUC is the primary metric** — it summarises performance across all thresholds and is independent of class balance.
- **AP is also important** when false positives are costly (e.g., incorrectly flagging a human artist's work).
- The default **threshold of 0.5** is a starting point. Raise it for fewer false positives (at the cost of more misses); lower it to catch more fakes.
- Always evaluate on **held-out datasets** to measure generalisation. AIGC detectors are prone to overfitting to the generator distribution they were trained on.

### Inference & Deployment

- Inference uses `force_load_visual=True` so the full CLIP ViT is always available for arbitrary new images — do not remove this.
- The model calls `model.eval()` and wraps inference in `torch.no_grad()`. Never skip these in custom inference code — they affect Dropout and BatchNorm behaviour.
- GradCAM heatmaps highlight spatial regions from the **fused SFDF feature map**, showing which parts of the image most influenced the prediction. Useful for debugging; adds ~2× latency per request.
- `server.py` is single-threaded for GPU inference. For production, consider Triton or TorchServe with batching.

### Common Pitfalls

| Symptom | Likely cause | Fix |
|---|---|---|
| `AssertionError: Token count N ≠ grid_size²` | Changed CLIP model without updating `grid_size` | Set `grid_size` in `build_detector` to match the new patch grid |
| `AssertionError: CLIP visual encoder not loaded` | Cache mode active (`load_visual=False`) but `clip_tokens=None` | Ensure cache files exist and are loaded; or set `clip_cache_dir: null` |
| `RuntimeError: size mismatch` loading checkpoint | Config changed after training | Use the same `default.yaml` that was active during training |
| AUC stuck near 0.5 | CLIP weights accidentally unfrozen, or LRs too high | Verify `requires_grad=False` on `clip_extractor.visual`; lower LRs |
| RAM exhausted during training | Cache + model + activations exceed available RAM | Reduce `max_samples_per_dataset` to 3340 (10 k total) or use `clip_cache_dir: null` |
| Cache creation very slow | Default `--batch-size 4` is conservative | Try `--batch-size 8` if you have enough RAM |
| `[hf_auth] No HF_TOKEN found` | `.env` file missing or token not set | Create `.env` with `HF_TOKEN=your_token` in the project root |
| OOM on GPU | Batch size too large | Reduce `batch_size`; use `accumulation_steps` to maintain effective batch |
| Heatmap is all one colour | GradCAM hooks not on the right layer | Check `utils/visualization.py` GradCAM target layer |
| `/detect` returns 500 | Server started without `--checkpoint` (non-demo mode) | Pass a valid checkpoint path or use `--demo` |
| Training extremely slow (>10 s/it on CPU) | `use_amp: true` on AMD CPU — oneDNN BF16 is 5–10× slower than FP32 on AMD | Set `use_amp: false` in `configs/default.yaml` (already the default now) |
| Training slow despite AMP disabled | Intel MKL not using AMD's AVX2 paths | Run via `.\run_train.ps1` which sets `MKL_DEBUG_CPU_TYPE=5` before torch imports |
| `RuntimeError: size mismatch` after wavelet CNN change | Checkpoint saved with `mid=192` (old wide CNN); new code uses `mid=64` | The architecture changed — start training fresh (no resume from old checkpoint) |
