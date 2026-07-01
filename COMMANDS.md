# AIGC Detector — PowerShell Command Reference

All commands assume the virtual environment is **activated** and you are in the project root.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Data Preparation](#2-data-preparation)
3. [CLIP Feature Cache](#3-clip-feature-cache)
4. [Training](#4-training)
5. [Per-Generator Split Training](#5-per-generator-split-training)
6. [Finetuning](#6-finetuning)
7. [Evaluation](#7-evaluation)
8. [Per-Generator Evaluation (Tables 1–3)](#8-per-generator-evaluation-tables-13)
9. [Calibration](#9-calibration)
10. [Inference](#10-inference)
11. [Flask Server (Browser Extension)](#11-flask-server-browser-extension)
12. [Heatmaps](#12-heatmaps)
13. [TensorBoard](#13-tensorboard)

---

## 1. Environment Setup

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1
# OR use the .venv folder if that is what was created:
.\.venv\Scripts\Activate.ps1

# Install all dependencies
pip install -r requirements.txt

# GPU install (if you have a CUDA GPU — run BEFORE pip install -r requirements.txt)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

> Create a `.env` file in the project root with your Hugging Face token to avoid rate limits:
> ```
> HF_TOKEN=your_token_here
> ```

---

## 2. Data Preparation

### Equalize Training Data (one-time, fixes resolution/format bias)

```powershell
# Default: resize + center-crop to 224×224, save as PNG into datasets_eq/
python equalize_training_data.py

# Custom target size and output folder
python equalize_training_data.py --target 256 --out datasets_eq

# Save as JPEG instead of PNG
python equalize_training_data.py --format jpg --quality 95

# Process only one dataset (dfdc | forensynths | genimage)
python equalize_training_data.py --only genimage
```

### Download ForenSynths (ProGAN subset)

```powershell
Set-Location .\datasets\ForenSynths

# Download 7-part split archive from HuggingFace
1..7 | ForEach-Object {
    $part = "{0:D3}" -f $_
    $url  = "https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_train.7z.$part"
    $out  = "progan_train.7z.$part"
    if (-Not (Test-Path $out)) {
        Write-Host "Downloading $out ..."
        Invoke-WebRequest -Uri $url -OutFile $out -ErrorAction Stop
    } else {
        Write-Host "$out already exists, skipping."
    }
}

# Extract archive (requires 7-Zip in PATH)
7z x progan_train.7z.001 -y

# Remove downloaded parts after extraction
Remove-Item progan_train.7z.* -Force

# Expand zip and clean up
Expand-Archive -Path .\progan_train.zip -DestinationPath . -Force
Remove-Item .\progan_train.zip -Force

# Create real/ and fake/ folders and copy 2 500 images each
$RealDir = ".\real"; $FakeDir = ".\fake"
New-Item -ItemType Directory -Force -Path $RealDir | Out-Null
New-Item -ItemType Directory -Force -Path $FakeDir | Out-Null

Get-ChildItem -Path .\progan\0_real -Filter *.png | Sort-Object Name | Select-Object -First 2500 |
    ForEach-Object { Copy-Item $_.FullName -Destination $RealDir }

Get-ChildItem -Path .\progan\1_fake -Filter *.png | Sort-Object Name | Select-Object -First 2500 |
    ForEach-Object { Copy-Item $_.FullName -Destination $FakeDir }
```

---

## 3. CLIP Feature Cache

> Run **once** before training. After the cache exists, `train.py` detects it automatically and skips the frozen ViT every batch.

```powershell
# Default (batch-size 4, all datasets)
python cache_clip_features.py

# Larger batch — faster but uses more RAM
python cache_clip_features.py --batch-size 8

# Force re-cache an already-cached dataset
python cache_clip_features.py --overwrite

# Cache only one dataset (dfdc | forensynths | genimage)
python cache_clip_features.py --dataset dfdc

# Cache per-generator dataset only (into datasets_eq/clip_cache/)
python cache_clip_features.py --pergen_only per-gen-dataset

# Skip per-generator caching
python cache_clip_features.py --skip-pergen

# Custom output directory
python cache_clip_features.py --cache_dir datasets_eq/clip_cache
```

---

## 4. Training

### Recommended (AMD CPU) — sets MKL AVX2 fix before Python starts

```powershell
# Standard training run (AMD CPU recommended launcher)
.\run_train.ps1

# Resume from a checkpoint
.\run_train.ps1 --resume checkpoints/epoch_050.pt
.\run_train.ps1 --resume checkpoints/best_model.pt

# Pass any train.py flag through the launcher
.\run_train.ps1 --config configs/default.yaml
.\run_train.ps1 --device cpu
```

### Direct Python (works on any CPU/GPU)

```powershell
# Standard training
python train.py

# Resume from a checkpoint
python train.py --resume checkpoints/epoch_050.pt
python train.py --resume checkpoints/best_model.pt

# Custom config file
python train.py --config configs/default.yaml

# Force CPU
python train.py --device cpu
```

---

## 5. Per-Generator Split Training

> "Seen-generator 80/20" protocol — trains on 80% of every generator, tests on the disjoint 20%.

```powershell
# Default settings
python train_pergen_split.py

# Custom epochs, test fraction, and random seed
python train_pergen_split.py --epochs 20 --test_frac 0.2 --seed 42
```

---

## 6. Finetuning

> Finetunes an existing checkpoint on the per-generator dataset.

```powershell
# Variation 1 — DFDC only (fastest, fixes near-random DFDC performance)
python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --include_generators DFDC `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_dfdc_only.pt

# Variation 2 — All generators equally
python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_all_gens.pt

# Variation 3 — All generators, underperformers boosted 3×
python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --target_generators DFDC Deepfake StyleGAN StyleGAN2 BigGAN CycleGAN StarGAN GauGAN "DALL-E" `
    --target_weight 3 `
    --epochs 20 `
    --patience 7 `
    --save checkpoints/finetuned_boosted.pt

# Resume a cancelled finetune run
python finetune.py `
    --checkpoint checkpoints/best_model.pt `
    --include_generators DFDC `
    --resume checkpoints/finetune_epoch_010.pt `
    --save checkpoints/finetuned_dfdc_only.pt
```

### Additional finetune flags

| Flag | Default | Description |
|------|---------|-------------|
| `--backbone_lr` | `1e-6` | LR for Swin backbone |
| `--cnn_lr` | `1e-5` | LR for wavelet CNN |
| `--batch_size` | `8` | Batch size |
| `--accum_steps` | `4` | Gradient accumulation steps |
| `--val_frac` | `0.10` | Fraction of data used for validation |
| `--save_every` | `5` | Save a checkpoint every N epochs |
| `--seed` | `42` | Random seed |

---

## 7. Evaluation

```powershell
# Evaluate on all configured datasets
python evaluate.py --checkpoint checkpoints/best_model.pt

# Evaluate on a custom dataset folder (real/ + fake/ layout)
python evaluate.py `
    --checkpoint checkpoints/best_model.pt `
    --data_dir datasets/MyDataset `
    --output_dir results/my_eval

# Adjust decision threshold (default 0.5; raise = fewer false positives)
python evaluate.py --checkpoint checkpoints/best_model.pt --threshold 0.4

# Force a specific device
python evaluate.py --checkpoint checkpoints/best_model.pt --device cpu

# Override batch size
python evaluate.py --checkpoint checkpoints/best_model.pt --batch_size 4
```

Outputs saved to `results/eval/<DatasetName>/`:
- `roc_curve.png`, `pr_curve.png`, `confusion_matrix.png`, `metrics.csv`
- `results/eval/summary.csv` — cross-dataset summary

---

## 8. Per-Generator Evaluation (Tables 1–3)

```powershell
# Single checkpoint — produces Tables 1, 2, 3
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --generators_root per-gen-dataset-test `
    --clip_cache_dir datasets_eq/clip_cache_test

# Two checkpoints side-by-side (with labels)
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt --label "Cross-gen model" `
    --checkpoint checkpoints/pergen_split_best_model.pt --label "Seen-gen 80/20" `
    --generators_root per-gen-dataset-test `
    --clip_cache_dir datasets_eq/clip_cache_test

# Ablation mode (no separate training needed)
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --ablation clip `
    --generators_root per-gen-dataset-test `
    --clip_cache_dir datasets_eq/clip_cache_test

# Training-set generators (not test)
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --generators_root per-gen-dataset `
    --clip_cache_dir datasets_eq/clip_cache
```

### `--ablation` options

| Value | What is disabled |
|-------|-----------------|
| `clip` | CLIP branch (wavelet + Swin only) |
| `wavelet` | Wavelet branch (CLIP + Swin only) |
| `sfdf` | Cross-attention fusion (simple addition) |
| `swin` | Swin backbone (global average pool instead) |

---

## 9. Calibration

> Picks an honest decision threshold on validation data — no test-set leakage.

```powershell
# Single checkpoint
python calibrate_and_eval.py --checkpoints checkpoints/epoch_020.pt

# Checkpoint ensemble (averages probabilities)
python calibrate_and_eval.py `
    --checkpoints checkpoints/epoch_010.pt `
                  checkpoints/epoch_015.pt `
                  checkpoints/epoch_020.pt
```

```powershell
# Held-out calibration (30% calib / 70% eval split)
python calibrate_heldout.py --checkpoints checkpoints/epoch_020.pt

# Ensemble version
python calibrate_heldout.py `
    --checkpoints checkpoints/epoch_010.pt `
                  checkpoints/epoch_015.pt `
                  checkpoints/epoch_020.pt
```

---

## 10. Inference

```powershell
# Single image
python inference.py `
    --checkpoint checkpoints/best_model.pt `
    --image path/to/image.jpg

# Entire folder
python inference.py `
    --checkpoint checkpoints/best_model.pt `
    --image_dir path/to/folder/

# With GradCAM heatmap overlay (saved as <stem>_heatmap.png)
python inference.py `
    --checkpoint checkpoints/best_model.pt `
    --image path/to/image.jpg `
    --heatmap `
    --output_dir results/inference

# Adjust threshold and force device
python inference.py `
    --checkpoint checkpoints/best_model.pt `
    --image_dir path/to/folder/ `
    --threshold 0.4 `
    --device cpu
```

Output CSV saved to `results/inference/predictions.csv`.

---

## 11. Flask Server (Browser Extension)

```powershell
# Start server with a trained model (default port 5000)
python server.py --checkpoint checkpoints/best_model.pt

# Demo mode — no checkpoint required (returns random results for UI testing)
python server.py --demo

# With GPU
python server.py --checkpoint checkpoints/best_model.pt --device cuda

# Custom host and port
python server.py --checkpoint checkpoints/best_model.pt --host 0.0.0.0 --port 8080

# Using the .venv Python directly (if venv not activated)
.\.venv\Scripts\python.exe server.py --checkpoint checkpoints\best_model.pt
```

Server runs at `http://127.0.0.1:5000` by default.

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if server + model are loaded |
| `/detect` | POST | Single-image detection (+ optional heatmap) |
| `/batch_detect` | POST | Batch detection (no heatmaps) |

---

## 12. Heatmaps

```powershell
# Generate sample heatmaps — 2×3 composite panel (GradCAM + FFT spectrum)
python generate_sample_heatmaps.py --checkpoint checkpoints/pergen_split_best_model.pt

# Using .venv Python directly
.\.venv\Scripts\python.exe generate_sample_heatmaps.py
```

Outputs land in `heatmap/output/`:
- `heatmap_image1-6.png` — GradCAM overlays
- `freq_image7_real.png`, `freq_image8_fake.png` — FFT spectrum panels
- `frequency_composite.png` — full 2×3 composite

---

## 13. TensorBoard

```powershell
# Launch TensorBoard to monitor training
tensorboard --logdir runs/

# Then open in browser:
# http://localhost:6006
```

---

## Quick Reference — Common Workflow

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Cache CLIP features (one-time, ~2–6 h)
python cache_clip_features.py

# 3. Train (AMD CPU — recommended)
.\run_train.ps1

# 4. Monitor in another terminal
tensorboard --logdir runs/

# 5. Evaluate
python evaluate.py --checkpoint checkpoints/best_model.pt

# 6. Per-generator tables
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --generators_root per-gen-dataset-test `
    --clip_cache_dir datasets_eq/clip_cache_test

# 7. Start server for browser extension
python server.py --checkpoint checkpoints/best_model.pt
```
