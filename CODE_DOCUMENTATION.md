# Code Documentation — Transformer Based on Multi-Domain Feature Fusion for AI-Generated Image Detection

This document maps the **Man & Cho (2026)** framework described in the thesis to the
actual implementation in this repository. It covers the five major architectural
components, the training configuration, the training-loop implementation, and the
evaluation metrics.

> **Label convention everywhere:** `0 = Real (human-made)`, `1 = AI-generated`.

---

## 1. Architecture Overview

The detector is a dual-branch, multi-domain feature-fusion network. An input image
is processed by two parallel branches (semantic + frequency), fused by a
cross-attention + gating module, modelled globally by a Swin Transformer, and
classified by a lightweight MLP.

```
Image [B, 3, 224, 224]
   │
   ├─► CLIPExtractor    →  Fs  [B, N, D]     spatial / semantic   (frozen CLIP ViT-L/14)
   ├─► WaveletExtractor →  Ff  [B, N, D]     frequency / texture  (DWT db4 + light CNN)
   │
   ├─► SFDF             →  F_fused [B, N, D] cross-attention alignment + gated fusion
   │
   ├─► reshape          →  [B, D, G, G]      token sequence → 2-D feature map (G = 16)
   │
   ├─► SwinBackbone     →  z  [B, D]         global contextual modelling
   │
   └─► MLP head         →  logit [B, 1]      sigmoid → P(AI-generated)
```

Where `N = G × G = 16 × 16 = 256` tokens and `D = feature_dim = 768`.

> **Implementation note (token count).** The thesis text mentions a 14×14 grid /
> 196 tokens, but with a 224×224 input and a /14 patch size the CLIP ViT-L/14 grid
> is **16×16 = 256 tokens**. The implementation uses 256 tokens consistently across
> both branches so their sequences align in the fusion module.

### File map

| File | Role |
|---|---|
| `models/clip_extractor.py` | Spatial branch — frozen CLIP ViT-L/14 patch tokens |
| `models/wavelet_extractor.py` | Frequency branch — 2-D DWT (db4) + lightweight CNN |
| `models/sfdf.py` | Cross-attention alignment + gated feature fusion |
| `models/swin_backbone.py` | Swin-Tiny global contextual backbone |
| `models/detector.py` | Full model: wires all branches + classifier head |
| `losses/focal_loss.py` | Binary focal loss (paper Eq. 6) |
| `train.py` | Training loop, optimiser, scheduler, early stopping |
| `cache_clip_features.py` | Pre-computes frozen-CLIP tokens to disk (CPU speedup) |
| `utils/metrics.py` | ACC / AP / Recall / F1 / AUC, confusion matrix |
| `evaluate_per_generator.py` | Per-generator Tables 1/2/3 evaluation |
| `configs/default.yaml` | All hyper-parameters |

---

## 2. Component 1 — CLIP Spatial Branch (`models/clip_extractor.py`)

**Paper role:** extract high-level **semantic** features using a frozen CLIP-ViT-L/14
(OpenAI pretrained). The encoder is *locked* (never updated) to leverage its prior
knowledge of semantic inconsistencies; only a small trainable projection adapts CLIP
space → the shared dimension `D`.

### How CLIP is integrated

```python
class CLIPExtractor(nn.Module):
    def __init__(self, model_name="ViT-L-14-quickgelu", pretrained="openai",
                 out_dim=768, load_visual=True):
        if load_visual:
            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained)
            self.visual = clip_model.visual
            clip_dim = self.visual.transformer.width      # 1024 for ViT-L/14
            for p in self.visual.parameters():            # ── FREEZE CLIP ──
                p.requires_grad = False
        else:
            self.visual = None                            # cache-only mode
            clip_dim = 1024
        # Only trainable part of this branch: 1024 → 768 projection
        self.proj = nn.Linear(clip_dim, out_dim, bias=False) if clip_dim != out_dim \
                    else nn.Identity()
```

Key points:
- **Frozen backbone.** `requires_grad = False` on every CLIP parameter — the ViT is
  pure inference; gradients flow only through `self.proj`.
- **Trainable adapter.** The `1024 → 768` linear projection (`self.proj`) is the only
  learnable component, trained at the low `backbone_lr` (1e-5).

### Extracting patch tokens

`_extract_patch_tokens` runs the ViT manually so we can keep the per-patch token
sequence (CLS token dropped):

```python
def _extract_patch_tokens(self, x):
    v = self.visual
    x = v.conv1(x)                              # patch embedding [B, width, 16, 16]
    B, C, gh, gw = x.shape
    x = x.reshape(B, C, -1).permute(0, 2, 1)   # [B, 256, width]
    cls = v.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
    x = torch.cat([cls, x], dim=1) + v.positional_embedding
    x = v.ln_pre(x)
    x = v.transformer(x)                        # batch-first transformer
    return x[:, 1:, :]                          # drop CLS → [B, 256, 1024]
```

> **Critical correctness fix (QuickGELU + batch-first).** Two settings here are
> essential and were the source of a major bug during development:
> 1. **`ViT-L-14-quickgelu`**, not plain `ViT-L-14`. OpenAI's CLIP weights were
>    trained with the QuickGELU activation; loading them into a plain-GELU model
>    silently degrades every feature.
> 2. **No `permute` before the transformer.** This open_clip build is *batch-first*
>    (`[batch, seq, dim]`). An earlier version permuted to `[seq, batch, dim]`,
>    which made the images in a batch attend to *each other* and corrupted all
>    tokens. With the fix, the output matches open_clip's official `visual(x)`
>    forward exactly (cosine similarity = 1.0) and is batch-independent.

### `forward` — live vs. cached

```python
def forward(self, x, precomputed_tokens=None):
    if precomputed_tokens is not None:
        tokens = precomputed_tokens            # cache hit: skip the ViT entirely
    else:
        with torch.no_grad():
            tokens = self._extract_patch_tokens(x)
    return self.proj(tokens)                    # [B, 256, 768]
```

Because the ViT is frozen, its tokens are deterministic per image and can be
**pre-computed once and cached** (see §8). When cached tokens are supplied the heavy
1.2 GB ViT is never loaded or run — the single biggest CPU speedup.

---

## 3. Component 2 — Wavelet Frequency Branch (`models/wavelet_extractor.py`)

**Paper role:** capture subtle, low-level **frequency artifacts** that semantic
methods miss. A 2-D Discrete Wavelet Transform (Daubechies-4, *db4*) decomposes the
image; a lightweight CNN encodes the high-frequency sub-bands into a token sequence
that matches `Fs`.

### Step 1 — Differentiable 2-D DWT (db4)

The DWT is implemented as **fixed (non-learnable) convolutions** using db4 filters,
so it runs on-device with no NumPy round-trips:

```python
class DWT2D(nn.Module):
    def __init__(self, wavelet="db4"):
        w = pywt.Wavelet(wavelet)
        lo = np.array(w.dec_lo[::-1], dtype=np.float32)   # decomposition filters
        hi = np.array(w.dec_hi[::-1], dtype=np.float32)
        # 2-D separable filters via outer products
        self.register_buffer("filt_lh", _2d(hi, lo))      # horizontal high-freq
        self.register_buffer("filt_hl", _2d(lo, hi))      # vertical   high-freq
        self.register_buffer("filt_hh", _2d(hi, hi))      # diagonal   high-freq

    def forward(self, x):                                  # x: [B, 3, H, W]
        bands = []
        for c in range(x.shape[1]):                        # per colour channel R,G,B
            xc = x[:, c:c+1]
            bands.append(self._apply_filter(xc, self.filt_lh))   # stride-2 conv
            bands.append(self._apply_filter(xc, self.filt_hl))
            bands.append(self._apply_filter(xc, self.filt_hh))
        return torch.cat(bands, dim=1)                     # [B, 9, H/2, W/2]
```

- **LL discarded.** Only `LH`, `HL`, `HH` (the high-frequency sub-bands where
  generation artifacts live — ringing, texture regularity) are kept. `LL` (the
  low-frequency approximation) is dropped because it overlaps with what CLIP already
  captures.
- **9 channels.** 3 colour channels × 3 high-freq sub-bands = **9-channel** map at
  half spatial resolution (`H/2 × W/2`).
- `filt_*` are `register_buffer`s (fixed, not parameters) → the DWT itself is **not
  trained**.

### Step 2 — Lightweight CNN encoder (the part that *is* trained)

```python
def _conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class WaveletExtractor(nn.Module):
    def __init__(self, in_channels=3, out_dim=768, grid_size=16):
        self.dwt = DWT2D(wavelet="db4")
        mid = 64
        self.cnn = nn.Sequential(
            _conv_block(in_channels * 3, mid),     #  9 →  64 , /2
            _conv_block(mid, mid * 4),             # 64 → 256 , /2
            _conv_block(mid * 4, out_dim),         # 256 → 768, /2
        )
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))  # match Fs length

    def forward(self, x):
        hf   = self.dwt(x)                         # [B, 9, H/2, W/2]
        feat = self.cnn(hf)                        # [B, 768, h, w]
        feat = self.pool(feat)                     # [B, 768, 16, 16]
        return feat.flatten(2).permute(0, 2, 1)    # Ff: [B, 256, 768]
```

- **3 conv blocks**, each `3×3 conv → BatchNorm → ReLU` with **stride-2** downsampling
  (paper §3.1).
- **Adaptive average pooling** forces the output to a `16×16` grid so the frequency
  token sequence `Ff` has the **same `[B, 256, 768]` shape** as the spatial `Fs`,
  which is required for cross-attention.

> **Implementation note (lightweight).** Intermediate channels are fixed at `mid=64`
> (`9 → 64 → 256 → 768`). The paper does not specify channel counts; the narrow
> middle keeps this branch genuinely lightweight (~4× fewer FLOPs than a wide
> variant) while remaining sufficient for frequency-artifact discrimination.

### How this branch is trained

- The DWT layer is a **fixed buffer** — no gradients.
- The **CNN encoder is the trainable frequency feature extractor**. In the optimiser
  it gets its **own, higher learning rate** (`cnn_lr = 1e-4`), separate from the rest
  of the network (`backbone_lr = 1e-5`) — see §9.
- During training the CNN receives the **augmented** image (random flip / crop / JPEG /
  blur), so it learns artifacts that survive common post-processing — this is what
  gives robustness to compression/resize/blur at test time.

---

## 4. Component 3 — Cross-Attention Alignment + Gating (`models/sfdf.py`)

**Paper role (§3.2):** fuse the spatial (`Fs`) and frequency (`Ff`) features. *Spatial
features are the query; frequency features are the key/value*, so fine-grained
frequency artifacts are aligned to their semantic/spatial locations. A gating module
then dynamically balances the two sources. The module is named **SFDF**
(Spatial-Frequency Dynamic Fusion).

### Stage 1 — Cross-Attention Alignment

```python
# Q from spatial (CLIP), K and V from frequency (Wavelet-CNN)
Q = self._split_heads(self.W_Q(Fs))     # [B, h, N, d_h]
K = self._split_heads(self.W_K(Ff))
V = self._split_heads(self.W_V(Ff))

A_heads = F.scaled_dot_product_attention(Q, K, V,
              dropout_p=self.attn_drop.p if self.training else 0.0)
A = self.out_proj(self._merge_heads(A_heads))      # [B, N, D]
```

- **Multi-head** cross-attention (`num_heads = 8`).
- `Fs → Query`, `Ff → Key/Value` — the spatial branch "asks" the frequency branch
  where the relevant artifacts are. The result `A = Softmax(QKᵀ/√d)·V`.
- Uses PyTorch 2's fused `scaled_dot_product_attention` for speed.
- Pre-norm `LayerNorm` is applied to `Fs` and `Ff` before projection.

### Stage 2 — Gated Feature Integration

```python
if use_gate:
    concat  = torch.cat([Fs, A], dim=-1)            # [B, N, 2D]
    G       = self.sigmoid(self.gate_mlp(concat))   # gate value ∈ (0,1), [B, N, D]
    F_fused = G * Fs + (1.0 - G) * A
else:
    F_fused = Fs + A                                 # ablation: no gating (Clip+F+A)
```

- The gate `G = σ( MLP([Fs ; A]) )` is produced by a **two-layer MLP** on the
  concatenation of the spatial features and the cross-attention output.
- `F_fused = G ⊙ Fs + (1−G) ⊙ A` — per-token, per-channel adaptive weighting between
  the semantic stream and the aligned-frequency stream. The gate acts as an adaptive
  filter that suppresses ambiguous regions and balances the two domains.

```python
self.gate_mlp = nn.Sequential(
    nn.Linear(dim * 2, dim), nn.GELU(), nn.Linear(dim, dim)
)
```

> **Implementation note (gating activation).** The paper describes the gating MLP as
> using "ReLU and sigmoid". The implementation uses **GELU** in the hidden layer and
> **sigmoid** on the output gate. GELU is the smoother, standard choice in transformer
> blocks; the output sigmoid (which produces the gate value) matches the paper.

### Ablation support

`use_gate=False` reduces fusion to `Fs + A` (the **Clip+F+A** ablation). The
`detector.py` `forward` selects the fusion path by ablation mode (see §6), enabling
all four paper variants (Clip / Clip+F / Clip+F+A / Clip+F+A+G) from one model.

---

## 5. Component 4 — Swin Transformer Backbone (`models/swin_backbone.py`)

**Paper role (§3.3):** the main feature-learning backbone. The fused features are
patch-partitioned and processed through four stages of Swin blocks using
Window-based (W-MSA) and Shifted-Window (SW-MSA) self-attention to learn both local
and global patterns.

```python
from timm.models.swin_transformer import SwinTransformer

class SwinBackbone(nn.Module):
    def __init__(self, in_dim=768, grid_size=16, window_size=4, embed_dim=96,
                 depths=[2,2,6,2], num_heads=[3,6,12,24], out_dim=768):
        # 1×1 conv projection: fused dim → Swin embedding dim
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim), nn.GELU())
        self.swin = SwinTransformer(
            img_size=grid_size, patch_size=1, in_chans=embed_dim,
            num_classes=0,                       # drop built-in classifier
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=4.0, qkv_bias=True,
            drop_path_rate=0.1, global_pool="avg")
        swin_out = embed_dim * (2 ** (len(depths)-1))    # 96 × 8 = 768
        self.head_proj = nn.Linear(swin_out, out_dim, bias=False) \
                         if swin_out != out_dim else nn.Identity()

    def forward(self, F_fused):                  # [B, 768, 16, 16]
        x = self.input_proj(F_fused)             # [B, 96, 16, 16]
        z = self.swin(x)                         # [B, 768]  (global avg pool inside)
        return self.head_proj(z)
```

- **Swin-Tiny configuration:** `depths=[2,2,6,2]`, `num_heads=[3,6,12,24]`,
  `embed_dim=96`. After 4 stages the channel dimension is `96 × 2³ = 768`.
- **`patch_size=1`.** The input is already a `16×16` token grid (not raw pixels), so
  patch size 1 preserves resolution; the four stages still halve spatial size and
  double channels as in standard Swin. W-MSA / SW-MSA, LayerNorm, MLP and residual
  connections are all internal to the timm `SwinTransformer`.
- **`window_size=4`** divides the `16×16` grid evenly.
- **Global average pooling** (`global_pool="avg"`) produces the single `768`-d vector
  `z` passed to the classifier.

### Token sequence → 2-D map (in `detector.py`)

Swin needs a 2-D feature map, so the fused token sequence is reshaped first:

```python
B, N, D = F_fused_seq.shape
G = self.grid_size                                       # 16
F_fused = F_fused_seq.permute(0, 2, 1).reshape(B, D, G, G)   # [B, 768, 16, 16]
z = self.backbone(F_fused)                              # [B, 768]
```

---

## 6. Component 5 — Full Detector + Classifier Head (`models/detector.py`)

`AIGCDetector` wires the branches together and exposes the **ablation switch**.

### Forward pass (ablation-aware)

```python
def forward(self, x, clip_tokens=None, return_features=False):
    Fs = self.clip_extractor(x, precomputed_tokens=clip_tokens)   # [B, N, D]

    abl = self.ablation
    if abl == "clip":                                  # Ours (Clip)
        F_fused_seq = Fs
    elif abl == "clip_f":                              # Ours (Clip+F): simple add
        Ff = self.wavelet_extractor(x)
        F_fused_seq = Fs + Ff
    elif abl == "clip_f_a":                            # Ours (Clip+F+A): attn, no gate
        Ff = self.wavelet_extractor(x)
        F_fused_seq = self.sfdf(Fs, Ff, use_gate=False)
    else:                                              # Ours (Clip+F+A+G): full model
        Ff = self.wavelet_extractor(x)
        F_fused_seq = self.sfdf(Fs, Ff, use_gate=True)

    B, N, D = F_fused_seq.shape
    F_fused = F_fused_seq.permute(0, 2, 1).reshape(B, D, self.grid_size, self.grid_size)
    z = self.backbone(F_fused)                         # Swin → [B, D]
    logits = self.classifier(z)                        # [B, 1]
    return logits
```

The four branches of the `if/elif` correspond exactly to the paper's four ablation
configurations, so a single checkpoint can be evaluated in any mode (see
`evaluate_per_generator.py --ablation`).

### Classification head (lightweight MLP)

```python
self.classifier = nn.Sequential(
    nn.Linear(feature_dim, feature_dim // 2),   # 768 → 384
    nn.GELU(),
    nn.Dropout(dropout),                        # dropout = 0.3
    nn.Linear(feature_dim // 2, 1),             # 384 → 1  (logit)
)
```

A two-layer MLP (`768 → 384 → 1`) with GELU and dropout produces a single logit;
`sigmoid(logit)` is the probability of *AI-generated*.

### Cache-aware construction (`build_detector`)

```python
def build_detector(cfg, force_load_visual=False, ablation=None):
    # If a CLIP cache exists, set load_visual=False → the frozen ViT is never loaded.
    ...
    return AIGCDetector(clip_model=m.get("clip_model", "ViT-L-14-quickgelu"),
                        feature_dim=..., dropout=m.get("dropout", 0.1),
                        load_visual=load_visual, ablation=effective_ablation)
```

---

## 7. Training Configuration (`configs/default.yaml`)

All hyper-parameters live in one YAML. Annotated:

```yaml
model:
  clip_model: "ViT-L-14-quickgelu"  # frozen CLIP backbone (correct activation for
                                    # OpenAI weights — see §2 fix)
  feature_dim: 768                  # shared embedding dimension D
  swin_window_size: 4               # Swin window (divides the 16×16 grid)
  swin_depths: [2, 2, 6, 2]         # Swin-Tiny stage depths
  swin_num_heads: [3, 6, 12, 24]    # Swin-Tiny heads per stage
  swin_embed_dim: 96                # Swin-Tiny base channels
  dropout: 0.3                      # dropout in SFDF + classifier head

data:
  image_size: 224
  dfdc_root:        "datasets_eq/DFDC"          # equalized training data (see §11)
  forensynths_root: "datasets_eq/ForenSynths"
  genimage_root:    "datasets_eq/GenImage"
  clip_cache_dir:   "datasets_eq/clip_cache"    # pre-computed frozen-CLIP tokens
  train_split: 0.8                  # 80/20 train/val split
  num_workers: 0                    # 0 on Windows/CPU (avoids subprocess overhead)
  augmentation:
    random_flip: true
    random_crop: true
    jpeg_compression: true          # random JPEG quality 50–100

training:
  batch_size: 8                     # per-step batch
  accumulation_steps: 4             # gradient accumulation → effective batch = 32
  epochs: 100                       # early stopping ends it sooner
  early_stop_patience: 15           # stop if val AUC flat for 15 epochs
  cnn_lr: 1.0e-4                    # lightweight wavelet-CNN branch (higher LR)
  backbone_lr: 1.0e-5              # Swin + CLIP projection adapter (lower LR)
  weight_decay: 1.0e-4
  focal_gamma: 2.0                  # focal-loss focusing parameter γ (paper Eq. 6)
  scheduler: "cosine"
  warmup_epochs: 5                  # linear warmup before cosine decay
  gradient_clip: 1.0
  use_amp: false                    # bf16 autocast (off by default; AMD-CPU caveat)
  use_compile: false                # torch.compile (optional speedup)

logging:
  save_dir: "checkpoints/"
  save_every: 5                     # periodic checkpoint cadence
  eval_every: 1                     # validate every epoch
```

**Data augmentation** (training only) — implemented in `data/dataset.py`
`build_transforms`:

```python
ops = [T.RandomHorizontalFlip(),
       T.RandomResizedCrop(224, scale=(0.6, 1.0)),
       T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.3)]
if jpeg_compress:
    ops.append(RandomJPEGCompression(min_quality=50))     # random JPEG 50–100
# always: Resize/CenterCrop (eval) + ToTensor + CLIP-normalisation
```

Augmentation is applied only during training and feeds the wavelet branch; it makes
the frequency features robust to JPEG compression, resizing, and blurring (the paper's
robustness tests). The cached CLIP tokens always come from the clean image.

---

## 8. CLIP Feature Cache (`cache_clip_features.py`)

Because the CLIP ViT is frozen, its tokens are constant per image and can be computed
once and reused — turning every training epoch from "run a 1.2 GB ViT on every image"
into a memory lookup.

```python
tokens = extractor._extract_patch_tokens(imgs_t)   # [B, 256, 1024]
torch.save({"paths": paths_out, "features": tokens.to(torch.float16)}, cache_path)
```

- Stored as `{name}_clip.pt` (training) and `pergen_{generator}_clip.pt` (eval),
  each a dict of `paths: list[str]` + `features: Tensor[N, 256, 1024]` in float16.
- At train/eval time, `ClipFeatureCache` (in `data/dataset.py`) serves tokens by image
  path, and `build_detector` sets `load_visual=False` so the ViT is never loaded.
- **Verification step (important):** after building, a cached feature must match a
  fresh live forward (cosine ≈ 1.0). This catches the batch/activation bug described
  in §2 before any training is wasted on corrupt features.

---

## 9. Training-Loop Implementation (`train.py`)

### Loss — Binary Focal Loss (paper Eq. 6)

```python
class BinaryFocalLoss(nn.Module):          # γ = 2, no alpha term (matches Eq. 6)
    def forward(self, logits, targets):
        logits, targets = logits.squeeze(-1), targets.float()
        p   = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        return ((1 - p_t).pow(self.gamma) * bce).mean()
```

Focal loss down-weights easy examples (`(1−p_t)^γ`) so training focuses on the hard,
near-boundary images.

### Optimiser — two parameter groups (paper §4.2)

```python
def build_optimizer(model, cfg):
    cnn_params   = list(model.wavelet_extractor.cnn.parameters())   # lr = 1e-4
    other_params = [p for p in model.parameters()
                    if id(p) not in {id(q) for q in cnn_params} and p.requires_grad]
    return AdamW([{"params": cnn_params,   "lr": 1e-4},
                  {"params": other_params, "lr": 1e-5}], weight_decay=1e-4)
```

- The **wavelet CNN** trains faster (`1e-4`) since it learns from scratch.
- The **Swin backbone + CLIP projection** use a gentler `1e-5` (adapting/fine work).
- The **frozen CLIP ViT** has no trainable params, so it never appears here.

### Scheduler — linear warmup → cosine annealing

```python
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                  total_iters=warmup_epochs * opt_steps_per_epoch)
cosine = CosineAnnealingLR(optimizer, T_max=(epochs-warmup_epochs)*opt_steps_per_epoch,
                           eta_min=1e-7)
scheduler = SequentialLR(optimizer, [warmup, cosine],
                         milestones=[warmup_epochs * opt_steps_per_epoch])
```

5 epochs of linear warmup (10% → 100% of base LR), then cosine decay to ~0. Steps are
counted in **optimizer steps** (after gradient accumulation), not loader iterations.

### One epoch — gradient accumulation

```python
def train_one_epoch(model, loader, optimizer, scheduler, criterion, ...):
    model.train(); optimizer.zero_grad()
    for i, batch in enumerate(loader):
        images, labels, clip_tokens = _unpack_batch(batch)   # 2- or 3-tuple
        logits = model(images, clip_tokens=clip_tokens)
        loss   = criterion(logits, labels) / accum_steps      # normalise
        loss.backward()
        if (i + 1) % accum_steps == 0 or last_batch:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # clip = 1.0
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
```

- **Gradient accumulation** (`accum_steps = 4`) gives an effective batch of 32 while
  only holding 8 images in memory per step.
- **Gradient clipping** at norm 1.0 stabilises training.
- `_unpack_batch` transparently supports cached (`img, clip_tokens, label`) and
  non-cached (`img, label`) batches.

### Validation, checkpointing, early stopping (per epoch)

```python
y_true, y_prob = evaluate(model, val_loader, device, autocast_ctx)
metrics = compute_all_metrics(y_true, y_prob)
if metrics["auc"] > best_auc:                       # save best by val AUC
    best_auc = metrics["auc"]
    torch.save({...}, save_dir / "best_model.pt")
if early_stop.step(metrics["auc"]):                 # patience = 15
    break
```

- **Model selection** is by **validation AUC** — `best_model.pt` is overwritten
  whenever AUC improves.
- **Early stopping** ends training when val AUC has not improved by `>1e-4` for 15
  consecutive epochs.
- Periodic `epoch_XXX.pt` checkpoints are saved every 5 epochs for resuming
  (`train.py --resume`).

---

## 10. Evaluation Metrics (`utils/metrics.py`)

The paper reports **Accuracy (ACC)** and **Average Precision (AP)**; the code also
tracks Recall, F1, and AUC-ROC for monitoring.

```python
def compute_all_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "acc":    accuracy_score(y_true, y_pred),
        "ap":     average_precision_score(y_true, y_prob),   # paper's AP
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1":     f1_score(y_true, y_pred, zero_division=0),
        "auc":    roc_auc_score(y_true, y_prob),             # model-selection metric
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]),
        ...
    }
```

- **ACC** — fraction correct at threshold 0.5.
- **AP** — area under the precision–recall curve, computed from the continuous
  probabilities (threshold-independent); this is the paper's headline metric alongside
  ACC.
- **AUC-ROC** — used to select the best checkpoint during training.
- **Confusion matrix** — printed each validation with TN/FP/FN/TP and plain-language
  labels (`utils/metrics.py::print_confusion_matrix`).

### Per-generator evaluation (`evaluate_per_generator.py`)

Reproduces the paper's **Table 1 (DFDC), Table 2 (ForenSynths GANs), Table 3
(GenImage diffusion)**. It evaluates each generator's folder independently, reports
`ACC / AP` per generator plus per-table means, and writes `tables.txt` + `results.csv`.
It can evaluate any ablation mode from a single checkpoint via `--ablation`, and reads
the per-generator CLIP cache via `--clip_cache_dir`.

```powershell
python evaluate_per_generator.py `
    --checkpoint checkpoints/best_model.pt `
    --generators_root per-gen-dataset `
    --clip_cache_dir datasets_eq/clip_cache
```

---

## 11. Key Implementation Decisions & Fixes (development notes)

These are implementation realities that differ from or extend the paper, documented
for reproducibility:

1. **CLIP activation + batch-first extraction (correctness).** Use
   `ViT-L-14-quickgelu` and do **not** permute before the transformer. Both are
   required for the CLIP features to match the official model (cosine = 1.0). The
   original code violated both and produced corrupted, batch-contaminated features,
   which capped cross-generator accuracy at chance until fixed.

2. **CLIP feature caching.** Frozen-ViT tokens are pre-computed to disk
   (`cache_clip_features.py`) and looked up by path at train/eval time — the main
   speedup on CPU. Caches must be verified (cosine ≈ 1.0 vs. live) before use.

3. **Data equalization (`equalize_training_data.py`).** The raw training sets had a
   real/fake *source shortcut* (e.g., GenImage real = 640×480 JPG vs. fake = 128×128
   PNG). All training images are re-processed through one identical pipeline
   (resize→center-crop→224, single PNG format) so the only thing distinguishing real
   from fake is the generation artifact, not resolution/format. Originals are kept;
   cleaned copies live in `datasets_eq/`.

4. **Data-leakage check (`check_leakage.py`).** MD5 + perceptual-hash comparison
   between the training set and the per-generator evaluation set ensures no evaluation
   image appears in training (held-out test integrity).

5. **Dropout = 0.3** (config), applied in both the SFDF attention and the classifier
   head, for regularisation.

6. **Token count = 256** (16×16), not 196 — consistent with a 224×224 input and the
   ViT-L/14 patch size (see §1).

7. **Wavelet CNN mid-channels = 64** — an unspecified-by-paper choice that keeps the
   frequency branch lightweight.

8. **Lightweight linear-probe variant (`lite_head.py`).** A diagnostic/auxiliary head
   (mean-pooled CLIP tokens → standardise → linear classifier) used to verify the
   features carry cross-generator signal independently of the heavy Swin head.

---

## 12. System Integration

The five components are not standalone — they are wired into one end-to-end network
(`models/detector.py::AIGCDetector`) and trained jointly. This section describes how
they connect and summarises the parameters of each.

### 12.1 Integration data flow (with tensor shapes)

```
Input image  x : [B, 3, 224, 224]
   │
   ├── Spatial branch ───────────────────────────────────────────────
   │     CLIP ViT-L/14 (frozen) ─► patch tokens   [B, 256, 1024]
   │     Linear projection       ─► Fs            [B, 256, 768]
   │
   ├── Frequency branch ─────────────────────────────────────────────
   │     DWT db4 (fixed)          ─► 9-ch HF map   [B, 9, 112, 112]
   │     Lightweight CNN          ─► feature map   [B, 768, h, w]
   │     AdaptiveAvgPool + flatten─► Ff            [B, 256, 768]
   │
   ├── SFDF fusion ──────────────────────────────────────────────────
   │     Cross-attention (Fs=Q, Ff=K,V) ─► A       [B, 256, 768]
   │     Gated integration              ─► F_fused [B, 256, 768]
   │
   ├── Sequence → 2-D map  (permute + reshape)     [B, 768, 16, 16]
   │
   ├── Swin-Tiny backbone (W-MSA/SW-MSA, 4 stages) ─► z  [B, 768]
   │
   └── MLP classifier head ─► logit [B, 1] ─► sigmoid ─► P(AI) ∈ (0,1)
```

### 12.2 Integration contracts (why the pieces fit)

- **Token-length alignment.** The cross-attention in SFDF requires `Fs` and `Ff` to
  have the *same* sequence length `N`. The CLIP branch produces a 16×16 = 256 token
  grid; the wavelet branch's `AdaptiveAvgPool2d((16, 16))` forces its output to the
  same 256 tokens. This is the key shape contract that lets the two domains fuse.
- **Shared dimension `D = 768`.** Every branch emits `D = 768` per token (CLIP via its
  `1024→768` projection; wavelet CNN's final conv outputs 768; SFDF preserves `D`), so
  fusion and the Swin input are dimension-compatible.
- **Sequence ↔ map conversion.** SFDF outputs a token *sequence* `[B, 256, 768]`; the
  Swin backbone needs a 2-D *map*, so `detector.forward` permutes/reshapes to
  `[B, 768, 16, 16]` before the backbone.
- **Ablation routing.** A single `ablation` flag selects the fusion path
  (`clip` / `clip_f` / `clip_f_a` / full) inside `forward`, so all four paper variants
  share one set of weights and one checkpoint.
- **CLIP-cache integration.** `clip_tokens` can be injected at `forward(x, clip_tokens=…)`;
  when present the frozen ViT is bypassed and only the trainable head runs. The same
  hook powers training (cached tokens), evaluation, and the heat map.
- **Training integration (dual learning rates).** The optimiser groups parameters by
  component: the from-scratch **wavelet CNN** trains at `1e-4`, while the adapting
  **Swin + CLIP projection + SFDF + head** train at `1e-5`; the **frozen CLIP ViT**
  contributes no gradients.

### 12.3 Parameter summary per component

Measured from the built model (`configs/default.yaml`, `feature_dim = 768`):

| Component | Module | Total params | Trainable | State | LR group | Output shape |
|---|---|---:|---:|---|---|---|
| Spatial — CLIP backbone | `clip_extractor.visual` | 303,966,208 | 0 | **frozen** | — | `[B, 256, 1024]` |
| Spatial — projection | `clip_extractor.proj` | 786,432 | 786,432 | trainable | `backbone_lr` 1e-5 | `[B, 256, 768]` |
| Frequency — DWT (db4) | `wavelet_extractor.dwt` | 0 | 0 | **fixed buffers** | — | `[B, 9, 112, 112]` |
| Frequency — CNN encoder | `wavelet_extractor.cnn` | 1,924,288 | 1,924,288 | trainable | `cnn_lr` 1e-4 | `[B, 256, 768]` |
| Fusion — SFDF | `sfdf` | 4,133,376 | 4,133,376 | trainable | `backbone_lr` 1e-5 | `[B, 256, 768]` |
| Backbone — Swin-Tiny | `backbone` | 27,579,402 | 27,579,402 | trainable | `backbone_lr` 1e-5 | `[B, 768]` |
| Head — MLP classifier | `classifier` | 295,681 | 295,681 | trainable | `backbone_lr` 1e-5 | `[B, 1]` |
| **TOTAL** | | **338,685,387** | **34,719,179** | — | — | — |

- **Only 10.25% of parameters are trainable** (34.7 M of 338.7 M). The remaining
  ~304 M are the frozen CLIP ViT — the model leverages CLIP's prior knowledge rather
  than re-learning it, which is the source of its data efficiency.
- The **Swin-Tiny backbone dominates the trainable budget** (27.6 M ≈ 79% of trainable
  params); the wavelet CNN and SFDF fusion are deliberately lightweight.

---

## 13. Region-of-Interest Heat Map (`utils/visualization.py`)

**Paper reference:** Man & Cho (2026) Figure 6 — *"Visual analysis of model attention
focuses for real images and AI-generated fake images"*. The heat map highlights the
regions the model relies on for its real/fake decision.

### 13.1 What it visualizes and where it is taken

The CAM is computed at the **post-fusion projection** (`backbone.input_proj`), i.e. on
the **fused spatial-frequency feature map `F_fused`**. This is the natural place for the
heat map: it shows where the *combined* semantic (CLIP) and frequency (wavelet) evidence
concentrates — the same "regions of interest" the paper reports.

### 13.2 Two methods

| Method | Function | How it works | Cost (cached tokens, CPU) |
|---|---|---|---|
| **Grad-CAM** (default, faithful) | `GradCAM(model)(x, clip_tokens=…)` | GAP of gradients × activations of `F_fused`, ReLU, upsample | ~**0.8 s** |
| **Fast activation map** | `activation_heatmap(model, x, clip_tokens=…)` | per-token L2 magnitude of `F_fused`, forward-only (no backprop) | ~**0.16 s** |

Grad-CAM is gradient-based and matches the paper's saliency; the fast variant is a
forward-only proxy (~5× faster) for interactive/real-time use.

### 13.3 Why it is fast (the two optimizations)

The frozen CLIP ViT (~304 M params) is the only heavy part. The heat map avoids paying
for it twice:

1. **The ViT is kept out of the autograd graph.** Tokens are extracted once under
   `torch.no_grad()`, then fed to the model as `clip_tokens`. The backward pass then
   only traverses the ~34.7 M trainable head — not 304 M frozen parameters. (The
   earlier implementation ran the ViT *inside* the graph with `image.requires_grad_()`,
   which built and back-propagated through the full ViT — slow and memory-heavy.)
2. **The ViT can be skipped entirely.** If pre-computed `clip_tokens` are supplied
   (e.g. from the CLIP cache, or computed once and reused), the heat map costs only a
   tiny head forward (+backward for Grad-CAM) — the timings above.

For a brand-new image with no cache, the unavoidable cost is **one** ViT forward
(in `no_grad`); everything after it is cheap.

### 13.4 Usage

```python
from utils import GradCAM, activation_heatmap, generate_heatmap
from data.dataset import build_transforms

tf = build_transforms(image_size=224, augment=False)

# One-call overlay (Man & Cho Figure 6 style) — returns a blended PIL image
overlay = generate_heatmap(model, image_pil, tf, device, method="gradcam")
overlay.save("roi_heatmap.png")

# Fast forward-only variant for interactive use
overlay = generate_heatmap(model, image_pil, tf, device, method="fast")

# Maximum speed: pass cached CLIP tokens to skip the ViT entirely
overlay = generate_heatmap(model, image_pil, tf, device,
                           method="fast", clip_tokens=cached_tokens)

# Low-level: get the raw [H,W] heat map in [0,1]
x   = tf(image_pil).unsqueeze(0).to(device)
cam = GradCAM(model)(x)                 # ndarray (224, 224), values in [0,1]
```

To reproduce the paper's side-by-side (real vs. AI) figure, call `generate_heatmap`
on a real image and an AI-generated image and place the two overlays next to each
other.

### 13.5 Server integration

`server.py` exposes the heat map via the `/detect` endpoint (`generate_heatmap: true`
in the request body) and returns it as a base64 data-URL (`heatmap_overlay`). The
server builds the detector with `force_load_visual=True` so the ViT is available to
extract tokens for arbitrary uploaded images. For batch/offline use over the cached
datasets, pass `clip_tokens` to get the near-instant timings above.
