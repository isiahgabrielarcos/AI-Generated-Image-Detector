"""
detector.py
───────────
Full Man & Cho (2026) AIGC detector.

Data flow:
  Image [B,3,224,224]
      │
      ├─ CLIPExtractor   →  Fs [B, N, D]   (spatial / semantic)
      ├─ WaveletExtractor→  Ff [B, N, D]   (frequency / texture)
      │
      ├─ SFDF            →  F_fused [B, N, D]
      │
      ├─ reshape         →  [B, D, G, G]
      │
      ├─ SwinBackbone    →  z [B, D]
      │
      └─ MLP head        →  logit [B, 1]   (sigmoid → ŷ ∈ (0,1))
              0 = Real,  1 = AI-generated

CPU cache mode
──────────────
When clip_cache_dir is configured and cache files exist, the frozen CLIP ViT
is never loaded or run.  build_detector() auto-detects the cache and sets
load_visual=False in CLIPExtractor, saving ~1.2 GB RAM and the entire ViT
forward pass per batch.  The trainable 1024→768 projection still runs.
"""

from pathlib import Path

import torch
import torch.nn as nn

from .clip_extractor import CLIPExtractor
from .wavelet_extractor import WaveletExtractor
from .sfdf import SFDF
from .swin_backbone import SwinBackbone


class AIGCDetector(nn.Module):
    """
    Args:
        clip_model      : CLIP model tag  (default 'ViT-L-14')
        clip_pretrained : CLIP weights    (default 'openai')
        feature_dim     : shared embedding dimension D
        grid_size       : spatial grid of token sequence (CLIP ViT-L/14 → 16)
        swin_window     : Swin window size (must divide grid_size)
        swin_depths     : Swin stage depths
        swin_heads      : Swin attention heads per stage
        swin_embed      : Swin base channels
        dropout         : dropout in SFDF
        load_visual     : if False, skip loading frozen CLIP ViT weights
                          (use when running with pre-cached CLIP features)
        ablation        : ablation variant string (None = full model):
                          "clip"      → Ours (Clip):      CLIP only, no wavelet/SFDF
                          "clip_f"    → Ours (Clip+F):    CLIP + Wavelet, simple addition
                          "clip_f_a"  → Ours (Clip+F+A):  + cross-attention, no gating
                          None/"full" → Ours (Clip+F+A+G): full model
    """

    def __init__(
        self,
        clip_model: str = "ViT-L-14-quickgelu",
        clip_pretrained: str = "openai",
        feature_dim: int = 768,
        grid_size: int = 16,
        swin_window: int = 4,
        swin_depths: list = None,
        swin_heads: list = None,
        swin_embed: int = 96,
        dropout: float = 0.1,
        load_visual: bool = True,
        ablation: str = None,
    ):
        super().__init__()
        swin_depths = swin_depths or [2, 2, 6, 2]
        swin_heads  = swin_heads  or [3, 6, 12, 24]

        valid = {None, "full", "clip", "clip_f", "clip_f_a"}
        if ablation not in valid:
            raise ValueError(f"ablation must be one of {valid}, got {ablation!r}")
        self.ablation = ablation

        # ── Dual-branch feature extractors ───────────────────────────
        self.clip_extractor = CLIPExtractor(
            model_name=clip_model,
            pretrained=clip_pretrained,
            out_dim=feature_dim,
            load_visual=load_visual,
        )
        # Wavelet branch not used in clip-only ablation, but we still
        # instantiate it so checkpoint keys remain consistent.
        self.wavelet_extractor = WaveletExtractor(
            in_channels=3,
            out_dim=feature_dim,
            grid_size=grid_size,
        )

        # ── Cross-domain fusion ───────────────────────────────────────
        # Instantiated for all modes; only called for clip_f_a and full.
        self.sfdf = SFDF(dim=feature_dim, num_heads=8, dropout=dropout)

        # ── Global contextual modelling ───────────────────────────────
        self.backbone = SwinBackbone(
            in_dim=feature_dim,
            grid_size=grid_size,
            window_size=swin_window,
            embed_dim=swin_embed,
            depths=swin_depths,
            num_heads=swin_heads,
            out_dim=feature_dim,
        )

        # ── Classification head ───────────────────────────────────────
        #  z_pool → MLP → sigmoid  (Eq. 5 in paper)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, 1),
        )

        self.grid_size = grid_size
        self.feature_dim = feature_dim

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        clip_tokens: torch.Tensor = None,
        return_features: bool = False,
    ):
        """
        Args:
            x              : [B, 3, 224, 224] – normalised input images
            clip_tokens    : [B, N, 1024] pre-cached raw CLIP patch tokens.
                             When provided, the frozen ViT is skipped.
            return_features: if True also return F_fused for visualisation

        Returns:
            logits   : [B, 1] raw logits  (sigmoid → probability)
            F_fused  : (optional) [B, D, G, G] for GradCAM / attention maps
        """
        # ── 1. Feature extraction ────────────────────────────────────
        # [DEMO] Fs = "the eyes" (clip_extractor.py) — runs unconditionally,
        # every ablation variant uses CLIP as its baseline.
        Fs = self.clip_extractor(x, precomputed_tokens=clip_tokens)  # [B, N, D]

        # ── 2. Cross-domain fusion (ablation-aware) ──────────────────
        # [DEMO] This if/elif IS the entire ablation study (Table 1-3 in the
        # writeup, configs/ablation_*.yaml). Each branch is a literal on/off
        # switch for one architectural component — nothing else changes.
        # Walking through these four branches live is the fastest way to
        # show "what does each piece actually contribute".
        abl = self.ablation
        if abl == "clip":
            # Ours (Clip): spatial features only — no Wavelet, no SFDF.
            F_fused_seq = Fs
        elif abl == "clip_f":
            # Ours (Clip+F): naive addition, no cross-attention/gating.
            # [DEMO] THIS is the variant that scores worse than CLIP alone —
            # proof that Fs and Ff conflict without SFDF to mediate them.
            Ff = self.wavelet_extractor(x)
            F_fused_seq = Fs + Ff
        elif abl == "clip_f_a":
            # Ours (Clip+F+A): cross-attention added, gating still off.
            Ff = self.wavelet_extractor(x)
            F_fused_seq = self.sfdf(Fs, Ff, use_gate=False)
        else:
            # Ours (Clip+F+A+G): full model (default) — the heart (sfdf.py)
            # fully engaged with both cross-attention AND gating.
            Ff = self.wavelet_extractor(x)
            F_fused_seq = self.sfdf(Fs, Ff, use_gate=True)

        # Reshape token sequence → 2-D spatial feature map
        B, N, D = F_fused_seq.shape
        G = self.grid_size
        assert N == G * G, f"Token count {N} != grid_size^2 {G*G}"
        F_fused = F_fused_seq.permute(0, 2, 1).reshape(B, D, G, G)  # [B,D,G,G]

        # ── 3. Swin backbone ─────────────────────────────────────────
        # [DEMO] z = "the brain" (swin_backbone.py) — global reasoning over
        # the fused map, regardless of which ablation branch produced it.
        z = self.backbone(F_fused)          # [B, D]

        # ── 4. Classification head ───────────────────────────────────
        # [DEMO] Final decision: one number, squashed by sigmoid elsewhere
        # (see predict() below) into P(AI-generated) ∈ (0,1).
        logits = self.classifier(z)         # [B, 1]

        if return_features:
            return logits, F_fused
        return logits

    # ─────────────────────────────────────────────────────────────────
    def predict(self, x: torch.Tensor, clip_tokens: torch.Tensor = None) -> dict:
        """
        Convenience method for inference.  Returns a dict with:
            probability : float in (0,1)   – P(AI-generated)
            prediction  : 'AI-Generated' | 'Real'
            confidence  : float in (0,1)   – max(p, 1-p)
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, clip_tokens=clip_tokens)
            prob = torch.sigmoid(logits).squeeze(-1)  # [B]

        results = []
        for p in prob.cpu().tolist():
            label = "AI-Generated" if p >= 0.5 else "Real"
            results.append({
                "probability": p,
                "prediction": label,
                "confidence": max(p, 1.0 - p),
            })
        return results[0] if len(results) == 1 else results


# ──────────────────────────────────────────────────────────────────────
#  Factory helper
# ──────────────────────────────────────────────────────────────────────

def build_detector(cfg: dict, force_load_visual: bool = False,
                   ablation: str = None) -> AIGCDetector:
    """
    Build detector from a loaded YAML config dict.

    Auto-detects clip_cache_dir: if cache files exist, sets load_visual=False
    so the 1.2 GB frozen ViT is never loaded during training.

    Args:
        cfg              : loaded YAML config dict
        force_load_visual: if True, always load the CLIP ViT regardless of
                           cache. Use this in inference.py where images are
                           arbitrary and not guaranteed to be in the cache.
    """
    m = cfg.get("model", {})
    d = cfg.get("data", {})

    # Auto-detect: skip loading CLIP ViT if cache files are present
    cache_dir = d.get("clip_cache_dir", None)
    load_visual = True
    if not force_load_visual and cache_dir:
        cache_path = Path(cache_dir)
        if cache_path.exists() and any(cache_path.glob("*_clip.pt")):
            load_visual = False
            print(f"[detector] CLIP cache found at '{cache_dir}' -> load_visual=False "
                  f"(ViT skipped, saves ~1.2 GB RAM)")

    if force_load_visual and not load_visual:
        print("[detector] force_load_visual=True: loading CLIP ViT regardless of cache")

    # ablation can come from the config file OR be overridden by the caller
    cfg_ablation = m.get("ablation", None)
    effective_ablation = ablation if ablation is not None else cfg_ablation

    return AIGCDetector(
        clip_model=m.get("clip_model", "ViT-L-14-quickgelu"),
        clip_pretrained="openai",
        feature_dim=m.get("feature_dim", 768),
        grid_size=16,
        swin_window=m.get("swin_window_size", 4),
        swin_depths=m.get("swin_depths", [2, 2, 6, 2]),
        swin_heads=m.get("swin_num_heads", [3, 6, 12, 24]),
        swin_embed=m.get("swin_embed_dim", 96),
        dropout=m.get("dropout", 0.1),
        load_visual=load_visual,
        ablation=effective_ablation,
    )
