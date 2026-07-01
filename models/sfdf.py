"""
sfdf.py  –  Spatial-Frequency Cross-Domain Feature Fusion
──────────────────────────────────────────────────────────
Implements paper §3.2 exactly:

  Stage 1 – Cross-Attention Alignment
      Q  = Fs @ W_Q
      K  = Ff @ W_K
      V  = Ff @ W_V
      A  = Softmax(QKᵀ / √D) V

  Stage 2 – Gated Feature Integration
      G        = σ( MLP([Fs ; A]) )
      F_fused  = G ⊙ Fs + (1-G) ⊙ A

[DEMO] THIS FILE IS THE HEART OF THE SYSTEM.
CLIP (clip_extractor.py) and Wavelet (wavelet_extractor.py) each produce a
strong feature stream on their own, but naively adding them together
(Fs + Ff) makes accuracy WORSE than CLIP alone — our own ablation results
show this (Clip-only 84.7% mean ACC vs Clip+F 77.6%, see Table 1-3 results).
The two feature domains conflict with each other. Cross-attention (Stage 1)
+ gating (Stage 2) below is the only component whose presence/absence flips
the model from "branches fighting" to "branches cooperating" — that's why
this, not CLIP or Swin, is the component to point to when asked "what makes
this work".
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SFDF(nn.Module):
    """
    Spatial-Frequency Cross-Domain Feature Fusion.

    Args:
        dim        : shared feature dimension D
        num_heads  : number of attention heads (multi-head cross-attention)
        dropout    : attention dropout
    """

    def __init__(self, dim: int = 768, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        # ── Projection matrices ───────────────────────────────────────
        self.W_Q = nn.Linear(dim, dim, bias=False)
        self.W_K = nn.Linear(dim, dim, bias=False)
        self.W_V = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.attn_drop = nn.Dropout(dropout)

        # ── Gating MLP  [Fs ; A] → dim*2 → dim → dim ─────────────────
        self.gate_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim, bias=True),
            nn.GELU(),
            nn.Linear(dim, dim, bias=True),
        )
        self.sigmoid = nn.Sigmoid()

        # ── Layer norms (pre-norm style) ───────────────────────────────
        self.norm_s = nn.LayerNorm(dim)
        self.norm_f = nn.LayerNorm(dim)

    # ─────────────────────────────────────────────────────────────────
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, N, D] → [B, h, N, d_h]"""
        B, N, D = x.shape
        x = x.reshape(B, N, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, h, N, d_h] → [B, N, D]"""
        B, h, N, dh = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, N, h * dh)
        return x

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        Fs: torch.Tensor,   # [B, N, D]  spatial (CLIP)
        Ff: torch.Tensor,   # [B, N, D]  frequency (Wavelet-CNN)
        use_gate: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            Fs       : [B, N, D] spatial features from CLIP
            Ff       : [B, N, D] frequency features from Wavelet-CNN
            use_gate : if False, skip gating — F_fused = Fs + A  (ablation Clip+F+A)

        Returns:
            F_fused : [B, N, D]
        """
        Fs = self.norm_s(Fs)
        Ff = self.norm_f(Ff)

        # ── Stage 1: Multi-head Cross-Attention ──────────────────────
        # [DEMO] Q comes from CLIP (Fs); K and V come from the wavelet branch
        # (Ff). Plain English: "for every semantic patch, go look up which
        # frequency-domain patches are relevant, and pull in their info."
        # This is what lets the model ask "does this object's high-frequency
        # texture match what a real object like this should look like?"
        Q = self._split_heads(self.W_Q(Fs))   # [B, h, N, d_h]
        K = self._split_heads(self.W_K(Ff))
        V = self._split_heads(self.W_V(Ff))

        A_heads = F.scaled_dot_product_attention(
            Q, K, V,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        A = self.out_proj(self._merge_heads(A_heads))  # [B, N, D] ← frequency info, now "spatially aware"

        # ── Stage 2: Gated Feature Integration ──────────────────────
        # [DEMO] G is a learned, per-token, per-channel mixing weight in [0,1].
        # It decides — separately for every patch — "trust CLIP's semantic
        # read here" (G→1) vs "trust the frequency-attended evidence here"
        # (G→0). This is the mechanism that resolves the Fs vs. Ff conflict;
        # `use_gate=False` reproduces the weaker "Clip+F+A" ablation variant.
        if use_gate:
            concat = torch.cat([Fs, A], dim=-1)      # [B, N, 2D]
            G = self.sigmoid(self.gate_mlp(concat))   # [B, N, D]
            F_fused = G * Fs + (1.0 - G) * A
        else:
            F_fused = Fs + A                           # ablation: no gating

        return F_fused
