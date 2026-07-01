"""
wavelet_extractor.py
────────────────────
Frequency-domain feature extractor (paper §3.1).

Pipeline:
  1. Differentiable 2D DWT (Daubechies-4) applied per colour channel.
  2. Discard low-freq LL sub-band; keep LH, HL, HH → 9-channel map.
  3. Lightweight CNN encoder (3 × Conv-BN-ReLU, stride-2) projects the
     high-frequency map to a token sequence matching Fs shape.

The DWT is implemented as fixed (non-learnable) separable convolutions
so it runs entirely on GPU without NumPy round-trips.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt


# ──────────────────────────────────────────────────────────────────────
#  Differentiable 2-D DWT layer
# ──────────────────────────────────────────────────────────────────────

class DWT2D(nn.Module):
    """
    Single-level 2-D DWT using fixed db4 filters as convolutions.

    Input  : [B, C_in, H, W]
    Output : [B, 3*C_in, H/2, W/2]   (LH, HL, HH per channel; LL discarded)
    """

    def __init__(self, wavelet: str = "db4"):
        super().__init__()

        w = pywt.Wavelet(wavelet)

        # Decomposition filters (reversed for convolution == correlation)
        lo = np.array(w.dec_lo[::-1], dtype=np.float32)
        hi = np.array(w.dec_hi[::-1], dtype=np.float32)

        # 2-D outer products → [1, 1, k, k] filters
        def _2d(a, b):
            return torch.from_numpy(np.outer(a, b)).unsqueeze(0).unsqueeze(0)

        self.register_buffer("filt_lh", _2d(hi, lo))  # row-hi, col-lo
        self.register_buffer("filt_hl", _2d(lo, hi))  # row-lo, col-hi
        self.register_buffer("filt_hh", _2d(hi, hi))  # row-hi, col-hi

        k = len(lo)
        self.pad = k // 2

    def _apply_filter(self, xc: torch.Tensor, filt: torch.Tensor) -> torch.Tensor:
        """xc: [B, 1, H, W]; filt: [1, 1, k, k]"""
        xp = F.pad(xc, [self.pad] * 4, mode="reflect")
        return F.conv2d(xp, filt, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [DEMO] This is "the hidden senses" — it decomposes the image into
        # high-frequency sub-bands (LH=horizontal edges, HL=vertical edges,
        # HH=diagonal/texture detail). The smooth LL band is thrown away on
        # purpose: AI-generator artifacts (upsampling grids, GAN checkerboard
        # patterns) live in the HIGH-frequency detail, not the smooth content
        # a human eye looks at.
        B, C, H, W = x.shape
        bands = []
        for c in range(C):
            xc = x[:, c : c + 1]          # [B, 1, H, W]
            bands.append(self._apply_filter(xc, self.filt_lh))
            bands.append(self._apply_filter(xc, self.filt_hl))
            bands.append(self._apply_filter(xc, self.filt_hh))
        return torch.cat(bands, dim=1)     # [B, 3*C, H/2, W/2]


# ──────────────────────────────────────────────────────────────────────
#  Lightweight CNN encoder
# ──────────────────────────────────────────────────────────────────────

def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class WaveletExtractor(nn.Module):
    """
    Args:
        in_channels : colour channels of input image (3)
        out_dim     : output feature dimension per token (=feature_dim)
        grid_size   : spatial grid of the target token sequence, e.g. 16
                      (must match CLIP's patch-grid so shapes align in SFDF)
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 768,
        grid_size: int = 16,
    ):
        super().__init__()

        self.dwt = DWT2D(wavelet="db4")

        # "Lightweight" CNN encoder (paper §3.1): 9 → 64 → 256 → 768 channels.
        # Keeping mid=64 makes this branch genuinely lightweight: ~3.8 B MACs
        # vs ~17 B MACs with mid=192 (the wider variant that max(64,768//4) gives).
        # The paper does not specify channel counts; narrow intermediate channels
        # are sufficient for frequency-artifact discrimination.
        mid = 64
        self.cnn = nn.Sequential(
            _conv_block(in_channels * 3, mid),          # 9  →  64, /2
            _conv_block(mid, mid * 4),                  # 64 → 256, /2
            _conv_block(mid * 4, out_dim),              # 256 → 768, /2
        )

        # Adaptive pool to exactly (grid_size, grid_size)
        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.grid_size = grid_size
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x  : [B, 3, H, W]  – raw input images
        Returns:
            Ff : [B, N, out_dim]  – frequency token sequence (N = grid_size²)
        """
        hf = self.dwt(x)                        # [B, 9, H/2, W/2]  ← high-freq-only map
        feat = self.cnn(hf)                     # [B, out_dim, h, w]  CNN reads the artifact texture
        feat = self.pool(feat)                  # [B, out_dim, G, G]  forced to match CLIP's 16×16 grid

        B, D, G, _ = feat.shape
        # [DEMO] Ff is the "frequency" feature stream — same token-grid shape
        # as Fs from clip_extractor.py, so SFDF (sfdf.py) can directly align
        # and cross-attend between them token-for-token.
        Ff = feat.flatten(2).permute(0, 2, 1)  # [B, G*G, D]
        return Ff
