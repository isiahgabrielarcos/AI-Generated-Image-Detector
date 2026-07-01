"""
clip_extractor.py
─────────────────
Frozen CLIP ViT-L/14 spatial-domain feature extractor.

Returns patch-level token embeddings [B, N, clip_dim] then projects
them down to [B, N, feature_dim].  The CLIP visual weights are completely
frozen – we only detect "semantic inconsistencies" using existing
prior knowledge (paper §3.1).

CPU optimisation note
─────────────────────
When pre-computed CLIP tokens are available (from cache_clip_features.py),
pass them as `precomputed_tokens` to forward().  The frozen ViT is then
bypassed entirely, which is the single biggest CPU speedup.

Set `load_visual=False` when you know you will always supply
precomputed_tokens; the ViT weights are never loaded, saving ~1.2 GB RAM.
The trainable projection adapter (clip_dim → feature_dim) is still
instantiated and trained in both modes.
"""

import torch
import torch.nn as nn
import open_clip


# ViT-L/14 patch-token width – hard-coded so we can skip loading the model
# in cache-only mode.  Change this if you switch CLIP variants.
_VITL14_WIDTH = 1024


class CLIPExtractor(nn.Module):
    """
    Wraps frozen CLIP ViT-L/14 and exposes patch-token features.

    Args:
        model_name       : CLIP model tag, e.g. 'ViT-L-14'
        pretrained       : CLIP checkpoint, e.g. 'openai'
        out_dim          : projected output dimension (= feature_dim in config)
        load_visual      : if False, skip loading the heavy ViT weights.
                           Use when you always pass precomputed_tokens.
    """

    def __init__(
        self,
        model_name: str = "ViT-L-14-quickgelu",
        pretrained: str = "openai",
        out_dim: int = 768,
        load_visual: bool = True,
    ):
        super().__init__()

        if load_visual:
            # ── load CLIP ────────────────────────────────────────────
            # [DEMO] This is "the eyes" of the system — pretrained CLIP ViT-L/14,
            # already knows what real-world objects/scenes look like from
            # ~400M image-text pairs. We don't train this from scratch.
            clip_model, _, _ = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            self.visual = clip_model.visual
            clip_dim = self.visual.transformer.width  # 1024 for ViT-L/14

            # ── freeze all CLIP weights ───────────────────────────────
            # [DEMO] requires_grad = False → these ~300M params NEVER update
            # during training. Only the small projection layer below learns.
            # This is why training is fast and doesn't need a huge dataset.
            for p in self.visual.parameters():
                p.requires_grad = False
        else:
            # Skip the ViT entirely – saves ~1.2 GB RAM on CPU.
            # forward() requires precomputed_tokens in this mode.
            self.visual = None
            clip_dim = _VITL14_WIDTH

        # ── trainable projection to shared feature_dim ───────────────
        # Trained at backbone_lr (paper §4.2); adapts CLIP space → D.
        self.proj = (
            nn.Linear(clip_dim, out_dim, bias=False)
            if clip_dim != out_dim
            else nn.Identity()
        )
        self.clip_dim = clip_dim
        self.out_dim = out_dim

    # ─────────────────────────────────────────────────────────────────
    def _extract_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Manually run the CLIP visual transformer to obtain per-patch
        token embeddings (CLS token excluded).

        Args:
            x : [B, 3, H, W]  – pre-processed images (CLIP normalisation)
        Returns:
            [B, N, clip_dim]  – N = (H/patch_size)^2
        """
        v = self.visual

        # [DEMO] Image gets cut into a 16×16 grid of patches (224/14 ≈ 16),
        # each patch becomes one "token" — same idea as splitting a sentence
        # into words before feeding it to a language transformer.
        # Patch embedding
        x = v.conv1(x)                             # [B, width, grid, grid]
        B, C, gh, gw = x.shape
        x = x.reshape(B, C, -1).permute(0, 2, 1)  # [B, N, width]

        # Prepend CLS token and add positional embedding
        cls = v.class_embedding.unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)             # [B, N+1, width]
        x = x + v.positional_embedding

        x = v.ln_pre(x)

        # This open_clip build's transformer is batch-first (expects
        # [batch, seq, dim]).  Do NOT permute to [seq, batch, dim]: that
        # swaps the batch and sequence axes, making the images in a batch
        # attend to each other and silently corrupting every token.
        # Verified: without the permute, output matches open_clip's official
        # visual forward exactly (cos = 1.0) and is batch-independent.
        x = v.transformer(x)                      # [B, N+1, width]

        # Drop CLS, keep patch tokens
        return x[:, 1:, :]                         # [B, N, clip_dim]

    # ─────────────────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        precomputed_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            x                  : [B, 3, 224, 224]  – always required (wavelet
                                   branch still processes it even when tokens
                                   are cached)
            precomputed_tokens : [B, N, clip_dim]  – raw ViT patch tokens from
                                   cache_clip_features.py.  When provided the
                                   frozen ViT is skipped entirely.
        Returns:
            Fs : [B, N, out_dim]
        """
        if precomputed_tokens is not None:
            # Fast path: bypass the frozen ViT, run only the tiny projection
            tokens = precomputed_tokens  # [B, N, 1024]
        else:
            assert self.visual is not None, (
                "CLIP visual encoder not loaded (load_visual=False). "
                "Either pass precomputed_tokens or set load_visual=True."
            )
            with torch.no_grad():
                tokens = self._extract_patch_tokens(x)   # [B, N, 1024]

        # [DEMO] Fs is the "spatial/semantic" feature stream — one of the two
        # inputs that feed into SFDF (sfdf.py) for cross-domain fusion.
        # Compare this to Ff produced by wavelet_extractor.py — same shape,
        # different domain (semantic content vs. frequency texture).
        Fs = self.proj(tokens)                            # [B, N, out_dim]
        return Fs
