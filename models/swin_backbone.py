"""
swin_backbone.py
─────────────────
Swin-Tiny backbone for global contextual modelling (paper §3.3).

Accepts the fused 2-D feature map F_fused ∈ R^{B × C × G × G} from SFDF,
applies a patch-partition (patch_size=1 so spatial resolution is preserved),
then processes it through Swin-Tiny blocks and global average-pooling to
produce a 768-d vector ready for the classification head.

Window size is set to 4 so it divides the default 16×16 feature grid evenly.
"""

import torch
import torch.nn as nn

try:
    from timm.models.swin_transformer import SwinTransformer
except ImportError:
    raise ImportError("Please install timm >= 0.9.0:  pip install timm")


class SwinBackbone(nn.Module):
    """
    Wraps timm SwinTransformer with settings matched to our 16×16 feature grid.

    Args:
        in_dim      : channels of F_fused (= feature_dim from SFDF output)
        grid_size   : spatial size of F_fused (default 16)
        window_size : Swin window size; must divide grid_size (default 4)
        embed_dim   : Swin base channels (96 for Tiny)
        depths      : Swin stage depths
        num_heads   : attention heads per stage
        out_dim     : dimension of the output pooled vector
    """

    def __init__(
        self,
        in_dim: int = 768,
        grid_size: int = 16,
        window_size: int = 4,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        out_dim: int = 768,
    ):
        super().__init__()
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        # Input projection: align in_dim → embed_dim for Swin
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )

        # Swin-Tiny with patch_size=1 so 16×16 feature map stays 16×16 tokens
        self.swin = SwinTransformer(
            img_size=grid_size,
            patch_size=1,
            in_chans=embed_dim,
            num_classes=0,          # remove built-in classifier
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.1,
            global_pool="avg",
        )

        # Swin-Tiny final channels = embed_dim * 2^(num_stages-1) * 1  (num_features)
        swin_out = embed_dim * (2 ** (len(depths) - 1))  # 96*8 = 768

        # Map to desired out_dim if different
        self.head_proj = (
            nn.Linear(swin_out, out_dim, bias=False)
            if swin_out != out_dim
            else nn.Identity()
        )
        self.out_dim = out_dim

    def forward(self, F_fused: torch.Tensor) -> torch.Tensor:
        """
        Args:
            F_fused : [B, C, G, G]  – spatial 2-D fused feature map
        Returns:
            z       : [B, out_dim]  – globally pooled representation
        """
        # [DEMO] This is "the brain" — by this point CLIP + Wavelet have
        # already been fused token-by-token by SFDF. Swin's job is different:
        # it looks at the WHOLE fused map together (shifted windows = local
        # patches gradually see more and more of the image across stages),
        # catching global inconsistencies a single patch can't reveal alone
        # — e.g. lighting/shadow direction that doesn't agree across the image.
        x = self.input_proj(F_fused)   # [B, embed_dim, G, G]
        z = self.swin(x)               # [B, swin_out]  (global avg pool inside)
        # [DEMO] z is the single vector that summarizes "is this image
        # internally consistent?" — feeds straight into the classifier head
        # (see detector.py) which outputs the real/fake probability.
        z = self.head_proj(z)          # [B, out_dim]
        return z
