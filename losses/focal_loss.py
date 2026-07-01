"""
focal_loss.py
─────────────
Binary Focal Loss from paper Eq. (6):

  L_cls = -(1/B) Σ [ (1-p_i)^γ · y_i · log(p_i)
                    + p_i^γ · (1-y_i) · log(1-p_i) ]

  where γ=2, p_i = sigmoid(logit_i), y_i ∈ {0,1}
  No alpha term — matches paper Eq. (6) exactly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Binary focal loss for AI-generated image classification.

    Args:
        gamma     : focusing parameter  (paper uses γ=2)
        reduction : 'mean' | 'sum' | 'none'
    """

    def __init__(
        self,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # [B, 1] or [B]
        targets: torch.Tensor,  # [B]  – float labels in {0.0, 1.0}
    ) -> torch.Tensor:
        logits = logits.squeeze(-1)                 # [B]
        targets = targets.float()

        p   = torch.sigmoid(logits)                 # [B]
        p_t = p * targets + (1.0 - p) * (1.0 - targets)

        # BCE per sample (numerically stable)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )

        # Focal weight — Eq. (6): (1-p_t)^γ where p_t = p_i for y=1, (1-p_i) for y=0
        focal_weight = (1.0 - p_t).pow(self.gamma)

        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
