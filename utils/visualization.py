"""
visualization.py
────────────────
Visualisation helpers:

  • GradCAM heatmap overlay  (used by server.py for the Artify extension)
  • Plot ROC curve
  • Plot Precision-Recall curve
  • Plot confusion matrix
  • Frequency spectrum visualisation
"""

from __future__ import annotations
import io
import base64
import math

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")           # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image


# ──────────────────────────────────────────────────────────────────────
#  GradCAM  –  gradient × activation at CLIP patch-token level
# ──────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Map at the CLIP patch-token level.

    WHY tokens, not backbone.input_proj
    ─────────────────────────────────────
    The Swin backbone ends with global-average-pool (GAP), which makes every
    spatial position contribute identically to the gradient at any earlier
    conv layer.  Hooking ``backbone.input_proj`` therefore produces a flat,
    spatially uninformative map.

    Instead we target the 256 CLIP patch tokens [B, 256, 1024].  Each token
    maps to a 14×14 pixel patch of the 224×224 input (16×16 grid = 256 patches).
    Gradient × activation at this level is spatially meaningful and sidesteps
    the GAP problem entirely.

    Speed
    ─────
    The frozen CLIP ViT (~304 M params) runs once under ``torch.no_grad()``;
    no autograd graph is built through it.  Gradient flows only through the
    ~34.7 M trainable head (proj → SFDF → backbone → classifier).

    Usage:
        gradcam = GradCAM(model)
        cam = gradcam(image_tensor)                 # ndarray [H, W] in [0,1]
        cam = gradcam(image_tensor, clip_tokens=t)  # cached tokens → fastest
    """

    def __init__(self, model):
        self.model = model

    @torch.enable_grad()
    def __call__(
        self,
        image_tensor: torch.Tensor,               # [1, 3, 224, 224]
        clip_tokens: torch.Tensor | None = None,  # [1, N, 1024] cached → skip ViT
        target_size: tuple = (224, 224),
    ) -> np.ndarray:
        """Return a (H, W) numpy heat map in [0, 1]."""
        model = self.model
        model.eval()

        # ── 1. Extract CLIP patch tokens without building a ViT graph ────
        if clip_tokens is not None:
            raw_tokens = clip_tokens                        # [B, N, 1024]
        else:
            extractor = model.clip_extractor
            if getattr(extractor, "visual", None) is None:
                raise RuntimeError(
                    "GradCAM: model built with load_visual=False and no "
                    "clip_tokens provided. Pass clip_tokens=... or build "
                    "the detector with force_load_visual=True.")
            with torch.no_grad():
                raw_tokens = extractor._extract_patch_tokens(image_tensor)

        # ── 2. Attach requires_grad so gradients flow from the trainable
        #       head back to this tensor (NOT through the frozen 304 M ViT) ──
        tokens = raw_tokens.detach().requires_grad_(True)

        # ── 3. Run the trainable head forward ────────────────────────────
        Fs = model.clip_extractor.proj(tokens)   # [B, N, D]

        # Wavelet branch — compute once with no_grad (we only want CLIP grads)
        with torch.no_grad():
            Ff = model.wavelet_extractor(image_tensor).detach()

        # Ablation-aware fusion
        abl = getattr(model, "ablation", None)
        if abl == "clip":
            F_fused_seq = Fs
        elif abl == "clip_f":
            F_fused_seq = Fs + Ff
        elif abl == "clip_f_a":
            F_fused_seq = model.sfdf(Fs, Ff, use_gate=False)
        else:                                    # full model (default)
            F_fused_seq = model.sfdf(Fs, Ff, use_gate=True)

        B, N, D = F_fused_seq.shape
        G = model.grid_size                      # 16
        F_fused = F_fused_seq.permute(0, 2, 1).reshape(B, D, G, G)

        z      = model.backbone(F_fused)
        logits = model.classifier(z)
        score  = torch.sigmoid(logits).squeeze()

        # ── 4. Backprop to the token tensor ──────────────────────────────
        model.zero_grad(set_to_none=True)
        score.backward()

        grads = tokens.grad                      # [B, N, clip_dim]
        if grads is None:
            return np.zeros(target_size, dtype=np.float32)

        # ── 5. GradCAM at patch-token level ──────────────────────────────
        # For each of the N=256 spatial positions: importance = grad · token
        # (dot product over the feature dimension D, then ReLU)
        cam_tokens = F.relu(
            (grads * raw_tokens).sum(dim=-1)     # [B, N]
        )

        # Reshape to 2-D spatial grid
        cam = cam_tokens.reshape(B, G, G)[0]     # [G, G] = [16, 16]
        cam_np = cam.detach().float().cpu().numpy()

        # Gaussian blur at native 16x16 resolution to merge adjacent hot spots
        # into coherent blobs before upscaling (matches paper-style visualisations)
        cam_np = cv2.GaussianBlur(cam_np, ksize=(0, 0), sigmaX=1.5)

        # Upsample with bicubic for smooth gradients
        cam_up = cv2.resize(cam_np, (target_size[1], target_size[0]),
                            interpolation=cv2.INTER_CUBIC)
        cam_up = np.maximum(cam_up, 0)           # re-apply ReLU after cubic may go negative

        mn, mx = cam_up.min(), cam_up.max()
        return (cam_up - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(cam_up)


# ──────────────────────────────────────────────────────────────────────
#  Fast forward-only heat map (no backprop)
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def activation_heatmap(
    model,
    image_tensor: torch.Tensor,                 # [1, 3, 224, 224]
    clip_tokens: torch.Tensor | None = None,
    target_size: tuple = (224, 224),
) -> np.ndarray:
    """
    Fastest heat map: a single forward pass, no backward.

    Uses the per-token L2 magnitude of the CLIP patch tokens as a saliency
    proxy — tokens the network encodes most strongly = regions of interest.
    ~2× faster than GradCAM (no backward pass needed).

    Returns a (H, W) numpy heat map in [0, 1].
    """
    model.eval()
    extractor = model.clip_extractor
    if clip_tokens is None and getattr(extractor, "visual", None) is None:
        raise RuntimeError("activation_heatmap: provide clip_tokens or build "
                           "the model with force_load_visual=True.")

    # Get CLIP patch tokens [B, N, D]
    if clip_tokens is not None:
        raw_tokens = clip_tokens
    else:
        raw_tokens = extractor._extract_patch_tokens(image_tensor)

    # Per-token L2 norm → [B, N] → reshape to [B, G, G]
    B, N, D = raw_tokens.shape
    G = model.grid_size
    cam = raw_tokens.norm(dim=-1).reshape(B, G, G)[0]   # [G, G]
    cam_np = cam.float().cpu().numpy()
    cam_up = cv2.resize(cam_np, (target_size[1], target_size[0]),
                        interpolation=cv2.INTER_LINEAR)
    mn, mx = cam_up.min(), cam_up.max()
    return (cam_up - mn) / (mx - mn) if mx - mn > 1e-8 else np.zeros_like(cam_up)


# ──────────────────────────────────────────────────────────────────────
#  Heatmap overlay helpers (used by server.py)
# ──────────────────────────────────────────────────────────────────────

def heatmap_to_overlay(
    image_pil: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Blend a GradCAM heat map onto the original image.

    Args:
        image_pil : original RGB PIL Image
        cam       : (H, W) float array in [0,1]
        alpha     : heatmap opacity
        colormap  : OpenCV colormap constant

    Returns:
        blended PIL Image (RGB)
    """
    img_np = np.array(image_pil.convert("RGB"))
    h, w   = img_np.shape[:2]

    # Resize cam to image size
    cam_resized = cv2.resize(cam, (w, h))
    heat_uint8  = (cam_resized * 255).astype(np.uint8)
    heat_color  = cv2.applyColorMap(heat_uint8, colormap)   # BGR
    heat_rgb    = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

    blended = (img_np * (1 - alpha) + heat_rgb * alpha).astype(np.uint8)
    return Image.fromarray(blended)


def generate_heatmap(
    model,
    image_pil: Image.Image,
    transform,
    device,
    method: str = "gradcam",          # "gradcam" (faithful) | "fast" (forward-only)
    clip_tokens: torch.Tensor | None = None,
    alpha: float = 0.5,
) -> Image.Image:
    """
    One-call "region of interest" heat map (Man & Cho 2026 Figure 6 style).

    Takes a raw PIL image, runs the chosen heat-map method, and returns the
    colour overlay blended on the original image — ready to display or save.

    Args:
        model       : trained AIGCDetector (eval mode)
        image_pil   : original RGB PIL image
        transform   : the eval transform (build_transforms(augment=False))
        device      : torch device
        method      : "gradcam" = Grad-CAM (default, matches the paper);
                      "fast"    = forward-only activation map (no backprop)
        clip_tokens : optional pre-cached CLIP tokens [1, N, 1024] → skips the
                      ViT for maximum speed
        alpha       : heat-map opacity over the original image
    Returns:
        blended RGB PIL image (same size as image_pil)
    """
    x = transform(image_pil.convert("RGB")).unsqueeze(0).to(device)
    if clip_tokens is not None:
        clip_tokens = clip_tokens.to(device)

    if method == "fast":
        cam = activation_heatmap(model, x, clip_tokens=clip_tokens)
    else:
        cam = GradCAM(model)(x, clip_tokens=clip_tokens)

    return heatmap_to_overlay(image_pil, cam, alpha=alpha)


def pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 data-URL string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


# ──────────────────────────────────────────────────────────────────────
#  Matplotlib plotting helpers
# ──────────────────────────────────────────────────────────────────────

def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, save_path: str | None = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color="#667eea", label=f"ROC (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ap: float,
    save_path: str | None = None,
):
    from sklearn.metrics import precision_recall_curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="#764ba2", label=f"AP = {ap:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list = None,
    save_path: str | None = None,
):
    class_names = class_names or ["Real", "AI-Generated"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks); ax.set_xticklabels(class_names)
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.close(fig)
