"""
generate_sample_heatmaps.py
────────────────────────────
Generate GradCAM heatmaps and frequency-domain visualisations for the
Man & Cho sample images in  heatmap/Man and Cho Sample Heatmap/.

  Images 1–3  →  real faces   → individual GradCAM overlays
  Images 4–6  →  fake faces   → individual GradCAM overlays
  Images 7–8  →  real + fake  → individual FFT spectra
                              → composite 2×3 panel (original | FFT | GradCAM)

All outputs land in  heatmap/output/.

Usage:
    python generate_sample_heatmaps.py
    python generate_sample_heatmaps.py --checkpoint checkpoints/best_model.pt
    python generate_sample_heatmaps.py --device cpu
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import cv2
from PIL import Image
import yaml

from models import build_detector
from data.dataset import build_transforms
from utils import setup_hf_auth
from utils.visualization import GradCAM, heatmap_to_overlay


# ── Paths ──────────────────────────────────────────────────────────────
INPUT_DIR  = Path("heatmap/Man and Cho Sample Heatmap")
OUTPUT_DIR = Path("heatmap/output")

GRADCAM_INDICES = [1, 2, 3, 4, 5, 6]   # images for GradCAM-only outputs
FREQ_REAL_IDX   = 7                      # image 7 → real face for freq figure
FREQ_FAKE_IDX   = 8                      # image 8 → fake face for freq figure


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--device",     default=None)
    p.add_argument("--threshold",  type=float, default=0.5)
    return p.parse_args()


# ── Helpers ─────────────────────────────────────────────────────────────
def load_pil(idx: int) -> Image.Image:
    return Image.open(INPUT_DIR / f"image{idx}.png").convert("RGB")


def infer(model, tensor: torch.Tensor, threshold: float) -> tuple[float, str]:
    """Return (probability, label) for a preprocessed [1,3,224,224] tensor."""
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor, clip_tokens=None)).item()
    return prob, ("AI-Generated" if prob >= threshold else "Real")


def compute_fft_spectrum(pil_img: Image.Image) -> np.ndarray:
    """
    Log-magnitude 2-D FFT spectrum of the grayscale image.
    Returns a float32 array in [0, 1], same spatial size as the input.

    The centre of the returned array is DC (zero frequency); high-frequency
    energy appears at the edges.  GAN up-sampling grids, periodic compression
    artefacts, and spectral peaks from specific generators are visible here —
    this is a fundamentally different diagnostic from GradCAM.
    """
    gray = np.array(pil_img.convert("L"), dtype=np.float64)
    fft_shift = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(fft_shift)).astype(np.float32)
    mn, mx = magnitude.min(), magnitude.max()
    return (magnitude - mn) / (mx - mn) if mx > mn else np.zeros_like(magnitude)


def save_gradcam_overlay(
    pil_img: Image.Image,
    cam: np.ndarray,
    out_path: Path,
    label: str,
    prob: float,
) -> None:
    overlay = heatmap_to_overlay(pil_img, cam)
    overlay.save(str(out_path))
    verdict = "[AI]" if label == "AI-Generated" else "[OK]"
    print(f"  {verdict} {out_path.name:<40} label={label}  p={prob:.4f}")


def save_fft_figure(
    pil_img: Image.Image,
    spectrum: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a side-by-side original + FFT spectrum figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.patch.set_facecolor("#111122")

    ax1.imshow(pil_img)
    ax1.set_title("Original", color="white", fontsize=12)
    ax1.axis("off")

    ax2.imshow(spectrum, cmap="inferno")
    ax2.set_title("Frequency Spectrum (log |FFT|)", color="white", fontsize=12)
    ax2.axis("off")

    fig.suptitle(title, color="white", fontsize=13, y=1.01)
    plt.tight_layout(pad=1.2)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  FFT  {out_path.name}")


def save_composite_figure(
    real_img:  Image.Image,
    real_fft:  np.ndarray,
    real_cam:  np.ndarray,
    fake_img:  Image.Image,
    fake_fft:  np.ndarray,
    fake_cam:  np.ndarray,
    real_prob: float,
    fake_prob: float,
    out_path:  Path,
) -> None:
    """
    2×3 composite panel matching Man & Cho (2026) Figure 5:
      Row 1  Real image:  original | FFT spectrum | GradCAM attention
      Row 2  Fake image:  original | FFT spectrum | GradCAM attention
    """
    real_overlay = heatmap_to_overlay(real_img, real_cam)
    fake_overlay = heatmap_to_overlay(fake_img, fake_cam)

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    fig.patch.set_facecolor("#111122")

    row_labels = [
        f"Real image  (p={real_prob:.4f})",
        f"Fake image  (p={fake_prob:.4f})",
    ]
    col_labels = ["Original", "Frequency Spectrum", "Attention Heatmap"]

    grid = [
        [real_img,  real_fft,  real_overlay],
        [fake_img,  fake_fft,  fake_overlay],
    ]
    cmaps = [None, "inferno", None]

    for r in range(2):
        for c in range(3):
            ax = axes[r][c]
            if cmaps[c] is not None:
                ax.imshow(grid[r][c], cmap=cmaps[c])
            else:
                ax.imshow(grid[r][c])
            ax.axis("off")
            if r == 0:
                ax.set_title(col_labels[c], color="white", fontsize=11, pad=6)

        # row label as y-axis text
        axes[r][0].text(
            -0.05, 0.5, row_labels[r],
            transform=axes[r][0].transAxes,
            color="white", fontsize=10,
            rotation=90, va="center", ha="right",
        )

    fig.suptitle(
        "Frequency Domain Features & Model Attention  (Man & Cho 2026 Fig. 5)",
        color="white", fontsize=13, y=1.01,
    )
    plt.tight_layout(pad=1.2)
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Composite -> {out_path.name}")


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    setup_hf_auth()

    cfg = yaml.safe_load(open(args.config))
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[generate_sample_heatmaps] Loading {args.checkpoint} on {device}\n")
    model = build_detector(cfg, force_load_visual=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: checkpoint was saved in CLIP-cache mode (load_visual=False)
    # so frozen ViT keys are absent; they are already correct from open_clip.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if unexpected:
        print(f"  [warn] unexpected keys in checkpoint: {len(unexpected)}")
    print(f"  Loaded {len(ckpt['model'])} checkpoint keys "
          f"({len(missing)} CLIP ViT keys kept from open_clip init)")
    model.eval()

    transform = build_transforms(image_size=224, augment=False)
    gradcam   = GradCAM(model)

    # ── Phase 1: GradCAM overlays for images 1–6 ──────────────────────
    print("=" * 60)
    print("Phase 1 — GradCAM attention heatmaps (images 1–6)")
    print("=" * 60)
    for idx in GRADCAM_INDICES:
        pil_img = load_pil(idx)
        tensor  = transform(pil_img).unsqueeze(0).to(device)
        prob, label = infer(model, tensor, args.threshold)
        cam     = gradcam(tensor)
        save_gradcam_overlay(
            pil_img, cam,
            OUTPUT_DIR / f"heatmap_image{idx}.png",
            label, prob,
        )

    # ── Phase 2: Frequency domain figures for images 7–8 ──────────────
    print()
    print("=" * 60)
    print("Phase 2 — Frequency domain + composite (images 7–8)")
    print("=" * 60)

    real_img = load_pil(FREQ_REAL_IDX)
    fake_img = load_pil(FREQ_FAKE_IDX)

    real_tensor = transform(real_img).unsqueeze(0).to(device)
    fake_tensor = transform(fake_img).unsqueeze(0).to(device)

    real_prob, real_label = infer(model, real_tensor, args.threshold)
    fake_prob, fake_label = infer(model, fake_tensor, args.threshold)

    print(f"  image7 -> {real_label}  (p={real_prob:.4f})")
    print(f"  image8 -> {fake_label}  (p={fake_prob:.4f})")
    print()

    # Individual FFT figures
    real_fft = compute_fft_spectrum(real_img)
    fake_fft = compute_fft_spectrum(fake_img)

    save_fft_figure(
        real_img, real_fft,
        OUTPUT_DIR / "freq_image7_real.png",
        title=f"Real image  (p={real_prob:.4f})",
    )
    save_fft_figure(
        fake_img, fake_fft,
        OUTPUT_DIR / "freq_image8_fake.png",
        title=f"Fake image  (p={fake_prob:.4f})",
    )

    # GradCAM for images 7–8 (needed for composite)
    real_cam = gradcam(real_tensor)
    fake_cam = gradcam(fake_tensor)

    # Composite 2×3 panel
    save_composite_figure(
        real_img, real_fft, real_cam,
        fake_img, fake_fft, fake_cam,
        real_prob, fake_prob,
        OUTPUT_DIR / "frequency_composite.png",
    )

    print(f"\nDone — all outputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
