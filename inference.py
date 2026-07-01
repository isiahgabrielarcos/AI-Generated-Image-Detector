"""
inference.py
────────────
Run the trained detector on a single image or folder of images.

Usage:
    # Single image
    python inference.py --checkpoint checkpoints/best_model.pt --image path/to/img.jpg

    # Folder of images
    python inference.py --checkpoint checkpoints/best_model.pt --image_dir path/to/folder/

    # Save GradCAM heatmap overlay
    python inference.py --checkpoint checkpoints/best_model.pt --image img.jpg --heatmap

Note: inference always loads the full CLIP ViT (force_load_visual=True) because
input images are arbitrary and are not guaranteed to be in the CLIP cache.
"""

import argparse
import time
from pathlib import Path

import torch
from PIL import Image
import yaml

from models import build_detector
from data.dataset import build_transforms
from utils import setup_hf_auth
from utils.visualization import GradCAM, heatmap_to_overlay


_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--image",      default=None, help="Path to a single image")
    p.add_argument("--image_dir",  default=None, help="Folder of images")
    p.add_argument("--heatmap",    action="store_true", help="Generate GradCAM heatmap")
    p.add_argument("--output_dir", default="results/inference")
    p.add_argument("--device",     default=None)
    p.add_argument("--threshold",  type=float, default=0.5)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def predict_image(
    image_path: str,
    model,
    transform,
    device,
    threshold: float,
    gradcam,
    output_dir: Path,
):
    image_path = Path(image_path)
    pil_img    = Image.open(image_path).convert("RGB")
    tensor     = transform(pil_img).unsqueeze(0).to(device)   # [1, 3, 224, 224]

    t0 = time.perf_counter()
    with torch.no_grad():
        # clip_tokens=None -> model runs the live CLIP ViT
        logits = model(tensor, clip_tokens=None)
        prob   = torch.sigmoid(logits).item()
    elapsed_ms = (time.perf_counter() - t0) * 1000

    label      = "AI-Generated" if prob >= threshold else "Real"
    confidence = max(prob, 1.0 - prob)

    verdict = "[AI]" if label == "AI-Generated" else "[OK]"
    print(
        f"  {verdict} {image_path.name:<40} "
        f"p={prob:.4f}  conf={confidence:.4f}  ({elapsed_ms:.1f} ms)"
    )

    if gradcam is not None:
        cam      = gradcam(tensor)
        overlay  = heatmap_to_overlay(pil_img, cam)
        out_path = output_dir / f"{image_path.stem}_heatmap.png"
        overlay.save(str(out_path))
        print(f"    heatmap -> {out_path}")

    return {
        "image":       str(image_path),
        "prediction":  label,
        "probability": prob,
        "confidence":  confidence,
        "time_ms":     elapsed_ms,
    }


def main():
    args   = parse_args()
    cfg    = load_config(args.config)

    setup_hf_auth()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Model ─────────────────────────────────────────────────────────
    # force_load_visual=True: inference images are not in the CLIP cache,
    # so the full ViT must be available for live feature extraction.
    print(f"[inference] loading {args.checkpoint} on {device}")
    model = build_detector(cfg, force_load_visual=True).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # strict=False: checkpoint may have been saved in CLIP-cache mode
    # (load_visual=False), so frozen ViT keys are absent in the checkpoint
    # but already correctly initialised by open_clip.
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    transform = build_transforms(image_size=224, augment=False)
    gradcam   = GradCAM(model) if args.heatmap else None

    # ── Collect image paths ───────────────────────────────────────────
    if args.image:
        image_paths = [args.image]
    elif args.image_dir:
        image_paths = [
            str(p)
            for p in sorted(Path(args.image_dir).rglob("*"))
            if p.suffix.lower() in _IMG_EXTS
        ]
    else:
        print("Provide --image or --image_dir.")
        return

    print(f"[inference] {len(image_paths)} image(s)\n")

    results = []
    for path in image_paths:
        r = predict_image(
            path, model, transform, device,
            args.threshold, gradcam, output_dir,
        )
        results.append(r)

    # ── Save CSV ──────────────────────────────────────────────────────
    import csv
    csv_path = output_dir / "predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n[inference] predictions saved -> {csv_path}")


if __name__ == "__main__":
    main()
