"""
equalize_training_data.py
─────────────────────────
Remove the real-vs-fake "source shortcut" from the training data.

Problem (confirmed by diagnose_data_bias.py):
  GenImage REAL = 640x480 JPG (~143KB),  FAKE = 128x128 PNG (~28KB).
  The model learns "big JPG = real, small PNG = fake" instead of artifacts,
  scoring 99% in-distribution but ~50% on matched per-gen test sets.

Fix:
  Pass EVERY training image (real and fake, all datasets) through one
  identical pipeline:
     resize-shortest-side + center-crop  ->  TARGET x TARGET
     re-encode to a single format (PNG, lossless)
  After this the only thing distinguishing real from fake is the actual
  generation artifact — resolution / format / file-size cues are gone.

Non-destructive: originals under datasets/ are untouched; cleaned copies
are written to datasets_eq/.  Drives off the existing CLIP cache path lists
so the cleaned set is exactly the same samples used for training.

Usage:
    python equalize_training_data.py
    python equalize_training_data.py --target 256 --out datasets_eq
    python equalize_training_data.py --format jpg --quality 95
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

# cache name -> (output dataset folder, source cache file)
DATASETS = {
    "dfdc":        "DFDC",
    "forensynths": "ForenSynths",
    "genimage":    "GenImage",
}
CACHE_DIR = Path("datasets/clip_cache")


def label_of(path: str) -> str:
    p = path.lower()
    if "fake" in p:
        return "fake"
    if "real" in p:
        return "real"
    return "unknown"


def equalize_one(img: Image.Image, target: int) -> Image.Image:
    """Resize shortest side + center-crop to target x target (aspect-preserving)."""
    img = img.convert("RGB")
    # ImageOps.fit: scales so the whole target is covered, then center-crops.
    return ImageOps.fit(img, (target, target), method=Image.BICUBIC,
                        centering=(0.5, 0.5))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--target",  type=int, default=224,
                    help="Output square size (default 224 = CLIP input size; "
                         "matches the per-gen cache pipeline exactly, so "
                         "training and test images are resampled identically)")
    ap.add_argument("--out",     default="datasets_eq",
                    help="Output root (non-destructive; default datasets_eq)")
    ap.add_argument("--format",  default="png", choices=["png", "jpg"],
                    help="Single output format for BOTH classes (default png, "
                         "lossless — adds no compression artifact)")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality if --format jpg (ignored for png)")
    ap.add_argument("--only",    default=None, choices=list(DATASETS),
                    help="Only process this one dataset")
    args = ap.parse_args()

    out_root = Path(args.out)
    ext = ".png" if args.format == "png" else ".jpg"

    print(f"[equalize] target={args.target}px  format={args.format}  out={out_root}\n")

    grand_total = 0
    for cache_name, ds_folder in DATASETS.items():
        if args.only and cache_name != args.only:
            continue
        cache_path = CACHE_DIR / f"{cache_name}_clip.pt"
        if not cache_path.exists():
            print(f"  [{cache_name}] cache not found: {cache_path} — skipping")
            continue

        paths = torch.load(cache_path, map_location="cpu",
                           weights_only=True)["paths"]
        print(f"  [{cache_name}] {len(paths)} images -> {out_root / ds_folder}")

        # Pre-create class dirs
        for cls in ("real", "fake"):
            (out_root / ds_folder / cls).mkdir(parents=True, exist_ok=True)

        counts = {"real": 0, "fake": 0, "skipped": 0}
        for src in tqdm(paths, desc=f"    {cache_name}", unit="img"):
            cls = label_of(src)
            if cls == "unknown":
                counts["skipped"] += 1
                continue
            src_p = Path(src)
            try:
                with Image.open(src_p) as im:
                    out_im = equalize_one(im, args.target)
            except Exception:
                counts["skipped"] += 1
                continue

            # Keep original stem for traceability; index suffix avoids collisions
            dst = out_root / ds_folder / cls / f"{src_p.stem}_{counts[cls]:06d}{ext}"
            if args.format == "png":
                out_im.save(dst, "PNG", optimize=False)
            else:
                out_im.save(dst, "JPEG", quality=args.quality)
            counts[cls] += 1

        print(f"    -> real={counts['real']}  fake={counts['fake']}  "
              f"skipped={counts['skipped']}")
        grand_total += counts["real"] + counts["fake"]

    print(f"\n[equalize] done. {grand_total} images written under {out_root}/")
    print("[equalize] Next:")
    print("  1. Point config data roots at the equalized copies, e.g.:")
    print(f"       dfdc_root:        {out_root}/DFDC")
    print(f"       forensynths_root: {out_root}/ForenSynths")
    print(f"       genimage_root:    {out_root}/GenImage")
    print("  2. Rebuild the CLIP cache:  python cache_clip_features.py --overwrite")
    print("  3. Re-run the linear probe to confirm the ceiling rose:")
    print("       python diagnose_clip_features.py")


if __name__ == "__main__":
    main()
