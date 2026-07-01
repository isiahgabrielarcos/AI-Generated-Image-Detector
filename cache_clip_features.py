"""
cache_clip_features.py
──────────────────────
Pre-compute and cache CLIP ViT-L/14 patch tokens for all dataset images.
Run ONCE before training to get the largest possible CPU speedup.

Usage:
    python cache_clip_features.py
    python cache_clip_features.py --config configs/default.yaml --batch-size 4

What it saves
─────────────
For each dataset (dfdc, forensynths, genimage) it writes:

    datasets/clip_cache/<name>_clip.pt
        "paths"    : list[str]               # absolute image paths
        "features" : Tensor[N, 256, 1024]    # float16, raw ViT-L/14 tokens

Storage (float16):
    ~524 KB per image  ->  ~5.2 GB for 10 k images  /  ~15.7 GB for 30 k

Expected run time on CPU (batch_size=4):
    Batch time is dominated by CLIP ViT-L/14 (24 transformer layers).
    Rough guide: ~10-40 s per batch of 4, depending on CPU.
    10 k images -> ~30 min - 2 h   (one-time cost)
    30 k images -> ~1.5 h - 6 h

After the cache is built, train.py will detect it automatically and skip
the ViT each training step, saving the same compute 50-100x over.
"""

import argparse
import time
import yaml
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from utils.hf_auth import setup_hf_auth
from models.clip_extractor import CLIPExtractor
from data.dataset import _collect_images, build_transforms


# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Images per CLIP batch (lower = less RAM during caching)")
    p.add_argument("--device",     default="cpu")
    p.add_argument("--overwrite",  action="store_true",
                   help="Re-cache datasets that already have a cache file")
    p.add_argument("--dataset",    default=None,
                   choices=["dfdc", "forensynths", "genimage", "pergen"],
                   help="Only re-cache this one dataset (skips the others)")
    p.add_argument("--pergen_root", default="per-gen-dataset",
                   help="Root folder of per-gen-dataset (used with --dataset pergen)")
    p.add_argument("--skip-pergen", action="store_true",
                   help="Build only the training caches (dfdc/forensynths/"
                        "genimage); skip the per-gen sets (cache those later, "
                        "they're only needed for evaluation)")
    p.add_argument("--pergen_only", default=None,
                   help="When caching per-gen, only (re)build this one generator "
                        "subfolder (e.g. DFDC). Skips the other generators.")
    p.add_argument("--cache_dir", default=None,
                   help="Override the cache output directory (default: read from config). "
                        "Useful for building a separate test-set cache without overwriting "
                        "the training cache.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────

def cache_one_dataset(
    name: str,
    root: Path,
    extractor: CLIPExtractor,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    transform,
    overwrite: bool,
):
    if cache_path.exists() and not overwrite:
        size_gb = cache_path.stat().st_size / 1e9
        print(f"  [{name}] already cached ({size_gb:.1f} GB) -> skipping "
              f"(use --overwrite to redo)")
        return

    def _find_dir(base, names):
        for n in names:
            d = base / n
            if d.is_dir():
                return d
        return None

    real_dir = _find_dir(root, ("real", "real-625", "real-5000", "1k_real"))
    fake_dir = _find_dir(root, ("fake", "fake-625", "fake-5000", "1k_fake"))

    real_imgs = sorted(_collect_images(real_dir)) if real_dir else []
    fake_imgs = sorted(_collect_images(fake_dir)) if fake_dir else []
    all_imgs  = real_imgs + fake_imgs

    if not all_imgs:
        print(f"  [{name}] WARNING: no images found in {root}")
        return

    print(f"  [{name}] {len(all_imgs)} images ({len(real_imgs)} real + "
          f"{len(fake_imgs)} fake)")

    paths_out:    list[str]          = []
    features_out: list[torch.Tensor] = []  # each: [B, 256, 1024] float16

    extractor.eval()
    t0 = time.time()

    with torch.no_grad():
        for start in tqdm(range(0, len(all_imgs), batch_size),
                          desc=f"  caching {name}", unit="batch"):
            batch_paths = all_imgs[start : start + batch_size]
            batch_imgs  = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_imgs.append(transform(img))
                except Exception:
                    batch_imgs.append(torch.zeros(3, 224, 224))

            imgs_t  = torch.stack(batch_imgs).to(device)         # [B, 3, 224, 224]
            # Run only the frozen ViT (skip proj; we cache raw 1024-d tokens
            # so the trainable projection can still be learned during training)
            tokens  = extractor._extract_patch_tokens(imgs_t)    # [B, 256, 1024]
            tokens  = tokens.cpu().to(torch.float16)             # save as float16

            paths_out.extend(str(p) for p in batch_paths)
            features_out.append(tokens)

    features_t = torch.cat(features_out, dim=0)  # [N, 256, 1024]
    torch.save({"paths": paths_out, "features": features_t}, cache_path)

    elapsed  = time.time() - t0
    size_gb  = cache_path.stat().st_size / 1e9
    print(f"  [{name}] saved {len(paths_out)} entries | "
          f"{size_gb:.2f} GB | {elapsed/60:.1f} min")


# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)
    device = torch.device(args.device)

    # Authenticate with HF Hub before any model download
    setup_hf_auth()

    cache_dir = Path(
        args.cache_dir if args.cache_dir
        else cfg.get("data", {}).get("clip_cache_dir", "datasets/clip_cache")
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Build extractor (load_visual=True so we can run _extract_patch_tokens)
    m_cfg = cfg.get("model", {})
    print("[cache] Loading CLIP ViT-L/14 (this may take a minute) ...")
    extractor = CLIPExtractor(
        model_name=m_cfg.get("clip_model",   "ViT-L-14-quickgelu"),
        pretrained="openai",
        out_dim=m_cfg.get("feature_dim", 768),
        load_visual=True,   # need the ViT to extract tokens
    ).to(device)

    # Use eval (deterministic) transforms – no augmentation for caching.
    # During training, augmentation is applied to the raw image for the
    # wavelet branch; the cached tokens come from the clean image.
    transform = build_transforms(image_size=224, augment=False)

    d_cfg = cfg.get("data", {})
    datasets = {
        "dfdc":        d_cfg.get("dfdc_root",        ""),
        "forensynths": d_cfg.get("forensynths_root", ""),
        "genimage":    d_cfg.get("genimage_root",    ""),
    }

    print(f"[cache] Writing to: {cache_dir}")
    print(f"[cache] Batch size: {args.batch_size}  Device: {device}\n")

    for name, root_str in datasets.items():
        if args.dataset and name != args.dataset:
            print(f"  [{name}] skipped (--dataset={args.dataset})")
            continue
        if not root_str or not Path(root_str).exists():
            print(f"  [{name}] path not found, skipping.")
            continue
        cache_path = cache_dir / f"{name}_clip.pt"
        cache_one_dataset(
            name=name,
            root=Path(root_str),
            extractor=extractor,
            cache_path=cache_path,
            device=device,
            batch_size=args.batch_size,
            transform=transform,
            overwrite=args.overwrite,
        )

    # ── Per-gen-dataset ──────────────────────────────────────────────
    if args.skip_pergen:
        print("\n[cache] --skip-pergen set: skipping per-gen caches "
              "(build them later for evaluation).")
    if args.dataset in (None, "pergen") and not args.skip_pergen:
        pergen_root = Path(args.pergen_root)
        if pergen_root.exists():
            print(f"\n[cache] Caching per-gen-dataset from: {pergen_root}")
            for gen_dir in sorted(pergen_root.iterdir()):
                if not gen_dir.is_dir():
                    continue
                gen_name   = gen_dir.name
                if args.pergen_only and gen_name != args.pergen_only:
                    continue
                cache_path = cache_dir / f"pergen_{gen_name}_clip.pt"
                cache_one_dataset(
                    name=f"pergen_{gen_name}",
                    root=gen_dir,
                    extractor=extractor,
                    cache_path=cache_path,
                    device=device,
                    batch_size=args.batch_size,
                    transform=transform,
                    overwrite=args.overwrite,
                )
        else:
            print(f"  [pergen] path not found: {pergen_root}, skipping.")

    print(f"\n[cache] All done.  Cache directory: {cache_dir}")
    print("[cache] Run  python train.py  to start training.")


if __name__ == "__main__":
    main()
