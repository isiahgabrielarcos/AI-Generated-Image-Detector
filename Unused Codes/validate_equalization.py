"""
validate_equalization.py
────────────────────────
Cheap go/no-go test: did equalizing the training data raise the cross-gen
ceiling?  Computes CLIP features for a SUBSET of the equalized training
images (no full cache needed), fits a linear probe, and tests it on the
EXISTING per-gen caches (per-gen data was not modified, so those caches
are still valid).

Compare the printed cross-gen MEAN against the pre-equalization baseline
of 58.5%.  If it rises meaningfully -> equalization worked, do the full
cache rebuild + retrain.  If flat -> the problem is deeper than data bias.
"""

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.clip_extractor import CLIPExtractor
from data.dataset import build_transforms
from utils import setup_hf_auth

random.seed(42)

EQ_ROOT   = Path("datasets_eq")
EQ_SETS   = ["DFDC", "ForenSynths", "GenImage"]
CACHE_DIR = Path("datasets/clip_cache")
PERGEN = ["DFDC", "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN",
          "StarGAN", "GauGAN", "Deepfake", "PNDM", "Guided", "DALL-E",
          "VQ-Diffusion"]
IMG_EXTS = {".png", ".jpg", ".jpeg"}


def label_of(path: str) -> int:
    p = path.lower()
    if "fake" in p: return 1
    if "real" in p: return 0
    return -1


@torch.no_grad()
def clip_meanpool(paths, extractor, transform, device, bs=16):
    """Return [N,1024] mean-pooled CLIP patch tokens for given image paths."""
    out = []
    for i in tqdm(range(0, len(paths), bs), desc="    clip", unit="batch"):
        batch = paths[i:i+bs]
        ims = []
        for p in batch:
            try:
                ims.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                ims.append(torch.zeros(3, 224, 224))
        x = torch.stack(ims).to(device)
        tok = extractor._extract_patch_tokens(x)      # [B,256,1024]
        out.append(tok.mean(dim=1).cpu().float())     # mean-pool -> [B,1024]
    return torch.cat(out).numpy()


def load_pergen_pooled(gen):
    cp = CACHE_DIR / f"pergen_{gen}_clip.pt"
    if not cp.exists():
        return None, None
    d = torch.load(cp, map_location="cpu", weights_only=True)
    X = d["features"].float().mean(dim=1).numpy()
    y = np.array([label_of(p) for p in d["paths"]])
    keep = y >= 0
    return X[keep], y[keep]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500,
                    help="images sampled per class per dataset (default 500)")
    ap.add_argument("--threads", type=int, default=4,
                    help="torch CPU threads (default 4; high counts can stall)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    setup_hf_auth()
    device = torch.device("cpu")

    print(f"[validate] threads={args.threads}  n_per_class={args.n}", flush=True)
    print("[validate] loading CLIP ViT-L/14 ...", flush=True)
    extractor = CLIPExtractor(model_name="ViT-L-14", pretrained="openai",
                              out_dim=768, load_visual=True).to(device).eval()
    transform = build_transforms(image_size=224, augment=False)

    # ── sample equalized training images ─────────────────────────────
    train_paths = []
    for ds in EQ_SETS:
        for cls in ("real", "fake"):
            d = EQ_ROOT / ds / cls
            files = [p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS]
            random.shuffle(files)
            train_paths += files[:args.n]
    print(f"[validate] sampled {len(train_paths)} equalized training images",
          flush=True)
    print("[validate] extracting CLIP features (this is the slow part) ...",
          flush=True)

    X_train = clip_meanpool(train_paths, extractor, transform, device)
    y_train = np.array([label_of(str(p)) for p in train_paths])

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(X_train), y_train)
    print(f"[validate] linear-probe TRAIN acc = "
          f"{clf.score(scaler.transform(X_train), y_train)*100:.1f}%\n")

    # ── test on existing per-gen caches ──────────────────────────────
    print(f"  {'Generator':<15} {'ACC':>7}")
    print("  " + "-" * 25)
    accs = []
    for gen in PERGEN:
        Xg, yg = load_pergen_pooled(gen)
        if Xg is None:
            print(f"  {gen:<15} (no cache)")
            continue
        acc = clf.score(scaler.transform(Xg), yg)
        accs.append(acc)
        print(f"  {gen:<15} {acc*100:>6.1f}%")
    print("  " + "-" * 25)
    mean = float(np.mean(accs)) if accs else 0.0
    print(f"  {'MEAN':<15} {mean*100:>6.1f}%")
    print(f"\n  Baseline (biased training): 58.5%")
    print(f"  After equalization:         {mean*100:.1f}%")
    delta = (mean - 0.585) * 100
    print(f"  Change: {delta:+.1f} points  "
          f"-> {'WORKED, proceed to retrain' if delta > 3 else 'flat, problem is deeper'}")


if __name__ == "__main__":
    main()
