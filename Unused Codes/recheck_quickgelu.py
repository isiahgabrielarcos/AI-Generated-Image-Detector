"""
recheck_quickgelu.py
────────────────────
Re-check the feature ceiling AFTER two fixes are applied together:
  1. Equalized training data (datasets_eq/, no resolution/format shortcut)
  2. Correct CLIP activation (ViT-L-14-quickgelu instead of plain ViT-L-14)

Computes mean-pooled CLIP features LIVE for a subset of the equalized
training images and a subset of every per-gen test set (the old per-gen
caches use the WRONG activation, so they cannot be reused here), fits a
linear probe, and reports cross-gen accuracy.

Compare the MEAN against:
    biased + wrong-gelu          : 58.5%
    equalized + wrong-gelu       : 57.2%
If quickgelu lifts it meaningfully -> proceed to full cache rebuild + retrain.
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from models.clip_extractor import CLIPExtractor
from data.dataset import build_transforms
from utils import setup_hf_auth

random.seed(42)
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

EQ_ROOT  = Path("datasets_eq")
EQ_SETS  = ["DFDC", "ForenSynths", "GenImage"]
PG_ROOT  = Path("per-gen-dataset")
PERGEN = ["DFDC", "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN",
          "StarGAN", "GauGAN", "Deepfake", "PNDM", "Guided", "DALL-E",
          "VQ-Diffusion"]


def find_dir(root, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def list_imgs(d, limit):
    files = [p for p in d.glob("*") if p.suffix.lower() in IMG_EXTS]
    random.shuffle(files)
    return files[:limit]


@torch.no_grad()
def meanpool(paths, extractor, transform, device, bs=16, desc="clip"):
    out = []
    for i in tqdm(range(0, len(paths), bs), desc=f"    {desc}", unit="batch"):
        batch = paths[i:i+bs]
        ims = []
        for p in batch:
            try:
                ims.append(transform(Image.open(p).convert("RGB")))
            except Exception:
                ims.append(torch.zeros(3, 224, 224))
        x = torch.stack(ims).to(device)
        tok = extractor._extract_patch_tokens(x)
        out.append(tok.mean(dim=1).cpu().float())
    return torch.cat(out).numpy() if out else np.zeros((0, 1024))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=500,
                    help="train images per class per dataset")
    ap.add_argument("--n_test", type=int, default=300,
                    help="test images per class per generator")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    setup_hf_auth()
    device = torch.device("cpu")

    print(f"[recheck] threads={args.threads}  n_train={args.n_train}  "
          f"n_test={args.n_test}", flush=True)
    print("[recheck] loading CLIP ViT-L-14-quickgelu ...", flush=True)
    extractor = CLIPExtractor(model_name="ViT-L-14-quickgelu",
                              pretrained="openai", out_dim=768,
                              load_visual=True).to(device).eval()
    transform = build_transforms(image_size=224, augment=False)

    # ── training subset (equalized) ──────────────────────────────────
    tr_paths, tr_y = [], []
    for ds in EQ_SETS:
        for cls, lab in (("real", 0), ("fake", 1)):
            d = EQ_ROOT / ds / cls
            if not d.is_dir():
                print(f"  [warn] missing {d}")
                continue
            fs = list_imgs(d, args.n_train)
            tr_paths += fs; tr_y += [lab] * len(fs)
    print(f"[recheck] {len(tr_paths)} train imgs -> extracting", flush=True)
    X_train = meanpool(tr_paths, extractor, transform, device, desc="train")
    y_train = np.array(tr_y)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(X_train)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(
        scaler.transform(X_train), y_train)
    print(f"[recheck] linear-probe TRAIN acc = "
          f"{clf.score(scaler.transform(X_train), y_train)*100:.1f}%\n", flush=True)

    # ── per-gen test ─────────────────────────────────────────────────
    print(f"  {'Generator':<15} {'ACC':>7}", flush=True)
    print("  " + "-" * 25)
    accs = []
    for gen in PERGEN:
        groot = PG_ROOT / gen
        rdir = find_dir(groot, ("1k_real", "real", "real-625", "0_real"))
        fdir = find_dir(groot, ("1k_fake", "fake", "fake-625", "1_fake"))
        if rdir is None or fdir is None:
            print(f"  {gen:<15} (dirs not found)")
            continue
        rp = list_imgs(rdir, args.n_test); fp = list_imgs(fdir, args.n_test)
        Xg = meanpool(rp + fp, extractor, transform, device, desc=gen)
        yg = np.array([0]*len(rp) + [1]*len(fp))
        acc = clf.score(scaler.transform(Xg), yg)
        accs.append(acc)
        print(f"  {gen:<15} {acc*100:>6.1f}%", flush=True)
    print("  " + "-" * 25)
    mean = float(np.mean(accs)) if accs else 0.0
    print(f"  {'MEAN':<15} {mean*100:>6.1f}%\n")
    print("  Reference points:")
    print("    biased   + wrong-gelu : 58.5%")
    print("    equalized + wrong-gelu: 57.2%")
    print(f"    equalized + quickgelu : {mean*100:.1f}%   <-- this run")


if __name__ == "__main__":
    main()
