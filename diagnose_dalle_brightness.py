"""
diagnose_dalle_brightness.py
────────────────────────────
Quick diagnostic: evaluates DALL-E with the existing best_model.pt twice —
  (A) images as-is (dark originals)
  (B) images with per-image histogram stretch to [0,255] in memory

No files are touched on disk. If (B) is materially better than (A),
brightness is the root cause and fixing the dataset is worth doing.

Run:
  python diagnose_dalle_brightness.py
  python diagnose_dalle_brightness.py --checkpoint checkpoints/best_model.pt
"""
import argparse
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from models import build_detector
from utils import compute_all_metrics, setup_hf_auth
from data.dataset import ClipFeatureCache, clip_collate_fn

_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
_FAKE_NAMES = ("fake", "fake-625", "fake-5000", "1k_fake")
_REAL_NAMES = ("real", "real-625", "real-5000", "1k_real")


def _find(root, names):
    for n in names:
        d = root / n
        if d.is_dir():
            return d
    return None


def _collect(folder):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in _IMG_EXTS)


class DalleDataset(Dataset):
    def __init__(self, gen_root: Path, stretch: bool, clip_cache=None):
        fake_dir = _find(gen_root, _FAKE_NAMES)
        real_dir = _find(gen_root, _REAL_NAMES)
        self.samples = (
            [(p, 1) for p in _collect(fake_dir)] +
            [(p, 0) for p in _collect(real_dir)]
        )
        self.stretch = stretch
        self.clip_cache = clip_cache
        self.base_tf = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))

        if self.stretch:
            # Per-image histogram stretch: expand darkest→0, brightest→255
            img = ImageOps.autocontrast(img, cutoff=0)

        img_t = self.base_tf(img)
        lab_t = torch.tensor(label, dtype=torch.float32)

        if self.clip_cache is not None:
            tok = self.clip_cache.get(path)
            if tok is None:
                tok = torch.zeros(256, 1024, dtype=torch.float32)
            return img_t, tok, lab_t
        return img_t, lab_t


@contextmanager
def noctx():
    yield


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    yt, yp = [], []
    for batch in loader:
        if len(batch) == 3:
            imgs, toks, labels = batch
            imgs = imgs.to(device); toks = toks.to(device)
            logits = model(imgs, clip_tokens=toks)
        else:
            imgs, labels = batch
            imgs = imgs.to(device)
            logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze(-1)
        yt.extend(labels.cpu().tolist())
        yp.extend(probs.cpu().tolist())
    return yt, yp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    ap.add_argument("--generators_root", default="per-gen-dataset")
    ap.add_argument("--clip_cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")
    cfg = yaml.safe_load(open(args.config))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_detector(cfg, force_load_visual=False).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}  (epoch={ckpt.get('epoch','?')}  best_auc={ckpt.get('best_auc','?')})\n")

    # Try to load CLIP cache for DALL-E
    cache_dir = Path(args.clip_cache_dir)
    cache_names = [n for n in ["pergen_DALL-E", "DALL-E"] if (cache_dir / f"{n}_clip.pt").exists()]
    clip_cache = ClipFeatureCache(cache_dir, cache_names) if cache_names else None
    if clip_cache and clip_cache.available:
        print(f"Using CLIP cache: {cache_names}")
    else:
        clip_cache = None
        print("No CLIP cache found for DALL-E — CLIP branch will use zeros (same for both runs, so comparison is still fair)\n")

    gen_root = Path(args.generators_root) / "DALL-E"
    collate = clip_collate_fn if clip_cache else None

    results = {}
    for label, stretch in [("(A) Original (dark)", False), ("(B) Brightness-stretched", True)]:
        ds = DalleDataset(gen_root, stretch=stretch, clip_cache=clip_cache)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=collate)
        yt, yp = run_eval(model, loader, device)
        m = compute_all_metrics(yt, yp)
        results[label] = m
        print(f"{label}")
        print(f"  ACC={m['acc']:.4f}  AUC={m['auc']:.4f}  AP={m['ap']:.4f}  "
              f"Recall={m['recall']:.4f}  F1={m['f1']:.4f}")
        print()

    # Delta
    a = results["(A) Original (dark)"]
    b = results["(B) Brightness-stretched"]
    d_acc = b['acc'] - a['acc']
    d_auc = b['auc'] - a['auc']
    d_ap  = b['ap']  - a['ap']
    d_rec = b['recall'] - a['recall']
    print("-" * 52)
    print(f"Delta (B-A):  ACC={d_acc:+.4f}  AUC={d_auc:+.4f}  AP={d_ap:+.4f}  Recall={d_rec:+.4f}")
    print()

    if d_auc > 0.03 or d_ap > 0.03:
        print("VERDICT: Brightness IS the issue — fixing the dataset is worth doing.")
        print("         Plan: stretch images on disk → rebuild DALL-E CLIP cache → retrain.")
    elif d_auc > 0.01 or d_ap > 0.01:
        print("VERDICT: Brightness has a modest effect. Fixing may help slightly.")
        print("         DALL-E may also just be a genuinely hard generator (it is).")
    else:
        print("VERDICT: Brightness is NOT the main issue.")
        print("         DALL-E is simply a hard generator for the current model.")


if __name__ == "__main__":
    main()
