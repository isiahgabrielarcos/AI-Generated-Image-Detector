"""
train_pergen_split.py
─────────────────────
Training VARIATION that replicates Man & Cho's likely protocol:
"seen-generator 80/20" — train on 80% of every generator, test on the
DISJOINT 20% held-out of each generator.

This is NOT the unseen cross-generator setup. The model sees each generator's
distribution during training and is tested on held-out images of the same
generators. Splits are strictly disjoint (stratified, fixed seed) so there is
no leakage. Report results as the "seen-generator (80/20)" protocol — a
separate experiment from the unseen cross-gen tables.

Reuses the existing per-gen CLIP cache (datasets_eq/clip_cache/pergen_*.pt) so
the frozen ViT is never run. The wavelet branch sees the (augmented) image;
the CLIP branch uses cached clean tokens — same design as the main training.

Usage:
  python train_pergen_split.py
  python train_pergen_split.py --epochs 20 --test_frac 0.2 --seed 42
"""
import argparse, csv, sys, time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from models import build_detector
from losses import BinaryFocalLoss
from data.dataset import ClipFeatureCache, clip_collate_fn, build_transforms, _collect_images
from utils import compute_all_metrics, print_metrics, print_confusion_matrix, setup_hf_auth
from train import train_one_epoch, build_optimizer, build_scheduler, EarlyStopping, evaluate
from evaluate_per_generator import TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS

ALL_GEN = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS
FAKE_NAMES = ("1k_fake", "fake", "fake-625", "fake-5000", "1_fake")
REAL_NAMES = ("1k_real", "real", "real-625", "real-5000", "0_real")


@contextmanager
def noctx():
    yield


def _find(root, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


class ListDataset(Dataset):
    def __init__(self, samples, transform, clip_cache):
        self.samples = samples
        self.transform = transform
        self.clip_cache = clip_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        x = self.transform(img)
        lab = torch.tensor(label, dtype=torch.float32)
        if self.clip_cache is not None:
            tok = self.clip_cache.get(path)
            if tok is None:
                tok = torch.zeros(256, 1024)
            return x, tok, lab
        return x, lab


def collect(gen_dir):
    fdir = _find(gen_dir, FAKE_NAMES); rdir = _find(gen_dir, REAL_NAMES)
    fakes = _collect_images(fdir) if fdir else []
    reals = _collect_images(rdir) if rdir else []
    return [(str(p), 1) for p in fakes] + [(str(p), 0) for p in reals]


def strat_3way(samples, test_frac, val_frac, seed):
    """Stratified disjoint split -> (train, val, test) lists of (path,label)."""
    rng = np.random.default_rng(seed)
    by = {0: [], 1: []}
    for s in samples:
        by[s[1]].append(s)
    train, val, test = [], [], []
    for c in (0, 1):
        items = by[c]; rng.shuffle(items)
        n = len(items); n_te = int(round(n*test_frac)); n_va = int(round(n*val_frac))
        test += items[:n_te]
        val += items[n_te:n_te+n_va]
        train += items[n_te+n_va:]
    rng.shuffle(train)
    return train, val, test


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--generators_root", default="per-gen-dataset")
    ap.add_argument("--clip_cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="checkpoints/pergen_split_best_model.pt")
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--out", default="results/pergen_split")
    ap.add_argument("--resume", default=None,
                    help="Path to a pergen_split_epoch_*.pt checkpoint to resume from")
    ap.add_argument("--eval_only", action="store_true",
                    help="Skip training; load --save checkpoint and evaluate on the held-out test split")
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 8))
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    gen_root = Path(args.generators_root)

    # ── cache ─────────────────────────────────────────────────────────
    cache_dir = Path(args.clip_cache_dir)
    names = [f"pergen_{g}" for g in ALL_GEN if (cache_dir / f"pergen_{g}_clip.pt").exists()]
    clip_cache = ClipFeatureCache(cache_dir, names) if names else None
    use_cache = clip_cache is not None and clip_cache.available
    collate = clip_collate_fn if use_cache else None

    # ── disjoint stratified split per generator ───────────────────────
    train_samples, val_samples = [], []
    test_by_gen = {}
    for g in ALL_GEN:
        gdir = gen_root / g
        if not gdir.exists():
            continue
        s = collect(gdir)
        if not s:
            continue
        tr, va, te = strat_3way(s, args.test_frac, args.val_frac, args.seed)
        train_samples += tr; val_samples += va; test_by_gen[g] = te
    print(f"[pergen-split] train={len(train_samples)}  val={len(val_samples)}  "
          f"test={sum(len(v) for v in test_by_gen.values())} "
          f"(80/20 seen-generator, disjoint, seed={args.seed})")

    # safety: assert train/test disjoint
    train_paths = {p for p, _ in train_samples} | {p for p, _ in val_samples}
    test_paths = {p for g in test_by_gen for p, _ in test_by_gen[g]}
    overlap = train_paths & test_paths
    assert not overlap, f"LEAKAGE: {len(overlap)} images in both train and test!"
    print(f"[pergen-split] disjoint check OK — 0 overlap between train/val and test\n")

    tf_train = build_transforms(image_size=224, augment=True)
    tf_eval = build_transforms(image_size=224, augment=False)
    train_loader = DataLoader(ListDataset(train_samples, tf_train, clip_cache),
                              batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate)
    val_loader = DataLoader(ListDataset(val_samples, tf_eval, clip_cache),
                            batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)

    # ── eval-only shortcut ────────────────────────────────────────────
    if args.eval_only:
        model = build_detector(cfg, force_load_visual=(not use_cache)).to(device)
        ckpt = torch.load(args.save, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        tf_eval = build_transforms(image_size=224, augment=False)
        rows = {}
        for g, te in test_by_gen.items():
            loader = DataLoader(ListDataset(te, tf_eval, clip_cache),
                                batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
            yt, yp = evaluate(model, loader, device, noctx)
            m = compute_all_metrics(yt, yp)
            rows[g] = (m["acc"]*100, m["ap"]*100)

        def block(title, gens):
            print(f"\n{title}\n  {'Generator':<14}{'ACC':>8}{'AP':>8}")
            accs=[]; aps=[]
            for g in gens:
                if g not in rows: continue
                a,p = rows[g]; accs.append(a); aps.append(p)
                print(f"  {g:<14}{a:>7.1f}{p:>8.1f}")
            if accs: print(f"  {'MEAN':<14}{np.mean(accs):>7.1f}{np.mean(aps):>8.1f}")

        print("\n" + "="*44)
        print(f"  SEEN-GENERATOR 80/20 — EVAL ONLY (held-out {int(args.test_frac*100)}% test)")
        print("="*44)
        block("Table 1 — DFDC (face)", TABLE1_GENERATORS)
        block("Table 2 — GAN (ForenSynths)", TABLE2_GENERATORS)
        block("Table 3 — Diffusion (GenImage)", TABLE3_GENERATORS)

        with open(out_dir / "pergen_split_results.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["generator", "acc", "ap"])
            for g in ALL_GEN:
                if g in rows: w.writerow([g, f"{rows[g][0]:.2f}", f"{rows[g][1]:.2f}"])
        print(f"\n[pergen-split] results -> {out_dir/'pergen_split_results.csv'}")
        return

    # ── model / optim ─────────────────────────────────────────────────
    model = build_detector(cfg, force_load_visual=(not use_cache)).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[pergen-split] trainable params: {n/1e6:.1f} M")
    criterion = BinaryFocalLoss(gamma=cfg.get("training", {}).get("focal_gamma", 2.0))
    optimizer = build_optimizer(model, cfg)
    accum = cfg.get("training", {}).get("accumulation_steps", 4)
    sched_cfg = {**cfg, "training": {**cfg["training"], "epochs": args.epochs}}
    scheduler = build_scheduler(optimizer, sched_cfg, len(train_loader))
    grad_clip = cfg.get("training", {}).get("gradient_clip", 1.0)
    stopper = EarlyStopping(patience=args.patience)
    best_auc = 0.0
    start_epoch = 0
    ckpt_dir = Path(args.save).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── resume ────────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "optimizer" not in ckpt:
            print(f"[pergen-split] WARNING: {args.resume} has no optimizer state — "
                  f"model weights loaded but training restarts with a fresh optimizer. "
                  f"Use a pergen_split_epoch_*.pt checkpoint for a clean resume.")
        model.load_state_dict(ckpt["model"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc = ckpt.get("best_auc", 0.0)
        print(f"[pergen-split] resumed from epoch {start_epoch}  best_auc={best_auc:.4f}\n")

    # ── train ─────────────────────────────────────────────────────────
    epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               criterion, device, grad_clip, accum, noctx)
        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        avg_t = sum(epoch_times[-3:]) / len(epoch_times[-3:])
        remaining = avg_t * (args.epochs - epoch - 1)
        eta_h = int(remaining // 3600)
        eta_m = int((remaining % 3600) // 60)
        print(f"Epoch [{epoch+1:3d}/{args.epochs}]  loss={loss:.4f}  "
              f"({elapsed:.0f}s)  ETA {eta_h}h {eta_m}m")

        yt, yp = evaluate(model, val_loader, device, noctx)
        m = compute_all_metrics(yt, yp)
        print_metrics(m, prefix=f"val ep{epoch+1}")
        print_confusion_matrix(m)

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc": best_auc, "metrics": m,
                "cfg": cfg, "protocol": "seen-generator-80-20",
            }, args.save)
            print(f"  * new best val AUC={best_auc:.4f} -> saved {args.save}")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = ckpt_dir / f"pergen_split_epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "best_auc": best_auc,
            }, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path.name}")

        if stopper.step(m["auc"]):
            print(f"[pergen-split] early stop at epoch {epoch+1}")
            break

    # ── load best, evaluate per-generator on the held-out 20% ─────────
    model.load_state_dict(torch.load(args.save, map_location=device, weights_only=False)["model"], strict=False)
    model.eval()
    rows = {}
    for g, te in test_by_gen.items():
        loader = DataLoader(ListDataset(te, tf_eval, clip_cache),
                            batch_size=16, shuffle=False, num_workers=0, collate_fn=collate)
        yt, yp = evaluate(model, loader, device, noctx)
        m = compute_all_metrics(yt, yp)
        rows[g] = (m["acc"]*100, m["ap"]*100)

    def block(title, gens):
        print(f"\n{title}\n  {'Generator':<14}{'ACC':>8}{'AP':>8}")
        accs=[]; aps=[]
        for g in gens:
            if g not in rows: continue
            a,p = rows[g]; accs.append(a); aps.append(p)
            print(f"  {g:<14}{a:>7.1f}{p:>8.1f}")
        if accs: print(f"  {'MEAN':<14}{np.mean(accs):>7.1f}{np.mean(aps):>8.1f}")

    print("\n" + "="*44)
    print(f"  SEEN-GENERATOR 80/20 — FULL MODEL (held-out {int(args.test_frac*100)}% test)")
    print("="*44)
    block("Table 1 — DFDC (face)", TABLE1_GENERATORS)
    block("Table 2 — GAN (ForenSynths)", TABLE2_GENERATORS)
    block("Table 3 — Diffusion (GenImage)", TABLE3_GENERATORS)

    with open(out_dir / "pergen_split_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["generator", "acc", "ap"])
        for g in ALL_GEN:
            if g in rows: w.writerow([g, f"{rows[g][0]:.2f}", f"{rows[g][1]:.2f}"])
    print(f"\n[pergen-split] best model  -> {args.save}")
    print(f"[pergen-split] results     -> {out_dir/'pergen_split_results.csv'}")
    print("[pergen-split] Report as 'seen-generator 80/20' protocol (disjoint splits, not unseen cross-gen).")


if __name__ == "__main__":
    main()
