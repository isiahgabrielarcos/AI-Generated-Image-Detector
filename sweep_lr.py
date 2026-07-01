"""
sweep_lr.py
───────────
Learning-rate sweep: train the SAME model (same data, same random init) at a
series of learning rates and compare how much each one learns.  Built to show
the panel the threshold where float32 training goes from "frozen" to "works".

Design for a fair comparison
────────────────────────────
  • Data is loaded ONCE and reused for every learning rate.
  • Before each run the RNG is re-seeded, so every learning rate starts from the
    IDENTICAL random initialization — the learning rate is the only variable.
  • Every run uses the same epoch count (so the x-axis is comparable).
  • Per-epoch metrics are written to one CSV per LR, plus two summary plots.

Resumable
─────────
  Each LR's CSV is written as it completes.  Re-running skips any LR whose CSV
  already has the full epoch count, so a long 45-epoch sweep can be done across
  several nights (Ctrl-C any time, just run again).

Usage
─────
    # default: 1e-14 ... 1e-7, 25 epochs each (~10 h)
    python sweep_lr.py

    # full rigor, matches your 1e-15/1e-4 runs (~19 h, resumable)
    python sweep_lr.py --epochs 45

    # quick look (~6 h)
    python sweep_lr.py --epochs 15

    # custom LR list
    python sweep_lr.py --lrs 1e-12 1e-10 1e-8 1e-7

Outputs (results/lr_sweep/)
    lr_1e-XX.csv              per-epoch metrics for each LR
    summary.csv              final/best AUC + weights-moved per LR
    auc_vs_epoch.png         all LR curves overlaid
    final_auc_vs_lr.png      the money plot: final AUC vs learning rate
"""
import argparse
import csv
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from models import build_detector
from data.dataset import build_dataloaders
from losses import BinaryFocalLoss
from train import build_optimizer, build_scheduler, train_one_epoch, evaluate
from utils import compute_all_metrics, setup_hf_auth


@contextmanager
def noctx():
    yield


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_tag(lr: float) -> str:
    return f"{lr:.0e}".replace("-0", "-")   # 1e-14 etc.


def run_one_lr(lr, cfg, epochs, train_loader, val_loader, seed, csv_path, device):
    """Train one LR from a fresh seeded init; write per-epoch CSV; return rows."""
    seed_all(seed)                                   # identical init for every LR
    cfg = {**cfg, "training": {**cfg["training"],
                               "cnn_lr": lr, "backbone_lr": lr, "epochs": epochs}}
    model = build_detector(cfg).to(device)
    criterion = BinaryFocalLoss(gamma=cfg["training"].get("focal_gamma", 2.0))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    grad_clip = cfg["training"].get("gradient_clip", 1.0)
    accum     = cfg["training"].get("accumulation_steps", 4)

    w0 = torch.cat([p.detach().flatten().clone()
                    for p in model.parameters() if p.requires_grad])

    rows = []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_acc", "val_auc", "val_ap",
             "weights_changed_pct"])

    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               criterion, device, grad_clip, accum, noctx)
        yt, yp = evaluate(model, val_loader, device, noctx)
        m = compute_all_metrics(yt, yp)
        wn = torch.cat([p.detach().flatten()
                        for p in model.parameters() if p.requires_grad])
        changed = 100.0 * float((wn != w0).sum()) / w0.numel()
        rows.append({"epoch": epoch + 1, "val_acc": m["acc"],
                     "val_auc": m["auc"], "weights": changed})
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch + 1, f"{loss:.6f}", f"{m['acc']:.6f}",
                                    f"{m['auc']:.6f}", f"{m['ap']:.6f}",
                                    f"{changed:.4f}"])
        print(f"    lr={lr:.0e}  epoch {epoch+1:>2}/{epochs}  "
              f"auc={m['auc']:.4f}  acc={m['acc']:.4f}  "
              f"weights_moved={changed:.2f}%  ({time.time()-t0:.0f}s)")

    del model, optimizer, scheduler
    return rows


def make_plots(all_results, out_dir):
    """all_results: dict {lr: list of per-epoch row dicts}"""
    lrs = sorted(all_results.keys())
    cmap = plt.get_cmap("viridis")

    # ── Plot 1: AUC vs epoch, one line per LR ─────────────────────────
    plt.figure(figsize=(10, 6))
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="random (0.5)")
    for i, lr in enumerate(lrs):
        rows = all_results[lr]
        eps = [r["epoch"] for r in rows]; aucs = [r["val_auc"] for r in rows]
        plt.plot(eps, aucs, marker=".", color=cmap(i / max(1, len(lrs)-1)),
                 label=f"{lr:.0e}")
    plt.ylim(0.0, 1.0); plt.xlabel("epoch"); plt.ylabel("val AUC")
    plt.title("Val AUC per epoch across learning rates\n"
              "(flat ~0.5 = no learning; climbing = learning)")
    plt.legend(title="learning rate", ncol=2, fontsize=8)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "auc_vs_epoch.png", dpi=120); plt.close()

    # ── Plot 2: final AUC vs learning rate (the threshold plot) ───────
    finals = [all_results[lr][-1]["val_auc"] for lr in lrs]
    bests  = [max(r["val_auc"] for r in all_results[lr]) for lr in lrs]
    plt.figure(figsize=(9, 5.5))
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="random (0.5)")
    plt.semilogx(lrs, finals, marker="o", label="final-epoch AUC")
    plt.semilogx(lrs, bests, marker="s", ls="--", alpha=0.6, label="best AUC")
    plt.xlabel("learning rate (log scale)"); plt.ylabel("val AUC")
    plt.ylim(0.0, 1.0)
    plt.title("Learning rate vs final val AUC\n"
              "the float32 'wall': below it the model can't learn")
    plt.legend(); plt.grid(alpha=0.3, which="both"); plt.tight_layout()
    plt.savefig(out_dir / "final_auc_vs_lr.png", dpi=120); plt.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr_1e-15_overnight.yaml",
                    help="Config used for data/model settings (its LR is ignored)")
    ap.add_argument("--lrs", nargs="*", type=float,
                    default=[1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/lr_sweep")
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")
    torch.set_num_threads(os.cpu_count() or 8)

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  LEARNING-RATE SWEEP")
    print("=" * 64)
    print(f"  learning rates : {[f'{l:.0e}' for l in args.lrs]}")
    print(f"  epochs each    : {args.epochs}  (same for all -> fair comparison)")
    print(f"  seed           : {args.seed}  (identical init for every LR)")
    est_h = len(args.lrs) * args.epochs * 190 / 3600
    print(f"  rough estimate : ~{est_h:.1f} h total  (resumable — Ctrl-C anytime)")
    print("=" * 64)

    print("[sweep] loading data once (reused for all LRs) ...")
    train_loader, val_loader = build_dataloaders(cfg)

    all_results = {}
    for lr in args.lrs:
        csv_path = out_dir / f"lr_{lr_tag(lr)}.csv"
        # resume: skip if this LR already finished all epochs
        if csv_path.exists():
            done = sum(1 for _ in open(csv_path)) - 1   # minus header
            if done >= args.epochs:
                print(f"[sweep] lr={lr:.0e} already done ({done} epochs) — skipping")
                all_results[lr] = [
                    {"epoch": int(r["epoch"]), "val_acc": float(r["val_acc"]),
                     "val_auc": float(r["val_auc"]),
                     "weights": float(r["weights_changed_pct"])}
                    for r in csv.DictReader(open(csv_path))]
                continue
        print(f"\n[sweep] === learning rate {lr:.0e} ===")
        rows = run_one_lr(lr, cfg, args.epochs, train_loader, val_loader,
                          args.seed, csv_path, device)
        all_results[lr] = rows
        make_plots(all_results, out_dir)      # refresh plots after each LR

    # ── summary table ─────────────────────────────────────────────────
    with open(out_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["learning_rate", "final_auc", "best_auc",
                    "final_acc", "final_weights_moved_pct"])
        for lr in sorted(all_results.keys()):
            rows = all_results[lr]
            w.writerow([f"{lr:.0e}", f"{rows[-1]['val_auc']:.4f}",
                        f"{max(r['val_auc'] for r in rows):.4f}",
                        f"{rows[-1]['val_acc']:.4f}",
                        f"{rows[-1]['weights']:.3f}"])

    make_plots(all_results, out_dir)
    print("\n" + "=" * 64)
    print("  SWEEP COMPLETE")
    print("=" * 64)
    print(f"  {'lr':>8}  {'final AUC':>10}  {'weights moved':>14}")
    for lr in sorted(all_results.keys()):
        rows = all_results[lr]
        print(f"  {lr:>8.0e}  {rows[-1]['val_auc']:>10.4f}  "
              f"{rows[-1]['weights']:>13.2f}%")
    print(f"\n  plots -> {out_dir/'auc_vs_epoch.png'}")
    print(f"        -> {out_dir/'final_auc_vs_lr.png'}")
    print(f"  table -> {out_dir/'summary.csv'}")


if __name__ == "__main__":
    main()
