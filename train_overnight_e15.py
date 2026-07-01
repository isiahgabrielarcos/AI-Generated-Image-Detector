"""
train_overnight_e15.py
──────────────────────
Long unattended 1e-15 training run designed to be started before bed and
reviewed in the morning.

What it does
────────────
  • Uses ALL CPU cores (no affinity cap) at normal priority.
  • Loads all three dataset caches (uses your RAM, as requested).
  • Trains for the configured number of epochs with NO early stopping, so it
    cannot quit early on the flat AUC.
  • After EVERY epoch it appends a row to a CSV and regenerates a PNG plot of
    val ACC / AUC vs epoch — so even if you wake up mid-run, the progress so
    far is already on disk.
  • Prints a clear verdict at the end (flatline confirmed or not).

Run:
    python train_overnight_e15.py
    python train_overnight_e15.py --config configs/lr_1e-15_overnight.yaml --epochs 45

Outputs:
    results/lr_1e-15_overnight/progress.csv
    results/lr_1e-15_overnight/progress.png
    checkpoints/lr_1e-15_overnight/  (periodic checkpoints, resumable)
"""
import argparse
import csv
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

import matplotlib
matplotlib.use("Agg")               # headless: just save PNGs, no display
import matplotlib.pyplot as plt

from models import build_detector
from data.dataset import build_dataloaders
from losses import BinaryFocalLoss
from train import build_optimizer, build_scheduler, train_one_epoch, evaluate
from utils import compute_all_metrics, setup_hf_auth


@contextmanager
def noctx():
    yield


def save_plot(rows, png_path: Path, lr_str: str):
    if not rows:
        return
    eps   = [r["epoch"]    for r in rows]
    accs  = [r["val_acc"]  for r in rows]
    aucs  = [r["val_auc"]  for r in rows]
    plt.figure(figsize=(9, 5))
    plt.axhline(0.5, color="gray", ls="--", lw=1, label="random baseline (0.5)")
    plt.plot(eps, aucs, marker="o", label="val AUC")
    plt.plot(eps, accs, marker="s", label="val ACC")
    plt.ylim(0.0, 1.0)
    plt.xlabel("epoch")
    plt.ylabel("score")
    plt.title(f"Learning rate = {lr_str}: val metrics per epoch\n"
              f"(flat near 0.5 = the model never learns)")
    plt.legend(loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    plt.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lr_1e-15_overnight.yaml")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs from config")
    ap.add_argument("--out", default="results/lr_1e-15_overnight")
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")

    # ── Use ALL cores, normal priority (no throttling) ────────────────
    n_threads = os.cpu_count() or 8
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(2)

    cfg = yaml.safe_load(open(args.config))
    t_cfg = cfg["training"]
    epochs = args.epochs or t_cfg.get("epochs", 45)
    cnn_lr = t_cfg.get("cnn_lr"); bb_lr = t_cfg.get("backbone_lr")
    lr_str = f"{cnn_lr:.0e}"

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "progress.csv"
    png_path = out_dir / "progress.png"
    ckpt_dir = Path(cfg["logging"].get("save_dir", "checkpoints/lr_1e-15_overnight/"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("  OVERNIGHT 1e-15 RUN")
    print("=" * 64)
    print(f"  config       : {args.config}")
    print(f"  learning rate: cnn={cnn_lr:.0e}  backbone={bb_lr:.0e}")
    print(f"  epochs       : {epochs}  (early stopping disabled)")
    print(f"  CPU threads  : {n_threads} (all cores)")
    print(f"  CSV  -> {csv_path}")
    print(f"  PLOT -> {png_path}")
    print("=" * 64)

    # ── Data / model / optim ──────────────────────────────────────────
    print("[overnight] building datasets (loads all CLIP caches) ...")
    train_loader, val_loader = build_dataloaders(cfg)
    print("[overnight] building model ...")
    model = build_detector(cfg).to(device)
    criterion = BinaryFocalLoss(gamma=t_cfg.get("focal_gamma", 2.0))
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    grad_clip = t_cfg.get("gradient_clip", 1.0)
    accum     = t_cfg.get("accumulation_steps", 4)

    # snapshot initial weights so we can report how much they moved overnight
    w0 = torch.cat([p.detach().flatten().clone()
                    for p in model.parameters() if p.requires_grad])

    # ── CSV header ────────────────────────────────────────────────────
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "val_acc", "val_ap", "val_auc", "val_f1",
             "epoch_sec", "elapsed_min", "weights_changed_pct"])

    rows = []
    t_start = time.time()
    for epoch in range(epochs):
        t0 = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, scheduler,
                               criterion, device, grad_clip, accum, noctx)
        yt, yp = evaluate(model, val_loader, device, noctx)
        m = compute_all_metrics(yt, yp)
        dt = time.time() - t0
        elapsed_min = (time.time() - t_start) / 60.0

        # how much have weights drifted from init (proves frozen-ness)
        wn = torch.cat([p.detach().flatten()
                        for p in model.parameters() if p.requires_grad])
        changed_pct = 100.0 * float((wn != w0).sum()) / w0.numel()

        row = {"epoch": epoch + 1, "train_loss": loss,
               "val_acc": m["acc"], "val_ap": m["ap"], "val_auc": m["auc"],
               "val_f1": m["f1"], "epoch_sec": dt, "elapsed_min": elapsed_min,
               "weights_changed_pct": changed_pct}
        rows.append(row)

        eta_min = (dt * (epochs - epoch - 1)) / 60.0
        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss={loss:.4f}  "
              f"val_acc={m['acc']:.4f}  val_auc={m['auc']:.4f}  "
              f"({dt:.0f}s, elapsed {elapsed_min:.0f}m, ETA {eta_min:.0f}m)  "
              f"weights_moved={changed_pct:.3f}%")

        # append CSV row + refresh plot every epoch (crash/wake safe)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [row["epoch"], f"{loss:.6f}", f"{m['acc']:.6f}",
                 f"{m['ap']:.6f}", f"{m['auc']:.6f}", f"{m['f1']:.6f}",
                 f"{dt:.1f}", f"{elapsed_min:.2f}", f"{changed_pct:.4f}"])
        save_plot(rows, png_path, lr_str)

        save_every = cfg["logging"].get("save_every", 10)
        if (epoch + 1) % save_every == 0:
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "optimizer": optimizer.state_dict()},
                       ckpt_dir / f"epoch_{epoch+1:03d}.pt")

    # ── Verdict ───────────────────────────────────────────────────────
    aucs = [r["val_auc"] for r in rows]
    accs = [r["val_acc"] for r in rows]
    total_h = (time.time() - t_start) / 3600.0
    print("\n" + "=" * 64)
    print("  OVERNIGHT RUN COMPLETE")
    print("=" * 64)
    print(f"  epochs run        : {len(rows)}")
    print(f"  total time        : {total_h:.2f} h")
    print(f"  val AUC  min/max  : {min(aucs):.4f} / {max(aucs):.4f}  "
          f"(spread {max(aucs)-min(aucs):.4f})")
    print(f"  val ACC  min/max  : {min(accs):.4f} / {max(accs):.4f}")
    print(f"  final weights moved from init: {rows[-1]['weights_changed_pct']:.3f}%")
    verdict = ("FLATLINE CONFIRMED — AUC never escaped the ~0.5 random band. "
               "No learning occurred at 1e-15."
               if max(aucs) < 0.60 else
               "AUC moved unexpectedly — inspect the curve.")
    print(f"  VERDICT: {verdict}")
    print(f"\n  See {png_path} and {csv_path}")


if __name__ == "__main__":
    main()
