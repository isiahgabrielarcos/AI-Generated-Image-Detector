"""
calibrate_and_eval.py
─────────────────────
Honest ways to raise per-generator ACC WITHOUT retraining and WITHOUT touching
the test set during training:

  1. Threshold calibration. ACC is reported at a 0.5 cutoff by default, but a
     model with high AP/AUC can have a sub-optimal 0.5 operating point. We pick
     ONE global decision threshold on the in-distribution VALIDATION split
     (never the test set) and report ACC at it. This is standard practice.

  2. Checkpoint ensembling (optional). Average the predicted probabilities of
     several checkpoints (e.g. epoch_010/015/020). Pure inference-time, no
     retraining, no leakage.

For transparency the script prints three ACC columns per generator:
  ACC@0.5      – the blind default (what you have now)
  ACC@val-thr  – HONEST, reportable (threshold chosen on validation only)
  ACC@oracle   – DIAGNOSTIC ONLY (best threshold on that test set; an upper
                 bound, NOT reportable — shows how much is pure thresholding)

AP and AUC are threshold-independent and unchanged.

Usage:
  python calibrate_and_eval.py --checkpoints checkpoints/epoch_020.pt
  python calibrate_and_eval.py --checkpoints checkpoints/epoch_010.pt checkpoints/epoch_015.pt checkpoints/epoch_020.pt
"""

import argparse, csv, gc, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             balanced_accuracy_score)

from models import build_detector
from data import build_dataloaders
from data.dataset import ClipFeatureCache, clip_collate_fn
from utils import setup_hf_auth
from evaluate_per_generator import (GeneratorDataset, infer,
                                    TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS)

ALL_GEN = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS


def best_threshold(y_true, y_prob):
    """Threshold maximizing balanced accuracy on the validation set."""
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    grid = np.linspace(0.02, 0.98, 193)
    best_t, best_b = 0.5, -1.0
    for t in grid:
        b = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if b > best_b:
            best_b, best_t = b, t
    return float(best_t), float(best_b)


def acc_at(y_true, y_prob, t):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    return float(((y_prob >= t).astype(int) == y_true).mean())


def oracle_acc(y_true, y_prob):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    grid = np.linspace(0.02, 0.98, 193)
    return max(acc_at(y_true, y_prob, t) for t in grid)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["checkpoints/epoch_020.pt"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--generators_root", default="per-gen-dataset")
    ap.add_argument("--clip_cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="results/calibrated")
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")
    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    gen_root = Path(args.generators_root)
    ckpts = args.checkpoints
    print(f"[calib] checkpoints: {ckpts}")
    print(f"[calib] ensembling {len(ckpts)} checkpoint(s)\n")

    # ── Phase 1: calibration on the in-distribution validation split ──
    print("[calib] Phase 1: validation calibration (loads training cache) ...")
    _, val_loader = build_dataloaders(cfg)
    val_true = None
    val_prob_sum = None
    for c in ckpts:
        model = build_detector(cfg, force_load_visual=False).to(device)
        model.load_state_dict(torch.load(c, map_location=device, weights_only=False)["model"], strict=False)
        model.eval()
        yt, yp = infer(model, val_loader, device)
        val_true = yt
        val_prob_sum = np.array(yp) if val_prob_sum is None else val_prob_sum + np.array(yp)
        del model; gc.collect()
    val_prob = val_prob_sum / len(ckpts)
    thr, bal = best_threshold(val_true, val_prob)
    val_acc_05 = acc_at(val_true, val_prob, 0.5)
    val_acc_thr = acc_at(val_true, val_prob, thr)
    print(f"[calib] validation: ACC@0.5={val_acc_05*100:.1f}  "
          f"chosen threshold={thr:.3f}  ACC@thr={val_acc_thr*100:.1f} "
          f"(bal-acc {bal*100:.1f})\n")

    del val_loader; gc.collect()   # free the ~15 GB training cache

    # ── Phase 2: per-generator evaluation (loads per-gen cache) ───────
    print("[calib] Phase 2: per-generator evaluation ...")
    cache_dir = Path(args.clip_cache_dir)
    cache_names = [f"pergen_{g}" for g in ALL_GEN
                   if (cache_dir / f"pergen_{g}_clip.pt").exists()]
    clip_cache = ClipFeatureCache(cache_dir, cache_names) if cache_names else None
    use_cache = clip_cache is not None and clip_cache.available
    collate = clip_collate_fn if use_cache else None

    # gather ensemble-averaged probs per generator
    per_gen_probs = {g: None for g in ALL_GEN}
    per_gen_true = {g: None for g in ALL_GEN}
    for c in ckpts:
        model = build_detector(cfg, force_load_visual=(not use_cache)).to(device)
        model.load_state_dict(torch.load(c, map_location=device, weights_only=False)["model"], strict=False)
        model.eval()
        for g in ALL_GEN:
            gdir = gen_root / g
            if not gdir.exists():
                continue
            try:
                ds = GeneratorDataset(gdir, clip_cache=clip_cache)
            except Exception:
                continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=0, collate_fn=collate)
            yt, yp = infer(model, loader, device)
            per_gen_true[g] = yt
            per_gen_probs[g] = np.array(yp) if per_gen_probs[g] is None else per_gen_probs[g] + np.array(yp)
        del model; gc.collect()

    # ── compute + print ──────────────────────────────────────────────
    rows = {}
    for g in ALL_GEN:
        if per_gen_probs[g] is None:
            continue
        yt = per_gen_true[g]; yp = per_gen_probs[g] / len(ckpts)
        rows[g] = {
            "acc05": acc_at(yt, yp, 0.5) * 100,
            "accthr": acc_at(yt, yp, thr) * 100,
            "accora": oracle_acc(yt, yp) * 100,
            "ap": average_precision_score(yt, yp) * 100,
            "auc": roc_auc_score(yt, yp) * 100,
        }

    def block(title, gens):
        print(f"\n{title}")
        print(f"  {'Generator':<14} {'ACC@0.5':>8} {'ACC@val':>8} {'ACC@orac':>9} {'AP':>7} {'AUC':>7}")
        a5=[]; at=[]; ao=[]; ap_=[]; au=[]
        for g in gens:
            r = rows.get(g)
            if not r: continue
            print(f"  {g:<14} {r['acc05']:>7.1f} {r['accthr']:>8.1f} {r['accora']:>8.1f} {r['ap']:>7.1f} {r['auc']:>7.1f}")
            a5.append(r['acc05']); at.append(r['accthr']); ao.append(r['accora']); ap_.append(r['ap']); au.append(r['auc'])
        if a5:
            print(f"  {'MEAN':<14} {np.mean(a5):>7.1f} {np.mean(at):>8.1f} {np.mean(ao):>8.1f} {np.mean(ap_):>7.1f} {np.mean(au):>7.1f}")
        return a5, at, ap_

    print("\n" + "="*64)
    print(f"  CALIBRATED PER-GENERATOR RESULTS  (val threshold = {thr:.3f})")
    print("="*64)
    block("Table 1 — DFDC (face)", TABLE1_GENERATORS)
    block("Table 2 — GAN (ForenSynths)", TABLE2_GENERATORS)
    block("Table 3 — Diffusion (GenImage)", TABLE3_GENERATORS)

    # save CSV
    with open(out_dir / "calibrated_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generator","acc_0.5","acc_val_thr","acc_oracle","ap","auc","val_threshold"])
        for g in ALL_GEN:
            r = rows.get(g)
            if r:
                w.writerow([g, f"{r['acc05']:.2f}", f"{r['accthr']:.2f}", f"{r['accora']:.2f}",
                            f"{r['ap']:.2f}", f"{r['auc']:.2f}", f"{thr:.3f}"])
    print(f"\n[calib] saved -> {out_dir/'calibrated_results.csv'}")
    print("[calib] Report the ACC@val column (honest). ACC@orac is a diagnostic ceiling only.")


if __name__ == "__main__":
    main()
