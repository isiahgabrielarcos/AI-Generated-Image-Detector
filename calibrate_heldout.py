"""
calibrate_heldout.py
────────────────────
Transparent, honest threshold calibration via a HELD-OUT split.

For each generator the 2,000 samples are split (stratified by label, fixed
seed) into:
    calibration  (default 30%)  – used ONLY to choose the decision threshold
    evaluation   (default 70%)  – the scored set; the threshold never saw it

Because the threshold is a single scalar fit on data that is not scored, the
reported evaluation ACC is honest (proper calib/eval separation, standard
experimental design). The model never trains on any of it.

Three ACC columns are reported on the EVALUATION split for full transparency:
    ACC@0.5     – the blind 0.5 cutoff
    ACC@global  – ONE threshold fit on the pooled calibration set, applied to
                  every generator (a single deployable operating point)
    ACC@domain  – a per-generator threshold fit on that generator's own
                  calibration split (the standard cross-domain-calibrated ACC)

AP and AUC are threshold-independent and reported on the evaluation split too.
Optional checkpoint ensembling: pass several --checkpoints to average probs.

Usage:
  python calibrate_heldout.py --checkpoints checkpoints/epoch_020.pt
  python calibrate_heldout.py --checkpoints checkpoints/epoch_010.pt checkpoints/epoch_015.pt checkpoints/epoch_020.pt
"""

import argparse, csv, gc, sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score, balanced_accuracy_score

from models import build_detector
from data.dataset import ClipFeatureCache, clip_collate_fn
from utils import setup_hf_auth
from evaluate_per_generator import (GeneratorDataset, infer,
                                    TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS)

ALL_GEN = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS


def strat_split(y, frac, seed):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    calib = np.zeros(len(y), bool)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        calib[idx[:int(round(len(idx) * frac))]] = True
    return calib, ~calib


def best_thr(y_true, y_prob):
    grid = np.linspace(0.02, 0.98, 193)
    bt, bb = 0.5, -1.0
    for t in grid:
        b = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if b > bb: bb, bt = b, t
    return float(bt)


def acc(y_true, y_prob, t):
    return float(((y_prob >= t).astype(int) == np.asarray(y_true)).mean())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", default=["checkpoints/epoch_020.pt"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--generators_root", default="per-gen-dataset")
    ap.add_argument("--clip_cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--calib_frac", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--out", default="results/calibrated")
    args = ap.parse_args()

    setup_hf_auth()
    device = torch.device("cpu")
    cfg = yaml.safe_load(open(args.config))
    gen_root = Path(args.generators_root)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = args.checkpoints
    print(f"[heldout] checkpoints={ckpts}  calib_frac={args.calib_frac}  seed={args.seed}\n")

    cache_dir = Path(args.clip_cache_dir)
    names = [f"pergen_{g}" for g in ALL_GEN if (cache_dir / f"pergen_{g}_clip.pt").exists()]
    clip_cache = ClipFeatureCache(cache_dir, names) if names else None
    use_cache = clip_cache is not None and clip_cache.available
    collate = clip_collate_fn if use_cache else None

    # ensemble-averaged probabilities per generator
    probs = {g: None for g in ALL_GEN}
    trues = {g: None for g in ALL_GEN}
    for c in ckpts:
        model = build_detector(cfg, force_load_visual=(not use_cache)).to(device)
        model.load_state_dict(torch.load(c, map_location=device, weights_only=False)["model"], strict=False)
        model.eval()
        for g in ALL_GEN:
            gdir = gen_root / g
            if not gdir.exists(): continue
            try: ds = GeneratorDataset(gdir, clip_cache=clip_cache)
            except Exception: continue
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate)
            yt, yp = infer(model, loader, device)
            trues[g] = np.asarray(yt)
            probs[g] = np.asarray(yp) if probs[g] is None else probs[g] + np.asarray(yp)
        del model; gc.collect()
    for g in ALL_GEN:
        if probs[g] is not None: probs[g] /= len(ckpts)

    # held-out split + global threshold from pooled calibration
    calib_masks = {}
    pooled_t, pooled_p = [], []
    for g in ALL_GEN:
        if probs[g] is None: continue
        cmask, _ = strat_split(trues[g], args.calib_frac, args.seed)
        calib_masks[g] = cmask
        pooled_t.append(trues[g][cmask]); pooled_p.append(probs[g][cmask])
    global_thr = best_thr(np.concatenate(pooled_t), np.concatenate(pooled_p))
    print(f"[heldout] global threshold (pooled calibration) = {global_thr:.3f}\n")

    rows = {}
    for g in ALL_GEN:
        if probs[g] is None: continue
        cmask = calib_masks[g]; emask = ~cmask
        yt_c, yp_c = trues[g][cmask], probs[g][cmask]
        yt_e, yp_e = trues[g][emask], probs[g][emask]
        dthr = best_thr(yt_c, yp_c)
        rows[g] = {
            "acc05":  acc(yt_e, yp_e, 0.5) * 100,
            "accg":   acc(yt_e, yp_e, global_thr) * 100,
            "accd":   acc(yt_e, yp_e, dthr) * 100,
            "ap":     average_precision_score(yt_e, yp_e) * 100,
            "auc":    roc_auc_score(yt_e, yp_e) * 100,
            "dthr":   dthr,
        }

    def block(title, gens):
        print(f"\n{title}")
        print(f"  {'Generator':<14} {'ACC@0.5':>8} {'ACC@glob':>9} {'ACC@dom':>8} {'AP':>7} {'AUC':>7}")
        agg = {k: [] for k in ('acc05','accg','accd','ap','auc')}
        for g in gens:
            r = rows.get(g)
            if not r: continue
            print(f"  {g:<14} {r['acc05']:>7.1f} {r['accg']:>8.1f} {r['accd']:>8.1f} {r['ap']:>7.1f} {r['auc']:>7.1f}")
            for k in agg: agg[k].append(r[k])
        if agg['acc05']:
            print(f"  {'MEAN':<14} {np.mean(agg['acc05']):>7.1f} {np.mean(agg['accg']):>8.1f} "
                  f"{np.mean(agg['accd']):>8.1f} {np.mean(agg['ap']):>7.1f} {np.mean(agg['auc']):>7.1f}")

    print("="*66)
    print(f"  HELD-OUT CALIBRATED RESULTS  (eval split = {int((1-args.calib_frac)*100)}% of each generator)")
    print("="*66)
    block("Table 1 — DFDC (face)", TABLE1_GENERATORS)
    block("Table 2 — GAN (ForenSynths)", TABLE2_GENERATORS)
    block("Table 3 — Diffusion (GenImage)", TABLE3_GENERATORS)

    with open(out_dir / "heldout_calibrated.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["generator","acc_0.5","acc_global","acc_domain","ap","auc","domain_threshold","global_threshold"])
        for g in ALL_GEN:
            r = rows.get(g)
            if r: w.writerow([g, f"{r['acc05']:.2f}", f"{r['accg']:.2f}", f"{r['accd']:.2f}",
                              f"{r['ap']:.2f}", f"{r['auc']:.2f}", f"{r['dthr']:.3f}", f"{global_thr:.3f}"])
    print(f"\n[heldout] saved -> {out_dir/'heldout_calibrated.csv'}")
    print("[heldout] Disclose the protocol in the paper: ACC reported on a held-out 70% eval")
    print("[heldout] split, threshold fit on the disjoint 30% calibration split. AP/AUC are")
    print("[heldout] threshold-free. ACC@dom = per-generator calibration; ACC@glob = single op-point.")


if __name__ == "__main__":
    main()
