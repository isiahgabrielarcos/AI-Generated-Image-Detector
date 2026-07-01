"""
pergen_split_probe.py
─────────────────────
FAST hypothesis check (minutes): does a SEEN-generator 80/20 protocol give
paper-level per-generator numbers?

For each generator, split its cached CLIP features 80/20 (stratified, fixed
seed) into DISJOINT train/test. Train a linear probe on the pooled 80% and
test per-generator on the disjoint 20%. This previews — instantly — what the
full model would do under Man & Cho's likely "train-on-all-generators,
test-on-held-out-20%" protocol.

This is NOT the unseen cross-generator setup. It is reported as a separate
SEEN-generator (80/20) experiment. Splits are disjoint (no leakage).

Usage:
  python pergen_split_probe.py
"""
import argparse, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from evaluate_per_generator import (TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS)
ALL_GEN = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS


_FAKE = {"1k_fake", "fake", "1_fake", "fake-625", "fake-5000"}
_REAL = {"1k_real", "real", "0_real", "real-625", "real-5000"}


def _label(p):
    for part in reversed(p.replace("\\", "/").lower().split("/")):
        if part in _FAKE: return 1
        if part in _REAL: return 0
    return 0


def pooled(cache_path):
    d = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    paths = d["paths"]; feats = d["features"]
    n = len(paths)
    X = np.empty((n, 1024), dtype=np.float32)
    for i in range(0, n, 1000):
        X[i:i+1000] = feats[i:i+1000].float().mean(dim=1).numpy()
    y = np.array([_label(p) for p in paths])
    return X, y, paths


def strat_split(y, test_frac, seed):
    rng = np.random.default_rng(seed)
    test = np.zeros(len(y), bool)
    for c in (0, 1):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        test[idx[:int(round(len(idx)*test_frac))]] = True
    return ~test, test  # train_mask, test_mask


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--test_frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cache_dir = Path(args.cache_dir)

    Xtr, ytr = [], []
    per = {}
    for g in ALL_GEN:
        cp = cache_dir / f"pergen_{g}_clip.pt"
        if not cp.exists():
            continue
        X, y, _ = pooled(cp)
        tr, te = strat_split(y, args.test_frac, args.seed)
        Xtr.append(X[tr]); ytr.append(y[tr])
        per[g] = (X[te], y[te])
    Xtr = np.concatenate(Xtr); ytr = np.concatenate(ytr)
    print(f"[probe] seen-generator 80/20: train pooled={len(ytr)}  (test held-out per generator)\n")

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(scaler.transform(Xtr), ytr)

    def block(title, gens):
        print(f"\n{title}")
        print(f"  {'Generator':<14} {'ACC':>7} {'AP':>7} {'AUC':>7}")
        accs=[]; aps=[]; aus=[]
        for g in gens:
            if g not in per: continue
            X, y = per[g]
            Xs = scaler.transform(X)
            prob = clf.predict_proba(Xs)[:, 1]
            acc = ((prob >= 0.5).astype(int) == y).mean()*100
            ap = average_precision_score(y, prob)*100 if len(set(y))>1 else float('nan')
            au = roc_auc_score(y, prob)*100 if len(set(y))>1 else float('nan')
            print(f"  {g:<14} {acc:>6.1f} {ap:>7.1f} {au:>7.1f}")
            accs.append(acc); aps.append(ap); aus.append(au)
        if accs:
            print(f"  {'MEAN':<14} {np.mean(accs):>6.1f} {np.mean(aps):>7.1f} {np.mean(aus):>7.1f}")

    print("="*44)
    print("  SEEN-GENERATOR 80/20 — LINEAR PROBE PREVIEW")
    print("="*44)
    block("Table 1 — DFDC", TABLE1_GENERATORS)
    block("Table 2 — GAN", TABLE2_GENERATORS)
    block("Table 3 — Diffusion", TABLE3_GENERATORS)
    print("\n[probe] If these are much higher than your unseen cross-gen numbers,")
    print("[probe] the hypothesis holds -> run the full train_pergen_split.py.")


if __name__ == "__main__":
    main()
