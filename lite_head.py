"""
lite_head.py  —  Lightweight detection head (clean, standalone variant)
───────────────────────────────────────────────────────────────────────
Why this exists
  The full Man & Cho model (CLIP + wavelet + SFDF + Swin, 34.7M trainable
  params) reaches 0.96 in-distribution AUC but collapses to ~chance on
  cross-generator evaluation: the heavy spatial head overfits and discards
  the transferable signal that IS present in the frozen CLIP features
  (a linear probe on the same features scores 70-80% on GANs).

  This script trains a deliberately TINY head — mean-pooled CLIP tokens ->
  standardize -> linear (logistic) classifier — which is the configuration
  that generalizes best. It is the "we found the heavy head hurts; a
  lightweight head restores cross-generator generalization" result.

What it does NOT touch
  Nothing. It only READS the cached features under --cache_dir and writes
  its own results/ + a saved head. Your detector.py / train.py /
  checkpoints / evaluate_per_generator.py are untouched.

Run
  python lite_head.py
  python lite_head.py --head mlp        # small MLP head instead of linear
  python lite_head.py --pool meanstd    # mean++std pooling (2048-d)
"""

import argparse
import sys
import pickle
from pathlib import Path

import numpy as np
import torch

# ── Table groupings (match evaluate_per_generator.py) ─────────────────
T1 = ["DFDC"]
T2 = ["ProGAN", "StyleGAN", "StyleGAN2", "BigGAN",
      "CycleGAN", "StarGAN", "GauGAN", "Deepfake"]
T3 = ["PNDM", "Guided", "DALL-E", "VQ-Diffusion"]
ALL_GEN = T1 + T2 + T3

TRAIN_CACHES = ["dfdc", "forensynths", "genimage"]


# Match the CLASS FOLDER exactly (by path component), never a substring —
# otherwise the generator folder "Deepfake" matches "fake" and mislabels
# every real image as fake.
_FAKE_DIRS = {"fake", "1k_fake", "fake-625", "fake-5000", "1_fake"}
_REAL_DIRS = {"real", "1k_real", "real-625", "real-5000", "0_real"}


def label_of(path: str) -> int:
    parts = path.replace("\\", "/").lower().split("/")
    for part in reversed(parts):          # leaf-most folder wins
        if part in _FAKE_DIRS:
            return 1
        if part in _REAL_DIRS:
            return 0
    return -1


def pooled_from_cache(cache_path: Path, pool: str, chunk: int = 1000):
    """Memory-safe (mmap + chunked) mean / mean+std pooling of a cache file."""
    d = torch.load(cache_path, map_location="cpu", weights_only=True, mmap=True)
    paths = d["paths"]
    feats = d["features"]                       # [N,256,1024] fp16 (mmap)
    n = len(paths)
    dim = 1024 if pool == "mean" else 2048
    out = np.empty((n, dim), dtype=np.float32)
    for i in range(0, n, chunk):
        blk = feats[i:i+chunk].float()          # [b,256,1024]
        if pool == "mean":
            out[i:i+chunk] = blk.mean(dim=1).numpy()
        else:
            mu = blk.mean(dim=1)
            sd = blk.std(dim=1)
            out[i:i+chunk] = torch.cat([mu, sd], dim=1).numpy()
    y = np.array([label_of(p) for p in paths])
    keep = y >= 0
    return out[keep], y[keep]


def metrics(clf, scaler, X, y):
    from sklearn.metrics import average_precision_score, roc_auc_score
    Xs = scaler.transform(X)
    prob = clf.predict_proba(Xs)[:, 1]
    pred = (prob >= 0.5).astype(int)
    acc = float((pred == y).mean())
    ap  = float(average_precision_score(y, prob)) if len(set(y)) > 1 else float("nan")
    auc = float(roc_auc_score(y, prob)) if len(set(y)) > 1 else float("nan")
    return acc, ap, auc


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="datasets_eq/clip_cache")
    ap.add_argument("--head", default="linear", choices=["linear", "mlp"])
    ap.add_argument("--pool", default="mean", choices=["mean", "meanstd"])
    ap.add_argument("--C", type=float, default=1.0,
                    help="inverse reg strength for linear head (lower=stronger reg)")
    ap.add_argument("--n_train", type=int, default=0,
                    help="if >0, subsample this many per class PER dataset "
                         "(e.g. 500 reproduces the earlier 3k-image re-check)")
    ap.add_argument("--train_on", default="all",
                    help="which training caches to use: 'all' or a comma list "
                         "of dfdc,forensynths,genimage (single-source protocol)")
    ap.add_argument("--out", default="results/per_generator_lite")
    args = ap.parse_args()
    rng = np.random.default_rng(42)

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    # ── load + pool training features ────────────────────────────────
    if args.train_on == "all":
        train_list = TRAIN_CACHES
    else:
        train_list = [s.strip() for s in args.train_on.split(",") if s.strip()]
    print(f"[lite] cache_dir={cache_dir}  head={args.head}  pool={args.pool}  "
          f"train_on={train_list}")
    Xs, ys = [], []
    for name in train_list:
        cp = cache_dir / f"{name}_clip.pt"
        if not cp.exists():
            print(f"  [warn] missing {cp}")
            continue
        X, y = pooled_from_cache(cp, args.pool)
        if args.n_train > 0:               # subsample n per class within this dataset
            idx = []
            for lab in (0, 1):
                pool = np.where(y == lab)[0]
                take = rng.choice(pool, size=min(args.n_train, len(pool)),
                                  replace=False)
                idx.append(take)
            idx = np.concatenate(idx)
            X, y = X[idx], y[idx]
        print(f"  loaded {name}: {len(y)} ({int(y.sum())} fake / {int((y==0).sum())} real)")
        Xs.append(X); ys.append(y)
    X_train = np.concatenate(Xs); y_train = np.concatenate(ys)
    print(f"  TRAIN total: {X_train.shape}\n")

    # ── train the lite head ──────────────────────────────────────────
    scaler = StandardScaler().fit(X_train)
    if args.head == "linear":
        clf = LogisticRegression(max_iter=3000, C=args.C)
    else:
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=400,
                            early_stopping=True, alpha=1e-3, random_state=42)
    clf.fit(scaler.transform(X_train), y_train)
    tr_acc = clf.score(scaler.transform(X_train), y_train)
    print(f"[lite] trained. in-distribution TRAIN acc = {tr_acc*100:.1f}%\n")

    # ── per-generator evaluation ─────────────────────────────────────
    rows = {}
    for gen in ALL_GEN:
        cp = cache_dir / f"pergen_{gen}_clip.pt"
        if not cp.exists():
            rows[gen] = None
            continue
        Xg, yg = pooled_from_cache(cp, args.pool)
        rows[gen] = metrics(clf, scaler, Xg, yg)

    def grp(gens):
        accs = [rows[g][0] for g in gens if rows.get(g)]
        aps  = [rows[g][1] for g in gens if rows.get(g)]
        return (np.mean(accs), np.mean(aps)) if accs else (float("nan"), float("nan"))

    # ── print tables ─────────────────────────────────────────────────
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    out("=" * 60)
    out(f"  LIGHTWEIGHT HEAD  ({args.head}, pool={args.pool})  — per generator")
    out(f"  Format: ACC / AP  (×100)")
    out("=" * 60)
    for title, gens in [("Table 1  Face Deepfake (DFDC)", T1),
                        ("Table 2  GAN-based (ForenSynths)", T2),
                        ("Table 3  Diffusion (GenImage)", T3)]:
        out(f"\n  {title}")
        out("  " + "-" * 40)
        for g in gens:
            m = rows.get(g)
            if m:
                out(f"    {g:<14} {m[0]*100:5.1f} / {m[1]*100:5.1f}   (AUC {m[2]*100:4.1f})")
            else:
                out(f"    {g:<14}   (no cache)")
        ga, gp = grp(gens)
        out(f"    {'MEAN':<14} {ga*100:5.1f} / {gp*100:5.1f}")

    out("\n" + "=" * 60)
    out("  SUMMARY (ACC / AP)")
    for tname, gens in [("T1 DFDC", T1), ("T2 GAN", T2), ("T3 Diffusion", T3)]:
        ga, gp = grp(gens)
        out(f"    {tname:<14} {ga*100:5.1f} / {gp*100:5.1f}")
    oa = grp(ALL_GEN)
    out(f"    {'OVERALL':<14} {oa[0]*100:5.1f} / {oa[1]*100:5.1f}")
    out("=" * 60)

    # ── save head + results ──────────────────────────────────────────
    with open(out_dir / "lite_head.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "clf": clf,
                     "head": args.head, "pool": args.pool}, f)
    (out_dir / "tables_lite.txt").write_text("\n".join(lines), encoding="utf-8")

    import csv
    with open(out_dir / "results_lite.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["generator", "table", "acc", "ap", "auc"])
        for g in ALL_GEN:
            m = rows.get(g)
            if m:
                t = 1 if g in T1 else 2 if g in T2 else 3
                w.writerow([g, t, f"{m[0]:.6f}", f"{m[1]:.6f}", f"{m[2]:.6f}"])

    print(f"\n[lite] saved head -> {out_dir/'lite_head.pkl'}")
    print(f"[lite] saved tables -> {out_dir/'tables_lite.txt'}")
    print(f"[lite] saved csv -> {out_dir/'results_lite.csv'}")


if __name__ == "__main__":
    main()
