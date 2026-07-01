"""
diagnose_clip_features.py
─────────────────────────
Decisive diagnostic: is the per-gen failure caused by the heavy trainable
head overfitting, or are the frozen CLIP features themselves uninformative
across generators?

Method:
  1. Load the TRAINING CLIP caches (dfdc/forensynths/genimage).
  2. Mean-pool each image's 256 tokens -> a single 1024-d vector.
  3. Train a plain logistic regression (almost no capacity -> can't memorize
     deep shortcuts) on those features.
  4. Evaluate that linear probe on every per-gen cache.

Interpretation:
  - Linear probe gets >75% cross-gen  -> CLIP features ARE general; your
    34.7M-param Swin head is what kills generalization. Fix = shrink the head.
  - Linear probe also ~50%            -> the problem is in the DATA: training
    reals vs fakes differ by a trivial cue absent in per-gen. Fix = data.
"""

import sys
from pathlib import Path

import numpy as np
import torch

CACHE_DIR = Path("datasets/clip_cache")
TRAIN_CACHES = ["dfdc", "forensynths", "genimage"]
PERGEN = ["DFDC", "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN", "CycleGAN",
          "StarGAN", "GauGAN", "Deepfake", "PNDM", "Guided", "DALL-E",
          "VQ-Diffusion"]


def label_of(path: str) -> int:
    p = path.lower()
    if "fake" in p:
        return 1
    if "real" in p:
        return 0
    return -1  # unknown


def load_pooled(cache_path: Path, mode: str = "mean"):
    """
    Return (X[N,D] float32, y[N] int) with tokens pooled.
      mode='mean'     -> [N,1024]       mean over tokens
      mode='meanstd'  -> [N,2048]       mean ++ std over tokens
    """
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    paths = data["paths"]
    feats = data["features"].float()        # [N,256,1024] float16 -> float32
    if mode == "mean":
        X = feats.mean(dim=1).numpy()
    elif mode == "meanstd":
        mu = feats.mean(dim=1)
        sd = feats.std(dim=1)
        X = torch.cat([mu, sd], dim=1).numpy()   # [N,2048]
    else:
        raise ValueError(mode)
    y = np.array([label_of(p) for p in paths])
    keep = y >= 0
    return X[keep], y[keep]


def run_probe(clf_factory, pool_mode: str, label: str):
    from sklearn.preprocessing import StandardScaler

    # Build training matrix
    Xs, ys = [], []
    for name in TRAIN_CACHES:
        cp = CACHE_DIR / f"{name}_clip.pt"
        if not cp.exists():
            continue
        X, y = load_pooled(cp, mode=pool_mode)
        Xs.append(X); ys.append(y)
    X_train = np.concatenate(Xs); y_train = np.concatenate(ys)

    scaler = StandardScaler().fit(X_train)
    clf = clf_factory()
    clf.fit(scaler.transform(X_train), y_train)
    train_acc = clf.score(scaler.transform(X_train), y_train)

    print(f"\n{'='*46}")
    print(f"  {label}")
    print(f"  pooling={pool_mode}   TRAIN acc={train_acc*100:.1f}%")
    print(f"{'='*46}")
    print(f"  {'Generator':<15} {'ACC':>7}  {'n':>6}")
    print("  " + "-" * 32)
    accs = []
    for gen in PERGEN:
        cp = CACHE_DIR / f"pergen_{gen}_clip.pt"
        if not cp.exists():
            print(f"  {gen:<15} {'(no cache)':>7}")
            continue
        Xg, yg = load_pooled(cp, mode=pool_mode)
        acc = clf.score(scaler.transform(Xg), yg)
        accs.append(acc)
        print(f"  {gen:<15} {acc*100:>6.1f}%  {len(yg):>6}")
    print("  " + "-" * 32)
    mean_acc = float(np.mean(accs)) if accs else 0.0
    print(f"  {'MEAN':<15} {mean_acc*100:>6.1f}%")
    return mean_acc


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    summary = {}

    # 1. Linear probe, mean pooling (baseline – already known ~58.5%)
    summary["Linear / mean"] = run_probe(
        lambda: LogisticRegression(max_iter=2000, C=1.0),
        "mean", "Logistic regression (linear)")

    # 2. Linear probe, mean+std pooling (richer features)
    summary["Linear / mean+std"] = run_probe(
        lambda: LogisticRegression(max_iter=2000, C=1.0),
        "meanstd", "Logistic regression (linear)")

    # 3. Small MLP, mean+std pooling (a little non-linear capacity)
    summary["MLP / mean+std"] = run_probe(
        lambda: MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                              early_stopping=True, alpha=1e-3,
                              random_state=42),
        "meanstd", "Small MLP (1x256, weight decay)")

    # ── Overall comparison ───────────────────────────────────────────
    print(f"\n{'#'*46}")
    print("  FEATURE-CEILING SUMMARY  (cross-gen MEAN acc)")
    print(f"{'#'*46}")
    for k, v in summary.items():
        print(f"  {k:<22} {v*100:>6.1f}%")
    print(f"\n  Your full 34.7M model (for reference):   ~50.0%")


if __name__ == "__main__":
    main()
