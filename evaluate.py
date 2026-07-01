"""
evaluate.py
───────────
Evaluate a trained AIGC detector on one or more datasets.

Outputs: Accuracy, Average Precision, Recall, F1-Score, AUC-ROC
         + saves ROC curve, PR curve, and Confusion Matrix as PNG.

Usage:
    # Evaluate on all three datasets defined in config
    python evaluate.py --checkpoint checkpoints/best_model.pt

    # Evaluate on a single custom folder (real/ + fake/ sub-dirs)
    python evaluate.py --checkpoint checkpoints/best_model.pt \
                       --data_dir datasets/MyDataset \
                       --output_dir results/my_eval
"""

import argparse
import csv
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import build_detector
from data.dataset import (
    ImageFolderBinary,
    ClipFeatureCache,
    build_transforms,
    clip_collate_fn,
)
from utils import compute_all_metrics, print_metrics, print_confusion_matrix, setup_hf_auth
from utils.visualization import plot_roc_curve, plot_pr_curve, plot_confusion_matrix


# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",  required=True,         help="Path to .pt checkpoint")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--data_dir",    default=None,          help="Custom dataset root (optional)")
    p.add_argument("--output_dir",  default="results/eval",help="Where to save plots & CSV")
    p.add_argument("--batch_size",  type=int, default=None, help="Override batch size")
    p.add_argument("--device",      default=None)
    p.add_argument("--threshold",   type=float, default=0.5)
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────

def _unpack_batch(batch):
    """Support both (img, label) and (img, clip_tokens, label) batches."""
    if len(batch) == 3:
        images, clip_tokens, labels = batch
        return images, labels, clip_tokens
    images, labels = batch
    return images, labels, None


@torch.no_grad()
def run_inference(model, loader, device) -> tuple:
    model.eval()
    y_true_all, y_prob_all = [], []

    for batch in tqdm(loader, desc="  inference"):
        images, labels, clip_tokens = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if clip_tokens is not None:
            clip_tokens = clip_tokens.to(device, non_blocking=True)

        logits = model(images, clip_tokens=clip_tokens)
        probs  = torch.sigmoid(logits).squeeze(-1)

        y_true_all.extend(labels.cpu().tolist())
        y_prob_all.extend(probs.cpu().tolist())

    return y_true_all, y_prob_all


# ──────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    name: str,
    dataset_root: str,
    model,
    device,
    batch_size: int,
    threshold: float,
    output_dir: Path,
    clip_cache: "ClipFeatureCache | None" = None,
):
    transform  = build_transforms(image_size=224, augment=False)
    ds         = ImageFolderBinary(dataset_root, transform, clip_cache=clip_cache)
    collate    = clip_collate_fn if (clip_cache is not None and clip_cache.available) else None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,          # 0 avoids subprocess overhead on CPU / Windows
        pin_memory=False,
        collate_fn=collate,
    )

    y_true, y_prob = run_inference(model, loader, device)
    metrics = compute_all_metrics(y_true, y_prob, threshold=threshold)

    # ── print ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print(f" Dataset : {name}")
    print(f"{'─'*60}")
    print_metrics(metrics, prefix=name)
    print_confusion_matrix(metrics)

    # ── save plots ────────────────────────────────────────────────────
    ds_dir = output_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)

    plot_roc_curve(
        metrics["fpr"], metrics["tpr"], metrics["auc"],
        save_path=str(ds_dir / "roc_curve.png"),
    )
    plot_pr_curve(
        np.array(y_true), np.array(y_prob), metrics["ap"],
        save_path=str(ds_dir / "pr_curve.png"),
    )
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        class_names=["Real", "AI-Generated"],
        save_path=str(ds_dir / "confusion_matrix.png"),
    )

    # ── save CSV ──────────────────────────────────────────────────────
    with open(ds_dir / "metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key in ("acc", "ap", "recall", "f1", "auc"):
            writer.writerow([key, f"{metrics[key]:.6f}"])

    print(f"  Plots & CSV saved -> {ds_dir}/")
    return metrics


# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg  = load_config(args.config)

    setup_hf_auth()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[evaluate] device = {device}")

    batch_size = args.batch_size or cfg.get("training", {}).get("batch_size", 8)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load CLIP cache (same cache used in training) ─────────────────
    # evaluate.py runs on the same datasets → all images are in cache
    d_cfg     = cfg.get("data", {})
    cache_dir = d_cfg.get("clip_cache_dir", None)
    clip_cache = None
    if cache_dir and Path(cache_dir).exists():
        existing = [
            name for name in ("dfdc", "forensynths", "genimage")
            if (Path(cache_dir) / f"{name}_clip.pt").exists()
        ]
        if existing:
            clip_cache = ClipFeatureCache(cache_dir, ["dfdc", "forensynths", "genimage"])

    # ── Load model ────────────────────────────────────────────────────
    # build_detector auto-detects cache → sets load_visual=False if cache exists
    print(f"[evaluate] loading checkpoint: {args.checkpoint}")
    model = build_detector(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # ── Single custom dataset ─────────────────────────────────────────
    if args.data_dir:
        # Custom folder: not guaranteed to be in cache → use live CLIP
        # Force reload visual if it was skipped
        if model.clip_extractor.visual is None:
            print("[evaluate] WARNING: custom --data_dir images may not be in "
                  "the CLIP cache. Reload model with force_load_visual=True.")
            model = build_detector(cfg, force_load_visual=True).to(device)
            ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            model.eval()
        evaluate_dataset(
            name=Path(args.data_dir).name,
            dataset_root=args.data_dir,
            model=model,
            device=device,
            batch_size=batch_size,
            threshold=args.threshold,
            output_dir=output_dir,
            clip_cache=None,  # custom data not in cache
        )
        return

    # ── All datasets from config ──────────────────────────────────────
    datasets = {
        "DFDC":        d_cfg.get("dfdc_root", ""),
        "ForenSynths": d_cfg.get("forensynths_root", ""),
        "GenImage":    d_cfg.get("genimage_root", ""),
    }

    summary_rows = []
    for name, root in datasets.items():
        if not root or not Path(root).exists():
            print(f"[evaluate] {name}: path not found, skipping.")
            continue

        metrics = evaluate_dataset(
            name=name,
            dataset_root=root,
            model=model,
            device=device,
            batch_size=batch_size,
            threshold=args.threshold,
            output_dir=output_dir,
            clip_cache=clip_cache,
        )
        summary_rows.append({
            "dataset": name,
            **{k: metrics[k] for k in ("acc", "ap", "recall", "f1", "auc")},
        })

    # ── Summary table ─────────────────────────────────────────────────
    if summary_rows:
        summary_path = output_dir / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

        print(f"\n{'='*60}")
        print(" SUMMARY")
        print(f"{'='*60}")
        header = f"{'Dataset':<15}  {'ACC':>6}  {'AP':>6}  {'Recall':>6}  {'F1':>6}  {'AUC':>6}"
        print(header)
        print("-" * len(header))
        for r in summary_rows:
            print(
                f"{r['dataset']:<15}  "
                f"{r['acc']:>6.4f}  {r['ap']:>6.4f}  "
                f"{r['recall']:>6.4f}  {r['f1']:>6.4f}  {r['auc']:>6.4f}"
            )
        print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
