"""
metrics.py
──────────
Evaluation metrics used throughout the project.

  • Accuracy  (ACC)
  • Average Precision  (AP / mAP)
  • Recall  @ threshold 0.5
  • F1-Score @ threshold 0.5
  • AUC-ROC

All functions accept plain Python lists or numpy/torch arrays.
"""

from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
)


def compute_all_metrics(
    y_true: list | np.ndarray,
    y_prob: list | np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all evaluation metrics from ground-truth labels and
    predicted probabilities.

    Args:
        y_true    : binary labels (0=Real, 1=AI-generated)
        y_prob    : predicted probability of being AI-generated ∈ [0,1]
        threshold : decision threshold (default 0.5)

    Returns:
        dict with keys: acc, ap, recall, f1, auc,
                        fpr (array), tpr (array), thresholds_roc (array)
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    acc    = accuracy_score(y_true, y_pred)
    ap     = average_precision_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1     = f1_score(y_true, y_pred, zero_division=0)

    # AUC-ROC (guard against single-class edge case)
    try:
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    except ValueError:
        auc, fpr, tpr, thr_roc = float("nan"), np.array([]), np.array([]), np.array([])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "acc":            acc,
        "ap":             ap,
        "recall":         recall,
        "f1":             f1,
        "auc":            auc,
        "fpr":            fpr,
        "tpr":            tpr,
        "thresholds_roc": thr_roc,
        "confusion_matrix": cm,
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    """Pretty-print the scalar metrics to stdout."""
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}"
        f"ACC={metrics['acc']:.4f}  "
        f"AP={metrics['ap']:.4f}  "
        f"Recall={metrics['recall']:.4f}  "
        f"F1={metrics['f1']:.4f}  "
        f"AUC={metrics['auc']:.4f}"
    )


def print_confusion_matrix(metrics: dict) -> None:
    """
    Pretty-print the confusion matrix from a compute_all_metrics() result.

    Example output:
      Confusion Matrix (n=2000):
                                Pred: Real  Pred: AI-Gen
        Actual: Real                   921            79   (TN / FP)
        Actual: AI-Gen                  97           903   (FN / TP)

        True  Positive (TP) =  903   |  correctly caught AI
        True  Negative (TN) =  921   |  correctly passed Real
        False Positive (FP) =   79   |  Real wrongly flagged as AI
        False Negative (FN) =   97   |  AI missed as Real
    """
    cm = metrics.get("confusion_matrix")
    if cm is None:
        return

    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp

    print(f"\n  Confusion Matrix (n={total:,}):")
    print(f"  {'':20s}  {'Pred: Real':>12}  {'Pred: AI-Gen':>12}")
    print(f"  {'Actual: Real':20s}  {tn:>12,d}  {fp:>12,d}   (TN / FP)")
    print(f"  {'Actual: AI-Gen':20s}  {fn:>12,d}  {tp:>12,d}   (FN / TP)")
    print()
    print(f"  True  Positive (TP) = {tp:>5,}   |  correctly caught AI")
    print(f"  True  Negative (TN) = {tn:>5,}   |  correctly passed Real")
    print(f"  False Positive (FP) = {fp:>5,}   |  Real wrongly flagged as AI")
    print(f"  False Negative (FN) = {fn:>5,}   |  AI missed as Real")


def log_metrics_to_tensorboard(writer, metrics: dict, step: int, prefix: str = "val") -> None:
    """Log scalar metrics to a TensorBoard SummaryWriter."""
    for key in ("acc", "ap", "recall", "f1", "auc"):
        writer.add_scalar(f"{prefix}/{key}", metrics[key], step)
