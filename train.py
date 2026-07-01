"""
train.py
────────
Training entry-point for the Man & Cho (2026) AIGC detector.

Usage:
    # First time: pre-compute CLIP cache (run once, takes ~2-6 h on CPU)
    python cache_clip_features.py --config configs/default.yaml

    # Then train
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/epoch_10.pt

CPU optimisations applied here (architecture unchanged):
    1. CLIP feature cache   – ViT bypassed each batch (biggest speedup)
    2. Gradient accumulation– effective batch=32 with per-step batch=8
    3. bfloat16 autocast    – DISABLED by default (use_amp: false).
                               On AMD CPUs without AVX-512 BF16 hardware, Intel
                               oneDNN BF16 ops are 5-10x SLOWER than plain FP32.
                               Enable only on Intel CPUs with VNNI/BF16 support.
    4. Lightweight wavelet  – CNN mid-channels fixed to 64 (9→64→256→768)
                               vs the wider 9→192→768→768; ~4.5x less FLOPs.
    5. Fused attention      – SFDF uses F.scaled_dot_product_attention (PyTorch 2+)
    6. torch.compile        – optional; set use_compile: true in config
    7. Early stopping       – quits when val AUC plateaus (saves epochs)
    8. CPU thread tuning    – sets torch thread count to available cores
    9. num_workers=0        – avoids Windows subprocess overhead

AMD CPU tip: set MKL_DEBUG_CPU_TYPE=5 in your shell BEFORE running Python
(MKL initialises at torch import time; the .env value arrives too late).
Use run_train.ps1 to apply this automatically.
"""

import os
import argparse
import time
import yaml
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import build_detector
from losses import BinaryFocalLoss
from data import build_dataloaders
from utils import compute_all_metrics, print_metrics, print_confusion_matrix, log_metrics_to_tensorboard, setup_hf_auth


# ──────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    p.add_argument("--device", default=None, help="cuda / cpu (auto-detected if omitted)")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _unpack_batch(batch):
    """
    Support both 2-tuple (img, label) and 3-tuple (img, clip_tokens, label)
    batches transparently.
    """
    if len(batch) == 3:
        images, clip_tokens, labels = batch
        return images, labels, clip_tokens
    images, labels = batch
    return images, labels, None


# ──────────────────────────────────────────────────────────────────────
#  Optimiser + scheduler
# ──────────────────────────────────────────────────────────────────────

def build_optimizer(model: nn.Module, cfg: dict):
    """
    Two param groups with different learning rates (paper §4.2):
      - Lightweight CNN branch    : lr = 1e-4
      - Swin backbone + CLIP proj : lr = 1e-5
    CLIP visual encoder is frozen (or not loaded), so it has no grad params.
    """
    t_cfg = cfg.get("training", {})
    cnn_lr      = t_cfg.get("cnn_lr", 1e-4)
    backbone_lr = t_cfg.get("backbone_lr", 1e-5)
    wd          = t_cfg.get("weight_decay", 1e-4)

    cnn_params    = list(model.wavelet_extractor.cnn.parameters())
    cnn_param_ids = {id(p) for p in cnn_params}
    other_params  = [p for p in model.parameters()
                     if id(p) not in cnn_param_ids and p.requires_grad]

    return AdamW(
        [
            {"params": cnn_params,   "lr": cnn_lr},
            {"params": other_params, "lr": backbone_lr},
        ],
        weight_decay=wd,
    )


def build_scheduler(optimizer, cfg: dict, steps_per_epoch: int):
    t_cfg         = cfg.get("training", {})
    epochs        = t_cfg.get("epochs", 50)
    warmup_epochs = t_cfg.get("warmup_epochs", 5)
    accum_steps   = t_cfg.get("accumulation_steps", 1)

    # "steps" here = optimizer steps (after accumulation), not loader iterations
    opt_steps_per_epoch = max(1, steps_per_epoch // accum_steps)

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs * opt_steps_per_epoch,
    )
    # eta_min must stay BELOW every group's base LR, otherwise CosineAnnealingLR
    # runs "backwards" and ramps the LR UP toward eta_min instead of decaying.
    # With normal LRs (1e-4/1e-5) the 1e-7 floor is fine; for tiny experimental
    # LRs (e.g. 1e-15) it would silently turn the run into a ~1e-7 run.
    min_base_lr = min(g["lr"] for g in optimizer.param_groups)
    eta_min = min(1e-7, min_base_lr * 1e-2)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=(epochs - warmup_epochs) * opt_steps_per_epoch,
        eta_min=eta_min,
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs * opt_steps_per_epoch],
    )


# ──────────────────────────────────────────────────────────────────────
#  Early stopping
# ──────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when val AUC has not improved for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.wait      = 0

    def step(self, metric: float) -> bool:
        """Return True if training should stop."""
        if metric > self.best + self.min_delta:
            self.best = metric
            self.wait = 0
            return False
        self.wait += 1
        if self.wait >= self.patience:
            print(f"  [early stop] no AUC improvement for {self.patience} epochs "
                  f"(best={self.best:.4f})")
            return True
        return False


# ──────────────────────────────────────────────────────────────────────
#  One epoch of training
# ──────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model, loader, optimizer, scheduler, criterion,
    device, grad_clip, accum_steps, autocast_ctx
) -> float:
    model.train()
    total_loss  = 0.0
    opt_step    = 0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
        images, labels, clip_tokens = _unpack_batch(batch)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if clip_tokens is not None:
            clip_tokens = clip_tokens.to(device, non_blocking=True)

        with autocast_ctx():
            logits = model(images, clip_tokens=clip_tokens)   # [B, 1]
            loss   = criterion(logits, labels)
            loss   = loss / accum_steps                        # normalise

        loss.backward()
        total_loss += loss.item() * accum_steps

        # Optimizer step after every `accum_steps` mini-batches
        is_last = (i + 1 == len(loader))
        if (i + 1) % accum_steps == 0 or is_last:
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            opt_step += 1

    return total_loss / len(loader)


# ──────────────────────────────────────────────────────────────────────
#  Evaluation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, autocast_ctx) -> tuple:
    """Returns (y_true, y_prob) lists."""
    model.eval()
    y_true_all, y_prob_all = [], []

    for batch in tqdm(loader, desc="  eval ", leave=False):
        images, labels, clip_tokens = _unpack_batch(batch)
        images = images.to(device, non_blocking=True)
        if clip_tokens is not None:
            clip_tokens = clip_tokens.to(device, non_blocking=True)

        with autocast_ctx():
            logits = model(images, clip_tokens=clip_tokens)
        probs = torch.sigmoid(logits).squeeze(-1)

        y_true_all.extend(labels.cpu().tolist())
        y_prob_all.extend(probs.cpu().tolist())

    return y_true_all, y_prob_all


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    cfg   = load_config(args.config)
    t_cfg = cfg.get("training", {})

    # ── HF authentication (faster model downloads) ───────────────────
    setup_hf_auth()

    # ── Device ───────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] device = {device}")

    # ── CPU thread tuning ────────────────────────────────────────────
    # Use all physical cores for matrix ops; limit inter-op to 2 threads
    # to avoid contention between PyTorch's internal parallelism.
    if device.type == "cpu":
        # Honour AIGC_MAX_THREADS (set by run_capped.ps1) so the run can be
        # throttled to a fraction of cores and leave the PC responsive.
        default_threads = min(os.cpu_count() or 4, 16)
        n_threads = int(os.environ.get("AIGC_MAX_THREADS", default_threads))
        n_threads = max(1, min(n_threads, default_threads))
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(2)
        print(f"[train] CPU threads: intra={n_threads}, inter=2")

    # ── AMP context ──────────────────────────────────────────────────
    # Default False: on AMD CPUs without AVX-512 BF16, oneDNN's BF16 path
    # is significantly slower than FP32.  Enable only on Intel Alder Lake+.
    use_amp   = t_cfg.get("use_amp", False)
    amp_dtype = torch.bfloat16   # bfloat16; no GradScaler needed on CPU

    @contextmanager
    def autocast_ctx():
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=amp_dtype):
                yield
        else:
            yield

    # ── Dirs ─────────────────────────────────────────────────────────
    log_cfg  = cfg.get("logging", {})
    log_dir  = Path(log_cfg.get("log_dir",  "runs/"))
    save_dir = Path(log_cfg.get("save_dir", "checkpoints/"))
    log_dir.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    # ── Per-epoch CSV log (easy morning review of long runs) ─────────
    metrics_csv = save_dir / "epoch_metrics.csv"
    if not metrics_csv.exists():
        with open(metrics_csv, "w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_acc,val_ap,val_auc,val_f1,epoch_sec\n")
    print(f"[train] per-epoch metrics -> {metrics_csv}")

    # ── Data ─────────────────────────────────────────────────────────
    print("[train] building datasets ...")
    train_loader, val_loader = build_dataloaders(cfg)

    # ── Model ────────────────────────────────────────────────────────
    print("[train] building model ...")
    model = build_detector(cfg).to(device)

    # ── Optional torch.compile ───────────────────────────────────────
    # Adds ~5-10 min of compilation overhead once, then gives ~20-50%
    # faster iterations by fusing ops and using optimised kernels.
    if t_cfg.get("use_compile", False):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[train] torch.compile applied – first batch will take several minutes")
        except Exception as e:
            print(f"[train] torch.compile unavailable, running uncompiled: {e}")

    # Count trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_params / 1e6:.1f} M")

    # ── Loss / optimiser / scheduler ─────────────────────────────────
    criterion   = BinaryFocalLoss(gamma=t_cfg.get("focal_gamma", 2.0))
    optimizer   = build_optimizer(model, cfg)
    scheduler   = build_scheduler(optimizer, cfg, len(train_loader))
    grad_clip   = t_cfg.get("gradient_clip", 1.0)
    epochs      = t_cfg.get("epochs", 50)
    accum_steps = t_cfg.get("accumulation_steps", 1)
    patience    = t_cfg.get("early_stop_patience", 10)
    save_every  = log_cfg.get("save_every", 5)
    eval_every  = log_cfg.get("eval_every", 1)

    effective_bs = cfg.get("training", {}).get("batch_size", 8) * accum_steps
    print(f"[train] batch_size={cfg['training']['batch_size']}  "
          f"accum_steps={accum_steps}  effective_batch={effective_bs}  "
          f"epochs={epochs}  patience={patience}  amp={use_amp}")

    start_epoch  = 0
    best_auc     = 0.0
    early_stop   = EarlyStopping(patience=patience)

    # ── Resume ───────────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_auc    = ckpt.get("best_auc", 0.0)
        print(f"[train] resumed from epoch {start_epoch}, best_auc={best_auc:.4f}")

    # ── Training loop ────────────────────────────────────────────────
    epoch_times = []
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, grad_clip, accum_steps, autocast_ctx
        )
        writer.add_scalar("train/loss", train_loss, epoch)

        elapsed = time.time() - t0
        epoch_times.append(elapsed)

        # ETA estimate (average of last 3 epochs)
        avg_t = sum(epoch_times[-3:]) / len(epoch_times[-3:])
        remaining = avg_t * (epochs - epoch - 1)
        eta_h = int(remaining // 3600)
        eta_m = int((remaining % 3600) // 60)

        print(f"Epoch [{epoch+1:3d}/{epochs}]  loss={train_loss:.4f}  "
              f"({elapsed:.0f}s)  ETA {eta_h}h {eta_m}m")

        # ── Validation ───────────────────────────────────────────────
        if (epoch + 1) % eval_every == 0:
            y_true, y_prob = evaluate(model, val_loader, device, autocast_ctx)
            metrics = compute_all_metrics(y_true, y_prob)
            print_metrics(metrics, prefix=f"val ep{epoch+1}")
            print_confusion_matrix(metrics)
            log_metrics_to_tensorboard(writer, metrics, epoch, prefix="val")

            # Append this epoch's numbers to the CSV
            with open(metrics_csv, "a", encoding="utf-8") as f:
                f.write(f"{epoch+1},{train_loss:.6f},{metrics['acc']:.6f},"
                        f"{metrics['ap']:.6f},{metrics['auc']:.6f},"
                        f"{metrics['f1']:.6f},{elapsed:.1f}\n")

            # Save best model
            if metrics["auc"] > best_auc:
                best_auc = metrics["auc"]
                torch.save(
                    {
                        "epoch":     epoch,
                        "model":     model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_auc":  best_auc,
                        "metrics":   metrics,
                        "cfg":       cfg,
                    },
                    save_dir / "best_model.pt",
                )
                print(f"  * New best AUC={best_auc:.4f} -> saved best_model.pt")

            # Early stopping check
            if early_stop.step(metrics["auc"]):
                print(f"[train] Early stopping at epoch {epoch+1}.")
                break

        # ── Periodic checkpoint ───────────────────────────────────────
        if (epoch + 1) % save_every == 0:
            torch.save(
                {
                    "epoch":     epoch,
                    "model":     model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_auc":  best_auc,
                },
                save_dir / f"epoch_{epoch+1:03d}.pt",
            )

    writer.close()
    total_h = sum(epoch_times) / 3600
    print(f"\n[train] Done. Best AUC = {best_auc:.4f}  "
          f"(total training time: {total_h:.1f} h)")


if __name__ == "__main__":
    main()
