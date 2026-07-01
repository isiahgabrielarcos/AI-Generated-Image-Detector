"""
finetune.py
───────────
Fine-tune a trained checkpoint to improve per-generator performance
without catastrophic forgetting.

Strategy
────────
All generators are included in every epoch (replay keeps existing generators
from degrading).  Target generators are *oversampled* by --target_weight so
the model sees more of what it currently struggles with.

Typical settings for safe improvement:
  backbone_lr  1e-6   (10× lower than original 1e-5)
  cnn_lr       1e-5   (10× lower than original 1e-4)
  epochs       20     (early stopping will cut this short if needed)
  patience     7

Usage
─────
    # Fine-tune cross-gen model on ALL generators (general improvement)
    python finetune.py --checkpoint checkpoints/best_model.pt

    # Focus on specific underperformers (upweight 3x)
    python finetune.py ^
        --checkpoint checkpoints/best_model.pt ^
        --target_generators DFDC Deepfake StyleGAN CycleGAN DALL-E ^
        --target_weight 3

    # Fine-tune seen-gen model
    python finetune.py ^
        --checkpoint checkpoints/pergen_split_best_model.pt ^
        --save checkpoints/finetuned_pergen_model.pt ^
        --target_generators DFDC PNDM DALL-E ^
        --target_weight 3
"""

import argparse
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from models import build_detector
from losses import BinaryFocalLoss
from data.dataset import ClipFeatureCache, clip_collate_fn, build_transforms, _collect_images
from utils import compute_all_metrics, print_metrics, print_confusion_matrix, setup_hf_auth
from train import train_one_epoch, EarlyStopping, evaluate
from evaluate_per_generator import TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS

ALL_GEN   = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS
FAKE_NAMES = ("1k_fake", "fake", "fake-625", "fake-5000", "1_fake")
REAL_NAMES = ("1k_real", "real", "real-625", "real-5000", "0_real")


@contextmanager
def noctx():
    yield


# ──────────────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────────────

class ListDataset(Dataset):
    def __init__(self, samples, transform, clip_cache):
        self.samples    = samples
        self.transform  = transform
        self.clip_cache = clip_cache

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        x   = self.transform(img)
        lab = torch.tensor(label, dtype=torch.float32)
        if self.clip_cache is not None:
            tok = self.clip_cache.get(path)
            if tok is None:
                tok = torch.zeros(256, 1024)
            return x, tok, lab
        return x, lab


def _find_dir(root, names):
    for n in names:
        if (root / n).is_dir():
            return root / n
    return None


def collect_generator(gen_dir: Path):
    """Return [(abs_path_str, label)] for one generator folder."""
    fdir = _find_dir(gen_dir, FAKE_NAMES)
    rdir = _find_dir(gen_dir, REAL_NAMES)
    fakes = _collect_images(fdir) if fdir else []
    reals = _collect_images(rdir) if rdir else []
    return [(str(p), 1) for p in fakes] + \
           [(str(p), 0) for p in reals]


# ──────────────────────────────────────────────────────────────────────
#  Optimizer — smaller LRs for fine-tuning
# ──────────────────────────────────────────────────────────────────────

def build_finetune_optimizer(model, backbone_lr: float, cnn_lr: float,
                              weight_decay: float = 1e-4):
    cnn_params    = list(model.wavelet_extractor.cnn.parameters())
    cnn_param_ids = {id(p) for p in cnn_params}
    other_params  = [p for p in model.parameters()
                     if id(p) not in cnn_param_ids and p.requires_grad]
    from torch.optim import AdamW
    return AdamW(
        [
            {"params": cnn_params,   "lr": cnn_lr},
            {"params": other_params, "lr": backbone_lr},
        ],
        weight_decay=weight_decay,
    )


# ──────────────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",        required=True,
                   help="Checkpoint to start from (best_model.pt or pergen_split_best_model.pt)")
    p.add_argument("--save",              default=None,
                   help="Output path (default: checkpoints/finetuned_<original>.pt)")
    p.add_argument("--config",            default="configs/default.yaml")
    p.add_argument("--generators_root",   default="per-gen-dataset")
    p.add_argument("--clip_cache_dir",    default="datasets_eq/clip_cache")
    p.add_argument("--include_generators", nargs="*", default=None,
                   help="Whitelist: only use these generators (default: all). "
                        "e.g. --include_generators DFDC")
    p.add_argument("--target_generators", nargs="*", default=[],
                   help="Generators to oversample (e.g. DFDC Deepfake StyleGAN)")
    p.add_argument("--target_weight",     type=float, default=3.0,
                   help="How many times more often to see each target generator sample")
    p.add_argument("--backbone_lr",       type=float, default=1e-6,
                   help="LR for Swin backbone + CLIP proj (default 1e-6, 10x lower than train)")
    p.add_argument("--cnn_lr",            type=float, default=1e-5,
                   help="LR for wavelet CNN (default 1e-5, 10x lower than train)")
    p.add_argument("--epochs",            type=int,   default=20)
    p.add_argument("--patience",          type=int,   default=7)
    p.add_argument("--batch_size",        type=int,   default=8)
    p.add_argument("--accum_steps",       type=int,   default=4)
    p.add_argument("--val_frac",          type=float, default=0.10,
                   help="Fraction of each generator held out for validation")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--save_every",        type=int,   default=5,
                   help="Save a resumable epoch checkpoint every N epochs")
    p.add_argument("--resume",            default=None,
                   help="Resume an interrupted fine-tune from a finetune_epoch_*.pt checkpoint")
    return p.parse_args()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args   = parse_args()
    device = torch.device("cpu")
    torch.set_num_threads(min(torch.get_num_threads(), 8))

    cfg = yaml.safe_load(open(args.config))

    # ── output path ───────────────────────────────────────────────────
    if args.save is None:
        stem = Path(args.checkpoint).stem
        args.save = f"checkpoints/finetuned_{stem}.pt"
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    setup_hf_auth()

    # ── CLIP cache ────────────────────────────────────────────────────
    cache_dir = Path(args.clip_cache_dir)
    names     = [f"pergen_{g}" for g in ALL_GEN
                 if (cache_dir / f"pergen_{g}_clip.pt").exists()]
    clip_cache = ClipFeatureCache(cache_dir, names) if names else None
    use_cache  = clip_cache is not None and clip_cache.available
    collate    = clip_collate_fn if use_cache else None

    # ── collect per-generator samples ─────────────────────────────────
    rng      = np.random.default_rng(args.seed)
    gen_root = Path(args.generators_root)
    target_set  = set(args.target_generators)
    include_set = set(args.include_generators) if args.include_generators else None

    all_train, all_val = [], []
    sample_weights     = []   # parallel to all_train
    found_gens         = []

    for g in ALL_GEN:
        if include_set is not None and g not in include_set:
            continue
        gdir = gen_root / g
        if not gdir.exists():
            continue
        samples = collect_generator(gdir)
        if not samples:
            continue
        found_gens.append(g)

        # Light val split (stratified per generator, same seed for reproducibility)
        rng_g = np.random.default_rng(args.seed + hash(g) % (2**31))
        by_label = {0: [], 1: []}
        for s in samples:
            by_label[s[1]].append(s)
        gen_train, gen_val = [], []
        for lbl, lst in by_label.items():
            rng_g.shuffle(lst)
            n_val = max(1, int(round(len(lst) * args.val_frac)))
            gen_val   += lst[:n_val]
            gen_train += lst[n_val:]

        w = args.target_weight if g in target_set else 1.0
        all_train += gen_train
        all_val   += gen_val
        sample_weights += [w] * len(gen_train)

    print(f"[finetune] generators found : {found_gens}")
    print(f"[finetune] target generators: {list(target_set) or '(all equal weight)'}")
    print(f"[finetune] train={len(all_train)}  val={len(all_val)}")

    # ── dataloaders ───────────────────────────────────────────────────
    tf_train = build_transforms(image_size=224, augment=True)
    tf_eval  = build_transforms(image_size=224, augment=False)

    train_ds = ListDataset(all_train, tf_train, clip_cache)
    val_ds   = ListDataset(all_val,   tf_eval,  clip_cache)

    # WeightedRandomSampler ensures target generators appear more often
    sampler = WeightedRandomSampler(
        weights     = torch.tensor(sample_weights, dtype=torch.float64),
        num_samples = len(all_train),
        replacement = True,
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=0, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=16,
                              shuffle=False, num_workers=0, collate_fn=collate)

    # ── model ─────────────────────────────────────────────────────────
    model = build_detector(cfg, force_load_visual=(not use_cache)).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"[finetune] loaded weights from {args.checkpoint}")
    print(f"[finetune] trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f} M\n")

    # ── optimizer (reduced LR) ────────────────────────────────────────
    optimizer = build_finetune_optimizer(
        model,
        backbone_lr  = args.backbone_lr,
        cnn_lr       = args.cnn_lr,
        weight_decay = cfg.get("training", {}).get("weight_decay", 1e-4),
    )

    # Simple cosine schedule over fine-tune epochs (no warm-up needed)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max   = args.epochs * steps_per_epoch,
        eta_min = 1e-8,
    )

    criterion  = BinaryFocalLoss(
        gamma=cfg.get("training", {}).get("focal_gamma", 2.0)
    )
    stopper    = EarlyStopping(patience=args.patience)
    best_auc   = ckpt.get("best_auc", 0.0)
    grad_clip  = cfg.get("training", {}).get("gradient_clip", 1.0)
    start_epoch = 0

    # ── resume from an interrupted fine-tune ──────────────────────────
    if args.resume:
        resume_ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(resume_ckpt["model"], strict=False)
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_epoch = resume_ckpt.get("epoch", 0) + 1
        best_auc    = resume_ckpt.get("best_auc", best_auc)
        print(f"[finetune] resumed from {args.resume}  "
              f"(epoch {start_epoch}, best_auc={best_auc:.4f})\n")

    ckpt_dir = Path(args.save).parent

    print(f"[finetune] backbone_lr={args.backbone_lr}  cnn_lr={args.cnn_lr}  "
          f"epochs={args.epochs}  patience={args.patience}")
    print(f"[finetune] saving best to: {args.save}")
    print(f"[finetune] epoch checkpoints every {args.save_every} epochs "
          f"(resume with --resume <epoch_ckpt>)\n")

    # ── training loop ─────────────────────────────────────────────────
    epoch_times = []
    for epoch in range(start_epoch, args.epochs):
        t0   = time.time()
        loss = train_one_epoch(
            model, train_loader, optimizer, scheduler,
            criterion, device, grad_clip, args.accum_steps, noctx,
        )
        elapsed = time.time() - t0
        epoch_times.append(elapsed)
        avg_t   = sum(epoch_times[-3:]) / len(epoch_times[-3:])
        eta_m   = int(avg_t * (args.epochs - epoch - 1) / 60)
        print(f"Epoch [{epoch+1:3d}/{args.epochs}]  loss={loss:.4f}  "
              f"({elapsed:.0f}s)  ETA ~{eta_m}m")

        yt, yp = evaluate(model, val_loader, device, noctx)
        m      = compute_all_metrics(yt, yp)
        print_metrics(m, prefix=f"val ep{epoch+1}")
        print_confusion_matrix(m)

        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc":  best_auc,
                "metrics":   m,
                "cfg":       cfg,
                "protocol":  "finetune",
                "from_ckpt": args.checkpoint,
                "target_generators": args.target_generators,
            }, args.save)
            print(f"  * new best val AUC={best_auc:.4f} -> saved {args.save}")

        # Periodic epoch checkpoint — always saved so you can resume if cancelled
        if (epoch + 1) % args.save_every == 0:
            epoch_ckpt = ckpt_dir / f"finetune_epoch_{epoch+1:03d}.pt"
            torch.save({
                "epoch":     epoch,
                "model":     model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_auc":  best_auc,
                "cfg":       cfg,
                "protocol":  "finetune",
                "from_ckpt": args.checkpoint,
            }, epoch_ckpt)
            print(f"  [ckpt] {epoch_ckpt.name} saved (resume with --resume {epoch_ckpt})")

        if stopper.step(m["auc"]):
            print(f"[finetune] early stop at epoch {epoch+1}")
            break

    print(f"\n[finetune] done. Best val AUC={best_auc:.4f}")
    print(f"[finetune] fine-tuned model: {args.save}")
    print(f"\nEvaluate with:")
    print(f"  python evaluate_per_generator.py \\")
    print(f"      --checkpoint {args.save} \\")
    print(f"      --generators_root per-gen-dataset-test \\")
    print(f"      --clip_cache_dir datasets_eq/clip_cache_test")


if __name__ == "__main__":
    main()
