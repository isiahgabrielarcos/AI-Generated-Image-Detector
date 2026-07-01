"""
evaluate_per_generator.py
─────────────────────────
Reproduce Tables 1, 2, 3 from Man & Cho (2026) for your own trained model.

Each generator is evaluated independently using its fake-625/ + real-625/
folders (or fake/ + real/ if that's the naming convention).
ACC and AP are reported per generator and averaged (Mean column).

Tables produced:
  Table 1 – Face deepfake dataset (DFDC)
  Table 2 – GAN-based generators   (ForenSynths: ProGAN, StyleGAN, …)
  Table 3 – Diffusion generators   (GenImage: PNDM, Guided, DALL-E, VQ-Diffusion)

Usage examples:
  # Single checkpoint (full model)
  python evaluate_per_generator.py \\
      --checkpoint checkpoints/best_model.pt \\
      --generators_root D:\\Dataset

  # Ablation study: pass multiple checkpoints with labels
  python evaluate_per_generator.py \\
      --checkpoint checkpoints/clip_only.pt      --label "Ours (Clip)" \\
      --checkpoint checkpoints/clip_f.pt         --label "Ours (Clip+F)" \\
      --checkpoint checkpoints/clip_f_a.pt       --label "Ours (Clip+F+A)" \\
      --checkpoint checkpoints/best_model.pt     --label "Ours (Clip+F+A+G)" \\
      --generators_root D:\\Dataset

  # Override ablation mode at eval time (no separate training needed)
  python evaluate_per_generator.py \\
      --checkpoint checkpoints/best_model.pt \\
      --ablation clip \\
      --generators_root D:\\Dataset

Folder layout expected under --generators_root:
  <generators_root>/
    DFDC/           fake-625/  real-625/
    ProGAN/         fake-625/  real-625/
    StyleGAN/       fake-625/  real-625/
    StyleGAN2/      fake-625/  real-625/
    BigGAN/         fake-625/  real-625/
    CycleGAN/       fake-625/  real-625/
    StarGAN/        fake-625/  real-625/
    GauGAN/         fake-625/  real-625/
    Deepfake/       fake-625/  real-625/
    PNDM/           fake-625/  real-625/
    Guided/         fake-625/  real-625/
    DALL-E/         fake-625/  real-625/
    VQ-Diffusion/   fake-625/  real-625/

  Also accepts fake/ + real/ naming (legacy).

Output:
  results/per_generator/tables.txt   – formatted ASCII tables (copy to paper)
  results/per_generator/results.csv  – full numeric results
"""

import argparse
import csv
import sys
import yaml
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

from models import build_detector
from utils import compute_all_metrics, setup_hf_auth
from data.dataset import ClipFeatureCache, clip_collate_fn

# ──────────────────────────────────────────────────────────────────────
# Generator groups matching Tables 1 / 2 / 3 of Man & Cho (2026)
# ──────────────────────────────────────────────────────────────────────

TABLE1_GENERATORS = ["DFDC"]

TABLE2_GENERATORS = [
    "ProGAN", "StyleGAN", "StyleGAN2", "BigGAN",
    "CycleGAN", "StarGAN", "GauGAN", "Deepfake",
]

TABLE3_GENERATORS = ["PNDM", "Guided", "DALL-E", "VQ-Diffusion"]

_CLIP_MEAN = (0.48145466, 0.4578275,  0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".JPEG"}


# ──────────────────────────────────────────────────────────────────────
# Minimal dataset that accepts fake-625/ + real-625/  OR  fake/ + real/
# ──────────────────────────────────────────────────────────────────────

class GeneratorDataset(Dataset):
    """Load real + fake images for a single generator."""

    def __init__(self, generator_root: Path,
                 clip_cache: Optional[ClipFeatureCache] = None):
        # Try both naming conventions
        fake_dir = self._find_dir(generator_root, ("fake", "fake-625", "fake-5000", "1k_fake"))
        real_dir = self._find_dir(generator_root, ("real", "real-625", "real-5000", "1k_real"))

        if fake_dir is None:
            raise FileNotFoundError(
                f"No fake/ or fake-625/ folder under {generator_root}")
        if real_dir is None:
            raise FileNotFoundError(
                f"No real/ or real-625/ folder under {generator_root}")

        self.samples = (
            [(p, 1) for p in self._collect(fake_dir)] +
            [(p, 0) for p in self._collect(real_dir)]
        )

        self.transform = T.Compose([
            T.Resize(224),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])
        self.clip_cache = clip_cache

    @staticmethod
    def _find_dir(root: Path, names: tuple) -> Optional[Path]:
        for name in names:
            d = root / name
            if d.is_dir():
                return d
        return None

    @staticmethod
    def _collect(folder: Path):
        # resolve() ensures absolute paths so they match the CLIP cache keys
        # regardless of whether --generators_root was passed as relative or absolute
        return sorted(
            p.resolve() for p in folder.rglob("*")
            if p.suffix.lower() in {e.lower() for e in _IMG_EXTS}
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        img_t   = self.transform(img)
        label_t = torch.tensor(label, dtype=torch.float32)

        if self.clip_cache is not None:
            tokens = self.clip_cache.get(path)
            if tokens is None:
                tokens = torch.zeros(256, 1024, dtype=torch.float32)
            return img_t, tokens, label_t

        return img_t, label_t


# ──────────────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer(model, loader, device) -> tuple[list, list]:
    model.eval()
    y_true, y_prob = [], []
    for batch in tqdm(loader, desc="    batches", leave=False, unit="batch"):
        if len(batch) == 3:
            imgs, clip_tokens, labels = batch
            imgs        = imgs.to(device, non_blocking=True)
            clip_tokens = clip_tokens.to(device, non_blocking=True)
            logits = model(imgs, clip_tokens=clip_tokens)
        else:
            imgs, labels = batch
            imgs   = imgs.to(device, non_blocking=True)
            logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze(-1)
        y_true.extend(labels.cpu().tolist())
        y_prob.extend(probs.cpu().tolist())
    return y_true, y_prob


def evaluate_generator(
    name: str,
    gen_root: Path,
    model,
    device,
    batch_size: int,
    clip_cache: Optional[ClipFeatureCache] = None,
) -> Optional[dict]:
    """Evaluate a single generator. Returns None if folder missing."""
    if not gen_root.exists():
        return None
    try:
        ds = GeneratorDataset(gen_root, clip_cache=clip_cache)
    except FileNotFoundError as e:
        print(f"  [skip] {name}: {e}")
        return None

    if len(ds) == 0:
        print(f"  [skip] {name}: empty dataset")
        return None

    use_cache  = clip_cache is not None and clip_cache.available
    collate_fn = clip_collate_fn if use_cache else None
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False,
                        collate_fn=collate_fn)
    y_true, y_prob = infer(model, loader, device)
    metrics = compute_all_metrics(y_true, y_prob)
    return metrics


# ──────────────────────────────────────────────────────────────────────
# Table printing
# ──────────────────────────────────────────────────────────────────────

def _fmt(val: Optional[float]) -> str:
    if val is None:
        return "  —  "
    return f"{val * 100:5.1f}"


def print_table(
    table_num: int,
    title: str,
    generators: list[str],
    results: dict,           # { model_label: { generator: metrics_dict } }
    out_lines: list,
):
    """Print one comparison table (like paper Tables 1/2/3)."""
    model_labels = list(results.keys())

    # Header
    gen_w = max(len(g) for g in generators) + 2
    col_w = 13   # "ACC / AP" per generator

    sep  = "─" * (22 + col_w * len(generators) + 13)
    hdr  = f"\nTable {table_num}: {title}"
    subh = f"  Format: ACC / AP  (× 100)"

    lines = [sep, hdr, subh, sep]

    # Column headers
    row = f"  {'Methods':<20}"
    for g in generators:
        row += f"  {g:^{col_w-2}}"
    row += f"  {'Mean':^11}"
    lines.append(row)
    lines.append("─" * len(sep))

    for label in model_labels:
        gen_metrics = results[label]
        row = f"  {label:<20}"
        accs, aps = [], []
        for g in generators:
            m = gen_metrics.get(g)
            if m:
                acc_s = _fmt(m["acc"])
                ap_s  = _fmt(m["ap"])
                cell  = f"{acc_s}/{ap_s}"
                accs.append(m["acc"])
                aps.append(m["ap"])
            else:
                cell = " " * (col_w - 2)
            row += f"  {cell:^{col_w-2}}"

        if accs:
            mean_acc = np.mean(accs)
            mean_ap  = np.mean(aps)
            row += f"  {_fmt(mean_acc)}/{_fmt(mean_ap)}"
        lines.append(row)

    lines.append(sep)

    for line in lines:
        print(line)
        out_lines.append(line)


# ──────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────

class _CheckpointAction(argparse.Action):
    """Accumulate (checkpoint, label) pairs from interleaved flags."""
    def __call__(self, parser, namespace, values, option_string=None):
        pairs = getattr(namespace, "checkpoint_pairs", None) or []
        pairs.append({"ckpt": values, "label": None})
        namespace.checkpoint_pairs = pairs


class _LabelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        pairs = getattr(namespace, "checkpoint_pairs", None) or []
        if pairs:
            pairs[-1]["label"] = values
        namespace.checkpoint_pairs = pairs


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-generator evaluation – reproduces Man & Cho Tables 1/2/3")
    p.add_argument("--checkpoint",  action=_CheckpointAction, dest="checkpoint_pairs",
                   metavar="PATH", help="Checkpoint .pt file (repeatable)")
    p.add_argument("--label",       action=_LabelAction,      dest="checkpoint_pairs",
                   metavar="NAME",  help="Label for the preceding --checkpoint")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--generators_root", required=True,
                   help="Parent folder containing one subfolder per generator")
    p.add_argument("--output_dir",  default="results/per_generator")
    p.add_argument("--batch_size",  type=int, default=16)
    p.add_argument("--device",      default=None)
    p.add_argument("--ablation",    default=None,
                   choices=[None, "clip", "clip_f", "clip_f_a"],
                   help="Override ablation mode (applies to all checkpoints)")
    p.add_argument("--clip_cache_dir", default="datasets/clip_cache",
                   help="Directory containing pergen_*_clip.pt cache files "
                        "(default: datasets/clip_cache). Pass 'none' to disable.")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    setup_hf_auth()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[eval] device = {device}")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    gen_root   = Path(args.generators_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint list
    pairs = getattr(args, "checkpoint_pairs", None) or []
    if not pairs:
        p_obj = argparse.ArgumentParser()
        p_obj.error("At least one --checkpoint is required.")
    for i, pair in enumerate(pairs):
        if pair["label"] is None:
            pair["label"] = f"Model-{i+1}" if len(pairs) > 1 else "Ours (Clip+F+A+G)"

    all_generators = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS

    # ── Load CLIP cache (one-time; shared across all checkpoints) ────
    clip_cache: Optional[ClipFeatureCache] = None
    cache_dir_str = args.clip_cache_dir
    if cache_dir_str and cache_dir_str.lower() != "none":
        cache_dir = Path(cache_dir_str)
        if cache_dir.exists():
            cache_names = [
                f"pergen_{g}" for g in all_generators
                if (cache_dir / f"pergen_{g}_clip.pt").exists()
            ]
            if cache_names:
                print(f"\n[eval] Loading CLIP cache ({len(cache_names)} generators) ...")
                clip_cache = ClipFeatureCache(cache_dir, cache_names)
            else:
                print(f"[eval] No pergen_*_clip.pt files found in {cache_dir} — "
                      "running live CLIP inference (slow).")
        else:
            print(f"[eval] clip_cache_dir not found: {cache_dir} — "
                  "running live CLIP inference (slow).")

    # Skip loading 1.2 GB frozen ViT when every present generator has a cache file
    if clip_cache is not None and clip_cache.available:
        present_gens = [g for g in all_generators if (gen_root / g).exists()]
        force_visual = not all(
            (Path(cache_dir_str) / f"pergen_{g}_clip.pt").exists()
            for g in present_gens
        )
    else:
        force_visual = True

    # results[model_label][generator] = metrics_dict
    results: dict[str, dict[str, Optional[dict]]] = {}

    for pair in pairs:
        ckpt_path = pair["ckpt"]
        label     = pair["label"]

        print(f"\n{'='*60}")
        print(f"  Model: {label}")
        print(f"  Checkpoint: {ckpt_path}")
        if not force_visual:
            print("  (ViT skipped — all generators have CLIP cache)")
        print(f"{'='*60}")

        model = build_detector(cfg, force_load_visual=force_visual,
                               ablation=args.ablation).to(device)
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()

        results[label] = {}

        for gen_name in all_generators:
            gen_dir = gen_root / gen_name
            print(f"  {gen_name:<15} ", end="", flush=True)
            m = evaluate_generator(gen_name, gen_dir, model, device,
                                   args.batch_size, clip_cache=clip_cache)
            results[label][gen_name] = m
            if m:
                print(f"ACC={m['acc']*100:.1f}  AP={m['ap']*100:.1f}")
            else:
                print("(skipped)")

    # ── Print and save tables ─────────────────────────────────────────
    out_lines: list[str] = []

    print_table(1, "Face Deepfake Dataset (DFDC)",
                TABLE1_GENERATORS, results, out_lines)

    print_table(2, "GAN-Based Generators (ForenSynths)",
                TABLE2_GENERATORS, results, out_lines)

    print_table(3, "Diffusion-Based Generators (GenImage)",
                TABLE3_GENERATORS, results, out_lines)

    # ── Save tables.txt ───────────────────────────────────────────────
    txt_path = output_dir / "tables.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\nTables saved -> {txt_path}")

    # ── Save CSV ──────────────────────────────────────────────────────
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "generator", "table",
                         "acc", "ap", "recall", "f1", "auc",
                         "n_samples"])

        def _table_of(g):
            if g in TABLE1_GENERATORS: return 1
            if g in TABLE2_GENERATORS: return 2
            if g in TABLE3_GENERATORS: return 3
            return 0

        for label, gen_dict in results.items():
            for gen_name, m in gen_dict.items():
                if m is None:
                    continue
                writer.writerow([
                    label, gen_name, _table_of(gen_name),
                    f"{m['acc']:.6f}", f"{m['ap']:.6f}",
                    f"{m['recall']:.6f}", f"{m['f1']:.6f}", f"{m['auc']:.6f}",
                    int(m["confusion_matrix"].sum()),
                ])

    print(f"CSV saved    -> {csv_path}")

    # ── Mean summary per table ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  MEAN SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Model':<22}  {'T1 ACC/AP':>10}  {'T2 ACC/AP':>10}  {'T3 ACC/AP':>10}")
    print("  " + "─" * 56)

    for label, gen_dict in results.items():
        row_parts = [f"  {label:<22}"]
        for group in [TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS]:
            accs = [gen_dict[g]["acc"] for g in group if gen_dict.get(g)]
            aps  = [gen_dict[g]["ap"]  for g in group if gen_dict.get(g)]
            if accs:
                row_parts.append(
                    f"  {np.mean(accs)*100:5.1f}/{np.mean(aps)*100:5.1f}")
            else:
                row_parts.append("      —    ")
        print("".join(row_parts))


if __name__ == "__main__":
    main()
