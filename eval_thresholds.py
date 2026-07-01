"""
Quick threshold sweep for ablations: evaluates mean ACC at thresholds 0.3, 0.4, 0.5
Usage:
  python eval_thresholds.py --checkpoint checkpoints/pergen_split_best_model.pt \
      --generators_root per-gen-dataset-test --clip_cache_dir datasets_eq/clip_cache_test
"""

import argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader

import yaml

from evaluate_per_generator import GeneratorDataset, infer, TABLE1_GENERATORS, TABLE2_GENERATORS, TABLE3_GENERATORS
from models import build_detector
from data.dataset import ClipFeatureCache
from utils.metrics import compute_all_metrics
from utils import setup_hf_auth

ALL_GENERATORS = TABLE1_GENERATORS + TABLE2_GENERATORS + TABLE3_GENERATORS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--generators_root", required=True)
    p.add_argument("--clip_cache_dir", default="datasets/clip_cache")
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def load_clip_cache(cache_dir, present_gens):
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return None
    cache_names = [f"pergen_{g}" for g in present_gens if (cache_dir / f"pergen_{g}_clip.pt").exists()]
    if cache_names:
        return ClipFeatureCache(cache_dir, cache_names)
    return None


@torch.no_grad()
def eval_model_for_ablation(cfg, ckpt_path, ablation, gen_root, clip_cache, device, batch_size):
    model = build_detector(cfg, ablation=ablation).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    thr_list = [0.3, 0.4, 0.5]
    accs_by_thr = {thr: [] for thr in thr_list}

    for g in ALL_GENERATORS:
        gen_dir = Path(gen_root) / g
        if not gen_dir.exists():
            continue
        try:
            ds = GeneratorDataset(gen_dir, clip_cache=clip_cache)
        except FileNotFoundError:
            continue
        if len(ds) == 0:
            continue
        use_cache = clip_cache is not None and clip_cache.available
        collate_fn = None
        if use_cache:
            collate_fn = getattr(__import__('data.dataset', fromlist=['clip_collate_fn']), 'clip_collate_fn')
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, collate_fn=collate_fn)
        y_true, y_prob = infer(model, loader, device)
        for thr in thr_list:
            m = compute_all_metrics(y_true, y_prob, threshold=thr)
            accs_by_thr[thr].append(m['acc'])

    mean_accs = {thr: (np.mean(v) if v else float('nan')) for thr, v in accs_by_thr.items()}
    return mean_accs


def main():
    args = parse_args()
    setup_hf_auth()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    with open('configs/default.yaml') as f:
        cfg = yaml.safe_load(f)

    gen_root = Path(args.generators_root)
    present_gens = [g for g in ALL_GENERATORS if (gen_root / g).exists()]
    clip_cache = load_clip_cache(args.clip_cache_dir, present_gens)

    print(f"Device = {device}")
    print(f"Present generators: {present_gens}")

    results = {}
    for ab in ['clip', 'clip_f', 'clip_f_a']:
        print(f"\nEvaluating ablation={ab} ...")
        mean_accs = eval_model_for_ablation(cfg, args.checkpoint, ab, args.generators_root, clip_cache, device, args.batch_size)
        results[ab] = mean_accs
        print(f"  Mean ACCs: { {k: f'{v*100:.2f}%' if not np.isnan(v) else 'n/a' for k,v in mean_accs.items()} }")

    print("\nSummary (mean ACC across present generators):")
    for ab, mean_accs in results.items():
        print(f"  {ab}: 0.3={mean_accs[0.3]*100:5.2f}%  0.4={mean_accs[0.4]*100:5.2f}%  0.5={mean_accs[0.5]*100:5.2f}%")


if __name__ == '__main__':
    main()
