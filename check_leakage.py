"""
check_leakage.py
────────────────
Check for image duplicates between training datasets and per-gen-dataset.

Two passes:
  1. Exact hash (MD5)  – catches identical files regardless of filename
  2. pHash             – catches same image at different resolution/compression
                         (only runs if exact-hash pass finds nothing suspicious)

Usage:
    python check_leakage.py

    # Only exact-hash check (fast, ~30 sec)
    python check_leakage.py --no-phash

    # Custom paths
    python check_leakage.py --train-root datasets --eval-root "D:\\Dataset\\per-gen-dataset"
"""

import argparse
import hashlib
import sys
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".JPEG"}


def collect_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in {e.lower() for e in IMG_EXTS})


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def exact_check(train_files: list[Path], eval_files: list[Path]) -> list[tuple]:
    print("\n[1/2] Exact hash check ...")

    print(f"  Hashing {len(train_files):,} training images ...")
    train_hashes: dict[str, Path] = {}
    for p in tqdm(train_files, desc="  train", unit="img"):
        train_hashes[md5(p)] = p

    print(f"  Hashing {len(eval_files):,} eval images ...")
    duplicates = []
    for p in tqdm(eval_files, desc="  eval ", unit="img"):
        h = md5(p)
        if h in train_hashes:
            duplicates.append((train_hashes[h], p))

    return duplicates


def phash_check(train_files: list[Path], eval_files: list[Path],
                threshold: int = 8) -> list[tuple]:
    try:
        import imagehash
        from PIL import Image
    except ImportError:
        print("  [skip] imagehash not installed — skipping pHash check")
        return []

    print(f"\n[2/2] pHash near-duplicate check (threshold={threshold}) ...")

    print(f"  Hashing {len(train_files):,} training images ...")
    train_hashes: list[tuple] = []
    for p in tqdm(train_files, desc="  train", unit="img"):
        try:
            with Image.open(p) as img:
                train_hashes.append((imagehash.phash(img.convert("RGB")), p))
        except Exception:
            pass

    print(f"  Comparing {len(eval_files):,} eval images against training ...")
    near_dupes = []
    for p in tqdm(eval_files, desc="  eval ", unit="img"):
        try:
            with Image.open(p) as img:
                h = imagehash.phash(img.convert("RGB"))
            for th, tp in train_hashes:
                if (h - th) <= threshold:
                    near_dupes.append((tp, p, int(h - th)))
                    break
        except Exception:
            pass

    return near_dupes


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    p = argparse.ArgumentParser()
    p.add_argument("--train-root", default="datasets",
                   help="Root of training datasets (contains DFDC/, ForenSynths/, GenImage/)")
    p.add_argument("--eval-root",  default=r"D:\Dataset\per-gen-dataset",
                   help="Per-generator evaluation dataset root")
    p.add_argument("--no-phash",   action="store_true",
                   help="Skip pHash near-duplicate check (exact hash only)")
    args = p.parse_args()

    train_root = Path(args.train_root)
    eval_root  = Path(args.eval_root)

    if not train_root.exists():
        print(f"[error] train-root not found: {train_root}")
        return
    if not eval_root.exists():
        print(f"[error] eval-root not found: {eval_root}")
        return

    # ── Collect files ─────────────────────────────────────────────────
    print(f"Training root : {train_root.resolve()}")
    print(f"Eval root     : {eval_root.resolve()}")

    print("\nCollecting training images ...")
    train_files = collect_images(train_root)
    print(f"  {len(train_files):,} training images found")

    print("Collecting eval images ...")
    eval_files = collect_images(eval_root)
    print(f"  {len(eval_files):,} eval images found")

    if not train_files or not eval_files:
        print("[error] One or both folders are empty.")
        return

    # ── Pass 1: Exact hash ────────────────────────────────────────────
    exact_dupes = exact_check(train_files, eval_files)

    print(f"\n  Exact duplicates found: {len(exact_dupes)}")
    if exact_dupes:
        print("  !! DATA LEAKAGE DETECTED !!")
        for train_p, eval_p in exact_dupes[:20]:
            print(f"    TRAIN: {train_p}")
            print(f"    EVAL : {eval_p}")
            print()
        if len(exact_dupes) > 20:
            print(f"  ... and {len(exact_dupes) - 20} more")
    else:
        print("  Clean — no exact duplicates between training and eval sets.")

    # ── Pass 2: pHash ─────────────────────────────────────────────────
    if not args.no_phash:
        near_dupes = phash_check(train_files, eval_files)
        print(f"\n  Near-duplicates found (pHash ≤8): {len(near_dupes)}")
        if near_dupes:
            print("  !! NEAR-DUPLICATES DETECTED !!")
            for train_p, eval_p, dist in near_dupes[:20]:
                print(f"    dist={dist}  TRAIN: {train_p}")
                print(f"           EVAL : {eval_p}")
                print()
            if len(near_dupes) > 20:
                print(f"  ... and {len(near_dupes) - 20} more")
        else:
            print("  Clean — no near-duplicates found.")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print("  LEAKAGE SUMMARY")
    print(f"{'='*50}")
    print(f"  Training images checked : {len(train_files):,}")
    print(f"  Eval images checked     : {len(eval_files):,}")
    print(f"  Exact duplicates        : {len(exact_dupes)}")
    if not args.no_phash:
        print(f"  Near-duplicates (pHash) : {len(near_dupes)}")
    verdict = "CLEAN" if len(exact_dupes) == 0 else "LEAKAGE DETECTED"
    if not args.no_phash and near_dupes:
        verdict = "NEAR-DUPLICATES DETECTED"
    print(f"  Verdict                 : {verdict}")


if __name__ == "__main__":
    main()
