"""
remap_cache_paths.py
────────────────────
Update the absolute image paths stored inside pergen_*_clip.pt cache files
after the dataset has been moved to a different drive / directory.

Usage:
    python remap_cache_paths.py \
        --cache_dir datasets_eq/clip_cache_test \
        --old_prefix "D:\\Dataset\\per-gen-dataset-test" \
        --new_prefix "per-gen-dataset-test"

    # --new_prefix can be relative (resolved to absolute automatically)
    # or already absolute.
"""

import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir",   required=True,
                   help="Folder containing pergen_*_clip.pt files")
    p.add_argument("--old_prefix",  required=True,
                   help="Path prefix to replace (as stored in the cache)")
    p.add_argument("--new_prefix",  required=True,
                   help="Replacement prefix (relative or absolute)")
    p.add_argument("--dry_run",     action="store_true",
                   help="Print what would change without writing")
    return p.parse_args()


def remap_file(cache_path: Path, old_prefix: str, new_prefix: str, dry_run: bool):
    data = torch.load(cache_path, map_location="cpu", weights_only=True)
    paths: list = data["paths"]

    # Normalise separators so comparison is OS-independent
    old_norm = old_prefix.replace("/", "\\").rstrip("\\")
    new_norm = str(Path(new_prefix).resolve())

    remapped = []
    changed  = 0
    for p in paths:
        p_norm = p.replace("/", "\\")
        if p_norm.startswith(old_norm):
            suffix = p_norm[len(old_norm):]               # e.g. \DFDC\1k_fake\img.jpg
            new_p  = new_norm + suffix.replace("\\", "\\")
            remapped.append(new_p)
            changed += 1
        else:
            remapped.append(p)

    print(f"  {cache_path.name}: {changed}/{len(paths)} paths remapped", end="")

    if dry_run or changed == 0:
        print("  [dry-run, not saved]" if dry_run else "  [no change]")
        return

    data["paths"] = remapped
    torch.save(data, cache_path)
    print("  -> saved")


def main():
    args     = parse_args()
    cache_dir = Path(args.cache_dir)

    if not cache_dir.exists():
        print(f"[remap] Cache dir not found: {cache_dir}")
        return

    cache_files = sorted(cache_dir.glob("pergen_*_clip.pt"))
    if not cache_files:
        print(f"[remap] No pergen_*_clip.pt files found in {cache_dir}")
        return

    print(f"[remap] {len(cache_files)} cache files in {cache_dir}")
    print(f"[remap] old prefix: {args.old_prefix}")
    print(f"[remap] new prefix: {Path(args.new_prefix).resolve()}\n")

    for cf in cache_files:
        remap_file(cf, args.old_prefix, args.new_prefix, args.dry_run)

    print("\n[remap] Done.")


if __name__ == "__main__":
    main()
