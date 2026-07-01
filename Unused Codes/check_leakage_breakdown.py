"""Quick per-generator leakage breakdown."""
import hashlib, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect(root):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

sys.stdout.reconfigure(encoding="utf-8")
train_root = Path("datasets")
eval_root  = Path(r"D:\Dataset\per-gen-dataset")

print("Hashing training images ...")
train_hashes = {}
for p in tqdm(collect(train_root), unit="img"):
    train_hashes[md5(p)] = p

print("\nChecking per generator ...")
by_gen = defaultdict(int)
for gen_dir in sorted(eval_root.iterdir()):
    if not gen_dir.is_dir(): continue
    hits = 0
    for p in collect(gen_dir):
        if md5(p) in train_hashes:
            hits += 1
    by_gen[gen_dir.name] = hits
    print(f"  {gen_dir.name:<15} {hits:4d} duplicates")

print(f"\nTotal: {sum(by_gen.values())} duplicates across {len(by_gen)} generators")
