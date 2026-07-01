"""Find which training files the PNDM eval duplicates come from."""
import hashlib, sys
from pathlib import Path
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

def collect(root): return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""): h.update(c)
    return h.hexdigest()

sys.stdout.reconfigure(encoding="utf-8")
train_root = Path("datasets")
pndm_eval  = Path(r"D:\Dataset\per-gen-dataset\PNDM\1k_fake")

print("Hashing training images ...")
train_hashes = {}
for p in tqdm(collect(train_root), unit="img"):
    train_hashes[md5(p)] = p

print("Checking PNDM eval images ...")
hits = 0
seen_dirs = set()
for p in tqdm(collect(pndm_eval), unit="img"):
    h = md5(p)
    if h in train_hashes:
        hits += 1
        tp = train_hashes[h]
        seen_dirs.add(str(tp.parent))
        if hits <= 5:
            print(f"  PNDM eval: {p.name}  <- TRAIN: {tp}")

print(f"\nTotal PNDM duplicates: {hits}")
print("Training source directories:")
for d in sorted(seen_dirs):
    print(f"  {d}")
