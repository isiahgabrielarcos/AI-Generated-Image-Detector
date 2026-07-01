"""
Extract ForenSynths substitute and GenImage into their final dataset folders.

ForenSynths → datasets/ForenSynths/real/  +  fake/
GenImage    → datasets/GenImage/real/     +  fake/
"""
import zipfile, random, re
from pathlib import Path
from collections import defaultdict

random.seed(42)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ─────────────────────────────────────────────────────────────────────
# 1.  ForenSynths substitute
# ─────────────────────────────────────────────────────────────────────
FS_ZIP   = Path("datasets/_download_tmp/forensynths_sub/forensynths_sub.zip")
FS_REAL  = Path("datasets/ForenSynths/real")
FS_FAKE  = Path("datasets/ForenSynths/fake")
FS_REAL.mkdir(parents=True, exist_ok=True)
FS_FAKE.mkdir(parents=True, exist_ok=True)

print("=== ForenSynths substitute ===")
with zipfile.ZipFile(FS_ZIP) as zf:
    real_imgs, fake_imgs = [], []
    for name in zf.namelist():
        if Path(name).suffix.lower() not in IMG_EXTS:
            continue
        low = name.lower()
        if "/real/" in low:
            real_imgs.append(name)
        elif "/fake" in low:          # Fake_GAN and Fake_Diffusion
            fake_imgs.append(name)

    print(f"  Found {len(real_imgs)} real, {len(fake_imgs)} fake")

    # Extract all (already balanced at 5k/5k)
    for i, name in enumerate(real_imgs):
        dest = FS_REAL / f"real_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  real: {i+1}/{len(real_imgs)}")

    for i, name in enumerate(fake_imgs):
        dest = FS_FAKE / f"fake_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  fake: {i+1}/{len(fake_imgs)}")

print(f"  Done → real: {len(list(FS_REAL.glob('*.jpg')))}, fake: {len(list(FS_FAKE.glob('*.jpg')))}")

# ─────────────────────────────────────────────────────────────────────
# 2.  GenImage
# ─────────────────────────────────────────────────────────────────────
GI_ZIP   = Path("datasets/_download_tmp/genimage/genimage.zip")
GI_REAL  = Path("datasets/GenImage/real")
GI_FAKE  = Path("datasets/GenImage/fake")
GI_REAL.mkdir(parents=True, exist_ok=True)
GI_FAKE.mkdir(parents=True, exist_ok=True)

TARGET_REAL = 5000
TARGET_FAKE = 5000   # spread across 7 generators

FAKE_GENERATORS = {"adm", "biggan", "midjourney", "vqdm", "glide",
                   "stable_diffusion_v_1_5", "wukong"}

print("\n=== GenImage ===")
with zipfile.ZipFile(GI_ZIP) as zf:
    real_imgs = []
    fake_by_gen = defaultdict(list)

    for name in zf.namelist():
        if Path(name).suffix.lower() not in IMG_EXTS:
            continue
        top = Path(name).parts[0].lower()
        if top == "nature":
            real_imgs.append(name)
        elif top in FAKE_GENERATORS:
            fake_by_gen[top].append(name)

    print(f"  Real (Nature): {len(real_imgs)}")
    for gen, imgs in sorted(fake_by_gen.items()):
        print(f"    {gen}: {len(imgs)}")

    # Sample real
    real_selected = random.sample(real_imgs, min(TARGET_REAL, len(real_imgs)))

    # Sample fake: equal per generator
    per_gen = TARGET_FAKE // len(fake_by_gen)
    fake_selected = []
    for gen, imgs in fake_by_gen.items():
        chosen = random.sample(imgs, min(per_gen, len(imgs)))
        fake_selected.extend(chosen)
    # Top up if needed
    all_fake = [img for imgs in fake_by_gen.values() for img in imgs
                if img not in set(fake_selected)]
    random.shuffle(all_fake)
    fake_selected += all_fake[: TARGET_FAKE - len(fake_selected)]
    random.shuffle(fake_selected)

    print(f"\n  Selected {len(real_selected)} real, {len(fake_selected)} fake")

    for i, name in enumerate(real_selected):
        dest = GI_REAL / f"real_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  real: {i+1}/{len(real_selected)}")

    for i, name in enumerate(fake_selected):
        dest = GI_FAKE / f"fake_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  fake: {i+1}/{len(fake_selected)}")

print(f"  Done → real: {len(list(GI_REAL.glob('*.jpg')))}, fake: {len(list(GI_FAKE.glob('*.jpg')))}")

# ─────────────────────────────────────────────────────────────────────
print("\n\n=== Final dataset summary ===")
for name, path in [("DFDC",        "datasets/DFDC"),
                   ("ForenSynths", "datasets/ForenSynths"),
                   ("GenImage",    "datasets/GenImage")]:
    r = len(list(Path(path, "real").glob("*")))
    f = len(list(Path(path, "fake").glob("*")))
    print(f"  {name:<15} real={r:5d}  fake={f:5d}  total={r+f:6d}")
