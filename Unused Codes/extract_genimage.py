"""Extract GenImage: 5k real (Nature) + 5k fake (7 generators balanced)."""
import zipfile, random
from pathlib import Path
from collections import defaultdict

random.seed(42)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

GI_ZIP  = Path("datasets/_download_tmp/genimage/genimage.zip")
GI_REAL = Path("datasets/GenImage/real")
GI_FAKE = Path("datasets/GenImage/fake")
GI_REAL.mkdir(parents=True, exist_ok=True)
GI_FAKE.mkdir(parents=True, exist_ok=True)

TARGET_REAL = 5000
TARGET_FAKE = 5000

FAKE_GENERATORS = {"adm", "biggan", "midjourney", "vqdm", "glide",
                   "stable_diffusion_v_1_5", "wukong"}

print("Scanning GenImage zip...")
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

    print(f"Real (Nature): {len(real_imgs)}")
    for gen, imgs in sorted(fake_by_gen.items()):
        print(f"  {gen}: {len(imgs)}")

    # Sample real
    real_selected = random.sample(real_imgs, min(TARGET_REAL, len(real_imgs)))

    # Equal per generator
    per_gen = TARGET_FAKE // len(fake_by_gen)
    fake_selected = []
    for gen, imgs in fake_by_gen.items():
        fake_selected.extend(random.sample(imgs, min(per_gen, len(imgs))))
    # Top up to hit exactly 5000
    all_remaining = [img for imgs in fake_by_gen.values() for img in imgs
                     if img not in set(fake_selected)]
    random.shuffle(all_remaining)
    fake_selected += all_remaining[:TARGET_FAKE - len(fake_selected)]
    random.shuffle(fake_selected)

    print(f"\nSelected: {len(real_selected)} real, {len(fake_selected)} fake")

    print("Extracting real...")
    for i, name in enumerate(real_selected):
        dest = GI_REAL / f"real_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(real_selected)}")

    print("Extracting fake...")
    for i, name in enumerate(fake_selected):
        dest = GI_FAKE / f"fake_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{len(fake_selected)}")

n_real = len(list(GI_REAL.glob("*.jpg")))
n_fake = len(list(GI_FAKE.glob("*.jpg")))
print(f"GenImage done: real={n_real}, fake={n_fake}")

# Final summary
print("\n=== Final dataset summary ===")
for ds_name, path in [("DFDC",        "datasets/DFDC"),
                       ("ForenSynths", "datasets/ForenSynths"),
                       ("GenImage",    "datasets/GenImage")]:
    r = len(list(Path(path, "real").glob("*")))
    f = len(list(Path(path, "fake").glob("*")))
    print(f"  {ds_name:<15} real={r:5d}  fake={f:5d}  total={r+f:6d}")
