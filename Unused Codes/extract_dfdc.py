"""
Extract 5k real + 5k fake face images from the downloaded zip into
datasets/DFDC/real/ and datasets/DFDC/fake/

Filename pattern:  celeb_fake_id0_id16_0000_f468.jpg
Video ID         = everything before the last _f{N} suffix
Max frames/video = MAX_PER_VIDEO (default 5)
"""
import zipfile, re, random, shutil
from pathlib import Path
from collections import defaultdict

ZIP_PATH      = Path("datasets/_download_tmp/dfdc_frames/ff_celebdf_frames.zip")
OUT_REAL      = Path("datasets/DFDC/real")
OUT_FAKE      = Path("datasets/DFDC/fake")
TARGET        = 5000
MAX_PER_VIDEO = 5
IMG_EXTS      = {".jpg", ".jpeg", ".png"}

OUT_REAL.mkdir(parents=True, exist_ok=True)
OUT_FAKE.mkdir(parents=True, exist_ok=True)

# ── Parse zip index ───────────────────────────────────────────────────
print("Scanning zip ...")
real_by_video = defaultdict(list)   # video_id → [zip_name, ...]
fake_by_video = defaultdict(list)

# Regex: strip trailing _f{digits} to get video ID
_vid_re = re.compile(r'^(.+)_f\d+$')

with zipfile.ZipFile(ZIP_PATH) as zf:
    all_names = zf.namelist()
    for name in all_names:
        p = Path(name)
        if p.suffix.lower() not in IMG_EXTS:
            continue
        stem = p.stem          # e.g. "celeb_fake_id0_id16_0000_f468"
        m = _vid_re.match(stem)
        vid_id = m.group(1) if m else stem

        # Classify by folder name containing "real" or "fake"
        parts_lower = name.lower()
        if "/real/" in parts_lower or "\\real\\" in parts_lower:
            real_by_video[vid_id].append(name)
        elif "/fake/" in parts_lower or "\\fake\\" in parts_lower:
            fake_by_video[vid_id].append(name)

print(f"Real videos: {len(real_by_video)}  |  total real frames: {sum(len(v) for v in real_by_video.values())}")
print(f"Fake videos: {len(fake_by_video)}  |  total fake frames: {sum(len(v) for v in fake_by_video.values())}")

# ── Apply max-frames-per-video cap, then sample ───────────────────────
random.seed(42)

def sample_capped(by_video, target, max_per_vid):
    """Pick at most max_per_vid frames from each video, shuffle, take target."""
    pool = []
    for vid, frames in by_video.items():
        chosen = random.sample(frames, min(len(frames), max_per_vid))
        pool.extend(chosen)
    random.shuffle(pool)
    return pool[:target]

real_selected = sample_capped(real_by_video, TARGET, MAX_PER_VIDEO)
fake_selected = sample_capped(fake_by_video, TARGET, MAX_PER_VIDEO)

print(f"\nSelected {len(real_selected)} real, {len(fake_selected)} fake")

# ── Extract selected files ────────────────────────────────────────────
print("Extracting ...")
with zipfile.ZipFile(ZIP_PATH) as zf:
    for i, name in enumerate(real_selected):
        dest = OUT_REAL / f"real_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 500 == 0:
            print(f"  real: {i+1}/{len(real_selected)}")

    for i, name in enumerate(fake_selected):
        dest = OUT_FAKE / f"fake_{i:05d}.jpg"
        with zf.open(name) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        if (i + 1) % 500 == 0:
            print(f"  fake: {i+1}/{len(fake_selected)}")

# ── Final count ───────────────────────────────────────────────────────
n_real = len(list(OUT_REAL.glob("*.jpg")))
n_fake = len(list(OUT_FAKE.glob("*.jpg")))
print(f"\nDone. datasets/DFDC/real/: {n_real} images, fake/: {n_fake} images")
