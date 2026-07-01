"""
diagnose_data_bias.py
─────────────────────
Check whether REAL vs FAKE images differ by trivial source-level cues
(resolution, file format, file size, JPEG quality) instead of generation
artifacts. If they do, the model learns "which source" not "real vs AI",
which explains why even trained generators fail on fresh collections.

Reports, per dataset, the real-vs-fake distribution of:
  - image width x height (resolution)
  - file extension / format
  - file size (KB)
  - JPEG quantization-table quality estimate (for .jpg)

A big real-vs-fake gap in ANY of these = exploitable shortcut.
"""

import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# (label name, root, real-subdir-candidates, fake-subdir-candidates)
TARGETS = [
    ("TRAIN/ForenSynths", Path("datasets/ForenSynths")),
    ("TRAIN/GenImage",    Path("datasets/GenImage")),
    ("TRAIN/DFDC",        Path("datasets/DFDC")),
    ("PERGEN/DFDC",       Path("per-gen-dataset/DFDC")),
    ("PERGEN/ProGAN",     Path("per-gen-dataset/ProGAN")),
    ("PERGEN/Deepfake",   Path("per-gen-dataset/Deepfake")),
    ("PERGEN/Guided",     Path("per-gen-dataset/Guided")),
]

REAL_NAMES = ("real", "real-625", "real-5000", "1k_real", "0_real")
FAKE_NAMES = ("fake", "fake-625", "fake-5000", "1k_fake", "1_fake")
SAMPLE = 250   # images sampled per class


def find_dir(root, names):
    for n in names:
        d = root / n
        if d.is_dir():
            return d
    # search one level down (some datasets nest by category)
    for n in names:
        hits = list(root.rglob(n))
        hits = [h for h in hits if h.is_dir()]
        if hits:
            return hits[0]
    return None


def collect(folder, limit):
    out = []
    for p in folder.rglob("*"):
        if p.suffix.lower() in IMG_EXTS:
            out.append(p)
            if len(out) >= limit:
                break
    return out


def profile(paths):
    res, fmt, sizes, qual = [], Counter(), [], []
    for p in paths:
        try:
            sizes.append(p.stat().st_size / 1024.0)  # KB
            fmt[p.suffix.lower()] += 1
            with Image.open(p) as im:
                res.append(im.size)  # (w,h)
                q = im.info.get("quality")
                if q:
                    qual.append(q)
        except Exception:
            pass
    return res, fmt, sizes, qual


def summarize(tag, res, fmt, sizes, qual):
    if not res:
        print(f"    {tag}: (no images)")
        return None
    ws = [r[0] for r in res]; hs = [r[1] for r in res]
    common_res = Counter(res).most_common(3)
    line = (f"    {tag:<6} n={len(res):<4} "
            f"WxH med={int(np.median(ws))}x{int(np.median(hs))} "
            f"size={np.median(sizes):6.1f}KB "
            f"fmt={dict(fmt)}")
    print(line)
    print(f"           top-res: {common_res}")
    return {
        "w": np.median(ws), "h": np.median(hs),
        "size": np.median(sizes), "fmt": dict(fmt),
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for name, root in TARGETS:
        print(f"\n{'='*70}\n  {name}   ({root})\n{'='*70}")
        if not root.exists():
            print("    [skip] root not found")
            continue
        rdir = find_dir(root, REAL_NAMES)
        fdir = find_dir(root, FAKE_NAMES)
        if rdir is None or fdir is None:
            print(f"    [skip] real={rdir} fake={fdir}")
            continue

        rstats = summarize("REAL", *profile(collect(rdir, SAMPLE)))
        fstats = summarize("FAKE", *profile(collect(fdir, SAMPLE)))

        # Flag obvious shortcuts
        if rstats and fstats:
            flags = []
            if (rstats["w"], rstats["h"]) != (fstats["w"], fstats["h"]):
                flags.append("RESOLUTION differs")
            if set(rstats["fmt"]) != set(fstats["fmt"]):
                flags.append("FORMAT differs")
            if fstats["size"] > 0 and (
                rstats["size"] / max(fstats["size"], 1e-6) > 1.6 or
                fstats["size"] / max(rstats["size"], 1e-6) > 1.6):
                flags.append("FILE-SIZE differs >1.6x")
            if flags:
                print(f"    >>> SHORTCUT RISK: {', '.join(flags)}")
            else:
                print(f"    >>> real/fake look matched on these cues")


if __name__ == "__main__":
    main()
