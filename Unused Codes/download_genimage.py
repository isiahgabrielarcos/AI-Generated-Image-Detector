"""
Download cartografia/unbiased-tiny-genimage (~2.35 GB) and inspect structure.
"""
import ssl, os, json, zipfile
from pathlib import Path

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import requests as _req
for method in ('send', 'request'):
    orig = getattr(_req.Session, method)
    def patched(self, *a, _orig=orig, **kw):
        kw['verify'] = False
        return _orig(self, *a, **kw)
    setattr(_req.Session, method, patched)

os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")
from kaggle import api as kaggle_api
kaggle_api.authenticate()

SLUG         = "cartografia/unbiased-tiny-genimage"
DOWNLOAD_DIR = Path("datasets/_download_tmp/genimage")
ZIP_PATH     = DOWNLOAD_DIR / "genimage.zip"
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print(f"Downloading {SLUG} (~2.35 GB) ...")
    kaggle_api.dataset_download_files(SLUG, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)
    zips = [p for p in DOWNLOAD_DIR.glob("*.zip") if p != ZIP_PATH]
    if zips:
        zips[0].rename(ZIP_PATH)
    print(f"Saved: {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1024/1024/1024:.2f} GB)")
else:
    print(f"Already downloaded ({ZIP_PATH.stat().st_size/1024/1024/1024:.2f} GB)")

print("\nInspecting structure ...")
with zipfile.ZipFile(ZIP_PATH) as zf:
    all_names = zf.namelist()
    img_names = [n for n in all_names if Path(n).suffix.lower() in IMG_EXTS]
    print(f"Total entries: {len(all_names)}  |  Images: {len(img_names)}")

    # Show first 50 entries
    print("\nFirst 50 entries:")
    for n in all_names[:50]:
        print(" ", n)

    # Count by top-level and second-level dirs
    from collections import Counter
    lvl2 = Counter()
    for n in img_names:
        parts = Path(n).parts
        key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
        lvl2[key] += 1
    print("\nImages per folder (top 40):")
    for folder, count in sorted(lvl2.items())[:40]:
        print(f"  {folder:<70} {count}")
