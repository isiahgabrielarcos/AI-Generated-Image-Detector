"""
Download FF++/CelebDF pre-extracted face frame dataset from Kaggle.
wish096/ff-andcelebdf-frame-dataset-by-wish  (~545 MB, already images)
Samples 5k real + 5k fake → datasets/DFDC/real/ and fake/

If the structure uses "real" and "fake" folders we copy directly.
If frames are named with video IDs, we enforce max 5 per video.
"""
import ssl, os, json, re, shutil, random, zipfile
from pathlib import Path
from collections import defaultdict

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import requests as _req
_orig_send = _req.Session.send
def _no_ssl_send(self, request, **kwargs):
    kwargs['verify'] = False
    return _orig_send(self, request, **kwargs)
_req.Session.send = _no_ssl_send

_orig_request = _req.Session.request
def _no_ssl_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)
_req.Session.request = _no_ssl_request

os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")
from kaggle import api as kaggle_api
kaggle_api.authenticate()

DATASET_SLUG  = "wish096/ff-andcelebdf-frame-dataset-by-wish"
DOWNLOAD_DIR  = Path("datasets/_download_tmp/dfdc_frames")
OUT_ROOT      = Path("datasets/DFDC")
TARGET        = 5000
MAX_PER_VIDEO = 5
IMG_EXTS      = {".jpg", ".jpeg", ".png"}

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "real").mkdir(parents=True, exist_ok=True)
(OUT_ROOT / "fake").mkdir(parents=True, exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────
zip_path = DOWNLOAD_DIR / "ff_celebdf_frames.zip"
if not zip_path.exists():
    print(f"Downloading {DATASET_SLUG} ...")
    kaggle_api.dataset_download_files(DATASET_SLUG, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)
    zips = [p for p in DOWNLOAD_DIR.glob("*.zip") if p != zip_path]
    if zips:
        zips[0].rename(zip_path)
    print(f"Saved: {zip_path}  ({zip_path.stat().st_size/1024/1024:.0f} MB)")
else:
    print(f"Already downloaded ({zip_path.stat().st_size/1024/1024:.0f} MB)")

# ── Inspect structure ─────────────────────────────────────────────────
print("\nInspecting zip structure ...")
with zipfile.ZipFile(zip_path) as zf:
    all_names = zf.namelist()
    img_names = [n for n in all_names if Path(n).suffix.lower() in IMG_EXTS]
    print(f"Total entries: {len(all_names)}  |  Images: {len(img_names)}")
    print("First 40 entries:")
    for n in all_names[:40]:
        print(" ", n)
    print("\nSample image paths:")
    for n in img_names[:20]:
        print(" ", n)

    # Detect folder structure
    top_dirs = set()
    for n in all_names[:200]:
        parts = Path(n).parts
        if len(parts) >= 2:
            top_dirs.add(parts[0])
    print(f"\nTop-level folders: {sorted(top_dirs)}")
