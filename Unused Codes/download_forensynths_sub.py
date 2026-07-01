"""Download a small ForenSynths-substitute dataset and inspect its structure."""
import ssl, os, json, zipfile
from pathlib import Path
from collections import Counter

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

SLUG         = "shanmuk4622/real-and-fake-ai-generated-512px-dataset"
DOWNLOAD_DIR = Path("datasets/_download_tmp/forensynths_sub")
ZIP_PATH     = DOWNLOAD_DIR / "forensynths_sub.zip"
IMG_EXTS     = {".jpg", ".jpeg", ".png", ".webp"}

DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print(f"Downloading {SLUG} (~0.8 GB) ...")
    kaggle_api.dataset_download_files(SLUG, path=str(DOWNLOAD_DIR), unzip=False, quiet=False)
    zips = [p for p in DOWNLOAD_DIR.glob("*.zip") if p != ZIP_PATH]
    if zips:
        zips[0].rename(ZIP_PATH)
    print(f"Saved: {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1024/1024:.0f} MB)")
else:
    print(f"Already downloaded ({ZIP_PATH.stat().st_size/1024/1024:.0f} MB)")

print("\nInspecting structure ...")
with zipfile.ZipFile(ZIP_PATH) as zf:
    all_names = zf.namelist()
    img_names = [n for n in all_names if Path(n).suffix.lower() in IMG_EXTS]
    print(f"Total entries: {len(all_names)}  |  Images: {len(img_names)}")
    print("\nFirst 40 entries:")
    for n in all_names[:40]:
        print(" ", n)
    # Count by folder depth
    lvl2 = Counter()
    for n in img_names:
        parts = Path(n).parts
        key = "/".join(parts[:3]) if len(parts) >= 3 else "/".join(parts)
        lvl2[key] += 1
    print("\nImages per folder (top 30):")
    for folder, count in sorted(lvl2.items())[:30]:
        print(f"  {folder:<70} {count}")

///////////////////////////////////////////////////////////////////////// 

import zipfile
from pathlib import Path
import shutil

ARCHIVE = Path(r'D:\Dataset\Progan\progan_train.zip')
TARGET_DIR = Path('datasets/ForenSynths/fake')
IMG_EXTS = {'.png', '.jpg', '.jpeg', '.webp'}

if TARGET_DIR.exists():
    shutil.rmtree(TARGET_DIR)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(ARCHIVE, 'r') as zf:
    fake_infos = [info for info in zf.infolist()
                  if not info.is_dir()
                  and info.filename.lower().count('/') >= 2
                  and info.filename.split('/')[1] == '1_fake'
                  and Path(info.filename).suffix.lower() in IMG_EXTS]
    print(f'Found {len(fake_infos)} fake image entries in archive.')

    for i, info in enumerate(fake_infos, start=1):
        dest = TARGET_DIR / f'fake_{i:06d}{Path(info.filename).suffix.lower()}'
        with zf.open(info, 'r') as src, open(dest, 'wb') as dst:
            dst.write(src.read())
        if i % 10000 == 0:
            print(f'  Extracted {i}/{len(fake_infos)}')

print(f'Done: extracted {len(list(TARGET_DIR.iterdir()))} fake images to {TARGET_DIR}')

