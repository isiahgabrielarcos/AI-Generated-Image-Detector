"""
preprocess_dfdc_test.py
───────────────────────
Face-crop the DFDC test images to match the training format.

Training DFDC images are 256x256 face crops.
Test DFDC images are 1920x1080 (or 1080x1920) full video frames.
This mismatch causes the model to see CLIP features of the video background
instead of the face, resulting in ~55% random accuracy.

This script:
  1. Detects faces in each test DFDC image using OpenCV Haar cascade
  2. Crops the face region (with padding) and resizes to 256x256
  3. Saves back in-place (replaces the full frames with face crops)
  4. After running this, rebuild the DFDC test cache:
       python cache_clip_features.py --dataset pergen --pergen_only DFDC
           --pergen_root per-gen-dataset-test
           --cache_dir datasets_eq/clip_cache_test
           --overwrite

Usage:
    python preprocess_dfdc_test.py
    python preprocess_dfdc_test.py --dfdc_dir per-gen-dataset-test/DFDC --target_size 256
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_IMG_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
PADDING_RATIO = 0.30   # 30% padding around detected face box


def detect_face(img_bgr, cascade_front, cascade_profile):
    """Return (x, y, w, h) of the largest detected face, or None."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    for detector, scale, neighbors in [
        (cascade_front,   1.10, 5),
        (cascade_front,   1.05, 3),
        (cascade_profile, 1.10, 5),
        (cascade_profile, 1.05, 3),
    ]:
        faces = detector.detectMultiScale(
            gray, scaleFactor=scale, minNeighbors=neighbors,
            minSize=(40, 40),
        )
        if len(faces) > 0:
            return max(faces, key=lambda f: f[2] * f[3])
    return None


def crop_face(img_bgr, face_rect, target_size, padding_ratio):
    """Crop face region with padding, square-expand, resize."""
    h, w = img_bgr.shape[:2]
    x, y, fw, fh = face_rect
    pad = int(max(fw, fh) * padding_ratio)

    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + fw + pad)
    y2 = min(h, y + fh + pad)

    crop = img_bgr[y1:y2, x1:x2]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def fallback_crop(img_bgr, target_size):
    """Smart center crop when no face is detected."""
    h, w = img_bgr.shape[:2]
    if h > w:
        # Portrait: face likely in upper portion — take upper square
        crop_size = w
        y1 = max(0, h // 8)           # start 1/8 from top
        y2 = min(h, y1 + crop_size)
        crop = img_bgr[y1:y2, 0:w]
    else:
        # Landscape: center square
        crop_size = h
        x1 = (w - crop_size) // 2
        crop = img_bgr[0:h, x1:x1 + crop_size]
    return cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_LANCZOS4)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser()
    ap.add_argument('--dfdc_dir',    default='per-gen-dataset-test/DFDC')
    ap.add_argument('--target_size', type=int, default=256)
    args = ap.parse_args()

    dfdc_dir    = Path(args.dfdc_dir)
    target_size = args.target_size

    cascade_front   = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cascade_profile = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_profileface.xml')

    all_imgs = sorted(
        p for sub in ['1k_real', '1k_fake']
        for p in (dfdc_dir / sub).rglob('*')
        if p.suffix.lower() in _IMG_EXTS
    )
    print(f'[preprocess] Found {len(all_imgs)} DFDC test images to process')

    detected = missed = 0
    for i, img_path in enumerate(all_imgs, 1):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f'  WARN: could not read {img_path.name}')
            continue

        face = detect_face(img_bgr, cascade_front, cascade_profile)
        if face is not None:
            cropped = crop_face(img_bgr, face, target_size, PADDING_RATIO)
            detected += 1
        else:
            cropped = fallback_crop(img_bgr, target_size)
            missed += 1
            print(f'  [fallback] {img_path.name} ({img_bgr.shape[1]}x{img_bgr.shape[0]})')

        # Save back in-place (convert BGR→RGB for PIL to save as JPEG properly)
        Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)).save(
            img_path, quality=95)

        if i % 100 == 0:
            print(f'  [{i}/{len(all_imgs)}] detected={detected} fallback={missed}')

    print(f'\n[preprocess] Done. Face detected: {detected}/{len(all_imgs)}, '
          f'fallback: {missed}/{len(all_imgs)}')
    print()
    print('Next step — rebuild the DFDC test CLIP cache:')
    print('  python cache_clip_features.py --dataset pergen --pergen_only DFDC '
          '--pergen_root per-gen-dataset-test '
          '--cache_dir datasets_eq/clip_cache_test --overwrite')


if __name__ == '__main__':
    main()
