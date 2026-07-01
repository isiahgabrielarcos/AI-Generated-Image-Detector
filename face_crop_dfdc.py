"""
face_crop_dfdc.py
─────────────────
Fix the DFDC cross-generator distribution mismatch (Option A).

Problem: the model was trained on tight DFDC FACE CROPS (299x299), but the
per-gen DFDC test set contains FULL VIDEO FRAMES (1080x1920) where the face is
~5% of the image. After resize to 224, the deepfake artifacts vanish, so DFDC
cross-gen sits at chance (~55%).

Fix: preprocess the per-gen DFDC frames the SAME way as training — detect the
face and crop it — so the test distribution matches training. The test images
stay HELD-OUT (they are different DFDC videos than training); only the
preprocessing is made consistent. This is standard practice, NOT leakage.

Detector: OpenCV Haar cascade (bundled with cv2, offline). For each frame it
takes the largest detected face, expands the box by a margin to include some
context (like the training crops), squares it, crops, and resizes.

Non-destructive: writes to a NEW folder; the original full-frame folder is kept.

Usage:
    python face_crop_dfdc.py
    python face_crop_dfdc.py --src per-gen-dataset/DFDC --out per-gen-dataset/DFDC_facecrop
    python face_crop_dfdc.py --margin 0.4 --size 256
"""

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUBDIRS = ["1k_real", "1k_fake", "real", "fake"]   # whichever exist

# Multiple bundled cascades catch more faces (frontal + profile) than one alone.
CASCADE_FILES = [
    "haarcascade_frontalface_default.xml",
    "haarcascade_frontalface_alt2.xml",
    "haarcascade_profileface.xml",
]


YUNET_DEFAULT = "models_aux/face_detection_yunet_2023mar.onnx"


def load_cascades():
    cs = []
    for f in CASCADE_FILES:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + f)
        if not c.empty():
            cs.append((f, c))
    return cs


def yunet_largest(detector, img_bgr, min_face):
    """YuNet DNN detection. Returns the largest face box above the score
    threshold (set at create time), or None. Far fewer false positives than Haar."""
    H, W = img_bgr.shape[:2]
    detector.setInputSize((W, H))
    _, faces = detector.detect(img_bgr)
    if faces is None or len(faces) == 0:
        return None
    boxes = []
    for f in faces:
        x, y, w, h = int(f[0]), int(f[1]), int(f[2]), int(f[3])
        if w >= min_face and h >= min_face:
            boxes.append((max(0, x), max(0, y), w, h))
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])


def largest_face(gray, cascades, min_face):
    """Detect with several cascades (incl. flipped profile) and return the largest box."""
    boxes = []
    for name, c in cascades:
        for b in c.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6,
                                    minSize=(min_face, min_face)):
            boxes.append(tuple(int(v) for v in b))
        if "profile" in name:
            # profileface only finds left-facing profiles; flip to catch right-facing
            flip = cv2.flip(gray, 1)
            Wg = gray.shape[1]
            for (x, y, w, h) in c.detectMultiScale(flip, scaleFactor=1.1, minNeighbors=6,
                                                   minSize=(min_face, min_face)):
                boxes.append((int(Wg - x - w), int(y), int(w), int(h)))
    if not boxes:
        # looser retry on the primary frontal cascade for hard frames
        for b in cascades[0][1].detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4,
                                                 minSize=(min_face // 2, min_face // 2)):
            boxes.append(tuple(int(v) for v in b))
    if not boxes:
        return None
    return max(boxes, key=lambda b: b[2] * b[3])   # (x, y, w, h)


def square_crop_box(x, y, w, h, margin, W, H):
    """Expand the face box by `margin`, make it square, clip to image."""
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * margin)
    half = side / 2.0
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    # clip
    x0 = max(0, x0); y0 = max(0, y0); x1 = min(W, x1); y1 = min(H, y1)
    return x0, y0, x1, y1


def center_square(W, H):
    s = min(W, H)
    x0 = (W - s) // 2; y0 = (H - s) // 2
    return x0, y0, x0 + s, y0 + s


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="per-gen-dataset/DFDC",
                    help="source folder with full-frame DFDC images")
    ap.add_argument("--out", default="per-gen-dataset/DFDC_facecrop",
                    help="destination folder for face crops (non-destructive)")
    ap.add_argument("--margin", type=float, default=0.35,
                    help="fraction of face size added as context on each side")
    ap.add_argument("--size", type=int, default=256,
                    help="output square size (cache later resizes to 224)")
    ap.add_argument("--min_face", type=int, default=60,
                    help="minimum face size in pixels to accept a detection")
    ap.add_argument("--skip_noface", action="store_true",
                    help="skip images with no detected face instead of center-cropping")
    ap.add_argument("--detector", default="yunet", choices=["yunet", "haar"],
                    help="face detector: yunet (DNN, accurate, default) or haar (cascades)")
    ap.add_argument("--model", default=YUNET_DEFAULT,
                    help="path to the YuNet ONNX model")
    ap.add_argument("--score", type=float, default=0.7,
                    help="YuNet confidence threshold (higher = stricter, fewer false positives)")
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out)
    yunet = None; cascades = None
    if args.detector == "yunet":
        if not Path(args.model).exists():
            print(f"[error] YuNet model not found at {args.model}. "
                  f"Download it or use --detector haar.")
            return
        yunet = cv2.FaceDetectorYN.create(args.model, "", (320, 320),
                                          score_threshold=args.score)
        print(f"[facecrop] detector=YuNet (DNN)  score_threshold={args.score}")
    else:
        cascades = load_cascades()
        if not cascades:
            print("[error] could not load any Haar cascade")
            return
        print(f"[facecrop] detector=Haar  cascades={[c[0] for c in cascades]}")

    subdirs = [d for d in SUBDIRS if (src / d).is_dir()]
    if not subdirs:
        print(f"[error] no class subfolders {SUBDIRS} found under {src}")
        return

    print(f"[facecrop] src={src}  out={out}  margin={args.margin}  size={args.size}")
    grand = {"detected": 0, "fallback": 0, "skipped": 0, "failed": 0}

    for sub in subdirs:
        in_dir = src / sub
        out_dir = out / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        files = [p for p in in_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        stats = {"detected": 0, "fallback": 0, "skipped": 0, "failed": 0}

        for p in tqdm(files, desc=f"  {sub}", unit="img"):
            img = cv2.imread(str(p))
            if img is None:
                stats["failed"] += 1
                continue
            H, W = img.shape[:2]
            if yunet is not None:
                box = yunet_largest(yunet, img, args.min_face)
            else:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                box = largest_face(gray, cascades, args.min_face)

            if box is not None:
                x, y, w, h = box
                x0, y0, x1, y1 = square_crop_box(x, y, w, h, args.margin, W, H)
                stats["detected"] += 1
            else:
                if args.skip_noface:
                    stats["skipped"] += 1
                    continue
                x0, y0, x1, y1 = center_square(W, H)   # best-effort fallback
                stats["fallback"] += 1

            crop = img[y0:y1, x0:x1]
            if crop.size == 0:
                stats["failed"] += 1
                continue
            crop = cv2.resize(crop, (args.size, args.size), interpolation=cv2.INTER_AREA)
            # keep original stem; save as jpg (DFDC is jpg)
            cv2.imwrite(str(out_dir / (p.stem + ".jpg")), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        n = len(files)
        print(f"    {sub}: {n} imgs  ->  detected={stats['detected']}  "
              f"fallback={stats['fallback']}  skipped={stats['skipped']}  failed={stats['failed']}")
        for k in grand:
            grand[k] += stats[k]

    det = grand["detected"]; fb = grand["fallback"]
    total = det + fb + grand["skipped"] + grand["failed"]
    rate = (100.0 * det / total) if total else 0.0
    print(f"\n[facecrop] done. face-detected={det}  fallback(center)={fb}  "
          f"skipped={grand['skipped']}  failed={grand['failed']}")
    print(f"[facecrop] face-detection rate: {rate:.1f}%")
    if fb > 0 and not args.skip_noface:
        print("[facecrop] NOTE: 'fallback' images used a center crop (no face found). "
              "If this count is high, re-run with --margin 0.5 or inspect those frames; "
              "or use --skip_noface to drop them.")
    print("\n[facecrop] Next steps:")
    print(f"  1. Verify no leakage:")
    print(f"       python check_leakage.py --eval-root {out}")
    print(f"  2. Swap folders (keep the original full-frame set as backup):")
    print(f"       rename {src}  ->  per-gen-dataset/DFDC_fullframe")
    print(f"       rename {out}  ->  per-gen-dataset/DFDC")
    print(f"  3. Rebuild ONLY the DFDC cache:")
    print(f"       python cache_clip_features.py --dataset pergen --pergen_only DFDC --overwrite")
    print(f"  4. Re-run per-generator evaluation.")


if __name__ == "__main__":
    main()
