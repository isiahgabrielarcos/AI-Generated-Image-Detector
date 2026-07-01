"""
dataset.py
──────────
Dataset classes for DFDC, ForenSynths, and GenImage.

Each dataset folder is expected to have the structure:

  <root>/
    real/      <- real images (label 0)
    fake/      <- AI-generated images (label 1)

Augmentation pipeline follows paper §4.2:
  - Random horizontal flip
  - Random crop
  - JPEG compression (quality 70-100, random)
  - Resize to 224x224
  - CLIP-compatible normalisation

CLIP feature cache
──────────────────
Run cache_clip_features.py once to pre-compute frozen CLIP patch tokens
for every image.  When the cache directory is configured, ImageFolderBinary
returns a 3-tuple (image, clip_tokens, label) instead of the usual 2-tuple.
The frozen ViT is then bypassed each training step, giving a large speedup.
"""

import io
import random
from pathlib import Path
from typing import List, Tuple, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as T


# ──────────────────────────────────────────────────────────────────────
#  JPEG augmentation
# ──────────────────────────────────────────────────────────────────────

class RandomJPEGCompression:
    """Randomly compress an image with JPEG at quality q in [min_q, 100]."""

    def __init__(self, min_quality: int = 70):
        self.min_quality = min_quality

    def __call__(self, img: Image.Image) -> Image.Image:
        quality = random.randint(self.min_quality, 100)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


# ──────────────────────────────────────────────────────────────────────
#  Transform factories
# ──────────────────────────────────────────────────────────────────────

# CLIP canonical normalisation
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)


def build_transforms(
    image_size: int = 224,
    augment: bool = True,
    jpeg_compress: bool = True,
) -> T.Compose:
    ops = []

    if augment:
        ops += [
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.3),
        ]
        if jpeg_compress:
            ops.append(RandomJPEGCompression(min_quality=50))
    else:
        ops += [
            T.Resize(image_size),
            T.CenterCrop(image_size),
        ]

    ops += [
        T.ToTensor(),
        T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ]
    return T.Compose(ops)


# ──────────────────────────────────────────────────────────────────────
#  CLIP feature cache
# ──────────────────────────────────────────────────────────────────────

class ClipFeatureCache:
    """
    Loads pre-computed CLIP ViT-L/14 patch tokens from disk and serves
    them by image path.

    Cache files are created by cache_clip_features.py.
    Each file contains:
        "paths"    : list[str]       – absolute image paths
        "features" : Tensor[N, 256, 1024] float16

    Memory: ~5.2 GB float16 for 10 k images; ~15.7 GB for 30 k.
    All entries are kept in RAM for fast per-batch lookup.
    """

    def __init__(self, cache_dir: str | Path, dataset_names: list):
        self._cache: dict[str, torch.Tensor] = {}  # path -> [256, 1024] float16
        cache_dir = Path(cache_dir)
        total = 0

        for name in dataset_names:
            cache_path = cache_dir / f"{name}_clip.pt"
            if not cache_path.exists():
                print(f"  [ClipCache] '{name}': cache file not found "
                      f"(run cache_clip_features.py first)")
                continue

            size_gb = cache_path.stat().st_size / 1e9
            print(f"  [ClipCache] loading '{name}' ({size_gb:.1f} GB) ...",
                  end="", flush=True)
            data = torch.load(cache_path, map_location="cpu", weights_only=True)
            paths: list = data["paths"]
            feats: torch.Tensor = data["features"]  # [N, 256, 1024] float16
            for p, f in zip(paths, feats):
                self._cache[str(Path(p).resolve())] = f  # normalize to absolute path
            total += len(paths)
            print(f" {len(paths)} entries")

        print(f"  [ClipCache] ready – {total} total entries in RAM")

    def get(self, path: str | Path) -> torch.Tensor | None:
        """Return [256, 1024] float32 tensor, or None if path not cached."""
        t = self._cache.get(str(Path(path).resolve()))
        return t.float() if t is not None else None   # float16 -> float32

    def __len__(self) -> int:
        return len(self._cache)

    @property
    def available(self) -> bool:
        return len(self._cache) > 0


# ──────────────────────────────────────────────────────────────────────
#  Image collection helper
# ──────────────────────────────────────────────────────────────────────

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def _collect_images(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    return sorted(
        p for p in folder.rglob("*") if p.suffix.lower() in _IMG_EXTS
    )


# ──────────────────────────────────────────────────────────────────────
#  Base dataset
# ──────────────────────────────────────────────────────────────────────

class ImageFolderBinary(Dataset):
    """
    Loads images from a folder with `real/` and `fake/` sub-directories.

    label 0 = Real
    label 1 = AI-Generated

    Return signature depends on whether a ClipFeatureCache is attached:
        cache present  ->  (image_tensor, clip_tokens [256, 1024], label)
        no cache       ->  (image_tensor, label)

    The DataLoader collate_fn must match – use clip_collate_fn() when
    a cache is in use.
    """

    def __init__(
        self,
        root: str,
        transform: T.Compose,
        max_samples: Optional[int] = None,
        clip_cache: Optional[ClipFeatureCache] = None,
    ):
        root = Path(root)

        def _find(base: Path, names: tuple) -> Path:
            for n in names:
                d = base / n
                if d.is_dir():
                    return d
            raise FileNotFoundError(
                f"Expected one of {names} under {base}")

        real_dir = _find(root, ("real", "real-625", "real-5000"))
        fake_dir = _find(root, ("fake", "fake-625", "fake-5000"))

        real_paths = _collect_images(real_dir)
        fake_paths = _collect_images(fake_dir)

        # Cap per-class if requested (preserves balance)
        if max_samples is not None:
            n = max_samples // 2
            real_paths = real_paths[:n]
            fake_paths = fake_paths[:n]

        self.samples: List[Tuple[Path, int]] = (
            [(p, 0) for p in real_paths] + [(p, 1) for p in fake_paths]
        )
        self.transform = transform
        self.clip_cache = clip_cache

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        label_t = torch.tensor(label, dtype=torch.float32)

        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        img_t = self.transform(img)

        # Return cached CLIP tokens if available
        if self.clip_cache is not None:
            clip_tok = self.clip_cache.get(path)
            if clip_tok is not None:
                return img_t, clip_tok, label_t
            # Fallback: zeros (shouldn't happen with a complete cache)
            return img_t, torch.zeros(256, 1024), label_t

        return img_t, label_t


# ──────────────────────────────────────────────────────────────────────
#  Collate helpers
# ──────────────────────────────────────────────────────────────────────

def clip_collate_fn(batch):
    """Collate 3-tuples: (img, clip_tokens, label) -> stacked tensors."""
    imgs, clips, labels = zip(*batch)
    return torch.stack(imgs), torch.stack(clips), torch.stack(labels)


# ──────────────────────────────────────────────────────────────────────
#  Named dataset builders
# ──────────────────────────────────────────────────────────────────────

def build_dfdc_dataset(
    root: str,
    split: str = "train",
    cfg: dict = None,
    clip_cache: ClipFeatureCache = None,
) -> Dataset:
    augment = (split == "train")
    transform = build_transforms(
        image_size=224,
        augment=augment,
        jpeg_compress=augment and cfg.get("data", {}).get(
            "augmentation", {}).get("jpeg_compression", True),
    )
    max_s = cfg.get("data", {}).get("max_samples_per_dataset", None) if cfg else None
    return ImageFolderBinary(root, transform, max_samples=max_s, clip_cache=clip_cache)


def build_forensynths_dataset(
    root: str,
    split: str = "train",
    cfg: dict = None,
    clip_cache: ClipFeatureCache = None,
) -> Dataset:
    augment = (split == "train")
    transform = build_transforms(image_size=224, augment=augment)
    max_s = cfg.get("data", {}).get("max_samples_per_dataset", None) if cfg else None
    return ImageFolderBinary(root, transform, max_samples=max_s, clip_cache=clip_cache)


def build_genimage_dataset(
    root: str,
    split: str = "train",
    cfg: dict = None,
    clip_cache: ClipFeatureCache = None,
) -> Dataset:
    augment = (split == "train")
    transform = build_transforms(image_size=224, augment=augment)
    max_s = cfg.get("data", {}).get("max_samples_per_dataset", None) if cfg else None
    return ImageFolderBinary(root, transform, max_samples=max_s, clip_cache=clip_cache)


# ──────────────────────────────────────────────────────────────────────
#  DataLoader builder  (called from train.py / evaluate.py)
# ──────────────────────────────────────────────────────────────────────

def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders from config.
    Combines DFDC + ForenSynths + GenImage automatically
    (skips any dataset whose root path doesn't exist).

    Returns:
        (train_loader, val_loader)
    """
    data_cfg    = cfg.get("data", {})
    train_split = data_cfg.get("train_split", 0.8)
    batch_size  = cfg.get("training", {}).get("batch_size", 8)
    num_workers = data_cfg.get("num_workers", 0)

    roots = {
        "dfdc":        data_cfg.get("dfdc_root", ""),
        "forensynths": data_cfg.get("forensynths_root", ""),
        "genimage":    data_cfg.get("genimage_root", ""),
    }
    builders = {
        "dfdc":        build_dfdc_dataset,
        "forensynths": build_forensynths_dataset,
        "genimage":    build_genimage_dataset,
    }

    # ── Load CLIP feature cache if configured ──────────────────────
    cache_dir = data_cfg.get("clip_cache_dir", None)
    clip_cache = None
    if cache_dir and Path(cache_dir).exists():
        # Only load caches for datasets whose root actually exists — datasets
        # with a missing root are skipped below, so loading their (multi-GB)
        # cache would waste RAM.  Pointing a root at a non-existent path is the
        # supported way to train on a subset and shrink the memory footprint.
        existing = [
            name for name in roots
            if roots[name] and Path(roots[name]).exists()
            and (Path(cache_dir) / f"{name}_clip.pt").exists()
        ]
        if existing:
            clip_cache = ClipFeatureCache(cache_dir, existing)

    use_cache = clip_cache is not None and clip_cache.available
    collate   = clip_collate_fn if use_cache else None

    # num_workers > 0 + cache in RAM = each worker spawns its own copy
    # of the full cache dict (Windows multiprocessing uses spawn, not fork).
    # Force 0 when cache is loaded to avoid multiplying RAM usage.
    if use_cache and num_workers > 0:
        print("  [dataset] CLIP cache active: forcing num_workers=0 "
              "(prevents cache duplication across worker processes)")
        num_workers = 0

    # ── Build each dataset ─────────────────────────────────────────
    train_datasets, val_datasets = [], []

    for name, root in roots.items():
        if not root or not Path(root).exists():
            print(f"  [dataset] {name}: path not found, skipping.")
            continue
        print(f"  [dataset] {name}: loading from {root}")

        # Probe for split sizing (eval transform, no augmentation)
        probe_ds = builders[name](root, split="eval", cfg=cfg, clip_cache=None)
        n        = len(probe_ds)
        n_train  = int(n * train_split)

        indices = torch.randperm(
            n, generator=torch.Generator().manual_seed(42)
        ).tolist()
        train_indices = indices[:n_train]
        val_indices   = indices[n_train:]

        # Build separate datasets so train gets augmentation, val gets eval transforms
        train_ds_full = builders[name](root, split="train", cfg=cfg, clip_cache=clip_cache)
        val_ds_full   = builders[name](root, split="eval",  cfg=cfg, clip_cache=clip_cache)

        train_datasets.append(torch.utils.data.Subset(train_ds_full, train_indices))
        val_datasets.append(torch.utils.data.Subset(val_ds_full,   val_indices))

    if not train_datasets:
        raise RuntimeError(
            "No datasets found. Check dataset paths in configs/default.yaml."
        )

    combined_train = ConcatDataset(train_datasets)
    combined_val   = ConcatDataset(val_datasets)

    train_loader = DataLoader(
        combined_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,       # pin_memory only benefits CUDA transfers
        drop_last=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        combined_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate,
    )

    cache_tag = f" (CLIP cache: {len(clip_cache)} entries)" if use_cache else ""
    print(f"  [dataset] train={len(combined_train)}  val={len(combined_val)}{cache_tag}")
    return train_loader, val_loader
