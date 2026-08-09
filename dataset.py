"""
dataset.py — Paired LR/HR grayscale restoration dataset.

Key correctness requirement: LR images may legitimately contain pixel values
outside [0, 1] (negative values from additive Gaussian noise, values > 1 from
multiplicative speckle noise). We NEVER clip these during loading — clipping
would destroy exactly the information the model needs to invert the noise.

Key throughput requirement: DataLoader uses pin_memory + persistent_workers +
prefetch_factor so the GPU is never starved waiting on CPU-side decode/augment.
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


def load_array(path: str) -> np.ndarray:
    """
    Load a single-channel image/array as float32, WITHOUT clipping.

    - .npy files are trusted as-is (the natural format for storing raw,
      unbounded degraded arrays — GT is typically saved this way too, or as
      a standard 8/16-bit image since GT is always in [0, 1]).
    - Standard image formats are read via cv2.IMREAD_UNCHANGED so we don't
      silently reinterpret bit depth, then normalized by their native range.
      NOTE: 8/16-bit integer image formats physically cannot store negative
      values, so if your LR degradation pipeline produces true out-of-range
      values, store LR as .npy. This loader supports both so either pipeline
      works without code changes.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
    else:
        raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        if raw.ndim == 3:  # collapse any accidental 3-channel read to grayscale
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        if raw.dtype == np.uint8:
            arr = raw.astype(np.float32) / 255.0
        elif raw.dtype == np.uint16:
            arr = raw.astype(np.float32) / 65535.0
        else:
            arr = raw.astype(np.float32)  # already float — trust native range
    return arr  # <-- no np.clip() here, intentionally


class RestorationDataset(Dataset):
    """
    Pairs LR/HR files by matching filename stem. Applies:
      1. A custom paired random crop (crop coords scaled by `scale` so the
         LR patch and HR patch depict exactly the same scene region).
      2. Identical flip/90-rotation to both images via Albumentations'
         ReplayCompose — this lets us reuse the exact same transform
         parameters across two images of *different* resolution (LR is
         `scale`x smaller than HR), which a normal Compose can't do safely.
    """

    def __init__(self, lr_dir, hr_dir, patch_size=128, scale=4, augment=True,
                 exts=(".png", ".npy", ".tif", ".tiff")):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir
        self.patch_size = patch_size          # HR patch size in pixels
        self.scale = scale                     # must match your degradation's downsample factor
        self.augment = augment
        assert patch_size % 8 == 0, "patch_size must be a multiple of 8 (3 U-Net downsample stages)"

        hr_files = sorted(f for f in os.listdir(hr_dir) if f.lower().endswith(exts))
        self.pairs = []
        for f in hr_files:
            stem = os.path.splitext(f)[0]
            lr_path = next(
                (os.path.join(lr_dir, stem + e) for e in exts
                 if os.path.exists(os.path.join(lr_dir, stem + e))),
                None,
            )
            if lr_path:
                self.pairs.append((lr_path, os.path.join(hr_dir, f)))
        if not self.pairs:
            raise RuntimeError(f"No matching LR/HR pairs found between {lr_dir} and {hr_dir}")

        # Geometry-only ops: safe to apply to unbounded LR values because they
        # never touch pixel *magnitudes*, only spatial layout.
        self.geo_transform = A.ReplayCompose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
        ])

    def __len__(self):
        return len(self.pairs)

    def _paired_random_crop(self, lr, hr):
        lr_ps = self.patch_size // self.scale
        h_lr, w_lr = lr.shape[:2]

        if h_lr < lr_ps or w_lr < lr_ps:  # pad tiny samples up to the required patch
            pad_h, pad_w = max(0, lr_ps - h_lr), max(0, lr_ps - w_lr)
            lr = np.pad(lr, ((0, pad_h), (0, pad_w)), mode="reflect")
            hr = np.pad(hr, ((0, pad_h * self.scale), (0, pad_w * self.scale)), mode="reflect")
            h_lr, w_lr = lr.shape[:2]

        top = np.random.randint(0, h_lr - lr_ps + 1)
        left = np.random.randint(0, w_lr - lr_ps + 1)
        lr_patch = lr[top:top + lr_ps, left:left + lr_ps]
        hr_patch = hr[top * self.scale: top * self.scale + self.patch_size,
                       left * self.scale: left * self.scale + self.patch_size]
        return lr_patch, hr_patch

    def __getitem__(self, idx):
        lr_path, hr_path = self.pairs[idx]
        lr = load_array(lr_path)
        hr = load_array(hr_path)

        if self.augment:
            lr, hr = self._paired_random_crop(lr, hr)
            replayed = self.geo_transform(image=hr)
            hr = replayed["image"]
            lr = A.ReplayCompose.replay(replayed["replay"], image=lr)["image"]

        lr_t = torch.from_numpy(np.ascontiguousarray(lr)).unsqueeze(0).float()
        hr_t = torch.from_numpy(np.ascontiguousarray(hr)).unsqueeze(0).float()
        return lr_t, hr_t


def build_dataloaders(train_lr, train_hr, val_lr, val_hr, patch_size=128, scale=4,
                       batch_size=16, num_workers=8):
    train_ds = RestorationDataset(train_lr, train_hr, patch_size, scale, augment=True)
    val_ds = RestorationDataset(val_lr, val_hr, patch_size, scale, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,  # keeps workers ahead of the GPU
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=max(2, num_workers // 2), pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
