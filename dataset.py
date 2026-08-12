"""
dataset.py
SEMDataset: loads single-channel grayscale SEM imagery stored as .npy
arrays, for paired (explicit LR/HR directories) or unpaired/synthetic
(LR generated on-the-fly from HR via antialiased bicubic downsampling +
Gaussian noise) super-resolution / denoising training.
"""

import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _list_npy(directory):
    return sorted(f for f in os.listdir(directory) if f.lower().endswith(".npy"))


def _load_npy_as_float01(path):
    """Loads a .npy array and normalizes it to a float32 [0, 1] range.
    Accepts uint8 (0-255) or already-float arrays (assumed 0-255 range
    unless already within [0, 1])."""
    arr = np.load(path)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3:
        # collapse an incidental trailing/leading channel dim to grayscale
        arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
    return arr  # (H, W) float32


class SEMDataset(Dataset):
    """
    Args:
        hr_dir (str): directory of high-resolution ground-truth .npy files.
        lr_dir (str or None): directory of paired low-resolution .npy
            files. If None, LR samples are synthesized on-the-fly from HR
            via `scale`x antialiased bicubic downsampling plus additive
            Gaussian noise.
        patch_size (int): HR crop size used during training (default 128).
        scale (int): super-resolution factor between LR and HR (default 2).
        train (bool): if True, applies random crop + hflip/vflip/90-degree
            rotation augmentations; if False, returns full, unaugmented
            images (used for validation).
        noise_sigma_range (tuple): min/max standard deviation (in [0, 1]
            normalized units) of synthetic Gaussian noise added when
            synthesizing LR samples.
    """

    def __init__(
        self,
        hr_dir,
        lr_dir=None,
        patch_size=128,
        scale=2,
        train=True,
        noise_sigma_range=(0.01, 0.05),
    ):
        super().__init__()
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale = scale
        self.train = train
        self.noise_sigma_range = noise_sigma_range
        self.paired = lr_dir is not None

        self.hr_files = _list_npy(hr_dir)
        if len(self.hr_files) == 0:
            raise FileNotFoundError(f"No .npy files found in hr_dir={hr_dir}")

        if self.paired:
            self.lr_files = _list_npy(lr_dir)
            if len(self.lr_files) != len(self.hr_files):
                raise ValueError(
                    "Paired mode requires hr_dir and lr_dir to contain the "
                    f"same number of files (got {len(self.hr_files)} HR, "
                    f"{len(self.lr_files)} LR)."
                )

    def __len__(self):
        return len(self.hr_files)

    def _synthesize_lr(self, hr_patch_t):
        # hr_patch_t: (1, 1, H, W) float32 tensor
        lr = F.interpolate(
            hr_patch_t,
            scale_factor=1.0 / self.scale,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        sigma = random.uniform(*self.noise_sigma_range)
        noise = torch.randn_like(lr) * sigma
        lr = (lr + noise).clamp(0.0, 1.0)
        return lr

    def _augment(self, lr, hr):
        # lr, hr: (1, H, W) tensors, spatially matched (up to `scale`)
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[-1])
            hr = torch.flip(hr, dims=[-1])
        if random.random() < 0.5:
            lr = torch.flip(lr, dims=[-2])
            hr = torch.flip(hr, dims=[-2])
        k = random.randint(0, 3)
        if k:
            lr = torch.rot90(lr, k, dims=[-2, -1])
            hr = torch.rot90(hr, k, dims=[-2, -1])
        return lr, hr

    def _random_crop_paired(self, lr_arr, hr_arr):
        # hr_arr is assumed to be exactly `scale`x the spatial size of lr_arr
        lh, lw = lr_arr.shape
        lp = self.patch_size // self.scale
        if lh < lp or lw < lp:
            raise ValueError(
                f"LR image ({lh}x{lw}) is smaller than the required LR "
                f"patch ({lp}x{lp}); reduce --patch_size."
            )
        top = random.randint(0, lh - lp)
        left = random.randint(0, lw - lp)
        lr_crop = lr_arr[top:top + lp, left:left + lp]
        hr_crop = hr_arr[
            top * self.scale: (top + lp) * self.scale,
            left * self.scale: (left + lp) * self.scale,
        ]
        return lr_crop, hr_crop

    def _random_crop_single(self, hr_arr):
        h, w = hr_arr.shape
        ps = self.patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            hr_arr = np.pad(hr_arr, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = hr_arr.shape
        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        return hr_arr[top:top + ps, left:left + ps]

    def __getitem__(self, idx):
        hr_arr = _load_npy_as_float01(os.path.join(self.hr_dir, self.hr_files[idx]))

        if self.paired:
            lr_arr = _load_npy_as_float01(os.path.join(self.lr_dir, self.lr_files[idx]))
            if self.train:
                lr_arr, hr_arr = self._random_crop_paired(lr_arr, hr_arr)
            hr_t = torch.from_numpy(np.ascontiguousarray(hr_arr)).unsqueeze(0).float()
            lr_t = torch.from_numpy(np.ascontiguousarray(lr_arr)).unsqueeze(0).float()
        else:
            if self.train:
                hr_arr = self._random_crop_single(hr_arr)
            hr_t = torch.from_numpy(np.ascontiguousarray(hr_arr)).unsqueeze(0).float()
            lr_t = self._synthesize_lr(hr_t.unsqueeze(0)).squeeze(0)

        if self.train:
            lr_t, hr_t = self._augment(lr_t, hr_t)

        return lr_t.contiguous(), hr_t.contiguous()
