# SEM Image Super-Resolution & Denoising (NAFNetSR)

A lightweight NAFNet-based PyTorch pipeline for 2x super-resolution and
denoising of single-channel grayscale SEM (Scanning Electron Microscope)
metrology imagery. Built and tuned for Google Colab's T4 GPU environment,
with automatic mixed-precision selection for both Turing (T4) and Ampere+
GPUs.

## Repository Layout

```
├── dataset.py       # SEMDataset: paired / synthetic LR-HR loading + augmentation
├── infer.py         # 8-fold TTA inference -> predictions/ + submission.zip
├── losses.py        # CharbonnierLoss, FFTLoss, SobelGradientLoss, KLAPhysicsLoss
├── model.py          # NAFNetSR architecture (NAFBlock / SimpleGate based)
├── README.md         # This file
├── requirements.txt  # Python dependencies
└── train.py          # Training driver (baseline + finetune modes)
```

## Architecture Summary

`NAFNetSR` is a U-Net-style encoder/bottleneck/decoder built from
NAFBlocks (Non-Linear Activation Free Blocks — SimpleGate + simplified
channel attention, no GELU/ReLU):

- **Encoder depths**: `(1, 2, 2, 4)` NAFBlocks at channel widths
  `(32, 64, 128, 256)`, with strided-convolution downsampling between
  stages.
- **Bottleneck**: the depth-4 NAFBlock stack at 256 channels.
- **Decoder**: mirrors the encoder with `ConvTranspose2d` upsampling and
  additive skip connections.
- **Reconstruction head**: `PixelShuffle(scale=2)` upsampler for the 2x
  super-resolution output.
- **Residual shortcut**: a bilinear-upsampled copy of the input is added
  directly to the reconstructed output, enforcing shortcut residual
  learning and suppressing line/space boundary haloing and ringing
  artifacts.

## Environment Setup (Google Colab, T4 GPU)

```bash
!pip install -r requirements.txt
```

Precision is auto-selected at runtime by `get_amp_settings()` /
`get_amp_dtype()` based on `torch.cuda.get_device_capability()`:

| GPU compute capability | Autocast dtype | GradScaler |
|---|---|---|
| >= 8.0 (Ampere, e.g. A100/L4) | `bfloat16` | not used |
| 7.5 (Turing T4) | `float16` | `torch.cuda.amp.GradScaler` |

## Data Layout

`.npy` files, single-channel grayscale, either `uint8` (0-255) or float
(0-255 or already 0-1) — `SEMDataset` normalizes automatically.

- **Paired mode**: pass both `--hr_dir` and `--lr_dir` with matching,
  pixel-aligned file counts; `hr_dir` images must be exactly `--scale`
  times the spatial resolution of the corresponding `lr_dir` images.
- **Synthetic mode**: pass only `--hr_dir`. LR inputs are synthesized
  on-the-fly via `--scale`x antialiased bicubic downsampling plus
  additive Gaussian noise.

Training augmentations (synthetic and paired): random `--patch_size`
crop (default `128x128`, HR-space), random horizontal flip, random
vertical flip, random 90-degree rotation.

## Phase 1 — Baseline Training

40 epochs, `lr=2e-4`, plain L1 pixel loss:

```bash
python train.py \
  --mode baseline \
  --hr_dir /content/data/train_hr \
  --val_hr_dir /content/data/val_hr \
  --patch_size 128 \
  --batch_size 16 \
  --checkpoint_dir /content/checkpoints
```

Add `--lr_dir /content/data/train_lr` (and `--val_lr_dir`) if you have
paired real LR/HR SEM captures instead of relying on synthetic
degradation.

This saves the best (by validation PSNR, evaluated on EMA shadow
weights) checkpoint to `/content/checkpoints/best_ema_weights.pth`.

## Phase 2 — Physics-Informed Fine-Tuning

15 epochs, `lr=2e-5`, `KLAPhysicsLoss` (Charbonnier + spectral FFT +
Sobel-gradient terms), resuming from the Phase 1 checkpoint:

```bash
python train.py \
  --mode finetune \
  --hr_dir /content/data/train_hr \
  --val_hr_dir /content/data/val_hr \
  --resume /content/checkpoints/best_ema_weights.pth \
  --patch_size 128 \
  --batch_size 16 \
  --checkpoint_dir /content/checkpoints
```

`KLAPhysicsLoss` combines:

- Charbonnier pixel loss (`eps=1e-3`, weight `1.0`)
- Real-FFT spectral loss (`torch.fft.rfft2`, `norm="ortho"`, weight `0.02`)
- Sobel gradient loss (`Gx` + `Gy` L1, weight `0.6`)

Per-component loss values are logged every `--log_interval` iterations.

## Inference (8-fold TTA)

```bash
python infer.py \
  --checkpoint /content/checkpoints/best_ema_weights.pth \
  --input_dir /content/data/test_lr \
  --output_dir /content/predictions \
  --zip_path /content/submission.zip
```

Runs 4 rotations x 2 horizontal-flip states (8 total forward passes),
exactly inverse-transforms each prediction, and averages them in mixed
precision. Outputs are clipped to `[0.0, 1.0]`, written as
`<name>_pred.npy` under `--output_dir`, and zipped to `--zip_path`.

## Key CLI Flags (`train.py`)

| Flag | Default | Notes |
|---|---|---|
| `--mode` | required | `baseline` or `finetune` |
| `--hr_dir` | required | training HR `.npy` directory |
| `--lr_dir` | `None` | paired LR dir; omit for synthetic LR |
| `--val_hr_dir` / `--val_lr_dir` | `None` | validation set (enables PSNR/SSIM tracking + checkpointing) |
| `--patch_size` | `128` | HR-space training crop size |
| `--scale` | `2` | super-resolution factor |
| `--batch_size` | `16` | |
| `--epochs` / `--lr` | mode default | override the `baseline`/`finetune` defaults |
| `--resume` | `None` | required for `--mode finetune` |
| `--ema_decay` | `0.999` | |
| `--checkpoint_dir` | `/content/checkpoints` | |

## Notes

- `AdamW` optimizer (`betas=(0.9, 0.999)`, `weight_decay=1e-4`) with
  `CosineAnnealingLR` (`eta_min=1e-6`), stepped every iteration.
- EMA (`decay=0.999`) is updated every optimizer step; validation and the
  final saved checkpoint both use the EMA shadow weights.
- PSNR and SSIM are computed with dependency-free, pure-PyTorch
  implementations in `train.py` (no `skimage` / `pytorch-msssim` needed).
