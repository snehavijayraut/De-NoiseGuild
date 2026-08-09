# Lightweight Restormer — Degraded Image Restoration (SEMICON KLA Track)

Single-channel restoration of images degraded by (order-unknown) multiplicative
speckle noise, additive Gaussian noise, and downsampling.

## Files

| File | Purpose |
|---|---|
| `dataset.py` | Paired LR/HR loading with **no clipping** on LR values; geometry-only augmentation applied identically to both resolutions via Albumentations `ReplayCompose`. |
| `model.py` | Lightweight Restormer: bicubic pre-upsample → MDTA/GDFN U-Net refinement → global residual → clamp to `[0,1]`. |
| `losses.py` | `0.7·L1 + 0.2·SSIM + 0.1·FFT-magnitude` composite loss. |
| `train.py` | AMP + AdamW + CosineAnnealingLR training loop with best/last checkpointing. |
| `infer.py` | Batched, `torch.compile`-accelerated inference script for the throughput benchmark. |

## Why this architecture

Restormer natively operates at a **fixed** spatial resolution — it's a
denoising/deblurring transformer, not a super-resolution network. Since this
task requires LR → HR upscaling, the model first bicubic-upsamples the LR
input to HR resolution (this is where the "downsampling" degradation gets
inverted), then runs a lightweight Restormer U-Net as a **residual refiner**
on top of that baseline to remove the speckle/Gaussian noise and recover
high-frequency detail. This keeps `in_channels=1`/`out_channels=1` and the
requested MDTA/GDFN blocks architecturally faithful to the paper while still
solving all three degradations end-to-end.

**Important:** `--scale` in `train.py`/`infer.py` must match the actual
downsampling factor used to generate your LR set.

## Quickstart

```bash
pip install -r requirements.txt

python train.py \
  --train_lr data/train/LR --train_hr data/train/HR \
  --val_lr   data/val/LR   --val_hr   data/val/HR \
  --scale 4 --patch_size 128 --batch_size 16 --epochs 100

python infer.py \
  --input_dir data/test/LR --output_dir results/ \
  --weights checkpoints/best_weights_only.pth --scale 4
```

## Notes on LR value ranges

- GT/HR images are assumed normalized to `[0,1]`.
- LR images may exceed `[0,1]` (raw noise physics) — `dataset.load_array`
  performs **zero clipping**. If your LR degradation pipeline produces true
  out-of-range floats, save LR as `.npy`; standard 8/16-bit image formats
  cannot represent negative values.
- Only the final model *output* is clamped to `[0,1]`, since that's what GT
  and the scoring metrics (L1/SSIM/LPIPS) expect.

## Reproducibility

`train.py --seed 42` seeds `random`, `numpy`, and `torch` (CPU + all CUDA
devices). Checkpoints store the full optimizer/scheduler/scaler state plus
the CLI args used, so any run can be exactly resumed via `--resume`.
