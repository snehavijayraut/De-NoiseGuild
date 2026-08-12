"""
train.py
Training driver for NAFNetSR SEM super-resolution / denoising.

Modes:
  --mode baseline : 40 epochs, lr=2e-4, plain L1 pixel loss.
  --mode finetune  : 15 epochs, lr=2e-5, KLAPhysicsLoss, requires --resume
                      to load a baseline pre-trained checkpoint.

Precision:
  Auto-detects GPU compute capability. Ampere+ GPUs (compute capability
  >= 8.0) use bfloat16 autocast (no GradScaler needed, bf16 has fp32-like
  dynamic range). Turing T4 GPUs (compute capability 7.5) use float16
  autocast with torch.cuda.amp.GradScaler for loss/gradient scaling.

Example (Colab T4):
  python train.py --mode baseline --hr_dir /content/data/hr \
      --val_hr_dir /content/data/val_hr --batch_size 16
  python train.py --mode finetune --hr_dir /content/data/hr \
      --val_hr_dir /content/data/val_hr \
      --resume /content/checkpoints/best_ema_weights.pth
"""

import argparse
import copy
import math
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from model import NAFNetSR
from losses import KLAPhysicsLoss
from dataset import SEMDataset


MODE_DEFAULTS = {
    "baseline": {"epochs": 40, "lr": 2e-4},
    "finetune": {"epochs": 15, "lr": 2e-5},
}


# --------------------------------------------------------------------------
# Exponential Moving Average
# --------------------------------------------------------------------------
class EMA:
    """Exponential Moving Average of model parameters/buffers, updated at
    every optimizer step with the given decay."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model.state_dict())
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.shadow.items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def apply_shadow(self, model):
        """Swaps model weights for the EMA shadow weights in-place, first
        backing up the raw (non-EMA) weights so they can be restored."""
        self._backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)

    def restore(self, model):
        assert self._backup is not None, "apply_shadow must be called before restore"
        model.load_state_dict(self._backup)
        self._backup = None

    def state_dict(self):
        return self.shadow


# --------------------------------------------------------------------------
# Precision helpers
# --------------------------------------------------------------------------
def get_amp_settings(device):
    """Auto-detects GPU compute capability and returns
    (autocast_dtype, use_grad_scaler)."""
    if device.type != "cuda":
        return torch.float32, False
    major, minor = torch.cuda.get_device_capability(device)
    cc = major + minor / 10.0
    if cc >= 8.0:
        # Ampere+ : bfloat16 autocast, no GradScaler required.
        return torch.bfloat16, False
    # Turing T4 (7.5) and older : float16 autocast + GradScaler.
    return torch.float16, True


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_psnr(pred, target, max_val=1.0, eps=1e-10):
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3])
    psnr = 10.0 * torch.log10((max_val ** 2) / (mse + eps))
    return psnr.mean().item()


def _gaussian_window(window_size, sigma, device, dtype):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_2d = g.unsqueeze(1) @ g.unsqueeze(0)
    return window_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, ws, ws)


def compute_ssim(pred, target, window_size=11, sigma=1.5, max_val=1.0):
    """Pure PyTorch single-channel SSIM (Wang et al., 2004), gaussian
    window, no external dependencies (skimage/pytorch-msssim not required)."""
    device, dtype = pred.device, pred.dtype
    window = _gaussian_window(window_size, sigma, device, dtype)
    channel = pred.shape[1]
    window = window.expand(channel, 1, window_size, window_size)
    pad = window_size // 2

    mu1 = torch.nn.functional.conv2d(pred, window, padding=pad, groups=channel)
    mu2 = torch.nn.functional.conv2d(target, window, padding=pad, groups=channel)

    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = torch.nn.functional.conv2d(pred * pred, window, padding=pad, groups=channel) - mu1_sq
    sigma2_sq = torch.nn.functional.conv2d(target * target, window, padding=pad, groups=channel) - mu2_sq
    sigma12 = torch.nn.functional.conv2d(pred * target, window, padding=pad, groups=channel) - mu1_mu2

    c1 = (0.01 * max_val) ** 2
    c2 = (0.03 * max_val) ** 2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )
    return ssim_map.mean().item()


# --------------------------------------------------------------------------
# Argparse
# --------------------------------------------------------------------------
def build_argparser():
    p = argparse.ArgumentParser(description="Train NAFNetSR on SEM imagery")
    p.add_argument("--mode", choices=["baseline", "finetune"], required=True)
    p.add_argument("--hr_dir", type=str, required=True)
    p.add_argument("--lr_dir", type=str, default=None,
                    help="Optional paired LR directory; if omitted, LR is synthesized on-the-fly")
    p.add_argument("--val_hr_dir", type=str, default=None)
    p.add_argument("--val_lr_dir", type=str, default=None)
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--epochs", type=int, default=None, help="Overrides the mode default if set")
    p.add_argument("--lr", type=float, default=None, help="Overrides the mode default if set")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to load (required for --mode finetune)")
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--checkpoint_dir", type=str, default="/content/checkpoints")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)

    if args.mode == "finetune" and args.resume is None:
        raise ValueError("--mode finetune requires --resume <path_to_baseline_checkpoint>")

    defaults = MODE_DEFAULTS[args.mode]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    lr = args.lr if args.lr is not None else defaults["lr"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype, use_scaler = get_amp_settings(device)
    print(f"[setup] device={device} amp_dtype={amp_dtype} grad_scaler={use_scaler} "
          f"mode={args.mode} epochs={epochs} lr={lr}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # ---------------- Data ----------------
    train_ds = SEMDataset(
        hr_dir=args.hr_dir,
        lr_dir=args.lr_dir,
        patch_size=args.patch_size,
        scale=args.scale,
        train=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    val_loader = None
    if args.val_hr_dir is not None:
        val_ds = SEMDataset(
            hr_dir=args.val_hr_dir,
            lr_dir=args.val_lr_dir,
            patch_size=args.patch_size,
            scale=args.scale,
            train=False,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # ---------------- Model ----------------
    model = NAFNetSR(scale=args.scale).to(device)

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state)
        print(f"[setup] loaded weights from {args.resume}")

    ema = EMA(model, decay=args.ema_decay)

    # ---------------- Loss ----------------
    if args.mode == "baseline":
        criterion = nn.L1Loss()
    else:
        criterion = KLAPhysicsLoss(w_pixel=1.0, w_fft=0.02, w_grad=0.6)

    # ---------------- Optimizer / Scheduler ----------------
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    total_iters = max(1, epochs * len(train_loader))
    scheduler = CosineAnnealingLR(optimizer, T_max=total_iters, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    best_metric = -math.inf
    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_components = {}

        for it, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                pred = model(lr_img)
                if args.mode == "baseline":
                    loss = criterion(pred, hr_img)
                    loss_dict = {"l1": loss.detach()}
                else:
                    loss, loss_dict = criterion(pred, hr_img)

            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            scheduler.step()
            ema.update(model)
            global_step += 1

            running_loss += loss.item()
            for k, v in loss_dict.items():
                running_components[k] = running_components.get(k, 0.0) + float(v)

            if (it + 1) % args.log_interval == 0:
                avg_loss = running_loss / (it + 1)
                comp_str = " ".join(
                    f"{k}={running_components[k] / (it + 1):.5f}" for k in running_components
                )
                cur_lr = scheduler.get_last_lr()[0]
                print(
                    f"[epoch {epoch}/{epochs}] iter {it + 1}/{len(train_loader)} "
                    f"loss={avg_loss:.5f} ({comp_str}) lr={cur_lr:.3e}"
                )

        epoch_time = time.time() - epoch_start
        print(
            f"[epoch {epoch}/{epochs}] completed in {epoch_time:.1f}s "
            f"avg_loss={running_loss / len(train_loader):.5f}"
        )

        # ---------------- Validation (on EMA shadow weights) ----------------
        if val_loader is not None:
            ema.apply_shadow(model)
            model.eval()
            psnr_total, ssim_total, n = 0.0, 0.0, 0
            with torch.no_grad():
                for lr_img, hr_img in val_loader:
                    lr_img = lr_img.to(device, non_blocking=True)
                    hr_img = hr_img.to(device, non_blocking=True)
                    with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                        pred = model(lr_img)
                    pred = pred.float().clamp(0.0, 1.0)
                    hr_f = hr_img.float()
                    psnr_total += compute_psnr(pred, hr_f)
                    ssim_total += compute_ssim(pred, hr_f)
                    n += 1
            val_psnr = psnr_total / max(n, 1)
            val_ssim = ssim_total / max(n, 1)
            print(f"[epoch {epoch}/{epochs}] [val] PSNR={val_psnr:.3f}dB SSIM={val_ssim:.4f}")

            if val_psnr > best_metric:
                best_metric = val_psnr
                ckpt_path = os.path.join(args.checkpoint_dir, "best_ema_weights.pth")
                torch.save(
                    {
                        "model": ema.state_dict(),
                        "epoch": epoch,
                        "val_psnr": val_psnr,
                        "val_ssim": val_ssim,
                        "mode": args.mode,
                    },
                    ckpt_path,
                )
                print(f"[epoch {epoch}/{epochs}] [checkpoint] new best PSNR={val_psnr:.3f}dB -> {ckpt_path}")

            ema.restore(model)
        else:
            # No validation set provided: persist the current EMA weights
            # each epoch so training always yields a usable checkpoint.
            ckpt_path = os.path.join(args.checkpoint_dir, "best_ema_weights.pth")
            torch.save({"model": ema.state_dict(), "epoch": epoch, "mode": args.mode}, ckpt_path)

    print("[done] training complete.")


if __name__ == "__main__":
    main()
