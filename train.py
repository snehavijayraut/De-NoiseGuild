"""
train.py — Training loop for the Lightweight Restormer.

Throughput/robustness choices:
  - torch.backends.cudnn.benchmark = True: patches are fixed-size, so cuDNN
    can autotune the fastest conv algorithms once and reuse them every step.
  - AMP (autocast + GradScaler): fp16 forward pass roughly halves memory
    and time on Tensor-Core GPUs; GradScaler prevents fp16 gradient underflow.
  - optimizer.zero_grad(set_to_none=True): skips a memset per step vs zeroing
    tensors in place.
  - Checkpointing saves both a full-state (resumable) checkpoint and a
    weights-only file — the latter is what infer.py should load, since it
    avoids pulling in optimizer/scheduler state at deployment time.
"""
import os
import time
import random
import argparse

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler

from dataset import build_dataloaders
from model import LightweightRestormer
from losses import CompositeRestorationLoss


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_lr", required=True, help="dir of training LR images")
    p.add_argument("--train_hr", required=True, help="dir of training HR/GT images")
    p.add_argument("--val_lr", required=True)
    p.add_argument("--val_hr", required=True)
    p.add_argument("--out_dir", default="./checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--patch_size", type=int, default=128, help="HR patch size, multiple of 8")
    p.add_argument("--scale", type=int, default=4, help="must match dataset's LR downsample factor")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--resume", default=None, help="path to a full checkpoint to resume from")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # reproducibility across runs/GPUs


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total, n = 0.0, 0
    for lr_img, hr_img in loader:
        lr_img = lr_img.to(device, non_blocking=True)
        hr_img = hr_img.to(device, non_blocking=True)
        with autocast():
            pred = model(lr_img)
            loss, _ = criterion(pred, hr_img)
        total += loss.item()
        n += 1
    model.train()
    return total / max(n, 1)


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # fixed patch sizes -> autotune conv algos once

    train_loader, val_loader = build_dataloaders(
        args.train_lr, args.train_hr, args.val_lr, args.val_hr,
        patch_size=args.patch_size, scale=args.scale,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    model = LightweightRestormer(
        in_channels=1, out_channels=1, dim=32,
        depths=(1, 2, 2, 4), heads=(1, 2, 4, 8), scale=args.scale,
    ).to(device)

    criterion = CompositeRestorationLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    start_epoch, best_val = 0, float("inf")

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        running = 0.0

        for i, (lr_img, hr_img) in enumerate(train_loader):
            lr_img = lr_img.to(device, non_blocking=True)
            hr_img = hr_img.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                pred = model(lr_img)
                loss, logs = criterion(pred, hr_img)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if i % args.log_every == 0:
                print(f"Epoch {epoch} [{i}/{len(train_loader)}] "
                      f"loss={logs['total']:.4f} (l1={logs['l1']:.4f} "
                      f"ssim={logs['ssim']:.4f} fft={logs['fft']:.4f})")

        scheduler.step()
        val_loss = validate(model, val_loader, criterion, device)
        dt = time.time() - t0
        print(f"== Epoch {epoch} done in {dt:.1f}s | "
              f"train={running/len(train_loader):.4f} val={val_loss:.4f} "
              f"lr={scheduler.get_last_lr()[0]:.2e} ==")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }
        torch.save(ckpt, os.path.join(args.out_dir, "last.pth"))

        if val_loss < best_val:
            best_val = val_loss
            ckpt["best_val"] = best_val
            torch.save(ckpt, os.path.join(args.out_dir, "best.pth"))
            # weights-only file: what infer.py should load (smaller, no optimizer state)
            torch.save(model.state_dict(), os.path.join(args.out_dir, "best_weights_only.pth"))
            print(f"  -> new best model saved (val={best_val:.4f})")


if __name__ == "__main__":
    main()
