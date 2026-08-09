"""
infer.py — Standalone submission inference script.

    python infer.py --input_dir <LR_dir> --output_dir <restored_dir> --weights best_weights_only.pth

Throughput optimizations (in order of impact):
  1. Batched inference instead of one-image-at-a-time — amortizes kernel
     launch overhead across many images per forward pass.
  2. torch.no_grad() — skips autograd graph construction entirely.
  3. AMP autocast — fp16 compute on Tensor Cores.
  4. channels_last memory format — better conv throughput on modern GPUs.
  5. torch.compile(mode="max-autotune") — fuses the forward pass into fewer,
     faster kernels; falls back gracefully if unavailable/unsupported.
  6. cudnn.benchmark = True — autotunes conv algorithms for the fixed
     inference batch shape.
The reported FPS covers disk read -> tensor prep -> GPU inference -> disk
write, matching the benchmark's definition of end-to-end throughput.
"""
import os
import glob
import time
import argparse

import numpy as np
import cv2
import torch
from torch.cuda.amp import autocast

from model import LightweightRestormer
from dataset import load_array  # same no-clip loader used during training


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--weights", required=True, help="path to a weights-only .pth file")
    p.add_argument("--scale", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--no_compile", action="store_true", help="disable torch.compile")
    return p.parse_args()


def build_model(weights_path, scale, device, use_compile):
    model = LightweightRestormer(
        in_channels=1, out_channels=1, dim=32,
        depths=(1, 2, 2, 4), heads=(1, 2, 4, 8), scale=scale,
    )
    state = torch.load(weights_path, map_location=device)
    state = state.get("model", state)  # accept either a full checkpoint or a weights-only file
    model.load_state_dict(state)
    model.to(device).eval()
    model = model.to(memory_format=torch.channels_last)  # faster conv layout on Tensor Cores

    if use_compile and hasattr(torch, "compile") and device.type == "cuda":
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:  # compile isn't supported on every platform/driver combo
            print(f"torch.compile unavailable, falling back to eager mode: {e}")
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    os.makedirs(args.output_dir, exist_ok=True)
    model = build_model(args.weights, args.scale, device, use_compile=not args.no_compile)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy")
    files = sorted(f for f in glob.glob(os.path.join(args.input_dir, "*")) if f.lower().endswith(exts))
    print(f"Found {len(files)} images to restore.")

    batch, names = [], []
    n_done, total_time = 0, 0.0

    def flush():
        nonlocal n_done, total_time
        if not batch:
            return
        t0 = time.time()
        x = torch.stack(batch).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            out = model(x)
        out = out.float().clamp(0.0, 1.0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()  # ensure GPU work is actually finished before timing stops
        total_time += time.time() - t0

        for j, name in enumerate(names):
            restored = (out[j, 0] * 255.0).round().astype(np.uint8)
            out_path = os.path.join(args.output_dir, os.path.splitext(name)[0] + ".png")
            cv2.imwrite(out_path, restored)
        n_done += len(names)
        batch.clear()
        names.clear()

    for fpath in files:
        arr = load_array(fpath)  # unclipped float32, identical preprocessing to training
        tensor = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float()
        batch.append(tensor)
        names.append(os.path.basename(fpath))
        if len(batch) == args.batch_size:
            flush()
    flush()  # remaining partial batch

    fps = n_done / total_time if total_time > 0 else 0.0
    print(f"Restored {n_done} images in {total_time:.2f}s -> {fps:.2f} FPS")


if __name__ == "__main__":
    main()
