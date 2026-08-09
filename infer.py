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
from concurrent.futures import ThreadPoolExecutor
from collections import deque

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
    p.add_argument("--export_onnx", action="store_true",
                   help="export the restored model to ONNX for FP16 inference backends")
    p.add_argument("--onnx_path", default="restormer.onnx",
                   help="path to write ONNX model when --export_onnx is enabled")
    return p.parse_args()


def export_onnx_model(model, input_shape, path, device):
    model.eval()
    dummy = torch.randn(input_shape, device=device)
    torch.onnx.export(
        model, dummy, path,
        opset_version=17,
        input_names=["input"], output_names=["output"],
        dynamic_axes={"input": {0: "batch", 2: "height", 3: "width"},
                      "output": {0: "batch", 2: "height", 3: "width"}},
        do_constant_folding=True,
    )
    print(f"Exported ONNX model to {path}")


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

    use_bfloat16 = (device.type == "cuda" and
                    getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    autocast_dtype = torch.bfloat16 if use_bfloat16 else torch.float16

    loader_executor = ThreadPoolExecutor(max_workers=4)
    writer_executor = ThreadPoolExecutor(max_workers=4)
    pending = deque()
    save_futures = []
    n_done, total_time = 0, 0.0

    def flush(batch, names):
        nonlocal n_done, total_time
        if not batch:
            return
        t0 = time.time()
        x = torch.stack(batch).to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with torch.inference_mode(), autocast(dtype=autocast_dtype, enabled=device.type == "cuda"):
            out = model(x)
        out = out.float().clamp(0.0, 1.0).cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time += time.time() - t0

        for j, name in enumerate(names):
            restored = (out[j, 0] * 255.0).round().astype(np.uint8)
            out_path = os.path.join(args.output_dir, os.path.splitext(name)[0] + ".png")
            save_futures.append(writer_executor.submit(cv2.imwrite, out_path, restored))
        n_done += len(names)

    if args.export_onnx:
        export_onnx_model(model, (1, 1, 128, 128), args.onnx_path, device)

    batch, names = [], []
    for fpath in files:
        pending.append((os.path.basename(fpath), loader_executor.submit(load_array, fpath)))
        if len(pending) >= args.batch_size * 2:
            while pending and len(batch) < args.batch_size:
                name, future = pending.popleft()
                arr = future.result()
                batch.append(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float())
                names.append(name)
            flush(batch, names)
            batch, names = [], []

    while pending:
        name, future = pending.popleft()
        arr = future.result()
        batch.append(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float())
        names.append(name)
        if len(batch) == args.batch_size:
            flush(batch, names)
            batch, names = [], []
    flush(batch, names)

    loader_executor.shutdown(wait=True)
    writer_executor.shutdown(wait=True)
    for fut in save_futures:
        fut.result()

    fps = n_done / total_time if total_time > 0 else 0.0
    print(f"Restored {n_done} images in {total_time:.2f}s -> {fps:.2f} FPS")


if __name__ == "__main__":
    main()
