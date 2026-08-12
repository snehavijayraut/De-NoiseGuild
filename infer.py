"""
infer.py
Standalone inference script for NAFNetSR.

Runs 8-fold test-time augmentation (4x 90-degree rotations x 2 horizontal
flip states), inverse-transforms each prediction back to the canonical
orientation, and averages them in mixed precision. Predictions are clipped
to [0.0, 1.0], written as `<name>_pred.npy` to --output_dir, and zipped
into a single submission archive.

Example (Colab T4):
  python infer.py --checkpoint /content/checkpoints/best_ema_weights.pth \
      --input_dir /content/data/test_lr
"""

import argparse
import os
import zipfile

import numpy as np
import torch

from model import NAFNetSR


def get_amp_dtype(device):
    if device.type != "cuda":
        return torch.float32
    major, minor = torch.cuda.get_device_capability(device)
    cc = major + minor / 10.0
    return torch.bfloat16 if cc >= 8.0 else torch.float16


def _load_npy_as_float01(path):
    arr = np.load(path).astype(np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[-1] in (1, 3, 4) else arr[0]
    return arr


def _tta_forward(model, x, device, amp_dtype):
    """8-fold TTA: {identity, rot90, rot180, rot270} x {no-flip, hflip}.
    Every forward pass is inverse-transformed back to the canonical
    orientation before averaging, with the inverse flip undone before the
    inverse rotation (exact reverse of the forward transform order)."""
    preds = []
    with torch.no_grad():
        for k in range(4):  # rotations: 0, 90, 180, 270 degrees
            for flip in (False, True):
                xt = torch.rot90(x, k, dims=[-2, -1])
                if flip:
                    xt = torch.flip(xt, dims=[-1])

                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                    yt = model(xt)
                yt = yt.float()

                # exact inverse transform (reverse order: undo flip, then rotation)
                if flip:
                    yt = torch.flip(yt, dims=[-1])
                if k:
                    yt = torch.rot90(yt, -k, dims=[-2, -1])

                preds.append(yt)
    return torch.stack(preds, dim=0).mean(dim=0)


def main():
    parser = argparse.ArgumentParser(description="NAFNetSR 8-fold TTA inference")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="/content/predictions")
    parser.add_argument("--zip_path", type=str, default="/content/submission.zip")
    parser.add_argument("--scale", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"[setup] device={device} amp_dtype={amp_dtype}")

    os.makedirs(args.output_dir, exist_ok=True)

    model = NAFNetSR(scale=args.scale).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state)
    model.eval()

    input_files = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(".npy"))
    if not input_files:
        raise FileNotFoundError(f"No .npy files found in input_dir={args.input_dir}")

    out_paths = []
    for fname in input_files:
        arr = _load_npy_as_float01(os.path.join(args.input_dir, fname))
        x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().to(device)

        pred = _tta_forward(model, x, device, amp_dtype)
        pred = pred.clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)

        stem = os.path.splitext(fname)[0]
        out_name = f"{stem}_pred.npy"
        out_path = os.path.join(args.output_dir, out_name)
        np.save(out_path, pred)
        out_paths.append(out_path)
        print(f"[infer] {fname} -> {out_name}  shape={pred.shape}")

    with zipfile.ZipFile(args.zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in out_paths:
            zf.write(p, arcname=os.path.basename(p))

    print(f"[done] wrote {len(out_paths)} predictions to {args.output_dir}")
    print(f"[done] compressed submission -> {args.zip_path}")


if __name__ == "__main__":
    main()
