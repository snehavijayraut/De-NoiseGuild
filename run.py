"""
run.py
Submission entrypoint for NAFNetSR-based SEM restoration.

Usage:
    python run.py <input-dir> <output-dir> [--scale N]

Notes:
- Reads all .npy files from <input-dir>
- Creates <output-dir> if missing
- Produces one restored .npy per input with the same filename
- Outputs are grayscale arrays shape (H, W) or (H, W, 1), clipped to [0,1], no NaN/Inf
- Loads model architecture from model.py in the same folder (or parent repo)
- Prefers checkpoint at ./models/best_ema_weights.pth; falls back to ../best_ema_weights.pth or ./best_ema_weights.pth

This script is written to run offline on an NVIDIA GPU (if available) without internet
access. Place the checkpoint file best_ema_weights.pth into the models/ directory to
make the submission self-contained.
"""

import os
import sys
import argparse
import numpy as np
import torch

# Ensure repository imports work (model.py is in repo root)
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from model import NAFNetSR
except Exception as e:
    raise RuntimeError(f"Failed to import model.py from repository root: {e}")


def get_amp_dtype(device):
    if device.type != "cuda":
        return torch.float32
    major, minor = torch.cuda.get_device_capability(device)
    cc = major + minor / 10.0
    return torch.bfloat16 if cc >= 8.0 else torch.float16


def _load_npy_as_float01(path):
    arr = np.load(path)
    arr = arr.astype(np.float32)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    # Collapse incidental channel axis if present
    if arr.ndim == 3:
        # If last dim is channel-like, prefer that
        if arr.shape[-1] in (1, 3, 4):
            arr = arr[..., 0]
        elif arr.shape[0] in (1, 3, 4):
            arr = arr[0]
        else:
            # If ambiguous, reduce to first channel
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Loaded array at {path} has unexpected shape {arr.shape}; expected 2D or 3D with channel axis.")
    return arr


@torch.no_grad()
def tta_predict(model, x, device, amp_dtype):
    # x: (1,1,H,W)
    preds = []
    for k in range(4):
        for flip in (False, True):
            xt = torch.rot90(x, k, dims=[-2, -1])
            if flip:
                xt = torch.flip(xt, dims=[-1])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=(device.type == "cuda")):
                yt = model(xt)
            yt = yt.float()
            if flip:
                yt = torch.flip(yt, dims=[-1])
            if k:
                yt = torch.rot90(yt, -k, dims=[-2, -1])
            preds.append(yt)
    return torch.stack(preds, dim=0).mean(dim=0)


def find_checkpoint():
    candidates = [
        os.path.join(REPO_ROOT, "models", "best_ema_weights.pth"),
        os.path.join(REPO_ROOT, "best_ema_weights.pth"),
        os.path.join(REPO_ROOT, "../best_ema_weights.pth"),
    ]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.exists(c):
            return c
    return None


def save_output_array(arr, out_path):
    # arr: numpy 2D or 3D (H,W) or (H,W,1) already in [0,1]
    np.save(out_path, arr.astype(np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir")
    p.add_argument("output_dir")
    p.add_argument("--scale", type=int, default=2, help="SR scale factor expected by the model (default: 2)")
    args = p.parse_args()

    input_dir = args.input_dir
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"ERROR: input_dir does not exist or is not a directory: {input_dir}")
        sys.exit(2)

    os.makedirs(output_dir, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith('.npy'))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = get_amp_dtype(device)
    print(f"[setup] device={device} amp_dtype={amp_dtype}")

    model = NAFNetSR(scale=args.scale).to(device)
    ckpt = find_checkpoint()
    if ckpt is None:
        print("WARNING: checkpoint not found. Place best_ema_weights.pth into ./models/ or repo root. Model will be randomly initialized.")
    else:
        print(f"Loading checkpoint: {ckpt}")
        state = torch.load(ckpt, map_location=device)
        state = state.get('model', state) if isinstance(state, dict) else state
        model.load_state_dict(state)
    model.eval()

    for fname in npy_files:
        in_path = os.path.join(input_dir, fname)
        try:
            arr = _load_npy_as_float01(in_path)
        except Exception as e:
            print(f"Failed to load {in_path}: {e}")
            continue

        h, w = arr.shape[:2]
        x = torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).unsqueeze(0).float().to(device)

        pred_t = tta_predict(model, x, device, amp_dtype)
        pred_t = pred_t.clamp(0.0, 1.0).squeeze(0).squeeze(0).cpu()
        pred_np = pred_t.numpy()

        # Ensure finite
        if not np.all(np.isfinite(pred_np)):
            pred_np = np.nan_to_num(pred_np, nan=0.0, posinf=1.0, neginf=0.0)
            pred_np = np.clip(pred_np, 0.0, 1.0)

        target_h, target_w = h * args.scale, w * args.scale
        if pred_np.shape[0] != target_h or pred_np.shape[1] != target_w:
            # Try to resize to expected resolution using available libs
            print(f"Prediction shape {pred_np.shape} != expected ({target_h},{target_w}) for {fname}; attempting resize.")
            try:
                import cv2
                pred_np = cv2.resize(pred_np, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            except Exception:
                # Fallback: simple numpy repeat/resample
                zoom_h = (target_h / pred_np.shape[0])
                zoom_w = (target_w / pred_np.shape[1])
                # Nearest neighbor repeat as fallback
                nh = int(np.ceil(zoom_h))
                nw = int(np.ceil(zoom_w))
                y = np.repeat(np.repeat(pred_np, nh, axis=0), nw, axis=1)
                pred_np = y[:target_h, :target_w]
            pred_np = np.clip(pred_np, 0.0, 1.0)

        # Ensure and save as (H,W) or (H,W,1) — choose (H,W)
        if pred_np.ndim == 3 and pred_np.shape[2] == 1:
            pred_np = pred_np[:, :, 0]

        out_path = os.path.join(output_dir, fname)
        save_output_array(pred_np, out_path)
        print(f"Saved: {out_path}  shape={pred_np.shape}")

    print("[done] All files processed.")


if __name__ == '__main__':
    main()
