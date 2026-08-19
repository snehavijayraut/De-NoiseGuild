# 🔬 De-Noise Guild: AI-Based SEM Image Restoration & 2x Super-Resolution

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20%7C%20CUDA%20%7C%20CPU-success.svg)]()
[![Competition](https://img.shields.io/badge/SEMICON-Hackathon%202026-orange.svg)]()

An end-to-end, physics-informed deep learning pipeline engineered for joint **denoising** (suppression of high-frequency speckle noise and Gaussian sensor noise) and **2x super-resolution** of single-channel matrix arrays (`.npy`). Designed for the **KLA Problem Statement**, this solution pairs a Non-Linear Activation-Free Network (**NAFNetSR**) with frequency and edge-preserving losses to restore critical line and space geometries in Scanning Electron Microscope (SEM) metrology imagery.

---

## 👥 Team Information

* **Team Name:** De-Noise Guild
* **College:** AISSMS College of Engineering, Pune (SPPU)
* **Team Members:**
  * Sneha Vijay Raut (Team Leader)
  * Sairaj Bandu Potgantwar
  * Sairaj Harish Kalushe

---

## ⚡ Submission Quick Start (Official Interface)

This solution is fully self-contained, validated offline, and runs without internet connectivity, API tokens, or external downloads.

### 1. Environment Setup
```bash
python -m pip install -r requirements.txt
```

### 2. Model Checkpoint Location
Ensure model weights are located in the `models/` directory:
```text
models/best_ema_weights.pth
```

### 3. Execute Inference
Run the standard entrypoint using positional arguments:
```bash
python run.py <input-dir> <output-dir>
```

---

## 📋 Evaluation Compliance Checklist

- [x] **Execution Signature:** Entrypoint is strictly named `run.py` and accepts positional CLI arguments `python run.py <input-dir> <output-dir>`.
- [x] **Batch Loading:** Discovers and parses all `.npy` matrix arrays from the provided `<input-dir>`.
- [x] **Automatic Directory Creation:** Automatically initializes `<output-dir>` if it does not already exist.
- [x] **Exact 1:1 Mapping:** Writes exactly one restored output per input file using the **identical base filename** without extra suffixes.
- [x] **Array Dimensions:** Output tensors are formatted as 2D grayscale arrays with shape `(H, W)` (or `(H, W, 1)`).
- [x] **Target Resolution:** Accurately scales images to 2x spatial resolution ($H_{\text{out}} = 2 \times H_{\text{in}}$, $W_{\text{out}} = 2 \times W_{\text{in}}$) with dynamic interpolation fallback.
- [x] **Value Normalization & Sanitization:** Output floats are strictly bounded to $[0.0, 1.0]$ and sanitized using `nan_to_num` to ensure zero `NaN` or `Inf` values.
- [x] **Fully Offline & Self-Contained:** Executes on NVIDIA GPUs without network calls, Hugging Face downloads, API keys, or manual intervention.

---

## 🏗️ Repository Layout

```text
De-NoiseGuild/
├── models/
│   └── best_ema_weights.pth   # Best EMA model weights
├── dataset.py                 # Dataset loader with synthetic/paired augmentation pipeline
├── losses.py                  # Charbonnier, Real-FFT, and Sobel gradient loss functions
├── model.py                   # NAFNetSR model architecture (SimpleGate + SCA blocks)
├── README.md                  # Comprehensive documentation and reproduction guide
├── requirements.txt           # Explicitly pinned Python dependencies
├── run.py                     # Official evaluation entrypoint with 8-fold TTA
└── train.py                   # Two-stage progressive training engine
```

---

## 🛠️ Step-by-Step Engineering Changelog & Implementation History

Throughout project development and benchmark verification, the following technical updates were applied:

1. **Standardized Entrypoint (`run.py`):** Replaced legacy evaluation scripts with `run.py` to match positional format `python run.py <input-dir> <output-dir>`.
2. **Robust Checkpoint Unpacking:** Handled nested checkpoint structures and stripped DataParallel `module.` prefixes.
3. **Dynamic Normalization:** Standardized arbitrary input ranges cleanly into $[0.0, 1.0]$.
4. **8-Fold TTA:** Integrated 4 rotations x 2 flip states with mixed-precision inference and inverse transform averaging.
5. **Output Conformance:** Enforced 2D array output `(H, W)`, 2x spatial dimensions, and non-finite value sanitization.
6. **Repository Streamlining:** Cleaned legacy test artifacts and placed weights in `models/best_ema_weights.pth`.

---

## 🔬 Architecture Highlights

* **Activation-Free Core:** Uses **SimpleGate** and **Simplified Channel Attention (SCA)** to capture long-range context without heavy GELU/ReLU overhead.
* **Sub-Pixel Upsampling:** `PixelShuffle(scale=2)` head reconstructs 2x spatial feature representations.
* **Global Residual Shortcut:** Bilinear skip connection routes low-resolution input directly to the final layer, eliminating boundary haloing.

---

## 🏋️ Compound Training Pipeline

Two-stage progressive training with AdamW and Exponential Moving Average (EMA decay=0.999):

### Phase 1 — Structural Baseline (40 Epochs)
```bash
python train.py --mode baseline --hr_dir /content/data/train_hr --val_hr_dir /content/data/val_hr --patch_size 128 --batch_size 16 --checkpoint_dir /content/checkpoints
```

### Phase 2 — Physics-Informed Optimization (15 Epochs)
```bash
python train.py --mode finetune --hr_dir /content/data/train_hr --val_hr_dir /content/data/val_hr --resume /content/checkpoints/best_ema_weights.pth --patch_size 128 --batch_size 16 --checkpoint_dir /content/checkpoints
```

**Compound Objective ($\mathcal{L}_{\text{total}}$):**

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Charbonnier}} + 0.02 \cdot \mathcal{L}_{\text{FFT}} + 0.6 \cdot \mathcal{L}_{\text{Sobel}}$$

---

## 🚀 Inference & 8-Fold Test-Time Augmentation (TTA)

The submission entrypoint `run.py` executes an 8-pass geometric ensemble during test evaluation (4 rotations x 2 flips) with mixed-precision averaging and $[0.0, 1.0]$ clamping.
