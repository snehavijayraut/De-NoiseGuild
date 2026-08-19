# 🔬 De-Noise Guild — AI-Based SEM Image Restoration & 2× Super-Resolution

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware](https://img.shields.io/badge/Hardware-NVIDIA%20GPU%20%7C%20CUDA%20%7C%20CPU-success.svg)]()
[![Competition](https://img.shields.io/badge/SEMICON-Hackathon%202026-orange.svg)]()

> **Non-linear activation-free deep learning architecture for single-channel SEM denoising and 2× super-resolution.**

**De-Noise Guild** is an end-to-end deep learning restoration pipeline developed for the **KLA Problem Statement: AI-Based Restoration of Degraded Images** at the SEMICON India Hackathon 2026.

The pipeline performs joint **denoising** (suppression of high-frequency speckle noise and Gaussian sensor noise) and **2× super-resolution** on single-channel `.npy` matrix arrays. It utilizes **NAFNetSR**—a Non-Linear Activation-Free architecture paired with an 8-fold test-time augmentation (TTA) ensemble—to reconstruct critical line/space feature geometries in Scanning Electron Microscope (SEM) metrology imagery.

---

## 👥 Team Information

* **Team Name:** De-Noise Guild
* **College:** AISSMS College of Engineering, Pune (SPPU)
* **Team Members:**
  * **Sneha Vijay Raut** — Team Leader
  * **Sairaj Bandu Potgantwar** — Member
  * **Sairaj Harish Kalushe** — Member

---

## ⚡ Submission Quick Start (Official Interface)

This solution is fully self-contained, validated offline, and executes without network calls, external downloads, API tokens, or manual intervention.

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
- [x] **Batch Loading:** Automatically discovers and parses all single-channel `.npy` arrays from `<input-dir>`.
- [x] **Automatic Directory Creation:** Automatically initializes `<output-dir>` if it does not already exist.
- [x] **Exact 1:1 Mapping:** Produces exactly one restored array per input file using the **identical base filename** without extra suffixes or stems.
- [x] **Array Dimensions:** Output tensors are formatted as 2D grayscale float32 arrays with shape `(H, W)`.
- [x] **Target Resolution:** Accurately scales images to 2× spatial resolution ($H_{\text{out}} = 2 \times H_{\text{in}}$, $W_{\text{out}} = 2 \times W_{\text{in}}$) with dynamic fallback.
- [x] **Value Normalization & Sanitization:** Output floats are strictly bounded to $[0.0, 1.0]$ and sanitized using `nan_to_num` to ensure zero `NaN` or `Inf` values.
- [x] **Fully Offline & Self-Contained:** Executes on local GPU without network dependencies, API keys, or external downloads.

---

## 🏗️ Repository Layout

```text
De-NoiseGuild/
├── models/
│   └── best_ema_weights.pth   # 50-epoch optimized EMA shadow weights
├── dataset.py                 # Dataset loader with bicubic downsampling & Gaussian noise synthesis
├── losses.py                  # Charbonnier, Real-FFT, and Sobel gradient loss definitions
├── model.py                   # NAFNetSR model architecture (SimpleGate + SCA blocks)
├── README.md                  # Model architecture and execution documentation
├── requirements.txt           # Explicitly pinned Python dependencies
├── run.py                     # Official evaluation entrypoint with 8-fold TTA
└── train.py                   # Model training driver with EMA tracking
```

---

## 🔬 Architecture Specifications (`NAFNetSR`)

`NAFNetSR` eliminates compute-heavy nonlinear activations (GELU/ReLU) and self-attention layers in favor of lightweight, element-wise linear blocks:

* **Encoder Hierarchy:** 4 hierarchical stages with block depths `(1, 2, 2, 4)` across channel widths `(32, 64, 128, 256)` using strided convolutions for spatial downsampling.
* **Activation-Free Core:**
  * **SimpleGate:** Splits feature channels in half ($C \rightarrow C/2$) and performs element-wise multiplication, introducing non-linearity without activation overhead.
  * **Simplified Channel Attention (SCA):** Uses global average pooling followed by a $1\times 1$ convolution to model inter-channel dependencies efficiently.
* **Symmetric Decoder:** Mirrored decoder stages with block depths `(4, 2, 2, 1)` utilizing `ConvTranspose2d` upsampling and additive skip connections.
* **Sub-Pixel Upsampling:** Reconstruction head features a $3\times 3$ convolution followed by `PixelShuffle(scale=2)` to expand feature maps to target $2\times$ spatial resolution.
* **Global Residual Shortcut:** A bilinear-interpolated skip path directly routes the low-resolution input to the final reconstructed layer ($I_{\text{out}} = \mathcal{F}(I_{\text{in}}) + \text{Bilinear}(I_{\text{in}}, 2\times)$), eliminating boundary haloing and edge ringing.

---

## 🏋️ Training Setup & Optimization

The model was trained for **50 epochs** using **Exponential Moving Average (EMA)** tracking:

```bash
python train.py \
  --mode baseline \
  --hr_dir /content/data/train_hr \
  --val_hr_dir /content/data/val_hr \
  --patch_size 128 \
  --batch_size 8 \
  --epochs 50 \
  --lr 2e-4 \
  --checkpoint_dir /content/checkpoints
```

* **Objective Function:** Mean Absolute Error ($\text{L1 Loss} = \frac{1}{N} \sum |I_{\text{pred}} - I_{\text{target}}|$) ensuring sharp, artifact-free baseline reconstruction.
* **Optimizer:** AdamW ($\beta_1 = 0.9, \beta_2 = 0.999$, weight decay = $10^{-4}$).
* **Learning Rate Schedule:** Cosine Annealing decay starting at $2 \times 10^{-4}$ down to $\eta_{\text{min}} = 10^{-6}$.
* **EMA Weight Shadowing:** Tracked shadow weights with a decay factor of $\alpha = 0.999$, saving parameters via `best_ema_weights.pth` for test generalization.

---

## 🚀 Inference & 8-Fold Test-Time Augmentation (TTA)

The evaluation script `run.py` runs an 8-pass geometric ensemble during test evaluation:
1. **Geometric Transformations:** Generates 8 transformed variants per input matrix ($4\text{ rotations } [0^\circ, 90^\circ, 180^\circ, 270^\circ] \times 2\text{ horizontal flip states}$).
2. **Precision Execution:** Automatically selects `torch.bfloat16` for Ampere+ GPUs or `torch.float16` for Turing T4 GPUs during forward passes with the loaded EMA weights.
3. **Inverse Mapping & Ensemble Mean:** Inverts all geometric rotations/flips in reverse sequence and calculates the arithmetic mean of all 8 passes.
4. **Sanitization:** Clamps values strictly to $[0.0, 1.0]$ and writes float32 `.npy` arrays to `<output-dir>`.
