# 🔬 De-Noise Guild — SEM Image Cleanup & 2× Upscaling

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Competition](https://img.shields.io/badge/SEMICON-Hackathon%202026-orange.svg)]()

## 📌 What This Project Does

This project cleans up noisy SEM (Scanning Electron Microscope) images and makes them **2× bigger and sharper**, using a deep learning model.

Think of it like this:
- **Input:** a small, grainy, noisy image
- **Output:** a bigger, cleaner, sharper version of the same image

It was built for the **KLA Problem Statement** at the **SEMICON India Hackathon 2026**.

---

## 👤 Author

**Sneha Vijay Raut**
AISSMS College of Engineering, Pune (SPPU)

---

## 🚀 How to Run It

### Step 1 — Install requirements
```bash
pip install -r requirements.txt
```

### Step 2 — Check the model file is here
```text
models/best_ema_weights.pth
```

### Step 3 — Run the program
```bash
python run.py <input-folder> <output-folder>
```

**Example:**
```bash
python run.py ./test_images ./results
```

That's it. It reads every image from the input folder and saves the cleaned-up, bigger version in the output folder.

---

## 🧠 How It Works (Simple Version)

1. **Input:** Grainy, low-resolution grayscale images (`.npy` files)
2. **Model:** A neural network called **NAFNetSR** looks at the image and learns to remove noise while adding detail
3. **Output:** A clean image that is **twice as tall and twice as wide** as the input
4. **Extra trick:** The model looks at the image 8 different ways (rotated/flipped) and averages the results — this makes the output more accurate and stable

---

## 📋 What the Output Looks Like

- Same filename as the input
- 2× the height and width of the input
- Pixel values kept between `0.0` and `1.0`
- No broken/invalid values (`NaN` or `Inf` are automatically fixed)

---

## 🏗️ Project Files

```text
De-NoiseGuild/
├── models/
│   └── best_ema_weights.pth   # trained model weights
├── dataset.py                 # loads images & creates noisy training data
├── losses.py                  # math used to measure how "wrong" the output is
├── model.py                   # the NAFNetSR neural network
├── run.py                     # main script — run this to clean images
├── train.py                   # script used to train the model
└── requirements.txt           # list of Python packages needed
```

---

## 🏋️ How It Was Trained (Simple Version)

If you want to retrain the model yourself:

```bash
python train.py \
  --hr_dir ./data/train_hr \
  --val_hr_dir ./data/val_hr \
  --patch_size 128 \
  --batch_size 8 \
  --epochs 50 \
  --lr 2e-4 \
  --checkpoint_dir ./checkpoints
```

**In plain terms:**
- Trained for **50 rounds (epochs)** over the training images
- Used a smart way of saving the "best average" version of the model weights (called EMA), so results stay stable
- Learns by comparing its cleaned-up guess to the real, high-quality image and slowly getting closer

---

## ✅ Quick Summary

| Feature | Details |
|---|---|
| Task | Denoise + 2× upscale SEM images |
| Input format | `.npy` grayscale arrays |
| Output format | `.npy` grayscale arrays, 2× size |
| Model | NAFNetSR (lightweight, no heavy attention layers) |
| Works offline? | ✅ Yes, no internet needed |
| Hardware | Runs on GPU or CPU |
