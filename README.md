# Submission README

This repository has been prepared for packaging as a competition submission.
Place the files below into the submission root (or rename this repository
folder to your team name before archiving):

- run.py              # entrypoint: python run.py <input-dir> <output-dir>
- requirements.txt    # Python dependencies
- README.md           # this file (overwritten)
- models/             # include best_ema_weights.pth here

Quick start (offline):
1. Install dependencies (in a virtualenv):
   python -m pip install -r requirements.txt

2. Ensure the checkpoint is present:
   ./models/best_ema_weights.pth
   (The repository root also contains best_ema_weights.pth; it will be
    copied into models/ during packaging if needed.)

3. Run inference:
   python run.py <input-dir> <output-dir>

Behavior and guarantees:
- run.py reads all .npy files from the input directory and writes exactly
  one .npy file per input into the output directory using the same
  filenames.
- Outputs are grayscale arrays with shape (H, W) (or optionally
  (H, W, 1)), values are clipped to [0.0, 1.0], and checked for NaN/Inf.
- Output resolution is the input resolution scaled by the model `--scale`
  (default 2). If the model output shape differs, run.py will attempt
  a safe resize fallback.
- The solution runs entirely offline on an NVIDIA GPU (if available) and
  does not require network access or external model downloads.

Notes:
- If you want this folder renamed to your team name for final submission,
  do so after verifying inference works locally.
