"""
losses.py — Composite restoration loss: L1 (0.7) + SSIM (0.2) + 2D real-FFT (0.1).

The FFT term operates on magnitude spectra from torch.fft.rfft2 and explicitly
penalizes high-frequency error — exactly the content speckle noise and
downsampling destroy first, and that pure L1/SSIM under-weight since they're
dominated by low-frequency intensity/structure error.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import SSIM


class CompositeRestorationLoss(nn.Module):
    def __init__(self, l1_weight=0.7, ssim_weight=0.2, fft_weight=0.1, data_range=1.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight
        self.l1 = nn.L1Loss()
        self.ssim = SSIM(data_range=data_range, size_average=True, channel=1)

    def fft_loss(self, pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        # compare magnitude spectra — robust to the sub-pixel phase shifts that
        # a plain complex-value L1 would over-penalize
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

    def forward(self, pred, target):
        l1_val = self.l1(pred, target)
        ssim_val = 1.0 - self.ssim(pred, target)  # SSIM is a similarity score; loss wants 1 - SSIM
        fft_val = self.fft_loss(pred, target)

        total = (self.l1_weight * l1_val
                 + self.ssim_weight * ssim_val
                 + self.fft_weight * fft_val)

        logs = {
            "l1": l1_val.item(),
            "ssim": ssim_val.item(),
            "fft": fft_val.item(),
            "total": total.item(),
        }
        return total, logs
