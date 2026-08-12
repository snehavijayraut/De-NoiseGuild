"""
losses.py
Loss functions for SEM image restoration:

  - CharbonnierLoss   : smooth L1 approximation, robust to outliers.
  - FFTLoss           : real-FFT (rfft2, orthonormal) magnitude L1 loss,
                         matching high-frequency spectral density.
  - SobelGradientLoss : Gx / Gy Sobel-gradient L1 loss, enforcing sharp,
                         well-aligned edges.
  - KLAPhysicsLoss    : composite objective combining all three, tuned
                         for SEM line/space edge fidelity during
                         fine-tuning. Returns (total_loss, loss_dict).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    """Smooth L1 (Charbonnier) loss: mean(sqrt((pred - target)^2 + eps^2))."""

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


def _sobel_kernels(device, dtype):
    gx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    gy = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=dtype, device=device
    ).view(1, 1, 3, 3)
    return gx, gy


class SobelGradientLoss(nn.Module):
    """L1 loss between Sobel gradients (Gx, Gy) of prediction and target,
    enforcing sharp edge alignment along both axes."""

    def forward(self, pred, target):
        gx, gy = _sobel_kernels(pred.device, pred.dtype)
        c = pred.shape[1]
        if c > 1:
            gx = gx.repeat(c, 1, 1, 1)
            gy = gy.repeat(c, 1, 1, 1)
        pred_gx = F.conv2d(pred, gx, padding=1, groups=c)
        pred_gy = F.conv2d(pred, gy, padding=1, groups=c)
        target_gx = F.conv2d(target, gx, padding=1, groups=c)
        target_gy = F.conv2d(target, gy, padding=1, groups=c)
        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)
        return loss_x + loss_y


class FFTLoss(nn.Module):
    """L1 loss between real-FFT (torch.fft.rfft2, norm='ortho') magnitude
    spectra of prediction and target, matching high-frequency spectral
    density (fine SEM texture / noise statistics)."""

    def forward(self, pred, target):
        pred_fft = torch.fft.rfft2(pred.float(), norm="ortho")
        target_fft = torch.fft.rfft2(target.float(), norm="ortho")
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2 + 1e-12)
        target_mag = torch.sqrt(target_fft.real ** 2 + target_fft.imag ** 2 + 1e-12)
        return F.l1_loss(pred_mag, target_mag)


class KLAPhysicsLoss(nn.Module):
    """Composite physics-informed loss for SEM fine-tuning.

    total = w_pixel * CharbonnierLoss
          + w_fft   * FFTLoss   (torch.fft.rfft2, norm='ortho')
          + w_grad  * SobelGradientLoss (Gx + Gy)

    Returns:
        total_loss (torch.Tensor): scalar tensor, differentiable.
        loss_dict (dict[str, torch.Tensor]): detached per-component
            values (and the total) for logging.
    """

    def __init__(self, w_pixel=1.0, w_fft=0.02, w_grad=0.6, charbonnier_eps=1e-3):
        super().__init__()
        self.w_pixel = w_pixel
        self.w_fft = w_fft
        self.w_grad = w_grad
        self.pixel_loss = CharbonnierLoss(eps=charbonnier_eps)
        self.fft_loss = FFTLoss()
        self.grad_loss = SobelGradientLoss()

    def forward(self, pred, target):
        l_pixel = self.pixel_loss(pred, target)
        l_fft = self.fft_loss(pred, target)
        l_grad = self.grad_loss(pred, target)

        total = self.w_pixel * l_pixel + self.w_fft * l_fft + self.w_grad * l_grad

        loss_dict = {
            "pixel": l_pixel.detach(),
            "fft": l_fft.detach(),
            "grad": l_grad.detach(),
            "total": total.detach(),
        }
        return total, loss_dict
