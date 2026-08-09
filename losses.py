"""
losses.py — Physics-informed restoration loss for KLA SEM restoration.

This loss combines pixel fidelity, edge sharpness, and frequency-domain
spectral preservation to avoid oversmoothing line-space arrays and nanoscale
SEM defects.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class KLAPhysicsLoss(nn.Module):
    def __init__(self, grad_weight=0.3, fft_weight=0.1):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.grad_weight = grad_weight
        self.fft_weight = fft_weight

    def forward(self, pred, target):
        loss_l1 = self.l1(pred, target)

        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        target_dx = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        target_dy = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_grad = self.l1(pred_dx, target_dx) + self.l1(pred_dy, target_dy)

        fft_pred = torch.fft.rfft2(pred, norm="ortho")
        fft_target = torch.fft.rfft2(target, norm="ortho")
        loss_fft = self.l1(torch.abs(fft_pred), torch.abs(fft_target))

        total = loss_l1 + (self.grad_weight * loss_grad) + (self.fft_weight * loss_fft)
        logs = {
            "l1": loss_l1.item(),
            "grad": loss_grad.item(),
            "fft": loss_fft.item(),
            "total": total.item(),
        }
        return total, logs
