import cv2
import numpy as np


def apply_kla_sem_degradation(gt_img: np.ndarray, scale: int = 2) -> np.ndarray:
    """Apply the official KLA SEM degradation sequence.

    The degradation intentionally keeps values outside the nominal [0, 1]
    range to match realistic SEM sensor output and to challenge restoration.
    """
    gt = gt_img.astype(np.float32)
    if gt.max() <= 1.0:
        gt = gt * 255.0

    h, w = gt.shape[:2]
    lr = cv2.resize(gt, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    sobelx = cv2.Sobel(lr, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(lr, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sobelx**2 + sobely**2)
    lr_brightened = lr + 0.15 * edge_mag

    speckle = np.random.normal(1.0, 0.08, lr.shape).astype(np.float32)
    gaussian = np.random.normal(0.0, 12.0, lr.shape).astype(np.float32)
    degraded = (lr_brightened * speckle) + gaussian
    return degraded.astype(np.float32) / 255.0
