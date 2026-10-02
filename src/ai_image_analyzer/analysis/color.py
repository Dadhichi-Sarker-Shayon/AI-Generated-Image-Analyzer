from __future__ import annotations
import numpy as np
import cv2
from pydantic import BaseModel


class ColorMetrics(BaseModel):
    banded_fraction: float
    colorfulness: float
    saturation_std: float
    gray_region_rb_cast: float


def compute_color_metrics(signals: ImageSignals) -> ColorMetrics:
    rgb = signals.rgb.astype(np.float32) / 255.0
    h, w = rgb.shape[:2]

    # Banding detection in smooth regions
    gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.hypot(grad_x, grad_y)
    smooth_mask = grad_mag < 0.02  # low gradient in [0,1] scale

    banded_fracs = []
    for c in range(3):
        ch = rgb[..., c]
        # Runs of equal values along rows (vectorized)
        diff = np.diff(ch, axis=1)
        # A run of >=3 equal pixels means two consecutive zero diffs
        run_mask = (diff[:, :-1] == 0) & (diff[:, 1:] == 0)
        # Align back to original pixel positions (center of the 3-pixel run)
        run_pixels = np.zeros_like(ch, dtype=bool)
        run_pixels[:, 1:-1] = run_mask
        banded_px = run_pixels & smooth_mask
        banded_fracs.append(float(banded_px.sum() / max(1, smooth_mask.sum())))

    banded_fraction = max(banded_fracs)

    # Colorfulness (Hasler & Süsstrunk)
    rg = rgb[..., 0] - rgb[..., 1]
    yb = 0.5 * (rgb[..., 0] + rgb[..., 1]) - rgb[..., 2]
    std_rg = np.std(rg)
    std_yb = np.std(yb)
    mean_rg = np.mean(rg)
    mean_yb = np.mean(yb)
    colorfulness = float(np.sqrt(std_rg**2 + std_yb**2) + 0.3 * np.sqrt(mean_rg**2 + mean_yb**2))

    # Saturation std
    hsv = cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation_std = float(hsv[..., 1].std() / 255.0)

    # Gray region color cast (R-B on near-gray pixels)
    max_c = rgb.max(axis=2)
    min_c = rgb.min(axis=2)
    gray_mask = (max_c - min_c) < (8 / 255)
    if gray_mask.any():
        gray_rb = float(np.mean(rgb[gray_mask, 0] - rgb[gray_mask, 2]))
    else:
        gray_rb = 0.0

    return ColorMetrics(
        banded_fraction=banded_fraction,
        colorfulness=colorfulness,
        saturation_std=saturation_std,
        gray_region_rb_cast=gray_rb,
    )