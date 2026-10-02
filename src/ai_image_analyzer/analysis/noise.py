from __future__ import annotations
import numpy as np
import cv2
from scipy import fft, ndimage
from pydantic import BaseModel


class NoiseMetrics(BaseModel):
    noise_sigma: float
    spatial_uniformity: float
    noise_hf_ratio: float
    sensor_periodicity_db: float
    tile_std_cv: float


def compute_noise_metrics(
    signals: ImageSignals,
    tile_size: int = 64,
    sensor_period_db_natural: float = 8.0,
) -> NoiseMetrics:
    gray = signals.gray
    h, w = signals.shape

    # Noise residual = image - gaussian blur
    blur = cv2.GaussianBlur(gray, (3, 3), 1.2)
    noise = gray - blur

    # Global noise sigma
    noise_sigma = float(noise.std())

    # Spatial uniformity via tiling
    tile_stds = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            y1 = min(y + tile_size, h)
            x1 = min(x + tile_size, w)
            if y1 - y >= 8 and x1 - x >= 8:
                tile_stds.append(noise[y:y1, x:x1].std())
    tile_stds = np.array(tile_stds)
    if len(tile_stds) > 1 and tile_stds.mean() > 0:
        spatial_uniformity = float(1.0 - (tile_stds.std() / tile_stds.mean()))
    else:
        spatial_uniformity = 1.0

    # Noise FFT for high-freq ratio and sensor periodicity
    nF = fft.fftshift(fft.fft2(noise - noise.mean()))
    npower = np.abs(nF) ** 2
    nfreq_r = signals.freq_r

    n_total = npower.sum()
    n_hf_mask = nfreq_r > 0.25
    noise_hf_ratio = float(npower[n_hf_mask].sum() / n_total) if n_total > 0 else 0.0

    # Sensor periodicity (CFA period-2 at Nyquist corners)
    median_np = float(np.median(npower[nfreq_r > 0.02])) if (nfreq_r > 0.02).any() else 1.0
    # Check energy near (0.45..0.5, 0) and (0, 0.45..0.5)
    sensor_mask = (
        ((np.abs(signals.freq_x) > 0.42) & (np.abs(signals.freq_x) < 0.5) & (np.abs(signals.freq_y) < 0.05)) |
        ((np.abs(signals.freq_y) > 0.42) & (np.abs(signals.freq_y) < 0.5) & (np.abs(signals.freq_x) < 0.05))
    )
    sensor_energy = npower[sensor_mask].mean() if sensor_mask.any() else 0
    sensor_periodicity_db = 10 * np.log10(sensor_energy / median_np) if median_np > 0 and sensor_energy > 0 else 0.0

    return NoiseMetrics(
        noise_sigma=noise_sigma,
        spatial_uniformity=spatial_uniformity,
        noise_hf_ratio=noise_hf_ratio,
        sensor_periodicity_db=sensor_periodicity_db,
        tile_std_cv=float(tile_stds.std() / tile_stds.mean()) if len(tile_stds) > 1 and tile_stds.mean() > 0 else 0.0,
    )