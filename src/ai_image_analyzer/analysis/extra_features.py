"""Additional interpretable forensic descriptors (candidate features; see scripts/analyze_metrics.py for validation).

All computed on a square uint8 RGB crop. Every value is a plain float with a physical meaning.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy import stats


def _radial_profile(power: np.ndarray, bins: int) -> np.ndarray:
    h, w = power.shape
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - h // 2, xx - w // 2) / (min(h, w) / 2)
    idx = np.clip((r * bins).astype(int), 0, bins)
    s = np.bincount(idx.ravel(), power.ravel(), minlength=bins + 1)[:bins]
    c = np.bincount(idx.ravel(), minlength=bins + 1)[:bins]
    return s / np.maximum(c, 1)


def compute_extra_features(rgb: np.ndarray) -> dict[str, float]:
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    f: dict[str, float] = {}
    # 1. radial spectrum (log10 power, normalized to the lowest band) + high-frequency slope
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    p = np.abs(np.fft.fftshift(np.fft.fft2((g - g.mean()) * win))) ** 2
    prof = np.log10(_radial_profile(p, 16) + 1e-9)
    prof = prof - prof[1]
    for i in (4, 8, 12, 14, 15):
        f[f"spec.band{i:02d}"] = float(prof[i])
    x = np.arange(8, 16)
    f["spec.hf_slope"] = float(np.polyfit(x, prof[8:16], 1)[0])
    f["spec.hf_drop"] = float(prof[10] - prof[15])
    # 2. noise residual statistics
    res = g - cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    f["res.std"] = float(res.std())
    f["res.kurtosis"] = float(stats.kurtosis(res.ravel()))
    f["res.skew"] = float(stats.skew(res.ravel()))
    f["res.zero_frac"] = float((np.abs(res) < 0.5).mean())
    for lag in (1, 2, 4, 8):
        a, b = res[:, :-lag], res[:, lag:]
        f[f"res.autocorr_x{lag}"] = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if res.std() > 0 else 0.0
    # 3. cross-channel noise correlation (demosaicing / shared-latent signature)
    rs = [(rgb[..., c].astype(np.float32) - cv2.medianBlur(rgb[..., c], 3).astype(np.float32)).ravel() for c in range(3)]
    f["chan.res_corr_rg"] = float(np.corrcoef(rs[0], rs[1])[0, 1]) if rs[0].std() > 0 and rs[1].std() > 0 else 0.0
    f["chan.res_corr_gb"] = float(np.corrcoef(rs[1], rs[2])[0, 1]) if rs[1].std() > 0 and rs[2].std() > 0 else 0.0
    # 4. gradient statistics
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag = np.hypot(gx, gy)
    f["grad.mean"] = float(mag.mean())
    f["grad.p99_over_mean"] = float(np.percentile(mag, 99) / (mag.mean() + 1e-6))
    f["grad.flat_frac"] = float((mag < 2.0).mean())
    # 5. JPEG blockiness: boundary vs interior pixel differences on the 8px grid
    dx = np.abs(np.diff(g, axis=1))
    cols = np.arange(dx.shape[1])
    f["jpeg.blockiness"] = float(dx[:, cols % 8 == 7].mean() / (dx[:, cols % 8 != 7].mean() + 1e-6))
    # 6. local-variance uniformity (16px tiles)
    t = 16
    tiles = g[: g.shape[0] // t * t, : g.shape[1] // t * t].reshape(g.shape[0] // t, t, g.shape[1] // t, t).std(axis=(1, 3))
    f["local.std_cv"] = float(tiles.std() / (tiles.mean() + 1e-6))
    f["local.min_over_med"] = float(np.percentile(tiles, 5) / (np.median(tiles) + 1e-6))
    # 7. tonal statistics
    f["tone.clip_frac"] = float(((rgb == 0) | (rgb == 255)).mean())
    f["tone.unique_colors"] = float(len(np.unique(rgb.reshape(-1, 3), axis=0)) / (rgb.shape[0] * rgb.shape[1]))
    # 8. Nyquist / checkerboard energy (transposed-convolution signature)
    f["spec.nyquist_ratio"] = float(np.log10((p[0, :].mean() + p[:, 0].mean()) / (p.mean() + 1e-9) + 1e-9))
    return f
