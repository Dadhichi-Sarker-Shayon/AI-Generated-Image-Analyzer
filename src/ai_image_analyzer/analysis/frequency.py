from __future__ import annotations
import dataclasses
from typing import Any
import numpy as np
from scipy import ndimage
from scipy.fft import fftshift
from pydantic import BaseModel, Field


class FrequencyMetrics(BaseModel):
    power_law_exponent: float | None = None
    radial_profile: list[list[float]] = Field(default_factory=list)
    hf_energy_ratio: float | None = None
    mid_energy_ratio: float | None = None
    n_peaks_significant: int = 0
    max_peak_db: float = 0.0
    top_peaks: list[dict[str, float]] = Field(default_factory=list)
    lattice_db: dict[str, float] = Field(default_factory=dict)
    best_lattice_period: int | None = None
    best_lattice_db: float = 0.0


def compute_frequency_metrics(
    signals: ImageSignals,
    exponent_center: float = -2.3,
    exponent_band: float = 0.35,
    peak_db_threshold: float = 10.0,
    lattice_db_threshold: float = 8.0,
    lattice_periods: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10),
    hf_cutoff: float = 0.35,
) -> FrequencyMetrics:
    power = signals.power
    freq_r = signals.freq_r
    h, w = signals.shape
    nyquist = min(h, w) / 2

    # Radial profile (log-spaced bins)
    f_min = 2.0 / min(h, w)  # ~2px period
    f_max = min(hf_cutoff * 1.2, 0.48)
    n_bins = 16
    bins = np.logspace(np.log10(f_min), np.log10(f_max), n_bins + 1)
    radial = []
    for i in range(n_bins):
        mask = (freq_r >= bins[i]) & (freq_r < bins[i + 1])
        if mask.any():
            radial.append([float((bins[i] + bins[i + 1]) / 2), float(power[mask].mean())])

    radial = np.array(radial)
    profile = radial.tolist()

    # Power-law fit on low-mid frequencies (f <= 0.30)
    fit_mask = (radial[:, 0] > 0) & (radial[:, 0] <= 0.30) & (radial[:, 1] > 0)
    exponent = None
    if fit_mask.sum() >= 4:
        x = np.log10(radial[fit_mask, 0])
        y = np.log10(radial[fit_mask, 1])
        coeff = np.polyfit(x, y, 1)
        exponent = float(coeff[0])

    # Energy ratios
    total_energy = power.sum()
    hf_mask = freq_r > hf_cutoff
    mid_mask = (freq_r > 0.08) & (freq_r <= hf_cutoff)
    hf_energy_ratio = float(power[hf_mask].sum() / total_energy) if total_energy > 0 else None
    mid_energy_ratio = float(power[mid_mask].sum() / total_energy) if total_energy > 0 else None

    # Spectral peak detection
    center_mask = freq_r > 0.02
    median_power = float(np.median(power[center_mask])) if center_mask.any() else 1.0
    peak_threshold = median_power * (10 ** (peak_db_threshold / 10.0))

    max_filter = ndimage.maximum_filter(power, size=9, mode="constant", cval=0)
    peak_mask = (power == max_filter) & (power > peak_threshold) & (freq_r > 0.02) & (freq_r < 0.45)
    peak_coords = np.argwhere(peak_mask)

    top_peaks = []
    max_peak_db = 0.0
    for py, px in peak_coords[:20]:
        fx = signals.freq_x[py, px]
        fy = signals.freq_y[py, px]
        pval = power[py, px]
        db = 10 * np.log10(pval / median_power) if median_power > 0 else 0
        max_peak_db = max(max_peak_db, db)
        top_peaks.append({"fx": float(fx), "fy": float(fy), "db": float(db)})

    n_significant = sum(1 for p in top_peaks if p["db"] >= peak_db_threshold)

    # Lattice energy detection (GAN grid artifacts)
    lattice_db = {}
    best_db = 0.0
    best_period = None
    for period in lattice_periods:
        f0 = 1.0 / period
        positions = [
            (f0, 0), (-f0, 0), (0, f0), (0, -f0),
            (f0, f0), (-f0, f0), (f0, -f0), (-f0, -f0),
        ]
        vals = []
        for fx_t, fy_t in positions:
            ix = int(round(fx_t * w)) + w // 2
            iy = int(round(fy_t * h)) + h // 2
            if 0 <= ix < w and 0 <= iy < h:
                # 3x3 window
                ix0, ix1 = max(ix - 1, 0), min(ix + 2, w)
                iy0, iy1 = max(iy - 1, 0), min(iy + 2, h)
                vals.append(power[iy0:iy1, ix0:ix1].mean())
        if vals:
            avg_db = 10 * np.log10(np.mean(vals) / median_power) if median_power > 0 else 0
            lattice_db[str(period)] = float(avg_db)
            if avg_db > best_db:
                best_db = avg_db
                best_period = period

    return FrequencyMetrics(
        power_law_exponent=exponent,
        radial_profile=profile,
        hf_energy_ratio=hf_energy_ratio,
        mid_energy_ratio=mid_energy_ratio,
        n_peaks_significant=n_significant,
        max_peak_db=max_peak_db,
        top_peaks=top_peaks[:5],
        lattice_db=lattice_db,
        best_lattice_period=best_period,
        best_lattice_db=best_db,
    )