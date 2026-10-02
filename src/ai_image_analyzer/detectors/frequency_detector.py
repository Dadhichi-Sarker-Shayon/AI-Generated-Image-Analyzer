from __future__ import annotations
from typing import Any
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class FrequencyDetector(BaseDetector):
    """Heuristic frequency-domain detector for AI-generated images."""
    name = "frequency"

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig.default()

    def evaluate(self, data: AnalysisData) -> DetectorResult:
        freq = data.frequency
        h, w = data.signals.shape

        # Insufficient data check
        if freq.power_law_exponent is None or min(h, w) < 128:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.INSUFFICIENT_DATA,
                reasoning=[f"Image too small ({min(h,w)}px) or no exponent computed"],
            )

        # Exponent deviation
        exp_dev = abs(freq.power_law_exponent - self.config.exponent_center) - self.config.exponent_band
        s_exp = _clamp01(max(0.0, exp_dev) / self.config.exponent_band)

        # Spectral peaks
        s_peak = _clamp01((freq.max_peak_db - self.config.peak_db_threshold) / 20.0)

        # Lattice (GAN grid)
        s_lat = _clamp01((freq.best_lattice_db - self.config.lattice_db_threshold) / 14.0)

        # HF energy ratio deviation (both high and low are suspicious)
        hf = freq.hf_energy_ratio or 0.0
        s_hf_high = _clamp01(max(0.0, hf - 0.10) / 0.15)
        s_hf_low = _clamp01(max(0.0, 0.0005 - hf) / 0.0015)
        s_hf = _clamp01(s_hf_high + s_hf_low)

        # Weighted score
        score = 0.30 * s_exp + 0.25 * s_peak + 0.25 * s_lat + 0.20 * s_hf

        # Confidence: how many signals fired
        fired = sum(1 for s in [s_exp, s_peak, s_lat, s_hf] if s > 0.1)
        confidence = _clamp01(fired * 0.22 + (1 if score > 0.7 else 0) * 0.12)

        reasoning = []
        if s_exp > 0.1:
            reasoning.append(f"Shallow power spectrum (alpha={freq.power_law_exponent:.2f}, natural: {self.config.exponent_center:.2f}±{self.config.exponent_band})")
        if s_peak > 0.1:
            reasoning.append(f"Significant spectral peaks ({freq.n_peaks_significant} peaks >{self.config.peak_db_threshold}dB, max {freq.max_peak_db:.1f}dB)")
        if s_lat > 0.1:
            reasoning.append(f"Lattice artifact at period {freq.best_lattice_period}px ({freq.best_lattice_db:.1f}dB)")
        if s_hf > 0.1:
            reasoning.append(f"Anomalous HF energy ratio ({hf:.4f})")

        return DetectorResult(
            name=self.name,
            status=DetectorStatus.OK,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            features={
                "exponent": freq.power_law_exponent or 0,
                "max_peak_db": freq.max_peak_db,
                "best_lattice_db": freq.best_lattice_db,
                "hf_energy_ratio": hf,
            },
        )