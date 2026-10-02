from __future__ import annotations
import dataclasses
from typing import Any
from .signals import ImageSignals
from .frequency import FrequencyMetrics, compute_frequency_metrics
from .noise import NoiseMetrics, compute_noise_metrics
from .texture import TextureMetrics, compute_texture_metrics
from .color import ColorMetrics, compute_color_metrics
from .jpeg import JpegMetrics, compute_jpeg_metrics
from ..config import AnalysisConfig
from ..io_utils import load_image


@dataclasses.dataclass
class AnalysisData:
    """Complete analysis output containing all metric groups."""
    image_meta: dict[str, Any]
    signals: ImageSignals
    frequency: FrequencyMetrics
    noise: NoiseMetrics
    texture: TextureMetrics
    color: ColorMetrics
    jpeg: JpegMetrics


class SignalAnalyzer:
    """Orchestrates all analysis modules."""

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig.default()

    def analyze(self, rgb: np.ndarray, source_meta: dict[str, Any] | None = None) -> AnalysisData:
        signals = ImageSignals.build(rgb, max_dim=self.config.max_analysis_dim)

        freq = compute_frequency_metrics(
            signals,
            exponent_center=self.config.exponent_center,
            exponent_band=self.config.exponent_band,
            peak_db_threshold=self.config.peak_db_threshold,
            lattice_db_threshold=self.config.lattice_db_threshold,
            lattice_periods=self.config.lattice_periods,
            hf_cutoff=self.config.hf_cutoff,
        )
        noise = compute_noise_metrics(
            signals,
            tile_size=self.config.tile_size,
            sensor_period_db_natural=self.config.sensor_period_db_natural,
        )
        texture = compute_texture_metrics(signals)
        color = compute_color_metrics(signals)
        jpeg = compute_jpeg_metrics(source_meta.get("file_bytes") if source_meta else None)

        return AnalysisData(
            image_meta=source_meta or {},
            signals=signals,
            frequency=freq,
            noise=noise,
            texture=texture,
            color=color,
            jpeg=jpeg,
        )