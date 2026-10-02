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
        
        # Conditional expensive analyses
        if self.config.enable_fractal or self.config.enable_glcm:
            texture = compute_texture_metrics(signals, enable_fractal=self.config.enable_fractal, enable_glcm=self.config.enable_glcm)
        else:
            # Fast path: minimal texture metrics
            texture = TextureMetrics(
                lbp_entropy=0.0, lbp_uniform_ratio=1.0,
                glcm_contrast=0.0, glcm_energy=0.0, glcm_homogeneity=0.0,
                fractal_dimension=None
            )
        
        if self.config.enable_color_analysis:
            color = compute_color_metrics(signals)
        else:
            color = ColorMetrics(banded_fraction=0.0, colorfulness=0.0, saturation_std=0.0, gray_region_rb_cast=0.0)
        
        if self.config.enable_jpeg_analysis:
            jpeg = compute_jpeg_metrics(source_meta.get("file_bytes") if source_meta else None)
        else:
            jpeg = JpegMetrics(is_jpeg=False, estimated_quality=None, quantization_table_source=None, re_encode_size_ratio=None, histogram_quantization_step=None)

        return AnalysisData(
            image_meta=source_meta or {},
            signals=signals,
            frequency=freq,
            noise=noise,
            texture=texture,
            color=color,
            jpeg=jpeg,
        )