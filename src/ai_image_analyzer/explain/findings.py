from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from ..analysis.analyzer import AnalysisData
from ..detectors.base import DetectorResult
from ..config import AnalysisConfig


class Finding(BaseModel):
    code: str
    title: str
    severity: str  # "info" | "warning" | "strong"
    supports: str  # "ai" | "natural" | "neutral"
    value: str
    threshold: str
    explanation: str


def build_findings(
    data: AnalysisData,
    ensemble_result: DetectorResult,
    config: AnalysisConfig | None = None,
) -> list[Finding]:
    cfg = config or AnalysisConfig.default()
    freq = data.frequency
    noise = data.noise
    texture = data.texture
    color = data.color
    jpeg = data.jpeg
    findings = []

    # 1. Shallow spectrum
    exp = freq.power_law_exponent
    if exp is not None:
        if exp > -2.0:
            findings.append(Finding(
                code="FFT_SHALLOW",
                title="Shallow power-law spectrum",
                severity="strong" if exp > -1.7 else "warning",
                supports="ai",
                value=f"{exp:.2f}",
                threshold=f"natural: {cfg.exponent_center:.2f}±{cfg.exponent_band}",
                explanation=f"Radial power spectrum decays as f^{exp:.2f} (natural photographs typically f^{-2.2} to f^{-2.6}). Shallow decay indicates excess mid-high frequency energy consistent with generative priors.",
            ))
        elif exp < -3.0:
            findings.append(Finding(
                code="FFT_DEEP",
                title="Steep power-law spectrum",
                severity="info",
                supports="natural",
                value=f"{exp:.2f}",
                threshold="natural: -2.2 to -2.6",
                explanation="Spectrum steeper than typical natural photos; may indicate heavy compression or specific camera processing.",
            ))

    # 2. Spectral peaks
    if freq.n_peaks_significant >= 3 and freq.max_peak_db >= cfg.peak_db_threshold:
        findings.append(Finding(
            code="FFT_PEAKS",
            title="Significant spectral peaks",
            severity="strong",
            supports="ai",
            value=f"{freq.n_peaks_significant} peaks, max {freq.max_peak_db:.1f} dB",
            threshold=f">{cfg.peak_db_threshold} dB above median",
            explanation=f"Found {freq.n_peaks_significant} localized spectral peaks exceeding {cfg.peak_db_threshold} dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.",
        ))
    elif freq.max_peak_db > 6:
        findings.append(Finding(
            code="FFT_PEAKS_WEAK",
            title="Moderate spectral peaks",
            severity="info",
            supports="neutral",
            value=f"max {freq.max_peak_db:.1f} dB",
            threshold=f">{cfg.peak_db_threshold} dB",
            explanation="Some spectral peaks present but below strong-evidence threshold.",
        ))

    # 3. Lattice periodicity
    if freq.best_lattice_db >= cfg.lattice_db_threshold and freq.best_lattice_period:
        findings.append(Finding(
            code="FFT_LATTICE",
            title="Lattice artifact detected",
            severity="strong",
            supports="ai",
            value=f"period={freq.best_lattice_period}px, {freq.best_lattice_db:.1f} dB",
            threshold=f">{cfg.lattice_db_threshold} dB",
            explanation=f"Energy concentrated at frequency grid consistent with period-{freq.best_lattice_period} upsampling lattice ({freq.best_lattice_db:.1f} dB). Strongly indicative of transposed convolution in GAN generators.",
        ))

    # 4. HF energy ratio
    hf = freq.hf_energy_ratio
    if hf is not None:
        if hf > 0.10:
            findings.append(Finding(
                code="HF_HIGH",
                title="Excessive high-frequency energy",
                severity="warning",
                supports="ai",
                value=f"{hf:.4f}",
                threshold="> 0.10",
                explanation=f"Top-octave energy ratio ({hf:.4f}) significantly exceeds natural photo range (~0.001-0.06). Suggests synthetic high-frequency content (checkerboard, ringing).",
            ))
        elif hf < 0.0004:
            findings.append(Finding(
                code="HF_LOW",
                title="Unusually low high-frequency energy",
                severity="warning",
                supports="ai",
                value=f"{hf:.4f}",
                threshold="< 0.0004",
                explanation=f"High-frequency energy extremely suppressed ({hf:.4f}), indicating over-smoothing typical of some diffusion/VAE outputs.",
            ))

    # 5. Noise uniformity
    if noise.spatial_uniformity > cfg.uniformity_ai:
        findings.append(Finding(
            code="NOISE_UNIFORM",
            title="Artificially uniform noise",
            severity="warning",
            supports="ai",
            value=f"{noise.spatial_uniformity:.3f}",
            threshold=f">{cfg.uniformity_ai:.2f}",
            explanation=f"Spatial noise uniformity ({noise.spatial_uniformity:.3f}) exceeds natural range. Real sensors exhibit spatially varying noise (PRNU); AI generators often produce spatially homogeneous noise.",
        ))

    # 6. Sensor periodicity
    if noise.sensor_periodicity_db < cfg.sensor_period_db_natural:
        findings.append(Finding(
            code="SENSOR_PERIOD_LOW",
            title="Low sensor-pattern periodicity",
            severity="info",
            supports="ai",
            value=f"{noise.sensor_periodicity_db:.1f} dB",
            threshold=f"< {cfg.sensor_period_db_natural} dB",
            explanation=f"Noise residual lacks the period-2 Bayer CFA periodicity ({noise.sensor_periodicity_db:.1f} dB). Consistent with computer-generated imagery lacking sensor hardware fingerprint.",
        ))

    # 7. Noise sigma
    if noise.noise_sigma < cfg.noise_sigma_low:
        findings.append(Finding(
            code="NOISE_LOW",
            title="Very low noise level",
            severity="info",
            supports="ai",
            value=f"{noise.noise_sigma:.2f}",
            threshold=f"< {cfg.noise_sigma_low}",
            explanation="Unusually clean image with minimal sensor noise; may indicate synthetic origin or heavy denoising.",
        ))
    elif noise.noise_sigma > cfg.noise_sigma_high:
        findings.append(Finding(
            code="NOISE_HIGH",
            title="High noise level",
            severity="info",
            supports="natural",
            value=f"{noise.noise_sigma:.2f}",
            threshold=f"> {cfg.noise_sigma_high}",
            explanation="Elevated noise consistent with high-ISO photography or film grain.",
        ))

    # 8. LBP entropy
    if texture.lbp_entropy < cfg.lbp_entropy_ai:
        findings.append(Finding(
            code="LBP_ENTROPY_LOW",
            title="Low LBP texture entropy",
            severity="info",
            supports="ai",
            value=f"{texture.lbp_entropy:.3f}",
            threshold=f"< {cfg.lbp_entropy_ai:.2f}",
            explanation=f"Local Binary Pattern entropy ({texture.lbp_entropy:.3f}) below natural texture complexity. Synthetic images often have overly regular micro-texture.",
        ))

    # 9. GLCM contrast
    if texture.glcm_contrast < cfg.glcm_contrast_ai:
        findings.append(Finding(
            code="GLCM_CONTRAST_LOW",
            title="Low GLCM contrast",
            severity="info",
            supports="ai",
            value=f"{texture.glcm_contrast:.3f}",
            threshold=f"< {cfg.glcm_contrast_ai:.2f}",
            explanation=f"Gray-level co-occurrence contrast ({texture.glcm_contrast:.3f}) suggests reduced local variation compared to natural textures.",
        ))

    # 10. Fractal dimension
    if texture.fractal_dimension is not None:
        if texture.fractal_dimension < cfg.fractal_dim_low:
            findings.append(Finding(
                code="FRACTAL_LOW",
                title="Low fractal dimension (over-smooth)",
                severity="warning",
                supports="ai",
                value=f"{texture.fractal_dimension:.3f}",
                threshold=f"< {cfg.fractal_dim_low:.2f}",
                explanation=f"Box-counting fractal dimension ({texture.fractal_dimension:.3f}) below natural range ({cfg.fractal_dim_low}-{cfg.fractal_dim_high}). Indicates geometrically simplified edges/textures.",
            ))
        elif texture.fractal_dimension > cfg.fractal_dim_high:
            findings.append(Finding(
                code="FRACTAL_HIGH",
                title="High fractal dimension (complex edges)",
                severity="info",
                supports="natural",
                value=f"{texture.fractal_dimension:.3f}",
                threshold=f"> {cfg.fractal_dim_high:.2f}",
                explanation="Fractal dimension in upper natural range; fine natural detail.",
            ))

    # 11. Color banding
    if color.banded_fraction > cfg.banded_fraction_ai:
        findings.append(Finding(
            code="COLOR_BANDING",
            title="Color quantization banding",
            severity="warning",
            supports="ai",
            value=f"{color.banded_fraction:.3f}",
            threshold=f">{cfg.banded_fraction_ai:.2f}",
            explanation=f"Detected banding in smooth regions ({color.banded_fraction:.1%} of smooth pixels). Suggests limited color depth or posterization from generation pipeline.",
        ))

    # 12. Color cast
    if abs(color.gray_region_rb_cast) > cfg.cast_threshold:
        findings.append(Finding(
            code="COLOR_CAST",
            title="Color cast in neutral regions",
            severity="info",
            supports="ai",
            value=f"{color.gray_region_rb_cast:.2f}",
            threshold=f"±{cfg.cast_threshold}",
            explanation=f"Near-gray pixels show R-B offset of {color.gray_region_rb_cast:.2f}. May indicate color-space artifacts from generative models.",
        ))

    # 13. JPEG double compression / quantization
    if jpeg.is_jpeg:
        if jpeg.histogram_quantization_step and jpeg.histogram_quantization_step > 1:
            findings.append(Finding(
                code="JPEG_QUANT_STEP",
                title="Histogram quantization step detected",
                severity="info",
                supports="neutral",
                value=f"step={jpeg.histogram_quantization_step:.1f}",
                threshold="> 1",
                explanation=f"Grayscale histogram shows regular spacing of {jpeg.histogram_quantization_step:.1f} levels, suggesting quantization/posterization.",
            ))
        if jpeg.re_encode_size_ratio and jpeg.re_encode_size_ratio < 0.85:
            findings.append(Finding(
                code="JPEG_REENCODE",
                title="Re-encoding size anomaly",
                severity="info",
                supports="neutral",
                value=f"{jpeg.re_encode_size_ratio:.2f}",
                threshold="< 0.85",
                explanation="File size differs significantly from re-encoding at estimated quality; may indicate double compression or non-standard quantization.",
            ))

    return findings