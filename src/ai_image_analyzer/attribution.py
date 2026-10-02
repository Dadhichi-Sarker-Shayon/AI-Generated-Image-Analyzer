from __future__ import annotations
from typing import Any
from pydantic import BaseModel
from .analysis.analyzer import AnalysisData
from .config import AnalysisConfig


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class AttributionResult(BaseModel):
    family: str  # "unknown" | "gan" | "diffusion" | "vae"
    family_scores: dict[str, float]
    specific_model: str
    reasoning: list[str]


def attribute(data: AnalysisData, config: AnalysisConfig | None = None) -> AttributionResult:
    """Heuristic model-family attribution from forensic metrics."""
    freq = data.frequency
    noise = data.noise
    texture = data.texture

    # GAN: strong lattice + peaks
    gan_lattice = _clamp01((freq.best_lattice_db - 6) / 14)
    gan_peaks = _clamp01((freq.max_peak_db - 8) / 12)
    gan_score = 0.6 * gan_lattice + 0.4 * gan_peaks

    # Diffusion: shallow exponent + mid-energy plateau
    exp = freq.power_law_exponent or -2.3
    exp_shallow = _clamp01((-2.0 - exp) / 1.2) if exp > -2.0 else 0.0  # shallower than -2.0
    mid_plateau = _clamp01((freq.mid_energy_ratio or 0) / 0.15)  # relative
    diff_score = 0.5 * exp_shallow + 0.5 * mid_plateau

    # VAE: over-smooth (very low HF + very low noise)
    vae_smooth = _clamp01((0.0012 - (freq.hf_energy_ratio or 0)) / 0.0012)
    vae_noise = _clamp01((1.5 - noise.noise_sigma) / 1.5)
    vae_score = 0.7 * vae_smooth + 0.3 * vae_noise

    scores = {
        "gan": gan_score,
        "diffusion": diff_score,
        "vae": vae_score,
    }
    best_family = max(scores, key=scores.get)
    best_score = scores[best_family]

    if best_score < 0.35:
        family = "unknown"
    else:
        family = best_family

    specific_model = "unknown (heuristic attribution; plug in trained classifier for model-level attribution)"

    reasoning = [
        f"GAN score: {gan_score:.3f} (lattice={gan_lattice:.2f}, peaks={gan_peaks:.2f})",
        f"Diffusion score: {diff_score:.3f} (shallow_exp={exp_shallow:.2f}, mid_plateau={mid_plateau:.2f})",
        f"VAE score: {vae_score:.3f} (smooth={vae_smooth:.2f}, low_noise={vae_noise:.2f})",
        f"Best match: {family} ({best_score:.3f})",
    ]

    return AttributionResult(
        family=family,
        family_scores=scores,
        specific_model=specific_model,
        reasoning=reasoning,
    )