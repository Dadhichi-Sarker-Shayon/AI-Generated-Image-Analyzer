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
    method: str = "heuristic"  # "trained" (ONNX generator classifier) | "heuristic" (unvalidated rules)
    top_generators: list[tuple[str, float]] = []  # trained method only: most likely generators with probabilities


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


# Generator classes of the bundled attribution model -> display name and model family.
GENERATOR_INFO = {
    "adm": ("ADM (guided diffusion)", "diffusion"),
    "biggan": ("BigGAN", "gan"),
    "dalle3": ("DALL-E 3", "diffusion"),
    "glide": ("GLIDE", "diffusion"),
    "midjourney": ("Midjourney", "diffusion"),
    "sd15": ("Stable Diffusion 1.x", "diffusion"),
    "sd21": ("Stable Diffusion 2.1", "diffusion"),
    "sd3": ("Stable Diffusion 3", "diffusion"),
    "sdxl": ("Stable Diffusion XL", "diffusion"),
    "vqdm": ("VQ-Diffusion", "diffusion"),
    "wukong": ("Wukong", "diffusion"),
}


def attribute_trained(attributor, rgb, ai_probability: float) -> AttributionResult:
    """Generator attribution with the trained classifier. Abstains when unsure or when the image is not judged AI."""
    gen, conf, probs = attributor.attribute(rgb)
    top = sorted(probs.items(), key=lambda kv: -kv[1])[:3]
    fam_scores: dict[str, float] = {}
    for g, pr in probs.items():
        fam = GENERATOR_INFO.get(g, (g, "unknown"))[1]
        fam_scores[fam] = fam_scores.get(fam, 0.0) + pr
    info = attributor.info
    metrics = info.get("metrics", {})
    known = ", ".join(GENERATOR_INFO[g][0] for g in attributor.classes if g in GENERATOR_INFO)
    reasoning = ["Top candidates: " + "; ".join(f"{GENERATOR_INFO.get(g, (g,))[0]} {pr:.0%}" for g, pr in top)]
    if "closed_set_acc_same_pipeline" in metrics:
        reasoning.append(
            f"Reliability: on unseen images of generators it knows, it picks the right one {metrics['closed_set_acc_same_pipeline']:.0%} "
            f"of the time when images come from the same pipeline as its training data, but only "
            f"{metrics['closed_set_acc_other_pipeline']:.0%} on a different benchmark. It therefore abstains below "
            f"{attributor.min_confidence:.0%} confidence (at that level it answers ~{metrics['at_min_confidence_known_answered']:.0%} of "
            f"images with {metrics['at_min_confidence_known_precision']:.0%} precision, and still wrongly names "
            f"{metrics['at_min_confidence_unseen_wrongly_named']:.0%} of images from generators it has never seen).")
    reasoning.append(f"It can only recognize: {known}. Images from other generators (e.g. FLUX, Imagen, Firefly) "
                     f"cannot be named correctly.")
    if ai_probability < 0.5:
        return AttributionResult(family="unknown", family_scores=fam_scores, method="trained", top_generators=top,
                                 specific_model="n/a (image not judged AI-generated)",
                                 reasoning=["Attribution is only meaningful for images judged AI-generated."] + reasoning)
    if gen is None:
        return AttributionResult(family="unknown", family_scores=fam_scores, method="trained", top_generators=top,
                                 specific_model=f"not confidently attributable (best guess confidence {conf:.0%} < {attributor.min_confidence:.0%}; "
                                                f"may be a generator the classifier does not know)",
                                 reasoning=reasoning)
    name, family = GENERATOR_INFO.get(gen, (gen, "unknown"))
    return AttributionResult(family=family, family_scores=fam_scores, method="trained", top_generators=top,
                             specific_model=f"{name} (confidence {conf:.0%})", reasoning=reasoning)
