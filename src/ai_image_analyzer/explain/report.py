from __future__ import annotations
import datetime
from typing import Any
from pydantic import BaseModel, Field
from ..analysis.analyzer import AnalysisData
from ..detectors.base import DetectorResult
from ..attribution import AttributionResult
from ..explain.findings import Finding
from ..config import AnalysisConfig


class ImageMeta(BaseModel):
    filename: str | None = None
    width: int
    height: int
    mode: str
    file_bytes: int | None = None


class AnalysisReport(BaseModel):
    version: str = "0.2.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"))
    image: ImageMeta
    metrics: dict[str, Any] = Field(default_factory=dict)
    detectors: dict[str, Any] = Field(default_factory=dict)
    ai_probability: float
    verdict: str  # "likely_ai" | "likely_natural" | "inconclusive"
    attribution: AttributionResult
    findings: list[Finding] = Field(default_factory=list)
    explanation: str = ""
    description: str | None = None  # content description (optional; not evidence of AI generation)
    model_evidence: dict[str, Any] | None = None  # per-region evidence of the trained model (see explain/model_evidence.py)

    def to_markdown(self) -> str:
        # Build full markdown report with all sections
        lines = []
        lines.append(f"# AI Image Analysis Report")
        lines.append(f"**Version:** {self.version}  ")
        lines.append(f"**Generated:** {self.generated_at}  ")
        lines.append("")
        lines.append(f"## Image")
        lines.append(f"- **File:** {self.image.filename or 'N/A'}")
        lines.append(f"- **Dimensions:** {self.image.width}×{self.image.height}")
        lines.append(f"- **Mode:** {self.image.mode}")
        if self.image.file_bytes:
            lines.append(f"- **Size:** {self.image.file_bytes:,} bytes")
        lines.append("")
        if self.description:
            lines.append(f"## Description")
            lines.append(f"{self.description}  ")
            lines.append("*(What the image shows; this is not evidence about whether it is AI-generated.)*")
            lines.append("")
        lines.append(f"## Verdict")
        lines.append(f"- **AI Probability:** {self.ai_probability:.1%}")
        lines.append(f"- **Verdict:** {self.verdict}")
        lines.append("")
        lines.append("## Attribution" + (" (trained generator classifier)" if self.attribution.method == "trained"
                                         else " (heuristic, unvalidated - not reliable)"))
        lines.append(f"- **Family:** {self.attribution.family.upper()}")
        lines.append(f"- **Score:** {self.attribution.family_scores.get(self.attribution.family, 0):.3f}")
        lines.append(f"- **Specific Model:** {self.attribution.specific_model}")
        lines.append("")
        if self.attribution.reasoning:
            lines.append("### Attribution Reasoning")
            for r in self.attribution.reasoning:
                lines.append(f"- {r}")
            lines.append("")
        lines.append(self.explanation)
        return "\n".join(lines)


def build_report(
    data: AnalysisData,
    detector_results: dict[str, DetectorResult],
    ensemble_result: DetectorResult,
    attribution: AttributionResult,
    findings: list[Finding],
    config: AnalysisConfig | None = None,
) -> AnalysisReport:
    meta = data.image_meta
    image_meta = ImageMeta(
        filename=meta.get("filename"),
        width=meta.get("width", data.signals.shape[1]),
        height=meta.get("height", data.signals.shape[0]),
        mode=meta.get("mode", "RGB"),
        file_bytes=len(meta["file_bytes"]) if meta.get("file_bytes") else None,
    )

    # Flatten metrics for report
    metrics = {
        "frequency": data.frequency.model_dump(),
        "noise": data.noise.model_dump(),
        "texture": data.texture.model_dump(),
        "color": data.color.model_dump(),
        "jpeg": data.jpeg.model_dump(),
    }

    detectors = {name: r.model_dump() for name, r in detector_results.items()}
    detectors["ensemble"] = ensemble_result.model_dump()

    ai_prob = ensemble_result.score or 0.0
    verdict = ensemble_result.features.get("verdict", "inconclusive")

    # Build narrative explanation
    explanation = _build_explanation(ai_prob, verdict, attribution, findings, data)
    me = getattr(data, "model_evidence", None)

    return AnalysisReport(
        image=image_meta,
        metrics=metrics,
        detectors=detectors,
        ai_probability=ai_prob,
        verdict=verdict,
        attribution=attribution,
        findings=findings,
        explanation=explanation,
        model_evidence=me.to_dict() if me is not None else None,
    )


def _build_explanation(
    ai_prob: float,
    verdict: str,
    attr: AttributionResult,
    findings: list[Finding],
    data: AnalysisData,
) -> str:
    lines = []

    # Verdict
    if verdict == "likely_ai":
        lines.append(f"**Verdict: Likely AI-generated** (confidence {ai_prob:.1%})")
    elif verdict == "likely_natural":
        lines.append(f"**Verdict: Likely natural photograph** (AI probability {ai_prob:.1%})")
    else:
        lines.append(f"**Verdict: Inconclusive** (AI probability {ai_prob:.1%})")

    lines.append("")

    # Attribution
    if attr.method == "trained":
        lines.append(f"**Likely generator:** {attr.specific_model}")
        if attr.family != "unknown":
            lines.append(f"*Model family: {attr.family}*")
    else:
        lines.append(f"**Model family attribution (heuristic, unvalidated - e.g. diffusion images are often labeled GAN):** {attr.family.upper()} (score: {attr.family_scores.get(attr.family, 0):.3f})")
        if attr.family != "unknown":
            lines.append(f"*Specific model: {attr.specific_model}*")
    lines.append("")

    me_f = next((f for f in findings if f.code == "MODEL_EVIDENCE_MAP"), None)
    if me_f is not None:
        lines.append(f"**Where the model found its evidence:** {me_f.explanation}")
        lines.append("")

    # Top findings
    # Findings measured as non-discriminative on unseen images are not presented as evidence.
    def _is_evidence(f): return f.validated is not False
    ai_findings = [f for f in findings if f.supports == "ai" and f.severity in ("strong", "warning") and _is_evidence(f)]
    nat_findings = [f for f in findings if f.supports == "natural" and _is_evidence(f)]
    weak = [f for f in findings if f.validated is False]

    if ai_findings:
        lines.append("**Key evidence for AI generation:**")
        for f in ai_findings[:6]:
            lines.append(f"- {f.title}: {f.explanation}")
        lines.append("")

    if nat_findings:
        lines.append("**Evidence for natural origin:**")
        for f in nat_findings[:3]:
            lines.append(f"- {f.title}: {f.explanation}")
        lines.append("")

    if weak:
        lines.append("**Other measurements (fire about as often on real images as on AI images, so they are not evidence on their own):**")
        for f in weak:
            lines.append(f"- {f.title} ({f.value}); fires on {f.measured_ai_rate:.0%} of AI vs {f.measured_real_rate:.0%} of real images")
        lines.append("")

    # Metrics summary
    freq = data.frequency
    noise = data.noise
    lines.append("**Metric summary:**")
    if freq.power_law_exponent is not None:
        lines.append(f"- Power-law exponent (alpha): {freq.power_law_exponent:.2f}")
    lines.append(f"- Spectral peaks >10dB: {freq.n_peaks_significant}")
    if freq.best_lattice_period:
        lines.append(f"- Lattice artifact: period {freq.best_lattice_period}px ({freq.best_lattice_db:.1f} dB)")
    lines.append(f"- HF energy ratio: {freq.hf_energy_ratio:.4f}" if freq.hf_energy_ratio is not None else "- HF energy ratio: N/A")
    lines.append(f"- Noise sigma: {noise.noise_sigma:.2f}")
    lines.append(f"- Noise spatial uniformity: {noise.spatial_uniformity:.3f}")
    lines.append(f"- Sensor periodicity: {noise.sensor_periodicity_db:.1f} dB")
    if data.texture.fractal_dimension is not None:
        lines.append(f"- Fractal dimension: {data.texture.fractal_dimension:.3f}")
    lines.append(f"- Color banding fraction: {data.color.banded_fraction:.3f}")
    lines.append("")

    lines.append("*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*")

    return "\n".join(lines)


def render_markdown(report: AnalysisReport) -> str:
    return report.to_markdown()