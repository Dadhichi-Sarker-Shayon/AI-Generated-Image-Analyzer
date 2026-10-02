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
    version: str = "0.1.0"
    generated_at: str = Field(default_factory=lambda: datetime.datetime.now(datetime.UTC).isoformat().replace("+00:00", "Z"))
    image: ImageMeta
    metrics: dict[str, Any] = Field(default_factory=dict)
    detectors: dict[str, Any] = Field(default_factory=dict)
    ai_probability: float
    verdict: str  # "likely_ai" | "likely_natural" | "inconclusive"
    attribution: AttributionResult
    findings: list[Finding] = Field(default_factory=list)
    explanation: str = ""

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
        lines.append(f"## Verdict")
        lines.append(f"- **AI Probability:** {self.ai_probability:.1%}")
        lines.append(f"- **Verdict:** {self.verdict}")
        lines.append("")
        lines.append(f"## Attribution")
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

    return AnalysisReport(
        image=image_meta,
        metrics=metrics,
        detectors=detectors,
        ai_probability=ai_prob,
        verdict=verdict,
        attribution=attribution,
        findings=findings,
        explanation=explanation,
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
    lines.append(f"**Model family attribution:** {attr.family.upper()} (score: {attr.family_scores.get(attr.family, 0):.3f})")
    if attr.family != "unknown":
        lines.append(f"*Specific model: {attr.specific_model}*")
    lines.append("")

    # Top findings
    ai_findings = [f for f in findings if f.supports == "ai" and f.severity in ("strong", "warning")]
    nat_findings = [f for f in findings if f.supports == "natural"]

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