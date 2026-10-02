from __future__ import annotations
from typing import Any
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..analysis.analyzer import AnalysisData
from ..config import AnalysisConfig


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


class EnsembleDetector:
    """Weighted ensemble of detectors with configurable verdict thresholds."""

    def __init__(
        self,
        detectors: list[BaseDetector],
        weights: dict[str, float] | None = None,
        verdict_ai: float = 0.60,
        verdict_natural: float = 0.35,
    ):
        self.detectors = detectors
        self.weights = weights or {d.name: 1.0 for d in detectors}
        self.verdict_ai = verdict_ai
        self.verdict_natural = verdict_natural

    def evaluate(self, data: AnalysisData) -> DetectorResult:
        results = [d.score_safe(data) for d in self.detectors]

        # Filter available with scores
        available = [(r, self.weights.get(r.name, 1.0)) for r in results if r.status == DetectorStatus.OK and r.score is not None]

        if not available:
            return DetectorResult(
                name="ensemble",
                status=DetectorStatus.UNAVAILABLE,
                reasoning=["No detectors produced valid scores"],
            )

        # Renormalize weights
        total_w = sum(w for _, w in available)
        weights_norm = [w / total_w for _, w in available]

        score = sum(r.score * w for (r, _), w in zip(available, weights_norm))
        confidence = sum((r.confidence or 0.5) * w for (r, _), w in zip(available, weights_norm))

        # Verdict
        if score >= self.verdict_ai:
            verdict = "likely_ai"
        elif score <= self.verdict_natural:
            verdict = "likely_natural"
        else:
            verdict = "inconclusive"

        reasoning = [f"Ensemble score: {score:.3f} (verdict: {verdict})"]
        for r, w in available:
            reasoning.append(f"  {r.name}: {r.score:.3f} (w={w:.2f})" + (f" [{r.confidence:.2f}]" if r.confidence else ""))

        return DetectorResult(
            name="ensemble",
            status=DetectorStatus.OK,
            score=score,
            confidence=confidence,
            reasoning=reasoning,
            features={"verdict": verdict, "sub_detectors": {r.name: r.score for r, _ in available}},
        )