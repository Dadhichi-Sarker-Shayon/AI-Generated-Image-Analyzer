from __future__ import annotations
from pathlib import Path
from typing import Any, Optional
from .config import AnalysisConfig
from .io_utils import load_image
from .analysis.analyzer import SignalAnalyzer
from .detectors import FrequencyDetector, ClipZeroShotDetector, EnsembleDetector
from .attribution import attribute
from .explain.findings import build_findings
from .explain.report import build_report, AnalysisReport
from .explain.visualize import anomaly_map, save_heatmap


class AIImageAnalyzer:
    """Main facade for AI-generated image detection and forensic analysis."""

    def __init__(
        self,
        config: AnalysisConfig | str | Path | None = None,
        use_clip: bool = True,
        clip_device: Optional[str] = None,
    ):
        if isinstance(config, (str, Path)):
            base_config = AnalysisConfig.from_yaml(config)
        elif config is None:
            base_config = AnalysisConfig.default()
        else:
            base_config = config

        if clip_device:
            # Create new config with updated clip_device (dataclass is frozen)
            config_dict = base_config.to_dict()
            config_dict["clip_device"] = clip_device
            self.config = AnalysisConfig(**config_dict)
        else:
            self.config = base_config

        self.analyzer = SignalAnalyzer(self.config)
        self.detectors = [FrequencyDetector(self.config)]
        if use_clip:
            self.detectors.append(ClipZeroShotDetector(self.config))
        self.ensemble = EnsembleDetector(
            self.detectors,
            weights=self.config.detector_weights,
            verdict_ai=self.config.verdict_ai,
            verdict_natural=self.config.verdict_natural,
        )

    def analyze(self, source: str | Path | bytes | Any) -> AnalysisReport:
        """
        Analyze an image from file path, bytes, or numpy array.
        Returns a complete AnalysisReport with verdict, metrics, and explanation.
        """
        rgb, meta = load_image(source)
        data = self.analyzer.analyze(rgb, meta)

        detector_results = {d.name: d.score_safe(data) for d in self.detectors}
        ensemble_result = self.ensemble.evaluate(data)
        detector_results["ensemble"] = ensemble_result

        attr = attribute(data, self.config)
        findings = build_findings(data, ensemble_result, self.config)
        report = build_report(data, detector_results, ensemble_result, attr, findings, self.config)

        return report

    def analyze_and_save_heatmap(
        self,
        source: str | Path | bytes | Any,
        heatmap_path: str | Path,
        alpha: float = 0.5,
    ) -> tuple[AnalysisReport, str]:
        """Analyze and save anomaly heatmap overlay."""
        report = self.analyze(source)
        # Reload original RGB for visualization
        rgb, _ = load_image(source)
        signals = self.analyzer.analyzer(data=None)._signals if hasattr(self.analyzer, '_signals') else None
        # Rebuild signals for heatmap (could cache but fine)
        from .analysis.signals import ImageSignals
        sig = ImageSignals.build(rgb, max_dim=self.config.max_analysis_dim)
        hm = anomaly_map(sig, tile_size=self.config.tile_size)
        out = save_heatmap(rgb, hm, str(heatmap_path), alpha)
        return report, out

    @property
    def available_detectors(self) -> list[str]:
        return [d.name for d in self.detectors] + ["ensemble"]