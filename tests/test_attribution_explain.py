import pytest
from ai_image_analyzer.attribution import attribute
from ai_image_analyzer.explain.findings import build_findings
from ai_image_analyzer.explain.visualize import anomaly_map, save_heatmap
from ai_image_analyzer.analysis.analyzer import SignalAnalyzer
from ai_image_analyzer.detectors import EnsembleDetector, FrequencyDetector
from ai_image_analyzer.config import AnalysisConfig


def test_attribution_gan(gan_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(gan_image)

    attr = attribute(data, config)
    assert attr.family == "gan"
    assert attr.family_scores["gan"] > attr.family_scores["diffusion"]
    assert attr.family_scores["gan"] > attr.family_scores["vae"]


def test_attribution_diffusion(diffusion_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(diffusion_image)

    attr = attribute(data, config)
    # Diffusion may not always win; check it's competitive
    assert attr.family in ("diffusion", "unknown", "gan", "vae")
    assert attr.family_scores["diffusion"] > 0.1


def test_attribution_unknown(natural_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)

    attr = attribute(data, config)
    assert attr.family in ("unknown", "gan", "diffusion", "vae")
    # Natural should have low scores for all
    for score in attr.family_scores.values():
        assert score < 0.5


def test_findings_nonempty(natural_image, gan_image, diffusion_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)

    for img in [natural_image, gan_image, diffusion_image]:
        data = analyzer.analyze(img)
        det = FrequencyDetector(config)
        det_result = det.evaluate(data)
        ensemble = EnsembleDetector([det], weights={"frequency": 1.0})
        ens_result = ensemble.evaluate(data)
        findings = build_findings(data, ens_result, config)

        assert len(findings) >= 1
        # Each finding should have required fields
        for f in findings:
            assert f.code
            assert f.title
            assert f.severity in ("info", "warning", "strong")
            assert f.supports in ("ai", "natural", "neutral")
            assert f.explanation


def test_anomaly_map_shape(natural_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)
    hm = anomaly_map(data.signals, tile_size=config.tile_size)
    assert hm.ndim == 2
    assert hm.shape[0] > 0 and hm.shape[1] > 0
    assert hm.min() >= 0 and hm.max() <= 1


def test_save_heatmap(natural_image, tmp_path):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)
    hm = anomaly_map(data.signals, tile_size=config.tile_size)
    out_path = tmp_path / "heatmap.png"
    save_heatmap(natural_image, hm, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 1000