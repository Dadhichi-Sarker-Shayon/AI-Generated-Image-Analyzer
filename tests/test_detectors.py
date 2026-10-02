import pytest
from ai_image_analyzer.detectors import FrequencyDetector, ClipZeroShotDetector, EnsembleDetector
from ai_image_analyzer.analysis.analyzer import SignalAnalyzer
from ai_image_analyzer.config import AnalysisConfig


def test_frequency_detector_natural(natural_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)

    det = FrequencyDetector(config)
    result = det.evaluate(data)

    assert result.status.name == "OK"
    assert result.score is not None
    assert 0 <= result.score <= 1
    # Natural should score low
    assert result.score < 0.5


def test_frequency_detector_gan(gan_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(gan_image)

    det = FrequencyDetector(config)
    result = det.evaluate(data)

    assert result.status.name == "OK"
    assert result.score is not None
    # GAN should score higher than natural
    assert result.score > 0.25


def test_frequency_detector_diffusion(diffusion_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(diffusion_image)

    det = FrequencyDetector(config)
    result = det.evaluate(data)

    assert result.status.name == "OK"
    assert result.score is not None
    # Diffusion should score higher
    assert result.score > 0.3


def test_clip_detector_unavailable_without_model(natural_image):
    """CLIP detector should be unavailable without network/model."""
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)

    det = ClipZeroShotDetector(config)
    result = det.evaluate(data)

    # Should be unavailable (no network in test env, or lazy load fails)
    # Either UNAVAILABLE or OK if somehow loaded
    assert result.status.name in ("UNAVAILABLE", "OK")
    if result.status.name == "OK":
        assert result.score is not None


def test_ensemble_detector(natural_image, gan_image, diffusion_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)

    detectors = [FrequencyDetector(config)]
    ensemble = EnsembleDetector(detectors, weights={"frequency": 1.0})

    for img, name in [(natural_image, "natural"), (gan_image, "gan"), (diffusion_image, "diffusion")]:
        data = analyzer.analyze(img)
        result = ensemble.evaluate(data)

        assert result.status.name == "OK"
        assert result.score is not None
        assert "verdict" in result.features
        assert result.features["verdict"] in ("likely_ai", "likely_natural", "inconclusive")


def test_ensemble_natural_low_score(natural_image):
    config = AnalysisConfig.default()
    analyzer = SignalAnalyzer(config)
    data = analyzer.analyze(natural_image)

    detectors = [FrequencyDetector(config)]
    ensemble = EnsembleDetector(detectors, weights={"frequency": 1.0})
    result = ensemble.evaluate(data)

    # Natural should tend toward "likely_natural" or "inconclusive"
    assert result.features["verdict"] in ("likely_natural", "inconclusive")
    assert result.score < 0.5