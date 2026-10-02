import pytest
from ai_image_analyzer.analyzer import AIImageAnalyzer
from ai_image_analyzer.config import AnalysisConfig


def test_full_pipeline_natural(natural_image):
    analyzer = AIImageAnalyzer(use_clip=False)
    report = analyzer.analyze(natural_image)

    assert report.ai_probability >= 0
    assert report.verdict in ("likely_ai", "likely_natural", "inconclusive")
    assert report.attribution.family in ("unknown", "gan", "diffusion", "vae")
    assert len(report.findings) >= 1
    assert len(report.explanation) > 100
    assert report.metrics["frequency"]["power_law_exponent"] is not None


def test_full_pipeline_gan(gan_image):
    analyzer = AIImageAnalyzer(use_clip=False)
    report = analyzer.analyze(gan_image)

    assert report.ai_probability > 0.2
    assert report.verdict in ("likely_ai", "inconclusive")
    assert len(report.findings) >= 2


def test_full_pipeline_diffusion(diffusion_image):
    analyzer = AIImageAnalyzer(use_clip=False)
    report = analyzer.analyze(diffusion_image)

    assert report.ai_probability > 0.2
    assert report.verdict in ("likely_ai", "inconclusive")
    assert len(report.findings) >= 2


def test_report_json_serializable(natural_image):
    analyzer = AIImageAnalyzer(use_clip=False)
    report = analyzer.analyze(natural_image)
    # Should not raise
    json_str = report.model_dump_json()
    assert len(json_str) > 1000


def test_report_markdown(natural_image):
    analyzer = AIImageAnalyzer(use_clip=False)
    report = analyzer.analyze(natural_image)
    md = report.to_markdown()
    assert "Verdict" in md
    assert "Metric summary" in md
    assert "Disclaimer" in md