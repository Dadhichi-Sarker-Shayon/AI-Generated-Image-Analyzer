import numpy as np
import pytest

from ai_image_analyzer import AIImageAnalyzer
from ai_image_analyzer.attribution import attribute_trained


class FakeAttributor:
    """Stands in for OnnxGeneratorAttributor with fixed probabilities."""
    classes = ["biggan", "dalle3", "sdxl"]
    info = {"metrics": {"closed_set_acc_unseen_images": 0.7}}

    def __init__(self, probs, min_conf=0.6):
        self._probs, self.min_confidence = probs, min_conf

    def attribute(self, rgb):
        best = max(self._probs, key=self._probs.get)
        conf = self._probs[best]
        return (best if conf >= self.min_confidence else None), conf, self._probs


RGB = np.zeros((64, 64, 3), np.uint8)


def test_confident_attribution_names_generator_and_family():
    a = attribute_trained(FakeAttributor({"biggan": 0.05, "dalle3": 0.9, "sdxl": 0.05}), RGB, ai_probability=0.9)
    assert a.method == "trained" and a.family == "diffusion"
    assert "DALL-E 3" in a.specific_model and "90%" in a.specific_model
    assert a.top_generators[0][0] == "dalle3"


def test_gan_family_for_biggan():
    a = attribute_trained(FakeAttributor({"biggan": 0.8, "dalle3": 0.1, "sdxl": 0.1}), RGB, 0.9)
    assert a.family == "gan"


def test_abstains_when_unsure():
    a = attribute_trained(FakeAttributor({"biggan": 0.3, "dalle3": 0.4, "sdxl": 0.3}), RGB, 0.9)
    assert a.family == "unknown" and "not confidently attributable" in a.specific_model


def test_no_attribution_for_images_judged_natural():
    a = attribute_trained(FakeAttributor({"biggan": 0.05, "dalle3": 0.9, "sdxl": 0.05}), RGB, ai_probability=0.1)
    assert a.family == "unknown" and "not judged AI" in a.specific_model


class FakeDescriber:
    def describe(self, rgb):
        return "a test description"


def test_description_is_attached_to_report_and_markdown():
    an = AIImageAnalyzer(use_clip=False, describer=FakeDescriber())
    rep = an.analyze(np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8))
    assert rep.description == "a test description"
    md = rep.to_markdown()
    assert "## Description" in md and "a test description" in md and "not evidence" in md


def test_no_description_by_default():
    rep = AIImageAnalyzer(use_clip=False).analyze(np.zeros((80, 80, 3), np.uint8))
    assert rep.description is None and "## Description" not in rep.to_markdown()


def test_describe_without_extra_fails_early_with_install_hint(monkeypatch):
    import importlib.util
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a, **k: None if name in ("torch", "transformers") else real(name, *a, **k))
    with pytest.raises(ImportError, match=r"ai-image-analyzer\[describe\]"):
        AIImageAnalyzer(use_clip=False, describe=True)
