import numpy as np
import pytest

from ai_image_analyzer import AIImageAnalyzer
from ai_image_analyzer.detectors import OnnxBinaryDetector, bundled_model_path

pytestmark = pytest.mark.skipif(bundled_model_path() is None, reason="bundled ONNX weights missing")


def _img(seed=0, size=300):
    return np.random.default_rng(seed).integers(0, 256, (size, size + 40, 3), dtype=np.uint8)


def test_bundled_model_loads_and_outputs_probability():
    det = OnnxBinaryDetector()
    assert det.available
    assert det.model_info["input_size"] == 192
    p = det.predict_proba(_img())
    assert 0.0 <= p <= 1.0


def test_preprocess_contract():
    det = OnnxBinaryDetector()
    det._lazy_load()
    x = det.preprocess(_img())
    assert x.shape == (1, 3, 192, 192) and x.dtype == np.float32


def test_prediction_is_deterministic():
    det = OnnxBinaryDetector()
    assert det.predict_proba(_img(1)) == det.predict_proba(_img(1))


def test_analyzer_uses_bundled_model_by_default():
    an = AIImageAnalyzer(use_clip=False)
    assert "binary_trained" in [d.name for d in an.detectors]
    assert an.ensemble.weights["binary_trained"] > an.ensemble.weights["frequency"]
    rep = an.analyze(_img(2))
    assert "binary_trained" in rep.detectors
    assert rep.findings[0].code == "TRAINED_MODEL"


def test_use_trained_false_gives_heuristics_only():
    an = AIImageAnalyzer(use_clip=False, use_trained=False)
    assert "binary_trained" not in [d.name for d in an.detectors]


def test_pil_image_input(tmp_path):
    from PIL import Image
    rep = AIImageAnalyzer(use_clip=False).analyze(Image.fromarray(_img(3)))
    assert 0.0 <= rep.ai_probability <= 1.0


def test_numpy_dtypes_are_not_corrupted():
    from ai_image_analyzer.io_utils import load_image
    base = _img(4)
    ref, _ = load_image(base)
    f01, _ = load_image(base.astype(np.float32) / 255)
    u16, _ = load_image(base.astype(np.uint16) * 257)
    assert np.abs(f01.astype(int) - ref).max() <= 1
    assert np.abs(u16.astype(int) - ref).max() <= 1


def test_missing_checkpoint_raises():
    with pytest.raises(FileNotFoundError):
        AIImageAnalyzer(use_clip=False, binary_checkpoint="does_not_exist.onnx")


def test_tiny_image_warns():
    with pytest.warns(RuntimeWarning, match="unreliable"):
        AIImageAnalyzer(use_clip=False).analyze(_img(5, size=20)[:20, :20])


# ---- explanation: model evidence map + measured findings ----
def test_evidence_map_sums_exactly_to_logit():
    import math
    from ai_image_analyzer.explain.model_evidence import explain_model_evidence
    det = OnnxBinaryDetector()
    rgb = _img(7, size=256)
    me = explain_model_evidence(det, rgb)
    p = det.predict_proba(rgb)
    assert me.evidence.shape == (6, 6)
    assert abs(me.logit - math.log(p / (1 - p))) < 1e-3      # sum(evidence)+bias == logit_ai - logit_real
    assert abs(me.ai_probability - p) < 1e-6
    for r in me.regions:                                       # regions lie inside the image and support the verdict
        assert 0 <= r.x0 < r.x1 <= rgb.shape[1] and 0 <= r.y0 < r.y1 <= rgb.shape[0]
        assert (r.contribution > 0) == (me.logit > 0)


def test_report_contains_model_evidence_and_heatmap(tmp_path):
    an = AIImageAnalyzer(use_clip=False)
    rep = an.analyze(_img(8))
    assert rep.model_evidence and rep.model_evidence["method"] == "class_activation_decomposition"
    assert "MODEL_EVIDENCE_MAP" in [f.code for f in rep.findings]
    assert "Where the model found its evidence" in rep.explanation
    out = tmp_path / "hm.png"
    _, path = an.analyze_and_save_heatmap(_img(8), out)
    assert out.exists() and out.stat().st_size > 0
    assert AIImageAnalyzer(use_clip=False, explain_model=False).analyze(_img(8)).model_evidence is None


def test_unvalidated_findings_are_not_presented_as_evidence():
    rep = AIImageAnalyzer(use_clip=False).analyze(_img(9, size=512))
    weak = [f for f in rep.findings if f.validated is False]
    assert weak, "expected some heuristic findings to be measured as non-discriminative"
    key = rep.explanation.split("**Key evidence for AI generation:**")[-1].split("**Other measurements")[0] \
        if "**Key evidence for AI generation:**" in rep.explanation else ""
    for f in weak:
        assert f.measured_ai_rate is not None and f.measured_real_rate is not None
        assert f"- {f.title}:" not in key
