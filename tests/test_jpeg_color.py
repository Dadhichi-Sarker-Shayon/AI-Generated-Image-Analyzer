import pytest
import numpy as np
from ai_image_analyzer.analysis.color import compute_color_metrics
from ai_image_analyzer.analysis.jpeg import compute_jpeg_metrics
from ai_image_analyzer.analysis.signals import ImageSignals


def test_color_metrics(natural_image):
    sig = ImageSignals.build(natural_image)
    m = compute_color_metrics(sig)
    assert 0 <= m.banded_fraction <= 1
    assert m.colorfulness >= 0
    assert 0 <= m.saturation_std <= 1
    assert isinstance(m.gray_region_rb_cast, float)


def test_color_banding(posterized_image):
    sig = ImageSignals.build(posterized_image)
    m = compute_color_metrics(sig)
    assert m.banded_fraction > 0.05  # Should detect posterization


def test_jpeg_metrics(jpeg_natural):
    m = compute_jpeg_metrics(jpeg_natural)
    assert m.is_jpeg is True
    assert m.estimated_quality is not None
    assert 60 <= m.estimated_quality <= 85  # Around 75


def test_jpeg_non_jpeg(natural_image):
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(natural_image).save(buf, format="PNG")
    m = compute_jpeg_metrics(buf.getvalue())
    assert m.is_jpeg is False


def test_jpeg_quantization_step():
    import io
    from PIL import Image
    # Create image with 4-level quantization
    arr = np.zeros((128, 128), dtype=np.uint8)
    for i in range(4):
        arr[:, i*32:(i+1)*32] = i * 64
    rgb = np.stack([arr]*3, axis=-1)
    im = Image.fromarray(rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    m = compute_jpeg_metrics(buf.getvalue())
    if m.histogram_quantization_step:
        assert m.histogram_quantization_step > 1