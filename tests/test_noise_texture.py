import pytest
from ai_image_analyzer.analysis.noise import compute_noise_metrics
from ai_image_analyzer.analysis.texture import compute_texture_metrics
from ai_image_analyzer.analysis.signals import ImageSignals


def test_noise_metrics(natural_image):
    sig = ImageSignals.build(natural_image)
    m = compute_noise_metrics(sig)
    assert m.noise_sigma > 0
    assert 0 <= m.spatial_uniformity <= 1
    assert 0 <= m.noise_hf_ratio <= 1
    assert m.tile_std_cv >= 0


def test_noise_uniformity_ai_like():
    # Synthetic perfectly uniform noise
    import numpy as np
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    img = img + np.random.default_rng(0).normal(0, 5, img.shape).astype(np.uint8)
    sig = ImageSignals.build(img)
    m = compute_noise_metrics(sig, tile_size=32)
    assert m.spatial_uniformity > 0.95  # Very uniform


def test_texture_metrics(natural_image):
    sig = ImageSignals.build(natural_image)
    m = compute_texture_metrics(sig)
    assert 0 <= m.lbp_entropy <= 1
    assert 0 <= m.lbp_uniform_ratio <= 1
    assert m.glcm_contrast >= 0
    assert m.glcm_energy >= 0
    assert m.glcm_homogeneity >= 0
    if m.fractal_dimension is not None:
        assert 1.0 < m.fractal_dimension < 2.0


def test_fractal_dimension_range():
    import numpy as np
    # High-frequency noise -> higher fractal dim
    rng = np.random.default_rng(1)
    noisy = rng.random((256, 256, 3))
    sig = ImageSignals.build((noisy * 255).astype(np.uint8))
    m = compute_texture_metrics(sig)
    if m.fractal_dimension is not None:
        assert m.fractal_dimension > 1.2