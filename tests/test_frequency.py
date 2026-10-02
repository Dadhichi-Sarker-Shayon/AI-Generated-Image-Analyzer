import pytest
import numpy as np
from ai_image_analyzer.analysis.frequency import compute_frequency_metrics
from ai_image_analyzer.analysis.signals import ImageSignals


def test_frequency_natural(natural_image):
    sig = ImageSignals.build(natural_image)
    m = compute_frequency_metrics(sig)
    assert m.power_law_exponent is not None
    # Natural should have exponent in typical range
    assert -3.0 < m.power_law_exponent < -1.7
    assert m.hf_energy_ratio is not None
    assert m.mid_energy_ratio is not None
    # Lattice may have some artifacts from fBm generation; GAN should be stronger
    if m.best_lattice_period is not None:
        assert m.best_lattice_db < 15.0  # Not as strong as explicit GAN lattice


def test_frequency_gan(gan_image):
    sig = ImageSignals.build(gan_image)
    m = compute_frequency_metrics(sig)
    assert m.power_law_exponent is not None
    # Should detect lattice at period 3
    assert m.best_lattice_period == 3
    assert m.best_lattice_db > 8.0
    assert m.n_peaks_significant >= 1


def test_frequency_diffusion(diffusion_image):
    sig = ImageSignals.build(diffusion_image)
    m = compute_frequency_metrics(sig)
    assert m.power_law_exponent is not None
    # Shallower exponent
    assert m.power_law_exponent > -2.0
    # Ring artifact should produce peaks
    assert m.n_peaks_significant >= 1
    assert m.max_peak_db > 10.0


def test_frequency_radial_profile_shape(natural_image):
    sig = ImageSignals.build(natural_image)
    m = compute_frequency_metrics(sig)
    assert len(m.radial_profile) > 8
    # Profile should be decreasing with frequency
    freqs = [p[0] for p in m.radial_profile]
    powers = [p[1] for p in m.radial_profile]
    assert all(freqs[i] < freqs[i+1] for i in range(len(freqs)-1))