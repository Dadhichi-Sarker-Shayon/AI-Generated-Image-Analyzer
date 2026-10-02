import pytest
import numpy as np
import cv2


@pytest.fixture(scope="session")
def natural_image():
    """Generate a natural-looking test image (fBm + sensor noise + vignette)."""
    return _generate_fixture("natural", 384, 384)


@pytest.fixture(scope="session")
def diffusion_image():
    """Generate a diffusion-like test image (shallow spectrum + ring + smooth)."""
    return _generate_fixture("diffusion", 384, 384)


@pytest.fixture(scope="session")
def gan_image():
    """Generate a GAN-like test image (fBm + period-3 lattice + noise)."""
    return _generate_fixture("gan", 384, 384)


def _generate_fixture(kind: str, h: int, w: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Grids for spatial patterns
    y_grid, x_grid = np.indices((h, w))

    # Base fBm spectrum
    if kind == "natural":
        beta = 2.3
    elif kind == "diffusion":
        beta = 1.3
    elif kind == "gan":
        beta = 2.3
    else:
        beta = 2.3

    # Generate fBm via inverse FFT
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    fr = np.hypot(fx, fy)
    fr[0, 0] = 1e-6  # avoid div by zero
    amp = fr ** (-beta / 2)
    phase = rng.uniform(0, 2 * np.pi, (h, w))
    F = amp * np.exp(1j * phase)
    F[0, 0] = 0
    img = np.fft.ifft2(F).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if kind == "natural":
        # Add sensor noise + CFA period-2 pattern + vignette
        noise = rng.normal(0, 2.5 / 255, (h, w))
        # Period-2 CFA pattern
        cfa = 0.01 * np.sin(np.pi * y_grid) * np.sin(np.pi * x_grid)
        vignette = 1 - 0.15 * (fr / fr.max())
        img = img * vignette + noise + cfa
    elif kind == "diffusion":
        # Add narrowband ring at f=0.25
        ring_mask = np.exp(-((fr - 0.25) ** 2) / (2 * 0.04 ** 2))
        ring_phase = rng.uniform(0, 2 * np.pi, (h, w))
        ring_F = 0.15 * ring_mask * np.exp(1j * ring_phase)
        ring = np.fft.ifft2(ring_F).real
        img = img + ring
        # Mild blur
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        # Uniform noise
        img = img + rng.normal(0, 1.2 / 255, (h, w))
    elif kind == "gan":
        # Period-3 lattice (sin*sin grid)
        lattice = 0.06 * np.sin(2 * np.pi * x_grid / 3) * np.sin(2 * np.pi * y_grid / 3)
        img = img + lattice
        # Noise
        img = img + rng.normal(0, 3.0 / 255, (h, w))

    img = np.clip(img, 0, 1)
    # Add slight channel variation
    rgb = np.stack([
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


@pytest.fixture(scope="session")
def jpeg_natural(natural_image):
    """Natural fixture encoded as JPEG bytes."""
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(natural_image).save(buf, format="JPEG", quality=75)
    return buf.getvalue()


@pytest.fixture(scope="session")
def posterized_image():
    """Posterized image for banding detection."""
    h, w = 384, 384
    y_grid, x_grid = np.indices((h, w))
    img = (0.3 + 0.4 * np.sin(x_grid * 0.02) * np.sin(y_grid * 0.03))
    img = np.clip(img, 0, 1)
    # Posterize to 8 levels
    img = np.round(img * 7) / 7
    rgb = np.stack([img] * 3, axis=-1)
    return (rgb * 255).astype(np.uint8)