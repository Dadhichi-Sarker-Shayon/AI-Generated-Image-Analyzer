from __future__ import annotations
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import sobel
from skimage.filters.rank import entropy as rank_entropy
from skimage.morphology import disk
from pydantic import BaseModel


class TextureMetrics(BaseModel):
    lbp_entropy: float
    lbp_uniform_ratio: float
    glcm_contrast: float
    glcm_energy: float
    glcm_homogeneity: float
    fractal_dimension: float | None


def compute_texture_metrics(signals: ImageSignals, max_dim: int = 384, enable_fractal: bool = False, enable_glcm: bool = False) -> TextureMetrics:
    gray = signals.gray
    h, w = gray.shape

    # Downscale for texture if needed
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA).astype(np.uint8)
    else:
        gray_small = gray.astype(np.uint8)

    # LBP (fast)
    lbp = local_binary_pattern(gray_small, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist_norm = hist / hist_sum
        entropy = -np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])) / np.log2(256)
        uniform_ratio = float(hist_norm[:37].sum())
    else:
        entropy = 0.0
        uniform_ratio = 1.0

    # GLCM (optional - expensive)
    if enable_glcm:
        levels = 32
        glcm_gray = (gray_small / 255 * (levels - 1)).astype(np.uint8)
        glcm = graycomatrix(glcm_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=levels, symmetric=True, normed=True)
        contrast = float(np.mean(graycoprops(glcm, "contrast")))
        energy = float(np.mean(graycoprops(glcm, "energy")))
        homogeneity = float(np.mean(graycoprops(glcm, "homogeneity")))
    else:
        contrast = energy = homogeneity = 0.0

    # Fractal dimension (optional - expensive)
    if enable_fractal:
        edges = sobel(gray_small)
        edge_thresh = edges > (0.3 * edges.max()) if edges.max() > 0 else edges > 0
        fractal_dim = _box_counting_dimension(edge_thresh)
    else:
        fractal_dim = None

    return TextureMetrics(
        lbp_entropy=float(entropy),
        lbp_uniform_ratio=uniform_ratio,
        glcm_contrast=contrast,
        glcm_energy=energy,
        glcm_homogeneity=homogeneity,
        fractal_dimension=fractal_dim,
    )


def _box_counting_dimension(binary: np.ndarray) -> float | None:
    h, w = binary.shape
    max_box = min(h, w) // 4
    sizes = []
    counts = []
    box = 1
    while box <= max_box:
        boxes_y = (h + box - 1) // box
        boxes_x = (w + box - 1) // box
        # Reshape and check any True in each box
        padded = np.zeros((boxes_y * box, boxes_x * box), dtype=bool)
        padded[:h, :w] = binary
        blocks = padded.reshape(boxes_y, box, boxes_x, box).any(axis=(1, 3))
        n = blocks.sum()
        if n > 0:
            sizes.append(box)
            counts.append(n)
        box *= 2
    if len(sizes) >= 3:
        x = np.log2(sizes)
        y = np.log2(counts)
        coeff = np.polyfit(x, y, 1)
        return float(-coeff[0])
    return None