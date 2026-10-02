from __future__ import annotations
import io
import numpy as np
from PIL import Image
from pydantic import BaseModel


class JpegMetrics(BaseModel):
    is_jpeg: bool
    estimated_quality: float | None
    quantization_table_source: str | None
    re_encode_size_ratio: float | None
    histogram_quantization_step: float | None


def compute_jpeg_metrics(source_bytes: bytes | None) -> JpegMetrics:
    if not source_bytes:
        return JpegMetrics(
            is_jpeg=False,
            estimated_quality=None,
            quantization_table_source=None,
            re_encode_size_ratio=None,
            histogram_quantization_step=None,
        )

    try:
        im = Image.open(io.BytesIO(source_bytes))
        if im.format != "JPEG":
            return JpegMetrics(
                is_jpeg=False,
                estimated_quality=None,
                quantization_table_source=None,
                re_encode_size_ratio=None,
                histogram_quantization_step=None,
            )

        # Quality from JPEG metadata (Pillow stores this in info)
        estimated_quality = im.info.get("quality")
        if estimated_quality is None:
            # Fallback: estimate by re-encoding at various qualities and matching file size
            estimated_quality = _estimate_quality_by_size(im, source_bytes)

        # Quantization table analysis
        qtables = im.info.get("jpeg_qtables")
        qt_source = "standard" if qtables and _is_standard_qtable(qtables) else ("custom" if qtables else None)

        # Re-encode size ratio (double-compression hint)
        re_encode_ratio = None
        if estimated_quality:
            buf = io.BytesIO()
            im.save(buf, format="JPEG", quality=int(estimated_quality), subsampling="keep")
            re_encode_ratio = len(buf.getvalue()) / len(source_bytes)

        # Histogram quantization step (posterization detection)
        hist_step = _detect_histogram_quantization(im)

        return JpegMetrics(
            is_jpeg=True,
            estimated_quality=estimated_quality,
            quantization_table_source=qt_source,
            re_encode_size_ratio=re_encode_ratio,
            histogram_quantization_step=hist_step,
        )
    except Exception:
        return JpegMetrics(
            is_jpeg=False,
            estimated_quality=None,
            quantization_table_source=None,
            re_encode_size_ratio=None,
            histogram_quantization_step=None,
        )


def _estimate_quality_by_size(im: Image.Image, original_bytes: bytes) -> float | None:
    """Binary search quality that matches original file size."""
    original_size = len(original_bytes)
    low, high = 10, 95
    best_q = None
    best_diff = float("inf")
    for q in [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]:
        buf = io.BytesIO()
        try:
            im.save(buf, format="JPEG", quality=q, subsampling="keep")
        except Exception:
            continue
        size = len(buf.getvalue())
        diff = abs(size - original_size) / original_size
        if diff < best_diff:
            best_diff = diff
            best_q = q
        if diff < 0.05:
            return float(q)
    return float(best_q) if best_q else None


def _is_standard_qtable(qtables) -> bool:
    """Check if quantization table matches IJG standard."""
    if not qtables:
        return False
    try:
        # Standard luminance Q-table (quality ~75)
        std_lum = np.array([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99],
        ])
        for q in qtables.values():
            if isinstance(q, (list, tuple)) and len(q) >= 64:
                arr = np.array(q).reshape(8, 8)
                if np.allclose(arr, std_lum, atol=2):
                    return True
    except Exception:
        pass
    return False


def _detect_histogram_quantization(im: Image.Image) -> float | None:
    """Detect dominant step in grayscale histogram (posterization)."""
    try:
        gray = im.convert("L")
        arr = np.array(gray, dtype=np.uint8)
        hist = np.bincount(arr.ravel(), minlength=256)
        # Smooth and find peaks
        from scipy.ndimage import gaussian_filter1d
        smooth = gaussian_filter1d(hist.astype(float), sigma=2)
        peaks = np.where((smooth[1:-1] > smooth[:-2]) & (smooth[1:-1] > smooth[2:]) & (smooth[1:-1] > 1.5 * np.median(smooth)))[0] + 1
        if len(peaks) >= 2:
            diffs = np.diff(peaks)
            diffs = diffs[diffs > 0]
            if len(diffs) > 0:
                from scipy import stats
                mode_result = stats.mode(diffs, keepdims=True)
                return float(mode_result.mode[0]) if mode_result.count[0] > 1 else float(np.median(diffs))
    except Exception:
        pass
    return None