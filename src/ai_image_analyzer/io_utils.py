from __future__ import annotations
import io
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image


def load_image(source: str | Path | bytes | io.BytesIO | np.ndarray | Image.Image) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Load an image from various sources and return as RGB uint8 array plus metadata.

    Returns:
        (rgb_array, meta_dict) where rgb_array is HxWx3 uint8 RGB,
        meta contains: width, height, mode, file_bytes (if from file/bytes)
    """
    meta: dict[str, Any] = {"source_type": type(source).__name__}

    if isinstance(source, np.ndarray):
        arr = source
        if arr.dtype == np.uint16:
            arr = (arr >> 8).astype(np.uint8)
        elif arr.dtype != np.uint8:
            # floats: [0, 1] range is scaled to 8 bit; values already above 1 are taken as 0-255
            scale = 255.0 if np.issubdtype(arr.dtype, np.floating) and float(np.nanmax(arr)) <= 1.0 else 1.0
            arr = (np.nan_to_num(arr.astype(np.float64)) * scale).clip(0, 255).astype(np.uint8)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)
        meta.update({"width": arr.shape[1], "height": arr.shape[0], "mode": "RGB", "file_bytes": None})
        return arr, meta

    if isinstance(source, Image.Image):
        img = source
        meta["file_bytes"] = None
    elif isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            file_bytes = f.read()
        img = Image.open(io.BytesIO(file_bytes))
        meta["file_bytes"] = file_bytes
        meta["filename"] = str(source)
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source))
        meta["file_bytes"] = source
    elif isinstance(source, io.BytesIO):
        img = Image.open(source)
        meta["file_bytes"] = source.getvalue()
    else:
        raise TypeError(f"Unsupported source type: {type(source)}")

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr = np.array(img, dtype=np.uint8)
    meta.update({"width": arr.shape[1], "height": arr.shape[0], "mode": "RGB"})
    return arr, meta


def to_grayscale(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB uint8 to float32 grayscale [0, 255]."""
    if rgb.ndim == 2:
        return rgb.astype(np.float32)
    if rgb.shape[2] == 3:
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    if rgb.shape[2] == 4:
        return (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    raise ValueError(f"Unexpected shape: {rgb.shape}")


def downscale_to_max(rgb: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image so max dimension <= max_dim using area interpolation."""
    h, w = rgb.shape[:2]
    if max(h, w) <= max_dim:
        return rgb
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    import cv2
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)