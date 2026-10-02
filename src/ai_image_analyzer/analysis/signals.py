from __future__ import annotations
import dataclasses
from typing import Any
import numpy as np
from scipy import fft


@dataclasses.dataclass
class ImageSignals:
    """Precomputed signal representations for efficient multi-analysis."""
    rgb: np.ndarray
    gray: np.ndarray
    gray_centered: np.ndarray
    F: np.ndarray
    power: np.ndarray
    freq_x: np.ndarray
    freq_y: np.ndarray
    freq_r: np.ndarray
    shape: tuple[int, int]

    @classmethod
    def build(cls, rgb: np.ndarray, max_dim: int | None = None) -> "ImageSignals":
        h, w = rgb.shape[:2]
        if max_dim and max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            import cv2
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            h, w = new_h, new_w

        gray = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
        gray_centered = gray - gray.mean()

        F = fft.fftshift(fft.fft2(gray_centered))
        power = np.abs(F) ** 2

        fx = fft.fftshift(fft.fftfreq(w, d=1.0))
        fy = fft.fftshift(fft.fftfreq(h, d=1.0))
        freq_x, freq_y = np.meshgrid(fx, fy)
        freq_r = np.hypot(freq_x, freq_y)

        return cls(
            rgb=rgb,
            gray=gray,
            gray_centered=gray_centered,
            F=F,
            power=power,
            freq_x=freq_x,
            freq_y=freq_y,
            freq_r=freq_r,
            shape=(h, w),
        )

    def tile_coords(self, tile_size: int) -> list[tuple[int, int, int, int]]:
        """Return list of (y0, y1, x0, x1) for non-overlapping tiles."""
        h, w = self.shape
        tiles = []
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1 = min(y + tile_size, h)
                x1 = min(x + tile_size, w)
                if y1 - y >= 8 and x1 - x >= 8:
                    tiles.append((y, y1, x, x1))
        return tiles