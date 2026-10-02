from __future__ import annotations
import numpy as np
import cv2
from PIL import Image


def anomaly_map(signals, tile_size: int = 64) -> np.ndarray:
    """
    Compute per-tile anomaly score combining local HF energy and noise residual.
    Returns float array [tiles_y, tiles_x] in [0, 1].
    """
    gray = signals.gray
    h, w = signals.shape

    # Local high-frequency energy via Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap_energy = np.abs(lap)

    # Local noise residual
    blur = cv2.GaussianBlur(gray, (3, 3), 1.2)
    noise_res = np.abs(gray - blur)

    tiles_y = (h + tile_size - 1) // tile_size
    tiles_x = (w + tile_size - 1) // tile_size
    map_arr = np.zeros((tiles_y, tiles_x), dtype=np.float32)

    for ty in range(tiles_y):
        y0 = ty * tile_size
        y1 = min(y0 + tile_size, h)
        for tx in range(tiles_x):
            x0 = tx * tile_size
            x1 = min(x0 + tile_size, w)
            if y1 - y0 < 8 or x1 - x0 < 8:
                continue
            le = lap_energy[y0:y1, x0:x1].mean()
            nr = noise_res[y0:y1, x0:x1].mean()
            map_arr[ty, tx] = 0.6 * le + 0.4 * nr

    # Normalize to [0,1] via percentile clipping
    vmin, vmax = np.percentile(map_arr[map_arr > 0], [5, 95]) if (map_arr > 0).any() else (0, 1)
    if vmax > vmin:
        map_arr = np.clip((map_arr - vmin) / (vmax - vmin), 0, 1)
    else:
        map_arr = np.zeros_like(map_arr)

    return map_arr


def save_heatmap(rgb: np.ndarray, heatmap: np.ndarray, out_path: str, alpha: float = 0.5) -> str:
    """Overlay heatmap on image and save. Uses matplotlib if available, else grayscale PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        h, w = rgb.shape[:2]
        th, tw = heatmap.shape
        # Upscale heatmap to image size
        heatmap_up = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap_colored = cm.jet(heatmap_up)[..., :3]  # RGB
        overlay = (rgb.astype(float) / 255 * (1 - alpha) + heatmap_colored * alpha * 255).clip(0, 255).astype(np.uint8)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb)
        axes[0].set_title("Original")
        axes[0].axis("off")
        im = axes[1].imshow(overlay)
        axes[1].set_title("Anomaly Heatmap")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        return out_path
    except Exception:
        # Fallback: save grayscale heatmap upscaled
        heatmap_up = cv2.resize(heatmap, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_CUBIC)
        hm_img = (heatmap_up * 255).astype(np.uint8)
        Image.fromarray(hm_img, mode="L").save(out_path)
        return out_path