#!/usr/bin/env python3
"""
Demo script: generates test fixtures and runs full analysis pipeline.
Outputs markdown reports and saves anomaly heatmaps.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Add src to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2
from PIL import Image

from ai_image_analyzer.analyzer import AIImageAnalyzer
from ai_image_analyzer.config import AnalysisConfig


def _generate_fixture(kind: str, h: int, w: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic test images with known forensic signatures."""
    rng = np.random.default_rng(seed)
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
    fr[0, 0] = 1e-6
    amp = fr ** (-beta / 2)
    phase = rng.uniform(0, 2 * np.pi, (h, w))
    F = amp * np.exp(1j * phase)
    F[0, 0] = 0
    img = np.fft.ifft2(F).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if kind == "natural":
        noise = rng.normal(0, 2.5 / 255, (h, w))
        cfa = 0.01 * np.sin(np.pi * y_grid) * np.sin(np.pi * x_grid)
        vignette = 1 - 0.15 * (fr / fr.max())
        img = img * vignette + noise + cfa
    elif kind == "diffusion":
        ring_mask = np.exp(-((fr - 0.25) ** 2) / (2 * 0.04 ** 2))
        ring_phase = rng.uniform(0, 2 * np.pi, (h, w))
        ring_F = 0.15 * ring_mask * np.exp(1j * ring_phase)
        ring = np.fft.ifft2(ring_F).real
        img = img + ring
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        img = img + rng.normal(0, 1.2 / 255, (h, w))
    elif kind == "gan":
        lattice = 0.06 * np.sin(2 * np.pi * x_grid / 3) * np.sin(2 * np.pi * y_grid / 3)
        img = img + lattice
        img = img + rng.normal(0, 3.0 / 255, (h, w))

    img = np.clip(img, 0, 1)
    rgb = np.stack([
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def generate_fixtures(out_dir: Path) -> dict[str, Path]:
    """Generate test fixture images and save them."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind in ["natural", "diffusion", "gan"]:
        img = _generate_fixture(kind, 512, 512)
        path = out_dir / f"fixture_{kind}.png"
        Image.fromarray(img).save(path)
        paths[kind] = path
        print(f"Generated {path}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="AI Image Analyzer Demo")
    parser.add_argument("--image", type=Path, help="Path to image to analyze (optional, generates fixtures if omitted)")
    parser.add_argument("--out-dir", type=Path, default=Path("demo_out"), help="Output directory for reports/heatmaps")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP detector")
    parser.add_argument("--device", default="cpu", help="Device for CLIP (cpu/cuda)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AIImageAnalyzer(use_clip=not args.no_clip, clip_device=args.device)

    if args.image:
        images = {"custom": args.image}
    else:
        print("Generating test fixtures...")
        images = generate_fixtures(args.out_dir / "fixtures")

    for name, path in images.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {name} ({path})")
        print(f"{'='*60}")

        # Full analysis
        report = analyzer.analyze(path)
        print(report.to_markdown())

        # Save markdown report
        md_path = args.out_dir / f"report_{name}.md"
        md_path.write_text(report.to_markdown())
        print(f"\nMarkdown report saved to {md_path}")

        # Save JSON report
        json_path = args.out_dir / f"report_{name}.json"
        json_path.write_text(report.model_dump_json(indent=2))
        print(f"JSON report saved to {json_path}")

        # Save heatmap
        hm_path = args.out_dir / f"heatmap_{name}.png"
        try:
            analyzer.analyze_and_save_heatmap(path, hm_path)
            print(f"Heatmap saved to {hm_path}")
        except Exception as e:
            print(f"Heatmap generation failed: {e}")

    print(f"\nAll outputs in {args.out_dir}")


if __name__ == "__main__":
    main()