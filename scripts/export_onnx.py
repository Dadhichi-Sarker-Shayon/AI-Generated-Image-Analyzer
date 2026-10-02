#!/usr/bin/env python3
"""
Export CLIP vision encoder to ONNX for deployment.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_image_analyzer.models.onnx_export import export_clip_vision_encoder


def main():
    parser = argparse.ArgumentParser(description="Export CLIP vision encoder to ONNX")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="HF model name")
    parser.add_argument("--out", default="clip_vision.onnx", help="Output ONNX path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    args = parser.parse_args()

    print(f"Exporting {args.model} to {args.out}...")
    try:
        path, size_mb = export_clip_vision_encoder(
            model_name=args.model,
            out_path=args.out,
            device=args.device,
            opset=args.opset,
        )
        print(f"Success! Saved to {path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()