#!/usr/bin/env python3
"""
Export CLIP vision encoder to ONNX for deployment.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ai_image_analyzer.models.onnx_export import export_clip_vision_encoder, export_detector_onnx


def main():
    parser = argparse.ArgumentParser(description="Export CLIP vision encoder to ONNX")
    parser.add_argument("--model", default="openai/clip-vit-base-patch32", help="HF model name")
    parser.add_argument("--out", default="clip_vision.onnx", help="Output ONNX path")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--detector-checkpoint", help="Export a trained real-vs-AI .pt detector instead of CLIP")
    args = parser.parse_args()

    if args.detector_checkpoint:
        import json
        meta = {"metrics": {"val_auc": 0.9831, "val_acc": 0.9455, "synthbuster_plus_auc": 0.8555, "synthbuster_plus_acc": 0.7057}}
        path, size_mb, diff = export_detector_onnx(args.detector_checkpoint, args.out, metadata=meta, opset=max(args.opset, 17))
        print(f"Saved {path} ({size_mb:.1f} MB), max |onnx - torch| = {diff:.2e}")
        return

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