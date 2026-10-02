from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from .analyzer import AIImageAnalyzer
from .config import AnalysisConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AI Image Analyzer - Detect AI-generated images with forensic metrics"
    )
    parser.add_argument("image", nargs="?", help="Path to image file (or - for stdin)")
    parser.add_argument("--config", "-c", type=Path, help="Path to YAML config")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON report")
    parser.add_argument("--markdown", "-m", action="store_true", help="Output markdown explanation (default)")
    parser.add_argument("--no-clip", action="store_true", help="Disable CLIP zero-shot detector")
    parser.add_argument("--heatmap", type=Path, help="Save anomaly heatmap to path")
    parser.add_argument("--alpha", type=float, default=0.5, help="Heatmap overlay alpha")
    parser.add_argument("--device", default="cpu", help="Device for CLIP (cpu/cuda)")
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    args = parser.parse_args(argv)

    if not args.image:
        parser.print_help()
        return 1

    # Load image
    source = args.image
    if args.image == "-":
        source = sys.stdin.buffer.read()

    analyzer = AIImageAnalyzer(
        config=args.config,
        use_clip=not args.no_clip,
        clip_device=args.device,
    )

    if args.heatmap:
        report, hm_path = analyzer.analyze_and_save_heatmap(source, args.heatmap, args.alpha)
        print(f"Heatmap saved to {hm_path}", file=sys.stderr)
    else:
        report = analyzer.analyze(source)

    if args.json:
        print(report.model_dump_json(indent=2))
    else:
        print(report.to_markdown())

    return 0


if __name__ == "__main__":
    sys.exit(main())