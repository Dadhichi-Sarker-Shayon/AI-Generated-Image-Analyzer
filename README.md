# AI Image Analyzer

Detect AI-generated images and explain exactly why — with pixel-level forensic metrics including FFT/DCT spectral analysis, noise fingerprints, texture statistics, color quantization, and JPEG artifact detection.

## Features

- **Multi-modal detection**: Heuristic frequency-domain detector + optional CLIP zero-shot semantic detector
- **Forensic metrics**: 20+ quantitative metrics (spectral exponent, lattice artifacts, noise uniformity, fractal dimension, color banding, etc.)
- **Model attribution**: Heuristic family classification (GAN / Diffusion / VAE / Unknown)
- **Explainable output**: Human-readable markdown report with exact metric values, thresholds, and reasoning
- **Anomaly heatmaps**: Per-tile spatial localization of synthetic artifacts
- **ONNX export**: Deploy CLIP vision encoder to ONNX for edge/cloud inference
- **Python library**: Clean API for integration into pipelines

## Installation

```bash
pip install -e .                    # Core dependencies
pip install -e ".[ml]"              # With CLIP detector (torch + transformers)
pip install -e ".[ml,onnx,viz]"     # All features
```

Requires Python ≥ 3.10.

## Quick Start

```python
from ai_image_analyzer import AIImageAnalyzer

analyzer = AIImageAnalyzer(use_clip=False)  # Set True for CLIP (requires torch)
report = analyzer.analyze("path/to/image.jpg")

print(report.to_markdown())
print(f"AI probability: {report.ai_probability:.1%}")
print(f"Verdict: {report.verdict}")
print(f"Attribution: {report.attribution.family}")
```

### Command Line

```bash
# Analyze single image
ai-image-analyzer path/to/image.jpg

# With heatmap
ai-image-analyzer path/to/image.jpg --heatmap heatmap.png

# JSON output
ai-image-analyzer path/to/image.jpg --json
```

## Analysis Pipeline

| Module | Metrics | Purpose |
|--------|---------|---------|
| **Frequency** | Power-law exponent (alpha), HF energy ratio, spectral peaks (dB), lattice periods (px) | Detect upsampling grids, ring artifacts, spectral deviations |
| **Noise** | Noise sigma, spatial uniformity, sensor periodicity (dB), HF noise ratio | Identify missing PRNU, artificial noise homogeneity |
| **Texture** | LBP entropy, GLCM contrast/energy/homogeneity, fractal dimension | Measure micro-texture complexity |
| **Color** | Banding fraction, colorfulness, saturation std, gray-region color cast | Detect quantization, color-space artifacts |
| **JPEG** | Quality estimate, quantization table source, re-encode ratio, histogram step | Identify double-compression, posterization |

## Output Structure

```json
{
  "ai_probability": 0.82,
  "verdict": "likely_ai",
  "attribution": {
    "family": "gan",
    "family_scores": {"gan": 0.78, "diffusion": 0.31, "vae": 0.12},
    "specific_model": "unknown (heuristic attribution)"
  },
  "findings": [
    {"code": "FFT_LATTICE", "title": "Lattice artifact detected", "severity": "strong", "supports": "ai", "value": "period=3px, 14.2 dB", "explanation": "..."}
  ],
  "explanation": "Verdict: Likely AI-generated...",
  "metrics": { "frequency": {...}, "noise": {...}, ... }
}
```

## Extending with Trained Detectors

The heuristic detectors are baselines. For production accuracy:

1. **Train a classifier** on GenImage / DiffusionDB / DRCT using the extracted features (`FrequencyMetrics`, `NoiseMetrics`, etc.)
2. **Replace** `FrequencyDetector` with your trained model (implements `BaseDetector`)
3. **Add** model-specific detectors (e.g., `MidjourneyDetector`, `StableDiffusionDetector`)
4. **Calibrate** ensemble weights via validation set

The `DetectorResult` interface makes swapping detectors seamless.

## Configuration

Copy `configs/detector_config.yaml` and adjust thresholds. Key parameters:

```yaml
frequency:
  exponent_center: -2.3      # Natural photo power-law center
  exponent_band: 0.35        # Tolerance band
  peak_db_threshold: 10.0    # Spectral peak significance
  lattice_db_threshold: 8.0  # Lattice artifact significance
ensemble:
  weights:
    frequency: 0.5
    clip: 0.5
  verdict_ai: 0.60
  verdict_natural: 0.35
```

## ONNX Export

```bash
python scripts/export_onnx.py --model openai/clip-vit-base-patch32 --out clip_vision.onnx
```

## Docker

```dockerfile
# Build
docker build -t ai-image-analyzer .

# Run
docker run --rm -v $(pwd)/images:/images ai-image-analyzer /images/test.jpg --json
```

## Project Structure

```
src/ai_image_analyzer/
├── analyzer.py           # Main facade
├── config.py             # YAML-configurable AnalysisConfig
├── io_utils.py           # Image loading
├── analysis/             # Signal processing modules
├── detectors/            # Detector implementations
├── attribution.py        # Model family attribution
├── explain/              # Findings, reports, visualizations
├── models/               # ONNX export utilities
└── __main__.py           # CLI entry point
```

## License

MIT

## Disclaimer

This library provides **heuristic forensic analysis** as a baseline. The default detectors use unsupervised statistical thresholds and zero-shot CLIP — they are **not a substitute for trained classifiers** on your target distribution. For production deployment, train supervised detectors on representative data (GenImage, DRCT, DiffusionDB, etc.) and plug them into the `BaseDetector` interface.