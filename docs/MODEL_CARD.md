---
license: cc-by-nc-sa-4.0
library_name: onnx
pipeline_tag: image-classification
tags:
  - ai-generated-image-detection
  - image-forensics
  - deepfake
  - onnx
  - efficientnet
  - timm
  - explainability
datasets:
  - Rajarshi-Roy-research/Defactify_Image_Dataset
  - TheKernel01/Tiny-GenImage
  - marco-willi/synthbuster-plus
metrics:
  - accuracy
  - roc_auc
model-index:
  - name: detector_v2 (real vs AI-generated)
    results:
      - task: {type: image-classification, name: Real vs AI-generated detection}
        dataset: {type: Rajarshi-Roy-research/Defactify_Image_Dataset, name: Defactify test (11,250 images, unseen)}
        metrics:
          - {type: roc_auc, value: 0.974, name: AUC}
          - {type: accuracy, value: 0.839, name: Accuracy @0.5}
      - task: {type: image-classification, name: Real vs AI-generated detection}
        dataset: {type: marco-willi/synthbuster-plus, name: Synthbuster-plus test (2,800 images, unseen; 13 generators incl. FLUX, Imagen 3)}
        metrics:
          - {type: roc_auc, value: 0.856, name: AUC}
          - {type: accuracy, value: 0.706, name: Accuracy @0.5}
---

# AI Image Analyzer — weights (alpha, v0.2)

Two small EfficientNet-B0 models used by the [AI Image Analyzer](https://github.com/Dadhichi-Sarker-Shayon/AI-Generated-Image-Analyzer) library:

| File | What it does | Size |
|---|---|---|
| `detector_v2.onnx` (+ `.json`, `.pt`) | real vs AI-generated, plus a 6×6 **evidence map** (exact per-region contribution to the decision) | 15 MB |
| `attribution_v1.onnx` (+ `.json`, `.pt`) | which of 11 generators made an AI image (abstains below 0.80 confidence) | 15 MB |

> **Alpha release — not a reliable detector for arbitrary images.** It is right ~84% of the time on unseen images from generators it was trained on,
> **~71% on a different benchmark with new generators**, and flags some real photos as AI (13.5% of RAISE camera raws).
> Do not use a single score to make decisions about a person or an image's origin. Full analysis: [REPORT.md](https://github.com/Dadhichi-Sarker-Shayon/AI-Generated-Image-Analyzer/blob/main/docs/REPORT.md).

## Use with the library (recommended)

```bash
pip install git+https://github.com/Dadhichi-Sarker-Shayon/AI-Generated-Image-Analyzer.git   # weights are bundled
```
```python
from ai_image_analyzer import AIImageAnalyzer
report = AIImageAnalyzer(use_clip=False).analyze("photo.jpg")
print(report.verdict, report.ai_probability, report.attribution.specific_model)
```

## Use the ONNX file directly

Preprocessing **must** match training: center-crop square → 256 px (Lanczos) → JPEG quality 95 → resize 192 px → ImageNet normalization.

```python
import io, json, numpy as np, onnxruntime as ort
from PIL import Image

meta = json.load(open("detector_v2.json"))

def preprocess(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size; s = min(w, h); left, top = (w - s) // 2, (h - s) // 2
    im = im.crop((left, top, left + s, top + s)).resize((256, 256), Image.LANCZOS)
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=95)
    im = Image.open(buf).convert("RGB").resize((meta["input_size"], meta["input_size"]), Image.BILINEAR)
    x = (np.asarray(im, np.float32) / 255 - np.array(meta["mean"], np.float32)) / np.array(meta["std"], np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)

sess = ort.InferenceSession("detector_v2.onnx", providers=["CPUExecutionProvider"])
probs, evidence = sess.run(None, {"pixel_values": preprocess("photo.jpg")})
print("P(AI) =", float(probs[0, 1]))                      # probs = [P(real), P(AI)]
# evidence: (1, 6, 6); sum(evidence) + meta["evidence_bias"] == logit_AI - logit_real  (exact)
```

`attribution_v1.onnx` takes the same input and returns 11 probabilities in the order of `classes` in `attribution_v1.json`
(`adm, biggan, dalle3, glide, midjourney, sd15, sd21, sd3, sdxl, vqdm, wukong`). Treat the answer as "unknown" if the top probability is below `min_confidence` (0.80).

## Training

- **Architecture:** `efficientnet_b0` (timm), ImageNet-pretrained, 192×192 input, 2-class head (detector) / 11-class head (attribution).
- **Data:** 70,000 training images — Defactify (COCO real vs SD 2.1, SDXL, SD3, DALL-E 3, Midjourney v6; 42,000) + Tiny-GenImage (ImageNet real vs ADM, BigGAN, GLIDE, Midjourney, SD 1.x, VQDM, Wukong; 28,000). 21,000 real / 49,000 AI.
  Every image: center-crop → 256 px → JPEG q95 (removes file-format/resolution shortcuts).
- **Recipe:** AdamW lr 1e-4, batch 32 (16×2 accumulation), label smoothing 0.1, class-balanced sampling, horizontal flip + color jitter, fp16, early stopping on validation AUC (detector: 15 epochs of 28k samples, best epoch 11; attribution: 6 epochs, best epoch 5).
- **Hardware:** one GTX 1650 (4 GB); ~2 h for the detector.

## Evaluation (threshold 0.5; 95% CIs in the report)

| Benchmark | Images | AUC | Accuracy | Real kept real | AI caught |
|---|---|---|---|---|---|
| Validation (used to select the epoch — optimistic) | 16,000 | 0.983 | 94.5% | 94.6% | 94.5% |
| **Defactify test** (unseen images; 5 generators vs COCO) | 11,250 | **0.974** | **83.9%** | 97.8% | 81.1% |
| **Synthbuster-plus test** (unseen; 13 generators incl. FLUX, Imagen 3, Firefly; real = RAISE raws) | 2,800 | **0.856** | **70.6%** | 86.5% | 69.3% |

At a 1% false-positive rate the detector catches 70% (Defactify) and only 9% (Synthbuster-plus) of AI images.
Synthbuster-plus detection by generator: DALL-E 3 98%, GLIDE 96%, SDXL 91%, FLUX.1-schnell 83%, Midjourney v5 81%, SD 1.x 66–69%, FLUX.1-dev 68%,
SD3-medium 60%, Firefly 53%, SD 2 52%, Imagen 3 46%, DALL-E 2 42%.

**Attribution:** 78% correct on unseen images of the training pipeline, **47.5% on a different benchmark**. At the shipped 0.80 threshold it
answers ~26% of images with 87% precision, and names ~12% of images from generators it never saw (e.g. FLUX) with a wrong label.

## Intended use and limitations

**Intended:** research, education, building explainable-forensics prototypes, one weak signal among several.
**Out of scope:** verifying the authenticity of a specific image for legal, journalistic, academic-integrity or moderation decisions; detecting edited
(inpainted) real photos; adversarial settings.

- Generalization to new generators and pipelines is limited (see above); newer high-fidelity generators (Imagen 3, Firefly) are mostly missed.
- False positives occur on real photos, especially raw-camera images (13.5% on RAISE).
- Scores change with re-encoding (same photo: JPEG 0.20, WebP 0.27, grayscale 0.16); robustness to compression/resizing/screenshots was not systematically tested.
- The probability is not calibrated. The input is reduced to 192 px, discarding fine pixel-level artifacts.
- Real and AI images in the training data come from different content distributions (COCO/ImageNet vs prompt-driven images), so content or style may contribute to the signal.
- The evidence map shows where the network's evidence lies (typically spread over the whole image); it is not proof of generation.

## License

**CC BY-NC-SA 4.0** — non-commercial, attribution, share-alike. The weights inherit this from the most restrictive training dataset (Tiny-GenImage, CC BY-NC-SA 4.0);
two other datasets declare no license. This is the authors' conservative reading, not legal advice; see
[LICENSE-WEIGHTS.md](https://github.com/Dadhichi-Sarker-Shayon/AI-Generated-Image-Analyzer/blob/main/LICENSE-WEIGHTS.md). The library code is MIT.

## Acknowledgements

Datasets: GenImage / Tiny-GenImage, Defactify (arXiv:2601.00553), Synthbuster, MS COCO, RAISE. Models/libraries: EfficientNet (Tan & Le 2019), timm, ONNX Runtime.
Details in [NOTICE.md](https://github.com/Dadhichi-Sarker-Shayon/AI-Generated-Image-Analyzer/blob/main/NOTICE.md).
