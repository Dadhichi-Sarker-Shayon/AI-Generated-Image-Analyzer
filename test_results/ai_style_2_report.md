# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:56.311249Z  

## Image
- **File:** real_test_images/ai_style_2.jpg
- **Dimensions:** 800×1000
- **Mode:** RGB
- **Size:** 234,512 bytes

## Verdict
- **AI Probability:** 29.0%
- **Verdict:** inconclusive

## Attribution
- **Family:** GAN
- **Score:** 0.877
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 0.877 (lattice=0.80, peaks=1.00)
- Diffusion score: 0.383 (shallow_exp=0.00, mid_plateau=0.77)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (0.877)

**Verdict: Inconclusive** (AI probability 29.0%)

**Model family attribution:** GAN (score: 0.877)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (17.1 dB). Strongly indicative of transposed convolution in GAN generators.
- Color quantization banding: Detected banding in smooth regions (40.7% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Metric summary:**
- Power-law exponent (alpha): -2.27
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (17.1 dB)
- HF energy ratio: 0.0275
- Noise sigma: 12.65
- Noise spatial uniformity: 0.415
- Sensor periodicity: 7.5 dB
- Fractal dimension: 1.420
- Color banding fraction: 0.407

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*