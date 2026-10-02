# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:47.194184Z  

## Image
- **File:** real_test_images/real_portrait_1.jpg
- **Dimensions:** 800×1000
- **Mode:** RGB
- **Size:** 105,846 bytes

## Verdict
- **AI Probability:** 51.9%
- **Verdict:** inconclusive

## Attribution
- **Family:** GAN
- **Score:** 0.782
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 0.782 (lattice=0.97, peaks=0.51)
- Diffusion score: 0.032 (shallow_exp=0.00, mid_plateau=0.06)
- VAE score: 0.139 (smooth=0.20, low_noise=0.00)
- Best match: gan (0.782)

**Verdict: Inconclusive** (AI probability 51.9%)

**Model family attribution:** GAN (score: 0.782)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (19.5 dB). Strongly indicative of transposed convolution in GAN generators.
- Color quantization banding: Detected banding in smooth regions (16.0% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Metric summary:**
- Power-law exponent (alpha): -2.96
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (19.5 dB)
- HF energy ratio: 0.0010
- Noise sigma: 2.12
- Noise spatial uniformity: 0.443
- Sensor periodicity: 3.0 dB
- Fractal dimension: 1.183
- Color banding fraction: 0.160

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*