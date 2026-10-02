# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:44.395726Z  

## Image
- **File:** real_test_images/real_city_1.jpg
- **Dimensions:** 800×533
- **Mode:** RGB
- **Size:** 115,262 bytes

## Verdict
- **AI Probability:** 37.8%
- **Verdict:** inconclusive

## Attribution
- **Family:** GAN
- **Score:** 1.000
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 1.000 (lattice=1.00, peaks=1.00)
- Diffusion score: 0.433 (shallow_exp=0.00, mid_plateau=0.87)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (1.000)

**Verdict: Inconclusive** (AI probability 37.8%)

**Model family attribution:** GAN (score: 1.000)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-6 upsampling lattice (29.6 dB). Strongly indicative of transposed convolution in GAN generators.
- Color quantization banding: Detected banding in smooth regions (49.0% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Metric summary:**
- Power-law exponent (alpha): -2.22
- Spectral peaks >10dB: 20
- Lattice artifact: period 6px (29.6 dB)
- HF energy ratio: 0.0165
- Noise sigma: 11.88
- Noise spatial uniformity: 0.462
- Sensor periodicity: 12.6 dB
- Fractal dimension: 1.414
- Color banding fraction: 0.490

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*