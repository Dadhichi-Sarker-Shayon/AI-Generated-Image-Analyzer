# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:39.708785Z  

## Image
- **File:** real_test_images/real_nature_1.jpg
- **Dimensions:** 800×534
- **Mode:** RGB
- **Size:** 139,130 bytes

## Verdict
- **AI Probability:** 18.4%
- **Verdict:** likely_natural

## Attribution
- **Family:** GAN
- **Score:** 0.606
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 0.606 (lattice=0.71, peaks=0.45)
- Diffusion score: 0.267 (shallow_exp=0.00, mid_plateau=0.53)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (0.606)

**Verdict: Likely natural photograph** (AI probability 18.4%)

**Model family attribution:** GAN (score: 0.606)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (15.9 dB). Strongly indicative of transposed convolution in GAN generators.
- Color quantization banding: Detected banding in smooth regions (42.9% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Metric summary:**
- Power-law exponent (alpha): -2.42
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (15.9 dB)
- HF energy ratio: 0.0181
- Noise sigma: 9.47
- Noise spatial uniformity: 0.527
- Sensor periodicity: 4.4 dB
- Fractal dimension: 1.396
- Color banding fraction: 0.429

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*