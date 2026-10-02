# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:52.670849Z  

## Image
- **File:** real_test_images/ai_style_1.jpg
- **Dimensions:** 800×608
- **Mode:** RGB
- **Size:** 26,843 bytes

## Verdict
- **AI Probability:** 74.8%
- **Verdict:** likely_ai

## Attribution
- **Family:** GAN
- **Score:** 1.000
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 1.000 (lattice=1.00, peaks=1.00)
- Diffusion score: 0.018 (shallow_exp=0.00, mid_plateau=0.04)
- VAE score: 0.453 (smooth=0.49, low_noise=0.36)
- Best match: gan (1.000)

**Verdict: Likely AI-generated** (confidence 74.8%)

**Model family attribution:** GAN (score: 1.000)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (36.1 dB). Strongly indicative of transposed convolution in GAN generators.
- Color quantization banding: Detected banding in smooth regions (58.3% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Evidence for natural origin:**
- Steep power-law spectrum: Spectrum steeper than typical natural photos; may indicate heavy compression or specific camera processing.

**Metric summary:**
- Power-law exponent (alpha): -3.09
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (36.1 dB)
- HF energy ratio: 0.0006
- Noise sigma: 0.96
- Noise spatial uniformity: -0.208
- Sensor periodicity: 5.5 dB
- Fractal dimension: 1.163
- Color banding fraction: 0.583

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*