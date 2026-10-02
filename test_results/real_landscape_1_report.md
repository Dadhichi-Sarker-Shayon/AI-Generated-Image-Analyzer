# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:29:50.346055Z  

## Image
- **File:** real_test_images/real_landscape_1.jpg
- **Dimensions:** 800×533
- **Mode:** RGB
- **Size:** 64,487 bytes

## Verdict
- **AI Probability:** 56.5%
- **Verdict:** inconclusive

## Attribution
- **Family:** GAN
- **Score:** 1.000
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 1.000 (lattice=1.00, peaks=1.00)
- Diffusion score: 0.078 (shallow_exp=0.00, mid_plateau=0.16)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (1.000)

**Verdict: Inconclusive** (AI probability 56.5%)

**Model family attribution:** GAN (score: 1.000)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (25.5 dB). Strongly indicative of transposed convolution in GAN generators.
- Low fractal dimension (over-smooth): Box-counting fractal dimension (1.081) below natural range (1.15-1.75). Indicates geometrically simplified edges/textures.
- Color quantization banding: Detected banding in smooth regions (66.9% of smooth pixels). Suggests limited color depth or posterization from generation pipeline.

**Metric summary:**
- Power-law exponent (alpha): -2.80
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (25.5 dB)
- HF energy ratio: 0.0023
- Noise sigma: 3.69
- Noise spatial uniformity: 0.132
- Sensor periodicity: 3.9 dB
- Fractal dimension: 1.081
- Color banding fraction: 0.669

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*