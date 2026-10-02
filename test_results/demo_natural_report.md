# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:30:00.190934Z  

## Image
- **File:** demo_out/fixtures/fixture_natural.png
- **Dimensions:** 512×512
- **Mode:** RGB
- **Size:** 434,958 bytes

## Verdict
- **AI Probability:** 11.5%
- **Verdict:** likely_natural

## Attribution
- **Family:** GAN
- **Score:** 0.433
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 0.433 (lattice=0.52, peaks=0.30)
- Diffusion score: 0.371 (shallow_exp=0.00, mid_plateau=0.74)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (0.433)

**Verdict: Likely natural photograph** (AI probability 11.5%)

**Model family attribution:** GAN (score: 0.433)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (13.3 dB). Strongly indicative of transposed convolution in GAN generators.
- Artificially uniform noise: Spatial noise uniformity (0.969) exceeds natural range. Real sensors exhibit spatially varying noise (PRNU); AI generators often produce spatially homogeneous noise.

**Evidence for natural origin:**
- High fractal dimension (complex edges): Fractal dimension in upper natural range; fine natural detail.

**Metric summary:**
- Power-law exponent (alpha): -2.30
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (13.3 dB)
- HF energy ratio: 0.0297
- Noise sigma: 6.77
- Noise spatial uniformity: 0.969
- Sensor periodicity: 2.0 dB
- Fractal dimension: 1.771
- Color banding fraction: 0.003

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*