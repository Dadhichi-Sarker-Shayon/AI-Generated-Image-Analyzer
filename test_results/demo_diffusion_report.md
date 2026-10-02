# AI Image Analysis Report
**Version:** 0.1.0  
**Generated:** 2026-10-02T08:30:02.221582Z  

## Image
- **File:** demo_out/fixtures/fixture_diffusion.png
- **Dimensions:** 512×512
- **Mode:** RGB
- **Size:** 380,633 bytes

## Verdict
- **AI Probability:** 57.8%
- **Verdict:** inconclusive

## Attribution
- **Family:** GAN
- **Score:** 0.741
- **Specific Model:** unknown (heuristic attribution; plug in trained classifier for model-level attribution)

### Attribution Reasoning
- GAN score: 0.741 (lattice=1.00, peaks=0.35)
- Diffusion score: 0.500 (shallow_exp=0.00, mid_plateau=1.00)
- VAE score: 0.000 (smooth=0.00, low_noise=0.00)
- Best match: gan (0.741)

**Verdict: Inconclusive** (AI probability 57.8%)

**Model family attribution:** GAN (score: 0.741)
*Specific model: unknown (heuristic attribution; plug in trained classifier for model-level attribution)*

**Key evidence for AI generation:**
- Shallow power-law spectrum: Radial power spectrum decays as f^-1.53 (natural photographs typically f^-2.2 to f^-2.6). Shallow decay indicates excess mid-high frequency energy consistent with generative priors.
- Significant spectral peaks: Found 20 localized spectral peaks exceeding 10.0 dB above background. Such peaks suggest periodic upsampling artifacts (transposed convolution lattices) common in GAN architectures.
- Lattice artifact detected: Energy concentrated at frequency grid consistent with period-10 upsampling lattice (24.1 dB). Strongly indicative of transposed convolution in GAN generators.
- Artificially uniform noise: Spatial noise uniformity (0.985) exceeds natural range. Real sensors exhibit spatially varying noise (PRNU); AI generators often produce spatially homogeneous noise.

**Evidence for natural origin:**
- High fractal dimension (complex edges): Fractal dimension in upper natural range; fine natural detail.

**Metric summary:**
- Power-law exponent (alpha): -1.53
- Spectral peaks >10dB: 20
- Lattice artifact: period 10px (24.1 dB)
- HF energy ratio: 0.0125
- Noise sigma: 5.13
- Noise spatial uniformity: 0.985
- Sensor periodicity: -1.8 dB
- Fractal dimension: 1.767
- Color banding fraction: 0.006

*Disclaimer: This analysis uses heuristic forensic metrics. For production use, replace heuristic detectors with trained classifiers on representative data. Heuristic thresholds are starting points and may require calibration.*