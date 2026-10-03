| Benchmark | Images | AUC (95% CI) | Accuracy @0.5 (95% CI) | Real kept real | AI caught |
|---|---|---|---|---|---|
| Validation (selection set, 16,000) | 16,000 | 0.983 (0.981-0.985) | 94.5% (94.1%-94.9%) | 94.6% | 94.5% |
| Defactify test (unseen, 11,250) | 11,250 | 0.974 (0.970-0.977) | 83.9% (83.3%-84.6%) | 97.8% | 81.1% |
| Synthbuster-plus test (unseen, 2,800) | 2,800 | 0.856 (0.832-0.877) | 70.6% (68.9%-72.1%) | 86.5% | 69.3% |

**Operating points** (threshold chosen to hit a false-positive rate on that benchmark's own real photos):

| Benchmark | TPR @ FPR 1% | TPR @ FPR 5% | TPR @ FPR 10% |
|---|---|---|---|
| Validation | 76.1% | 94.2% | 97.0% |
| Defactify test | 70.0% | 90.0% | 95.0% |
| Synthbuster-plus test | 8.7% | 54.5% | 65.9% |

**Detection rate by source** (share flagged as AI at 0.5; real-photo rows show the false-positive rate):

*Validation (selection set, 16,000)*

| Source | n | Flagged as AI |
|---|---|---|
| DALL-E 3 | 1,500 | 99.0% |
| SDXL | 1,500 | 99.0% |
| ADM | 500 | 98.6% |
| BigGAN | 500 | 98.0% |
| SD 2.1 | 1,500 | 96.7% |
| GLIDE | 500 | 96.6% |
| SD 3 | 1,500 | 95.9% |
| Midjourney | 2,000 | 94.0% |
| Wukong | 500 | 85.4% |
| SD 1.x | 500 | 83.2% |
| VQ-Diffusion | 500 | 70.0% |
| real photos (real) | 5,000 | 5.4% |

*Defactify test (unseen, 11,250)*

| Source | n | Flagged as AI |
|---|---|---|
| DALL-E 3 | 1,866 | 90.9% |
| SDXL | 1,882 | 86.1% |
| Midjourney | 1,859 | 79.1% |
| SD 3 | 1,884 | 78.3% |
| SD 2.1 | 1,868 | 71.1% |
| real photos (COCO) (real) | 1,891 | 2.2% |

*Synthbuster-plus test (unseen, 2,800)*

| Source | n | Flagged as AI |
|---|---|---|
| DALL-E 3 | 200 | 98.0% |
| GLIDE | 200 | 96.0% |
| SDXL | 200 | 90.5% |
| FLUX.1-schnell | 200 | 82.5% |
| Midjourney v5 | 200 | 80.5% |
| SD 1.3 | 200 | 68.5% |
| FLUX.1-dev | 200 | 68.0% |
| SD 1.4 | 200 | 66.0% |
| SD3-medium | 200 | 59.5% |
| Firefly | 200 | 52.5% |
| SD 2 | 200 | 51.5% |
| Imagen 3 | 200 | 46.0% |
| DALL-E 2 | 200 | 42.0% |
| real photos (RAISE) (real) | 200 | 13.5% |

**Attribution** (closed-set accuracy on unseen images of generators it was trained on):

| Benchmark | n | Accuracy |
|---|---|---|
| Defactify test (same pipeline as training) | 500 | 78.0% |
| Synthbuster-plus (different pipeline) | 800 | 47.5% |

| Min. confidence | Known-generator images answered | Precision when answered | Unseen generators wrongly named |
|---|---|---|---|
| 0.00 | 100.0% | 59.2% | 100.0% |
| 0.50 | 67.4% | 70.7% | 51.0% |
| 0.60 | 53.0% | 74.6% | 36.2% |
| 0.70 | 39.4% | 80.3% | 21.6% |
| 0.80 | 25.5% | 87.0% | 11.8% |
| 0.90 | 11.2% | 89.7% | 3.6% |
| 0.95 | 3.6% | 91.5% | 1.0% |