#!/usr/bin/env python3
"""AUC of every detector inside the full pipeline, and of candidate ensemble weightings (results/raw/pipeline_scores.csv).

The CSV comes from scripts/eval_pipeline.py (420 Synthbuster-plus images: 30 per source, 30 real).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
d = pd.read_csv(ROOT / "results" / "raw" / "pipeline_scores.csv")
cols = ["binary_trained", "frequency", "exif_forensics", "patch_cnn", "model_lattice"]
out = {"n": int(len(d)), "n_real": int((d.label == 0).sum()), "n_ai": int((d.label == 1).sum()), "detector_auc": {}, "ensembles": {}}
print(f"{len(d)} images ({out['n_real']} real / {out['n_ai']} AI)\n\nAUC per detector:")
for c in cols + ["ensemble"]:
    x = d[["label", c]].dropna()
    out["detector_auc"][c] = {"auc": round(float(roc_auc_score(x.label, x[c])), 3),
                              "mean_score_real": round(float(x[x.label == 0][c].mean()), 3),
                              "mean_score_ai": round(float(x[x.label == 1][c].mean()), 3)}
    print(f"  {c:16s} AUC {out['detector_auc'][c]['auc']:.3f}   mean score real {out['detector_auc'][c]['mean_score_real']:.2f} / AI {out['detector_auc'][c]['mean_score_ai']:.2f}")
schemes = {"original weights (1.5, .5, 1, 1, 1)": [1.5, .5, 1, 1, 1], "trained model only": [1, 0, 0, 0, 0],
           "shipped: trained 4, others .3, patch_cnn 0": [4, .3, .3, 0, .3], "trained 8, freq/lattice .5": [8, .5, 0, 0, .5]}
print("\nEnsemble weightings (order: trained, frequency, exif, patch_cnn, lattice):")
for n, w in schemes.items():
    w = np.array(w, float)
    s = (d[cols].fillna(0).values * w).sum(1) / w.sum()
    row = {"auc": round(float(roc_auc_score(d.label, s)), 3)}
    for t in (0.5, 0.6):
        p = s >= t
        row[f"thr_{t}"] = {"ai_caught": round(float(p[d.label == 1].mean()), 3), "real_ok": round(float((~p[d.label == 0]).mean()), 3),
                           "acc": round(float((p == d.label).mean()), 3)}
    out["ensembles"][n] = row
    print(f"  {n:44s} AUC {row['auc']:.3f} | @0.5 AI caught {row['thr_0.5']['ai_caught']:.0%}, real ok {row['thr_0.5']['real_ok']:.0%} | @0.6 AI caught {row['thr_0.6']['ai_caught']:.0%}, real ok {row['thr_0.6']['real_ok']:.0%}")
(ROOT / "results" / "pipeline_detector_auc.json").write_text(json.dumps(out, indent=2))
