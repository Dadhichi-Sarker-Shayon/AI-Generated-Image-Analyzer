#!/usr/bin/env python3
"""Which continuous forensic metrics really separate AI from real images?

Inputs: metrics_crop_*.csv (controlled native 256px crop) and metrics_*.csv (whole image) from extract_metrics.py.
A metric is KEPT only if its AUC is >= 0.65 (or <= 0.35) in EVERY benchmark with the same direction on the controlled crop.
Thresholds are learned on one benchmark and scored on the other (cross-dataset), never on the data they are tuned on.
Writes src/ai_image_analyzer/explain/metric_calibration.json.
"""
import argparse, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

EXCLUDE = {"label", "w", "h", "bytes"}


def auc(y, x):
    x = np.nan_to_num(np.asarray(x, float), nan=0.0)
    return float(roc_auc_score(y, x)) if len(set(y)) > 1 and np.ptp(x) > 0 else 0.5


def youden(y, x, sign):
    best, t_best = -1, None
    for t in np.unique(np.quantile(x, np.linspace(0.02, 0.98, 97))):
        pred = (sign * x) >= (sign * t)
        j = pred[y == 1].mean() - pred[y == 0].mean()
        if j > best: best, t_best = j, t
    return float(t_best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="checkpoints_v2")
    ap.add_argument("--out", default="src/ai_image_analyzer/explain/metric_calibration.json")
    ap.add_argument("--crop-prefix", default="metrics_crop_")
    ap.add_argument("--whole-prefix", default="metrics_", help="'' to skip the whole-image comparison")
    a = ap.parse_args()
    crop = {d: pd.read_csv(f"{a.dir}/{a.crop_prefix}{d}.csv") for d in ("defactify", "synthbuster")}
    whole = {d: pd.read_csv(f"{a.dir}/{a.whole_prefix or a.crop_prefix}{d}.csv") for d in ("defactify", "synthbuster")}
    metrics = sorted(c for c in crop["defactify"].columns if c not in EXCLUDE and crop["defactify"][c].dtype != object
                     and c not in ("dataset", "source", "fmt") and c in crop["synthbuster"].columns)

    print("== confound check: AUC of image size alone (whole images) ==")
    for d, df in whole.items():
        print(f"  {d}: log(pixels) AUC = {auc(df.label, np.log(df.w * df.h)):.3f}   file-bytes AUC = {auc(df.label, df.bytes):.3f}")
    print("\n== per-metric AUC (>0.5: higher value = AI). crop = controlled 256px crop, whole = full image ==")
    print(f"{'metric':36s}{'crop D':>8s}{'crop S':>8s}{'whole D':>9s}{'whole S':>9s}  keep")
    rows, kept = [], []
    for m in metrics:
        ac = {d: auc(crop[d].label, crop[d][m].fillna(0)) for d in crop}
        aw = {d: auc(whole[d].label, whole[d][m].fillna(0)) if m in whole[d] else float("nan") for d in whole}
        same = (ac["defactify"] - .5) * (ac["synthbuster"] - .5) > 0
        keep = bool(same and min(abs(v - .5) for v in ac.values()) >= 0.15)
        rows.append((m, ac, aw, keep))
        if keep: kept.append(m)
        print(f"{m:36s}{ac['defactify']:8.3f}{ac['synthbuster']:8.3f}{aw['defactify']:9.3f}{aw['synthbuster']:9.3f}  {'KEEP' if keep else ''}")

    cal = {"controlled_crop_px": 256, "datasets": {d: {"real": int((crop[d].label == 0).sum()), "ai": int((crop[d].label == 1).sum())} for d in crop},
           "metrics": {}, "combined": None}
    print("\n== kept metrics: threshold learned on one benchmark, tested on the other ==")
    for m in kept:
        ac = {d: auc(crop[d].label, crop[d][m].fillna(0)) for d in crop}
        sign = 1 if np.mean(list(ac.values())) > .5 else -1
        out = {"direction": "higher_is_ai" if sign == 1 else "lower_is_ai", "auc": {d: round(v, 3) for d, v in ac.items()}, "cross": {}}
        for tr, te in (("defactify", "synthbuster"), ("synthbuster", "defactify")):
            x_tr, y_tr = crop[tr][m].fillna(0).values, crop[tr].label.values
            t = youden(y_tr, x_tr, sign)
            x_te, y_te = crop[te][m].fillna(0).values, crop[te].label.values
            pred = (sign * x_te) >= (sign * t)
            out["cross"][f"{tr}->{te}"] = {"threshold": round(t, 5), "ai_flagged": round(float(pred[y_te == 1].mean()), 3),
                                           "real_flagged": round(float(pred[y_te == 0].mean()), 3)}
        allx = np.concatenate([crop[d][m].fillna(0).values for d in crop]); ally = np.concatenate([crop[d].label.values for d in crop])
        out["threshold"] = round(youden(ally, allx, sign), 5)
        pr = (sign * allx) >= (sign * out["threshold"])
        out["pooled_ai_flagged"], out["pooled_real_flagged"] = round(float(pr[ally == 1].mean()), 3), round(float(pr[ally == 0].mean()), 3)
        cal["metrics"][m] = out
        for k, v in out["cross"].items():
            print(f"  {m:34s} {k:24s} thr={v['threshold']:>10} flags {v['ai_flagged']:.0%} of AI, {v['real_flagged']:.0%} of real")

    print("\n== all metrics together (logistic regression), trained on one benchmark, tested on the other ==")
    cols = [m for m in metrics]
    comb = {}
    for tr, te in (("defactify", "synthbuster"), ("synthbuster", "defactify")):
        sc = StandardScaler().fit(crop[tr][cols].fillna(0))
        lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(crop[tr][cols].fillna(0)), crop[tr].label)
        p = lr.predict_proba(sc.transform(crop[te][cols].fillna(0)))[:, 1]
        comb[f"{tr}->{te}"] = round(float(roc_auc_score(crop[te].label, p)), 3)
        print(f"  {tr} -> {te}: AUC {comb[f'{tr}->{te}']}")
    cal["combined"] = {"cross_dataset_auc": comb}
    json.dump(cal, open(a.out, "w"), indent=2)
    print("\nkept:", kept)
