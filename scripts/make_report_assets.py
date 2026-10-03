#!/usr/bin/env python3
"""Build every figure and table used by docs/REPORT.md from the saved results (nothing is typed in by hand).

Inputs : results/probs_*.csv, results/*.json, results/logs/*.csv, docs/validation/*, src/.../explain/finding_stats.json
Outputs: docs/figures/*.png, results/summary_metrics.json, results/tables.md
"""
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
RES, FIG, DOC = ROOT / "results", ROOT / "docs" / "figures", ROOT / "docs" / "validation"
FIG.mkdir(parents=True, exist_ok=True)

# validated categorical slots 1-3 (blue, orange, aqua) on the light chart surface
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2dd"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 9.5, "axes.axisbelow": True,
})
BENCH = {"validation": ("Validation (selection set, 16,000)", "probs_validation.csv"),
         "defactify": ("Defactify test (unseen, 11,250)", "probs_defactify_test.csv"),
         "synthbuster": ("Synthbuster-plus test (unseen, 2,800)", "probs_synthbuster_plus.csv")}
PRETTY = {"real": "real photos", "coco_real": "real photos (COCO)", "raise1k": "real photos (RAISE)", "sd15": "SD 1.x", "sd21": "SD 2.1",
          "sd3": "SD 3", "sdxl": "SDXL", "dalle3": "DALL-E 3", "midjourney": "Midjourney", "adm": "ADM", "biggan": "BigGAN",
          "glide": "GLIDE", "vqdm": "VQ-Diffusion", "wukong": "Wukong", "dalle2": "DALL-E 2", "firefly": "Firefly",
          "imagen3": "Imagen 3", "FLUX.1-dev": "FLUX.1-dev", "FLUX.1-schnell": "FLUX.1-schnell", "SD3-medium": "SD3-medium",
          "midjourney-v5": "Midjourney v5", "stable-diffusion-1-3": "SD 1.3", "stable-diffusion-1-4": "SD 1.4",
          "stable-diffusion-2": "SD 2", "stable-diffusion-xl": "SDXL"}
REAL = {"real", "coco_real", "raise1k"}
rng = np.random.default_rng(0)


def boot(y, p, fn, n=1000):
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 2:
            continue
        vals.append(fn(y[i], p[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def save(fig, name):
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def clean_logs():
    """Reduce the raw tqdm-polluted training logs to per-epoch tables."""
    out = {}
    for tag, src in (("detector", "C:/ai_data/train_v2.log"), ("attribution", "C:/ai_data/train_attr.log")):
        if not Path(src).exists():
            continue
        text = Path(src).read_text(encoding="utf8", errors="ignore").replace("\r", "\n")
        rows, cur = [], {}
        for line in text.split("\n"):
            m = re.match(r"Train: Loss=([\d.]+), Acc=([\d.]+)", line)
            if m:
                cur = {"train_loss": float(m[1]), "train_acc": float(m[2])}
            m = re.match(r"Val:\s+Loss=([\d.]+), Acc=([\d.]+), AUC=([\d.]+)", line)
            if m and cur:
                cur.update(val_loss=float(m[1]), val_acc=float(m[2]), val_auc=float(m[3]))
                cur["epoch"] = len(rows) + 1
                rows.append(cur)
                cur = {}
        df = pd.DataFrame(rows)[["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_auc"]]
        (RES / "logs").mkdir(exist_ok=True)
        df.to_csv(RES / "logs" / f"training_{tag}.csv", index=False)
        out[tag] = df
    return out


def fig_training(logs):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1))
    d, a = logs["detector"], logs["attribution"]
    axes[0].plot(d.epoch, d.val_acc * 100, color=BLUE, lw=2, marker="o", ms=4)
    axes[0].plot(d.epoch, d.train_acc * 100, color=BLUE, lw=1.4, ls=(0, (4, 3)), alpha=.8)
    axes[0].set_title("Detector: accuracy (%)")
    axes[0].text(d.epoch.iloc[-1], d.val_acc.iloc[-1] * 100 + 1.4, "validation", color=INK, ha="right", fontsize=8.5)
    axes[0].text(d.epoch.iloc[-1], d.train_acc.iloc[-1] * 100 - 4.4, "training (dashed)", color=INK2, ha="right", fontsize=8.5)
    axes[1].plot(d.epoch, d.val_auc, color=BLUE, lw=2, marker="o", ms=4)
    axes[1].set_title("Detector: validation AUC")
    axes[2].plot(a.epoch, a.val_acc * 100, color=ORANGE, lw=2, marker="o", ms=4)
    axes[2].plot(a.epoch, a.train_acc * 100, color=ORANGE, lw=1.4, ls=(0, (4, 3)), alpha=.8)
    axes[2].set_title("Attribution (11 classes): accuracy (%)")
    for ax in axes:
        ax.grid(axis="y")
        ax.set_xlabel("epoch (28,000 samples each)")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, "training_curves.png")


def load_probs():
    return {k: pd.read_csv(RES / f) for k, (_, f) in BENCH.items()}


def fig_roc(P):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot([0, 1], [0, 1], color=GRID, lw=1.2)
    summary = {}
    for (k, (title, _)), color, (tx, ty) in zip(BENCH.items(), (BLUE, AQUA, ORANGE), ((.38, .93), (.20, .77), (.30, .58))):
        d = P[k]
        y, p = d.label.values, d.prob.values
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        lo, hi = boot(y, p, roc_auc_score, 300)
        ax.plot(fpr, tpr, color=color, lw=2)
        ax.text(tx, ty, f"{title.split(' (')[0]}  AUC {auc:.3f}", color=INK, fontsize=8.8)
        ax.plot([tx - .03], [ty + .008], marker="s", color=color, ms=6, ls="")
        summary[k] = {"auc": auc, "auc_ci": [lo, hi]}
    ax.set_xlabel("false positive rate (real photos flagged as AI)")
    ax.set_ylabel("true positive rate (AI images caught)")
    ax.set_title("ROC curves of the detector")
    ax.grid(True)
    ax.set_aspect("equal")
    save(fig, "roc_curves.png")
    return summary


def fig_by_generator(P):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1, 0.62, 1.05]})
    for ax, (k, (title, _)) in zip(axes, BENCH.items()):
        d = P[k].copy()
        d["flag"] = (d.prob >= 0.5).astype(float)
        g = d.groupby("source").flag.agg(["mean", "size"]).reset_index()
        g["real"] = g.source.isin(REAL)
        g = g.sort_values(["real", "mean"], ascending=[True, True])
        colors = [ORANGE if r else BLUE for r in g.real]
        y = np.arange(len(g))
        ax.barh(y, g["mean"] * 100, color=colors, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([PRETTY.get(s, s) for s in g.source])
        for yi, (m, n) in enumerate(zip(g["mean"], g["size"])):
            ax.text(m * 100 + 1.2, yi, f"{m * 100:.0f}%", va="center", fontsize=8.5, color=INK)
        ax.axvline(50, color=GRID, lw=1)
        ax.set_xlim(0, 112)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel("% of images flagged as AI")
        ax.grid(axis="x")
    fig.tight_layout()
    fig.text(0.5, -0.02, "blue = AI generators (higher is better)      orange = real photos (lower is better)", ha="center",
             color=INK2, fontsize=9)
    save(fig, "detection_by_generator.png")


def fig_attribution(attr):
    t = pd.DataFrame(attr["threshold_table"])
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for col, color, label, dy in (("known_precision", BLUE, "correct when it answers (known generators)", 0.0),
                                  ("known_answered", AQUA, "share of known-generator images it answers", 14.0),
                                  ("unseen_wrongly_named", ORANGE, "unseen generators wrongly named", 4.0)):
        ax.plot(t.threshold, t[col] * 100, color=color, lw=2, marker="o", ms=4)
        ax.text(t.threshold.iloc[-1] + .01, t[col].iloc[-1] * 100 + dy, label, color=INK, fontsize=8.3, va="center")
    ax.axvline(0.8, color=INK2, lw=1, ls=(0, (3, 3)))
    ax.text(0.785, 55, "shipped threshold 0.80", color=INK2, fontsize=8, rotation=90, va="center", ha="right")
    ax.set_xlim(0, 1.45)
    ax.set_ylim(0, 105)
    ax.set_xlabel("minimum confidence required to name a generator")
    ax.set_ylabel("%")
    ax.set_title("Attribution: trading coverage for precision")
    ax.grid(axis="y")
    save(fig, "attribution_tradeoff.png")


def fig_findings():
    stats = json.loads((ROOT / "src/ai_image_analyzer/explain/finding_stats.json").read_text())["findings"]
    rows = sorted(stats.items(), key=lambda kv: kv[1]["ai_rate"] - kv[1]["real_rate"])
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    y = np.arange(len(rows))
    for yi, (k, v) in enumerate(rows):
        ax.plot([v["real_rate"] * 100, v["ai_rate"] * 100], [yi, yi], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter([v["ai_rate"] * 100 for _, v in rows], y, color=BLUE, s=42, zorder=3, label="fires on AI images")
    ax.scatter([v["real_rate"] * 100 for _, v in rows], y, color=ORANGE, s=42, zorder=3, label="fires on real images")
    ax.set_yticks(y)
    ax.set_yticklabels([k.split("|")[0] for k, _ in rows], fontsize=8.5)
    ax.set_xlim(-3, 103)
    ax.set_xlabel("% of images on which the forensic finding fires (1,195 unseen images)")
    ax.set_title("Hand-written forensic findings do not separate AI from real")
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.45, 0.5), fontsize=9)
    ax.grid(axis="x")
    save(fig, "findings_fire_rates.png")


def fig_metrics():
    out = {}
    for tag, title in (("library_metrics", "Library metrics"), ("extra_descriptors", "Extra descriptors")):
        txt = (DOC / f"metric_analysis_{tag}.txt").read_text(encoding="utf8")
        rows = []
        for line in txt.splitlines():
            m = re.match(r"^(\S+\.\S+|\S+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*(KEEP)?$", line.strip())
            if m and ("." in m[1]):
                rows.append((m[1], float(m[2]), float(m[3])))
        out[tag] = rows
    rows = out["library_metrics"] + out["extra_descriptors"]
    names = [r[0] for r in rows]
    fig, ax = plt.subplots(figsize=(7.4, 8.4))
    y = np.arange(len(rows))
    ax.axvspan(0.35, 0.65, color=GRID, alpha=.55, lw=0)
    ax.axvline(0.5, color=INK2, lw=0.8)
    ax.scatter([r[1] for r in rows], y, color=BLUE, s=26, zorder=3, label="Defactify test")
    ax.scatter([r[2] for r in rows], y, color=ORANGE, s=26, zorder=3, label="Synthbuster-plus test")
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.4)
    ax.invert_yaxis()
    ax.set_xlim(0.2, 0.8)
    ax.set_xlabel("AUC of the single metric (0.5 = no separation; shaded = |AUC-0.5| < 0.15, the keep threshold)")
    ax.set_title("None of ~50 interpretable metrics holds across both benchmarks")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    ax.grid(axis="x")
    save(fig, "metric_auc.png")


def tables(P, attr, roc_summary):
    md = []
    summ = {}
    md.append("| Benchmark | Images | AUC (95% CI) | Accuracy @0.5 (95% CI) | Real kept real | AI caught |\n|---|---|---|---|---|---|")
    for k, (title, _) in BENCH.items():
        d = P[k]
        y, p = d.label.values, d.prob.values
        pred = (p >= 0.5).astype(int)
        acc = (pred == y).mean()
        alo, ahi = boot(y, pred.astype(float), lambda a, b: (a == b).mean(), 400)
        tnr, tpr = (pred[y == 0] == 0).mean(), (pred[y == 1] == 1).mean()
        a = roc_summary[k]
        summ[k] = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_ai": int((y == 1).sum()), "auc": a["auc"], "auc_ci": a["auc_ci"],
                   "acc": float(acc), "acc_ci": [alo, ahi], "real_correct": float(tnr), "ai_caught": float(tpr)}
        md.append(f"| {title} | {len(y):,} | {a['auc']:.3f} ({a['auc_ci'][0]:.3f}-{a['auc_ci'][1]:.3f}) | {acc:.1%} ({alo:.1%}-{ahi:.1%}) | {tnr:.1%} | {tpr:.1%} |")
    md.append("\n**Operating points** (threshold chosen to hit a false-positive rate on that benchmark's own real photos):\n")
    md.append("| Benchmark | TPR @ FPR 1% | TPR @ FPR 5% | TPR @ FPR 10% |\n|---|---|---|---|")
    for k, (title, _) in BENCH.items():
        y, p = P[k].label.values, P[k].prob.values
        fpr, tpr, _ = roc_curve(y, p)
        vals = [float(np.interp(f, fpr, tpr)) for f in (0.01, 0.05, 0.10)]
        summ[k]["tpr_at_fpr"] = dict(zip(("1%", "5%", "10%"), vals))
        md.append(f"| {title.split(' (')[0]} | {vals[0]:.1%} | {vals[1]:.1%} | {vals[2]:.1%} |")
    md.append("\n**Detection rate by source** (share flagged as AI at 0.5; real-photo rows show the false-positive rate):\n")
    for k, (title, _) in BENCH.items():
        d = P[k].copy()
        d["flag"] = (d.prob >= 0.5).astype(float)
        g = d.groupby("source").flag.agg(["mean", "size"]).reset_index()
        g["real"] = g.source.isin(REAL)
        g = g.sort_values(["real", "mean"], ascending=[True, False])
        md.append(f"*{title}*\n\n| Source | n | Flagged as AI |\n|---|---|---|")
        for _, r in g.iterrows():
            md.append(f"| {PRETTY.get(r.source, r.source)}{' (real)' if r.real else ''} | {int(r['size']):,} | {r['mean']:.1%} |")
        md.append("")
    md.append("**Attribution** (closed-set accuracy on unseen images of generators it was trained on):\n")
    md.append("| Benchmark | n | Accuracy |\n|---|---|---|")
    for k, t in (("defactify", "Defactify test (same pipeline as training)"), ("synthbuster", "Synthbuster-plus (different pipeline)")):
        md.append(f"| {t} | {attr[k]['n']} | {attr[k]['closed_set_acc']:.1%} |")
    md.append("\n| Min. confidence | Known-generator images answered | Precision when answered | Unseen generators wrongly named |\n|---|---|---|---|")
    for r in attr["threshold_table"]:
        md.append(f"| {r['threshold']:.2f} | {r['known_answered']:.1%} | {r['known_precision']:.1%} | {r['unseen_wrongly_named']:.1%} |")
    (RES / "tables.md").write_text("\n".join(md), encoding="utf8")
    (RES / "summary_metrics.json").write_text(json.dumps(summ, indent=2), encoding="utf8")


def main():
    logs = clean_logs()
    P = load_probs()
    attr = json.loads((DOC / "attribution_eval.json").read_text())
    if len(logs) == 2:
        fig_training(logs)
    roc = fig_roc(P)
    fig_by_generator(P)
    fig_attribution(attr)
    fig_findings()
    fig_metrics()
    tables(P, attr, roc)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print((RES / "tables.md").read_text(encoding="utf8"))


if __name__ == "__main__":
    main()
