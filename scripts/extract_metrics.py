#!/usr/bin/env python3
"""Extract every continuous forensic metric (plus size info) for labeled images -> CSV. Parallel; heuristics only."""
import argparse, glob, io, sys
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path
import numpy as np, pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

SCHEMAS = {"defactify": ("Image", lambda r: int(r["Label_A"]), "Label_B"),
           "synthbuster": ("image", lambda r: int(r["label"]), "source")}
_an = None
CROP = 0
EXTRA = False


def _init(crop=0, extra=False):
    global _an, CROP, EXTRA
    CROP, EXTRA = crop, extra
    from ai_image_analyzer.analysis.analyzer import SignalAnalyzer
    from ai_image_analyzer.config import AnalysisConfig
    cfg = AnalysisConfig.default()
    cfg = replace(cfg, enable_glcm=True, enable_fractal=True) if hasattr(cfg, "__dataclass_fields__") else cfg
    _an = SignalAnalyzer(cfg)


def work(job):
    ds, src, lab, b = job
    try:
        img = Image.open(io.BytesIO(b)); fmt = img.format; rgb = np.asarray(img.convert("RGB"))
        if CROP:  # identical native-pixel center crop for every image: no resampling, no size shortcut
            h, w = rgb.shape[:2]
            if min(h, w) < CROP: return None
            t, l = (h - CROP) // 2, (w - CROP) // 2
            rgb = np.ascontiguousarray(rgb[t:t + CROP, l:l + CROP])
        if EXTRA:
            from ai_image_analyzer.analysis.extra_features import compute_extra_features
            row = {"dataset": ds, "source": src, "label": lab, "w": rgb.shape[1], "h": rgb.shape[0], "bytes": len(b), "fmt": fmt}
            row.update(compute_extra_features(rgb))
            return row
        d = _an.analyze(rgb, {})
        row = {"dataset": ds, "source": src, "label": lab, "w": rgb.shape[1], "h": rgb.shape[0], "bytes": len(b), "fmt": fmt}
        for g in ("frequency", "noise", "texture", "color", "jpeg"):
            for k, v in getattr(d, g).model_dump().items():
                if isinstance(v, (int, float, bool)) and v is not None: row[f"{g}.{k}"] = float(v)
        return row
    except Exception as e:
        return None


def jobs(ds, pattern, cap_real, cap_ai):
    icol, labfn, scol = SCHEMAS[ds]; seen = {}
    for f in sorted(glob.glob(pattern)):
        names = pq.ParquetFile(f).schema_arrow.names
        cols = list({icol, scol, "Label_A", "label"} & set(names))
        for batch in pq.ParquetFile(f).iter_batches(batch_size=32, columns=cols):
            for r in batch.to_pylist():
                lab, src = labfn(r), str(r[scol])
                if seen.get(src, 0) >= (cap_real if lab == 0 else cap_ai): continue
                seen[src] = seen.get(src, 0) + 1
                yield ds, src, lab, r[icol]["bytes"]


if __name__ == "__main__":
    import csv
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SCHEMAS), required=True)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--cap-real", type=int, default=600)
    ap.add_argument("--cap-ai-per-source", type=int, default=120)
    ap.add_argument("--crop", type=int, default=0, help="native center-crop size (0 = whole image)")
    ap.add_argument("--extra", action="store_true", help="compute extra_features instead of the library metrics")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    with Pool(a.procs, initializer=_init, initargs=(a.crop, a.extra)) as pool:
        for i, r in enumerate(pool.imap_unordered(work, jobs(a.dataset, a.parquet_glob, a.cap_real, a.cap_ai_per_source), chunksize=4)):
            if r: rows.append(r)
            if (i + 1) % 100 == 0: print(i + 1, flush=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("dataset", "source", "label"), k))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    print("DONE", len(rows))
