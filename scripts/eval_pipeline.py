#!/usr/bin/env python3
"""Run the FULL AIImageAnalyzer on a labeled parquet benchmark and save every detector's score (CSV)."""
import argparse, csv, glob, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_image_analyzer import AIImageAnalyzer

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--per-source", type=int, default=30)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = AIImageAnalyzer(use_clip=False, binary_checkpoint=a.checkpoint)
    seen, rows = {}, []
    for f in sorted(glob.glob(a.parquet_glob)):
        for batch in pq.ParquetFile(f).iter_batches(batch_size=8, columns=["image", "label", "source"]):
            for r in batch.to_pylist():
                if seen.get(r["source"], 0) >= a.per_source: continue
                seen[r["source"]] = seen.get(r["source"], 0) + 1
                rep = an.analyze(r["image"]["bytes"])
                row = {"source": r["source"], "label": r["label"], "ensemble": rep.ai_probability, "verdict": rep.verdict}
                for k, v in rep.detectors.items(): row[k] = v["score"] if getattr(v["status"], "value", v["status"]) == "ok" else ""
                rows.append(row)
        print(len(rows), flush=True)
    keys = sorted({k for r in rows for k in r})
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
    print("DONE")
