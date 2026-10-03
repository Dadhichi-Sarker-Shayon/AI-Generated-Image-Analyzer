#!/usr/bin/env python3
"""Record which forensic findings fire on labeled images (heuristics only) -> JSONL.

Used to measure, per finding code, how often it fires on AI vs real images.
"""
import argparse, glob, json, sys
from pathlib import Path
import pyarrow.parquet as pq
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ai_image_analyzer import AIImageAnalyzer

SCHEMAS = {  # image col, binary-label fn, source col
    "defactify": ("Image", lambda r: int(r["Label_A"]), "Label_B"),
    "synthbuster": ("image", lambda r: int(r["label"]), "source"),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(SCHEMAS), required=True)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--cap-real", type=int, default=400)
    ap.add_argument("--cap-ai-per-source", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    icol, labfn, scol = SCHEMAS[a.dataset]
    an = AIImageAnalyzer(use_clip=False, use_trained=False, explain_model=False)
    seen, n = {}, 0
    with open(a.out, "w") as out:
        for f in sorted(glob.glob(a.parquet_glob)):
            cols = list({icol, scol, "Label_A", "label"} & set(pq.ParquetFile(f).schema_arrow.names))
            for batch in pq.ParquetFile(f).iter_batches(batch_size=16, columns=cols):
                for r in batch.to_pylist():
                    lab, src = labfn(r), str(r[scol])
                    cap = a.cap_real if lab == 0 else a.cap_ai_per_source
                    if seen.get(src, 0) >= cap: continue
                    seen[src] = seen.get(src, 0) + 1
                    rep = an.analyze(r[icol]["bytes"])
                    out.write(json.dumps({"dataset": a.dataset, "source": src, "label": lab,
                                          "findings": [{"code": x.code, "supports": x.supports, "severity": x.severity} for x in rep.findings],
                                          "w": rep.image.width, "h": rep.image.height}) + "\n")
                    n += 1
                    if n % 100 == 0: out.flush(); print(n, flush=True)
    print("DONE", n)
