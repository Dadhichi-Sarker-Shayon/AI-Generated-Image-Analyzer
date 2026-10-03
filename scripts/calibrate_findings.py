#!/usr/bin/env python3
"""Turn findings_*.jsonl (scripts/eval_findings.py) into src/ai_image_analyzer/explain/finding_stats.json.

A finding counts as *validated evidence* for a direction only if, on unseen images, it fires clearly more often on
that class than on the other (rate ratio >= 1.5, absolute gap >= 5 points, non-overlapping 95% Wilson intervals)
and the direction holds in every benchmark individually.
"""
import argparse, json, math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="src/ai_image_analyzer/explain/finding_stats.json")
    a = ap.parse_args()
    rows = [json.loads(l) for f in a.files for l in open(f)]
    ds_names = sorted({r["dataset"] for r in rows})
    n = defaultdict(int)                      # (dataset|ALL, label) -> images
    fired = defaultdict(int)                  # (dataset|ALL, label, code, supports) -> images where it fired
    keys = set()
    for r in rows:
        for d in (r["dataset"], "ALL"):
            n[(d, r["label"])] += 1
        for k in {(f["code"], f["supports"]) for f in r["findings"]}:
            keys.add(k)
            for d in (r["dataset"], "ALL"):
                fired[(d, r["label"], *k)] += 1
    stats = {"datasets": {d: {"real": n[(d, 0)], "ai": n[(d, 1)]} for d in ds_names + ["ALL"]}, "findings": {}}
    for code, sup in sorted(keys):
        def rate(d, lab): return fired[(d, lab, code, sup)] / max(1, n[(d, lab)])
        ra, rr = rate("ALL", 1), rate("ALL", 0)
        lo_a, hi_a = wilson(fired[("ALL", 1, code, sup)], n[("ALL", 1)])
        lo_r, hi_r = wilson(fired[("ALL", 0, code, sup)], n[("ALL", 0)])
        want_ai = sup == "ai"
        hi_cls, lo_cls = (ra, rr) if want_ai else (rr, ra)           # rate on the class it claims to support / the other
        ci_ok = (lo_a > hi_r) if want_ai else (lo_r > hi_a)
        per_ds = all((rate(d, 1) > rate(d, 0)) if want_ai else (rate(d, 0) > rate(d, 1)) for d in ds_names)
        validated = bool(hi_cls >= 1.5 * lo_cls and hi_cls - lo_cls >= 0.05 and ci_ok and per_ds)
        stats["findings"][f"{code}|{sup}"] = {
            "ai_rate": round(ra, 4), "real_rate": round(rr, 4),
            "ai_ci": [round(lo_a, 4), round(hi_a, 4)], "real_ci": [round(lo_r, 4), round(hi_r, 4)],
            "per_dataset": {d: {"ai_rate": round(rate(d, 1), 4), "real_rate": round(rate(d, 0), 4)} for d in ds_names},
            "validated": validated,
        }
    Path(a.out).write_text(json.dumps(stats, indent=2))
    print(f"{'finding':28s}{'fires on AI':>12s}{'fires on real':>15s}  validated")
    for k, v in sorted(stats["findings"].items(), key=lambda kv: -kv[1]["validated"]):
        print(f"{k:28s}{v['ai_rate']:>11.1%}{v['real_rate']:>15.1%}  {v['validated']}")
    print(stats["datasets"])
