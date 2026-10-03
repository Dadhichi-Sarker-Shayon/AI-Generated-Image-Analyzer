#!/usr/bin/env python3
"""Evaluate a trained generator-attribution checkpoint (.pt) on unseen benchmark images.

Reports (a) closed-set accuracy on generators the model was trained on, (b) what it does on generators it has never seen
(FLUX, Imagen 3, Firefly, DALL-E 2): the share it would wrongly name at each confidence threshold.
"""
import argparse, glob, json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, timm, torch
sys.path.insert(0, str(Path(__file__).parent))
from eval_external import prep

GENS = "adm biggan dalle3 glide midjourney sd15 sd21 sd3 sdxl vqdm wukong".split()
SB_MAP = {"dalle3": "dalle3", "midjourney-v5": "midjourney", "stable-diffusion-1-3": "sd15", "stable-diffusion-1-4": "sd15",
          "stable-diffusion-2": "sd21", "stable-diffusion-xl": "sdxl", "SD3-medium": "sd3", "glide": "glide"}
DF_MAP = {1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
UNSEEN = {"FLUX.1-dev", "FLUX.1-schnell", "dalle2", "firefly", "imagen3"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-source", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(GENS))
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    m.to(dev).eval()
    recs = []  # (benchmark, true label or 'unseen:<src>', probs)

    def run(images):
        x = torch.stack([prep(b, 192) for b in images]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
            return torch.softmax(m(x).float(), 1).cpu().numpy()

    def feed(bench, items):
        for i in range(0, len(items), 16):
            chunk = items[i:i + 16]
            for (tag, _), p in zip(chunk, run([b for _, b in chunk])):
                recs.append((bench, tag, p))

    seen = Counter()
    for f in sorted(glob.glob("C:/ai_data/raw/synthbuster/data/test-*.parquet")):
        items = []
        for r in pq.ParquetFile(f).read(columns=["image", "label", "source"]).to_pylist():
            if r["label"] != 1 or seen[r["source"]] >= a.per_source:
                continue
            seen[r["source"]] += 1
            tag = SB_MAP.get(r["source"]) or ("unseen:" + r["source"] if r["source"] in UNSEEN else None)
            if tag:
                items.append((tag, r["image"]["bytes"]))
        feed("synthbuster", items)
        print(f, len(recs), flush=True)
    seen = Counter()
    for f in sorted(glob.glob("C:/ai_data/raw/defactify_test/data/test-*.parquet")):
        items = []
        for r in pq.ParquetFile(f).read(columns=["Image", "Label_B"]).to_pylist():
            if r["Label_B"] == 0 or seen[r["Label_B"]] >= a.per_source:
                continue
            seen[r["Label_B"]] += 1
            items.append((DF_MAP[r["Label_B"]], r["Image"]["bytes"]))
        feed("defactify", items)
        print(f, len(recs), flush=True)

    res = {}
    for bench in ("defactify", "synthbuster"):
        known = [(t, p) for b, t, p in recs if b == bench and not t.startswith("unseen:")]
        acc = float(np.mean([GENS[int(p.argmax())] == t for t, p in known]))
        per = defaultdict(list)
        for t, p in known:
            per[t].append(GENS[int(p.argmax())] == t)
        res[bench] = {"n": len(known), "closed_set_acc": round(acc, 3),
                      "per_generator": {k: round(float(np.mean(v)), 3) for k, v in sorted(per.items())}}
        print(f"\n{bench}: closed-set accuracy {acc:.1%} on {len(known)} images")
        for k, v in sorted(per.items()):
            print(f"   {k:12s}{np.mean(v):6.1%}  (n={len(v)})")
    known = [(t, p) for b, t, p in recs if not t.startswith("unseen:")]
    unseen = [(t, p) for b, t, p in recs if t.startswith("unseen:")]
    print("\nthreshold | known generators: answered / correct-when-answered | UNSEEN generators wrongly named")
    rows = []
    for th in (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        ans = [(t, p) for t, p in known if p.max() >= th]
        cor = np.mean([GENS[int(p.argmax())] == t for t, p in ans]) if ans else 0
        wrong = np.mean([p.max() >= th for _, p in unseen]) if unseen else 0
        rows.append({"threshold": th, "known_answered": round(len(ans) / len(known), 3),
                     "known_precision": round(float(cor), 3), "unseen_wrongly_named": round(float(wrong), 3)})
        print(f"   {th:4.2f}   |   {len(ans) / len(known):5.1%} / {cor:6.1%}                         | {wrong:6.1%}")
    res["threshold_table"] = rows
    res["n_unseen"] = len(unseen)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
