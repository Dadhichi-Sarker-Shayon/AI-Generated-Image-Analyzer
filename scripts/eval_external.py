#!/usr/bin/env python3
"""Evaluate a trained binary detector on an external parquet benchmark (per-generator breakdown).

Uses the same preprocessing as scripts/build_dataset.py + the training val transform.
"""
import argparse, glob, io, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, timm, torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from build_dataset import encode  # crop -> 256 -> JPEG q95

MEAN, STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)


def prep(b, size):
    im = Image.open(io.BytesIO(encode(b))).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--parquet-glob", required=True)
    ap.add_argument("--model", default="efficientnet_b0")
    ap.add_argument("--img-size", type=int, default=192)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None)
    ap.add_argument("--schema", choices=["synthbuster", "defactify"], default="synthbuster")
    ap.add_argument("--probs-out", default=None, help="CSV with one row per image: source,label,prob")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = timm.create_model(a.model, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    model.to(dev).eval()
    probs, labels, srcs = [], [], []
    for f in sorted(glob.glob(a.parquet_glob)):
        pf = pq.ParquetFile(f)
        cols = ["image", "label", "source"] if a.schema == "synthbuster" else ["Image", "Label_A", "Label_B"]
        for batch in pf.iter_batches(batch_size=16, columns=cols):
            d = batch.to_pydict()
            if a.schema == "defactify":  # normalise to the synthbuster column names
                names = {0: "coco_real", 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
                d = {"image": d["Image"], "label": d["Label_A"], "source": [names[int(b)] for b in d["Label_B"]]}
            x = torch.stack([prep(i["bytes"], a.img_size) for i in d["image"]]).to(dev)
            with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
                p = torch.softmax(model(x).float(), 1)[:, 1].cpu().numpy()
            probs += p.tolist(); labels += d["label"]; srcs += d["source"]
        print("done", Path(f).name, len(probs), flush=True)
    probs, labels = np.array(probs), np.array(labels)
    pred = probs >= a.threshold
    res = {"n": len(labels), "auc": float(roc_auc_score(labels, probs)),
           "acc": float((pred == labels).mean()),
           "real_correct": float((~pred[labels == 0]).mean()), "per_source": {}}
    by = defaultdict(list)
    for p, l, s in zip(pred, labels, srcs): by[s].append((p, l))
    for s, v in sorted(by.items()):
        v = np.array(v); res["per_source"][s] = {"n": len(v), "flagged_as_ai": float(v[:, 0].mean()), "label": int(v[0, 1])}
    print(json.dumps(res, indent=1))
    if a.probs_out:
        import csv
        with open(a.probs_out, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["source", "label", "prob"])
            w.writerows(zip(srcs, labels, np.round(probs, 5)))
    if a.out: Path(a.out).write_text(json.dumps(res, indent=1))
