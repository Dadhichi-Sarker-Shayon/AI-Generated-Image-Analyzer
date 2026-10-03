#!/usr/bin/env python3
"""Evaluate a binary detector (.pt) on a folder dataset  <root>/<split>/{real,ai/<generator>}/*.jpg.

Writes a per-image CSV (source,label,prob) and prints per-generator detection rates.
"""
import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load(path, size):
    im = Image.open(path).convert("RGB").resize((size, size), Image.BILINEAR)
    return torch.from_numpy(((np.asarray(im, np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--img-size", type=int, default=192)
    ap.add_argument("--probs-out", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=2)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    m.to(dev).eval()

    items = []
    base = Path(a.root) / a.split
    for f in os.listdir(base / "real"):
        items.append(("real", 0, base / "real" / f))
    for g in sorted(os.listdir(base / "ai")):
        for f in os.listdir(base / "ai" / g):
            items.append((g, 1, base / "ai" / g / f))
    probs = []
    for i in range(0, len(items), 64):
        x = torch.stack([load(p, a.img_size) for _, _, p in items[i:i + 64]]).to(dev)
        with torch.no_grad(), torch.autocast("cuda", enabled=dev == "cuda"):
            probs += torch.softmax(m(x).float(), 1)[:, 1].cpu().tolist()
        if (i // 64) % 40 == 0:
            print(i, len(items), flush=True)
    labels = np.array([l for _, l, _ in items])
    probs = np.array(probs)
    with open(a.probs_out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "label", "prob"])
        w.writerows((s, l, round(p, 5)) for (s, l, _), p in zip(items, probs))
    pred = probs >= 0.5
    per = defaultdict(list)
    for (s, l, _), p in zip(items, pred):
        per[s].append(p if l == 1 else (not p))
    res = {"n": len(items), "auc": float(roc_auc_score(labels, probs)), "acc": float((pred == labels).mean()),
           "per_source_correct": {k: round(float(np.mean(v)), 4) for k, v in sorted(per.items())}}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
