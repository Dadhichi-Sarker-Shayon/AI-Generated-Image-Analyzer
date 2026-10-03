#!/usr/bin/env python3
"""Download Defactify + Tiny-GenImage parquet shards (train/validation only) to a local dir."""
import argparse, os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
from huggingface_hub import snapshot_download

SOURCES = {
    "defactify": ("Rajarshi-Roy-research/Defactify_Image_Dataset", ["data/train-*", "data/validation-*", "README.md"]),
    "tinygenimage": ("TheKernel01/Tiny-GenImage", ["data/*", "README.md"]),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"C:\ai_data\raw")
    ap.add_argument("--only", choices=list(SOURCES), nargs="*", default=list(SOURCES))
    a = ap.parse_args()
    for k in a.only:
        repo, pats = SOURCES[k]
        print("Downloading", repo, flush=True)
        snapshot_download(repo, repo_type="dataset", allow_patterns=pats,
                          local_dir=os.path.join(a.out, k), max_workers=4)
    print("DONE")
