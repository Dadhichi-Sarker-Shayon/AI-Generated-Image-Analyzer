#!/usr/bin/env python3
"""Convert downloaded HF parquet shards into  <out>/{train,val}/{real,ai/<generator>}/*.jpg

Every image gets the identical treatment (center-crop square -> 256px -> JPEG q95) so that
file format / resolution cannot become a shortcut separating real from AI.
"""
import argparse, io, os
from multiprocessing import Pool
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image

DEFACTIFY = {0: None, 1: "sd21", 2: "sdxl", 3: "sd3", 4: "dalle3", 5: "midjourney"}
TINY = ["real", "adm", "biggan", "glide", "midjourney", "sd14", "sd15", "vqdm", "wukong"]
SIZE, QUALITY = 256, 95


def encode(b: bytes) -> bytes:
    im = Image.open(io.BytesIO(b)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    im = im.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s))
    im = im.resize((SIZE, SIZE), Image.LANCZOS)
    out = io.BytesIO()
    im.save(out, "JPEG", quality=QUALITY)
    return out.getvalue()


def work(job):
    b, dest = job
    try:
        dest = Path(dest)
        if dest.exists():
            return 1
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(encode(b))
        return 1
    except Exception:
        return 0


def jobs_for(src: str, files, split: str, out: Path):
    for f in files:
        pf = pq.ParquetFile(f)
        names = pf.schema_arrow.names
        imgcol = "Image" if "Image" in names else "image"
        cols = [imgcol] + [c for c in ("Label_B", "generator") if c in names]
        n = 0
        for batch in pf.iter_batches(batch_size=128, columns=cols):
            d = batch.to_pydict()
            for i, img in enumerate(d[imgcol]):
                if src == "defactify":
                    gen = DEFACTIFY[int(d["Label_B"][i])]
                else:
                    g = TINY[int(d["generator"][i])]
                    gen = None if g == "real" else g
                sub = "real" if gen is None else f"ai/{gen}"
                yield img["bytes"], str(out / split / sub / f"{src}_{Path(f).stem}_{n:06d}.jpg")
                n += 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=r"C:\ai_data\raw")
    ap.add_argument("--out", default=r"C:\ai_data\detector")
    ap.add_argument("--procs", type=int, default=os.cpu_count() or 4)
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    plan = [
        ("defactify", "train", sorted((raw / "defactify/data").glob("train-*.parquet"))),
        ("defactify", "val", sorted((raw / "defactify/data").glob("validation-*.parquet"))),
        ("tiny", "train", sorted((raw / "tinygenimage/data").glob("train-*.parquet"))),
        ("tiny", "val", sorted((raw / "tinygenimage/data").glob("validation-*.parquet"))),
    ]
    with Pool(a.procs) as pool:
        for src, split, files in plan:
            if not files:
                print("skip (no files)", src, split); continue
            ok = tot = 0
            for r in pool.imap_unordered(work, jobs_for("defactify" if src == "defactify" else "tiny", files, split, out), chunksize=16):
                ok += r; tot += 1
                if tot % 2000 == 0: print(src, split, tot, flush=True)
            print(f"{src} {split}: {ok}/{tot} written", flush=True)
    print("DONE")
