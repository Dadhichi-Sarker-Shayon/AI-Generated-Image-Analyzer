#!/usr/bin/env python3
"""
Download and prepare datasets for AI image detection training.
Supports: GenImage (subset), DRCT, SynthBuster, CIFAKE.
"""
from __future__ import annotations
import argparse
import os
import sys
import zipfile
import tarfile
import gzip
import shutil
from pathlib import Path
from typing import Optional
import urllib.request
from tqdm import tqdm
import numpy as np
from PIL import Image


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url: str, output_path: Path, desc: str = "Downloading"):
    """Download a file with progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)


def extract_archive(archive_path: Path, extract_dir: Path):
    """Extract zip, tar.gz, or tar archive."""
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {archive_path}...")
    
    if archive_path.suffix == ".zip":
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)
    elif archive_path.suffix in [".gz", ".tgz"] or ".tar" in str(archive_path):
        with tarfile.open(archive_path, 'r:*') as tf:
            tf.extractall(extract_dir)
    else:
        raise ValueError(f"Unknown archive format: {archive_path}")


def organize_dataset(source_dir: Path, output_dir: Path, max_per_class: Optional[int] = None):
    """
    Reorganize downloaded dataset into standard structure:
    output_dir/
      train/
        real/
        ai/
      val/
        real/
        ai/
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find real and AI images
    real_images = []
    ai_images = []
    
    # Common patterns for real images
    real_patterns = ["real", "nature", "photo", "camera", "original", "0_real"]
    ai_patterns = ["ai", "fake", "generated", "synthetic", "diffusion", "gan", 
                   "midjourney", "stable_diffusion", "dalle", "stylegan", "progan",
                   "1_fake", "fake_images"]
    
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        for img_path in source_dir.rglob(ext):
            rel = img_path.relative_to(source_dir)
            path_str = str(rel).lower()
            
            is_real = any(p in path_str for p in real_patterns)
            is_ai = any(p in path_str for p in ai_patterns)
            
            if is_real and not is_ai:
                real_images.append(img_path)
            elif is_ai and not is_real:
                ai_images.append(img_path)
            else:
                # Ambiguous - try to infer from parent dirs
                parent_names = [p.lower() for p in img_path.parents]
                if any("real" in p or "nature" in p or "photo" in p for p in parent_names):
                    real_images.append(img_path)
                elif any("fake" in p or "ai" in p or "gen" in p for p in parent_names):
                    ai_images.append(img_path)
    
    print(f"Found {len(real_images)} real, {len(ai_images)} AI images")
    
    if len(real_images) == 0 or len(ai_images) == 0:
        print("WARNING: Could not auto-detect real/AI split. You may need to organize manually.")
        return
    
    # Shuffle and split
    np.random.seed(42)
    np.random.shuffle(real_images)
    np.random.shuffle(ai_images)
    
    if max_per_class:
        real_images = real_images[:max_per_class]
        ai_images = ai_images[:max_per_class]
    
    # 80/20 train/val split
    split_idx_real = int(0.8 * len(real_images))
    split_idx_ai = int(0.8 * len(ai_images))
    
    splits = {
        "train": {"real": real_images[:split_idx_real], "ai": ai_images[:split_idx_ai]},
        "val": {"real": real_images[split_idx_real:], "ai": ai_images[split_idx_ai:]},
    }
    
    # Copy to organized structure
    for split, classes in splits.items():
        for class_name, images in classes.items():
            out_dir = output_dir / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            for i, src in enumerate(images):
                dst = out_dir / f"{class_name}_{i:06d}{src.suffix}"
                shutil.copy2(src, dst)
            
            print(f"  {split}/{class_name}: {len(images)} images")
    
    print(f"\nDataset organized at: {output_dir}")


def _generate_fixture(kind: str, h: int, w: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic test images with known forensic signatures."""
    rng = np.random.default_rng(seed)
    y_grid, x_grid = np.indices((h, w))

    if kind == "natural":
        beta = 2.3
    elif kind == "diffusion":
        beta = 1.3
    elif kind == "gan":
        beta = 2.3
    else:
        beta = 2.3

    # Generate fBm via inverse FFT
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    fr = np.hypot(fx, fy)
    fr[0, 0] = 1e-6
    amp = fr ** (-beta / 2)
    phase = rng.uniform(0, 2 * np.pi, (h, w))
    F = amp * np.exp(1j * phase)
    F[0, 0] = 0
    img = np.fft.ifft2(F).real
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if kind == "natural":
        noise = rng.normal(0, 2.5 / 255, (h, w))
        cfa = 0.01 * np.sin(np.pi * y_grid) * np.sin(np.pi * x_grid)
        vignette = 1 - 0.15 * (fr / fr.max())
        img = img * vignette + noise + cfa
    elif kind == "diffusion":
        ring_mask = np.exp(-((fr - 0.25) ** 2) / (2 * 0.04 ** 2))
        ring_phase = rng.uniform(0, 2 * np.pi, (h, w))
        ring_F = 0.15 * ring_mask * np.exp(1j * ring_phase)
        ring = np.fft.ifft2(ring_F).real
        img = img + ring
        import cv2
        img = cv2.GaussianBlur(img, (3, 3), 0.8)
        img = img + rng.normal(0, 1.2 / 255, (h, w))
    elif kind == "gan":
        lattice = 0.06 * np.sin(2 * np.pi * x_grid / 3) * np.sin(2 * np.pi * y_grid / 3)
        img = img + lattice
        img = img + rng.normal(0, 3.0 / 255, (h, w))

    img = np.clip(img, 0, 1)
    rgb = np.stack([
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
        img * (1 + rng.normal(0, 0.005)),
    ], axis=-1)
    rgb = np.clip(rgb, 0, 1)
    return (rgb * 255).astype(np.uint8)


def prepare_synthetic_dataset(output_dir: Path, num_samples: int = 5000):
    """Generate synthetic training data using our fixtures (for testing only)."""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": {"real": int(0.8 * num_samples // 2), "ai": int(0.8 * num_samples // 2)},
        "val": {"real": int(0.2 * num_samples // 2), "ai": int(0.2 * num_samples // 2)},
    }
    
    for split, classes in splits.items():
        for class_name, count in classes.items():
            out_dir = output_dir / split / class_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            split_offset = 10000 if split == "val" else 0
            
            for i in range(count):
                seed = i + split_offset
                if class_name == "real":
                    img = _generate_fixture("natural", 224, 224, seed=seed)
                else:
                    # Alternate between diffusion and GAN for AI class
                    if i % 2 == 0:
                        img = _generate_fixture("diffusion", 224, 224, seed=seed)
                    else:
                        img = _generate_fixture("gan", 224, 224, seed=seed)
                
                dst = out_dir / f"{class_name}_{i:06d}.png"
                Image.fromarray(img).save(dst)
            
            print(f"  Generated {split}/{class_name}: {count} images")
    
    print(f"\nSynthetic dataset created at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare training datasets")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Download command
    dl_parser = subparsers.add_parser("download", help="Download a dataset")
    dl_parser.add_argument("--dataset", choices=["cifake", "synthbuster", "drct_subset"], required=True)
    dl_parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    
    # Organize command
    org_parser = subparsers.add_parser("organize", help="Organize downloaded dataset")
    org_parser.add_argument("--source-dir", type=Path, required=True)
    org_parser.add_argument("--output-dir", type=Path, required=True)
    org_parser.add_argument("--max-per-class", type=int, default=None)
    
    # Synthetic command
    synth_parser = subparsers.add_parser("synthetic", help="Generate synthetic test dataset")
    synth_parser.add_argument("--output-dir", type=Path, default=Path("./data/synthetic"))
    synth_parser.add_argument("--num-samples", type=int, default=5000)
    
    args = parser.parse_args()
    
    if args.command == "download":
        if args.dataset == "cifake":
            # CIFAKE - CIFAR-10 real vs synthetic
            # Try multiple mirrors
            urls = [
                "https://huggingface.co/datasets/birdy654/CIFAKE/resolve/main/CIFAKE.zip",
                "https://github.com/birdy654/CIFAKE/releases/download/v1.0/CIFAKE.zip",
            ]
            zip_path = args.output_dir / "CIFAKE.zip"
            success = False
            for url in urls:
                try:
                    print(f"Trying {url}...")
                    download_url(url, zip_path, "CIFAKE")
                    success = True
                    break
                except Exception as e:
                    print(f"  Failed: {e}")
            if not success:
                print("All CIFAKE mirrors failed. Try manual download from https://github.com/birdy654/CIFAKE")
                return
            extract_archive(zip_path, args.output_dir / "CIFAKE")
            print("CIFAKE downloaded. Run 'organize' to prepare for training.")
            
        elif args.dataset == "synthbuster":
            # SynthBuster benchmark
            print("SynthBuster download requires manual steps:")
            print("1. Go to https://github.com/kkkls/SynthBuster")
            print("2. Download the dataset")
            print("3. Extract and run 'organize'")
            
        elif args.dataset == "drct_subset":
            # DRCT subset - using Hugging Face
            print("For DRCT, use Hugging Face datasets:")
            print("  from datasets import load_dataset")
            print("  ds = load_dataset('DRCT/DRCT')")
            print("Then save images and run 'organize'")
        
        elif args.dataset == "deepfake":
            # DeepFake detection dataset (smaller subset)
            url = "https://huggingface.co/datasets/ashishpatel26/deepfake-detection/resolve/main/data.zip"
            zip_path = args.output_dir / "deepfake.zip"
            print(f"Downloading from {url}...")
            download_url(url, zip_path, "DeepFake")
            extract_archive(zip_path, args.output_dir / "deepfake")
            print("Downloaded. Run 'organize' to prepare for training.")
    
    elif args.command == "organize":
        organize_dataset(args.source_dir, args.output_dir, args.max_per_class)
    
    elif args.command == "synthetic":
        # Need to add src to path
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        prepare_synthetic_dataset(args.output_dir, args.num_samples)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()