#!/usr/bin/env python3
"""
Cross-validation script for AI image detection.
Implements k-fold cross-validation on parquet dataset.
"""
from __future__ import annotations
import argparse
import os
import sys
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd


def create_folds(parquet_dir: str, n_folds: int = 5, seed: int = 42) -> list:
    """Create k-fold splits from parquet files."""
    train_files = sorted(Path("data/tanaji").glob("train-*.parquet"))
    val_files = sorted(Path("data/tanaji").glob("validation-*.parquet"))
    
    all_files = train_files + val_files
    print(f"Total parquet files: {len(all_files)}")
    
    # Shuffle with seed
    np.random.seed(seed)
    np.random.shuffle(all_files)
    
    # Create folds
    fold_size = len(all_files) // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else len(all_files)
        test_files = all_files[start:end]
        train_files = [f for f in all_files if f not in test_files]
        folds.append({
            'fold': i,
            'train': train_files,
            'test': test_files
        })
    
    return folds


def extract_fold_images(fold_files: list, output_dir: str, max_per_class: int = None):
    """Extract images from parquet files to folder structure."""
    import pandas as pd
    from PIL import Image
    import io
    
    output_path = Path(output_dir)
    (output_dir / "train" / "real").mkdir(parents=True, exist_ok=True)
    (output_dir / "train" / "ai").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "real").mkdir(parents=True, exist_ok=True)
    (output_dir / "val" / "ai").mkdir(parents=True, exist_ok=True)
    
    real_count = 0
    ai_count = 0
    
    for pf in fold_files:
        df = pd.read_parquet(pf)
        for _, row in df.iterrows():
            label = int(row['label'])
            img_bytes = row['image']['bytes']
            
            try:
                img = Image.open(io.BytesIO(img_bytes))
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img = img.resize((192, 192), Image.LANCZOS)
                
                if label == 0:
                    img.save(f"data/cv_fold/{output_dir}/train/real/real_{real_count:07d}.jpg", quality=95)
                    real_count += 1
                else:
                    img.save(f"data/cv_fold/{output_dir}/train/ai/ai_{ai_count:07d}.jpg", quality=95)
                    ai_count += 1
            except Exception as e:
                pass
    
    print(f"  Extracted: {real_count} real, {ai_count} AI images")


def run_cv_training(fold_idx: int, data_dir: str, output_dir: str):
    """Run training for a single fold."""
    cmd = [
        sys.executable, "-m", "scripts.train",
        "--data-dir", data_dir,
        "--model", "mobilenetv3_small_100",
        "--task", "binary",
        "--batch-size", "8",
        "--accum-steps", "2",
        "--grad-checkpoint",
        "--precision", "fp16",
        "--img-size", "192",
        "--epochs", "10",
        "--checkpoint-interval", "1",
        "--output-dir", output_dir,
        "--early-stopping-patience", "5",
        "--num-workers", "0",
        "--freeze-backbone",
        "--unfreeze-epoch", "5",
        "--use-warmup",
        "--lr-warmup-epochs", "2",
        "--use-reduce-lr",
        "--use-mixup",
        "--mixup-alpha", "0.2",
        "--mixup-prob", "0.5",
        "--use-cutmix",
        "--cutmix-alpha", "1.0",
        "--cutmix-prob", "0.5",
        "--use-randaugment",
        "--label-smoothing", "0.1",
        "--use-focal",
        "--focal-gamma", "2.0",
        "--focal-alpha", "1.0",
        "--lr-warmup-epochs", "2",
        "--unfreeze-epoch", "5",
        "--use-reduce-lr",
        "--use-warmup",
    ]
    
    output = f"checkpoints_cv/fold_{fold_idx}"
    cmd.extend(["--output-dir", output])
    
    print(f"\n{'='*60}")
    print(f"Training fold {fold_idx}")
    print(f"Output: {output}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd="G:/My Drive/AI Image analyzer")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="K-fold cross-validation for AI image detection")
    parser.add_argument("--folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per class per fold")
    args = parser.parse_args()
    
    print(f"Creating {args.folds}-fold cross-validation splits...")
    
    # Create folds from parquet files
    folds = create_folds("data/tanaji", n_folds=args.folds, seed=args.seed)
    
    # Extract images for each fold
    for fold in folds:
        fold_idx = fold['fold']
        print(f"\nExtracting images for fold {fold_idx}...")
        
        # Create fold directory
        fold_dir = f"data/cv_fold/fold_{fold_idx}"
        os.makedirs(f"data/cv_fold/fold_{fold_idx}/train/real", exist_ok=True)
        os.makedirs(f"data/cv_fold/fold_{fold_idx}/train/ai", exist_ok=True)
        
        # Extract training data
        extract_fold_images(fold['train'], fold_dir)
        
        # For validation, use a subset of test files
        # For simplicity, we'll use the balanced dataset for validation
        # and only extract training data for cross-validation
        
    print("\nAll folds extracted. Starting cross-validation training...")
    
    # Run training for each fold
    for fold in folds:
        fold_idx = fold['fold']
        data_dir = f"data/cv_fold/fold_{fold_idx}"
        output_dir = f"checkpoints_cv/fold_{fold_idx}"
        
        success = run_cv_training(fold_idx, data_dir, output_dir)
        if success:
            print(f"✓ Fold {fold_idx} completed")
        else:
            print(f"✗ Fold {fold_idx} failed")


if __name__ == "__main__":
    main()