#!/usr/bin/env python3
"""
Ensemble training script for AI image detection.
Trains multiple models with different seeds for ensemble voting.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_training(seed: int, output_dir: str, data_dir: str, **kwargs):
    """Run a single training run with specified seed."""
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
    
    # Add seed to output dir
    seed_output = os.path.join(output_dir, f"seed_{seed}")
    cmd.extend(["--output-dir", seed_output])
    
    print(f"\n{'='*60}")
    print(f"Starting training with seed {seed}")
    print(f"Output: {seed_output}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, cwd="G:/My Drive/AI Image analyzer")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Train ensemble of models")
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--output-dir", type=str, default="./checkpoints_ensemble", help="Base output directory")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456, 789, 999], help="Random seeds for ensemble")
    parser.add_argument("--n-models", type=int, default=5, help="Number of models to train")
    args = parser.parse_args()
    
    # Limit number of models
    seeds = args.seeds[:args.n_models]
    
    print(f"Training ensemble with {len(seeds)} models: {seeds}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    
    success_count = 0
    for seed in seeds:
        try:
            if run_training(seed, args.output_dir, args.data_dir):
                success_count += 1
                print(f"✓ Seed {seed} completed successfully")
            else:
                print(f"✗ Seed {seed} failed")
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        except Exception as e:
            print(f"✗ Seed {seed} failed with error: {e}")
    
    print(f"\nCompleted {success_count}/{len(seeds)} models successfully")
    
    if success_count > 0:
        print(f"\nEnsemble models saved to: {args.output_dir}/seed_*/")
        print("Next step: Run ensemble evaluation")


if __name__ == "__main__":
    main()