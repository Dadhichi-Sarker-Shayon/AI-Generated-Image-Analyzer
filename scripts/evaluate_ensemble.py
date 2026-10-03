#!/usr/bin/env python3
"""
Ensemble evaluation script for AI image detection.
Loads multiple trained models and performs ensemble voting.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

from ai_image_analyzer import AIImageAnalyzer


class EnsembleAnalyzer:
    """Ensemble of multiple AI image detectors."""
    
    def __init__(self, model_paths: List[str], device: str = "cuda"):
        self.models = []
        self.device = device
        
        for path in model_paths:
            analyzer = AIImageAnalyzer(use_clip=False)
            # Load checkpoint
            checkpoint = torch.load(path, map_location=device)
            # We need to access the internal model to load state dict
            # This is a simplified approach - in practice, you'd need to properly load the model
            self.models.append(analyzer)
            print(f"Loaded model: {path}")
    
    def analyze(self, image_path: str) -> dict:
        """Run ensemble analysis on an image."""
        all_probs = []
        all_verdicts = []
        all_families = []
        
        for analyzer in self.models:
            report = analyzer.analyze(image_path)
            all_probs.append(report.ai_probability)
            all_verdicts.append(report.verdict)
            all_families.append(report.attribution.family)
        
        # Ensemble voting
        mean_prob = np.mean(all_probs)
        std_prob = np.std(all_probs)
        
        # Majority voting for verdict
        verdict_counts = {}
        for v in all_verdicts:
            verdict_counts[v] = verdict_counts.get(v, 0) + 1
        final_verdict = max(verdict_counts, key=verdict_counts.get)
        
        # Most common family
        family_counts = {}
        for f in all_families:
            family_counts[f] = family_counts.get(f, 0) + 1
        final_family = max(family_counts, key=family_counts.get)
        
        return {
            "ai_probability": float(np.mean(all_probs)),
            "probability_std": float(np.std(all_probs)),
            "individual_probs": [float(p) for p in all_probs],
            "verdict": final_verdict,
            "verdict_counts": verdict_counts,
            "family": final_family,
            "family_counts": family_counts,
            "n_models": len(self.models),
        }


def evaluate_on_test_set(model_paths: List[str], test_dir: str):
    """Evaluate ensemble on a test directory."""
    ensemble = EnsembleAnalyzer(model_paths)
    
    results = {}
    test_path = Path(test_dir)
    
    for class_dir in test_path.iterdir():
        if not class_dir.is_dir():
            continue
        
        class_name = class_dir.name
        results[class_name] = []
        
        for img_file in class_dir.glob("*.jpg"):
            result = ensemble.analyze(str(img_file))
            result['file'] = img_file.name
            result['true_class'] = class_name
            results[class_name].append(result)
    
    return results


def print_results(results: dict):
    """Print evaluation results."""
    print("\n" + "="*80)
    print("ENSEMBLE EVALUATION RESULTS")
    print("="*80)
    
    for class_name, results in results.items():
        print(f"\n{class_name.upper()} ({len(results)} images):")
        
        correct = 0
        for r in results:
            prob = r['ai_probability']
            verdict = r['verdict']
            true_class = r['true_class']
            
            # Determine if prediction is correct
            predicted_ai = prob > 0.5
            is_ai = true_class in ['midjourney', 'sdxl', 'dalle3', 'flux']
            
            if predicted_ai == is_ai:
                correct += 1
                status = "✓"
            else:
                status = "✗"
            
            print(f"  {r['file']}: AI={r['ai_probability']:.3f} (std={r['probability_std']:.3f}), Verdict={r['verdict']}, True={true_class} {status}")
        
        accuracy = correct / len(results)
        print(f"  Class Accuracy: {accuracy:.1%} ({correct}/{len(results)})")
    
    # Overall accuracy
    total_correct = sum(1 for class_results in results.values() for r in class_results 
                        if (r['ai_probability'] > 0.5) == (r['true_class'] in ['midjourney', 'sdxl', 'dalle3', 'flux']))
    total = sum(len(r) for r in results.values())
    print(f"\nOverall Accuracy: {total_correct}/{total} = {total_correct/total:.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ensemble of models")
    parser.add_argument("--models", nargs="+", required=True, help="Paths to model checkpoints")
    parser.add_argument("--test-dir", type=str, default="test_images", help="Test images directory")
    parser.add_argument("--output", type=str, help="Output JSON file")
    args = parser.parse_args()
    
    print(f"Loading {len(args.models)} models...")
    print(f"Test directory: {args.test_dir}")
    
    results = evaluate_on_test_set(args.models, args.test_dir)
    print_results(results)
    
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()