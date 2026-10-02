from __future__ import annotations
from typing import Any, Optional, List
import numpy as np
import cv2
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


class PatchCNNDetector(BaseDetector):
    """
    Patch-level CNN detector for local artifact detection.
    Uses a sliding window approach with a lightweight CNN.
    Can be used with a trained patch classifier or heuristic patch analysis.
    """
    name = "patch_cnn"
    
    def __init__(
        self,
        config: Optional[AnalysisConfig] = None,
        patch_size: int = 64,
        stride: int = 32,
        threshold: float = 0.5,
        use_trained_model: bool = False,
        checkpoint_path: Optional[str] = None,
    ):
        self.config = config or AnalysisConfig.default()
        self.patch_size = patch_size
        self.stride = stride
        self.threshold = threshold
        self.use_trained_model = use_trained_model
        self.checkpoint_path = checkpoint_path
        self._model = None
        self._load_error: Optional[str] = None
    
    def _lazy_load_model(self):
        if self._model is not None or self._load_error is not None or not self.use_trained_model:
            return
        try:
            import torch
            import timm
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self._model = timm.create_model(
                checkpoint.get("model_name", "efficientnet_b0"),
                pretrained=False,
                num_classes=2
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.eval()
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        if self.use_trained_model:
            return self._evaluate_trained(data)
        else:
            return self._evaluate_heuristic(data)
    
    def _evaluate_heuristic(self, data: AnalysisData) -> DetectorResult:
        """Heuristic patch analysis without trained model."""
        rgb = data.signals.rgb
        h, w = rgb.shape[:2]
        
        patch_scores = []
        patch_coords = []
        
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = rgb[y:y+self.patch_size, x:x+self.patch_size]
                score = self._heuristic_patch_score(patch)
                patch_scores.append(score)
                patch_coords.append((y, x))
        
        if not patch_scores:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.INSUFFICIENT_DATA,
                reasoning=["Image too small for patch analysis"],
            )
        
        patch_scores = np.array(patch_scores)
        max_score = float(patch_scores.max())
        mean_score = float(patch_scores.mean())
        suspicious_patches = int((patch_scores > self.threshold).sum())
        total_patches = len(patch_scores)
        
        # Overall image score
        score = max(max_score, mean_score * 1.5)  # Combine max and mean
        score = min(1.0, score)
        
        reasoning = []
        if suspicious_patches > 0:
            reasoning.append(f"{suspicious_patches}/{total_patches} patches exceed threshold ({self.threshold})")
        if max_score > 0.7:
            reasoning.append(f"Max patch score: {max_score:.3f} (strong local artifact)")
        
        # Find top suspicious regions
        top_indices = np.argsort(patch_scores)[-5:][::-1]
        top_patches = [{"y": patch_coords[i][0], "x": patch_coords[i][1], "score": float(patch_scores[i])} 
                       for i in top_indices]
        
        return DetectorResult(
            name=self.name,
            status=DetectorStatus.OK,
            score=score,
            confidence=min(1.0, suspicious_patches / max(1, total_patches * 0.1)),
            reasoning=reasoning,
            features={
                "patch_scores_mean": mean_score,
                "patch_scores_max": max_score,
                "suspicious_patches": suspicious_patches,
                "total_patches": total_patches,
                "top_patches": top_patches,
            },
        )
    
    def _heuristic_patch_score(self, patch: np.ndarray) -> float:
        """Heuristic scoring of a single patch."""
        # Convert to grayscale
        if patch.ndim == 3:
            gray = 0.299 * patch[..., 0] + 0.587 * patch[..., 1] + 0.114 * patch[..., 2]
        else:
            gray = patch
        gray = gray.astype(np.float32)
        
        scores = []
        
        # 1. High-frequency energy (Laplacian)
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        hf_energy = np.abs(lap).mean() / 255.0
        scores.append(min(1.0, hf_energy * 5))  # Normalize
        
        # 2. Local spectral flatness (FFT)
        try:
            f = np.fft.fft2(gray - gray.mean())
            power = np.abs(f) ** 2
            flatness = np.exp(np.mean(np.log(power[power > 0] + 1e-10))) / (np.mean(power) + 1e-10)
            scores.append(min(1.0, flatness * 2))
        except Exception:
            scores.append(0.0)
        
        # 3. Noise residual uniformity
        blur = cv2.GaussianBlur(gray, (3, 3), 1.0)
        noise = gray - blur
        noise_uniformity = 1.0 - (noise.std() / (gray.std() + 1e-6))
        scores.append(noise_uniformity)
        
        # 4. Color banding in patch
        if patch.ndim == 3:
            banding = 0.0
            for c in range(3):
                ch = patch[..., c]
                diff = np.diff(ch.astype(float), axis=1)
                runs = (diff == 0).sum() / max(1, diff.size)
                banding = max(banding, runs)
            scores.append(min(1.0, banding * 10))
        
        return float(np.mean(scores))
    
    def _evaluate_trained(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load_model()
        if self._model is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Trained model not loaded: {self._load_error}"],
            )
        
        try:
            import torch
            from torchvision import transforms
            
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            
            rgb = data.signals.rgb
            h, w = rgb.shape[:2]
            
            patch_probs = []
            patch_coords = []
            
            for y in range(0, h - self.patch_size + 1, self.stride):
                for x in range(0, w - self.patch_size + 1, self.stride):
                    patch = rgb[y:y+self.patch_size, x:x+self.patch_size]
                    img_tensor = transform(patch).unsqueeze(0)
                    
                    with torch.no_grad():
                        logits = self._model(img_tensor)
                        prob = torch.softmax(logits, dim=1)[0, 1].item()  # AI class
                    
                    patch_probs.append(prob)
                    patch_coords.append((y, x))
            
            patch_probs = np.array(patch_probs)
            max_prob = float(patch_probs.max())
            mean_prob = float(patch_probs.mean())
            suspicious = int((patch_probs > self.threshold).sum())
            
            score = max(max_prob, mean_prob)
            
            reasoning = [f"Trained patch CNN: max AI prob={max_prob:.3f}, mean={mean_prob:.3f}"]
            if suspicious > 0:
                reasoning.append(f"{suspicious} patches exceed threshold")
            
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=score,
                confidence=min(1.0, suspicious / max(1, len(patch_probs) * 0.1)),
                reasoning=reasoning,
                features={
                    "patch_probs_mean": mean_prob,
                    "patch_probs_max": max_prob,
                    "suspicious_patches": suspicious,
                    "total_patches": len(patch_probs),
                },
            )
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Trained inference error: {type(e).__name__}: {e}"],
            )