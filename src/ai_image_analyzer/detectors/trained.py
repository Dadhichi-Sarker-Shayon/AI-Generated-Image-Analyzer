from __future__ import annotations
from typing import Any, Optional
import numpy as np
try:  # torch is optional: the bundled ONNX detector (onnx_detector.py) does not need it
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus


class TrainedModelDetector(BaseDetector):
    """Base class for trained model detectors (ViT, ResNet, EfficientNet, etc.)."""
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_base_patch16_224",
        device: str = "cpu",
        input_size: int = 224,
        threshold: float = 0.5,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device = device
        self.input_size = input_size
        self.threshold = threshold
        self._model = None
        self._transform = None
        self._load_error: Optional[str] = None
        self._forensic_preprocess = False
    
    def _lazy_load(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            import torch
            import timm
            from torchvision import transforms
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.input_size = int(checkpoint.get("img_size", self.input_size))
            self._forensic_preprocess = checkpoint.get("preprocess") == "center_crop_256_jpeg95"
            self._model = timm.create_model(
                checkpoint.get("model_name", self.model_name),
                pretrained=False,
                num_classes=checkpoint.get("num_classes", 2)
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self.device).eval()
            
            self._transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
    
    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._model is not None
    
    def _preprocess(self, rgb: np.ndarray):
        import io
        from PIL import Image
        img = Image.fromarray(rgb)
        if self._forensic_preprocess:
            # Must mirror scripts/build_dataset.py: center-crop square -> 256px -> JPEG q95
            w, h = img.size
            side = min(w, h)
            img = img.crop(((w - side) // 2, (h - side) // 2, (w - side) // 2 + side, (h - side) // 2 + side))
            img = img.resize((256, 256), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=95)
            img = Image.open(buf).convert("RGB")
        return self._transform(img).unsqueeze(0).to(self.device)
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._model is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Model not loaded: {self._load_error}"],
            )
        
        try:
            import torch
            with torch.no_grad():
                x = self._preprocess(getattr(data, "original_rgb", data.signals.rgb))
                logits = self._model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # For binary: class 1 = AI
            ai_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
            score = ai_prob
            confidence = abs(ai_prob - self.threshold) * 2  # 0-1
            
            reasoning = []
            if score > self.threshold:
                reasoning.append(f"Trained {self.model_name} predicts AI-generated (p={ai_prob:.3f})")
            else:
                reasoning.append(f"Trained {self.model_name} predicts natural (p={ai_prob:.3f})")
            
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                features={"ai_probability": ai_prob, "model": self.model_name},
            )
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Inference error: {type(e).__name__}: {e}"],
            )


class TrainedBinaryDetector(TrainedModelDetector):
    """Real-vs-AI detector; architecture, input size and preprocessing come from the checkpoint."""
    name = "binary_trained"

    def __init__(self, checkpoint_path: str, device: str = "cpu", threshold: float = 0.5):
        super().__init__(checkpoint_path, "efficientnet_b0", device, 192, threshold)


class TrainedViTDetector(TrainedModelDetector):
    """Fine-tuned Vision Transformer detector."""
    name = "vit_trained"
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        super().__init__(checkpoint_path, "vit_base_patch16_224", device, 224, threshold)


class TrainedResNetDetector(TrainedModelDetector):
    """Fine-tuned ResNet detector."""
    name = "resnet_trained"
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        super().__init__(checkpoint_path, "resnet50", device, 224, threshold)


class TrainedEfficientNetDetector(TrainedModelDetector):
    """Fine-tuned EfficientNet detector."""
    name = "efficientnet_trained"
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        super().__init__(checkpoint_path, "efficientnet_b0", device, 224, threshold)


class TrainedGeneratorDetector(BaseDetector):
    """Trained detector for specific generator attribution (multi-class)."""
    name = "generator_trained"
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_base_patch16_224",
        device: str = "cpu",
        generator_names: Optional[list[str]] = None,
        input_size: int = 224,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device = device
        self.generator_names = generator_names or ["midjourney", "stable_diffusion", "dall_e", "other"]
        self.input_size = input_size
        self._model = None
        self._transform = None
        self._load_error: Optional[str] = None
    
    def _lazy_load(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            import torch
            import timm
            from torchvision import transforms
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self._model = timm.create_model(
                checkpoint.get("model_name", self.model_name),
                pretrained=False,
                num_classes=len(self.generator_names)
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self.device).eval()
            
            self._transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
    
    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._model is not None
    
    def _preprocess(self, rgb: np.ndarray):
        from PIL import Image
        img = Image.fromarray(rgb)
        return self._transform(img).unsqueeze(0).to(self.device)
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._model is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Model not loaded: {self._load_error}"],
            )
        
        try:
            import torch
            with torch.no_grad():
                x = self._preprocess(data.signals.rgb)
                logits = self._model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Get top prediction
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
            predicted_generator = self.generator_names[top_idx] if top_idx < len(self.generator_names) else "unknown"
            
            # Also return binary AI probability (sum of all AI generators vs real)
            # For generator task, all classes are AI generators
            ai_prob = 1.0  # All generator classes are AI
            score = ai_prob
            
            reasoning = [
                f"Trained {self.model_name} predicts: {predicted_generator} (p={top_prob:.3f})",
                f"Top-3: {', '.join(f'{self.generator_names[i]}:{probs[i]:.3f}' for i in np.argsort(probs)[-3:][::-1] if i < len(self.generator_names))}",
            ]
            
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=score,
                confidence=top_prob,
                reasoning=reasoning,
                features={
                    "predicted_generator": predicted_generator,
                    "generator_probabilities": {name: float(probs[i]) for i, name in enumerate(self.generator_names) if i < len(probs)},
                    "model": self.model_name,
                },
            )
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Inference error: {type(e).__name__}: {e}"],
            )


class TrainedAttributionClassifier(BaseDetector):
    """Supervised attribution classifier (multi-head: family + specific model)."""
    name = "attribution_trained"
    
    def __init__(
        self,
        checkpoint_path: str,
        model_name: str = "vit_base_patch16_224",
        device: str = "cpu",
        families: Optional[list[str]] = None,
        models: Optional[list[str]] = None,
        input_size: int = 224,
    ):
        self.checkpoint_path = checkpoint_path
        self.model_name = model_name
        self.device = device
        self.families = families or ["gan", "diffusion", "vae", "unknown"]
        self.models = models or [
            "midjourney", "stable_diffusion", "dall_e", "stylegan", "progan", "other"
        ]
        self.input_size = input_size
        self._model = None
        self._transform = None
        self._load_error: Optional[str] = None
    
    def _lazy_load(self):
        if self._model is not None or self._load_error is not None:
            return
        try:
            import torch
            import timm
            from torchvision import transforms
            
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            # Multi-head model: backbone + two heads
            self._model = AttributionModel(
                backbone_name=checkpoint.get("model_name", self.model_name),
                num_families=len(self.families),
                num_models=len(self.models),
                pretrained=False,
            )
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self.device).eval()
            
            self._transform = transforms.Compose([
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"
    
    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._model is not None
    
    def _preprocess(self, rgb: np.ndarray):
        from PIL import Image
        img = Image.fromarray(rgb)
        return self._transform(img).unsqueeze(0).to(self.device)
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._model is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Model not loaded: {self._load_error}"],
            )
        
        try:
            import torch
            with torch.no_grad():
                x = self._preprocess(data.signals.rgb)
                family_logits, model_logits = self._model(x)
                family_probs = torch.softmax(family_logits, dim=1).cpu().numpy()[0]
                model_probs = torch.softmax(model_logits, dim=1).cpu().numpy()[0]
            
            family_idx = int(np.argmax(family_probs))
            model_idx = int(np.argmax(model_probs))
            
            family = self.families[family_idx] if family_idx < len(self.families) else "unknown"
            model = self.models[model_idx] if model_idx < len(self.models) else "unknown"
            
            family_conf = float(family_probs[family_idx])
            model_conf = float(model_probs[model_idx])
            
            # AI probability = 1 - P(real) if we had real class, else use family confidence
            ai_prob = family_conf  # Simplified
            
            reasoning = [
                f"Attribution: family={family} ({family_conf:.3f}), model={model} ({model_conf:.3f})",
                f"Families: {', '.join(f'{self.families[i]}:{family_probs[i]:.3f}' for i in np.argsort(family_probs)[::-1])}",
            ]
            
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=ai_prob,
                confidence=min(family_conf, model_conf),
                reasoning=reasoning,
                features={
                    "family": family,
                    "family_probs": {name: float(family_probs[i]) for i, name in enumerate(self.families)},
                    "model": model,
                    "model_probs": {name: float(model_probs[i]) for i, name in enumerate(self.models)},
                },
            )
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Inference error: {type(e).__name__}: {e}"],
            )


class AttributionModel(torch.nn.Module if torch is not None else object):
    """Multi-head model for attribution (family + specific model)."""
    
    def __init__(self, backbone_name: str, num_families: int, num_models: int, pretrained: bool = True):
        super().__init__()
        import timm
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        feat_dim = self.backbone.num_features
        self.family_head = torch.nn.Linear(feat_dim, num_families)
        self.model_head = torch.nn.Linear(feat_dim, num_models)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.family_head(features), self.model_head(features)


def create_trained_detectors(
    binary_checkpoint: Optional[str] = None,
    generator_checkpoint: Optional[str] = None,
    attribution_checkpoint: Optional[str] = None,
    device: str = "cpu",
) -> list[BaseDetector]:
    """Factory to create all trained detectors from checkpoints."""
    detectors = []
    
    if binary_checkpoint:
        detectors.append(TrainedBinaryDetector(binary_checkpoint, device))
        # Can also add ResNet/EfficientNet ensemble
        # detectors.append(TrainedResNetDetector(binary_checkpoint, device))
        # detectors.append(TrainedEfficientNetDetector(binary_checkpoint, device))
    
    if generator_checkpoint:
        detectors.append(TrainedGeneratorDetector(generator_checkpoint, device=device))
    
    if attribution_checkpoint:
        detectors.append(TrainedAttributionClassifier(attribution_checkpoint, device=device))
    
    return detectors