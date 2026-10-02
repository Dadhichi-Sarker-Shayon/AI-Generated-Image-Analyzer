from __future__ import annotations
from typing import Any, Optional
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


class ClipZeroShotDetector(BaseDetector):
    """Zero-shot CLIP-based detector (optional, supports PyTorch + ONNX Runtime)."""
    name = "clip"

    PROMPTS_REAL = [
        "a real photograph",
        "a natural scene photo",
        "a candid photo of a real scene",
        "a high-dynamic-range photograph",
    ]
    PROMPTS_GEN = [
        "an AI-generated image",
        "a computer-generated image",
        "an image produced by a text-to-image model",
        "an artificial digital artwork",
    ]

    def __init__(self, config: AnalysisConfig | None = None):
        self.config = config or AnalysisConfig.default()
        self._model = None
        self._processor = None
        self._onnx_session = None
        self._device = self.config.clip_device
        self._load_error: Optional[str] = None

    def _lazy_load(self):
        if self._model is not None or self._onnx_session is not None or self._load_error is not None:
            return
        
        # Try ONNX first if enabled
        if getattr(self.config, 'use_onnx_clip', False):
            try:
                import onnxruntime as ort
                import os
                onnx_path = "clip_vision.onnx"
                if os.path.exists(onnx_path):
                    self._onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                    # Still need processor for text
                    from transformers import CLIPProcessor
                    self._processor = CLIPProcessor.from_pretrained(self.config.clip_model)
                    return
            except Exception:
                pass  # Fall back to PyTorch

        # PyTorch fallback
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
            model_name = self.config.clip_model
            self._processor = CLIPProcessor.from_pretrained(model_name)
            self._model = CLIPModel.from_pretrained(model_name).to(self._device).eval()
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._model is not None or self._onnx_session is not None

    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._model is None and self._onnx_session is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"CLIP not available: {self._load_error or 'model not loaded'}"],
            )

        try:
            from PIL import Image
            pil_img = Image.fromarray(data.signals.rgb)

            inputs = self._processor(
                text=self.PROMPTS_REAL + self.PROMPTS_GEN,
                images=pil_img,
                return_tensors="np" if self._onnx_session else "pt",
                padding=True,
            )

            if self._onnx_session:
                # ONNX inference
                import numpy as np
                pixel_values = inputs["pixel_values"]
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                
                # Run vision encoder
                image_embeds = self._onnx_session.run(None, {"pixel_values": pixel_values})[0]
                
                # For text, we'd need a separate ONNX text encoder or fall back to PyTorch
                # Simplified: use PyTorch for full model if ONNX text not available
                import torch
                from transformers import CLIPModel
                model = CLIPModel.from_pretrained(self.config.clip_model).to(self._device).eval()
                with torch.no_grad():
                    text_inputs = {k: torch.tensor(v).to(self._device) for k, v in inputs.items() if k != "pixel_values"}
                    text_outputs = model.get_text_features(**text_inputs)
                    image_features = torch.tensor(image_embeds).to(self._device)
                    logits_per_image = image_features @ text_outputs.T * model.logit_scale.exp()
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
            else:
                # PyTorch inference
                import torch
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self._model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

            p_real = probs[:4].mean()
            p_gen = probs[4:].mean()
            total = p_real + p_gen
            score = float(p_gen / total) if total > 0 else 0.5
            confidence = float(abs(p_gen - p_real))

            reasoning = []
            if score > 0.55:
                reasoning.append(f"CLIP semantic similarity favors generated ({p_gen:.3f} vs {p_real:.3f})")
            elif score < 0.45:
                reasoning.append(f"CLIP semantic similarity favors real ({p_real:.3f} vs {p_gen:.3f})")
            else:
                reasoning.append("CLIP semantic similarity inconclusive")

            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=score,
                confidence=confidence,
                reasoning=reasoning,
                features={"p_real": p_real, "p_gen": p_gen},
            )
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"CLIP inference error: {type(e).__name__}: {e}"],
            )