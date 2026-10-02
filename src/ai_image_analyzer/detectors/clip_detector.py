from __future__ import annotations
from typing import Any, Optional
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


class ClipZeroShotDetector(BaseDetector):
    """Zero-shot CLIP-based detector (optional, requires transformers + torch)."""
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
        self._device = self.config.clip_device
        self._load_error: Optional[str] = None

    def _lazy_load(self):
        if self._model is not None or self._load_error is not None:
            return
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
        return self._model is not None

    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._model is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"CLIP not available: {self._load_error or 'model not loaded'}"],
            )

        try:
            import torch
            rgb = data.signals.rgb
            pil_img = None
            try:
                from PIL import Image
                pil_img = Image.fromarray(rgb)
            except Exception:
                return DetectorResult(
                    name=self.name,
                    status=DetectorStatus.UNAVAILABLE,
                    reasoning=["Could not convert to PIL image"],
                )

            inputs = self._processor(
                text=self.PROMPTS_REAL + self.PROMPTS_GEN,
                images=pil_img,
                return_tensors="pt",
                padding=True,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits_per_image = outputs.logits_per_image  # [1, n_prompts]
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