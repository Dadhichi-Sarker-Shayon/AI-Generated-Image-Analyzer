"""Optional natural-language image description (captioning).

Requires the `describe` extra (torch + transformers) and downloads the captioning model (~1 GB) on first use.
The description summarizes *content*; it is not evidence about whether the image is AI-generated.
"""
from __future__ import annotations
import numpy as np

DEFAULT_MODEL = "Salesforce/blip-image-captioning-base"


class ImageDescriber:
    def __init__(self, model_name: str = DEFAULT_MODEL, device: str = "cpu", max_new_tokens: int = 40):
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._processor = None
        import importlib.util
        missing = [m for m in ("torch", "transformers") if importlib.util.find_spec(m) is None]
        if missing:  # fail at construction with an actionable message, not at the first image
            raise ImportError(f"Image descriptions need {' and '.join(missing)}: pip install 'ai-image-analyzer[describe]'")

    def _lazy_load(self):
        if self._model is not None:
            return
        try:
            from transformers import BlipForConditionalGeneration, BlipProcessor
        except ImportError as e:  # pragma: no cover
            raise ImportError("Image descriptions need extra packages: pip install 'ai-image-analyzer[describe]'") from e
        self._processor = BlipProcessor.from_pretrained(self.model_name)
        self._model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device).eval()

    def describe(self, rgb: np.ndarray) -> str:
        """One-sentence description of an HxWx3 uint8 RGB image."""
        import torch
        from PIL import Image
        self._lazy_load()
        img = Image.fromarray(rgb)
        longest = max(img.size)
        if longest > 768:  # captioning does not need full resolution
            s = 768 / longest
            img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))), Image.BICUBIC)
        inputs = self._processor(images=img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, num_beams=3)
        return self._processor.decode(out[0], skip_special_tokens=True).strip()
