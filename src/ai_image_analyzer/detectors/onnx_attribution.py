"""Torch-free generator attribution (which AI generator made the image) on ONNX Runtime."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
from .onnx_detector import BUNDLED_ATTRIBUTION, OnnxBinaryDetector


def bundled_attribution_path() -> Optional[Path]:
    return BUNDLED_ATTRIBUTION if BUNDLED_ATTRIBUTION.exists() else None


class OnnxGeneratorAttributor:
    """Closed-set classifier over the generators it was trained on, with an abstain option.

    It can only name generators it has seen; for anything else it should abstain. `min_confidence`
    (from the JSON sidecar, calibrated on benchmark generators the model never saw) controls that.
    """

    def __init__(self, model_path: str | Path | None = None, min_confidence: float | None = None):
        self.model_path = Path(model_path) if model_path else BUNDLED_ATTRIBUTION
        self._pre = OnnxBinaryDetector(self.model_path)      # same loader + preprocessing contract
        self._min_conf = min_confidence

    @property
    def available(self) -> bool:
        return self._pre.available

    @property
    def info(self) -> dict:
        self._pre._lazy_load()
        return dict(self._pre._meta)

    @property
    def classes(self) -> list[str]:
        return list(self.info.get("classes", []))

    @property
    def min_confidence(self) -> float:
        return self._min_conf if self._min_conf is not None else float(self.info.get("min_confidence", 0.5))

    def predict_proba(self, rgb: np.ndarray) -> dict[str, float]:
        self._pre._lazy_load()
        if self._pre._session is None:
            raise RuntimeError(f"ONNX model not loaded: {self._pre._load_error}")
        p = self._pre._session.run(None, {"pixel_values": self._pre.preprocess(rgb)})[0][0]
        return {c: float(v) for c, v in zip(self.classes, p)}

    def attribute(self, rgb: np.ndarray) -> tuple[Optional[str], float, dict[str, float]]:
        """(generator or None if it abstains, confidence, all probabilities)."""
        probs = self.predict_proba(rgb)
        best = max(probs, key=probs.get)
        conf = probs[best]
        return (best if conf >= self.min_confidence else None), conf, probs
