"""Torch-free trained real-vs-AI detector running on ONNX Runtime."""
from __future__ import annotations
import io
import json
from pathlib import Path
from typing import Optional
import numpy as np
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "models" / "weights"
BUNDLED_MODEL = WEIGHTS_DIR / "detector_v2.onnx"
BUNDLED_ATTRIBUTION = WEIGHTS_DIR / "attribution_v1.onnx"


def bundled_model_path() -> Optional[Path]:
    return BUNDLED_MODEL if BUNDLED_MODEL.exists() else None


class OnnxBinaryDetector(BaseDetector):
    """EfficientNet-B0 real-vs-AI classifier exported to ONNX.

    Preprocessing contract (read from the JSON sidecar, must match training):
    center-crop square -> 256px (Lanczos) -> JPEG q95 -> resize to input_size -> ImageNet normalize.
    """
    name = "binary_trained"

    def __init__(self, model_path: str | Path | None = None, threshold: float = 0.5, providers: list[str] | None = None):
        self.model_path = Path(model_path) if model_path else BUNDLED_MODEL
        self.threshold = threshold
        self.providers = providers or ["CPUExecutionProvider"]
        self._session = None
        self._meta: dict = {}
        self._load_error: Optional[str] = None

    def _lazy_load(self):
        if self._session is not None or self._load_error is not None:
            return
        try:
            import onnxruntime as ort
            self._session = ort.InferenceSession(str(self.model_path), providers=self.providers)
            sidecar = self.model_path.with_suffix(".json")
            self._meta = json.loads(sidecar.read_text(encoding="utf8")) if sidecar.exists() else {}
        except Exception as e:
            self._load_error = f"{type(e).__name__}: {e}"

    @property
    def available(self) -> bool:
        self._lazy_load()
        return self._session is not None

    @property
    def model_info(self) -> dict:
        self._lazy_load()
        return dict(self._meta)

    def preprocess(self, rgb: np.ndarray) -> np.ndarray:
        from PIL import Image
        size = int(self._meta.get("input_size", 192))
        img = Image.fromarray(rgb)
        if self._meta.get("preprocess", "center_crop_256_jpeg95") == "center_crop_256_jpeg95":
            w, h = img.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            img = img.crop((left, top, left + side, top + side)).resize((256, 256), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=95)
            img = Image.open(buf).convert("RGB")
        img = img.resize((size, size), Image.BILINEAR)
        mean = np.array(self._meta.get("mean", [0.485, 0.456, 0.406]), np.float32)
        std = np.array(self._meta.get("std", [0.229, 0.224, 0.225]), np.float32)
        x = (np.asarray(img, np.float32) / 255.0 - mean) / std
        return x.transpose(2, 0, 1)[None].astype(np.float32)

    def predict_proba(self, rgb: np.ndarray) -> float:
        """P(AI-generated) for an HxWx3 uint8 RGB array."""
        self._lazy_load()
        if self._session is None:
            raise RuntimeError(f"ONNX model not loaded: {self._load_error}")
        probs = self._session.run(None, {"pixel_values": self.preprocess(rgb)})[0][0]
        return float(probs[1])

    def predict_with_evidence(self, rgb: np.ndarray) -> tuple[float, np.ndarray, float]:
        """(P(AI), evidence_map[h,w], bias). sum(evidence_map) + bias == logit_ai - logit_real."""
        self._lazy_load()
        if self._session is None:
            raise RuntimeError(f"ONNX model not loaded: {self._load_error}")
        outs = self._session.run(None, {"pixel_values": self.preprocess(rgb)})
        if len(outs) < 2:
            raise RuntimeError("This ONNX model has no evidence_map output (re-export with scripts/export_onnx.py).")
        return float(outs[0][0][1]), outs[1][0], float(self._meta.get("evidence_bias", 0.0))

    def evaluate(self, data: AnalysisData) -> DetectorResult:
        self._lazy_load()
        if self._session is None:
            return DetectorResult(name=self.name, status=DetectorStatus.UNAVAILABLE,
                                  reasoning=[f"Model not loaded: {self._load_error}"])
        p = self.predict_proba(getattr(data, "original_rgb", data.signals.rgb))
        arch = self._meta.get("architecture", "cnn")
        verdict = "AI-generated" if p >= self.threshold else "natural"
        return DetectorResult(
            name=self.name, status=DetectorStatus.OK, score=p,
            confidence=abs(p - self.threshold) * 2,
            reasoning=[f"Trained {arch} classifier predicts {verdict} (P(AI)={p:.3f})"],
            features={"ai_probability": p, "model": arch, "model_name": self._meta.get("name", self.model_path.stem),
                      "metrics": self._meta.get("metrics", {})},
        )
