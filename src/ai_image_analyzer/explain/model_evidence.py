"""Model-grounded explanation: where in the image the trained detector found its evidence.

The bundled network is  features -> global average pool -> linear, so its decision decomposes EXACTLY:
    logit_ai - logit_real = sum(evidence_map) + bias
Each cell of the map is that location's additive contribution (positive = pushes toward AI,
negative = toward natural). It is the model's own arithmetic, not a separate heuristic and not an
occlusion approximation (blanking patches was tried and rejected: the blank patches themselves
look artificial to the model). Resolution is coarse (6x6 for a 192px input).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class Region:
    x0: int
    y0: int
    x1: int
    y1: int
    contribution: float  # share of the logit (ai - real) contributed by this region

    def to_dict(self) -> dict:
        return {"box": [self.x0, self.y0, self.x1, self.y1], "contribution": round(self.contribution, 4)}


@dataclass
class ModelEvidence:
    ai_probability: float
    logit: float                                  # logit_ai - logit_real
    bias: float
    evidence: np.ndarray                          # (h, w) additive contributions
    crop_box: tuple[int, int, int, int]           # square part of the original the model sees
    regions: list[Region] = field(default_factory=list)   # strongest cells supporting the verdict
    top_share: float = 0.0                        # share of same-sign evidence held by those cells
    uniform_share: float = 0.0                    # what a perfectly even spread would give

    @property
    def localized(self) -> bool:
        return self.top_share >= 2.0 * self.uniform_share and self.top_share >= 0.4

    def to_dict(self) -> dict:
        return {
            "method": "class_activation_decomposition", "ai_probability": round(self.ai_probability, 4),
            "logit": round(self.logit, 4), "bias": round(self.bias, 4), "crop_box": list(self.crop_box),
            "evidence": np.round(self.evidence, 4).tolist(), "regions": [r.to_dict() for r in self.regions],
            "top_share": round(self.top_share, 4), "uniform_share": round(self.uniform_share, 4),
            "localized": self.localized,
        }


def explain_model_evidence(detector, rgb: np.ndarray, top_k: int = 3) -> ModelEvidence:
    """Evidence map for an OnnxBinaryDetector on an HxWx3 uint8 image (one forward pass)."""
    p, ev, bias = detector.predict_with_evidence(rgb)
    hh, ww = ev.shape
    h, w = rgb.shape[:2]
    side = min(h, w)
    left, top = (w - side) // 2, (h - side) // 2
    logit = float(ev.sum() + bias)
    sign = 1.0 if logit >= 0 else -1.0
    flat = sign * ev.ravel()
    order = np.argsort(-flat)[:top_k]
    regions = []
    for k in order:
        if flat[k] <= 0:
            continue
        i, j = divmod(int(k), ww)
        regions.append(Region(int(left + j * side / ww), int(top + i * side / hh),
                              int(left + (j + 1) * side / ww), int(top + (i + 1) * side / hh), float(ev.ravel()[k])))
    same = flat[flat > 0].sum()
    top_share = float(sum(flat[k] for k in order if flat[k] > 0) / same) if same > 0 else 0.0
    return ModelEvidence(p, logit, bias, ev, (left, top, left + side, top + side), regions,
                         top_share, min(1.0, top_k / ev.size))


def save_model_heatmap(rgb: np.ndarray, me: ModelEvidence, out_path: str, alpha: float = 0.55) -> str:
    """Save original | overlay (red = supports AI, blue = supports natural). Needs only Pillow/numpy."""
    x0, y0, x1, y1 = me.crop_box
    crop = rgb[y0:y1, x0:x1]
    side = crop.shape[0]
    m = np.asarray(Image.fromarray(me.evidence.astype(np.float32), mode="F").resize((side, side), Image.BICUBIC))
    a = np.clip(np.abs(m) / max(float(np.abs(m).max()), 1e-6), 0, 1)[..., None] * alpha
    color = np.where(m[..., None] >= 0, np.array([255, 40, 40]), np.array([40, 90, 255])).astype(np.float32)
    overlay = (crop * (1 - a) + color * a).clip(0, 255).astype(np.uint8)
    Image.fromarray(np.concatenate([crop, overlay], axis=1)).save(out_path)
    return out_path
