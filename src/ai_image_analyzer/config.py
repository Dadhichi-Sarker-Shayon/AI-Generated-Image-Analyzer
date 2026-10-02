from __future__ import annotations
import dataclasses
from pathlib import Path
from typing import Any, Optional
import yaml


@dataclasses.dataclass(frozen=True)
class AnalysisConfig:
    max_analysis_dim: int = 1024
    tile_size: int = 64

    # Frequency analysis
    exponent_center: float = -2.3
    exponent_band: float = 0.35
    peak_db_threshold: float = 10.0
    lattice_db_threshold: float = 8.0
    lattice_periods: tuple[int, ...] = (2, 3, 4, 5, 6, 8, 10)
    hf_cutoff: float = 0.35

    # Noise analysis
    noise_sigma_low: float = 1.5
    noise_sigma_high: float = 18.0
    uniformity_ai: float = 0.93
    sensor_period_db_natural: float = 8.0

    # Texture analysis
    lbp_entropy_ai: float = 0.90
    glcm_contrast_ai: float = 0.6
    fractal_dim_low: float = 1.15
    fractal_dim_high: float = 1.75

    # Color analysis
    banded_fraction_ai: float = 0.05
    cast_threshold: float = 3.0

    # Ensemble
    detector_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {"frequency": 0.5, "clip": 0.5}
    )
    verdict_ai: float = 0.60
    verdict_natural: float = 0.25

    # CLIP
    clip_model: str = "openai/clip-vit-base-patch32"
    clip_device: str = "cpu"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AnalysisConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        flat = _flatten_config(data)
        field_names = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in flat.items() if k in field_names}
        return cls(**filtered)

    @classmethod
    def default(cls) -> "AnalysisConfig":
        return cls()


def _flatten_config(data: dict[str, Any]) -> dict[str, Any]:
    """Flatten nested YAML sections (e.g., frequency: {exponent_center: -2.3}) to flat keys."""
    flat = {}
    for k, v in data.items():
        if isinstance(v, dict):
            for sk, sv in v.items():
                flat[sk] = sv
        else:
            flat[k] = v
    return flat