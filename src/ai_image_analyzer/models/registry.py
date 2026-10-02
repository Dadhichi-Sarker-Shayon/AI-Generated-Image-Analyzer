from __future__ import annotations
import json
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelCheckpoint:
    """Metadata for a model checkpoint."""
    name: str
    path: str
    model_type: str  # "binary", "generator", "attribution"
    architecture: str  # "vit_base_patch16_224", "resnet50", etc.
    task: str
    training_data: str  # e.g., "GenImage", "DRCT", "DiffusionDB"
    generators: Optional[list[str]] = None
    families: Optional[list[str]] = None
    metrics: Optional[dict[str, float]] = None  # val_auc, val_acc, etc.
    created_at: str = ""
    file_hash: str = ""
    file_size_mb: float = 0.0
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.file_hash and Path(self.path).exists():
            self.file_hash = self._compute_hash()
            self.file_size_mb = Path(self.path).stat().st_size / (1024 * 1024)
    
    def _compute_hash(self) -> str:
        """Compute SHA256 of checkpoint file."""
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]


class ModelRegistry:
    """Registry for managing trained model checkpoints."""
    
    def __init__(self, registry_path: str | Path = "model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: dict[str, ModelCheckpoint] = {}
        self._load()
    
    def _load(self):
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                data = json.load(f)
                for name, info in data.items():
                    self.models[name] = ModelCheckpoint(**info)
    
    def _save(self):
        data = {name: asdict(model) for name, model in self.models.items()}
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        name: str,
        path: str,
        model_type: str,
        architecture: str,
        task: str,
        training_data: str,
        generators: Optional[list[str]] = None,
        families: Optional[list[str]] = None,
        metrics: Optional[dict[str, float]] = None,
    ) -> ModelCheckpoint:
        """Register a new model checkpoint."""
        if name in self.models:
            raise ValueError(f"Model '{name}' already registered. Use update() or unregister() first.")
        
        model = ModelCheckpoint(
            name=name,
            path=str(Path(path).resolve()),
            model_type=model_type,
            architecture=architecture,
            task=task,
            training_data=training_data,
            generators=generators,
            families=families,
            metrics=metrics,
        )
        self.models[name] = model
        self._save()
        return model
    
    def get(self, name: str) -> Optional[ModelCheckpoint]:
        return self.models.get(name)
    
    def list_models(self, model_type: Optional[str] = None) -> list[ModelCheckpoint]:
        models = list(self.models.values())
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        return sorted(models, key=lambda m: m.created_at, reverse=True)
    
    def unregister(self, name: str) -> bool:
        if name in self.models:
            del self.models[name]
            self._save()
            return True
        return False
    
    def update_metrics(self, name: str, metrics: dict[str, float]) -> bool:
        if name in self.models:
            self.models[name].metrics = metrics
            self._save()
            return True
        return False
    
    def get_best(self, model_type: str, metric: str = "val_auc") -> Optional[ModelCheckpoint]:
        """Get best model by metric for a given type."""
        candidates = [m for m in self.models.values() if m.model_type == model_type and m.metrics and metric in m.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda m: m.metrics[metric])


# Default registry instance
default_registry = ModelRegistry()


def register_trained_model(
    name: str,
    checkpoint_path: str,
    model_type: str,
    architecture: str,
    training_data: str,
    **kwargs,
) -> ModelCheckpoint:
    """Convenience function to register a trained model."""
    return default_registry.register(
        name=name,
        path=checkpoint_path,
        model_type=model_type,
        architecture=architecture,
        task=model_type,
        training_data=training_data,
        **kwargs,
    )


def get_model_for_analyzer(
    model_type: str = "binary",
    registry: Optional[ModelRegistry] = None,
) -> Optional[str]:
    """Get best checkpoint path for analyzer."""
    reg = registry or default_registry
    best = reg.get_best(model_type)
    return best.path if best else None