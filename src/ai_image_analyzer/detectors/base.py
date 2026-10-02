from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any
from pydantic import BaseModel


class DetectorStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    INSUFFICIENT_DATA = "insufficient_data"


class DetectorResult(BaseModel):
    name: str
    status: DetectorStatus
    score: float | None = None
    confidence: float | None = None
    reasoning: list[str] = []
    features: dict[str, Any] = {}


class BaseDetector(ABC):
    name: str = "base"

    @abstractmethod
    def evaluate(self, data: Any) -> DetectorResult:
        pass

    def score_safe(self, data: Any) -> DetectorResult:
        try:
            return self.evaluate(data)
        except Exception as e:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.UNAVAILABLE,
                reasoning=[f"Error: {type(e).__name__}: {e}"],
            )