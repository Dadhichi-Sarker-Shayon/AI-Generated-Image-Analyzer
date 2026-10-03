"""AI Image Analyzer - Detect AI-generated images with forensic metrics."""
from __future__ import annotations

__version__ = "0.2.0"

from .analyzer import AIImageAnalyzer
from .config import AnalysisConfig
from .explain.report import AnalysisReport, Finding

__all__ = [
    "AIImageAnalyzer",
    "AnalysisConfig",
    "AnalysisReport",
    "Finding",
    "__version__",
]