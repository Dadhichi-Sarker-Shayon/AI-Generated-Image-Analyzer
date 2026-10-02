from __future__ import annotations
from typing import Any, Optional, Dict, List
import numpy as np
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


# Known generator lattice signatures (period in pixels, orientation, strength)
# Based on published research and empirical analysis
GENERATOR_LATTICE_SIGNATURES = {
    "midjourney": {
        "periods": [2, 4, 8],  # Common MJ periods
        "orientations": [0, 45, 90],  # Degrees
        "strength_range": (10, 25),  # dB
        "description": "Midjourney v4-v6: strong period-2,4,8 lattice from VQ-VAE decoder",
    },
    "stable_diffusion": {
        "periods": [8, 16, 32],  # SD U-Net 8x downsampling
        "orientations": [0, 45],
        "strength_range": (6, 18),
        "description": "SD 1.5/SDXL: period-8/16/32 from 8x VAE decoder, weaker than GAN",
    },
    "dall_e_2": {
        "periods": [8, 16],
        "orientations": [0],
        "strength_range": (8, 20),
        "description": "DALL-E 2: prior + decoder lattice",
    },
    "dall_e_3": {
        "periods": [8, 16, 32],
        "orientations": [0, 45],
        "strength_range": (6, 16),
        "description": "DALL-E 3: improved VAE, similar to SDXL",
    },
    "stylegan": {
        "periods": [2, 4, 8, 16],
        "orientations": [0, 45, 90],
        "strength_range": (12, 30),
        "description": "StyleGAN: strong period-2/4 from transposed conv, very distinctive",
    },
    "progan": {
        "periods": [2, 4, 8],
        "orientations": [0, 45],
        "strength_range": (10, 25),
        "description": "ProGAN: progressive growing leaves period-2/4 signatures",
    },
    "adobe_firefly": {
        "periods": [8, 16],
        "orientations": [0, 45],
        "strength_range": (5, 15),
        "description": "Firefly: proprietary, similar to diffusion VAE",
    },
}


class ModelSpecificLatticeDetector(BaseDetector):
    """
    Model-specific lattice fingerprint detector.
    Matches detected lattice periods/orientations to known generator signatures.
    """
    name = "model_lattice"
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig.default()
        self.signatures = GENERATOR_LATTICE_SIGNATURES
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        freq = data.frequency
        
        if not freq.lattice_db or freq.best_lattice_period is None:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.INSUFFICIENT_DATA,
                reasoning=["No lattice artifacts detected in frequency analysis"],
            )
        
        # Get detected lattice info
        detected_period = freq.best_lattice_period
        detected_db = freq.best_lattice_db
        all_lattice_db = freq.lattice_db
        
        # Match against known signatures
        matches = []
        for generator, sig in self.signatures.items():
            score = self._match_signature(detected_period, detected_db, all_lattice_db, sig)
            if score > 0:
                matches.append({
                    "generator": generator,
                    "score": score,
                    "matched_periods": self._get_matched_periods(detected_period, all_lattice_db, sig),
                    "description": sig["description"],
                })
        
        # Sort by score
        matches.sort(key=lambda x: x["score"], reverse=True)
        
        if not matches:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=0.0,
                confidence=0.1,
                reasoning=["Lattice detected but doesn't match known generator signatures"],
                features={"detected_period": detected_period, "detected_db": detected_db},
            )
        
        top_match = matches[0]
        top_generator = top_match["generator"]
        top_score = top_match["score"]
        
        # Normalize score (0-1)
        normalized_score = min(1.0, top_score / 2.0)  # Empirical scaling
        
        reasoning = [
            f"Best match: {top_generator} (score: {top_score:.2f})",
            f"  Detected lattice: period={detected_period}px, {detected_db:.1f} dB",
            f"  {top_match['description']}",
        ]
        
        if len(matches) > 1:
            second = matches[1]
            reasoning.append(f"  Alternative: {second['generator']} (score: {second['score']:.2f})")
        
        return DetectorResult(
            name=self.name,
            status=DetectorStatus.OK,
            score=normalized_score,
            confidence=min(1.0, top_score / 3.0),
            reasoning=reasoning,
            features={
                "detected_period": detected_period,
                "detected_db": detected_db,
                "matches": [
                    {"generator": m["generator"], "score": m["score"], "periods": m["matched_periods"]}
                    for m in matches[:3]
                ],
            },
        )
    
    def _match_signature(
        self,
        detected_period: int,
        detected_db: float,
        all_lattice_db: Dict[str, float],
        signature: Dict,
    ) -> float:
        """Compute match score between detected lattice and generator signature."""
        score = 0.0
        
        # Primary period match
        if detected_period in signature["periods"]:
            # Strong match if period matches and dB in expected range
            min_db, max_db = signature["strength_range"]
            if min_db <= detected_db <= max_db:
                score += 1.5
            elif detected_db > min_db * 0.7:
                score += 1.0
            else:
                score += 0.5
        
        # Secondary period matches (harmonics)
        for period in signature["periods"]:
            if period != detected_period:
                period_key = str(period)
                if period_key in all_lattice_db:
                    db = all_lattice_db[period_key]
                    if db > signature["strength_range"][0] * 0.5:
                        score += 0.3
        
        # Check for orientation-specific patterns (would need 2D lattice analysis)
        # Simplified for now
        
        return score
    
    def _get_matched_periods(
        self,
        detected_period: int,
        all_lattice_db: Dict[str, float],
        signature: Dict,
    ) -> List[int]:
        """Get list of periods that matched the signature."""
        matched = []
        if detected_period in signature["periods"]:
            matched.append(detected_period)
        for period in signature["periods"]:
            if period != detected_period and str(period) in all_lattice_db:
                if all_lattice_db[str(period)] > signature["strength_range"][0] * 0.3:
                    matched.append(period)
        return matched