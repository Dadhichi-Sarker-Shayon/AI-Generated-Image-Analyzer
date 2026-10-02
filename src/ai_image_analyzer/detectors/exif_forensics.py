from __future__ import annotations
from typing import Any, Optional
from dataclasses import dataclass
from ..analysis.analyzer import AnalysisData
from .base import BaseDetector, DetectorResult, DetectorStatus
from ..config import AnalysisConfig


@dataclass
class ExifMetadata:
    has_exif: bool = False
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    software: Optional[str] = None
    datetime_original: Optional[str] = None
    gps_info: Optional[dict] = None
    orientation: Optional[int] = None
    x_resolution: Optional[float] = None
    y_resolution: Optional[float] = None
    color_space: Optional[str] = None
    exif_version: Optional[str] = None
    user_comment: Optional[str] = None
    ai_keywords: list[str] = None  # midjourney, stable diffusion, etc.


class ExifForensicsDetector(BaseDetector):
    """EXIF/metadata forensics detector - checks for AI generation signatures in metadata."""
    name = "exif_forensics"
    
    AI_SOFTWARE_KEYWORDS = [
        "midjourney", "stable diffusion", "dall-e", "dalle", "openai",
        "adobe firefly", "canva", "bing image creator", "leonardo",
        "playground", "nightcafe", "artbreeder", "deepai", "craiyon",
        "stablediffusion", "sd.xl", "sdxl", "flux", "ideogram",
    ]
    
    AI_USER_COMMENT_KEYWORDS = [
        "midjourney", "stable diffusion", "dall-e", "prompt:", "negative prompt:",
        "steps:", "cfg:", "sampler:", "seed:", "model:", "loras:",
    ]
    
    def evaluate(self, data: AnalysisData) -> DetectorResult:
        exif_data = self._extract_exif(data.image_meta.get("file_bytes"))
        
        if not exif_data.has_exif:
            return DetectorResult(
                name=self.name,
                status=DetectorStatus.OK,
                score=0.0,
                confidence=0.1,
                reasoning=["No EXIF data found - could be stripped or AI-generated"],
                features={"has_exif": False},
            )
        
        reasoning = []
        ai_score = 0.0
        evidence_count = 0
        
        # Check software field
        if exif_data.software:
            software_lower = exif_data.software.lower()
            for kw in self.AI_SOFTWARE_KEYWORDS:
                if kw in software_lower:
                    ai_score += 0.4
                    evidence_count += 1
                    reasoning.append(f"Software field contains AI keyword: '{kw}'")
        
        # Check user comment
        if exif_data.user_comment:
            comment_lower = exif_data.user_comment.lower()
            for kw in self.AI_USER_COMMENT_KEYWORDS:
                if kw in comment_lower:
                    ai_score += 0.3
                    evidence_count += 1
                    reasoning.append(f"UserComment contains AI parameter: '{kw}'")
        
        # Check for missing camera info (typical of AI)
        if not exif_data.camera_make and not exif_data.camera_model:
            ai_score += 0.15
            evidence_count += 1
            reasoning.append("No camera make/model in EXIF (common in AI images)")
        
        # Check for AI-specific EXIF tags (some generators add custom tags)
        # This would need per-generator knowledge
        
        # Normalize score
        ai_score = min(1.0, ai_score)
        confidence = min(1.0, evidence_count * 0.3 + (0.2 if exif_data.has_exif else 0))
        
        if not reasoning:
            reasoning.append("EXIF present with camera metadata - consistent with real photo")
        
        return DetectorResult(
            name=self.name,
            status=DetectorStatus.OK,
            score=ai_score,
            confidence=confidence,
            reasoning=reasoning,
            features={
                "has_exif": exif_data.has_exif,
                "camera_make": exif_data.camera_make,
                "camera_model": exif_data.camera_model,
                "software": exif_data.software,
                "has_gps": exif_data.gps_info is not None,
                "ai_keywords_found": evidence_count > 0,
            },
        )
    
    def _extract_exif(self, file_bytes: Optional[bytes]) -> ExifMetadata:
        if not file_bytes:
            return ExifMetadata(has_exif=False)
        
        try:
            from PIL import Image
            import io
            
            img = Image.open(io.BytesIO(file_bytes))
            exif = img.getexif()
            
            if not exif:
                return ExifMetadata(has_exif=False)
            
            # Standard EXIF tags
            exif_data = ExifMetadata(has_exif=True)
            
            # Tag IDs: https://exiv2.org/tags.html
            exif_data.camera_make = exif.get(271)  # Make
            exif_data.camera_model = exif.get(272)  # Model
            exif_data.software = exif.get(305)      # Software
            exif_data.datetime_original = exif.get(36867)  # DateTimeOriginal
            exif_data.orientation = exif.get(274)   # Orientation
            exif_data.x_resolution = exif.get(282)  # XResolution
            exif_data.y_resolution = exif.get(283)  # YResolution
            exif_data.color_space = exif.get(40961) # ColorSpace
            exif_data.exif_version = exif.get(36864) # ExifVersion
            exif_data.user_comment = exif.get(37510) # UserComment
            
            # GPS info
            gps_ifd = exif.get_ifd(34853) if hasattr(exif, 'get_ifd') else None
            if gps_ifd:
                exif_data.gps_info = dict(gps_ifd)
            
            return exif_data
            
        except Exception:
            return ExifMetadata(has_exif=False)