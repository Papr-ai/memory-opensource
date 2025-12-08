"""
DeepSeek-OCR Provider Types

Pydantic models for DeepSeek-OCR responses from deepseek-ai/DeepSeek-OCR
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected text"""
    x: float
    y: float
    width: float
    height: float


class TextDetection(BaseModel):
    """Single text detection result"""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBox] = None
    language: Optional[str] = None


class OCRPage(BaseModel):
    """OCR result for a single page"""
    page_number: int
    detections: List[TextDetection]
    full_text: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DeepSeekOCRResponse(BaseModel):
    """
    Response model for DeepSeek-OCR
    
    DeepSeek-OCR is a multimodal model for OCR and document understanding.
    """
    ocr_results: List[OCRPage]
    model_version: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="allow")

