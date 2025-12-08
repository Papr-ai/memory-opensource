"""
PaddleOCR Provider Types

Pydantic models for PaddleOCR responses from PaddlePaddle/PaddleOCR
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class OCRBox(BaseModel):
    """Bounding box for detected text (4 corner points)"""
    points: List[List[float]]  # [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]


class OCRDetection(BaseModel):
    """Single OCR detection result from PaddleOCR"""
    box: OCRBox
    text: str
    confidence: float


class TableCell(BaseModel):
    """Table cell information"""
    bbox: List[float]  # [x1, y1, x2, y2]
    text: str
    row: int
    col: int
    row_span: int = 1
    col_span: int = 1


class TableResult(BaseModel):
    """Table recognition result"""
    bbox: List[float]  # [x1, y1, x2, y2]
    cells: List[TableCell]
    html: Optional[str] = None


class LayoutElement(BaseModel):
    """Layout analysis element"""
    bbox: List[float]  # [x1, y1, x2, y2]
    type: str  # text, title, figure, table, etc.
    confidence: float
    text: Optional[str] = None


class PageResult(BaseModel):
    """OCR result for a single page"""
    page_number: int = 1
    ocr_results: List[OCRDetection] = Field(default_factory=list)
    tables: List[TableResult] = Field(default_factory=list)
    layout: List[LayoutElement] = Field(default_factory=list)
    full_text: Optional[str] = None


class PaddleOCRResponse(BaseModel):
    """
    Response model for PaddleOCR
    
    PaddleOCR supports:
    - Multi-language OCR (80+ languages)
    - Table recognition
    - Layout analysis
    - Document structure recognition
    """
    results: Union[List[OCRDetection], List[PageResult]]
    model_version: Optional[str] = "PP-OCRv4"
    processing_time: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(extra="allow")

