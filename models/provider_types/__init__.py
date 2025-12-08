"""
Provider Type Definitions

Pydantic models for various document processing provider responses.
"""

try:
    from .deepseek import DeepSeekOCRResponse
except ImportError:
    DeepSeekOCRResponse = None

try:
    from .paddleocr import PaddleOCRResponse
except ImportError:
    PaddleOCRResponse = None

__all__ = [
    "DeepSeekOCRResponse",
    "PaddleOCRResponse",
]

