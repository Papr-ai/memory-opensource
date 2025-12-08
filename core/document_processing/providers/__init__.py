"""
Document Processing Providers

Pluggable document processing providers for PAPR Memory.
"""

from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from .tensorlake import TensorLakeProvider
from .reducto import ReductoProvider
from .gemini import GeminiVisionProvider

__all__ = [
    "DocumentProvider",
    "ProcessingResult",
    "DocumentPage",
    "ProcessingStatus",
    "TensorLakeProvider",
    "ReductoProvider",
    "GeminiVisionProvider"
]