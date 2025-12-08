"""
PAPR Memory Document Processing Module

This module provides pluggable document processing capabilities with support for:
- TensorLake.ai document processing
- Reducto AI document processing
- Gemini Vision fallback processing
- Multi-tenant organization/namespace scoping
- Temporal workflow integration for durable execution
- Real-time WebSocket status updates
"""

from .providers.base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from .providers.tensorlake import TensorLakeProvider
from .providers.reducto import ReductoProvider
from .providers.gemini import GeminiVisionProvider
from .provider_manager import ProviderRegistry, DocumentProcessorFactory
from .security import FileValidator, SecureFileStorage
from .websocket_manager import WebSocketManager

__all__ = [
    "DocumentProvider",
    "ProcessingResult",
    "DocumentPage",
    "ProcessingStatus",
    "TensorLakeProvider",
    "ReductoProvider",
    "GeminiVisionProvider",
    "ProviderRegistry",
    "DocumentProcessorFactory",
    "FileValidator",
    "SecureFileStorage",
    "WebSocketManager"
]