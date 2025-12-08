"""
Base classes for document processing providers
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from pydantic import BaseModel
from datetime import datetime


class ProcessingStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class DocumentPage(BaseModel):
    page_number: int
    content: str
    confidence: float
    metadata: Dict[str, Any] = {}


class ProcessingResult(BaseModel):
    pages: List[DocumentPage]
    total_pages: int
    processing_time: float
    confidence: float
    metadata: Dict[str, Any] = {}
    provider_specific: Dict[str, Any] = {}


class ProcessingProgress(BaseModel):
    upload_id: str
    status: ProcessingStatus
    progress: float  # 0.0 to 1.0
    current_page: Optional[int] = None
    total_pages: Optional[int] = None
    current_filename: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime
    provider: str


class DocumentProvider(ABC):
    """Abstract base class for document processing providers"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider_name = self.__class__.__name__.replace("Provider", "").lower()

    @abstractmethod
    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process a document and return extracted content"""
        pass

    @abstractmethod
    async def validate_config(self) -> bool:
        """Validate provider configuration"""
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file formats"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is healthy"""
        pass