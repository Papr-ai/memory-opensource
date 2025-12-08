"""
DeepSeek-OCR Provider

Provider implementation for deepseek-ai/DeepSeek-OCR from HuggingFace
"""

import os
import io
from typing import Dict, Any, Optional, Callable, List
from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from services.logger_singleton import LoggerSingleton
import httpx

logger = LoggerSingleton.get_logger(__name__)


class DeepSeekOCRProvider(DocumentProvider):
    """DeepSeek-OCR document processing provider
    
    Supports:
    - DeepInfra API (recommended, requires DEEPINFRA_TOKEN)
    - Self-hosted deployment (requires custom base_url)
    
    Note: DeepSeek models are NOT available on HuggingFace Inference API.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Primary: DeepInfra API token
        self.api_key = (
            config.get("api_key") or 
            os.getenv("DEEPINFRA_TOKEN") or 
            os.getenv("DEEPSEEK_API_KEY")
        )
        self.model_id = config.get("model_id", "deepseek-ai/DeepSeek-Janus-1.3B")
        
        # Default to DeepInfra - the primary supported platform
        self.base_url = config.get("base_url") or os.getenv("DEEPSEEK_API_URL", "https://api.deepinfra.com/v1/inference")
        self.timeout = config.get("timeout", 120)
        
        logger.info(f"DeepSeek provider initialized with base_url: {self.base_url}")
        logger.info(f"DeepSeek model: {self.model_id}")

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process a document using DeepSeek-OCR"""
        start_time = __import__('time').time()
        
        try:
            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.PROCESSING,
                    "progress": 0.0,
                    "current_filename": filename,
                    "provider": "deepseek-ocr"
                })

            # Call DeepSeek-OCR via HuggingFace API
            pages = await self._process_with_deepseek(file_content, filename, upload_id, progress_callback)
            
            processing_time = __import__('time').time() - start_time
            
            # Calculate overall confidence
            total_conf = sum(p.confidence for p in pages)
            avg_confidence = total_conf / len(pages) if pages else 0.0

            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.COMPLETED,
                    "progress": 1.0,
                    "total_pages": len(pages),
                    "provider": "deepseek-ocr"
                })

            return ProcessingResult(
                pages=pages,
                total_pages=len(pages),
                processing_time=processing_time,
                confidence=avg_confidence,
                metadata={
                    "provider": "deepseek-ocr",
                    "model_id": self.model_id,
                    "filename": filename
                },
                provider_specific={
                    "model_id": self.model_id,
                    "ocr_results": [
                        {
                            "page_number": p.page_number,
                            "text": p.content,
                            "confidence": p.confidence
                        }
                        for p in pages
                    ]
                }
            )

        except Exception as e:
            logger.error(f"DeepSeek-OCR processing failed: {e}", exc_info=True)
            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.FAILED,
                    "error": str(e),
                    "provider": "deepseek-ocr"
                })
            raise

    async def _process_with_deepseek(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable]
    ) -> List[DocumentPage]:
        """Process document with DeepSeek-OCR API"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        
        # Prepare the image/document for OCR
        files = {
            "inputs": file_content
        }
        
        api_url = f"{self.base_url}/{self.model_id}"
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                api_url,
                headers=headers,
                files=files
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek-OCR API error: {response.status_code} - {response.text}")
            
            result = response.json()
            
        # Parse DeepSeek-OCR response
        pages = []
        
        # DeepSeek-OCR typically returns OCR text directly
        if isinstance(result, list) and result:
            for idx, item in enumerate(result):
                if isinstance(item, dict):
                    text = item.get("generated_text", "") or item.get("text", "")
                    confidence = item.get("confidence", 0.9)
                else:
                    text = str(item)
                    confidence = 0.9
                    
                pages.append(DocumentPage(
                    page_number=idx + 1,
                    content=text,
                    confidence=confidence,
                    metadata={"model": self.model_id}
                ))
        elif isinstance(result, dict):
            text = result.get("generated_text", "") or result.get("text", "") or str(result)
            pages.append(DocumentPage(
                page_number=1,
                content=text,
                confidence=result.get("confidence", 0.9),
                metadata={"model": self.model_id}
            ))
        else:
            # Fallback: treat entire response as text
            pages.append(DocumentPage(
                page_number=1,
                content=str(result),
                confidence=0.8,
                metadata={"model": self.model_id}
            ))
        
        return pages

    async def validate_config(self) -> bool:
        """Validate DeepSeek-OCR configuration"""
        if not self.api_key:
            logger.error("DeepSeek-OCR API key not provided")
            return False
        return True

    def get_supported_formats(self) -> List[str]:
        return ["pdf", "png", "jpg", "jpeg", "webp", "tiff"]

    async def health_check(self) -> bool:
        """Check if DeepSeek-OCR API is accessible"""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self.base_url}/{self.model_id}",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"DeepSeek-OCR health check failed: {e}")
            return False

