"""
Reducto AI document processing provider
"""

import httpx
import asyncio
import tempfile
from pathlib import Path
import os
from os import environ as env
from typing import Dict, Any, Optional, Callable, List
from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from services.logger_singleton import LoggerSingleton

try:
    from reducto import Reducto, AsyncReducto
    REDUCTO_AVAILABLE = True
except ImportError:
    REDUCTO_AVAILABLE = False
    Reducto = None
    AsyncReducto = None

logger = LoggerSingleton.get_logger(__name__)


class ReductoProvider(DocumentProvider):
    """Reducto AI document processing provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or env.get("REDUCTO_API_KEY")
        # Reducto SDK prefers environment selection, not base URL
        # Supported: 'production', 'eu', 'au' (SDK does not accept 'us', use 'production')
        env_name = config.get("environment") or env.get("REDUCTO_ENVIRONMENT", "us")
        if env_name == "us":
            env_name = "production"
        self.environment = env_name
        self.timeout = config.get("timeout", 300)

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process document using Reducto AI SDK"""

        if not REDUCTO_AVAILABLE:
            raise Exception("Reducto library not available")

        logger.info(f"Processing document {filename} with Reducto (upload_id: {upload_id})")

        if progress_callback:
            await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.1, 1, 1)

        try:
            # Initialize Reducto client (sync API used with temporary file)
            client = Reducto(api_key=self.api_key, environment=self.environment)

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file.flush()
                temp_path = Path(temp_file.name)

            try:
                # Upload file to Reducto
                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.3, 1, 1)

                upload = client.upload(file=temp_path)

                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.6, 1, 1)

                # Process with pipeline
                pipeline_id = self.config.get("pipeline_id") or os.getenv("REDUCTO_PIPELINE_ID", "k977pgfmaqm5h9p0nqr5x2hs7d7s2v5g")
                
                # Wrap synchronous Reducto call in asyncio executor with timeout
                # Reducto can take 20+ minutes for large documents, so use a generous timeout
                timeout_seconds = self.config.get("pipeline_timeout", 3600)  # Default: 1 hour
                logger.info(f"Starting Reducto pipeline with {timeout_seconds}s timeout")
                
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            client.pipeline.run,
                            document_url=upload,
                            pipeline_id=pipeline_id
                        ),
                        timeout=timeout_seconds
                    )
                except asyncio.TimeoutError:
                    raise Exception(f"Reducto processing timed out after {timeout_seconds} seconds")

                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.9, 1, 1)

                # Parse results into our format
                pages = []
                if hasattr(result, 'content') and result.content:
                    pages.append(DocumentPage(
                        page_number=1,
                        content=str(result.content),
                        confidence=getattr(result, 'confidence', 0.9),
                        metadata=getattr(result, 'metadata', {})
                    ))
                elif isinstance(result, dict):
                    content = result.get("content", "")
                    pages.append(DocumentPage(
                        page_number=1,
                        content=str(content),
                        confidence=result.get("confidence", 0.9),
                        metadata=result.get("metadata", {})
                    ))
                else:
                    # Fallback: convert result to string
                    pages.append(DocumentPage(
                        page_number=1,
                        content=str(result),
                        confidence=0.9,
                        metadata={}
                    ))

                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.COMPLETED, 1.0, 1, 1)

                # Convert result to dict for Pydantic validation
                provider_specific = {}
                if hasattr(result, '__dict__'):
                    provider_specific = result.__dict__
                elif isinstance(result, dict):
                    provider_specific = result
                else:
                    provider_specific = {"raw_result": str(result)}

                return ProcessingResult(
                    pages=pages,
                    total_pages=len(pages),
                    processing_time=0,  # Reducto doesn't provide this
                    confidence=pages[0].confidence if pages else 0.9,
                    metadata={},
                    provider_specific=provider_specific
                )

            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()

        except Exception as e:
            logger.error(f"Reducto processing failed: {e}")
            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.FAILED, 0.0, 1, 1)
            raise Exception(f"Reducto processing failed: {str(e)}")

    async def validate_config(self) -> bool:
        """Validate Reducto configuration"""
        if not REDUCTO_AVAILABLE:
            return False
        if not self.api_key:
            return False

        # Try a lightweight SDK call to validate environment/connectivity
        try:
            sdk = Reducto(api_key=self.api_key, environment=self.environment)
            # Try to access a simple attribute or method to validate the SDK
            # The version attribute might not exist in all SDK versions
            if hasattr(sdk, 'version') and hasattr(sdk.version, 'retrieve'):
                _ = sdk.version.retrieve()
            else:
                # Just check if the SDK was created successfully
                logger.info("Reducto SDK created successfully (version check not available)")
            return True
        except Exception as e:
            logger.error(f"Reducto config validation failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        return ["pdf", "docx", "pptx", "html", "txt"]

    async def health_check(self) -> bool:
        return await self.validate_config()