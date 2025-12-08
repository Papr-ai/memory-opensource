"""
Google Gemini Vision document processing provider using the NEW google-genai SDK
Reference: https://googleapis.github.io/python-genai/
"""

import os
import asyncio
from typing import Dict, Any, Optional, Callable, List
from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


class GeminiVisionProvider(DocumentProvider):
    """Google Gemini Vision document processing provider using new google-genai SDK"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.model = config.get("model", "gemini-2.0-flash-exp")  # Using latest model

        # Import the NEW google-genai SDK
        try:
            from google import genai
            from google.genai import types
            
            self.genai = genai
            self.types = types
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Gemini client initialized with new google-genai SDK (model: {self.model})")
        except ImportError:
            logger.error("google-genai package not installed. Install with: pip install google-genai")
            self.genai = None
            self.types = None
            self.client = None

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process document using Gemini Vision API with new SDK"""

        if not self.client:
            raise Exception("Gemini client not available - check google-genai installation and API key")

        logger.info(f"Processing document {filename} with Gemini Vision (upload_id: {upload_id})")

        if progress_callback:
            await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.1, None, None)

        try:
            # Convert document to images if PDF, or process directly if image
            if filename.lower().endswith('.pdf'):
                pages = await self._process_pdf_pages(file_content, upload_id, progress_callback)
            else:
                pages = await self._process_single_image(file_content, filename, upload_id, progress_callback)

            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.COMPLETED, 1.0, None, None)

            # Combine all page content for provider_specific storage
            full_content = "\n\n".join([page.content for page in pages])
            
            logger.info(f"✅ Successfully extracted {len(full_content)} chars from {len(pages)} pages")

            return ProcessingResult(
                pages=pages,
                total_pages=len(pages),
                processing_time=0,  # Gemini doesn't provide timing
                confidence=0.95,    # Default confidence for Gemini
                metadata={},
                provider_specific={
                    "model": self.model,
                    "sdk": "google-genai",
                    "content": full_content,  # ← Store actual content!
                    "pages_count": len(pages)
                }
            )

        except Exception as e:
            logger.error(f"Gemini processing error: {e}")
            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.FAILED, 0.0, None, None)
            raise Exception(f"Gemini processing failed: {str(e)}")

    async def _process_pdf_pages(
        self,
        file_content: bytes,
        upload_id: str,
        progress_callback: Optional[Callable]
    ) -> List[DocumentPage]:
        """Process PDF by converting pages to images"""

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise Exception("PyMuPDF package not installed - required for PDF processing")

        doc = fitz.open(stream=file_content, filetype="pdf")
        pages = []
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            pix = page.get_pixmap()
            img_data = pix.tobytes("png")

            # Process with Gemini using new SDK
            prompt = "Extract all text content from this document page. Maintain formatting and structure."

            try:
                # Use new SDK API - pass image as Part
                image_part = self.types.Part.from_bytes(
                    data=img_data,
                    mime_type="image/png"
                )
                
                response = await self._generate_content_async([prompt, image_part])

                page_content = DocumentPage(
                    page_number=page_num + 1,
                    content=response.text if response.text else "",
                    confidence=0.95,  # Gemini doesn't provide confidence scores
                    metadata={"page_size": [pix.width, pix.height]}
                )
                pages.append(page_content)

                if progress_callback:
                    progress = 0.2 + (0.7 * (page_num + 1) / total_pages)
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, progress, page_num + 1, total_pages)

            except Exception as e:
                logger.error(f"Error processing page {page_num + 1}: {e}")
                # Continue with other pages
                continue

        doc.close()
        return pages

    async def _process_single_image(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable]
    ) -> List[DocumentPage]:
        """Process single image file"""

        # Determine MIME type from filename
        ext = filename.split('.')[-1].lower()
        mime_type = f"image/{ext}" if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif'] else "image/png"

        prompt = "Extract all text content from this document. Maintain formatting and structure."

        # Use new SDK API
        image_part = self.types.Part.from_bytes(
            data=file_content,
            mime_type=mime_type
        )
        
        response = await self._generate_content_async([prompt, image_part])

        if progress_callback:
            await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.8, 1, 1)

        return [DocumentPage(
            page_number=1,
            content=response.text if response.text else "",
            confidence=0.95,
            metadata={"mime_type": mime_type}
        )]

    async def _generate_content_async(self, contents):
        """Async wrapper for Gemini content generation using new SDK
        
        Args:
            contents: List of prompt string and Part objects
            
        Returns:
            Response object with .text attribute
        """

        # New SDK's generate_content is synchronous, so run in thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._generate_content_sync,
            contents
        )
    
    def _generate_content_sync(self, contents):
        """Synchronous content generation using new SDK"""
        return self.client.models.generate_content(
            model=self.model,
            contents=contents
        )

    async def validate_config(self) -> bool:
        """Validate Gemini configuration"""
        if not self.api_key or not self.client:
            return False

        try:
            # Test with a simple request
            response = await self._generate_content_async(["test"])
            return True
        except Exception as e:
            logger.error(f"Gemini config validation failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        return ["pdf", "png", "jpg", "jpeg", "webp", "gif"]

    async def health_check(self) -> bool:
        return await self.validate_config()
