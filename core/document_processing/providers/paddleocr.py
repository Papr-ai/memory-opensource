"""
PaddleOCR Provider

Provider implementation for PaddlePaddle/PaddleOCR
"""

import os
import io
from typing import Dict, Any, Optional, Callable, List
from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


class PaddleOCRProvider(DocumentProvider):
    """PaddleOCR document processing provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lang = config.get("lang", "en")
        self.use_angle_cls = config.get("use_angle_cls", True)
        # Note: use_gpu is deprecated in PaddleOCR 3.3.0+
        # GPU is auto-detected, use device parameter for explicit control
        self.device = config.get("device", "cpu")  # "cpu", "gpu", or "gpu:0"
        self.det_model_dir = config.get("det_model_dir")
        self.rec_model_dir = config.get("rec_model_dir")
        self.cls_model_dir = config.get("cls_model_dir")
        self.ocr_engine = None
        self._initialized = False

    def _initialize_ocr(self):
        """Lazy initialization of PaddleOCR engine"""
        if self._initialized:
            return
            
        try:
            from paddleocr import PaddleOCR
            
            # PaddleOCR 3.3.0+ uses 'device' instead of 'use_gpu'
            kwargs = {
                "use_angle_cls": self.use_angle_cls,
                "lang": self.lang,
                "device": self.device,
                "show_log": False
            }
            
            if self.det_model_dir:
                kwargs["det_model_dir"] = self.det_model_dir
            if self.rec_model_dir:
                kwargs["rec_model_dir"] = self.rec_model_dir
            if self.cls_model_dir:
                kwargs["cls_model_dir"] = self.cls_model_dir
                
            self.ocr_engine = PaddleOCR(**kwargs)
            self._initialized = True
            logger.info(f"PaddleOCR initialized with lang={self.lang}")
            
        except ImportError:
            logger.error("PaddleOCR library not installed. Install with: pip install paddleocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process a document using PaddleOCR"""
        start_time = __import__('time').time()
        
        try:
            self._initialize_ocr()
            
            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.PROCESSING,
                    "progress": 0.0,
                    "current_filename": filename,
                    "provider": "paddleocr"
                })

            # Process with PaddleOCR
            pages = await self._process_with_paddleocr(
                file_content, filename, upload_id, progress_callback
            )
            
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
                    "provider": "paddleocr"
                })

            return ProcessingResult(
                pages=pages,
                total_pages=len(pages),
                processing_time=processing_time,
                confidence=avg_confidence,
                metadata={
                    "provider": "paddleocr",
                    "lang": self.lang,
                    "filename": filename
                },
                provider_specific={
                    "lang": self.lang,
                    "results": [
                        {
                            "page_number": p.page_number,
                            "text": p.content,
                            "confidence": p.confidence,
                            "metadata": p.metadata
                        }
                        for p in pages
                    ]
                }
            )

        except Exception as e:
            logger.error(f"PaddleOCR processing failed: {e}", exc_info=True)
            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.FAILED,
                    "error": str(e),
                    "provider": "paddleocr"
                })
            raise

    async def _process_with_paddleocr(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable]
    ) -> List[DocumentPage]:
        """Process document with PaddleOCR"""
        import asyncio
        from PIL import Image
        import io as io_lib
        
        pages = []
        
        # Check if PDF or image
        if filename.lower().endswith('.pdf'):
            # Convert PDF to images
            pages_data = await self._pdf_to_images(file_content)
        else:
            # Single image
            pages_data = [(1, file_content)]
        
        total_pages = len(pages_data)
        
        for page_idx, (page_num, img_bytes) in enumerate(pages_data):
            if progress_callback:
                await progress_callback({
                    "upload_id": upload_id,
                    "status": ProcessingStatus.PROCESSING,
                    "progress": page_idx / total_pages,
                    "current_page": page_num,
                    "total_pages": total_pages,
                    "provider": "paddleocr"
                })
            
            # Run OCR in thread pool to avoid blocking
            result = await asyncio.to_thread(self._ocr_image, img_bytes)
            
            if result:
                # Extract text and confidence from PaddleOCR result
                text_lines = []
                confidences = []
                
                for line in result:
                    if len(line) >= 2:
                        # line[0] is bbox, line[1] is (text, confidence)
                        text, conf = line[1]
                        text_lines.append(text)
                        confidences.append(conf)
                
                full_text = "\n".join(text_lines)
                avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
                
                pages.append(DocumentPage(
                    page_number=page_num,
                    content=full_text,
                    confidence=avg_conf,
                    metadata={
                        "lang": self.lang,
                        "detections": len(result),
                        "raw_result": result
                    }
                ))
        
        return pages

    def _ocr_image(self, img_bytes: bytes) -> List:
        """Run OCR on single image (synchronous)"""
        from PIL import Image
        import io as io_lib
        
        img = Image.open(io_lib.BytesIO(img_bytes))
        result = self.ocr_engine.ocr(img, cls=self.use_angle_cls)
        
        # PaddleOCR returns nested list for batch processing
        if result and isinstance(result, list) and len(result) > 0:
            return result[0] if result[0] else []
        return []

    async def _pdf_to_images(self, pdf_bytes: bytes) -> List[tuple]:
        """Convert PDF to images"""
        try:
            import fitz  # PyMuPDF
            import io as io_lib
            
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            pages = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                img_bytes = pix.tobytes("png")
                pages.append((page_num + 1, img_bytes))
            
            doc.close()
            return pages
            
        except ImportError:
            logger.error("PyMuPDF not installed. Install with: pip install pymupdf")
            raise
        except Exception as e:
            logger.error(f"PDF conversion failed: {e}")
            raise

    async def validate_config(self) -> bool:
        """Validate PaddleOCR configuration"""
        try:
            self._initialize_ocr()
            return self._initialized
        except Exception as e:
            logger.error(f"PaddleOCR config validation failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        return ["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "webp"]

    async def health_check(self) -> bool:
        """Check if PaddleOCR is working"""
        try:
            self._initialize_ocr()
            return self._initialized
        except Exception as e:
            logger.error(f"PaddleOCR health check failed: {e}")
            return False

