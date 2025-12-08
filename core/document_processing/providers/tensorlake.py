"""
TensorLake.ai document processing provider
"""

import httpx
# Ensure .env is loaded for API keys/base URL (conditionally based on USE_DOTENV)
try:
    from dotenv import find_dotenv, load_dotenv
    use_dotenv = env.get("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        _ENV_FILE = find_dotenv()
        if _ENV_FILE:
            load_dotenv(_ENV_FILE)
except Exception:
    pass
import asyncio
from os import environ as env
from typing import Dict, Any, Optional, Callable, List
from .base import DocumentProvider, ProcessingResult, DocumentPage, ProcessingStatus
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)


class TensorLakeProvider(DocumentProvider):
    """TensorLake.ai document processing provider"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url", "https://api.tensorlake.ai")
        self.timeout = config.get("timeout", 300)

    async def process_document(
        self,
        file_content: bytes,
        filename: str,
        upload_id: str,
        progress_callback: Optional[Callable] = None
    ) -> ProcessingResult:
        """Process document using TensorLake SDK
        
        Uses the official tensorlake.documentai SDK to upload and parse documents.
        The SDK returns a ParseResult object with chunks containing actual markdown content.
        
        Reference: https://pypi.org/project/tensorlake/
        """

        logger.info(f"Processing document {filename} with TensorLake SDK (upload_id: {upload_id})")

        try:
            # Import TensorLake SDK - now using the correct module!
            from tensorlake.documentai import DocumentAI, ParseStatus
            import tempfile
            import os
            
            # Initial progress update
            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.1, None, None)
            
            # Fix SSL certificate issue - TensorLake SDK uses httpx which needs proper SSL config
            # Save original SSL_CERT_FILE if it exists
            original_ssl_cert = env.get("SSL_CERT_FILE")
            try:
                # Use certifi's certificate bundle
                import certifi
                env["SSL_CERT_FILE"] = certifi.where()
                logger.info(f"Set SSL_CERT_FILE to certifi bundle: {certifi.where()}")
            except ImportError:
                # If certifi not available, unset SSL_CERT_FILE to use system defaults
                if "SSL_CERT_FILE" in env:
                    del env["SSL_CERT_FILE"]
                logger.info("Unset SSL_CERT_FILE to use system defaults")
            
            try:
                # Initialize DocumentAI client
                doc_ai = DocumentAI(api_key=self.api_key)
                logger.info("TensorLake DocumentAI client initialized")
            finally:
                # Restore original SSL_CERT_FILE setting
                if original_ssl_cert is not None:
                    env["SSL_CERT_FILE"] = original_ssl_cert
                elif "SSL_CERT_FILE" in env:
                    del env["SSL_CERT_FILE"]
            
            # Step 1: Write file to temp location (SDK requires file path)
            with tempfile.NamedTemporaryFile(suffix=f"_{filename}", delete=False) as temp_file:
                temp_file.write(file_content)
                temp_path = temp_file.name
            
            try:
                # Step 2: Upload file using SDK
                logger.info(f"Uploading file to TensorLake using SDK...")
                file_id = await asyncio.to_thread(doc_ai.upload, temp_path)
                logger.info(f"File uploaded successfully, file_id: {file_id}")
                
                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.3, None, None)
                
                # Step 3: Parse document using SDK
                logger.info(f"Starting document parsing...")
                parse_id = await asyncio.to_thread(doc_ai.parse, file_id)
                logger.info(f"Document parsing started, parse_id: {parse_id}")
                
                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.5, None, None)
                
                # Step 4: Wait for completion using SDK
                logger.info(f"Waiting for parsing to complete...")
                result = await asyncio.to_thread(doc_ai.wait_for_completion, parse_id)
                
                logger.info(f"Parsing completed with status: {result.status}")
                
                if progress_callback:
                    await progress_callback(upload_id, ProcessingStatus.PROCESSING, 0.9, None, None)
                
                # Step 5: Get the full parsed result with all chunks
                # wait_for_completion might not return all data, so fetch explicitly
                logger.info(f"Fetching full parse result with chunks...")
                result = await asyncio.to_thread(doc_ai.get_parsed_result, parse_id)
                logger.info(f"Fetched result: status={result.status}, chunks={len(result.chunks) if result.chunks else 0}")
                
                # Step 6: Check status and extract content from chunks
                if result.status == ParseStatus.SUCCESSFUL or result.status == ParseStatus.COMPLETED:
                    # Extract content from chunks - this is the KEY part!
                    content_parts = []
                    
                    if result.chunks:
                        logger.info(f"Extracting content from {len(result.chunks)} chunks")
                        for chunk in result.chunks:
                            if hasattr(chunk, 'content') and chunk.content:
                                content_parts.append(chunk.content)
                                logger.debug(f"Chunk content length: {len(chunk.content)}")
                    else:
                        logger.error(f"❌ TensorLake returned NO CHUNKS! Result type: {type(result)}")
                        logger.error(f"   parse_id: {parse_id}, status: {result.status}")
                        logger.error(f"   parsed_pages_count: {result.parsed_pages_count}")
                        logger.error(f"   This means the SDK returned success but no content!")
                    
                    # Combine all chunks into full content
                    full_content = "\n".join(content_parts)
                    
                    if not full_content:
                        # CRITICAL: Content extraction failed, RAISE EXCEPTION instead of silent fallback
                        error_msg = (
                            f"TensorLake parsing succeeded but NO CONTENT was extracted. "
                            f"parse_id={parse_id}, file_id={file_id}, "
                            f"status={result.status}, chunks={len(result.chunks) if result.chunks else 0}"
                        )
                        logger.error(f"❌ {error_msg}")
                        raise Exception(error_msg)
                    
                    logger.info(f"✅ Successfully extracted {len(full_content)} chars from document")
                    
                    # Create DocumentPage with actual content
                    pages = [DocumentPage(
                        page_number=1,
                        content=full_content,  # ← ACTUAL CONTENT from chunks!
                        confidence=0.95,
                        metadata={
                            "file_id": file_id,
                            "parse_id": parse_id,
                            "parsed_pages_count": result.parsed_pages_count,
                            "total_chunks": len(content_parts)
                        }
                    )]
                    
                    if progress_callback:
                        await progress_callback(upload_id, ProcessingStatus.COMPLETED, 1.0, len(pages), len(pages))
                    
                    # Return result with actual content
                    return ProcessingResult(
                        pages=pages,
                        total_pages=len(pages),
                        processing_time=0,  # SDK handles timing internally
                        confidence=0.95,
                        metadata={
                            "file_id": file_id,
                            "parse_id": parse_id,
                            "parsed_pages_count": result.parsed_pages_count
                        },
                        provider_specific={
                            "file_id": file_id,
                            "parse_id": parse_id,
                            "content": full_content,  # ← Store actual content!
                            "status": str(result.status),
                            "parsed_pages_count": result.parsed_pages_count,
                            "chunks_count": len(content_parts)
                        }
                    )
                else:
                    error_msg = result.error or f"Parsing failed with status: {result.status}"
                    logger.error(f"Document parsing failed: {error_msg}")
                    raise Exception(f"Document parsing failed: {error_msg}")
            
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass
                    
        except ImportError as ie:
            logger.error(f"TensorLake SDK not installed: {ie}. Install with: pip install tensorlake")
            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.FAILED, 0.0, None, None)
            raise Exception("TensorLake SDK required. Install with: pip install tensorlake")
        except Exception as e:
            logger.error(f"TensorLake processing failed: {e}", exc_info=True)
            if progress_callback:
                await progress_callback(upload_id, ProcessingStatus.FAILED, 0.0, None, None)
            raise Exception(f"TensorLake processing failed: {str(e)}")

    async def fetch_parse_result(self, parse_id: str) -> Dict[str, Any]:
        """Fetch the full parsed result from TensorLake using SDK
        
        Uses doc_ai.get_parsed_result() to fetch the ParseResult object.
        
        Args:
            parse_id: The TensorLake parse ID
            
        Returns:
            Dict with parsed content including actual text content from chunks
            
        Reference: https://docs.tensorlake.ai/document-ingestion/parsing/parsed-document-reference
        """
        logger.info(f"Fetching TensorLake parse result for parse_id: {parse_id} using SDK")
        
        try:
            from tensorlake.documentai import DocumentAI, ParseStatus
            
            # Fix SSL certificate issue - same as in process_document
            original_ssl_cert = env.get("SSL_CERT_FILE")
            try:
                import certifi
                env["SSL_CERT_FILE"] = certifi.where()
            except ImportError:
                if "SSL_CERT_FILE" in env:
                    del env["SSL_CERT_FILE"]
            
            try:
                # Initialize SDK client
                doc_ai = DocumentAI(api_key=self.api_key)
            finally:
                # Restore original SSL_CERT_FILE setting
                if original_ssl_cert is not None:
                    env["SSL_CERT_FILE"] = original_ssl_cert
                elif "SSL_CERT_FILE" in env:
                    del env["SSL_CERT_FILE"]
            
            # Get parsed result using SDK
            result = await asyncio.to_thread(doc_ai.get_parsed_result, parse_id)
            
            logger.info(f"Fetched TensorLake result: status={result.status}")
            
            # Extract content from chunks - KEY FIX!
            content_parts = []
            
            if result.chunks:
                logger.info(f"Extracting content from {len(result.chunks)} chunks")
                for chunk in result.chunks:
                    if hasattr(chunk, 'content') and chunk.content:
                        content_parts.append(chunk.content)
            
            full_content = "\n".join(content_parts)
            
            if full_content:
                logger.info(f"Successfully extracted {len(full_content)} chars of actual content")
            else:
                logger.warning("No actual content found in TensorLake chunks")
            
            # Return with actual content
            return {
                "parse_id": parse_id,
                "status": str(result.status),
                "content": full_content,  # ← This is the key field with actual text!
                "parsed_pages_count": result.parsed_pages_count,
                "chunks_count": len(content_parts),
                "full_result": result.model_dump() if hasattr(result, 'model_dump') else {}
            }
            
        except ImportError:
            logger.error("TensorLake SDK not installed. Install with: pip install tensorlake")
            raise Exception("TensorLake SDK required. Install with: pip install tensorlake")
        except Exception as e:
            logger.error(f"Failed to fetch TensorLake parse result: {e}")
            raise Exception(f"Failed to fetch TensorLake result: {str(e)}")
    
    async def validate_config(self) -> bool:
        """Validate TensorLake configuration"""
        if not self.api_key:
            return False

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{self.base_url}/health",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"TensorLake config validation failed: {e}")
            return False

    def get_supported_formats(self) -> List[str]:
        return ["pdf", "png", "jpg", "jpeg", "webp", "tiff"]

    async def health_check(self) -> bool:
        return await self.validate_config()