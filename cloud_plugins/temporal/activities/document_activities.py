"""
Temporal activities for document processing
"""

import hashlib
import httpx
import hmac
import json
import gzip
from temporalio import activity
import os
from os import environ as env
from typing import Dict, Any, Optional, List
from models.shared_types import MemoryMetadata, PreferredProvider
from datetime import datetime

from core.document_processing.provider_manager import TenantConfigManager, DocumentProcessorFactory
from core.document_processing.parse_integration import ParseDocumentIntegration
from core.document_processing.security import FileValidator
from services.multi_tenant_utils import apply_multi_tenant_scoping_to_batch_request
from models.memory_models import AddMemoryRequest, BatchMemoryRequest
# Lazy import inside function to avoid importing heavy modules (Stripe, routes) during test collection
from memory.memory_graph import MemoryGraph
from services.logger_singleton import LoggerSingleton

logger = LoggerSingleton.get_logger(__name__)

# Load environment variables from .env if present (activities run in worker process)
try:
    import os
    from dotenv import find_dotenv, load_dotenv
    # Respect USE_DOTENV setting
    use_dotenv = os.getenv("USE_DOTENV", "true").lower() == "true"
    if use_dotenv:
        _ENV_FILE = find_dotenv()
        if _ENV_FILE:
            load_dotenv(_ENV_FILE)
except Exception:
    pass

# Import hierarchical processing components
try:
    from models.hierarchical_models import (
        DocumentToMemoryTransformer, ProviderContentExtractor, MemoryTransformer,
        HierarchicalProcessingConfig, ContentElement, TableElement, ImageElement,
        ContentType
    )
    from models.shared_types import MemoryMetadata
    from core.document_processing.llm_memory_generator import (
        LLMMemoryStructureGenerator, generate_optimized_memory_structures
    )
    # Note: Using inline grouping logic instead of HierarchicalChunker class (see lines 1462-1542)
    HIERARCHICAL_AVAILABLE = True
except ImportError as e:
    HIERARCHICAL_AVAILABLE = False
    logger.warning(f"Hierarchical processing models not available: {e}")


@activity.defn
async def validate_document(
    file_content: bytes,
    filename: str,
    organization_id: Optional[str],
    namespace_id: Optional[str]
) -> Dict[str, Any]:
    """Validate document before processing"""

    activity.heartbeat("Validating document")

    # File size validation
    max_size = 50 * 1024 * 1024  # 50MB default
    if len(file_content) > max_size:
        return {"valid": False, "error": f"File too large: {len(file_content)} bytes"}

    # File type validation
    allowed_extensions = [".pdf", ".png", ".jpg", ".jpeg", ".webp", ".html", ".txt", ".docx"]
    if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
        return {"valid": False, "error": f"Unsupported file type: {filename}"}

    # Content validation (basic checks)
    if len(file_content) < 100:  # Minimum file size
        return {"valid": False, "error": "File appears to be empty or corrupted"}

    # Use FileValidator for comprehensive validation
    try:
        is_valid, validation_error = await FileValidator.validate_file(
            file_content, filename
        )
        if not is_valid:
            return {"valid": False, "error": validation_error}
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}

    # Calculate file hash for deduplication
    file_hash = hashlib.sha256(file_content).hexdigest()

    return {
        "valid": True,
        "file_hash": file_hash,
        "file_size": len(file_content),
        "file_type": filename.split('.')[-1].lower()
    }


@activity.defn
async def process_document_with_provider(
    file_content: bytes,
    filename: str,
    upload_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str],
    preferred_provider: Optional[PreferredProvider]
) -> Dict[str, Any]:
    """Process document using configured provider with fallback logic"""

    activity.heartbeat("Starting document processing")

    memory_graph = None
    try:
        # Get memory graph instance
        memory_graph = MemoryGraph()

        # Create processor factory
        config_manager = TenantConfigManager(memory_graph)
        factory = DocumentProcessorFactory(config_manager)

        # Progress callback for heartbeats
        async def progress_callback(upload_id, status, progress, current_page, total_pages):
            activity.heartbeat({
                "progress": progress,
                "status": status.value if hasattr(status, 'value') else str(status),
                "current_page": current_page,
                "total_pages": total_pages
            })

            # Also send real-time update
            await send_status_update({
                "upload_id": upload_id,
                "status": status.value if hasattr(status, 'value') else str(status),
                "progress": progress,
                "current_page": current_page,
                "total_pages": total_pages,
                "timestamp": datetime.now(),
                "organization_id": organization_id,
                "namespace_id": namespace_id
            })

        # Create processor for tenant
        # Coerce enum to string if needed
        pp = preferred_provider.value if hasattr(preferred_provider, "value") else preferred_provider
        processor = await factory.create_processor(
            organization_id or "default",
            namespace_id,
            pp
        )

        # Process document
        result = await processor.process_document(
            file_content,
            filename,
            upload_id,
            progress_callback
        )

        return {
            "pages": [page.model_dump() for page in result.pages],
            "stats": {
                "total_pages": result.total_pages,
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                "provider": processor.provider_name
            },
            "metadata": result.metadata,
            "provider_specific": result.provider_specific
        }

    except Exception as e:
        # Keep failure payloads small to avoid Temporal failure size limits
        msg = str(e)
        if len(msg) > 512:
            msg = msg[:512]
        logger.error("Document processing failed: %s", msg)
        from temporalio import activity as _activity
        raise _activity.ApplicationError(msg)
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in process_document_with_provider: {cleanup_error}")


@activity.defn
async def store_in_memory_batch(
    processing_result: Dict[str, Any],
    metadata: Dict[str, Any],
    upload_id: str,
    user_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str],
    workspace_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """Store processed content directly in memory using batch API"""
    # Allow calling outside Temporal by guarding heartbeat
    try:
        activity.heartbeat("Creating memories using batch API")
    except Exception:
        pass

    try:
        from fastapi import BackgroundTasks
        from unittest.mock import Mock
        from models.memory_models import OptimizedAuthResponse

        # Create memory graph instance
        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()

        # Extract document title from content
        document_title = extract_document_title(processing_result["pages"][0]["content"])

        # Convert pages to memory requests
        memory_requests = []
        for i, page in enumerate(processing_result["pages"]):
            page_title = f"{document_title} - Page {i + 1}" if len(processing_result["pages"]) > 1 else document_title

            # Create memory request for each page
            memory_request = AddMemoryRequest(
                content=page["content"],
                title=page_title,
                external_user_id=user_id,
                metadata={
                    "upload_id": upload_id,
                    "page_number": i + 1,
                    "total_pages": len(processing_result["pages"]),
                    "document_type": metadata.get("content_type", "document"),
                    "provider": processing_result["stats"]["provider"],
                    "confidence": processing_result["stats"]["confidence"],
                    "processing_time": processing_result["stats"]["processing_time"],
                    **metadata
                }
            )
            memory_requests.append(memory_request)

        # Create batch request
        batch_request = BatchMemoryRequest(memories=memory_requests)

        # Apply multi-tenant scoping
        batch_request = apply_multi_tenant_scoping_to_batch_request(
            batch_request, organization_id, namespace_id
        )

        # Create auth response
        auth_response = OptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            workspace_id=workspace_id or "default",
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_qwen_route=False
        )

        # Create mock request object
        mock_request = Mock()
        mock_request.headers = {
            "X-Client-Type": "temporal_document_processor",
            "X-API-Key": api_key or "temporal_internal",
            "Authorization": f"APIKey {api_key or 'temporal_internal'}"
        }

        # Call the memory batch handler directly
        from routes.memory_routes import common_add_memory_batch_handler
        result = await common_add_memory_batch_handler(
            request=mock_request,
            memory_graph=memory_graph,
            background_tasks=BackgroundTasks(),
            auth_response=auth_response,
            memory_request_batch=batch_request,
            skip_background_processing=True,  # Already in background via Temporal
            legacy_route=False
        )

        # Report progress to Temporal
        try:
            activity.heartbeat(f"Memory batch completed: {result.total_successful}/{result.total_processed} successful")
        except Exception:
            pass

        # Return full structured result so we can link memories later
        return {
            "memory_batch_result": result.status,
            "batch": result.model_dump() if hasattr(result, "model_dump") else result
        }

    except Exception as e:
        logger.error(f"Memory batch operation failed: {e}")
        raise
    finally:
        try:
            await memory_graph.cleanup()
        except Exception:
            pass


def extract_document_title(first_page_content: str) -> str:
    """Extract a meaningful title from document content"""
    lines = first_page_content.strip().split('\n')

    # Look for title-like content in first few lines
    for line in lines[:5]:
        line = line.strip()
        if len(line) > 10 and len(line) < 100:
            # Check if it looks like a title (not too long, has meaningful words)
            words = line.split()
            if len(words) >= 2 and len(words) <= 12:
                return line

    # Fallback to first non-empty line
    for line in lines:
        line = line.strip()
        if line and len(line) > 5:
            return line[:80] + ("..." if len(line) > 80 else "")

    return "Document Upload"


@activity.defn
async def send_status_update(update: Dict[str, Any]):
    """Send status update via WebSocket"""

    try:
        from core.document_processing.websocket_manager import WebSocketManager

        ws_manager = WebSocketManager()
        await ws_manager.broadcast_status_update(
            update,
            update.get("organization_id", "default"),
            update.get("namespace_id")
        )

    except Exception as e:
        logger.warning(f"Failed to send WebSocket status update: {e}")
        # Don't fail the activity if WebSocket update fails


@activity.defn(name="send_document_webhook_notification")
async def send_webhook_notification(webhook_data: Dict[str, Any]) -> Dict[str, Any]:
    """Send webhook notification for document processing completion"""

    webhook_url = webhook_data.get("webhook_url")
    webhook_secret = webhook_data.get("webhook_secret")

    if not webhook_url:
        return {"status": "skipped", "reason": "No webhook URL provided"}

    try:
        # Prepare webhook payload
        payload = {
            "upload_id": webhook_data["upload_id"],
            "status": webhook_data["status"],
            "timestamp": datetime.now().isoformat(),
            "results": webhook_data.get("results"),
            "error": webhook_data.get("error")
        }

        payload_json = json.dumps(payload)

        # Calculate HMAC signature if secret provided
        headers = {"Content-Type": "application/json"}
        if webhook_secret:
            signature = hmac.new(
                webhook_secret.encode(),
                payload_json.encode(),
                hashlib.sha256
            ).hexdigest()
            headers["X-PAPR-Signature"] = f"sha256={signature}"

        # Send webhook
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                webhook_url,
                content=payload_json,
                headers=headers
            )

            return {
                "status": "success",
                "response_code": response.status_code,
                "response_body": response.text[:1000] if response.text else None
            }

    except Exception as e:
        logger.error(f"Webhook notification failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@activity.defn
async def cleanup_failed_processing(
    upload_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str],
    error: str
) -> Dict[str, Any]:
    """Cleanup resources after failed processing"""

    try:
        logger.info(f"Cleaning up failed processing for upload_id: {upload_id}")

        # Here you could:
        # 1. Delete any partial memory records
        # 2. Update Parse Server records with failure status
        # 3. Clean up temporary files
        # 4. Send failure notifications

        # For now, just log the cleanup
        cleanup_actions = [
            f"Marked upload {upload_id} as failed",
            f"Cleaned up temporary resources",
            f"Recorded error: {error}"
        ]

        return {
            "status": "completed",
            "actions": cleanup_actions
        }

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }


@activity.defn
async def store_in_parse_only(
    processing_result: Dict[str, Any],
    metadata: Dict[str, Any],
    upload_id: str,
    user_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str],
    workspace_id: Optional[str]
) -> Dict[str, Any]:
    """
    Store processed document content in Parse Server only.
    Creates/updates Post for document context and links memories to that Post.
    """
    # Allow calling outside Temporal by guarding heartbeat
    try:
        activity.heartbeat("Creating Parse Post and linking memories")
    except Exception:
        pass

    try:
        from models.parse_server import MemoryParseServer, ParsePointer, AddMemoryItem
        from core.document_processing.parse_integration import ParseDocumentIntegration

        # Create memory graph instance (for consistency/tenant info)
        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()

        # Extract document title from content
        document_title = extract_document_title(processing_result["pages"][0]["content"])

        # Create Parse integration instance
        parse_integration = ParseDocumentIntegration(memory_graph)

        # Create/update Parse Post for document context
        post_data = {
            "title": document_title,
            "filename": metadata.get("filename", ""),
            "total_pages": processing_result["stats"]["total_pages"],
            "upload_id": upload_id,
            "content_type": metadata.get("content_type", "document"),
            "provider": processing_result["stats"]["provider"],
            "user_id": user_id,
            "organization_id": organization_id,
            "namespace_id": namespace_id,
            "workspace_id": workspace_id
        }

        # Create or update Post (Parse will auto-create PageVersion on updates)
        post_result = await parse_integration.create_or_update_document_post(
            upload_id=upload_id,
            post_data=post_data,
            organization_id=organization_id,
            namespace_id=namespace_id
        )

        post_id = post_result.get("post_id")
        post_social_id = post_result.get("post_social_id")
        page_version_id = post_result.get("page_version_id")

        # Convert pages to memory items and create them in Parse
        memory_items = []
        for i, page in enumerate(processing_result["pages"]):
            page_title = f"{document_title} - Page {i + 1}" if len(processing_result["pages"]) > 1 else document_title

            # Create memory in Parse Server
            memory_data = {
                "content": page["content"],
                "title": page_title,
                "user_id": user_id,
                "organization_id": organization_id,
                "namespace_id": namespace_id,
                "workspace_id": workspace_id,
                "metadata": {
                    "upload_id": upload_id,
                    "page_number": i + 1,
                    "total_pages": len(processing_result["pages"]),
                    "document_type": metadata.get("content_type", "document"),
                    "provider": processing_result["stats"]["provider"],
                    "confidence": processing_result["stats"]["confidence"],
                    "processing_time": processing_result["stats"]["processing_time"],
                    **metadata
                },
                # Link to Post
                "post": ParsePointer(objectId=post_id, className="Post") if post_id else None,
                "page_number": i + 1,
                "total_pages": len(processing_result["pages"]),
                "upload_id": upload_id,
                "filename": metadata.get("filename", ""),
                "page": f"{i + 1} of {len(processing_result['pages'])}"
            }

            memory_result = await parse_integration.create_memory_record(memory_data)

            if memory_result.get("success"):
                memory_item = AddMemoryItem(
                    objectId=memory_result.get("objectId"),
                    content=page["content"],
                    title=page_title,
                    createdAt=memory_result.get("createdAt"),
                    metadata=memory_data["metadata"]
                )
                memory_items.append(memory_item)

        # Report progress
        try:
            activity.heartbeat(f"Created Post and {len(memory_items)} memory records")
        except Exception:
            pass

        return {
            "memory_items": [item.model_dump() if hasattr(item, 'model_dump') else dict(item) for item in memory_items],
            "parse_records": {
                "post": post_id,
                "postSocial": post_social_id,
                "pageVersion": page_version_id
            }
        }

    except Exception as e:
        logger.error(f"Parse storage operation failed: {e}")
        raise
    finally:
        try:
            await memory_graph.cleanup()
        except Exception:
            pass


@activity.defn
async def store_in_memory_and_parse(
    processing_result: Dict[str, Any],
    metadata: Dict[str, Any],
    upload_id: str,
    user_id: str,
    organization_id: Optional[str],
    namespace_id: Optional[str],
    workspace_id: Optional[str] = None,
    api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Store processed document content using batch memory API and Parse Post.
    First adds memories via batch API, then creates Parse Post and links memories.
    """
    # Allow calling outside Temporal by guarding heartbeat
    try:
        activity.heartbeat("Starting memory batch and Parse operations")
    except Exception:
        pass

    try:
        # Step 1: Add memories via batch API
        batch_result = await store_in_memory_batch(
            processing_result=processing_result,
            metadata=metadata,
            upload_id=upload_id,
            user_id=user_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            workspace_id=workspace_id,
            api_key=api_key
        )

        # Step 2: Create Parse Post and link memories
        parse_result = await store_in_parse_only(
            processing_result=processing_result,
            metadata=metadata,
            upload_id=upload_id,
            user_id=user_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            workspace_id=workspace_id
        )

        # Merge results
        memory_items = []

        # Prefer memory items from batch result if available
        if batch_result.get("memory_batch_result") in {"success", "partial"}:
            # Extract parse memory items from batch dump
            from models.parse_server import AddMemoryItem
            batch_dump = batch_result.get("batch", {})
            successful = batch_dump.get("successful", []) or []
            for resp in successful:
                for item in (resp.get("data") or []):
                    try:
                        memory_items.append(AddMemoryItem(**item))
                    except Exception:
                        pass
        else:
            # Fallback to parse result memory items
            memory_items = parse_result.get("memory_items", [])

        # Report final progress
        try:
            activity.heartbeat(f"Completed: {len(memory_items)} memories and Parse Post")
        except Exception:
            pass

        # Link memories back to Post (if we have objectIds and a Post)
        try:
            post_id = (parse_result.get("parse_records") or {}).get("post") or (parse_result.get("parse_records") or {}).get("post_id")
            if post_id and memory_items:
                from core.document_processing.parse_integration import ParseDocumentIntegration
                from memory.memory_graph import MemoryGraph
                mg = MemoryGraph()
                linker = ParseDocumentIntegration(mg)
                memory_object_ids = [getattr(mi, 'objectId', None) or (mi.get('objectId') if isinstance(mi, dict) else None) for mi in memory_items]
                memory_object_ids = [x for x in memory_object_ids if x]
                if memory_object_ids:
                    await linker.link_memories_to_post(post_id, memory_object_ids)
                try:
                    await mg.cleanup()
                except Exception:
                    pass
        except Exception:
            pass

        return {
            "memory_items": [item.model_dump() if hasattr(item, 'model_dump') else dict(item) for item in memory_items],
            "parse_records": parse_result.get("parse_records", {}),
            "batch_result": batch_result
        }

    except Exception as e:
        logger.error(f"Memory and Parse storage operation failed: {e}")
        raise


@activity.defn
async def download_and_validate_file(file_reference: Dict[str, Any], organization_id: str, namespace_id: str) -> Dict[str, Any]:
    """Download file from Parse Server storage and validate it"""

    try:
        file_url = file_reference.get("file_url")
        file_name = file_reference.get("file_name")
        file_size = file_reference.get("file_size", 0)

        logger.info(f"Downloading and validating file: {file_name} ({file_size:,} bytes) from {file_url}")

        # Download file bytes
        file_bytes: bytes = b""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(file_url)
            if resp.status_code != 200:
                return {"valid": False, "error": f"Failed to download file: {resp.status_code}"}
            file_bytes = resp.content or b""

        # Validate with FileValidator to ensure parity with direct upload path
        try:
            is_valid, validation_error = await FileValidator.validate_file(file_bytes, file_name)
            if not is_valid:
                return {"valid": False, "error": validation_error}
        except Exception as ve:
            return {"valid": False, "error": f"Security validation error: {ve}"}

        return {
            "valid": True,
            "file_info": {
                "name": file_name,
                "size": len(file_bytes) if file_bytes else file_size,
                "url": file_url
            }
        }

    except Exception as e:
        logger.error(f"Failed to download/validate file: {e}")
        return {
            "valid": False,
            "error": f"File validation failed: {str(e)}"
        }


@activity.defn
async def process_document_with_provider_from_reference(
    file_reference: Dict[str, Any],
    upload_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: Optional[str],
    preferred_provider: Optional[PreferredProvider] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """Process document using file reference instead of file content.

    Returns a minimized payload suitable for Temporal workflow history, and also
    persists the full provider output into a Parse `Post` as JSON content, returning
    the `post_id` so downstream activities can fetch details on demand.
    """

    memory_graph = None
    try:
        # Import processing components
        from core.document_processing.provider_manager import TenantConfigManager, DocumentProcessorFactory
        from memory.memory_graph import MemoryGraph

        logger.info(f"Processing document from reference for upload_id: {upload_id}")

        # Get memory graph (Temporal activities do not have a FastAPI request)
        memory_graph = MemoryGraph()

        # Create processor factory
        config_manager = TenantConfigManager(memory_graph)
        factory = DocumentProcessorFactory(config_manager)

        # Create processor
        pp = preferred_provider.value if hasattr(preferred_provider, "value") else preferred_provider
        processor = await factory.create_processor(
            organization_id or "default",
            namespace_id,
            pp
        )

        file_url = file_reference.get("file_url")
        file_name = file_reference.get("file_name")

        import httpx
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.get(file_url)
            file_content = response.content

        # Process with provider
        async def progress_callback(upload_id, status, progress, current_page, total_pages):
            logger.info(f"Processing progress: {progress:.1%} (page {current_page}/{total_pages})")

        result = await processor.process_document(
            file_content,
            file_name,
            upload_id,
            progress_callback
        )

        logger.info(f"Document processing completed for upload_id: {upload_id}")

        # Persist provider JSON into Parse Post and return only identifiers
        from core.document_processing.parse_integration import ParseDocumentIntegration
        parse = ParseDocumentIntegration(memory_graph)

        # Extract actual page count and verify content from provider response
        actual_total_pages = result.total_pages
        provider_name_lower = processor.provider_name.lower()
        logger.info(f"🔍 Provider: {provider_name_lower}, Initial total_pages: {actual_total_pages}")
        
        # Convert provider_specific to dict for processing
        provider_dict = result.provider_specific
        if hasattr(provider_dict, 'model_dump'):
            provider_dict = provider_dict.model_dump()
        elif hasattr(provider_dict, '__dict__'):
            provider_dict = provider_dict.__dict__
        
        # Deep-convert nested models to plain dicts
        try:
            from core.document_processing.parse_integration import ParseDocumentIntegration as _PDI
            provider_dict = _PDI._json_safe(provider_dict)
        except Exception as conv_err:
            logger.info(f"Deep conversion to dict failed (continuing): {conv_err}")
        
        logger.info(f"🔍 provider_specific keys: {list(provider_dict.keys()) if isinstance(provider_dict, dict) else 'not a dict'}")
        
        # Helper function for nested dict access
        def _get_nested(obj, keys):
            cur = obj
            for k in keys:
                if isinstance(cur, dict):
                    cur = cur.get(k)
                else:
                    return None
            return cur
        
        # Provider-specific page count extraction and content verification
        if provider_name_lower == "reducto":
            try:
                # Reducto stores page count at result.parse.result.usage.num_pages
                usage = _get_nested(provider_dict, ["result", "parse", "result", "usage"]) or provider_dict.get("usage", {})
                logger.info(f"🔍 Reducto usage: {usage}")
                
                if usage and "num_pages" in usage and usage.get("num_pages"):
                    actual_total_pages = usage.get("num_pages")
                    logger.info(f"✅ Extracted actual page count from Reducto usage: {actual_total_pages}")
                else:
                    # Fallback: count unique pages from chunks
                    chunks = _get_nested(provider_dict, ["result", "parse", "result", "chunks"]) or []
                    logger.info(f"🔍 Found {len(chunks)} chunks for page counting")
                    
                    pages = set()
                    for chunk in chunks:
                        for block in chunk.get("blocks", []):
                            bbox = block.get("bbox", {})
                            if "page" in bbox:
                                pages.add(bbox["page"])
                    
                    if pages:
                        actual_total_pages = max(pages)
                        logger.info(f"✅ Extracted actual page count from Reducto chunks: {actual_total_pages}")
                    else:
                        logger.warning(f"⚠️  Could not extract page count from Reducto, using default: {actual_total_pages}")
            except Exception as e:
                logger.error(f"❌ Failed to extract page count from Reducto: {e}", exc_info=True)
        
        elif provider_name_lower == "tensorlake":
            try:
                # TensorLake returns parsed_pages_count and total_pages
                parsed_pages_count = provider_dict.get("parsed_pages_count", 0)
                total_pages = result.total_pages or parsed_pages_count
                
                if parsed_pages_count > 0:
                    actual_total_pages = parsed_pages_count
                    logger.info(f"✅ TensorLake parsed_pages_count: {actual_total_pages}")
                else:
                    logger.info(f"⚠️  TensorLake parsed_pages_count is 0, using result.total_pages: {total_pages}")
                    actual_total_pages = total_pages
                
                # CRITICAL: Verify content is present
                content = provider_dict.get("content")
                if content:
                    content_length = len(content) if isinstance(content, str) else 0
                    logger.info(f"✅ TensorLake content verified: {content_length} chars")
                else:
                    logger.error(f"❌ CRITICAL: TensorLake provider_specific has NO 'content' field!")
                    logger.error(f"   Available keys: {list(provider_dict.keys())}")
                    logger.error(f"   parse_id: {provider_dict.get('parse_id')}")
                    logger.error(f"   file_id: {provider_dict.get('file_id')}")
                    logger.error(f"   This means content extraction in TensorLake provider FAILED!")
                    
            except Exception as e:
                logger.error(f"❌ Failed to process TensorLake response: {e}", exc_info=True)
        
        elif provider_name_lower == "gemini":
            # Gemini processes entire PDF as one document, count from pages list
            if result.pages:
                # Gemini typically returns 1 page with all content
                # But if we extracted pages from PyMuPDF, count those
                actual_total_pages = len(result.pages)
                logger.info(f"✅ Gemini page count from result.pages: {actual_total_pages}")
                
                # Verify content
                if result.pages[0].content:
                    logger.info(f"✅ Gemini content verified: {len(result.pages[0].content)} chars")
                else:
                    logger.warning(f"⚠️  Gemini first page has no content")
        
        elif provider_name_lower in ["paddleocr", "deepseek-ocr", "deepseekocr"]:
            # These process page-by-page, count from pages list
            if result.pages:
                actual_total_pages = len(result.pages)
                logger.info(f"✅ {provider_name_lower} page count from result.pages: {actual_total_pages}")
        
        else:
            logger.info(f"ℹ️  Using default page count for provider '{provider_name_lower}': {actual_total_pages}")

        # Compose minimal processing metadata
        stats = {
            "total_pages": actual_total_pages,
            "processing_time": result.processing_time,
            "confidence": result.confidence,
            "provider": processor.provider_name,
        }

        # Attach file pointer info if available in file_reference
        base_meta = {
            "file_url": file_reference.get("file_url"),
            "file_name": file_reference.get("file_name"),
            "content_type": file_reference.get("content_type"),
        }

        # Ensure JSON-safe payloads (guards against non-serializable provider SDK objects)
        from core.document_processing.parse_integration import ParseDocumentIntegration as _PDI
        safe_provider_specific = _PDI._json_safe(result.provider_specific or {})
        safe_stats = _PDI._json_safe(stats)
        safe_meta = _PDI._json_safe(base_meta)
        
        # DEBUG: Log provider_specific keys to verify content is present
        logger.info(f"🔍 provider_specific keys after JSON-safe: {list(safe_provider_specific.keys())}")
        if "content" in safe_provider_specific:
            content_len = len(safe_provider_specific["content"]) if isinstance(safe_provider_specific["content"], str) else 0
            logger.info(f"✅ provider_specific HAS 'content' field: {content_len} chars")
        else:
            logger.warning(f"⚠️  provider_specific MISSING 'content' field!")
            logger.warning(f"   Available fields: {safe_provider_specific}")
        
        # CRITICAL CHECK: For TensorLake, verify content is present before proceeding
        if provider_name_lower == "tensorlake":
            content = safe_provider_specific.get("content")
            if not content or (isinstance(content, str) and len(content) < 50):
                logger.error(f"❌ CRITICAL: TensorLake provider_specific has insufficient content!")
                logger.error(f"   Content length: {len(content) if content else 0}")
                logger.error(f"   provider_specific keys: {list(safe_provider_specific.keys())}")
                logger.error(f"   parse_id: {safe_provider_specific.get('parse_id')}")
                logger.error(f"   file_id: {safe_provider_specific.get('file_id')}")
                raise Exception(f"TensorLake returned no content. This should have been caught in the provider. parse_id={safe_provider_specific.get('parse_id')}")

        provider_post = await parse.create_post_with_provider_json(
            upload_id=upload_id,
            provider_name=processor.provider_name,
            provider_specific=safe_provider_specific,
            user_id=user_id,  # Set ACL to user
            organization_id=organization_id,
            namespace_id=namespace_id,
            workspace_id=workspace_id,
            metadata=safe_meta,
            processing_metadata=safe_stats,
            document_title=file_reference.get("file_name")
        )

        # CRITICAL: Validate Post was created successfully
        # Expect a PostParseServer model instance
        post_id = getattr(provider_post, "objectId", None)
        if not post_id:
            error_msg = f"Parse provider-json Post creation did not return objectId for upload_id {upload_id}"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        # CRITICAL: Verify Post actually exists in Parse Server (not just returned an ID)
        # This prevents workflow replay issues where an old Post ID is returned but Post doesn't exist
        try:
            post_verification = await parse.get_post(post_id)
            if not post_verification:
                error_msg = f"Post {post_id} was created but verification failed - Post does not exist in Parse Server! upload_id={upload_id}"
                logger.error(error_msg)
                raise Exception(error_msg)
            logger.info(f"✅ Post {post_id} created and verified successfully for upload_id {upload_id}")
        except Exception as verify_err:
            error_msg = f"Failed to verify Post {post_id} exists after creation: {verify_err}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Return minimal payload to workflow
        first_page_preview = (result.pages[0].content[:1000] if result.pages and result.pages[0].content else "")
        return {
            "post": {"objectId": post_id},
            "preview": first_page_preview,
            "stats": {
                "total_pages": actual_total_pages,  # Use corrected page count
                "processing_time": result.processing_time,
                "confidence": result.confidence,
                "provider": processor.provider_name,
            }
        }

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in process_document_with_provider_from_reference: {cleanup_error}")


@activity.defn
async def create_memory_batch_for_pages(
    page_batch: List[Dict[str, Any]],
    upload_id: str,
    user_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: str,
    batch_start_index: int = 0
) -> Dict[str, Any]:
    """Create memory items for a batch of pages, treating each page as separate memory"""

    try:
        # Build a BatchMemoryRequest and reuse the same Temporal-backed path as v1 /memory/batch
        from models.memory_models import AddMemoryRequest, BatchMemoryRequest, OptimizedAuthResponse
        from models.shared_types import MemoryType, MemoryMetadata
        from services.batch_processor import process_batch_with_temporal
        from services.logger_singleton import LoggerSingleton

        logger = LoggerSingleton.get_logger(__name__)
        logger.info(f"Creating memory items for {len(page_batch)} pages starting at index {batch_start_index}")

        memories: list[AddMemoryRequest] = []

        def chunk_text_by_bytes(text: str, max_bytes: int = 14900) -> List[str]:
            """Chunk text so each chunk's UTF-8 byte size <= max_bytes."""
            if not text:
                return []
            b = text.encode("utf-8")
            if len(b) <= max_bytes:
                return [text]
            chunks: List[str] = []
            start = 0
            n = len(b)
            while start < n:
                end = min(start + max_bytes, n)
                # ensure we don't split in the middle of a multibyte char
                while end > start and (b[end - 1] & 0xC0) == 0x80:
                    end -= 1
                chunk_bytes = b[start:end]
                chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))
                start = end
            return chunks
        
        for i, page_data in enumerate(page_batch):
            page_index = batch_start_index + i + 1
            page_content = page_data.get("content", "")
            if not page_content.strip():
                logger.warning(f"Skipping empty page {page_index}")
                continue

            chunks = chunk_text_by_bytes(page_content, 14900)
            total_chunks = len(chunks)
            for chunk_idx, chunk_text in enumerate(chunks):
                # Create metadata with ACL fields - similar to handle_incoming_memory
                # Extract any existing metadata from page_data (user might have provided ACLs)
                existing_metadata = page_data.get("metadata", {})
                
                # Helper function to merge ACL lists (from handle_incoming_memory pattern)
                def merge_acl_lists(existing, new):
                    return list(set((existing or []) + (new or [])))
                
                # Get existing ACL fields from user-provided metadata (if any)
                existing_user_read = existing_metadata.get("user_read_access", [])
                existing_user_write = existing_metadata.get("user_write_access", [])
                existing_workspace_read = existing_metadata.get("workspace_read_access", [])
                existing_workspace_write = existing_metadata.get("workspace_write_access", [])
                existing_role_read = existing_metadata.get("role_read_access", [])
                existing_role_write = existing_metadata.get("role_write_access", [])
                existing_namespace_read = existing_metadata.get("namespace_read_access", [])
                existing_namespace_write = existing_metadata.get("namespace_write_access", [])
                existing_organization_read = existing_metadata.get("organization_read_access", [])
                existing_organization_write = existing_metadata.get("organization_write_access", [])
                
                # Merge resolved user_id with any existing ACLs
                # This ensures the user who uploaded always has access, plus any ACLs they specified
                merged_user_read = merge_acl_lists(existing_user_read, [user_id])
                merged_user_write = merge_acl_lists(existing_user_write, [user_id])
                merged_workspace_read = merge_acl_lists(existing_workspace_read, [])
                merged_workspace_write = merge_acl_lists(existing_workspace_write, [])
                merged_role_read = merge_acl_lists(existing_role_read, [])
                merged_role_write = merge_acl_lists(existing_role_write, [])
                merged_namespace_read = merge_acl_lists(existing_namespace_read, [])
                merged_namespace_write = merge_acl_lists(existing_namespace_write, [])
                merged_organization_read = merge_acl_lists(existing_organization_read, [])
                merged_organization_write = merge_acl_lists(existing_organization_write, [])
                
                # Create metadata with merged ACL fields
                memory_metadata = MemoryMetadata(
                    user_id=user_id,
                    workspace_id=workspace_id,
                    # Merged ACL fields - preserves user-provided ACLs + adds resolved user
                    user_read_access=merged_user_read,
                    user_write_access=merged_user_write,
                    workspace_read_access=merged_workspace_read,
                    workspace_write_access=merged_workspace_write,
                    role_read_access=merged_role_read,
                    role_write_access=merged_role_write,
                    namespace_read_access=merged_namespace_read,
                    namespace_write_access=merged_namespace_write,
                    organization_read_access=merged_organization_read,
                    organization_write_access=merged_organization_write,
                    # Custom metadata
                    customMetadata={
                        "upload_id": upload_id,
                        "page_number": page_index,
                        "source": "document_processing",
                        "organization_id": organization_id,
                        "namespace_id": namespace_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": total_chunks,
                    }
                )
                
                memories.append(
                    AddMemoryRequest(
                        content=chunk_text,
                        type=MemoryType.DOCUMENT,
                        metadata=memory_metadata,
                        external_user_id=user_id,
                        organization_id=organization_id,
                        namespace_id=namespace_id
                    )
                )

        batch_request = BatchMemoryRequest(
            external_user_id=user_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            memories=memories,
            batch_size=min(50, len(memories) or 1)
        )

        # Minimal auth envelope for processor (developer=end user in this path)
        auth_response = OptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_qwen_route=False
        )

        # Start Temporal batch workflow used by v1 route
        batch_result = await process_batch_with_temporal(
            batch_request=batch_request,
            auth_response=auth_response,
            api_key="temporal_internal",
            webhook_url=None,
            webhook_secret=None
        )

        return {
            "memory_batch": batch_result.model_dump() if hasattr(batch_result, "model_dump") else batch_result
        }

    except Exception as e:
        logger.error(f"Failed to create memory batch: {e}")
        raise


@activity.defn
async def store_document_in_parse(
    processing_result: Dict[str, Any],
    metadata: Dict[str, Any],
    upload_id: str,
    user_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: str
) -> Dict[str, Any]:
    """Store the document record in Parse Server"""

    memory_graph = None
    try:
        from core.document_processing.parse_integration import ParseDocumentIntegration
        from memory.memory_graph import MemoryGraph

        logger.info(f"Storing document in Parse Server for upload_id: {upload_id}")

        memory_graph = MemoryGraph()
        parse_integration = ParseDocumentIntegration(memory_graph)

        # Prepare post data
        post_data = {
            "pages": processing_result.get("pages", []),
            "metadata": metadata,
            "processing_metadata": processing_result.get("stats", {}),
            "user_id": user_id,
            "workspace_id": workspace_id
        }

        # Create document post
        result = await parse_integration.create_or_update_document_post(
            upload_id=upload_id,
            post_data=post_data,
            organization_id=organization_id,
            namespace_id=namespace_id
        )

        logger.info(f"Successfully stored document in Parse Server: {result}")

        return {
            "parse_records": result
        }

    except Exception as e:
        logger.error(f"Failed to store document in Parse Server: {e}")
        raise
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in store_document_in_parse: {cleanup_error}")


# New Hierarchical Chunking Activities

@activity.defn
async def process_document_with_hierarchical_chunking(
    processing_result: Dict[str, Any],
    base_metadata: Dict[str, Any],
    organization_id: str,
    namespace_id: str,
    user_id: str
) -> Dict[str, Any]:
    """Process document result using hierarchical chunking and multi-modal extraction"""

    if not HIERARCHICAL_AVAILABLE:
        raise Exception("Hierarchical processing not available")

    activity.heartbeat("Starting hierarchical content extraction")

    try:
        logger.info(f"Processing document with hierarchical chunking for user: {user_id}")

        # Create base metadata object
        metadata = MemoryMetadata(**base_metadata) if base_metadata else MemoryMetadata()

        # Extract structured content from provider results
        memory_requests = DocumentToMemoryTransformer.process_document_result(
            processing_result=processing_result,
            base_metadata=metadata,
            organization_id=organization_id,
            namespace_id=namespace_id
        )

        activity.heartbeat(f"Generated {len(memory_requests)} memory requests")

        # Log what we found
        content_types = {}
        for req in memory_requests:
            content_type = req.metadata.customMetadata.get('content_type', 'text') if req.metadata and req.metadata.customMetadata else 'text'
            content_types[content_type] = content_types.get(content_type, 0) + 1

        logger.info(f"Hierarchical extraction found: {content_types}")

        return {
            "memory_requests": [req.model_dump() for req in memory_requests],
            "content_summary": content_types,
            "total_memories": len(memory_requests)
        }

    except Exception as e:
        logger.error(f"Hierarchical chunking failed: {e}")
        raise


@activity.defn
async def extract_structured_content_from_provider(
    provider_specific: Dict[str, Any],
    provider_name: str,
    base_metadata: MemoryMetadata,
    organization_id: str,
    namespace_id: str
) -> Dict[str, Any]:
    """Extract structured content (tables, images, charts) from provider-specific results"""

    if not HIERARCHICAL_AVAILABLE:
        raise Exception("Hierarchical processing not available")

    memory_graph = None
    try:
        activity.heartbeat("Extracting structured content from provider results")
    except Exception:
        pass

    try:
        logger.info(f"Extracting structured content from {provider_name} provider")

        # If we were passed a reference to a Post, fetch provider JSON from Parse
        post_id = None
        if isinstance(provider_specific, dict):
            # New shape: {"post": {"objectId": "..."}} or {"objectId": "..."}
            if provider_specific.get("post") and isinstance(provider_specific.get("post"), dict):
                post_id = provider_specific.get("post", {}).get("objectId")
            if not post_id:
                post_id = provider_specific.get("objectId")
            # Backward compatibility: {"post_id": "..."}
            if not post_id:
                post_id = provider_specific.get("post_id")
        
        if post_id:
            logger.info(f"Detected Post reference: {post_id}, fetching provider result from Parse")
            
            # Use the reusable memory_management method to fetch Post with all data
            from services.memory_management import fetch_post_with_provider_result_async
            
            post_data = await fetch_post_with_provider_result_async(post_id)
            
            if not post_data:
                raise Exception(f"Failed to fetch Post {post_id} from Parse")
            
            # Extract all the useful information from the typed response
            provider_specific = post_data.provider_specific
            provider_name = post_data.provider_name or provider_name
            
            # Update organization_id and namespace_id if they exist in the Post
            # (This ensures consistency with the stored document)
            if post_data.organization_id:
                organization_id = post_data.organization_id
            if post_data.namespace_id:
                namespace_id = post_data.namespace_id
            
            # Log the metadata we extracted
            logger.info(f"Post {post_id} metadata: provider={provider_name}, org={organization_id}, ns={namespace_id}, upload_id={post_data.upload_id}")
            logger.info(f"Provider result size: {len(str(provider_specific))} chars")
            
            if not provider_specific:
                raise Exception(f"Post {post_id} has no provider_result_file or provider_result")

        # Dereference TensorLake parse_id if present (fetch actual parsed content)
        if provider_name.lower() == "tensorlake" and isinstance(provider_specific, dict):
            parse_id = provider_specific.get("parse_id")
            file_id = provider_specific.get("file_id")
            
            if parse_id and not provider_specific.get("content"):
                logger.info(f"Detected TensorLake parse_id reference: {parse_id}, fetching actual content via HTTP API")
                try:
                    # Create TensorLake provider instance to fetch result
                    from core.document_processing.providers.tensorlake import TensorLakeProvider
                    from os import environ as env
                    
                    tensorlake_config = {
                        "api_key": env.get("TENSORLAKE_API_KEY"),
                        "base_url": env.get("TENSORLAKE_BASE_URL", "https://api.tensorlake.ai"),
                        "timeout": 300
                    }
                    
                    tensorlake_provider = TensorLakeProvider(tensorlake_config)
                    parse_result = await tensorlake_provider.fetch_parse_result(parse_id)
                    
                    # Extract actual content from parse result
                    content = parse_result.get("content") or parse_result.get("text", "")
                    
                    if content:
                        logger.info(f"Successfully fetched TensorLake content: {len(content)} chars")
                        # Replace provider_specific with full result
                        provider_specific = {
                            "file_id": file_id,
                            "parse_id": parse_id,
                            "content": content,  # ← Actual parsed text!
                            "status": parse_result.get("status"),
                            "full_result": parse_result  # Keep full result for potential structured data
                        }
                    else:
                        logger.warning(f"TensorLake parse result has no content field")
                        
                except Exception as fetch_err:
                    logger.error(f"Failed to fetch TensorLake content: {fetch_err}")
                    # Continue with reference-only data (will fail downstream but be traceable)
        
        # Attempt typed parse with provider SDK if available (best-effort)
        try:
            from core.document_processing.provider_type_parser import parse_with_provider_sdk
            provider_specific = parse_with_provider_sdk(provider_name, provider_specific)
        except Exception as e:
            logger.info(f"Provider typed parsing failed, using raw response: {e}")
            # Never fail extraction due to typed parsing attempt

        # Ensure provider_specific is a dict after parsing/typed-parse
        if isinstance(provider_specific, str):
            try:
                provider_specific = json.loads(provider_specific)
            except Exception:
                provider_specific = {}

        # Branch: simple vs complex based on structure (tables/images/charts presence)
        def _detect_complex(ps: Dict[str, Any]) -> bool:
            try:
                # Direct flags in provider JSON
                s = json.dumps(ps)[:200000].lower()  # cap to avoid huge dumps
                if any(k in s for k in ("\"tables\"", "\"images\"", "\"charts\"", "table", "image", "chart")):
                    return True
                # Reducto blocks type
                result = (ps or {}).get("result") or {}
                chunks = result.get("chunks") or []
                for ch in chunks:
                    for blk in (ch.get("blocks") or []):
                        btype = str(blk.get("type") or "").lower()
                        if btype in ("table", "image", "figure", "chart"):
                            return True
                return False
            except Exception:
                return False

        # Compute analysis for logging/observability
        try:
            total_pages = 0
            # Get processing metadata from post_data if available (from Post fetch)
            pm = (post_data.processing_metadata if 'post_data' in locals() else {}) or {}
            total_pages = pm.get("total_pages") or 0
            s = json.dumps(provider_specific)[:200000].lower()
            has_tables = ("table" in s) or ("\"tables\"" in s)
            has_images = ("image" in s) or ("\"images\"" in s) or ("figure" in s)
            has_charts = ("chart" in s) or ("\"charts\"" in s)
            analysis = {
                "total_pages": total_pages,
                "has_tables": bool(has_tables),
                "has_images": bool(has_images),
                "has_charts": bool(has_charts),
            }
        except Exception:
            analysis = {"total_pages": None, "has_tables": False, "has_images": False, "has_charts": False}

        is_complex = _detect_complex(provider_specific)
        logger.info(f"Structure analysis decision: {'complex' if is_complex else 'simple'} | analysis={analysis}")

        # Simple path: render full provider JSON into Markdown and build direct memories
        if not is_complex:
            try:
                from core.document_processing.provider_adapter import provider_to_markdown
                from models.shared_types import MemoryType

                markdown = provider_to_markdown(provider_name, provider_specific)

                def _chunk_text_by_bytes(text: str, max_bytes: int = 14900) -> List[str]:
                    if not text:
                        return []
                    b = text.encode("utf-8")
                    if len(b) <= max_bytes:
                        return [text]
                    chunks: List[str] = []
                    start = 0
                    n = len(b)
                    while start < n:
                        end = min(start + max_bytes, n)
                        while end > start and (b[end - 1] & 0xC0) == 0x80:
                            end -= 1
                        chunk_bytes = b[start:end]
                        chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))
                        start = end
                    return chunks

                chunks = _chunk_text_by_bytes(markdown, 14900)
                memory_requests = []
                for idx, ch in enumerate(chunks):
                    memory_requests.append(
                        AddMemoryRequest(
                            content=ch,
                            type=MemoryType.DOCUMENT,
                            metadata={
                                "organization_id": organization_id,
                                "namespace_id": namespace_id,
                                "customMetadata": {
                                    "content_type": "markdown",
                                    "chunk_index": idx,
                                    "total_chunks": len(chunks),
                                    "source": "provider_markdown",
                                }
                            }
                        )
                    )

                return {
                    "decision": "simple",
                    "structured_elements": [],
                    "memory_requests": [mr.model_dump() for mr in memory_requests],
                    "element_summary": {},
                    "structure_analysis": analysis,
                    "provider": provider_name
                }
            except Exception as e:
                logger.warning(f"Simple-path markdown generation failed, falling back to complex path: {e}")

        # Extract structured elements based on provider (complex path)
        structured_elements = []

        try:
            # Prefer adapter that maps full provider JSON to ContentElements
            from core.document_processing.provider_adapter import extract_structured_elements as _extract_elems
            base_meta_dict = base_metadata.model_dump() if isinstance(base_metadata, MemoryMetadata) else (base_metadata or {})
            structured_elements = _extract_elems(
                provider_name,
                provider_specific,
                base_meta_dict,
                organization_id,
                namespace_id,
            )
            logger.info(f"Provider adapter extracted {len(structured_elements)} elements")
        except Exception as e:
            logger.warning(f"Provider adapter content-element extraction failed: {e}", exc_info=True)

        if not structured_elements:
            # Provider-specific fallback paths
            if provider_name.lower() == 'reducto':
                try:
                    structured_elements = ProviderContentExtractor.extract_from_reducto(provider_specific)
                    logger.info(f"ProviderContentExtractor.extract_from_reducto returned {len(structured_elements)} elements")
                except Exception as e:
                    logger.warning(f"ProviderContentExtractor reducto fallback failed: {e}", exc_info=True)
            elif provider_name.lower() == 'tensorlake':
                try:
                    structured_elements = ProviderContentExtractor.extract_from_tensorlake(provider_specific)
                    logger.info(f"ProviderContentExtractor.extract_from_tensorlake returned {len(structured_elements)} elements")
                except Exception as e:
                    logger.warning(f"ProviderContentExtractor tensorlake fallback failed: {e}", exc_info=True)

        try:
            activity.heartbeat(f"Found {len(structured_elements)} structured elements")
        except Exception:
            pass

        # Note: Chunking is now handled by separate chunk_document_elements activity in workflow
        # This allows chunking to be optional and visible in Temporal UI

        # Convert elements to memory requests
        metadata = base_metadata if isinstance(base_metadata, MemoryMetadata) else MemoryMetadata(**(base_metadata or {}))
        memory_requests = []

        for element in structured_elements:
            memory_request = MemoryTransformer.content_element_to_memory_request(
                element=element,
                base_metadata=metadata,
                organization_id=organization_id,
                namespace_id=namespace_id
            )
            memory_requests.append(memory_request)

        # Analyze what we extracted
        element_types = {}
        for element in structured_elements:
            element_type = element.content_type.value
            element_types[element_type] = element_types.get(element_type, 0) + 1

        logger.info(f"Structured extraction from {provider_name}: {element_types}")

        # Final safety net: if we still have nothing, fall back to simple markdown path
        if not structured_elements and not memory_requests:
            try:
                from core.document_processing.provider_adapter import provider_to_markdown
                from models.shared_types import MemoryType
                markdown = provider_to_markdown(provider_name, provider_specific)
                def _chunk_text_by_bytes(text: str, max_bytes: int = 14900) -> List[str]:
                    if not text:
                        return []
                    b = text.encode("utf-8")
                    if len(b) <= max_bytes:
                        return [text]
                    chunks: List[str] = []
                    start = 0
                    n = len(b)
                    while start < n:
                        end = min(start + max_bytes, n)
                        while end > start and (b[end - 1] & 0xC0) == 0x80:
                            end -= 1
                        chunk_bytes = b[start:end]
                        chunks.append(chunk_bytes.decode("utf-8", errors="ignore"))
                        start = end
                    return chunks
                chunks = _chunk_text_by_bytes(markdown, 14900)
                for idx, ch in enumerate(chunks):
                    memory_requests.append(
                        MemoryTransformer.content_element_to_memory_request(
                            element=ContentElement(
                                element_id=f"md_{idx}",
                                content_type="text",
                                content=ch,
                                metadata={
                                    "organization_id": organization_id,
                                    "namespace_id": namespace_id,
                                    "content_type": "markdown",
                                    "chunk_index": idx,
                                    "total_chunks": len(chunks),
                                    "source": "provider_markdown_fallback"
                                }
                            ),
                            base_metadata=metadata,
                            organization_id=organization_id,
                            namespace_id=namespace_id
                        )
                    )
                logger.info(f"Fallback markdown path produced {len(memory_requests)} memories")
            except Exception as e:
                logger.warning(f"Fallback markdown path failed: {e}")

        # Helper: Detect if image URLs are temporary/expiring (need re-upload)
        def _has_temporary_image_urls(elements: List) -> bool:
            """Check if any elements have temporary/expiring image URLs that need re-upload"""
            for elem in elements:
                if elem.content_type in [ContentType.IMAGE, "image"]:
                    image_url = elem.image_url if hasattr(elem, 'image_url') else None
                    if image_url and isinstance(image_url, str):
                        # Check for common temporary URL patterns:
                        # - AWS S3 signed URLs (X-Amz-Signature, X-Amz-Expires)
                        # - Azure SAS tokens (sig=, se=)
                        # - Google Cloud signed URLs (Expires=, Signature=)
                        temp_patterns = [
                            "X-Amz-Signature", "X-Amz-Expires",  # AWS S3
                            "sig=", "se=",  # Azure Blob SAS
                            "Expires=", "Signature=",  # Google Cloud Storage
                            "x-goog-signature"  # Google Cloud Storage alt
                        ]
                        if any(pattern in image_url for pattern in temp_patterns):
                            return True
            return False
        
        # For large documents, store structured elements in Parse to avoid Temporal payload limits
        # Temporal has a ~2MB recommended limit per activity result
        # ALSO store if document has temporary image URLs (need re-upload before they expire)
        extraction_size_estimate = len(str(structured_elements)) + len(str(memory_requests))
        has_temp_images = _has_temporary_image_urls(structured_elements)
        should_store_in_parse = (extraction_size_estimate > 500_000) or has_temp_images  # 500KB threshold OR temporary images
        
        if has_temp_images:
            logger.info(f"Extraction has temporary image URLs - will store in Parse for re-upload (provider: {provider_name})")
        if extraction_size_estimate > 500_000:
            logger.info(f"Extraction is large ({extraction_size_estimate:,} bytes) - will store in Parse")
        
        extraction_result_id = None
        if should_store_in_parse and post_id:
            try:
                # Store extraction results in the existing Post's extractionResult field
                from services.memory_management import store_extraction_result_in_post
                extraction_result_id = await store_extraction_result_in_post(
                    post_id=post_id,
                    structured_elements=[elem.model_dump() for elem in structured_elements],
                    memory_requests=[req.model_dump() for req in memory_requests],
                    element_summary=element_types,
                    decision="complex"
                )
                logger.info(f"Stored large extraction result ({extraction_size_estimate:,} bytes) in Post {post_id}")

                # PIPELINE TRACKING: Update provider_extraction field
                try:
                    from core.document_processing.parse_integration import ParseDocumentIntegration
                    from memory.memory_graph import MemoryGraph
                    from datetime import datetime

                    memory_graph = MemoryGraph()
                    await memory_graph.ensure_async_connection()
                    parse_integration = ParseDocumentIntegration(memory_graph)

                    await parse_integration.update_post(post_id, {
                        # STAGE 1: Provider Extraction (pipeline tracking)
                        "provider_extraction": {
                            "url": extraction_result_id,  # Filename stored in Parse
                            "provider": provider_name,
                            "stats": {
                                "total_elements": len(structured_elements),
                                "text": element_types.get("text", 0),
                                "table": element_types.get("table", 0),
                                "image": element_types.get("image", 0),
                                "extraction_size_bytes": extraction_size_estimate,
                                "compression_ratio": None  # Set by store_extraction_result_in_post
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                            "duration_ms": None  # Would need activity start time
                        },
                        "pipeline_status": "processing"  # Set overall pipeline status
                    })
                    logger.info(f"Updated provider_extraction pipeline tracking in Post {post_id}")
                except Exception as track_err:
                    logger.warning(f"Failed to update provider_extraction tracking: {track_err}")

                # Return minimal payload with reference
                return {
                    "decision": "complex",
                    "extraction_stored": True,
                    "extraction_result_id": extraction_result_id,
                    "post_id": post_id,
                    "element_summary": element_types,
                    "structure_analysis": analysis,
                    "provider": provider_name,
                    "extraction_size": extraction_size_estimate
                }
            except Exception as store_err:
                logger.warning(f"Failed to store extraction in Parse, returning full payload: {store_err}")
                # Fall through to return full payload
        
        # For small documents or if storage failed, return full payload
        return {
            "decision": "complex",
            "structured_elements": [elem.model_dump() for elem in structured_elements],
            "memory_requests": [req.model_dump() for req in memory_requests],
            "element_summary": element_types,
            "structure_analysis": analysis,
            "provider": provider_name,
            "extraction_stored": False
        }

    except Exception as e:
        logger.error(f"Structured content extraction failed: {e}")
        raise
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in extract_structured_content_from_provider: {cleanup_error}")


def _deep_clean_metadata(obj: Any) -> Any:
    """Recursively clean metadata objects to be compatible with AddMemoryRequest.customMetadata
    
    customMetadata only accepts: Dict[str, Union[str, int, float, bool, List[str]]]
    This function converts nested dicts/objects to JSON strings
    """
    import json
    
    if obj is None:
        return None
    elif isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if v is None:
                continue  # Skip None values
            elif isinstance(v, (str, int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, list):
                # Check if it's a list of strings
                if all(isinstance(item, str) for item in v):
                    cleaned[k] = v
                else:
                    # Convert complex list to JSON string
                    cleaned[k] = json.dumps(v)
            elif isinstance(v, dict):
                # Convert nested dict to JSON string
                cleaned[k] = json.dumps(v)
            else:
                # Convert any other object to JSON string
                try:
                    cleaned[k] = json.dumps(v)
                except:
                    cleaned[k] = str(v)
        return cleaned
    else:
        return obj


@activity.defn
async def chunk_document_elements(
    content_elements: List[Dict[str, Any]],
    chunking_config: Optional[Dict[str, Any]] = None,
    post_id: Optional[str] = None,
    extraction_stored: bool = False
) -> Dict[str, Any]:
    """
    Temporal activity to apply hierarchical semantic chunking to extracted content elements.

    This groups small elements together and splits large ones to create optimal-sized chunks
    for embedding and search. Tables and images are preserved as separate chunks.

    Args:
        content_elements: List of ContentElement dicts (empty if extraction_stored=True)
        chunking_config: Optional chunking configuration (strategy, max_size, etc.)
        post_id: Parse Server Post ID containing stored extraction (if extraction_stored=True)
        extraction_stored: If True, fetch elements from Parse Server using post_id

    Returns:
        Dict with chunked_elements (or empty if stored back to Parse), stats, and metadata
    """
    from core.document_processing.hierarchical_chunker import HierarchicalChunker
    from models.hierarchical_models import ChunkingConfig, ChunkingStrategy

    # Fetch elements from Parse if they were stored there (large documents)
    if extraction_stored and post_id:
        logger.info(f"Fetching stored extraction from Parse Server Post chunk_document_elements {post_id}")
        try:
            from services.memory_management import fetch_extraction_result_from_post

            # Fetch the decompressed extraction data
            extraction_data = await fetch_extraction_result_from_post(post_id)

            if not extraction_data:
                raise Exception(f"Failed to fetch extraction from Post {post_id}")

            content_elements = extraction_data.get("structured_elements", [])
            logger.info(f"Fetched {len(content_elements)} elements from Parse Server")

        except Exception as e:
            logger.error(f"Failed to fetch extraction from Parse: {e}", exc_info=True)
            return {
                "chunked_elements": [],
                "stats": {
                    "original_count": 0,
                    "chunked_count": 0,
                    "reduction_percent": 0,
                    "error": f"Failed to fetch extraction: {str(e)}"
                },
                "config_used": {}
            }

    logger.info(f"Starting hierarchical chunking for {len(content_elements)} elements")

    memory_graph = None
    try:
        # Create chunking configuration
        if chunking_config:
            config = ChunkingConfig(**chunking_config)
        else:
            # Default configuration optimized for 1-2 page chunks with Qwen embeddings
            # Use HIERARCHICAL strategy to maintain section boundaries and document structure
            config = ChunkingConfig(
                strategy=ChunkingStrategy.HIERARCHICAL,
                max_chunk_size=6000,  # Support 1-2 pages (works with qwen 3 4b 2650-dim embeddings)
                min_chunk_size=1000,  # Avoid too-small fragments
                overlap_size=200,     # Small overlap for context continuity
                preserve_tables=True,  # Keep tables as separate chunks
                preserve_images=True,  # Keep images as separate chunks
                semantic_threshold=0.75  # Group elements with >75% semantic similarity
            )

        # Deserialize ContentElements from dicts
        from models.hierarchical_models import ContentElement, ContentType

        elements = []
        for elem_dict in content_elements:
            # Handle content_type enum conversion
            content_type_value = elem_dict.get("content_type", "text")
            if isinstance(content_type_value, str):
                try:
                    content_type = ContentType(content_type_value.lower())
                except ValueError:
                    logger.warning(f"Unknown content_type '{content_type_value}', defaulting to TEXT")
                    content_type = ContentType.TEXT
            else:
                content_type = content_type_value

            elem_dict["content_type"] = content_type

            # Reconstruct ContentElement
            element = ContentElement(**elem_dict)
            elements.append(element)

        # Apply hierarchical semantic chunking
        chunker = HierarchicalChunker(config)
        chunked_elements_objs = chunker._apply_chunking_strategy(elements, config)

        # Serialize back to dicts for Temporal
        chunked_elements = []
        for elem in chunked_elements_objs:
            elem_dict = {
                "element_id": elem.element_id,
                "content_type": elem.content_type.value,
                "content": elem.content,
                "metadata": elem.metadata or {}
            }

            # Preserve additional fields if present
            if hasattr(elem, 'structured_data') and elem.structured_data:
                elem_dict["structured_data"] = elem.structured_data
            if hasattr(elem, 'image_url') and elem.image_url:
                elem_dict["image_url"] = elem.image_url
            if hasattr(elem, 'image_description') and elem.image_description:
                elem_dict["image_description"] = elem.image_description

            chunked_elements.append(elem_dict)

        # Calculate statistics
        original_count = len(content_elements)
        chunked_count = len(chunked_elements)
        reduction_pct = ((original_count - chunked_count) / original_count * 100) if original_count > 0 else 0

        # Count element types
        element_types = {}
        for elem in chunked_elements:
            elem_type = elem.get("content_type", "text")
            element_types[elem_type] = element_types.get(elem_type, 0) + 1

        logger.info(f"Hierarchical chunking complete: {chunked_count} chunks from {original_count} elements ({reduction_pct:.1f}% reduction)")
        logger.info(f"Chunked element types: {element_types}")

        # If extraction was stored in Parse, store the chunked version back
        # This replaces the original extraction with the chunked version
        if extraction_stored and post_id:
            try:
                logger.info(f"Storing chunked elements back to Parse Server Post {post_id}")

                # Compress and upload chunked elements
                import gzip
                import json
                from services.memory_management import compress_extraction

                chunked_data = {
                    "structured_elements": chunked_elements,
                    "stats": {
                        "original_count": original_count,
                        "chunked_count": chunked_count,
                        "reduction_percent": reduction_pct,
                        "element_types": element_types
                    }
                }

                # Use the same compression logic as extraction
                compressed_bytes, compression_ratio = compress_extraction(chunked_data)

                # Upload to Parse Server storage
                from core.document_processing.parse_integration import ParseDocumentIntegration
                from memory.memory_graph import MemoryGraph

                memory_graph = MemoryGraph()
                await memory_graph.ensure_async_connection()
                parse_integration = ParseDocumentIntegration(memory_graph)

                # Generate filename for chunked extraction
                import hashlib
                content_hash = hashlib.sha256(str(chunked_data).encode()).hexdigest()[:32]
                chunked_filename = f"{content_hash}_chunked_{post_id}.json.gz"

                # Upload compressed file
                file_url = await parse_integration.upload_file(
                    compressed_bytes,
                    chunked_filename,
                    "application/gzip"
                )

                # Update Post with chunked extraction (preserving pipeline tracking)
                from datetime import datetime

                await parse_integration.update_post(post_id, {
                    # Quick access (backward compatibility)
                    "extraction_result": file_url,
                    "extraction_chunked": True,

                    # STAGE 2: Hierarchical Chunking (pipeline tracking)
                    "chunked_extraction": {
                        "url": file_url,
                        "stats": {
                            "original_count": original_count,
                            "chunked_count": chunked_count,
                            "reduction_percent": reduction_pct,
                            "element_types": element_types,
                            "avg_chunk_size": sum(len(e.get("content", "")) for e in chunked_elements) // chunked_count if chunked_count > 0 else 0,
                            "compression_ratio": compression_ratio
                        },
                        "config": {
                            "strategy": config.strategy.value,
                            "max_chunk_size": config.max_chunk_size,
                            "min_chunk_size": config.min_chunk_size
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                        "duration_ms": None  # Would need to track start time
                    }
                })

                logger.info(f"Stored chunked extraction to Parse: {file_url} (compression: {compression_ratio:.1f}%)")

                # Return empty chunked_elements since they're now stored in Parse
                # The LLM activity will fetch them
                return {
                    "chunked_elements": [],  # Empty - stored in Parse
                    "extraction_stored": True,
                    "post_id": post_id,
                    "stats": {
                        "original_count": original_count,
                        "chunked_count": chunked_count,
                        "reduction_percent": reduction_pct,
                        "element_types": element_types
                    },
                    "config_used": {
                        "strategy": config.strategy.value,
                        "max_chunk_size": config.max_chunk_size,
                        "min_chunk_size": config.min_chunk_size
                    }
                }

            except Exception as store_error:
                logger.error(f"Failed to store chunked elements to Parse: {store_error}", exc_info=True)
                # Fall through to return inline elements

        # Return chunked elements inline (small documents or storage failed)
        return {
            "chunked_elements": chunked_elements,
            "stats": {
                "original_count": original_count,
                "chunked_count": chunked_count,
                "reduction_percent": reduction_pct,
                "element_types": element_types
            },
            "config_used": {
                "strategy": config.strategy.value,
                "max_chunk_size": config.max_chunk_size,
                "min_chunk_size": config.min_chunk_size
            }
        }

    except Exception as e:
        logger.error(f"Hierarchical chunking failed: {e}", exc_info=True)
        # Return original elements on failure
        return {
            "chunked_elements": content_elements,
            "stats": {
                "original_count": len(content_elements),
                "chunked_count": len(content_elements),
                "reduction_percent": 0,
                "error": str(e)
            },
            "config_used": {}
        }
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in chunk_document_elements: {cleanup_error}")


@activity.defn
async def create_hierarchical_memory_batch(
    memory_requests: List[Dict[str, Any]],
    user_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: str,
    batch_size: int = 20,
    post_id: Optional[str] = None
) -> Dict[str, Any]:
    """Create memory items using hierarchical memory requests via the batch API
    
    Args:
        post_id: Optional existing Post ID to update (instead of creating a new Post for batch memories)
    """

    activity.heartbeat("Creating hierarchical memory batch")

    memory_graph = None
    try:
        from models.memory_models import AddMemoryRequest, BatchMemoryRequest, OptimizedAuthResponse
        from services.batch_processor import process_batch_with_temporal, validate_batch_size

        logger.info(f"Creating hierarchical memory batch with {len(memory_requests)} items")

        # Respect caller-provided batch_size but cap at Pydantic limit (50)
        MAX_BATCH_SIZE = 50
        effective_chunk_size = min(max(1, int(batch_size or MAX_BATCH_SIZE)), MAX_BATCH_SIZE)
        logger.info(f"Using chunk size: {effective_chunk_size} (cap={MAX_BATCH_SIZE}, requested={batch_size})")

        # Convert dict memory requests back to AddMemoryRequest objects
        # Deep clean metadata to ensure customMetadata is compatible
        memories = []
        for idx, req_dict in enumerate(memory_requests):
            try:
                # Deep clean the customMetadata to remove None and convert nested dicts
                if 'metadata' in req_dict and isinstance(req_dict['metadata'], dict):
                    if 'customMetadata' in req_dict['metadata'] and isinstance(req_dict['metadata']['customMetadata'], dict):
                        req_dict['metadata']['customMetadata'] = _deep_clean_metadata(req_dict['metadata']['customMetadata'])
                
                # Use model_validate which handles nested models better
                memory_req = AddMemoryRequest.model_validate(req_dict)
                memories.append(memory_req)
            except Exception as e:
                logger.error(f"Failed to validate memory request {idx}: {e}")
                logger.error(f"Request dict keys: {req_dict.keys()}")
                if 'metadata' in req_dict:
                    logger.error(f"Metadata keys: {req_dict['metadata'].keys() if isinstance(req_dict['metadata'], dict) else 'not a dict'}")
                # Skip this memory rather than failing the entire batch
                continue

        if not memories:
            raise Exception("No valid memory requests to process after validation")

        logger.info(f"Successfully validated {len(memories)} memory requests out of {len(memory_requests)} total")

        # Auth response for processor
        auth_response = OptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_qwen_route=False
        )

        activity.heartbeat("Starting batch memory creation")

        # Split into chunks of effective_chunk_size (<=50)
        total_memories_created = 0
        batch_results = []
        
        for i in range(0, len(memories), effective_chunk_size):
            chunk = memories[i:i + effective_chunk_size]
            chunk_num = (i // MAX_BATCH_SIZE) + 1
            total_chunks = (len(memories) + effective_chunk_size - 1) // effective_chunk_size
            
            # Safety check: ensure chunk never exceeds MAX_BATCH_SIZE
            if len(chunk) > MAX_BATCH_SIZE:
                logger.error(f"⚠️ Chunk {chunk_num} has {len(chunk)} items, truncating to {MAX_BATCH_SIZE}")
                chunk = chunk[:MAX_BATCH_SIZE]
            
            activity.heartbeat(f"Processing batch {chunk_num}/{total_chunks} ({len(chunk)} items)")
            logger.info(f"Creating BatchMemoryRequest with {len(chunk)} memories (max allowed: {MAX_BATCH_SIZE})")
            
            # Store batch chunk in Parse Server to avoid Temporal payload limits
            # Use existing post_id if provided (for first chunk) to avoid duplicate Posts
            from services.memory_management import store_batch_memories_in_parse
            
            chunk_post_id = await store_batch_memories_in_parse(
                memories=chunk,
                organization_id=organization_id,
                namespace_id=namespace_id,
                user_id=user_id,
                workspace_id=workspace_id,
                batch_metadata={
                    "chunk_num": chunk_num,
                    "total_chunks": total_chunks,
                    "source": "hierarchical_document_processing"
                },
                existing_post_id=post_id if chunk_num == 1 else None  # Only use existing post for first chunk
            )
            
            logger.info(f"Stored batch chunk {chunk_num} with {len(chunk)} memories in Post {chunk_post_id}")
            
            # Process batch from Parse reference (avoids GRPC payload limits)
            try:
                from cloud_plugins.temporal.activities.memory_activities import process_batch_memories_from_parse_reference
                
                activity.heartbeat(f"Processing batch chunk {chunk_num}/{total_chunks} from Parse")
                
                batch_result = await process_batch_memories_from_parse_reference(
                    post_id=chunk_post_id,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                    user_id=user_id,
                    workspace_id=workspace_id
                )
                
                batch_result["chunk_num"] = chunk_num
                batch_result["total_chunks"] = total_chunks
                
                logger.info(f"Successfully processed batch chunk {chunk_num}/{total_chunks}: {batch_result.get('successful', 0)}/{len(chunk)} memories created")
                
            except Exception as e:
                logger.error(f"Failed to process batch chunk {chunk_num}/{total_chunks}: {e}")
                batch_result = {
                    "post_id": chunk_post_id,
                    "status": "failed",
                    "error": str(e),
                    "chunk_num": chunk_num,
                    "total_chunks": total_chunks,
                    "successful": 0,
                    "failed": len(chunk)
                }
            
            batch_results.append(batch_result)
            total_memories_created += len(chunk)
            
            logger.info(f"Successfully processed batch {chunk_num}/{total_chunks} with {len(chunk)} memories")

        logger.info(f"Successfully created all hierarchical memory batches: {total_memories_created} memories in {len(batch_results)} batches")

        # PIPELINE TRACKING: Update indexing_results and mark pipeline complete
        if post_id:
            try:
                from core.document_processing.parse_integration import ParseDocumentIntegration
                from memory.memory_graph import MemoryGraph
                from datetime import datetime

                memory_graph = MemoryGraph()
                await memory_graph.ensure_async_connection()
                parse_integration = ParseDocumentIntegration(memory_graph)

                # Calculate total successful/failed from batch results
                total_successful = sum(br.get("successful", 0) for br in batch_results)
                total_failed = sum(br.get("failed", 0) for br in batch_results)

                # Count Neo4j nodes, Qdrant vectors, Parse memories
                # (For now, assume they're equal to successful - can be refined)
                neo4j_count = total_successful
                qdrant_count = total_successful
                parse_count = total_successful

                # Determine overall status
                if total_failed == 0:
                    status = "completed"
                elif total_successful > 0:
                    status = "partial"
                else:
                    status = "failed"

                await parse_integration.update_post(post_id, {
                    # STAGE 5: Indexing Results (pipeline tracking)
                    "indexing_results": {
                        "status": status,
                        "stats": {
                            "successful": total_successful,
                            "failed": total_failed,
                            "neo4j_nodes": neo4j_count,
                            "qdrant_vectors": qdrant_count,
                            "parse_memories": parse_count
                        },
                        "timestamp": datetime.utcnow().isoformat(),
                        "duration_ms": None  # Would need activity start time
                    },
                    "pipeline_status": status,  # Mark overall pipeline status
                    "pipeline_end": datetime.utcnow()  # Mark pipeline completion time
                })
                logger.info(f"Updated indexing_results pipeline tracking in Post {post_id}: {total_successful} successful, {total_failed} failed")
            except Exception as track_err:
                logger.warning(f"Failed to update indexing_results tracking: {track_err}")

        return {
            "batch_results": [br.model_dump() if hasattr(br, "model_dump") else br for br in batch_results],
            "memories_created": total_memories_created,
            "total_batches": len(batch_results),
            "batch_ids": [getattr(br, 'batch_id', None) for br in batch_results]
        }

    except Exception as e:
        logger.error(f"Hierarchical memory batch creation failed: {e}")
        raise
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in create_hierarchical_memory_batch: {cleanup_error}")


@activity.defn
async def update_post_pipeline_start(post_id: str, pipeline_start: str) -> None:
    """Update Post with pipeline_start timestamp
    
    IDEMPOTENCY: Verifies Post exists before updating to prevent workflow replay issues
    """
    memory_graph = None
    try:
        from core.document_processing.parse_integration import ParseDocumentIntegration
        from memory.memory_graph import MemoryGraph

        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()
        parse_integration = ParseDocumentIntegration(memory_graph)

        # CRITICAL: Verify Post exists before updating (idempotency check)
        # This prevents "Cannot use MongoClient after close" and 404 errors during workflow replay
        try:
            post_exists = await parse_integration.get_post(post_id)
            if not post_exists:
                # This is likely a Temporal workflow replay issue - the Post was created in a previous run
                # that was interrupted. Skip the update gracefully rather than failing the workflow.
                logger.warning(f"⚠️  Post {post_id} does not exist (likely workflow replay). Skipping pipeline_start update.")
                return  # Skip update gracefully
        except Exception as check_err:
            # If we can't verify the Post, skip gracefully (likely replay issue)
            logger.warning(f"⚠️  Could not verify Post {post_id} exists ({check_err}). Skipping pipeline_start update.")
            return  # Skip update gracefully

        # Convert ISO string to Parse Date format
        parse_date = {
            "__type": "Date",
            "iso": pipeline_start
        }

        success = await parse_integration.update_post(post_id, {
            "pipeline_start": parse_date,
            "pipeline_status": "processing"
        })
        
        if not success:
            raise Exception(f"Failed to update Post {post_id} with pipeline_start (update returned False)")
        
        logger.info(f"✅ Set pipeline_start for Post {post_id}")
    except Exception as e:
        logger.error(f"❌ Failed to set pipeline_start for Post {post_id}: {e}", exc_info=True)
        raise
    finally:
        # Ensure MemoryGraph cleanup happens even on exceptions
        if memory_graph:
            try:
                await memory_graph.cleanup()
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup MemoryGraph in update_post_pipeline_start: {cleanup_error}")


@activity.defn
async def update_post_llm_extraction(
    post_id: str,
    llm_result_stored: bool,
    total_generated: int,
    llm_result_id: Optional[str] = None,
    generation_summary: Optional[Dict[str, Any]] = None
) -> None:
    """Update Post with LLM-enhanced extraction results (Stage 3)
    
    This activity only stores metadata about LLM processing, not the actual results.
    The full LLM results are either stored in Parse (if llm_result_stored=True) or 
    passed inline in the workflow (if small).
    
    IDEMPOTENCY: Verifies Post exists before updating to prevent workflow replay issues
    
    Args:
        post_id: Parse Post objectId
        llm_result_stored: Whether LLM result was stored in Parse (large documents)
        total_generated: Total number of memory requests generated
        llm_result_id: Optional URL/ID of stored LLM result in Parse
        generation_summary: Optional summary of generated content types
    """
    try:
        from core.document_processing.parse_integration import ParseDocumentIntegration
        from memory.memory_graph import MemoryGraph
        from datetime import datetime, timezone

        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()
        parse_integration = ParseDocumentIntegration(memory_graph)

        # CRITICAL: Verify Post exists before updating (idempotency check)
        try:
            post_exists = await parse_integration.get_post(post_id)
            if not post_exists:
                error_msg = f"Post {post_id} does not exist. Cannot update LLM extraction metadata."
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as check_err:
            error_msg = f"Failed to verify Post {post_id} exists before updating LLM extraction: {check_err}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Build llm_enhanced_extraction metadata (no large payloads!)
        llm_enhanced = {
            "stats": {
                "elements_processed": total_generated,
                "llm_calls": total_generated,  # Approximate (1 call per element)
                "total_tokens_used": None,  # TODO: Track tokens from LLM calls
                "model": "gemini-2.5-flash",  # Default model
                "failed_elements": 0,  # TODO: Track failures
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": None  # Would need activity start time
        }
        
        # If LLM result is stored in Parse (large documents), add URL
        if llm_result_stored and llm_result_id:
            llm_enhanced["url"] = llm_result_id
        
        # Add generation summary if provided
        if generation_summary:
            llm_enhanced["generation_summary"] = generation_summary
        
        success = await parse_integration.update_post(post_id, {
            "llm_enhanced_extraction": llm_enhanced,
            "pipeline_status": "llm_enhanced"
        })
        
        if not success:
            raise Exception(f"Failed to update Post {post_id} with LLM extraction metadata (update returned False)")
        
        logger.info(f"✅ Updated llm_enhanced_extraction for Post {post_id}: {total_generated} elements processed")
    except Exception as e:
        logger.error(f"❌ Failed to update llm_enhanced_extraction for Post {post_id}: {e}", exc_info=True)
        raise


@activity.defn
async def update_post_indexing_results(
    post_id: str,
    indexing_result: Dict[str, Any],
    pipeline_start_iso: Optional[str] = None,
    pipeline_end_iso: Optional[str] = None
) -> None:
    """Update Post with final indexing results (Stage 5) and pipeline completion
    
    IDEMPOTENCY: Verifies Post exists before updating to prevent workflow replay issues
    
    Args:
        post_id: Parse Post objectId
        indexing_result: Results from batch indexing workflow
        pipeline_start_iso: Pipeline start time (ISO string) for duration calculation
        pipeline_end_iso: Pipeline end time (ISO string)
    """
    try:
        from core.document_processing.parse_integration import ParseDocumentIntegration
        from memory.memory_graph import MemoryGraph
        from datetime import datetime, timezone
        import time

        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()
        parse_integration = ParseDocumentIntegration(memory_graph)

        # CRITICAL: Verify Post exists before updating (idempotency check)
        try:
            post_exists = await parse_integration.get_post(post_id)
            if not post_exists:
                error_msg = f"Post {post_id} does not exist. Cannot update indexing results."
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as check_err:
            error_msg = f"Failed to verify Post {post_id} exists before updating indexing results: {check_err}"
            logger.error(error_msg)
            raise Exception(error_msg)

        # Build indexing_results metadata
        indexing_results = {
            "status": indexing_result.get("status", "unknown"),
            "stats": {
                "successful": indexing_result.get("successful", 0),
                "failed": indexing_result.get("failed", 0),
                "total_processed": indexing_result.get("total_processed", 0),
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Use provided pipeline_end_iso or generate current time
        if pipeline_end_iso:
            pipeline_end_dict = {
                "__type": "Date",
                "iso": pipeline_end_iso
            }
        else:
            pipeline_end = datetime.now(timezone.utc)
            pipeline_end_dict = {
                "__type": "Date",
                "iso": pipeline_end.isoformat()
            }
        
        update_fields = {
            "indexing_results": indexing_results,
            "pipeline_end": pipeline_end_dict,
            "pipeline_status": "completed" if indexing_result.get("status") == "completed" else "failed"
        }
        
        # Calculate total duration if both start and end times are provided
        if pipeline_start_iso and pipeline_end_iso:
            try:
                start_dt = datetime.fromisoformat(pipeline_start_iso.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(pipeline_end_iso.replace('Z', '+00:00'))
                total_duration_ms = int((end_dt - start_dt).total_seconds() * 1000)
                update_fields["total_duration_ms"] = total_duration_ms
                logger.info(f"Pipeline duration for Post {post_id}: {total_duration_ms}ms ({total_duration_ms/1000:.2f}s)")
            except Exception as e:
                logger.warning(f"Could not calculate pipeline duration: {e}")
        
        success = await parse_integration.update_post(post_id, update_fields)
        
        if not success:
            raise Exception(f"Failed to update Post {post_id} with indexing results (update returned False)")
        
        logger.info(f"✅ Updated indexing_results for Post {post_id}: {indexing_results['stats']['successful']}/{indexing_results['stats']['total_processed']} successful")
    except Exception as e:
        logger.error(f"❌ Failed to update indexing_results for Post {post_id}: {e}", exc_info=True)
        raise


@activity.defn
async def analyze_document_structure(
    processing_result: Dict[str, Any],
    organization_id: str,
    namespace_id: str
) -> Dict[str, Any]:
    """Analyze document structure to understand content organization"""

    activity.heartbeat("Analyzing document structure")

    try:
        logger.info("Analyzing document structure for hierarchical processing")

        pages = processing_result.get("pages", [])
        provider_specific = processing_result.get("provider_specific", {})
        stats = processing_result.get("stats", {})

        # Basic structure analysis
        analysis = {
            "total_pages": len(pages),
            "average_page_length": sum(len(p.get("content", "")) for p in pages) / len(pages) if pages else 0,
            "provider": stats.get("provider", "unknown"),
            "confidence": stats.get("confidence", 0.0),
            "has_structured_data": bool(provider_specific),
            "processing_time": stats.get("processing_time", 0)
        }

        # Analyze content patterns
        content_patterns = {
            "has_tables": False,
            "has_images": False,
            "has_charts": False,
            "text_density": "normal"
        }

        # Check provider-specific data for structured content indicators
        if provider_specific:
            if 'tables' in provider_specific or 'table' in str(provider_specific).lower():
                content_patterns["has_tables"] = True
            if 'images' in provider_specific or 'image' in str(provider_specific).lower():
                content_patterns["has_images"] = True
            if 'charts' in provider_specific or 'chart' in str(provider_specific).lower():
                content_patterns["has_charts"] = True

        # Estimate processing strategy
        recommended_strategy = "basic"
        if content_patterns["has_tables"] or content_patterns["has_images"]:
            recommended_strategy = "hierarchical"
        if content_patterns["has_tables"] and content_patterns["has_images"]:
            recommended_strategy = "multi_modal"

        analysis.update(content_patterns)
        analysis["recommended_strategy"] = recommended_strategy

        logger.info(f"Document structure analysis: {analysis}")

        return {
            "structure_analysis": analysis,
            "requires_hierarchical_processing": recommended_strategy in ["hierarchical", "multi_modal"]
        }

    except Exception as e:
        logger.error(f"Document structure analysis failed: {e}")
        raise


@activity.defn
async def generate_llm_optimized_memory_structures(
    content_elements: List[Dict[str, Any]],
    domain: Optional[str],
    base_metadata: MemoryMetadata,
    organization_id: str,
    namespace_id: str,
    use_llm: bool = True,
    post_id: Optional[str] = None,  # For fetching stored extraction if needed
    extraction_stored: bool = False  # Flag indicating extraction was stored in Parse
) -> Dict[str, Any]:
    """Generate LLM-optimized memory structures from content elements
    
    Args:
        content_elements: List of ContentElement dicts (or empty if extraction_stored=True)
        domain: Optional domain for context
        base_metadata: Base memory metadata
        organization_id: Organization ID
        namespace_id: Namespace ID
        use_llm: Whether to use LLM for optimization
        post_id: Post ID to fetch extraction from if extraction_stored=True
        extraction_stored: Whether extraction was stored in Parse (for large documents)
    """

    if not HIERARCHICAL_AVAILABLE:
        raise Exception("Hierarchical processing not available")

    try:
        activity.heartbeat("Starting LLM memory structure generation")
    except Exception:
        pass

    try:
        # Determine the source of content_elements based on extraction_stored flag
        elements_to_process = []
        
        if extraction_stored and post_id:
            # Large document: fetch extraction from Parse Server
            try:
                from services.memory_management import fetch_extraction_result_from_post
                
                logger.info(f"Fetching stored extraction result from Post {post_id} (extraction_stored={extraction_stored})")
                extraction_data = await fetch_extraction_result_from_post(post_id)
                
                if not extraction_data:
                    error_msg = f"Extraction was marked as stored (extraction_stored=True) but fetch returned None for Post {post_id}. This likely means the extraction wasn't stored properly in Parse Server."
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
                # Use structured_elements from stored extraction
                elements_to_process = extraction_data.get("structured_elements", [])
                logger.info(f"Fetched {len(elements_to_process)} elements from stored extraction")
                
                if not elements_to_process:
                    error_msg = f"Fetched extraction from Post {post_id} but structured_elements is empty. Extraction data: {extraction_data.keys()}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                
            except Exception as fetch_err:
                logger.error(f"Failed to fetch stored extraction, cannot proceed: {fetch_err}", exc_info=True)
                raise
        else:
            # Small document: use inline content_elements from function parameter
            elements_to_process = content_elements
            logger.info(f"Using {len(elements_to_process)} inline content elements (extraction_stored={extraction_stored})")
        
        logger.info(f"Generating LLM-optimized memory structures for {len(elements_to_process)} elements in {domain or 'general'} domain")

        # Convert dict elements back to ContentElement objects
        elements = []
        for elem_dict in elements_to_process:
            # Reconstruct ContentElement from dict based on type
            content_type_value = elem_dict.get("content_type", "text")

            # Convert string to ContentType enum if needed
            if isinstance(content_type_value, str):
                try:
                    content_type = ContentType(content_type_value.lower())
                except ValueError:
                    logger.warning(f"Unknown content_type '{content_type_value}', defaulting to TEXT")
                    content_type = ContentType.TEXT
            else:
                content_type = content_type_value

            # Update dict with proper enum before passing to constructors
            elem_dict["content_type"] = content_type

            if content_type in [ContentType.TABLE, "table"]:
                element = TableElement(**elem_dict)
            elif content_type in [ContentType.IMAGE, "image"]:
                element = ImageElement(**elem_dict)
            else:
                element = ContentElement(**elem_dict)

            elements.append(element)

        try:
            activity.heartbeat(f"Reconstructed {len(elements)} content elements")
        except Exception:
            pass

        # Create base metadata object
        metadata = base_metadata if isinstance(base_metadata, MemoryMetadata) else MemoryMetadata(**(base_metadata or {}))

        # Collect document metadata for contextual retrieval (2025 research - Phase 1)
        # Extract from base_metadata or elements
        document_metadata = {}

        # Try to get document title from various sources
        if metadata and metadata.customMetadata:
            document_metadata['title'] = metadata.customMetadata.get('document_title') or \
                                        metadata.customMetadata.get('title') or \
                                        metadata.customMetadata.get('file_name', 'Unknown Document')
            document_metadata['type'] = metadata.customMetadata.get('document_type', 'document')
            document_metadata['domain'] = domain or 'general'

        # Try to infer total pages from elements
        if elements:
            page_numbers = [e.metadata.get('page_number', 0) for e in elements if hasattr(e, 'metadata') and e.metadata]
            page_numbers = [p for p in page_numbers if isinstance(p, int) and p > 0]
            if page_numbers:
                document_metadata['total_pages'] = max(page_numbers)
            else:
                document_metadata['total_pages'] = len(elements) if elements else 'Unknown'

        logger.info(f"Using document context for LLM: {document_metadata.get('title', 'Unknown')} ({document_metadata.get('total_pages', 'Unknown')} pages)")

        # Generate optimized memory structures using LLM (or deterministic transformer when disabled)
        if use_llm:
            memory_requests = await generate_optimized_memory_structures(
                content_elements=elements,
                domain=domain,
                base_metadata=metadata,
                document_metadata=document_metadata  # Pass document context to LLM
            )
        else:
            memory_requests = [
                MemoryTransformer.content_element_to_memory_request(
                    element=e,
                    base_metadata=metadata,
                    organization_id=organization_id,
                    namespace_id=namespace_id,
                ) for e in elements
            ]

        # Add organization/namespace and document metadata fields to each memory
        for req in memory_requests:
            req.organization_id = organization_id
            req.namespace_id = namespace_id

            # Add document-level metadata fields for filtering/search (Phase 1)
            if req.metadata and req.metadata.customMetadata:
                # Add document context fields if not already present
                if 'document_title' not in req.metadata.customMetadata:
                    req.metadata.customMetadata['document_title'] = document_metadata.get('title', 'Unknown')
                if 'document_type' not in req.metadata.customMetadata:
                    req.metadata.customMetadata['document_type'] = document_metadata.get('type', 'document')
                if 'total_pages' not in req.metadata.customMetadata:
                    req.metadata.customMetadata['total_pages'] = document_metadata.get('total_pages', 'Unknown')
                if 'domain' not in req.metadata.customMetadata:
                    req.metadata.customMetadata['domain'] = document_metadata.get('domain', 'general')

            # Sanitize customMetadata to primitives
            try:
                if getattr(req, "metadata", None) and isinstance(getattr(req.metadata, "customMetadata", None), dict):
                    req.metadata.customMetadata = _deep_clean_metadata(req.metadata.customMetadata)
            except Exception:
                pass

        try:
            activity.heartbeat(f"Generated {len(memory_requests)} LLM-optimized memory structures")
        except Exception:
            pass

        # Analyze what was generated
        generated_summary = {}
        for req in memory_requests:
            llm_type = req.metadata.customMetadata.get('content_type', 'text') if req.metadata and req.metadata.customMetadata else 'text'
            generated_summary[llm_type] = generated_summary.get(llm_type, 0) + 1

        logger.info(f"LLM memory generation summary for {domain or 'general'} domain: {generated_summary}")

        # Convert memory requests to dicts for return (exclude None to avoid validation issues)
        memory_requests_dicts = [req.model_dump(exclude_none=True, mode='json') for req in memory_requests]

        # Enforce schema via Pydantic validation to guarantee downstream types
        try:
            from models.memory_models import AddMemoryRequest as _AddMemoryRequest
            validated_dicts = []
            for d in memory_requests_dicts:
                try:
                    v = _AddMemoryRequest.model_validate(d)
                    # Ensure metadata.customMetadata is primitives-only
                    if getattr(v, "metadata", None) and isinstance(getattr(v.metadata, "customMetadata", None), dict):
                        v.metadata.customMetadata = _deep_clean_metadata(v.metadata.customMetadata)
                    validated_dicts.append(v.model_dump(exclude_none=True, mode='json'))
                except Exception:
                    # Best-effort fallback: keep original dict if validation fails
                    validated_dicts.append(d)
            memory_requests_dicts = validated_dicts
        except Exception:
            pass
        
        # Check if response would exceed Temporal payload limits (~2MB recommended)
        # If so, store in Parse Server and return reference
        response_size_estimate = len(str(memory_requests_dicts))
        should_store_in_parse = response_size_estimate > 500_000  # 500KB threshold
        
        if should_store_in_parse and post_id:
            try:
                # Store LLM-generated memory requests in Parse Server
                from services.memory_management import store_extraction_result_in_post
                
                llm_result_id = await store_extraction_result_in_post(
                    post_id=post_id,
                    structured_elements=[],  # Already processed into memory requests
                    memory_requests=memory_requests_dicts,
                    element_summary=generated_summary,
                    decision="llm_optimized"
                )
                
                logger.info(f"Stored large LLM result ({response_size_estimate:,} bytes) in Post {post_id}")

                # PIPELINE TRACKING: Update llm_enhanced_extraction field
                try:
                    from core.document_processing.parse_integration import ParseDocumentIntegration
                    from memory.memory_graph import MemoryGraph
                    from datetime import datetime

                    memory_graph = MemoryGraph()
                    await memory_graph.ensure_async_connection()
                    parse_integration = ParseDocumentIntegration(memory_graph)

                    # Count chunking validation results from memory metadata
                    chunking_validation_results = {"coherent": 0, "incomplete": 0, "mixed_topics": 0}
                    for req in memory_requests:
                        if req.metadata and req.metadata.customMetadata:
                            validation = req.metadata.customMetadata.get("chunking_validation", {})
                            if isinstance(validation, dict):
                                completeness = validation.get("completeness", "complete")
                                if completeness == "complete":
                                    chunking_validation_results["coherent"] += 1
                                elif "incomplete" in completeness:
                                    chunking_validation_results["incomplete"] += 1
                                elif "mixed" in completeness:
                                    chunking_validation_results["mixed_topics"] += 1

                    await parse_integration.update_post(post_id, {
                        # STAGE 3: LLM Metadata Enhancement (pipeline tracking)
                        "llm_enhanced_extraction": {
                            "url": llm_result_id,  # Filename stored in Parse
                            "stats": {
                                "elements_processed": len(elements_to_process),
                                "llm_calls": len(memory_requests),
                                "total_tokens_used": None,  # TODO: Track tokens from LLM calls
                                "avg_tokens_per_chunk": None,  # TODO: Calculate from token tracking
                                "model": "gpt-5-nano",  # TODO: Get from actual model used
                                "failed_elements": 0,  # TODO: Track failures
                                "chunking_validation_results": chunking_validation_results
                            },
                            "timestamp": datetime.utcnow().isoformat(),
                            "duration_ms": None  # Would need activity start time
                        },
                        "pipeline_status": "processing"
                    })
                    logger.info(f"Updated llm_enhanced_extraction pipeline tracking in Post {post_id}")
                except Exception as track_err:
                    logger.warning(f"Failed to update llm_enhanced_extraction tracking: {track_err}")

                # Return minimal payload with reference
                return {
                    "llm_result_stored": True,
                    "llm_result_id": llm_result_id,
                    "post_id": post_id,
                    "generation_summary": generated_summary,
                    "domain": domain,
                    "llm_optimized": True,
                    "total_generated": len(memory_requests),
                    "external_user_id": metadata.user_id,
                    "organization_id": organization_id,
                    "namespace_id": namespace_id,
                    "result_size": response_size_estimate
                }
            except Exception as store_err:
                logger.warning(f"Failed to store LLM result in Parse, returning full payload: {store_err}")
                # Fall through to return full payload
        
        # For small results or if storage failed, return full payload
        response: Dict[str, Any] = {
            "memory_requests": memory_requests_dicts,
            "generation_summary": generated_summary,
            "domain": domain,
            "llm_optimized": True,
            "total_generated": len(memory_requests),
            "external_user_id": metadata.user_id,
            "organization_id": organization_id,
            "namespace_id": namespace_id,
            "llm_result_stored": False
        }
        return response

    except Exception as e:
        logger.error(f"LLM memory structure generation failed: {e}")
        raise


@activity.defn
async def fetch_llm_result_from_post(post_id: str) -> Dict[str, Any]:
    """Fetch LLM-generated memory requests from Parse Server (for large documents)"""
    
    try:
        activity.heartbeat(f"Fetching LLM result from Post {post_id}")
        
        from services.memory_management import fetch_extraction_result_from_post
        
        logger.info(f"Fetching LLM-generated memory requests from Post {post_id}")
        
        # Reuse the same fetch function - it works for both extraction and LLM results
        extraction_data = await fetch_extraction_result_from_post(post_id)
        
        if not extraction_data:
            raise Exception(f"Failed to fetch LLM result from Post {post_id}")
        
        memory_requests = extraction_data.get("memory_requests", [])
        
        if not memory_requests:
            raise Exception(f"LLM result in Post {post_id} has no memory_requests")
        
        logger.info(f"Successfully fetched {len(memory_requests)} memory requests from Post {post_id}")
        
        return {
            "memory_requests": memory_requests,
            "generation_summary": extraction_data.get("element_summary", {}),
            "decision": extraction_data.get("decision", "llm_optimized")
        }
        
    except Exception as e:
        logger.error(f"Failed to fetch LLM result from Post {post_id}: {e}")
        raise


@activity.defn
async def extract_and_upload_images_from_pdf(
    pdf_url: str,
    structured_elements: List[Dict[str, Any]],
    organization_id: str,
    namespace_id: str,
    workspace_id: Optional[str],
    post_id: Optional[str] = None
) -> Dict[str, Any]:
    """Extract images from PDF using bbox coordinates and upload to Parse Server
    
    This activity:
    1. Downloads the PDF from pdf_url (typically from Reducto/provider)
    2. Uses provided structured_elements with ImageElement bbox coordinates
    3. Uses PyMuPDF to crop images from the PDF based on bbox coordinates
    4. Uploads each cropped image to Parse Server as a File
    5. Optionally updates the Post's extractionResultFile with image_url mappings
    
    Args:
        pdf_url: URL to the PDF file (from provider response)
        structured_elements: List of structured elements containing image elements with bbox
        organization_id: Organization ID
        namespace_id: Namespace ID
        workspace_id: Workspace ID
        post_id: Optional Parse Post ID for updating extraction results
    
    Returns:
        Dict with uploaded image URLs mapped to element IDs
    """
    
    try:
        activity.heartbeat(f"Extracting images from PDF: {pdf_url}")
        
        import fitz  # PyMuPDF
        from io import BytesIO
        import base64
        from services.memory_management import store_extraction_result_in_post
        
        logger.info(f"Starting image extraction from PDF: {pdf_url}")
        
        # Step 1: Download PDF
        activity.heartbeat("Downloading PDF")
        async with httpx.AsyncClient(timeout=300.0) as client:
            pdf_response = await client.get(pdf_url)
            if pdf_response.status_code != 200:
                raise Exception(f"Failed to download PDF: {pdf_response.status_code}")
            pdf_bytes = pdf_response.content
        
        logger.info(f"Downloaded PDF: {len(pdf_bytes):,} bytes")
        
        # Step 2: Use provided structured_elements to find images with bbox
        activity.heartbeat("Processing image elements")
        
        # Filter for image elements that have bbox but no image_url
        image_elements_to_extract = []
        for elem in structured_elements:
            if elem.get("content_type") == "image":
                metadata = elem.get("metadata", {})
                bbox = metadata.get("bbox")
                image_url = elem.get("image_url")
                
                # Only extract if we have bbox but no URL
                if bbox and not image_url:
                    image_elements_to_extract.append(elem)
        
        logger.info(f"Found {len(image_elements_to_extract)} images to extract from PDF")
        
        if not image_elements_to_extract:
            logger.info("No images need extraction (all have URLs or no bbox)")
            return {
                "images_extracted": 0,
                "image_url_mappings": {}
            }
        
        # Step 3: Open PDF with PyMuPDF
        activity.heartbeat(f"Processing {len(image_elements_to_extract)} images")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        # Parse Server configuration
        parse_url = env.get("PARSE_SERVER_URL")
        parse_app_id = env.get("PARSE_APPLICATION_ID")
        parse_master_key = env.get("PARSE_MASTER_KEY")
        
        headers = {
            "X-Parse-Application-Id": parse_app_id,
            "X-Parse-Master-Key": parse_master_key,
        }
        
        image_url_mappings = {}
        uploaded_count = 0
        
        # Step 4: Extract and upload each image
        for idx, elem in enumerate(image_elements_to_extract):
            try:
                element_id = elem.get("element_id")
                metadata = elem.get("metadata", {})
                bbox = metadata.get("bbox", {})
                image_description = elem.get("image_description", "image")
                
                # Extract bbox coordinates
                page_num = bbox.get("page", 1) - 1  # PyMuPDF uses 0-based pages
                left = bbox.get("left", 0)
                top = bbox.get("top", 0)
                width = bbox.get("width", 0)
                height = bbox.get("height", 0)
                
                # Validate page number
                if page_num < 0 or page_num >= pdf_document.page_count:
                    logger.warning(f"Invalid page number {page_num + 1} for element {element_id}")
                    continue
                
                # Get the PDF page
                page = pdf_document[page_num]
                page_rect = page.rect
                
                # Convert normalized coordinates (0-1) to absolute pixel coordinates
                abs_left = left * page_rect.width
                abs_top = top * page_rect.height
                abs_width = width * page_rect.width
                abs_height = height * page_rect.height
                
                # Create crop rectangle (left, top, right, bottom)
                crop_rect = fitz.Rect(
                    abs_left,
                    abs_top,
                    abs_left + abs_width,
                    abs_top + abs_height
                )
                
                # Render the cropped area as an image
                # Use matrix for higher resolution (2x scale for better quality)
                mat = fitz.Matrix(2, 2)
                pix = page.get_pixmap(matrix=mat, clip=crop_rect)
                
                # Convert to PNG bytes
                image_bytes = pix.pil_tobytes(format="PNG")
                
                # Generate filename
                safe_description = "".join(c if c.isalnum() else "_" for c in image_description[:30])
                filename = f"{element_id}_{safe_description}.png"
                
                # Upload to Parse Server as a File
                activity.heartbeat(f"Uploading image {idx + 1}/{len(image_elements_to_extract)}: {filename}")
                
                # Parse File upload endpoint
                async with httpx.AsyncClient(timeout=60.0) as client:
                    file_upload_response = await client.post(
                        f"{parse_url}/parse/files/{filename}",
                        content=image_bytes,
                        headers={
                            **headers,
                            "Content-Type": "image/png"
                        }
                    )
                    
                    if file_upload_response.status_code in [200, 201]:
                        file_data = file_upload_response.json()
                        image_url = file_data.get("url")
                        
                        if image_url:
                            image_url_mappings[element_id] = image_url
                            uploaded_count += 1
                            logger.info(f"Uploaded image {element_id}: {image_url}")
                        else:
                            logger.warning(f"File upload succeeded but no URL returned for {element_id}")
                    else:
                        logger.error(f"Failed to upload image {element_id}: {file_upload_response.status_code} {file_upload_response.text}")
                
            except Exception as img_err:
                logger.error(f"Failed to extract/upload image {element_id}: {img_err}")
                continue
        
        # Close PDF document
        pdf_document.close()
        
        logger.info(f"Successfully extracted and uploaded {uploaded_count}/{len(image_elements_to_extract)} images")
        
        # Step 5: Update structured_elements in-place with new image URLs
        if image_url_mappings:
            activity.heartbeat("Updating elements with image URLs")
            
            # Update structured_elements with new image URLs
            for elem in structured_elements:
                element_id = elem.get("element_id")
                if element_id in image_url_mappings:
                    # Add the image_url to the element
                    elem["image_url"] = image_url_mappings[element_id]
                    
                    # Also update the content to include the markdown image link
                    image_url = image_url_mappings[element_id]
                    image_description = elem.get("image_description", "image")
                    elem["content"] = f"![{image_description}]({image_url})\n\n*{image_description}*"
                    
                    logger.info(f"Updated element {element_id} with image URL")
            
            logger.info(f"Updated {len(image_url_mappings)} elements with image URLs")
        
        return {
            "images_extracted": uploaded_count,
            "total_candidates": len(image_elements_to_extract),
            "image_url_mappings": image_url_mappings,
            "post_id": post_id
        }
        
    except Exception as e:
        logger.error(f"Image extraction from PDF failed: {e}", exc_info=True)
        raise


@activity.defn
async def download_and_reupload_provider_images(
    post_id: str,
    provider: str = "unknown"
) -> Dict[str, Any]:
    """Download images with temporary URLs from any provider and re-upload to Parse with permanent URLs
    
    This activity works with ALL providers that return temporary/expiring image URLs:
    - AWS S3 signed URLs (Reducto, TensorLake) - expire after 1 hour
    - Azure Blob Storage SAS tokens - configurable expiration
    - Google Cloud Storage signed URLs - configurable expiration
    
    Process:
    1. Fetches structured_elements from Parse Server
    2. Identifies images with temporary URLs (detects AWS/Azure/GCS signature patterns)
    3. Downloads each image from the temporary provider URL
    4. Uploads to Parse Server storage (permanent URLs)
    5. Replaces temporary URLs with permanent Parse URLs in structured_elements
    6. Updates the stored extraction back to Parse
    
    Args:
        post_id: Parse Post ID containing the extraction results
        provider: Provider name for logging (e.g., "reducto", "tensorlake", "gemini")
    
    Returns:
        Dict with count of images re-uploaded and URL mappings
    """
    
    try:
        activity.heartbeat(f"Re-uploading {provider} images for Post {post_id}")
        
        from services.memory_management import fetch_extraction_result_from_post, store_extraction_result_in_post
        
        logger.info(f"Fetching stored extraction from Post {post_id} to re-upload {provider} images")
        
        # Step 1: Fetch extraction results from Post
        extraction_data = await fetch_extraction_result_from_post(post_id)
        
        if not extraction_data:
            raise Exception(f"No extraction data found for Post {post_id}")
        
        structured_elements = extraction_data.get("structured_elements", [])
        
        if not structured_elements:
            logger.info("No structured elements found in stored extraction")
            return {
                "images_reuploaded": 0,
                "total_candidates": 0,
                "url_mappings": {},
                "post_id": post_id
            }
        
        logger.info(f"Fetched {len(structured_elements)} elements from stored extraction")
        
        # Step 2: Find images with temporary URLs that need re-uploading
        # Check for common temporary URL patterns across all cloud providers
        def _is_temporary_url(url: str) -> bool:
            """Check if URL is temporary/expiring (AWS S3, Azure SAS, Google Cloud signed)"""
            if not url:
                return False
            temp_patterns = [
                "X-Amz-Signature", "X-Amz-Expires",  # AWS S3 signed URLs
                "sig=", "se=",  # Azure Blob Storage SAS tokens
                "Expires=", "Signature=",  # Google Cloud Storage signed URLs
                "x-goog-signature"  # Google Cloud Storage alternative
            ]
            return any(pattern in url for pattern in temp_patterns)
        
        images_to_reupload = []
        for elem in structured_elements:
            if elem.get("content_type") == "image":
                image_url = elem.get("image_url")
                if _is_temporary_url(image_url):
                    images_to_reupload.append(elem)
        
        logger.info(f"Found {len(images_to_reupload)} images with temporary URLs to re-upload (provider: {provider})")
        
        if not images_to_reupload:
            logger.info(f"No {provider} images need re-uploading")
            return {
                "images_reuploaded": 0,
                "total_candidates": 0,
                "url_mappings": {},
                "post_id": post_id
            }
        
        # Step 3: Download and re-upload each image
        parse_url = env.get("PARSE_SERVER_URL")
        parse_app_id = env.get("PARSE_APPLICATION_ID")
        parse_master_key = env.get("PARSE_MASTER_KEY")
        
        headers = {
            "X-Parse-Application-Id": parse_app_id,
            "X-Parse-Master-Key": parse_master_key,
        }
        
        url_mappings = {}
        reuploaded_count = 0
        
        for idx, elem in enumerate(images_to_reupload):
            try:
                element_id = elem.get("element_id")
                old_url = elem.get("image_url")
                image_description = elem.get("image_description", "image")
                
                activity.heartbeat(f"Re-uploading image {idx + 1}/{len(images_to_reupload)}: {element_id}")
                
                # Download image from provider URL
                async with httpx.AsyncClient(timeout=60.0) as client:
                    logger.info(f"Downloading {provider} image from: {old_url[:100]}...")
                    image_response = await client.get(old_url)
                    
                    if image_response.status_code != 200:
                        logger.error(f"Failed to download {provider} image {element_id}: {image_response.status_code}")
                        continue
                    
                    image_bytes = image_response.content
                    content_type = image_response.headers.get("content-type", "image/png")
                    
                    # Determine file extension from content-type
                    if "png" in content_type:
                        ext = "png"
                    elif "jpeg" in content_type or "jpg" in content_type:
                        ext = "jpg"
                    elif "webp" in content_type:
                        ext = "webp"
                    else:
                        ext = "png"  # Default
                    
                    # Generate filename
                    safe_description = "".join(c if c.isalnum() else "_" for c in image_description[:30])
                    filename = f"{element_id}_{safe_description}.{ext}"
                    
                    logger.info(f"Uploading {len(image_bytes):,} bytes as {filename} to Parse")
                    
                    # Upload to Parse Server
                    file_upload_response = await client.post(
                        f"{parse_url}/parse/files/{filename}",
                        content=image_bytes,
                        headers={
                            **headers,
                            "Content-Type": content_type
                        }
                    )
                    
                    if file_upload_response.status_code in [200, 201]:
                        file_data = file_upload_response.json()
                        new_url = file_data.get("url")
                        
                        if new_url:
                            url_mappings[element_id] = {
                                "old_url": old_url,
                                "new_url": new_url
                            }
                            
                            # Replace URL in element
                            elem["image_url"] = new_url
                            
                            # Update content with new URL
                            elem["content"] = f"![{image_description}]({new_url})\n\n*{image_description}*"
                            
                            reuploaded_count += 1
                            logger.info(f"✅ Re-uploaded {element_id}: {new_url}")
                        else:
                            logger.warning(f"File upload succeeded but no URL returned for {element_id}")
                    else:
                        logger.error(f"Failed to upload image {element_id}: {file_upload_response.status_code} {file_upload_response.text}")
                
            except Exception as img_err:
                logger.error(f"Failed to re-upload image {element_id}: {img_err}")
                continue
        
        logger.info(f"Successfully re-uploaded {reuploaded_count}/{len(images_to_reupload)} {provider} images")
        
        # Step 4: Update stored extraction with permanent Parse URLs
        if reuploaded_count > 0:
            activity.heartbeat("Updating stored extraction with permanent Parse URLs")
            
            await store_extraction_result_in_post(
                post_id=post_id,
                structured_elements=structured_elements,  # Updated in-place with Parse URLs
                memory_requests=extraction_data.get("memory_requests", []),
                element_summary=extraction_data.get("element_summary", {}),
                decision=extraction_data.get("decision", "complex")
            )
            
            logger.info(f"✅ Updated Post {post_id} with {reuploaded_count} permanent Parse image URLs")
        
        return {
            "images_reuploaded": reuploaded_count,
            "total_candidates": len(images_to_reupload),
            "url_mappings": url_mappings,
            "post_id": post_id
        }
        
    except Exception as e:
        logger.error(f"Image re-upload from {provider} failed: {e}", exc_info=True)
        raise


@activity.defn
async def extract_and_upload_images_from_pdf_stored(
    pdf_url: str,
    post_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: Optional[str]
) -> Dict[str, Any]:
    """Extract images from PDF for stored extractions (fetches from Parse, extracts, updates back)
    
    This activity:
    1. Fetches structured_elements from Parse Server
    2. Downloads the PDF and extracts images using PyMuPDF
    3. Uploads images to Parse Server
    4. Updates the structured_elements back to Parse with image URLs
    
    Args:
        pdf_url: URL to the PDF file
        post_id: Parse Post ID containing the extraction results
        organization_id: Organization ID
        namespace_id: Namespace ID
        workspace_id: Workspace ID
    
    Returns:
        Dict with uploaded image URLs mapped to element IDs
    """
    
    try:
        activity.heartbeat(f"Extracting images from PDF for stored extraction in Post {post_id}")
        
        from services.memory_management import fetch_extraction_result_from_post, store_extraction_result_in_post
        
        logger.info(f"Fetching stored extraction from Post {post_id}")
        
        # Step 1: Fetch extraction results from Post
        extraction_data = await fetch_extraction_result_from_post(post_id)
        
        if not extraction_data:
            raise Exception(f"No extraction data found for Post {post_id}")
        
        structured_elements = extraction_data.get("structured_elements", [])
        
        if not structured_elements:
            logger.info("No structured elements found in stored extraction")
            return {
                "images_extracted": 0,
                "total_candidates": 0,
                "image_url_mappings": {},
                "post_id": post_id
            }
        
        logger.info(f"Fetched {len(structured_elements)} elements from stored extraction")
        
        # Step 2: Call the regular image extraction activity logic
        # (reuse the same logic but pass the fetched elements)
        result = await extract_and_upload_images_from_pdf(
            pdf_url=pdf_url,
            structured_elements=structured_elements,
            organization_id=organization_id,
            namespace_id=namespace_id,
            workspace_id=workspace_id,
            post_id=post_id
        )
        
        # Step 3: Update the stored extraction with the modified elements
        if result.get('images_extracted', 0) > 0:
            activity.heartbeat("Updating stored extraction with image URLs")
            
            await store_extraction_result_in_post(
                post_id=post_id,
                structured_elements=structured_elements,  # Already updated in-place by extract_and_upload_images_from_pdf
                memory_requests=extraction_data.get("memory_requests", []),
                element_summary=extraction_data.get("element_summary", {}),
                decision=extraction_data.get("decision", "complex")
            )
            
            logger.info(f"Updated stored extraction in Post {post_id} with {result.get('images_extracted')} image URLs")
        
        return result
        
    except Exception as e:
        logger.error(f"Image extraction from PDF (stored) failed: {e}", exc_info=True)
        raise


@activity.defn
async def store_batch_memories_in_parse_for_processing(
    memory_requests: List[Dict[str, Any]],
    organization_id: str,
    namespace_id: str,
    user_id: str,
    workspace_id: Optional[str],
    existing_post_id: Optional[str] = None,
    schema_specification: Optional[Dict[str, Any]] = None,
    source_url: Optional[str] = None  # PDF/document URL for sourceUrl field
) -> Dict[str, Any]:
    """
    Store batch memory requests in Parse Post for processing by batch memory workflow.
    
    This activity is called from DocumentProcessingWorkflow to store memories before
    triggering the full indexing pipeline via process_batch_memories_from_parse_reference.
    
    Args:
        memory_requests: List of memory request dictionaries from LLM generation
        organization_id: Organization ID
        namespace_id: Namespace ID  
        user_id: End user ID
        workspace_id: Workspace ID
        existing_post_id: Optional Post ID to update (reuse document Post)
        schema_specification: Optional schema specification dict with schema_id, graph_override
        source_url: Optional PDF/document URL to set as sourceUrl in memory metadata
        
    Returns:
        {"post_id": str} - The Post ID where memories are stored
    """
    try:
        activity.heartbeat("Storing batch memories in Parse for processing")
        
        from services.memory_management import store_batch_memories_in_parse
        
        # Extract schema_id from schema_specification for logging
        schema_id = schema_specification.get("schema_id") if schema_specification else None
        logger.info(f"Storing {len(memory_requests)} memories in Parse (post_id: {existing_post_id}, schema_id: {schema_id}, source_url: {source_url})")
        
        # Inject schema_id, post pointer, and sourceUrl into each memory's metadata for downstream processing
        # Note: We inject schema_id as a string because MemoryMetadata
        # expects customMetadata values to be primitives (str, int, float, bool, list[str])
        # The full schema_specification will be passed separately via the batch workflow
        for mem_dict in memory_requests:
            if not isinstance(mem_dict, dict):
                continue
            if "metadata" not in mem_dict:
                mem_dict["metadata"] = {}
            if "customMetadata" not in mem_dict["metadata"]:
                mem_dict["metadata"]["customMetadata"] = {}
            
            # Inject schema_id as a string for backward compatibility
            if schema_specification:
                schema_id = schema_specification.get("schema_id")
                if schema_id:
                    mem_dict["metadata"]["customMetadata"]["schema_id"] = schema_id
                    logger.info(f"✅ Injected schema_id into memory metadata: schema_id={schema_id}")
            
            # Inject post pointer (document Post ID) - this links memories to the document
            # store_generic_memory_item looks for 'pageId' in metadata to create the post pointer
            if existing_post_id:
                mem_dict["metadata"]["pageId"] = existing_post_id
                # ALSO set post_objectId in customMetadata for easy querying
                mem_dict["metadata"]["customMetadata"]["post_objectId"] = existing_post_id
                logger.info(f"✅ Injected post pointer into memory metadata: pageId={existing_post_id}, post_objectId={existing_post_id}")
            
            # Inject sourceUrl (PDF/document URL) - this is the source document URL
            if source_url:
                mem_dict["metadata"]["sourceUrl"] = source_url
                logger.info(f"✅ Injected sourceUrl into memory metadata: sourceUrl={source_url}")
        
        # Store in Parse (will create or update Post)
        # Extract schema_id for batch metadata
        schema_id_for_metadata = None
        if schema_specification:
            # schema_specification now uses "schema_id" (string) not "schema_ids" (list)
            schema_id_for_metadata = schema_specification.get("schema_id")
        
        post_id = await store_batch_memories_in_parse(
            memories=memory_requests,
            organization_id=organization_id,
            namespace_id=namespace_id,
            user_id=user_id,
            workspace_id=workspace_id,
            batch_metadata={
                "source": "document_processing_workflow",
                "schema_id": schema_id_for_metadata
            },
            existing_post_id=existing_post_id  # Reuse document Post if provided
        )
        
        logger.info(f"Successfully stored {len(memory_requests)} memories in Post {post_id}")
        
        return {"post_id": post_id}
        
    except Exception as e:
        logger.error(f"Failed to store batch memories in Parse: {e}", exc_info=True)
        raise