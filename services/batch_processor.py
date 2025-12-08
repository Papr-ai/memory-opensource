"""
Batch Processing Service

Handles batch memory operations with feature flag support for Temporal.
"""

from typing import List, Dict, Any
import os
from config import get_features
from services.logger_singleton import LoggerSingleton
from models.parse_server import BatchMemoryResponse, BatchMemoryError
from models.memory_models import BatchMemoryRequest, OptimizedAuthResponse

logger = LoggerSingleton.get_logger(__name__)


async def validate_batch_size(batch_size: int) -> tuple[bool, str, int]:
    """
    Validate batch size against edition limits.
    
    Args:
        batch_size: Number of items in batch
        
    Returns:
        (is_valid, error_message, max_allowed)
    """
    features = get_features()
    max_batch = features.config.get("batch_processing", {}).get("max_batch_size", 50)
    
    if batch_size > max_batch:
        message = features.config.get("messaging", {}).get("batch_limit_exceeded", "")
        return False, message, max_batch
    
    return True, "", max_batch


async def should_use_temporal(batch_size: int) -> bool:
    """
    Determine if batch should use Temporal workflows.
    
    Args:
        batch_size: Number of items in batch
        
    Returns:
        True if should use Temporal, False for background tasks
    """
    features = get_features()
    # Detailed logging for decision path
    from services.logger_singleton import LoggerSingleton
    _log = LoggerSingleton.get_logger(__name__)

    if not features.is_enabled("temporal"):
        _log.info("Temporal disabled by feature flag or edition; skipping Temporal")
        return False

    threshold = features.config.get("temporal", {}).get("temporal_threshold", 2)
    use_temporal = batch_size > threshold
    _log.info(f"Temporal decision: batch_size={batch_size}, threshold={threshold}, use_temporal={use_temporal}")
    return use_temporal


async def process_batch_with_temporal(
    batch_request: BatchMemoryRequest,  # Validated BatchMemoryRequest
    auth_response: OptimizedAuthResponse,  # Validated OptimizedAuthResponse
    api_key: str,  # Real API key
    webhook_url: str = None,
    webhook_secret: str = None
) -> BatchMemoryResponse:
    """
    Process batch using Temporal with real authentication data (cloud-only, guaranteed delivery).

    Args:
        batch_request: Validated BatchMemoryRequest with memories to process
        auth_response: Validated OptimizedAuthResponse with authentication data
        api_key: Real API key for internal authentication
        webhook_url: Optional webhook for completion notification
        webhook_secret: Optional webhook secret for validation

    Returns:
        BatchMemoryResponse with workflow ID
    """
    try:
        from cloud_plugins.temporal.client import get_temporal_client
        from cloud_plugins.temporal.workflows import (
            process_batch_workflow,
            process_batch_workflow_from_post,
        )
        from services.memory_management import store_batch_memories_in_parse
        import uuid

        # Defensive: coerce to correct types if callers passed dicts by mistake
        if not isinstance(batch_request, BatchMemoryRequest):
            try:
                batch_request = BatchMemoryRequest(**(batch_request.model_dump() if hasattr(batch_request, "model_dump") else dict(batch_request)))  # type: ignore[arg-type]
            except Exception as e:
                raise TypeError(f"batch_request must be BatchMemoryRequest, got {type(batch_request)}: {e}")
        if not isinstance(auth_response, OptimizedAuthResponse):
            try:
                auth_response = OptimizedAuthResponse(**(auth_response.model_dump() if hasattr(auth_response, "model_dump") else dict(auth_response)))  # type: ignore[arg-type]
            except Exception as e:
                raise TypeError(f"auth_response must be OptimizedAuthResponse, got {type(auth_response)}: {e}")

        # Generate batch ID
        batch_id = str(uuid.uuid4())

        # Get Temporal client
        client = await get_temporal_client()

        # Use the full multi-stage ProcessBatchMemoryWorkflow
        # This runs add_memory_quick → idx_generate_graph_schema → update_relationships → idx_update_metrics
        from cloud_plugins.temporal.workflows import process_batch_workflow

        # Start the proper multi-stage workflow with full auth data
        workflow_id = await process_batch_workflow(
            client=client,
            batch_id=batch_id,
            batch_request=batch_request,
            auth_response=auth_response,
            api_key=api_key,
            webhook_url=webhook_url,
            webhook_secret=webhook_secret
        )

        logger.info(f"Started Temporal batch processing: {workflow_id}")

        # Return immediate response
        return BatchMemoryResponse.success(
            successful=[],  # Will be populated when workflow completes
            total_processed=0,
            total_successful=0,
            total_failed=0,
            details={
                "status": "processing",
                "workflow_id": workflow_id,
                "batch_id": batch_id,
                "message": "Batch processing started. You will receive a webhook notification when complete.",
                "webhook_url": webhook_url,
                "estimated_completion_minutes": len(batch_request.memories) // 10  # Rough estimate
            }
        )
        
    except ImportError:
        logger.error("Temporal plugin not available but temporal feature enabled")
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(index=-1, error="Temporal service unavailable")],
            code=503,
            error="Batch processing service unavailable"
        )
    except Exception as e:
        logger.error(f"Error starting Temporal workflow: {e}")
        return BatchMemoryResponse.failure(
            errors=[BatchMemoryError(index=-1, error=str(e))],
            code=500,
            error="Failed to start batch processing"
        )

