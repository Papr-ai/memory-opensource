"""
Batch Memory Processing Workflow

Durable workflow for processing large batches of memories.
Guarantees completion even through server restarts.
"""

from datetime import timedelta
import asyncio
from typing import List, Dict, Any, Optional
from temporalio import workflow
from temporalio.common import RetryPolicy
from services.logger_singleton import LoggerSingleton
from models.temporal_models import BatchWorkflowData

logger = LoggerSingleton.get_logger(__name__)


@workflow.defn
class ProcessBatchMemoryWorkflow:
    """
    Workflow for processing batch memory creation.
    
    Features:
    - Automatic retries on failures
    - Progress tracking
    - Webhook notification when complete
    - Survives server restarts
    """
    
    @workflow.run
    async def run(self, batch_data: BatchWorkflowData) -> Dict[str, Any]:
        """
        Process a batch of memories.
        
        Args:
            batch_data: BatchWorkflowData containing:
                - batch_id: Unique batch identifier
                - batch_request: Validated BatchMemoryRequest with memories to process
                - auth_response: OptimizedAuthResponse with authentication data
                - api_key: API key for internal authentication
                - webhook_url: Optional webhook URL for completion notification
                - webhook_secret: Optional webhook secret for authentication
                - schema_specification: Optional schema specification for graph extraction
        
        Returns:
            {
                "status": "completed" | "failed",
                "total_processed": int,
                "successful": int,
                "failed": int,
                "errors": List[dict]
            }
        """
        batch_id = batch_data.batch_id
        batch_request = batch_data.batch_request
        memories = batch_request.memories

        logger.info(f"Starting batch workflow {batch_id} with {len(memories)} memories")
        
        # Process each memory as a set of stage activities for per-stage visibility
        total = len(memories)
        successful = 0
        errors: list[dict] = []

        # Helper for common retry policy
        stage_retry = RetryPolicy(
            maximum_attempts=5,
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=30),
            backoff_coefficient=2.0,
        )

        # Build a minimal per-memory batch_data to avoid exceeding gRPC payload limits.
        # Each activity only needs: batch_id, api_key, auth_response, and a batch_request
        # containing ONE memory at index 0. We pass index=0 to activities accordingly.
        def build_shrunk_batch_data(idx: int) -> Dict[str, Any]:
            # Extract single memory as plain dict
            mem = memories[idx]
            try:
                mem_dict = mem.model_dump(exclude_none=True) if hasattr(mem, "model_dump") else dict(mem)
            except Exception:
                mem_dict = {
                    "content": getattr(mem, "content", None),
                    "type": getattr(mem, "type", None),
                    "metadata": getattr(mem, "metadata", None),
                    "title": getattr(mem, "title", None),
                    "external_user_id": getattr(mem, "external_user_id", None),
                }

            # Slim auth response
            ar = batch_data.auth_response
            try:
                auth_response = ar.model_dump(exclude_none=True) if hasattr(ar, "model_dump") else dict(ar)
            except Exception:
                auth_response = {
                    "developer_id": getattr(ar, "developer_id", None),
                    "end_user_id": getattr(ar, "end_user_id", None),
                    "workspace_id": getattr(ar, "workspace_id", None),
                    "organization_id": getattr(ar, "organization_id", None),
                    "namespace_id": getattr(ar, "namespace_id", None),
                    "is_qwen_route": getattr(ar, "is_qwen_route", None),
                    "session_token": getattr(ar, "session_token", None),
                }

            # Slim batch_request with just one memory
            br = batch_data.batch_request
            try:
                user_id = getattr(br, "user_id", None)
                external_user_id = getattr(br, "external_user_id", None)
                organization_id = getattr(br, "organization_id", None)
                namespace_id = getattr(br, "namespace_id", None)
            except Exception:
                user_id = None
                external_user_id = None
                organization_id = None
                namespace_id = None

            shrunk = {
                "batch_id": batch_id,
                "api_key": batch_data.api_key,
                # keep flag explicit to avoid default disagreements
                "legacy_route": True,
                "auth_response": auth_response,
                "batch_request": {
                    "user_id": user_id,
                    "external_user_id": external_user_id,
                    "organization_id": organization_id,
                    "namespace_id": namespace_id,
                    # Critically: send only ONE memory to activities
                    "memories": [mem_dict],
                    # Keep minimal other fields to avoid large payload
                },
                "schema_specification": batch_data.schema_specification,  # Pass schema specification for graph extraction
            }
            return shrunk

        # Stage 1: Process memories using TRUE batch activities to avoid MongoDB connection pool exhaustion.
        #
        # DEFAULT BEHAVIOR (NEW):
        #     All fresh workflows should run through batch_add_memory_quick even when triggered directly.
        #
        # BACKWARD COMPATIBILITY:
        #     Workflows whose histories predate this rollout must stay on add_memory_quick to avoid nondeterministic
        #     replays.
        #
        # PATCH STRATEGY:
        #     workflow.patched("default-batch-processing") returns True for binaries that include this change.
        #     Older histories (started before the patch existed) will see False and automatically stay on the legacy path.
        default_batch_patch = workflow.patched("default-batch-processing")
        is_child_workflow = workflow.info().parent is not None

        # Removed force-legacy-individual-memory patch - default-batch-processing is now the standard
        if default_batch_patch:
            use_batch_processing = True
            logger.info(
                f"✅ default-batch-processing patch active: using batch_add_memory_quick for {total} memories"
            )
        elif is_child_workflow:
            # Child workflows already run under unique workflow IDs, so we can safely default to batch even if the
            # history predates the patch (they won't collide with older runs).
            use_batch_processing = True
            logger.info(
                f"✅ Child workflow detected (pre-patch replay): using batch_add_memory_quick for {total} memories"
            )
        else:
            # History predates the default-batch-processing patch and there's no parent workflow.
            # Stay on the legacy path for determinism.
            use_batch_processing = False
            logger.warning(
                f"⚠️  Workflow history predates default-batch-processing patch: using add_memory_quick for backward compatibility ({total} memories)"
            )
        
        quick_results = [None] * total  # Pre-allocate results list
        
        if use_batch_processing:
            # NEW CODE PATH: Batch processing with parallel execution
            BATCH_SIZE = 20  # Process 20 memories per activity
            logger.info(f"📦 Processing {total} memories in batches of {BATCH_SIZE} (PARALLEL execution using batch_add_memory_quick)")
        else:
            # LEGACY CODE PATH: Individual processing (only if explicitly patched)
            logger.warning(f"⚠️  Using legacy individual processing for {total} memories")
        
        if use_batch_processing:
            # Prepare ALL batch activity tasks upfront
            batch_activity_tasks = []
            batch_ranges = []  # Track which indices each batch covers
            
            for batch_start in range(0, total, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total)
                batch_size = batch_end - batch_start
                
                logger.info(f"Preparing parallel batch {batch_start}-{batch_end} ({batch_size} memories)")
                
                # Prepare batch data for all memories in this batch
                batch_data_list = [
                    {"batch_data": build_shrunk_batch_data(idx), "index": 0}
                    for idx in range(batch_start, batch_end)
                ]
                
                # Create activity task (don't await yet - we'll gather them)
                task = workflow.start_activity(
                    "batch_add_memory_quick",  # ✅ TRUE BATCH ACTIVITY
                    batch_data_list,
                    start_to_close_timeout=timedelta(minutes=10),  # Increased for batch DB operations
                    retry_policy=stage_retry,
                )
                
                batch_activity_tasks.append(task)
                batch_ranges.append((batch_start, batch_end))
            
            # Execute ALL batch activities in PARALLEL 🚀
            logger.info(f"🚀 Executing {len(batch_activity_tasks)} batch activities in PARALLEL using batch_add_memory_quick")
            batch_results = await asyncio.gather(*batch_activity_tasks, return_exceptions=True)
            
            # Process results from all batches
            for batch_idx, (batch_result, (batch_start, batch_end)) in enumerate(zip(batch_results, batch_ranges)):
                batch_size = batch_end - batch_start
                if isinstance(batch_result, Exception):
                    logger.error(f"❌ Batch {batch_start}-{batch_end} failed: {batch_result}")
                    # Mark all memories in this batch as failed
                    for i in range(batch_size):
                        quick_results[batch_start + i] = batch_result
                else:
                    # batch_result is List[Dict], one per memory
                    for i, result in enumerate(batch_result):
                        quick_results[batch_start + i] = result
                    logger.info(f"✅ Batch {batch_start}-{batch_end} completed")
        else:
            # LEGACY CODE PATH: Individual processing (only for workflows that predate default-batch-processing patch)
            logger.info(f"🚀 Executing {total} individual activities in PARALLEL using add_memory_quick")
            
            quick_handles = []
            for idx in range(total):
                quick_handles.append(
                    workflow.start_activity(
                        "add_memory_quick",  # Legacy individual activity
                        {"batch_data": build_shrunk_batch_data(idx), "index": 0},
                        start_to_close_timeout=timedelta(minutes=5),
                        retry_policy=stage_retry,
                    )
                )
            quick_results = await asyncio.gather(*quick_handles, return_exceptions=True)

        # Determine which items can proceed past quick stage
        valid_indices: list[int] = []
        for idx in range(total):
            quick = quick_results[idx]
            if isinstance(quick, Exception):
                errors.append({"index": idx, "error": str(quick)})
                continue
            if not isinstance(quick, dict) or not quick.get("memory_id"):
                errors.append({"index": idx, "error": "quick add returned no memory_id; skipping indexing pipeline"})
                continue
            valid_indices.append(idx)

        # Stage 2: Generate/store schema in parallel for all valid items
        schema_handles = []
        for idx in valid_indices:
            schema_handles.append(
                workflow.start_activity(
                    "idx_generate_graph_schema",
                    {"batch_data": build_shrunk_batch_data(idx), "index": 0, "quick": quick_results[idx]},
                    start_to_close_timeout=timedelta(minutes=60),  # Increased for LLM processing (can be slow for large batches)
                    retry_policy=stage_retry,
                )
            )
        schema_results = await asyncio.gather(*schema_handles, return_exceptions=True)

        # Collect relationships and metrics per index for next stages
        idx_to_relationships: dict[int, list[dict]] = {}
        idx_to_metrics: dict[int, dict] = {}
        next_indices_after_schema: list[int] = []
        for i, idx in enumerate(valid_indices):
            res = schema_results[i]
            if isinstance(res, Exception):
                errors.append({"index": idx, "error": f"idx_generate_graph_schema failed: {res}"})
                continue
            if isinstance(res, dict):
                idx_to_relationships[idx] = res.get("relationships_json") or []
                idx_to_metrics[idx] = res.get("schema_metrics") or {}
            next_indices_after_schema.append(idx)

        # Stage 3: Update relationships in parallel
        rel_handles = []
        for idx in next_indices_after_schema:
            rel_handles.append(
                workflow.start_activity(
                    "update_relationships",
                    {
                        "batch_data": build_shrunk_batch_data(idx),
                        "index": 0,
                        "quick": quick_results[idx],
                        "relationships_json": idx_to_relationships.get(idx) or [],
                    },
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=stage_retry,
                )
            )
        rel_results = await asyncio.gather(*rel_handles, return_exceptions=True)
        # Record relationship-stage errors but proceed
        for i, idx in enumerate(next_indices_after_schema):
            rr = rel_results[i]
            if isinstance(rr, Exception):
                errors.append({"index": idx, "error": f"update_relationships failed: {rr}"})

        # Stage 4: Persist metrics in parallel
        metrics_handles = []
        for idx in next_indices_after_schema:
            metrics_handles.append(
                workflow.start_activity(
                    "idx_update_metrics",
                    {
                        "batch_data": build_shrunk_batch_data(idx),
                        "index": 0,
                        "quick": quick_results[idx],
                        "metrics": idx_to_metrics.get(idx) or {},
                    },
                    start_to_close_timeout=timedelta(minutes=2),
                    retry_policy=stage_retry,
                )
            )
        metrics_results = await asyncio.gather(*metrics_handles, return_exceptions=True)
        for i, idx in enumerate(next_indices_after_schema):
            mr = metrics_results[i]
            if isinstance(mr, Exception):
                errors.append({"index": idx, "error": f"idx_update_metrics failed: {mr}"})

        # Compute successes/failures: only count items that produced a valid memory_id
        successful = len([idx for idx in range(total) if not isinstance(quick_results[idx], Exception) and isinstance(quick_results[idx], dict) and quick_results[idx].get("memory_id")])

        # Collect memory_object_ids for linking to Post
        memory_object_ids = []
        for idx in range(total):
            quick = quick_results[idx]
            if isinstance(quick, dict) and quick.get("object_id"):
                memory_object_ids.append(quick["object_id"])

        results = {
            "status": "completed" if successful == total else ("failed" if successful == 0 else "partial"),
            "total_processed": total,
            "successful": successful,
            "failed": total - successful,
            "errors": errors,
            "memory_object_ids": memory_object_ids,  # Add memory_object_ids for linking
        }
        
        # Send webhook notification if configured
        if batch_data.webhook_url:
            await workflow.execute_activity(
                "send_webhook_notification",
                {
                    "batch_id": batch_id,
                    "webhook_url": batch_data.webhook_url,
                    "webhook_secret": batch_data.webhook_secret,
                    "results": results
                },
                start_to_close_timeout=timedelta(seconds=30),
                retry_policy=RetryPolicy(
                    maximum_attempts=5,
                    initial_interval=timedelta(seconds=2),
                    maximum_interval=timedelta(seconds=10)
                )
            )
        
        logger.info(f"Completed batch workflow {batch_id}: {results['successful']}/{results['total_processed']} successful")
        
        return results


# Helper function to start the workflow
async def process_batch_workflow(
    client: Any,  # Temporal client
    batch_id: str,
    batch_request: Any,  # Original BatchMemoryRequest from models.memory_models
    auth_response: Any,  # Original OptimizedAuthResponse from models.memory_models
    api_key: str | None,  # Real API key (optional)
    webhook_url: str = None,
    webhook_secret: str = None
) -> str:
    """
    Start a batch processing workflow with real authentication data.

    Returns:
        workflow_id: ID to track the workflow
    """
    from config import get_features

    features = get_features()
    task_queue = features.config.get("temporal", {}).get("task_queue", "memory-processing")

    workflow_id = f"batch-{batch_id}"

    # Import the temporal-safe models here to avoid sandbox issues
    from models.temporal_models import (
        BatchMemoryRequest as TemporalBatchMemoryRequest,
        OptimizedAuthResponse as TemporalOptimizedAuthResponse,
        AddMemoryRequest as TemporalAddMemoryRequest
    )

    # Convert original models to temporal-safe versions
    temporal_memories = []
    for memory in batch_request.memories:
        # Normalize to dict to safely access optional/variant fields
        try:
            mem_dict = memory.model_dump() if hasattr(memory, "model_dump") else dict(memory)
        except Exception:
            mem_dict = {
                "content": getattr(memory, "content", None),
                "type": getattr(memory, "type", None),
                "metadata": getattr(memory, "metadata", None),
            }

        content = mem_dict.get("content")
        m_type = mem_dict.get("type")
        if hasattr(m_type, "value"):
            m_type = m_type.value  # Convert enum to string

        metadata = mem_dict.get("metadata")
        if hasattr(metadata, "model_dump"):
            metadata = metadata.model_dump(exclude_none=True)

        # Title is optional; derive from explicit field or metadata if present
        title = mem_dict.get("title")
        if not title and isinstance(metadata, dict):
            title = metadata.get("title")

        # External user may be provided via memory field or inside metadata
        external_user_id = mem_dict.get("external_user_id")
        if not external_user_id and isinstance(metadata, dict):
            external_user_id = metadata.get("external_user_id")

        temporal_memory = TemporalAddMemoryRequest(
            content=content,
            title=title,
            type=m_type or "text",
            external_user_id=external_user_id,
            metadata=metadata,
        )
        temporal_memories.append(temporal_memory)

    # Extract schema specification from batch request's memory_policy or graph_generation
    schema_specification = None
    schema_id = None

    # NEW: Check for memory_policy first (unified API)
    if hasattr(batch_request, 'memory_policy') and batch_request.memory_policy:
        mp = batch_request.memory_policy
        mp_dict = mp.model_dump() if hasattr(mp, 'model_dump') else mp

        schema_id = mp_dict.get('schema_id')

        # Build schema_specification dict for workflow
        schema_specification = {
            "schema_id": schema_id,
            "mode": mp_dict.get('mode', 'auto'),
            "node_constraints": mp_dict.get('node_constraints'),
            "consent": mp_dict.get('consent', 'implicit'),
            "risk": mp_dict.get('risk', 'none'),
            "acl": mp_dict.get('acl'),
        }

        # Handle manual mode - extract nodes/relationships into graph_override
        # Note: 'structured' is accepted as deprecated alias for 'manual'
        mode = mp_dict.get('mode', 'auto')
        if mode in ('manual', 'structured'):
            nodes = mp_dict.get('nodes')
            relationships = mp_dict.get('relationships')
            if nodes:
                schema_specification['graph_override'] = {
                    'nodes': nodes,
                    'relationships': relationships or []
                }

        logger.info(f"📋 Batch workflow: Using memory_policy - mode={mode}, schema_id={schema_id}")

    # LEGACY: Fall back to graph_generation if memory_policy not provided
    elif hasattr(batch_request, 'graph_generation') and batch_request.graph_generation:
        from models.temporal_models import flatten_graph_generation_for_temporal
        flattened = flatten_graph_generation_for_temporal(batch_request.graph_generation)
        schema_id = flattened.get('schema_id')
        schema_specification = flattened
        logger.info(f"📋 Batch workflow: Using legacy graph_generation - schema_id={schema_id}")

    temporal_batch_request = TemporalBatchMemoryRequest(
        user_id=batch_request.user_id,
        external_user_id=batch_request.external_user_id,
        organization_id=batch_request.organization_id,
        namespace_id=batch_request.namespace_id,
        schema_id=schema_id,
        memories=temporal_memories,
        batch_size=batch_request.batch_size,
        webhook_url=batch_request.webhook_url,
        webhook_secret=batch_request.webhook_secret
    )

    temporal_auth_response = TemporalOptimizedAuthResponse(
        developer_id=auth_response.developer_id,
        end_user_id=auth_response.end_user_id,
        workspace_id=auth_response.workspace_id,
        organization_id=auth_response.organization_id,
        namespace_id=auth_response.namespace_id,
        is_qwen_route=auth_response.is_qwen_route
    )

    # Create properly typed workflow data
    workflow_data = BatchWorkflowData(
        batch_id=batch_id,
        batch_request=temporal_batch_request,
        auth_response=temporal_auth_response,
        api_key=api_key,
        webhook_url=webhook_url,
        webhook_secret=webhook_secret,
        schema_specification=schema_specification  # Pass schema specification for graph extraction
    )

    # Note: When using Worker Versioning, workflows started via client.start_workflow
    # are automatically routed to the default build ID configured on the task queue.
    # This workflow will use whatever build ID is set as default via:
    # tctl task-queue version-set update --add-build-ids v0.2.0+batch-default.20251117
    handle = await client.start_workflow(
        ProcessBatchMemoryWorkflow.run,
        workflow_data,
        id=workflow_id,
        task_queue=task_queue
    )

    logger.info(f"Started Temporal workflow: {workflow_id} (will use default build ID on task queue)")

    return workflow_id



@workflow.defn
class ProcessBatchMemoryFromPostWorkflow:
    """
    Reference-based workflow: fetch memories from a Parse Post (compressed file)
    inside an activity, then run the full multi-stage processing pipeline via ProcessBatchMemoryWorkflow.
    
    This workflow:
    1. Fetches memories from Parse Post (avoiding gRPC payload limits)
    2. Constructs BatchWorkflowData with fetched memories
    3. Delegates to ProcessBatchMemoryWorkflow to run the full multi-stage pipeline:
       - Stage 1: add_memory_quick (parallel quick adds)
       - Stage 2: idx_generate_graph_schema (parallel LLM schema generation with custom schema enforcement)
       - Stage 3: update_relationships (parallel Neo4j relationship building)
       - Stage 4: idx_update_metrics (parallel metrics updates)
    """

    @workflow.run
    async def run(self, ref_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            ref_data: {
                "batch_id": str,
                "post_id": str,
                "organization_id": str,
                "namespace_id": str,
                "user_id": str,
                "workspace_id": Optional[str],
                "schema_specification": Optional[Dict[str, Any]]  # SchemaSpecificationMixin data for graph enforcement
            }
        Returns summary with counts and errors.
        """
        logger.info(f"🎬 ProcessBatchMemoryFromPostWorkflow.run() INVOKED with ref_data keys: {list(ref_data.keys())}")
        
        batch_id = ref_data.get("batch_id")
        post_id = ref_data.get("post_id")
        organization_id = ref_data.get("organization_id")
        namespace_id = ref_data.get("namespace_id")
        user_id = ref_data.get("user_id")
        workspace_id = ref_data.get("workspace_id")
        schema_specification = ref_data.get("schema_specification")
        
        # Extract schema_id from schema_specification
        schema_id = None
        if schema_specification and isinstance(schema_specification, dict):
            schema_id = schema_specification.get("schema_id")
        logger.info(f"Starting reference batch workflow {batch_id} for Post {post_id} with schema_specification={schema_specification}, schema_id={schema_id}")

        retry = RetryPolicy(
            maximum_attempts=5,
            initial_interval=timedelta(seconds=2),
            maximum_interval=timedelta(seconds=30),
            backoff_coefficient=2.0,
        )

        # Step 1: Fetch memories from Parse Post (this returns AddMemoryRequest list)
        fetch_result = await workflow.execute_activity(
            "fetch_batch_memories_from_post",
            args=[post_id, organization_id, namespace_id, user_id, workspace_id, schema_specification],
            start_to_close_timeout=timedelta(minutes=5),
            retry_policy=retry,
        )
        
        memories = fetch_result.get("memories", [])
        if not memories:
            logger.warning(f"No memories fetched from Post {post_id}")
            return {
                "status": "failed",
                "total_processed": 0,
                "successful": 0,
                "failed": 0,
                "errors": [{"error": "No memories found in Post"}]
            }
        
        logger.info(f"Fetched {len(memories)} memories from Post {post_id}, starting multi-stage processing pipeline")
        
        # Step 2: Build BatchWorkflowData for ProcessBatchMemoryWorkflow
        from models.temporal_models import (
            BatchWorkflowData,
            BatchMemoryRequest as TemporalBatchMemoryRequest,
            OptimizedAuthResponse as TemporalOptimizedAuthResponse,
            AddMemoryRequest as TemporalAddMemoryRequest
        )
        
        # Convert memory dicts to TemporalAddMemoryRequest objects
        temporal_memories = []
        for mem_dict in memories:
            try:
                temporal_memory = TemporalAddMemoryRequest(**mem_dict)
                temporal_memories.append(temporal_memory)
            except Exception as e:
                logger.error(f"Failed to convert memory to TemporalAddMemoryRequest: {e}")
                continue
        
        if not temporal_memories:
            logger.error(f"Failed to convert any memories to TemporalAddMemoryRequest")
            return {
                "status": "failed",
                "total_processed": 0,
                "successful": 0,
                "failed": len(memories),
                "errors": [{"error": "Failed to convert memories to TemporalAddMemoryRequest"}]
            }
        
        # Extract schema fields from schema_specification
        # schema_specification is a flattened dict that may contain:
        # - Legacy format: schema_id, graph_override directly
        # - New format: mode, auto{schema_id}, manual{nodes, relationships}
        schema_id = None
        if schema_specification and isinstance(schema_specification, dict):
            schema_id = schema_specification.get("schema_id")

        # Create temporal batch request
        temporal_batch_request = TemporalBatchMemoryRequest(
            user_id=user_id,
            external_user_id=user_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            schema_id=schema_id,
            memories=temporal_memories,
            batch_size=len(temporal_memories)
        )
        
        # Create temporal auth response
        temporal_auth_response = TemporalOptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_qwen_route=False
        )
        
        # Create workflow data
        # Note: schema_specification should be passed as a dict, not a SchemaSpecification object
        # because BatchWorkflowData.schema_specification is typed as Optional[Dict[str, Any]]
        workflow_data = BatchWorkflowData(
            batch_id=batch_id,
            batch_request=temporal_batch_request,
            auth_response=temporal_auth_response,
            api_key="temporal_internal",
            webhook_url=None,
            webhook_secret=None,
            schema_specification=schema_specification  # Pass as dict directly
        )
        
        # Step 3: Start ProcessBatchMemoryWorkflow as child workflow to run the full multi-stage pipeline
        logger.info(f"Starting child workflow ProcessBatchMemoryWorkflow for {len(temporal_memories)} memories")
        
        try:
            # Log workflow_data structure for debugging
            logger.info(f"🔍 DEBUG: workflow_data keys: {list(workflow_data.keys()) if isinstance(workflow_data, dict) else 'not a dict'}")
            logger.info(f"🔍 DEBUG: workflow_data.batch_request.memories count: {len(workflow_data.batch_request.memories) if hasattr(workflow_data, 'batch_request') and hasattr(workflow_data.batch_request, 'memories') else 'N/A'}")
            
            # Use a unique workflow ID to prevent replay of old workflows
            # Combine batch_id with parent run_id to force new workflow execution instead of replaying old history
            parent_run_id = workflow.info().run_id
            unique_workflow_id = f"{batch_id}-processing-{parent_run_id[:8]}"  # Use first 8 chars of run_id for uniqueness
            
            logger.info(f"🔍 Starting child workflow with unique ID: {unique_workflow_id}")
            
            # Use versioned task queue to match the parent workflow
            memory_task_queue = "memory-processing"  # v2 for batch-default version
            
            batch_workflow_handle = await workflow.start_child_workflow(
                ProcessBatchMemoryWorkflow.run,
                args=[workflow_data],
                id=unique_workflow_id,
                task_queue=memory_task_queue,
                task_timeout=timedelta(minutes=30)  # Increased for large file processing
            )
            
            logger.info(f"✅ Child workflow started successfully, waiting for completion...")
            
            # Wait for child workflow to complete
            results = await batch_workflow_handle
            
        except Exception as child_error:
            logger.error(f"❌ CRITICAL: Failed to start or execute child workflow: {type(child_error).__name__}: {child_error}", exc_info=True)
            # Return failure result instead of crashing
            return {
                "status": "failed",
                "total_processed": len(temporal_memories),
                "successful": 0,
                "failed": len(temporal_memories),
                "errors": [{"error": f"Child workflow failed: {child_error}"}]
            }
        
        logger.info(f"Completed reference batch workflow {batch_id}: {results.get('successful', 0)}/{results.get('total_processed', 0)} successful")
        
        # Step 4: Link created memories to the Post (for document processing workflows)
        # Use memory_object_ids from results instead of searching
        memory_object_ids = results.get("memory_object_ids", [])
            
        if memory_object_ids and post_id:
            logger.info(f"Linking {len(memory_object_ids)} memories from batch {batch_id} to Post {post_id}")
            try:
                link_result = await workflow.execute_activity(
                    "link_batch_memories_to_post",
                    args=[memory_object_ids, post_id, user_id, organization_id, namespace_id, workspace_id],
                    start_to_close_timeout=timedelta(minutes=5),
                    retry_policy=retry
                )
                logger.info(f"Linked {link_result.get('linked_count', 0)} memories to Post {post_id}")
            except Exception as link_err:
                logger.error(f"Failed to link memories to Post (non-fatal): {link_err}")
                # Don't fail the workflow if linking fails
        else:
            logger.warning(f"No memory_object_ids to link to Post {post_id}")
        
        return results


async def process_batch_workflow_from_post(
    client: Any,
    batch_id: str,
    post_id: str,
    organization_id: str,
    namespace_id: str,
    user_id: str,
    workspace_id: Optional[str] = None,
) -> str:
    """Helper to start the reference-based workflow with a small payload."""
    from config import get_features

    features = get_features()
    task_queue = features.config.get("temporal", {}).get("task_queue", "memory-processing")

    workflow_id = f"batch-ref-{batch_id}"

    # Note: When using Worker Versioning, workflows started via client.start_workflow
    # are automatically routed to the default build ID configured on the task queue.
    handle = await client.start_workflow(
        ProcessBatchMemoryFromPostWorkflow.run,
        {
            "batch_id": batch_id,
            "post_id": post_id,
            "organization_id": organization_id,
            "namespace_id": namespace_id,
            "user_id": user_id,
            "workspace_id": workspace_id,
        },
        id=workflow_id,
        task_queue=task_queue,
    )

    logger.info(f"Started Temporal reference workflow: {workflow_id} (will use default build ID on task queue)")
    return workflow_id


@workflow.defn
class ProcessBatchMemoryFromRequestWorkflow:
    """
    Simplified workflow using BatchMemoryRequest for all batch sizes.

    This workflow accepts only a BatchMemoryRequest objectId reference,
    eliminating gRPC payload size concerns completely. The activity fetches
    the batch data from Parse and processes all memories.
    """

    @workflow.run
    async def run(self, request_ref: Dict[str, str]) -> Dict[str, Any]:
        """
        Process batch memories by fetching data from BatchMemoryRequest.

        Args:
            request_ref: {
                "request_id": "abc123",  # BatchMemoryRequest objectId
                "batch_id": "batch-uuid"
            }

        Returns:
            {
                "status": "completed" | "partial_failure" | "failed",
                "total": int,
                "successful": int,
                "failed": int,
                "errors": List[Dict]
            }
        """
        request_id = request_ref["request_id"]
        batch_id = request_ref["batch_id"]

        workflow.logger.info(
            f"Starting batch processing for request {request_id}, batch {batch_id}"
        )

        # Single activity call handles everything
        # This is much simpler than the per-memory activity approach
        retry = RetryPolicy(
            maximum_attempts=3,
            initial_interval=timedelta(seconds=2),
            backoff_coefficient=2.0,
            maximum_interval=timedelta(minutes=5)
        )

        result = await workflow.execute_activity(
            "fetch_and_process_batch_request",
            args=[request_id, batch_id],
            start_to_close_timeout=timedelta(hours=1),
            retry_policy=retry
        )

        workflow.logger.info(
            f"Completed batch {batch_id}: {result['successful']}/{result['total']} successful"
        )

        return result


async def process_batch_workflow_from_request(
    client: Any,
    batch_id: str,
    request_id: str
) -> str:
    """
    Start the simplified batch workflow with only BatchMemoryRequest reference.

    This is the preferred approach for all batch sizes as it avoids gRPC payload
    limits and simplifies the workflow code significantly.

    Args:
        client: Temporal client
        batch_id: Unique batch identifier
        request_id: BatchMemoryRequest objectId

    Returns:
        workflow_id: ID to track the workflow
    """
    from config import get_features

    features = get_features()
    task_queue = features.config.get("temporal", {}).get("task_queue", "memory-processing")

    workflow_id = f"batch-request-{batch_id}"

    # Note: When using Worker Versioning, workflows started via client.start_workflow
    # are automatically routed to the default build ID configured on the task queue.
    handle = await client.start_workflow(
        ProcessBatchMemoryFromRequestWorkflow.run,
        {"request_id": request_id, "batch_id": batch_id},
        id=workflow_id,
        task_queue=task_queue
    )

    logger.info(f"Started Temporal workflow with BatchMemoryRequest reference: {workflow_id} (will use default build ID on task queue)")
    return workflow_id
