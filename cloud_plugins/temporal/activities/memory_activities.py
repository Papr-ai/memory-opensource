"""
Temporal Activities for Memory Processing

Activities are executed by workers and contain the actual business logic
for processing memories in durable workflows.
"""

import os
# Ensure pure-Python protobuf before any imports that may transitively load protobuf/sentencepiece
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import asyncio
import httpx
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from urllib.parse import urlparse
from temporalio import activity
from services.logger_singleton import LoggerSingleton
from starlette.requests import Request as StarletteRequest
from datetime import datetime, UTC

logger = LoggerSingleton.get_logger(__name__)

# Only import for type checking to avoid runtime import/circular deps
if TYPE_CHECKING:
    from memory.memory_graph import MemoryGraph

# Module-level singleton for MemoryGraph to avoid re-creating per activity
_MEMORY_GRAPH_SINGLETON: Optional["MemoryGraph"] = None

async def _get_memory_graph_singleton():
    """Return a process-wide MemoryGraph instance with an ensured connection."""
    global _MEMORY_GRAPH_SINGLETON
    if _MEMORY_GRAPH_SINGLETON is None:
        from memory.memory_graph import MemoryGraph
        _MEMORY_GRAPH_SINGLETON = MemoryGraph()
    # Ensure the async connections are established before use
    await _MEMORY_GRAPH_SINGLETON.ensure_async_connection()
    return _MEMORY_GRAPH_SINGLETON
# Helpers to normalize Temporal payloads when Pydantic converter is not applied to activities
def _normalize_bd(raw_bd: Any) -> Dict[str, Any]:
    try:
        if hasattr(raw_bd, "model_dump"):
            return raw_bd.model_dump()
        if isinstance(raw_bd, dict):
            return raw_bd
        # Fallback best-effort
        return dict(raw_bd)
    except Exception:
        return {}

def _get_mem(bd: Dict[str, Any], idx: int) -> Dict[str, Any]:
    br = bd.get("batch_request") or {}
    mems = br.get("memories") or []
    if idx < 0 or idx >= len(mems):
        return {}
    m = mems[idx]
    if isinstance(m, dict):
        return m
    # Convert object-like to dict
    out = {}
    for key in ("content", "type", "metadata", "graph_generation"):
        out[key] = getattr(m, key, None)
    return out

def _get_auth(bd: Dict[str, Any]) -> Dict[str, Any]:
    ar = bd.get("auth_response") or {}
    if isinstance(ar, dict):
        return ar
    out = {}
    for key in ("developer_id", "end_user_id", "workspace_id", "organization_id", "namespace_id", "is_qwen_route", "session_token"):
        out[key] = getattr(ar, key, None)
    return out
# -------- Per-stage activities for visibility and durability --------

@activity.defn(name="add_memory_quick")
async def add_memory_quick(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create the base memory item (shell) and return identifiers.
    Payload: {"batch_data": BatchWorkflowData, "index": int}
    """
    from models.memory_models import AddMemoryRequest as AppAddMemoryRequest
    from models.shared_types import MemoryMetadata
    from models.memory_models import BatchMemoryRequest, OptimizedAuthResponse
    from fastapi import BackgroundTasks
    try:
        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]

        # Build app models
        mem = _get_mem(bd, idx)
        
        # Log the raw metadata before Pydantic validation
        raw_metadata = mem.get("metadata") or {}
        logger.info(f"🔍 add_memory_quick: Raw metadata keys: {list(raw_metadata.keys())}")
        logger.info(f"🔍 add_memory_quick: pageId in raw metadata: {raw_metadata.get('pageId')}")
        logger.info(f"🔍 add_memory_quick: customMetadata in raw metadata: {raw_metadata.get('customMetadata', {})}")
        
        # Clean memory data to match API AddMemoryRequest model
        # This removes extra fields like 'title' and moves 'external_user_id' to metadata
        cleaned_mem = _clean_memory_for_api(mem)
        
        app_req = AppAddMemoryRequest(
            content=cleaned_mem.get("content"),
            type=cleaned_mem.get("type"),
            metadata=MemoryMetadata(**(cleaned_mem.get("metadata") or {})) if cleaned_mem.get("metadata") else None,
        )
        
        # Log after Pydantic validation
        if app_req.metadata:
            logger.info(f"🔍 add_memory_quick: After Pydantic - pageId: {app_req.metadata.pageId}")
            logger.info(f"🔍 add_memory_quick: After Pydantic - customMetadata: {app_req.metadata.customMetadata}")

        # Ensure multi-tenant IDs propagate on metadata so Parse pointers can be set later
        if app_req.metadata is None:
            app_req.metadata = MemoryMetadata()
        try:
            if ar := _get_auth(payload.get("batch_data") or {}):
                org_id_tmp = ar.get("organization_id")
                ns_id_tmp = ar.get("namespace_id")
                if org_id_tmp:
                    app_req.metadata.organization_id = org_id_tmp
                if ns_id_tmp:
                    app_req.metadata.namespace_id = ns_id_tmp
        except Exception:
            pass

        ar = _get_auth(bd)
        # Honor legacy_route if provided on batch_data; default to True to match OSS
        legacy_route = bool(bd.get("legacy_route", True))
        # Build full auth payload with optional fields when available
        auth_kwargs: Dict[str, Any] = {
            "developer_id": ar.get("developer_id"),
            "end_user_id": ar.get("end_user_id"),
            "workspace_id": ar.get("workspace_id"),
            "is_qwen_route": bool(ar.get("is_qwen_route")),
            "session_token": ar.get("session_token"),
            "api_key": bd.get("api_key"),
        }
        # Multi-tenant optional fields
        org_id = ar.get("organization_id")
        ns_id = ar.get("namespace_id")
        if org_id and ns_id:
            auth_kwargs["organization_id"] = org_id
            auth_kwargs["namespace_id"] = ns_id
            auth_kwargs["auth_type"] = "organization"
            auth_kwargs["is_legacy_auth"] = False
        # Optional enrichment fields
        # If original auth_response dict is available on bd, pull richer options
        raw_ar = bd.get("auth_response") or {}
        if isinstance(raw_ar, dict):
            for opt_key in (
                "user_roles",
                "user_workspace_ids",
                "user_schemas",
                "cached_schema",
                "api_key_info",
                "updated_metadata",
                "updated_batch_request",
            ):
                if opt_key in raw_ar and raw_ar.get(opt_key) is not None:
                    auth_kwargs[opt_key] = raw_ar.get(opt_key)

        auth = OptimizedAuthResponse(**auth_kwargs)

        memory_graph = await _get_memory_graph_singleton()

        # Quick add via existing handler path with skip_background_processing=True
        # Build a minimal Request with auth headers
        header_items = [(b"x-client-type", b"temporal_worker"), (b"content-type", b"application/json")]
        api_key = bd.get("api_key")
        if api_key:
            header_items.extend([
                (b"x-api-key", api_key.encode()),
                (b"authorization", f"APIKey {api_key}".encode()),
            ])
        api_key = bd.get("api_key")
        if api_key:
            header_items.extend([
                (b"x-api-key", api_key.encode()),
                (b"authorization", f"APIKey {api_key}".encode())
            ])
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": "/temporal-quick", "headers": header_items}
        async def _empty_receive():
            return {"type": "http.request"}
        request = StarletteRequest(scope, _empty_receive)

        # Optional fields for handler
        upload_id = None
        try:
            meta_dict = mem.get("metadata") if isinstance(mem, dict) else None
            if isinstance(meta_dict, dict):
                # Extract upload_id from metadata or customMetadata
                upload_id = meta_dict.get("upload_id")
                if not upload_id and "customMetadata" in meta_dict:
                    custom_meta = meta_dict.get("customMetadata", {})
                    if isinstance(custom_meta, dict):
                        upload_id = custom_meta.get("upload_id")
                
                # If we found upload_id, set it at the top level of metadata for Parse query compatibility
                if upload_id and app_req.metadata:
                    app_req.metadata.upload_id = upload_id
                    logger.info(f"✅ Set upload_id as top-level metadata field: {upload_id}")
        except Exception as e:
            logger.warning(f"Failed to extract/set upload_id: {e}")
            
        post_object_id = None
        try:
            meta_dict = mem.get("metadata") if isinstance(mem, dict) else None
            if isinstance(meta_dict, dict):
                # Prefer explicit post_objectId if present; else fallback to pageId used by parsers
                post_object_id = meta_dict.get("post_objectId") or meta_dict.get("pageId")
        except Exception:
            pass

        from routes.memory_routes import common_add_memory_handler
        
        # Log what we're passing to common_add_memory_handler
        logger.info(f"🔍 add_memory_quick: Calling common_add_memory_handler with:")
        logger.info(f"  - user_id (developer_id): {auth.developer_id}")
        logger.info(f"  - end_user_id: {auth.end_user_id}")
        logger.info(f"  - upload_id: {upload_id}")
        logger.info(f"  - post_objectId: {post_object_id}")
        logger.info(f"  - pageId in metadata: {app_req.metadata.pageId if app_req.metadata else None}")
        
        # Create Neo4j session for this memory to ensure proper session lifecycle
        # Each parallel memory gets its own isolated session that lives for the entire operation
        await memory_graph.ensure_async_connection()
        
        logger.info(f"🔍 add_memory_quick: About to call common_add_memory_handler with:")
        logger.info(f"  - auth org_id: {auth.organization_id}, namespace_id: {auth.namespace_id}")
        logger.info(f"  - metadata org_id: {app_req.metadata.organization_id if app_req.metadata else None}, namespace_id: {app_req.metadata.namespace_id if app_req.metadata else None}")
        logger.info(f"  - post_objectId: {post_object_id}")
        logger.info(f"  - upload_id: {upload_id}")
        
        async with memory_graph.async_neo_conn.get_session() as task_neo_session:
            try:
                result = await common_add_memory_handler(
                        request=request,
                        memory_graph=memory_graph,
                        background_tasks=BackgroundTasks(),
                        neo_session=task_neo_session,  # Pass session that will live for entire operation
                        auth_response=auth,
                        memory_request=app_req,
                        skip_background_processing=True,
                        upload_id=upload_id,
                        post_objectId=post_object_id,
                        legacy_route=legacy_route,
                    )
                logger.info(f"✅ add_memory_quick: common_add_memory_handler returned successfully")
            except Exception as handler_error:
                logger.error(f"❌ add_memory_quick: common_add_memory_handler FAILED: {type(handler_error).__name__}: {handler_error}", exc_info=True)
                raise

        # Extract identifiers (handle dict or object)
        mem_id = None
        obj_id = None
        memory_chunk_ids = []
        
        logger.info(f"🔍 add_memory_quick: Extracting identifiers from result...")
        logger.info(f"  - result type: {type(result)}")
        logger.info(f"  - result.data exists: {hasattr(result, 'data') if result else False}")
        
        if result and result.data:
            logger.info(f"  - result.data length: {len(result.data)}")
            first = result.data[0]
            logger.info(f"  - first element type: {type(first)}")
            try:
                mem_id = first.memoryId
            except Exception:
                mem_id = first.get("memoryId") if isinstance(first, dict) else None
            try:
                obj_id = first.objectId
            except Exception:
                obj_id = first.get("objectId") if isinstance(first, dict) else None
            # Extract memoryChunkIds to preserve through the workflow
            try:
                memory_chunk_ids = first.memoryChunkIds if hasattr(first, 'memoryChunkIds') else []
            except Exception:
                memory_chunk_ids = first.get("memoryChunkIds", []) if isinstance(first, dict) else []
        else:
            logger.warning(f"⚠️  add_memory_quick: result or result.data is empty!")
            logger.warning(f"  - result: {result}")
            
        logger.info(f"add_memory_quick identifiers: memory_id={mem_id}, object_id={obj_id}, batch_id={bd.get('batch_id')}, memoryChunkIds={memory_chunk_ids}")
        # Return memory_id, object_id, batch_id, AND memoryChunkIds to preserve through workflow
        return {"memory_id": mem_id, "object_id": obj_id, "batch_id": bd.get("batch_id"), "memory_chunk_ids": memory_chunk_ids}
    except Exception as e:
        activity.heartbeat({"stage": "add_memory_quick", "error": str(e)})
        raise


def _clean_memory_for_api(mem: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert TemporalAddMemoryRequest format to API AddMemoryRequest format.
    
    TemporalAddMemoryRequest has:
    - title (Optional[str]) at top level
    - external_user_id (Optional[str]) at top level
    - metadata (Optional[Dict]) at top level
    
    API AddMemoryRequest has:
    - No title field
    - external_user_id inside metadata
    - metadata (MemoryMetadata) at top level
    
    Args:
        mem: Memory dictionary in TemporalAddMemoryRequest format
        
    Returns:
        Memory dictionary in API AddMemoryRequest format
    """
    cleaned = mem.copy()
    
    # Remove title field (not part of API AddMemoryRequest)
    cleaned.pop("title", None)
    
    # Move external_user_id from top level to metadata.external_user_id
    # API AddMemoryRequest expects it inside metadata, not at top level
    if "external_user_id" in cleaned:
        external_user_id = cleaned.pop("external_user_id")
        if "metadata" not in cleaned:
            cleaned["metadata"] = {}
        if isinstance(cleaned["metadata"], dict):
            cleaned["metadata"]["external_user_id"] = external_user_id
    
    return cleaned


async def _build_temporal_auth(bd: Dict[str, Any]):
    """Helper function to build OptimizedAuthResponse from batch_data."""
    from models.memory_models import OptimizedAuthResponse
    
    ar = _get_auth(bd)
    # Build full auth payload with optional fields when available
    auth_kwargs: Dict[str, Any] = {
        "developer_id": ar.get("developer_id"),
        "end_user_id": ar.get("end_user_id"),
        "workspace_id": ar.get("workspace_id"),
        "is_qwen_route": bool(ar.get("is_qwen_route")),
        "session_token": ar.get("session_token"),
        "api_key": bd.get("api_key"),
    }
    # Multi-tenant optional fields
    org_id = ar.get("organization_id")
    ns_id = ar.get("namespace_id")
    if org_id and ns_id:
        auth_kwargs["organization_id"] = org_id
        auth_kwargs["namespace_id"] = ns_id
        auth_kwargs["auth_type"] = "organization"
        auth_kwargs["is_legacy_auth"] = False
    # Optional enrichment fields
    raw_ar = bd.get("auth_response") or {}
    if isinstance(raw_ar, dict):
        for opt_key in (
            "user_roles",
            "user_workspace_ids",
            "user_schemas",
            "cached_schema",
            "api_key_info",
            "updated_metadata",
            "updated_batch_request",
        ):
            if opt_key in raw_ar and raw_ar.get(opt_key) is not None:
                auth_kwargs[opt_key] = raw_ar.get(opt_key)
    
    return OptimizedAuthResponse(**auth_kwargs)


@activity.defn(name="batch_add_memory_quick")
async def batch_add_memory_quick(batch_data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    TRUE BATCH version - processes multiple memories in a SINGLE database transaction using batch handlers.
    This replaces the for-loop approach with full batch processing.
    
    Args:
        batch_data_list: List of batch_data dictionaries (up to 10 memories per batch)
    
    Returns:
        List of results with memory_id, object_id, batch_id, memory_chunk_ids
    """
    try:
        from memory.memory_graph import MemoryGraph
        from routes.memory_routes import batch_common_add_memory_handler
        from models.memory_models import AddMemoryRequest as AppAddMemoryRequest
        from fastapi import BackgroundTasks
        
        logger.info(f"📦 batch_add_memory_quick: Processing {len(batch_data_list)} memories in TRUE BATCH MODE")
        
        # Initialize memory graph ONCE for all memories
        memory_graph = MemoryGraph()
        await memory_graph.ensure_async_connection()
        
        # Extract all memory requests and metadata from the batch
        memory_requests = []
        bd = None  # Will be set from first item
        upload_id = None
        post_object_id = None
        
        for idx, payload in enumerate(batch_data_list):
            try:
                bd = _normalize_bd(payload.get("batch_data"))
                mem = bd.get("batch_request", {}).get("memories", [{}])[payload.get("index", 0)]
                
                # Clean memory data to match API AddMemoryRequest model
                # This removes extra fields like 'title' and moves 'external_user_id' to metadata
                cleaned_mem = _clean_memory_for_api(mem)
                
                # Create AddMemoryRequest (API model, not Temporal model)
                app_req = AppAddMemoryRequest(**cleaned_mem)
                memory_requests.append(app_req)
                
                # Extract post info from first memory
                if idx == 0:
                    upload_id = bd.get("upload_id")
                    try:
                        meta_dict = mem.get("metadata") if isinstance(mem, dict) else None
                        if isinstance(meta_dict, dict):
                            post_object_id = meta_dict.get("post_objectId") or meta_dict.get("pageId")
                    except Exception:
                        pass
                        
            except Exception as item_error:
                logger.error(f"❌ Failed to prepare memory {idx}: {item_error}", exc_info=True)
                # Add a placeholder request
                memory_requests.append(None)
        
        # Filter out None values
        memory_requests = [req for req in memory_requests if req is not None]
        
        if not memory_requests:
            logger.error("No valid memory requests to process")
            return []
        
        # Build auth using helper function
        auth = await _build_temporal_auth(bd)
        
        # Create request object
        from starlette.requests import Request
        request_obj = Request(scope={
            "type": "http",
            "method": "POST",
            "headers": [(b"content-type", b"application/json")],
            "path": "/v1/memory/batch",
        })
        
        # Process all memories with SINGLE Neo4j session using TRUE BATCH handler
        async with memory_graph.async_neo_conn.get_session() as batch_neo_session:
            # Call TRUE BATCH handler - processes ALL memories in a single transaction
            responses = await batch_common_add_memory_handler(
                request=request_obj,
                memory_graph=memory_graph,
                background_tasks=BackgroundTasks(),
                neo_session=batch_neo_session,  # SHARED SESSION for entire batch
                auth_response=auth,
                memory_requests=memory_requests,  # ALL memories at once
                skip_background_processing=True,
                upload_id=upload_id,
                post_objectId=post_object_id,
                legacy_route=False,
            )
            
            # Extract identifiers from batch responses
            results = []
            for idx, response in enumerate(responses):
                try:
                    mem_id = None
                    obj_id = None
                    memory_chunk_ids = []
                    
                    if response and response.data:
                        first = response.data[0]
                        try:
                            mem_id = first.memoryId if hasattr(first, 'memoryId') else first.get("memoryId")
                            obj_id = first.objectId if hasattr(first, 'objectId') else first.get("objectId")
                            memory_chunk_ids = first.memoryChunkIds if hasattr(first, 'memoryChunkIds') else first.get("memoryChunkIds", [])
                        except Exception as e:
                            logger.error(f"Failed to extract IDs from response {idx}: {e}")
                    
                    results.append({
                        "memory_id": mem_id,
                        "object_id": obj_id,
                        "batch_id": bd.get("batch_id"),
                        "memory_chunk_ids": memory_chunk_ids
                    })
                    
                except Exception as item_error:
                    logger.error(f"❌ Failed to extract result {idx}: {item_error}", exc_info=True)
                    # Return empty result for this memory
                    results.append({
                        "memory_id": None,
                        "object_id": None,
                        "batch_id": bd.get("batch_id") if 'bd' in locals() else None,
                        "memory_chunk_ids": []
                    })
        
        logger.info(f"📊 batch_add_memory_quick: Completed {len(results)}/{len(batch_data_list)} memories")
        return results
        
    except Exception as e:
        activity.heartbeat({"stage": "batch_add_memory_quick", "error": str(e)})
        logger.error(f"❌ batch_add_memory_quick FAILED: {e}", exc_info=True)
        raise


@activity.defn(name="index_and_enrich_memory")
async def index_and_enrich_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run indexing/LLM enrichment for a memory that was quick-added.
    Payload: {"batch_data": BatchWorkflowData, "index": int, "quick": {"memory_id": str}}
    """
    try:
        from memory.memory_item import memory_item_from_dict, MemoryItem
        from memory.memory_item import memory_item_to_dict
        from models.shared_types import MemoryMetadata
        from models.memory_models import AddMemoryRequest as AppAddMemoryRequest
        from memory.memory_graph import MemoryGraph

        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]
        quick: Dict[str, Any] = payload.get("quick", {})

        memory_graph = await _get_memory_graph_singleton()

        # Build memory_dict minimally from batch request for the index/enrich step
        m = _get_mem(bd, idx)
        metadata = (m.get("metadata") or {})
        # Ensure org/namespace are present in metadata for downstream indexing/enrichment
        try:
            ar = _get_auth(bd)
            org_id_ie = ar.get("organization_id")
            ns_id_ie = ar.get("namespace_id")
            if org_id_ie and not metadata.get("organization_id"):
                metadata["organization_id"] = org_id_ie
            if ns_id_ie and not metadata.get("namespace_id"):
                metadata["namespace_id"] = ns_id_ie
        except Exception:
            pass
        memory_dict = {
            "id": quick.get("memory_id"),
            "objectId": quick.get("object_id"),  # Parse objectId distinct from memory_id
            "batch_id": bd.get("batch_id"),
            "content": m.get("content"),
            "metadata": metadata,
            "type": m.get("type"),
        }
        logger.info(f"index_and_enrich_memory payload.quick={quick}; memory_dict.keys={list(memory_dict.keys())}")

        activity.heartbeat({"stage": "index_and_enrich:start", "index": idx})

        # Execute the existing indexing/enrichment pipeline
        ar = _get_auth(bd)
        session_token = ar.get("session_token")
        user_id = ar.get("end_user_id")
        api_key = payload.get("api_key") or bd.get("api_key")
        
        # Extract GraphGeneration parameters from memory object
        graph_generation = m.get("graph_generation")
        graph_override = None
        schema_id = None
        property_overrides = None

        if graph_generation:
            if graph_generation.get("mode") == "manual" and graph_generation.get("manual"):
                graph_override = graph_generation["manual"]
            elif graph_generation.get("mode") == "auto" and graph_generation.get("auto"):
                auto_config = graph_generation["auto"]
                schema_id = auto_config.get("schema_id")
                property_overrides = auto_config.get("property_overrides")

        result = await memory_graph.process_memory_item_async(
            session_token=session_token,
            memory_dict=memory_dict,
            workspace_id=ar.get("workspace_id"),
            user_id=user_id,
            api_key=api_key,
            legacy_route=False,
            graph_override=graph_override,
            schema_id=schema_id,
            property_overrides=property_overrides
        )

        ok = bool(result and result.get("success"))
        activity.heartbeat({"stage": "index_and_enrich:end", "index": idx, "ok": ok})
        return {"ok": ok}
    except Exception as e:
        activity.heartbeat({"stage": "index_and_enrich", "error": str(e)})
        raise


@activity.defn(name="update_relationships")
async def update_relationships(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Create relationships after enrichment.
    Payload: {"batch_data": BatchWorkflowData, "index": int}
    """
    try:
        from memory.memory_item import memory_item_from_dict
        from memory.memory_graph import MemoryGraph

        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]

        m = _get_mem(bd, idx)
        metadata = (m.get("metadata") or {})

        memory_graph = await _get_memory_graph_singleton()

        # Use memory_id from quick stage if provided to satisfy memory_item_from_dict
        quick = payload.get("quick", {})
        memory_id = quick.get("memory_id")
        if not memory_id:
            # Nothing to relate if we didn't persist a base memory; treat as no-op
            activity.heartbeat({"stage": "update_relationships", "index": idx, "skipped": True})
            return {"ok": True, "skipped": True}
        memory_dict = {
            "id": memory_id,
            "objectId": quick.get("object_id"),  # keep Parse objectId separate from memory_id
            "batch_id": bd.get("batch_id"),
            "content": m.get("content"),
            "metadata": metadata,
            "type": m.get("type"),
        }
        logger.info(f"update_relationships payload.quick={quick}; relationships_json_in_payload={payload.get('relationships_json')}")
        memory_item_obj = memory_item_from_dict(memory_dict)

        activity.heartbeat({"stage": "update_relationships:start", "index": idx})
        # Use internal session management inside the method (it opens session if None)
        # Prefer relationships_json passed from previous stage if present
        relationships_json = payload.get("relationships_json")
        if not relationships_json:
            relationships_json = metadata.get("relationships_json", []) or []

        await memory_graph.update_memory_item_with_relationships(
            memory_item=memory_item_obj,
            relationships_json=relationships_json,
            workspace_id=_get_auth(bd).get("workspace_id"),
            user_id=_get_auth(bd).get("end_user_id"),
            neo_session=None,
            legacy_route=False,
        )
        activity.heartbeat({"stage": "update_relationships:end", "index": idx})
        return {"ok": True}
    except Exception as e:
        activity.heartbeat({"stage": "update_relationships", "error": str(e)})
        raise


from models.temporal_models import BatchWorkflowData


@activity.defn(name="process_memory_batch")
async def process_memory_batch(batch_data: BatchWorkflowData) -> Dict[str, Any]:
    """
    Process a batch of memories using real authentication and proper batch handler.

    Args:
        batch_data: {
            "batch_id": str,
            "batch_request": dict,  # Full BatchMemoryRequest data
            "auth_response": dict,  # Real OptimizedAuthResponse data
            "api_key": str,         # Real API key
            "webhook_url": Optional[str],
            "webhook_secret": Optional[str]
        }

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

    logger.info(f"Starting batch processing activity for batch {batch_id} with real authentication")

    try:
        # Import here to avoid circular dependencies
        from memory.memory_graph import MemoryGraph
        from routes.memory_routes import common_add_memory_batch_handler
        from fastapi import BackgroundTasks
        from unittest.mock import Mock
        from models.memory_models import BatchMemoryRequest, OptimizedAuthResponse, AddMemoryRequest as AppAddMemoryRequest
        from models.shared_types import MemoryMetadata

        # Create memory graph instance
        memory_graph = await _get_memory_graph_singleton()

        # Convert Temporal-safe batch request to application model using real AddMemoryRequest type
        app_memories: List[AppAddMemoryRequest] = []
        for m in batch_data.batch_request.memories:
            app_memories.append(
                AppAddMemoryRequest(
                    content=m.content,
                    type=m.type,
                    metadata=MemoryMetadata(**m.metadata) if m.metadata else None,
                )
            )
        batch_request = BatchMemoryRequest(
            user_id=batch_data.batch_request.user_id,
            external_user_id=batch_data.batch_request.external_user_id,
            organization_id=batch_data.batch_request.organization_id,
            namespace_id=batch_data.batch_request.namespace_id,
            memories=app_memories,
            batch_size=batch_data.batch_request.batch_size,
            webhook_url=batch_data.batch_request.webhook_url,
            webhook_secret=batch_data.batch_request.webhook_secret,
        )

        # Convert Temporal-safe auth to application model
        auth_response = OptimizedAuthResponse(
            developer_id=batch_data.auth_response.developer_id,
            end_user_id=batch_data.auth_response.end_user_id,
            workspace_id=batch_data.auth_response.workspace_id,
            organization_id=batch_data.auth_response.organization_id,
            namespace_id=batch_data.auth_response.namespace_id,
            is_qwen_route=batch_data.auth_response.is_qwen_route,
            session_token=batch_data.auth_response.session_token,
        )

        logger.info(f"Processing batch with end_user_id: {auth_response.end_user_id}, workspace_id: {auth_response.workspace_id}")

        # Build a minimal real Starlette Request with headers
        api_key = batch_data.api_key
        header_items = [(b"x-client-type", b"temporal_worker"), (b"content-type", b"application/json")]
        if api_key:
            header_items.extend([
                (b"x-api-key", api_key.encode()),
                (b"authorization", f"APIKey {api_key}".encode()),
            ])

        scope = {
            "type": "http",
            "http_version": "1.1",
            "method": "POST",
            "path": "/temporal-activity",
            "headers": header_items,
        }

        async def _empty_receive():
            return {"type": "http.request"}

        mock_request = StarletteRequest(scope, _empty_receive)

        # Use the real batch handler that handles proper validation and external user ID resolution
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
        activity.heartbeat(f"Batch processing completed")

        # Convert result to expected format
        if result.status == "success":
            return {
                "status": "completed",
                "total_processed": result.total_processed,
                "successful": result.total_successful,
                "failed": result.total_failed,
                "errors": [{"index": err.index, "error": err.error} for err in result.errors]
            }
        else:
            return {
                "status": "failed",
                "total_processed": result.total_processed,
                "successful": result.total_successful,
                "failed": result.total_failed,
                "errors": [{"index": err.index, "error": err.error} for err in result.errors]
            }

    except Exception as e:
        logger.error(f"Fatal error in batch processing: {e}")
        return {
            "status": "failed",
            "total_processed": 0,
            "successful": 0,
            "failed": len(batch_data.get("batch_request", {}).get("memories", [])),
            "errors": [{"index": -1, "error": f"Fatal error: {str(e)}"}]
        }


@activity.defn(name="send_webhook_notification")
async def send_webhook_notification(webhook_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send webhook notification about batch completion.

    Args:
        webhook_data: {
            "batch_id": str,
            "webhook_url": str,
            "webhook_secret": Optional[str],
            "results": dict
        }

    Returns:
        {"status": "success" | "failed", "response_code": int}
    """
    batch_id = webhook_data["batch_id"]
    webhook_url = (webhook_data["webhook_url"] or "").strip()
    # Default scheme if missing
    if webhook_url and "://" not in webhook_url:
        webhook_url = f"http://{webhook_url}"
    webhook_secret = webhook_data.get("webhook_secret")
    results = webhook_data["results"]

    logger.info(f"Sending webhook notification for batch {batch_id} to {webhook_url}")

    # Prepare webhook payload
    payload = {
        "event": "batch_completed",
        "batch_id": batch_id,
        "timestamp": activity.info().current_attempt_scheduled_time.isoformat(),
        "results": results
    }

    headers = {
        "Content-Type": "application/json",
        "User-Agent": "PAPR-Memory-Temporal/1.0"
    }

    # Add webhook secret if provided
    if webhook_secret:
        import hmac
        import hashlib
        import json

        signature = hmac.new(
            webhook_secret.encode(),
            json.dumps(payload).encode(),
            hashlib.sha256
        ).hexdigest()
        headers["X-PAPR-Signature"] = f"sha256={signature}"

    try:
        # In-process test fallback: allow posting directly to the FastAPI app when enabled
        parsed_target = urlparse(webhook_url)
        host_lower = (parsed_target.hostname or "").lower()
        # Single flag with backward-compat: prefer PAPR_ENABLE_TEST_WEBHOOK, fallback to TEST_WEBHOOK_INPROC
        def _truthy(name: str) -> bool:
            return os.getenv(name, "").lower() in {"1", "true", "yes"}
        enable_inproc = _truthy("PAPR_ENABLE_TEST_WEBHOOK") or _truthy("TEST_WEBHOOK_INPROC")
        if not enable_inproc and host_lower in {"test", "localhost", "127.0.0.1"}:
            # Auto-enable for local/test targets
            enable_inproc = True

        if enable_inproc:
            try:
                from main import app  # Use the same FastAPI app instance as tests
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
                    path = parsed_target.path or "/"
                    response = await client.post(path, json=payload, headers=headers)
            except Exception as e:
                logger.warning(f"In-process webhook fallback failed, falling back to network HTTP: {e}")
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(webhook_url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(webhook_url, json=payload, headers=headers)

        if response.status_code in [200, 201, 202]:
            logger.info(f"Webhook notification sent successfully for batch {batch_id}")
            return {"status": "success", "response_code": response.status_code}
        else:
            logger.warning(f"Webhook returned non-success status {response.status_code} for batch {batch_id}")
            return {"status": "failed", "response_code": response.status_code}

    except Exception as e:
        # If DNS resolution failed, try in-process as a last resort
        err_text = str(e)
        if "nodename nor servname provided" in err_text:
            try:
                from main import app  # Only available in same environment
                transport = httpx.ASGITransport(app=app)
                async with httpx.AsyncClient(transport=transport, base_url="http://test", timeout=30.0) as client:
                    # Best effort: use path from parsed URL or root
                    parsed_target = urlparse(webhook_url)
                    path = parsed_target.path or "/"
                    response = await client.post(path, json=payload, headers=headers)
                if response.status_code in [200, 201, 202]:
                    logger.info(f"Webhook notification sent via in-process fallback for batch {batch_id}")
                    return {"status": "success", "response_code": response.status_code}
                else:
                    logger.warning(f"In-process fallback returned non-success status {response.status_code} for batch {batch_id}")
                    return {"status": "failed", "response_code": response.status_code, "error": err_text}
            except Exception as e2:
                logger.error(f"Webhook DNS error and in-process fallback failed for batch {batch_id}: {e2}")
                return {"status": "failed", "error": err_text, "response_code": 0}
        logger.error(f"Failed to send webhook notification for batch {batch_id}: {e}")
        return {"status": "failed", "error": err_text, "response_code": 0}


# ------------ Fine-grained indexing activities for per-step visibility ------------

@activity.defn(name="idx_index_grouped_memory")
async def idx_index_grouped_memory(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Index memory in vector store (Qdrant) using BigBird embedding path.
    Payload: {"batch_data": BatchWorkflowData, "index": int, "quick": {"memory_id","object_id","batch_id"}}
    """
    try:
        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]
        quick: Dict[str, Any] = payload.get("quick", {})

        memory_graph = await _get_memory_graph_singleton()

        m = _get_mem(bd, idx)
        metadata = (m.get("metadata") or {})
        memory_dict = {
            "id": quick.get("memory_id"),
            "objectId": quick.get("object_id"),
            "batch_id": bd.get("batch_id"),
            "content": m.get("content"),
            "metadata": metadata,
            "type": m.get("type"),
        }

        # Prepare related mems list empty for indexing-only step
        related: List[Dict[str, Any]] = []
        bigbird_memory_dict = dict(memory_dict)
        if 'metadata' in bigbird_memory_dict:
            from memory.memory_graph import MemoryGraph as MG
            bigbird_memory_dict['metadata'] = MG.pinecone_compatible_metadata(bigbird_memory_dict['metadata'])
        await memory_graph.add_grouped_memory_item_to_qdrant(bigbird_memory_dict, related)
        return {"ok": True}
    except Exception as e:
        activity.heartbeat({"stage": "idx_index_grouped_memory", "error": str(e)})
        raise


@activity.defn(name="idx_generate_graph_schema")
async def idx_generate_graph_schema(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate and store schema graph for the memory item.
    Payload: {"batch_data": BatchWorkflowData, "index": int, "quick": {...}}
    Returns {"ok": bool, "schema_metrics": {...}, "relationships_json": [...]} for later steps.
    """
    try:
        from memory.memory_graph import MemoryGraph
        from memory.memory_item import memory_item_from_dict

        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]
        quick: Dict[str, Any] = payload.get("quick", {})
        ar = _get_auth(bd)

        memory_graph = await _get_memory_graph_singleton()

        m = _get_mem(bd, idx)
        metadata = (m.get("metadata") or {})
        # If objectId missing, try to resolve it from Parse by memoryId
        object_id = quick.get("object_id")
        if not object_id:
            try:
                from services.memory_management import retrieve_memory_item_parse
                resolved = await retrieve_memory_item_parse(
                    session_token=ar.get("session_token"),
                    memory_item_id=quick.get("memory_id"),
                    api_key=bd.get("api_key")
                )
                if resolved and isinstance(resolved, dict):
                    object_id = resolved.get("objectId") or object_id
            except Exception as _:
                pass

        memory_dict = {
            "id": quick.get("memory_id"),
            "objectId": object_id,
            "batch_id": bd.get("batch_id"),
            "content": m.get("content"),
            "metadata": metadata,
            "type": m.get("type"),
            "temporal_orchestrated": True,
        }

        # Minimal related memories and override to drive generation
        user_id = ar.get("end_user_id")
        # Fallback: get user_id from batch_request if not in auth_response
        if not user_id:
            br = bd.get("batch_request") or {}
            user_id = br.get("user_id") or br.get("external_user_id")
            logger.info(f"📋 idx_generate_graph_schema: Using user_id from batch_request: {user_id}")
        
        workspace_id = ar.get("workspace_id")
        session_token = ar.get("session_token")
        api_key = bd.get("api_key")
        # Honor legacy_route propagated from top-level batch data; default True for OSS parity
        legacy_route = bool(bd.get("legacy_route", True))

        # Extract schema specification from batch_data.schema_specification (preferred) or fallback to metadata
        schema_id = None
        graph_override = None
        property_overrides = None
        
        # First, try to get schema specification from batch_data
        schema_specification = bd.get("schema_specification")
        if schema_specification and isinstance(schema_specification, dict):
            schema_id = schema_specification.get("schema_id")
            graph_override = schema_specification.get("graph_override")
            property_overrides = schema_specification.get("property_overrides")

            # NEW: Extract mode and handle structured mode
            mode = schema_specification.get("mode", "auto")
            if mode in ("manual", "structured"):  # Accept both, 'structured' is deprecated
                # For structured mode, nodes/relationships should be in graph_override
                # If not already set, check for nodes directly in schema_specification
                if not graph_override:
                    nodes = schema_specification.get("nodes")
                    relationships = schema_specification.get("relationships")
                    if nodes:
                        graph_override = {"nodes": nodes, "relationships": relationships or []}
                        logger.info(f"📋 idx_generate_graph_schema: STRUCTURED mode - using exact nodes from schema_specification")

            # NEW: Extract node_constraints and convert to property_overrides format
            node_constraints = schema_specification.get("node_constraints")
            if node_constraints:
                if not property_overrides:
                    property_overrides = []
                # Convert node_constraints to property_overrides format
                for constraint in node_constraints:
                    if constraint.get("force"):
                        property_overrides.append({
                            "node_type": constraint.get("node_type"),
                            "properties": constraint["force"]
                        })
                logger.info(f"📋 idx_generate_graph_schema: Converted {len(node_constraints)} node_constraints to property_overrides")

            # NEW: Extract OMO fields and inject into metadata
            consent = schema_specification.get("consent", "implicit")
            risk = schema_specification.get("risk", "none")
            acl = schema_specification.get("acl")
            if consent or risk or acl:
                if not metadata:
                    metadata = {}
                metadata["consent"] = consent
                metadata["risk"] = risk
                if acl:
                    metadata["acl"] = acl
                # Update memory_dict with OMO-enhanced metadata
                memory_dict["metadata"] = metadata
                logger.info(f"📋 idx_generate_graph_schema: Injected OMO fields - consent={consent}, risk={risk}")

            logger.info(f"📋 idx_generate_graph_schema: Using schema_specification from batch_data: schema_id={schema_id}, mode={mode}")
        else:
            # Fallback: extract schema_id from metadata.customMetadata.schema_id (legacy)
            if metadata and isinstance(metadata, dict):
                custom_metadata = metadata.get("customMetadata", {})
                if isinstance(custom_metadata, dict):
                    schema_id = custom_metadata.get("schema_id")
                    if schema_id:
                        logger.info(f"📋 idx_generate_graph_schema: Fallback to schema_id from customMetadata: {schema_id}")

        # Ask the graph layer to produce schema and relationships via the same internal method
        # by running only the schema-extraction part; we reuse process_memory_item_async but
        # accept that it performs full pipeline; we only harvest schema metrics/relationships.
        logger.info(f"idx_generate_graph_schema quick={quick}")
        # Thread all parameters used by the core path, keep neo_session=None
        result = await memory_graph.process_memory_item_async(
            session_token=session_token,
            memory_dict=memory_dict,
            relationships_json=None,
            workspace_id=workspace_id,
            user_id=user_id,
            user_workspace_ids=None,
            api_key=api_key,
            neo_session=None,
            legacy_route=legacy_route,
            graph_override=graph_override,  # Pass graph_override from schema specification
            schema_id=schema_id,  # Pass schema_id to enforce custom schema
            property_overrides=property_overrides  # Pass property_overrides for node customization
        )
        ok = bool(result and result.get("success"))
        relationships = []
        schema_metrics = {}
        if result and result.get("data"):
            rel = result["data"].get("related_memories_relationships") or []
            relationships = rel
            schema_metrics = result["data"].get("metrics", {})
        return {"ok": ok, "schema_metrics": schema_metrics, "relationships_json": relationships}
    except Exception as e:
        activity.heartbeat({"stage": "idx_generate_graph_schema", "error": str(e)})
        raise


@activity.defn(name="process_batch_memories_from_parse_reference")
async def process_batch_memories_from_parse_reference(
    post_id: str,
    organization_id: str,
    namespace_id: str,
    user_id: str,
    workspace_id: Optional[str] = None,
    schema_specification: Optional[Dict[str, Any]] = None  # SchemaSpecificationMixin data for graph enforcement
) -> Dict[str, Any]:
    """Process batch memories stored in Parse Server (avoids Temporal GRPC limits)."""
    try:
        from services.memory_management import fetch_batch_memories_from_parse
        from routes.memory_routes import common_add_memory_batch_handler
        from models.memory_models import AddMemoryRequest, BatchMemoryRequest, OptimizedAuthResponse
        from models.shared_types import MemoryMetadata
        from fastapi import BackgroundTasks
        from starlette.requests import Request as StarletteRequest
        
        logger.info(f"Fetching batch memories from Parse Post {post_id}")
        batch_data = await fetch_batch_memories_from_parse(post_id)
        
        if not batch_data:
            raise Exception(f"Failed to fetch batch data from Post {post_id}")
        
        memories_dicts = batch_data.get("memories", [])
        logger.info(f"Processing {len(memories_dicts)} memories from Post {post_id}")
        
        memory_graph = await _get_memory_graph_singleton()

        auth_response = OptimizedAuthResponse(
            developer_id=user_id,
            end_user_id=user_id,
            workspace_id=workspace_id,
            organization_id=organization_id,
            namespace_id=namespace_id,
            is_qwen_route=False
        )

        # Extract schema information from schema_specification (preferred) or fallback to legacy schema_id
        schema_id = None
        graph_override = None

        if schema_specification and isinstance(schema_specification, dict):
            schema_id = schema_specification.get("schema_id")
            graph_override = schema_specification.get("graph_override")
            property_overrides = schema_specification.get("property_overrides")
            logger.info(f"📋 Using schema_specification from document workflow: schema_id={schema_id}")
        else:
            # Legacy fallback - this shouldn't happen with the new workflow but keeping for safety
            logger.info(f"📋 No schema_specification provided, using default schema processing")
        
        # Build AddMemoryRequest objects with schema information in metadata
        add_memory_requests: List[AddMemoryRequest] = []
        for mem_dict in memories_dicts:
            metadata_dict = mem_dict.get("metadata", {})
            
            # Clean up customMetadata to remove any non-primitive values that might have been
            # stored from previous runs (e.g., nested schema_specification dict)
            # MemoryMetadata expects customMetadata values to be primitives (str, int, float, bool, list[str])
            if metadata_dict and "customMetadata" in metadata_dict and isinstance(metadata_dict["customMetadata"], dict):
                cleaned_custom_metadata = {}
                for key, value in metadata_dict["customMetadata"].items():
                    # Only keep primitive types and lists of strings
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_custom_metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        cleaned_custom_metadata[key] = value
                    else:
                        logger.debug(f"Skipping non-primitive customMetadata field: {key}={type(value)}")
                metadata_dict["customMetadata"] = cleaned_custom_metadata
            
            # Inject schema_id into customMetadata for backward compatibility
            if schema_id:
                if not metadata_dict:
                    metadata_dict = {}
                if "customMetadata" not in metadata_dict:
                    metadata_dict["customMetadata"] = {}
                metadata_dict["customMetadata"]["schema_id"] = schema_id
            
            metadata = MemoryMetadata(**metadata_dict) if metadata_dict else None
            
            # Only pass fields allowed by AddMemoryRequest schema
            add_memory_requests.append(
                AddMemoryRequest(
                    content=mem_dict.get("content"),
                    type=mem_dict.get("type", "text"),
                    metadata=metadata,
                    schema_id=schema_id  # Pass schema_id for indexing
                )
            )

        # Create single BatchMemoryRequest (use <=50 per internal processing)
        batch_request = BatchMemoryRequest(
            memories=add_memory_requests,
            organization_id=organization_id,
            namespace_id=namespace_id,
            user_id=user_id,
            external_user_id=user_id,
            workspace_id=workspace_id,
            batch_size=min(50, len(add_memory_requests) or 1),
            schema_id=schema_id  # Pass schema_id to batch request
        )

        # Create a minimal Starlette request for handler
        header_items = [(b"x-client-type", b"temporal_batch_worker"), (b"content-type", b"application/json")]
        scope = {"type": "http", "http_version": "1.1", "method": "POST", "path": "/temporal-batch", "headers": header_items}
        async def _empty_receive():
            return {"type": "http.request"}
        request = StarletteRequest(scope, _empty_receive)

        # Invoke batch handler with synchronous processing
        # We're already in a Temporal workflow, so skip background processing to avoid nested workflows
        # This ensures the full indexing pipeline runs synchronously:
        # 1. add_memory_item_async (quick add)
        # 2. process_memory_item_async (LLM indexing/enrichment) 
        # 3. update_memory_item_with_relationships (Neo4j relationships)
        batch_response = await common_add_memory_batch_handler(
            request=request,
            memory_graph=memory_graph,
            background_tasks=BackgroundTasks(),
            auth_response=auth_response,
            memory_request_batch=batch_request,
            skip_background_processing=True,  # We're already in Temporal, avoid nested workflows
            legacy_route=False
        )

        logger.info(
            f"Batch processing complete for Post {post_id}: "
            f"{getattr(batch_response, 'total_successful', 0)}/{len(memories_dicts)} successful"
        )

        errors_list = [
            {"index": err.index, "error": err.error}
            for err in getattr(batch_response, 'errors', []) or []
        ]

        return {
            "post_id": post_id,
            "status": getattr(batch_response, 'status', 'completed'),
            "total_processed": getattr(batch_response, 'total_processed', len(memories_dicts)),
            "successful": getattr(batch_response, 'total_successful', 0),
            "failed": getattr(batch_response, 'total_failed', 0),
            "errors": errors_list
        }
        
    except Exception as e:
        logger.error(f"Failed to process batch from Parse Post {post_id}: {e}", exc_info=True)
        raise


@activity.defn(name="fetch_batch_memories_from_post")
async def fetch_batch_memories_from_post(
    post_id: str,
    organization_id: str,
    namespace_id: str,
    user_id: str,
    workspace_id: Optional[str] = None,
    schema_specification: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Fetch batch memory requests from Parse Post and return them as a list.
    
    This activity is called from ProcessBatchMemoryFromPostWorkflow to fetch memories
    before delegating to ProcessBatchMemoryWorkflow for multi-stage processing.
    
    Args:
        post_id: Parse Post ID containing batch memories
        organization_id: Organization ID
        namespace_id: Namespace ID
        user_id: End user ID
        workspace_id: Workspace ID
        schema_specification: Optional schema specification dict with schema_id, graph_override, property_overrides
        
    Returns:
        {"memories": List[Dict[str, Any]]} - List of AddMemoryRequest dicts ready for processing
    """
    try:
        activity.heartbeat(f"Fetching batch memories from Post {post_id}")
        
        from services.memory_management import fetch_batch_memories_from_parse
        
        logger.info(f"Fetching batch memories from Post {post_id}")
        logger.info(f"   Organization: {organization_id}, Namespace: {namespace_id}")
        logger.info(f"   User: {user_id}, Workspace: {workspace_id}")
        
        # Fetch memories from Parse Server
        # No authentication needed - get_parse_headers() uses PARSE_MASTER_KEY
        batch_data = await fetch_batch_memories_from_parse(post_id)
        
        if not batch_data:
            error_msg = f"Failed to fetch batch data from Post {post_id} - fetch returned None (likely due to exception)"
            logger.error(error_msg)
            raise Exception(error_msg)
        
        memories_dicts = batch_data.get("memories", [])
        
        logger.info(f"Fetched {len(memories_dicts)} memories from Post {post_id}")
        
        # Extract schema information from schema_specification
        schema_id = None
        graph_override = None
        property_overrides = None

        if schema_specification and isinstance(schema_specification, dict):
            schema_id = schema_specification.get("schema_id")
            graph_override = schema_specification.get("graph_override")
            property_overrides = schema_specification.get("property_overrides")
            logger.info(f"📋 Schema specification: schema_id={schema_id}")
        
        # Clean up and prepare memories for ProcessBatchMemoryWorkflow
        prepared_memories = []
        for mem_dict in memories_dicts:
            # Ensure required fields exist
            if not mem_dict.get("content"):
                logger.warning(f"Skipping memory without content: {mem_dict}")
                continue
            
            metadata_dict = mem_dict.get("metadata", {})
            
            # Clean up customMetadata to remove non-primitive values
            if metadata_dict and "customMetadata" in metadata_dict and isinstance(metadata_dict["customMetadata"], dict):
                cleaned_custom_metadata = {}
                for key, value in metadata_dict["customMetadata"].items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_custom_metadata[key] = value
                    elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                        cleaned_custom_metadata[key] = value
                metadata_dict["customMetadata"] = cleaned_custom_metadata
            
            # Ensure metadata structure exists
            if not metadata_dict:
                metadata_dict = {}
            if "customMetadata" not in metadata_dict:
                metadata_dict["customMetadata"] = {}
            
            # Inject organization_id, namespace_id, and schema_id at BOTH levels:
            # 1. Top-level metadata (for auth/validation checks in common_add_memory_handler)
            # 2. customMetadata (for activities that process individual memories)
            # This ensures the namespace_id from the workflow parameters overrides any stale value from Parse
            metadata_dict["organization_id"] = organization_id
            metadata_dict["namespace_id"] = namespace_id
            metadata_dict["customMetadata"]["organization_id"] = organization_id
            metadata_dict["customMetadata"]["namespace_id"] = namespace_id
            
            if schema_id:
                metadata_dict["customMetadata"]["schema_id"] = schema_id
            
            # Prepare memory dict compatible with TemporalAddMemoryRequest
            # Note: organization_id, namespace_id, and schema_id are:
            # 1. In metadata.customMetadata (for individual memory processing)
            # 2. At the batch level in BatchMemoryRequest (for batch-wide scoping)
            prepared_memory = {
                "content": mem_dict.get("content"),
                "type": mem_dict.get("type", "text"),
                "metadata": metadata_dict,
                "title": mem_dict.get("title"),
                "external_user_id": user_id
            }
            
            prepared_memories.append(prepared_memory)
        
        logger.info(f"Prepared {len(prepared_memories)} memories for multi-stage processing")
        
        # Return memories and schema specification for ProcessBatchMemoryWorkflow
        result = {"memories": prepared_memories}
        
        # Include schema specification if provided
        if schema_specification:
            result["schema_specification"] = schema_specification
            logger.info(f"Returning schema_specification with schema_id={schema_id}")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to fetch batch memories from Post {post_id}: {e}", exc_info=True)
        raise


@activity.defn(name="fetch_and_process_batch_request")
async def fetch_and_process_batch_request(
    request_id: str,
    batch_id: str
) -> Dict[str, Any]:
    """
    Fetch BatchMemoryRequest from Parse and process all memories.

    This consolidated activity handles the entire batch processing pipeline:
    1. Fetch BatchMemoryRequest from Parse
    2. Download and decompress batch data
    3. Process each memory through the pipeline
    4. Update BatchMemoryRequest with results
    5. Send webhook if configured

    Args:
        request_id: BatchMemoryRequest objectId
        batch_id: Batch identifier for logging

    Returns:
        {
            "status": "completed" | "partial_failure",
            "total": int,
            "successful": int,
            "failed": int,
            "errors": List[Dict]
        }
    """
    try:
        from services.memory_management import (
            fetch_batch_memory_request_from_parse,
            update_batch_request_status
        )
        from memory.memory_graph import MemoryGraph
        from routers.v1.memory_routes_v1 import common_add_memory_handler
        from models.memory_models import AddMemoryRequest
        from models.shared_types import MemoryMetadata
        from fastapi import BackgroundTasks
        from starlette.requests import Request as StarletteRequest
        from datetime import datetime, UTC

        activity.logger.info(f"Fetching batch request {request_id}")

        # Step 1: Fetch BatchMemoryRequest from Parse
        # heartbeat_details is a list, not a dict - extract api_key if present
        heartbeat_details = activity.info().heartbeat_details
        api_key = None
        if heartbeat_details and isinstance(heartbeat_details, (list, tuple)) and len(heartbeat_details) > 0:
            if isinstance(heartbeat_details[0], dict):
                api_key = heartbeat_details[0].get("api_key")
        
        batch_request = await fetch_batch_memory_request_from_parse(
            request_id=request_id,
            api_key=api_key
        )

        if not batch_request:
            raise ValueError(f"BatchMemoryRequest {request_id} not found")

        memories = getattr(batch_request, 'memories', [])
        total_memories = len(memories)

        activity.logger.info(f"Processing {total_memories} memories for batch {batch_id}")

        # Step 2: Update status to "processing"
        await update_batch_request_status(
            request_id=request_id,
            status="processing"
        )

        # Step 3: Process memories
        memory_graph = await _get_memory_graph_singleton()

        # Build auth response from batch request data
        auth_response = {
            "organization_id": batch_request.organization.objectId if batch_request.organization else None,
            "namespace_id": batch_request.namespace.objectId if batch_request.namespace else None,
            "end_user_id": batch_request.user.objectId if batch_request.user else None,
            "workspace_id": batch_request.workspace.objectId if batch_request.workspace else None,
        }

        # Create mock request for common_add_memory_handler
        async def _empty_receive():
            return {"type": "http.request"}

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/v1/memory/batch",
            "headers": []
        }
        request = StarletteRequest(scope, _empty_receive)

        results = []
        success_count = 0
        errors = []

        for idx, memory_data in enumerate(memories):
            try:
                # Send heartbeat every 10 memories
                if idx % 10 == 0:
                    activity.heartbeat(f"Processing {idx}/{total_memories}")

                # Convert memory dict to AddMemoryRequest
                metadata_dict = memory_data.get("metadata", {})
                metadata = MemoryMetadata(**metadata_dict) if metadata_dict else None

                memory_request = AddMemoryRequest(
                    content=memory_data.get("content"),
                    type=memory_data.get("type", "text"),
                    metadata=metadata,
                    title=memory_data.get("title"),
                    external_user_id=memory_data.get("external_user_id")
                )

                # Process memory through existing pipeline with full indexing/enrichment
                # Set skip_background_processing=False to enable:
                # 1. process_memory_item_async (LLM indexing, usecase/goals, related memories, schema generation)
                # 2. update_memory_item_with_relationships (Neo4j graph relationships)
                result = await common_add_memory_handler(
                    request=request,
                    memory_graph=memory_graph,
                    background_tasks=BackgroundTasks(),
                    neo_session=None,
                    auth_response=auth_response,
                    memory_request=memory_request,
                    skip_background_processing=False,  # Enable full pipeline!
                    upload_id=None,
                    post_objectId=None,
                    legacy_route=False
                )

                if result and result.data:
                    results.append({
                        "success": True,
                        "memory_id": result.data.objectId,
                        "index": idx
                    })
                    success_count += 1
                else:
                    results.append({
                        "success": False,
                        "index": idx,
                        "error": "No result returned"
                    })
                    errors.append({
                        "index": idx,
                        "error": "No result returned",
                        "timestamp": datetime.now(UTC).isoformat()
                    })

            except Exception as e:
                activity.logger.error(
                    f"Failed to process memory {idx}: {str(e)}",
                    exc_info=True
                )
                results.append({
                    "success": False,
                    "index": idx,
                    "error": str(e)
                })
                errors.append({
                    "index": idx,
                    "error": str(e),
                    "timestamp": datetime.now(UTC).isoformat()
                })

            # Update progress every 10 memories
            if (idx + 1) % 10 == 0:
                await update_batch_request_status(
                    request_id=request_id,
                    status="processing",
                    processed_count=idx + 1
                )

        # Step 4: Finalize batch request
        fail_count = len(errors)
        final_status = "completed" if fail_count == 0 else "partial_failure"

        await update_batch_request_status(
            request_id=request_id,
            status=final_status,
            processed_count=total_memories,
            success_count=success_count,
            fail_count=fail_count,
            errors=errors
        )

        # Step 5: Send webhook if configured
        if batch_request.webhookUrl:
            try:
                await _send_batch_webhook(
                    webhook_url=batch_request.webhookUrl,
                    webhook_secret=batch_request.webhookSecret,
                    batch_id=batch_id,
                    status=final_status,
                    total=total_memories,
                    successful=success_count,
                    failed=fail_count
                )
            except Exception as e:
                activity.logger.error(f"Failed to send webhook: {e}")

        activity.logger.info(
            f"Completed batch {batch_id}: {success_count}/{total_memories} successful, "
            f"{fail_count} failed"
        )

        return {
            "status": final_status,
            "total": total_memories,
            "successful": success_count,
            "failed": fail_count,
            "errors": errors
        }

    except Exception as e:
        activity.logger.error(f"Fatal error processing batch {batch_id}: {e}", exc_info=True)

        # Update batch request with failure
        try:
            await update_batch_request_status(
                request_id=request_id,
                status="failed",
                error=str(e)
            )
        except:
            pass

        raise


async def _send_batch_webhook(
    webhook_url: str,
    webhook_secret: Optional[str],
    batch_id: str,
    status: str,
    total: int,
    successful: int,
    failed: int
) -> None:
    """Send webhook notification for batch completion"""
    import httpx
    import hmac
    import hashlib
    import json

    payload = {
        "batch_id": batch_id,
        "status": status,
        "total": total,
        "successful": successful,
        "failed": failed,
        "timestamp": datetime.now(UTC).isoformat()
    }

    headers = {"Content-Type": "application/json"}

    # Add HMAC signature if secret provided
    if webhook_secret:
        payload_str = json.dumps(payload)
        signature = hmac.new(
            webhook_secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        headers["X-Papr-Signature"] = f"sha256={signature}"

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            webhook_url,
            json=payload,
            headers=headers
        )

        if response.status_code not in [200, 201, 202, 204]:
            raise Exception(f"Webhook failed: {response.status_code}")


@activity.defn(name="idx_update_metrics")
async def idx_update_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Persist operation costs/metrics back to Parse Memory item.
    Payload: {"batch_data": BatchWorkflowData, "index": int, "quick": {...}, "metrics": {...}}
    """
    try:
        from memory.memory_graph import MemoryGraph, update_memory_item

        bd = _normalize_bd(payload.get("batch_data"))
        idx: int = payload["index"]
        quick: Dict[str, Any] = payload.get("quick", {})
        metrics = payload.get("metrics") or {}
        ar = _get_auth(bd)
        legacy_route = bool(bd.get("legacy_route", True))

        memory_graph = await _get_memory_graph_singleton()

        m = _get_mem(bd, idx)
        memory_dict = {
            "id": quick.get("memory_id"),
            "objectId": quick.get("object_id"),
            "batch_id": bd.get("batch_id"),
            "content": m.get("content"),
            "metadata": (m.get("metadata") or {}),
            "type": m.get("type"),
            "metrics": metrics,
            # Preserve memoryChunkIds from quick_add result
            "memoryChunkIds": quick.get("memory_chunk_ids", []),
        }

        # Persist metrics via existing helper
        logger.info(f"idx_update_metrics metrics keys={list(metrics.keys()) if isinstance(metrics, dict) else type(metrics)} for objectId={memory_dict.get('objectId')}, memoryChunkIds={memory_dict.get('memoryChunkIds')}")
        await update_memory_item(
            ar.get("session_token"),
            memory_dict,
            None,
            api_key=bd.get("api_key"),
        )
        return {"ok": True}
    except Exception as e:
        activity.heartbeat({"stage": "idx_update_metrics", "error": str(e)})
        raise


@activity.defn(name="link_batch_memories_to_post")
async def link_batch_memories_to_post(
    memory_object_ids: List[str],
    post_id: str,
    user_id: str,
    organization_id: str,
    namespace_id: str,
    workspace_id: str
) -> Dict[str, Any]:
    """
    Link a list of Memory objectIds to a Post.
    
    Args:
        memory_object_ids: List of Parse Memory objectIds to link
        post_id: Parse Post objectId to link to
        user_id: User ID for auth
        organization_id: Organization ID
        namespace_id: Namespace ID
        workspace_id: Workspace ID
    
    Returns:
        Dict with linked_count, post_id, and success status
    """
    try:
        activity.heartbeat(f"Linking {len(memory_object_ids)} memories to Post {post_id}")
        
        from core.document_processing.parse_integration import ParseDocumentIntegration
        
        if not memory_object_ids:
            logger.warning(f"No memory_object_ids provided to link to Post {post_id}")
            return {"linked_count": 0, "post_id": post_id, "success": False, "error": "No memory IDs provided"}
        
        memory_graph = await _get_memory_graph_singleton()
        
        logger.info(f"Linking {len(memory_object_ids)} memories to Post {post_id}")
        logger.debug(f"Memory IDs: {memory_object_ids[:10]}..." if len(memory_object_ids) > 10 else f"Memory IDs: {memory_object_ids}")
        
        # Link them to the Post
        integration = ParseDocumentIntegration(memory_graph)
        success = await integration.link_memories_to_post(post_id, memory_object_ids)
        
        return {
            "linked_count": len(memory_object_ids) if success else 0,
            "post_id": post_id,
            "success": success
        }
                
    except Exception as e:
        logger.error(f"Failed to link memories to post: {e}", exc_info=True)
        raise